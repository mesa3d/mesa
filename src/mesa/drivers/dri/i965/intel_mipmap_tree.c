/*
 * Copyright 2006 VMware, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <GL/gl.h>
#include <GL/internal/dri_interface.h>
#include <drm_fourcc.h>

#include "intel_batchbuffer.h"
#include "intel_image.h"
#include "intel_mipmap_tree.h"
#include "intel_tex.h"
#include "intel_blit.h"
#include "intel_fbo.h"

#include "brw_blorp.h"
#include "brw_context.h"
#include "brw_state.h"

#include "main/enums.h"
#include "main/fbobject.h"
#include "main/formats.h"
#include "main/glformats.h"
#include "main/texcompress_etc.h"
#include "main/teximage.h"
#include "main/streaming-load-memcpy.h"

#include "util/format_srgb.h"

#include "x86/common_x86_asm.h"

#define FILE_DEBUG_FLAG DEBUG_MIPTREE

static void *intel_miptree_map_raw(struct brw_context *brw,
                                   struct intel_mipmap_tree *mt,
                                   GLbitfield mode);

static void intel_miptree_unmap_raw(struct intel_mipmap_tree *mt);

static bool
intel_miptree_supports_mcs(struct brw_context *brw,
                           const struct intel_mipmap_tree *mt)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;

   /* MCS compression only applies to multisampled miptrees */
   if (mt->surf.samples <= 1)
      return false;

   /* Prior to Gen7, all MSAA surfaces used IMS layout. */
   if (devinfo->gen < 7)
      return false;

   /* See isl_surf_get_mcs_surf for details. */
   if (mt->surf.samples == 16 && mt->surf.logical_level0_px.width > 8192)
      return false;

   /* In Gen7, IMS layout is only used for depth and stencil buffers. */
   switch (_mesa_get_format_base_format(mt->format)) {
   case GL_DEPTH_COMPONENT:
   case GL_STENCIL_INDEX:
   case GL_DEPTH_STENCIL:
      return false;
   default:
      /* From the Ivy Bridge PRM, Vol4 Part1 p77 ("MCS Enable"):
       *
       *   This field must be set to 0 for all SINT MSRTs when all RT channels
       *   are not written
       *
       * In practice this means that we have to disable MCS for all signed
       * integer MSAA buffers.  The alternative, to disable MCS only when one
       * of the render target channels is disabled, is impractical because it
       * would require converting between CMS and UMS MSAA layouts on the fly,
       * which is expensive.
       */
      if (devinfo->gen == 7 && _mesa_get_format_datatype(mt->format) == GL_INT) {
         return false;
      } else {
         return true;
      }
   }
}

static bool
intel_tiling_supports_ccs(const struct brw_context *brw,
                          enum isl_tiling tiling)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;

   /* From the Ivy Bridge PRM, Vol2 Part1 11.7 "MCS Buffer for Render
    * Target(s)", beneath the "Fast Color Clear" bullet (p326):
    *
    *     - Support is limited to tiled render targets.
    *
    * Gen9 changes the restriction to Y-tile only.
    */
   if (devinfo->gen >= 9)
      return tiling == ISL_TILING_Y0;
   else if (devinfo->gen >= 7)
      return tiling != ISL_TILING_LINEAR;
   else
      return false;
}

/**
 * For a single-sampled render target ("non-MSRT"), determine if an MCS buffer
 * can be used. This doesn't (and should not) inspect any of the properties of
 * the miptree's BO.
 *
 * From the Ivy Bridge PRM, Vol2 Part1 11.7 "MCS Buffer for Render Target(s)",
 * beneath the "Fast Color Clear" bullet (p326):
 *
 *     - Support is for non-mip-mapped and non-array surface types only.
 *
 * And then later, on p327:
 *
 *     - MCS buffer for non-MSRT is supported only for RT formats 32bpp,
 *       64bpp, and 128bpp.
 *
 * From the Skylake documentation, it is made clear that X-tiling is no longer
 * supported:
 *
 *     - MCS and Lossless compression is supported for TiledY/TileYs/TileYf
 *     non-MSRTs only.
 */
static bool
intel_miptree_supports_ccs(struct brw_context *brw,
                           const struct intel_mipmap_tree *mt)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;

   /* MCS support does not exist prior to Gen7 */
   if (devinfo->gen < 7)
      return false;

   /* This function applies only to non-multisampled render targets. */
   if (mt->surf.samples > 1)
      return false;

   /* MCS is only supported for color buffers */
   if (!_mesa_is_format_color_format(mt->format))
      return false;

   if (mt->cpp != 4 && mt->cpp != 8 && mt->cpp != 16)
      return false;

   const bool mip_mapped = mt->first_level != 0 || mt->last_level != 0;
   const bool arrayed = mt->surf.logical_level0_px.array_len > 1 ||
                        mt->surf.logical_level0_px.depth > 1;

   if (arrayed) {
       /* Multisample surfaces with the CMS layout are not layered surfaces,
        * yet still have physical_depth0 > 1. Assert that we don't
        * accidentally reject a multisampled surface here. We should have
        * rejected it earlier by explicitly checking the sample count.
        */
      assert(mt->surf.samples == 1);
   }

   /* Handle the hardware restrictions...
    *
    * All GENs have the following restriction: "MCS buffer for non-MSRT is
    * supported only for RT formats 32bpp, 64bpp, and 128bpp."
    *
    * From the HSW PRM Volume 7: 3D-Media-GPGPU, page 652: (Color Clear of
    * Non-MultiSampler Render Target Restrictions) Support is for
    * non-mip-mapped and non-array surface types only.
    *
    * From the BDW PRM Volume 7: 3D-Media-GPGPU, page 649: (Color Clear of
    * Non-MultiSampler Render Target Restriction). Mip-mapped and arrayed
    * surfaces are supported with MCS buffer layout with these alignments in
    * the RT space: Horizontal Alignment = 256 and Vertical Alignment = 128.
    *
    * From the SKL PRM Volume 7: 3D-Media-GPGPU, page 632: (Color Clear of
    * Non-MultiSampler Render Target Restriction). Mip-mapped and arrayed
    * surfaces are supported with MCS buffer layout with these alignments in
    * the RT space: Horizontal Alignment = 128 and Vertical Alignment = 64.
    */
   if (devinfo->gen < 8 && (mip_mapped || arrayed))
      return false;

   /* The PRM doesn't say this explicitly, but fast-clears don't appear to
    * work for 3D textures until gen9 where the layout of 3D textures changes
    * to match 2D array textures.
    */
   if (devinfo->gen <= 8 && mt->surf.dim != ISL_SURF_DIM_2D)
      return false;

   /* There's no point in using an MCS buffer if the surface isn't in a
    * renderable format.
    */
   if (!brw->mesa_format_supports_render[mt->format])
      return false;

   return true;
}

static bool
intel_tiling_supports_hiz(const struct brw_context *brw,
                          enum isl_tiling tiling)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;

   if (devinfo->gen < 6)
      return false;

   return tiling == ISL_TILING_Y0;
}

static bool
intel_miptree_supports_hiz(const struct brw_context *brw,
                           const struct intel_mipmap_tree *mt)
{
   if (!brw->has_hiz)
      return false;

   switch (mt->format) {
   case MESA_FORMAT_Z_FLOAT32:
   case MESA_FORMAT_Z32_FLOAT_S8X24_UINT:
   case MESA_FORMAT_Z24_UNORM_X8_UINT:
   case MESA_FORMAT_Z24_UNORM_S8_UINT:
   case MESA_FORMAT_Z_UNORM16:
      return true;
   default:
      return false;
   }
}

/**
 * Return true if the format that will be used to access the miptree is
 * CCS_E-compatible with the miptree's linear/non-sRGB format.
 *
 * Why use the linear format? Well, although the miptree may be specified with
 * an sRGB format, the usage of that color space/format can be toggled. Since
 * our HW tends to support more linear formats than sRGB ones, we use this
 * format variant for check for CCS_E compatibility.
 */
static bool
format_ccs_e_compat_with_miptree(const struct gen_device_info *devinfo,
                                 const struct intel_mipmap_tree *mt,
                                 enum isl_format access_format)
{
   assert(mt->aux_usage == ISL_AUX_USAGE_CCS_E);

   mesa_format linear_format = _mesa_get_srgb_format_linear(mt->format);
   enum isl_format isl_format = brw_isl_format_for_mesa_format(linear_format);
   return isl_formats_are_ccs_e_compatible(devinfo, isl_format, access_format);
}

static bool
intel_miptree_supports_ccs_e(struct brw_context *brw,
                             const struct intel_mipmap_tree *mt)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;

   if (devinfo->gen < 9)
      return false;

   /* For now compression is only enabled for integer formats even though
    * there exist supported floating point formats also. This is a heuristic
    * decision based on current public benchmarks. In none of the cases these
    * formats provided any improvement but a few cases were seen to regress.
    * Hence these are left to to be enabled in the future when they are known
    * to improve things.
    */
   if (_mesa_get_format_datatype(mt->format) == GL_FLOAT)
      return false;

   if (!intel_miptree_supports_ccs(brw, mt))
      return false;

   /* Many window system buffers are sRGB even if they are never rendered as
    * sRGB.  For those, we want CCS_E for when sRGBEncode is false.  When the
    * surface is used as sRGB, we fall back to CCS_D.
    */
   mesa_format linear_format = _mesa_get_srgb_format_linear(mt->format);
   enum isl_format isl_format = brw_isl_format_for_mesa_format(linear_format);
   return isl_format_supports_ccs_e(&brw->screen->devinfo, isl_format);
}

/**
 * Determine depth format corresponding to a depth+stencil format,
 * for separate stencil.
 */
mesa_format
intel_depth_format_for_depthstencil_format(mesa_format format) {
   switch (format) {
   case MESA_FORMAT_Z24_UNORM_S8_UINT:
      return MESA_FORMAT_Z24_UNORM_X8_UINT;
   case MESA_FORMAT_Z32_FLOAT_S8X24_UINT:
      return MESA_FORMAT_Z_FLOAT32;
   default:
      return format;
   }
}

static bool
create_mapping_table(GLenum target, unsigned first_level, unsigned last_level,
                     unsigned depth0, struct intel_mipmap_level *table)
{
   for (unsigned level = first_level; level <= last_level; level++) {
      const unsigned d =
         target == GL_TEXTURE_3D ? minify(depth0, level) : depth0;

      table[level].slice = calloc(d, sizeof(*table[0].slice));
      if (!table[level].slice)
         goto unwind;
   }

   return true;

unwind:
   for (unsigned level = first_level; level <= last_level; level++)
      free(table[level].slice);

   return false;
}

static bool
needs_separate_stencil(const struct brw_context *brw,
                       struct intel_mipmap_tree *mt,
                       mesa_format format)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;

   if (_mesa_get_format_base_format(format) != GL_DEPTH_STENCIL)
      return false;

   if (devinfo->must_use_separate_stencil)
      return true;

   return brw->has_separate_stencil &&
          intel_miptree_supports_hiz(brw, mt);
}

/**
 * Choose the aux usage for this miptree.  This function must be called fairly
 * late in the miptree create process after we have a tiling.
 */
static void
intel_miptree_choose_aux_usage(struct brw_context *brw,
                               struct intel_mipmap_tree *mt)
{
   assert(mt->aux_usage == ISL_AUX_USAGE_NONE);

   if (intel_miptree_supports_mcs(brw, mt)) {
      assert(mt->surf.msaa_layout == ISL_MSAA_LAYOUT_ARRAY);
      mt->aux_usage = ISL_AUX_USAGE_MCS;
   } else if (intel_tiling_supports_ccs(brw, mt->surf.tiling) &&
              intel_miptree_supports_ccs(brw, mt)) {
      if (!unlikely(INTEL_DEBUG & DEBUG_NO_RBC) &&
          intel_miptree_supports_ccs_e(brw, mt)) {
         mt->aux_usage = ISL_AUX_USAGE_CCS_E;
      } else {
         mt->aux_usage = ISL_AUX_USAGE_CCS_D;
      }
   } else if (intel_tiling_supports_hiz(brw, mt->surf.tiling) &&
              intel_miptree_supports_hiz(brw, mt)) {
      mt->aux_usage = ISL_AUX_USAGE_HIZ;
   }

   /* We can do fast-clear on all auxiliary surface types that are
    * allocated through the normal texture creation paths.
    */
   if (mt->aux_usage != ISL_AUX_USAGE_NONE)
      mt->supports_fast_clear = true;
}


/**
 * Choose an appropriate uncompressed format for a requested
 * compressed format, if unsupported.
 */
mesa_format
intel_lower_compressed_format(struct brw_context *brw, mesa_format format)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;

   /* No need to lower ETC formats on these platforms,
    * they are supported natively.
    */
   if (devinfo->gen >= 8 || devinfo->is_baytrail)
      return format;

   switch (format) {
   case MESA_FORMAT_ETC1_RGB8:
      return MESA_FORMAT_R8G8B8X8_UNORM;
   case MESA_FORMAT_ETC2_RGB8:
      return MESA_FORMAT_R8G8B8X8_UNORM;
   case MESA_FORMAT_ETC2_SRGB8:
   case MESA_FORMAT_ETC2_SRGB8_ALPHA8_EAC:
   case MESA_FORMAT_ETC2_SRGB8_PUNCHTHROUGH_ALPHA1:
      return MESA_FORMAT_B8G8R8A8_SRGB;
   case MESA_FORMAT_ETC2_RGBA8_EAC:
   case MESA_FORMAT_ETC2_RGB8_PUNCHTHROUGH_ALPHA1:
      return MESA_FORMAT_R8G8B8A8_UNORM;
   case MESA_FORMAT_ETC2_R11_EAC:
      return MESA_FORMAT_R_UNORM16;
   case MESA_FORMAT_ETC2_SIGNED_R11_EAC:
      return MESA_FORMAT_R_SNORM16;
   case MESA_FORMAT_ETC2_RG11_EAC:
      return MESA_FORMAT_R16G16_UNORM;
   case MESA_FORMAT_ETC2_SIGNED_RG11_EAC:
      return MESA_FORMAT_R16G16_SNORM;
   default:
      /* Non ETC1 / ETC2 format */
      return format;
   }
}

unsigned
brw_get_num_logical_layers(const struct intel_mipmap_tree *mt, unsigned level)
{
   if (mt->surf.dim == ISL_SURF_DIM_3D)
      return minify(mt->surf.logical_level0_px.depth, level);
   else
      return mt->surf.logical_level0_px.array_len;
}

UNUSED static unsigned
get_num_phys_layers(const struct isl_surf *surf, unsigned level)
{
   /* In case of physical dimensions one needs to consider also the layout.
    * See isl_calc_phys_level0_extent_sa().
    */
   if (surf->dim != ISL_SURF_DIM_3D)
      return surf->phys_level0_sa.array_len;

   if (surf->dim_layout == ISL_DIM_LAYOUT_GEN4_2D)
      return minify(surf->phys_level0_sa.array_len, level);

   return minify(surf->phys_level0_sa.depth, level);
}

/** \brief Assert that the level and layer are valid for the miptree. */
void
intel_miptree_check_level_layer(const struct intel_mipmap_tree *mt,
                                uint32_t level,
                                uint32_t layer)
{
   (void) mt;
   (void) level;
   (void) layer;

   assert(level >= mt->first_level);
   assert(level <= mt->last_level);
   assert(layer < get_num_phys_layers(&mt->surf, level));
}

static enum isl_aux_state **
create_aux_state_map(struct intel_mipmap_tree *mt,
                     enum isl_aux_state initial)
{
   const uint32_t levels = mt->last_level + 1;

   uint32_t total_slices = 0;
   for (uint32_t level = 0; level < levels; level++)
      total_slices += brw_get_num_logical_layers(mt, level);

   const size_t per_level_array_size = levels * sizeof(enum isl_aux_state *);

   /* We're going to allocate a single chunk of data for both the per-level
    * reference array and the arrays of aux_state.  This makes cleanup
    * significantly easier.
    */
   const size_t total_size = per_level_array_size +
                             total_slices * sizeof(enum isl_aux_state);
   void *data = malloc(total_size);
   if (data == NULL)
      return NULL;

   enum isl_aux_state **per_level_arr = data;
   enum isl_aux_state *s = data + per_level_array_size;
   for (uint32_t level = 0; level < levels; level++) {
      per_level_arr[level] = s;
      const unsigned level_layers = brw_get_num_logical_layers(mt, level);
      for (uint32_t a = 0; a < level_layers; a++)
         *(s++) = initial;
   }
   assert((void *)s == data + total_size);

   return per_level_arr;
}

static void
free_aux_state_map(enum isl_aux_state **state)
{
   free(state);
}

static bool
need_to_retile_as_linear(struct brw_context *brw, unsigned row_pitch,
                         enum isl_tiling tiling, unsigned samples)
{
   if (samples > 1)
      return false;

   if (tiling == ISL_TILING_LINEAR)
      return false;

    /* If the width is much smaller than a tile, don't bother tiling. */
   if (row_pitch < 64)
      return true;

   if (ALIGN(row_pitch, 512) >= 32768) {
      perf_debug("row pitch %u too large to blit, falling back to untiled",
                 row_pitch);
      return true;
   }

   return false;
}

static bool
need_to_retile_as_x(const struct brw_context *brw, uint64_t size,
                    enum isl_tiling tiling)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;

   /* If the BO is too large to fit in the aperture, we need to use the
    * BLT engine to support it.  Prior to Sandybridge, the BLT paths can't
    * handle Y-tiling, so we need to fall back to X.
    */
   if (devinfo->gen < 6 && size >= brw->max_gtt_map_object_size &&
       tiling == ISL_TILING_Y0)
      return true;

   return false;
}

static struct intel_mipmap_tree *
make_surface(struct brw_context *brw, GLenum target, mesa_format format,
             unsigned first_level, unsigned last_level,
             unsigned width0, unsigned height0, unsigned depth0,
             unsigned num_samples, isl_tiling_flags_t tiling_flags,
             isl_surf_usage_flags_t isl_usage_flags, uint32_t alloc_flags,
             unsigned row_pitch, struct brw_bo *bo)
{
   struct intel_mipmap_tree *mt = calloc(sizeof(*mt), 1);
   if (!mt)
      return NULL;

   if (!create_mapping_table(target, first_level, last_level, depth0,
                             mt->level)) {
      free(mt);
      return NULL;
   }

   mt->refcount = 1;

   if (target == GL_TEXTURE_CUBE_MAP ||
       target == GL_TEXTURE_CUBE_MAP_ARRAY)
      isl_usage_flags |= ISL_SURF_USAGE_CUBE_BIT;

   DBG("%s: %s %s %ux %u:%u:%u %d..%d <-- %p\n",
        __func__,
       _mesa_enum_to_string(target),
       _mesa_get_format_name(format),
       num_samples, width0, height0, depth0,
       first_level, last_level, mt);

   struct isl_surf_init_info init_info = {
      .dim = get_isl_surf_dim(target),
      .format = translate_tex_format(brw, format, false),
      .width = width0,
      .height = height0,
      .depth = target == GL_TEXTURE_3D ? depth0 : 1,
      .levels = last_level - first_level + 1,
      .array_len = target == GL_TEXTURE_3D ? 1 : depth0,
      .samples = num_samples,
      .row_pitch = row_pitch,
      .usage = isl_usage_flags,
      .tiling_flags = tiling_flags,
   };

   if (!isl_surf_init_s(&brw->isl_dev, &mt->surf, &init_info))
      goto fail;

   /* Depth surfaces are always Y-tiled and stencil is always W-tiled, although
    * on gen7 platforms we also need to create Y-tiled copies of stencil for
    * texturing since the hardware can't sample from W-tiled surfaces. For
    * everything else, check for corner cases needing special treatment.
    */
   bool is_depth_stencil =
      mt->surf.usage & (ISL_SURF_USAGE_STENCIL_BIT | ISL_SURF_USAGE_DEPTH_BIT);
   if (!is_depth_stencil) {
      if (need_to_retile_as_linear(brw, mt->surf.row_pitch,
                                   mt->surf.tiling, mt->surf.samples)) {
         init_info.tiling_flags = 1u << ISL_TILING_LINEAR;
         if (!isl_surf_init_s(&brw->isl_dev, &mt->surf, &init_info))
            goto fail;
      } else if (need_to_retile_as_x(brw, mt->surf.size, mt->surf.tiling)) {
         init_info.tiling_flags = 1u << ISL_TILING_X;
         if (!isl_surf_init_s(&brw->isl_dev, &mt->surf, &init_info))
            goto fail;
      }
   }

   /* In case of linear the buffer gets padded by fixed 64 bytes and therefore
    * the size may not be multiple of row_pitch.
    * See isl_apply_surface_padding().
    */
   if (mt->surf.tiling != ISL_TILING_LINEAR)
      assert(mt->surf.size % mt->surf.row_pitch == 0);

   if (!bo) {
      mt->bo = brw_bo_alloc_tiled(brw->bufmgr, "isl-miptree",
                                  mt->surf.size,
                                  BRW_MEMZONE_OTHER,
                                  isl_tiling_to_i915_tiling(
                                     mt->surf.tiling),
                                  mt->surf.row_pitch, alloc_flags);
      if (!mt->bo)
         goto fail;
   } else {
      mt->bo = bo;
   }

   mt->first_level = first_level;
   mt->last_level = last_level;
   mt->target = target;
   mt->format = format;
   mt->aux_state = NULL;
   mt->cpp = isl_format_get_layout(mt->surf.format)->bpb / 8;
   mt->compressed = _mesa_is_format_compressed(format);
   mt->drm_modifier = DRM_FORMAT_MOD_INVALID;

   return mt;

fail:
   intel_miptree_release(&mt);
   return NULL;
}

static bool
make_separate_stencil_surface(struct brw_context *brw,
                              struct intel_mipmap_tree *mt)
{
   mt->stencil_mt = make_surface(brw, mt->target, MESA_FORMAT_S_UINT8,
                                 0, mt->surf.levels - 1,
                                 mt->surf.logical_level0_px.width,
                                 mt->surf.logical_level0_px.height,
                                 mt->surf.dim == ISL_SURF_DIM_3D ?
                                    mt->surf.logical_level0_px.depth :
                                    mt->surf.logical_level0_px.array_len,
                                 mt->surf.samples, ISL_TILING_W_BIT,
                                 ISL_SURF_USAGE_STENCIL_BIT |
                                 ISL_SURF_USAGE_TEXTURE_BIT,
                                 BO_ALLOC_BUSY, 0, NULL);

   if (!mt->stencil_mt)
      return false;

   mt->stencil_mt->r8stencil_needs_update = true;

   return true;
}

static struct intel_mipmap_tree *
miptree_create(struct brw_context *brw,
               GLenum target,
               mesa_format format,
               GLuint first_level,
               GLuint last_level,
               GLuint width0,
               GLuint height0,
               GLuint depth0,
               GLuint num_samples,
               enum intel_miptree_create_flags flags)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;

   if (format == MESA_FORMAT_S_UINT8)
      return make_surface(brw, target, format, first_level, last_level,
                          width0, height0, depth0, num_samples,
                          ISL_TILING_W_BIT,
                          ISL_SURF_USAGE_STENCIL_BIT |
                          ISL_SURF_USAGE_TEXTURE_BIT,
                          BO_ALLOC_BUSY,
                          0,
                          NULL);

   const GLenum base_format = _mesa_get_format_base_format(format);
   if ((base_format == GL_DEPTH_COMPONENT ||
        base_format == GL_DEPTH_STENCIL) &&
       !(flags & MIPTREE_CREATE_LINEAR)) {
      /* Fix up the Z miptree format for how we're splitting out separate
       * stencil.  Gen7 expects there to be no stencil bits in its depth buffer.
       */
      const mesa_format depth_only_format =
         intel_depth_format_for_depthstencil_format(format);
      struct intel_mipmap_tree *mt = make_surface(
         brw, target, devinfo->gen >= 6 ? depth_only_format : format,
         first_level, last_level,
         width0, height0, depth0, num_samples, ISL_TILING_Y0_BIT,
         ISL_SURF_USAGE_DEPTH_BIT | ISL_SURF_USAGE_TEXTURE_BIT,
         BO_ALLOC_BUSY, 0, NULL);

      if (!mt)
         return NULL;

      if (needs_separate_stencil(brw, mt, format) &&
          !make_separate_stencil_surface(brw, mt)) {
         intel_miptree_release(&mt);
         return NULL;
      }

      if (!(flags & MIPTREE_CREATE_NO_AUX))
         intel_miptree_choose_aux_usage(brw, mt);

      return mt;
   }

   mesa_format tex_format = format;
   mesa_format etc_format = MESA_FORMAT_NONE;
   uint32_t alloc_flags = 0;

   format = intel_lower_compressed_format(brw, format);

   etc_format = (format != tex_format) ? tex_format : MESA_FORMAT_NONE;

   if (flags & MIPTREE_CREATE_BUSY)
      alloc_flags |= BO_ALLOC_BUSY;

   isl_tiling_flags_t tiling_flags = (flags & MIPTREE_CREATE_LINEAR) ?
      ISL_TILING_LINEAR_BIT : ISL_TILING_ANY_MASK;

   /* TODO: This used to be because there wasn't BLORP to handle Y-tiling. */
   if (devinfo->gen < 6)
      tiling_flags &= ~ISL_TILING_Y0_BIT;

   struct intel_mipmap_tree *mt = make_surface(
                                     brw, target, format,
                                     first_level, last_level,
                                     width0, height0, depth0,
                                     num_samples, tiling_flags,
                                     ISL_SURF_USAGE_RENDER_TARGET_BIT |
                                     ISL_SURF_USAGE_TEXTURE_BIT,
                                     alloc_flags, 0, NULL);
   if (!mt)
      return NULL;

   mt->etc_format = etc_format;

   if (!(flags & MIPTREE_CREATE_NO_AUX))
      intel_miptree_choose_aux_usage(brw, mt);

   return mt;
}

struct intel_mipmap_tree *
intel_miptree_create(struct brw_context *brw,
                     GLenum target,
                     mesa_format format,
                     GLuint first_level,
                     GLuint last_level,
                     GLuint width0,
                     GLuint height0,
                     GLuint depth0,
                     GLuint num_samples,
                     enum intel_miptree_create_flags flags)
{
   assert(num_samples > 0);

   struct intel_mipmap_tree *mt = miptree_create(
                                     brw, target, format,
                                     first_level, last_level,
                                     width0, height0, depth0, num_samples,
                                     flags);
   if (!mt)
      return NULL;

   mt->offset = 0;

   /* Create the auxiliary surface up-front. CCS_D, on the other hand, can only
    * compress clear color so we wait until an actual fast-clear to allocate
    * it.
    */
   if (mt->aux_usage != ISL_AUX_USAGE_CCS_D &&
       !intel_miptree_alloc_aux(brw, mt)) {
      intel_miptree_release(&mt);
      return NULL;
   }

   return mt;
}

struct intel_mipmap_tree *
intel_miptree_create_for_bo(struct brw_context *brw,
                            struct brw_bo *bo,
                            mesa_format format,
                            uint32_t offset,
                            uint32_t width,
                            uint32_t height,
                            uint32_t depth,
                            int pitch,
                            enum isl_tiling tiling,
                            enum intel_miptree_create_flags flags)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;
   struct intel_mipmap_tree *mt;
   const GLenum target = depth > 1 ? GL_TEXTURE_2D_ARRAY : GL_TEXTURE_2D;
   const GLenum base_format = _mesa_get_format_base_format(format);

   if ((base_format == GL_DEPTH_COMPONENT ||
        base_format == GL_DEPTH_STENCIL)) {
      const mesa_format depth_only_format =
         intel_depth_format_for_depthstencil_format(format);
      mt = make_surface(brw, target,
                        devinfo->gen >= 6 ? depth_only_format : format,
                        0, 0, width, height, depth, 1, ISL_TILING_Y0_BIT,
                        ISL_SURF_USAGE_DEPTH_BIT | ISL_SURF_USAGE_TEXTURE_BIT,
                        0, pitch, bo);
      if (!mt)
         return NULL;

      brw_bo_reference(bo);

      if (!(flags & MIPTREE_CREATE_NO_AUX))
         intel_miptree_choose_aux_usage(brw, mt);

      return mt;
   } else if (format == MESA_FORMAT_S_UINT8) {
      mt = make_surface(brw, target, MESA_FORMAT_S_UINT8,
                        0, 0, width, height, depth, 1,
                        ISL_TILING_W_BIT,
                        ISL_SURF_USAGE_STENCIL_BIT |
                        ISL_SURF_USAGE_TEXTURE_BIT,
                        0, pitch, bo);
      if (!mt)
         return NULL;

      assert(bo->size >= mt->surf.size);

      brw_bo_reference(bo);
      return mt;
   }

   /* Nothing will be able to use this miptree with the BO if the offset isn't
    * aligned.
    */
   if (tiling != ISL_TILING_LINEAR)
      assert(offset % 4096 == 0);

   /* miptrees can't handle negative pitch.  If you need flipping of images,
    * that's outside of the scope of the mt.
    */
   assert(pitch >= 0);

   /* The BO already has a tiling format and we shouldn't confuse the lower
    * layers by making it try to find a tiling format again.
    */
   assert((flags & MIPTREE_CREATE_LINEAR) == 0);

   mt = make_surface(brw, target, format,
                     0, 0, width, height, depth, 1,
                     1lu << tiling,
                     ISL_SURF_USAGE_RENDER_TARGET_BIT |
                     ISL_SURF_USAGE_TEXTURE_BIT,
                     0, pitch, bo);
   if (!mt)
      return NULL;

   brw_bo_reference(bo);
   mt->bo = bo;
   mt->offset = offset;

   if (!(flags & MIPTREE_CREATE_NO_AUX)) {
      intel_miptree_choose_aux_usage(brw, mt);

      /* Create the auxiliary surface up-front. CCS_D, on the other hand, can
       * only compress clear color so we wait until an actual fast-clear to
       * allocate it.
       */
      if (mt->aux_usage != ISL_AUX_USAGE_CCS_D &&
          !intel_miptree_alloc_aux(brw, mt)) {
         intel_miptree_release(&mt);
         return NULL;
      }
   }

   return mt;
}

static struct intel_mipmap_tree *
miptree_create_for_planar_image(struct brw_context *brw,
                                __DRIimage *image, GLenum target,
                                enum isl_tiling tiling)
{
   const struct intel_image_format *f = image->planar_format;
   struct intel_mipmap_tree *planar_mt = NULL;

   for (int i = 0; i < f->nplanes; i++) {
      const int index = f->planes[i].buffer_index;
      const uint32_t dri_format = f->planes[i].dri_format;
      const mesa_format format = driImageFormatToGLFormat(dri_format);
      const uint32_t width = image->width >> f->planes[i].width_shift;
      const uint32_t height = image->height >> f->planes[i].height_shift;

      /* Disable creation of the texture's aux buffers because the driver
       * exposes no EGL API to manage them. That is, there is no API for
       * resolving the aux buffer's content to the main buffer nor for
       * invalidating the aux buffer's content.
       */
      struct intel_mipmap_tree *mt =
         intel_miptree_create_for_bo(brw, image->bo, format,
                                     image->offsets[index],
                                     width, height, 1,
                                     image->strides[index],
                                     tiling,
                                     MIPTREE_CREATE_NO_AUX);
      if (mt == NULL)
         return NULL;

      mt->target = target;

      if (i == 0)
         planar_mt = mt;
      else
         planar_mt->plane[i - 1] = mt;
   }

   planar_mt->drm_modifier = image->modifier;

   return planar_mt;
}

static bool
create_ccs_buf_for_image(struct brw_context *brw,
                         __DRIimage *image,
                         struct intel_mipmap_tree *mt,
                         enum isl_aux_state initial_state)
{
   struct isl_surf temp_ccs_surf;

   /* CCS is only supported for very simple miptrees */
   assert(image->aux_offset != 0 && image->aux_pitch != 0);
   assert(image->tile_x == 0 && image->tile_y == 0);
   assert(mt->surf.samples == 1);
   assert(mt->surf.levels == 1);
   assert(mt->surf.logical_level0_px.depth == 1);
   assert(mt->surf.logical_level0_px.array_len == 1);
   assert(mt->first_level == 0);
   assert(mt->last_level == 0);

   /* We shouldn't already have a CCS */
   assert(!mt->aux_buf);

   if (!isl_surf_get_ccs_surf(&brw->isl_dev, &mt->surf, &temp_ccs_surf,
                              image->aux_pitch))
      return false;

   assert(image->aux_offset < image->bo->size);
   assert(temp_ccs_surf.size <= image->bo->size - image->aux_offset);

   mt->aux_buf = calloc(sizeof(*mt->aux_buf), 1);
   if (mt->aux_buf == NULL)
      return false;

   mt->aux_state = create_aux_state_map(mt, initial_state);
   if (!mt->aux_state) {
      free(mt->aux_buf);
      mt->aux_buf = NULL;
      return false;
   }

   /* On gen10+ we start using an extra space in the aux buffer to store the
    * indirect clear color. However, if we imported an image from the window
    * system with CCS, we don't have the extra space at the end of the aux
    * buffer. So create a new bo here that will store that clear color.
    */
   if (brw->isl_dev.ss.clear_color_state_size > 0) {
      mt->aux_buf->clear_color_bo =
         brw_bo_alloc_tiled(brw->bufmgr, "clear_color_bo",
                            brw->isl_dev.ss.clear_color_state_size,
                            BRW_MEMZONE_OTHER, I915_TILING_NONE, 0,
                            BO_ALLOC_ZEROED);
      if (!mt->aux_buf->clear_color_bo) {
         free(mt->aux_buf);
         mt->aux_buf = NULL;
         return false;
      }
   }

   mt->aux_buf->bo = image->bo;
   brw_bo_reference(image->bo);

   mt->aux_buf->offset = image->aux_offset;
   mt->aux_buf->surf = temp_ccs_surf;

   return true;
}

struct intel_mipmap_tree *
intel_miptree_create_for_dri_image(struct brw_context *brw,
                                   __DRIimage *image, GLenum target,
                                   mesa_format format,
                                   bool is_winsys_image)
{
   uint32_t bo_tiling, bo_swizzle;
   brw_bo_get_tiling(image->bo, &bo_tiling, &bo_swizzle);

   const struct isl_drm_modifier_info *mod_info =
      isl_drm_modifier_get_info(image->modifier);

   const enum isl_tiling tiling =
      mod_info ? mod_info->tiling : isl_tiling_from_i915_tiling(bo_tiling);

   if (image->planar_format && image->planar_format->nplanes > 1)
      return miptree_create_for_planar_image(brw, image, target, tiling);

   if (image->planar_format)
      assert(image->planar_format->planes[0].dri_format == image->dri_format);

   if (!brw->ctx.TextureFormatSupported[format]) {
      /* The texture storage paths in core Mesa detect if the driver does not
       * support the user-requested format, and then searches for a
       * fallback format. The DRIimage code bypasses core Mesa, though. So we
       * do the fallbacks here for important formats.
       *
       * We must support DRM_FOURCC_XBGR8888 textures because the Android
       * framework produces HAL_PIXEL_FORMAT_RGBX8888 winsys surfaces, which
       * the Chrome OS compositor consumes as dma_buf EGLImages.
       */
      format = _mesa_format_fallback_rgbx_to_rgba(format);
   }

   if (!brw->ctx.TextureFormatSupported[format])
      return NULL;

   enum intel_miptree_create_flags mt_create_flags = 0;

   /* If this image comes in from a window system, we have different
    * requirements than if it comes in via an EGL import operation.  Window
    * system images can use any form of auxiliary compression we wish because
    * they get "flushed" before being handed off to the window system and we
    * have the opportunity to do resolves.  Non window-system images, on the
    * other hand, have no resolve point so we can't have aux without a
    * modifier.
    */
   if (!is_winsys_image)
      mt_create_flags |= MIPTREE_CREATE_NO_AUX;

   /* If we have a modifier which specifies aux, don't create one yet */
   if (mod_info && mod_info->aux_usage != ISL_AUX_USAGE_NONE)
      mt_create_flags |= MIPTREE_CREATE_NO_AUX;

   /* Disable creation of the texture's aux buffers because the driver exposes
    * no EGL API to manage them. That is, there is no API for resolving the aux
    * buffer's content to the main buffer nor for invalidating the aux buffer's
    * content.
    */
   struct intel_mipmap_tree *mt =
      intel_miptree_create_for_bo(brw, image->bo, format,
                                  image->offset, image->width, image->height, 1,
                                  image->pitch, tiling, mt_create_flags);
   if (mt == NULL)
      return NULL;

   mt->target = target;
   mt->level[0].level_x = image->tile_x;
   mt->level[0].level_y = image->tile_y;
   mt->drm_modifier = image->modifier;

   /* From "OES_EGL_image" error reporting. We report GL_INVALID_OPERATION
    * for EGL images from non-tile aligned sufaces in gen4 hw and earlier which has
    * trouble resolving back to destination image due to alignment issues.
    */
   const struct gen_device_info *devinfo = &brw->screen->devinfo;
   if (!devinfo->has_surface_tile_offset) {
      uint32_t draw_x, draw_y;
      intel_miptree_get_tile_offsets(mt, 0, 0, &draw_x, &draw_y);

      if (draw_x != 0 || draw_y != 0) {
         _mesa_error(&brw->ctx, GL_INVALID_OPERATION, __func__);
         intel_miptree_release(&mt);
         return NULL;
      }
   }

   if (mod_info && mod_info->aux_usage != ISL_AUX_USAGE_NONE) {
      assert(mod_info->aux_usage == ISL_AUX_USAGE_CCS_E);

      mt->aux_usage = mod_info->aux_usage;
      /* If we are a window system buffer, then we can support fast-clears
       * even if the modifier doesn't support them by doing a partial resolve
       * as part of the flush operation.
       */
      mt->supports_fast_clear =
         is_winsys_image || mod_info->supports_clear_color;

      /* We don't know the actual state of the surface when we get it but we
       * can make a pretty good guess based on the modifier.  What we do know
       * for sure is that it isn't in the AUX_INVALID state, so we just assume
       * a worst case of compression.
       */
      enum isl_aux_state initial_state =
         isl_drm_modifier_get_default_aux_state(image->modifier);

      if (!create_ccs_buf_for_image(brw, image, mt, initial_state)) {
         intel_miptree_release(&mt);
         return NULL;
      }
   }

   /* Don't assume coherency for imported EGLimages.  We don't know what
    * external clients are going to do with it.  They may scan it out.
    */
   image->bo->cache_coherent = false;

   return mt;
}

/**
 * For a singlesample renderbuffer, this simply wraps the given BO with a
 * miptree.
 *
 * For a multisample renderbuffer, this wraps the window system's
 * (singlesample) BO with a singlesample miptree attached to the
 * intel_renderbuffer, then creates a multisample miptree attached to irb->mt
 * that will contain the actual rendering (which is lazily resolved to
 * irb->singlesample_mt).
 */
bool
intel_update_winsys_renderbuffer_miptree(struct brw_context *intel,
                                         struct intel_renderbuffer *irb,
                                         struct intel_mipmap_tree *singlesample_mt,
                                         uint32_t width, uint32_t height,
                                         uint32_t pitch)
{
   struct intel_mipmap_tree *multisample_mt = NULL;
   struct gl_renderbuffer *rb = &irb->Base.Base;
   mesa_format format = rb->Format;
   const unsigned num_samples = MAX2(rb->NumSamples, 1);

   /* Only the front and back buffers, which are color buffers, are allocated
    * through the image loader.
    */
   assert(_mesa_get_format_base_format(format) == GL_RGB ||
          _mesa_get_format_base_format(format) == GL_RGBA);

   assert(singlesample_mt);

   if (num_samples == 1) {
      intel_miptree_release(&irb->mt);
      irb->mt = singlesample_mt;

      assert(!irb->singlesample_mt);
   } else {
      intel_miptree_release(&irb->singlesample_mt);
      irb->singlesample_mt = singlesample_mt;

      if (!irb->mt ||
          irb->mt->surf.logical_level0_px.width != width ||
          irb->mt->surf.logical_level0_px.height != height) {
         multisample_mt = intel_miptree_create_for_renderbuffer(intel,
                                                                format,
                                                                width,
                                                                height,
                                                                num_samples);
         if (!multisample_mt)
            goto fail;

         irb->need_downsample = false;
         intel_miptree_release(&irb->mt);
         irb->mt = multisample_mt;
      }
   }
   return true;

fail:
   intel_miptree_release(&irb->mt);
   return false;
}

struct intel_mipmap_tree*
intel_miptree_create_for_renderbuffer(struct brw_context *brw,
                                      mesa_format format,
                                      uint32_t width,
                                      uint32_t height,
                                      uint32_t num_samples)
{
   struct intel_mipmap_tree *mt;
   uint32_t depth = 1;
   GLenum target = num_samples > 1 ? GL_TEXTURE_2D_MULTISAMPLE : GL_TEXTURE_2D;

   mt = intel_miptree_create(brw, target, format, 0, 0,
                             width, height, depth, num_samples,
                             MIPTREE_CREATE_BUSY);
   if (!mt)
      goto fail;

   return mt;

fail:
   intel_miptree_release(&mt);
   return NULL;
}

void
intel_miptree_reference(struct intel_mipmap_tree **dst,
                        struct intel_mipmap_tree *src)
{
   if (*dst == src)
      return;

   intel_miptree_release(dst);

   if (src) {
      src->refcount++;
      DBG("%s %p refcount now %d\n", __func__, src, src->refcount);
   }

   *dst = src;
}

static void
intel_miptree_aux_buffer_free(struct intel_miptree_aux_buffer *aux_buf)
{
   if (aux_buf == NULL)
      return;

   brw_bo_unreference(aux_buf->bo);
   brw_bo_unreference(aux_buf->clear_color_bo);

   free(aux_buf);
}

void
intel_miptree_release(struct intel_mipmap_tree **mt)
{
   if (!*mt)
      return;

   DBG("%s %p refcount will be %d\n", __func__, *mt, (*mt)->refcount - 1);
   if (--(*mt)->refcount <= 0) {
      GLuint i;

      DBG("%s deleting %p\n", __func__, *mt);

      brw_bo_unreference((*mt)->bo);
      intel_miptree_release(&(*mt)->stencil_mt);
      intel_miptree_release(&(*mt)->r8stencil_mt);
      intel_miptree_aux_buffer_free((*mt)->aux_buf);
      free_aux_state_map((*mt)->aux_state);

      intel_miptree_release(&(*mt)->plane[0]);
      intel_miptree_release(&(*mt)->plane[1]);

      for (i = 0; i < MAX_TEXTURE_LEVELS; i++) {
	 free((*mt)->level[i].slice);
      }

      free(*mt);
   }
   *mt = NULL;
}


void
intel_get_image_dims(struct gl_texture_image *image,
                     int *width, int *height, int *depth)
{
   switch (image->TexObject->Target) {
   case GL_TEXTURE_1D_ARRAY:
      /* For a 1D Array texture the OpenGL API will treat the image height as
       * the number of array slices. For Intel hardware, we treat the 1D array
       * as a 2D Array with a height of 1. So, here we want to swap image
       * height and depth.
       */
      assert(image->Depth == 1);
      *width = image->Width;
      *height = 1;
      *depth = image->Height;
      break;
   case GL_TEXTURE_CUBE_MAP:
      /* For Cube maps, the mesa/main api layer gives us a depth of 1 even
       * though we really have 6 slices.
       */
      assert(image->Depth == 1);
      *width = image->Width;
      *height = image->Height;
      *depth = 6;
      break;
   default:
      *width = image->Width;
      *height = image->Height;
      *depth = image->Depth;
      break;
   }
}

/**
 * Can the image be pulled into a unified mipmap tree?  This mirrors
 * the completeness test in a lot of ways.
 *
 * Not sure whether I want to pass gl_texture_image here.
 */
bool
intel_miptree_match_image(struct intel_mipmap_tree *mt,
                          struct gl_texture_image *image)
{
   struct intel_texture_image *intelImage = intel_texture_image(image);
   GLuint level = intelImage->base.Base.Level;
   int width, height, depth;

   /* glTexImage* choose the texture object based on the target passed in, and
    * objects can't change targets over their lifetimes, so this should be
    * true.
    */
   assert(image->TexObject->Target == mt->target);

   mesa_format mt_format = mt->format;
   if (mt->format == MESA_FORMAT_Z24_UNORM_X8_UINT && mt->stencil_mt)
      mt_format = MESA_FORMAT_Z24_UNORM_S8_UINT;
   if (mt->format == MESA_FORMAT_Z_FLOAT32 && mt->stencil_mt)
      mt_format = MESA_FORMAT_Z32_FLOAT_S8X24_UINT;
   if (mt->etc_format != MESA_FORMAT_NONE)
      mt_format = mt->etc_format;

   if (_mesa_get_srgb_format_linear(image->TexFormat) !=
       _mesa_get_srgb_format_linear(mt_format))
      return false;

   intel_get_image_dims(image, &width, &height, &depth);

   if (mt->target == GL_TEXTURE_CUBE_MAP)
      depth = 6;

   if (level >= mt->surf.levels)
      return false;

   const unsigned level_depth =
      mt->surf.dim == ISL_SURF_DIM_3D ?
         minify(mt->surf.logical_level0_px.depth, level) :
         mt->surf.logical_level0_px.array_len;

   return width == minify(mt->surf.logical_level0_px.width, level) &&
          height == minify(mt->surf.logical_level0_px.height, level) &&
          depth == level_depth &&
          MAX2(image->NumSamples, 1) == mt->surf.samples;
}

void
intel_miptree_get_image_offset(const struct intel_mipmap_tree *mt,
			       GLuint level, GLuint slice,
			       GLuint *x, GLuint *y)
{
   if (level == 0 && slice == 0) {
      *x = mt->level[0].level_x;
      *y = mt->level[0].level_y;
      return;
   }

   uint32_t x_offset_sa, y_offset_sa;

   /* Miptree itself can have an offset only if it represents a single
    * slice in an imported buffer object.
    * See intel_miptree_create_for_dri_image().
    */
   assert(mt->level[0].level_x == 0);
   assert(mt->level[0].level_y == 0);

   /* Given level is relative to level zero while the miptree may be
    * represent just a subset of all levels starting from 'first_level'.
    */
   assert(level >= mt->first_level);
   level -= mt->first_level;

   const unsigned z = mt->surf.dim == ISL_SURF_DIM_3D ? slice : 0;
   slice = mt->surf.dim == ISL_SURF_DIM_3D ? 0 : slice;
   isl_surf_get_image_offset_el(&mt->surf, level, slice, z,
                                &x_offset_sa, &y_offset_sa);

   *x = x_offset_sa;
   *y = y_offset_sa;
}


/**
 * This function computes the tile_w (in bytes) and tile_h (in rows) of
 * different tiling patterns. If the BO is untiled, tile_w is set to cpp
 * and tile_h is set to 1.
 */
void
intel_get_tile_dims(enum isl_tiling tiling, uint32_t cpp,
                    uint32_t *tile_w, uint32_t *tile_h)
{
   switch (tiling) {
   case ISL_TILING_X:
      *tile_w = 512;
      *tile_h = 8;
      break;
   case ISL_TILING_Y0:
      *tile_w = 128;
      *tile_h = 32;
      break;
   case ISL_TILING_LINEAR:
      *tile_w = cpp;
      *tile_h = 1;
      break;
   default:
      unreachable("not reached");
   }
}


/**
 * This function computes masks that may be used to select the bits of the X
 * and Y coordinates that indicate the offset within a tile.  If the BO is
 * untiled, the masks are set to 0.
 */
void
intel_get_tile_masks(enum isl_tiling tiling, uint32_t cpp,
                     uint32_t *mask_x, uint32_t *mask_y)
{
   uint32_t tile_w_bytes, tile_h;

   intel_get_tile_dims(tiling, cpp, &tile_w_bytes, &tile_h);

   *mask_x = tile_w_bytes / cpp - 1;
   *mask_y = tile_h - 1;
}

/**
 * Compute the offset (in bytes) from the start of the BO to the given x
 * and y coordinate.  For tiled BOs, caller must ensure that x and y are
 * multiples of the tile size.
 */
uint32_t
intel_miptree_get_aligned_offset(const struct intel_mipmap_tree *mt,
                                 uint32_t x, uint32_t y)
{
   int cpp = mt->cpp;
   uint32_t pitch = mt->surf.row_pitch;

   switch (mt->surf.tiling) {
   default:
      unreachable("not reached");
   case ISL_TILING_LINEAR:
      return y * pitch + x * cpp;
   case ISL_TILING_X:
      assert((x % (512 / cpp)) == 0);
      assert((y % 8) == 0);
      return y * pitch + x / (512 / cpp) * 4096;
   case ISL_TILING_Y0:
      assert((x % (128 / cpp)) == 0);
      assert((y % 32) == 0);
      return y * pitch + x / (128 / cpp) * 4096;
   }
}

/**
 * Rendering with tiled buffers requires that the base address of the buffer
 * be aligned to a page boundary.  For renderbuffers, and sometimes with
 * textures, we may want the surface to point at a texture image level that
 * isn't at a page boundary.
 *
 * This function returns an appropriately-aligned base offset
 * according to the tiling restrictions, plus any required x/y offset
 * from there.
 */
uint32_t
intel_miptree_get_tile_offsets(const struct intel_mipmap_tree *mt,
                               GLuint level, GLuint slice,
                               uint32_t *tile_x,
                               uint32_t *tile_y)
{
   uint32_t x, y;
   uint32_t mask_x, mask_y;

   intel_get_tile_masks(mt->surf.tiling, mt->cpp, &mask_x, &mask_y);
   intel_miptree_get_image_offset(mt, level, slice, &x, &y);

   *tile_x = x & mask_x;
   *tile_y = y & mask_y;

   return intel_miptree_get_aligned_offset(mt, x & ~mask_x, y & ~mask_y);
}

static void
intel_miptree_copy_slice_sw(struct brw_context *brw,
                            struct intel_mipmap_tree *src_mt,
                            unsigned src_level, unsigned src_layer,
                            struct intel_mipmap_tree *dst_mt,
                            unsigned dst_level, unsigned dst_layer,
                            unsigned width, unsigned height)
{
   void *src, *dst;
   ptrdiff_t src_stride, dst_stride;
   const unsigned cpp = (isl_format_get_layout(dst_mt->surf.format)->bpb / 8);

   intel_miptree_map(brw, src_mt,
                     src_level, src_layer,
                     0, 0,
                     width, height,
                     GL_MAP_READ_BIT | BRW_MAP_DIRECT_BIT,
                     &src, &src_stride);

   intel_miptree_map(brw, dst_mt,
                     dst_level, dst_layer,
                     0, 0,
                     width, height,
                     GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT |
                     BRW_MAP_DIRECT_BIT,
                     &dst, &dst_stride);

   DBG("sw blit %s mt %p %p/%"PRIdPTR" -> %s mt %p %p/%"PRIdPTR" (%dx%d)\n",
       _mesa_get_format_name(src_mt->format),
       src_mt, src, src_stride,
       _mesa_get_format_name(dst_mt->format),
       dst_mt, dst, dst_stride,
       width, height);

   int row_size = cpp * width;
   if (src_stride == row_size &&
       dst_stride == row_size) {
      memcpy(dst, src, row_size * height);
   } else {
      for (int i = 0; i < height; i++) {
         memcpy(dst, src, row_size);
         dst += dst_stride;
         src += src_stride;
      }
   }

   intel_miptree_unmap(brw, dst_mt, dst_level, dst_layer);
   intel_miptree_unmap(brw, src_mt, src_level, src_layer);

   /* Don't forget to copy the stencil data over, too.  We could have skipped
    * passing BRW_MAP_DIRECT_BIT, but that would have meant intel_miptree_map
    * shuffling the two data sources in/out of temporary storage instead of
    * the direct mapping we get this way.
    */
   if (dst_mt->stencil_mt) {
      assert(src_mt->stencil_mt);
      intel_miptree_copy_slice_sw(brw,
                                  src_mt->stencil_mt, src_level, src_layer,
                                  dst_mt->stencil_mt, dst_level, dst_layer,
                                  width, height);
   }
}

void
intel_miptree_copy_slice(struct brw_context *brw,
                         struct intel_mipmap_tree *src_mt,
                         unsigned src_level, unsigned src_layer,
                         struct intel_mipmap_tree *dst_mt,
                         unsigned dst_level, unsigned dst_layer)

{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;
   mesa_format format = src_mt->format;
   unsigned width = minify(src_mt->surf.phys_level0_sa.width,
                           src_level - src_mt->first_level);
   unsigned height = minify(src_mt->surf.phys_level0_sa.height,
                            src_level - src_mt->first_level);

   assert(src_layer < get_num_phys_layers(&src_mt->surf,
                                          src_level - src_mt->first_level));

   assert(_mesa_get_srgb_format_linear(src_mt->format) ==
          _mesa_get_srgb_format_linear(dst_mt->format));

   DBG("validate blit mt %s %p %d,%d -> mt %s %p %d,%d (%dx%d)\n",
       _mesa_get_format_name(src_mt->format),
       src_mt, src_level, src_layer,
       _mesa_get_format_name(dst_mt->format),
       dst_mt, dst_level, dst_layer,
       width, height);

   if (devinfo->gen >= 6) {
      /* On gen6 and above, we just use blorp.  It's faster than the blitter
       * and can handle everything without software fallbacks.
       */
      brw_blorp_copy_miptrees(brw,
                              src_mt, src_level, src_layer,
                              dst_mt, dst_level, dst_layer,
                              0, 0, 0, 0, width, height);

      if (src_mt->stencil_mt) {
         assert(dst_mt->stencil_mt);
         brw_blorp_copy_miptrees(brw,
                                 src_mt->stencil_mt, src_level, src_layer,
                                 dst_mt->stencil_mt, dst_level, dst_layer,
                                 0, 0, 0, 0, width, height);
      }
      return;
   }

   if (dst_mt->compressed) {
      unsigned int i, j;
      _mesa_get_format_block_size(dst_mt->format, &i, &j);
      height = ALIGN_NPOT(height, j) / j;
      width = ALIGN_NPOT(width, i) / i;
   }

   /* Gen4-5 doesn't support separate stencil */
   assert(!src_mt->stencil_mt);

   uint32_t dst_x, dst_y, src_x, src_y;
   intel_miptree_get_image_offset(dst_mt, dst_level, dst_layer,
                                  &dst_x, &dst_y);
   intel_miptree_get_image_offset(src_mt, src_level, src_layer,
                                  &src_x, &src_y);

   DBG("validate blit mt %s %p %d,%d/%d -> mt %s %p %d,%d/%d (%dx%d)\n",
       _mesa_get_format_name(src_mt->format),
       src_mt, src_x, src_y, src_mt->surf.row_pitch,
       _mesa_get_format_name(dst_mt->format),
       dst_mt, dst_x, dst_y, dst_mt->surf.row_pitch,
       width, height);

   if (!intel_miptree_blit(brw,
                           src_mt, src_level, src_layer, 0, 0, false,
                           dst_mt, dst_level, dst_layer, 0, 0, false,
                           width, height, COLOR_LOGICOP_COPY)) {
      perf_debug("miptree validate blit for %s failed\n",
                 _mesa_get_format_name(format));

      intel_miptree_copy_slice_sw(brw,
                                  src_mt, src_level, src_layer,
                                  dst_mt, dst_level, dst_layer,
                                  width, height);
   }
}

/**
 * Copies the image's current data to the given miptree, and associates that
 * miptree with the image.
 */
void
intel_miptree_copy_teximage(struct brw_context *brw,
			    struct intel_texture_image *intelImage,
			    struct intel_mipmap_tree *dst_mt)
{
   struct intel_mipmap_tree *src_mt = intelImage->mt;
   struct intel_texture_object *intel_obj =
      intel_texture_object(intelImage->base.Base.TexObject);
   int level = intelImage->base.Base.Level;
   const unsigned face = intelImage->base.Base.Face;
   unsigned start_layer, end_layer;

   if (intel_obj->base.Target == GL_TEXTURE_1D_ARRAY) {
      assert(face == 0);
      assert(intelImage->base.Base.Height);
      start_layer = 0;
      end_layer = intelImage->base.Base.Height - 1;
   } else if (face > 0) {
      start_layer = face;
      end_layer = face;
   } else {
      assert(intelImage->base.Base.Depth);
      start_layer = 0;
      end_layer = intelImage->base.Base.Depth - 1;
   }

   for (unsigned i = start_layer; i <= end_layer; i++) {
      intel_miptree_copy_slice(brw,
                               src_mt, level, i,
                               dst_mt, level, i);
   }

   intel_miptree_reference(&intelImage->mt, dst_mt);
   intel_obj->needs_validate = true;
}

static struct intel_miptree_aux_buffer *
intel_alloc_aux_buffer(struct brw_context *brw,
                       const struct isl_surf *aux_surf,
                       bool wants_memset,
                       uint8_t memset_value)
{
   struct intel_miptree_aux_buffer *buf = calloc(sizeof(*buf), 1);
   if (!buf)
      return false;

   uint64_t size = aux_surf->size;

   const bool has_indirect_clear = brw->isl_dev.ss.clear_color_state_size > 0;
   if (has_indirect_clear) {
      /* On CNL+, instead of setting the clear color in the SURFACE_STATE, we
       * will set a pointer to a dword somewhere that contains the color. So,
       * allocate the space for the clear color value here on the aux buffer.
       */
      buf->clear_color_offset = size;
      size += brw->isl_dev.ss.clear_color_state_size;
   }

   /* If the buffer needs to be initialised (requiring the buffer to be
    * immediately mapped to cpu space for writing), do not use the gpu access
    * flag which can cause an unnecessary delay if the backing pages happened
    * to be just used by the GPU.
    */
   const bool alloc_zeroed = wants_memset && memset_value == 0;
   const bool needs_memset =
      !alloc_zeroed && (wants_memset || has_indirect_clear);
   const uint32_t alloc_flags =
      alloc_zeroed ? BO_ALLOC_ZEROED : (needs_memset ? 0 : BO_ALLOC_BUSY);

   /* ISL has stricter set of alignment rules then the drm allocator.
    * Therefore one can pass the ISL dimensions in terms of bytes instead of
    * trying to recalculate based on different format block sizes.
    */
   buf->bo = brw_bo_alloc_tiled(brw->bufmgr, "aux-miptree", size,
                                BRW_MEMZONE_OTHER, I915_TILING_Y,
                                aux_surf->row_pitch, alloc_flags);
   if (!buf->bo) {
      free(buf);
      return NULL;
   }

   /* Initialize the bo to the desired value */
   if (needs_memset) {
      assert(!(alloc_flags & BO_ALLOC_BUSY));

      void *map = brw_bo_map(brw, buf->bo, MAP_WRITE | MAP_RAW);
      if (map == NULL) {
         intel_miptree_aux_buffer_free(buf);
         return NULL;
      }

      /* Memset the aux_surf portion of the BO. */
      if (wants_memset)
         memset(map, memset_value, aux_surf->size);

      /* Zero the indirect clear color to match ::fast_clear_color. */
      if (has_indirect_clear) {
         memset((char *)map + buf->clear_color_offset, 0,
                brw->isl_dev.ss.clear_color_state_size);
      }

      brw_bo_unmap(buf->bo);
   }

   if (has_indirect_clear) {
      buf->clear_color_bo = buf->bo;
      brw_bo_reference(buf->clear_color_bo);
   }

   buf->surf = *aux_surf;

   return buf;
}


/**
 * Helper for intel_miptree_alloc_aux() that sets
 * \c mt->level[level].has_hiz. Return true if and only if
 * \c has_hiz was set.
 */
static bool
intel_miptree_level_enable_hiz(struct brw_context *brw,
                               struct intel_mipmap_tree *mt,
                               uint32_t level)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;

   assert(mt->aux_buf);
   assert(mt->surf.size > 0);

   if (devinfo->gen >= 8 || devinfo->is_haswell) {
      uint32_t width = minify(mt->surf.phys_level0_sa.width, level);
      uint32_t height = minify(mt->surf.phys_level0_sa.height, level);

      /* Disable HiZ for LOD > 0 unless the width is 8 aligned
       * and the height is 4 aligned. This allows our HiZ support
       * to fulfill Haswell restrictions for HiZ ops. For LOD == 0,
       * we can grow the width & height to allow the HiZ op to
       * force the proper size alignments.
       */
      if (level > 0 && ((width & 7) || (height & 3))) {
         DBG("mt %p level %d: HiZ DISABLED\n", mt, level);
         return false;
      }
   }

   DBG("mt %p level %d: HiZ enabled\n", mt, level);
   mt->level[level].has_hiz = true;
   return true;
}


/**
 * Allocate the initial aux surface for a miptree based on mt->aux_usage
 *
 * Since MCS, HiZ, and CCS_E can compress more than just clear color, we
 * create the auxiliary surfaces up-front.  CCS_D, on the other hand, can only
 * compress clear color so we wait until an actual fast-clear to allocate it.
 */
bool
intel_miptree_alloc_aux(struct brw_context *brw,
                        struct intel_mipmap_tree *mt)
{
   assert(mt->aux_buf == NULL);

   /* Get the aux buf allocation parameters for this miptree. */
   enum isl_aux_state initial_state;
   uint8_t memset_value;
   struct isl_surf aux_surf;
   MAYBE_UNUSED bool aux_surf_ok;

   switch (mt->aux_usage) {
   case ISL_AUX_USAGE_NONE:
      aux_surf.size = 0;
      aux_surf_ok = true;
      break;
   case ISL_AUX_USAGE_HIZ:
      initial_state = ISL_AUX_STATE_AUX_INVALID;
      aux_surf_ok = isl_surf_get_hiz_surf(&brw->isl_dev, &mt->surf, &aux_surf);
      break;
   case ISL_AUX_USAGE_MCS:
      /* From the Ivy Bridge PRM, Vol 2 Part 1 p326:
       *
       *     When MCS buffer is enabled and bound to MSRT, it is required that
       *     it is cleared prior to any rendering.
       *
       * Since we don't use the MCS buffer for any purpose other than
       * rendering, it makes sense to just clear it immediately upon
       * allocation.
       *
       * Note: the clear value for MCS buffers is all 1's, so we memset to
       * 0xff.
       */
      initial_state = ISL_AUX_STATE_CLEAR;
      memset_value = 0xFF;
      aux_surf_ok = isl_surf_get_mcs_surf(&brw->isl_dev, &mt->surf, &aux_surf);
      break;
   case ISL_AUX_USAGE_CCS_D:
   case ISL_AUX_USAGE_CCS_E:
      /* When CCS_E is used, we need to ensure that the CCS starts off in a
       * valid state.  From the Sky Lake PRM, "MCS Buffer for Render
       * Target(s)":
       *
       *    "If Software wants to enable Color Compression without Fast
       *    clear, Software needs to initialize MCS with zeros."
       *
       * A CCS value of 0 indicates that the corresponding block is in the
       * pass-through state which is what we want.
       *
       * For CCS_D, do the same thing. On gen9+, this avoids having any
       * undefined bits in the aux buffer.
       */
      initial_state = ISL_AUX_STATE_PASS_THROUGH;
      memset_value = 0;
      aux_surf_ok =
         isl_surf_get_ccs_surf(&brw->isl_dev, &mt->surf, &aux_surf, 0);
      break;
   }

   /* We should have a valid aux_surf. */
   assert(aux_surf_ok);

   /* No work is needed for a zero-sized auxiliary buffer. */
   if (aux_surf.size == 0)
      return true;

   /* Create the aux_state for the auxiliary buffer. */
   mt->aux_state = create_aux_state_map(mt, initial_state);
   if (mt->aux_state == NULL)
      return false;

   /* Allocate the auxiliary buffer. */
   const bool needs_memset = initial_state != ISL_AUX_STATE_AUX_INVALID;
   mt->aux_buf = intel_alloc_aux_buffer(brw, &aux_surf, needs_memset,
                                        memset_value);
   if (mt->aux_buf == NULL) {
      free_aux_state_map(mt->aux_state);
      mt->aux_state = NULL;
      return false;
   }

   /* Perform aux_usage-specific initialization. */
   if (mt->aux_usage == ISL_AUX_USAGE_HIZ) {
      for (unsigned level = mt->first_level; level <= mt->last_level; ++level)
         intel_miptree_level_enable_hiz(brw, mt, level);
   }

   return true;
}


/**
 * Can the miptree sample using the hiz buffer?
 */
bool
intel_miptree_sample_with_hiz(struct brw_context *brw,
                              struct intel_mipmap_tree *mt)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;

   if (!devinfo->has_sample_with_hiz) {
      return false;
   }

   if (!mt->aux_buf) {
      return false;
   }

   /* It seems the hardware won't fallback to the depth buffer if some of the
    * mipmap levels aren't available in the HiZ buffer. So we need all levels
    * of the texture to be HiZ enabled.
    */
   for (unsigned level = 0; level < mt->surf.levels; ++level) {
      if (!intel_miptree_level_has_hiz(mt, level))
         return false;
   }

   /* If compressed multisampling is enabled, then we use it for the auxiliary
    * buffer instead.
    *
    * From the BDW PRM (Volume 2d: Command Reference: Structures
    *                   RENDER_SURFACE_STATE.AuxiliarySurfaceMode):
    *
    *  "If this field is set to AUX_HIZ, Number of Multisamples must be
    *   MULTISAMPLECOUNT_1, and Surface Type cannot be SURFTYPE_3D.
    *
    * There is no such blurb for 1D textures, but there is sufficient evidence
    * that this is broken on SKL+.
    */
   return (mt->surf.samples == 1 &&
           mt->target != GL_TEXTURE_3D &&
           mt->target != GL_TEXTURE_1D /* gen9+ restriction */);
}

/**
 * Does the miptree slice have hiz enabled?
 */
bool
intel_miptree_level_has_hiz(const struct intel_mipmap_tree *mt, uint32_t level)
{
   intel_miptree_check_level_layer(mt, level, 0);
   return mt->level[level].has_hiz;
}

static inline uint32_t
miptree_level_range_length(const struct intel_mipmap_tree *mt,
                           uint32_t start_level, uint32_t num_levels)
{
   assert(start_level >= mt->first_level);
   assert(start_level <= mt->last_level);

   if (num_levels == INTEL_REMAINING_LAYERS)
      num_levels = mt->last_level - start_level + 1;
   /* Check for overflow */
   assert(start_level + num_levels >= start_level);
   assert(start_level + num_levels <= mt->last_level + 1);

   return num_levels;
}

static inline uint32_t
miptree_layer_range_length(const struct intel_mipmap_tree *mt, uint32_t level,
                           uint32_t start_layer, uint32_t num_layers)
{
   assert(level <= mt->last_level);

   const uint32_t total_num_layers = brw_get_num_logical_layers(mt, level);
   assert(start_layer < total_num_layers);
   if (num_layers == INTEL_REMAINING_LAYERS)
      num_layers = total_num_layers - start_layer;
   /* Check for overflow */
   assert(start_layer + num_layers >= start_layer);
   assert(start_layer + num_layers <= total_num_layers);

   return num_layers;
}

bool
intel_miptree_has_color_unresolved(const struct intel_mipmap_tree *mt,
                                   unsigned start_level, unsigned num_levels,
                                   unsigned start_layer, unsigned num_layers)
{
   assert(_mesa_is_format_color_format(mt->format));

   if (!mt->aux_buf)
      return false;

   /* Clamp the level range to fit the miptree */
   num_levels = miptree_level_range_length(mt, start_level, num_levels);

   for (uint32_t l = 0; l < num_levels; l++) {
      const uint32_t level = start_level + l;
      const uint32_t level_layers =
         miptree_layer_range_length(mt, level, start_layer, num_layers);
      for (unsigned a = 0; a < level_layers; a++) {
         enum isl_aux_state aux_state =
            intel_miptree_get_aux_state(mt, level, start_layer + a);
         assert(aux_state != ISL_AUX_STATE_AUX_INVALID);
         if (aux_state != ISL_AUX_STATE_PASS_THROUGH)
            return true;
      }
   }

   return false;
}

static void
intel_miptree_check_color_resolve(const struct brw_context *brw,
                                  const struct intel_mipmap_tree *mt,
                                  unsigned level, unsigned layer)
{
   if (!mt->aux_buf)
      return;

   /* Fast color clear is supported for mipmapped surfaces only on Gen8+. */
   assert(brw->screen->devinfo.gen >= 8 ||
          (level == 0 && mt->first_level == 0 && mt->last_level == 0));

   /* Compression of arrayed msaa surfaces is supported. */
   if (mt->surf.samples > 1)
      return;

   /* Fast color clear is supported for non-msaa arrays only on Gen8+. */
   assert(brw->screen->devinfo.gen >= 8 ||
          (layer == 0 &&
           mt->surf.logical_level0_px.depth == 1 &&
           mt->surf.logical_level0_px.array_len == 1));

   (void)level;
   (void)layer;
}

static enum isl_aux_op
get_ccs_d_resolve_op(enum isl_aux_state aux_state,
                     enum isl_aux_usage aux_usage,
                     bool fast_clear_supported)
{
   assert(aux_usage == ISL_AUX_USAGE_NONE || aux_usage == ISL_AUX_USAGE_CCS_D);

   const bool ccs_supported = aux_usage == ISL_AUX_USAGE_CCS_D;

   assert(ccs_supported == fast_clear_supported);

   switch (aux_state) {
   case ISL_AUX_STATE_CLEAR:
   case ISL_AUX_STATE_PARTIAL_CLEAR:
      if (!ccs_supported)
         return ISL_AUX_OP_FULL_RESOLVE;
      else
         return ISL_AUX_OP_NONE;

   case ISL_AUX_STATE_PASS_THROUGH:
      return ISL_AUX_OP_NONE;

   case ISL_AUX_STATE_RESOLVED:
   case ISL_AUX_STATE_AUX_INVALID:
   case ISL_AUX_STATE_COMPRESSED_CLEAR:
   case ISL_AUX_STATE_COMPRESSED_NO_CLEAR:
      break;
   }

   unreachable("Invalid aux state for CCS_D");
}

static enum isl_aux_op
get_ccs_e_resolve_op(enum isl_aux_state aux_state,
                     enum isl_aux_usage aux_usage,
                     bool fast_clear_supported)
{
   /* CCS_E surfaces can be accessed as CCS_D if we're careful. */
   assert(aux_usage == ISL_AUX_USAGE_NONE ||
          aux_usage == ISL_AUX_USAGE_CCS_D ||
          aux_usage == ISL_AUX_USAGE_CCS_E);

   if (aux_usage == ISL_AUX_USAGE_CCS_D)
      assert(fast_clear_supported);

   switch (aux_state) {
   case ISL_AUX_STATE_CLEAR:
   case ISL_AUX_STATE_PARTIAL_CLEAR:
      if (fast_clear_supported)
         return ISL_AUX_OP_NONE;
      else if (aux_usage == ISL_AUX_USAGE_CCS_E)
         return ISL_AUX_OP_PARTIAL_RESOLVE;
      else
         return ISL_AUX_OP_FULL_RESOLVE;

   case ISL_AUX_STATE_COMPRESSED_CLEAR:
      if (aux_usage != ISL_AUX_USAGE_CCS_E)
         return ISL_AUX_OP_FULL_RESOLVE;
      else if (!fast_clear_supported)
         return ISL_AUX_OP_PARTIAL_RESOLVE;
      else
         return ISL_AUX_OP_NONE;

   case ISL_AUX_STATE_COMPRESSED_NO_CLEAR:
      if (aux_usage != ISL_AUX_USAGE_CCS_E)
         return ISL_AUX_OP_FULL_RESOLVE;
      else
         return ISL_AUX_OP_NONE;

   case ISL_AUX_STATE_PASS_THROUGH:
      return ISL_AUX_OP_NONE;

   case ISL_AUX_STATE_RESOLVED:
   case ISL_AUX_STATE_AUX_INVALID:
      break;
   }

   unreachable("Invalid aux state for CCS_E");
}

static void
intel_miptree_prepare_ccs_access(struct brw_context *brw,
                                 struct intel_mipmap_tree *mt,
                                 uint32_t level, uint32_t layer,
                                 enum isl_aux_usage aux_usage,
                                 bool fast_clear_supported)
{
   enum isl_aux_state aux_state = intel_miptree_get_aux_state(mt, level, layer);

   enum isl_aux_op resolve_op;
   if (mt->aux_usage == ISL_AUX_USAGE_CCS_E) {
      resolve_op = get_ccs_e_resolve_op(aux_state, aux_usage,
                                        fast_clear_supported);
   } else {
      assert(mt->aux_usage == ISL_AUX_USAGE_CCS_D);
      resolve_op = get_ccs_d_resolve_op(aux_state, aux_usage,
                                        fast_clear_supported);
   }

   if (resolve_op != ISL_AUX_OP_NONE) {
      intel_miptree_check_color_resolve(brw, mt, level, layer);
      brw_blorp_resolve_color(brw, mt, level, layer, resolve_op);

      switch (resolve_op) {
      case ISL_AUX_OP_FULL_RESOLVE:
         /* The CCS full resolve operation destroys the CCS and sets it to the
          * pass-through state.  (You can also think of this as being both a
          * resolve and an ambiguate in one operation.)
          */
         intel_miptree_set_aux_state(brw, mt, level, layer, 1,
                                     ISL_AUX_STATE_PASS_THROUGH);
         break;

      case ISL_AUX_OP_PARTIAL_RESOLVE:
         intel_miptree_set_aux_state(brw, mt, level, layer, 1,
                                     ISL_AUX_STATE_COMPRESSED_NO_CLEAR);
         break;

      default:
         unreachable("Invalid resolve op");
      }
   }
}

static void
intel_miptree_finish_ccs_write(struct brw_context *brw,
                               struct intel_mipmap_tree *mt,
                               uint32_t level, uint32_t layer,
                               enum isl_aux_usage aux_usage)
{
   assert(aux_usage == ISL_AUX_USAGE_NONE ||
          aux_usage == ISL_AUX_USAGE_CCS_D ||
          aux_usage == ISL_AUX_USAGE_CCS_E);

   enum isl_aux_state aux_state = intel_miptree_get_aux_state(mt, level, layer);

   if (mt->aux_usage == ISL_AUX_USAGE_CCS_E) {
      switch (aux_state) {
      case ISL_AUX_STATE_CLEAR:
      case ISL_AUX_STATE_PARTIAL_CLEAR:
         assert(aux_usage == ISL_AUX_USAGE_CCS_E ||
                aux_usage == ISL_AUX_USAGE_CCS_D);

         if (aux_usage == ISL_AUX_USAGE_CCS_E) {
            intel_miptree_set_aux_state(brw, mt, level, layer, 1,
                                        ISL_AUX_STATE_COMPRESSED_CLEAR);
         } else if (aux_state != ISL_AUX_STATE_PARTIAL_CLEAR) {
            intel_miptree_set_aux_state(brw, mt, level, layer, 1,
                                        ISL_AUX_STATE_PARTIAL_CLEAR);
         }
         break;

      case ISL_AUX_STATE_COMPRESSED_CLEAR:
      case ISL_AUX_STATE_COMPRESSED_NO_CLEAR:
         assert(aux_usage == ISL_AUX_USAGE_CCS_E);
         break; /* Nothing to do */

      case ISL_AUX_STATE_PASS_THROUGH:
         if (aux_usage == ISL_AUX_USAGE_CCS_E) {
            intel_miptree_set_aux_state(brw, mt, level, layer, 1,
                                        ISL_AUX_STATE_COMPRESSED_NO_CLEAR);
         } else {
            /* Nothing to do */
         }
         break;

      case ISL_AUX_STATE_RESOLVED:
      case ISL_AUX_STATE_AUX_INVALID:
         unreachable("Invalid aux state for CCS_E");
      }
   } else {
      assert(mt->aux_usage == ISL_AUX_USAGE_CCS_D);
      /* CCS_D is a bit simpler */
      switch (aux_state) {
      case ISL_AUX_STATE_CLEAR:
         assert(aux_usage == ISL_AUX_USAGE_CCS_D);
         intel_miptree_set_aux_state(brw, mt, level, layer, 1,
                                     ISL_AUX_STATE_PARTIAL_CLEAR);
         break;

      case ISL_AUX_STATE_PARTIAL_CLEAR:
         assert(aux_usage == ISL_AUX_USAGE_CCS_D);
         break; /* Nothing to do */

      case ISL_AUX_STATE_PASS_THROUGH:
         /* Nothing to do */
         break;

      case ISL_AUX_STATE_COMPRESSED_CLEAR:
      case ISL_AUX_STATE_COMPRESSED_NO_CLEAR:
      case ISL_AUX_STATE_RESOLVED:
      case ISL_AUX_STATE_AUX_INVALID:
         unreachable("Invalid aux state for CCS_D");
      }
   }
}

static void
intel_miptree_prepare_mcs_access(struct brw_context *brw,
                                 struct intel_mipmap_tree *mt,
                                 uint32_t layer,
                                 enum isl_aux_usage aux_usage,
                                 bool fast_clear_supported)
{
   assert(aux_usage == ISL_AUX_USAGE_MCS);

   switch (intel_miptree_get_aux_state(mt, 0, layer)) {
   case ISL_AUX_STATE_CLEAR:
   case ISL_AUX_STATE_COMPRESSED_CLEAR:
      if (!fast_clear_supported) {
         brw_blorp_mcs_partial_resolve(brw, mt, layer, 1);
         intel_miptree_set_aux_state(brw, mt, 0, layer, 1,
                                     ISL_AUX_STATE_COMPRESSED_NO_CLEAR);
      }
      break;

   case ISL_AUX_STATE_COMPRESSED_NO_CLEAR:
      break; /* Nothing to do */

   case ISL_AUX_STATE_RESOLVED:
   case ISL_AUX_STATE_PASS_THROUGH:
   case ISL_AUX_STATE_AUX_INVALID:
   case ISL_AUX_STATE_PARTIAL_CLEAR:
      unreachable("Invalid aux state for MCS");
   }
}

static void
intel_miptree_finish_mcs_write(struct brw_context *brw,
                               struct intel_mipmap_tree *mt,
                               uint32_t layer,
                               enum isl_aux_usage aux_usage)
{
   assert(aux_usage == ISL_AUX_USAGE_MCS);

   switch (intel_miptree_get_aux_state(mt, 0, layer)) {
   case ISL_AUX_STATE_CLEAR:
      intel_miptree_set_aux_state(brw, mt, 0, layer, 1,
                                  ISL_AUX_STATE_COMPRESSED_CLEAR);
      break;

   case ISL_AUX_STATE_COMPRESSED_CLEAR:
   case ISL_AUX_STATE_COMPRESSED_NO_CLEAR:
      break; /* Nothing to do */

   case ISL_AUX_STATE_RESOLVED:
   case ISL_AUX_STATE_PASS_THROUGH:
   case ISL_AUX_STATE_AUX_INVALID:
   case ISL_AUX_STATE_PARTIAL_CLEAR:
      unreachable("Invalid aux state for MCS");
   }
}

static void
intel_miptree_prepare_hiz_access(struct brw_context *brw,
                                 struct intel_mipmap_tree *mt,
                                 uint32_t level, uint32_t layer,
                                 enum isl_aux_usage aux_usage,
                                 bool fast_clear_supported)
{
   assert(aux_usage == ISL_AUX_USAGE_NONE || aux_usage == ISL_AUX_USAGE_HIZ);

   enum isl_aux_op hiz_op = ISL_AUX_OP_NONE;
   switch (intel_miptree_get_aux_state(mt, level, layer)) {
   case ISL_AUX_STATE_CLEAR:
   case ISL_AUX_STATE_COMPRESSED_CLEAR:
      if (aux_usage != ISL_AUX_USAGE_HIZ || !fast_clear_supported)
         hiz_op = ISL_AUX_OP_FULL_RESOLVE;
      break;

   case ISL_AUX_STATE_COMPRESSED_NO_CLEAR:
      if (aux_usage != ISL_AUX_USAGE_HIZ)
         hiz_op = ISL_AUX_OP_FULL_RESOLVE;
      break;

   case ISL_AUX_STATE_PASS_THROUGH:
   case ISL_AUX_STATE_RESOLVED:
      break;

   case ISL_AUX_STATE_AUX_INVALID:
      if (aux_usage == ISL_AUX_USAGE_HIZ)
         hiz_op = ISL_AUX_OP_AMBIGUATE;
      break;

   case ISL_AUX_STATE_PARTIAL_CLEAR:
      unreachable("Invalid HiZ state");
   }

   if (hiz_op != ISL_AUX_OP_NONE) {
      intel_hiz_exec(brw, mt, level, layer, 1, hiz_op);

      switch (hiz_op) {
      case ISL_AUX_OP_FULL_RESOLVE:
         intel_miptree_set_aux_state(brw, mt, level, layer, 1,
                                     ISL_AUX_STATE_RESOLVED);
         break;

      case ISL_AUX_OP_AMBIGUATE:
         /* The HiZ resolve operation is actually an ambiguate */
         intel_miptree_set_aux_state(brw, mt, level, layer, 1,
                                     ISL_AUX_STATE_PASS_THROUGH);
         break;

      default:
         unreachable("Invalid HiZ op");
      }
   }
}

static void
intel_miptree_finish_hiz_write(struct brw_context *brw,
                               struct intel_mipmap_tree *mt,
                               uint32_t level, uint32_t layer,
                               enum isl_aux_usage aux_usage)
{
   assert(aux_usage == ISL_AUX_USAGE_NONE || aux_usage == ISL_AUX_USAGE_HIZ);

   switch (intel_miptree_get_aux_state(mt, level, layer)) {
   case ISL_AUX_STATE_CLEAR:
      assert(aux_usage == ISL_AUX_USAGE_HIZ);
      intel_miptree_set_aux_state(brw, mt, level, layer, 1,
                                  ISL_AUX_STATE_COMPRESSED_CLEAR);
      break;

   case ISL_AUX_STATE_COMPRESSED_NO_CLEAR:
   case ISL_AUX_STATE_COMPRESSED_CLEAR:
      assert(aux_usage == ISL_AUX_USAGE_HIZ);
      break; /* Nothing to do */

   case ISL_AUX_STATE_RESOLVED:
      if (aux_usage == ISL_AUX_USAGE_HIZ) {
         intel_miptree_set_aux_state(brw, mt, level, layer, 1,
                                     ISL_AUX_STATE_COMPRESSED_NO_CLEAR);
      } else {
         intel_miptree_set_aux_state(brw, mt, level, layer, 1,
                                     ISL_AUX_STATE_AUX_INVALID);
      }
      break;

   case ISL_AUX_STATE_PASS_THROUGH:
      if (aux_usage == ISL_AUX_USAGE_HIZ) {
         intel_miptree_set_aux_state(brw, mt, level, layer, 1,
                                     ISL_AUX_STATE_COMPRESSED_NO_CLEAR);
      }
      break;

   case ISL_AUX_STATE_AUX_INVALID:
      assert(aux_usage != ISL_AUX_USAGE_HIZ);
      break;

   case ISL_AUX_STATE_PARTIAL_CLEAR:
      unreachable("Invalid HiZ state");
   }
}

void
intel_miptree_prepare_access(struct brw_context *brw,
                             struct intel_mipmap_tree *mt,
                             uint32_t start_level, uint32_t num_levels,
                             uint32_t start_layer, uint32_t num_layers,
                             enum isl_aux_usage aux_usage,
                             bool fast_clear_supported)
{
   num_levels = miptree_level_range_length(mt, start_level, num_levels);

   switch (mt->aux_usage) {
   case ISL_AUX_USAGE_NONE:
      /* Nothing to do */
      break;

   case ISL_AUX_USAGE_MCS:
      assert(mt->aux_buf);
      assert(start_level == 0 && num_levels == 1);
      const uint32_t level_layers =
         miptree_layer_range_length(mt, 0, start_layer, num_layers);
      for (uint32_t a = 0; a < level_layers; a++) {
         intel_miptree_prepare_mcs_access(brw, mt, start_layer + a,
                                          aux_usage, fast_clear_supported);
      }
      break;

   case ISL_AUX_USAGE_CCS_D:
   case ISL_AUX_USAGE_CCS_E:
      if (!mt->aux_buf)
         return;

      for (uint32_t l = 0; l < num_levels; l++) {
         const uint32_t level = start_level + l;
         const uint32_t level_layers =
            miptree_layer_range_length(mt, level, start_layer, num_layers);
         for (uint32_t a = 0; a < level_layers; a++) {
            intel_miptree_prepare_ccs_access(brw, mt, level,
                                             start_layer + a,
                                             aux_usage, fast_clear_supported);
         }
      }
      break;

   case ISL_AUX_USAGE_HIZ:
      assert(mt->aux_buf);
      for (uint32_t l = 0; l < num_levels; l++) {
         const uint32_t level = start_level + l;
         if (!intel_miptree_level_has_hiz(mt, level))
            continue;

         const uint32_t level_layers =
            miptree_layer_range_length(mt, level, start_layer, num_layers);
         for (uint32_t a = 0; a < level_layers; a++) {
            intel_miptree_prepare_hiz_access(brw, mt, level, start_layer + a,
                                             aux_usage, fast_clear_supported);
         }
      }
      break;

   default:
      unreachable("Invalid aux usage");
   }
}

void
intel_miptree_finish_write(struct brw_context *brw,
                           struct intel_mipmap_tree *mt, uint32_t level,
                           uint32_t start_layer, uint32_t num_layers,
                           enum isl_aux_usage aux_usage)
{
   num_layers = miptree_layer_range_length(mt, level, start_layer, num_layers);

   switch (mt->aux_usage) {
   case ISL_AUX_USAGE_NONE:
      /* Nothing to do */
      break;

   case ISL_AUX_USAGE_MCS:
      assert(mt->aux_buf);
      for (uint32_t a = 0; a < num_layers; a++) {
         intel_miptree_finish_mcs_write(brw, mt, start_layer + a,
                                        aux_usage);
      }
      break;

   case ISL_AUX_USAGE_CCS_D:
   case ISL_AUX_USAGE_CCS_E:
      if (!mt->aux_buf)
         return;

      for (uint32_t a = 0; a < num_layers; a++) {
         intel_miptree_finish_ccs_write(brw, mt, level, start_layer + a,
                                        aux_usage);
      }
      break;

   case ISL_AUX_USAGE_HIZ:
      if (!intel_miptree_level_has_hiz(mt, level))
         return;

      for (uint32_t a = 0; a < num_layers; a++) {
         intel_miptree_finish_hiz_write(brw, mt, level, start_layer + a,
                                        aux_usage);
      }
      break;

   default:
      unreachable("Invavlid aux usage");
   }
}

enum isl_aux_state
intel_miptree_get_aux_state(const struct intel_mipmap_tree *mt,
                            uint32_t level, uint32_t layer)
{
   intel_miptree_check_level_layer(mt, level, layer);

   if (_mesa_is_format_color_format(mt->format)) {
      assert(mt->aux_buf != NULL);
      assert(mt->surf.samples == 1 ||
             mt->surf.msaa_layout == ISL_MSAA_LAYOUT_ARRAY);
   } else if (mt->format == MESA_FORMAT_S_UINT8) {
      unreachable("Cannot get aux state for stencil");
   } else {
      assert(intel_miptree_level_has_hiz(mt, level));
   }

   return mt->aux_state[level][layer];
}

void
intel_miptree_set_aux_state(struct brw_context *brw,
                            struct intel_mipmap_tree *mt, uint32_t level,
                            uint32_t start_layer, uint32_t num_layers,
                            enum isl_aux_state aux_state)
{
   num_layers = miptree_layer_range_length(mt, level, start_layer, num_layers);

   if (_mesa_is_format_color_format(mt->format)) {
      assert(mt->aux_buf != NULL);
      assert(mt->surf.samples == 1 ||
             mt->surf.msaa_layout == ISL_MSAA_LAYOUT_ARRAY);
   } else if (mt->format == MESA_FORMAT_S_UINT8) {
      unreachable("Cannot get aux state for stencil");
   } else {
      assert(intel_miptree_level_has_hiz(mt, level));
   }

   for (unsigned a = 0; a < num_layers; a++) {
      if (mt->aux_state[level][start_layer + a] != aux_state) {
         mt->aux_state[level][start_layer + a] = aux_state;
         brw->ctx.NewDriverState |= BRW_NEW_AUX_STATE;
      }
   }
}

/* On Gen9 color buffers may be compressed by the hardware (lossless
 * compression). There are, however, format restrictions and care needs to be
 * taken that the sampler engine is capable for re-interpreting a buffer with
 * format different the buffer was originally written with.
 *
 * For example, SRGB formats are not compressible and the sampler engine isn't
 * capable of treating RGBA_UNORM as SRGB_ALPHA. In such a case the underlying
 * color buffer needs to be resolved so that the sampling surface can be
 * sampled as non-compressed (i.e., without the auxiliary MCS buffer being
 * set).
 */
static bool
can_texture_with_ccs(struct brw_context *brw,
                     struct intel_mipmap_tree *mt,
                     enum isl_format view_format)
{
   if (mt->aux_usage != ISL_AUX_USAGE_CCS_E)
      return false;

   if (!format_ccs_e_compat_with_miptree(&brw->screen->devinfo,
                                         mt, view_format)) {
      perf_debug("Incompatible sampling format (%s) for rbc (%s)\n",
                 isl_format_get_layout(view_format)->name,
                 _mesa_get_format_name(mt->format));
      return false;
   }

   return true;
}

enum isl_aux_usage
intel_miptree_texture_aux_usage(struct brw_context *brw,
                                struct intel_mipmap_tree *mt,
                                enum isl_format view_format)
{
   switch (mt->aux_usage) {
   case ISL_AUX_USAGE_HIZ:
      if (intel_miptree_sample_with_hiz(brw, mt))
         return ISL_AUX_USAGE_HIZ;
      break;

   case ISL_AUX_USAGE_MCS:
      return ISL_AUX_USAGE_MCS;

   case ISL_AUX_USAGE_CCS_D:
   case ISL_AUX_USAGE_CCS_E:
      if (!mt->aux_buf) {
         assert(mt->aux_usage == ISL_AUX_USAGE_CCS_D);
         return ISL_AUX_USAGE_NONE;
      }

      /* If we don't have any unresolved color, report an aux usage of
       * ISL_AUX_USAGE_NONE.  This way, texturing won't even look at the
       * aux surface and we can save some bandwidth.
       */
      if (!intel_miptree_has_color_unresolved(mt, 0, INTEL_REMAINING_LEVELS,
                                              0, INTEL_REMAINING_LAYERS))
         return ISL_AUX_USAGE_NONE;

      if (can_texture_with_ccs(brw, mt, view_format))
         return ISL_AUX_USAGE_CCS_E;
      break;

   default:
      break;
   }

   return ISL_AUX_USAGE_NONE;
}

static bool
isl_formats_are_fast_clear_compatible(enum isl_format a, enum isl_format b)
{
   /* On gen8 and earlier, the hardware was only capable of handling 0/1 clear
    * values so sRGB curve application was a no-op for all fast-clearable
    * formats.
    *
    * On gen9+, the hardware supports arbitrary clear values.  For sRGB clear
    * values, the hardware interprets the floats, not as what would be
    * returned from the sampler (or written by the shader), but as being
    * between format conversion and sRGB curve application.  This means that
    * we can switch between sRGB and UNORM without having to whack the clear
    * color.
    */
   return isl_format_srgb_to_linear(a) == isl_format_srgb_to_linear(b);
}

void
intel_miptree_prepare_texture(struct brw_context *brw,
                              struct intel_mipmap_tree *mt,
                              enum isl_format view_format,
                              uint32_t start_level, uint32_t num_levels,
                              uint32_t start_layer, uint32_t num_layers)
{
   enum isl_aux_usage aux_usage =
      intel_miptree_texture_aux_usage(brw, mt, view_format);
   bool clear_supported = aux_usage != ISL_AUX_USAGE_NONE;

   /* Clear color is specified as ints or floats and the conversion is done by
    * the sampler.  If we have a texture view, we would have to perform the
    * clear color conversion manually.  Just disable clear color.
    */
   if (!isl_formats_are_fast_clear_compatible(mt->surf.format, view_format))
      clear_supported = false;

   intel_miptree_prepare_access(brw, mt, start_level, num_levels,
                                start_layer, num_layers,
                                aux_usage, clear_supported);
}

void
intel_miptree_prepare_image(struct brw_context *brw,
                            struct intel_mipmap_tree *mt)
{
   /* The data port doesn't understand any compression */
   intel_miptree_prepare_access(brw, mt, 0, INTEL_REMAINING_LEVELS,
                                0, INTEL_REMAINING_LAYERS,
                                ISL_AUX_USAGE_NONE, false);
}

enum isl_aux_usage
intel_miptree_render_aux_usage(struct brw_context *brw,
                               struct intel_mipmap_tree *mt,
                               enum isl_format render_format,
                               bool blend_enabled,
                               bool draw_aux_disabled)
{
   struct gen_device_info *devinfo = &brw->screen->devinfo;

   if (draw_aux_disabled)
      return ISL_AUX_USAGE_NONE;

   switch (mt->aux_usage) {
   case ISL_AUX_USAGE_MCS:
      assert(mt->aux_buf);
      return ISL_AUX_USAGE_MCS;

   case ISL_AUX_USAGE_CCS_D:
   case ISL_AUX_USAGE_CCS_E:
      if (!mt->aux_buf) {
         assert(mt->aux_usage == ISL_AUX_USAGE_CCS_D);
         return ISL_AUX_USAGE_NONE;
      }

      /* gen9+ hardware technically supports non-0/1 clear colors with sRGB
       * formats.  However, there are issues with blending where it doesn't
       * properly apply the sRGB curve to the clear color when blending.
       */
      if (devinfo->gen >= 9 && blend_enabled &&
          isl_format_is_srgb(render_format) &&
          !isl_color_value_is_zero_one(mt->fast_clear_color, render_format))
         return ISL_AUX_USAGE_NONE;

      if (mt->aux_usage == ISL_AUX_USAGE_CCS_E &&
          format_ccs_e_compat_with_miptree(&brw->screen->devinfo,
                                           mt, render_format))
         return ISL_AUX_USAGE_CCS_E;

      /* Otherwise, we have to fall back to CCS_D */
      return ISL_AUX_USAGE_CCS_D;

   default:
      return ISL_AUX_USAGE_NONE;
   }
}

void
intel_miptree_prepare_render(struct brw_context *brw,
                             struct intel_mipmap_tree *mt, uint32_t level,
                             uint32_t start_layer, uint32_t layer_count,
                             enum isl_aux_usage aux_usage)
{
   intel_miptree_prepare_access(brw, mt, level, 1, start_layer, layer_count,
                                aux_usage, aux_usage != ISL_AUX_USAGE_NONE);
}

void
intel_miptree_finish_render(struct brw_context *brw,
                            struct intel_mipmap_tree *mt, uint32_t level,
                            uint32_t start_layer, uint32_t layer_count,
                            enum isl_aux_usage aux_usage)
{
   assert(_mesa_is_format_color_format(mt->format));

   intel_miptree_finish_write(brw, mt, level, start_layer, layer_count,
                              aux_usage);
}

void
intel_miptree_prepare_depth(struct brw_context *brw,
                            struct intel_mipmap_tree *mt, uint32_t level,
                            uint32_t start_layer, uint32_t layer_count)
{
   intel_miptree_prepare_access(brw, mt, level, 1, start_layer, layer_count,
                                mt->aux_usage, mt->aux_buf != NULL);
}

void
intel_miptree_finish_depth(struct brw_context *brw,
                           struct intel_mipmap_tree *mt, uint32_t level,
                           uint32_t start_layer, uint32_t layer_count,
                           bool depth_written)
{
   if (depth_written) {
      intel_miptree_finish_write(brw, mt, level, start_layer, layer_count,
                                 mt->aux_buf != NULL);
   }
}

void
intel_miptree_prepare_external(struct brw_context *brw,
                               struct intel_mipmap_tree *mt)
{
   enum isl_aux_usage aux_usage = ISL_AUX_USAGE_NONE;
   bool supports_fast_clear = false;

   const struct isl_drm_modifier_info *mod_info =
      isl_drm_modifier_get_info(mt->drm_modifier);

   if (mod_info && mod_info->aux_usage != ISL_AUX_USAGE_NONE) {
      /* CCS_E is the only supported aux for external images and it's only
       * supported on very simple images.
       */
      assert(mod_info->aux_usage == ISL_AUX_USAGE_CCS_E);
      assert(_mesa_is_format_color_format(mt->format));
      assert(mt->first_level == 0 && mt->last_level == 0);
      assert(mt->surf.logical_level0_px.depth == 1);
      assert(mt->surf.logical_level0_px.array_len == 1);
      assert(mt->surf.samples == 1);
      assert(mt->aux_buf != NULL);

      aux_usage = mod_info->aux_usage;
      supports_fast_clear = mod_info->supports_clear_color;
   }

   intel_miptree_prepare_access(brw, mt, 0, INTEL_REMAINING_LEVELS,
                                0, INTEL_REMAINING_LAYERS,
                                aux_usage, supports_fast_clear);
}

void
intel_miptree_finish_external(struct brw_context *brw,
                              struct intel_mipmap_tree *mt)
{
   if (!mt->aux_buf)
      return;

   /* We don't know the actual aux state of the aux surface.  The previous
    * owner could have given it to us in a number of different states.
    * Because we don't know the aux state, we reset the aux state to the
    * least common denominator of possible valid states.
    */
   enum isl_aux_state default_aux_state =
      isl_drm_modifier_get_default_aux_state(mt->drm_modifier);
   assert(mt->last_level == mt->first_level);
   intel_miptree_set_aux_state(brw, mt, 0, 0, INTEL_REMAINING_LAYERS,
                               default_aux_state);
}

/**
 * Make it possible to share the BO backing the given miptree with another
 * process or another miptree.
 *
 * Fast color clears are unsafe with shared buffers, so we need to resolve and
 * then discard the MCS buffer, if present.  We also set the no_ccs flag to
 * ensure that no MCS buffer gets allocated in the future.
 *
 * HiZ is similarly unsafe with shared buffers.
 */
void
intel_miptree_make_shareable(struct brw_context *brw,
                             struct intel_mipmap_tree *mt)
{
   /* MCS buffers are also used for multisample buffers, but we can't resolve
    * away a multisample MCS buffer because it's an integral part of how the
    * pixel data is stored.  Fortunately this code path should never be
    * reached for multisample buffers.
    */
   assert(mt->surf.msaa_layout == ISL_MSAA_LAYOUT_NONE ||
          mt->surf.samples == 1);

   intel_miptree_prepare_access(brw, mt, 0, INTEL_REMAINING_LEVELS,
                                0, INTEL_REMAINING_LAYERS,
                                ISL_AUX_USAGE_NONE, false);

   if (mt->aux_buf) {
      intel_miptree_aux_buffer_free(mt->aux_buf);
      mt->aux_buf = NULL;

      /* Make future calls of intel_miptree_level_has_hiz() return false. */
      for (uint32_t l = mt->first_level; l <= mt->last_level; ++l) {
         mt->level[l].has_hiz = false;
      }

      free(mt->aux_state);
      mt->aux_state = NULL;
      brw->ctx.NewDriverState |= BRW_NEW_AUX_STATE;
   }

   mt->aux_usage = ISL_AUX_USAGE_NONE;
   mt->supports_fast_clear = false;
}


/**
 * \brief Get pointer offset into stencil buffer.
 *
 * The stencil buffer is W tiled. Since the GTT is incapable of W fencing, we
 * must decode the tile's layout in software.
 *
 * See
 *   - PRM, 2011 Sandy Bridge, Volume 1, Part 2, Section 4.5.2.1 W-Major Tile
 *     Format.
 *   - PRM, 2011 Sandy Bridge, Volume 1, Part 2, Section 4.5.3 Tiling Algorithm
 *
 * Even though the returned offset is always positive, the return type is
 * signed due to
 *    commit e8b1c6d6f55f5be3bef25084fdd8b6127517e137
 *    mesa: Fix return type of  _mesa_get_format_bytes() (#37351)
 */
static intptr_t
intel_offset_S8(uint32_t stride, uint32_t x, uint32_t y, bool swizzled)
{
   uint32_t tile_size = 4096;
   uint32_t tile_width = 64;
   uint32_t tile_height = 64;
   uint32_t row_size = 64 * stride / 2; /* Two rows are interleaved. */

   uint32_t tile_x = x / tile_width;
   uint32_t tile_y = y / tile_height;

   /* The byte's address relative to the tile's base addres. */
   uint32_t byte_x = x % tile_width;
   uint32_t byte_y = y % tile_height;

   uintptr_t u = tile_y * row_size
               + tile_x * tile_size
               + 512 * (byte_x / 8)
               +  64 * (byte_y / 8)
               +  32 * ((byte_y / 4) % 2)
               +  16 * ((byte_x / 4) % 2)
               +   8 * ((byte_y / 2) % 2)
               +   4 * ((byte_x / 2) % 2)
               +   2 * (byte_y % 2)
               +   1 * (byte_x % 2);

   if (swizzled) {
      /* adjust for bit6 swizzling */
      if (((byte_x / 8) % 2) == 1) {
	 if (((byte_y / 8) % 2) == 0) {
	    u += 64;
	 } else {
	    u -= 64;
	 }
      }
   }

   return u;
}

void
intel_miptree_updownsample(struct brw_context *brw,
                           struct intel_mipmap_tree *src,
                           struct intel_mipmap_tree *dst)
{
   unsigned src_w = src->surf.logical_level0_px.width;
   unsigned src_h = src->surf.logical_level0_px.height;
   unsigned dst_w = dst->surf.logical_level0_px.width;
   unsigned dst_h = dst->surf.logical_level0_px.height;

   brw_blorp_blit_miptrees(brw,
                           src, 0 /* level */, 0 /* layer */,
                           src->format, SWIZZLE_XYZW,
                           dst, 0 /* level */, 0 /* layer */, dst->format,
                           0, 0, src_w, src_h,
                           0, 0, dst_w, dst_h,
                           GL_NEAREST, false, false /*mirror x, y*/,
                           false, false);

   if (src->stencil_mt) {
      src_w = src->stencil_mt->surf.logical_level0_px.width;
      src_h = src->stencil_mt->surf.logical_level0_px.height;
      dst_w = dst->stencil_mt->surf.logical_level0_px.width;
      dst_h = dst->stencil_mt->surf.logical_level0_px.height;

      brw_blorp_blit_miptrees(brw,
                              src->stencil_mt, 0 /* level */, 0 /* layer */,
                              src->stencil_mt->format, SWIZZLE_XYZW,
                              dst->stencil_mt, 0 /* level */, 0 /* layer */,
                              dst->stencil_mt->format,
                              0, 0, src_w, src_h,
                              0, 0, dst_w, dst_h,
                              GL_NEAREST, false, false /*mirror x, y*/,
                              false, false /* decode/encode srgb */);
   }
}

void
intel_update_r8stencil(struct brw_context *brw,
                       struct intel_mipmap_tree *mt)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;

   assert(devinfo->gen >= 7);
   struct intel_mipmap_tree *src =
      mt->format == MESA_FORMAT_S_UINT8 ? mt : mt->stencil_mt;
   if (!src || devinfo->gen >= 8 || !src->r8stencil_needs_update)
      return;

   assert(src->surf.size > 0);

   if (!mt->r8stencil_mt) {
      assert(devinfo->gen > 6); /* Handle MIPTREE_LAYOUT_GEN6_HIZ_STENCIL */
      mt->r8stencil_mt = make_surface(
                            brw,
                            src->target,
                            MESA_FORMAT_R_UINT8,
                            src->first_level, src->last_level,
                            src->surf.logical_level0_px.width,
                            src->surf.logical_level0_px.height,
                            src->surf.dim == ISL_SURF_DIM_3D ?
                               src->surf.logical_level0_px.depth :
                               src->surf.logical_level0_px.array_len,
                            src->surf.samples,
                            ISL_TILING_Y0_BIT,
                            ISL_SURF_USAGE_TEXTURE_BIT,
                            BO_ALLOC_BUSY, 0, NULL);
      assert(mt->r8stencil_mt);
   }

   struct intel_mipmap_tree *dst = mt->r8stencil_mt;

   for (int level = src->first_level; level <= src->last_level; level++) {
      const unsigned depth = src->surf.dim == ISL_SURF_DIM_3D ?
         minify(src->surf.phys_level0_sa.depth, level) :
         src->surf.phys_level0_sa.array_len;

      for (unsigned layer = 0; layer < depth; layer++) {
         brw_blorp_copy_miptrees(brw,
                                 src, level, layer,
                                 dst, level, layer,
                                 0, 0, 0, 0,
                                 minify(src->surf.logical_level0_px.width,
                                        level),
                                 minify(src->surf.logical_level0_px.height,
                                        level));
      }
   }

   brw_cache_flush_for_read(brw, dst->bo);
   src->r8stencil_needs_update = false;
}

static void *
intel_miptree_map_raw(struct brw_context *brw,
                      struct intel_mipmap_tree *mt,
                      GLbitfield mode)
{
   struct brw_bo *bo = mt->bo;

   if (brw_batch_references(&brw->batch, bo))
      intel_batchbuffer_flush(brw);

   return brw_bo_map(brw, bo, mode);
}

static void
intel_miptree_unmap_raw(struct intel_mipmap_tree *mt)
{
   brw_bo_unmap(mt->bo);
}

static void
intel_miptree_unmap_gtt(struct brw_context *brw,
                        struct intel_mipmap_tree *mt,
                        struct intel_miptree_map *map,
                        unsigned int level, unsigned int slice)
{
   intel_miptree_unmap_raw(mt);
}

static void
intel_miptree_map_gtt(struct brw_context *brw,
		      struct intel_mipmap_tree *mt,
		      struct intel_miptree_map *map,
		      unsigned int level, unsigned int slice)
{
   unsigned int bw, bh;
   void *base;
   unsigned int image_x, image_y;
   intptr_t x = map->x;
   intptr_t y = map->y;

   /* For compressed formats, the stride is the number of bytes per
    * row of blocks.  intel_miptree_get_image_offset() already does
    * the divide.
    */
   _mesa_get_format_block_size(mt->format, &bw, &bh);
   assert(y % bh == 0);
   assert(x % bw == 0);
   y /= bh;
   x /= bw;

   intel_miptree_access_raw(brw, mt, level, slice,
                            map->mode & GL_MAP_WRITE_BIT);

   base = intel_miptree_map_raw(brw, mt, map->mode);

   if (base == NULL)
      map->ptr = NULL;
   else {
      base += mt->offset;

      /* Note that in the case of cube maps, the caller must have passed the
       * slice number referencing the face.
      */
      intel_miptree_get_image_offset(mt, level, slice, &image_x, &image_y);
      x += image_x;
      y += image_y;

      map->stride = mt->surf.row_pitch;
      map->ptr = base + y * map->stride + x * mt->cpp;
   }

   DBG("%s: %d,%d %dx%d from mt %p (%s) "
       "%"PRIiPTR",%"PRIiPTR" = %p/%d\n", __func__,
       map->x, map->y, map->w, map->h,
       mt, _mesa_get_format_name(mt->format),
       x, y, map->ptr, map->stride);

   map->unmap = intel_miptree_unmap_gtt;
}

static void
intel_miptree_unmap_blit(struct brw_context *brw,
			 struct intel_mipmap_tree *mt,
			 struct intel_miptree_map *map,
			 unsigned int level,
			 unsigned int slice)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;
   struct gl_context *ctx = &brw->ctx;

   intel_miptree_unmap_raw(map->linear_mt);

   if (map->mode & GL_MAP_WRITE_BIT) {
      if (devinfo->gen >= 6) {
         brw_blorp_copy_miptrees(brw, map->linear_mt, 0, 0,
                                 mt, level, slice,
                                 0, 0, map->x, map->y, map->w, map->h);
      } else {
         bool ok = intel_miptree_copy(brw,
                                      map->linear_mt, 0, 0, 0, 0,
                                      mt, level, slice, map->x, map->y,
                                      map->w, map->h);
         WARN_ONCE(!ok, "Failed to blit from linear temporary mapping");
      }
   }

   intel_miptree_release(&map->linear_mt);
}

static void
intel_miptree_map_blit(struct brw_context *brw,
		       struct intel_mipmap_tree *mt,
		       struct intel_miptree_map *map,
		       unsigned int level, unsigned int slice)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;
   map->linear_mt = intel_miptree_create(brw, GL_TEXTURE_2D, mt->format,
                                         /* first_level */ 0,
                                         /* last_level */ 0,
                                         map->w, map->h, 1,
                                         /* samples */ 1,
                                         MIPTREE_CREATE_LINEAR);

   if (!map->linear_mt) {
      fprintf(stderr, "Failed to allocate blit temporary\n");
      goto fail;
   }
   map->stride = map->linear_mt->surf.row_pitch;

   /* One of either READ_BIT or WRITE_BIT or both is set.  READ_BIT implies no
    * INVALIDATE_RANGE_BIT.  WRITE_BIT needs the original values read in unless
    * invalidate is set, since we'll be writing the whole rectangle from our
    * temporary buffer back out.
    */
   if (!(map->mode & GL_MAP_INVALIDATE_RANGE_BIT)) {
      if (devinfo->gen >= 6) {
         brw_blorp_copy_miptrees(brw, mt, level, slice,
                                 map->linear_mt, 0, 0,
                                 map->x, map->y, 0, 0, map->w, map->h);
      } else {
         if (!intel_miptree_copy(brw,
                                 mt, level, slice, map->x, map->y,
                                 map->linear_mt, 0, 0, 0, 0,
                                 map->w, map->h)) {
            fprintf(stderr, "Failed to blit\n");
            goto fail;
         }
      }
   }

   map->ptr = intel_miptree_map_raw(brw, map->linear_mt, map->mode);

   DBG("%s: %d,%d %dx%d from mt %p (%s) %d,%d = %p/%d\n", __func__,
       map->x, map->y, map->w, map->h,
       mt, _mesa_get_format_name(mt->format),
       level, slice, map->ptr, map->stride);

   map->unmap = intel_miptree_unmap_blit;
   return;

fail:
   intel_miptree_release(&map->linear_mt);
   map->ptr = NULL;
   map->stride = 0;
}

/**
 * "Map" a buffer by copying it to an untiled temporary using MOVNTDQA.
 */
#if defined(USE_SSE41)
static void
intel_miptree_unmap_movntdqa(struct brw_context *brw,
                             struct intel_mipmap_tree *mt,
                             struct intel_miptree_map *map,
                             unsigned int level,
                             unsigned int slice)
{
   _mesa_align_free(map->buffer);
   map->buffer = NULL;
   map->ptr = NULL;
}

static void
intel_miptree_map_movntdqa(struct brw_context *brw,
                           struct intel_mipmap_tree *mt,
                           struct intel_miptree_map *map,
                           unsigned int level, unsigned int slice)
{
   assert(map->mode & GL_MAP_READ_BIT);
   assert(!(map->mode & GL_MAP_WRITE_BIT));

   intel_miptree_access_raw(brw, mt, level, slice, false);

   DBG("%s: %d,%d %dx%d from mt %p (%s) %d,%d = %p/%d\n", __func__,
       map->x, map->y, map->w, map->h,
       mt, _mesa_get_format_name(mt->format),
       level, slice, map->ptr, map->stride);

   /* Map the original image */
   uint32_t image_x;
   uint32_t image_y;
   intel_miptree_get_image_offset(mt, level, slice, &image_x, &image_y);
   image_x += map->x;
   image_y += map->y;

   void *src = intel_miptree_map_raw(brw, mt, map->mode);
   if (!src)
      return;

   src += mt->offset;

   src += image_y * mt->surf.row_pitch;
   src += image_x * mt->cpp;

   /* Due to the pixel offsets for the particular image being mapped, our
    * src pointer may not be 16-byte aligned.  However, if the pitch is
    * divisible by 16, then the amount by which it's misaligned will remain
    * consistent from row to row.
    */
   assert((mt->surf.row_pitch % 16) == 0);
   const int misalignment = ((uintptr_t) src) & 15;

   /* Create an untiled temporary buffer for the mapping. */
   const unsigned width_bytes = _mesa_format_row_stride(mt->format, map->w);

   map->stride = ALIGN(misalignment + width_bytes, 16);

   map->buffer = _mesa_align_malloc(map->stride * map->h, 16);
   /* Offset the destination so it has the same misalignment as src. */
   map->ptr = map->buffer + misalignment;

   assert((((uintptr_t) map->ptr) & 15) == misalignment);

   for (uint32_t y = 0; y < map->h; y++) {
      void *dst_ptr = map->ptr + y * map->stride;
      void *src_ptr = src + y * mt->surf.row_pitch;

      _mesa_streaming_load_memcpy(dst_ptr, src_ptr, width_bytes);
   }

   intel_miptree_unmap_raw(mt);

   map->unmap = intel_miptree_unmap_movntdqa;
}
#endif

static void
intel_miptree_unmap_s8(struct brw_context *brw,
		       struct intel_mipmap_tree *mt,
		       struct intel_miptree_map *map,
		       unsigned int level,
		       unsigned int slice)
{
   if (map->mode & GL_MAP_WRITE_BIT) {
      unsigned int image_x, image_y;
      uint8_t *untiled_s8_map = map->ptr;
      uint8_t *tiled_s8_map = intel_miptree_map_raw(brw, mt, GL_MAP_WRITE_BIT);

      intel_miptree_get_image_offset(mt, level, slice, &image_x, &image_y);

      for (uint32_t y = 0; y < map->h; y++) {
	 for (uint32_t x = 0; x < map->w; x++) {
	    ptrdiff_t offset = intel_offset_S8(mt->surf.row_pitch,
	                                       image_x + x + map->x,
	                                       image_y + y + map->y,
					       brw->has_swizzling);
	    tiled_s8_map[offset] = untiled_s8_map[y * map->w + x];
	 }
      }

      intel_miptree_unmap_raw(mt);
   }

   free(map->buffer);
}

static void
intel_miptree_map_s8(struct brw_context *brw,
		     struct intel_mipmap_tree *mt,
		     struct intel_miptree_map *map,
		     unsigned int level, unsigned int slice)
{
   map->stride = map->w;
   map->buffer = map->ptr = malloc(map->stride * map->h);
   if (!map->buffer)
      return;

   intel_miptree_access_raw(brw, mt, level, slice,
                            map->mode & GL_MAP_WRITE_BIT);

   /* One of either READ_BIT or WRITE_BIT or both is set.  READ_BIT implies no
    * INVALIDATE_RANGE_BIT.  WRITE_BIT needs the original values read in unless
    * invalidate is set, since we'll be writing the whole rectangle from our
    * temporary buffer back out.
    */
   if (!(map->mode & GL_MAP_INVALIDATE_RANGE_BIT)) {
      uint8_t *untiled_s8_map = map->ptr;
      uint8_t *tiled_s8_map = intel_miptree_map_raw(brw, mt, GL_MAP_READ_BIT);
      unsigned int image_x, image_y;

      intel_miptree_get_image_offset(mt, level, slice, &image_x, &image_y);

      for (uint32_t y = 0; y < map->h; y++) {
	 for (uint32_t x = 0; x < map->w; x++) {
	    ptrdiff_t offset = intel_offset_S8(mt->surf.row_pitch,
	                                       x + image_x + map->x,
	                                       y + image_y + map->y,
					       brw->has_swizzling);
	    untiled_s8_map[y * map->w + x] = tiled_s8_map[offset];
	 }
      }

      intel_miptree_unmap_raw(mt);

      DBG("%s: %d,%d %dx%d from mt %p %d,%d = %p/%d\n", __func__,
	  map->x, map->y, map->w, map->h,
	  mt, map->x + image_x, map->y + image_y, map->ptr, map->stride);
   } else {
      DBG("%s: %d,%d %dx%d from mt %p = %p/%d\n", __func__,
	  map->x, map->y, map->w, map->h,
	  mt, map->ptr, map->stride);
   }

   map->unmap = intel_miptree_unmap_s8;
}

static void
intel_miptree_unmap_etc(struct brw_context *brw,
                        struct intel_mipmap_tree *mt,
                        struct intel_miptree_map *map,
                        unsigned int level,
                        unsigned int slice)
{
   uint32_t image_x;
   uint32_t image_y;
   intel_miptree_get_image_offset(mt, level, slice, &image_x, &image_y);

   image_x += map->x;
   image_y += map->y;

   uint8_t *dst = intel_miptree_map_raw(brw, mt, GL_MAP_WRITE_BIT)
                + image_y * mt->surf.row_pitch
                + image_x * mt->cpp;

   if (mt->etc_format == MESA_FORMAT_ETC1_RGB8)
      _mesa_etc1_unpack_rgba8888(dst, mt->surf.row_pitch,
                                 map->ptr, map->stride,
                                 map->w, map->h);
   else
      _mesa_unpack_etc2_format(dst, mt->surf.row_pitch,
                               map->ptr, map->stride,
                               map->w, map->h, mt->etc_format);

   intel_miptree_unmap_raw(mt);
   free(map->buffer);
}

static void
intel_miptree_map_etc(struct brw_context *brw,
                      struct intel_mipmap_tree *mt,
                      struct intel_miptree_map *map,
                      unsigned int level,
                      unsigned int slice)
{
   assert(mt->etc_format != MESA_FORMAT_NONE);
   if (mt->etc_format == MESA_FORMAT_ETC1_RGB8) {
      assert(mt->format == MESA_FORMAT_R8G8B8X8_UNORM);
   }

   assert(map->mode & GL_MAP_WRITE_BIT);
   assert(map->mode & GL_MAP_INVALIDATE_RANGE_BIT);

   intel_miptree_access_raw(brw, mt, level, slice, true);

   map->stride = _mesa_format_row_stride(mt->etc_format, map->w);
   map->buffer = malloc(_mesa_format_image_size(mt->etc_format,
                                                map->w, map->h, 1));
   map->ptr = map->buffer;
   map->unmap = intel_miptree_unmap_etc;
}

/**
 * Mapping functions for packed depth/stencil miptrees backed by real separate
 * miptrees for depth and stencil.
 *
 * On gen7, and to support HiZ pre-gen7, we have to have the stencil buffer
 * separate from the depth buffer.  Yet at the GL API level, we have to expose
 * packed depth/stencil textures and FBO attachments, and Mesa core expects to
 * be able to map that memory for texture storage and glReadPixels-type
 * operations.  We give Mesa core that access by mallocing a temporary and
 * copying the data between the actual backing store and the temporary.
 */
static void
intel_miptree_unmap_depthstencil(struct brw_context *brw,
				 struct intel_mipmap_tree *mt,
				 struct intel_miptree_map *map,
				 unsigned int level,
				 unsigned int slice)
{
   struct intel_mipmap_tree *z_mt = mt;
   struct intel_mipmap_tree *s_mt = mt->stencil_mt;
   bool map_z32f_x24s8 = mt->format == MESA_FORMAT_Z_FLOAT32;

   if (map->mode & GL_MAP_WRITE_BIT) {
      uint32_t *packed_map = map->ptr;
      uint8_t *s_map = intel_miptree_map_raw(brw, s_mt, GL_MAP_WRITE_BIT);
      uint32_t *z_map = intel_miptree_map_raw(brw, z_mt, GL_MAP_WRITE_BIT);
      unsigned int s_image_x, s_image_y;
      unsigned int z_image_x, z_image_y;

      intel_miptree_get_image_offset(s_mt, level, slice,
				     &s_image_x, &s_image_y);
      intel_miptree_get_image_offset(z_mt, level, slice,
				     &z_image_x, &z_image_y);

      for (uint32_t y = 0; y < map->h; y++) {
	 for (uint32_t x = 0; x < map->w; x++) {
	    ptrdiff_t s_offset = intel_offset_S8(s_mt->surf.row_pitch,
						 x + s_image_x + map->x,
						 y + s_image_y + map->y,
						 brw->has_swizzling);
	    ptrdiff_t z_offset = ((y + z_image_y + map->y) *
                                  (z_mt->surf.row_pitch / 4) +
				  (x + z_image_x + map->x));

	    if (map_z32f_x24s8) {
	       z_map[z_offset] = packed_map[(y * map->w + x) * 2 + 0];
	       s_map[s_offset] = packed_map[(y * map->w + x) * 2 + 1];
	    } else {
	       uint32_t packed = packed_map[y * map->w + x];
	       s_map[s_offset] = packed >> 24;
	       z_map[z_offset] = packed;
	    }
	 }
      }

      intel_miptree_unmap_raw(s_mt);
      intel_miptree_unmap_raw(z_mt);

      DBG("%s: %d,%d %dx%d from z mt %p (%s) %d,%d, s mt %p %d,%d = %p/%d\n",
	  __func__,
	  map->x, map->y, map->w, map->h,
	  z_mt, _mesa_get_format_name(z_mt->format),
	  map->x + z_image_x, map->y + z_image_y,
	  s_mt, map->x + s_image_x, map->y + s_image_y,
	  map->ptr, map->stride);
   }

   free(map->buffer);
}

static void
intel_miptree_map_depthstencil(struct brw_context *brw,
			       struct intel_mipmap_tree *mt,
			       struct intel_miptree_map *map,
			       unsigned int level, unsigned int slice)
{
   struct intel_mipmap_tree *z_mt = mt;
   struct intel_mipmap_tree *s_mt = mt->stencil_mt;
   bool map_z32f_x24s8 = mt->format == MESA_FORMAT_Z_FLOAT32;
   int packed_bpp = map_z32f_x24s8 ? 8 : 4;

   map->stride = map->w * packed_bpp;
   map->buffer = map->ptr = malloc(map->stride * map->h);
   if (!map->buffer)
      return;

   intel_miptree_access_raw(brw, mt, level, slice,
                            map->mode & GL_MAP_WRITE_BIT);

   /* One of either READ_BIT or WRITE_BIT or both is set.  READ_BIT implies no
    * INVALIDATE_RANGE_BIT.  WRITE_BIT needs the original values read in unless
    * invalidate is set, since we'll be writing the whole rectangle from our
    * temporary buffer back out.
    */
   if (!(map->mode & GL_MAP_INVALIDATE_RANGE_BIT)) {
      uint32_t *packed_map = map->ptr;
      uint8_t *s_map = intel_miptree_map_raw(brw, s_mt, GL_MAP_READ_BIT);
      uint32_t *z_map = intel_miptree_map_raw(brw, z_mt, GL_MAP_READ_BIT);
      unsigned int s_image_x, s_image_y;
      unsigned int z_image_x, z_image_y;

      intel_miptree_get_image_offset(s_mt, level, slice,
				     &s_image_x, &s_image_y);
      intel_miptree_get_image_offset(z_mt, level, slice,
				     &z_image_x, &z_image_y);

      for (uint32_t y = 0; y < map->h; y++) {
	 for (uint32_t x = 0; x < map->w; x++) {
	    int map_x = map->x + x, map_y = map->y + y;
	    ptrdiff_t s_offset = intel_offset_S8(s_mt->surf.row_pitch,
						 map_x + s_image_x,
						 map_y + s_image_y,
						 brw->has_swizzling);
	    ptrdiff_t z_offset = ((map_y + z_image_y) *
                                  (z_mt->surf.row_pitch / 4) +
				  (map_x + z_image_x));
	    uint8_t s = s_map[s_offset];
	    uint32_t z = z_map[z_offset];

	    if (map_z32f_x24s8) {
	       packed_map[(y * map->w + x) * 2 + 0] = z;
	       packed_map[(y * map->w + x) * 2 + 1] = s;
	    } else {
	       packed_map[y * map->w + x] = (s << 24) | (z & 0x00ffffff);
	    }
	 }
      }

      intel_miptree_unmap_raw(s_mt);
      intel_miptree_unmap_raw(z_mt);

      DBG("%s: %d,%d %dx%d from z mt %p %d,%d, s mt %p %d,%d = %p/%d\n",
	  __func__,
	  map->x, map->y, map->w, map->h,
	  z_mt, map->x + z_image_x, map->y + z_image_y,
	  s_mt, map->x + s_image_x, map->y + s_image_y,
	  map->ptr, map->stride);
   } else {
      DBG("%s: %d,%d %dx%d from mt %p = %p/%d\n", __func__,
	  map->x, map->y, map->w, map->h,
	  mt, map->ptr, map->stride);
   }

   map->unmap = intel_miptree_unmap_depthstencil;
}

/**
 * Create and attach a map to the miptree at (level, slice). Return the
 * attached map.
 */
static struct intel_miptree_map*
intel_miptree_attach_map(struct intel_mipmap_tree *mt,
                         unsigned int level,
                         unsigned int slice,
                         unsigned int x,
                         unsigned int y,
                         unsigned int w,
                         unsigned int h,
                         GLbitfield mode)
{
   struct intel_miptree_map *map = calloc(1, sizeof(*map));

   if (!map)
      return NULL;

   assert(mt->level[level].slice[slice].map == NULL);
   mt->level[level].slice[slice].map = map;

   map->mode = mode;
   map->x = x;
   map->y = y;
   map->w = w;
   map->h = h;

   return map;
}

/**
 * Release the map at (level, slice).
 */
static void
intel_miptree_release_map(struct intel_mipmap_tree *mt,
                         unsigned int level,
                         unsigned int slice)
{
   struct intel_miptree_map **map;

   map = &mt->level[level].slice[slice].map;
   free(*map);
   *map = NULL;
}

static bool
can_blit_slice(struct intel_mipmap_tree *mt,
               unsigned int level, unsigned int slice)
{
   /* See intel_miptree_blit() for details on the 32k pitch limit. */
   if (mt->surf.row_pitch >= 32768)
      return false;

   return true;
}

static bool
use_intel_mipree_map_blit(struct brw_context *brw,
                          struct intel_mipmap_tree *mt,
                          GLbitfield mode,
                          unsigned int level,
                          unsigned int slice)
{
   const struct gen_device_info *devinfo = &brw->screen->devinfo;

   if (devinfo->has_llc &&
      /* It's probably not worth swapping to the blit ring because of
       * all the overhead involved.
       */
       !(mode & GL_MAP_WRITE_BIT) &&
       !mt->compressed &&
       (mt->surf.tiling == ISL_TILING_X ||
        /* Prior to Sandybridge, the blitter can't handle Y tiling */
        (devinfo->gen >= 6 && mt->surf.tiling == ISL_TILING_Y0) ||
        /* Fast copy blit on skl+ supports all tiling formats. */
        devinfo->gen >= 9) &&
       can_blit_slice(mt, level, slice))
      return true;

   if (mt->surf.tiling != ISL_TILING_LINEAR &&
       mt->bo->size >= brw->max_gtt_map_object_size) {
      assert(can_blit_slice(mt, level, slice));
      return true;
   }

   return false;
}

/**
 * Parameter \a out_stride has type ptrdiff_t not because the buffer stride may
 * exceed 32 bits but to diminish the likelihood subtle bugs in pointer
 * arithmetic overflow.
 *
 * If you call this function and use \a out_stride, then you're doing pointer
 * arithmetic on \a out_ptr. The type of \a out_stride doesn't prevent all
 * bugs.  The caller must still take care to avoid 32-bit overflow errors in
 * all arithmetic expressions that contain buffer offsets and pixel sizes,
 * which usually have type uint32_t or GLuint.
 */
void
intel_miptree_map(struct brw_context *brw,
                  struct intel_mipmap_tree *mt,
                  unsigned int level,
                  unsigned int slice,
                  unsigned int x,
                  unsigned int y,
                  unsigned int w,
                  unsigned int h,
                  GLbitfield mode,
                  void **out_ptr,
                  ptrdiff_t *out_stride)
{
   struct intel_miptree_map *map;

   assert(mt->surf.samples == 1);

   map = intel_miptree_attach_map(mt, level, slice, x, y, w, h, mode);
   if (!map){
      *out_ptr = NULL;
      *out_stride = 0;
      return;
   }

   if (mt->format == MESA_FORMAT_S_UINT8) {
      intel_miptree_map_s8(brw, mt, map, level, slice);
   } else if (mt->etc_format != MESA_FORMAT_NONE &&
              !(mode & BRW_MAP_DIRECT_BIT)) {
      intel_miptree_map_etc(brw, mt, map, level, slice);
   } else if (mt->stencil_mt && !(mode & BRW_MAP_DIRECT_BIT)) {
      intel_miptree_map_depthstencil(brw, mt, map, level, slice);
   } else if (use_intel_mipree_map_blit(brw, mt, mode, level, slice)) {
      intel_miptree_map_blit(brw, mt, map, level, slice);
#if defined(USE_SSE41)
   } else if (!(mode & GL_MAP_WRITE_BIT) &&
              !mt->compressed && cpu_has_sse4_1 &&
              (mt->surf.row_pitch % 16 == 0)) {
      intel_miptree_map_movntdqa(brw, mt, map, level, slice);
#endif
   } else {
      intel_miptree_map_gtt(brw, mt, map, level, slice);
   }

   *out_ptr = map->ptr;
   *out_stride = map->stride;

   if (map->ptr == NULL)
      intel_miptree_release_map(mt, level, slice);
}

void
intel_miptree_unmap(struct brw_context *brw,
                    struct intel_mipmap_tree *mt,
                    unsigned int level,
                    unsigned int slice)
{
   struct intel_miptree_map *map = mt->level[level].slice[slice].map;

   assert(mt->surf.samples == 1);

   if (!map)
      return;

   DBG("%s: mt %p (%s) level %d slice %d\n", __func__,
       mt, _mesa_get_format_name(mt->format), level, slice);

   if (map->unmap)
	   map->unmap(brw, mt, map, level, slice);

   intel_miptree_release_map(mt, level, slice);
}

enum isl_surf_dim
get_isl_surf_dim(GLenum target)
{
   switch (target) {
   case GL_TEXTURE_1D:
   case GL_TEXTURE_1D_ARRAY:
      return ISL_SURF_DIM_1D;

   case GL_TEXTURE_2D:
   case GL_TEXTURE_2D_ARRAY:
   case GL_TEXTURE_RECTANGLE:
   case GL_TEXTURE_CUBE_MAP:
   case GL_TEXTURE_CUBE_MAP_ARRAY:
   case GL_TEXTURE_2D_MULTISAMPLE:
   case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
   case GL_TEXTURE_EXTERNAL_OES:
      return ISL_SURF_DIM_2D;

   case GL_TEXTURE_3D:
      return ISL_SURF_DIM_3D;
   }

   unreachable("Invalid texture target");
}

enum isl_dim_layout
get_isl_dim_layout(const struct gen_device_info *devinfo,
                   enum isl_tiling tiling, GLenum target)
{
   switch (target) {
   case GL_TEXTURE_1D:
   case GL_TEXTURE_1D_ARRAY:
      return (devinfo->gen >= 9 && tiling == ISL_TILING_LINEAR ?
              ISL_DIM_LAYOUT_GEN9_1D : ISL_DIM_LAYOUT_GEN4_2D);

   case GL_TEXTURE_2D:
   case GL_TEXTURE_2D_ARRAY:
   case GL_TEXTURE_RECTANGLE:
   case GL_TEXTURE_2D_MULTISAMPLE:
   case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
   case GL_TEXTURE_EXTERNAL_OES:
      return ISL_DIM_LAYOUT_GEN4_2D;

   case GL_TEXTURE_CUBE_MAP:
   case GL_TEXTURE_CUBE_MAP_ARRAY:
      return (devinfo->gen == 4 ? ISL_DIM_LAYOUT_GEN4_3D :
              ISL_DIM_LAYOUT_GEN4_2D);

   case GL_TEXTURE_3D:
      return (devinfo->gen >= 9 ?
              ISL_DIM_LAYOUT_GEN4_2D : ISL_DIM_LAYOUT_GEN4_3D);
   }

   unreachable("Invalid texture target");
}

bool
intel_miptree_set_clear_color(struct brw_context *brw,
                              struct intel_mipmap_tree *mt,
                              union isl_color_value clear_color)
{
   if (memcmp(&mt->fast_clear_color, &clear_color, sizeof(clear_color)) != 0) {
      mt->fast_clear_color = clear_color;
      if (mt->aux_buf->clear_color_bo) {
         /* We can't update the clear color while the hardware is still using
          * the previous one for a resolve or sampling from it. Make sure that
          * there are no pending commands at this point.
          */
         brw_emit_pipe_control_flush(brw, PIPE_CONTROL_CS_STALL);
         for (int i = 0; i < 4; i++) {
            brw_store_data_imm32(brw, mt->aux_buf->clear_color_bo,
                                 mt->aux_buf->clear_color_offset + i * 4,
                                 mt->fast_clear_color.u32[i]);
         }
         brw_emit_pipe_control_flush(brw, PIPE_CONTROL_STATE_CACHE_INVALIDATE);
      }
      brw->ctx.NewDriverState |= BRW_NEW_AUX_STATE;
      return true;
   }
   return false;
}

union isl_color_value
intel_miptree_get_clear_color(const struct gen_device_info *devinfo,
                              const struct intel_mipmap_tree *mt,
                              enum isl_format view_format, bool sampling,
                              struct brw_bo **clear_color_bo,
                              uint32_t *clear_color_offset)
{
   assert(mt->aux_buf);

   if (devinfo->gen == 10 && isl_format_is_srgb(view_format) && sampling) {
      /* The gen10 sampler doesn't gamma-correct the clear color. In this case,
       * we switch to using the inline clear color and do the sRGB color
       * conversion process defined in the OpenGL spec. The red, green, and
       * blue channels take part in gamma correction, while the alpha channel
       * is unchanged.
       */
      union isl_color_value srgb_decoded_value = mt->fast_clear_color;
      for (unsigned i = 0; i < 3; i++) {
         srgb_decoded_value.f32[i] =
            util_format_srgb_to_linear_float(mt->fast_clear_color.f32[i]);
      }
      *clear_color_bo = 0;
      *clear_color_offset = 0;
      return srgb_decoded_value;
   } else {
      *clear_color_bo = mt->aux_buf->clear_color_bo;
      *clear_color_offset = mt->aux_buf->clear_color_offset;
      return mt->fast_clear_color;
   }
}
