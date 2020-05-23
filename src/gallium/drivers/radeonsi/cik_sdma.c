/*
 * Copyright 2010 Jerome Glisse <glisse@freedesktop.org>
 * Copyright 2015 Advanced Micro Devices, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * on the rights to use, copy, modify, merge, publish, distribute, sub
 * license, and/or sell copies of the Software, and to permit persons to whom
 * the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHOR(S) AND/OR THEIR SUPPLIERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "si_pipe.h"
#include "sid.h"

static unsigned minify_as_blocks(unsigned width, unsigned level, unsigned blk_w)
{
   width = u_minify(width, level);
   return DIV_ROUND_UP(width, blk_w);
}

static unsigned encode_tile_info(struct si_context *sctx, struct si_texture *tex, unsigned level,
                                 bool set_bpp)
{
   struct radeon_info *info = &sctx->screen->info;
   unsigned tile_index = tex->surface.u.legacy.tiling_index[level];
   unsigned macro_tile_index = tex->surface.u.legacy.macro_tile_index;
   unsigned tile_mode = info->si_tile_mode_array[tile_index];
   unsigned macro_tile_mode = info->cik_macrotile_mode_array[macro_tile_index];

   return (set_bpp ? util_logbase2(tex->surface.bpe) : 0) | (G_009910_ARRAY_MODE(tile_mode) << 3) |
          (G_009910_MICRO_TILE_MODE_NEW(tile_mode) << 8) |
          /* Non-depth modes don't have TILE_SPLIT set. */
          ((util_logbase2(tex->surface.u.legacy.tile_split >> 6)) << 11) |
          (G_009990_BANK_WIDTH(macro_tile_mode) << 15) |
          (G_009990_BANK_HEIGHT(macro_tile_mode) << 18) |
          (G_009990_NUM_BANKS(macro_tile_mode) << 21) |
          (G_009990_MACRO_TILE_ASPECT(macro_tile_mode) << 24) |
          (G_009910_PIPE_CONFIG(tile_mode) << 26);
}

static bool si_sdma_v4_copy_texture(struct si_context *sctx, struct pipe_resource *dst,
                                    unsigned dst_level, unsigned dstx, unsigned dsty, unsigned dstz,
                                    struct pipe_resource *src, unsigned src_level,
                                    const struct pipe_box *src_box)
{
   struct si_texture *ssrc = (struct si_texture *)src;
   struct si_texture *sdst = (struct si_texture *)dst;

   unsigned bpp = sdst->surface.bpe;
   uint64_t dst_address = sdst->buffer.gpu_address + sdst->surface.u.gfx9.surf_offset;
   uint64_t src_address = ssrc->buffer.gpu_address + ssrc->surface.u.gfx9.surf_offset;
   unsigned dst_pitch = sdst->surface.u.gfx9.surf_pitch;
   unsigned src_pitch = ssrc->surface.u.gfx9.surf_pitch;
   uint64_t dst_slice_pitch = ((uint64_t)sdst->surface.u.gfx9.surf_slice_size) / bpp;
   uint64_t src_slice_pitch = ((uint64_t)ssrc->surface.u.gfx9.surf_slice_size) / bpp;
   unsigned srcx = src_box->x / ssrc->surface.blk_w;
   unsigned srcy = src_box->y / ssrc->surface.blk_h;
   unsigned srcz = src_box->z;
   unsigned copy_width = DIV_ROUND_UP(src_box->width, ssrc->surface.blk_w);
   unsigned copy_height = DIV_ROUND_UP(src_box->height, ssrc->surface.blk_h);
   unsigned copy_depth = src_box->depth;
   unsigned xalign = MAX2(1, 4 / bpp);

   assert(src_level <= src->last_level);
   assert(dst_level <= dst->last_level);
   assert(sdst->surface.u.gfx9.surf_offset + dst_slice_pitch * bpp * (dstz + src_box->depth) <=
          sdst->buffer.buf->size);
   assert(ssrc->surface.u.gfx9.surf_offset + src_slice_pitch * bpp * (srcz + src_box->depth) <=
          ssrc->buffer.buf->size);

   if (!si_prepare_for_dma_blit(sctx, sdst, dst_level, dstx, dsty, dstz, ssrc, src_level, src_box))
      return false;

   dstx /= sdst->surface.blk_w;
   dsty /= sdst->surface.blk_h;

   if (srcx >= (1 << 14) || srcy >= (1 << 14) || srcz >= (1 << 11) || dstx >= (1 << 14) ||
       dsty >= (1 << 14) || dstz >= (1 << 11))
      return false;

   /* Linear -> linear sub-window copy. */
   if (ssrc->surface.is_linear && sdst->surface.is_linear) {
      struct radeon_cmdbuf *cs = sctx->sdma_cs;

      /* Check if everything fits into the bitfields */
      if (!(src_pitch <= (1 << 19) && dst_pitch <= (1 << 19) && src_slice_pitch <= (1 << 28) &&
            dst_slice_pitch <= (1 << 28) && copy_width <= (1 << 14) && copy_height <= (1 << 14) &&
            copy_depth <= (1 << 11)))
         return false;

      si_need_dma_space(sctx, 13, &sdst->buffer, &ssrc->buffer);

      src_address += ssrc->surface.u.gfx9.offset[src_level];
      dst_address += sdst->surface.u.gfx9.offset[dst_level];

      /* Check alignments */
      if ((src_address % 4) != 0 || (dst_address % 4) != 0 || (src_pitch % xalign) != 0)
         return false;

      radeon_emit(
         cs, CIK_SDMA_PACKET(CIK_SDMA_OPCODE_COPY, CIK_SDMA_COPY_SUB_OPCODE_LINEAR_SUB_WINDOW,
                             sctx->ws->cs_is_secure(cs) ? (1u << 2) : 0) |
                             (util_logbase2(bpp) << 29));
      radeon_emit(cs, src_address);
      radeon_emit(cs, src_address >> 32);
      radeon_emit(cs, srcx | (srcy << 16));
      radeon_emit(cs, srcz | ((src_pitch - 1) << 13));
      radeon_emit(cs, src_slice_pitch - 1);
      radeon_emit(cs, dst_address);
      radeon_emit(cs, dst_address >> 32);
      radeon_emit(cs, dstx | (dsty << 16));
      radeon_emit(cs, dstz | ((dst_pitch - 1) << 13));
      radeon_emit(cs, dst_slice_pitch - 1);
      radeon_emit(cs, (copy_width - 1) | ((copy_height - 1) << 16));
      radeon_emit(cs, (copy_depth - 1));
      return true;
   }

   /* Linear <-> Tiled sub-window copy */
   if (ssrc->surface.is_linear != sdst->surface.is_linear) {
      struct si_texture *tiled = ssrc->surface.is_linear ? sdst : ssrc;
      struct si_texture *linear = tiled == ssrc ? sdst : ssrc;
      unsigned tiled_level = tiled == ssrc ? src_level : dst_level;
      unsigned linear_level = linear == ssrc ? src_level : dst_level;
      unsigned tiled_x = tiled == ssrc ? srcx : dstx;
      unsigned linear_x = linear == ssrc ? srcx : dstx;
      unsigned tiled_y = tiled == ssrc ? srcy : dsty;
      unsigned linear_y = linear == ssrc ? srcy : dsty;
      unsigned tiled_z = tiled == ssrc ? srcz : dstz;
      unsigned linear_z = linear == ssrc ? srcz : dstz;
      unsigned tiled_width = tiled == ssrc
                                ? DIV_ROUND_UP(ssrc->buffer.b.b.width0, ssrc->surface.blk_w)
                                : DIV_ROUND_UP(sdst->buffer.b.b.width0, sdst->surface.blk_w);
      unsigned tiled_height = tiled == ssrc
                                 ? DIV_ROUND_UP(ssrc->buffer.b.b.height0, ssrc->surface.blk_h)
                                 : DIV_ROUND_UP(sdst->buffer.b.b.height0, sdst->surface.blk_h);
      unsigned tiled_depth = tiled == ssrc ? ssrc->buffer.b.b.depth0 : sdst->buffer.b.b.depth0;
      unsigned linear_pitch = linear == ssrc ? src_pitch : dst_pitch;
      unsigned linear_slice_pitch = linear == ssrc ? src_slice_pitch : dst_slice_pitch;
      uint64_t tiled_address = tiled == ssrc ? src_address : dst_address;
      uint64_t linear_address = linear == ssrc ? src_address : dst_address;
      struct radeon_cmdbuf *cs = sctx->sdma_cs;

      linear_address += linear->surface.u.gfx9.offset[linear_level];

      /* Check if everything fits into the bitfields */
      if (!(tiled_x <= (1 << 14) && tiled_y <= (1 << 14) && tiled_z <= (1 << 11) &&
            tiled_width <= (1 << 14) && tiled_height <= (1 << 14) && tiled_depth <= (1 << 11) &&
            linear_x <= (1 << 14) && linear_y <= (1 << 14) && linear_z <= (1 << 11) &&
            linear_pitch <= (1 << 14) && linear_slice_pitch <= (1 << 28) &&
            copy_width <= (1 << 14) && copy_height <= (1 << 14) && copy_depth <= (1 << 11)))
         return false;

      /* Check alignments */
      if ((tiled_address % 256 != 0) || (linear_address % 4 != 0) || (linear_pitch % xalign != 0) ||
          (linear_slice_pitch % xalign != 0))
         return false;

      si_need_dma_space(sctx, 14, &sdst->buffer, &ssrc->buffer);

      radeon_emit(
         cs, CIK_SDMA_PACKET(CIK_SDMA_OPCODE_COPY, CIK_SDMA_COPY_SUB_OPCODE_TILED_SUB_WINDOW,
                             sctx->ws->cs_is_secure(cs) ? (1u << 2) : 0) |
             tiled->buffer.b.b.last_level << 20 | tiled_level << 24 |
             (linear == sdst ? 1u : 0) << 31);
      radeon_emit(cs, (uint32_t)tiled_address);
      radeon_emit(cs, (uint32_t)(tiled_address >> 32));
      radeon_emit(cs, tiled_x | (tiled_y << 16));
      radeon_emit(cs, tiled_z | ((tiled_width - 1) << 16));
      radeon_emit(cs, (tiled_height - 1) | (tiled_depth - 1) << 16);
      radeon_emit(cs, util_logbase2(bpp) | tiled->surface.u.gfx9.surf.swizzle_mode << 3 |
                         tiled->surface.u.gfx9.resource_type << 9 |
                         tiled->surface.u.gfx9.surf.epitch << 16);
      radeon_emit(cs, (uint32_t)linear_address);
      radeon_emit(cs, (uint32_t)(linear_address >> 32));
      radeon_emit(cs, linear_x | (linear_y << 16));
      radeon_emit(cs, linear_z | ((linear_pitch - 1) << 16));
      radeon_emit(cs, linear_slice_pitch - 1);
      radeon_emit(cs, (copy_width - 1) | ((copy_height - 1) << 16));
      radeon_emit(cs, (copy_depth - 1));
      return true;
   }

   return false;
}

static bool cik_sdma_copy_texture(struct si_context *sctx, struct pipe_resource *dst,
                                  unsigned dst_level, unsigned dstx, unsigned dsty, unsigned dstz,
                                  struct pipe_resource *src, unsigned src_level,
                                  const struct pipe_box *src_box)
{
   struct radeon_info *info = &sctx->screen->info;
   struct si_texture *ssrc = (struct si_texture *)src;
   struct si_texture *sdst = (struct si_texture *)dst;
   unsigned bpp = sdst->surface.bpe;
   uint64_t dst_address = sdst->buffer.gpu_address + sdst->surface.u.legacy.level[dst_level].offset;
   uint64_t src_address = ssrc->buffer.gpu_address + ssrc->surface.u.legacy.level[src_level].offset;
   unsigned dst_mode = sdst->surface.u.legacy.level[dst_level].mode;
   unsigned src_mode = ssrc->surface.u.legacy.level[src_level].mode;
   unsigned dst_tile_index = sdst->surface.u.legacy.tiling_index[dst_level];
   unsigned src_tile_index = ssrc->surface.u.legacy.tiling_index[src_level];
   unsigned dst_tile_mode = info->si_tile_mode_array[dst_tile_index];
   unsigned src_tile_mode = info->si_tile_mode_array[src_tile_index];
   unsigned dst_micro_mode = G_009910_MICRO_TILE_MODE_NEW(dst_tile_mode);
   unsigned src_micro_mode = G_009910_MICRO_TILE_MODE_NEW(src_tile_mode);
   unsigned dst_tile_swizzle = dst_mode == RADEON_SURF_MODE_2D ? sdst->surface.tile_swizzle : 0;
   unsigned src_tile_swizzle = src_mode == RADEON_SURF_MODE_2D ? ssrc->surface.tile_swizzle : 0;
   unsigned dst_pitch = sdst->surface.u.legacy.level[dst_level].nblk_x;
   unsigned src_pitch = ssrc->surface.u.legacy.level[src_level].nblk_x;
   uint64_t dst_slice_pitch =
      ((uint64_t)sdst->surface.u.legacy.level[dst_level].slice_size_dw * 4) / bpp;
   uint64_t src_slice_pitch =
      ((uint64_t)ssrc->surface.u.legacy.level[src_level].slice_size_dw * 4) / bpp;
   unsigned dst_width = minify_as_blocks(sdst->buffer.b.b.width0, dst_level, sdst->surface.blk_w);
   unsigned src_width = minify_as_blocks(ssrc->buffer.b.b.width0, src_level, ssrc->surface.blk_w);
   unsigned dst_height = minify_as_blocks(sdst->buffer.b.b.height0, dst_level, sdst->surface.blk_h);
   unsigned src_height = minify_as_blocks(ssrc->buffer.b.b.height0, src_level, ssrc->surface.blk_h);
   unsigned srcx = src_box->x / ssrc->surface.blk_w;
   unsigned srcy = src_box->y / ssrc->surface.blk_h;
   unsigned srcz = src_box->z;
   unsigned copy_width = DIV_ROUND_UP(src_box->width, ssrc->surface.blk_w);
   unsigned copy_height = DIV_ROUND_UP(src_box->height, ssrc->surface.blk_h);
   unsigned copy_depth = src_box->depth;

   assert(src_level <= src->last_level);
   assert(dst_level <= dst->last_level);
   assert(sdst->surface.u.legacy.level[dst_level].offset +
             dst_slice_pitch * bpp * (dstz + src_box->depth) <=
          sdst->buffer.buf->size);
   assert(ssrc->surface.u.legacy.level[src_level].offset +
             src_slice_pitch * bpp * (srcz + src_box->depth) <=
          ssrc->buffer.buf->size);

   if (!si_prepare_for_dma_blit(sctx, sdst, dst_level, dstx, dsty, dstz, ssrc, src_level, src_box))
      return false;

   dstx /= sdst->surface.blk_w;
   dsty /= sdst->surface.blk_h;

   if (srcx >= (1 << 14) || srcy >= (1 << 14) || srcz >= (1 << 11) || dstx >= (1 << 14) ||
       dsty >= (1 << 14) || dstz >= (1 << 11))
      return false;

   dst_address |= dst_tile_swizzle << 8;
   src_address |= src_tile_swizzle << 8;

   /* Linear -> linear sub-window copy. */
   if (dst_mode == RADEON_SURF_MODE_LINEAR_ALIGNED && src_mode == RADEON_SURF_MODE_LINEAR_ALIGNED &&
       /* check if everything fits into the bitfields */
       src_pitch <= (1 << 14) && dst_pitch <= (1 << 14) && src_slice_pitch <= (1 << 28) &&
       dst_slice_pitch <= (1 << 28) && copy_width <= (1 << 14) && copy_height <= (1 << 14) &&
       copy_depth <= (1 << 11) &&
       /* HW limitation - GFX7: */
       (sctx->chip_class != GFX7 ||
        (copy_width < (1 << 14) && copy_height < (1 << 14) && copy_depth < (1 << 11))) &&
       /* HW limitation - some GFX7 parts: */
       ((sctx->family != CHIP_BONAIRE && sctx->family != CHIP_KAVERI) ||
        (srcx + copy_width != (1 << 14) && srcy + copy_height != (1 << 14)))) {
      struct radeon_cmdbuf *cs = sctx->sdma_cs;

      si_need_dma_space(sctx, 13, &sdst->buffer, &ssrc->buffer);

      radeon_emit(
         cs, CIK_SDMA_PACKET(CIK_SDMA_OPCODE_COPY, CIK_SDMA_COPY_SUB_OPCODE_LINEAR_SUB_WINDOW, 0) |
                (util_logbase2(bpp) << 29));
      radeon_emit(cs, src_address);
      radeon_emit(cs, src_address >> 32);
      radeon_emit(cs, srcx | (srcy << 16));
      radeon_emit(cs, srcz | ((src_pitch - 1) << 16));
      radeon_emit(cs, src_slice_pitch - 1);
      radeon_emit(cs, dst_address);
      radeon_emit(cs, dst_address >> 32);
      radeon_emit(cs, dstx | (dsty << 16));
      radeon_emit(cs, dstz | ((dst_pitch - 1) << 16));
      radeon_emit(cs, dst_slice_pitch - 1);
      if (sctx->chip_class == GFX7) {
         radeon_emit(cs, copy_width | (copy_height << 16));
         radeon_emit(cs, copy_depth);
      } else {
         radeon_emit(cs, (copy_width - 1) | ((copy_height - 1) << 16));
         radeon_emit(cs, (copy_depth - 1));
      }
      return true;
   }

   /* Tiled <-> linear sub-window copy. */
   if ((src_mode >= RADEON_SURF_MODE_1D) != (dst_mode >= RADEON_SURF_MODE_1D)) {
      struct si_texture *tiled = src_mode >= RADEON_SURF_MODE_1D ? ssrc : sdst;
      struct si_texture *linear = tiled == ssrc ? sdst : ssrc;
      unsigned tiled_level = tiled == ssrc ? src_level : dst_level;
      unsigned linear_level = linear == ssrc ? src_level : dst_level;
      unsigned tiled_x = tiled == ssrc ? srcx : dstx;
      unsigned linear_x = linear == ssrc ? srcx : dstx;
      unsigned tiled_y = tiled == ssrc ? srcy : dsty;
      unsigned linear_y = linear == ssrc ? srcy : dsty;
      unsigned tiled_z = tiled == ssrc ? srcz : dstz;
      unsigned linear_z = linear == ssrc ? srcz : dstz;
      unsigned tiled_width = tiled == ssrc ? src_width : dst_width;
      unsigned linear_width = linear == ssrc ? src_width : dst_width;
      unsigned tiled_pitch = tiled == ssrc ? src_pitch : dst_pitch;
      unsigned linear_pitch = linear == ssrc ? src_pitch : dst_pitch;
      unsigned tiled_slice_pitch = tiled == ssrc ? src_slice_pitch : dst_slice_pitch;
      unsigned linear_slice_pitch = linear == ssrc ? src_slice_pitch : dst_slice_pitch;
      uint64_t tiled_address = tiled == ssrc ? src_address : dst_address;
      uint64_t linear_address = linear == ssrc ? src_address : dst_address;
      unsigned tiled_micro_mode = tiled == ssrc ? src_micro_mode : dst_micro_mode;

      assert(tiled_pitch % 8 == 0);
      assert(tiled_slice_pitch % 64 == 0);
      unsigned pitch_tile_max = tiled_pitch / 8 - 1;
      unsigned slice_tile_max = tiled_slice_pitch / 64 - 1;
      unsigned xalign = MAX2(1, 4 / bpp);
      unsigned copy_width_aligned = copy_width;

      /* If the region ends at the last pixel and is unaligned, we
       * can copy the remainder of the line that is not visible to
       * make it aligned.
       */
      if (copy_width % xalign != 0 && linear_x + copy_width == linear_width &&
          tiled_x + copy_width == tiled_width &&
          linear_x + align(copy_width, xalign) <= linear_pitch &&
          tiled_x + align(copy_width, xalign) <= tiled_pitch)
         copy_width_aligned = align(copy_width, xalign);

      /* HW limitations. */
      if ((sctx->family == CHIP_BONAIRE || sctx->family == CHIP_KAVERI) &&
          linear_pitch - 1 == 0x3fff && bpp == 16)
         return false;

      if (sctx->chip_class == GFX7 &&
          (copy_width_aligned == (1 << 14) || copy_height == (1 << 14) || copy_depth == (1 << 11)))
         return false;

      if ((sctx->family == CHIP_BONAIRE || sctx->family == CHIP_KAVERI ||
           sctx->family == CHIP_KABINI) &&
          (tiled_x + copy_width == (1 << 14) || tiled_y + copy_height == (1 << 14)))
         return false;

      /* The hw can read outside of the given linear buffer bounds,
       * or access those pages but not touch the memory in case
       * of writes. (it still causes a VM fault)
       *
       * Out-of-bounds memory access or page directory access must
       * be prevented.
       */
      int64_t start_linear_address, end_linear_address;
      unsigned granularity;

      /* Deduce the size of reads from the linear surface. */
      switch (tiled_micro_mode) {
      case V_009910_ADDR_SURF_DISPLAY_MICRO_TILING:
         granularity = bpp == 1 ? 64 / (8 * bpp) : 128 / (8 * bpp);
         break;
      case V_009910_ADDR_SURF_THIN_MICRO_TILING:
      case V_009910_ADDR_SURF_DEPTH_MICRO_TILING:
         if (0 /* TODO: THICK microtiling */)
            granularity =
               bpp == 1 ? 32 / (8 * bpp)
                        : bpp == 2 ? 64 / (8 * bpp) : bpp <= 8 ? 128 / (8 * bpp) : 256 / (8 * bpp);
         else
            granularity = bpp <= 2 ? 64 / (8 * bpp) : bpp <= 8 ? 128 / (8 * bpp) : 256 / (8 * bpp);
         break;
      default:
         return false;
      }

      /* The linear reads start at tiled_x & ~(granularity - 1).
       * If linear_x == 0 && tiled_x % granularity != 0, the hw
       * starts reading from an address preceding linear_address!!!
       */
      start_linear_address =
         linear->surface.u.legacy.level[linear_level].offset +
         bpp * (linear_z * linear_slice_pitch + linear_y * linear_pitch + linear_x);
      start_linear_address -= (int)(bpp * (tiled_x % granularity));

      end_linear_address =
         linear->surface.u.legacy.level[linear_level].offset +
         bpp * ((linear_z + copy_depth - 1) * linear_slice_pitch +
                (linear_y + copy_height - 1) * linear_pitch + (linear_x + copy_width));

      if ((tiled_x + copy_width) % granularity)
         end_linear_address += granularity - (tiled_x + copy_width) % granularity;

      if (start_linear_address < 0 || end_linear_address > linear->surface.surf_size)
         return false;

      /* Check requirements. */
      if (tiled_address % 256 == 0 && linear_address % 4 == 0 && linear_pitch % xalign == 0 &&
          linear_x % xalign == 0 && tiled_x % xalign == 0 && copy_width_aligned % xalign == 0 &&
          tiled_micro_mode != V_009910_ADDR_SURF_ROTATED_MICRO_TILING &&
          /* check if everything fits into the bitfields */
          tiled->surface.u.legacy.tile_split <= 4096 && pitch_tile_max < (1 << 11) &&
          slice_tile_max < (1 << 22) && linear_pitch <= (1 << 14) &&
          linear_slice_pitch <= (1 << 28) && copy_width_aligned <= (1 << 14) &&
          copy_height <= (1 << 14) && copy_depth <= (1 << 11)) {
         struct radeon_cmdbuf *cs = sctx->sdma_cs;
         uint32_t direction = linear == sdst ? 1u << 31 : 0;

         si_need_dma_space(sctx, 14, &sdst->buffer, &ssrc->buffer);

         radeon_emit(cs, CIK_SDMA_PACKET(CIK_SDMA_OPCODE_COPY,
                                         CIK_SDMA_COPY_SUB_OPCODE_TILED_SUB_WINDOW, 0) |
                            direction);
         radeon_emit(cs, tiled_address);
         radeon_emit(cs, tiled_address >> 32);
         radeon_emit(cs, tiled_x | (tiled_y << 16));
         radeon_emit(cs, tiled_z | (pitch_tile_max << 16));
         radeon_emit(cs, slice_tile_max);
         radeon_emit(cs, encode_tile_info(sctx, tiled, tiled_level, true));
         radeon_emit(cs, linear_address);
         radeon_emit(cs, linear_address >> 32);
         radeon_emit(cs, linear_x | (linear_y << 16));
         radeon_emit(cs, linear_z | ((linear_pitch - 1) << 16));
         radeon_emit(cs, linear_slice_pitch - 1);
         if (sctx->chip_class == GFX7) {
            radeon_emit(cs, copy_width_aligned | (copy_height << 16));
            radeon_emit(cs, copy_depth);
         } else {
            radeon_emit(cs, (copy_width_aligned - 1) | ((copy_height - 1) << 16));
            radeon_emit(cs, (copy_depth - 1));
         }
         return true;
      }
   }

   /* Tiled -> Tiled sub-window copy. */
   if (dst_mode >= RADEON_SURF_MODE_1D && src_mode >= RADEON_SURF_MODE_1D &&
       /* check if these fit into the bitfields */
       src_address % 256 == 0 && dst_address % 256 == 0 &&
       ssrc->surface.u.legacy.tile_split <= 4096 && sdst->surface.u.legacy.tile_split <= 4096 &&
       dstx % 8 == 0 && dsty % 8 == 0 && srcx % 8 == 0 && srcy % 8 == 0 &&
       /* this can either be equal, or display->rotated (GFX8+ only) */
       (src_micro_mode == dst_micro_mode ||
        (sctx->chip_class >= GFX8 && src_micro_mode == V_009910_ADDR_SURF_DISPLAY_MICRO_TILING &&
         dst_micro_mode == V_009910_ADDR_SURF_ROTATED_MICRO_TILING))) {
      assert(src_pitch % 8 == 0);
      assert(dst_pitch % 8 == 0);
      assert(src_slice_pitch % 64 == 0);
      assert(dst_slice_pitch % 64 == 0);
      unsigned src_pitch_tile_max = src_pitch / 8 - 1;
      unsigned dst_pitch_tile_max = dst_pitch / 8 - 1;
      unsigned src_slice_tile_max = src_slice_pitch / 64 - 1;
      unsigned dst_slice_tile_max = dst_slice_pitch / 64 - 1;
      unsigned copy_width_aligned = copy_width;
      unsigned copy_height_aligned = copy_height;

      /* If the region ends at the last pixel and is unaligned, we
       * can copy the remainder of the tile that is not visible to
       * make it aligned.
       */
      if (copy_width % 8 != 0 && srcx + copy_width == src_width && dstx + copy_width == dst_width)
         copy_width_aligned = align(copy_width, 8);

      if (copy_height % 8 != 0 && srcy + copy_height == src_height &&
          dsty + copy_height == dst_height)
         copy_height_aligned = align(copy_height, 8);

      /* check if these fit into the bitfields */
      if (src_pitch_tile_max < (1 << 11) && dst_pitch_tile_max < (1 << 11) &&
          src_slice_tile_max < (1 << 22) && dst_slice_tile_max < (1 << 22) &&
          copy_width_aligned <= (1 << 14) && copy_height_aligned <= (1 << 14) &&
          copy_depth <= (1 << 11) && copy_width_aligned % 8 == 0 && copy_height_aligned % 8 == 0 &&
          /* HW limitation - GFX7: */
          (sctx->chip_class != GFX7 ||
           (copy_width_aligned < (1 << 14) && copy_height_aligned < (1 << 14) &&
            copy_depth < (1 << 11))) &&
          /* HW limitation - some GFX7 parts: */
          ((sctx->family != CHIP_BONAIRE && sctx->family != CHIP_KAVERI &&
            sctx->family != CHIP_KABINI) ||
           (srcx + copy_width_aligned != (1 << 14) && srcy + copy_height_aligned != (1 << 14) &&
            dstx + copy_width != (1 << 14)))) {
         struct radeon_cmdbuf *cs = sctx->sdma_cs;

         si_need_dma_space(sctx, 15, &sdst->buffer, &ssrc->buffer);

         radeon_emit(
            cs, CIK_SDMA_PACKET(CIK_SDMA_OPCODE_COPY, CIK_SDMA_COPY_SUB_OPCODE_T2T_SUB_WINDOW, 0));
         radeon_emit(cs, src_address);
         radeon_emit(cs, src_address >> 32);
         radeon_emit(cs, srcx | (srcy << 16));
         radeon_emit(cs, srcz | (src_pitch_tile_max << 16));
         radeon_emit(cs, src_slice_tile_max);
         radeon_emit(cs, encode_tile_info(sctx, ssrc, src_level, true));
         radeon_emit(cs, dst_address);
         radeon_emit(cs, dst_address >> 32);
         radeon_emit(cs, dstx | (dsty << 16));
         radeon_emit(cs, dstz | (dst_pitch_tile_max << 16));
         radeon_emit(cs, dst_slice_tile_max);
         radeon_emit(cs, encode_tile_info(sctx, sdst, dst_level, false));
         if (sctx->chip_class == GFX7) {
            radeon_emit(cs, copy_width_aligned | (copy_height_aligned << 16));
            radeon_emit(cs, copy_depth);
         } else {
            radeon_emit(cs, (copy_width_aligned - 8) | ((copy_height_aligned - 8) << 16));
            radeon_emit(cs, (copy_depth - 1));
         }
         return true;
      }
   }

   return false;
}

static void cik_sdma_copy(struct pipe_context *ctx, struct pipe_resource *dst, unsigned dst_level,
                          unsigned dstx, unsigned dsty, unsigned dstz, struct pipe_resource *src,
                          unsigned src_level, const struct pipe_box *src_box)
{
   struct si_context *sctx = (struct si_context *)ctx;

   assert(src->target != PIPE_BUFFER);

   if (!sctx->sdma_cs || src->flags & PIPE_RESOURCE_FLAG_SPARSE ||
       dst->flags & PIPE_RESOURCE_FLAG_SPARSE)
      goto fallback;

   /* SDMA causes corruption. See:
    *   https://bugs.freedesktop.org/show_bug.cgi?id=110575
    *   https://bugs.freedesktop.org/show_bug.cgi?id=110635
    *
    * Keep SDMA enabled on APUs.
    */
   if (sctx->screen->debug_flags & DBG(FORCE_SDMA) ||
       (!sctx->screen->info.has_dedicated_vram &&
        !(sctx->screen->debug_flags & DBG(NO_SDMA_COPY_IMAGE)))) {
      if ((sctx->chip_class == GFX7 || sctx->chip_class == GFX8) &&
          cik_sdma_copy_texture(sctx, dst, dst_level, dstx, dsty, dstz, src, src_level, src_box))
         return;
      else if (sctx->chip_class == GFX9 && si_sdma_v4_copy_texture(sctx, dst, dst_level, dstx, dsty,
                                                                   dstz, src, src_level, src_box))
         return;
   }

fallback:
   si_resource_copy_region(ctx, dst, dst_level, dstx, dsty, dstz, src, src_level, src_box);
}

void cik_init_sdma_functions(struct si_context *sctx)
{
   sctx->dma_copy = cik_sdma_copy;
}
