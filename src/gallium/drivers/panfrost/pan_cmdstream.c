/*
 * Copyright (C) 2018 Alyssa Rosenzweig
 * Copyright (C) 2020 Collabora Ltd.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the next
 * paragraph) shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "util/macros.h"
#include "util/u_prim.h"
#include "util/u_vbuf.h"

#include "panfrost-quirks.h"

#include "pan_allocate.h"
#include "pan_bo.h"
#include "pan_cmdstream.h"
#include "pan_context.h"
#include "pan_job.h"

/* If a BO is accessed for a particular shader stage, will it be in the primary
 * batch (vertex/tiler) or the secondary batch (fragment)? Anything but
 * fragment will be primary, e.g. compute jobs will be considered
 * "vertex/tiler" by analogy */

static inline uint32_t
panfrost_bo_access_for_stage(enum pipe_shader_type stage)
{
        assert(stage == PIPE_SHADER_FRAGMENT ||
               stage == PIPE_SHADER_VERTEX ||
               stage == PIPE_SHADER_COMPUTE);

        return stage == PIPE_SHADER_FRAGMENT ?
               PAN_BO_ACCESS_FRAGMENT :
               PAN_BO_ACCESS_VERTEX_TILER;
}

static void
panfrost_vt_emit_shared_memory(struct panfrost_context *ctx,
                               struct mali_vertex_tiler_postfix *postfix)
{
        struct panfrost_device *dev = pan_device(ctx->base.screen);
        struct panfrost_batch *batch = panfrost_get_batch_for_fbo(ctx);

        unsigned shift = panfrost_get_stack_shift(batch->stack_size);
        struct mali_shared_memory shared = {
                .stack_shift = shift,
                .scratchpad = panfrost_batch_get_scratchpad(batch, shift, dev->thread_tls_alloc, dev->core_count)->gpu,
                .shared_workgroup_count = ~0,
        };
        postfix->shared_memory = panfrost_upload_transient(batch, &shared, sizeof(shared));
}

static void
panfrost_vt_attach_framebuffer(struct panfrost_context *ctx,
                               struct mali_vertex_tiler_postfix *postfix)
{
        struct panfrost_device *dev = pan_device(ctx->base.screen);
        struct panfrost_batch *batch = panfrost_get_batch_for_fbo(ctx);

        /* If we haven't, reserve space for the framebuffer */

        if (!batch->framebuffer.gpu) {
                unsigned size = (dev->quirks & MIDGARD_SFBD) ?
                        sizeof(struct mali_single_framebuffer) :
                        sizeof(struct mali_framebuffer);

                batch->framebuffer = panfrost_allocate_transient(batch, size);

                /* Tag the pointer */
                if (!(dev->quirks & MIDGARD_SFBD))
                        batch->framebuffer.gpu |= MALI_MFBD;
        }

        postfix->shared_memory = batch->framebuffer.gpu;
}

static void
panfrost_vt_update_rasterizer(struct panfrost_context *ctx,
                              struct mali_vertex_tiler_prefix *prefix,
                              struct mali_vertex_tiler_postfix *postfix)
{
        struct panfrost_rasterizer *rasterizer = ctx->rasterizer;

        postfix->gl_enables |= 0x7;
        SET_BIT(postfix->gl_enables, MALI_FRONT_CCW_TOP,
                rasterizer && rasterizer->base.front_ccw);
        SET_BIT(postfix->gl_enables, MALI_CULL_FACE_FRONT,
                rasterizer && (rasterizer->base.cull_face & PIPE_FACE_FRONT));
        SET_BIT(postfix->gl_enables, MALI_CULL_FACE_BACK,
                rasterizer && (rasterizer->base.cull_face & PIPE_FACE_BACK));
        SET_BIT(prefix->unknown_draw, MALI_DRAW_FLATSHADE_FIRST,
                rasterizer && rasterizer->base.flatshade_first);
}

void
panfrost_vt_update_primitive_size(struct panfrost_context *ctx,
                                  struct mali_vertex_tiler_prefix *prefix,
                                  union midgard_primitive_size *primitive_size)
{
        struct panfrost_rasterizer *rasterizer = ctx->rasterizer;

        if (!panfrost_writes_point_size(ctx)) {
                bool points = prefix->draw_mode == MALI_POINTS;
                float val = 0.0f;

                if (rasterizer)
                        val = points ?
                              rasterizer->base.point_size :
                              rasterizer->base.line_width;

                primitive_size->constant = val;
        }
}

static void
panfrost_vt_update_occlusion_query(struct panfrost_context *ctx,
                                   struct mali_vertex_tiler_postfix *postfix)
{
        SET_BIT(postfix->gl_enables, MALI_OCCLUSION_QUERY, ctx->occlusion_query);
        if (ctx->occlusion_query)
                postfix->occlusion_counter = ctx->occlusion_query->bo->gpu;
        else
                postfix->occlusion_counter = 0;
}

void
panfrost_vt_init(struct panfrost_context *ctx,
                 enum pipe_shader_type stage,
                 struct mali_vertex_tiler_prefix *prefix,
                 struct mali_vertex_tiler_postfix *postfix)
{
        struct panfrost_device *device = pan_device(ctx->base.screen);

        if (!ctx->shader[stage])
                return;

        memset(prefix, 0, sizeof(*prefix));
        memset(postfix, 0, sizeof(*postfix));

        if (device->quirks & IS_BIFROST) {
                postfix->gl_enables = 0x2;
                panfrost_vt_emit_shared_memory(ctx, postfix);
        } else {
                postfix->gl_enables = 0x6;
                panfrost_vt_attach_framebuffer(ctx, postfix);
        }

        if (stage == PIPE_SHADER_FRAGMENT) {
                panfrost_vt_update_occlusion_query(ctx, postfix);
                panfrost_vt_update_rasterizer(ctx, prefix, postfix);
        }
}

static unsigned
panfrost_translate_index_size(unsigned size)
{
        switch (size) {
        case 1:
                return MALI_DRAW_INDEXED_UINT8;

        case 2:
                return MALI_DRAW_INDEXED_UINT16;

        case 4:
                return MALI_DRAW_INDEXED_UINT32;

        default:
                unreachable("Invalid index size");
        }
}

/* Gets a GPU address for the associated index buffer. Only gauranteed to be
 * good for the duration of the draw (transient), could last longer. Also get
 * the bounds on the index buffer for the range accessed by the draw. We do
 * these operations together because there are natural optimizations which
 * require them to be together. */

static mali_ptr
panfrost_get_index_buffer_bounded(struct panfrost_context *ctx,
                                  const struct pipe_draw_info *info,
                                  unsigned *min_index, unsigned *max_index)
{
        struct panfrost_resource *rsrc = pan_resource(info->index.resource);
        struct panfrost_batch *batch = panfrost_get_batch_for_fbo(ctx);
        off_t offset = info->start * info->index_size;
        bool needs_indices = true;
        mali_ptr out = 0;

        if (info->max_index != ~0u) {
                *min_index = info->min_index;
                *max_index = info->max_index;
                needs_indices = false;
        }

        if (!info->has_user_indices) {
                /* Only resources can be directly mapped */
                panfrost_batch_add_bo(batch, rsrc->bo,
                                      PAN_BO_ACCESS_SHARED |
                                      PAN_BO_ACCESS_READ |
                                      PAN_BO_ACCESS_VERTEX_TILER);
                out = rsrc->bo->gpu + offset;

                /* Check the cache */
                needs_indices = !panfrost_minmax_cache_get(rsrc->index_cache,
                                                           info->start,
                                                           info->count,
                                                           min_index,
                                                           max_index);
        } else {
                /* Otherwise, we need to upload to transient memory */
                const uint8_t *ibuf8 = (const uint8_t *) info->index.user;
                out = panfrost_upload_transient(batch, ibuf8 + offset,
                                                info->count *
                                                info->index_size);
        }

        if (needs_indices) {
                /* Fallback */
                u_vbuf_get_minmax_index(&ctx->base, info, min_index, max_index);

                if (!info->has_user_indices)
                        panfrost_minmax_cache_add(rsrc->index_cache,
                                                  info->start, info->count,
                                                  *min_index, *max_index);
        }

        return out;
}

void
panfrost_vt_set_draw_info(struct panfrost_context *ctx,
                          const struct pipe_draw_info *info,
                          enum mali_draw_mode draw_mode,
                          struct mali_vertex_tiler_postfix *vertex_postfix,
                          struct mali_vertex_tiler_prefix *tiler_prefix,
                          struct mali_vertex_tiler_postfix *tiler_postfix,
                          unsigned *vertex_count,
                          unsigned *padded_count)
{
        tiler_prefix->draw_mode = draw_mode;

        unsigned draw_flags = 0;

        if (panfrost_writes_point_size(ctx))
                draw_flags |= MALI_DRAW_VARYING_SIZE;

        if (info->primitive_restart)
                draw_flags |= MALI_DRAW_PRIMITIVE_RESTART_FIXED_INDEX;

        /* These doesn't make much sense */

        draw_flags |= 0x3000;

        if (info->index_size) {
                unsigned min_index = 0, max_index = 0;

                tiler_prefix->indices = panfrost_get_index_buffer_bounded(ctx,
                                                                       info,
                                                                       &min_index,
                                                                       &max_index);

                /* Use the corresponding values */
                *vertex_count = max_index - min_index + 1;
                tiler_postfix->offset_start = vertex_postfix->offset_start = min_index + info->index_bias;
                tiler_prefix->offset_bias_correction = -min_index;
                tiler_prefix->index_count = MALI_POSITIVE(info->count);
                draw_flags |= panfrost_translate_index_size(info->index_size);
        } else {
                tiler_prefix->indices = 0;
                *vertex_count = ctx->vertex_count;
                tiler_postfix->offset_start = vertex_postfix->offset_start = info->start;
                tiler_prefix->offset_bias_correction = 0;
                tiler_prefix->index_count = MALI_POSITIVE(ctx->vertex_count);
        }

        tiler_prefix->unknown_draw = draw_flags;

        /* Encode the padded vertex count */

        if (info->instance_count > 1) {
                *padded_count = panfrost_padded_vertex_count(*vertex_count);

                unsigned shift = __builtin_ctz(ctx->padded_count);
                unsigned k = ctx->padded_count >> (shift + 1);

                tiler_postfix->instance_shift = vertex_postfix->instance_shift = shift;
                tiler_postfix->instance_odd = vertex_postfix->instance_odd = k;
        } else {
                *padded_count = *vertex_count;

                /* Reset instancing state */
                tiler_postfix->instance_shift = vertex_postfix->instance_shift = 0;
                tiler_postfix->instance_odd = vertex_postfix->instance_odd = 0;
        }
}

static void
panfrost_shader_meta_init(struct panfrost_context *ctx,
                          enum pipe_shader_type st,
                          struct mali_shader_meta *meta)
{
        const struct panfrost_device *dev = pan_device(ctx->base.screen);
        struct panfrost_shader_state *ss = panfrost_get_shader_state(ctx, st);

        memset(meta, 0, sizeof(*meta));
        meta->shader = (ss->bo ? ss->bo->gpu : 0) | ss->first_tag;
        meta->attribute_count = ss->attribute_count;
        meta->varying_count = ss->varying_count;
        meta->texture_count = ctx->sampler_view_count[st];
        meta->sampler_count = ctx->sampler_count[st];

        if (dev->quirks & IS_BIFROST) {
                if (st == PIPE_SHADER_VERTEX)
                        meta->bifrost1.unk1 = 0x800000;
                else {
                        /* First clause ATEST |= 0x4000000.
                         * Less than 32 regs |= 0x200 */
                        meta->bifrost1.unk1 = 0x958020;
                }

                meta->bifrost1.uniform_buffer_count = panfrost_ubo_count(ctx, st);
                if (st == PIPE_SHADER_VERTEX)
                        meta->bifrost2.preload_regs = 0xC0;
                else
                        meta->bifrost2.preload_regs = 0x1;
                meta->bifrost2.uniform_count = MIN2(ss->uniform_count,
                                                    ss->uniform_cutoff);
        } else {
                meta->midgard1.uniform_count = MIN2(ss->uniform_count,
                                                    ss->uniform_cutoff);
                meta->midgard1.work_count = ss->work_reg_count;
                meta->midgard1.flags_hi = 0x8; /* XXX */
                meta->midgard1.flags_lo = 0x220;
                meta->midgard1.uniform_buffer_count = panfrost_ubo_count(ctx, st);
        }
}

static unsigned
panfrost_translate_compare_func(enum pipe_compare_func in)
{
        switch (in) {
        case PIPE_FUNC_NEVER:
                return MALI_FUNC_NEVER;

        case PIPE_FUNC_LESS:
                return MALI_FUNC_LESS;

        case PIPE_FUNC_EQUAL:
                return MALI_FUNC_EQUAL;

        case PIPE_FUNC_LEQUAL:
                return MALI_FUNC_LEQUAL;

        case PIPE_FUNC_GREATER:
                return MALI_FUNC_GREATER;

        case PIPE_FUNC_NOTEQUAL:
                return MALI_FUNC_NOTEQUAL;

        case PIPE_FUNC_GEQUAL:
                return MALI_FUNC_GEQUAL;

        case PIPE_FUNC_ALWAYS:
                return MALI_FUNC_ALWAYS;

        default:
                unreachable("Invalid func");
        }
}

static unsigned
panfrost_translate_stencil_op(enum pipe_stencil_op in)
{
        switch (in) {
        case PIPE_STENCIL_OP_KEEP:
                return MALI_STENCIL_KEEP;

        case PIPE_STENCIL_OP_ZERO:
                return MALI_STENCIL_ZERO;

        case PIPE_STENCIL_OP_REPLACE:
               return MALI_STENCIL_REPLACE;

        case PIPE_STENCIL_OP_INCR:
                return MALI_STENCIL_INCR;

        case PIPE_STENCIL_OP_DECR:
                return MALI_STENCIL_DECR;

        case PIPE_STENCIL_OP_INCR_WRAP:
                return MALI_STENCIL_INCR_WRAP;

        case PIPE_STENCIL_OP_DECR_WRAP:
                return MALI_STENCIL_DECR_WRAP;

        case PIPE_STENCIL_OP_INVERT:
                return MALI_STENCIL_INVERT;

        default:
                unreachable("Invalid stencil op");
        }
}

static unsigned
translate_tex_wrap(enum pipe_tex_wrap w)
{
        switch (w) {
        case PIPE_TEX_WRAP_REPEAT:
                return MALI_WRAP_REPEAT;

        case PIPE_TEX_WRAP_CLAMP:
                return MALI_WRAP_CLAMP;

        case PIPE_TEX_WRAP_CLAMP_TO_EDGE:
                return MALI_WRAP_CLAMP_TO_EDGE;

        case PIPE_TEX_WRAP_CLAMP_TO_BORDER:
                return MALI_WRAP_CLAMP_TO_BORDER;

        case PIPE_TEX_WRAP_MIRROR_REPEAT:
                return MALI_WRAP_MIRRORED_REPEAT;

        case PIPE_TEX_WRAP_MIRROR_CLAMP:
                return MALI_WRAP_MIRRORED_CLAMP;

        case PIPE_TEX_WRAP_MIRROR_CLAMP_TO_EDGE:
                return MALI_WRAP_MIRRORED_CLAMP_TO_EDGE;

        case PIPE_TEX_WRAP_MIRROR_CLAMP_TO_BORDER:
                return MALI_WRAP_MIRRORED_CLAMP_TO_BORDER;

        default:
                unreachable("Invalid wrap");
        }
}

void panfrost_sampler_desc_init(const struct pipe_sampler_state *cso,
                                struct mali_sampler_descriptor *hw)
{
        unsigned func = panfrost_translate_compare_func(cso->compare_func);
        bool min_nearest = cso->min_img_filter == PIPE_TEX_FILTER_NEAREST;
        bool mag_nearest = cso->mag_img_filter == PIPE_TEX_FILTER_NEAREST;
        bool mip_linear  = cso->min_mip_filter == PIPE_TEX_MIPFILTER_LINEAR;
        unsigned min_filter = min_nearest ? MALI_SAMP_MIN_NEAREST : 0;
        unsigned mag_filter = mag_nearest ? MALI_SAMP_MAG_NEAREST : 0;
        unsigned mip_filter = mip_linear  ?
                              (MALI_SAMP_MIP_LINEAR_1 | MALI_SAMP_MIP_LINEAR_2) : 0;
        unsigned normalized = cso->normalized_coords ? MALI_SAMP_NORM_COORDS : 0;

        *hw = (struct mali_sampler_descriptor) {
                .filter_mode = min_filter | mag_filter | mip_filter |
                               normalized,
                .wrap_s = translate_tex_wrap(cso->wrap_s),
                .wrap_t = translate_tex_wrap(cso->wrap_t),
                .wrap_r = translate_tex_wrap(cso->wrap_r),
                .compare_func = panfrost_flip_compare_func(func),
                .border_color = {
                        cso->border_color.f[0],
                        cso->border_color.f[1],
                        cso->border_color.f[2],
                        cso->border_color.f[3]
                },
                .min_lod = FIXED_16(cso->min_lod, false), /* clamp at 0 */
                .max_lod = FIXED_16(cso->max_lod, false),
                .lod_bias = FIXED_16(cso->lod_bias, true), /* can be negative */
                .seamless_cube_map = cso->seamless_cube_map,
        };

        /* If necessary, we disable mipmapping in the sampler descriptor by
         * clamping the LOD as tight as possible (from 0 to epsilon,
         * essentially -- remember these are fixed point numbers, so
         * epsilon=1/256) */

        if (cso->min_mip_filter == PIPE_TEX_MIPFILTER_NONE)
                hw->max_lod = hw->min_lod + 1;
}

void panfrost_sampler_desc_init_bifrost(const struct pipe_sampler_state *cso,
                                        struct bifrost_sampler_descriptor *hw)
{
        *hw = (struct bifrost_sampler_descriptor) {
                .unk1 = 0x1,
                .wrap_s = translate_tex_wrap(cso->wrap_s),
                .wrap_t = translate_tex_wrap(cso->wrap_t),
                .wrap_r = translate_tex_wrap(cso->wrap_r),
                .unk8 = 0x8,
                .min_filter = cso->min_img_filter == PIPE_TEX_FILTER_NEAREST,
                .norm_coords = cso->normalized_coords,
                .mip_filter = cso->min_mip_filter == PIPE_TEX_MIPFILTER_LINEAR,
                .mag_filter = cso->mag_img_filter == PIPE_TEX_FILTER_LINEAR,
                .min_lod = FIXED_16(cso->min_lod, false), /* clamp at 0 */
                .max_lod = FIXED_16(cso->max_lod, false),
        };

        /* If necessary, we disable mipmapping in the sampler descriptor by
         * clamping the LOD as tight as possible (from 0 to epsilon,
         * essentially -- remember these are fixed point numbers, so
         * epsilon=1/256) */

        if (cso->min_mip_filter == PIPE_TEX_MIPFILTER_NONE)
                hw->max_lod = hw->min_lod + 1;
}

static void
panfrost_make_stencil_state(const struct pipe_stencil_state *in,
                            struct mali_stencil_test *out)
{
        out->ref = 0; /* Gallium gets it from elsewhere */

        out->mask = in->valuemask;
        out->func = panfrost_translate_compare_func(in->func);
        out->sfail = panfrost_translate_stencil_op(in->fail_op);
        out->dpfail = panfrost_translate_stencil_op(in->zfail_op);
        out->dppass = panfrost_translate_stencil_op(in->zpass_op);
}

static void
panfrost_frag_meta_rasterizer_update(struct panfrost_context *ctx,
                                     struct mali_shader_meta *fragmeta)
{
        if (!ctx->rasterizer) {
                SET_BIT(fragmeta->unknown2_4, MALI_NO_MSAA, true);
                SET_BIT(fragmeta->unknown2_3, MALI_HAS_MSAA, false);
                fragmeta->depth_units = 0.0f;
                fragmeta->depth_factor = 0.0f;
                SET_BIT(fragmeta->unknown2_4, MALI_DEPTH_RANGE_A, false);
                SET_BIT(fragmeta->unknown2_4, MALI_DEPTH_RANGE_B, false);
                return;
        }

        bool msaa = ctx->rasterizer->base.multisample;

        /* TODO: Sample size */
        SET_BIT(fragmeta->unknown2_3, MALI_HAS_MSAA, msaa);
        SET_BIT(fragmeta->unknown2_4, MALI_NO_MSAA, !msaa);
        fragmeta->depth_units = ctx->rasterizer->base.offset_units * 2.0f;
        fragmeta->depth_factor = ctx->rasterizer->base.offset_scale;

        /* XXX: Which bit is which? Does this maybe allow offseting not-tri? */

        SET_BIT(fragmeta->unknown2_4, MALI_DEPTH_RANGE_A,
                ctx->rasterizer->base.offset_tri);
        SET_BIT(fragmeta->unknown2_4, MALI_DEPTH_RANGE_B,
                ctx->rasterizer->base.offset_tri);
}

static void
panfrost_frag_meta_zsa_update(struct panfrost_context *ctx,
                              struct mali_shader_meta *fragmeta)
{
        const struct pipe_depth_stencil_alpha_state *zsa = ctx->depth_stencil;
        int zfunc = PIPE_FUNC_ALWAYS;

        if (!zsa) {
                struct pipe_stencil_state default_stencil = {
                        .enabled = 0,
                        .func = PIPE_FUNC_ALWAYS,
                        .fail_op = MALI_STENCIL_KEEP,
                        .zfail_op = MALI_STENCIL_KEEP,
                        .zpass_op = MALI_STENCIL_KEEP,
                        .writemask = 0xFF,
                        .valuemask = 0xFF
                };

                panfrost_make_stencil_state(&default_stencil,
                                            &fragmeta->stencil_front);
                fragmeta->stencil_mask_front = default_stencil.writemask;
                fragmeta->stencil_back = fragmeta->stencil_front;
                fragmeta->stencil_mask_back = default_stencil.writemask;
                SET_BIT(fragmeta->unknown2_4, MALI_STENCIL_TEST, false);
                SET_BIT(fragmeta->unknown2_3, MALI_DEPTH_WRITEMASK, false);
        } else {
                SET_BIT(fragmeta->unknown2_4, MALI_STENCIL_TEST,
                        zsa->stencil[0].enabled);
                panfrost_make_stencil_state(&zsa->stencil[0],
                                            &fragmeta->stencil_front);
                fragmeta->stencil_mask_front = zsa->stencil[0].writemask;
                fragmeta->stencil_front.ref = ctx->stencil_ref.ref_value[0];

                /* If back-stencil is not enabled, use the front values */

                if (zsa->stencil[1].enabled) {
                        panfrost_make_stencil_state(&zsa->stencil[1],
                                                    &fragmeta->stencil_back);
                        fragmeta->stencil_mask_back = zsa->stencil[1].writemask;
                        fragmeta->stencil_back.ref = ctx->stencil_ref.ref_value[1];
                } else {
                        fragmeta->stencil_back = fragmeta->stencil_front;
                        fragmeta->stencil_mask_back = fragmeta->stencil_mask_front;
                        fragmeta->stencil_back.ref = fragmeta->stencil_front.ref;
                }

                if (zsa->depth.enabled)
                        zfunc = zsa->depth.func;

                /* Depth state (TODO: Refactor) */

                SET_BIT(fragmeta->unknown2_3, MALI_DEPTH_WRITEMASK,
                        zsa->depth.writemask);
        }

        fragmeta->unknown2_3 &= ~MALI_DEPTH_FUNC_MASK;
        fragmeta->unknown2_3 |= MALI_DEPTH_FUNC(panfrost_translate_compare_func(zfunc));
}

static bool
panfrost_fs_required(
                struct panfrost_shader_state *fs,
                struct panfrost_blend_final *blend,
                unsigned rt_count)
{
        /* If we generally have side effects */
        if (fs->fs_sidefx)
                return true;

        /* If colour is written we need to execute */
        for (unsigned i = 0; i < rt_count; ++i) {
                if (!blend[i].no_colour)
                        return true;
        }

        /* If depth is written and not implied we need to execute.
         * TODO: Predicate on Z/S writes being enabled */
        return (fs->writes_depth || fs->writes_stencil);
}

static void
panfrost_frag_meta_blend_update(struct panfrost_context *ctx,
                                struct mali_shader_meta *fragmeta,
                                void *rts)
{
        const struct panfrost_device *dev = pan_device(ctx->base.screen);
        struct panfrost_shader_state *fs;
        fs = panfrost_get_shader_state(ctx, PIPE_SHADER_FRAGMENT);

        SET_BIT(fragmeta->unknown2_4, MALI_NO_DITHER,
                (dev->quirks & MIDGARD_SFBD) && ctx->blend &&
                !ctx->blend->base.dither);

        /* Get blending setup */
        unsigned rt_count = MAX2(ctx->pipe_framebuffer.nr_cbufs, 1);

        struct panfrost_blend_final blend[PIPE_MAX_COLOR_BUFS];
        unsigned shader_offset = 0;
        struct panfrost_bo *shader_bo = NULL;

        for (unsigned c = 0; c < rt_count; ++c)
                blend[c] = panfrost_get_blend_for_context(ctx, c, &shader_bo,
                                                          &shader_offset);

        /* Disable shader execution if we can */
        if (dev->quirks & MIDGARD_SHADERLESS
                        && !panfrost_fs_required(fs, blend, rt_count)) {
                fragmeta->shader = 0;
                fragmeta->attribute_count = 0;
                fragmeta->varying_count = 0;
                fragmeta->texture_count = 0;
                fragmeta->sampler_count = 0;

                /* This feature is not known to work on Bifrost */
                fragmeta->midgard1.work_count = 1;
                fragmeta->midgard1.uniform_count = 0;
                fragmeta->midgard1.uniform_buffer_count = 0;
        }

         /* If there is a blend shader, work registers are shared. We impose 8
          * work registers as a limit for blend shaders. Should be lower XXX */

        if (!(dev->quirks & IS_BIFROST)) {
                for (unsigned c = 0; c < rt_count; ++c) {
                        if (blend[c].is_shader) {
                                fragmeta->midgard1.work_count =
                                        MAX2(fragmeta->midgard1.work_count, 8);
                        }
                }
        }

        /* Even on MFBD, the shader descriptor gets blend shaders. It's *also*
         * copied to the blend_meta appended (by convention), but this is the
         * field actually read by the hardware. (Or maybe both are read...?).
         * Specify the last RTi with a blend shader. */

        fragmeta->blend.shader = 0;

        for (signed rt = (rt_count - 1); rt >= 0; --rt) {
                if (!blend[rt].is_shader)
                        continue;

                fragmeta->blend.shader = blend[rt].shader.gpu |
                                         blend[rt].shader.first_tag;
                break;
        }

        if (dev->quirks & MIDGARD_SFBD) {
                /* When only a single render target platform is used, the blend
                 * information is inside the shader meta itself. We additionally
                 * need to signal CAN_DISCARD for nontrivial blend modes (so
                 * we're able to read back the destination buffer) */

                SET_BIT(fragmeta->unknown2_3, MALI_HAS_BLEND_SHADER,
                        blend[0].is_shader);

                if (!blend[0].is_shader) {
                        fragmeta->blend.equation = *blend[0].equation.equation;
                        fragmeta->blend.constant = blend[0].equation.constant;
                }

                SET_BIT(fragmeta->unknown2_3, MALI_CAN_DISCARD,
                        !blend[0].no_blending || fs->can_discard); 
                return;
        }

        /* Additional blend descriptor tacked on for jobs using MFBD */

        for (unsigned i = 0; i < rt_count; ++i) {
                if (dev->quirks & IS_BIFROST) {
                        struct bifrost_blend_rt *brts = rts;

                        brts[i].flags = 0x200;
                        if (blend[i].is_shader) {
                                /* The blend shader's address needs to be at
                                 * the same top 32 bit as the fragment shader.
                                 * TODO: Ensure that's always the case.
                                 */
                                assert((blend[i].shader.gpu & (0xffffffffull << 32)) ==
                                       (fs->bo->gpu & (0xffffffffull << 32)));
                                brts[i].shader = blend[i].shader.gpu;
                                brts[i].unk2 = 0x0;
                        } else {
                                enum pipe_format format = ctx->pipe_framebuffer.cbufs[i]->format;
                                const struct util_format_description *format_desc;
                                format_desc = util_format_description(format);

                                brts[i].equation = *blend[i].equation.equation;

                                /* TODO: this is a bit more complicated */
                                brts[i].constant = blend[i].equation.constant;

                                brts[i].format = panfrost_format_to_bifrost_blend(format_desc);
                                brts[i].unk2 = 0x19;

                                brts[i].shader_type = fs->blend_types[i];
                        }
                } else {
                        struct midgard_blend_rt *mrts = rts;

                        if (!blend[i].no_colour) {
                                mrts[i].flags = 0x200;

                                bool is_srgb = (ctx->pipe_framebuffer.nr_cbufs > i) &&
                                               (ctx->pipe_framebuffer.cbufs[i]) &&
                                               util_format_is_srgb(ctx->pipe_framebuffer.cbufs[i]->format);

                                SET_BIT(mrts[i].flags, MALI_BLEND_MRT_SHADER, blend[i].is_shader);
                                SET_BIT(mrts[i].flags, MALI_BLEND_LOAD_TIB, !blend[i].no_blending);
                                SET_BIT(mrts[i].flags, MALI_BLEND_SRGB, is_srgb);
                                SET_BIT(mrts[i].flags, MALI_BLEND_NO_DITHER, !ctx->blend->base.dither);
                        }

                        if (blend[i].is_shader) {
                                mrts[i].blend.shader = blend[i].shader.gpu | blend[i].shader.first_tag;
                        } else {
                                mrts[i].blend.equation = *blend[i].equation.equation;
                                mrts[i].blend.constant = blend[i].equation.constant;
                        }
                }
        }
}

static void
panfrost_frag_shader_meta_init(struct panfrost_context *ctx,
                               struct mali_shader_meta *fragmeta,
                               void *rts)
{
        const struct panfrost_device *dev = pan_device(ctx->base.screen);
        struct panfrost_shader_state *fs;

        fs = panfrost_get_shader_state(ctx, PIPE_SHADER_FRAGMENT);

        fragmeta->alpha_coverage = ~MALI_ALPHA_COVERAGE(0.000000);
        fragmeta->unknown2_3 = MALI_DEPTH_FUNC(MALI_FUNC_ALWAYS) | 0x3010;
        fragmeta->unknown2_4 = 0x4e0;

        /* unknown2_4 has 0x10 bit set on T6XX and T720. We don't know why this
         * is required (independent of 32-bit/64-bit descriptors), or why it's
         * not used on later GPU revisions. Otherwise, all shader jobs fault on
         * these earlier chips (perhaps this is a chicken bit of some kind).
         * More investigation is needed. */

        SET_BIT(fragmeta->unknown2_4, 0x10, dev->quirks & MIDGARD_SFBD);

        if (dev->quirks & IS_BIFROST) {
                /* TODO */
        } else {
                /* Depending on whether it's legal to in the given shader, we try to
                 * enable early-z testing (or forward-pixel kill?) */

                SET_BIT(fragmeta->midgard1.flags_lo, MALI_EARLY_Z,
                        !fs->can_discard && !fs->writes_depth);

                /* Add the writes Z/S flags if needed. */
                SET_BIT(fragmeta->midgard1.flags_lo, MALI_WRITES_Z, fs->writes_depth);
                SET_BIT(fragmeta->midgard1.flags_hi, MALI_WRITES_S, fs->writes_stencil);

                /* Any time texturing is used, derivatives are implicitly calculated,
                 * so we need to enable helper invocations */

                SET_BIT(fragmeta->midgard1.flags_lo, MALI_HELPER_INVOCATIONS,
                        fs->helper_invocations);

                const struct pipe_depth_stencil_alpha_state *zsa = ctx->depth_stencil;

                bool depth_enabled = fs->writes_depth ||
                   (zsa && zsa->depth.enabled && zsa->depth.func != PIPE_FUNC_ALWAYS);

                SET_BIT(fragmeta->midgard1.flags_lo, 0x400, !depth_enabled && fs->can_discard);
                SET_BIT(fragmeta->midgard1.flags_lo, MALI_READS_ZS, depth_enabled && fs->can_discard);
        }

        panfrost_frag_meta_rasterizer_update(ctx, fragmeta);
        panfrost_frag_meta_zsa_update(ctx, fragmeta);
        panfrost_frag_meta_blend_update(ctx, fragmeta, rts);
}

void
panfrost_emit_shader_meta(struct panfrost_batch *batch,
                          enum pipe_shader_type st,
                          struct mali_vertex_tiler_postfix *postfix)
{
        struct panfrost_context *ctx = batch->ctx;
        struct panfrost_shader_state *ss = panfrost_get_shader_state(ctx, st);

        if (!ss) {
                postfix->shader = 0;
                return;
        }

        struct mali_shader_meta meta;

        panfrost_shader_meta_init(ctx, st, &meta);

        /* Add the shader BO to the batch. */
        panfrost_batch_add_bo(batch, ss->bo,
                              PAN_BO_ACCESS_PRIVATE |
                              PAN_BO_ACCESS_READ |
                              panfrost_bo_access_for_stage(st));

        mali_ptr shader_ptr;

        if (st == PIPE_SHADER_FRAGMENT) {
                struct panfrost_device *dev = pan_device(ctx->base.screen);
                unsigned rt_count = MAX2(ctx->pipe_framebuffer.nr_cbufs, 1);
                size_t desc_size = sizeof(meta);
                void *rts = NULL;
                struct panfrost_transfer xfer;
                unsigned rt_size;

                if (dev->quirks & MIDGARD_SFBD)
                        rt_size = 0;
                else if (dev->quirks & IS_BIFROST)
                        rt_size = sizeof(struct bifrost_blend_rt);
                else
                        rt_size = sizeof(struct midgard_blend_rt);

                desc_size += rt_size * rt_count;

                if (rt_size)
                        rts = rzalloc_size(ctx, rt_size * rt_count);

                panfrost_frag_shader_meta_init(ctx, &meta, rts);

                xfer = panfrost_allocate_transient(batch, desc_size);

                memcpy(xfer.cpu, &meta, sizeof(meta));
                memcpy(xfer.cpu + sizeof(meta), rts, rt_size * rt_count);

                if (rt_size)
                        ralloc_free(rts);

                shader_ptr = xfer.gpu;
        } else {
                shader_ptr = panfrost_upload_transient(batch, &meta,
                                                       sizeof(meta));
        }

        postfix->shader = shader_ptr;
}

static void
panfrost_mali_viewport_init(struct panfrost_context *ctx,
                            struct mali_viewport *mvp)
{
        const struct pipe_viewport_state *vp = &ctx->pipe_viewport;

        /* Clip bounds are encoded as floats. The viewport itself is encoded as
         * (somewhat) asymmetric ints. */

        const struct pipe_scissor_state *ss = &ctx->scissor;

        memset(mvp, 0, sizeof(*mvp));

        /* By default, do no viewport clipping, i.e. clip to (-inf, inf) in
         * each direction. Clipping to the viewport in theory should work, but
         * in practice causes issues when we're not explicitly trying to
         * scissor */

        *mvp = (struct mali_viewport) {
                .clip_minx = -INFINITY,
                .clip_miny = -INFINITY,
                .clip_maxx = INFINITY,
                .clip_maxy = INFINITY,
        };

        /* Always scissor to the viewport by default. */
        float vp_minx = (int) (vp->translate[0] - fabsf(vp->scale[0]));
        float vp_maxx = (int) (vp->translate[0] + fabsf(vp->scale[0]));

        float vp_miny = (int) (vp->translate[1] - fabsf(vp->scale[1]));
        float vp_maxy = (int) (vp->translate[1] + fabsf(vp->scale[1]));

        float minz = (vp->translate[2] - fabsf(vp->scale[2]));
        float maxz = (vp->translate[2] + fabsf(vp->scale[2]));

        /* Apply the scissor test */

        unsigned minx, miny, maxx, maxy;

        if (ss && ctx->rasterizer && ctx->rasterizer->base.scissor) {
                minx = MAX2(ss->minx, vp_minx);
                miny = MAX2(ss->miny, vp_miny);
                maxx = MIN2(ss->maxx, vp_maxx);
                maxy = MIN2(ss->maxy, vp_maxy);
        } else {
                minx = vp_minx;
                miny = vp_miny;
                maxx = vp_maxx;
                maxy = vp_maxy;
        }

        /* Hardware needs the min/max to be strictly ordered, so flip if we
         * need to. The viewport transformation in the vertex shader will
         * handle the negatives if we don't */

        if (miny > maxy) {
                unsigned temp = miny;
                miny = maxy;
                maxy = temp;
        }

        if (minx > maxx) {
                unsigned temp = minx;
                minx = maxx;
                maxx = temp;
        }

        if (minz > maxz) {
                float temp = minz;
                minz = maxz;
                maxz = temp;
        }

        /* Clamp to the framebuffer size as a last check */

        minx = MIN2(ctx->pipe_framebuffer.width, minx);
        maxx = MIN2(ctx->pipe_framebuffer.width, maxx);

        miny = MIN2(ctx->pipe_framebuffer.height, miny);
        maxy = MIN2(ctx->pipe_framebuffer.height, maxy);

        /* Upload */

        mvp->viewport0[0] = minx;
        mvp->viewport1[0] = MALI_POSITIVE(maxx);

        mvp->viewport0[1] = miny;
        mvp->viewport1[1] = MALI_POSITIVE(maxy);

        mvp->clip_minz = minz;
        mvp->clip_maxz = maxz;
}

void
panfrost_emit_viewport(struct panfrost_batch *batch,
                       struct mali_vertex_tiler_postfix *tiler_postfix)
{
        struct panfrost_context *ctx = batch->ctx;
        struct mali_viewport mvp;

        panfrost_mali_viewport_init(batch->ctx,  &mvp);

        /* Update the job, unless we're doing wallpapering (whose lack of
         * scissor we can ignore, since if we "miss" a tile of wallpaper, it'll
         * just... be faster :) */

        if (!ctx->wallpaper_batch)
                panfrost_batch_union_scissor(batch, mvp.viewport0[0],
                                             mvp.viewport0[1],
                                             mvp.viewport1[0] + 1,
                                             mvp.viewport1[1] + 1);

        tiler_postfix->viewport = panfrost_upload_transient(batch, &mvp,
                                                            sizeof(mvp));
}

static mali_ptr
panfrost_map_constant_buffer_gpu(struct panfrost_batch *batch,
                                 enum pipe_shader_type st,
                                 struct panfrost_constant_buffer *buf,
                                 unsigned index)
{
        struct pipe_constant_buffer *cb = &buf->cb[index];
        struct panfrost_resource *rsrc = pan_resource(cb->buffer);

        if (rsrc) {
                panfrost_batch_add_bo(batch, rsrc->bo,
                                      PAN_BO_ACCESS_SHARED |
                                      PAN_BO_ACCESS_READ |
                                      panfrost_bo_access_for_stage(st));

                /* Alignment gauranteed by
                 * PIPE_CAP_CONSTANT_BUFFER_OFFSET_ALIGNMENT */
                return rsrc->bo->gpu + cb->buffer_offset;
        } else if (cb->user_buffer) {
                return panfrost_upload_transient(batch,
                                                 cb->user_buffer +
                                                 cb->buffer_offset,
                                                 cb->buffer_size);
        } else {
                unreachable("No constant buffer");
        }
}

struct sysval_uniform {
        union {
                float f[4];
                int32_t i[4];
                uint32_t u[4];
                uint64_t du[2];
        };
};

static void
panfrost_upload_viewport_scale_sysval(struct panfrost_batch *batch,
                                      struct sysval_uniform *uniform)
{
        struct panfrost_context *ctx = batch->ctx;
        const struct pipe_viewport_state *vp = &ctx->pipe_viewport;

        uniform->f[0] = vp->scale[0];
        uniform->f[1] = vp->scale[1];
        uniform->f[2] = vp->scale[2];
}

static void
panfrost_upload_viewport_offset_sysval(struct panfrost_batch *batch,
                                       struct sysval_uniform *uniform)
{
        struct panfrost_context *ctx = batch->ctx;
        const struct pipe_viewport_state *vp = &ctx->pipe_viewport;

        uniform->f[0] = vp->translate[0];
        uniform->f[1] = vp->translate[1];
        uniform->f[2] = vp->translate[2];
}

static void panfrost_upload_txs_sysval(struct panfrost_batch *batch,
                                       enum pipe_shader_type st,
                                       unsigned int sysvalid,
                                       struct sysval_uniform *uniform)
{
        struct panfrost_context *ctx = batch->ctx;
        unsigned texidx = PAN_SYSVAL_ID_TO_TXS_TEX_IDX(sysvalid);
        unsigned dim = PAN_SYSVAL_ID_TO_TXS_DIM(sysvalid);
        bool is_array = PAN_SYSVAL_ID_TO_TXS_IS_ARRAY(sysvalid);
        struct pipe_sampler_view *tex = &ctx->sampler_views[st][texidx]->base;

        assert(dim);
        uniform->i[0] = u_minify(tex->texture->width0, tex->u.tex.first_level);

        if (dim > 1)
                uniform->i[1] = u_minify(tex->texture->height0,
                                         tex->u.tex.first_level);

        if (dim > 2)
                uniform->i[2] = u_minify(tex->texture->depth0,
                                         tex->u.tex.first_level);

        if (is_array)
                uniform->i[dim] = tex->texture->array_size;
}

static void
panfrost_upload_ssbo_sysval(struct panfrost_batch *batch,
                            enum pipe_shader_type st,
                            unsigned ssbo_id,
                            struct sysval_uniform *uniform)
{
        struct panfrost_context *ctx = batch->ctx;

        assert(ctx->ssbo_mask[st] & (1 << ssbo_id));
        struct pipe_shader_buffer sb = ctx->ssbo[st][ssbo_id];

        /* Compute address */
        struct panfrost_bo *bo = pan_resource(sb.buffer)->bo;

        panfrost_batch_add_bo(batch, bo,
                              PAN_BO_ACCESS_SHARED | PAN_BO_ACCESS_RW |
                              panfrost_bo_access_for_stage(st));

        /* Upload address and size as sysval */
        uniform->du[0] = bo->gpu + sb.buffer_offset;
        uniform->u[2] = sb.buffer_size;
}

static void
panfrost_upload_sampler_sysval(struct panfrost_batch *batch,
                               enum pipe_shader_type st,
                               unsigned samp_idx,
                               struct sysval_uniform *uniform)
{
        struct panfrost_context *ctx = batch->ctx;
        struct pipe_sampler_state *sampl = &ctx->samplers[st][samp_idx]->base;

        uniform->f[0] = sampl->min_lod;
        uniform->f[1] = sampl->max_lod;
        uniform->f[2] = sampl->lod_bias;

        /* Even without any errata, Midgard represents "no mipmapping" as
         * fixing the LOD with the clamps; keep behaviour consistent. c.f.
         * panfrost_create_sampler_state which also explains our choice of
         * epsilon value (again to keep behaviour consistent) */

        if (sampl->min_mip_filter == PIPE_TEX_MIPFILTER_NONE)
                uniform->f[1] = uniform->f[0] + (1.0/256.0);
}

static void
panfrost_upload_num_work_groups_sysval(struct panfrost_batch *batch,
                                       struct sysval_uniform *uniform)
{
        struct panfrost_context *ctx = batch->ctx;

        uniform->u[0] = ctx->compute_grid->grid[0];
        uniform->u[1] = ctx->compute_grid->grid[1];
        uniform->u[2] = ctx->compute_grid->grid[2];
}

static void
panfrost_upload_sysvals(struct panfrost_batch *batch, void *buf,
                        struct panfrost_shader_state *ss,
                        enum pipe_shader_type st)
{
        struct sysval_uniform *uniforms = (void *)buf;

        for (unsigned i = 0; i < ss->sysval_count; ++i) {
                int sysval = ss->sysval[i];

                switch (PAN_SYSVAL_TYPE(sysval)) {
                case PAN_SYSVAL_VIEWPORT_SCALE:
                        panfrost_upload_viewport_scale_sysval(batch,
                                                              &uniforms[i]);
                        break;
                case PAN_SYSVAL_VIEWPORT_OFFSET:
                        panfrost_upload_viewport_offset_sysval(batch,
                                                               &uniforms[i]);
                        break;
                case PAN_SYSVAL_TEXTURE_SIZE:
                        panfrost_upload_txs_sysval(batch, st,
                                                   PAN_SYSVAL_ID(sysval),
                                                   &uniforms[i]);
                        break;
                case PAN_SYSVAL_SSBO:
                        panfrost_upload_ssbo_sysval(batch, st,
                                                    PAN_SYSVAL_ID(sysval),
                                                    &uniforms[i]);
                        break;
                case PAN_SYSVAL_NUM_WORK_GROUPS:
                        panfrost_upload_num_work_groups_sysval(batch,
                                                               &uniforms[i]);
                        break;
                case PAN_SYSVAL_SAMPLER:
                        panfrost_upload_sampler_sysval(batch, st,
                                                       PAN_SYSVAL_ID(sysval),
                                                       &uniforms[i]);
                        break;
                default:
                        assert(0);
                }
        }
}

static const void *
panfrost_map_constant_buffer_cpu(struct panfrost_constant_buffer *buf,
                                 unsigned index)
{
        struct pipe_constant_buffer *cb = &buf->cb[index];
        struct panfrost_resource *rsrc = pan_resource(cb->buffer);

        if (rsrc)
                return rsrc->bo->cpu;
        else if (cb->user_buffer)
                return cb->user_buffer;
        else
                unreachable("No constant buffer");
}

void
panfrost_emit_const_buf(struct panfrost_batch *batch,
                        enum pipe_shader_type stage,
                        struct mali_vertex_tiler_postfix *postfix)
{
        struct panfrost_context *ctx = batch->ctx;
        struct panfrost_shader_variants *all = ctx->shader[stage];

        if (!all)
                return;

        struct panfrost_constant_buffer *buf = &ctx->constant_buffer[stage];

        struct panfrost_shader_state *ss = &all->variants[all->active_variant];

        /* Uniforms are implicitly UBO #0 */
        bool has_uniforms = buf->enabled_mask & (1 << 0);

        /* Allocate room for the sysval and the uniforms */
        size_t sys_size = sizeof(float) * 4 * ss->sysval_count;
        size_t uniform_size = has_uniforms ? (buf->cb[0].buffer_size) : 0;
        size_t size = sys_size + uniform_size;
        struct panfrost_transfer transfer = panfrost_allocate_transient(batch,
                                                                        size);

        /* Upload sysvals requested by the shader */
        panfrost_upload_sysvals(batch, transfer.cpu, ss, stage);

        /* Upload uniforms */
        if (has_uniforms && uniform_size) {
                const void *cpu = panfrost_map_constant_buffer_cpu(buf, 0);
                memcpy(transfer.cpu + sys_size, cpu, uniform_size);
        }

        /* Next up, attach UBOs. UBO #0 is the uniforms we just
         * uploaded */

        unsigned ubo_count = panfrost_ubo_count(ctx, stage);
        assert(ubo_count >= 1);

        size_t sz = sizeof(uint64_t) * ubo_count;
        uint64_t ubos[PAN_MAX_CONST_BUFFERS];
        int uniform_count = ss->uniform_count;

        /* Upload uniforms as a UBO */
        ubos[0] = MALI_MAKE_UBO(2 + uniform_count, transfer.gpu);

        /* The rest are honest-to-goodness UBOs */

        for (unsigned ubo = 1; ubo < ubo_count; ++ubo) {
                size_t usz = buf->cb[ubo].buffer_size;
                bool enabled = buf->enabled_mask & (1 << ubo);
                bool empty = usz == 0;

                if (!enabled || empty) {
                        /* Stub out disabled UBOs to catch accesses */
                        ubos[ubo] = MALI_MAKE_UBO(0, 0xDEAD0000);
                        continue;
                }

                mali_ptr gpu = panfrost_map_constant_buffer_gpu(batch, stage,
                                                                buf, ubo);

                unsigned bytes_per_field = 16;
                unsigned aligned = ALIGN_POT(usz, bytes_per_field);
                ubos[ubo] = MALI_MAKE_UBO(aligned / bytes_per_field, gpu);
        }

        mali_ptr ubufs = panfrost_upload_transient(batch, ubos, sz);
        postfix->uniforms = transfer.gpu;
        postfix->uniform_buffers = ubufs;

        buf->dirty_mask = 0;
}

void
panfrost_emit_shared_memory(struct panfrost_batch *batch,
                            const struct pipe_grid_info *info,
                            struct midgard_payload_vertex_tiler *vtp)
{
        struct panfrost_context *ctx = batch->ctx;
        struct panfrost_shader_variants *all = ctx->shader[PIPE_SHADER_COMPUTE];
        struct panfrost_shader_state *ss = &all->variants[all->active_variant];
        unsigned single_size = util_next_power_of_two(MAX2(ss->shared_size,
                                                           128));
        unsigned shared_size = single_size * info->grid[0] * info->grid[1] *
                               info->grid[2] * 4;
        struct panfrost_bo *bo = panfrost_batch_get_shared_memory(batch,
                                                                  shared_size,
                                                                  1);

        struct mali_shared_memory shared = {
                .shared_memory = bo->gpu,
                .shared_workgroup_count =
                        util_logbase2_ceil(info->grid[0]) +
                        util_logbase2_ceil(info->grid[1]) +
                        util_logbase2_ceil(info->grid[2]),
                .shared_unk1 = 0x2,
                .shared_shift = util_logbase2(single_size) - 1
        };

        vtp->postfix.shared_memory = panfrost_upload_transient(batch, &shared,
                                                               sizeof(shared));
}

static mali_ptr
panfrost_get_tex_desc(struct panfrost_batch *batch,
                      enum pipe_shader_type st,
                      struct panfrost_sampler_view *view)
{
        if (!view)
                return (mali_ptr) 0;

        struct pipe_sampler_view *pview = &view->base;
        struct panfrost_resource *rsrc = pan_resource(pview->texture);

        /* Add the BO to the job so it's retained until the job is done. */

        panfrost_batch_add_bo(batch, rsrc->bo,
                              PAN_BO_ACCESS_SHARED | PAN_BO_ACCESS_READ |
                              panfrost_bo_access_for_stage(st));

        panfrost_batch_add_bo(batch, view->midgard_bo,
                              PAN_BO_ACCESS_SHARED | PAN_BO_ACCESS_READ |
                              panfrost_bo_access_for_stage(st));

        return view->midgard_bo->gpu;
}

void
panfrost_emit_texture_descriptors(struct panfrost_batch *batch,
                                  enum pipe_shader_type stage,
                                  struct mali_vertex_tiler_postfix *postfix)
{
        struct panfrost_context *ctx = batch->ctx;
        struct panfrost_device *device = pan_device(ctx->base.screen);

        if (!ctx->sampler_view_count[stage])
                return;

        if (device->quirks & IS_BIFROST) {
                struct bifrost_texture_descriptor *descriptors;

                descriptors = malloc(sizeof(struct bifrost_texture_descriptor) *
                                     ctx->sampler_view_count[stage]);

                for (int i = 0; i < ctx->sampler_view_count[stage]; ++i) {
                        struct panfrost_sampler_view *view = ctx->sampler_views[stage][i];
                        struct pipe_sampler_view *pview = &view->base;
                        struct panfrost_resource *rsrc = pan_resource(pview->texture);

                        /* Add the BOs to the job so they are retained until the job is done. */

                        panfrost_batch_add_bo(batch, rsrc->bo,
                                              PAN_BO_ACCESS_SHARED | PAN_BO_ACCESS_READ |
                                              panfrost_bo_access_for_stage(stage));

                        panfrost_batch_add_bo(batch, view->bifrost_bo,
                                              PAN_BO_ACCESS_SHARED | PAN_BO_ACCESS_READ |
                                              panfrost_bo_access_for_stage(stage));

                        memcpy(&descriptors[i], view->bifrost_descriptor, sizeof(*view->bifrost_descriptor));
                }

                postfix->textures = panfrost_upload_transient(batch,
                                                              descriptors,
                                                              sizeof(struct bifrost_texture_descriptor) *
                                                                      ctx->sampler_view_count[stage]);

                free(descriptors);
        } else {
                uint64_t trampolines[PIPE_MAX_SHADER_SAMPLER_VIEWS];

                for (int i = 0; i < ctx->sampler_view_count[stage]; ++i)
                        trampolines[i] = panfrost_get_tex_desc(batch, stage,
                                                               ctx->sampler_views[stage][i]);

                postfix->textures = panfrost_upload_transient(batch,
                                                              trampolines,
                                                              sizeof(uint64_t) *
                                                              ctx->sampler_view_count[stage]);
        }
}

void
panfrost_emit_sampler_descriptors(struct panfrost_batch *batch,
                                  enum pipe_shader_type stage,
                                  struct mali_vertex_tiler_postfix *postfix)
{
        struct panfrost_context *ctx = batch->ctx;
        struct panfrost_device *device = pan_device(ctx->base.screen);

        if (!ctx->sampler_count[stage])
                return;

        if (device->quirks & IS_BIFROST) {
                size_t desc_size = sizeof(struct bifrost_sampler_descriptor);
                size_t transfer_size = desc_size * ctx->sampler_count[stage];
                struct panfrost_transfer transfer = panfrost_allocate_transient(batch,
                                                                                transfer_size);
                struct bifrost_sampler_descriptor *desc = (struct bifrost_sampler_descriptor *)transfer.cpu;

                for (int i = 0; i < ctx->sampler_count[stage]; ++i)
                        desc[i] = ctx->samplers[stage][i]->bifrost_hw;

                postfix->sampler_descriptor = transfer.gpu;
        } else {
                size_t desc_size = sizeof(struct mali_sampler_descriptor);
                size_t transfer_size = desc_size * ctx->sampler_count[stage];
                struct panfrost_transfer transfer = panfrost_allocate_transient(batch,
                                                                                transfer_size);
                struct mali_sampler_descriptor *desc = (struct mali_sampler_descriptor *)transfer.cpu;

                for (int i = 0; i < ctx->sampler_count[stage]; ++i)
                        desc[i] = ctx->samplers[stage][i]->midgard_hw;

                postfix->sampler_descriptor = transfer.gpu;
        }
}

void
panfrost_emit_vertex_attr_meta(struct panfrost_batch *batch,
                               struct mali_vertex_tiler_postfix *vertex_postfix)
{
        struct panfrost_context *ctx = batch->ctx;

        if (!ctx->vertex)
                return;

        struct panfrost_vertex_state *so = ctx->vertex;

        panfrost_vertex_state_upd_attr_offs(ctx, vertex_postfix);
        vertex_postfix->attribute_meta = panfrost_upload_transient(batch, so->hw,
                                                               sizeof(*so->hw) *
                                                               PAN_MAX_ATTRIBUTE);
}

void
panfrost_emit_vertex_data(struct panfrost_batch *batch,
                          struct mali_vertex_tiler_postfix *vertex_postfix)
{
        struct panfrost_context *ctx = batch->ctx;
        struct panfrost_vertex_state *so = ctx->vertex;

        /* Staged mali_attr, and index into them. i =/= k, depending on the
         * vertex buffer mask and instancing. Twice as much room is allocated,
         * for a worst case of NPOT_DIVIDEs which take up extra slot */
        union mali_attr attrs[PIPE_MAX_ATTRIBS * 2];
        unsigned k = 0;

        for (unsigned i = 0; i < so->num_elements; ++i) {
                /* We map a mali_attr to be 1:1 with the mali_attr_meta, which
                 * means duplicating some vertex buffers (who cares? aside from
                 * maybe some caching implications but I somehow doubt that
                 * matters) */

                struct pipe_vertex_element *elem = &so->pipe[i];
                unsigned vbi = elem->vertex_buffer_index;

                /* The exception to 1:1 mapping is that we can have multiple
                 * entries (NPOT divisors), so we fixup anyways */

                so->hw[i].index = k;

                if (!(ctx->vb_mask & (1 << vbi)))
                        continue;

                struct pipe_vertex_buffer *buf = &ctx->vertex_buffers[vbi];
                struct panfrost_resource *rsrc;

                rsrc = pan_resource(buf->buffer.resource);
                if (!rsrc)
                        continue;

                /* Align to 64 bytes by masking off the lower bits. This
                 * will be adjusted back when we fixup the src_offset in
                 * mali_attr_meta */

                mali_ptr raw_addr = rsrc->bo->gpu + buf->buffer_offset;
                mali_ptr addr = raw_addr & ~63;
                unsigned chopped_addr = raw_addr - addr;

                /* Add a dependency of the batch on the vertex buffer */
                panfrost_batch_add_bo(batch, rsrc->bo,
                                      PAN_BO_ACCESS_SHARED |
                                      PAN_BO_ACCESS_READ |
                                      PAN_BO_ACCESS_VERTEX_TILER);

                /* Set common fields */
                attrs[k].elements = addr;
                attrs[k].stride = buf->stride;

                /* Since we advanced the base pointer, we shrink the buffer
                 * size */
                attrs[k].size = rsrc->base.width0 - buf->buffer_offset;

                /* We need to add the extra size we masked off (for
                 * correctness) so the data doesn't get clamped away */
                attrs[k].size += chopped_addr;

                /* For non-instancing make sure we initialize */
                attrs[k].shift = attrs[k].extra_flags = 0;

                /* Instancing uses a dramatically different code path than
                 * linear, so dispatch for the actual emission now that the
                 * common code is finished */

                unsigned divisor = elem->instance_divisor;

                if (divisor && ctx->instance_count == 1) {
                        /* Silly corner case where there's a divisor(=1) but
                         * there's no legitimate instancing. So we want *every*
                         * attribute to be the same. So set stride to zero so
                         * we don't go anywhere. */

                        attrs[k].size = attrs[k].stride + chopped_addr;
                        attrs[k].stride = 0;
                        attrs[k++].elements |= MALI_ATTR_LINEAR;
                } else if (ctx->instance_count <= 1) {
                        /* Normal, non-instanced attributes */
                        attrs[k++].elements |= MALI_ATTR_LINEAR;
                } else {
                        unsigned instance_shift = vertex_postfix->instance_shift;
                        unsigned instance_odd = vertex_postfix->instance_odd;

                        k += panfrost_vertex_instanced(ctx->padded_count,
                                                       instance_shift,
                                                       instance_odd,
                                                       divisor, &attrs[k]);
                }
        }

        /* Add special gl_VertexID/gl_InstanceID buffers */

        panfrost_vertex_id(ctx->padded_count, &attrs[k]);
        so->hw[PAN_VERTEX_ID].index = k++;
        panfrost_instance_id(ctx->padded_count, &attrs[k]);
        so->hw[PAN_INSTANCE_ID].index = k++;

        /* Upload whatever we emitted and go */

        vertex_postfix->attributes = panfrost_upload_transient(batch, attrs,
                                                           k * sizeof(*attrs));
}

static mali_ptr
panfrost_emit_varyings(struct panfrost_batch *batch, union mali_attr *slot,
                       unsigned stride, unsigned count)
{
        /* Fill out the descriptor */
        slot->stride = stride;
        slot->size = stride * count;
        slot->shift = slot->extra_flags = 0;

        struct panfrost_transfer transfer = panfrost_allocate_transient(batch,
                                                                        slot->size);

        slot->elements = transfer.gpu | MALI_ATTR_LINEAR;

        return transfer.gpu;
}

static void
panfrost_emit_streamout(struct panfrost_batch *batch, union mali_attr *slot,
                        unsigned stride, unsigned offset, unsigned count,
                        struct pipe_stream_output_target *target)
{
        /* Fill out the descriptor */
        slot->stride = stride * 4;
        slot->shift = slot->extra_flags = 0;

        unsigned max_size = target->buffer_size;
        unsigned expected_size = slot->stride * count;

        slot->size = MIN2(max_size, expected_size);

        /* Grab the BO and bind it to the batch */
        struct panfrost_bo *bo = pan_resource(target->buffer)->bo;

        /* Varyings are WRITE from the perspective of the VERTEX but READ from
         * the perspective of the TILER and FRAGMENT.
         */
        panfrost_batch_add_bo(batch, bo,
                              PAN_BO_ACCESS_SHARED |
                              PAN_BO_ACCESS_RW |
                              PAN_BO_ACCESS_VERTEX_TILER |
                              PAN_BO_ACCESS_FRAGMENT);

        mali_ptr addr = bo->gpu + target->buffer_offset + (offset * slot->stride);
        slot->elements = addr;
}

/* Given a shader and buffer indices, link varying metadata together */

static bool
is_special_varying(gl_varying_slot loc)
{
        switch (loc) {
        case VARYING_SLOT_POS:
        case VARYING_SLOT_PSIZ:
        case VARYING_SLOT_PNTC:
        case VARYING_SLOT_FACE:
                return true;
        default:
                return false;
        }
}

static void
panfrost_emit_varying_meta(void *outptr, struct panfrost_shader_state *ss,
                           signed general, signed gl_Position,
                           signed gl_PointSize, signed gl_PointCoord,
                           signed gl_FrontFacing)
{
        struct mali_attr_meta *out = (struct mali_attr_meta *) outptr;

        for (unsigned i = 0; i < ss->varying_count; ++i) {
                gl_varying_slot location = ss->varyings_loc[i];
                int index = -1;

                switch (location) {
                case VARYING_SLOT_POS:
                        index = gl_Position;
                        break;
                case VARYING_SLOT_PSIZ:
                        index = gl_PointSize;
                        break;
                case VARYING_SLOT_PNTC:
                        index = gl_PointCoord;
                        break;
                case VARYING_SLOT_FACE:
                        index = gl_FrontFacing;
                        break;
                default:
                        index = general;
                        break;
                }

                assert(index >= 0);
                out[i].index = index;
        }
}

static bool
has_point_coord(unsigned mask, gl_varying_slot loc)
{
        if ((loc >= VARYING_SLOT_TEX0) && (loc <= VARYING_SLOT_TEX7))
                return (mask & (1 << (loc - VARYING_SLOT_TEX0)));
        else if (loc == VARYING_SLOT_PNTC)
                return (mask & (1 << 8));
        else
                return false;
}

/* Helpers for manipulating stream out information so we can pack varyings
 * accordingly. Compute the src_offset for a given captured varying */

static struct pipe_stream_output *
pan_get_so(struct pipe_stream_output_info *info, gl_varying_slot loc)
{
        for (unsigned i = 0; i < info->num_outputs; ++i) {
                if (info->output[i].register_index == loc)
                        return &info->output[i];
        }

        unreachable("Varying not captured");
}

void
panfrost_emit_varying_descriptor(struct panfrost_batch *batch,
                                 unsigned vertex_count,
                                 struct mali_vertex_tiler_postfix *vertex_postfix,
                                 struct mali_vertex_tiler_postfix *tiler_postfix,
                                 union midgard_primitive_size *primitive_size)
{
        /* Load the shaders */
        struct panfrost_context *ctx = batch->ctx;
        struct panfrost_shader_state *vs, *fs;
        unsigned int num_gen_varyings = 0;
        size_t vs_size, fs_size;

        /* Allocate the varying descriptor */

        vs = panfrost_get_shader_state(ctx, PIPE_SHADER_VERTEX);
        fs = panfrost_get_shader_state(ctx, PIPE_SHADER_FRAGMENT);
        vs_size = sizeof(struct mali_attr_meta) * vs->varying_count;
        fs_size = sizeof(struct mali_attr_meta) * fs->varying_count;

        struct panfrost_transfer trans = panfrost_allocate_transient(batch,
                                                                     vs_size +
                                                                     fs_size);

        struct pipe_stream_output_info *so = &vs->stream_output;

        /* Check if this varying is linked by us. This is the case for
         * general-purpose, non-captured varyings. If it is, link it. If it's
         * not, use the provided stream out information to determine the
         * offset, since it was already linked for us. */

        for (unsigned i = 0; i < vs->varying_count; i++) {
                gl_varying_slot loc = vs->varyings_loc[i];

                bool special = is_special_varying(loc);
                bool captured = ((vs->so_mask & (1ll << loc)) ? true : false);

                if (captured) {
                        struct pipe_stream_output *o = pan_get_so(so, loc);

                        unsigned dst_offset = o->dst_offset * 4; /* dwords */
                        vs->varyings[i].src_offset = dst_offset;
                } else if (!special) {
                        vs->varyings[i].src_offset = 16 * (num_gen_varyings++);
                }
        }

        /* Conversely, we need to set src_offset for the captured varyings.
         * Here, the layout is defined by the stream out info, not us */

        /* Link up with fragment varyings */
        bool reads_point_coord = fs->reads_point_coord;

        for (unsigned i = 0; i < fs->varying_count; i++) {
                gl_varying_slot loc = fs->varyings_loc[i];
                unsigned src_offset;
                signed vs_idx = -1;

                /* Link up */
                for (unsigned j = 0; j < vs->varying_count; ++j) {
                        if (vs->varyings_loc[j] == loc) {
                                vs_idx = j;
                                break;
                        }
                }

                /* Either assign or reuse */
                if (vs_idx >= 0)
                        src_offset = vs->varyings[vs_idx].src_offset;
                else
                        src_offset = 16 * (num_gen_varyings++);

                fs->varyings[i].src_offset = src_offset;

                if (has_point_coord(fs->point_sprite_mask, loc))
                        reads_point_coord = true;
        }

        memcpy(trans.cpu, vs->varyings, vs_size);
        memcpy(trans.cpu + vs_size, fs->varyings, fs_size);

        union mali_attr varyings[PIPE_MAX_ATTRIBS] = {0};

        /* Figure out how many streamout buffers could be bound */
        unsigned so_count = ctx->streamout.num_targets;
        for (unsigned i = 0; i < vs->varying_count; i++) {
                gl_varying_slot loc = vs->varyings_loc[i];

                bool captured = ((vs->so_mask & (1ll << loc)) ? true : false);
                if (!captured) continue;

                struct pipe_stream_output *o = pan_get_so(so, loc);
                so_count = MAX2(so_count, o->output_buffer + 1);
        }

        signed idx = so_count;
        signed general = idx++;
        signed gl_Position = idx++;
        signed gl_PointSize = vs->writes_point_size ? (idx++) : -1;
        signed gl_PointCoord = reads_point_coord ? (idx++) : -1;
        signed gl_FrontFacing = fs->reads_face ? (idx++) : -1;
        signed gl_FragCoord = fs->reads_frag_coord ? (idx++) : -1;

        /* Emit the stream out buffers */

        unsigned out_count = u_stream_outputs_for_vertices(ctx->active_prim,
                                                           ctx->vertex_count);

        for (unsigned i = 0; i < so_count; ++i) {
                if (i < ctx->streamout.num_targets) {
                        panfrost_emit_streamout(batch, &varyings[i],
                                                so->stride[i],
                                                ctx->streamout.offsets[i],
                                                out_count,
                                                ctx->streamout.targets[i]);
                } else {
                        /* Emit a dummy buffer */
                        panfrost_emit_varyings(batch, &varyings[i],
                                               so->stride[i] * 4,
                                               out_count);

                        /* Clear the attribute type */
                        varyings[i].elements &= ~0xF;
                }
        }

        panfrost_emit_varyings(batch, &varyings[general],
                               num_gen_varyings * 16,
                               vertex_count);

        mali_ptr varyings_p;

        /* fp32 vec4 gl_Position */
        varyings_p = panfrost_emit_varyings(batch, &varyings[gl_Position],
                                            sizeof(float) * 4, vertex_count);
        tiler_postfix->position_varying = varyings_p;


        if (panfrost_writes_point_size(ctx)) {
                varyings_p = panfrost_emit_varyings(batch,
                                                    &varyings[gl_PointSize],
                                                    2, vertex_count);
                primitive_size->pointer = varyings_p;
        }

        if (reads_point_coord)
                varyings[gl_PointCoord].elements = MALI_VARYING_POINT_COORD;

        if (fs->reads_face)
                varyings[gl_FrontFacing].elements = MALI_VARYING_FRONT_FACING;

        if (fs->reads_frag_coord)
                varyings[gl_FragCoord].elements = MALI_VARYING_FRAG_COORD;

        struct panfrost_device *device = pan_device(ctx->base.screen);
        assert(!(device->quirks & IS_BIFROST) || !(reads_point_coord));

        /* Let's go ahead and link varying meta to the buffer in question, now
         * that that information is available. VARYING_SLOT_POS is mapped to
         * gl_FragCoord for fragment shaders but gl_Positionf or vertex shaders
         * */

        panfrost_emit_varying_meta(trans.cpu, vs, general, gl_Position,
                                   gl_PointSize, gl_PointCoord,
                                   gl_FrontFacing);

        panfrost_emit_varying_meta(trans.cpu + vs_size, fs, general,
                                   gl_FragCoord, gl_PointSize,
                                   gl_PointCoord, gl_FrontFacing);

        /* Replace streamout */

        struct mali_attr_meta *ovs = (struct mali_attr_meta *)trans.cpu;
        struct mali_attr_meta *ofs = ovs + vs->varying_count;

        for (unsigned i = 0; i < vs->varying_count; i++) {
                gl_varying_slot loc = vs->varyings_loc[i];

                bool captured = ((vs->so_mask & (1ll << loc)) ? true : false);
                if (!captured)
                        continue;

                struct pipe_stream_output *o = pan_get_so(so, loc);
                ovs[i].index = o->output_buffer;

                assert(o->stream == 0);
                ovs[i].format = (vs->varyings[i].format & ~MALI_NR_CHANNELS(4))
                        | MALI_NR_CHANNELS(o->num_components);

                if (device->quirks & HAS_SWIZZLES)
                        ovs[i].swizzle = panfrost_get_default_swizzle(o->num_components);
                else
                        ovs[i].swizzle = panfrost_bifrost_swizzle(o->num_components);

                /* Link to the fragment */
                signed fs_idx = -1;

                /* Link up */
                for (unsigned j = 0; j < fs->varying_count; ++j) {
                        if (fs->varyings_loc[j] == loc) {
                                fs_idx = j;
                                break;
                        }
                }

                if (fs_idx >= 0) {
                        ofs[fs_idx].index = ovs[i].index;
                        ofs[fs_idx].format = ovs[i].format;
                        ofs[fs_idx].swizzle = ovs[i].swizzle;
                }
        }

        /* Replace point sprite */
        for (unsigned i = 0; i < fs->varying_count; i++) {
                /* If we have a point sprite replacement, handle that here. We
                 * have to translate location first.  TODO: Flip y in shader.
                 * We're already keying ... just time crunch .. */

                if (has_point_coord(fs->point_sprite_mask,
                                    fs->varyings_loc[i])) {
                        ofs[i].index = gl_PointCoord;

                        /* Swizzle out the z/w to 0/1 */
                        ofs[i].format = MALI_RG16F;
                        ofs[i].swizzle = panfrost_get_default_swizzle(2);
                }
        }

        /* Fix up unaligned addresses */
        for (unsigned i = 0; i < so_count; ++i) {
                if (varyings[i].elements < MALI_RECORD_SPECIAL)
                        continue;

                unsigned align = (varyings[i].elements & 63);

                /* While we're at it, the SO buffers are linear */

                if (!align) {
                        varyings[i].elements |= MALI_ATTR_LINEAR;
                        continue;
                }

                /* We need to adjust alignment */
                varyings[i].elements &= ~63;
                varyings[i].elements |= MALI_ATTR_LINEAR;
                varyings[i].size += align;

                for (unsigned v = 0; v < vs->varying_count; ++v) {
                        if (ovs[v].index != i)
                                continue;

                        ovs[v].src_offset = vs->varyings[v].src_offset + align;
                }

                for (unsigned f = 0; f < fs->varying_count; ++f) {
                        if (ofs[f].index != i)
                                continue;

                        ofs[f].src_offset = fs->varyings[f].src_offset + align;
                }
        }

        varyings_p = panfrost_upload_transient(batch, varyings,
                                               idx * sizeof(*varyings));
        vertex_postfix->varyings = varyings_p;
        tiler_postfix->varyings = varyings_p;

        vertex_postfix->varying_meta = trans.gpu;
        tiler_postfix->varying_meta = trans.gpu + vs_size;
}

void
panfrost_emit_vertex_tiler_jobs(struct panfrost_batch *batch,
                                struct mali_vertex_tiler_prefix *vertex_prefix,
                                struct mali_vertex_tiler_postfix *vertex_postfix,
                                struct mali_vertex_tiler_prefix *tiler_prefix,
                                struct mali_vertex_tiler_postfix *tiler_postfix,
                                union midgard_primitive_size *primitive_size)
{
        struct panfrost_context *ctx = batch->ctx;
        struct panfrost_device *device = pan_device(ctx->base.screen);
        bool wallpapering = ctx->wallpaper_batch && batch->tiler_dep;
        struct bifrost_payload_vertex bifrost_vertex = {0,};
        struct bifrost_payload_tiler bifrost_tiler = {0,};
        struct midgard_payload_vertex_tiler midgard_vertex = {0,};
        struct midgard_payload_vertex_tiler midgard_tiler = {0,};
        void *vp, *tp;
        size_t vp_size, tp_size;

        if (device->quirks & IS_BIFROST) {
                bifrost_vertex.prefix = *vertex_prefix;
                bifrost_vertex.postfix = *vertex_postfix;
                vp = &bifrost_vertex;
                vp_size = sizeof(bifrost_vertex);

                bifrost_tiler.prefix = *tiler_prefix;
                bifrost_tiler.tiler.primitive_size = *primitive_size;
                bifrost_tiler.tiler.tiler_meta = panfrost_batch_get_tiler_meta(batch, ~0);
                bifrost_tiler.postfix = *tiler_postfix;
                tp = &bifrost_tiler;
                tp_size = sizeof(bifrost_tiler);
        } else {
                midgard_vertex.prefix = *vertex_prefix;
                midgard_vertex.postfix = *vertex_postfix;
                vp = &midgard_vertex;
                vp_size = sizeof(midgard_vertex);

                midgard_tiler.prefix = *tiler_prefix;
                midgard_tiler.postfix = *tiler_postfix;
                midgard_tiler.primitive_size = *primitive_size;
                tp = &midgard_tiler;
                tp_size = sizeof(midgard_tiler);
        }

        if (wallpapering) {
                /* Inject in reverse order, with "predicted" job indices.
                 * THIS IS A HACK XXX */
                panfrost_new_job(batch, JOB_TYPE_TILER, false,
                                 batch->job_index + 2, tp, tp_size, true);
                panfrost_new_job(batch, JOB_TYPE_VERTEX, false, 0,
                                 vp, vp_size, true);
                return;
        }

        /* If rasterizer discard is enable, only submit the vertex */

        bool rasterizer_discard = ctx->rasterizer &&
                                  ctx->rasterizer->base.rasterizer_discard;

        unsigned vertex = panfrost_new_job(batch, JOB_TYPE_VERTEX, false, 0,
                                           vp, vp_size, false);

        if (rasterizer_discard)
                return;

        panfrost_new_job(batch, JOB_TYPE_TILER, false, vertex, tp, tp_size,
                         false);
}

/* TODO: stop hardcoding this */
mali_ptr
panfrost_emit_sample_locations(struct panfrost_batch *batch)
{
        uint16_t locations[] = {
            128, 128,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            0, 256,
            128, 128,
            0, 0,
            0, 0,
            0, 0,
            0, 0,
            0, 0,
            0, 0,
            0, 0,
            0, 0,
            0, 0,
            0, 0,
            0, 0,
            0, 0,
            0, 0,
            0, 0,
            0, 0,
        };

        return panfrost_upload_transient(batch, locations, 96 * sizeof(uint16_t));
}
