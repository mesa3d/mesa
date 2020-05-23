/*
 * Copyright (C) 2016 Rob Clark <robclark@freedesktop.org>
 * Copyright © 2018 Google, Inc.
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
 *
 * Authors:
 *    Rob Clark <robclark@freedesktop.org>
 */

#include "freedreno_query_acc.h"

#include "fd6_context.h"
#include "fd6_compute.h"
#include "fd6_blend.h"
#include "fd6_blitter.h"
#include "fd6_draw.h"
#include "fd6_emit.h"
#include "fd6_gmem.h"
#include "fd6_image.h"
#include "fd6_program.h"
#include "fd6_query.h"
#include "fd6_rasterizer.h"
#include "fd6_texture.h"
#include "fd6_zsa.h"

static void
fd6_context_destroy(struct pipe_context *pctx)
{
	struct fd6_context *fd6_ctx = fd6_context(fd_context(pctx));

	u_upload_destroy(fd6_ctx->border_color_uploader);

	fd_context_destroy(pctx);

	if (fd6_ctx->vsc_draw_strm)
		fd_bo_del(fd6_ctx->vsc_draw_strm);
	if (fd6_ctx->vsc_prim_strm)
		fd_bo_del(fd6_ctx->vsc_prim_strm);
	fd_bo_del(fd6_ctx->control_mem);

	fd_context_cleanup_common_vbos(&fd6_ctx->base);

	ir3_cache_destroy(fd6_ctx->shader_cache);

	fd6_texture_fini(pctx);

	free(fd6_ctx);
}

static const uint8_t primtypes[] = {
		[PIPE_PRIM_POINTS]                      = DI_PT_POINTLIST,
		[PIPE_PRIM_LINES]                       = DI_PT_LINELIST,
		[PIPE_PRIM_LINE_STRIP]                  = DI_PT_LINESTRIP,
		[PIPE_PRIM_LINE_LOOP]                   = DI_PT_LINELOOP,
		[PIPE_PRIM_TRIANGLES]                   = DI_PT_TRILIST,
		[PIPE_PRIM_TRIANGLE_STRIP]              = DI_PT_TRISTRIP,
		[PIPE_PRIM_TRIANGLE_FAN]                = DI_PT_TRIFAN,
		[PIPE_PRIM_LINES_ADJACENCY]             = DI_PT_LINE_ADJ,
		[PIPE_PRIM_LINE_STRIP_ADJACENCY]        = DI_PT_LINESTRIP_ADJ,
		[PIPE_PRIM_TRIANGLES_ADJACENCY]         = DI_PT_TRI_ADJ,
		[PIPE_PRIM_TRIANGLE_STRIP_ADJACENCY]    = DI_PT_TRISTRIP_ADJ,
		[PIPE_PRIM_PATCHES]                     = DI_PT_PATCHES0,
		[PIPE_PRIM_MAX]                         = DI_PT_RECTLIST,  /* internal clear blits */
};

struct pipe_context *
fd6_context_create(struct pipe_screen *pscreen, void *priv, unsigned flags)
{
	struct fd_screen *screen = fd_screen(pscreen);
	struct fd6_context *fd6_ctx = CALLOC_STRUCT(fd6_context);
	struct pipe_context *pctx;

	if (!fd6_ctx)
		return NULL;


	switch (screen->gpu_id) {
	case 618:
/*
GRAS_BIN_CONTROL:
RB_BIN_CONTROL:
  - a618 doesn't appear to set .USE_VIZ; also bin size diffs

RB_CCU_CNTL:
  - 0x3c400004 -> 0x3e400004
  - 0x10000000 -> 0x08000000

RB_UNKNOWN_8E04:               <-- see stencil-0000.rd.gz
  - 0x01000000 -> 0x00100000

SP_UNKNOWN_A0F8:
PC_UNKNOWN_9805:
  - 0x1 -> 0
 */
		fd6_ctx->magic.RB_UNKNOWN_8E04_blit = 0x00100000;
		fd6_ctx->magic.RB_CCU_CNTL_gmem = A6XX_RB_CCU_CNTL_OFFSET(0x7c000) |
										  A6XX_RB_CCU_CNTL_GMEM |
										  A6XX_RB_CCU_CNTL_UNK2;
		fd6_ctx->magic.RB_CCU_CNTL_bypass = A6XX_RB_CCU_CNTL_OFFSET(0x10000);
		fd6_ctx->magic.PC_UNKNOWN_9805 = 0x0;
		fd6_ctx->magic.SP_UNKNOWN_A0F8 = 0x0;
		break;
	case 630:
		fd6_ctx->magic.RB_UNKNOWN_8E04_blit = 0x01000000;
		fd6_ctx->magic.RB_CCU_CNTL_gmem = A6XX_RB_CCU_CNTL_OFFSET(0xf8000) |
										  A6XX_RB_CCU_CNTL_GMEM |
										  A6XX_RB_CCU_CNTL_UNK2;
		fd6_ctx->magic.RB_CCU_CNTL_bypass = A6XX_RB_CCU_CNTL_OFFSET(0x20000);
		fd6_ctx->magic.PC_UNKNOWN_9805 = 0x1;
		fd6_ctx->magic.SP_UNKNOWN_A0F8 = 0x1;
		break;
	case 640:
		fd6_ctx->magic.RB_UNKNOWN_8E04_blit = 0x00100000;
		fd6_ctx->magic.RB_CCU_CNTL_gmem = A6XX_RB_CCU_CNTL_OFFSET(0xf8000) |
										  A6XX_RB_CCU_CNTL_GMEM;
		fd6_ctx->magic.RB_CCU_CNTL_bypass = A6XX_RB_CCU_CNTL_OFFSET(0x20000);
		fd6_ctx->magic.PC_UNKNOWN_9805 = 0x1;
		fd6_ctx->magic.SP_UNKNOWN_A0F8 = 0x1;
		break;
	case 650:
		fd6_ctx->magic.RB_UNKNOWN_8E04_blit = 0x04100000;
		fd6_ctx->magic.RB_CCU_CNTL_gmem = A6XX_RB_CCU_CNTL_OFFSET(0x114000) |
										  A6XX_RB_CCU_CNTL_GMEM;
		fd6_ctx->magic.RB_CCU_CNTL_bypass = A6XX_RB_CCU_CNTL_OFFSET(0x30000);
		fd6_ctx->magic.PC_UNKNOWN_9805 = 0x2;
		fd6_ctx->magic.SP_UNKNOWN_A0F8 = 0x2;
		break;
	default:
		unreachable("missing magic config");
	}

	pctx = &fd6_ctx->base.base;
	pctx->screen = pscreen;

	fd6_ctx->base.dev = fd_device_ref(screen->dev);
	fd6_ctx->base.screen = fd_screen(pscreen);

	pctx->destroy = fd6_context_destroy;
	pctx->create_blend_state = fd6_blend_state_create;
	pctx->create_rasterizer_state = fd6_rasterizer_state_create;
	pctx->create_depth_stencil_alpha_state = fd6_zsa_state_create;

	fd6_draw_init(pctx);
	fd6_compute_init(pctx);
	fd6_gmem_init(pctx);
	fd6_texture_init(pctx);
	fd6_prog_init(pctx);
	fd6_emit_init(pctx);
	fd6_query_context_init(pctx);

	pctx = fd_context_init(&fd6_ctx->base, pscreen, primtypes, priv, flags);
	if (!pctx)
		return NULL;

	/* after fd_context_init() to override set_shader_images() */
	fd6_image_init(pctx);

	util_blitter_set_texture_multisample(fd6_ctx->base.blitter, true);

	/* fd_context_init overwrites delete_rasterizer_state, so set this
	 * here. */
	pctx->delete_rasterizer_state = fd6_rasterizer_state_delete;
	pctx->delete_blend_state = fd6_blend_state_delete;
	pctx->delete_depth_stencil_alpha_state = fd6_depth_stencil_alpha_state_delete;

	/* initial sizes for VSC buffers (or rather the per-pipe sizes
	 * which is used to derive entire buffer size:
	 */
	fd6_ctx->vsc_draw_strm_pitch = 0x440;
	fd6_ctx->vsc_prim_strm_pitch = 0x1040;

	fd6_ctx->control_mem = fd_bo_new(screen->dev, 0x1000,
			DRM_FREEDRENO_GEM_TYPE_KMEM, "control");

	fd_context_setup_common_vbos(&fd6_ctx->base);

	fd6_blitter_init(pctx);

	fd6_ctx->border_color_uploader = u_upload_create(pctx, 4096, 0,
                                                         PIPE_USAGE_STREAM, 0);

	return pctx;
}
