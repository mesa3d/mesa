/*
 * © Copyright 2018 Alyssa Rosenzweig
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
 */

#include <stdio.h>
#include "pan_blend_shaders.h"
#include "pan_util.h"
#include "midgard/midgard_compile.h"
#include "compiler/nir/nir_builder.h"
#include "nir/nir_lower_blend.h"
#include "gallium/auxiliary/util/u_blend.h"
#include "util/u_memory.h"

/*
 * Implements the command stream portion of programmatic blend shaders.
 *
 * On Midgard, common blending operations are accelerated by the fixed-function
 * blending pipeline. Panfrost supports this fast path via the code in
 * pan_blending.c. Nevertheless, uncommon blend modes (including some seemingly
 * simple modes present in ES2) require "blend shaders", a special internal
 * shader type used for programmable blending.
 *
 * Blend shaders operate during the normal blending time, but they bypass the
 * fixed-function blending pipeline and instead go straight to the Midgard
 * shader cores. The shaders themselves are essentially just fragment shaders,
 * making heavy use of uint8 arithmetic to manipulate RGB values for the
 * framebuffer.
 *
 * As is typical with Midgard, shader binaries must be accompanied by
 * information about the first tag (ORed with the bottom nibble of address,
 * like usual) and work registers. Work register count is assumed to be less
 * than or equal to the coresponding fragment shader's work count. This
 * suggests that blend shader invocation is tied to fragment shader
 * execution.
 *
 * The shaders themselves use the standard ISA. The source pixel colour,
 * including alpha, is preloaded into r0 as a vec4 of float32. The destination
 * pixel colour must be loaded explicitly via load/store ops, possibly
 * performing conversions in software. The blended colour must be stored with a
 * fragment writeout in the correct framebuffer format, either in software or
 * via conversion opcodes on the load/store pipe.
 *
 * Blend shaders hardcode constants. Naively, this requires recompilation each
 * time the blend color changes, which is a performance risk. Accordingly, we
 * 'cheat' a bit: instead of loading the constant, we compile a shader with a
 * dummy constant, exporting the offset to the immediate in the shader binary,
 * storing this generic binary and metadata in the CSO itself at CSO create
 * time.
 *
 * We then hot patch in the color into this shader at attachment / color change
 * time, allowing for CSO create to be the only expensive operation
 * (compilation).
 */

static nir_lower_blend_options
nir_make_options(const struct pipe_blend_state *blend, unsigned i)
{
        nir_lower_blend_options options;

        if (blend->logicop_enable) {
            options.logicop_enable = true;
            options.logicop_func = blend->logicop_func;
            return options;
        }

        options.logicop_enable = false;

        if (!blend->independent_blend_enable)
                i = 0;

        /* If blend is disabled, we just use replace mode */

        nir_lower_blend_channel rgb = {
                .func = BLEND_FUNC_ADD,
                .src_factor = BLEND_FACTOR_ZERO,
                .invert_src_factor = true,
                .dst_factor = BLEND_FACTOR_ZERO,
                .invert_dst_factor = false
        };

        nir_lower_blend_channel alpha = rgb;

        if (blend->rt[i].blend_enable) {
                rgb.func = util_blend_func_to_shader(blend->rt[i].rgb_func);
                rgb.src_factor = util_blend_factor_to_shader(blend->rt[i].rgb_src_factor);
                rgb.dst_factor = util_blend_factor_to_shader(blend->rt[i].rgb_dst_factor);
                rgb.invert_src_factor = util_blend_factor_is_inverted(blend->rt[i].rgb_src_factor);
                rgb.invert_dst_factor = util_blend_factor_is_inverted(blend->rt[i].rgb_dst_factor);

                alpha.func = util_blend_func_to_shader(blend->rt[i].alpha_func);
                alpha.src_factor = util_blend_factor_to_shader(blend->rt[i].alpha_src_factor);
                alpha.dst_factor = util_blend_factor_to_shader(blend->rt[i].alpha_dst_factor);
                alpha.invert_src_factor = util_blend_factor_is_inverted(blend->rt[i].alpha_src_factor);
                alpha.invert_dst_factor = util_blend_factor_is_inverted(blend->rt[i].alpha_dst_factor);
        }

        options.rgb = rgb;
        options.alpha = alpha;

        options.colormask = blend->rt[i].colormask;

        return options;
}

struct panfrost_blend_shader
panfrost_compile_blend_shader(
        struct panfrost_context *ctx,
        struct pipe_blend_state *cso,
        enum pipe_format format,
        unsigned rt)
{
        struct panfrost_device *dev = pan_device(ctx->base.screen);
        struct panfrost_blend_shader res;

        res.ctx = ctx;

        /* Build the shader */

        nir_shader *shader = nir_shader_create(NULL, MESA_SHADER_FRAGMENT, &midgard_nir_options, NULL);
        nir_function *fn = nir_function_create(shader, "main");
        nir_function_impl *impl = nir_function_impl_create(fn);

        /* Create the blend variables */

        nir_variable *c_src = nir_variable_create(shader, nir_var_shader_in, glsl_vector_type(GLSL_TYPE_FLOAT, 4), "gl_Color");
        nir_variable *c_out = nir_variable_create(shader, nir_var_shader_out, glsl_vector_type(GLSL_TYPE_FLOAT, 4), "gl_FragColor");

        c_src->data.location = VARYING_SLOT_COL0;
        c_out->data.location = FRAG_RESULT_COLOR;

        /* Setup nir_builder */

        nir_builder _b;
        nir_builder *b = &_b;
        nir_builder_init(b, impl);
        b->cursor = nir_before_block(nir_start_block(impl));

        /* Setup inputs */

        nir_ssa_def *s_src = nir_load_var(b, c_src);

        /* Build a trivial blend shader */
        nir_store_var(b, c_out, s_src, 0xFF);

        nir_lower_blend_options options =
                nir_make_options(cso, rt);
        options.format = format;

        NIR_PASS_V(shader, nir_lower_blend, options);

        NIR_PASS_V(shader, nir_lower_framebuffer, format, dev->gpu_id);

        /* Compile the built shader */

        panfrost_program program;
        midgard_compile_shader_nir(shader, &program, true, rt, dev->gpu_id, false);

        /* Allow us to patch later */
        res.patch_index = program.blend_patch_offset;
        res.first_tag = program.first_tag;
        res.size = program.compiled.size;
        res.buffer = program.compiled.data;

        return res;
}
