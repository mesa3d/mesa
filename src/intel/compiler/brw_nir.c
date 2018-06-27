/*
 * Copyright © 2014 Intel Corporation
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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include "brw_nir.h"
#include "brw_shader.h"
#include "common/gen_debug.h"
#include "compiler/glsl_types.h"
#include "compiler/nir/nir_builder.h"
#include "util/u_math.h"

static bool
is_input(nir_intrinsic_instr *intrin)
{
   return intrin->intrinsic == nir_intrinsic_load_input ||
          intrin->intrinsic == nir_intrinsic_load_per_vertex_input ||
          intrin->intrinsic == nir_intrinsic_load_interpolated_input;
}

static bool
is_output(nir_intrinsic_instr *intrin)
{
   return intrin->intrinsic == nir_intrinsic_load_output ||
          intrin->intrinsic == nir_intrinsic_load_per_vertex_output ||
          intrin->intrinsic == nir_intrinsic_store_output ||
          intrin->intrinsic == nir_intrinsic_store_per_vertex_output;
}

/**
 * In many cases, we just add the base and offset together, so there's no
 * reason to keep them separate.  Sometimes, combining them is essential:
 * if a shader only accesses part of a compound variable (such as a matrix
 * or array), the variable's base may not actually exist in the VUE map.
 *
 * This pass adds constant offsets to instr->const_index[0], and resets
 * the offset source to 0.  Non-constant offsets remain unchanged - since
 * we don't know what part of a compound variable is accessed, we allocate
 * storage for the entire thing.
 */

static bool
add_const_offset_to_base_block(nir_block *block, nir_builder *b,
                               nir_variable_mode mode)
{
   nir_foreach_instr_safe(instr, block) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

      if ((mode == nir_var_shader_in && is_input(intrin)) ||
          (mode == nir_var_shader_out && is_output(intrin))) {
         nir_src *offset = nir_get_io_offset_src(intrin);
         nir_const_value *const_offset = nir_src_as_const_value(*offset);

         if (const_offset) {
            intrin->const_index[0] += const_offset->u32[0];
            b->cursor = nir_before_instr(&intrin->instr);
            nir_instr_rewrite_src(&intrin->instr, offset,
                                  nir_src_for_ssa(nir_imm_int(b, 0)));
         }
      }
   }
   return true;
}

static void
add_const_offset_to_base(nir_shader *nir, nir_variable_mode mode)
{
   nir_foreach_function(f, nir) {
      if (f->impl) {
         nir_builder b;
         nir_builder_init(&b, f->impl);
         nir_foreach_block(block, f->impl) {
            add_const_offset_to_base_block(block, &b, mode);
         }
      }
   }
}

static bool
remap_tess_levels(nir_builder *b, nir_intrinsic_instr *intr,
                  GLenum primitive_mode)
{
   const int location = nir_intrinsic_base(intr);
   const unsigned component = nir_intrinsic_component(intr);
   bool out_of_bounds;

   if (location == VARYING_SLOT_TESS_LEVEL_INNER) {
      switch (primitive_mode) {
      case GL_QUADS:
         /* gl_TessLevelInner[0..1] lives at DWords 3-2 (reversed). */
         nir_intrinsic_set_base(intr, 0);
         nir_intrinsic_set_component(intr, 3 - component);
         out_of_bounds = false;
         break;
      case GL_TRIANGLES:
         /* gl_TessLevelInner[0] lives at DWord 4. */
         nir_intrinsic_set_base(intr, 1);
         out_of_bounds = component > 0;
         break;
      case GL_ISOLINES:
         out_of_bounds = true;
         break;
      default:
         unreachable("Bogus tessellation domain");
      }
   } else if (location == VARYING_SLOT_TESS_LEVEL_OUTER) {
      if (primitive_mode == GL_ISOLINES) {
         /* gl_TessLevelOuter[0..1] lives at DWords 6-7 (in order). */
         nir_intrinsic_set_base(intr, 1);
         nir_intrinsic_set_component(intr, 2 + nir_intrinsic_component(intr));
         out_of_bounds = component > 1;
      } else {
         /* Triangles use DWords 7-5 (reversed); Quads use 7-4 (reversed) */
         nir_intrinsic_set_base(intr, 1);
         nir_intrinsic_set_component(intr, 3 - nir_intrinsic_component(intr));
         out_of_bounds = component == 3 && primitive_mode == GL_TRIANGLES;
      }
   } else {
      return false;
   }

   if (out_of_bounds) {
      if (nir_intrinsic_infos[intr->intrinsic].has_dest) {
         b->cursor = nir_before_instr(&intr->instr);
         nir_ssa_def *undef = nir_ssa_undef(b, 1, 32);
         nir_ssa_def_rewrite_uses(&intr->dest.ssa, nir_src_for_ssa(undef));
      }
      nir_instr_remove(&intr->instr);
   }

   return true;
}

static bool
remap_patch_urb_offsets(nir_block *block, nir_builder *b,
                        const struct brw_vue_map *vue_map,
                        GLenum tes_primitive_mode)
{
   const bool is_passthrough_tcs = b->shader->info.name &&
      strcmp(b->shader->info.name, "passthrough") == 0;

   nir_foreach_instr_safe(instr, block) {
      if (instr->type != nir_instr_type_intrinsic)
         continue;

      nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

      gl_shader_stage stage = b->shader->info.stage;

      if ((stage == MESA_SHADER_TESS_CTRL && is_output(intrin)) ||
          (stage == MESA_SHADER_TESS_EVAL && is_input(intrin))) {

         if (!is_passthrough_tcs &&
             remap_tess_levels(b, intrin, tes_primitive_mode))
            continue;

         int vue_slot = vue_map->varying_to_slot[intrin->const_index[0]];
         assert(vue_slot != -1);
         intrin->const_index[0] = vue_slot;

         nir_src *vertex = nir_get_io_vertex_index_src(intrin);
         if (vertex) {
            nir_const_value *const_vertex = nir_src_as_const_value(*vertex);
            if (const_vertex) {
               intrin->const_index[0] += const_vertex->u32[0] *
                                         vue_map->num_per_vertex_slots;
            } else {
               b->cursor = nir_before_instr(&intrin->instr);

               /* Multiply by the number of per-vertex slots. */
               nir_ssa_def *vertex_offset =
                  nir_imul(b,
                           nir_ssa_for_src(b, *vertex, 1),
                           nir_imm_int(b,
                                       vue_map->num_per_vertex_slots));

               /* Add it to the existing offset */
               nir_src *offset = nir_get_io_offset_src(intrin);
               nir_ssa_def *total_offset =
                  nir_iadd(b, vertex_offset,
                           nir_ssa_for_src(b, *offset, 1));

               nir_instr_rewrite_src(&intrin->instr, offset,
                                     nir_src_for_ssa(total_offset));
            }
         }
      }
   }
   return true;
}

void
brw_nir_lower_vs_inputs(nir_shader *nir,
                        const uint8_t *vs_attrib_wa_flags)
{
   /* Start with the location of the variable's base. */
   foreach_list_typed(nir_variable, var, node, &nir->inputs) {
      var->data.driver_location = var->data.location;
   }

   /* Now use nir_lower_io to walk dereference chains.  Attribute arrays are
    * loaded as one vec4 or dvec4 per element (or matrix column), depending on
    * whether it is a double-precision type or not.
    */
   nir_lower_io(nir, nir_var_shader_in, type_size_vec4, 0);

   /* This pass needs actual constants */
   nir_opt_constant_folding(nir);

   add_const_offset_to_base(nir, nir_var_shader_in);

   brw_nir_apply_attribute_workarounds(nir, vs_attrib_wa_flags);

   /* The last step is to remap VERT_ATTRIB_* to actual registers */

   /* Whether or not we have any system generated values.  gl_DrawID is not
    * included here as it lives in its own vec4.
    */
   const bool has_sgvs =
      nir->info.system_values_read &
      (BITFIELD64_BIT(SYSTEM_VALUE_FIRST_VERTEX) |
       BITFIELD64_BIT(SYSTEM_VALUE_BASE_INSTANCE) |
       BITFIELD64_BIT(SYSTEM_VALUE_VERTEX_ID_ZERO_BASE) |
       BITFIELD64_BIT(SYSTEM_VALUE_INSTANCE_ID));

   const unsigned num_inputs = util_bitcount64(nir->info.inputs_read);

   nir_foreach_function(function, nir) {
      if (!function->impl)
         continue;

      nir_builder b;
      nir_builder_init(&b, function->impl);

      nir_foreach_block(block, function->impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

            switch (intrin->intrinsic) {
            case nir_intrinsic_load_first_vertex:
            case nir_intrinsic_load_base_instance:
            case nir_intrinsic_load_vertex_id_zero_base:
            case nir_intrinsic_load_instance_id:
            case nir_intrinsic_load_is_indexed_draw:
            case nir_intrinsic_load_draw_id: {
               b.cursor = nir_after_instr(&intrin->instr);

               /* gl_VertexID and friends are stored by the VF as the last
                * vertex element.  We convert them to load_input intrinsics at
                * the right location.
                */
               nir_intrinsic_instr *load =
                  nir_intrinsic_instr_create(nir, nir_intrinsic_load_input);
               load->src[0] = nir_src_for_ssa(nir_imm_int(&b, 0));

               nir_intrinsic_set_base(load, num_inputs);
               switch (intrin->intrinsic) {
               case nir_intrinsic_load_first_vertex:
                  nir_intrinsic_set_component(load, 0);
                  break;
               case nir_intrinsic_load_base_instance:
                  nir_intrinsic_set_component(load, 1);
                  break;
               case nir_intrinsic_load_vertex_id_zero_base:
                  nir_intrinsic_set_component(load, 2);
                  break;
               case nir_intrinsic_load_instance_id:
                  nir_intrinsic_set_component(load, 3);
                  break;
               case nir_intrinsic_load_draw_id:
               case nir_intrinsic_load_is_indexed_draw:
                  /* gl_DrawID and IsIndexedDraw are stored right after
                   * gl_VertexID and friends if any of them exist.
                   */
                  nir_intrinsic_set_base(load, num_inputs + has_sgvs);
                  if (intrin->intrinsic == nir_intrinsic_load_draw_id)
                     nir_intrinsic_set_component(load, 0);
                  else
                     nir_intrinsic_set_component(load, 1);
                  break;
               default:
                  unreachable("Invalid system value intrinsic");
               }

               load->num_components = 1;
               nir_ssa_dest_init(&load->instr, &load->dest, 1, 32, NULL);
               nir_builder_instr_insert(&b, &load->instr);

               nir_ssa_def_rewrite_uses(&intrin->dest.ssa,
                                        nir_src_for_ssa(&load->dest.ssa));
               nir_instr_remove(&intrin->instr);
               break;
            }

            case nir_intrinsic_load_input: {
               /* Attributes come in a contiguous block, ordered by their
                * gl_vert_attrib value.  That means we can compute the slot
                * number for an attribute by masking out the enabled attributes
                * before it and counting the bits.
                */
               int attr = nir_intrinsic_base(intrin);
               int slot = util_bitcount64(nir->info.inputs_read &
                                            BITFIELD64_MASK(attr));
               nir_intrinsic_set_base(intrin, slot);
               break;
            }

            default:
               break; /* Nothing to do */
            }
         }
      }
   }
}

void
brw_nir_lower_vue_inputs(nir_shader *nir,
                         const struct brw_vue_map *vue_map)
{
   foreach_list_typed(nir_variable, var, node, &nir->inputs) {
      var->data.driver_location = var->data.location;
   }

   /* Inputs are stored in vec4 slots, so use type_size_vec4(). */
   nir_lower_io(nir, nir_var_shader_in, type_size_vec4, 0);

   /* This pass needs actual constants */
   nir_opt_constant_folding(nir);

   add_const_offset_to_base(nir, nir_var_shader_in);

   nir_foreach_function(function, nir) {
      if (!function->impl)
         continue;

      nir_foreach_block(block, function->impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);

            if (intrin->intrinsic == nir_intrinsic_load_input ||
                intrin->intrinsic == nir_intrinsic_load_per_vertex_input) {
               /* Offset 0 is the VUE header, which contains
                * VARYING_SLOT_LAYER [.y], VARYING_SLOT_VIEWPORT [.z], and
                * VARYING_SLOT_PSIZ [.w].
                */
               int varying = nir_intrinsic_base(intrin);
               int vue_slot;
               switch (varying) {
               case VARYING_SLOT_PSIZ:
                  nir_intrinsic_set_base(intrin, 0);
                  nir_intrinsic_set_component(intrin, 3);
                  break;

               default:
                  vue_slot = vue_map->varying_to_slot[varying];
                  assert(vue_slot != -1);
                  nir_intrinsic_set_base(intrin, vue_slot);
                  break;
               }
            }
         }
      }
   }
}

void
brw_nir_lower_tes_inputs(nir_shader *nir, const struct brw_vue_map *vue_map)
{
   foreach_list_typed(nir_variable, var, node, &nir->inputs) {
      var->data.driver_location = var->data.location;
   }

   nir_lower_io(nir, nir_var_shader_in, type_size_vec4, 0);

   /* This pass needs actual constants */
   nir_opt_constant_folding(nir);

   add_const_offset_to_base(nir, nir_var_shader_in);

   nir_foreach_function(function, nir) {
      if (function->impl) {
         nir_builder b;
         nir_builder_init(&b, function->impl);
         nir_foreach_block(block, function->impl) {
            remap_patch_urb_offsets(block, &b, vue_map,
                                    nir->info.tess.primitive_mode);
         }
      }
   }
}

void
brw_nir_lower_fs_inputs(nir_shader *nir,
                        const struct gen_device_info *devinfo,
                        const struct brw_wm_prog_key *key)
{
   foreach_list_typed(nir_variable, var, node, &nir->inputs) {
      var->data.driver_location = var->data.location;

      /* Apply default interpolation mode.
       *
       * Everything defaults to smooth except for the legacy GL color
       * built-in variables, which might be flat depending on API state.
       */
      if (var->data.interpolation == INTERP_MODE_NONE) {
         const bool flat = key->flat_shade &&
            (var->data.location == VARYING_SLOT_COL0 ||
             var->data.location == VARYING_SLOT_COL1);

         var->data.interpolation = flat ? INTERP_MODE_FLAT
                                        : INTERP_MODE_SMOOTH;
      }

      /* On Ironlake and below, there is only one interpolation mode.
       * Centroid interpolation doesn't mean anything on this hardware --
       * there is no multisampling.
       */
      if (devinfo->gen < 6) {
         var->data.centroid = false;
         var->data.sample = false;
      }
   }

   nir_lower_io_options lower_io_options = 0;
   if (key->persample_interp)
      lower_io_options |= nir_lower_io_force_sample_interpolation;

   nir_lower_io(nir, nir_var_shader_in, type_size_vec4, lower_io_options);

   /* This pass needs actual constants */
   nir_opt_constant_folding(nir);

   add_const_offset_to_base(nir, nir_var_shader_in);
}

void
brw_nir_lower_vue_outputs(nir_shader *nir)
{
   nir_foreach_variable(var, &nir->outputs) {
      var->data.driver_location = var->data.location;
   }

   nir_lower_io(nir, nir_var_shader_out, type_size_vec4, 0);
}

void
brw_nir_lower_tcs_outputs(nir_shader *nir, const struct brw_vue_map *vue_map,
                          GLenum tes_primitive_mode)
{
   nir_foreach_variable(var, &nir->outputs) {
      var->data.driver_location = var->data.location;
   }

   nir_lower_io(nir, nir_var_shader_out, type_size_vec4, 0);

   /* This pass needs actual constants */
   nir_opt_constant_folding(nir);

   add_const_offset_to_base(nir, nir_var_shader_out);

   nir_foreach_function(function, nir) {
      if (function->impl) {
         nir_builder b;
         nir_builder_init(&b, function->impl);
         nir_foreach_block(block, function->impl) {
            remap_patch_urb_offsets(block, &b, vue_map, tes_primitive_mode);
         }
      }
   }
}

void
brw_nir_lower_fs_outputs(nir_shader *nir)
{
   nir_foreach_variable(var, &nir->outputs) {
      var->data.driver_location =
         SET_FIELD(var->data.index, BRW_NIR_FRAG_OUTPUT_INDEX) |
         SET_FIELD(var->data.location, BRW_NIR_FRAG_OUTPUT_LOCATION);
   }

   nir_lower_io(nir, nir_var_shader_out, type_size_dvec4, 0);
}

#define OPT(pass, ...) ({                                  \
   bool this_progress = false;                             \
   NIR_PASS(this_progress, nir, pass, ##__VA_ARGS__);      \
   if (this_progress)                                      \
      progress = true;                                     \
   this_progress;                                          \
})

static nir_variable_mode
brw_nir_no_indirect_mask(const struct brw_compiler *compiler,
                         gl_shader_stage stage)
{
   nir_variable_mode indirect_mask = 0;

   if (compiler->glsl_compiler_options[stage].EmitNoIndirectInput)
      indirect_mask |= nir_var_shader_in;
   if (compiler->glsl_compiler_options[stage].EmitNoIndirectOutput)
      indirect_mask |= nir_var_shader_out;
   if (compiler->glsl_compiler_options[stage].EmitNoIndirectTemp)
      indirect_mask |= nir_var_local;

   return indirect_mask;
}

nir_shader *
brw_nir_optimize(nir_shader *nir, const struct brw_compiler *compiler,
                 bool is_scalar, bool allow_copies)
{
   nir_variable_mode indirect_mask =
      brw_nir_no_indirect_mask(compiler, nir->info.stage);

   bool progress;
   do {
      progress = false;
      OPT(nir_split_array_vars, nir_var_local);
      OPT(nir_shrink_vec_array_vars, nir_var_local);
      OPT(nir_lower_vars_to_ssa);
      if (allow_copies) {
         /* Only run this pass in the first call to brw_nir_optimize.  Later
          * calls assume that we've lowered away any copy_deref instructions
          * and we don't want to introduce any more.
          */
         OPT(nir_opt_find_array_copies);
      }
      OPT(nir_opt_copy_prop_vars);
      OPT(nir_opt_dead_write_vars);

      if (is_scalar) {
         OPT(nir_lower_alu_to_scalar);
      }

      OPT(nir_copy_prop);

      if (is_scalar) {
         OPT(nir_lower_phis_to_scalar);
      }

      OPT(nir_copy_prop);
      OPT(nir_opt_dce);
      OPT(nir_opt_cse);

      /* For indirect loads of uniforms (push constants), we assume that array
       * indices will nearly always be in bounds and the cost of the load is
       * low.  Therefore there shouldn't be a performance benefit to avoid it.
       * However, in vec4 tessellation shaders, these loads operate by
       * actually pulling from memory.
       */
      const bool is_vec4_tessellation = !is_scalar &&
         (nir->info.stage == MESA_SHADER_TESS_CTRL ||
          nir->info.stage == MESA_SHADER_TESS_EVAL);
      OPT(nir_opt_peephole_select, 0, is_vec4_tessellation);

      OPT(nir_opt_intrinsics);
      OPT(nir_opt_algebraic);
      OPT(nir_opt_constant_folding);
      OPT(nir_opt_dead_cf);
      if (OPT(nir_opt_trivial_continues)) {
         /* If nir_opt_trivial_continues makes progress, then we need to clean
          * things up if we want any hope of nir_opt_if or nir_opt_loop_unroll
          * to make progress.
          */
         OPT(nir_copy_prop);
         OPT(nir_opt_dce);
      }
      OPT(nir_opt_if);
      if (nir->options->max_unroll_iterations != 0) {
         OPT(nir_opt_loop_unroll, indirect_mask);
      }
      OPT(nir_opt_remove_phis);
      OPT(nir_opt_undef);
      OPT(nir_lower_doubles, nir_lower_drcp |
                             nir_lower_dsqrt |
                             nir_lower_drsq |
                             nir_lower_dtrunc |
                             nir_lower_dfloor |
                             nir_lower_dceil |
                             nir_lower_dfract |
                             nir_lower_dround_even |
                             nir_lower_dmod);
      OPT(nir_lower_pack);
   } while (progress);

   /* Workaround Gfxbench unused local sampler variable which will trigger an
    * assert in the opt_large_constants pass.
    */
   OPT(nir_remove_dead_variables, nir_var_local);

   return nir;
}

static unsigned
lower_bit_size_callback(const nir_alu_instr *alu, UNUSED void *data)
{
   assert(alu->dest.dest.is_ssa);
   if (alu->dest.dest.ssa.bit_size != 16)
      return 0;

   switch (alu->op) {
   case nir_op_idiv:
   case nir_op_imod:
   case nir_op_irem:
   case nir_op_udiv:
   case nir_op_umod:
      return 32;
   default:
      return 0;
   }
}

/* Does some simple lowering and runs the standard suite of optimizations
 *
 * This is intended to be called more-or-less directly after you get the
 * shader out of GLSL or some other source.  While it is geared towards i965,
 * it is not at all generator-specific except for the is_scalar flag.  Even
 * there, it is safe to call with is_scalar = false for a shader that is
 * intended for the FS backend as long as nir_optimize is called again with
 * is_scalar = true to scalarize everything prior to code gen.
 */
nir_shader *
brw_preprocess_nir(const struct brw_compiler *compiler, nir_shader *nir)
{
   const struct gen_device_info *devinfo = compiler->devinfo;
   UNUSED bool progress; /* Written by OPT */

   const bool is_scalar = compiler->scalar_stage[nir->info.stage];

   if (nir->info.stage == MESA_SHADER_GEOMETRY)
      OPT(nir_lower_gs_intrinsics);

   /* See also brw_nir_trig_workarounds.py */
   if (compiler->precise_trig &&
       !(devinfo->gen >= 10 || devinfo->is_kabylake))
      OPT(brw_nir_apply_trig_workarounds);

   static const nir_lower_tex_options tex_options = {
      .lower_txp = ~0,
      .lower_txf_offset = true,
      .lower_rect_offset = true,
      .lower_txd_cube_map = true,
   };

   OPT(nir_lower_tex, &tex_options);
   OPT(nir_normalize_cubemap_coords);

   OPT(nir_lower_global_vars_to_local);

   OPT(nir_split_var_copies);
   OPT(nir_split_struct_vars, nir_var_local);

   /* Run opt_algebraic before int64 lowering so we can hopefully get rid
    * of some int64 instructions.
    */
   OPT(nir_opt_algebraic);

   /* Lower int64 instructions before nir_optimize so that loop unrolling
    * sees their actual cost.
    */
   OPT(nir_lower_int64, nir_lower_imul64 |
                        nir_lower_isign64 |
                        nir_lower_divmod64);

   nir = brw_nir_optimize(nir, compiler, is_scalar, true);

   /* This needs to be run after the first optimization pass but before we
    * lower indirect derefs away
    */
   if (compiler->supports_shader_constants) {
      OPT(nir_opt_large_constants, NULL, 32);
   }

   OPT(nir_lower_bit_size, lower_bit_size_callback, NULL);

   if (is_scalar) {
      OPT(nir_lower_load_const_to_scalar);
   }

   /* Lower a bunch of stuff */
   OPT(nir_lower_var_copies);

   OPT(nir_lower_system_values);

   const nir_lower_subgroups_options subgroups_options = {
      .subgroup_size = BRW_SUBGROUP_SIZE,
      .ballot_bit_size = 32,
      .lower_to_scalar = true,
      .lower_subgroup_masks = true,
      .lower_vote_trivial = !is_scalar,
      .lower_shuffle = true,
   };
   OPT(nir_lower_subgroups, &subgroups_options);

   OPT(nir_lower_clip_cull_distance_arrays);

   nir_variable_mode indirect_mask =
      brw_nir_no_indirect_mask(compiler, nir->info.stage);
   OPT(nir_lower_indirect_derefs, indirect_mask);

   OPT(brw_nir_lower_mem_access_bit_sizes);

   /* Get rid of split copies */
   nir = brw_nir_optimize(nir, compiler, is_scalar, false);

   return nir;
}

void
brw_nir_link_shaders(const struct brw_compiler *compiler,
                     nir_shader **producer, nir_shader **consumer)
{
   nir_lower_io_arrays_to_elements(*producer, *consumer);
   nir_validate_shader(*producer, "after nir_lower_io_arrays_to_elements");
   nir_validate_shader(*consumer, "after nir_lower_io_arrays_to_elements");

   const bool p_is_scalar =
      compiler->scalar_stage[(*producer)->info.stage];
   const bool c_is_scalar =
      compiler->scalar_stage[(*consumer)->info.stage];

   if (p_is_scalar && c_is_scalar) {
      NIR_PASS_V(*producer, nir_lower_io_to_scalar_early, nir_var_shader_out);
      NIR_PASS_V(*consumer, nir_lower_io_to_scalar_early, nir_var_shader_in);
      *producer = brw_nir_optimize(*producer, compiler, p_is_scalar, false);
      *consumer = brw_nir_optimize(*consumer, compiler, c_is_scalar, false);
   }

   if (nir_link_constant_varyings(*producer, *consumer))
      *consumer = brw_nir_optimize(*consumer, compiler, c_is_scalar, false);

   NIR_PASS_V(*producer, nir_remove_dead_variables, nir_var_shader_out);
   NIR_PASS_V(*consumer, nir_remove_dead_variables, nir_var_shader_in);

   if (nir_remove_unused_varyings(*producer, *consumer)) {
      NIR_PASS_V(*producer, nir_lower_global_vars_to_local);
      NIR_PASS_V(*consumer, nir_lower_global_vars_to_local);

      /* The backend might not be able to handle indirects on
       * temporaries so we need to lower indirects on any of the
       * varyings we have demoted here.
       */
      NIR_PASS_V(*producer, nir_lower_indirect_derefs,
                 brw_nir_no_indirect_mask(compiler, (*producer)->info.stage));
      NIR_PASS_V(*consumer, nir_lower_indirect_derefs,
                 brw_nir_no_indirect_mask(compiler, (*consumer)->info.stage));

      *producer = brw_nir_optimize(*producer, compiler, p_is_scalar, false);
      *consumer = brw_nir_optimize(*consumer, compiler, c_is_scalar, false);
   }
}

/* Prepare the given shader for codegen
 *
 * This function is intended to be called right before going into the actual
 * backend and is highly backend-specific.  Also, once this function has been
 * called on a shader, it will no longer be in SSA form so most optimizations
 * will not work.
 */
nir_shader *
brw_postprocess_nir(nir_shader *nir, const struct brw_compiler *compiler,
                    bool is_scalar)
{
   const struct gen_device_info *devinfo = compiler->devinfo;
   bool debug_enabled =
      (INTEL_DEBUG & intel_debug_flag_for_shader_stage(nir->info.stage));

   UNUSED bool progress; /* Written by OPT */


   do {
      progress = false;
      OPT(nir_opt_algebraic_before_ffma);
   } while (progress);

   nir = brw_nir_optimize(nir, compiler, is_scalar, false);

   if (devinfo->gen >= 6) {
      /* Try and fuse multiply-adds */
      OPT(brw_nir_opt_peephole_ffma);
   }

   OPT(nir_opt_algebraic_late);

   OPT(nir_lower_to_source_mods, nir_lower_all_source_mods);
   OPT(nir_copy_prop);
   OPT(nir_opt_dce);
   OPT(nir_opt_move_comparisons);

   OPT(nir_lower_locals_to_regs);

   if (unlikely(debug_enabled)) {
      /* Re-index SSA defs so we print more sensible numbers. */
      nir_foreach_function(function, nir) {
         if (function->impl)
            nir_index_ssa_defs(function->impl);
      }

      fprintf(stderr, "NIR (SSA form) for %s shader:\n",
              _mesa_shader_stage_to_string(nir->info.stage));
      nir_print_shader(nir, stderr);
   }

   OPT(nir_convert_from_ssa, true);

   if (!is_scalar) {
      OPT(nir_move_vec_src_uses_to_dest);
      OPT(nir_lower_vec_to_movs);
   }

   OPT(nir_opt_dce);

   /* This is the last pass we run before we start emitting stuff.  It
    * determines when we need to insert boolean resolves on Gen <= 5.  We
    * run it last because it stashes data in instr->pass_flags and we don't
    * want that to be squashed by other NIR passes.
    */
   if (devinfo->gen <= 5)
      brw_nir_analyze_boolean_resolves(nir);

   nir_sweep(nir);

   if (unlikely(debug_enabled)) {
      fprintf(stderr, "NIR (final form) for %s shader:\n",
              _mesa_shader_stage_to_string(nir->info.stage));
      nir_print_shader(nir, stderr);
   }

   return nir;
}

nir_shader *
brw_nir_apply_sampler_key(nir_shader *nir,
                          const struct brw_compiler *compiler,
                          const struct brw_sampler_prog_key_data *key_tex,
                          bool is_scalar)
{
   const struct gen_device_info *devinfo = compiler->devinfo;
   nir_lower_tex_options tex_options = { 0 };

   /* Iron Lake and prior require lowering of all rectangle textures */
   if (devinfo->gen < 6)
      tex_options.lower_rect = true;

   /* Prior to Broadwell, our hardware can't actually do GL_CLAMP */
   if (devinfo->gen < 8) {
      tex_options.saturate_s = key_tex->gl_clamp_mask[0];
      tex_options.saturate_t = key_tex->gl_clamp_mask[1];
      tex_options.saturate_r = key_tex->gl_clamp_mask[2];
   }

   /* Prior to Haswell, we have to fake texture swizzle */
   for (unsigned s = 0; s < MAX_SAMPLERS; s++) {
      if (key_tex->swizzles[s] == SWIZZLE_NOOP)
         continue;

      tex_options.swizzle_result |= (1 << s);
      for (unsigned c = 0; c < 4; c++)
         tex_options.swizzles[s][c] = GET_SWZ(key_tex->swizzles[s], c);
   }

   /* Prior to Haswell, we have to lower gradients on shadow samplers */
   tex_options.lower_txd_shadow = devinfo->gen < 8 && !devinfo->is_haswell;

   tex_options.lower_y_uv_external = key_tex->y_uv_image_mask;
   tex_options.lower_y_u_v_external = key_tex->y_u_v_image_mask;
   tex_options.lower_yx_xuxv_external = key_tex->yx_xuxv_image_mask;
   tex_options.lower_xy_uxvx_external = key_tex->xy_uxvx_image_mask;
   tex_options.lower_ayuv_external = key_tex->ayuv_image_mask;

   if (nir_lower_tex(nir, &tex_options)) {
      nir_validate_shader(nir, "after nir_lower_tex");
      nir = brw_nir_optimize(nir, compiler, is_scalar, false);
   }

   return nir;
}

enum brw_reg_type
brw_type_for_nir_type(const struct gen_device_info *devinfo, nir_alu_type type)
{
   switch (type) {
   case nir_type_uint:
   case nir_type_uint32:
      return BRW_REGISTER_TYPE_UD;
   case nir_type_bool:
   case nir_type_int:
   case nir_type_bool32:
   case nir_type_int32:
      return BRW_REGISTER_TYPE_D;
   case nir_type_float:
   case nir_type_float32:
      return BRW_REGISTER_TYPE_F;
   case nir_type_float16:
      return BRW_REGISTER_TYPE_HF;
   case nir_type_float64:
      return BRW_REGISTER_TYPE_DF;
   case nir_type_int64:
      return devinfo->gen < 8 ? BRW_REGISTER_TYPE_DF : BRW_REGISTER_TYPE_Q;
   case nir_type_uint64:
      return devinfo->gen < 8 ? BRW_REGISTER_TYPE_DF : BRW_REGISTER_TYPE_UQ;
   case nir_type_int16:
      return BRW_REGISTER_TYPE_W;
   case nir_type_uint16:
      return BRW_REGISTER_TYPE_UW;
   case nir_type_int8:
      return BRW_REGISTER_TYPE_B;
   case nir_type_uint8:
      return BRW_REGISTER_TYPE_UB;
   default:
      unreachable("unknown type");
   }

   return BRW_REGISTER_TYPE_F;
}

/* Returns the glsl_base_type corresponding to a nir_alu_type.
 * This is used by both brw_vec4_nir and brw_fs_nir.
 */
enum glsl_base_type
brw_glsl_base_type_for_nir_type(nir_alu_type type)
{
   switch (type) {
   case nir_type_float:
   case nir_type_float32:
      return GLSL_TYPE_FLOAT;

   case nir_type_float16:
      return GLSL_TYPE_FLOAT16;

   case nir_type_float64:
      return GLSL_TYPE_DOUBLE;

   case nir_type_int:
   case nir_type_int32:
      return GLSL_TYPE_INT;

   case nir_type_uint:
   case nir_type_uint32:
      return GLSL_TYPE_UINT;

   case nir_type_int16:
      return GLSL_TYPE_INT16;

   case nir_type_uint16:
      return GLSL_TYPE_UINT16;

   default:
      unreachable("bad type");
   }
}

nir_shader *
brw_nir_create_passthrough_tcs(void *mem_ctx, const struct brw_compiler *compiler,
                               const nir_shader_compiler_options *options,
                               const struct brw_tcs_prog_key *key)
{
   nir_builder b;
   nir_builder_init_simple_shader(&b, mem_ctx, MESA_SHADER_TESS_CTRL,
                                  options);
   nir_shader *nir = b.shader;
   nir_variable *var;
   nir_intrinsic_instr *load;
   nir_intrinsic_instr *store;
   nir_ssa_def *zero = nir_imm_int(&b, 0);
   nir_ssa_def *invoc_id =
      nir_load_system_value(&b, nir_intrinsic_load_invocation_id, 0);

   nir->info.inputs_read = key->outputs_written &
      ~(VARYING_BIT_TESS_LEVEL_INNER | VARYING_BIT_TESS_LEVEL_OUTER);
   nir->info.outputs_written = key->outputs_written;
   nir->info.tess.tcs_vertices_out = key->input_vertices;
   nir->info.name = ralloc_strdup(nir, "passthrough");
   nir->num_uniforms = 8 * sizeof(uint32_t);

   var = nir_variable_create(nir, nir_var_uniform, glsl_vec4_type(), "hdr_0");
   var->data.location = 0;
   var = nir_variable_create(nir, nir_var_uniform, glsl_vec4_type(), "hdr_1");
   var->data.location = 1;

   /* Write the patch URB header. */
   for (int i = 0; i <= 1; i++) {
      load = nir_intrinsic_instr_create(nir, nir_intrinsic_load_uniform);
      load->num_components = 4;
      load->src[0] = nir_src_for_ssa(zero);
      nir_ssa_dest_init(&load->instr, &load->dest, 4, 32, NULL);
      nir_intrinsic_set_base(load, i * 4 * sizeof(uint32_t));
      nir_builder_instr_insert(&b, &load->instr);

      store = nir_intrinsic_instr_create(nir, nir_intrinsic_store_output);
      store->num_components = 4;
      store->src[0] = nir_src_for_ssa(&load->dest.ssa);
      store->src[1] = nir_src_for_ssa(zero);
      nir_intrinsic_set_base(store, VARYING_SLOT_TESS_LEVEL_INNER - i);
      nir_intrinsic_set_write_mask(store, WRITEMASK_XYZW);
      nir_builder_instr_insert(&b, &store->instr);
   }

   /* Copy inputs to outputs. */
   uint64_t varyings = nir->info.inputs_read;

   while (varyings != 0) {
      const int varying = ffsll(varyings) - 1;

      load = nir_intrinsic_instr_create(nir,
                                        nir_intrinsic_load_per_vertex_input);
      load->num_components = 4;
      load->src[0] = nir_src_for_ssa(invoc_id);
      load->src[1] = nir_src_for_ssa(zero);
      nir_ssa_dest_init(&load->instr, &load->dest, 4, 32, NULL);
      nir_intrinsic_set_base(load, varying);
      nir_builder_instr_insert(&b, &load->instr);

      store = nir_intrinsic_instr_create(nir,
                                         nir_intrinsic_store_per_vertex_output);
      store->num_components = 4;
      store->src[0] = nir_src_for_ssa(&load->dest.ssa);
      store->src[1] = nir_src_for_ssa(invoc_id);
      store->src[2] = nir_src_for_ssa(zero);
      nir_intrinsic_set_base(store, varying);
      nir_intrinsic_set_write_mask(store, WRITEMASK_XYZW);
      nir_builder_instr_insert(&b, &store->instr);

      varyings &= ~BITFIELD64_BIT(varying);
   }

   nir_validate_shader(nir, "in brw_nir_create_passthrough_tcs");

   nir = brw_preprocess_nir(compiler, nir);

   return nir;
}
