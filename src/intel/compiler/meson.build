# Copyright © 2017 Intel Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

libintel_compiler_files = files(
  'brw_cfg.cpp',
  'brw_cfg.h',
  'brw_clip.h',
  'brw_clip_line.c',
  'brw_clip_point.c',
  'brw_clip_tri.c',
  'brw_clip_unfilled.c',
  'brw_clip_util.c',
  'brw_compile_clip.c',
  'brw_compile_sf.c',
  'brw_compiler.c',
  'brw_compiler.h',
  'brw_dead_control_flow.cpp',
  'brw_dead_control_flow.h',
  'brw_disasm.c',
  'brw_disasm_info.c',
  'brw_disasm_info.h',
  'brw_eu.c',
  'brw_eu_compact.c',
  'brw_eu_defines.h',
  'brw_eu_emit.c',
  'brw_eu.h',
  'brw_eu_util.c',
  'brw_eu_validate.c',
  'brw_fs_bank_conflicts.cpp',
  'brw_fs_builder.h',
  'brw_fs_cmod_propagation.cpp',
  'brw_fs_combine_constants.cpp',
  'brw_fs_copy_propagation.cpp',
  'brw_fs.cpp',
  'brw_fs_cse.cpp',
  'brw_fs_dead_code_eliminate.cpp',
  'brw_fs_generator.cpp',
  'brw_fs.h',
  'brw_fs_live_variables.cpp',
  'brw_fs_live_variables.h',
  'brw_fs_lower_conversions.cpp',
  'brw_fs_lower_pack.cpp',
  'brw_fs_nir.cpp',
  'brw_fs_reg_allocate.cpp',
  'brw_fs_register_coalesce.cpp',
  'brw_fs_saturate_propagation.cpp',
  'brw_fs_sel_peephole.cpp',
  'brw_fs_surface_builder.cpp',
  'brw_fs_surface_builder.h',
  'brw_fs_validate.cpp',
  'brw_fs_visitor.cpp',
  'brw_inst.h',
  'brw_interpolation_map.c',
  'brw_ir_allocator.h',
  'brw_ir_fs.h',
  'brw_ir_vec4.h',
  'brw_nir.h',
  'brw_nir.c',
  'brw_nir_analyze_boolean_resolves.c',
  'brw_nir_analyze_ubo_ranges.c',
  'brw_nir_attribute_workarounds.c',
  'brw_nir_lower_cs_intrinsics.c',
  'brw_nir_lower_image_load_store.c',
  'brw_nir_lower_mem_access_bit_sizes.c',
  'brw_nir_opt_peephole_ffma.c',
  'brw_nir_tcs_workarounds.c',
  'brw_packed_float.c',
  'brw_predicated_break.cpp',
  'brw_reg.h',
  'brw_reg_type.c',
  'brw_reg_type.h',
  'brw_schedule_instructions.cpp',
  'brw_shader.cpp',
  'brw_shader.h',
  'brw_vec4_builder.h',
  'brw_vec4_cmod_propagation.cpp',
  'brw_vec4_copy_propagation.cpp',
  'brw_vec4.cpp',
  'brw_vec4_cse.cpp',
  'brw_vec4_dead_code_eliminate.cpp',
  'brw_vec4_generator.cpp',
  'brw_vec4_gs_visitor.cpp',
  'brw_vec4_gs_visitor.h',
  'brw_vec4.h',
  'brw_vec4_live_variables.cpp',
  'brw_vec4_live_variables.h',
  'brw_vec4_nir.cpp',
  'brw_vec4_gs_nir.cpp',
  'brw_vec4_reg_allocate.cpp',
  'brw_vec4_surface_builder.cpp',
  'brw_vec4_surface_builder.h',
  'brw_vec4_tcs.cpp',
  'brw_vec4_tcs.h',
  'brw_vec4_tes.cpp',
  'brw_vec4_tes.h',
  'brw_vec4_visitor.cpp',
  'brw_vec4_vs_visitor.cpp',
  'brw_vec4_vs.h',
  'brw_vue_map.c',
  'brw_wm_iz.cpp',
  'gen6_gs_visitor.cpp',
  'gen6_gs_visitor.h',
)

brw_nir_trig = custom_target(
  'brw_nir_trig_workarounds.c',
  input : 'brw_nir_trig_workarounds.py',
  output : 'brw_nir_trig_workarounds.c',
  command : [
    prog_python, '@INPUT@',
    '-p', join_paths(meson.source_root(), 'src/compiler/nir/'),
  ],
  depend_files : nir_algebraic_py,
  capture : true,
)

libintel_compiler = static_library(
  'intel_compiler',
  [libintel_compiler_files, brw_nir_trig, ir_expression_operation_h],
  include_directories : [inc_common, inc_intel],
  c_args : [c_vis_args, no_override_init_args],
  cpp_args : [cpp_vis_args],
  dependencies : idep_nir_headers,
  build_by_default : false,
)

if with_tests
  # The last two tests are not C++ or gtest, pre comment in autotools make
  foreach t : ['fs_cmod_propagation', 'fs_copy_propagation',
               'fs_saturate_propagation', 'vf_float_conversions',
               'vec4_register_coalesce', 'vec4_copy_propagation',
               'vec4_cmod_propagation', 'vec4_dead_code_eliminate',
               'eu_compact', 'eu_validate']
    test(
      t,
      executable(
        [t, ir_expression_operation_h],
        'test_@0@.cpp'.format(t),
        include_directories : [inc_common, inc_intel],
        link_with : [
          libintel_compiler, libintel_common, libintel_dev, libmesa_util, libisl,
        ],
        dependencies : [dep_thread, dep_dl, idep_gtest, idep_nir],
      ),
      suite : ['intel'],
    )
  endforeach
endif
