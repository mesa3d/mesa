/*
 * Copyright 2017 Advanced Micro Devices, Inc.
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

#include "si_shader_internal.h"
#include "si_pipe.h"

#include "ac_nir_to_llvm.h"

#include "tgsi/tgsi_from_mesa.h"

#include "compiler/nir/nir.h"
#include "compiler/nir_types.h"

static nir_variable* tex_get_texture_var(nir_tex_instr *instr)
{
	for (unsigned i = 0; i < instr->num_srcs; i++) {
		switch (instr->src[i].src_type) {
		case nir_tex_src_texture_deref:
			return nir_deref_instr_get_variable(nir_src_as_deref(instr->src[i].src));
		default:
			break;
		}
	}

	return NULL;
}

static nir_variable* intrinsic_get_var(nir_intrinsic_instr *instr)
{
	return nir_deref_instr_get_variable(nir_src_as_deref(instr->src[0]));
}

static void scan_instruction(struct tgsi_shader_info *info,
			     nir_instr *instr)
{
	if (instr->type == nir_instr_type_alu) {
		nir_alu_instr *alu = nir_instr_as_alu(instr);

		switch (alu->op) {
		case nir_op_fddx:
		case nir_op_fddy:
		case nir_op_fddx_fine:
		case nir_op_fddy_fine:
		case nir_op_fddx_coarse:
		case nir_op_fddy_coarse:
			info->uses_derivatives = true;
			break;
		default:
			break;
		}
	} else if (instr->type == nir_instr_type_tex) {
		nir_tex_instr *tex = nir_instr_as_tex(instr);
		nir_variable *texture = tex_get_texture_var(tex);

		if (!texture) {
			info->samplers_declared |=
				u_bit_consecutive(tex->sampler_index, 1);
		} else {
			if (texture->data.bindless)
				info->uses_bindless_samplers = true;
		}

		switch (tex->op) {
		case nir_texop_tex:
		case nir_texop_txb:
		case nir_texop_lod:
			info->uses_derivatives = true;
			break;
		default:
			break;
		}
	} else if (instr->type == nir_instr_type_intrinsic) {
		nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);

		switch (intr->intrinsic) {
		case nir_intrinsic_load_front_face:
			info->uses_frontface = 1;
			break;
		case nir_intrinsic_load_instance_id:
			info->uses_instanceid = 1;
			break;
		case nir_intrinsic_load_invocation_id:
			info->uses_invocationid = true;
			break;
		case nir_intrinsic_load_num_work_groups:
			info->uses_grid_size = true;
			break;
		case nir_intrinsic_load_local_group_size:
			/* The block size is translated to IMM with a fixed block size. */
			if (info->properties[TGSI_PROPERTY_CS_FIXED_BLOCK_WIDTH] == 0)
				info->uses_block_size = true;
			break;
		case nir_intrinsic_load_local_invocation_id:
		case nir_intrinsic_load_work_group_id: {
			unsigned mask = nir_ssa_def_components_read(&intr->dest.ssa);
			while (mask) {
				unsigned i = u_bit_scan(&mask);

				if (intr->intrinsic == nir_intrinsic_load_work_group_id)
					info->uses_block_id[i] = true;
				else
					info->uses_thread_id[i] = true;
			}
			break;
		}
		case nir_intrinsic_load_vertex_id:
			info->uses_vertexid = 1;
			break;
		case nir_intrinsic_load_vertex_id_zero_base:
			info->uses_vertexid_nobase = 1;
			break;
		case nir_intrinsic_load_base_vertex:
			info->uses_basevertex = 1;
			break;
		case nir_intrinsic_load_primitive_id:
			info->uses_primid = 1;
			break;
		case nir_intrinsic_load_sample_mask_in:
			info->reads_samplemask = true;
			break;
		case nir_intrinsic_load_tess_level_inner:
		case nir_intrinsic_load_tess_level_outer:
			info->reads_tess_factors = true;
			break;
		case nir_intrinsic_image_deref_load: {
			nir_variable *var = intrinsic_get_var(intr);
			if (var->data.bindless) {
				info->uses_bindless_images = true;

				if (glsl_get_sampler_dim(var->type) == GLSL_SAMPLER_DIM_BUF)
					info->uses_bindless_buffer_load = true;
				else
					info->uses_bindless_image_load = true;
			}
			break;
		}
		case nir_intrinsic_image_deref_size:
		case nir_intrinsic_image_deref_samples: {
			nir_variable *var = intrinsic_get_var(intr);
			if (var->data.bindless)
				info->uses_bindless_images = true;
			break;
		}
		case nir_intrinsic_image_deref_store: {
			nir_variable *var = intrinsic_get_var(intr);
			if (var->data.bindless) {
				info->uses_bindless_images = true;

				if (glsl_get_sampler_dim(var->type) == GLSL_SAMPLER_DIM_BUF)
					info->uses_bindless_buffer_store = true;
				else
					info->uses_bindless_image_store = true;
			}
			info->writes_memory = true;
			break;
		}
		case nir_intrinsic_image_deref_atomic_add:
		case nir_intrinsic_image_deref_atomic_min:
		case nir_intrinsic_image_deref_atomic_max:
		case nir_intrinsic_image_deref_atomic_and:
		case nir_intrinsic_image_deref_atomic_or:
		case nir_intrinsic_image_deref_atomic_xor:
		case nir_intrinsic_image_deref_atomic_exchange:
		case nir_intrinsic_image_deref_atomic_comp_swap: {
			nir_variable *var = intrinsic_get_var(intr);
			if (var->data.bindless) {
				info->uses_bindless_images = true;

				if (glsl_get_sampler_dim(var->type) == GLSL_SAMPLER_DIM_BUF)
					info->uses_bindless_buffer_atomic = true;
				else
					info->uses_bindless_image_atomic = true;
			}
			info->writes_memory = true;
			break;
		}
		case nir_intrinsic_store_ssbo:
		case nir_intrinsic_ssbo_atomic_add:
		case nir_intrinsic_ssbo_atomic_imin:
		case nir_intrinsic_ssbo_atomic_umin:
		case nir_intrinsic_ssbo_atomic_imax:
		case nir_intrinsic_ssbo_atomic_umax:
		case nir_intrinsic_ssbo_atomic_and:
		case nir_intrinsic_ssbo_atomic_or:
		case nir_intrinsic_ssbo_atomic_xor:
		case nir_intrinsic_ssbo_atomic_exchange:
		case nir_intrinsic_ssbo_atomic_comp_swap:
			info->writes_memory = true;
			break;
		case nir_intrinsic_load_deref: {
			nir_variable *var = intrinsic_get_var(intr);
			nir_variable_mode mode = var->data.mode;
			enum glsl_base_type base_type =
				glsl_get_base_type(glsl_without_array(var->type));

			if (mode == nir_var_shader_in) {
				switch (var->data.interpolation) {
				case INTERP_MODE_NONE:
					if (glsl_base_type_is_integer(base_type))
						break;

					/* fall-through */
				case INTERP_MODE_SMOOTH:
					if (var->data.sample)
						info->uses_persp_sample = true;
					else if (var->data.centroid)
						info->uses_persp_centroid = true;
					else
						info->uses_persp_center = true;
					break;

				case INTERP_MODE_NOPERSPECTIVE:
					if (var->data.sample)
						info->uses_linear_sample = true;
					else if (var->data.centroid)
						info->uses_linear_centroid = true;
					else
						info->uses_linear_center = true;
					break;
				}
			}
			break;
		}
		case nir_intrinsic_interp_deref_at_centroid:
		case nir_intrinsic_interp_deref_at_sample:
		case nir_intrinsic_interp_deref_at_offset: {
			enum glsl_interp_mode interp = intrinsic_get_var(intr)->data.interpolation;
			switch (interp) {
			case INTERP_MODE_SMOOTH:
			case INTERP_MODE_NONE:
				if (intr->intrinsic == nir_intrinsic_interp_deref_at_centroid)
					info->uses_persp_opcode_interp_centroid = true;
				else if (intr->intrinsic == nir_intrinsic_interp_deref_at_sample)
					info->uses_persp_opcode_interp_sample = true;
				else
					info->uses_persp_opcode_interp_offset = true;
				break;
			case INTERP_MODE_NOPERSPECTIVE:
				if (intr->intrinsic == nir_intrinsic_interp_deref_at_centroid)
					info->uses_linear_opcode_interp_centroid = true;
				else if (intr->intrinsic == nir_intrinsic_interp_deref_at_sample)
					info->uses_linear_opcode_interp_sample = true;
				else
					info->uses_linear_opcode_interp_offset = true;
				break;
			case INTERP_MODE_FLAT:
				break;
			default:
				unreachable("Unsupported interpoation type");
			}
			break;
		}
		default:
			break;
		}
	}
}

void si_nir_scan_tess_ctrl(const struct nir_shader *nir,
			   const struct tgsi_shader_info *info,
			   struct tgsi_tessctrl_info *out)
{
	memset(out, 0, sizeof(*out));

	if (nir->info.stage != MESA_SHADER_TESS_CTRL)
		return;

	/* Initial value = true. Here the pass will accumulate results from
	 * multiple segments surrounded by barriers. If tess factors aren't
	 * written at all, it's a shader bug and we don't care if this will be
	 * true.
	 */
	out->tessfactors_are_def_in_all_invocs = true;

	/* TODO: Implement scanning of tess factors, see tgsi backend. */
}

void si_nir_scan_shader(const struct nir_shader *nir,
			struct tgsi_shader_info *info)
{
	nir_function *func;
	unsigned i;

	info->processor = pipe_shader_type_from_mesa(nir->info.stage);
	info->num_tokens = 2; /* indicate that the shader is non-empty */
	info->num_instructions = 2;

	info->properties[TGSI_PROPERTY_NEXT_SHADER] =
		pipe_shader_type_from_mesa(nir->info.next_stage);

	if (nir->info.stage == MESA_SHADER_TESS_CTRL) {
		info->properties[TGSI_PROPERTY_TCS_VERTICES_OUT] =
			nir->info.tess.tcs_vertices_out;
	}

	if (nir->info.stage == MESA_SHADER_TESS_EVAL) {
		if (nir->info.tess.primitive_mode == GL_ISOLINES)
			info->properties[TGSI_PROPERTY_TES_PRIM_MODE] = PIPE_PRIM_LINES;
		else
			info->properties[TGSI_PROPERTY_TES_PRIM_MODE] = nir->info.tess.primitive_mode;

		STATIC_ASSERT((TESS_SPACING_EQUAL + 1) % 3 == PIPE_TESS_SPACING_EQUAL);
		STATIC_ASSERT((TESS_SPACING_FRACTIONAL_ODD + 1) % 3 ==
			      PIPE_TESS_SPACING_FRACTIONAL_ODD);
		STATIC_ASSERT((TESS_SPACING_FRACTIONAL_EVEN + 1) % 3 ==
			      PIPE_TESS_SPACING_FRACTIONAL_EVEN);

		info->properties[TGSI_PROPERTY_TES_SPACING] = (nir->info.tess.spacing + 1) % 3;
		info->properties[TGSI_PROPERTY_TES_VERTEX_ORDER_CW] = !nir->info.tess.ccw;
		info->properties[TGSI_PROPERTY_TES_POINT_MODE] = nir->info.tess.point_mode;
	}

	if (nir->info.stage == MESA_SHADER_GEOMETRY) {
		info->properties[TGSI_PROPERTY_GS_INPUT_PRIM] = nir->info.gs.input_primitive;
		info->properties[TGSI_PROPERTY_GS_OUTPUT_PRIM] = nir->info.gs.output_primitive;
		info->properties[TGSI_PROPERTY_GS_MAX_OUTPUT_VERTICES] = nir->info.gs.vertices_out;
		info->properties[TGSI_PROPERTY_GS_INVOCATIONS] = nir->info.gs.invocations;
	}

	if (nir->info.stage == MESA_SHADER_FRAGMENT) {
		info->properties[TGSI_PROPERTY_FS_EARLY_DEPTH_STENCIL] =
			nir->info.fs.early_fragment_tests | nir->info.fs.post_depth_coverage;
		info->properties[TGSI_PROPERTY_FS_POST_DEPTH_COVERAGE] = nir->info.fs.post_depth_coverage;

		if (nir->info.fs.pixel_center_integer) {
			info->properties[TGSI_PROPERTY_FS_COORD_PIXEL_CENTER] =
				TGSI_FS_COORD_PIXEL_CENTER_INTEGER;
		}

		if (nir->info.fs.depth_layout != FRAG_DEPTH_LAYOUT_NONE) {
			switch (nir->info.fs.depth_layout) {
			case FRAG_DEPTH_LAYOUT_ANY:
				info->properties[TGSI_PROPERTY_FS_DEPTH_LAYOUT] = TGSI_FS_DEPTH_LAYOUT_ANY;
				break;
			case FRAG_DEPTH_LAYOUT_GREATER:
				info->properties[TGSI_PROPERTY_FS_DEPTH_LAYOUT] = TGSI_FS_DEPTH_LAYOUT_GREATER;
				break;
			case FRAG_DEPTH_LAYOUT_LESS:
				info->properties[TGSI_PROPERTY_FS_DEPTH_LAYOUT] = TGSI_FS_DEPTH_LAYOUT_LESS;
				break;
			case FRAG_DEPTH_LAYOUT_UNCHANGED:
				info->properties[TGSI_PROPERTY_FS_DEPTH_LAYOUT] = TGSI_FS_DEPTH_LAYOUT_UNCHANGED;
				break;
			default:
				unreachable("Unknow depth layout");
			}
		}
	}

	if (nir->info.stage == MESA_SHADER_COMPUTE) {
		info->properties[TGSI_PROPERTY_CS_FIXED_BLOCK_WIDTH] = nir->info.cs.local_size[0];
		info->properties[TGSI_PROPERTY_CS_FIXED_BLOCK_HEIGHT] = nir->info.cs.local_size[1];
		info->properties[TGSI_PROPERTY_CS_FIXED_BLOCK_DEPTH] = nir->info.cs.local_size[2];
	}

	i = 0;
	uint64_t processed_inputs = 0;
	unsigned num_inputs = 0;
	nir_foreach_variable(variable, &nir->inputs) {
		unsigned semantic_name, semantic_index;

		const struct glsl_type *type = variable->type;
		if (nir_is_per_vertex_io(variable, nir->info.stage)) {
			assert(glsl_type_is_array(type));
			type = glsl_get_array_element(type);
		}

		unsigned attrib_count = glsl_count_attribute_slots(type,
								   nir->info.stage == MESA_SHADER_VERTEX);

		i = variable->data.driver_location;

		/* Vertex shader inputs don't have semantics. The state
		 * tracker has already mapped them to attributes via
		 * variable->data.driver_location.
		 */
		if (nir->info.stage == MESA_SHADER_VERTEX) {
			/* TODO: gather the actual input useage and remove this. */
			info->input_usage_mask[i] = TGSI_WRITEMASK_XYZW;

			if (glsl_type_is_dual_slot(variable->type)) {
				num_inputs += 2;

				/* TODO: gather the actual input useage and remove this. */
				info->input_usage_mask[i+1] = TGSI_WRITEMASK_XYZW;
			} else
				num_inputs++;
			continue;
		}

		/* Fragment shader position is a system value. */
		if (nir->info.stage == MESA_SHADER_FRAGMENT &&
		    variable->data.location == VARYING_SLOT_POS) {
			if (variable->data.pixel_center_integer)
				info->properties[TGSI_PROPERTY_FS_COORD_PIXEL_CENTER] =
					TGSI_FS_COORD_PIXEL_CENTER_INTEGER;

			num_inputs++;
			continue;
		}

		for (unsigned j = 0; j < attrib_count; j++, i++) {

			if (processed_inputs & ((uint64_t)1 << i))
				continue;

			processed_inputs |= ((uint64_t)1 << i);
			num_inputs++;

			tgsi_get_gl_varying_semantic(variable->data.location + j, true,
						     &semantic_name, &semantic_index);

			info->input_semantic_name[i] = semantic_name;
			info->input_semantic_index[i] = semantic_index;

			if (semantic_name == TGSI_SEMANTIC_PRIMID)
				info->uses_primid = true;

			if (variable->data.sample)
				info->input_interpolate_loc[i] = TGSI_INTERPOLATE_LOC_SAMPLE;
			else if (variable->data.centroid)
				info->input_interpolate_loc[i] = TGSI_INTERPOLATE_LOC_CENTROID;
			else
				info->input_interpolate_loc[i] = TGSI_INTERPOLATE_LOC_CENTER;

			enum glsl_base_type base_type =
				glsl_get_base_type(glsl_without_array(variable->type));

			switch (variable->data.interpolation) {
			case INTERP_MODE_NONE:
				if (glsl_base_type_is_integer(base_type)) {
					info->input_interpolate[i] = TGSI_INTERPOLATE_CONSTANT;
					break;
				}

				if (semantic_name == TGSI_SEMANTIC_COLOR) {
					info->input_interpolate[i] = TGSI_INTERPOLATE_COLOR;
					break;
				}
				/* fall-through */

			case INTERP_MODE_SMOOTH:
				assert(!glsl_base_type_is_integer(base_type));

				info->input_interpolate[i] = TGSI_INTERPOLATE_PERSPECTIVE;
				break;

			case INTERP_MODE_NOPERSPECTIVE:
				assert(!glsl_base_type_is_integer(base_type));

				info->input_interpolate[i] = TGSI_INTERPOLATE_LINEAR;
				break;

			case INTERP_MODE_FLAT:
				info->input_interpolate[i] = TGSI_INTERPOLATE_CONSTANT;
				break;
			}

			/* TODO make this more precise */
			if (variable->data.location == VARYING_SLOT_COL0)
				info->colors_read |= 0x0f;
			else if (variable->data.location == VARYING_SLOT_COL1)
				info->colors_read |= 0xf0;
		}
	}

	info->num_inputs = num_inputs;


	i = 0;
	uint64_t processed_outputs = 0;
	unsigned num_outputs = 0;
	nir_foreach_variable(variable, &nir->outputs) {
		unsigned semantic_name, semantic_index;

		i = variable->data.driver_location;

		const struct glsl_type *type = variable->type;
		if (nir_is_per_vertex_io(variable, nir->info.stage)) {
			assert(glsl_type_is_array(type));
			type = glsl_get_array_element(type);
		}

		unsigned attrib_count = glsl_count_attribute_slots(type, false);
		for (unsigned k = 0; k < attrib_count; k++, i++) {

			if (nir->info.stage == MESA_SHADER_FRAGMENT) {
				tgsi_get_gl_frag_result_semantic(variable->data.location + k,
					&semantic_name, &semantic_index);

				/* Adjust for dual source blending */
				if (variable->data.index > 0) {
					semantic_index++;
				}
			} else {
				tgsi_get_gl_varying_semantic(variable->data.location + k, true,
							     &semantic_name, &semantic_index);
			}

			unsigned num_components = 4;
			unsigned vector_elements = glsl_get_vector_elements(glsl_without_array(variable->type));
			if (vector_elements)
				num_components = vector_elements;

			unsigned component = variable->data.location_frac;
			if (glsl_type_is_64bit(glsl_without_array(variable->type))) {
				if (glsl_type_is_dual_slot(glsl_without_array(variable->type)) && k % 2) {
					num_components = (num_components * 2) - 4;
					component = 0;
				} else {
					num_components = MIN2(num_components * 2, 4);
				}
			}

			ubyte usagemask = 0;
			for (unsigned j = component; j < num_components + component; j++) {
				switch (j) {
				case 0:
					usagemask |= TGSI_WRITEMASK_X;
					break;
				case 1:
					usagemask |= TGSI_WRITEMASK_Y;
					break;
				case 2:
					usagemask |= TGSI_WRITEMASK_Z;
					break;
				case 3:
					usagemask |= TGSI_WRITEMASK_W;
					break;
				default:
					unreachable("error calculating component index");
				}
			}

			unsigned gs_out_streams;
			if (variable->data.stream & (1u << 31)) {
				gs_out_streams = variable->data.stream & ~(1u << 31);
			} else {
				assert(variable->data.stream < 4);
				gs_out_streams = 0;
				for (unsigned j = 0; j < num_components; ++j)
					gs_out_streams |= variable->data.stream << (2 * (component + j));
			}

			unsigned streamx = gs_out_streams & 3;
			unsigned streamy = (gs_out_streams >> 2) & 3;
			unsigned streamz = (gs_out_streams >> 4) & 3;
			unsigned streamw = (gs_out_streams >> 6) & 3;

			if (usagemask & TGSI_WRITEMASK_X) {
				info->output_usagemask[i] |= TGSI_WRITEMASK_X;
				info->output_streams[i] |= streamx;
				info->num_stream_output_components[streamx]++;
			}
			if (usagemask & TGSI_WRITEMASK_Y) {
				info->output_usagemask[i] |= TGSI_WRITEMASK_Y;
				info->output_streams[i] |= streamy << 2;
				info->num_stream_output_components[streamy]++;
			}
			if (usagemask & TGSI_WRITEMASK_Z) {
				info->output_usagemask[i] |= TGSI_WRITEMASK_Z;
				info->output_streams[i] |= streamz << 4;
				info->num_stream_output_components[streamz]++;
			}
			if (usagemask & TGSI_WRITEMASK_W) {
				info->output_usagemask[i] |= TGSI_WRITEMASK_W;
				info->output_streams[i] |= streamw << 6;
				info->num_stream_output_components[streamw]++;
			}

			/* make sure we only count this location once against
			 * the num_outputs counter.
			 */
			if (processed_outputs & ((uint64_t)1 << i))
				continue;

			processed_outputs |= ((uint64_t)1 << i);
			num_outputs++;

			info->output_semantic_name[i] = semantic_name;
			info->output_semantic_index[i] = semantic_index;

			switch (semantic_name) {
			case TGSI_SEMANTIC_PRIMID:
				info->writes_primid = true;
				break;
			case TGSI_SEMANTIC_VIEWPORT_INDEX:
				info->writes_viewport_index = true;
				break;
			case TGSI_SEMANTIC_LAYER:
				info->writes_layer = true;
				break;
			case TGSI_SEMANTIC_PSIZE:
				info->writes_psize = true;
				break;
			case TGSI_SEMANTIC_CLIPVERTEX:
				info->writes_clipvertex = true;
				break;
			case TGSI_SEMANTIC_COLOR:
				info->colors_written |= 1 << semantic_index;
				break;
			case TGSI_SEMANTIC_STENCIL:
				info->writes_stencil = true;
				break;
			case TGSI_SEMANTIC_SAMPLEMASK:
				info->writes_samplemask = true;
				break;
			case TGSI_SEMANTIC_EDGEFLAG:
				info->writes_edgeflag = true;
				break;
			case TGSI_SEMANTIC_POSITION:
				if (info->processor == PIPE_SHADER_FRAGMENT)
					info->writes_z = true;
				else
					info->writes_position = true;
				break;
			}

			if (nir->info.stage == MESA_SHADER_TESS_CTRL) {
				switch (semantic_name) {
				case TGSI_SEMANTIC_PATCH:
					info->reads_perpatch_outputs = true;
				break;
				case TGSI_SEMANTIC_TESSINNER:
				case TGSI_SEMANTIC_TESSOUTER:
					info->reads_tessfactor_outputs = true;
				break;
				default:
					info->reads_pervertex_outputs = true;
				}
			}
		}

		unsigned loc = variable->data.location;
		if (nir->info.stage == MESA_SHADER_FRAGMENT &&
		    loc == FRAG_RESULT_COLOR &&
		    nir->info.outputs_written & (1ull << loc)) {
			assert(attrib_count == 1);
			info->properties[TGSI_PROPERTY_FS_COLOR0_WRITES_ALL_CBUFS] = true;
		}
	}

	info->num_outputs = num_outputs;

	struct set *ubo_set = _mesa_set_create(NULL, _mesa_hash_pointer,
					       _mesa_key_pointer_equal);

	/* Intialise const_file_max[0] */
	info->const_file_max[0] = -1;

	unsigned ubo_idx = 1;
	nir_foreach_variable(variable, &nir->uniforms) {
		const struct glsl_type *type = variable->type;
		enum glsl_base_type base_type =
			glsl_get_base_type(glsl_without_array(type));
		unsigned aoa_size = MAX2(1, glsl_get_aoa_size(type));

		/* Gather buffers declared bitmasks. Note: radeonsi doesn't
		 * really use the mask (other than ubo_idx == 1 for regular
		 * uniforms) its really only used for getting the buffer count
		 * so we don't need to worry about the ordering.
		 */
		if (variable->interface_type != NULL) {
			if (variable->data.mode == nir_var_uniform) {

				unsigned block_count;
				if (base_type != GLSL_TYPE_INTERFACE) {
					struct set_entry *entry =
						_mesa_set_search(ubo_set, variable->interface_type);

					/* Check if we have already processed
					 * a member from this ubo.
					 */
					if (entry)
						continue;

					block_count = 1;
				} else {
					block_count = aoa_size;
				}

				info->const_buffers_declared |= u_bit_consecutive(ubo_idx, block_count);
				ubo_idx += block_count;

				_mesa_set_add(ubo_set, variable->interface_type);
			}

			if (variable->data.mode == nir_var_shader_storage) {
				/* TODO: make this more accurate */
				info->shader_buffers_declared =
					u_bit_consecutive(0, SI_NUM_SHADER_BUFFERS);
			}

			continue;
		}

		/* We rely on the fact that nir_lower_samplers_as_deref has
		 * eliminated struct dereferences.
		 */
		if (base_type == GLSL_TYPE_SAMPLER) {
			if (variable->data.bindless) {
				info->const_buffers_declared |= 1;
				info->const_file_max[0] +=
					glsl_count_attribute_slots(type, false);
			} else {
				info->samplers_declared |=
					u_bit_consecutive(variable->data.binding, aoa_size);
			}
		} else if (base_type == GLSL_TYPE_IMAGE) {
			if (variable->data.bindless) {
				info->const_buffers_declared |= 1;
				info->const_file_max[0] +=
					glsl_count_attribute_slots(type, false);
			} else {
				info->images_declared |=
					u_bit_consecutive(variable->data.binding, aoa_size);
			}
		} else if (base_type != GLSL_TYPE_ATOMIC_UINT) {
			if (strncmp(variable->name, "state.", 6) == 0 ||
			    strncmp(variable->name, "gl_", 3) == 0) {
				/* FIXME: figure out why piglit tests with builtin
				 * uniforms are failing without this.
				 */
				info->const_buffers_declared =
					u_bit_consecutive(0, SI_NUM_CONST_BUFFERS);
			} else {
				info->const_buffers_declared |= 1;
				info->const_file_max[0] +=
					glsl_count_attribute_slots(type, false);
			}
		}
	}

	_mesa_set_destroy(ubo_set, NULL);

	info->num_written_clipdistance = nir->info.clip_distance_array_size;
	info->num_written_culldistance = nir->info.cull_distance_array_size;
	info->clipdist_writemask = u_bit_consecutive(0, info->num_written_clipdistance);
	info->culldist_writemask = u_bit_consecutive(0, info->num_written_culldistance);

	if (info->processor == PIPE_SHADER_FRAGMENT)
		info->uses_kill = nir->info.fs.uses_discard;

	func = (struct nir_function *)exec_list_get_head_const(&nir->functions);
	nir_foreach_block(block, func->impl) {
		nir_foreach_instr(instr, block)
			scan_instruction(info, instr);
	}
}

/**
 * Perform "lowering" operations on the NIR that are run once when the shader
 * selector is created.
 */
void
si_lower_nir(struct si_shader_selector* sel)
{
	/* Adjust the driver location of inputs and outputs. The state tracker
	 * interprets them as slots, while the ac/nir backend interprets them
	 * as individual components.
	 */
	nir_foreach_variable(variable, &sel->nir->inputs)
		variable->data.driver_location *= 4;

	nir_foreach_variable(variable, &sel->nir->outputs) {
		variable->data.driver_location *= 4;

		if (sel->nir->info.stage == MESA_SHADER_FRAGMENT) {
			if (variable->data.location == FRAG_RESULT_DEPTH)
				variable->data.driver_location += 2;
			else if (variable->data.location == FRAG_RESULT_STENCIL)
				variable->data.driver_location += 1;
		}
	}

	/* Perform lowerings (and optimizations) of code.
	 *
	 * Performance considerations aside, we must:
	 * - lower certain ALU operations
	 * - ensure constant offsets for texture instructions are folded
	 *   and copy-propagated
	 */
	NIR_PASS_V(sel->nir, nir_lower_returns);
	NIR_PASS_V(sel->nir, nir_lower_vars_to_ssa);
	NIR_PASS_V(sel->nir, nir_lower_alu_to_scalar);
	NIR_PASS_V(sel->nir, nir_lower_phis_to_scalar);

	static const struct nir_lower_tex_options lower_tex_options = {
		.lower_txp = ~0u,
	};
	NIR_PASS_V(sel->nir, nir_lower_tex, &lower_tex_options);

	const nir_lower_subgroups_options subgroups_options = {
		.subgroup_size = 64,
		.ballot_bit_size = 64,
		.lower_to_scalar = true,
		.lower_subgroup_masks = true,
		.lower_vote_trivial = false,
		.lower_vote_eq_to_ballot = true,
	};
	NIR_PASS_V(sel->nir, nir_lower_subgroups, &subgroups_options);

	ac_lower_indirect_derefs(sel->nir, sel->screen->info.chip_class);

	NIR_PASS_V(sel->nir, nir_lower_load_const_to_scalar);

	bool progress;
	do {
		progress = false;

		/* (Constant) copy propagation is needed for txf with offsets. */
		NIR_PASS(progress, sel->nir, nir_copy_prop);
		NIR_PASS(progress, sel->nir, nir_opt_remove_phis);
		NIR_PASS(progress, sel->nir, nir_opt_dce);
		if (nir_opt_trivial_continues(sel->nir)) {
			progress = true;
			NIR_PASS(progress, sel->nir, nir_copy_prop);
			NIR_PASS(progress, sel->nir, nir_opt_dce);
		}
		NIR_PASS(progress, sel->nir, nir_opt_if);
		NIR_PASS(progress, sel->nir, nir_opt_dead_cf);
		NIR_PASS(progress, sel->nir, nir_opt_cse);
		NIR_PASS(progress, sel->nir, nir_opt_peephole_select, 8, true, true);

		/* Needed for algebraic lowering */
		NIR_PASS(progress, sel->nir, nir_opt_algebraic);
		NIR_PASS(progress, sel->nir, nir_opt_constant_folding);

		NIR_PASS(progress, sel->nir, nir_opt_undef);
		NIR_PASS(progress, sel->nir, nir_opt_conditional_discard);
		if (sel->nir->options->max_unroll_iterations) {
			NIR_PASS(progress, sel->nir, nir_opt_loop_unroll, 0);
		}
	} while (progress);
}

static void declare_nir_input_vs(struct si_shader_context *ctx,
				 struct nir_variable *variable,
				 unsigned input_index,
				 LLVMValueRef out[4])
{
	si_llvm_load_input_vs(ctx, input_index, out);
}

static void declare_nir_input_fs(struct si_shader_context *ctx,
				 struct nir_variable *variable,
				 unsigned input_index,
				 LLVMValueRef out[4])
{
	unsigned slot = variable->data.location;
	if (slot == VARYING_SLOT_POS) {
		out[0] = LLVMGetParam(ctx->main_fn, SI_PARAM_POS_X_FLOAT);
		out[1] = LLVMGetParam(ctx->main_fn, SI_PARAM_POS_Y_FLOAT);
		out[2] = LLVMGetParam(ctx->main_fn, SI_PARAM_POS_Z_FLOAT);
		out[3] = ac_build_fdiv(&ctx->ac, ctx->ac.f32_1,
				LLVMGetParam(ctx->main_fn, SI_PARAM_POS_W_FLOAT));
		return;
	}

	si_llvm_load_input_fs(ctx, input_index, out);
}

LLVMValueRef
si_nir_lookup_interp_param(struct ac_shader_abi *abi,
			   enum glsl_interp_mode interp, unsigned location)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);
	int interp_param_idx = -1;

	switch (interp) {
	case INTERP_MODE_FLAT:
		return NULL;
	case INTERP_MODE_SMOOTH:
	case INTERP_MODE_NONE:
		if (location == INTERP_CENTER)
			interp_param_idx = SI_PARAM_PERSP_CENTER;
		else if (location == INTERP_CENTROID)
			interp_param_idx = SI_PARAM_PERSP_CENTROID;
		else if (location == INTERP_SAMPLE)
			interp_param_idx = SI_PARAM_PERSP_SAMPLE;
		break;
	case INTERP_MODE_NOPERSPECTIVE:
		if (location == INTERP_CENTER)
			interp_param_idx = SI_PARAM_LINEAR_CENTER;
		else if (location == INTERP_CENTROID)
			interp_param_idx = SI_PARAM_LINEAR_CENTROID;
		else if (location == INTERP_SAMPLE)
			interp_param_idx = SI_PARAM_LINEAR_SAMPLE;
		break;
	default:
		assert(!"Unhandled interpolation mode.");
		return NULL;
	}

	return interp_param_idx != -1 ?
		LLVMGetParam(ctx->main_fn, interp_param_idx) : NULL;
}

static LLVMValueRef
si_nir_load_sampler_desc(struct ac_shader_abi *abi,
		         unsigned descriptor_set, unsigned base_index,
		         unsigned constant_index, LLVMValueRef dynamic_index,
		         enum ac_descriptor_type desc_type, bool image,
			 bool write, bool bindless)
{
	struct si_shader_context *ctx = si_shader_context_from_abi(abi);
	const struct tgsi_shader_info *info = &ctx->shader->selector->info;
	LLVMBuilderRef builder = ctx->ac.builder;
	unsigned const_index = base_index + constant_index;
	bool dcc_off = write;

	/* TODO: images_store and images_atomic are not set */
	if (!dynamic_index && image &&
	    (info->images_store | info->images_atomic) & (1 << const_index))
		dcc_off = true;

	assert(!descriptor_set);
	assert(!image || desc_type == AC_DESC_IMAGE || desc_type == AC_DESC_BUFFER);

	if (bindless) {
		LLVMValueRef list =
			LLVMGetParam(ctx->main_fn, ctx->param_bindless_samplers_and_images);

		/* dynamic_index is the bindless handle */
		if (image) {
			return si_load_image_desc(ctx, list, dynamic_index, desc_type,
						  dcc_off, true);
		}

		/* Since bindless handle arithmetic can contain an unsigned integer
		 * wraparound and si_load_sampler_desc assumes there isn't any,
		 * use GEP without "inbounds" (inside ac_build_pointer_add)
		 * to prevent incorrect code generation and hangs.
		 */
		dynamic_index = LLVMBuildMul(ctx->ac.builder, dynamic_index,
					     LLVMConstInt(ctx->i32, 2, 0), "");
		list = ac_build_pointer_add(&ctx->ac, list, dynamic_index);
		return si_load_sampler_desc(ctx, list, ctx->i32_0, desc_type);
	}

	unsigned num_slots = image ? ctx->num_images : ctx->num_samplers;
	assert(const_index < num_slots);

	LLVMValueRef list = LLVMGetParam(ctx->main_fn, ctx->param_samplers_and_images);
	LLVMValueRef index = LLVMConstInt(ctx->ac.i32, const_index, false);

	if (dynamic_index) {
		index = LLVMBuildAdd(builder, index, dynamic_index, "");

		/* From the GL_ARB_shader_image_load_store extension spec:
		 *
		 *    If a shader performs an image load, store, or atomic
		 *    operation using an image variable declared as an array,
		 *    and if the index used to select an individual element is
		 *    negative or greater than or equal to the size of the
		 *    array, the results of the operation are undefined but may
		 *    not lead to termination.
		 */
		index = si_llvm_bound_index(ctx, index, num_slots);
	}

	if (image) {
		index = LLVMBuildSub(ctx->ac.builder,
				     LLVMConstInt(ctx->i32, SI_NUM_IMAGES - 1, 0),
				     index, "");
		return si_load_image_desc(ctx, list, index, desc_type, dcc_off, false);
	}

	index = LLVMBuildAdd(ctx->ac.builder, index,
			     LLVMConstInt(ctx->i32, SI_NUM_IMAGES / 2, 0), "");
	return si_load_sampler_desc(ctx, list, index, desc_type);
}

static void bitcast_inputs(struct si_shader_context *ctx,
			   LLVMValueRef data[4],
			   unsigned input_idx)
{
	for (unsigned chan = 0; chan < 4; chan++) {
		ctx->inputs[input_idx + chan] =
			LLVMBuildBitCast(ctx->ac.builder, data[chan], ctx->ac.i32, "");
	}
}

bool si_nir_build_llvm(struct si_shader_context *ctx, struct nir_shader *nir)
{
	struct tgsi_shader_info *info = &ctx->shader->selector->info;

	if (nir->info.stage == MESA_SHADER_VERTEX ||
	    nir->info.stage == MESA_SHADER_FRAGMENT) {
		uint64_t processed_inputs = 0;
		nir_foreach_variable(variable, &nir->inputs) {
			unsigned attrib_count = glsl_count_attribute_slots(variable->type,
									   nir->info.stage == MESA_SHADER_VERTEX);
			unsigned input_idx = variable->data.driver_location;

			LLVMValueRef data[4];
			unsigned loc = variable->data.location;

			for (unsigned i = 0; i < attrib_count; i++) {
				/* Packed components share the same location so skip
				 * them if we have already processed the location.
				 */
				if (processed_inputs & ((uint64_t)1 << (loc + i))) {
					input_idx += 4;
					continue;
				}

				if (nir->info.stage == MESA_SHADER_VERTEX) {
					declare_nir_input_vs(ctx, variable, input_idx / 4, data);
					bitcast_inputs(ctx, data, input_idx);
					if (glsl_type_is_dual_slot(variable->type)) {
						input_idx += 4;
						declare_nir_input_vs(ctx, variable, input_idx / 4, data);
						bitcast_inputs(ctx, data, input_idx);
					}
				} else if (nir->info.stage == MESA_SHADER_FRAGMENT) {
					declare_nir_input_fs(ctx, variable, input_idx / 4, data);
					bitcast_inputs(ctx, data, input_idx);
				}

				processed_inputs |= ((uint64_t)1 << (loc + i));
				input_idx += 4;
			}
		}
	}

	ctx->abi.inputs = &ctx->inputs[0];
	ctx->abi.load_sampler_desc = si_nir_load_sampler_desc;
	ctx->abi.clamp_shadow_reference = true;

	ctx->num_samplers = util_last_bit(info->samplers_declared);
	ctx->num_images = util_last_bit(info->images_declared);

	if (ctx->shader->selector->info.properties[TGSI_PROPERTY_CS_LOCAL_SIZE]) {
		assert(nir->info.stage == MESA_SHADER_COMPUTE);
		si_declare_compute_memory(ctx);
	}
	ac_nir_translate(&ctx->ac, &ctx->abi, nir);

	return true;
}
