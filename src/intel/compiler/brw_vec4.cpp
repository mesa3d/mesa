/*
 * Copyright © 2011 Intel Corporation
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

#include "brw_vec4.h"
#include "brw_fs.h"
#include "brw_cfg.h"
#include "brw_nir.h"
#include "brw_vec4_builder.h"
#include "brw_vec4_live_variables.h"
#include "brw_vec4_vs.h"
#include "brw_dead_control_flow.h"
#include "common/gen_debug.h"
#include "program/prog_parameter.h"
#include "util/u_math.h"

#define MAX_INSTRUCTION (1 << 30)

using namespace brw;

namespace brw {

void
src_reg::init()
{
   memset((void*)this, 0, sizeof(*this));
   this->file = BAD_FILE;
   this->type = BRW_REGISTER_TYPE_UD;
}

src_reg::src_reg(enum brw_reg_file file, int nr, const glsl_type *type)
{
   init();

   this->file = file;
   this->nr = nr;
   if (type && (type->is_scalar() || type->is_vector() || type->is_matrix()))
      this->swizzle = brw_swizzle_for_size(type->vector_elements);
   else
      this->swizzle = BRW_SWIZZLE_XYZW;
   if (type)
      this->type = brw_type_for_base_type(type);
}

/** Generic unset register constructor. */
src_reg::src_reg()
{
   init();
}

src_reg::src_reg(struct ::brw_reg reg) :
   backend_reg(reg)
{
   this->offset = 0;
   this->reladdr = NULL;
}

src_reg::src_reg(const dst_reg &reg) :
   backend_reg(reg)
{
   this->reladdr = reg.reladdr;
   this->swizzle = brw_swizzle_for_mask(reg.writemask);
}

void
dst_reg::init()
{
   memset((void*)this, 0, sizeof(*this));
   this->file = BAD_FILE;
   this->type = BRW_REGISTER_TYPE_UD;
   this->writemask = WRITEMASK_XYZW;
}

dst_reg::dst_reg()
{
   init();
}

dst_reg::dst_reg(enum brw_reg_file file, int nr)
{
   init();

   this->file = file;
   this->nr = nr;
}

dst_reg::dst_reg(enum brw_reg_file file, int nr, const glsl_type *type,
                 unsigned writemask)
{
   init();

   this->file = file;
   this->nr = nr;
   this->type = brw_type_for_base_type(type);
   this->writemask = writemask;
}

dst_reg::dst_reg(enum brw_reg_file file, int nr, brw_reg_type type,
                 unsigned writemask)
{
   init();

   this->file = file;
   this->nr = nr;
   this->type = type;
   this->writemask = writemask;
}

dst_reg::dst_reg(struct ::brw_reg reg) :
   backend_reg(reg)
{
   this->offset = 0;
   this->reladdr = NULL;
}

dst_reg::dst_reg(const src_reg &reg) :
   backend_reg(reg)
{
   this->writemask = brw_mask_for_swizzle(reg.swizzle);
   this->reladdr = reg.reladdr;
}

bool
dst_reg::equals(const dst_reg &r) const
{
   return (this->backend_reg::equals(r) &&
           (reladdr == r.reladdr ||
            (reladdr && r.reladdr && reladdr->equals(*r.reladdr))));
}

bool
vec4_instruction::is_send_from_grf()
{
   switch (opcode) {
   case SHADER_OPCODE_SHADER_TIME_ADD:
   case VS_OPCODE_PULL_CONSTANT_LOAD_GEN7:
   case SHADER_OPCODE_UNTYPED_ATOMIC:
   case SHADER_OPCODE_UNTYPED_SURFACE_READ:
   case SHADER_OPCODE_UNTYPED_SURFACE_WRITE:
   case SHADER_OPCODE_TYPED_ATOMIC:
   case SHADER_OPCODE_TYPED_SURFACE_READ:
   case SHADER_OPCODE_TYPED_SURFACE_WRITE:
   case VEC4_OPCODE_URB_READ:
   case TCS_OPCODE_URB_WRITE:
   case TCS_OPCODE_RELEASE_INPUT:
   case SHADER_OPCODE_BARRIER:
      return true;
   default:
      return false;
   }
}

/**
 * Returns true if this instruction's sources and destinations cannot
 * safely be the same register.
 *
 * In most cases, a register can be written over safely by the same
 * instruction that is its last use.  For a single instruction, the
 * sources are dereferenced before writing of the destination starts
 * (naturally).
 *
 * However, there are a few cases where this can be problematic:
 *
 * - Virtual opcodes that translate to multiple instructions in the
 *   code generator: if src == dst and one instruction writes the
 *   destination before a later instruction reads the source, then
 *   src will have been clobbered.
 *
 * The register allocator uses this information to set up conflicts between
 * GRF sources and the destination.
 */
bool
vec4_instruction::has_source_and_destination_hazard() const
{
   switch (opcode) {
   case TCS_OPCODE_SET_INPUT_URB_OFFSETS:
   case TCS_OPCODE_SET_OUTPUT_URB_OFFSETS:
   case TES_OPCODE_ADD_INDIRECT_URB_OFFSET:
      return true;
   default:
      /* 8-wide compressed DF operations are executed as two 4-wide operations,
       * so we have a src/dst hazard if the first half of the instruction
       * overwrites the source of the second half. Prevent this by marking
       * compressed instructions as having src/dst hazards, so the register
       * allocator assigns safe register regions for dst and srcs.
       */
      return size_written > REG_SIZE;
   }
}

unsigned
vec4_instruction::size_read(unsigned arg) const
{
   switch (opcode) {
   case SHADER_OPCODE_SHADER_TIME_ADD:
   case SHADER_OPCODE_UNTYPED_ATOMIC:
   case SHADER_OPCODE_UNTYPED_SURFACE_READ:
   case SHADER_OPCODE_UNTYPED_SURFACE_WRITE:
   case SHADER_OPCODE_TYPED_ATOMIC:
   case SHADER_OPCODE_TYPED_SURFACE_READ:
   case SHADER_OPCODE_TYPED_SURFACE_WRITE:
   case TCS_OPCODE_URB_WRITE:
      if (arg == 0)
         return mlen * REG_SIZE;
      break;
   case VS_OPCODE_PULL_CONSTANT_LOAD_GEN7:
      if (arg == 1)
         return mlen * REG_SIZE;
      break;
   default:
      break;
   }

   switch (src[arg].file) {
   case BAD_FILE:
      return 0;
   case IMM:
   case UNIFORM:
      return 4 * type_sz(src[arg].type);
   default:
      /* XXX - Represent actual vertical stride. */
      return exec_size * type_sz(src[arg].type);
   }
}

bool
vec4_instruction::can_do_source_mods(const struct gen_device_info *devinfo)
{
   if (devinfo->gen == 6 && is_math())
      return false;

   if (is_send_from_grf())
      return false;

   if (!backend_instruction::can_do_source_mods())
      return false;

   return true;
}

bool
vec4_instruction::can_do_cmod()
{
   if (!backend_instruction::can_do_cmod())
      return false;

   /* The accumulator result appears to get used for the conditional modifier
    * generation.  When negating a UD value, there is a 33rd bit generated for
    * the sign in the accumulator value, so now you can't check, for example,
    * equality with a 32-bit value.  See piglit fs-op-neg-uvec4.
    */
   for (unsigned i = 0; i < 3; i++) {
      if (src[i].file != BAD_FILE &&
          type_is_unsigned_int(src[i].type) && src[i].negate)
         return false;
   }

   return true;
}

bool
vec4_instruction::can_do_writemask(const struct gen_device_info *devinfo)
{
   switch (opcode) {
   case SHADER_OPCODE_GEN4_SCRATCH_READ:
   case VEC4_OPCODE_DOUBLE_TO_F32:
   case VEC4_OPCODE_DOUBLE_TO_D32:
   case VEC4_OPCODE_DOUBLE_TO_U32:
   case VEC4_OPCODE_TO_DOUBLE:
   case VEC4_OPCODE_PICK_LOW_32BIT:
   case VEC4_OPCODE_PICK_HIGH_32BIT:
   case VEC4_OPCODE_SET_LOW_32BIT:
   case VEC4_OPCODE_SET_HIGH_32BIT:
   case VS_OPCODE_PULL_CONSTANT_LOAD:
   case VS_OPCODE_PULL_CONSTANT_LOAD_GEN7:
   case VS_OPCODE_SET_SIMD4X2_HEADER_GEN9:
   case TCS_OPCODE_SET_INPUT_URB_OFFSETS:
   case TCS_OPCODE_SET_OUTPUT_URB_OFFSETS:
   case TES_OPCODE_CREATE_INPUT_READ_HEADER:
   case TES_OPCODE_ADD_INDIRECT_URB_OFFSET:
   case VEC4_OPCODE_URB_READ:
   case SHADER_OPCODE_MOV_INDIRECT:
      return false;
   default:
      /* The MATH instruction on Gen6 only executes in align1 mode, which does
       * not support writemasking.
       */
      if (devinfo->gen == 6 && is_math())
         return false;

      if (is_tex())
         return false;

      return true;
   }
}

bool
vec4_instruction::can_change_types() const
{
   return dst.type == src[0].type &&
          !src[0].abs && !src[0].negate && !saturate &&
          (opcode == BRW_OPCODE_MOV ||
           (opcode == BRW_OPCODE_SEL &&
            dst.type == src[1].type &&
            predicate != BRW_PREDICATE_NONE &&
            !src[1].abs && !src[1].negate));
}

/**
 * Returns how many MRFs an opcode will write over.
 *
 * Note that this is not the 0 or 1 implied writes in an actual gen
 * instruction -- the generate_* functions generate additional MOVs
 * for setup.
 */
int
vec4_visitor::implied_mrf_writes(vec4_instruction *inst)
{
   if (inst->mlen == 0 || inst->is_send_from_grf())
      return 0;

   switch (inst->opcode) {
   case SHADER_OPCODE_RCP:
   case SHADER_OPCODE_RSQ:
   case SHADER_OPCODE_SQRT:
   case SHADER_OPCODE_EXP2:
   case SHADER_OPCODE_LOG2:
   case SHADER_OPCODE_SIN:
   case SHADER_OPCODE_COS:
      return 1;
   case SHADER_OPCODE_INT_QUOTIENT:
   case SHADER_OPCODE_INT_REMAINDER:
   case SHADER_OPCODE_POW:
   case TCS_OPCODE_THREAD_END:
      return 2;
   case VS_OPCODE_URB_WRITE:
      return 1;
   case VS_OPCODE_PULL_CONSTANT_LOAD:
      return 2;
   case SHADER_OPCODE_GEN4_SCRATCH_READ:
      return 2;
   case SHADER_OPCODE_GEN4_SCRATCH_WRITE:
      return 3;
   case GS_OPCODE_URB_WRITE:
   case GS_OPCODE_URB_WRITE_ALLOCATE:
   case GS_OPCODE_THREAD_END:
      return 0;
   case GS_OPCODE_FF_SYNC:
      return 1;
   case TCS_OPCODE_URB_WRITE:
      return 0;
   case SHADER_OPCODE_SHADER_TIME_ADD:
      return 0;
   case SHADER_OPCODE_TEX:
   case SHADER_OPCODE_TXL:
   case SHADER_OPCODE_TXD:
   case SHADER_OPCODE_TXF:
   case SHADER_OPCODE_TXF_CMS:
   case SHADER_OPCODE_TXF_CMS_W:
   case SHADER_OPCODE_TXF_MCS:
   case SHADER_OPCODE_TXS:
   case SHADER_OPCODE_TG4:
   case SHADER_OPCODE_TG4_OFFSET:
   case SHADER_OPCODE_SAMPLEINFO:
   case SHADER_OPCODE_GET_BUFFER_SIZE:
      return inst->header_size;
   default:
      unreachable("not reached");
   }
}

bool
src_reg::equals(const src_reg &r) const
{
   return (this->backend_reg::equals(r) &&
	   !reladdr && !r.reladdr);
}

bool
src_reg::negative_equals(const src_reg &r) const
{
   return this->backend_reg::negative_equals(r) &&
          !reladdr && !r.reladdr;
}

bool
vec4_visitor::opt_vector_float()
{
   bool progress = false;

   foreach_block(block, cfg) {
      int last_reg = -1, last_offset = -1;
      enum brw_reg_file last_reg_file = BAD_FILE;

      uint8_t imm[4] = { 0 };
      int inst_count = 0;
      vec4_instruction *imm_inst[4];
      unsigned writemask = 0;
      enum brw_reg_type dest_type = BRW_REGISTER_TYPE_F;

      foreach_inst_in_block_safe(vec4_instruction, inst, block) {
         int vf = -1;
         enum brw_reg_type need_type;

         /* Look for unconditional MOVs from an immediate with a partial
          * writemask.  Skip type-conversion MOVs other than integer 0,
          * where the type doesn't matter.  See if the immediate can be
          * represented as a VF.
          */
         if (inst->opcode == BRW_OPCODE_MOV &&
             inst->src[0].file == IMM &&
             inst->predicate == BRW_PREDICATE_NONE &&
             inst->dst.writemask != WRITEMASK_XYZW &&
             type_sz(inst->src[0].type) < 8 &&
             (inst->src[0].type == inst->dst.type || inst->src[0].d == 0)) {

            vf = brw_float_to_vf(inst->src[0].d);
            need_type = BRW_REGISTER_TYPE_D;

            if (vf == -1) {
               vf = brw_float_to_vf(inst->src[0].f);
               need_type = BRW_REGISTER_TYPE_F;
            }
         } else {
            last_reg = -1;
         }

         /* If this wasn't a MOV, or the destination register doesn't match,
          * or we have to switch destination types, then this breaks our
          * sequence.  Combine anything we've accumulated so far.
          */
         if (last_reg != inst->dst.nr ||
             last_offset != inst->dst.offset ||
             last_reg_file != inst->dst.file ||
             (vf > 0 && dest_type != need_type)) {

            if (inst_count > 1) {
               unsigned vf;
               memcpy(&vf, imm, sizeof(vf));
               vec4_instruction *mov = MOV(imm_inst[0]->dst, brw_imm_vf(vf));
               mov->dst.type = dest_type;
               mov->dst.writemask = writemask;
               inst->insert_before(block, mov);

               for (int i = 0; i < inst_count; i++) {
                  imm_inst[i]->remove(block);
               }

               progress = true;
            }

            inst_count = 0;
            last_reg = -1;
            writemask = 0;
            dest_type = BRW_REGISTER_TYPE_F;

            for (int i = 0; i < 4; i++) {
               imm[i] = 0;
            }
         }

         /* Record this instruction's value (if it was representable). */
         if (vf != -1) {
            if ((inst->dst.writemask & WRITEMASK_X) != 0)
               imm[0] = vf;
            if ((inst->dst.writemask & WRITEMASK_Y) != 0)
               imm[1] = vf;
            if ((inst->dst.writemask & WRITEMASK_Z) != 0)
               imm[2] = vf;
            if ((inst->dst.writemask & WRITEMASK_W) != 0)
               imm[3] = vf;

            writemask |= inst->dst.writemask;
            imm_inst[inst_count++] = inst;

            last_reg = inst->dst.nr;
            last_offset = inst->dst.offset;
            last_reg_file = inst->dst.file;
            if (vf > 0)
               dest_type = need_type;
         }
      }
   }

   if (progress)
      invalidate_live_intervals();

   return progress;
}

/* Replaces unused channels of a swizzle with channels that are used.
 *
 * For instance, this pass transforms
 *
 *    mov vgrf4.yz, vgrf5.wxzy
 *
 * into
 *
 *    mov vgrf4.yz, vgrf5.xxzx
 *
 * This eliminates false uses of some channels, letting dead code elimination
 * remove the instructions that wrote them.
 */
bool
vec4_visitor::opt_reduce_swizzle()
{
   bool progress = false;

   foreach_block_and_inst_safe(block, vec4_instruction, inst, cfg) {
      if (inst->dst.file == BAD_FILE ||
          inst->dst.file == ARF ||
          inst->dst.file == FIXED_GRF ||
          inst->is_send_from_grf())
         continue;

      unsigned swizzle;

      /* Determine which channels of the sources are read. */
      switch (inst->opcode) {
      case VEC4_OPCODE_PACK_BYTES:
      case BRW_OPCODE_DP4:
      case BRW_OPCODE_DPH: /* FINISHME: DPH reads only three channels of src0,
                            *           but all four of src1.
                            */
         swizzle = brw_swizzle_for_size(4);
         break;
      case BRW_OPCODE_DP3:
         swizzle = brw_swizzle_for_size(3);
         break;
      case BRW_OPCODE_DP2:
         swizzle = brw_swizzle_for_size(2);
         break;

      case VEC4_OPCODE_TO_DOUBLE:
      case VEC4_OPCODE_DOUBLE_TO_F32:
      case VEC4_OPCODE_DOUBLE_TO_D32:
      case VEC4_OPCODE_DOUBLE_TO_U32:
      case VEC4_OPCODE_PICK_LOW_32BIT:
      case VEC4_OPCODE_PICK_HIGH_32BIT:
      case VEC4_OPCODE_SET_LOW_32BIT:
      case VEC4_OPCODE_SET_HIGH_32BIT:
         swizzle = brw_swizzle_for_size(4);
         break;

      default:
         swizzle = brw_swizzle_for_mask(inst->dst.writemask);
         break;
      }

      /* Update sources' swizzles. */
      for (int i = 0; i < 3; i++) {
         if (inst->src[i].file != VGRF &&
             inst->src[i].file != ATTR &&
             inst->src[i].file != UNIFORM)
            continue;

         const unsigned new_swizzle =
            brw_compose_swizzle(swizzle, inst->src[i].swizzle);
         if (inst->src[i].swizzle != new_swizzle) {
            inst->src[i].swizzle = new_swizzle;
            progress = true;
         }
      }
   }

   if (progress)
      invalidate_live_intervals();

   return progress;
}

void
vec4_visitor::split_uniform_registers()
{
   /* Prior to this, uniforms have been in an array sized according to
    * the number of vector uniforms present, sparsely filled (so an
    * aggregate results in reg indices being skipped over).  Now we're
    * going to cut those aggregates up so each .nr index is one
    * vector.  The goal is to make elimination of unused uniform
    * components easier later.
    */
   foreach_block_and_inst(block, vec4_instruction, inst, cfg) {
      for (int i = 0 ; i < 3; i++) {
	 if (inst->src[i].file != UNIFORM)
	    continue;

	 assert(!inst->src[i].reladdr);

         inst->src[i].nr += inst->src[i].offset / 16;
	 inst->src[i].offset %= 16;
      }
   }
}

/* This function returns the register number where we placed the uniform */
static int
set_push_constant_loc(const int nr_uniforms, int *new_uniform_count,
                      const int src, const int size, const int channel_size,
                      int *new_loc, int *new_chan,
                      int *new_chans_used)
{
   int dst;
   /* Find the lowest place we can slot this uniform in. */
   for (dst = 0; dst < nr_uniforms; dst++) {
      if (ALIGN(new_chans_used[dst], channel_size) + size <= 4)
         break;
   }

   assert(dst < nr_uniforms);

   new_loc[src] = dst;
   new_chan[src] = ALIGN(new_chans_used[dst], channel_size);
   new_chans_used[dst] = ALIGN(new_chans_used[dst], channel_size) + size;

   *new_uniform_count = MAX2(*new_uniform_count, dst + 1);
   return dst;
}

void
vec4_visitor::pack_uniform_registers()
{
   uint8_t chans_used[this->uniforms];
   int new_loc[this->uniforms];
   int new_chan[this->uniforms];
   bool is_aligned_to_dvec4[this->uniforms];
   int new_chans_used[this->uniforms];
   int channel_sizes[this->uniforms];

   memset(chans_used, 0, sizeof(chans_used));
   memset(new_loc, 0, sizeof(new_loc));
   memset(new_chan, 0, sizeof(new_chan));
   memset(new_chans_used, 0, sizeof(new_chans_used));
   memset(is_aligned_to_dvec4, 0, sizeof(is_aligned_to_dvec4));
   memset(channel_sizes, 0, sizeof(channel_sizes));

   /* Find which uniform vectors are actually used by the program.  We
    * expect unused vector elements when we've moved array access out
    * to pull constants, and from some GLSL code generators like wine.
    */
   foreach_block_and_inst(block, vec4_instruction, inst, cfg) {
      unsigned readmask;
      switch (inst->opcode) {
      case VEC4_OPCODE_PACK_BYTES:
      case BRW_OPCODE_DP4:
      case BRW_OPCODE_DPH:
         readmask = 0xf;
         break;
      case BRW_OPCODE_DP3:
         readmask = 0x7;
         break;
      case BRW_OPCODE_DP2:
         readmask = 0x3;
         break;
      default:
         readmask = inst->dst.writemask;
         break;
      }

      for (int i = 0 ; i < 3; i++) {
         if (inst->src[i].file != UNIFORM)
            continue;

         assert(type_sz(inst->src[i].type) % 4 == 0);
         int channel_size = type_sz(inst->src[i].type) / 4;

         int reg = inst->src[i].nr;
         for (int c = 0; c < 4; c++) {
            if (!(readmask & (1 << c)))
               continue;

            unsigned channel = BRW_GET_SWZ(inst->src[i].swizzle, c) + 1;
            unsigned used = MAX2(chans_used[reg], channel * channel_size);
            if (used <= 4) {
               chans_used[reg] = used;
               channel_sizes[reg] = MAX2(channel_sizes[reg], channel_size);
            } else {
               is_aligned_to_dvec4[reg] = true;
               is_aligned_to_dvec4[reg + 1] = true;
               chans_used[reg + 1] = used - 4;
               channel_sizes[reg + 1] = MAX2(channel_sizes[reg + 1], channel_size);
            }
         }
      }

      if (inst->opcode == SHADER_OPCODE_MOV_INDIRECT &&
          inst->src[0].file == UNIFORM) {
         assert(inst->src[2].file == BRW_IMMEDIATE_VALUE);
         assert(inst->src[0].subnr == 0);

         unsigned bytes_read = inst->src[2].ud;
         assert(bytes_read % 4 == 0);
         unsigned vec4s_read = DIV_ROUND_UP(bytes_read, 16);

         /* We just mark every register touched by a MOV_INDIRECT as being
          * fully used.  This ensures that it doesn't broken up piecewise by
          * the next part of our packing algorithm.
          */
         int reg = inst->src[0].nr;
         int channel_size = type_sz(inst->src[0].type) / 4;
         for (unsigned i = 0; i < vec4s_read; i++) {
            chans_used[reg + i] = 4;
            channel_sizes[reg + i] = MAX2(channel_sizes[reg + i], channel_size);
         }
      }
   }

   int new_uniform_count = 0;

   /* As the uniforms are going to be reordered, take the data from a temporary
    * copy of the original param[].
    */
   uint32_t *param = ralloc_array(NULL, uint32_t, stage_prog_data->nr_params);
   memcpy(param, stage_prog_data->param,
          sizeof(uint32_t) * stage_prog_data->nr_params);

   /* Now, figure out a packing of the live uniform vectors into our
    * push constants. Start with dvec{3,4} because they are aligned to
    * dvec4 size (2 vec4).
    */
   for (int src = 0; src < uniforms; src++) {
      int size = chans_used[src];

      if (size == 0 || !is_aligned_to_dvec4[src])
         continue;

      /* dvec3 are aligned to dvec4 size, apply the alignment of the size
       * to 4 to avoid moving last component of a dvec3 to the available
       * location at the end of a previous dvec3. These available locations
       * could be filled by smaller variables in next loop.
       */
      size = ALIGN(size, 4);
      int dst = set_push_constant_loc(uniforms, &new_uniform_count,
                                      src, size, channel_sizes[src],
                                      new_loc, new_chan,
                                      new_chans_used);
      /* Move the references to the data */
      for (int j = 0; j < size; j++) {
         stage_prog_data->param[dst * 4 + new_chan[src] + j] =
            param[src * 4 + j];
      }
   }

   /* Continue with the rest of data, which is aligned to vec4. */
   for (int src = 0; src < uniforms; src++) {
      int size = chans_used[src];

      if (size == 0 || is_aligned_to_dvec4[src])
         continue;

      int dst = set_push_constant_loc(uniforms, &new_uniform_count,
                                      src, size, channel_sizes[src],
                                      new_loc, new_chan,
                                      new_chans_used);
      /* Move the references to the data */
      for (int j = 0; j < size; j++) {
         stage_prog_data->param[dst * 4 + new_chan[src] + j] =
            param[src * 4 + j];
      }
   }

   ralloc_free(param);
   this->uniforms = new_uniform_count;

   /* Now, update the instructions for our repacked uniforms. */
   foreach_block_and_inst(block, vec4_instruction, inst, cfg) {
      for (int i = 0 ; i < 3; i++) {
         int src = inst->src[i].nr;

         if (inst->src[i].file != UNIFORM)
            continue;

         int chan = new_chan[src] / channel_sizes[src];
         inst->src[i].nr = new_loc[src];
         inst->src[i].swizzle += BRW_SWIZZLE4(chan, chan, chan, chan);
      }
   }
}

/**
 * Does algebraic optimizations (0 * a = 0, 1 * a = a, a + 0 = a).
 *
 * While GLSL IR also performs this optimization, we end up with it in
 * our instruction stream for a couple of reasons.  One is that we
 * sometimes generate silly instructions, for example in array access
 * where we'll generate "ADD offset, index, base" even if base is 0.
 * The other is that GLSL IR's constant propagation doesn't track the
 * components of aggregates, so some VS patterns (initialize matrix to
 * 0, accumulate in vertex blending factors) end up breaking down to
 * instructions involving 0.
 */
bool
vec4_visitor::opt_algebraic()
{
   bool progress = false;

   foreach_block_and_inst(block, vec4_instruction, inst, cfg) {
      switch (inst->opcode) {
      case BRW_OPCODE_MOV:
         if (inst->src[0].file != IMM)
            break;

         if (inst->saturate) {
            /* Full mixed-type saturates don't happen.  However, we can end up
             * with things like:
             *
             *    mov.sat(8) g21<1>DF       -1F
             *
             * Other mixed-size-but-same-base-type cases may also be possible.
             */
            if (inst->dst.type != inst->src[0].type &&
                inst->dst.type != BRW_REGISTER_TYPE_DF &&
                inst->src[0].type != BRW_REGISTER_TYPE_F)
               assert(!"unimplemented: saturate mixed types");

            if (brw_saturate_immediate(inst->src[0].type,
                                       &inst->src[0].as_brw_reg())) {
               inst->saturate = false;
               progress = true;
            }
         }
         break;

      case BRW_OPCODE_OR:
         if (inst->src[1].is_zero()) {
            inst->opcode = BRW_OPCODE_MOV;
            inst->src[1] = src_reg();
            progress = true;
         }
         break;

      case VEC4_OPCODE_UNPACK_UNIFORM:
         if (inst->src[0].file != UNIFORM) {
            inst->opcode = BRW_OPCODE_MOV;
            progress = true;
         }
         break;

      case BRW_OPCODE_ADD:
	 if (inst->src[1].is_zero()) {
	    inst->opcode = BRW_OPCODE_MOV;
	    inst->src[1] = src_reg();
	    progress = true;
	 }
	 break;

      case BRW_OPCODE_MUL:
	 if (inst->src[1].is_zero()) {
	    inst->opcode = BRW_OPCODE_MOV;
	    switch (inst->src[0].type) {
	    case BRW_REGISTER_TYPE_F:
	       inst->src[0] = brw_imm_f(0.0f);
	       break;
	    case BRW_REGISTER_TYPE_D:
	       inst->src[0] = brw_imm_d(0);
	       break;
	    case BRW_REGISTER_TYPE_UD:
	       inst->src[0] = brw_imm_ud(0u);
	       break;
	    default:
	       unreachable("not reached");
	    }
	    inst->src[1] = src_reg();
	    progress = true;
	 } else if (inst->src[1].is_one()) {
	    inst->opcode = BRW_OPCODE_MOV;
	    inst->src[1] = src_reg();
	    progress = true;
         } else if (inst->src[1].is_negative_one()) {
            inst->opcode = BRW_OPCODE_MOV;
            inst->src[0].negate = !inst->src[0].negate;
            inst->src[1] = src_reg();
            progress = true;
	 }
	 break;
      case SHADER_OPCODE_BROADCAST:
         if (is_uniform(inst->src[0]) ||
             inst->src[1].is_zero()) {
            inst->opcode = BRW_OPCODE_MOV;
            inst->src[1] = src_reg();
            inst->force_writemask_all = true;
            progress = true;
         }
         break;

      default:
	 break;
      }
   }

   if (progress)
      invalidate_live_intervals();

   return progress;
}

/**
 * Only a limited number of hardware registers may be used for push
 * constants, so this turns access to the overflowed constants into
 * pull constants.
 */
void
vec4_visitor::move_push_constants_to_pull_constants()
{
   int pull_constant_loc[this->uniforms];

   /* Only allow 32 registers (256 uniform components) as push constants,
    * which is the limit on gen6.
    *
    * If changing this value, note the limitation about total_regs in
    * brw_curbe.c.
    */
   int max_uniform_components = 32 * 8;
   if (this->uniforms * 4 <= max_uniform_components)
      return;

   /* Make some sort of choice as to which uniforms get sent to pull
    * constants.  We could potentially do something clever here like
    * look for the most infrequently used uniform vec4s, but leave
    * that for later.
    */
   for (int i = 0; i < this->uniforms * 4; i += 4) {
      pull_constant_loc[i / 4] = -1;

      if (i >= max_uniform_components) {
         uint32_t *values = &stage_prog_data->param[i];

         /* Try to find an existing copy of this uniform in the pull
          * constants if it was part of an array access already.
          */
         for (unsigned int j = 0; j < stage_prog_data->nr_pull_params; j += 4) {
            int matches;

            for (matches = 0; matches < 4; matches++) {
               if (stage_prog_data->pull_param[j + matches] != values[matches])
                  break;
            }

            if (matches == 4) {
               pull_constant_loc[i / 4] = j / 4;
               break;
            }
         }

         if (pull_constant_loc[i / 4] == -1) {
            assert(stage_prog_data->nr_pull_params % 4 == 0);
            pull_constant_loc[i / 4] = stage_prog_data->nr_pull_params / 4;

            for (int j = 0; j < 4; j++) {
               stage_prog_data->pull_param[stage_prog_data->nr_pull_params++] =
                  values[j];
            }
         }
      }
   }

   /* Now actually rewrite usage of the things we've moved to pull
    * constants.
    */
   foreach_block_and_inst_safe(block, vec4_instruction, inst, cfg) {
      for (int i = 0 ; i < 3; i++) {
         if (inst->src[i].file != UNIFORM ||
             pull_constant_loc[inst->src[i].nr] == -1)
            continue;

         int uniform = inst->src[i].nr;

         const glsl_type *temp_type = type_sz(inst->src[i].type) == 8 ?
            glsl_type::dvec4_type : glsl_type::vec4_type;
         dst_reg temp = dst_reg(this, temp_type);

         emit_pull_constant_load(block, inst, temp, inst->src[i],
                                 pull_constant_loc[uniform], src_reg());

         inst->src[i].file = temp.file;
         inst->src[i].nr = temp.nr;
         inst->src[i].offset %= 16;
         inst->src[i].reladdr = NULL;
      }
   }

   /* Repack push constants to remove the now-unused ones. */
   pack_uniform_registers();
}

/* Conditions for which we want to avoid setting the dependency control bits */
bool
vec4_visitor::is_dep_ctrl_unsafe(const vec4_instruction *inst)
{
#define IS_DWORD(reg) \
   (reg.type == BRW_REGISTER_TYPE_UD || \
    reg.type == BRW_REGISTER_TYPE_D)

#define IS_64BIT(reg) (reg.file != BAD_FILE && type_sz(reg.type) == 8)

   /* From the Cherryview and Broadwell PRMs:
    *
    * "When source or destination datatype is 64b or operation is integer DWord
    * multiply, DepCtrl must not be used."
    *
    * SKL PRMs don't include this restriction, however, gen7 seems to be
    * affected, at least by the 64b restriction, since DepCtrl with double
    * precision instructions seems to produce GPU hangs in some cases.
    */
   if (devinfo->gen == 8 || gen_device_info_is_9lp(devinfo)) {
      if (inst->opcode == BRW_OPCODE_MUL &&
         IS_DWORD(inst->src[0]) &&
         IS_DWORD(inst->src[1]))
         return true;
   }

   if (devinfo->gen >= 7 && devinfo->gen <= 8) {
      if (IS_64BIT(inst->dst) || IS_64BIT(inst->src[0]) ||
          IS_64BIT(inst->src[1]) || IS_64BIT(inst->src[2]))
      return true;
   }

#undef IS_64BIT
#undef IS_DWORD

   if (devinfo->gen >= 8) {
      if (inst->opcode == BRW_OPCODE_F32TO16)
         return true;
   }

   /*
    * mlen:
    * In the presence of send messages, totally interrupt dependency
    * control. They're long enough that the chance of dependency
    * control around them just doesn't matter.
    *
    * predicate:
    * From the Ivy Bridge PRM, volume 4 part 3.7, page 80:
    * When a sequence of NoDDChk and NoDDClr are used, the last instruction that
    * completes the scoreboard clear must have a non-zero execution mask. This
    * means, if any kind of predication can change the execution mask or channel
    * enable of the last instruction, the optimization must be avoided. This is
    * to avoid instructions being shot down the pipeline when no writes are
    * required.
    *
    * math:
    * Dependency control does not work well over math instructions.
    * NB: Discovered empirically
    */
   return (inst->mlen || inst->predicate || inst->is_math());
}

/**
 * Sets the dependency control fields on instructions after register
 * allocation and before the generator is run.
 *
 * When you have a sequence of instructions like:
 *
 * DP4 temp.x vertex uniform[0]
 * DP4 temp.y vertex uniform[0]
 * DP4 temp.z vertex uniform[0]
 * DP4 temp.w vertex uniform[0]
 *
 * The hardware doesn't know that it can actually run the later instructions
 * while the previous ones are in flight, producing stalls.  However, we have
 * manual fields we can set in the instructions that let it do so.
 */
void
vec4_visitor::opt_set_dependency_control()
{
   vec4_instruction *last_grf_write[BRW_MAX_GRF];
   uint8_t grf_channels_written[BRW_MAX_GRF];
   vec4_instruction *last_mrf_write[BRW_MAX_GRF];
   uint8_t mrf_channels_written[BRW_MAX_GRF];

   assert(prog_data->total_grf ||
          !"Must be called after register allocation");

   foreach_block (block, cfg) {
      memset(last_grf_write, 0, sizeof(last_grf_write));
      memset(last_mrf_write, 0, sizeof(last_mrf_write));

      foreach_inst_in_block (vec4_instruction, inst, block) {
         /* If we read from a register that we were doing dependency control
          * on, don't do dependency control across the read.
          */
         for (int i = 0; i < 3; i++) {
            int reg = inst->src[i].nr + inst->src[i].offset / REG_SIZE;
            if (inst->src[i].file == VGRF) {
               last_grf_write[reg] = NULL;
            } else if (inst->src[i].file == FIXED_GRF) {
               memset(last_grf_write, 0, sizeof(last_grf_write));
               break;
            }
            assert(inst->src[i].file != MRF);
         }

         if (is_dep_ctrl_unsafe(inst)) {
            memset(last_grf_write, 0, sizeof(last_grf_write));
            memset(last_mrf_write, 0, sizeof(last_mrf_write));
            continue;
         }

         /* Now, see if we can do dependency control for this instruction
          * against a previous one writing to its destination.
          */
         int reg = inst->dst.nr + inst->dst.offset / REG_SIZE;
         if (inst->dst.file == VGRF || inst->dst.file == FIXED_GRF) {
            if (last_grf_write[reg] &&
                last_grf_write[reg]->dst.offset == inst->dst.offset &&
                !(inst->dst.writemask & grf_channels_written[reg])) {
               last_grf_write[reg]->no_dd_clear = true;
               inst->no_dd_check = true;
            } else {
               grf_channels_written[reg] = 0;
            }

            last_grf_write[reg] = inst;
            grf_channels_written[reg] |= inst->dst.writemask;
         } else if (inst->dst.file == MRF) {
            if (last_mrf_write[reg] &&
                last_mrf_write[reg]->dst.offset == inst->dst.offset &&
                !(inst->dst.writemask & mrf_channels_written[reg])) {
               last_mrf_write[reg]->no_dd_clear = true;
               inst->no_dd_check = true;
            } else {
               mrf_channels_written[reg] = 0;
            }

            last_mrf_write[reg] = inst;
            mrf_channels_written[reg] |= inst->dst.writemask;
         }
      }
   }
}

bool
vec4_instruction::can_reswizzle(const struct gen_device_info *devinfo,
                                int dst_writemask,
                                int swizzle,
                                int swizzle_mask)
{
   /* Gen6 MATH instructions can not execute in align16 mode, so swizzles
    * are not allowed.
    */
   if (devinfo->gen == 6 && is_math() && swizzle != BRW_SWIZZLE_XYZW)
      return false;

   /* We can't swizzle implicit accumulator access.  We'd have to
    * reswizzle the producer of the accumulator value in addition
    * to the consumer (i.e. both MUL and MACH).  Just skip this.
    */
   if (reads_accumulator_implicitly())
      return false;

   if (!can_do_writemask(devinfo) && dst_writemask != WRITEMASK_XYZW)
      return false;

   /* If this instruction sets anything not referenced by swizzle, then we'd
    * totally break it when we reswizzle.
    */
   if (dst.writemask & ~swizzle_mask)
      return false;

   if (mlen > 0)
      return false;

   for (int i = 0; i < 3; i++) {
      if (src[i].is_accumulator())
         return false;
   }

   return true;
}

/**
 * For any channels in the swizzle's source that were populated by this
 * instruction, rewrite the instruction to put the appropriate result directly
 * in those channels.
 *
 * e.g. for swizzle=yywx, MUL a.xy b c -> MUL a.yy_x b.yy z.yy_x
 */
void
vec4_instruction::reswizzle(int dst_writemask, int swizzle)
{
   /* Destination write mask doesn't correspond to source swizzle for the dot
    * product and pack_bytes instructions.
    */
   if (opcode != BRW_OPCODE_DP4 && opcode != BRW_OPCODE_DPH &&
       opcode != BRW_OPCODE_DP3 && opcode != BRW_OPCODE_DP2 &&
       opcode != VEC4_OPCODE_PACK_BYTES) {
      for (int i = 0; i < 3; i++) {
         if (src[i].file == BAD_FILE || src[i].file == IMM)
            continue;

         src[i].swizzle = brw_compose_swizzle(swizzle, src[i].swizzle);
      }
   }

   /* Apply the specified swizzle and writemask to the original mask of
    * written components.
    */
   dst.writemask = dst_writemask &
                   brw_apply_swizzle_to_mask(swizzle, dst.writemask);
}

/*
 * Tries to reduce extra MOV instructions by taking temporary GRFs that get
 * just written and then MOVed into another reg and making the original write
 * of the GRF write directly to the final destination instead.
 */
bool
vec4_visitor::opt_register_coalesce()
{
   bool progress = false;
   int next_ip = 0;

   calculate_live_intervals();

   foreach_block_and_inst_safe (block, vec4_instruction, inst, cfg) {
      int ip = next_ip;
      next_ip++;

      if (inst->opcode != BRW_OPCODE_MOV ||
          (inst->dst.file != VGRF && inst->dst.file != MRF) ||
	  inst->predicate ||
	  inst->src[0].file != VGRF ||
	  inst->dst.type != inst->src[0].type ||
	  inst->src[0].abs || inst->src[0].negate || inst->src[0].reladdr)
	 continue;

      /* Remove no-op MOVs */
      if (inst->dst.file == inst->src[0].file &&
          inst->dst.nr == inst->src[0].nr &&
          inst->dst.offset == inst->src[0].offset) {
         bool is_nop_mov = true;

         for (unsigned c = 0; c < 4; c++) {
            if ((inst->dst.writemask & (1 << c)) == 0)
               continue;

            if (BRW_GET_SWZ(inst->src[0].swizzle, c) != c) {
               is_nop_mov = false;
               break;
            }
         }

         if (is_nop_mov) {
            inst->remove(block);
            progress = true;
            continue;
         }
      }

      bool to_mrf = (inst->dst.file == MRF);

      /* Can't coalesce this GRF if someone else was going to
       * read it later.
       */
      if (var_range_end(var_from_reg(alloc, dst_reg(inst->src[0])), 8) > ip)
	 continue;

      /* We need to check interference with the final destination between this
       * instruction and the earliest instruction involved in writing the GRF
       * we're eliminating.  To do that, keep track of which of our source
       * channels we've seen initialized.
       */
      const unsigned chans_needed =
         brw_apply_inv_swizzle_to_mask(inst->src[0].swizzle,
                                       inst->dst.writemask);
      unsigned chans_remaining = chans_needed;

      /* Now walk up the instruction stream trying to see if we can rewrite
       * everything writing to the temporary to write into the destination
       * instead.
       */
      vec4_instruction *_scan_inst = (vec4_instruction *)inst->prev;
      foreach_inst_in_block_reverse_starting_from(vec4_instruction, scan_inst,
                                                  inst) {
         _scan_inst = scan_inst;

         if (regions_overlap(inst->src[0], inst->size_read(0),
                             scan_inst->dst, scan_inst->size_written)) {
            /* Found something writing to the reg we want to coalesce away. */
            if (to_mrf) {
               /* SEND instructions can't have MRF as a destination. */
               if (scan_inst->mlen)
                  break;

               if (devinfo->gen == 6) {
                  /* gen6 math instructions must have the destination be
                   * VGRF, so no compute-to-MRF for them.
                   */
                  if (scan_inst->is_math()) {
                     break;
                  }
               }
            }

            /* VS_OPCODE_UNPACK_FLAGS_SIMD4X2 generates a bunch of mov(1)
             * instructions, and this optimization pass is not capable of
             * handling that.  Bail on these instructions and hope that some
             * later optimization pass can do the right thing after they are
             * expanded.
             */
            if (scan_inst->opcode == VS_OPCODE_UNPACK_FLAGS_SIMD4X2)
               break;

            /* This doesn't handle saturation on the instruction we
             * want to coalesce away if the register types do not match.
             * But if scan_inst is a non type-converting 'mov', we can fix
             * the types later.
             */
            if (inst->saturate &&
                inst->dst.type != scan_inst->dst.type &&
                !(scan_inst->opcode == BRW_OPCODE_MOV &&
                  scan_inst->dst.type == scan_inst->src[0].type))
               break;

            /* Only allow coalescing between registers of the same type size.
             * Otherwise we would need to make the pass aware of the fact that
             * channel sizes are different for single and double precision.
             */
            if (type_sz(inst->src[0].type) != type_sz(scan_inst->src[0].type))
               break;

            /* Check that scan_inst writes the same amount of data as the
             * instruction, otherwise coalescing would lead to writing a
             * different (larger or smaller) region of the destination
             */
            if (scan_inst->size_written != inst->size_written)
               break;

            /* If we can't handle the swizzle, bail. */
            if (!scan_inst->can_reswizzle(devinfo, inst->dst.writemask,
                                          inst->src[0].swizzle,
                                          chans_needed)) {
               break;
            }

            /* This only handles coalescing writes of 8 channels (1 register
             * for single-precision and 2 registers for double-precision)
             * starting at the source offset of the copy instruction.
             */
            if (DIV_ROUND_UP(scan_inst->size_written,
                             type_sz(scan_inst->dst.type)) > 8 ||
                scan_inst->dst.offset != inst->src[0].offset)
               break;

	    /* Mark which channels we found unconditional writes for. */
	    if (!scan_inst->predicate)
               chans_remaining &= ~scan_inst->dst.writemask;

	    if (chans_remaining == 0)
	       break;
	 }

         /* You can't read from an MRF, so if someone else reads our MRF's
          * source GRF that we wanted to rewrite, that stops us.  If it's a
          * GRF we're trying to coalesce to, we don't actually handle
          * rewriting sources so bail in that case as well.
          */
	 bool interfered = false;
	 for (int i = 0; i < 3; i++) {
            if (regions_overlap(inst->src[0], inst->size_read(0),
                                scan_inst->src[i], scan_inst->size_read(i)))
	       interfered = true;
	 }
	 if (interfered)
	    break;

         /* If somebody else writes the same channels of our destination here,
          * we can't coalesce before that.
          */
         if (regions_overlap(inst->dst, inst->size_written,
                             scan_inst->dst, scan_inst->size_written) &&
             (inst->dst.writemask & scan_inst->dst.writemask) != 0) {
            break;
         }

         /* Check for reads of the register we're trying to coalesce into.  We
          * can't go rewriting instructions above that to put some other value
          * in the register instead.
          */
         if (to_mrf && scan_inst->mlen > 0) {
            if (inst->dst.nr >= scan_inst->base_mrf &&
                inst->dst.nr < scan_inst->base_mrf + scan_inst->mlen) {
               break;
            }
         } else {
            for (int i = 0; i < 3; i++) {
               if (regions_overlap(inst->dst, inst->size_written,
                                   scan_inst->src[i], scan_inst->size_read(i)))
                  interfered = true;
            }
            if (interfered)
               break;
         }
      }

      if (chans_remaining == 0) {
	 /* If we've made it here, we have an MOV we want to coalesce out, and
	  * a scan_inst pointing to the earliest instruction involved in
	  * computing the value.  Now go rewrite the instruction stream
	  * between the two.
	  */
         vec4_instruction *scan_inst = _scan_inst;
	 while (scan_inst != inst) {
	    if (scan_inst->dst.file == VGRF &&
                scan_inst->dst.nr == inst->src[0].nr &&
		scan_inst->dst.offset == inst->src[0].offset) {
               scan_inst->reswizzle(inst->dst.writemask,
                                    inst->src[0].swizzle);
	       scan_inst->dst.file = inst->dst.file;
               scan_inst->dst.nr = inst->dst.nr;
	       scan_inst->dst.offset = inst->dst.offset;
               if (inst->saturate &&
                   inst->dst.type != scan_inst->dst.type) {
                  /* If we have reached this point, scan_inst is a non
                   * type-converting 'mov' and we can modify its register types
                   * to match the ones in inst. Otherwise, we could have an
                   * incorrect saturation result.
                   */
                  scan_inst->dst.type = inst->dst.type;
                  scan_inst->src[0].type = inst->src[0].type;
               }
	       scan_inst->saturate |= inst->saturate;
	    }
	    scan_inst = (vec4_instruction *)scan_inst->next;
	 }
	 inst->remove(block);
	 progress = true;
      }
   }

   if (progress)
      invalidate_live_intervals();

   return progress;
}

/**
 * Eliminate FIND_LIVE_CHANNEL instructions occurring outside any control
 * flow.  We could probably do better here with some form of divergence
 * analysis.
 */
bool
vec4_visitor::eliminate_find_live_channel()
{
   bool progress = false;
   unsigned depth = 0;

   if (!brw_stage_has_packed_dispatch(devinfo, stage, stage_prog_data)) {
      /* The optimization below assumes that channel zero is live on thread
       * dispatch, which may not be the case if the fixed function dispatches
       * threads sparsely.
       */
      return false;
   }

   foreach_block_and_inst_safe(block, vec4_instruction, inst, cfg) {
      switch (inst->opcode) {
      case BRW_OPCODE_IF:
      case BRW_OPCODE_DO:
         depth++;
         break;

      case BRW_OPCODE_ENDIF:
      case BRW_OPCODE_WHILE:
         depth--;
         break;

      case SHADER_OPCODE_FIND_LIVE_CHANNEL:
         if (depth == 0) {
            inst->opcode = BRW_OPCODE_MOV;
            inst->src[0] = brw_imm_d(0);
            inst->force_writemask_all = true;
            progress = true;
         }
         break;

      default:
         break;
      }
   }

   return progress;
}

/**
 * Splits virtual GRFs requesting more than one contiguous physical register.
 *
 * We initially create large virtual GRFs for temporary structures, arrays,
 * and matrices, so that the visitor functions can add offsets to work their
 * way down to the actual member being accessed.  But when it comes to
 * optimization, we'd like to treat each register as individual storage if
 * possible.
 *
 * So far, the only thing that might prevent splitting is a send message from
 * a GRF on IVB.
 */
void
vec4_visitor::split_virtual_grfs()
{
   int num_vars = this->alloc.count;
   int new_virtual_grf[num_vars];
   bool split_grf[num_vars];

   memset(new_virtual_grf, 0, sizeof(new_virtual_grf));

   /* Try to split anything > 0 sized. */
   for (int i = 0; i < num_vars; i++) {
      split_grf[i] = this->alloc.sizes[i] != 1;
   }

   /* Check that the instructions are compatible with the registers we're trying
    * to split.
    */
   foreach_block_and_inst(block, vec4_instruction, inst, cfg) {
      if (inst->dst.file == VGRF && regs_written(inst) > 1)
         split_grf[inst->dst.nr] = false;

      for (int i = 0; i < 3; i++) {
         if (inst->src[i].file == VGRF && regs_read(inst, i) > 1)
            split_grf[inst->src[i].nr] = false;
      }
   }

   /* Allocate new space for split regs.  Note that the virtual
    * numbers will be contiguous.
    */
   for (int i = 0; i < num_vars; i++) {
      if (!split_grf[i])
         continue;

      new_virtual_grf[i] = alloc.allocate(1);
      for (unsigned j = 2; j < this->alloc.sizes[i]; j++) {
         unsigned reg = alloc.allocate(1);
         assert(reg == new_virtual_grf[i] + j - 1);
         (void) reg;
      }
      this->alloc.sizes[i] = 1;
   }

   foreach_block_and_inst(block, vec4_instruction, inst, cfg) {
      if (inst->dst.file == VGRF && split_grf[inst->dst.nr] &&
          inst->dst.offset / REG_SIZE != 0) {
         inst->dst.nr = (new_virtual_grf[inst->dst.nr] +
                         inst->dst.offset / REG_SIZE - 1);
         inst->dst.offset %= REG_SIZE;
      }
      for (int i = 0; i < 3; i++) {
         if (inst->src[i].file == VGRF && split_grf[inst->src[i].nr] &&
             inst->src[i].offset / REG_SIZE != 0) {
            inst->src[i].nr = (new_virtual_grf[inst->src[i].nr] +
                                inst->src[i].offset / REG_SIZE - 1);
            inst->src[i].offset %= REG_SIZE;
         }
      }
   }
   invalidate_live_intervals();
}

void
vec4_visitor::dump_instruction(backend_instruction *be_inst)
{
   dump_instruction(be_inst, stderr);
}

void
vec4_visitor::dump_instruction(backend_instruction *be_inst, FILE *file)
{
   vec4_instruction *inst = (vec4_instruction *)be_inst;

   if (inst->predicate) {
      fprintf(file, "(%cf%d.%d%s) ",
              inst->predicate_inverse ? '-' : '+',
              inst->flag_subreg / 2,
              inst->flag_subreg % 2,
              pred_ctrl_align16[inst->predicate]);
   }

   fprintf(file, "%s(%d)", brw_instruction_name(devinfo, inst->opcode),
           inst->exec_size);
   if (inst->saturate)
      fprintf(file, ".sat");
   if (inst->conditional_mod) {
      fprintf(file, "%s", conditional_modifier[inst->conditional_mod]);
      if (!inst->predicate &&
          (devinfo->gen < 5 || (inst->opcode != BRW_OPCODE_SEL &&
                                inst->opcode != BRW_OPCODE_CSEL &&
                                inst->opcode != BRW_OPCODE_IF &&
                                inst->opcode != BRW_OPCODE_WHILE))) {
         fprintf(file, ".f%d.%d", inst->flag_subreg / 2, inst->flag_subreg % 2);
      }
   }
   fprintf(file, " ");

   switch (inst->dst.file) {
   case VGRF:
      fprintf(file, "vgrf%d", inst->dst.nr);
      break;
   case FIXED_GRF:
      fprintf(file, "g%d", inst->dst.nr);
      break;
   case MRF:
      fprintf(file, "m%d", inst->dst.nr);
      break;
   case ARF:
      switch (inst->dst.nr) {
      case BRW_ARF_NULL:
         fprintf(file, "null");
         break;
      case BRW_ARF_ADDRESS:
         fprintf(file, "a0.%d", inst->dst.subnr);
         break;
      case BRW_ARF_ACCUMULATOR:
         fprintf(file, "acc%d", inst->dst.subnr);
         break;
      case BRW_ARF_FLAG:
         fprintf(file, "f%d.%d", inst->dst.nr & 0xf, inst->dst.subnr);
         break;
      default:
         fprintf(file, "arf%d.%d", inst->dst.nr & 0xf, inst->dst.subnr);
         break;
      }
      break;
   case BAD_FILE:
      fprintf(file, "(null)");
      break;
   case IMM:
   case ATTR:
   case UNIFORM:
      unreachable("not reached");
   }
   if (inst->dst.offset ||
       (inst->dst.file == VGRF &&
        alloc.sizes[inst->dst.nr] * REG_SIZE != inst->size_written)) {
      const unsigned reg_size = (inst->dst.file == UNIFORM ? 16 : REG_SIZE);
      fprintf(file, "+%d.%d", inst->dst.offset / reg_size,
              inst->dst.offset % reg_size);
   }
   if (inst->dst.writemask != WRITEMASK_XYZW) {
      fprintf(file, ".");
      if (inst->dst.writemask & 1)
         fprintf(file, "x");
      if (inst->dst.writemask & 2)
         fprintf(file, "y");
      if (inst->dst.writemask & 4)
         fprintf(file, "z");
      if (inst->dst.writemask & 8)
         fprintf(file, "w");
   }
   fprintf(file, ":%s", brw_reg_type_to_letters(inst->dst.type));

   if (inst->src[0].file != BAD_FILE)
      fprintf(file, ", ");

   for (int i = 0; i < 3 && inst->src[i].file != BAD_FILE; i++) {
      if (inst->src[i].negate)
         fprintf(file, "-");
      if (inst->src[i].abs)
         fprintf(file, "|");
      switch (inst->src[i].file) {
      case VGRF:
         fprintf(file, "vgrf%d", inst->src[i].nr);
         break;
      case FIXED_GRF:
         fprintf(file, "g%d.%d", inst->src[i].nr, inst->src[i].subnr);
         break;
      case ATTR:
         fprintf(file, "attr%d", inst->src[i].nr);
         break;
      case UNIFORM:
         fprintf(file, "u%d", inst->src[i].nr);
         break;
      case IMM:
         switch (inst->src[i].type) {
         case BRW_REGISTER_TYPE_F:
            fprintf(file, "%fF", inst->src[i].f);
            break;
         case BRW_REGISTER_TYPE_DF:
            fprintf(file, "%fDF", inst->src[i].df);
            break;
         case BRW_REGISTER_TYPE_D:
            fprintf(file, "%dD", inst->src[i].d);
            break;
         case BRW_REGISTER_TYPE_UD:
            fprintf(file, "%uU", inst->src[i].ud);
            break;
         case BRW_REGISTER_TYPE_VF:
            fprintf(file, "[%-gF, %-gF, %-gF, %-gF]",
                    brw_vf_to_float((inst->src[i].ud >>  0) & 0xff),
                    brw_vf_to_float((inst->src[i].ud >>  8) & 0xff),
                    brw_vf_to_float((inst->src[i].ud >> 16) & 0xff),
                    brw_vf_to_float((inst->src[i].ud >> 24) & 0xff));
            break;
         default:
            fprintf(file, "???");
            break;
         }
         break;
      case ARF:
         switch (inst->src[i].nr) {
         case BRW_ARF_NULL:
            fprintf(file, "null");
            break;
         case BRW_ARF_ADDRESS:
            fprintf(file, "a0.%d", inst->src[i].subnr);
            break;
         case BRW_ARF_ACCUMULATOR:
            fprintf(file, "acc%d", inst->src[i].subnr);
            break;
         case BRW_ARF_FLAG:
            fprintf(file, "f%d.%d", inst->src[i].nr & 0xf, inst->src[i].subnr);
            break;
         default:
            fprintf(file, "arf%d.%d", inst->src[i].nr & 0xf, inst->src[i].subnr);
            break;
         }
         break;
      case BAD_FILE:
         fprintf(file, "(null)");
         break;
      case MRF:
         unreachable("not reached");
      }

      if (inst->src[i].offset ||
          (inst->src[i].file == VGRF &&
           alloc.sizes[inst->src[i].nr] * REG_SIZE != inst->size_read(i))) {
         const unsigned reg_size = (inst->src[i].file == UNIFORM ? 16 : REG_SIZE);
         fprintf(file, "+%d.%d", inst->src[i].offset / reg_size,
                 inst->src[i].offset % reg_size);
      }

      if (inst->src[i].file != IMM) {
         static const char *chans[4] = {"x", "y", "z", "w"};
         fprintf(file, ".");
         for (int c = 0; c < 4; c++) {
            fprintf(file, "%s", chans[BRW_GET_SWZ(inst->src[i].swizzle, c)]);
         }
      }

      if (inst->src[i].abs)
         fprintf(file, "|");

      if (inst->src[i].file != IMM) {
         fprintf(file, ":%s", brw_reg_type_to_letters(inst->src[i].type));
      }

      if (i < 2 && inst->src[i + 1].file != BAD_FILE)
         fprintf(file, ", ");
   }

   if (inst->force_writemask_all)
      fprintf(file, " NoMask");

   if (inst->exec_size != 8)
      fprintf(file, " group%d", inst->group);

   fprintf(file, "\n");
}


int
vec4_vs_visitor::setup_attributes(int payload_reg)
{
   foreach_block_and_inst(block, vec4_instruction, inst, cfg) {
      for (int i = 0; i < 3; i++) {
         if (inst->src[i].file == ATTR) {
            assert(inst->src[i].offset % REG_SIZE == 0);
            int grf = payload_reg + inst->src[i].nr +
                      inst->src[i].offset / REG_SIZE;

            struct brw_reg reg = brw_vec8_grf(grf, 0);
            reg.swizzle = inst->src[i].swizzle;
            reg.type = inst->src[i].type;
            reg.abs = inst->src[i].abs;
            reg.negate = inst->src[i].negate;
            inst->src[i] = reg;
         }
      }
   }

   return payload_reg + vs_prog_data->nr_attribute_slots;
}

int
vec4_visitor::setup_uniforms(int reg)
{
   prog_data->base.dispatch_grf_start_reg = reg;

   /* The pre-gen6 VS requires that some push constants get loaded no
    * matter what, or the GPU would hang.
    */
   if (devinfo->gen < 6 && this->uniforms == 0) {
      brw_stage_prog_data_add_params(stage_prog_data, 4);
      for (unsigned int i = 0; i < 4; i++) {
	 unsigned int slot = this->uniforms * 4 + i;
	 stage_prog_data->param[slot] = BRW_PARAM_BUILTIN_ZERO;
      }

      this->uniforms++;
      reg++;
   } else {
      reg += ALIGN(uniforms, 2) / 2;
   }

   for (int i = 0; i < 4; i++)
      reg += stage_prog_data->ubo_ranges[i].length;

   stage_prog_data->nr_params = this->uniforms * 4;

   prog_data->base.curb_read_length =
      reg - prog_data->base.dispatch_grf_start_reg;

   return reg;
}

void
vec4_vs_visitor::setup_payload(void)
{
   int reg = 0;

   /* The payload always contains important data in g0, which contains
    * the URB handles that are passed on to the URB write at the end
    * of the thread.  So, we always start push constants at g1.
    */
   reg++;

   reg = setup_uniforms(reg);

   reg = setup_attributes(reg);

   this->first_non_payload_grf = reg;
}

bool
vec4_visitor::lower_minmax()
{
   assert(devinfo->gen < 6);

   bool progress = false;

   foreach_block_and_inst_safe(block, vec4_instruction, inst, cfg) {
      const vec4_builder ibld(this, block, inst);

      if (inst->opcode == BRW_OPCODE_SEL &&
          inst->predicate == BRW_PREDICATE_NONE) {
         /* FIXME: Using CMP doesn't preserve the NaN propagation semantics of
          *        the original SEL.L/GE instruction
          */
         ibld.CMP(ibld.null_reg_d(), inst->src[0], inst->src[1],
                  inst->conditional_mod);
         inst->predicate = BRW_PREDICATE_NORMAL;
         inst->conditional_mod = BRW_CONDITIONAL_NONE;

         progress = true;
      }
   }

   if (progress)
      invalidate_live_intervals();

   return progress;
}

src_reg
vec4_visitor::get_timestamp()
{
   assert(devinfo->gen >= 7);

   src_reg ts = src_reg(brw_reg(BRW_ARCHITECTURE_REGISTER_FILE,
                                BRW_ARF_TIMESTAMP,
                                0,
                                0,
                                0,
                                BRW_REGISTER_TYPE_UD,
                                BRW_VERTICAL_STRIDE_0,
                                BRW_WIDTH_4,
                                BRW_HORIZONTAL_STRIDE_4,
                                BRW_SWIZZLE_XYZW,
                                WRITEMASK_XYZW));

   dst_reg dst = dst_reg(this, glsl_type::uvec4_type);

   vec4_instruction *mov = emit(MOV(dst, ts));
   /* We want to read the 3 fields we care about (mostly field 0, but also 2)
    * even if it's not enabled in the dispatch.
    */
   mov->force_writemask_all = true;

   return src_reg(dst);
}

void
vec4_visitor::emit_shader_time_begin()
{
   current_annotation = "shader time start";
   shader_start_time = get_timestamp();
}

void
vec4_visitor::emit_shader_time_end()
{
   current_annotation = "shader time end";
   src_reg shader_end_time = get_timestamp();


   /* Check that there weren't any timestamp reset events (assuming these
    * were the only two timestamp reads that happened).
    */
   src_reg reset_end = shader_end_time;
   reset_end.swizzle = BRW_SWIZZLE_ZZZZ;
   vec4_instruction *test = emit(AND(dst_null_ud(), reset_end, brw_imm_ud(1u)));
   test->conditional_mod = BRW_CONDITIONAL_Z;

   emit(IF(BRW_PREDICATE_NORMAL));

   /* Take the current timestamp and get the delta. */
   shader_start_time.negate = true;
   dst_reg diff = dst_reg(this, glsl_type::uint_type);
   emit(ADD(diff, shader_start_time, shader_end_time));

   /* If there were no instructions between the two timestamp gets, the diff
    * is 2 cycles.  Remove that overhead, so I can forget about that when
    * trying to determine the time taken for single instructions.
    */
   emit(ADD(diff, src_reg(diff), brw_imm_ud(-2u)));

   emit_shader_time_write(0, src_reg(diff));
   emit_shader_time_write(1, brw_imm_ud(1u));
   emit(BRW_OPCODE_ELSE);
   emit_shader_time_write(2, brw_imm_ud(1u));
   emit(BRW_OPCODE_ENDIF);
}

void
vec4_visitor::emit_shader_time_write(int shader_time_subindex, src_reg value)
{
   dst_reg dst =
      dst_reg(this, glsl_type::get_array_instance(glsl_type::vec4_type, 2));

   dst_reg offset = dst;
   dst_reg time = dst;
   time.offset += REG_SIZE;

   offset.type = BRW_REGISTER_TYPE_UD;
   int index = shader_time_index * 3 + shader_time_subindex;
   emit(MOV(offset, brw_imm_d(index * BRW_SHADER_TIME_STRIDE)));

   time.type = BRW_REGISTER_TYPE_UD;
   emit(MOV(time, value));

   vec4_instruction *inst =
      emit(SHADER_OPCODE_SHADER_TIME_ADD, dst_reg(), src_reg(dst));
   inst->mlen = 2;
}

static bool
is_align1_df(vec4_instruction *inst)
{
   switch (inst->opcode) {
   case VEC4_OPCODE_DOUBLE_TO_F32:
   case VEC4_OPCODE_DOUBLE_TO_D32:
   case VEC4_OPCODE_DOUBLE_TO_U32:
   case VEC4_OPCODE_TO_DOUBLE:
   case VEC4_OPCODE_PICK_LOW_32BIT:
   case VEC4_OPCODE_PICK_HIGH_32BIT:
   case VEC4_OPCODE_SET_LOW_32BIT:
   case VEC4_OPCODE_SET_HIGH_32BIT:
      return true;
   default:
      return false;
   }
}

/**
 * Three source instruction must have a GRF/MRF destination register.
 * ARF NULL is not allowed.  Fix that up by allocating a temporary GRF.
 */
void
vec4_visitor::fixup_3src_null_dest()
{
   bool progress = false;

   foreach_block_and_inst_safe (block, vec4_instruction, inst, cfg) {
      if (inst->is_3src(devinfo) && inst->dst.is_null()) {
         const unsigned size_written = type_sz(inst->dst.type);
         const unsigned num_regs = DIV_ROUND_UP(size_written, REG_SIZE);

         inst->dst = retype(dst_reg(VGRF, alloc.allocate(num_regs)),
                            inst->dst.type);
         progress = true;
      }
   }

   if (progress)
      invalidate_live_intervals();
}

void
vec4_visitor::convert_to_hw_regs()
{
   foreach_block_and_inst(block, vec4_instruction, inst, cfg) {
      for (int i = 0; i < 3; i++) {
         class src_reg &src = inst->src[i];
         struct brw_reg reg;
         switch (src.file) {
         case VGRF: {
            reg = byte_offset(brw_vecn_grf(4, src.nr, 0), src.offset);
            reg.type = src.type;
            reg.abs = src.abs;
            reg.negate = src.negate;
            break;
         }

         case UNIFORM: {
            reg = stride(byte_offset(brw_vec4_grf(
                                        prog_data->base.dispatch_grf_start_reg +
                                        src.nr / 2, src.nr % 2 * 4),
                                     src.offset),
                         0, 4, 1);
            reg.type = src.type;
            reg.abs = src.abs;
            reg.negate = src.negate;

            /* This should have been moved to pull constants. */
            assert(!src.reladdr);
            break;
         }

         case FIXED_GRF:
            if (type_sz(src.type) == 8) {
               reg = src.as_brw_reg();
               break;
            }
            /* fallthrough */
         case ARF:
         case IMM:
            continue;

         case BAD_FILE:
            /* Probably unused. */
            reg = brw_null_reg();
            reg = retype(reg, src.type);
            break;

         case MRF:
         case ATTR:
            unreachable("not reached");
         }

         apply_logical_swizzle(&reg, inst, i);
         src = reg;

         /* From IVB PRM, vol4, part3, "General Restrictions on Regioning
          * Parameters":
          *
          *   "If ExecSize = Width and HorzStride ≠ 0, VertStride must be set
          *    to Width * HorzStride."
          *
          * We can break this rule with DF sources on DF align1
          * instructions, because the exec_size would be 4 and width is 4.
          * As we know we are not accessing to next GRF, it is safe to
          * set vstride to the formula given by the rule itself.
          */
         if (is_align1_df(inst) && (cvt(inst->exec_size) - 1) == src.width)
            src.vstride = src.width + src.hstride;
      }

      if (inst->is_3src(devinfo)) {
         /* 3-src instructions with scalar sources support arbitrary subnr,
          * but don't actually use swizzles.  Convert swizzle into subnr.
          * Skip this for double-precision instructions: RepCtrl=1 is not
          * allowed for them and needs special handling.
          */
         for (int i = 0; i < 3; i++) {
            if (inst->src[i].vstride == BRW_VERTICAL_STRIDE_0 &&
                type_sz(inst->src[i].type) < 8) {
               assert(brw_is_single_value_swizzle(inst->src[i].swizzle));
               inst->src[i].subnr += 4 * BRW_GET_SWZ(inst->src[i].swizzle, 0);
            }
         }
      }

      dst_reg &dst = inst->dst;
      struct brw_reg reg;

      switch (inst->dst.file) {
      case VGRF:
         reg = byte_offset(brw_vec8_grf(dst.nr, 0), dst.offset);
         reg.type = dst.type;
         reg.writemask = dst.writemask;
         break;

      case MRF:
         reg = byte_offset(brw_message_reg(dst.nr), dst.offset);
         assert((reg.nr & ~BRW_MRF_COMPR4) < BRW_MAX_MRF(devinfo->gen));
         reg.type = dst.type;
         reg.writemask = dst.writemask;
         break;

      case ARF:
      case FIXED_GRF:
         reg = dst.as_brw_reg();
         break;

      case BAD_FILE:
         reg = brw_null_reg();
         reg = retype(reg, dst.type);
         break;

      case IMM:
      case ATTR:
      case UNIFORM:
         unreachable("not reached");
      }

      dst = reg;
   }
}

static bool
stage_uses_interleaved_attributes(unsigned stage,
                                  enum shader_dispatch_mode dispatch_mode)
{
   switch (stage) {
   case MESA_SHADER_TESS_EVAL:
      return true;
   case MESA_SHADER_GEOMETRY:
      return dispatch_mode != DISPATCH_MODE_4X2_DUAL_OBJECT;
   default:
      return false;
   }
}

/**
 * Get the closest native SIMD width supported by the hardware for instruction
 * \p inst.  The instruction will be left untouched by
 * vec4_visitor::lower_simd_width() if the returned value matches the
 * instruction's original execution size.
 */
static unsigned
get_lowered_simd_width(const struct gen_device_info *devinfo,
                       enum shader_dispatch_mode dispatch_mode,
                       unsigned stage, const vec4_instruction *inst)
{
   /* Do not split some instructions that require special handling */
   switch (inst->opcode) {
   case SHADER_OPCODE_GEN4_SCRATCH_READ:
   case SHADER_OPCODE_GEN4_SCRATCH_WRITE:
      return inst->exec_size;
   default:
      break;
   }

   unsigned lowered_width = MIN2(16, inst->exec_size);

   /* We need to split some cases of double-precision instructions that write
    * 2 registers. We only need to care about this in gen7 because that is the
    * only hardware that implements fp64 in Align16.
    */
   if (devinfo->gen == 7 && inst->size_written > REG_SIZE) {
      /* Align16 8-wide double-precision SEL does not work well. Verified
       * empirically.
       */
      if (inst->opcode == BRW_OPCODE_SEL && type_sz(inst->dst.type) == 8)
         lowered_width = MIN2(lowered_width, 4);

      /* HSW PRM, 3D Media GPGPU Engine, Region Alignment Rules for Direct
       * Register Addressing:
       *
       *    "When destination spans two registers, the source MUST span two
       *     registers."
       */
      for (unsigned i = 0; i < 3; i++) {
         if (inst->src[i].file == BAD_FILE)
            continue;
         if (inst->size_read(i) <= REG_SIZE)
            lowered_width = MIN2(lowered_width, 4);

         /* Interleaved attribute setups use a vertical stride of 0, which
          * makes them hit the associated instruction decompression bug in gen7.
          * Split them to prevent this.
          */
         if (inst->src[i].file == ATTR &&
             stage_uses_interleaved_attributes(stage, dispatch_mode))
            lowered_width = MIN2(lowered_width, 4);
      }
   }

   /* IvyBridge can manage a maximum of 4 DFs per SIMD4x2 instruction, since
    * it doesn't support compression in Align16 mode, no matter if it has
    * force_writemask_all enabled or disabled (the latter is affected by the
    * compressed instruction bug in gen7, which is another reason to enforce
    * this limit).
    */
   if (devinfo->gen == 7 && !devinfo->is_haswell &&
       (get_exec_type_size(inst) == 8 || type_sz(inst->dst.type) == 8))
      lowered_width = MIN2(lowered_width, 4);

   return lowered_width;
}

static bool
dst_src_regions_overlap(vec4_instruction *inst)
{
   if (inst->size_written == 0)
      return false;

   unsigned dst_start = inst->dst.offset;
   unsigned dst_end = dst_start + inst->size_written - 1;
   for (int i = 0; i < 3; i++) {
      if (inst->src[i].file == BAD_FILE)
         continue;

      if (inst->dst.file != inst->src[i].file ||
          inst->dst.nr != inst->src[i].nr)
         continue;

      unsigned src_start = inst->src[i].offset;
      unsigned src_end = src_start + inst->size_read(i) - 1;

      if ((dst_start >= src_start && dst_start <= src_end) ||
          (dst_end >= src_start && dst_end <= src_end) ||
          (dst_start <= src_start && dst_end >= src_end)) {
         return true;
      }
   }

   return false;
}

bool
vec4_visitor::lower_simd_width()
{
   bool progress = false;

   foreach_block_and_inst_safe(block, vec4_instruction, inst, cfg) {
      const unsigned lowered_width =
         get_lowered_simd_width(devinfo, prog_data->dispatch_mode, stage, inst);
      assert(lowered_width <= inst->exec_size);
      if (lowered_width == inst->exec_size)
         continue;

      /* We need to deal with source / destination overlaps when splitting.
       * The hardware supports reading from and writing to the same register
       * in the same instruction, but we need to be careful that each split
       * instruction we produce does not corrupt the source of the next.
       *
       * The easiest way to handle this is to make the split instructions write
       * to temporaries if there is an src/dst overlap and then move from the
       * temporaries to the original destination. We also need to consider
       * instructions that do partial writes via align1 opcodes, in which case
       * we need to make sure that the we initialize the temporary with the
       * value of the instruction's dst.
       */
      bool needs_temp = dst_src_regions_overlap(inst);
      for (unsigned n = 0; n < inst->exec_size / lowered_width; n++)  {
         unsigned channel_offset = lowered_width * n;

         unsigned size_written = lowered_width * type_sz(inst->dst.type);

         /* Create the split instruction from the original so that we copy all
          * relevant instruction fields, then set the width and calculate the
          * new dst/src regions.
          */
         vec4_instruction *linst = new(mem_ctx) vec4_instruction(*inst);
         linst->exec_size = lowered_width;
         linst->group = channel_offset;
         linst->size_written = size_written;

         /* Compute split dst region */
         dst_reg dst;
         if (needs_temp) {
            unsigned num_regs = DIV_ROUND_UP(size_written, REG_SIZE);
            dst = retype(dst_reg(VGRF, alloc.allocate(num_regs)),
                         inst->dst.type);
            if (inst->is_align1_partial_write()) {
               vec4_instruction *copy = MOV(dst, src_reg(inst->dst));
               copy->exec_size = lowered_width;
               copy->group = channel_offset;
               copy->size_written = size_written;
               inst->insert_before(block, copy);
            }
         } else {
            dst = horiz_offset(inst->dst, channel_offset);
         }
         linst->dst = dst;

         /* Compute split source regions */
         for (int i = 0; i < 3; i++) {
            if (linst->src[i].file == BAD_FILE)
               continue;

            bool is_interleaved_attr =
               linst->src[i].file == ATTR &&
               stage_uses_interleaved_attributes(stage,
                                                 prog_data->dispatch_mode);

            if (!is_uniform(linst->src[i]) && !is_interleaved_attr)
               linst->src[i] = horiz_offset(linst->src[i], channel_offset);
         }

         inst->insert_before(block, linst);

         /* If we used a temporary to store the result of the split
          * instruction, copy the result to the original destination
          */
         if (needs_temp) {
            vec4_instruction *mov =
               MOV(offset(inst->dst, lowered_width, n), src_reg(dst));
            mov->exec_size = lowered_width;
            mov->group = channel_offset;
            mov->size_written = size_written;
            mov->predicate = inst->predicate;
            inst->insert_before(block, mov);
         }
      }

      inst->remove(block);
      progress = true;
   }

   if (progress)
      invalidate_live_intervals();

   return progress;
}

static brw_predicate
scalarize_predicate(brw_predicate predicate, unsigned writemask)
{
   if (predicate != BRW_PREDICATE_NORMAL)
      return predicate;

   switch (writemask) {
   case WRITEMASK_X:
      return BRW_PREDICATE_ALIGN16_REPLICATE_X;
   case WRITEMASK_Y:
      return BRW_PREDICATE_ALIGN16_REPLICATE_Y;
   case WRITEMASK_Z:
      return BRW_PREDICATE_ALIGN16_REPLICATE_Z;
   case WRITEMASK_W:
      return BRW_PREDICATE_ALIGN16_REPLICATE_W;
   default:
      unreachable("invalid writemask");
   }
}

/* Gen7 has a hardware decompression bug that we can exploit to represent
 * handful of additional swizzles natively.
 */
static bool
is_gen7_supported_64bit_swizzle(vec4_instruction *inst, unsigned arg)
{
   switch (inst->src[arg].swizzle) {
   case BRW_SWIZZLE_XXXX:
   case BRW_SWIZZLE_YYYY:
   case BRW_SWIZZLE_ZZZZ:
   case BRW_SWIZZLE_WWWW:
   case BRW_SWIZZLE_XYXY:
   case BRW_SWIZZLE_YXYX:
   case BRW_SWIZZLE_ZWZW:
   case BRW_SWIZZLE_WZWZ:
      return true;
   default:
      return false;
   }
}

/* 64-bit sources use regions with a width of 2. These 2 elements in each row
 * can be addressed using 32-bit swizzles (which is what the hardware supports)
 * but it also means that the swizzle we apply on the first two components of a
 * dvec4 is coupled with the swizzle we use for the last 2. In other words,
 * only some specific swizzle combinations can be natively supported.
 *
 * FIXME: we can go an step further and implement even more swizzle
 *        variations using only partial scalarization.
 *
 * For more details see:
 * https://bugs.freedesktop.org/show_bug.cgi?id=92760#c82
 */
bool
vec4_visitor::is_supported_64bit_region(vec4_instruction *inst, unsigned arg)
{
   const src_reg &src = inst->src[arg];
   assert(type_sz(src.type) == 8);

   /* Uniform regions have a vstride=0. Because we use 2-wide rows with
    * 64-bit regions it means that we cannot access components Z/W, so
    * return false for any such case. Interleaved attributes will also be
    * mapped to GRF registers with a vstride of 0, so apply the same
    * treatment.
    */
   if ((is_uniform(src) ||
        (stage_uses_interleaved_attributes(stage, prog_data->dispatch_mode) &&
         src.file == ATTR)) &&
       (brw_mask_for_swizzle(src.swizzle) & 12))
      return false;

   switch (src.swizzle) {
   case BRW_SWIZZLE_XYZW:
   case BRW_SWIZZLE_XXZZ:
   case BRW_SWIZZLE_YYWW:
   case BRW_SWIZZLE_YXWZ:
      return true;
   default:
      return devinfo->gen == 7 && is_gen7_supported_64bit_swizzle(inst, arg);
   }
}

bool
vec4_visitor::scalarize_df()
{
   bool progress = false;

   foreach_block_and_inst_safe(block, vec4_instruction, inst, cfg) {
      /* Skip DF instructions that operate in Align1 mode */
      if (is_align1_df(inst))
         continue;

      /* Check if this is a double-precision instruction */
      bool is_double = type_sz(inst->dst.type) == 8;
      for (int arg = 0; !is_double && arg < 3; arg++) {
         is_double = inst->src[arg].file != BAD_FILE &&
                     type_sz(inst->src[arg].type) == 8;
      }

      if (!is_double)
         continue;

      /* Skip the lowering for specific regioning scenarios that we can
       * support natively.
       */
      bool skip_lowering = true;

      /* XY and ZW writemasks operate in 32-bit, which means that they don't
       * have a native 64-bit representation and they should always be split.
       */
      if (inst->dst.writemask == WRITEMASK_XY ||
          inst->dst.writemask == WRITEMASK_ZW) {
         skip_lowering = false;
      } else {
         for (unsigned i = 0; i < 3; i++) {
            if (inst->src[i].file == BAD_FILE || type_sz(inst->src[i].type) < 8)
               continue;
            skip_lowering = skip_lowering && is_supported_64bit_region(inst, i);
         }
      }

      if (skip_lowering)
         continue;

      /* Generate scalar instructions for each enabled channel */
      for (unsigned chan = 0; chan < 4; chan++) {
         unsigned chan_mask = 1 << chan;
         if (!(inst->dst.writemask & chan_mask))
            continue;

         vec4_instruction *scalar_inst = new(mem_ctx) vec4_instruction(*inst);

         for (unsigned i = 0; i < 3; i++) {
            unsigned swz = BRW_GET_SWZ(inst->src[i].swizzle, chan);
            scalar_inst->src[i].swizzle = BRW_SWIZZLE4(swz, swz, swz, swz);
         }

         scalar_inst->dst.writemask = chan_mask;

         if (inst->predicate != BRW_PREDICATE_NONE) {
            scalar_inst->predicate =
               scalarize_predicate(inst->predicate, chan_mask);
         }

         inst->insert_before(block, scalar_inst);
      }

      inst->remove(block);
      progress = true;
   }

   if (progress)
      invalidate_live_intervals();

   return progress;
}

bool
vec4_visitor::lower_64bit_mad_to_mul_add()
{
   bool progress = false;

   foreach_block_and_inst_safe(block, vec4_instruction, inst, cfg) {
      if (inst->opcode != BRW_OPCODE_MAD)
         continue;

      if (type_sz(inst->dst.type) != 8)
         continue;

      dst_reg mul_dst = dst_reg(this, glsl_type::dvec4_type);

      /* Use the copy constructor so we copy all relevant instruction fields
       * from the original mad into the add and mul instructions
       */
      vec4_instruction *mul = new(mem_ctx) vec4_instruction(*inst);
      mul->opcode = BRW_OPCODE_MUL;
      mul->dst = mul_dst;
      mul->src[0] = inst->src[1];
      mul->src[1] = inst->src[2];
      mul->src[2].file = BAD_FILE;

      vec4_instruction *add = new(mem_ctx) vec4_instruction(*inst);
      add->opcode = BRW_OPCODE_ADD;
      add->src[0] = src_reg(mul_dst);
      add->src[1] = inst->src[0];
      add->src[2].file = BAD_FILE;

      inst->insert_before(block, mul);
      inst->insert_before(block, add);
      inst->remove(block);

      progress = true;
   }

   if (progress)
      invalidate_live_intervals();

   return progress;
}

/* The align16 hardware can only do 32-bit swizzle channels, so we need to
 * translate the logical 64-bit swizzle channels that we use in the Vec4 IR
 * to 32-bit swizzle channels in hardware registers.
 *
 * @inst and @arg identify the original vec4 IR source operand we need to
 * translate the swizzle for and @hw_reg is the hardware register where we
 * will write the hardware swizzle to use.
 *
 * This pass assumes that Align16/DF instructions have been fully scalarized
 * previously so there is just one 64-bit swizzle channel to deal with for any
 * given Vec4 IR source.
 */
void
vec4_visitor::apply_logical_swizzle(struct brw_reg *hw_reg,
                                    vec4_instruction *inst, int arg)
{
   src_reg reg = inst->src[arg];

   if (reg.file == BAD_FILE || reg.file == BRW_IMMEDIATE_VALUE)
      return;

   /* If this is not a 64-bit operand or this is a scalar instruction we don't
    * need to do anything about the swizzles.
    */
   if(type_sz(reg.type) < 8 || is_align1_df(inst)) {
      hw_reg->swizzle = reg.swizzle;
      return;
   }

   /* Take the 64-bit logical swizzle channel and translate it to 32-bit */
   assert(brw_is_single_value_swizzle(reg.swizzle) ||
          is_supported_64bit_region(inst, arg));

   /* Apply the region <2, 2, 1> for GRF or <0, 2, 1> for uniforms, as align16
    * HW can only do 32-bit swizzle channels.
    */
   hw_reg->width = BRW_WIDTH_2;

   if (is_supported_64bit_region(inst, arg) &&
       !is_gen7_supported_64bit_swizzle(inst, arg)) {
      /* Supported 64-bit swizzles are those such that their first two
       * components, when expanded to 32-bit swizzles, match the semantics
       * of the original 64-bit swizzle with 2-wide row regioning.
       */
      unsigned swizzle0 = BRW_GET_SWZ(reg.swizzle, 0);
      unsigned swizzle1 = BRW_GET_SWZ(reg.swizzle, 1);
      hw_reg->swizzle = BRW_SWIZZLE4(swizzle0 * 2, swizzle0 * 2 + 1,
                                     swizzle1 * 2, swizzle1 * 2 + 1);
   } else {
      /* If we got here then we have one of the following:
       *
       * 1. An unsupported swizzle, which should be single-value thanks to the
       *    scalarization pass.
       *
       * 2. A gen7 supported swizzle. These can be single-value or double-value
       *    swizzles. If the latter, they are never cross-dvec2 channels. For
       *    these we always need to activate the gen7 vstride=0 exploit.
       */
      unsigned swizzle0 = BRW_GET_SWZ(reg.swizzle, 0);
      unsigned swizzle1 = BRW_GET_SWZ(reg.swizzle, 1);
      assert((swizzle0 < 2) == (swizzle1 < 2));

      /* To gain access to Z/W components we need to select the second half
       * of the register and then use a X/Y swizzle to select Z/W respectively.
       */
      if (swizzle0 >= 2) {
         *hw_reg = suboffset(*hw_reg, 2);
         swizzle0 -= 2;
         swizzle1 -= 2;
      }

      /* All gen7-specific supported swizzles require the vstride=0 exploit */
      if (devinfo->gen == 7 && is_gen7_supported_64bit_swizzle(inst, arg))
         hw_reg->vstride = BRW_VERTICAL_STRIDE_0;

      /* Any 64-bit source with an offset at 16B is intended to address the
       * second half of a register and needs a vertical stride of 0 so we:
       *
       * 1. Don't violate register region restrictions.
       * 2. Activate the gen7 instruction decompresion bug exploit when
       *    execsize > 4
       */
      if (hw_reg->subnr % REG_SIZE == 16) {
         assert(devinfo->gen == 7);
         hw_reg->vstride = BRW_VERTICAL_STRIDE_0;
      }

      hw_reg->swizzle = BRW_SWIZZLE4(swizzle0 * 2, swizzle0 * 2 + 1,
                                     swizzle1 * 2, swizzle1 * 2 + 1);
   }
}

bool
vec4_visitor::run()
{
   if (shader_time_index >= 0)
      emit_shader_time_begin();

   emit_prolog();

   emit_nir_code();
   if (failed)
      return false;
   base_ir = NULL;

   emit_thread_end();

   calculate_cfg();

   /* Before any optimization, push array accesses out to scratch
    * space where we need them to be.  This pass may allocate new
    * virtual GRFs, so we want to do it early.  It also makes sure
    * that we have reladdr computations available for CSE, since we'll
    * often do repeated subexpressions for those.
    */
   move_grf_array_access_to_scratch();
   move_uniform_array_access_to_pull_constants();

   pack_uniform_registers();
   move_push_constants_to_pull_constants();
   split_virtual_grfs();

#define OPT(pass, args...) ({                                          \
      pass_num++;                                                      \
      bool this_progress = pass(args);                                 \
                                                                       \
      if (unlikely(INTEL_DEBUG & DEBUG_OPTIMIZER) && this_progress) {  \
         char filename[64];                                            \
         snprintf(filename, 64, "%s-%s-%02d-%02d-" #pass,              \
                  stage_abbrev, nir->info.name, iteration, pass_num); \
                                                                       \
         backend_shader::dump_instructions(filename);                  \
      }                                                                \
                                                                       \
      progress = progress || this_progress;                            \
      this_progress;                                                   \
   })


   if (unlikely(INTEL_DEBUG & DEBUG_OPTIMIZER)) {
      char filename[64];
      snprintf(filename, 64, "%s-%s-00-00-start",
               stage_abbrev, nir->info.name);

      backend_shader::dump_instructions(filename);
   }

   bool progress;
   int iteration = 0;
   int pass_num = 0;
   do {
      progress = false;
      pass_num = 0;
      iteration++;

      OPT(opt_predicated_break, this);
      OPT(opt_reduce_swizzle);
      OPT(dead_code_eliminate);
      OPT(dead_control_flow_eliminate, this);
      OPT(opt_copy_propagation);
      OPT(opt_cmod_propagation);
      OPT(opt_cse);
      OPT(opt_algebraic);
      OPT(opt_register_coalesce);
      OPT(eliminate_find_live_channel);
   } while (progress);

   pass_num = 0;

   if (OPT(opt_vector_float)) {
      OPT(opt_cse);
      OPT(opt_copy_propagation, false);
      OPT(opt_copy_propagation, true);
      OPT(dead_code_eliminate);
   }

   if (devinfo->gen <= 5 && OPT(lower_minmax)) {
      OPT(opt_cmod_propagation);
      OPT(opt_cse);
      OPT(opt_copy_propagation);
      OPT(dead_code_eliminate);
   }

   if (OPT(lower_simd_width)) {
      OPT(opt_copy_propagation);
      OPT(dead_code_eliminate);
   }

   if (failed)
      return false;

   OPT(lower_64bit_mad_to_mul_add);

   /* Run this before payload setup because tesselation shaders
    * rely on it to prevent cross dvec2 regioning on DF attributes
    * that are setup so that XY are on the second half of register and
    * ZW are in the first half of the next.
    */
   OPT(scalarize_df);

   setup_payload();

   if (unlikely(INTEL_DEBUG & DEBUG_SPILL_VEC4)) {
      /* Debug of register spilling: Go spill everything. */
      const int grf_count = alloc.count;
      float spill_costs[alloc.count];
      bool no_spill[alloc.count];
      evaluate_spill_costs(spill_costs, no_spill);
      for (int i = 0; i < grf_count; i++) {
         if (no_spill[i])
            continue;
         spill_reg(i);
      }

      /* We want to run this after spilling because 64-bit (un)spills need to
       * emit code to shuffle 64-bit data for the 32-bit scratch read/write
       * messages that can produce unsupported 64-bit swizzle regions.
       */
      OPT(scalarize_df);
   }

   fixup_3src_null_dest();

   bool allocated_without_spills = reg_allocate();

   if (!allocated_without_spills) {
      compiler->shader_perf_log(log_data,
                                "%s shader triggered register spilling.  "
                                "Try reducing the number of live vec4 values "
                                "to improve performance.\n",
                                stage_name);

      while (!reg_allocate()) {
         if (failed)
            return false;
      }

      /* We want to run this after spilling because 64-bit (un)spills need to
       * emit code to shuffle 64-bit data for the 32-bit scratch read/write
       * messages that can produce unsupported 64-bit swizzle regions.
       */
      OPT(scalarize_df);
   }

   opt_schedule_instructions();

   opt_set_dependency_control();

   convert_to_hw_regs();

   if (last_scratch > 0) {
      prog_data->base.total_scratch =
         brw_get_scratch_size(last_scratch * REG_SIZE);
   }

   return !failed;
}

} /* namespace brw */

extern "C" {

/**
 * Compile a vertex shader.
 *
 * Returns the final assembly and the program's size.
 */
const unsigned *
brw_compile_vs(const struct brw_compiler *compiler, void *log_data,
               void *mem_ctx,
               const struct brw_vs_prog_key *key,
               struct brw_vs_prog_data *prog_data,
               nir_shader *shader,
               int shader_time_index,
               char **error_str)
{
   const bool is_scalar = compiler->scalar_stage[MESA_SHADER_VERTEX];
   shader = brw_nir_apply_sampler_key(shader, compiler, &key->tex, is_scalar);

   const unsigned *assembly = NULL;

   if (prog_data->base.vue_map.varying_to_slot[VARYING_SLOT_EDGE] != -1) {
      /* If the output VUE map contains VARYING_SLOT_EDGE then we need to copy
       * the edge flag from VERT_ATTRIB_EDGEFLAG.  This will be done
       * automatically by brw_vec4_visitor::emit_urb_slot but we need to
       * ensure that prog_data->inputs_read is accurate.
       *
       * In order to make late NIR passes aware of the change, we actually
       * whack shader->info.inputs_read instead.  This is safe because we just
       * made a copy of the shader.
       */
      assert(!is_scalar);
      assert(key->copy_edgeflag);
      shader->info.inputs_read |= VERT_BIT_EDGEFLAG;
   }

   prog_data->inputs_read = shader->info.inputs_read;
   prog_data->double_inputs_read = shader->info.vs.double_inputs;

   brw_nir_lower_vs_inputs(shader, key->gl_attrib_wa_flags);
   brw_nir_lower_vue_outputs(shader);
   shader = brw_postprocess_nir(shader, compiler, is_scalar);

   prog_data->base.clip_distance_mask =
      ((1 << shader->info.clip_distance_array_size) - 1);
   prog_data->base.cull_distance_mask =
      ((1 << shader->info.cull_distance_array_size) - 1) <<
      shader->info.clip_distance_array_size;

   unsigned nr_attribute_slots = util_bitcount64(prog_data->inputs_read);

   /* gl_VertexID and gl_InstanceID are system values, but arrive via an
    * incoming vertex attribute.  So, add an extra slot.
    */
   if (shader->info.system_values_read &
       (BITFIELD64_BIT(SYSTEM_VALUE_FIRST_VERTEX) |
        BITFIELD64_BIT(SYSTEM_VALUE_BASE_INSTANCE) |
        BITFIELD64_BIT(SYSTEM_VALUE_VERTEX_ID_ZERO_BASE) |
        BITFIELD64_BIT(SYSTEM_VALUE_INSTANCE_ID))) {
      nr_attribute_slots++;
   }

   /* gl_DrawID and IsIndexedDraw share its very own vec4 */
   if (shader->info.system_values_read &
       (BITFIELD64_BIT(SYSTEM_VALUE_DRAW_ID) |
        BITFIELD64_BIT(SYSTEM_VALUE_IS_INDEXED_DRAW))) {
      nr_attribute_slots++;
   }

   if (shader->info.system_values_read &
       BITFIELD64_BIT(SYSTEM_VALUE_IS_INDEXED_DRAW))
      prog_data->uses_is_indexed_draw = true;

   if (shader->info.system_values_read &
       BITFIELD64_BIT(SYSTEM_VALUE_FIRST_VERTEX))
      prog_data->uses_firstvertex = true;

   if (shader->info.system_values_read &
       BITFIELD64_BIT(SYSTEM_VALUE_BASE_INSTANCE))
      prog_data->uses_baseinstance = true;

   if (shader->info.system_values_read &
       BITFIELD64_BIT(SYSTEM_VALUE_VERTEX_ID_ZERO_BASE))
      prog_data->uses_vertexid = true;

   if (shader->info.system_values_read &
       BITFIELD64_BIT(SYSTEM_VALUE_INSTANCE_ID))
      prog_data->uses_instanceid = true;

   if (shader->info.system_values_read &
       BITFIELD64_BIT(SYSTEM_VALUE_DRAW_ID))
          prog_data->uses_drawid = true;

   /* The 3DSTATE_VS documentation lists the lower bound on "Vertex URB Entry
    * Read Length" as 1 in vec4 mode, and 0 in SIMD8 mode.  Empirically, in
    * vec4 mode, the hardware appears to wedge unless we read something.
    */
   if (is_scalar)
      prog_data->base.urb_read_length =
         DIV_ROUND_UP(nr_attribute_slots, 2);
   else
      prog_data->base.urb_read_length =
         DIV_ROUND_UP(MAX2(nr_attribute_slots, 1), 2);

   prog_data->nr_attribute_slots = nr_attribute_slots;

   /* Since vertex shaders reuse the same VUE entry for inputs and outputs
    * (overwriting the original contents), we need to make sure the size is
    * the larger of the two.
    */
   const unsigned vue_entries =
      MAX2(nr_attribute_slots, (unsigned)prog_data->base.vue_map.num_slots);

   if (compiler->devinfo->gen == 6) {
      prog_data->base.urb_entry_size = DIV_ROUND_UP(vue_entries, 8);
   } else {
      prog_data->base.urb_entry_size = DIV_ROUND_UP(vue_entries, 4);
      /* On Cannonlake software shall not program an allocation size that
       * specifies a size that is a multiple of 3 64B (512-bit) cachelines.
       */
      if (compiler->devinfo->gen == 10 &&
          prog_data->base.urb_entry_size % 3 == 0)
         prog_data->base.urb_entry_size++;
   }

   if (INTEL_DEBUG & DEBUG_VS) {
      fprintf(stderr, "VS Output ");
      brw_print_vue_map(stderr, &prog_data->base.vue_map);
   }

   if (is_scalar) {
      prog_data->base.dispatch_mode = DISPATCH_MODE_SIMD8;

      fs_visitor v(compiler, log_data, mem_ctx, key, &prog_data->base.base,
                   NULL, /* prog; Only used for TEXTURE_RECTANGLE on gen < 8 */
                   shader, 8, shader_time_index);
      if (!v.run_vs()) {
         if (error_str)
            *error_str = ralloc_strdup(mem_ctx, v.fail_msg);

         return NULL;
      }

      prog_data->base.base.dispatch_grf_start_reg = v.payload.num_regs;

      fs_generator g(compiler, log_data, mem_ctx,
                     &prog_data->base.base, v.promoted_constants,
                     v.runtime_check_aads_emit, MESA_SHADER_VERTEX);
      if (INTEL_DEBUG & DEBUG_VS) {
         const char *debug_name =
            ralloc_asprintf(mem_ctx, "%s vertex shader %s",
                            shader->info.label ? shader->info.label :
                               "unnamed",
                            shader->info.name);

         g.enable_debug(debug_name);
      }
      g.generate_code(v.cfg, 8);
      assembly = g.get_assembly();
   }

   if (!assembly) {
      prog_data->base.dispatch_mode = DISPATCH_MODE_4X2_DUAL_OBJECT;

      vec4_vs_visitor v(compiler, log_data, key, prog_data,
                        shader, mem_ctx, shader_time_index);
      if (!v.run()) {
         if (error_str)
            *error_str = ralloc_strdup(mem_ctx, v.fail_msg);

         return NULL;
      }

      assembly = brw_vec4_generate_assembly(compiler, log_data, mem_ctx,
                                            shader, &prog_data->base, v.cfg);
   }

   return assembly;
}

} /* extern "C" */
