/*
 * Copyright © 2010 Intel Corporation
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

/** @file brw_fs.cpp
 *
 * This file drives the GLSL IR -> LIR translation, contains the
 * optimizations on the LIR, and drives the generation of native code
 * from the LIR.
 */

#include "main/macros.h"
#include "brw_eu.h"
#include "brw_fs.h"
#include "brw_nir.h"
#include "brw_vec4_gs_visitor.h"
#include "brw_cfg.h"
#include "brw_dead_control_flow.h"
#include "common/gen_debug.h"
#include "compiler/glsl_types.h"
#include "compiler/nir/nir_builder.h"
#include "program/prog_parameter.h"
#include "util/u_math.h"

using namespace brw;

static unsigned get_lowered_simd_width(const struct gen_device_info *devinfo,
                                       const fs_inst *inst);

void
fs_inst::init(enum opcode opcode, uint8_t exec_size, const fs_reg &dst,
              const fs_reg *src, unsigned sources)
{
   memset((void*)this, 0, sizeof(*this));

   this->src = new fs_reg[MAX2(sources, 3)];
   for (unsigned i = 0; i < sources; i++)
      this->src[i] = src[i];

   this->opcode = opcode;
   this->dst = dst;
   this->sources = sources;
   this->exec_size = exec_size;
   this->base_mrf = -1;

   assert(dst.file != IMM && dst.file != UNIFORM);

   assert(this->exec_size != 0);

   this->conditional_mod = BRW_CONDITIONAL_NONE;

   /* This will be the case for almost all instructions. */
   switch (dst.file) {
   case VGRF:
   case ARF:
   case FIXED_GRF:
   case MRF:
   case ATTR:
      this->size_written = dst.component_size(exec_size);
      break;
   case BAD_FILE:
      this->size_written = 0;
      break;
   case IMM:
   case UNIFORM:
      unreachable("Invalid destination register file");
   }

   this->writes_accumulator = false;
}

fs_inst::fs_inst()
{
   init(BRW_OPCODE_NOP, 8, dst, NULL, 0);
}

fs_inst::fs_inst(enum opcode opcode, uint8_t exec_size)
{
   init(opcode, exec_size, reg_undef, NULL, 0);
}

fs_inst::fs_inst(enum opcode opcode, uint8_t exec_size, const fs_reg &dst)
{
   init(opcode, exec_size, dst, NULL, 0);
}

fs_inst::fs_inst(enum opcode opcode, uint8_t exec_size, const fs_reg &dst,
                 const fs_reg &src0)
{
   const fs_reg src[1] = { src0 };
   init(opcode, exec_size, dst, src, 1);
}

fs_inst::fs_inst(enum opcode opcode, uint8_t exec_size, const fs_reg &dst,
                 const fs_reg &src0, const fs_reg &src1)
{
   const fs_reg src[2] = { src0, src1 };
   init(opcode, exec_size, dst, src, 2);
}

fs_inst::fs_inst(enum opcode opcode, uint8_t exec_size, const fs_reg &dst,
                 const fs_reg &src0, const fs_reg &src1, const fs_reg &src2)
{
   const fs_reg src[3] = { src0, src1, src2 };
   init(opcode, exec_size, dst, src, 3);
}

fs_inst::fs_inst(enum opcode opcode, uint8_t exec_width, const fs_reg &dst,
                 const fs_reg src[], unsigned sources)
{
   init(opcode, exec_width, dst, src, sources);
}

fs_inst::fs_inst(const fs_inst &that)
{
   memcpy((void*)this, &that, sizeof(that));

   this->src = new fs_reg[MAX2(that.sources, 3)];

   for (unsigned i = 0; i < that.sources; i++)
      this->src[i] = that.src[i];
}

fs_inst::~fs_inst()
{
   delete[] this->src;
}

void
fs_inst::resize_sources(uint8_t num_sources)
{
   if (this->sources != num_sources) {
      fs_reg *src = new fs_reg[MAX2(num_sources, 3)];

      for (unsigned i = 0; i < MIN2(this->sources, num_sources); ++i)
         src[i] = this->src[i];

      delete[] this->src;
      this->src = src;
      this->sources = num_sources;
   }
}

void
fs_visitor::VARYING_PULL_CONSTANT_LOAD(const fs_builder &bld,
                                       const fs_reg &dst,
                                       const fs_reg &surf_index,
                                       const fs_reg &varying_offset,
                                       uint32_t const_offset)
{
   /* We have our constant surface use a pitch of 4 bytes, so our index can
    * be any component of a vector, and then we load 4 contiguous
    * components starting from that.
    *
    * We break down the const_offset to a portion added to the variable offset
    * and a portion done using fs_reg::offset, which means that if you have
    * GLSL using something like "uniform vec4 a[20]; gl_FragColor = a[i]",
    * we'll temporarily generate 4 vec4 loads from offset i * 4, and CSE can
    * later notice that those loads are all the same and eliminate the
    * redundant ones.
    */
   fs_reg vec4_offset = vgrf(glsl_type::uint_type);
   bld.ADD(vec4_offset, varying_offset, brw_imm_ud(const_offset & ~0xf));

   /* The pull load message will load a vec4 (16 bytes). If we are loading
    * a double this means we are only loading 2 elements worth of data.
    * We also want to use a 32-bit data type for the dst of the load operation
    * so other parts of the driver don't get confused about the size of the
    * result.
    */
   fs_reg vec4_result = bld.vgrf(BRW_REGISTER_TYPE_F, 4);
   fs_inst *inst = bld.emit(FS_OPCODE_VARYING_PULL_CONSTANT_LOAD_LOGICAL,
                            vec4_result, surf_index, vec4_offset);
   inst->size_written = 4 * vec4_result.component_size(inst->exec_size);

   shuffle_from_32bit_read(bld, dst, vec4_result,
                           (const_offset & 0xf) / type_sz(dst.type), 1);
}

/**
 * A helper for MOV generation for fixing up broken hardware SEND dependency
 * handling.
 */
void
fs_visitor::DEP_RESOLVE_MOV(const fs_builder &bld, int grf)
{
   /* The caller always wants uncompressed to emit the minimal extra
    * dependencies, and to avoid having to deal with aligning its regs to 2.
    */
   const fs_builder ubld = bld.annotate("send dependency resolve")
                              .half(0);

   ubld.MOV(ubld.null_reg_f(), fs_reg(VGRF, grf, BRW_REGISTER_TYPE_F));
}

bool
fs_inst::equals(fs_inst *inst) const
{
   return (opcode == inst->opcode &&
           dst.equals(inst->dst) &&
           src[0].equals(inst->src[0]) &&
           src[1].equals(inst->src[1]) &&
           src[2].equals(inst->src[2]) &&
           saturate == inst->saturate &&
           predicate == inst->predicate &&
           conditional_mod == inst->conditional_mod &&
           mlen == inst->mlen &&
           base_mrf == inst->base_mrf &&
           target == inst->target &&
           eot == inst->eot &&
           header_size == inst->header_size &&
           shadow_compare == inst->shadow_compare &&
           exec_size == inst->exec_size &&
           offset == inst->offset);
}

bool
fs_inst::is_send_from_grf() const
{
   switch (opcode) {
   case FS_OPCODE_VARYING_PULL_CONSTANT_LOAD_GEN7:
   case SHADER_OPCODE_SHADER_TIME_ADD:
   case FS_OPCODE_INTERPOLATE_AT_SAMPLE:
   case FS_OPCODE_INTERPOLATE_AT_SHARED_OFFSET:
   case FS_OPCODE_INTERPOLATE_AT_PER_SLOT_OFFSET:
   case SHADER_OPCODE_UNTYPED_ATOMIC:
   case SHADER_OPCODE_UNTYPED_ATOMIC_FLOAT:
   case SHADER_OPCODE_UNTYPED_SURFACE_READ:
   case SHADER_OPCODE_UNTYPED_SURFACE_WRITE:
   case SHADER_OPCODE_BYTE_SCATTERED_WRITE:
   case SHADER_OPCODE_BYTE_SCATTERED_READ:
   case SHADER_OPCODE_TYPED_ATOMIC:
   case SHADER_OPCODE_TYPED_SURFACE_READ:
   case SHADER_OPCODE_TYPED_SURFACE_WRITE:
   case SHADER_OPCODE_URB_WRITE_SIMD8:
   case SHADER_OPCODE_URB_WRITE_SIMD8_PER_SLOT:
   case SHADER_OPCODE_URB_WRITE_SIMD8_MASKED:
   case SHADER_OPCODE_URB_WRITE_SIMD8_MASKED_PER_SLOT:
   case SHADER_OPCODE_URB_READ_SIMD8:
   case SHADER_OPCODE_URB_READ_SIMD8_PER_SLOT:
      return true;
   case FS_OPCODE_UNIFORM_PULL_CONSTANT_LOAD:
      return src[1].file == VGRF;
   case FS_OPCODE_FB_WRITE:
   case FS_OPCODE_FB_READ:
      return src[0].file == VGRF;
   default:
      if (is_tex())
         return src[0].file == VGRF;

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
 * - SIMD16 compressed instructions with certain regioning (see below).
 *
 * The register allocator uses this information to set up conflicts between
 * GRF sources and the destination.
 */
bool
fs_inst::has_source_and_destination_hazard() const
{
   switch (opcode) {
   case FS_OPCODE_PACK_HALF_2x16_SPLIT:
      /* Multiple partial writes to the destination */
      return true;
   case SHADER_OPCODE_SHUFFLE:
      /* This instruction returns an arbitrary channel from the source and
       * gets split into smaller instructions in the generator.  It's possible
       * that one of the instructions will read from a channel corresponding
       * to an earlier instruction.
       */
   case SHADER_OPCODE_SEL_EXEC:
      /* This is implemented as
       *
       * mov(16)      g4<1>D      0D            { align1 WE_all 1H };
       * mov(16)      g4<1>D      g5<8,8,1>D    { align1 1H }
       *
       * Because the source is only read in the second instruction, the first
       * may stomp all over it.
       */
      return true;
   default:
      /* The SIMD16 compressed instruction
       *
       * add(16)      g4<1>F      g4<8,8,1>F   g6<8,8,1>F
       *
       * is actually decoded in hardware as:
       *
       * add(8)       g4<1>F      g4<8,8,1>F   g6<8,8,1>F
       * add(8)       g5<1>F      g5<8,8,1>F   g7<8,8,1>F
       *
       * Which is safe.  However, if we have uniform accesses
       * happening, we get into trouble:
       *
       * add(8)       g4<1>F      g4<0,1,0>F   g6<8,8,1>F
       * add(8)       g5<1>F      g4<0,1,0>F   g7<8,8,1>F
       *
       * Now our destination for the first instruction overwrote the
       * second instruction's src0, and we get garbage for those 8
       * pixels.  There's a similar issue for the pre-gen6
       * pixel_x/pixel_y, which are registers of 16-bit values and thus
       * would get stomped by the first decode as well.
       */
      if (exec_size == 16) {
         for (int i = 0; i < sources; i++) {
            if (src[i].file == VGRF && (src[i].stride == 0 ||
                                        src[i].type == BRW_REGISTER_TYPE_UW ||
                                        src[i].type == BRW_REGISTER_TYPE_W ||
                                        src[i].type == BRW_REGISTER_TYPE_UB ||
                                        src[i].type == BRW_REGISTER_TYPE_B)) {
               return true;
            }
         }
      }
      return false;
   }
}

bool
fs_inst::is_copy_payload(const brw::simple_allocator &grf_alloc) const
{
   if (this->opcode != SHADER_OPCODE_LOAD_PAYLOAD)
      return false;

   fs_reg reg = this->src[0];
   if (reg.file != VGRF || reg.offset != 0 || reg.stride != 1)
      return false;

   if (grf_alloc.sizes[reg.nr] * REG_SIZE != this->size_written)
      return false;

   for (int i = 0; i < this->sources; i++) {
      reg.type = this->src[i].type;
      if (!this->src[i].equals(reg))
         return false;

      if (i < this->header_size) {
         reg.offset += REG_SIZE;
      } else {
         reg = horiz_offset(reg, this->exec_size);
      }
   }

   return true;
}

bool
fs_inst::can_do_source_mods(const struct gen_device_info *devinfo)
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
fs_inst::can_do_cmod()
{
   if (!backend_instruction::can_do_cmod())
      return false;

   /* The accumulator result appears to get used for the conditional modifier
    * generation.  When negating a UD value, there is a 33rd bit generated for
    * the sign in the accumulator value, so now you can't check, for example,
    * equality with a 32-bit value.  See piglit fs-op-neg-uvec4.
    */
   for (unsigned i = 0; i < sources; i++) {
      if (type_is_unsigned_int(src[i].type) && src[i].negate)
         return false;
   }

   return true;
}

bool
fs_inst::can_change_types() const
{
   return dst.type == src[0].type &&
          !src[0].abs && !src[0].negate && !saturate &&
          (opcode == BRW_OPCODE_MOV ||
           (opcode == BRW_OPCODE_SEL &&
            dst.type == src[1].type &&
            predicate != BRW_PREDICATE_NONE &&
            !src[1].abs && !src[1].negate));
}

void
fs_reg::init()
{
   memset((void*)this, 0, sizeof(*this));
   type = BRW_REGISTER_TYPE_UD;
   stride = 1;
}

/** Generic unset register constructor. */
fs_reg::fs_reg()
{
   init();
   this->file = BAD_FILE;
}

fs_reg::fs_reg(struct ::brw_reg reg) :
   backend_reg(reg)
{
   this->offset = 0;
   this->stride = 1;
   if (this->file == IMM &&
       (this->type != BRW_REGISTER_TYPE_V &&
        this->type != BRW_REGISTER_TYPE_UV &&
        this->type != BRW_REGISTER_TYPE_VF)) {
      this->stride = 0;
   }
}

bool
fs_reg::equals(const fs_reg &r) const
{
   return (this->backend_reg::equals(r) &&
           stride == r.stride);
}

bool
fs_reg::negative_equals(const fs_reg &r) const
{
   return (this->backend_reg::negative_equals(r) &&
           stride == r.stride);
}

bool
fs_reg::is_contiguous() const
{
   return stride == 1;
}

unsigned
fs_reg::component_size(unsigned width) const
{
   const unsigned stride = ((file != ARF && file != FIXED_GRF) ? this->stride :
                            hstride == 0 ? 0 :
                            1 << (hstride - 1));
   return MAX2(width * stride, 1) * type_sz(type);
}

extern "C" int
type_size_scalar(const struct glsl_type *type)
{
   unsigned int size, i;

   switch (type->base_type) {
   case GLSL_TYPE_UINT:
   case GLSL_TYPE_INT:
   case GLSL_TYPE_FLOAT:
   case GLSL_TYPE_BOOL:
      return type->components();
   case GLSL_TYPE_UINT16:
   case GLSL_TYPE_INT16:
   case GLSL_TYPE_FLOAT16:
      return DIV_ROUND_UP(type->components(), 2);
   case GLSL_TYPE_UINT8:
   case GLSL_TYPE_INT8:
      return DIV_ROUND_UP(type->components(), 4);
   case GLSL_TYPE_DOUBLE:
   case GLSL_TYPE_UINT64:
   case GLSL_TYPE_INT64:
      return type->components() * 2;
   case GLSL_TYPE_ARRAY:
      return type_size_scalar(type->fields.array) * type->length;
   case GLSL_TYPE_STRUCT:
      size = 0;
      for (i = 0; i < type->length; i++) {
	 size += type_size_scalar(type->fields.structure[i].type);
      }
      return size;
   case GLSL_TYPE_SAMPLER:
   case GLSL_TYPE_ATOMIC_UINT:
   case GLSL_TYPE_IMAGE:
      /* Samplers, atomics, and images take up no register space, since
       * they're baked in at link time.
       */
      return 0;
   case GLSL_TYPE_SUBROUTINE:
      return 1;
   case GLSL_TYPE_VOID:
   case GLSL_TYPE_ERROR:
   case GLSL_TYPE_INTERFACE:
   case GLSL_TYPE_FUNCTION:
      unreachable("not reached");
   }

   return 0;
}

/**
 * Create a MOV to read the timestamp register.
 *
 * The caller is responsible for emitting the MOV.  The return value is
 * the destination of the MOV, with extra parameters set.
 */
fs_reg
fs_visitor::get_timestamp(const fs_builder &bld)
{
   assert(devinfo->gen >= 7);

   fs_reg ts = fs_reg(retype(brw_vec4_reg(BRW_ARCHITECTURE_REGISTER_FILE,
                                          BRW_ARF_TIMESTAMP,
                                          0),
                             BRW_REGISTER_TYPE_UD));

   fs_reg dst = fs_reg(VGRF, alloc.allocate(1), BRW_REGISTER_TYPE_UD);

   /* We want to read the 3 fields we care about even if it's not enabled in
    * the dispatch.
    */
   bld.group(4, 0).exec_all().MOV(dst, ts);

   return dst;
}

void
fs_visitor::emit_shader_time_begin()
{
   /* We want only the low 32 bits of the timestamp.  Since it's running
    * at the GPU clock rate of ~1.2ghz, it will roll over every ~3 seconds,
    * which is plenty of time for our purposes.  It is identical across the
    * EUs, but since it's tracking GPU core speed it will increment at a
    * varying rate as render P-states change.
    */
   shader_start_time = component(
      get_timestamp(bld.annotate("shader time start")), 0);
}

void
fs_visitor::emit_shader_time_end()
{
   /* Insert our code just before the final SEND with EOT. */
   exec_node *end = this->instructions.get_tail();
   assert(end && ((fs_inst *) end)->eot);
   const fs_builder ibld = bld.annotate("shader time end")
                              .exec_all().at(NULL, end);
   const fs_reg timestamp = get_timestamp(ibld);

   /* We only use the low 32 bits of the timestamp - see
    * emit_shader_time_begin()).
    *
    * We could also check if render P-states have changed (or anything
    * else that might disrupt timing) by setting smear to 2 and checking if
    * that field is != 0.
    */
   const fs_reg shader_end_time = component(timestamp, 0);

   /* Check that there weren't any timestamp reset events (assuming these
    * were the only two timestamp reads that happened).
    */
   const fs_reg reset = component(timestamp, 2);
   set_condmod(BRW_CONDITIONAL_Z,
               ibld.AND(ibld.null_reg_ud(), reset, brw_imm_ud(1u)));
   ibld.IF(BRW_PREDICATE_NORMAL);

   fs_reg start = shader_start_time;
   start.negate = true;
   const fs_reg diff = component(fs_reg(VGRF, alloc.allocate(1),
                                        BRW_REGISTER_TYPE_UD),
                                 0);
   const fs_builder cbld = ibld.group(1, 0);
   cbld.group(1, 0).ADD(diff, start, shader_end_time);

   /* If there were no instructions between the two timestamp gets, the diff
    * is 2 cycles.  Remove that overhead, so I can forget about that when
    * trying to determine the time taken for single instructions.
    */
   cbld.ADD(diff, diff, brw_imm_ud(-2u));
   SHADER_TIME_ADD(cbld, 0, diff);
   SHADER_TIME_ADD(cbld, 1, brw_imm_ud(1u));
   ibld.emit(BRW_OPCODE_ELSE);
   SHADER_TIME_ADD(cbld, 2, brw_imm_ud(1u));
   ibld.emit(BRW_OPCODE_ENDIF);
}

void
fs_visitor::SHADER_TIME_ADD(const fs_builder &bld,
                            int shader_time_subindex,
                            fs_reg value)
{
   int index = shader_time_index * 3 + shader_time_subindex;
   struct brw_reg offset = brw_imm_d(index * BRW_SHADER_TIME_STRIDE);

   fs_reg payload;
   if (dispatch_width == 8)
      payload = vgrf(glsl_type::uvec2_type);
   else
      payload = vgrf(glsl_type::uint_type);

   bld.emit(SHADER_OPCODE_SHADER_TIME_ADD, fs_reg(), payload, offset, value);
}

void
fs_visitor::vfail(const char *format, va_list va)
{
   char *msg;

   if (failed)
      return;

   failed = true;

   msg = ralloc_vasprintf(mem_ctx, format, va);
   msg = ralloc_asprintf(mem_ctx, "%s compile failed: %s\n", stage_abbrev, msg);

   this->fail_msg = msg;

   if (debug_enabled) {
      fprintf(stderr, "%s",  msg);
   }
}

void
fs_visitor::fail(const char *format, ...)
{
   va_list va;

   va_start(va, format);
   vfail(format, va);
   va_end(va);
}

/**
 * Mark this program as impossible to compile with dispatch width greater
 * than n.
 *
 * During the SIMD8 compile (which happens first), we can detect and flag
 * things that are unsupported in SIMD16+ mode, so the compiler can skip the
 * SIMD16+ compile altogether.
 *
 * During a compile of dispatch width greater than n (if one happens anyway),
 * this just calls fail().
 */
void
fs_visitor::limit_dispatch_width(unsigned n, const char *msg)
{
   if (dispatch_width > n) {
      fail("%s", msg);
   } else {
      max_dispatch_width = n;
      compiler->shader_perf_log(log_data,
                                "Shader dispatch width limited to SIMD%d: %s",
                                n, msg);
   }
}

/**
 * Returns true if the instruction has a flag that means it won't
 * update an entire destination register.
 *
 * For example, dead code elimination and live variable analysis want to know
 * when a write to a variable screens off any preceding values that were in
 * it.
 */
bool
fs_inst::is_partial_write() const
{
   return ((this->predicate && this->opcode != BRW_OPCODE_SEL) ||
           (this->exec_size * type_sz(this->dst.type)) < 32 ||
           !this->dst.is_contiguous() ||
           this->dst.offset % REG_SIZE != 0);
}

unsigned
fs_inst::components_read(unsigned i) const
{
   /* Return zero if the source is not present. */
   if (src[i].file == BAD_FILE)
      return 0;

   switch (opcode) {
   case FS_OPCODE_LINTERP:
      if (i == 0)
         return 2;
      else
         return 1;

   case FS_OPCODE_PIXEL_X:
   case FS_OPCODE_PIXEL_Y:
      assert(i == 0);
      return 2;

   case FS_OPCODE_FB_WRITE_LOGICAL:
      assert(src[FB_WRITE_LOGICAL_SRC_COMPONENTS].file == IMM);
      /* First/second FB write color. */
      if (i < 2)
         return src[FB_WRITE_LOGICAL_SRC_COMPONENTS].ud;
      else
         return 1;

   case SHADER_OPCODE_TEX_LOGICAL:
   case SHADER_OPCODE_TXD_LOGICAL:
   case SHADER_OPCODE_TXF_LOGICAL:
   case SHADER_OPCODE_TXL_LOGICAL:
   case SHADER_OPCODE_TXS_LOGICAL:
   case FS_OPCODE_TXB_LOGICAL:
   case SHADER_OPCODE_TXF_CMS_LOGICAL:
   case SHADER_OPCODE_TXF_CMS_W_LOGICAL:
   case SHADER_OPCODE_TXF_UMS_LOGICAL:
   case SHADER_OPCODE_TXF_MCS_LOGICAL:
   case SHADER_OPCODE_LOD_LOGICAL:
   case SHADER_OPCODE_TG4_LOGICAL:
   case SHADER_OPCODE_TG4_OFFSET_LOGICAL:
   case SHADER_OPCODE_SAMPLEINFO_LOGICAL:
      assert(src[TEX_LOGICAL_SRC_COORD_COMPONENTS].file == IMM &&
             src[TEX_LOGICAL_SRC_GRAD_COMPONENTS].file == IMM);
      /* Texture coordinates. */
      if (i == TEX_LOGICAL_SRC_COORDINATE)
         return src[TEX_LOGICAL_SRC_COORD_COMPONENTS].ud;
      /* Texture derivatives. */
      else if ((i == TEX_LOGICAL_SRC_LOD || i == TEX_LOGICAL_SRC_LOD2) &&
               opcode == SHADER_OPCODE_TXD_LOGICAL)
         return src[TEX_LOGICAL_SRC_GRAD_COMPONENTS].ud;
      /* Texture offset. */
      else if (i == TEX_LOGICAL_SRC_TG4_OFFSET)
         return 2;
      /* MCS */
      else if (i == TEX_LOGICAL_SRC_MCS && opcode == SHADER_OPCODE_TXF_CMS_W_LOGICAL)
         return 2;
      else
         return 1;

   case SHADER_OPCODE_UNTYPED_SURFACE_READ_LOGICAL:
   case SHADER_OPCODE_TYPED_SURFACE_READ_LOGICAL:
      assert(src[3].file == IMM);
      /* Surface coordinates. */
      if (i == 0)
         return src[3].ud;
      /* Surface operation source (ignored for reads). */
      else if (i == 1)
         return 0;
      else
         return 1;

   case SHADER_OPCODE_UNTYPED_SURFACE_WRITE_LOGICAL:
   case SHADER_OPCODE_TYPED_SURFACE_WRITE_LOGICAL:
      assert(src[3].file == IMM &&
             src[4].file == IMM);
      /* Surface coordinates. */
      if (i == 0)
         return src[3].ud;
      /* Surface operation source. */
      else if (i == 1)
         return src[4].ud;
      else
         return 1;

   case SHADER_OPCODE_BYTE_SCATTERED_READ_LOGICAL:
      /* Scattered logical opcodes use the following params:
       * src[0] Surface coordinates
       * src[1] Surface operation source (ignored for reads)
       * src[2] Surface
       * src[3] IMM with always 1 dimension.
       * src[4] IMM with arg bitsize for scattered read/write 8, 16, 32
       */
      assert(src[3].file == IMM &&
             src[4].file == IMM);
      return i == 1 ? 0 : 1;

   case SHADER_OPCODE_BYTE_SCATTERED_WRITE_LOGICAL:
      assert(src[3].file == IMM &&
             src[4].file == IMM);
      return 1;

   case SHADER_OPCODE_UNTYPED_ATOMIC_LOGICAL:
   case SHADER_OPCODE_TYPED_ATOMIC_LOGICAL: {
      assert(src[3].file == IMM &&
             src[4].file == IMM);
      const unsigned op = src[4].ud;
      /* Surface coordinates. */
      if (i == 0)
         return src[3].ud;
      /* Surface operation source. */
      else if (i == 1 && op == BRW_AOP_CMPWR)
         return 2;
      else if (i == 1 && (op == BRW_AOP_INC || op == BRW_AOP_DEC ||
                          op == BRW_AOP_PREDEC))
         return 0;
      else
         return 1;
   }
   case FS_OPCODE_INTERPOLATE_AT_PER_SLOT_OFFSET:
      return (i == 0 ? 2 : 1);

   case SHADER_OPCODE_UNTYPED_ATOMIC_FLOAT_LOGICAL: {
      assert(src[3].file == IMM &&
             src[4].file == IMM);
      const unsigned op = src[4].ud;
      /* Surface coordinates. */
      if (i == 0)
         return src[3].ud;
      /* Surface operation source. */
      else if (i == 1 && op == BRW_AOP_FCMPWR)
         return 2;
      else
         return 1;
   }

   default:
      return 1;
   }
}

unsigned
fs_inst::size_read(int arg) const
{
   switch (opcode) {
   case FS_OPCODE_FB_WRITE:
   case FS_OPCODE_REP_FB_WRITE:
      if (arg == 0) {
         if (base_mrf >= 0)
            return src[0].file == BAD_FILE ? 0 : 2 * REG_SIZE;
         else
            return mlen * REG_SIZE;
      }
      break;

   case FS_OPCODE_FB_READ:
   case SHADER_OPCODE_URB_WRITE_SIMD8:
   case SHADER_OPCODE_URB_WRITE_SIMD8_PER_SLOT:
   case SHADER_OPCODE_URB_WRITE_SIMD8_MASKED:
   case SHADER_OPCODE_URB_WRITE_SIMD8_MASKED_PER_SLOT:
   case SHADER_OPCODE_URB_READ_SIMD8:
   case SHADER_OPCODE_URB_READ_SIMD8_PER_SLOT:
   case SHADER_OPCODE_UNTYPED_ATOMIC:
   case SHADER_OPCODE_UNTYPED_ATOMIC_FLOAT:
   case SHADER_OPCODE_UNTYPED_SURFACE_READ:
   case SHADER_OPCODE_UNTYPED_SURFACE_WRITE:
   case SHADER_OPCODE_TYPED_ATOMIC:
   case SHADER_OPCODE_TYPED_SURFACE_READ:
   case SHADER_OPCODE_TYPED_SURFACE_WRITE:
   case FS_OPCODE_INTERPOLATE_AT_SAMPLE:
   case FS_OPCODE_INTERPOLATE_AT_SHARED_OFFSET:
   case SHADER_OPCODE_BYTE_SCATTERED_WRITE:
   case SHADER_OPCODE_BYTE_SCATTERED_READ:
      if (arg == 0)
         return mlen * REG_SIZE;
      break;

   case FS_OPCODE_SET_SAMPLE_ID:
      if (arg == 1)
         return 1;
      break;

   case FS_OPCODE_UNIFORM_PULL_CONSTANT_LOAD_GEN7:
      /* The payload is actually stored in src1 */
      if (arg == 1)
         return mlen * REG_SIZE;
      break;

   case FS_OPCODE_LINTERP:
      if (arg == 1)
         return 16;
      break;

   case SHADER_OPCODE_LOAD_PAYLOAD:
      if (arg < this->header_size)
         return REG_SIZE;
      break;

   case CS_OPCODE_CS_TERMINATE:
   case SHADER_OPCODE_BARRIER:
      return REG_SIZE;

   case SHADER_OPCODE_MOV_INDIRECT:
      if (arg == 0) {
         assert(src[2].file == IMM);
         return src[2].ud;
      }
      break;

   default:
      if (is_tex() && arg == 0 && src[0].file == VGRF)
         return mlen * REG_SIZE;
      break;
   }

   switch (src[arg].file) {
   case UNIFORM:
   case IMM:
      return components_read(arg) * type_sz(src[arg].type);
   case BAD_FILE:
   case ARF:
   case FIXED_GRF:
   case VGRF:
   case ATTR:
      return components_read(arg) * src[arg].component_size(exec_size);
   case MRF:
      unreachable("MRF registers are not allowed as sources");
   }
   return 0;
}

namespace {
   /* Return the subset of flag registers that an instruction could
    * potentially read or write based on the execution controls and flag
    * subregister number of the instruction.
    */
   unsigned
   flag_mask(const fs_inst *inst)
   {
      const unsigned start = inst->flag_subreg * 16 + inst->group;
      const unsigned end = start + inst->exec_size;
      return ((1 << DIV_ROUND_UP(end, 8)) - 1) & ~((1 << (start / 8)) - 1);
   }

   unsigned
   bit_mask(unsigned n)
   {
      return (n >= CHAR_BIT * sizeof(bit_mask(n)) ? ~0u : (1u << n) - 1);
   }

   unsigned
   flag_mask(const fs_reg &r, unsigned sz)
   {
      if (r.file == ARF) {
         const unsigned start = (r.nr - BRW_ARF_FLAG) * 4 + r.subnr;
         const unsigned end = start + sz;
         return bit_mask(end) & ~bit_mask(start);
      } else {
         return 0;
      }
   }
}

unsigned
fs_inst::flags_read(const gen_device_info *devinfo) const
{
   if (predicate == BRW_PREDICATE_ALIGN1_ANYV ||
       predicate == BRW_PREDICATE_ALIGN1_ALLV) {
      /* The vertical predication modes combine corresponding bits from
       * f0.0 and f1.0 on Gen7+, and f0.0 and f0.1 on older hardware.
       */
      const unsigned shift = devinfo->gen >= 7 ? 4 : 2;
      return flag_mask(this) << shift | flag_mask(this);
   } else if (predicate) {
      return flag_mask(this);
   } else {
      unsigned mask = 0;
      for (int i = 0; i < sources; i++) {
         mask |= flag_mask(src[i], size_read(i));
      }
      return mask;
   }
}

unsigned
fs_inst::flags_written() const
{
   if ((conditional_mod && (opcode != BRW_OPCODE_SEL &&
                            opcode != BRW_OPCODE_CSEL &&
                            opcode != BRW_OPCODE_IF &&
                            opcode != BRW_OPCODE_WHILE)) ||
       opcode == SHADER_OPCODE_FIND_LIVE_CHANNEL ||
       opcode == FS_OPCODE_FB_WRITE) {
      return flag_mask(this);
   } else {
      return flag_mask(dst, size_written);
   }
}

/**
 * Returns how many MRFs an FS opcode will write over.
 *
 * Note that this is not the 0 or 1 implied writes in an actual gen
 * instruction -- the FS opcodes often generate MOVs in addition.
 */
int
fs_visitor::implied_mrf_writes(fs_inst *inst) const
{
   if (inst->mlen == 0)
      return 0;

   if (inst->base_mrf == -1)
      return 0;

   switch (inst->opcode) {
   case SHADER_OPCODE_RCP:
   case SHADER_OPCODE_RSQ:
   case SHADER_OPCODE_SQRT:
   case SHADER_OPCODE_EXP2:
   case SHADER_OPCODE_LOG2:
   case SHADER_OPCODE_SIN:
   case SHADER_OPCODE_COS:
      return 1 * dispatch_width / 8;
   case SHADER_OPCODE_POW:
   case SHADER_OPCODE_INT_QUOTIENT:
   case SHADER_OPCODE_INT_REMAINDER:
      return 2 * dispatch_width / 8;
   case SHADER_OPCODE_TEX:
   case FS_OPCODE_TXB:
   case SHADER_OPCODE_TXD:
   case SHADER_OPCODE_TXF:
   case SHADER_OPCODE_TXF_CMS:
   case SHADER_OPCODE_TXF_MCS:
   case SHADER_OPCODE_TG4:
   case SHADER_OPCODE_TG4_OFFSET:
   case SHADER_OPCODE_TXL:
   case SHADER_OPCODE_TXS:
   case SHADER_OPCODE_LOD:
   case SHADER_OPCODE_SAMPLEINFO:
      return 1;
   case FS_OPCODE_FB_WRITE:
   case FS_OPCODE_REP_FB_WRITE:
      return inst->src[0].file == BAD_FILE ? 0 : 2;
   case FS_OPCODE_UNIFORM_PULL_CONSTANT_LOAD:
   case SHADER_OPCODE_GEN4_SCRATCH_READ:
      return 1;
   case FS_OPCODE_VARYING_PULL_CONSTANT_LOAD_GEN4:
      return inst->mlen;
   case SHADER_OPCODE_GEN4_SCRATCH_WRITE:
      return inst->mlen;
   default:
      unreachable("not reached");
   }
}

fs_reg
fs_visitor::vgrf(const glsl_type *const type)
{
   int reg_width = dispatch_width / 8;
   return fs_reg(VGRF, alloc.allocate(type_size_scalar(type) * reg_width),
                 brw_type_for_base_type(type));
}

fs_reg::fs_reg(enum brw_reg_file file, int nr)
{
   init();
   this->file = file;
   this->nr = nr;
   this->type = BRW_REGISTER_TYPE_F;
   this->stride = (file == UNIFORM ? 0 : 1);
}

fs_reg::fs_reg(enum brw_reg_file file, int nr, enum brw_reg_type type)
{
   init();
   this->file = file;
   this->nr = nr;
   this->type = type;
   this->stride = (file == UNIFORM ? 0 : 1);
}

/* For SIMD16, we need to follow from the uniform setup of SIMD8 dispatch.
 * This brings in those uniform definitions
 */
void
fs_visitor::import_uniforms(fs_visitor *v)
{
   this->push_constant_loc = v->push_constant_loc;
   this->pull_constant_loc = v->pull_constant_loc;
   this->uniforms = v->uniforms;
   this->subgroup_id = v->subgroup_id;
}

void
fs_visitor::emit_fragcoord_interpolation(fs_reg wpos)
{
   assert(stage == MESA_SHADER_FRAGMENT);

   /* gl_FragCoord.x */
   bld.MOV(wpos, this->pixel_x);
   wpos = offset(wpos, bld, 1);

   /* gl_FragCoord.y */
   bld.MOV(wpos, this->pixel_y);
   wpos = offset(wpos, bld, 1);

   /* gl_FragCoord.z */
   if (devinfo->gen >= 6) {
      bld.MOV(wpos, fetch_payload_reg(bld, payload.source_depth_reg));
   } else {
      bld.emit(FS_OPCODE_LINTERP, wpos,
               this->delta_xy[BRW_BARYCENTRIC_PERSPECTIVE_PIXEL],
               component(interp_reg(VARYING_SLOT_POS, 2), 0));
   }
   wpos = offset(wpos, bld, 1);

   /* gl_FragCoord.w: Already set up in emit_interpolation */
   bld.MOV(wpos, this->wpos_w);
}

enum brw_barycentric_mode
brw_barycentric_mode(enum glsl_interp_mode mode, nir_intrinsic_op op)
{
   /* Barycentric modes don't make sense for flat inputs. */
   assert(mode != INTERP_MODE_FLAT);

   unsigned bary;
   switch (op) {
   case nir_intrinsic_load_barycentric_pixel:
   case nir_intrinsic_load_barycentric_at_offset:
      bary = BRW_BARYCENTRIC_PERSPECTIVE_PIXEL;
      break;
   case nir_intrinsic_load_barycentric_centroid:
      bary = BRW_BARYCENTRIC_PERSPECTIVE_CENTROID;
      break;
   case nir_intrinsic_load_barycentric_sample:
   case nir_intrinsic_load_barycentric_at_sample:
      bary = BRW_BARYCENTRIC_PERSPECTIVE_SAMPLE;
      break;
   default:
      unreachable("invalid intrinsic");
   }

   if (mode == INTERP_MODE_NOPERSPECTIVE)
      bary += 3;

   return (enum brw_barycentric_mode) bary;
}

/**
 * Turn one of the two CENTROID barycentric modes into PIXEL mode.
 */
static enum brw_barycentric_mode
centroid_to_pixel(enum brw_barycentric_mode bary)
{
   assert(bary == BRW_BARYCENTRIC_PERSPECTIVE_CENTROID ||
          bary == BRW_BARYCENTRIC_NONPERSPECTIVE_CENTROID);
   return (enum brw_barycentric_mode) ((unsigned) bary - 1);
}

fs_reg *
fs_visitor::emit_frontfacing_interpolation()
{
   fs_reg *reg = new(this->mem_ctx) fs_reg(vgrf(glsl_type::bool_type));

   if (devinfo->gen >= 6) {
      /* Bit 15 of g0.0 is 0 if the polygon is front facing. We want to create
       * a boolean result from this (~0/true or 0/false).
       *
       * We can use the fact that bit 15 is the MSB of g0.0:W to accomplish
       * this task in only one instruction:
       *    - a negation source modifier will flip the bit; and
       *    - a W -> D type conversion will sign extend the bit into the high
       *      word of the destination.
       *
       * An ASR 15 fills the low word of the destination.
       */
      fs_reg g0 = fs_reg(retype(brw_vec1_grf(0, 0), BRW_REGISTER_TYPE_W));
      g0.negate = true;

      bld.ASR(*reg, g0, brw_imm_d(15));
   } else {
      /* Bit 31 of g1.6 is 0 if the polygon is front facing. We want to create
       * a boolean result from this (1/true or 0/false).
       *
       * Like in the above case, since the bit is the MSB of g1.6:UD we can use
       * the negation source modifier to flip it. Unfortunately the SHR
       * instruction only operates on UD (or D with an abs source modifier)
       * sources without negation.
       *
       * Instead, use ASR (which will give ~0/true or 0/false).
       */
      fs_reg g1_6 = fs_reg(retype(brw_vec1_grf(1, 6), BRW_REGISTER_TYPE_D));
      g1_6.negate = true;

      bld.ASR(*reg, g1_6, brw_imm_d(31));
   }

   return reg;
}

void
fs_visitor::compute_sample_position(fs_reg dst, fs_reg int_sample_pos)
{
   assert(stage == MESA_SHADER_FRAGMENT);
   struct brw_wm_prog_data *wm_prog_data = brw_wm_prog_data(this->prog_data);
   assert(dst.type == BRW_REGISTER_TYPE_F);

   if (wm_prog_data->persample_dispatch) {
      /* Convert int_sample_pos to floating point */
      bld.MOV(dst, int_sample_pos);
      /* Scale to the range [0, 1] */
      bld.MUL(dst, dst, brw_imm_f(1 / 16.0f));
   }
   else {
      /* From ARB_sample_shading specification:
       * "When rendering to a non-multisample buffer, or if multisample
       *  rasterization is disabled, gl_SamplePosition will always be
       *  (0.5, 0.5).
       */
      bld.MOV(dst, brw_imm_f(0.5f));
   }
}

fs_reg *
fs_visitor::emit_samplepos_setup()
{
   assert(devinfo->gen >= 6);

   const fs_builder abld = bld.annotate("compute sample position");
   fs_reg *reg = new(this->mem_ctx) fs_reg(vgrf(glsl_type::vec2_type));
   fs_reg pos = *reg;
   fs_reg int_sample_x = vgrf(glsl_type::int_type);
   fs_reg int_sample_y = vgrf(glsl_type::int_type);

   /* WM will be run in MSDISPMODE_PERSAMPLE. So, only one of SIMD8 or SIMD16
    * mode will be enabled.
    *
    * From the Ivy Bridge PRM, volume 2 part 1, page 344:
    * R31.1:0         Position Offset X/Y for Slot[3:0]
    * R31.3:2         Position Offset X/Y for Slot[7:4]
    * .....
    *
    * The X, Y sample positions come in as bytes in  thread payload. So, read
    * the positions using vstride=16, width=8, hstride=2.
    */
   const fs_reg sample_pos_reg =
      fetch_payload_reg(abld, payload.sample_pos_reg, BRW_REGISTER_TYPE_W);

   /* Compute gl_SamplePosition.x */
   abld.MOV(int_sample_x, subscript(sample_pos_reg, BRW_REGISTER_TYPE_B, 0));
   compute_sample_position(offset(pos, abld, 0), int_sample_x);

   /* Compute gl_SamplePosition.y */
   abld.MOV(int_sample_y, subscript(sample_pos_reg, BRW_REGISTER_TYPE_B, 1));
   compute_sample_position(offset(pos, abld, 1), int_sample_y);
   return reg;
}

fs_reg *
fs_visitor::emit_sampleid_setup()
{
   assert(stage == MESA_SHADER_FRAGMENT);
   brw_wm_prog_key *key = (brw_wm_prog_key*) this->key;
   assert(devinfo->gen >= 6);

   const fs_builder abld = bld.annotate("compute sample id");
   fs_reg *reg = new(this->mem_ctx) fs_reg(vgrf(glsl_type::uint_type));

   if (!key->multisample_fbo) {
      /* As per GL_ARB_sample_shading specification:
       * "When rendering to a non-multisample buffer, or if multisample
       *  rasterization is disabled, gl_SampleID will always be zero."
       */
      abld.MOV(*reg, brw_imm_d(0));
   } else if (devinfo->gen >= 8) {
      /* Sample ID comes in as 4-bit numbers in g1.0:
       *
       *    15:12 Slot 3 SampleID (only used in SIMD16)
       *     11:8 Slot 2 SampleID (only used in SIMD16)
       *      7:4 Slot 1 SampleID
       *      3:0 Slot 0 SampleID
       *
       * Each slot corresponds to four channels, so we want to replicate each
       * half-byte value to 4 channels in a row:
       *
       *    dst+0:    .7    .6    .5    .4    .3    .2    .1    .0
       *             7:4   7:4   7:4   7:4   3:0   3:0   3:0   3:0
       *
       *    dst+1:    .7    .6    .5    .4    .3    .2    .1    .0  (if SIMD16)
       *           15:12 15:12 15:12 15:12  11:8  11:8  11:8  11:8
       *
       * First, we read g1.0 with a <1,8,0>UB region, causing the first 8
       * channels to read the first byte (7:0), and the second group of 8
       * channels to read the second byte (15:8).  Then, we shift right by
       * a vector immediate of <4, 4, 4, 4, 0, 0, 0, 0>, moving the slot 1 / 3
       * values into place.  Finally, we AND with 0xf to keep the low nibble.
       *
       *    shr(16) tmp<1>W g1.0<1,8,0>B 0x44440000:V
       *    and(16) dst<1>D tmp<8,8,1>W  0xf:W
       *
       * TODO: These payload bits exist on Gen7 too, but they appear to always
       *       be zero, so this code fails to work.  We should find out why.
       */
      const fs_reg tmp = abld.vgrf(BRW_REGISTER_TYPE_UW);

      for (unsigned i = 0; i < DIV_ROUND_UP(dispatch_width, 16); i++) {
         const fs_builder hbld = abld.group(MIN2(16, dispatch_width), i);
         hbld.SHR(offset(tmp, hbld, i),
                  stride(retype(brw_vec1_grf(1 + i, 0), BRW_REGISTER_TYPE_UB),
                         1, 8, 0),
                  brw_imm_v(0x44440000));
      }

      abld.AND(*reg, tmp, brw_imm_w(0xf));
   } else {
      const fs_reg t1 = component(abld.vgrf(BRW_REGISTER_TYPE_UD), 0);
      const fs_reg t2 = abld.vgrf(BRW_REGISTER_TYPE_UW);

      /* The PS will be run in MSDISPMODE_PERSAMPLE. For example with
       * 8x multisampling, subspan 0 will represent sample N (where N
       * is 0, 2, 4 or 6), subspan 1 will represent sample 1, 3, 5 or
       * 7. We can find the value of N by looking at R0.0 bits 7:6
       * ("Starting Sample Pair Index (SSPI)") and multiplying by two
       * (since samples are always delivered in pairs). That is, we
       * compute 2*((R0.0 & 0xc0) >> 6) == (R0.0 & 0xc0) >> 5. Then
       * we need to add N to the sequence (0, 0, 0, 0, 1, 1, 1, 1) in
       * case of SIMD8 and sequence (0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
       * 2, 3, 3, 3, 3) in case of SIMD16. We compute this sequence by
       * populating a temporary variable with the sequence (0, 1, 2, 3),
       * and then reading from it using vstride=1, width=4, hstride=0.
       * These computations hold good for 4x multisampling as well.
       *
       * For 2x MSAA and SIMD16, we want to use the sequence (0, 1, 0, 1):
       * the first four slots are sample 0 of subspan 0; the next four
       * are sample 1 of subspan 0; the third group is sample 0 of
       * subspan 1, and finally sample 1 of subspan 1.
       */

      /* SKL+ has an extra bit for the Starting Sample Pair Index to
       * accomodate 16x MSAA.
       */
      abld.exec_all().group(1, 0)
          .AND(t1, fs_reg(retype(brw_vec1_grf(0, 0), BRW_REGISTER_TYPE_UD)),
               brw_imm_ud(0xc0));
      abld.exec_all().group(1, 0).SHR(t1, t1, brw_imm_d(5));

      /* This works for SIMD8-SIMD16.  It also works for SIMD32 but only if we
       * can assume 4x MSAA.  Disallow it on IVB+
       *
       * FINISHME: One day, we could come up with a way to do this that
       * actually works on gen7.
       */
      if (devinfo->gen >= 7)
         limit_dispatch_width(16, "gl_SampleId is unsupported in SIMD32 on gen7");
      abld.exec_all().group(8, 0).MOV(t2, brw_imm_v(0x32103210));

      /* This special instruction takes care of setting vstride=1,
       * width=4, hstride=0 of t2 during an ADD instruction.
       */
      abld.emit(FS_OPCODE_SET_SAMPLE_ID, *reg, t1, t2);
   }

   return reg;
}

fs_reg *
fs_visitor::emit_samplemaskin_setup()
{
   assert(stage == MESA_SHADER_FRAGMENT);
   struct brw_wm_prog_data *wm_prog_data = brw_wm_prog_data(this->prog_data);
   assert(devinfo->gen >= 6);

   fs_reg *reg = new(this->mem_ctx) fs_reg(vgrf(glsl_type::int_type));

   fs_reg coverage_mask =
      fetch_payload_reg(bld, payload.sample_mask_in_reg, BRW_REGISTER_TYPE_D);

   if (wm_prog_data->persample_dispatch) {
      /* gl_SampleMaskIn[] comes from two sources: the input coverage mask,
       * and a mask representing which sample is being processed by the
       * current shader invocation.
       *
       * From the OES_sample_variables specification:
       * "When per-sample shading is active due to the use of a fragment input
       *  qualified by "sample" or due to the use of the gl_SampleID or
       *  gl_SamplePosition variables, only the bit for the current sample is
       *  set in gl_SampleMaskIn."
       */
      const fs_builder abld = bld.annotate("compute gl_SampleMaskIn");

      if (nir_system_values[SYSTEM_VALUE_SAMPLE_ID].file == BAD_FILE)
         nir_system_values[SYSTEM_VALUE_SAMPLE_ID] = *emit_sampleid_setup();

      fs_reg one = vgrf(glsl_type::int_type);
      fs_reg enabled_mask = vgrf(glsl_type::int_type);
      abld.MOV(one, brw_imm_d(1));
      abld.SHL(enabled_mask, one, nir_system_values[SYSTEM_VALUE_SAMPLE_ID]);
      abld.AND(*reg, enabled_mask, coverage_mask);
   } else {
      /* In per-pixel mode, the coverage mask is sufficient. */
      *reg = coverage_mask;
   }
   return reg;
}

fs_reg
fs_visitor::resolve_source_modifiers(const fs_reg &src)
{
   if (!src.abs && !src.negate)
      return src;

   fs_reg temp = bld.vgrf(src.type);
   bld.MOV(temp, src);

   return temp;
}

void
fs_visitor::emit_discard_jump()
{
   assert(brw_wm_prog_data(this->prog_data)->uses_kill);

   /* For performance, after a discard, jump to the end of the
    * shader if all relevant channels have been discarded.
    */
   fs_inst *discard_jump = bld.emit(FS_OPCODE_DISCARD_JUMP);
   discard_jump->flag_subreg = 1;

   discard_jump->predicate = BRW_PREDICATE_ALIGN1_ANY4H;
   discard_jump->predicate_inverse = true;
}

void
fs_visitor::emit_gs_thread_end()
{
   assert(stage == MESA_SHADER_GEOMETRY);

   struct brw_gs_prog_data *gs_prog_data = brw_gs_prog_data(prog_data);

   if (gs_compile->control_data_header_size_bits > 0) {
      emit_gs_control_data_bits(this->final_gs_vertex_count);
   }

   const fs_builder abld = bld.annotate("thread end");
   fs_inst *inst;

   if (gs_prog_data->static_vertex_count != -1) {
      foreach_in_list_reverse(fs_inst, prev, &this->instructions) {
         if (prev->opcode == SHADER_OPCODE_URB_WRITE_SIMD8 ||
             prev->opcode == SHADER_OPCODE_URB_WRITE_SIMD8_MASKED ||
             prev->opcode == SHADER_OPCODE_URB_WRITE_SIMD8_PER_SLOT ||
             prev->opcode == SHADER_OPCODE_URB_WRITE_SIMD8_MASKED_PER_SLOT) {
            prev->eot = true;

            /* Delete now dead instructions. */
            foreach_in_list_reverse_safe(exec_node, dead, &this->instructions) {
               if (dead == prev)
                  break;
               dead->remove();
            }
            return;
         } else if (prev->is_control_flow() || prev->has_side_effects()) {
            break;
         }
      }
      fs_reg hdr = abld.vgrf(BRW_REGISTER_TYPE_UD, 1);
      abld.MOV(hdr, fs_reg(retype(brw_vec8_grf(1, 0), BRW_REGISTER_TYPE_UD)));
      inst = abld.emit(SHADER_OPCODE_URB_WRITE_SIMD8, reg_undef, hdr);
      inst->mlen = 1;
   } else {
      fs_reg payload = abld.vgrf(BRW_REGISTER_TYPE_UD, 2);
      fs_reg *sources = ralloc_array(mem_ctx, fs_reg, 2);
      sources[0] = fs_reg(retype(brw_vec8_grf(1, 0), BRW_REGISTER_TYPE_UD));
      sources[1] = this->final_gs_vertex_count;
      abld.LOAD_PAYLOAD(payload, sources, 2, 2);
      inst = abld.emit(SHADER_OPCODE_URB_WRITE_SIMD8, reg_undef, payload);
      inst->mlen = 2;
   }
   inst->eot = true;
   inst->offset = 0;
}

void
fs_visitor::assign_curb_setup()
{
   unsigned uniform_push_length = DIV_ROUND_UP(stage_prog_data->nr_params, 8);

   unsigned ubo_push_length = 0;
   unsigned ubo_push_start[4];
   for (int i = 0; i < 4; i++) {
      ubo_push_start[i] = 8 * (ubo_push_length + uniform_push_length);
      ubo_push_length += stage_prog_data->ubo_ranges[i].length;
   }

   prog_data->curb_read_length = uniform_push_length + ubo_push_length;

   /* Map the offsets in the UNIFORM file to fixed HW regs. */
   foreach_block_and_inst(block, fs_inst, inst, cfg) {
      for (unsigned int i = 0; i < inst->sources; i++) {
	 if (inst->src[i].file == UNIFORM) {
            int uniform_nr = inst->src[i].nr + inst->src[i].offset / 4;
            int constant_nr;
            if (inst->src[i].nr >= UBO_START) {
               /* constant_nr is in 32-bit units, the rest are in bytes */
               constant_nr = ubo_push_start[inst->src[i].nr - UBO_START] +
                             inst->src[i].offset / 4;
            } else if (uniform_nr >= 0 && uniform_nr < (int) uniforms) {
               constant_nr = push_constant_loc[uniform_nr];
            } else {
               /* Section 5.11 of the OpenGL 4.1 spec says:
                * "Out-of-bounds reads return undefined values, which include
                *  values from other variables of the active program or zero."
                * Just return the first push constant.
                */
               constant_nr = 0;
            }

	    struct brw_reg brw_reg = brw_vec1_grf(payload.num_regs +
						  constant_nr / 8,
						  constant_nr % 8);
            brw_reg.abs = inst->src[i].abs;
            brw_reg.negate = inst->src[i].negate;

            assert(inst->src[i].stride == 0);
            inst->src[i] = byte_offset(
               retype(brw_reg, inst->src[i].type),
               inst->src[i].offset % 4);
	 }
      }
   }

   /* This may be updated in assign_urb_setup or assign_vs_urb_setup. */
   this->first_non_payload_grf = payload.num_regs + prog_data->curb_read_length;
}

void
fs_visitor::calculate_urb_setup()
{
   assert(stage == MESA_SHADER_FRAGMENT);
   struct brw_wm_prog_data *prog_data = brw_wm_prog_data(this->prog_data);
   brw_wm_prog_key *key = (brw_wm_prog_key*) this->key;

   memset(prog_data->urb_setup, -1,
          sizeof(prog_data->urb_setup[0]) * VARYING_SLOT_MAX);

   int urb_next = 0;
   /* Figure out where each of the incoming setup attributes lands. */
   if (devinfo->gen >= 6) {
      if (util_bitcount64(nir->info.inputs_read &
                            BRW_FS_VARYING_INPUT_MASK) <= 16) {
         /* The SF/SBE pipeline stage can do arbitrary rearrangement of the
          * first 16 varying inputs, so we can put them wherever we want.
          * Just put them in order.
          *
          * This is useful because it means that (a) inputs not used by the
          * fragment shader won't take up valuable register space, and (b) we
          * won't have to recompile the fragment shader if it gets paired with
          * a different vertex (or geometry) shader.
          */
         for (unsigned int i = 0; i < VARYING_SLOT_MAX; i++) {
            if (nir->info.inputs_read & BRW_FS_VARYING_INPUT_MASK &
                BITFIELD64_BIT(i)) {
               prog_data->urb_setup[i] = urb_next++;
            }
         }
      } else {
         /* We have enough input varyings that the SF/SBE pipeline stage can't
          * arbitrarily rearrange them to suit our whim; we have to put them
          * in an order that matches the output of the previous pipeline stage
          * (geometry or vertex shader).
          */
         struct brw_vue_map prev_stage_vue_map;
         brw_compute_vue_map(devinfo, &prev_stage_vue_map,
                             key->input_slots_valid,
                             nir->info.separate_shader);

         int first_slot =
            brw_compute_first_urb_slot_required(nir->info.inputs_read,
                                                &prev_stage_vue_map);

         assert(prev_stage_vue_map.num_slots <= first_slot + 32);
         for (int slot = first_slot; slot < prev_stage_vue_map.num_slots;
              slot++) {
            int varying = prev_stage_vue_map.slot_to_varying[slot];
            if (varying != BRW_VARYING_SLOT_PAD &&
                (nir->info.inputs_read & BRW_FS_VARYING_INPUT_MASK &
                 BITFIELD64_BIT(varying))) {
               prog_data->urb_setup[varying] = slot - first_slot;
            }
         }
         urb_next = prev_stage_vue_map.num_slots - first_slot;
      }
   } else {
      /* FINISHME: The sf doesn't map VS->FS inputs for us very well. */
      for (unsigned int i = 0; i < VARYING_SLOT_MAX; i++) {
         /* Point size is packed into the header, not as a general attribute */
         if (i == VARYING_SLOT_PSIZ)
            continue;

	 if (key->input_slots_valid & BITFIELD64_BIT(i)) {
	    /* The back color slot is skipped when the front color is
	     * also written to.  In addition, some slots can be
	     * written in the vertex shader and not read in the
	     * fragment shader.  So the register number must always be
	     * incremented, mapped or not.
	     */
	    if (_mesa_varying_slot_in_fs((gl_varying_slot) i))
	       prog_data->urb_setup[i] = urb_next;
            urb_next++;
	 }
      }

      /*
       * It's a FS only attribute, and we did interpolation for this attribute
       * in SF thread. So, count it here, too.
       *
       * See compile_sf_prog() for more info.
       */
      if (nir->info.inputs_read & BITFIELD64_BIT(VARYING_SLOT_PNTC))
         prog_data->urb_setup[VARYING_SLOT_PNTC] = urb_next++;
   }

   prog_data->num_varying_inputs = urb_next;
}

void
fs_visitor::assign_urb_setup()
{
   assert(stage == MESA_SHADER_FRAGMENT);
   struct brw_wm_prog_data *prog_data = brw_wm_prog_data(this->prog_data);

   int urb_start = payload.num_regs + prog_data->base.curb_read_length;

   /* Offset all the urb_setup[] index by the actual position of the
    * setup regs, now that the location of the constants has been chosen.
    */
   foreach_block_and_inst(block, fs_inst, inst, cfg) {
      for (int i = 0; i < inst->sources; i++) {
         if (inst->src[i].file == ATTR) {
            /* ATTR regs in the FS are in units of logical scalar inputs each
             * of which consumes half of a GRF register.
             */
            assert(inst->src[i].offset < REG_SIZE / 2);
            const unsigned grf = urb_start + inst->src[i].nr / 2;
            const unsigned offset = (inst->src[i].nr % 2) * (REG_SIZE / 2) +
                                    inst->src[i].offset;
            const unsigned width = inst->src[i].stride == 0 ?
                                   1 : MIN2(inst->exec_size, 8);
            struct brw_reg reg = stride(
               byte_offset(retype(brw_vec8_grf(grf, 0), inst->src[i].type),
                           offset),
               width * inst->src[i].stride,
               width, inst->src[i].stride);
            reg.abs = inst->src[i].abs;
            reg.negate = inst->src[i].negate;
            inst->src[i] = reg;
         }
      }
   }

   /* Each attribute is 4 setup channels, each of which is half a reg. */
   this->first_non_payload_grf += prog_data->num_varying_inputs * 2;
}

void
fs_visitor::convert_attr_sources_to_hw_regs(fs_inst *inst)
{
   for (int i = 0; i < inst->sources; i++) {
      if (inst->src[i].file == ATTR) {
         int grf = payload.num_regs +
                   prog_data->curb_read_length +
                   inst->src[i].nr +
                   inst->src[i].offset / REG_SIZE;

         /* As explained at brw_reg_from_fs_reg, From the Haswell PRM:
          *
          * VertStride must be used to cross GRF register boundaries. This
          * rule implies that elements within a 'Width' cannot cross GRF
          * boundaries.
          *
          * So, for registers that are large enough, we have to split the exec
          * size in two and trust the compression state to sort it out.
          */
         unsigned total_size = inst->exec_size *
                               inst->src[i].stride *
                               type_sz(inst->src[i].type);

         assert(total_size <= 2 * REG_SIZE);
         const unsigned exec_size =
            (total_size <= REG_SIZE) ? inst->exec_size : inst->exec_size / 2;

         unsigned width = inst->src[i].stride == 0 ? 1 : exec_size;
         struct brw_reg reg =
            stride(byte_offset(retype(brw_vec8_grf(grf, 0), inst->src[i].type),
                               inst->src[i].offset % REG_SIZE),
                   exec_size * inst->src[i].stride,
                   width, inst->src[i].stride);
         reg.abs = inst->src[i].abs;
         reg.negate = inst->src[i].negate;

         inst->src[i] = reg;
      }
   }
}

void
fs_visitor::assign_vs_urb_setup()
{
   struct brw_vs_prog_data *vs_prog_data = brw_vs_prog_data(prog_data);

   assert(stage == MESA_SHADER_VERTEX);

   /* Each attribute is 4 regs. */
   this->first_non_payload_grf += 4 * vs_prog_data->nr_attribute_slots;

   assert(vs_prog_data->base.urb_read_length <= 15);

   /* Rewrite all ATTR file references to the hw grf that they land in. */
   foreach_block_and_inst(block, fs_inst, inst, cfg) {
      convert_attr_sources_to_hw_regs(inst);
   }
}

void
fs_visitor::assign_tcs_single_patch_urb_setup()
{
   assert(stage == MESA_SHADER_TESS_CTRL);

   /* Rewrite all ATTR file references to HW_REGs. */
   foreach_block_and_inst(block, fs_inst, inst, cfg) {
      convert_attr_sources_to_hw_regs(inst);
   }
}

void
fs_visitor::assign_tes_urb_setup()
{
   assert(stage == MESA_SHADER_TESS_EVAL);

   struct brw_vue_prog_data *vue_prog_data = brw_vue_prog_data(prog_data);

   first_non_payload_grf += 8 * vue_prog_data->urb_read_length;

   /* Rewrite all ATTR file references to HW_REGs. */
   foreach_block_and_inst(block, fs_inst, inst, cfg) {
      convert_attr_sources_to_hw_regs(inst);
   }
}

void
fs_visitor::assign_gs_urb_setup()
{
   assert(stage == MESA_SHADER_GEOMETRY);

   struct brw_vue_prog_data *vue_prog_data = brw_vue_prog_data(prog_data);

   first_non_payload_grf +=
      8 * vue_prog_data->urb_read_length * nir->info.gs.vertices_in;

   foreach_block_and_inst(block, fs_inst, inst, cfg) {
      /* Rewrite all ATTR file references to GRFs. */
      convert_attr_sources_to_hw_regs(inst);
   }
}


/**
 * Split large virtual GRFs into separate components if we can.
 *
 * This is mostly duplicated with what brw_fs_vector_splitting does,
 * but that's really conservative because it's afraid of doing
 * splitting that doesn't result in real progress after the rest of
 * the optimization phases, which would cause infinite looping in
 * optimization.  We can do it once here, safely.  This also has the
 * opportunity to split interpolated values, or maybe even uniforms,
 * which we don't have at the IR level.
 *
 * We want to split, because virtual GRFs are what we register
 * allocate and spill (due to contiguousness requirements for some
 * instructions), and they're what we naturally generate in the
 * codegen process, but most virtual GRFs don't actually need to be
 * contiguous sets of GRFs.  If we split, we'll end up with reduced
 * live intervals and better dead code elimination and coalescing.
 */
void
fs_visitor::split_virtual_grfs()
{
   /* Compact the register file so we eliminate dead vgrfs.  This
    * only defines split points for live registers, so if we have
    * too large dead registers they will hit assertions later.
    */
   compact_virtual_grfs();

   int num_vars = this->alloc.count;

   /* Count the total number of registers */
   int reg_count = 0;
   int vgrf_to_reg[num_vars];
   for (int i = 0; i < num_vars; i++) {
      vgrf_to_reg[i] = reg_count;
      reg_count += alloc.sizes[i];
   }

   /* An array of "split points".  For each register slot, this indicates
    * if this slot can be separated from the previous slot.  Every time an
    * instruction uses multiple elements of a register (as a source or
    * destination), we mark the used slots as inseparable.  Then we go
    * through and split the registers into the smallest pieces we can.
    */
   bool split_points[reg_count];
   memset(split_points, 0, sizeof(split_points));

   /* Mark all used registers as fully splittable */
   foreach_block_and_inst(block, fs_inst, inst, cfg) {
      if (inst->dst.file == VGRF) {
         int reg = vgrf_to_reg[inst->dst.nr];
         for (unsigned j = 1; j < this->alloc.sizes[inst->dst.nr]; j++)
            split_points[reg + j] = true;
      }

      for (int i = 0; i < inst->sources; i++) {
         if (inst->src[i].file == VGRF) {
            int reg = vgrf_to_reg[inst->src[i].nr];
            for (unsigned j = 1; j < this->alloc.sizes[inst->src[i].nr]; j++)
               split_points[reg + j] = true;
         }
      }
   }

   foreach_block_and_inst(block, fs_inst, inst, cfg) {
      if (inst->dst.file == VGRF) {
         int reg = vgrf_to_reg[inst->dst.nr] + inst->dst.offset / REG_SIZE;
         for (unsigned j = 1; j < regs_written(inst); j++)
            split_points[reg + j] = false;
      }
      for (int i = 0; i < inst->sources; i++) {
         if (inst->src[i].file == VGRF) {
            int reg = vgrf_to_reg[inst->src[i].nr] + inst->src[i].offset / REG_SIZE;
            for (unsigned j = 1; j < regs_read(inst, i); j++)
               split_points[reg + j] = false;
         }
      }
   }

   int new_virtual_grf[reg_count];
   int new_reg_offset[reg_count];

   int reg = 0;
   for (int i = 0; i < num_vars; i++) {
      /* The first one should always be 0 as a quick sanity check. */
      assert(split_points[reg] == false);

      /* j = 0 case */
      new_reg_offset[reg] = 0;
      reg++;
      int offset = 1;

      /* j > 0 case */
      for (unsigned j = 1; j < alloc.sizes[i]; j++) {
         /* If this is a split point, reset the offset to 0 and allocate a
          * new virtual GRF for the previous offset many registers
          */
         if (split_points[reg]) {
            assert(offset <= MAX_VGRF_SIZE);
            int grf = alloc.allocate(offset);
            for (int k = reg - offset; k < reg; k++)
               new_virtual_grf[k] = grf;
            offset = 0;
         }
         new_reg_offset[reg] = offset;
         offset++;
         reg++;
      }

      /* The last one gets the original register number */
      assert(offset <= MAX_VGRF_SIZE);
      alloc.sizes[i] = offset;
      for (int k = reg - offset; k < reg; k++)
         new_virtual_grf[k] = i;
   }
   assert(reg == reg_count);

   foreach_block_and_inst(block, fs_inst, inst, cfg) {
      if (inst->dst.file == VGRF) {
         reg = vgrf_to_reg[inst->dst.nr] + inst->dst.offset / REG_SIZE;
         inst->dst.nr = new_virtual_grf[reg];
         inst->dst.offset = new_reg_offset[reg] * REG_SIZE +
                            inst->dst.offset % REG_SIZE;
         assert((unsigned)new_reg_offset[reg] < alloc.sizes[new_virtual_grf[reg]]);
      }
      for (int i = 0; i < inst->sources; i++) {
	 if (inst->src[i].file == VGRF) {
            reg = vgrf_to_reg[inst->src[i].nr] + inst->src[i].offset / REG_SIZE;
            inst->src[i].nr = new_virtual_grf[reg];
            inst->src[i].offset = new_reg_offset[reg] * REG_SIZE +
                                  inst->src[i].offset % REG_SIZE;
            assert((unsigned)new_reg_offset[reg] < alloc.sizes[new_virtual_grf[reg]]);
         }
      }
   }
   invalidate_live_intervals();
}

/**
 * Remove unused virtual GRFs and compact the virtual_grf_* arrays.
 *
 * During code generation, we create tons of temporary variables, many of
 * which get immediately killed and are never used again.  Yet, in later
 * optimization and analysis passes, such as compute_live_intervals, we need
 * to loop over all the virtual GRFs.  Compacting them can save a lot of
 * overhead.
 */
bool
fs_visitor::compact_virtual_grfs()
{
   bool progress = false;
   int remap_table[this->alloc.count];
   memset(remap_table, -1, sizeof(remap_table));

   /* Mark which virtual GRFs are used. */
   foreach_block_and_inst(block, const fs_inst, inst, cfg) {
      if (inst->dst.file == VGRF)
         remap_table[inst->dst.nr] = 0;

      for (int i = 0; i < inst->sources; i++) {
         if (inst->src[i].file == VGRF)
            remap_table[inst->src[i].nr] = 0;
      }
   }

   /* Compact the GRF arrays. */
   int new_index = 0;
   for (unsigned i = 0; i < this->alloc.count; i++) {
      if (remap_table[i] == -1) {
         /* We just found an unused register.  This means that we are
          * actually going to compact something.
          */
         progress = true;
      } else {
         remap_table[i] = new_index;
         alloc.sizes[new_index] = alloc.sizes[i];
         invalidate_live_intervals();
         ++new_index;
      }
   }

   this->alloc.count = new_index;

   /* Patch all the instructions to use the newly renumbered registers */
   foreach_block_and_inst(block, fs_inst, inst, cfg) {
      if (inst->dst.file == VGRF)
         inst->dst.nr = remap_table[inst->dst.nr];

      for (int i = 0; i < inst->sources; i++) {
         if (inst->src[i].file == VGRF)
            inst->src[i].nr = remap_table[inst->src[i].nr];
      }
   }

   /* Patch all the references to delta_xy, since they're used in register
    * allocation.  If they're unused, switch them to BAD_FILE so we don't
    * think some random VGRF is delta_xy.
    */
   for (unsigned i = 0; i < ARRAY_SIZE(delta_xy); i++) {
      if (delta_xy[i].file == VGRF) {
         if (remap_table[delta_xy[i].nr] != -1) {
            delta_xy[i].nr = remap_table[delta_xy[i].nr];
         } else {
            delta_xy[i].file = BAD_FILE;
         }
      }
   }

   return progress;
}

static int
get_subgroup_id_param_index(const brw_stage_prog_data *prog_data)
{
   if (prog_data->nr_params == 0)
      return -1;

   /* The local thread id is always the last parameter in the list */
   uint32_t last_param = prog_data->param[prog_data->nr_params - 1];
   if (last_param == BRW_PARAM_BUILTIN_SUBGROUP_ID)
      return prog_data->nr_params - 1;

   return -1;
}

/**
 * Struct for handling complex alignments.
 *
 * A complex alignment is stored as multiplier and an offset.  A value is
 * considered to be aligned if it is {offset} larger than a multiple of {mul}.
 * For instance, with an alignment of {8, 2}, cplx_align_apply would do the
 * following:
 *
 *  N  | cplx_align_apply({8, 2}, N)
 * ----+-----------------------------
 *  4  | 6
 *  6  | 6
 *  8  | 14
 *  10 | 14
 *  12 | 14
 *  14 | 14
 *  16 | 22
 */
struct cplx_align {
   unsigned mul:4;
   unsigned offset:4;
};

#define CPLX_ALIGN_MAX_MUL 8

static void
cplx_align_assert_sane(struct cplx_align a)
{
   assert(a.mul > 0 && util_is_power_of_two_nonzero(a.mul));
   assert(a.offset < a.mul);
}

/**
 * Combines two alignments to produce a least multiple of sorts.
 *
 * The returned alignment is the smallest (in terms of multiplier) such that
 * anything aligned to both a and b will be aligned to the new alignment.
 * This function will assert-fail if a and b are not compatible, i.e. if the
 * offset parameters are such that no common alignment is possible.
 */
static struct cplx_align
cplx_align_combine(struct cplx_align a, struct cplx_align b)
{
   cplx_align_assert_sane(a);
   cplx_align_assert_sane(b);

   /* Assert that the alignments agree. */
   assert((a.offset & (b.mul - 1)) == (b.offset & (a.mul - 1)));

   return a.mul > b.mul ? a : b;
}

/**
 * Apply a complex alignment
 *
 * This function will return the smallest number greater than or equal to
 * offset that is aligned to align.
 */
static unsigned
cplx_align_apply(struct cplx_align align, unsigned offset)
{
   return ALIGN(offset - align.offset, align.mul) + align.offset;
}

#define UNIFORM_SLOT_SIZE 4

struct uniform_slot_info {
   /** True if the given uniform slot is live */
   unsigned is_live:1;

   /** True if this slot and the next slot must remain contiguous */
   unsigned contiguous:1;

   struct cplx_align align;
};

static void
mark_uniform_slots_read(struct uniform_slot_info *slots,
                        unsigned num_slots, unsigned alignment)
{
   assert(alignment > 0 && util_is_power_of_two_nonzero(alignment));
   assert(alignment <= CPLX_ALIGN_MAX_MUL);

   /* We can't align a slot to anything less than the slot size */
   alignment = MAX2(alignment, UNIFORM_SLOT_SIZE);

   struct cplx_align align = {alignment, 0};
   cplx_align_assert_sane(align);

   for (unsigned i = 0; i < num_slots; i++) {
      slots[i].is_live = true;
      if (i < num_slots - 1)
         slots[i].contiguous = true;

      align.offset = (i * UNIFORM_SLOT_SIZE) & (align.mul - 1);
      if (slots[i].align.mul == 0) {
         slots[i].align = align;
      } else {
         slots[i].align = cplx_align_combine(slots[i].align, align);
      }
   }
}

/**
 * Assign UNIFORM file registers to either push constants or pull constants.
 *
 * We allow a fragment shader to have more than the specified minimum
 * maximum number of fragment shader uniform components (64).  If
 * there are too many of these, they'd fill up all of register space.
 * So, this will push some of them out to the pull constant buffer and
 * update the program to load them.
 */
void
fs_visitor::assign_constant_locations()
{
   /* Only the first compile gets to decide on locations. */
   if (push_constant_loc) {
      assert(pull_constant_loc);
      return;
   }

   struct uniform_slot_info slots[uniforms];
   memset(slots, 0, sizeof(slots));

   foreach_block_and_inst_safe(block, fs_inst, inst, cfg) {
      for (int i = 0 ; i < inst->sources; i++) {
         if (inst->src[i].file != UNIFORM)
            continue;

         /* NIR tightly packs things so the uniform number might not be
          * aligned (if we have a double right after a float, for instance).
          * This is fine because the process of re-arranging them will ensure
          * that things are properly aligned.  The offset into that uniform,
          * however, must be aligned.
          *
          * In Vulkan, we have explicit offsets but everything is crammed
          * into a single "variable" so inst->src[i].nr will always be 0.
          * Everything will be properly aligned relative to that one base.
          */
         assert(inst->src[i].offset % type_sz(inst->src[i].type) == 0);

         unsigned u = inst->src[i].nr +
                      inst->src[i].offset / UNIFORM_SLOT_SIZE;

         if (u >= uniforms)
            continue;

         unsigned slots_read;
         if (inst->opcode == SHADER_OPCODE_MOV_INDIRECT && i == 0) {
            slots_read = DIV_ROUND_UP(inst->src[2].ud, UNIFORM_SLOT_SIZE);
         } else {
            unsigned bytes_read = inst->components_read(i) *
                                  type_sz(inst->src[i].type);
            slots_read = DIV_ROUND_UP(bytes_read, UNIFORM_SLOT_SIZE);
         }

         assert(u + slots_read <= uniforms);
         mark_uniform_slots_read(&slots[u], slots_read,
                                 type_sz(inst->src[i].type));
      }
   }

   int subgroup_id_index = get_subgroup_id_param_index(stage_prog_data);

   /* Only allow 16 registers (128 uniform components) as push constants.
    *
    * Just demote the end of the list.  We could probably do better
    * here, demoting things that are rarely used in the program first.
    *
    * If changing this value, note the limitation about total_regs in
    * brw_curbe.c.
    */
   unsigned int max_push_components = 16 * 8;
   if (subgroup_id_index >= 0)
      max_push_components--; /* Save a slot for the thread ID */

   /* We push small arrays, but no bigger than 16 floats.  This is big enough
    * for a vec4 but hopefully not large enough to push out other stuff.  We
    * should probably use a better heuristic at some point.
    */
   const unsigned int max_chunk_size = 16;

   unsigned int num_push_constants = 0;
   unsigned int num_pull_constants = 0;

   push_constant_loc = ralloc_array(mem_ctx, int, uniforms);
   pull_constant_loc = ralloc_array(mem_ctx, int, uniforms);

   /* Default to -1 meaning no location */
   memset(push_constant_loc, -1, uniforms * sizeof(*push_constant_loc));
   memset(pull_constant_loc, -1, uniforms * sizeof(*pull_constant_loc));

   int chunk_start = -1;
   struct cplx_align align;
   for (unsigned u = 0; u < uniforms; u++) {
      if (!slots[u].is_live) {
         assert(chunk_start == -1);
         continue;
      }

      /* Skip subgroup_id_index to put it in the last push register. */
      if (subgroup_id_index == (int)u)
         continue;

      if (chunk_start == -1) {
         chunk_start = u;
         align = slots[u].align;
      } else {
         /* Offset into the chunk */
         unsigned chunk_offset = (u - chunk_start) * UNIFORM_SLOT_SIZE;

         /* Shift the slot alignment down by the chunk offset so it is
          * comparable with the base chunk alignment.
          */
         struct cplx_align slot_align = slots[u].align;
         slot_align.offset =
            (slot_align.offset - chunk_offset) & (align.mul - 1);

         align = cplx_align_combine(align, slot_align);
      }

      /* Sanity check the alignment */
      cplx_align_assert_sane(align);

      if (slots[u].contiguous)
         continue;

      /* Adjust the alignment to be in terms of slots, not bytes */
      assert((align.mul & (UNIFORM_SLOT_SIZE - 1)) == 0);
      assert((align.offset & (UNIFORM_SLOT_SIZE - 1)) == 0);
      align.mul /= UNIFORM_SLOT_SIZE;
      align.offset /= UNIFORM_SLOT_SIZE;

      unsigned push_start_align = cplx_align_apply(align, num_push_constants);
      unsigned chunk_size = u - chunk_start + 1;
      if ((!compiler->supports_pull_constants && u < UBO_START) ||
          (chunk_size < max_chunk_size &&
           push_start_align + chunk_size <= max_push_components)) {
         /* Align up the number of push constants */
         num_push_constants = push_start_align;
         for (unsigned i = 0; i < chunk_size; i++)
            push_constant_loc[chunk_start + i] = num_push_constants++;
      } else {
         /* We need to pull this one */
         num_pull_constants = cplx_align_apply(align, num_pull_constants);
         for (unsigned i = 0; i < chunk_size; i++)
            pull_constant_loc[chunk_start + i] = num_pull_constants++;
      }

      /* Reset the chunk and start again */
      chunk_start = -1;
   }

   /* Add the CS local thread ID uniform at the end of the push constants */
   if (subgroup_id_index >= 0)
      push_constant_loc[subgroup_id_index] = num_push_constants++;

   /* As the uniforms are going to be reordered, stash the old array and
    * create two new arrays for push/pull params.
    */
   uint32_t *param = stage_prog_data->param;
   stage_prog_data->nr_params = num_push_constants;
   if (num_push_constants) {
      stage_prog_data->param = rzalloc_array(mem_ctx, uint32_t,
                                             num_push_constants);
   } else {
      stage_prog_data->param = NULL;
   }
   assert(stage_prog_data->nr_pull_params == 0);
   assert(stage_prog_data->pull_param == NULL);
   if (num_pull_constants > 0) {
      stage_prog_data->nr_pull_params = num_pull_constants;
      stage_prog_data->pull_param = rzalloc_array(mem_ctx, uint32_t,
                                                  num_pull_constants);
   }

   /* Now that we know how many regular uniforms we'll push, reduce the
    * UBO push ranges so we don't exceed the 3DSTATE_CONSTANT limits.
    */
   unsigned push_length = DIV_ROUND_UP(stage_prog_data->nr_params, 8);
   for (int i = 0; i < 4; i++) {
      struct brw_ubo_range *range = &prog_data->ubo_ranges[i];

      if (push_length + range->length > 64)
         range->length = 64 - push_length;

      push_length += range->length;
   }
   assert(push_length <= 64);

   /* Up until now, the param[] array has been indexed by reg + offset
    * of UNIFORM registers.  Move pull constants into pull_param[] and
    * condense param[] to only contain the uniforms we chose to push.
    *
    * NOTE: Because we are condensing the params[] array, we know that
    * push_constant_loc[i] <= i and we can do it in one smooth loop without
    * having to make a copy.
    */
   for (unsigned int i = 0; i < uniforms; i++) {
      uint32_t value = param[i];
      if (pull_constant_loc[i] != -1) {
         stage_prog_data->pull_param[pull_constant_loc[i]] = value;
      } else if (push_constant_loc[i] != -1) {
         stage_prog_data->param[push_constant_loc[i]] = value;
      }
   }
   ralloc_free(param);
}

bool
fs_visitor::get_pull_locs(const fs_reg &src,
                          unsigned *out_surf_index,
                          unsigned *out_pull_index)
{
   assert(src.file == UNIFORM);

   if (src.nr >= UBO_START) {
      const struct brw_ubo_range *range =
         &prog_data->ubo_ranges[src.nr - UBO_START];

      /* If this access is in our (reduced) range, use the push data. */
      if (src.offset / 32 < range->length)
         return false;

      *out_surf_index = prog_data->binding_table.ubo_start + range->block;
      *out_pull_index = (32 * range->start + src.offset) / 4;
      return true;
   }

   const unsigned location = src.nr + src.offset / 4;

   if (location < uniforms && pull_constant_loc[location] != -1) {
      /* A regular uniform push constant */
      *out_surf_index = stage_prog_data->binding_table.pull_constants_start;
      *out_pull_index = pull_constant_loc[location];
      return true;
   }

   return false;
}

/**
 * Replace UNIFORM register file access with either UNIFORM_PULL_CONSTANT_LOAD
 * or VARYING_PULL_CONSTANT_LOAD instructions which load values into VGRFs.
 */
void
fs_visitor::lower_constant_loads()
{
   unsigned index, pull_index;

   foreach_block_and_inst_safe (block, fs_inst, inst, cfg) {
      /* Set up the annotation tracking for new generated instructions. */
      const fs_builder ibld(this, block, inst);

      for (int i = 0; i < inst->sources; i++) {
	 if (inst->src[i].file != UNIFORM)
	    continue;

         /* We'll handle this case later */
         if (inst->opcode == SHADER_OPCODE_MOV_INDIRECT && i == 0)
            continue;

         if (!get_pull_locs(inst->src[i], &index, &pull_index))
	    continue;

         assert(inst->src[i].stride == 0);

         const unsigned block_sz = 64; /* Fetch one cacheline at a time. */
         const fs_builder ubld = ibld.exec_all().group(block_sz / 4, 0);
         const fs_reg dst = ubld.vgrf(BRW_REGISTER_TYPE_UD);
         const unsigned base = pull_index * 4;

         ubld.emit(FS_OPCODE_UNIFORM_PULL_CONSTANT_LOAD,
                   dst, brw_imm_ud(index), brw_imm_ud(base & ~(block_sz - 1)));

         /* Rewrite the instruction to use the temporary VGRF. */
         inst->src[i].file = VGRF;
         inst->src[i].nr = dst.nr;
         inst->src[i].offset = (base & (block_sz - 1)) +
                               inst->src[i].offset % 4;

         brw_mark_surface_used(prog_data, index);
      }

      if (inst->opcode == SHADER_OPCODE_MOV_INDIRECT &&
          inst->src[0].file == UNIFORM) {

         if (!get_pull_locs(inst->src[0], &index, &pull_index))
            continue;

         VARYING_PULL_CONSTANT_LOAD(ibld, inst->dst,
                                    brw_imm_ud(index),
                                    inst->src[1],
                                    pull_index * 4);
         inst->remove(block);

         brw_mark_surface_used(prog_data, index);
      }
   }
   invalidate_live_intervals();
}

bool
fs_visitor::opt_algebraic()
{
   bool progress = false;

   foreach_block_and_inst(block, fs_inst, inst, cfg) {
      switch (inst->opcode) {
      case BRW_OPCODE_MOV:
         if ((inst->conditional_mod == BRW_CONDITIONAL_Z ||
              inst->conditional_mod == BRW_CONDITIONAL_NZ) &&
             inst->dst.is_null() &&
             (inst->src[0].abs || inst->src[0].negate)) {
            inst->src[0].abs = false;
            inst->src[0].negate = false;
            progress = true;
            break;
         }

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

      case BRW_OPCODE_MUL:
         if (inst->src[1].file != IMM)
            continue;

         /* a * 1.0 = a */
         if (inst->src[1].is_one()) {
            inst->opcode = BRW_OPCODE_MOV;
            inst->src[1] = reg_undef;
            progress = true;
            break;
         }

         /* a * -1.0 = -a */
         if (inst->src[1].is_negative_one()) {
            inst->opcode = BRW_OPCODE_MOV;
            inst->src[0].negate = !inst->src[0].negate;
            inst->src[1] = reg_undef;
            progress = true;
            break;
         }

         /* a * 0.0 = 0.0 */
         if (inst->src[1].is_zero()) {
            inst->opcode = BRW_OPCODE_MOV;
            inst->src[0] = inst->src[1];
            inst->src[1] = reg_undef;
            progress = true;
            break;
         }

         if (inst->src[0].file == IMM) {
            assert(inst->src[0].type == BRW_REGISTER_TYPE_F);
            inst->opcode = BRW_OPCODE_MOV;
            inst->src[0].f *= inst->src[1].f;
            inst->src[1] = reg_undef;
            progress = true;
            break;
         }
         break;
      case BRW_OPCODE_ADD:
         if (inst->src[1].file != IMM)
            continue;

         /* a + 0.0 = a */
         if (inst->src[1].is_zero()) {
            inst->opcode = BRW_OPCODE_MOV;
            inst->src[1] = reg_undef;
            progress = true;
            break;
         }

         if (inst->src[0].file == IMM) {
            assert(inst->src[0].type == BRW_REGISTER_TYPE_F);
            inst->opcode = BRW_OPCODE_MOV;
            inst->src[0].f += inst->src[1].f;
            inst->src[1] = reg_undef;
            progress = true;
            break;
         }
         break;
      case BRW_OPCODE_OR:
         if (inst->src[0].equals(inst->src[1]) ||
             inst->src[1].is_zero()) {
            inst->opcode = BRW_OPCODE_MOV;
            inst->src[1] = reg_undef;
            progress = true;
            break;
         }
         break;
      case BRW_OPCODE_LRP:
         if (inst->src[1].equals(inst->src[2])) {
            inst->opcode = BRW_OPCODE_MOV;
            inst->src[0] = inst->src[1];
            inst->src[1] = reg_undef;
            inst->src[2] = reg_undef;
            progress = true;
            break;
         }
         break;
      case BRW_OPCODE_CMP:
         if ((inst->conditional_mod == BRW_CONDITIONAL_Z ||
              inst->conditional_mod == BRW_CONDITIONAL_NZ) &&
             inst->src[1].is_zero() &&
             (inst->src[0].abs || inst->src[0].negate)) {
            inst->src[0].abs = false;
            inst->src[0].negate = false;
            progress = true;
            break;
         }
         break;
      case BRW_OPCODE_SEL:
         if (inst->src[0].equals(inst->src[1])) {
            inst->opcode = BRW_OPCODE_MOV;
            inst->src[1] = reg_undef;
            inst->predicate = BRW_PREDICATE_NONE;
            inst->predicate_inverse = false;
            progress = true;
         } else if (inst->saturate && inst->src[1].file == IMM) {
            switch (inst->conditional_mod) {
            case BRW_CONDITIONAL_LE:
            case BRW_CONDITIONAL_L:
               switch (inst->src[1].type) {
               case BRW_REGISTER_TYPE_F:
                  if (inst->src[1].f >= 1.0f) {
                     inst->opcode = BRW_OPCODE_MOV;
                     inst->src[1] = reg_undef;
                     inst->conditional_mod = BRW_CONDITIONAL_NONE;
                     progress = true;
                  }
                  break;
               default:
                  break;
               }
               break;
            case BRW_CONDITIONAL_GE:
            case BRW_CONDITIONAL_G:
               switch (inst->src[1].type) {
               case BRW_REGISTER_TYPE_F:
                  if (inst->src[1].f <= 0.0f) {
                     inst->opcode = BRW_OPCODE_MOV;
                     inst->src[1] = reg_undef;
                     inst->conditional_mod = BRW_CONDITIONAL_NONE;
                     progress = true;
                  }
                  break;
               default:
                  break;
               }
            default:
               break;
            }
         }
         break;
      case BRW_OPCODE_MAD:
         if (inst->src[1].is_zero() || inst->src[2].is_zero()) {
            inst->opcode = BRW_OPCODE_MOV;
            inst->src[1] = reg_undef;
            inst->src[2] = reg_undef;
            progress = true;
         } else if (inst->src[0].is_zero()) {
            inst->opcode = BRW_OPCODE_MUL;
            inst->src[0] = inst->src[2];
            inst->src[2] = reg_undef;
            progress = true;
         } else if (inst->src[1].is_one()) {
            inst->opcode = BRW_OPCODE_ADD;
            inst->src[1] = inst->src[2];
            inst->src[2] = reg_undef;
            progress = true;
         } else if (inst->src[2].is_one()) {
            inst->opcode = BRW_OPCODE_ADD;
            inst->src[2] = reg_undef;
            progress = true;
         } else if (inst->src[1].file == IMM && inst->src[2].file == IMM) {
            inst->opcode = BRW_OPCODE_ADD;
            inst->src[1].f *= inst->src[2].f;
            inst->src[2] = reg_undef;
            progress = true;
         }
         break;
      case SHADER_OPCODE_BROADCAST:
         if (is_uniform(inst->src[0])) {
            inst->opcode = BRW_OPCODE_MOV;
            inst->sources = 1;
            inst->force_writemask_all = true;
            progress = true;
         } else if (inst->src[1].file == IMM) {
            inst->opcode = BRW_OPCODE_MOV;
            /* It's possible that the selected component will be too large and
             * overflow the register.  This can happen if someone does a
             * readInvocation() from GLSL or SPIR-V and provides an OOB
             * invocationIndex.  If this happens and we some how manage
             * to constant fold it in and get here, then component() may cause
             * us to start reading outside of the VGRF which will lead to an
             * assert later.  Instead, just let it wrap around if it goes over
             * exec_size.
             */
            const unsigned comp = inst->src[1].ud & (inst->exec_size - 1);
            inst->src[0] = component(inst->src[0], comp);
            inst->sources = 1;
            inst->force_writemask_all = true;
            progress = true;
         }
         break;

      case SHADER_OPCODE_SHUFFLE:
         if (is_uniform(inst->src[0])) {
            inst->opcode = BRW_OPCODE_MOV;
            inst->sources = 1;
            progress = true;
         } else if (inst->src[1].file == IMM) {
            inst->opcode = BRW_OPCODE_MOV;
            inst->src[0] = component(inst->src[0],
                                     inst->src[1].ud);
            inst->sources = 1;
            progress = true;
         }
         break;

      default:
	 break;
      }

      /* Swap if src[0] is immediate. */
      if (progress && inst->is_commutative()) {
         if (inst->src[0].file == IMM) {
            fs_reg tmp = inst->src[1];
            inst->src[1] = inst->src[0];
            inst->src[0] = tmp;
         }
      }
   }
   return progress;
}

/**
 * Optimize sample messages that have constant zero values for the trailing
 * texture coordinates. We can just reduce the message length for these
 * instructions instead of reserving a register for it. Trailing parameters
 * that aren't sent default to zero anyway. This will cause the dead code
 * eliminator to remove the MOV instruction that would otherwise be emitted to
 * set up the zero value.
 */
bool
fs_visitor::opt_zero_samples()
{
   /* Gen4 infers the texturing opcode based on the message length so we can't
    * change it.
    */
   if (devinfo->gen < 5)
      return false;

   bool progress = false;

   foreach_block_and_inst(block, fs_inst, inst, cfg) {
      if (!inst->is_tex())
         continue;

      fs_inst *load_payload = (fs_inst *) inst->prev;

      if (load_payload->is_head_sentinel() ||
          load_payload->opcode != SHADER_OPCODE_LOAD_PAYLOAD)
         continue;

      /* We don't want to remove the message header or the first parameter.
       * Removing the first parameter is not allowed, see the Haswell PRM
       * volume 7, page 149:
       *
       *     "Parameter 0 is required except for the sampleinfo message, which
       *      has no parameter 0"
       */
      while (inst->mlen > inst->header_size + inst->exec_size / 8 &&
             load_payload->src[(inst->mlen - inst->header_size) /
                               (inst->exec_size / 8) +
                               inst->header_size - 1].is_zero()) {
         inst->mlen -= inst->exec_size / 8;
         progress = true;
      }
   }

   if (progress)
      invalidate_live_intervals();

   return progress;
}

/**
 * Optimize sample messages which are followed by the final RT write.
 *
 * CHV, and GEN9+ can mark a texturing SEND instruction with EOT to have its
 * results sent directly to the framebuffer, bypassing the EU.  Recognize the
 * final texturing results copied to the framebuffer write payload and modify
 * them to write to the framebuffer directly.
 */
bool
fs_visitor::opt_sampler_eot()
{
   brw_wm_prog_key *key = (brw_wm_prog_key*) this->key;

   if (stage != MESA_SHADER_FRAGMENT || dispatch_width > 16)
      return false;

   if (devinfo->gen != 9 && !devinfo->is_cherryview)
      return false;

   /* FINISHME: It should be possible to implement this optimization when there
    * are multiple drawbuffers.
    */
   if (key->nr_color_regions != 1)
      return false;

   /* Requires emitting a bunch of saturating MOV instructions during logical
    * send lowering to clamp the color payload, which the sampler unit isn't
    * going to do for us.
    */
   if (key->clamp_fragment_color)
      return false;

   /* Look for a texturing instruction immediately before the final FB_WRITE. */
   bblock_t *block = cfg->blocks[cfg->num_blocks - 1];
   fs_inst *fb_write = (fs_inst *)block->end();
   assert(fb_write->eot);
   assert(fb_write->opcode == FS_OPCODE_FB_WRITE_LOGICAL);

   /* There wasn't one; nothing to do. */
   if (unlikely(fb_write->prev->is_head_sentinel()))
      return false;

   fs_inst *tex_inst = (fs_inst *) fb_write->prev;

   /* 3D Sampler » Messages » Message Format
    *
    * “Response Length of zero is allowed on all SIMD8* and SIMD16* sampler
    *  messages except sample+killpix, resinfo, sampleinfo, LOD, and gather4*”
    */
   if (tex_inst->opcode != SHADER_OPCODE_TEX_LOGICAL &&
       tex_inst->opcode != SHADER_OPCODE_TXD_LOGICAL &&
       tex_inst->opcode != SHADER_OPCODE_TXF_LOGICAL &&
       tex_inst->opcode != SHADER_OPCODE_TXL_LOGICAL &&
       tex_inst->opcode != FS_OPCODE_TXB_LOGICAL &&
       tex_inst->opcode != SHADER_OPCODE_TXF_CMS_LOGICAL &&
       tex_inst->opcode != SHADER_OPCODE_TXF_CMS_W_LOGICAL &&
       tex_inst->opcode != SHADER_OPCODE_TXF_UMS_LOGICAL)
      return false;

   /* XXX - This shouldn't be necessary. */
   if (tex_inst->prev->is_head_sentinel())
      return false;

   /* Check that the FB write sources are fully initialized by the single
    * texturing instruction.
    */
   for (unsigned i = 0; i < FB_WRITE_LOGICAL_NUM_SRCS; i++) {
      if (i == FB_WRITE_LOGICAL_SRC_COLOR0) {
         if (!fb_write->src[i].equals(tex_inst->dst) ||
             fb_write->size_read(i) != tex_inst->size_written)
         return false;
      } else if (i != FB_WRITE_LOGICAL_SRC_COMPONENTS) {
         if (fb_write->src[i].file != BAD_FILE)
            return false;
      }
   }

   assert(!tex_inst->eot); /* We can't get here twice */
   assert((tex_inst->offset & (0xff << 24)) == 0);

   const fs_builder ibld(this, block, tex_inst);

   tex_inst->offset |= fb_write->target << 24;
   tex_inst->eot = true;
   tex_inst->dst = ibld.null_reg_ud();
   tex_inst->size_written = 0;
   fb_write->remove(cfg->blocks[cfg->num_blocks - 1]);

   /* Marking EOT is sufficient, lower_logical_sends() will notice the EOT
    * flag and submit a header together with the sampler message as required
    * by the hardware.
    */
   invalidate_live_intervals();
   return true;
}

bool
fs_visitor::opt_register_renaming()
{
   bool progress = false;
   int depth = 0;

   int remap[alloc.count];
   memset(remap, -1, sizeof(int) * alloc.count);

   foreach_block_and_inst(block, fs_inst, inst, cfg) {
      if (inst->opcode == BRW_OPCODE_IF || inst->opcode == BRW_OPCODE_DO) {
         depth++;
      } else if (inst->opcode == BRW_OPCODE_ENDIF ||
                 inst->opcode == BRW_OPCODE_WHILE) {
         depth--;
      }

      /* Rewrite instruction sources. */
      for (int i = 0; i < inst->sources; i++) {
         if (inst->src[i].file == VGRF &&
             remap[inst->src[i].nr] != -1 &&
             remap[inst->src[i].nr] != inst->src[i].nr) {
            inst->src[i].nr = remap[inst->src[i].nr];
            progress = true;
         }
      }

      const int dst = inst->dst.nr;

      if (depth == 0 &&
          inst->dst.file == VGRF &&
          alloc.sizes[inst->dst.nr] * REG_SIZE == inst->size_written &&
          !inst->is_partial_write()) {
         if (remap[dst] == -1) {
            remap[dst] = dst;
         } else {
            remap[dst] = alloc.allocate(regs_written(inst));
            inst->dst.nr = remap[dst];
            progress = true;
         }
      } else if (inst->dst.file == VGRF &&
                 remap[dst] != -1 &&
                 remap[dst] != dst) {
         inst->dst.nr = remap[dst];
         progress = true;
      }
   }

   if (progress) {
      invalidate_live_intervals();

      for (unsigned i = 0; i < ARRAY_SIZE(delta_xy); i++) {
         if (delta_xy[i].file == VGRF && remap[delta_xy[i].nr] != -1) {
            delta_xy[i].nr = remap[delta_xy[i].nr];
         }
      }
   }

   return progress;
}

/**
 * Remove redundant or useless discard jumps.
 *
 * For example, we can eliminate jumps in the following sequence:
 *
 * discard-jump       (redundant with the next jump)
 * discard-jump       (useless; jumps to the next instruction)
 * placeholder-halt
 */
bool
fs_visitor::opt_redundant_discard_jumps()
{
   bool progress = false;

   bblock_t *last_bblock = cfg->blocks[cfg->num_blocks - 1];

   fs_inst *placeholder_halt = NULL;
   foreach_inst_in_block_reverse(fs_inst, inst, last_bblock) {
      if (inst->opcode == FS_OPCODE_PLACEHOLDER_HALT) {
         placeholder_halt = inst;
         break;
      }
   }

   if (!placeholder_halt)
      return false;

   /* Delete any HALTs immediately before the placeholder halt. */
   for (fs_inst *prev = (fs_inst *) placeholder_halt->prev;
        !prev->is_head_sentinel() && prev->opcode == FS_OPCODE_DISCARD_JUMP;
        prev = (fs_inst *) placeholder_halt->prev) {
      prev->remove(last_bblock);
      progress = true;
   }

   if (progress)
      invalidate_live_intervals();

   return progress;
}

/**
 * Compute a bitmask with GRF granularity with a bit set for each GRF starting
 * from \p r.offset which overlaps the region starting at \p s.offset and
 * spanning \p ds bytes.
 */
static inline unsigned
mask_relative_to(const fs_reg &r, const fs_reg &s, unsigned ds)
{
   const int rel_offset = reg_offset(s) - reg_offset(r);
   const int shift = rel_offset / REG_SIZE;
   const unsigned n = DIV_ROUND_UP(rel_offset % REG_SIZE + ds, REG_SIZE);
   assert(reg_space(r) == reg_space(s) &&
          shift >= 0 && shift < int(8 * sizeof(unsigned)));
   return ((1 << n) - 1) << shift;
}

bool
fs_visitor::opt_peephole_csel()
{
   if (devinfo->gen < 8)
      return false;

   bool progress = false;

   foreach_block_reverse(block, cfg) {
      int ip = block->end_ip + 1;

      foreach_inst_in_block_reverse_safe(fs_inst, inst, block) {
         ip--;

         if (inst->opcode != BRW_OPCODE_SEL ||
             inst->predicate != BRW_PREDICATE_NORMAL ||
             (inst->dst.type != BRW_REGISTER_TYPE_F &&
              inst->dst.type != BRW_REGISTER_TYPE_D &&
              inst->dst.type != BRW_REGISTER_TYPE_UD))
            continue;

         /* Because it is a 3-src instruction, CSEL cannot have an immediate
          * value as a source, but we can sometimes handle zero.
          */
         if ((inst->src[0].file != VGRF && inst->src[0].file != ATTR &&
              inst->src[0].file != UNIFORM) ||
             (inst->src[1].file != VGRF && inst->src[1].file != ATTR &&
              inst->src[1].file != UNIFORM && !inst->src[1].is_zero()))
            continue;

         foreach_inst_in_block_reverse_starting_from(fs_inst, scan_inst, inst) {
            if (!scan_inst->flags_written())
               continue;

            if ((scan_inst->opcode != BRW_OPCODE_CMP &&
                 scan_inst->opcode != BRW_OPCODE_MOV) ||
                scan_inst->predicate != BRW_PREDICATE_NONE ||
                (scan_inst->src[0].file != VGRF &&
                 scan_inst->src[0].file != ATTR &&
                 scan_inst->src[0].file != UNIFORM) ||
                scan_inst->src[0].type != BRW_REGISTER_TYPE_F)
               break;

            if (scan_inst->opcode == BRW_OPCODE_CMP && !scan_inst->src[1].is_zero())
               break;

            const brw::fs_builder ibld(this, block, inst);

            const enum brw_conditional_mod cond =
               inst->predicate_inverse
               ? brw_negate_cmod(scan_inst->conditional_mod)
               : scan_inst->conditional_mod;

            fs_inst *csel_inst = NULL;

            if (inst->src[1].file != IMM) {
               csel_inst = ibld.CSEL(inst->dst,
                                     inst->src[0],
                                     inst->src[1],
                                     scan_inst->src[0],
                                     cond);
            } else if (cond == BRW_CONDITIONAL_NZ) {
               /* Consider the sequence
                *
                * cmp.nz.f0  null<1>F   g3<8,8,1>F   0F
                * (+f0) sel  g124<1>UD  g2<8,8,1>UD  0x00000000UD
                *
                * The sel will pick the immediate value 0 if r0 is ±0.0.
                * Therefore, this sequence is equivalent:
                *
                * cmp.nz.f0  null<1>F   g3<8,8,1>F   0F
                * (+f0) sel  g124<1>F   g2<8,8,1>F   (abs)g3<8,8,1>F
                *
                * The abs is ensures that the result is 0UD when g3 is -0.0F.
                * By normal cmp-sel merging, this is also equivalent:
                *
                * csel.nz    g124<1>F   g2<4,4,1>F   (abs)g3<4,4,1>F  g3<4,4,1>F
                */
               csel_inst = ibld.CSEL(inst->dst,
                                     inst->src[0],
                                     scan_inst->src[0],
                                     scan_inst->src[0],
                                     cond);

               csel_inst->src[1].abs = true;
            }

            if (csel_inst != NULL) {
               progress = true;
               inst->remove(block);
            }

            break;
         }
      }
   }

   return progress;
}

bool
fs_visitor::compute_to_mrf()
{
   bool progress = false;
   int next_ip = 0;

   /* No MRFs on Gen >= 7. */
   if (devinfo->gen >= 7)
      return false;

   calculate_live_intervals();

   foreach_block_and_inst_safe(block, fs_inst, inst, cfg) {
      int ip = next_ip;
      next_ip++;

      if (inst->opcode != BRW_OPCODE_MOV ||
	  inst->is_partial_write() ||
	  inst->dst.file != MRF || inst->src[0].file != VGRF ||
	  inst->dst.type != inst->src[0].type ||
	  inst->src[0].abs || inst->src[0].negate ||
          !inst->src[0].is_contiguous() ||
          inst->src[0].offset % REG_SIZE != 0)
	 continue;

      /* Can't compute-to-MRF this GRF if someone else was going to
       * read it later.
       */
      if (this->virtual_grf_end[inst->src[0].nr] > ip)
	 continue;

      /* Found a move of a GRF to a MRF.  Let's see if we can go rewrite the
       * things that computed the value of all GRFs of the source region.  The
       * regs_left bitset keeps track of the registers we haven't yet found a
       * generating instruction for.
       */
      unsigned regs_left = (1 << regs_read(inst, 0)) - 1;

      foreach_inst_in_block_reverse_starting_from(fs_inst, scan_inst, inst) {
         if (regions_overlap(scan_inst->dst, scan_inst->size_written,
                             inst->src[0], inst->size_read(0))) {
	    /* Found the last thing to write our reg we want to turn
	     * into a compute-to-MRF.
	     */

	    /* If this one instruction didn't populate all the
	     * channels, bail.  We might be able to rewrite everything
	     * that writes that reg, but it would require smarter
	     * tracking.
	     */
	    if (scan_inst->is_partial_write())
	       break;

            /* Handling things not fully contained in the source of the copy
             * would need us to understand coalescing out more than one MOV at
             * a time.
             */
            if (!region_contained_in(scan_inst->dst, scan_inst->size_written,
                                     inst->src[0], inst->size_read(0)))
               break;

	    /* SEND instructions can't have MRF as a destination. */
	    if (scan_inst->mlen)
	       break;

	    if (devinfo->gen == 6) {
	       /* gen6 math instructions must have the destination be
		* GRF, so no compute-to-MRF for them.
		*/
	       if (scan_inst->is_math()) {
		  break;
	       }
	    }

            /* Clear the bits for any registers this instruction overwrites. */
            regs_left &= ~mask_relative_to(
               inst->src[0], scan_inst->dst, scan_inst->size_written);
            if (!regs_left)
               break;
	 }

	 /* We don't handle control flow here.  Most computation of
	  * values that end up in MRFs are shortly before the MRF
	  * write anyway.
	  */
	 if (block->start() == scan_inst)
	    break;

	 /* You can't read from an MRF, so if someone else reads our
	  * MRF's source GRF that we wanted to rewrite, that stops us.
	  */
	 bool interfered = false;
	 for (int i = 0; i < scan_inst->sources; i++) {
            if (regions_overlap(scan_inst->src[i], scan_inst->size_read(i),
                                inst->src[0], inst->size_read(0))) {
	       interfered = true;
	    }
	 }
	 if (interfered)
	    break;

         if (regions_overlap(scan_inst->dst, scan_inst->size_written,
                             inst->dst, inst->size_written)) {
	    /* If somebody else writes our MRF here, we can't
	     * compute-to-MRF before that.
	     */
            break;
         }

         if (scan_inst->mlen > 0 && scan_inst->base_mrf != -1 &&
             regions_overlap(fs_reg(MRF, scan_inst->base_mrf), scan_inst->mlen * REG_SIZE,
                             inst->dst, inst->size_written)) {
	    /* Found a SEND instruction, which means that there are
	     * live values in MRFs from base_mrf to base_mrf +
	     * scan_inst->mlen - 1.  Don't go pushing our MRF write up
	     * above it.
	     */
            break;
         }
      }

      if (regs_left)
         continue;

      /* Found all generating instructions of our MRF's source value, so it
       * should be safe to rewrite them to point to the MRF directly.
       */
      regs_left = (1 << regs_read(inst, 0)) - 1;

      foreach_inst_in_block_reverse_starting_from(fs_inst, scan_inst, inst) {
         if (regions_overlap(scan_inst->dst, scan_inst->size_written,
                             inst->src[0], inst->size_read(0))) {
            /* Clear the bits for any registers this instruction overwrites. */
            regs_left &= ~mask_relative_to(
               inst->src[0], scan_inst->dst, scan_inst->size_written);

            const unsigned rel_offset = reg_offset(scan_inst->dst) -
                                        reg_offset(inst->src[0]);

            if (inst->dst.nr & BRW_MRF_COMPR4) {
               /* Apply the same address transformation done by the hardware
                * for COMPR4 MRF writes.
                */
               assert(rel_offset < 2 * REG_SIZE);
               scan_inst->dst.nr = inst->dst.nr + rel_offset / REG_SIZE * 4;

               /* Clear the COMPR4 bit if the generating instruction is not
                * compressed.
                */
               if (scan_inst->size_written < 2 * REG_SIZE)
                  scan_inst->dst.nr &= ~BRW_MRF_COMPR4;

            } else {
               /* Calculate the MRF number the result of this instruction is
                * ultimately written to.
                */
               scan_inst->dst.nr = inst->dst.nr + rel_offset / REG_SIZE;
            }

            scan_inst->dst.file = MRF;
            scan_inst->dst.offset = inst->dst.offset + rel_offset % REG_SIZE;
            scan_inst->saturate |= inst->saturate;
            if (!regs_left)
               break;
         }
      }

      assert(!regs_left);
      inst->remove(block);
      progress = true;
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
fs_visitor::eliminate_find_live_channel()
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

   foreach_block_and_inst_safe(block, fs_inst, inst, cfg) {
      switch (inst->opcode) {
      case BRW_OPCODE_IF:
      case BRW_OPCODE_DO:
         depth++;
         break;

      case BRW_OPCODE_ENDIF:
      case BRW_OPCODE_WHILE:
         depth--;
         break;

      case FS_OPCODE_DISCARD_JUMP:
         /* This can potentially make control flow non-uniform until the end
          * of the program.
          */
         return progress;

      case SHADER_OPCODE_FIND_LIVE_CHANNEL:
         if (depth == 0) {
            inst->opcode = BRW_OPCODE_MOV;
            inst->src[0] = brw_imm_ud(0u);
            inst->sources = 1;
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
 * Once we've generated code, try to convert normal FS_OPCODE_FB_WRITE
 * instructions to FS_OPCODE_REP_FB_WRITE.
 */
void
fs_visitor::emit_repclear_shader()
{
   brw_wm_prog_key *key = (brw_wm_prog_key*) this->key;
   int base_mrf = 0;
   int color_mrf = base_mrf + 2;
   fs_inst *mov;

   if (uniforms > 0) {
      mov = bld.exec_all().group(4, 0)
               .MOV(brw_message_reg(color_mrf),
                    fs_reg(UNIFORM, 0, BRW_REGISTER_TYPE_F));
   } else {
      struct brw_reg reg =
         brw_reg(BRW_GENERAL_REGISTER_FILE, 2, 3, 0, 0, BRW_REGISTER_TYPE_F,
                 BRW_VERTICAL_STRIDE_8, BRW_WIDTH_2, BRW_HORIZONTAL_STRIDE_4,
                 BRW_SWIZZLE_XYZW, WRITEMASK_XYZW);

      mov = bld.exec_all().group(4, 0)
               .MOV(vec4(brw_message_reg(color_mrf)), fs_reg(reg));
   }

   fs_inst *write = NULL;
   if (key->nr_color_regions == 1) {
      write = bld.emit(FS_OPCODE_REP_FB_WRITE);
      write->saturate = key->clamp_fragment_color;
      write->base_mrf = color_mrf;
      write->target = 0;
      write->header_size = 0;
      write->mlen = 1;
   } else {
      assume(key->nr_color_regions > 0);

      struct brw_reg header =
         retype(brw_message_reg(base_mrf), BRW_REGISTER_TYPE_UD);
      bld.exec_all().group(16, 0)
         .MOV(header, retype(brw_vec8_grf(0, 0), BRW_REGISTER_TYPE_UD));

      for (int i = 0; i < key->nr_color_regions; ++i) {
         if (i > 0) {
            bld.exec_all().group(1, 0)
               .MOV(component(header, 2), brw_imm_ud(i));
         }

         write = bld.emit(FS_OPCODE_REP_FB_WRITE);
         write->saturate = key->clamp_fragment_color;
         write->base_mrf = base_mrf;
         write->target = i;
         write->header_size = 2;
         write->mlen = 3;
      }
   }
   write->eot = true;
   write->last_rt = true;

   calculate_cfg();

   assign_constant_locations();
   assign_curb_setup();

   /* Now that we have the uniform assigned, go ahead and force it to a vec4. */
   if (uniforms > 0) {
      assert(mov->src[0].file == FIXED_GRF);
      mov->src[0] = brw_vec4_grf(mov->src[0].nr, 0);
   }
}

/**
 * Walks through basic blocks, looking for repeated MRF writes and
 * removing the later ones.
 */
bool
fs_visitor::remove_duplicate_mrf_writes()
{
   fs_inst *last_mrf_move[BRW_MAX_MRF(devinfo->gen)];
   bool progress = false;

   /* Need to update the MRF tracking for compressed instructions. */
   if (dispatch_width >= 16)
      return false;

   memset(last_mrf_move, 0, sizeof(last_mrf_move));

   foreach_block_and_inst_safe (block, fs_inst, inst, cfg) {
      if (inst->is_control_flow()) {
	 memset(last_mrf_move, 0, sizeof(last_mrf_move));
      }

      if (inst->opcode == BRW_OPCODE_MOV &&
	  inst->dst.file == MRF) {
         fs_inst *prev_inst = last_mrf_move[inst->dst.nr];
	 if (prev_inst && inst->equals(prev_inst)) {
	    inst->remove(block);
	    progress = true;
	    continue;
	 }
      }

      /* Clear out the last-write records for MRFs that were overwritten. */
      if (inst->dst.file == MRF) {
         last_mrf_move[inst->dst.nr] = NULL;
      }

      if (inst->mlen > 0 && inst->base_mrf != -1) {
	 /* Found a SEND instruction, which will include two or fewer
	  * implied MRF writes.  We could do better here.
	  */
	 for (int i = 0; i < implied_mrf_writes(inst); i++) {
	    last_mrf_move[inst->base_mrf + i] = NULL;
	 }
      }

      /* Clear out any MRF move records whose sources got overwritten. */
      for (unsigned i = 0; i < ARRAY_SIZE(last_mrf_move); i++) {
         if (last_mrf_move[i] &&
             regions_overlap(inst->dst, inst->size_written,
                             last_mrf_move[i]->src[0],
                             last_mrf_move[i]->size_read(0))) {
            last_mrf_move[i] = NULL;
         }
      }

      if (inst->opcode == BRW_OPCODE_MOV &&
	  inst->dst.file == MRF &&
	  inst->src[0].file != ARF &&
	  !inst->is_partial_write()) {
         last_mrf_move[inst->dst.nr] = inst;
      }
   }

   if (progress)
      invalidate_live_intervals();

   return progress;
}

/**
 * Rounding modes for conversion instructions are included for each
 * conversion, but right now it is a state. So once it is set,
 * we don't need to call it again for subsequent calls.
 *
 * This is useful for vector/matrices conversions, as setting the
 * mode once is enough for the full vector/matrix
 */
bool
fs_visitor::remove_extra_rounding_modes()
{
   bool progress = false;

   foreach_block (block, cfg) {
      brw_rnd_mode prev_mode = BRW_RND_MODE_UNSPECIFIED;

      foreach_inst_in_block_safe (fs_inst, inst, block) {
         if (inst->opcode == SHADER_OPCODE_RND_MODE) {
            assert(inst->src[0].file == BRW_IMMEDIATE_VALUE);
            const brw_rnd_mode mode = (brw_rnd_mode) inst->src[0].d;
            if (mode == prev_mode) {
               inst->remove(block);
               progress = true;
            } else {
               prev_mode = mode;
            }
         }
      }
   }

   if (progress)
      invalidate_live_intervals();

   return progress;
}

static void
clear_deps_for_inst_src(fs_inst *inst, bool *deps, int first_grf, int grf_len)
{
   /* Clear the flag for registers that actually got read (as expected). */
   for (int i = 0; i < inst->sources; i++) {
      int grf;
      if (inst->src[i].file == VGRF || inst->src[i].file == FIXED_GRF) {
         grf = inst->src[i].nr;
      } else {
         continue;
      }

      if (grf >= first_grf &&
          grf < first_grf + grf_len) {
         deps[grf - first_grf] = false;
         if (inst->exec_size == 16)
            deps[grf - first_grf + 1] = false;
      }
   }
}

/**
 * Implements this workaround for the original 965:
 *
 *     "[DevBW, DevCL] Implementation Restrictions: As the hardware does not
 *      check for post destination dependencies on this instruction, software
 *      must ensure that there is no destination hazard for the case of ‘write
 *      followed by a posted write’ shown in the following example.
 *
 *      1. mov r3 0
 *      2. send r3.xy <rest of send instruction>
 *      3. mov r2 r3
 *
 *      Due to no post-destination dependency check on the ‘send’, the above
 *      code sequence could have two instructions (1 and 2) in flight at the
 *      same time that both consider ‘r3’ as the target of their final writes.
 */
void
fs_visitor::insert_gen4_pre_send_dependency_workarounds(bblock_t *block,
                                                        fs_inst *inst)
{
   int write_len = regs_written(inst);
   int first_write_grf = inst->dst.nr;
   bool needs_dep[BRW_MAX_MRF(devinfo->gen)];
   assert(write_len < (int)sizeof(needs_dep) - 1);

   memset(needs_dep, false, sizeof(needs_dep));
   memset(needs_dep, true, write_len);

   clear_deps_for_inst_src(inst, needs_dep, first_write_grf, write_len);

   /* Walk backwards looking for writes to registers we're writing which
    * aren't read since being written.  If we hit the start of the program,
    * we assume that there are no outstanding dependencies on entry to the
    * program.
    */
   foreach_inst_in_block_reverse_starting_from(fs_inst, scan_inst, inst) {
      /* If we hit control flow, assume that there *are* outstanding
       * dependencies, and force their cleanup before our instruction.
       */
      if (block->start() == scan_inst && block->num != 0) {
         for (int i = 0; i < write_len; i++) {
            if (needs_dep[i])
               DEP_RESOLVE_MOV(fs_builder(this, block, inst),
                               first_write_grf + i);
         }
         return;
      }

      /* We insert our reads as late as possible on the assumption that any
       * instruction but a MOV that might have left us an outstanding
       * dependency has more latency than a MOV.
       */
      if (scan_inst->dst.file == VGRF) {
         for (unsigned i = 0; i < regs_written(scan_inst); i++) {
            int reg = scan_inst->dst.nr + i;

            if (reg >= first_write_grf &&
                reg < first_write_grf + write_len &&
                needs_dep[reg - first_write_grf]) {
               DEP_RESOLVE_MOV(fs_builder(this, block, inst), reg);
               needs_dep[reg - first_write_grf] = false;
               if (scan_inst->exec_size == 16)
                  needs_dep[reg - first_write_grf + 1] = false;
            }
         }
      }

      /* Clear the flag for registers that actually got read (as expected). */
      clear_deps_for_inst_src(scan_inst, needs_dep, first_write_grf, write_len);

      /* Continue the loop only if we haven't resolved all the dependencies */
      int i;
      for (i = 0; i < write_len; i++) {
         if (needs_dep[i])
            break;
      }
      if (i == write_len)
         return;
   }
}

/**
 * Implements this workaround for the original 965:
 *
 *     "[DevBW, DevCL] Errata: A destination register from a send can not be
 *      used as a destination register until after it has been sourced by an
 *      instruction with a different destination register.
 */
void
fs_visitor::insert_gen4_post_send_dependency_workarounds(bblock_t *block, fs_inst *inst)
{
   int write_len = regs_written(inst);
   int first_write_grf = inst->dst.nr;
   bool needs_dep[BRW_MAX_MRF(devinfo->gen)];
   assert(write_len < (int)sizeof(needs_dep) - 1);

   memset(needs_dep, false, sizeof(needs_dep));
   memset(needs_dep, true, write_len);
   /* Walk forwards looking for writes to registers we're writing which aren't
    * read before being written.
    */
   foreach_inst_in_block_starting_from(fs_inst, scan_inst, inst) {
      /* If we hit control flow, force resolve all remaining dependencies. */
      if (block->end() == scan_inst && block->num != cfg->num_blocks - 1) {
         for (int i = 0; i < write_len; i++) {
            if (needs_dep[i])
               DEP_RESOLVE_MOV(fs_builder(this, block, scan_inst),
                               first_write_grf + i);
         }
         return;
      }

      /* Clear the flag for registers that actually got read (as expected). */
      clear_deps_for_inst_src(scan_inst, needs_dep, first_write_grf, write_len);

      /* We insert our reads as late as possible since they're reading the
       * result of a SEND, which has massive latency.
       */
      if (scan_inst->dst.file == VGRF &&
          scan_inst->dst.nr >= first_write_grf &&
          scan_inst->dst.nr < first_write_grf + write_len &&
          needs_dep[scan_inst->dst.nr - first_write_grf]) {
         DEP_RESOLVE_MOV(fs_builder(this, block, scan_inst),
                         scan_inst->dst.nr);
         needs_dep[scan_inst->dst.nr - first_write_grf] = false;
      }

      /* Continue the loop only if we haven't resolved all the dependencies */
      int i;
      for (i = 0; i < write_len; i++) {
         if (needs_dep[i])
            break;
      }
      if (i == write_len)
         return;
   }
}

void
fs_visitor::insert_gen4_send_dependency_workarounds()
{
   if (devinfo->gen != 4 || devinfo->is_g4x)
      return;

   bool progress = false;

   foreach_block_and_inst(block, fs_inst, inst, cfg) {
      if (inst->mlen != 0 && inst->dst.file == VGRF) {
         insert_gen4_pre_send_dependency_workarounds(block, inst);
         insert_gen4_post_send_dependency_workarounds(block, inst);
         progress = true;
      }
   }

   if (progress)
      invalidate_live_intervals();
}

/**
 * Turns the generic expression-style uniform pull constant load instruction
 * into a hardware-specific series of instructions for loading a pull
 * constant.
 *
 * The expression style allows the CSE pass before this to optimize out
 * repeated loads from the same offset, and gives the pre-register-allocation
 * scheduling full flexibility, while the conversion to native instructions
 * allows the post-register-allocation scheduler the best information
 * possible.
 *
 * Note that execution masking for setting up pull constant loads is special:
 * the channels that need to be written are unrelated to the current execution
 * mask, since a later instruction will use one of the result channels as a
 * source operand for all 8 or 16 of its channels.
 */
void
fs_visitor::lower_uniform_pull_constant_loads()
{
   foreach_block_and_inst (block, fs_inst, inst, cfg) {
      if (inst->opcode != FS_OPCODE_UNIFORM_PULL_CONSTANT_LOAD)
         continue;

      if (devinfo->gen >= 7) {
         const fs_builder ubld = fs_builder(this, block, inst).exec_all();
         const fs_reg payload = ubld.group(8, 0).vgrf(BRW_REGISTER_TYPE_UD);

         ubld.group(8, 0).MOV(payload,
                              retype(brw_vec8_grf(0, 0), BRW_REGISTER_TYPE_UD));
         ubld.group(1, 0).MOV(component(payload, 2),
                              brw_imm_ud(inst->src[1].ud / 16));

         inst->opcode = FS_OPCODE_UNIFORM_PULL_CONSTANT_LOAD_GEN7;
         inst->src[1] = payload;
         inst->header_size = 1;
         inst->mlen = 1;

         invalidate_live_intervals();
      } else {
         /* Before register allocation, we didn't tell the scheduler about the
          * MRF we use.  We know it's safe to use this MRF because nothing
          * else does except for register spill/unspill, which generates and
          * uses its MRF within a single IR instruction.
          */
         inst->base_mrf = FIRST_PULL_LOAD_MRF(devinfo->gen) + 1;
         inst->mlen = 1;
      }
   }
}

bool
fs_visitor::lower_load_payload()
{
   bool progress = false;

   foreach_block_and_inst_safe (block, fs_inst, inst, cfg) {
      if (inst->opcode != SHADER_OPCODE_LOAD_PAYLOAD)
         continue;

      assert(inst->dst.file == MRF || inst->dst.file == VGRF);
      assert(inst->saturate == false);
      fs_reg dst = inst->dst;

      /* Get rid of COMPR4.  We'll add it back in if we need it */
      if (dst.file == MRF)
         dst.nr = dst.nr & ~BRW_MRF_COMPR4;

      const fs_builder ibld(this, block, inst);
      const fs_builder hbld = ibld.exec_all().group(8, 0);

      for (uint8_t i = 0; i < inst->header_size; i++) {
         if (inst->src[i].file != BAD_FILE) {
            fs_reg mov_dst = retype(dst, BRW_REGISTER_TYPE_UD);
            fs_reg mov_src = retype(inst->src[i], BRW_REGISTER_TYPE_UD);
            hbld.MOV(mov_dst, mov_src);
         }
         dst = offset(dst, hbld, 1);
      }

      if (inst->dst.file == MRF && (inst->dst.nr & BRW_MRF_COMPR4) &&
          inst->exec_size > 8) {
         /* In this case, the payload portion of the LOAD_PAYLOAD isn't
          * a straightforward copy.  Instead, the result of the
          * LOAD_PAYLOAD is treated as interleaved and the first four
          * non-header sources are unpacked as:
          *
          * m + 0: r0
          * m + 1: g0
          * m + 2: b0
          * m + 3: a0
          * m + 4: r1
          * m + 5: g1
          * m + 6: b1
          * m + 7: a1
          *
          * This is used for gen <= 5 fb writes.
          */
         assert(inst->exec_size == 16);
         assert(inst->header_size + 4 <= inst->sources);
         for (uint8_t i = inst->header_size; i < inst->header_size + 4; i++) {
            if (inst->src[i].file != BAD_FILE) {
               if (devinfo->has_compr4) {
                  fs_reg compr4_dst = retype(dst, inst->src[i].type);
                  compr4_dst.nr |= BRW_MRF_COMPR4;
                  ibld.MOV(compr4_dst, inst->src[i]);
               } else {
                  /* Platform doesn't have COMPR4.  We have to fake it */
                  fs_reg mov_dst = retype(dst, inst->src[i].type);
                  ibld.half(0).MOV(mov_dst, half(inst->src[i], 0));
                  mov_dst.nr += 4;
                  ibld.half(1).MOV(mov_dst, half(inst->src[i], 1));
               }
            }

            dst.nr++;
         }

         /* The loop above only ever incremented us through the first set
          * of 4 registers.  However, thanks to the magic of COMPR4, we
          * actually wrote to the first 8 registers, so we need to take
          * that into account now.
          */
         dst.nr += 4;

         /* The COMPR4 code took care of the first 4 sources.  We'll let
          * the regular path handle any remaining sources.  Yes, we are
          * modifying the instruction but we're about to delete it so
          * this really doesn't hurt anything.
          */
         inst->header_size += 4;
      }

      for (uint8_t i = inst->header_size; i < inst->sources; i++) {
         if (inst->src[i].file != BAD_FILE)
            ibld.MOV(retype(dst, inst->src[i].type), inst->src[i]);
         dst = offset(dst, ibld, 1);
      }

      inst->remove(block);
      progress = true;
   }

   if (progress)
      invalidate_live_intervals();

   return progress;
}

bool
fs_visitor::lower_integer_multiplication()
{
   bool progress = false;

   foreach_block_and_inst_safe(block, fs_inst, inst, cfg) {
      const fs_builder ibld(this, block, inst);

      if (inst->opcode == BRW_OPCODE_MUL) {
         if (inst->dst.is_accumulator() ||
             (inst->dst.type != BRW_REGISTER_TYPE_D &&
              inst->dst.type != BRW_REGISTER_TYPE_UD))
            continue;

         if (devinfo->has_integer_dword_mul)
            continue;

         if (inst->src[1].file == IMM &&
             inst->src[1].ud < (1 << 16)) {
            /* The MUL instruction isn't commutative. On Gen <= 6, only the low
             * 16-bits of src0 are read, and on Gen >= 7 only the low 16-bits of
             * src1 are used.
             *
             * If multiplying by an immediate value that fits in 16-bits, do a
             * single MUL instruction with that value in the proper location.
             */
            if (devinfo->gen < 7) {
               fs_reg imm(VGRF, alloc.allocate(dispatch_width / 8),
                          inst->dst.type);
               ibld.MOV(imm, inst->src[1]);
               ibld.MUL(inst->dst, imm, inst->src[0]);
            } else {
               const bool ud = (inst->src[1].type == BRW_REGISTER_TYPE_UD);
               ibld.MUL(inst->dst, inst->src[0],
                        ud ? brw_imm_uw(inst->src[1].ud)
                           : brw_imm_w(inst->src[1].d));
            }
         } else {
            /* Gen < 8 (and some Gen8+ low-power parts like Cherryview) cannot
             * do 32-bit integer multiplication in one instruction, but instead
             * must do a sequence (which actually calculates a 64-bit result):
             *
             *    mul(8)  acc0<1>D   g3<8,8,1>D      g4<8,8,1>D
             *    mach(8) null       g3<8,8,1>D      g4<8,8,1>D
             *    mov(8)  g2<1>D     acc0<8,8,1>D
             *
             * But on Gen > 6, the ability to use second accumulator register
             * (acc1) for non-float data types was removed, preventing a simple
             * implementation in SIMD16. A 16-channel result can be calculated by
             * executing the three instructions twice in SIMD8, once with quarter
             * control of 1Q for the first eight channels and again with 2Q for
             * the second eight channels.
             *
             * Which accumulator register is implicitly accessed (by AccWrEnable
             * for instance) is determined by the quarter control. Unfortunately
             * Ivybridge (and presumably Baytrail) has a hardware bug in which an
             * implicit accumulator access by an instruction with 2Q will access
             * acc1 regardless of whether the data type is usable in acc1.
             *
             * Specifically, the 2Q mach(8) writes acc1 which does not exist for
             * integer data types.
             *
             * Since we only want the low 32-bits of the result, we can do two
             * 32-bit x 16-bit multiplies (like the mul and mach are doing), and
             * adjust the high result and add them (like the mach is doing):
             *
             *    mul(8)  g7<1>D     g3<8,8,1>D      g4.0<8,8,1>UW
             *    mul(8)  g8<1>D     g3<8,8,1>D      g4.1<8,8,1>UW
             *    shl(8)  g9<1>D     g8<8,8,1>D      16D
             *    add(8)  g2<1>D     g7<8,8,1>D      g8<8,8,1>D
             *
             * We avoid the shl instruction by realizing that we only want to add
             * the low 16-bits of the "high" result to the high 16-bits of the
             * "low" result and using proper regioning on the add:
             *
             *    mul(8)  g7<1>D     g3<8,8,1>D      g4.0<16,8,2>UW
             *    mul(8)  g8<1>D     g3<8,8,1>D      g4.1<16,8,2>UW
             *    add(8)  g7.1<2>UW  g7.1<16,8,2>UW  g8<16,8,2>UW
             *
             * Since it does not use the (single) accumulator register, we can
             * schedule multi-component multiplications much better.
             */

            bool needs_mov = false;
            fs_reg orig_dst = inst->dst;
            fs_reg low = inst->dst;
            if (orig_dst.is_null() || orig_dst.file == MRF ||
                regions_overlap(inst->dst, inst->size_written,
                                inst->src[0], inst->size_read(0)) ||
                regions_overlap(inst->dst, inst->size_written,
                                inst->src[1], inst->size_read(1))) {
               needs_mov = true;
               /* Get a new VGRF but keep the same stride as inst->dst */
               low = fs_reg(VGRF, alloc.allocate(regs_written(inst)),
                            inst->dst.type);
               low.stride = inst->dst.stride;
               low.offset = inst->dst.offset % REG_SIZE;
            }

            /* Get a new VGRF but keep the same stride as inst->dst */
            fs_reg high(VGRF, alloc.allocate(regs_written(inst)),
                        inst->dst.type);
            high.stride = inst->dst.stride;
            high.offset = inst->dst.offset % REG_SIZE;

            if (devinfo->gen >= 7) {
               if (inst->src[1].file == IMM) {
                  ibld.MUL(low, inst->src[0],
                           brw_imm_uw(inst->src[1].ud & 0xffff));
                  ibld.MUL(high, inst->src[0],
                           brw_imm_uw(inst->src[1].ud >> 16));
               } else {
                  ibld.MUL(low, inst->src[0],
                           subscript(inst->src[1], BRW_REGISTER_TYPE_UW, 0));
                  ibld.MUL(high, inst->src[0],
                           subscript(inst->src[1], BRW_REGISTER_TYPE_UW, 1));
               }
            } else {
               ibld.MUL(low, subscript(inst->src[0], BRW_REGISTER_TYPE_UW, 0),
                        inst->src[1]);
               ibld.MUL(high, subscript(inst->src[0], BRW_REGISTER_TYPE_UW, 1),
                        inst->src[1]);
            }

            ibld.ADD(subscript(low, BRW_REGISTER_TYPE_UW, 1),
                     subscript(low, BRW_REGISTER_TYPE_UW, 1),
                     subscript(high, BRW_REGISTER_TYPE_UW, 0));

            if (needs_mov || inst->conditional_mod) {
               set_condmod(inst->conditional_mod,
                           ibld.MOV(orig_dst, low));
            }
         }

      } else if (inst->opcode == SHADER_OPCODE_MULH) {
         /* Should have been lowered to 8-wide. */
         assert(inst->exec_size <= get_lowered_simd_width(devinfo, inst));
         const fs_reg acc = retype(brw_acc_reg(inst->exec_size),
                                   inst->dst.type);
         fs_inst *mul = ibld.MUL(acc, inst->src[0], inst->src[1]);
         fs_inst *mach = ibld.MACH(inst->dst, inst->src[0], inst->src[1]);

         if (devinfo->gen >= 8) {
            /* Until Gen8, integer multiplies read 32-bits from one source,
             * and 16-bits from the other, and relying on the MACH instruction
             * to generate the high bits of the result.
             *
             * On Gen8, the multiply instruction does a full 32x32-bit
             * multiply, but in order to do a 64-bit multiply we can simulate
             * the previous behavior and then use a MACH instruction.
             *
             * FINISHME: Don't use source modifiers on src1.
             */
            assert(mul->src[1].type == BRW_REGISTER_TYPE_D ||
                   mul->src[1].type == BRW_REGISTER_TYPE_UD);
            mul->src[1].type = BRW_REGISTER_TYPE_UW;
            mul->src[1].stride *= 2;

         } else if (devinfo->gen == 7 && !devinfo->is_haswell &&
                    inst->group > 0) {
            /* Among other things the quarter control bits influence which
             * accumulator register is used by the hardware for instructions
             * that access the accumulator implicitly (e.g. MACH).  A
             * second-half instruction would normally map to acc1, which
             * doesn't exist on Gen7 and up (the hardware does emulate it for
             * floating-point instructions *only* by taking advantage of the
             * extra precision of acc0 not normally used for floating point
             * arithmetic).
             *
             * HSW and up are careful enough not to try to access an
             * accumulator register that doesn't exist, but on earlier Gen7
             * hardware we need to make sure that the quarter control bits are
             * zero to avoid non-deterministic behaviour and emit an extra MOV
             * to get the result masked correctly according to the current
             * channel enables.
             */
            mach->group = 0;
            mach->force_writemask_all = true;
            mach->dst = ibld.vgrf(inst->dst.type);
            ibld.MOV(inst->dst, mach->dst);
         }
      } else {
         continue;
      }

      inst->remove(block);
      progress = true;
   }

   if (progress)
      invalidate_live_intervals();

   return progress;
}

bool
fs_visitor::lower_minmax()
{
   assert(devinfo->gen < 6);

   bool progress = false;

   foreach_block_and_inst_safe(block, fs_inst, inst, cfg) {
      const fs_builder ibld(this, block, inst);

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

static void
setup_color_payload(const fs_builder &bld, const brw_wm_prog_key *key,
                    fs_reg *dst, fs_reg color, unsigned components)
{
   if (key->clamp_fragment_color) {
      fs_reg tmp = bld.vgrf(BRW_REGISTER_TYPE_F, 4);
      assert(color.type == BRW_REGISTER_TYPE_F);

      for (unsigned i = 0; i < components; i++)
         set_saturate(true,
                      bld.MOV(offset(tmp, bld, i), offset(color, bld, i)));

      color = tmp;
   }

   for (unsigned i = 0; i < components; i++)
      dst[i] = offset(color, bld, i);
}

static void
lower_fb_write_logical_send(const fs_builder &bld, fs_inst *inst,
                            const struct brw_wm_prog_data *prog_data,
                            const brw_wm_prog_key *key,
                            const fs_visitor::thread_payload &payload)
{
   assert(inst->src[FB_WRITE_LOGICAL_SRC_COMPONENTS].file == IMM);
   const gen_device_info *devinfo = bld.shader->devinfo;
   const fs_reg &color0 = inst->src[FB_WRITE_LOGICAL_SRC_COLOR0];
   const fs_reg &color1 = inst->src[FB_WRITE_LOGICAL_SRC_COLOR1];
   const fs_reg &src0_alpha = inst->src[FB_WRITE_LOGICAL_SRC_SRC0_ALPHA];
   const fs_reg &src_depth = inst->src[FB_WRITE_LOGICAL_SRC_SRC_DEPTH];
   const fs_reg &dst_depth = inst->src[FB_WRITE_LOGICAL_SRC_DST_DEPTH];
   const fs_reg &src_stencil = inst->src[FB_WRITE_LOGICAL_SRC_SRC_STENCIL];
   fs_reg sample_mask = inst->src[FB_WRITE_LOGICAL_SRC_OMASK];
   const unsigned components =
      inst->src[FB_WRITE_LOGICAL_SRC_COMPONENTS].ud;

   /* We can potentially have a message length of up to 15, so we have to set
    * base_mrf to either 0 or 1 in order to fit in m0..m15.
    */
   fs_reg sources[15];
   int header_size = 2, payload_header_size;
   unsigned length = 0;

   if (devinfo->gen < 6) {
      /* TODO: Support SIMD32 on gen4-5 */
      assert(bld.group() < 16);

      /* For gen4-5, we always have a header consisting of g0 and g1.  We have
       * an implied MOV from g0,g1 to the start of the message.  The MOV from
       * g0 is handled by the hardware and the MOV from g1 is provided by the
       * generator.  This is required because, on gen4-5, the generator may
       * generate two write messages with different message lengths in order
       * to handle AA data properly.
       *
       * Also, since the pixel mask goes in the g0 portion of the message and
       * since render target writes are the last thing in the shader, we write
       * the pixel mask directly into g0 and it will get copied as part of the
       * implied write.
       */
      if (prog_data->uses_kill) {
         bld.exec_all().group(1, 0)
            .MOV(retype(brw_vec1_grf(0, 0), BRW_REGISTER_TYPE_UW),
                 brw_flag_reg(0, 1));
      }

      assert(length == 0);
      length = 2;
   } else if ((devinfo->gen <= 7 && !devinfo->is_haswell &&
               prog_data->uses_kill) ||
              color1.file != BAD_FILE ||
              key->nr_color_regions > 1) {
      /* From the Sandy Bridge PRM, volume 4, page 198:
       *
       *     "Dispatched Pixel Enables. One bit per pixel indicating
       *      which pixels were originally enabled when the thread was
       *      dispatched. This field is only required for the end-of-
       *      thread message and on all dual-source messages."
       */
      const fs_builder ubld = bld.exec_all().group(8, 0);

      fs_reg header = ubld.vgrf(BRW_REGISTER_TYPE_UD, 2);
      if (bld.group() < 16) {
         /* The header starts off as g0 and g1 for the first half */
         ubld.group(16, 0).MOV(header, retype(brw_vec8_grf(0, 0),
                                              BRW_REGISTER_TYPE_UD));
      } else {
         /* The header starts off as g0 and g2 for the second half */
         assert(bld.group() < 32);
         const fs_reg header_sources[2] = {
            retype(brw_vec8_grf(0, 0), BRW_REGISTER_TYPE_UD),
            retype(brw_vec8_grf(2, 0), BRW_REGISTER_TYPE_UD),
         };
         ubld.LOAD_PAYLOAD(header, header_sources, 2, 0);
      }

      uint32_t g00_bits = 0;

      /* Set "Source0 Alpha Present to RenderTarget" bit in message
       * header.
       */
      if (inst->target > 0 && key->replicate_alpha)
         g00_bits |= 1 << 11;

      /* Set computes stencil to render target */
      if (prog_data->computed_stencil)
         g00_bits |= 1 << 14;

      if (g00_bits) {
         /* OR extra bits into g0.0 */
         ubld.group(1, 0).OR(component(header, 0),
                             retype(brw_vec1_grf(0, 0),
                                    BRW_REGISTER_TYPE_UD),
                             brw_imm_ud(g00_bits));
      }

      /* Set the render target index for choosing BLEND_STATE. */
      if (inst->target > 0) {
         ubld.group(1, 0).MOV(component(header, 2), brw_imm_ud(inst->target));
      }

      if (prog_data->uses_kill) {
         assert(bld.group() < 16);
         ubld.group(1, 0).MOV(retype(component(header, 15),
                                     BRW_REGISTER_TYPE_UW),
                              brw_flag_reg(0, 1));
      }

      assert(length == 0);
      sources[0] = header;
      sources[1] = horiz_offset(header, 8);
      length = 2;
   }
   assert(length == 0 || length == 2);
   header_size = length;

   if (payload.aa_dest_stencil_reg[0]) {
      assert(inst->group < 16);
      sources[length] = fs_reg(VGRF, bld.shader->alloc.allocate(1));
      bld.group(8, 0).exec_all().annotate("FB write stencil/AA alpha")
         .MOV(sources[length],
              fs_reg(brw_vec8_grf(payload.aa_dest_stencil_reg[0], 0)));
      length++;
   }

   if (sample_mask.file != BAD_FILE) {
      sources[length] = fs_reg(VGRF, bld.shader->alloc.allocate(1),
                               BRW_REGISTER_TYPE_UD);

      /* Hand over gl_SampleMask.  Only the lower 16 bits of each channel are
       * relevant.  Since it's unsigned single words one vgrf is always
       * 16-wide, but only the lower or higher 8 channels will be used by the
       * hardware when doing a SIMD8 write depending on whether we have
       * selected the subspans for the first or second half respectively.
       */
      assert(sample_mask.file != BAD_FILE && type_sz(sample_mask.type) == 4);
      sample_mask.type = BRW_REGISTER_TYPE_UW;
      sample_mask.stride *= 2;

      bld.exec_all().annotate("FB write oMask")
         .MOV(horiz_offset(retype(sources[length], BRW_REGISTER_TYPE_UW),
                           inst->group % 16),
              sample_mask);
      length++;
   }

   payload_header_size = length;

   if (src0_alpha.file != BAD_FILE) {
      /* FIXME: This is being passed at the wrong location in the payload and
       * doesn't work when gl_SampleMask and MRTs are used simultaneously.
       * It's supposed to be immediately before oMask but there seems to be no
       * reasonable way to pass them in the correct order because LOAD_PAYLOAD
       * requires header sources to form a contiguous segment at the beginning
       * of the message and src0_alpha has per-channel semantics.
       */
      setup_color_payload(bld, key, &sources[length], src0_alpha, 1);
      length++;
   } else if (key->replicate_alpha && inst->target != 0) {
      /* Handle the case when fragment shader doesn't write to draw buffer
       * zero. No need to call setup_color_payload() for src0_alpha because
       * alpha value will be undefined.
       */
      length++;
   }

   setup_color_payload(bld, key, &sources[length], color0, components);
   length += 4;

   if (color1.file != BAD_FILE) {
      setup_color_payload(bld, key, &sources[length], color1, components);
      length += 4;
   }

   if (src_depth.file != BAD_FILE) {
      sources[length] = src_depth;
      length++;
   }

   if (dst_depth.file != BAD_FILE) {
      sources[length] = dst_depth;
      length++;
   }

   if (src_stencil.file != BAD_FILE) {
      assert(devinfo->gen >= 9);
      assert(bld.dispatch_width() == 8);

      /* XXX: src_stencil is only available on gen9+. dst_depth is never
       * available on gen9+. As such it's impossible to have both enabled at the
       * same time and therefore length cannot overrun the array.
       */
      assert(length < 15);

      sources[length] = bld.vgrf(BRW_REGISTER_TYPE_UD);
      bld.exec_all().annotate("FB write OS")
         .MOV(retype(sources[length], BRW_REGISTER_TYPE_UB),
              subscript(src_stencil, BRW_REGISTER_TYPE_UB, 0));
      length++;
   }

   fs_inst *load;
   if (devinfo->gen >= 7) {
      /* Send from the GRF */
      fs_reg payload = fs_reg(VGRF, -1, BRW_REGISTER_TYPE_F);
      load = bld.LOAD_PAYLOAD(payload, sources, length, payload_header_size);
      payload.nr = bld.shader->alloc.allocate(regs_written(load));
      load->dst = payload;

      inst->src[0] = payload;
      inst->resize_sources(1);
   } else {
      /* Send from the MRF */
      load = bld.LOAD_PAYLOAD(fs_reg(MRF, 1, BRW_REGISTER_TYPE_F),
                              sources, length, payload_header_size);

      /* On pre-SNB, we have to interlace the color values.  LOAD_PAYLOAD
       * will do this for us if we just give it a COMPR4 destination.
       */
      if (devinfo->gen < 6 && bld.dispatch_width() == 16)
         load->dst.nr |= BRW_MRF_COMPR4;

      if (devinfo->gen < 6) {
         /* Set up src[0] for the implied MOV from grf0-1 */
         inst->resize_sources(1);
         inst->src[0] = brw_vec8_grf(0, 0);
      } else {
         inst->resize_sources(0);
      }
      inst->base_mrf = 1;
   }

   inst->opcode = FS_OPCODE_FB_WRITE;
   inst->mlen = regs_written(load);
   inst->header_size = header_size;
}

static void
lower_fb_read_logical_send(const fs_builder &bld, fs_inst *inst)
{
   const fs_builder &ubld = bld.exec_all().group(8, 0);
   const unsigned length = 2;
   const fs_reg header = ubld.vgrf(BRW_REGISTER_TYPE_UD, length);

   if (bld.group() < 16) {
      ubld.group(16, 0).MOV(header, retype(brw_vec8_grf(0, 0),
                                           BRW_REGISTER_TYPE_UD));
   } else {
      assert(bld.group() < 32);
      const fs_reg header_sources[] = {
         retype(brw_vec8_grf(0, 0), BRW_REGISTER_TYPE_UD),
         retype(brw_vec8_grf(2, 0), BRW_REGISTER_TYPE_UD)
      };
      ubld.LOAD_PAYLOAD(header, header_sources, ARRAY_SIZE(header_sources), 0);
   }

   inst->resize_sources(1);
   inst->src[0] = header;
   inst->opcode = FS_OPCODE_FB_READ;
   inst->mlen = length;
   inst->header_size = length;
}

static void
lower_sampler_logical_send_gen4(const fs_builder &bld, fs_inst *inst, opcode op,
                                const fs_reg &coordinate,
                                const fs_reg &shadow_c,
                                const fs_reg &lod, const fs_reg &lod2,
                                const fs_reg &surface,
                                const fs_reg &sampler,
                                unsigned coord_components,
                                unsigned grad_components)
{
   const bool has_lod = (op == SHADER_OPCODE_TXL || op == FS_OPCODE_TXB ||
                         op == SHADER_OPCODE_TXF || op == SHADER_OPCODE_TXS);
   fs_reg msg_begin(MRF, 1, BRW_REGISTER_TYPE_F);
   fs_reg msg_end = msg_begin;

   /* g0 header. */
   msg_end = offset(msg_end, bld.group(8, 0), 1);

   for (unsigned i = 0; i < coord_components; i++)
      bld.MOV(retype(offset(msg_end, bld, i), coordinate.type),
              offset(coordinate, bld, i));

   msg_end = offset(msg_end, bld, coord_components);

   /* Messages other than SAMPLE and RESINFO in SIMD16 and TXD in SIMD8
    * require all three components to be present and zero if they are unused.
    */
   if (coord_components > 0 &&
       (has_lod || shadow_c.file != BAD_FILE ||
        (op == SHADER_OPCODE_TEX && bld.dispatch_width() == 8))) {
      for (unsigned i = coord_components; i < 3; i++)
         bld.MOV(offset(msg_end, bld, i), brw_imm_f(0.0f));

      msg_end = offset(msg_end, bld, 3 - coord_components);
   }

   if (op == SHADER_OPCODE_TXD) {
      /* TXD unsupported in SIMD16 mode. */
      assert(bld.dispatch_width() == 8);

      /* the slots for u and v are always present, but r is optional */
      if (coord_components < 2)
         msg_end = offset(msg_end, bld, 2 - coord_components);

      /*  P   = u, v, r
       * dPdx = dudx, dvdx, drdx
       * dPdy = dudy, dvdy, drdy
       *
       * 1-arg: Does not exist.
       *
       * 2-arg: dudx   dvdx   dudy   dvdy
       *        dPdx.x dPdx.y dPdy.x dPdy.y
       *        m4     m5     m6     m7
       *
       * 3-arg: dudx   dvdx   drdx   dudy   dvdy   drdy
       *        dPdx.x dPdx.y dPdx.z dPdy.x dPdy.y dPdy.z
       *        m5     m6     m7     m8     m9     m10
       */
      for (unsigned i = 0; i < grad_components; i++)
         bld.MOV(offset(msg_end, bld, i), offset(lod, bld, i));

      msg_end = offset(msg_end, bld, MAX2(grad_components, 2));

      for (unsigned i = 0; i < grad_components; i++)
         bld.MOV(offset(msg_end, bld, i), offset(lod2, bld, i));

      msg_end = offset(msg_end, bld, MAX2(grad_components, 2));
   }

   if (has_lod) {
      /* Bias/LOD with shadow comparator is unsupported in SIMD16 -- *Without*
       * shadow comparator (including RESINFO) it's unsupported in SIMD8 mode.
       */
      assert(shadow_c.file != BAD_FILE ? bld.dispatch_width() == 8 :
             bld.dispatch_width() == 16);

      const brw_reg_type type =
         (op == SHADER_OPCODE_TXF || op == SHADER_OPCODE_TXS ?
          BRW_REGISTER_TYPE_UD : BRW_REGISTER_TYPE_F);
      bld.MOV(retype(msg_end, type), lod);
      msg_end = offset(msg_end, bld, 1);
   }

   if (shadow_c.file != BAD_FILE) {
      if (op == SHADER_OPCODE_TEX && bld.dispatch_width() == 8) {
         /* There's no plain shadow compare message, so we use shadow
          * compare with a bias of 0.0.
          */
         bld.MOV(msg_end, brw_imm_f(0.0f));
         msg_end = offset(msg_end, bld, 1);
      }

      bld.MOV(msg_end, shadow_c);
      msg_end = offset(msg_end, bld, 1);
   }

   inst->opcode = op;
   inst->src[0] = reg_undef;
   inst->src[1] = surface;
   inst->src[2] = sampler;
   inst->resize_sources(3);
   inst->base_mrf = msg_begin.nr;
   inst->mlen = msg_end.nr - msg_begin.nr;
   inst->header_size = 1;
}

static void
lower_sampler_logical_send_gen5(const fs_builder &bld, fs_inst *inst, opcode op,
                                const fs_reg &coordinate,
                                const fs_reg &shadow_c,
                                const fs_reg &lod, const fs_reg &lod2,
                                const fs_reg &sample_index,
                                const fs_reg &surface,
                                const fs_reg &sampler,
                                unsigned coord_components,
                                unsigned grad_components)
{
   fs_reg message(MRF, 2, BRW_REGISTER_TYPE_F);
   fs_reg msg_coords = message;
   unsigned header_size = 0;

   if (inst->offset != 0) {
      /* The offsets set up by the visitor are in the m1 header, so we can't
       * go headerless.
       */
      header_size = 1;
      message.nr--;
   }

   for (unsigned i = 0; i < coord_components; i++)
      bld.MOV(retype(offset(msg_coords, bld, i), coordinate.type),
              offset(coordinate, bld, i));

   fs_reg msg_end = offset(msg_coords, bld, coord_components);
   fs_reg msg_lod = offset(msg_coords, bld, 4);

   if (shadow_c.file != BAD_FILE) {
      fs_reg msg_shadow = msg_lod;
      bld.MOV(msg_shadow, shadow_c);
      msg_lod = offset(msg_shadow, bld, 1);
      msg_end = msg_lod;
   }

   switch (op) {
   case SHADER_OPCODE_TXL:
   case FS_OPCODE_TXB:
      bld.MOV(msg_lod, lod);
      msg_end = offset(msg_lod, bld, 1);
      break;
   case SHADER_OPCODE_TXD:
      /**
       *  P   =  u,    v,    r
       * dPdx = dudx, dvdx, drdx
       * dPdy = dudy, dvdy, drdy
       *
       * Load up these values:
       * - dudx   dudy   dvdx   dvdy   drdx   drdy
       * - dPdx.x dPdy.x dPdx.y dPdy.y dPdx.z dPdy.z
       */
      msg_end = msg_lod;
      for (unsigned i = 0; i < grad_components; i++) {
         bld.MOV(msg_end, offset(lod, bld, i));
         msg_end = offset(msg_end, bld, 1);

         bld.MOV(msg_end, offset(lod2, bld, i));
         msg_end = offset(msg_end, bld, 1);
      }
      break;
   case SHADER_OPCODE_TXS:
      msg_lod = retype(msg_end, BRW_REGISTER_TYPE_UD);
      bld.MOV(msg_lod, lod);
      msg_end = offset(msg_lod, bld, 1);
      break;
   case SHADER_OPCODE_TXF:
      msg_lod = offset(msg_coords, bld, 3);
      bld.MOV(retype(msg_lod, BRW_REGISTER_TYPE_UD), lod);
      msg_end = offset(msg_lod, bld, 1);
      break;
   case SHADER_OPCODE_TXF_CMS:
      msg_lod = offset(msg_coords, bld, 3);
      /* lod */
      bld.MOV(retype(msg_lod, BRW_REGISTER_TYPE_UD), brw_imm_ud(0u));
      /* sample index */
      bld.MOV(retype(offset(msg_lod, bld, 1), BRW_REGISTER_TYPE_UD), sample_index);
      msg_end = offset(msg_lod, bld, 2);
      break;
   default:
      break;
   }

   inst->opcode = op;
   inst->src[0] = reg_undef;
   inst->src[1] = surface;
   inst->src[2] = sampler;
   inst->resize_sources(3);
   inst->base_mrf = message.nr;
   inst->mlen = msg_end.nr - message.nr;
   inst->header_size = header_size;

   /* Message length > MAX_SAMPLER_MESSAGE_SIZE disallowed by hardware. */
   assert(inst->mlen <= MAX_SAMPLER_MESSAGE_SIZE);
}

static bool
is_high_sampler(const struct gen_device_info *devinfo, const fs_reg &sampler)
{
   if (devinfo->gen < 8 && !devinfo->is_haswell)
      return false;

   return sampler.file != IMM || sampler.ud >= 16;
}

static void
lower_sampler_logical_send_gen7(const fs_builder &bld, fs_inst *inst, opcode op,
                                const fs_reg &coordinate,
                                const fs_reg &shadow_c,
                                fs_reg lod, const fs_reg &lod2,
                                const fs_reg &sample_index,
                                const fs_reg &mcs,
                                const fs_reg &surface,
                                const fs_reg &sampler,
                                const fs_reg &tg4_offset,
                                unsigned coord_components,
                                unsigned grad_components)
{
   const gen_device_info *devinfo = bld.shader->devinfo;
   unsigned reg_width = bld.dispatch_width() / 8;
   unsigned header_size = 0, length = 0;
   fs_reg sources[MAX_SAMPLER_MESSAGE_SIZE];
   for (unsigned i = 0; i < ARRAY_SIZE(sources); i++)
      sources[i] = bld.vgrf(BRW_REGISTER_TYPE_F);

   if (op == SHADER_OPCODE_TG4 || op == SHADER_OPCODE_TG4_OFFSET ||
       inst->offset != 0 || inst->eot ||
       op == SHADER_OPCODE_SAMPLEINFO ||
       is_high_sampler(devinfo, sampler)) {
      /* For general texture offsets (no txf workaround), we need a header to
       * put them in.
       *
       * TG4 needs to place its channel select in the header, for interaction
       * with ARB_texture_swizzle.  The sampler index is only 4-bits, so for
       * larger sampler numbers we need to offset the Sampler State Pointer in
       * the header.
       */
      fs_reg header = retype(sources[0], BRW_REGISTER_TYPE_UD);
      header_size = 1;
      length++;

      /* If we're requesting fewer than four channels worth of response,
       * and we have an explicit header, we need to set up the sampler
       * writemask.  It's reversed from normal: 1 means "don't write".
       */
      if (!inst->eot && regs_written(inst) != 4 * reg_width) {
         assert(regs_written(inst) % reg_width == 0);
         unsigned mask = ~((1 << (regs_written(inst) / reg_width)) - 1) & 0xf;
         inst->offset |= mask << 12;
      }

      /* Build the actual header */
      const fs_builder ubld = bld.exec_all().group(8, 0);
      const fs_builder ubld1 = ubld.group(1, 0);
      ubld.MOV(header, retype(brw_vec8_grf(0, 0), BRW_REGISTER_TYPE_UD));
      if (inst->offset) {
         ubld1.MOV(component(header, 2), brw_imm_ud(inst->offset));
      } else if (bld.shader->stage != MESA_SHADER_VERTEX &&
                 bld.shader->stage != MESA_SHADER_FRAGMENT) {
         /* The vertex and fragment stages have g0.2 set to 0, so
          * header0.2 is 0 when g0 is copied. Other stages may not, so we
          * must set it to 0 to avoid setting undesirable bits in the
          * message.
          */
         ubld1.MOV(component(header, 2), brw_imm_ud(0));
      }

      if (is_high_sampler(devinfo, sampler)) {
         if (sampler.file == BRW_IMMEDIATE_VALUE) {
            assert(sampler.ud >= 16);
            const int sampler_state_size = 16; /* 16 bytes */

            ubld1.ADD(component(header, 3),
                      retype(brw_vec1_grf(0, 3), BRW_REGISTER_TYPE_UD),
                      brw_imm_ud(16 * (sampler.ud / 16) * sampler_state_size));
         } else {
            fs_reg tmp = ubld1.vgrf(BRW_REGISTER_TYPE_UD);
            ubld1.AND(tmp, sampler, brw_imm_ud(0x0f0));
            ubld1.SHL(tmp, tmp, brw_imm_ud(4));
            ubld1.ADD(component(header, 3),
                      retype(brw_vec1_grf(0, 3), BRW_REGISTER_TYPE_UD),
                      tmp);
         }
      }
   }

   if (shadow_c.file != BAD_FILE) {
      bld.MOV(sources[length], shadow_c);
      length++;
   }

   bool coordinate_done = false;

   /* Set up the LOD info */
   switch (op) {
   case FS_OPCODE_TXB:
   case SHADER_OPCODE_TXL:
      if (devinfo->gen >= 9 && op == SHADER_OPCODE_TXL && lod.is_zero()) {
         op = SHADER_OPCODE_TXL_LZ;
         break;
      }
      bld.MOV(sources[length], lod);
      length++;
      break;
   case SHADER_OPCODE_TXD:
      /* TXD should have been lowered in SIMD16 mode. */
      assert(bld.dispatch_width() == 8);

      /* Load dPdx and the coordinate together:
       * [hdr], [ref], x, dPdx.x, dPdy.x, y, dPdx.y, dPdy.y, z, dPdx.z, dPdy.z
       */
      for (unsigned i = 0; i < coord_components; i++) {
         bld.MOV(sources[length++], offset(coordinate, bld, i));

         /* For cube map array, the coordinate is (u,v,r,ai) but there are
          * only derivatives for (u, v, r).
          */
         if (i < grad_components) {
            bld.MOV(sources[length++], offset(lod, bld, i));
            bld.MOV(sources[length++], offset(lod2, bld, i));
         }
      }

      coordinate_done = true;
      break;
   case SHADER_OPCODE_TXS:
      bld.MOV(retype(sources[length], BRW_REGISTER_TYPE_UD), lod);
      length++;
      break;
   case SHADER_OPCODE_TXF:
      /* Unfortunately, the parameters for LD are intermixed: u, lod, v, r.
       * On Gen9 they are u, v, lod, r
       */
      bld.MOV(retype(sources[length++], BRW_REGISTER_TYPE_D), coordinate);

      if (devinfo->gen >= 9) {
         if (coord_components >= 2) {
            bld.MOV(retype(sources[length], BRW_REGISTER_TYPE_D),
                    offset(coordinate, bld, 1));
         } else {
            sources[length] = brw_imm_d(0);
         }
         length++;
      }

      if (devinfo->gen >= 9 && lod.is_zero()) {
         op = SHADER_OPCODE_TXF_LZ;
      } else {
         bld.MOV(retype(sources[length], BRW_REGISTER_TYPE_D), lod);
         length++;
      }

      for (unsigned i = devinfo->gen >= 9 ? 2 : 1; i < coord_components; i++)
         bld.MOV(retype(sources[length++], BRW_REGISTER_TYPE_D),
                 offset(coordinate, bld, i));

      coordinate_done = true;
      break;

   case SHADER_OPCODE_TXF_CMS:
   case SHADER_OPCODE_TXF_CMS_W:
   case SHADER_OPCODE_TXF_UMS:
   case SHADER_OPCODE_TXF_MCS:
      if (op == SHADER_OPCODE_TXF_UMS ||
          op == SHADER_OPCODE_TXF_CMS ||
          op == SHADER_OPCODE_TXF_CMS_W) {
         bld.MOV(retype(sources[length], BRW_REGISTER_TYPE_UD), sample_index);
         length++;
      }

      if (op == SHADER_OPCODE_TXF_CMS || op == SHADER_OPCODE_TXF_CMS_W) {
         /* Data from the multisample control surface. */
         bld.MOV(retype(sources[length], BRW_REGISTER_TYPE_UD), mcs);
         length++;

         /* On Gen9+ we'll use ld2dms_w instead which has two registers for
          * the MCS data.
          */
         if (op == SHADER_OPCODE_TXF_CMS_W) {
            bld.MOV(retype(sources[length], BRW_REGISTER_TYPE_UD),
                    mcs.file == IMM ?
                    mcs :
                    offset(mcs, bld, 1));
            length++;
         }
      }

      /* There is no offsetting for this message; just copy in the integer
       * texture coordinates.
       */
      for (unsigned i = 0; i < coord_components; i++)
         bld.MOV(retype(sources[length++], BRW_REGISTER_TYPE_D),
                 offset(coordinate, bld, i));

      coordinate_done = true;
      break;
   case SHADER_OPCODE_TG4_OFFSET:
      /* More crazy intermixing */
      for (unsigned i = 0; i < 2; i++) /* u, v */
         bld.MOV(sources[length++], offset(coordinate, bld, i));

      for (unsigned i = 0; i < 2; i++) /* offu, offv */
         bld.MOV(retype(sources[length++], BRW_REGISTER_TYPE_D),
                 offset(tg4_offset, bld, i));

      if (coord_components == 3) /* r if present */
         bld.MOV(sources[length++], offset(coordinate, bld, 2));

      coordinate_done = true;
      break;
   default:
      break;
   }

   /* Set up the coordinate (except for cases where it was done above) */
   if (!coordinate_done) {
      for (unsigned i = 0; i < coord_components; i++)
         bld.MOV(sources[length++], offset(coordinate, bld, i));
   }

   int mlen;
   if (reg_width == 2)
      mlen = length * reg_width - header_size;
   else
      mlen = length * reg_width;

   const fs_reg src_payload = fs_reg(VGRF, bld.shader->alloc.allocate(mlen),
                                     BRW_REGISTER_TYPE_F);
   bld.LOAD_PAYLOAD(src_payload, sources, length, header_size);

   /* Generate the SEND. */
   inst->opcode = op;
   inst->src[0] = src_payload;
   inst->src[1] = surface;
   inst->src[2] = sampler;
   inst->resize_sources(3);
   inst->mlen = mlen;
   inst->header_size = header_size;

   /* Message length > MAX_SAMPLER_MESSAGE_SIZE disallowed by hardware. */
   assert(inst->mlen <= MAX_SAMPLER_MESSAGE_SIZE);
}

static void
lower_sampler_logical_send(const fs_builder &bld, fs_inst *inst, opcode op)
{
   const gen_device_info *devinfo = bld.shader->devinfo;
   const fs_reg &coordinate = inst->src[TEX_LOGICAL_SRC_COORDINATE];
   const fs_reg &shadow_c = inst->src[TEX_LOGICAL_SRC_SHADOW_C];
   const fs_reg &lod = inst->src[TEX_LOGICAL_SRC_LOD];
   const fs_reg &lod2 = inst->src[TEX_LOGICAL_SRC_LOD2];
   const fs_reg &sample_index = inst->src[TEX_LOGICAL_SRC_SAMPLE_INDEX];
   const fs_reg &mcs = inst->src[TEX_LOGICAL_SRC_MCS];
   const fs_reg &surface = inst->src[TEX_LOGICAL_SRC_SURFACE];
   const fs_reg &sampler = inst->src[TEX_LOGICAL_SRC_SAMPLER];
   const fs_reg &tg4_offset = inst->src[TEX_LOGICAL_SRC_TG4_OFFSET];
   assert(inst->src[TEX_LOGICAL_SRC_COORD_COMPONENTS].file == IMM);
   const unsigned coord_components = inst->src[TEX_LOGICAL_SRC_COORD_COMPONENTS].ud;
   assert(inst->src[TEX_LOGICAL_SRC_GRAD_COMPONENTS].file == IMM);
   const unsigned grad_components = inst->src[TEX_LOGICAL_SRC_GRAD_COMPONENTS].ud;

   if (devinfo->gen >= 7) {
      lower_sampler_logical_send_gen7(bld, inst, op, coordinate,
                                      shadow_c, lod, lod2, sample_index,
                                      mcs, surface, sampler, tg4_offset,
                                      coord_components, grad_components);
   } else if (devinfo->gen >= 5) {
      lower_sampler_logical_send_gen5(bld, inst, op, coordinate,
                                      shadow_c, lod, lod2, sample_index,
                                      surface, sampler,
                                      coord_components, grad_components);
   } else {
      lower_sampler_logical_send_gen4(bld, inst, op, coordinate,
                                      shadow_c, lod, lod2,
                                      surface, sampler,
                                      coord_components, grad_components);
   }
}

/**
 * Initialize the header present in some typed and untyped surface
 * messages.
 */
static fs_reg
emit_surface_header(const fs_builder &bld, const fs_reg &sample_mask)
{
   fs_builder ubld = bld.exec_all().group(8, 0);
   const fs_reg dst = ubld.vgrf(BRW_REGISTER_TYPE_UD);
   ubld.MOV(dst, brw_imm_d(0));
   ubld.group(1, 0).MOV(component(dst, 7), sample_mask);
   return dst;
}

static void
lower_surface_logical_send(const fs_builder &bld, fs_inst *inst, opcode op,
                           const fs_reg &sample_mask)
{
   const gen_device_info *devinfo = bld.shader->devinfo;

   /* Get the logical send arguments. */
   const fs_reg &addr = inst->src[0];
   const fs_reg &src = inst->src[1];
   const fs_reg &surface = inst->src[2];
   const UNUSED fs_reg &dims = inst->src[3];
   const fs_reg &arg = inst->src[4];

   /* Calculate the total number of components of the payload. */
   const unsigned addr_sz = inst->components_read(0);
   const unsigned src_sz = inst->components_read(1);
   /* From the BDW PRM Volume 7, page 147:
    *
    *  "For the Data Cache Data Port*, the header must be present for the
    *   following message types: [...] Typed read/write/atomics"
    *
    * Earlier generations have a similar wording.  Because of this restriction
    * we don't attempt to implement sample masks via predication for such
    * messages prior to Gen9, since we have to provide a header anyway.  On
    * Gen11+ the header has been removed so we can only use predication.
    */
   const unsigned header_sz = devinfo->gen < 9 &&
                              (op == SHADER_OPCODE_TYPED_SURFACE_READ ||
                               op == SHADER_OPCODE_TYPED_SURFACE_WRITE ||
                               op == SHADER_OPCODE_TYPED_ATOMIC) ? 1 : 0;
   const unsigned sz = header_sz + addr_sz + src_sz;

   /* Allocate space for the payload. */
   fs_reg *const components = new fs_reg[sz];
   const fs_reg payload = bld.vgrf(BRW_REGISTER_TYPE_UD, sz);
   unsigned n = 0;

   /* Construct the payload. */
   if (header_sz)
      components[n++] = emit_surface_header(bld, sample_mask);

   for (unsigned i = 0; i < addr_sz; i++)
      components[n++] = offset(addr, bld, i);

   for (unsigned i = 0; i < src_sz; i++)
      components[n++] = offset(src, bld, i);

   bld.LOAD_PAYLOAD(payload, components, sz, header_sz);

   /* Predicate the instruction on the sample mask if no header is
    * provided.
    */
   if (!header_sz && sample_mask.file != BAD_FILE &&
       sample_mask.file != IMM) {
      const fs_builder ubld = bld.group(1, 0).exec_all();
      if (inst->predicate) {
         assert(inst->predicate == BRW_PREDICATE_NORMAL);
         assert(!inst->predicate_inverse);
         assert(inst->flag_subreg < 2);
         /* Combine the sample mask with the existing predicate by using a
          * vertical predication mode.
          */
         inst->predicate = BRW_PREDICATE_ALIGN1_ALLV;
         ubld.MOV(retype(brw_flag_subreg(inst->flag_subreg + 2),
                         sample_mask.type),
                  sample_mask);
      } else {
         inst->flag_subreg = 2;
         inst->predicate = BRW_PREDICATE_NORMAL;
         inst->predicate_inverse = false;
         ubld.MOV(retype(brw_flag_subreg(inst->flag_subreg), sample_mask.type),
                  sample_mask);
      }
   }

   /* Update the original instruction. */
   inst->opcode = op;
   inst->mlen = header_sz + (addr_sz + src_sz) * inst->exec_size / 8;
   inst->header_size = header_sz;

   inst->src[0] = payload;
   inst->src[1] = surface;
   inst->src[2] = arg;
   inst->resize_sources(3);

   delete[] components;
}

static void
lower_varying_pull_constant_logical_send(const fs_builder &bld, fs_inst *inst)
{
   const gen_device_info *devinfo = bld.shader->devinfo;

   if (devinfo->gen >= 7) {
      /* We are switching the instruction from an ALU-like instruction to a
       * send-from-grf instruction.  Since sends can't handle strides or
       * source modifiers, we have to make a copy of the offset source.
       */
      fs_reg tmp = bld.vgrf(BRW_REGISTER_TYPE_UD);
      bld.MOV(tmp, inst->src[1]);
      inst->src[1] = tmp;

      inst->opcode = FS_OPCODE_VARYING_PULL_CONSTANT_LOAD_GEN7;
      inst->mlen = inst->exec_size / 8;
   } else {
      const fs_reg payload(MRF, FIRST_PULL_LOAD_MRF(devinfo->gen),
                           BRW_REGISTER_TYPE_UD);

      bld.MOV(byte_offset(payload, REG_SIZE), inst->src[1]);

      inst->opcode = FS_OPCODE_VARYING_PULL_CONSTANT_LOAD_GEN4;
      inst->resize_sources(1);
      inst->base_mrf = payload.nr;
      inst->header_size = 1;
      inst->mlen = 1 + inst->exec_size / 8;
   }
}

static void
lower_math_logical_send(const fs_builder &bld, fs_inst *inst)
{
   assert(bld.shader->devinfo->gen < 6);

   inst->base_mrf = 2;
   inst->mlen = inst->sources * inst->exec_size / 8;

   if (inst->sources > 1) {
      /* From the Ironlake PRM, Volume 4, Part 1, Section 6.1.13
       * "Message Payload":
       *
       * "Operand0[7].  For the INT DIV functions, this operand is the
       *  denominator."
       *  ...
       * "Operand1[7].  For the INT DIV functions, this operand is the
       *  numerator."
       */
      const bool is_int_div = inst->opcode != SHADER_OPCODE_POW;
      const fs_reg src0 = is_int_div ? inst->src[1] : inst->src[0];
      const fs_reg src1 = is_int_div ? inst->src[0] : inst->src[1];

      inst->resize_sources(1);
      inst->src[0] = src0;

      assert(inst->exec_size == 8);
      bld.MOV(fs_reg(MRF, inst->base_mrf + 1, src1.type), src1);
   }
}

bool
fs_visitor::lower_logical_sends()
{
   bool progress = false;

   foreach_block_and_inst_safe(block, fs_inst, inst, cfg) {
      const fs_builder ibld(this, block, inst);

      switch (inst->opcode) {
      case FS_OPCODE_FB_WRITE_LOGICAL:
         assert(stage == MESA_SHADER_FRAGMENT);
         lower_fb_write_logical_send(ibld, inst,
                                     brw_wm_prog_data(prog_data),
                                     (const brw_wm_prog_key *)key,
                                     payload);
         break;

      case FS_OPCODE_FB_READ_LOGICAL:
         lower_fb_read_logical_send(ibld, inst);
         break;

      case SHADER_OPCODE_TEX_LOGICAL:
         lower_sampler_logical_send(ibld, inst, SHADER_OPCODE_TEX);
         break;

      case SHADER_OPCODE_TXD_LOGICAL:
         lower_sampler_logical_send(ibld, inst, SHADER_OPCODE_TXD);
         break;

      case SHADER_OPCODE_TXF_LOGICAL:
         lower_sampler_logical_send(ibld, inst, SHADER_OPCODE_TXF);
         break;

      case SHADER_OPCODE_TXL_LOGICAL:
         lower_sampler_logical_send(ibld, inst, SHADER_OPCODE_TXL);
         break;

      case SHADER_OPCODE_TXS_LOGICAL:
         lower_sampler_logical_send(ibld, inst, SHADER_OPCODE_TXS);
         break;

      case FS_OPCODE_TXB_LOGICAL:
         lower_sampler_logical_send(ibld, inst, FS_OPCODE_TXB);
         break;

      case SHADER_OPCODE_TXF_CMS_LOGICAL:
         lower_sampler_logical_send(ibld, inst, SHADER_OPCODE_TXF_CMS);
         break;

      case SHADER_OPCODE_TXF_CMS_W_LOGICAL:
         lower_sampler_logical_send(ibld, inst, SHADER_OPCODE_TXF_CMS_W);
         break;

      case SHADER_OPCODE_TXF_UMS_LOGICAL:
         lower_sampler_logical_send(ibld, inst, SHADER_OPCODE_TXF_UMS);
         break;

      case SHADER_OPCODE_TXF_MCS_LOGICAL:
         lower_sampler_logical_send(ibld, inst, SHADER_OPCODE_TXF_MCS);
         break;

      case SHADER_OPCODE_LOD_LOGICAL:
         lower_sampler_logical_send(ibld, inst, SHADER_OPCODE_LOD);
         break;

      case SHADER_OPCODE_TG4_LOGICAL:
         lower_sampler_logical_send(ibld, inst, SHADER_OPCODE_TG4);
         break;

      case SHADER_OPCODE_TG4_OFFSET_LOGICAL:
         lower_sampler_logical_send(ibld, inst, SHADER_OPCODE_TG4_OFFSET);
         break;

      case SHADER_OPCODE_SAMPLEINFO_LOGICAL:
         lower_sampler_logical_send(ibld, inst, SHADER_OPCODE_SAMPLEINFO);
         break;

      case SHADER_OPCODE_UNTYPED_SURFACE_READ_LOGICAL:
         lower_surface_logical_send(ibld, inst,
                                    SHADER_OPCODE_UNTYPED_SURFACE_READ,
                                    fs_reg());
         break;

      case SHADER_OPCODE_UNTYPED_SURFACE_WRITE_LOGICAL:
         lower_surface_logical_send(ibld, inst,
                                    SHADER_OPCODE_UNTYPED_SURFACE_WRITE,
                                    ibld.sample_mask_reg());
         break;

      case SHADER_OPCODE_BYTE_SCATTERED_READ_LOGICAL:
         lower_surface_logical_send(ibld, inst,
                                    SHADER_OPCODE_BYTE_SCATTERED_READ,
                                    fs_reg());
         break;

      case SHADER_OPCODE_BYTE_SCATTERED_WRITE_LOGICAL:
         lower_surface_logical_send(ibld, inst,
                                    SHADER_OPCODE_BYTE_SCATTERED_WRITE,
                                    ibld.sample_mask_reg());
         break;

      case SHADER_OPCODE_UNTYPED_ATOMIC_LOGICAL:
         lower_surface_logical_send(ibld, inst,
                                    SHADER_OPCODE_UNTYPED_ATOMIC,
                                    ibld.sample_mask_reg());
         break;

      case SHADER_OPCODE_UNTYPED_ATOMIC_FLOAT_LOGICAL:
         lower_surface_logical_send(ibld, inst,
                                    SHADER_OPCODE_UNTYPED_ATOMIC_FLOAT,
                                    ibld.sample_mask_reg());
         break;

      case SHADER_OPCODE_TYPED_SURFACE_READ_LOGICAL:
         lower_surface_logical_send(ibld, inst,
                                    SHADER_OPCODE_TYPED_SURFACE_READ,
                                    brw_imm_d(0xffff));
         break;

      case SHADER_OPCODE_TYPED_SURFACE_WRITE_LOGICAL:
         lower_surface_logical_send(ibld, inst,
                                    SHADER_OPCODE_TYPED_SURFACE_WRITE,
                                    ibld.sample_mask_reg());
         break;

      case SHADER_OPCODE_TYPED_ATOMIC_LOGICAL:
         lower_surface_logical_send(ibld, inst,
                                    SHADER_OPCODE_TYPED_ATOMIC,
                                    ibld.sample_mask_reg());
         break;

      case FS_OPCODE_VARYING_PULL_CONSTANT_LOAD_LOGICAL:
         lower_varying_pull_constant_logical_send(ibld, inst);
         break;

      case SHADER_OPCODE_RCP:
      case SHADER_OPCODE_RSQ:
      case SHADER_OPCODE_SQRT:
      case SHADER_OPCODE_EXP2:
      case SHADER_OPCODE_LOG2:
      case SHADER_OPCODE_SIN:
      case SHADER_OPCODE_COS:
      case SHADER_OPCODE_POW:
      case SHADER_OPCODE_INT_QUOTIENT:
      case SHADER_OPCODE_INT_REMAINDER:
         /* The math opcodes are overloaded for the send-like and
          * expression-like instructions which seems kind of icky.  Gen6+ has
          * a native (but rather quirky) MATH instruction so we don't need to
          * do anything here.  On Gen4-5 we'll have to lower the Gen6-like
          * logical instructions (which we can easily recognize because they
          * have mlen = 0) into send-like virtual instructions.
          */
         if (devinfo->gen < 6 && inst->mlen == 0) {
            lower_math_logical_send(ibld, inst);
            break;

         } else {
            continue;
         }

      default:
         continue;
      }

      progress = true;
   }

   if (progress)
      invalidate_live_intervals();

   return progress;
}

/**
 * Get the closest allowed SIMD width for instruction \p inst accounting for
 * some common regioning and execution control restrictions that apply to FPU
 * instructions.  These restrictions don't necessarily have any relevance to
 * instructions not executed by the FPU pipeline like extended math, control
 * flow or send message instructions.
 *
 * For virtual opcodes it's really up to the instruction -- In some cases
 * (e.g. where a virtual instruction unrolls into a simple sequence of FPU
 * instructions) it may simplify virtual instruction lowering if we can
 * enforce FPU-like regioning restrictions already on the virtual instruction,
 * in other cases (e.g. virtual send-like instructions) this may be
 * excessively restrictive.
 */
static unsigned
get_fpu_lowered_simd_width(const struct gen_device_info *devinfo,
                           const fs_inst *inst)
{
   /* Maximum execution size representable in the instruction controls. */
   unsigned max_width = MIN2(32, inst->exec_size);

   /* According to the PRMs:
    *  "A. In Direct Addressing mode, a source cannot span more than 2
    *      adjacent GRF registers.
    *   B. A destination cannot span more than 2 adjacent GRF registers."
    *
    * Look for the source or destination with the largest register region
    * which is the one that is going to limit the overall execution size of
    * the instruction due to this rule.
    */
   unsigned reg_count = DIV_ROUND_UP(inst->size_written, REG_SIZE);

   for (unsigned i = 0; i < inst->sources; i++)
      reg_count = MAX2(reg_count, DIV_ROUND_UP(inst->size_read(i), REG_SIZE));

   /* Calculate the maximum execution size of the instruction based on the
    * factor by which it goes over the hardware limit of 2 GRFs.
    */
   if (reg_count > 2)
      max_width = MIN2(max_width, inst->exec_size / DIV_ROUND_UP(reg_count, 2));

   /* According to the IVB PRMs:
    *  "When destination spans two registers, the source MUST span two
    *   registers. The exception to the above rule:
    *
    *    - When source is scalar, the source registers are not incremented.
    *    - When source is packed integer Word and destination is packed
    *      integer DWord, the source register is not incremented but the
    *      source sub register is incremented."
    *
    * The hardware specs from Gen4 to Gen7.5 mention similar regioning
    * restrictions.  The code below intentionally doesn't check whether the
    * destination type is integer because empirically the hardware doesn't
    * seem to care what the actual type is as long as it's dword-aligned.
    */
   if (devinfo->gen < 8) {
      for (unsigned i = 0; i < inst->sources; i++) {
         /* IVB implements DF scalars as <0;2,1> regions. */
         const bool is_scalar_exception = is_uniform(inst->src[i]) &&
            (devinfo->is_haswell || type_sz(inst->src[i].type) != 8);
         const bool is_packed_word_exception =
            type_sz(inst->dst.type) == 4 && inst->dst.stride == 1 &&
            type_sz(inst->src[i].type) == 2 && inst->src[i].stride == 1;

         /* We check size_read(i) against size_written instead of REG_SIZE
          * because we want to properly handle SIMD32.  In SIMD32, you can end
          * up with writes to 4 registers and a source that reads 2 registers
          * and we may still need to lower all the way to SIMD8 in that case.
          */
         if (inst->size_written > REG_SIZE &&
             inst->size_read(i) != 0 &&
             inst->size_read(i) < inst->size_written &&
             !is_scalar_exception && !is_packed_word_exception) {
            const unsigned reg_count = DIV_ROUND_UP(inst->size_written, REG_SIZE);
            max_width = MIN2(max_width, inst->exec_size / reg_count);
         }
      }
   }

   if (devinfo->gen < 6) {
      /* From the G45 PRM, Volume 4 Page 361:
       *
       *    "Operand Alignment Rule: With the exceptions listed below, a
       *     source/destination operand in general should be aligned to even
       *     256-bit physical register with a region size equal to two 256-bit
       *     physical registers."
       *
       * Normally we enforce this by allocating virtual registers to the
       * even-aligned class.  But we need to handle payload registers.
       */
      for (unsigned i = 0; i < inst->sources; i++) {
         if (inst->src[i].file == FIXED_GRF && (inst->src[i].nr & 1) &&
             inst->size_read(i) > REG_SIZE) {
            max_width = MIN2(max_width, 8);
         }
      }
   }

   /* From the IVB PRMs:
    *  "When an instruction is SIMD32, the low 16 bits of the execution mask
    *   are applied for both halves of the SIMD32 instruction. If different
    *   execution mask channels are required, split the instruction into two
    *   SIMD16 instructions."
    *
    * There is similar text in the HSW PRMs.  Gen4-6 don't even implement
    * 32-wide control flow support in hardware and will behave similarly.
    */
   if (devinfo->gen < 8 && !inst->force_writemask_all)
      max_width = MIN2(max_width, 16);

   /* From the IVB PRMs (applies to HSW too):
    *  "Instructions with condition modifiers must not use SIMD32."
    *
    * From the BDW PRMs (applies to later hardware too):
    *  "Ternary instruction with condition modifiers must not use SIMD32."
    */
   if (inst->conditional_mod && (devinfo->gen < 8 || inst->is_3src(devinfo)))
      max_width = MIN2(max_width, 16);

   /* From the IVB PRMs (applies to other devices that don't have the
    * gen_device_info::supports_simd16_3src flag set):
    *  "In Align16 access mode, SIMD16 is not allowed for DW operations and
    *   SIMD8 is not allowed for DF operations."
    */
   if (inst->is_3src(devinfo) && !devinfo->supports_simd16_3src)
      max_width = MIN2(max_width, inst->exec_size / reg_count);

   /* Pre-Gen8 EUs are hardwired to use the QtrCtrl+1 (where QtrCtrl is
    * the 8-bit quarter of the execution mask signals specified in the
    * instruction control fields) for the second compressed half of any
    * single-precision instruction (for double-precision instructions
    * it's hardwired to use NibCtrl+1, at least on HSW), which means that
    * the EU will apply the wrong execution controls for the second
    * sequential GRF write if the number of channels per GRF is not exactly
    * eight in single-precision mode (or four in double-float mode).
    *
    * In this situation we calculate the maximum size of the split
    * instructions so they only ever write to a single register.
    */
   if (devinfo->gen < 8 && inst->size_written > REG_SIZE &&
       !inst->force_writemask_all) {
      const unsigned channels_per_grf = inst->exec_size /
         DIV_ROUND_UP(inst->size_written, REG_SIZE);
      const unsigned exec_type_size = get_exec_type_size(inst);
      assert(exec_type_size);

      /* The hardware shifts exactly 8 channels per compressed half of the
       * instruction in single-precision mode and exactly 4 in double-precision.
       */
      if (channels_per_grf != (exec_type_size == 8 ? 4 : 8))
         max_width = MIN2(max_width, channels_per_grf);

      /* Lower all non-force_writemask_all DF instructions to SIMD4 on IVB/BYT
       * because HW applies the same channel enable signals to both halves of
       * the compressed instruction which will be just wrong under
       * non-uniform control flow.
       */
      if (devinfo->gen == 7 && !devinfo->is_haswell &&
          (exec_type_size == 8 || type_sz(inst->dst.type) == 8))
         max_width = MIN2(max_width, 4);
   }

   /* Only power-of-two execution sizes are representable in the instruction
    * control fields.
    */
   return 1 << _mesa_logbase2(max_width);
}

/**
 * Get the maximum allowed SIMD width for instruction \p inst accounting for
 * various payload size restrictions that apply to sampler message
 * instructions.
 *
 * This is only intended to provide a maximum theoretical bound for the
 * execution size of the message based on the number of argument components
 * alone, which in most cases will determine whether the SIMD8 or SIMD16
 * variant of the message can be used, though some messages may have
 * additional restrictions not accounted for here (e.g. pre-ILK hardware uses
 * the message length to determine the exact SIMD width and argument count,
 * which makes a number of sampler message combinations impossible to
 * represent).
 */
static unsigned
get_sampler_lowered_simd_width(const struct gen_device_info *devinfo,
                               const fs_inst *inst)
{
   /* Calculate the number of coordinate components that have to be present
    * assuming that additional arguments follow the texel coordinates in the
    * message payload.  On IVB+ there is no need for padding, on ILK-SNB we
    * need to pad to four or three components depending on the message,
    * pre-ILK we need to pad to at most three components.
    */
   const unsigned req_coord_components =
      (devinfo->gen >= 7 ||
       !inst->components_read(TEX_LOGICAL_SRC_COORDINATE)) ? 0 :
      (devinfo->gen >= 5 && inst->opcode != SHADER_OPCODE_TXF_LOGICAL &&
                            inst->opcode != SHADER_OPCODE_TXF_CMS_LOGICAL) ? 4 :
      3;

   /* On Gen9+ the LOD argument is for free if we're able to use the LZ
    * variant of the TXL or TXF message.
    */
   const bool implicit_lod = devinfo->gen >= 9 &&
                             (inst->opcode == SHADER_OPCODE_TXL ||
                              inst->opcode == SHADER_OPCODE_TXF) &&
                             inst->src[TEX_LOGICAL_SRC_LOD].is_zero();

   /* Calculate the total number of argument components that need to be passed
    * to the sampler unit.
    */
   const unsigned num_payload_components =
      MAX2(inst->components_read(TEX_LOGICAL_SRC_COORDINATE),
           req_coord_components) +
      inst->components_read(TEX_LOGICAL_SRC_SHADOW_C) +
      (implicit_lod ? 0 : inst->components_read(TEX_LOGICAL_SRC_LOD)) +
      inst->components_read(TEX_LOGICAL_SRC_LOD2) +
      inst->components_read(TEX_LOGICAL_SRC_SAMPLE_INDEX) +
      (inst->opcode == SHADER_OPCODE_TG4_OFFSET_LOGICAL ?
       inst->components_read(TEX_LOGICAL_SRC_TG4_OFFSET) : 0) +
      inst->components_read(TEX_LOGICAL_SRC_MCS);

   /* SIMD16 messages with more than five arguments exceed the maximum message
    * size supported by the sampler, regardless of whether a header is
    * provided or not.
    */
   return MIN2(inst->exec_size,
               num_payload_components > MAX_SAMPLER_MESSAGE_SIZE / 2 ? 8 : 16);
}

/**
 * Get the closest native SIMD width supported by the hardware for instruction
 * \p inst.  The instruction will be left untouched by
 * fs_visitor::lower_simd_width() if the returned value is equal to the
 * original execution size.
 */
static unsigned
get_lowered_simd_width(const struct gen_device_info *devinfo,
                       const fs_inst *inst)
{
   switch (inst->opcode) {
   case BRW_OPCODE_MOV:
   case BRW_OPCODE_SEL:
   case BRW_OPCODE_NOT:
   case BRW_OPCODE_AND:
   case BRW_OPCODE_OR:
   case BRW_OPCODE_XOR:
   case BRW_OPCODE_SHR:
   case BRW_OPCODE_SHL:
   case BRW_OPCODE_ASR:
   case BRW_OPCODE_CMPN:
   case BRW_OPCODE_CSEL:
   case BRW_OPCODE_F32TO16:
   case BRW_OPCODE_F16TO32:
   case BRW_OPCODE_BFREV:
   case BRW_OPCODE_BFE:
   case BRW_OPCODE_ADD:
   case BRW_OPCODE_MUL:
   case BRW_OPCODE_AVG:
   case BRW_OPCODE_FRC:
   case BRW_OPCODE_RNDU:
   case BRW_OPCODE_RNDD:
   case BRW_OPCODE_RNDE:
   case BRW_OPCODE_RNDZ:
   case BRW_OPCODE_LZD:
   case BRW_OPCODE_FBH:
   case BRW_OPCODE_FBL:
   case BRW_OPCODE_CBIT:
   case BRW_OPCODE_SAD2:
   case BRW_OPCODE_MAD:
   case BRW_OPCODE_LRP:
   case FS_OPCODE_PACK:
   case SHADER_OPCODE_SEL_EXEC:
   case SHADER_OPCODE_CLUSTER_BROADCAST:
      return get_fpu_lowered_simd_width(devinfo, inst);

   case BRW_OPCODE_CMP: {
      /* The Ivybridge/BayTrail WaCMPInstFlagDepClearedEarly workaround says that
       * when the destination is a GRF the dependency-clear bit on the flag
       * register is cleared early.
       *
       * Suggested workarounds are to disable coissuing CMP instructions
       * or to split CMP(16) instructions into two CMP(8) instructions.
       *
       * We choose to split into CMP(8) instructions since disabling
       * coissuing would affect CMP instructions not otherwise affected by
       * the errata.
       */
      const unsigned max_width = (devinfo->gen == 7 && !devinfo->is_haswell &&
                                  !inst->dst.is_null() ? 8 : ~0);
      return MIN2(max_width, get_fpu_lowered_simd_width(devinfo, inst));
   }
   case BRW_OPCODE_BFI1:
   case BRW_OPCODE_BFI2:
      /* The Haswell WaForceSIMD8ForBFIInstruction workaround says that we
       * should
       *  "Force BFI instructions to be executed always in SIMD8."
       */
      return MIN2(devinfo->is_haswell ? 8 : ~0u,
                  get_fpu_lowered_simd_width(devinfo, inst));

   case BRW_OPCODE_IF:
      assert(inst->src[0].file == BAD_FILE || inst->exec_size <= 16);
      return inst->exec_size;

   case SHADER_OPCODE_RCP:
   case SHADER_OPCODE_RSQ:
   case SHADER_OPCODE_SQRT:
   case SHADER_OPCODE_EXP2:
   case SHADER_OPCODE_LOG2:
   case SHADER_OPCODE_SIN:
   case SHADER_OPCODE_COS:
      /* Unary extended math instructions are limited to SIMD8 on Gen4 and
       * Gen6.
       */
      return (devinfo->gen >= 7 ? MIN2(16, inst->exec_size) :
              devinfo->gen == 5 || devinfo->is_g4x ? MIN2(16, inst->exec_size) :
              MIN2(8, inst->exec_size));

   case SHADER_OPCODE_POW:
      /* SIMD16 is only allowed on Gen7+. */
      return (devinfo->gen >= 7 ? MIN2(16, inst->exec_size) :
              MIN2(8, inst->exec_size));

   case SHADER_OPCODE_INT_QUOTIENT:
   case SHADER_OPCODE_INT_REMAINDER:
      /* Integer division is limited to SIMD8 on all generations. */
      return MIN2(8, inst->exec_size);

   case FS_OPCODE_LINTERP:
   case SHADER_OPCODE_GET_BUFFER_SIZE:
   case FS_OPCODE_DDX_COARSE:
   case FS_OPCODE_DDX_FINE:
   case FS_OPCODE_DDY_COARSE:
   case FS_OPCODE_UNIFORM_PULL_CONSTANT_LOAD:
   case FS_OPCODE_VARYING_PULL_CONSTANT_LOAD_GEN7:
   case FS_OPCODE_PACK_HALF_2x16_SPLIT:
   case FS_OPCODE_UNPACK_HALF_2x16_SPLIT_X:
   case FS_OPCODE_UNPACK_HALF_2x16_SPLIT_Y:
   case FS_OPCODE_INTERPOLATE_AT_SAMPLE:
   case FS_OPCODE_INTERPOLATE_AT_SHARED_OFFSET:
   case FS_OPCODE_INTERPOLATE_AT_PER_SLOT_OFFSET:
      return MIN2(16, inst->exec_size);

   case FS_OPCODE_VARYING_PULL_CONSTANT_LOAD_LOGICAL:
      /* Pre-ILK hardware doesn't have a SIMD8 variant of the texel fetch
       * message used to implement varying pull constant loads, so expand it
       * to SIMD16.  An alternative with longer message payload length but
       * shorter return payload would be to use the SIMD8 sampler message that
       * takes (header, u, v, r) as parameters instead of (header, u).
       */
      return (devinfo->gen == 4 ? 16 : MIN2(16, inst->exec_size));

   case FS_OPCODE_DDY_FINE:
      /* The implementation of this virtual opcode may require emitting
       * compressed Align16 instructions, which are severely limited on some
       * generations.
       *
       * From the Ivy Bridge PRM, volume 4 part 3, section 3.3.9 (Register
       * Region Restrictions):
       *
       *  "In Align16 access mode, SIMD16 is not allowed for DW operations
       *   and SIMD8 is not allowed for DF operations."
       *
       * In this context, "DW operations" means "operations acting on 32-bit
       * values", so it includes operations on floats.
       *
       * Gen4 has a similar restriction.  From the i965 PRM, section 11.5.3
       * (Instruction Compression -> Rules and Restrictions):
       *
       *  "A compressed instruction must be in Align1 access mode. Align16
       *   mode instructions cannot be compressed."
       *
       * Similar text exists in the g45 PRM.
       *
       * Empirically, compressed align16 instructions using odd register
       * numbers don't appear to work on Sandybridge either.
       */
      return (devinfo->gen == 4 || devinfo->gen == 6 ||
              (devinfo->gen == 7 && !devinfo->is_haswell) ?
              MIN2(8, inst->exec_size) : MIN2(16, inst->exec_size));

   case SHADER_OPCODE_MULH:
      /* MULH is lowered to the MUL/MACH sequence using the accumulator, which
       * is 8-wide on Gen7+.
       */
      return (devinfo->gen >= 7 ? 8 :
              get_fpu_lowered_simd_width(devinfo, inst));

   case FS_OPCODE_FB_WRITE_LOGICAL:
      /* Gen6 doesn't support SIMD16 depth writes but we cannot handle them
       * here.
       */
      assert(devinfo->gen != 6 ||
             inst->src[FB_WRITE_LOGICAL_SRC_SRC_DEPTH].file == BAD_FILE ||
             inst->exec_size == 8);
      /* Dual-source FB writes are unsupported in SIMD16 mode. */
      return (inst->src[FB_WRITE_LOGICAL_SRC_COLOR1].file != BAD_FILE ?
              8 : MIN2(16, inst->exec_size));

   case FS_OPCODE_FB_READ_LOGICAL:
      return MIN2(16, inst->exec_size);

   case SHADER_OPCODE_TEX_LOGICAL:
   case SHADER_OPCODE_TXF_CMS_LOGICAL:
   case SHADER_OPCODE_TXF_UMS_LOGICAL:
   case SHADER_OPCODE_TXF_MCS_LOGICAL:
   case SHADER_OPCODE_LOD_LOGICAL:
   case SHADER_OPCODE_TG4_LOGICAL:
   case SHADER_OPCODE_SAMPLEINFO_LOGICAL:
   case SHADER_OPCODE_TXF_CMS_W_LOGICAL:
   case SHADER_OPCODE_TG4_OFFSET_LOGICAL:
      return get_sampler_lowered_simd_width(devinfo, inst);

   case SHADER_OPCODE_TXD_LOGICAL:
      /* TXD is unsupported in SIMD16 mode. */
      return 8;

   case SHADER_OPCODE_TXL_LOGICAL:
   case FS_OPCODE_TXB_LOGICAL:
      /* Only one execution size is representable pre-ILK depending on whether
       * the shadow reference argument is present.
       */
      if (devinfo->gen == 4)
         return inst->src[TEX_LOGICAL_SRC_SHADOW_C].file == BAD_FILE ? 16 : 8;
      else
         return get_sampler_lowered_simd_width(devinfo, inst);

   case SHADER_OPCODE_TXF_LOGICAL:
   case SHADER_OPCODE_TXS_LOGICAL:
      /* Gen4 doesn't have SIMD8 variants for the RESINFO and LD-with-LOD
       * messages.  Use SIMD16 instead.
       */
      if (devinfo->gen == 4)
         return 16;
      else
         return get_sampler_lowered_simd_width(devinfo, inst);

   case SHADER_OPCODE_TYPED_ATOMIC_LOGICAL:
   case SHADER_OPCODE_TYPED_SURFACE_READ_LOGICAL:
   case SHADER_OPCODE_TYPED_SURFACE_WRITE_LOGICAL:
      return 8;

   case SHADER_OPCODE_UNTYPED_ATOMIC_LOGICAL:
   case SHADER_OPCODE_UNTYPED_ATOMIC_FLOAT_LOGICAL:
   case SHADER_OPCODE_UNTYPED_SURFACE_READ_LOGICAL:
   case SHADER_OPCODE_UNTYPED_SURFACE_WRITE_LOGICAL:
   case SHADER_OPCODE_BYTE_SCATTERED_WRITE_LOGICAL:
   case SHADER_OPCODE_BYTE_SCATTERED_READ_LOGICAL:
      return MIN2(16, inst->exec_size);

   case SHADER_OPCODE_URB_READ_SIMD8:
   case SHADER_OPCODE_URB_READ_SIMD8_PER_SLOT:
   case SHADER_OPCODE_URB_WRITE_SIMD8:
   case SHADER_OPCODE_URB_WRITE_SIMD8_PER_SLOT:
   case SHADER_OPCODE_URB_WRITE_SIMD8_MASKED:
   case SHADER_OPCODE_URB_WRITE_SIMD8_MASKED_PER_SLOT:
      return MIN2(8, inst->exec_size);

   case SHADER_OPCODE_QUAD_SWIZZLE:
      return 8;

   case SHADER_OPCODE_MOV_INDIRECT: {
      /* From IVB and HSW PRMs:
       *
       * "2.When the destination requires two registers and the sources are
       *  indirect, the sources must use 1x1 regioning mode.
       *
       * In case of DF instructions in HSW/IVB, the exec_size is limited by
       * the EU decompression logic not handling VxH indirect addressing
       * correctly.
       */
      const unsigned max_size = (devinfo->gen >= 8 ? 2 : 1) * REG_SIZE;
      /* Prior to Broadwell, we only have 8 address subregisters. */
      return MIN3(devinfo->gen >= 8 ? 16 : 8,
                  max_size / (inst->dst.stride * type_sz(inst->dst.type)),
                  inst->exec_size);
   }

   case SHADER_OPCODE_LOAD_PAYLOAD: {
      const unsigned reg_count =
         DIV_ROUND_UP(inst->dst.component_size(inst->exec_size), REG_SIZE);

      if (reg_count > 2) {
         /* Only LOAD_PAYLOAD instructions with per-channel destination region
          * can be easily lowered (which excludes headers and heterogeneous
          * types).
          */
         assert(!inst->header_size);
         for (unsigned i = 0; i < inst->sources; i++)
            assert(type_sz(inst->dst.type) == type_sz(inst->src[i].type) ||
                   inst->src[i].file == BAD_FILE);

         return inst->exec_size / DIV_ROUND_UP(reg_count, 2);
      } else {
         return inst->exec_size;
      }
   }
   default:
      return inst->exec_size;
   }
}

/**
 * Return true if splitting out the group of channels of instruction \p inst
 * given by lbld.group() requires allocating a temporary for the i-th source
 * of the lowered instruction.
 */
static inline bool
needs_src_copy(const fs_builder &lbld, const fs_inst *inst, unsigned i)
{
   return !(is_periodic(inst->src[i], lbld.dispatch_width()) ||
            (inst->components_read(i) == 1 &&
             lbld.dispatch_width() <= inst->exec_size)) ||
          (inst->flags_written() &
           flag_mask(inst->src[i], type_sz(inst->src[i].type)));
}

/**
 * Extract the data that would be consumed by the channel group given by
 * lbld.group() from the i-th source region of instruction \p inst and return
 * it as result in packed form.
 */
static fs_reg
emit_unzip(const fs_builder &lbld, fs_inst *inst, unsigned i)
{
   /* Specified channel group from the source region. */
   const fs_reg src = horiz_offset(inst->src[i], lbld.group());

   if (needs_src_copy(lbld, inst, i)) {
      /* Builder of the right width to perform the copy avoiding uninitialized
       * data if the lowered execution size is greater than the original
       * execution size of the instruction.
       */
      const fs_builder cbld = lbld.group(MIN2(lbld.dispatch_width(),
                                              inst->exec_size), 0);
      const fs_reg tmp = lbld.vgrf(inst->src[i].type, inst->components_read(i));

      for (unsigned k = 0; k < inst->components_read(i); ++k)
         cbld.MOV(offset(tmp, lbld, k), offset(src, inst->exec_size, k));

      return tmp;

   } else if (is_periodic(inst->src[i], lbld.dispatch_width())) {
      /* The source is invariant for all dispatch_width-wide groups of the
       * original region.
       */
      return inst->src[i];

   } else {
      /* We can just point the lowered instruction at the right channel group
       * from the original region.
       */
      return src;
   }
}

/**
 * Return true if splitting out the group of channels of instruction \p inst
 * given by lbld.group() requires allocating a temporary for the destination
 * of the lowered instruction and copying the data back to the original
 * destination region.
 */
static inline bool
needs_dst_copy(const fs_builder &lbld, const fs_inst *inst)
{
   /* If the instruction writes more than one component we'll have to shuffle
    * the results of multiple lowered instructions in order to make sure that
    * they end up arranged correctly in the original destination region.
    */
   if (inst->size_written > inst->dst.component_size(inst->exec_size))
      return true;

   /* If the lowered execution size is larger than the original the result of
    * the instruction won't fit in the original destination, so we'll have to
    * allocate a temporary in any case.
    */
   if (lbld.dispatch_width() > inst->exec_size)
      return true;

   for (unsigned i = 0; i < inst->sources; i++) {
      /* If we already made a copy of the source for other reasons there won't
       * be any overlap with the destination.
       */
      if (needs_src_copy(lbld, inst, i))
         continue;

      /* In order to keep the logic simple we emit a copy whenever the
       * destination region doesn't exactly match an overlapping source, which
       * may point at the source and destination not being aligned group by
       * group which could cause one of the lowered instructions to overwrite
       * the data read from the same source by other lowered instructions.
       */
      if (regions_overlap(inst->dst, inst->size_written,
                          inst->src[i], inst->size_read(i)) &&
          !inst->dst.equals(inst->src[i]))
        return true;
   }

   return false;
}

/**
 * Insert data from a packed temporary into the channel group given by
 * lbld.group() of the destination region of instruction \p inst and return
 * the temporary as result.  Any copy instructions that are required for
 * unzipping the previous value (in the case of partial writes) will be
 * inserted using \p lbld_before and any copy instructions required for
 * zipping up the destination of \p inst will be inserted using \p lbld_after.
 */
static fs_reg
emit_zip(const fs_builder &lbld_before, const fs_builder &lbld_after,
         fs_inst *inst)
{
   assert(lbld_before.dispatch_width() == lbld_after.dispatch_width());
   assert(lbld_before.group() == lbld_after.group());

   /* Specified channel group from the destination region. */
   const fs_reg dst = horiz_offset(inst->dst, lbld_after.group());
   const unsigned dst_size = inst->size_written /
      inst->dst.component_size(inst->exec_size);

   if (needs_dst_copy(lbld_after, inst)) {
      const fs_reg tmp = lbld_after.vgrf(inst->dst.type, dst_size);

      if (inst->predicate) {
         /* Handle predication by copying the original contents of
          * the destination into the temporary before emitting the
          * lowered instruction.
          */
         const fs_builder gbld_before =
            lbld_before.group(MIN2(lbld_before.dispatch_width(),
                                   inst->exec_size), 0);
         for (unsigned k = 0; k < dst_size; ++k) {
            gbld_before.MOV(offset(tmp, lbld_before, k),
                            offset(dst, inst->exec_size, k));
         }
      }

      const fs_builder gbld_after =
         lbld_after.group(MIN2(lbld_after.dispatch_width(),
                               inst->exec_size), 0);
      for (unsigned k = 0; k < dst_size; ++k) {
         /* Use a builder of the right width to perform the copy avoiding
          * uninitialized data if the lowered execution size is greater than
          * the original execution size of the instruction.
          */
         gbld_after.MOV(offset(dst, inst->exec_size, k),
                        offset(tmp, lbld_after, k));
      }

      return tmp;

   } else {
      /* No need to allocate a temporary for the lowered instruction, just
       * take the right group of channels from the original region.
       */
      return dst;
   }
}

bool
fs_visitor::lower_simd_width()
{
   bool progress = false;

   foreach_block_and_inst_safe(block, fs_inst, inst, cfg) {
      const unsigned lower_width = get_lowered_simd_width(devinfo, inst);

      if (lower_width != inst->exec_size) {
         /* Builder matching the original instruction.  We may also need to
          * emit an instruction of width larger than the original, set the
          * execution size of the builder to the highest of both for now so
          * we're sure that both cases can be handled.
          */
         const unsigned max_width = MAX2(inst->exec_size, lower_width);
         const fs_builder ibld = bld.at(block, inst)
                                    .exec_all(inst->force_writemask_all)
                                    .group(max_width, inst->group / max_width);

         /* Split the copies in chunks of the execution width of either the
          * original or the lowered instruction, whichever is lower.
          */
         const unsigned n = DIV_ROUND_UP(inst->exec_size, lower_width);
         const unsigned dst_size = inst->size_written /
            inst->dst.component_size(inst->exec_size);

         assert(!inst->writes_accumulator && !inst->mlen);

         /* Inserting the zip, unzip, and duplicated instructions in all of
          * the right spots is somewhat tricky.  All of the unzip and any
          * instructions from the zip which unzip the destination prior to
          * writing need to happen before all of the per-group instructions
          * and the zip instructions need to happen after.  In order to sort
          * this all out, we insert the unzip instructions before \p inst,
          * insert the per-group instructions after \p inst (i.e. before
          * inst->next), and insert the zip instructions before the
          * instruction after \p inst.  Since we are inserting instructions
          * after \p inst, inst->next is a moving target and we need to save
          * it off here so that we insert the zip instructions in the right
          * place.
          *
          * Since we're inserting split instructions after after_inst, the
          * instructions will end up in the reverse order that we insert them.
          * However, certain render target writes require that the low group
          * instructions come before the high group.  From the Ivy Bridge PRM
          * Vol. 4, Pt. 1, Section 3.9.11:
          *
          *    "If multiple SIMD8 Dual Source messages are delivered by the
          *    pixel shader thread, each SIMD8_DUALSRC_LO message must be
          *    issued before the SIMD8_DUALSRC_HI message with the same Slot
          *    Group Select setting."
          *
          * And, from Section 3.9.11.1 of the same PRM:
          *
          *    "When SIMD32 or SIMD16 PS threads send render target writes
          *    with multiple SIMD8 and SIMD16 messages, the following must
          *    hold:
          *
          *    All the slots (as described above) must have a corresponding
          *    render target write irrespective of the slot's validity. A slot
          *    is considered valid when at least one sample is enabled. For
          *    example, a SIMD16 PS thread must send two SIMD8 render target
          *    writes to cover all the slots.
          *
          *    PS thread must send SIMD render target write messages with
          *    increasing slot numbers. For example, SIMD16 thread has
          *    Slot[15:0] and if two SIMD8 render target writes are used, the
          *    first SIMD8 render target write must send Slot[7:0] and the
          *    next one must send Slot[15:8]."
          *
          * In order to make low group instructions come before high group
          * instructions (this is required for some render target writes), we
          * split from the highest group to lowest.
          */
         exec_node *const after_inst = inst->next;
         for (int i = n - 1; i >= 0; i--) {
            /* Emit a copy of the original instruction with the lowered width.
             * If the EOT flag was set throw it away except for the last
             * instruction to avoid killing the thread prematurely.
             */
            fs_inst split_inst = *inst;
            split_inst.exec_size = lower_width;
            split_inst.eot = inst->eot && i == int(n - 1);

            /* Select the correct channel enables for the i-th group, then
             * transform the sources and destination and emit the lowered
             * instruction.
             */
            const fs_builder lbld = ibld.group(lower_width, i);

            for (unsigned j = 0; j < inst->sources; j++)
               split_inst.src[j] = emit_unzip(lbld.at(block, inst), inst, j);

            split_inst.dst = emit_zip(lbld.at(block, inst),
                                      lbld.at(block, after_inst), inst);
            split_inst.size_written =
               split_inst.dst.component_size(lower_width) * dst_size;

            lbld.at(block, inst->next).emit(split_inst);
         }

         inst->remove(block);
         progress = true;
      }
   }

   if (progress)
      invalidate_live_intervals();

   return progress;
}

void
fs_visitor::dump_instructions()
{
   dump_instructions(NULL);
}

void
fs_visitor::dump_instructions(const char *name)
{
   FILE *file = stderr;
   if (name && geteuid() != 0) {
      file = fopen(name, "w");
      if (!file)
         file = stderr;
   }

   if (cfg) {
      calculate_register_pressure();
      int ip = 0, max_pressure = 0;
      foreach_block_and_inst(block, backend_instruction, inst, cfg) {
         max_pressure = MAX2(max_pressure, regs_live_at_ip[ip]);
         fprintf(file, "{%3d} %4d: ", regs_live_at_ip[ip], ip);
         dump_instruction(inst, file);
         ip++;
      }
      fprintf(file, "Maximum %3d registers live at once.\n", max_pressure);
   } else {
      int ip = 0;
      foreach_in_list(backend_instruction, inst, &instructions) {
         fprintf(file, "%4d: ", ip++);
         dump_instruction(inst, file);
      }
   }

   if (file != stderr) {
      fclose(file);
   }
}

void
fs_visitor::dump_instruction(backend_instruction *be_inst)
{
   dump_instruction(be_inst, stderr);
}

void
fs_visitor::dump_instruction(backend_instruction *be_inst, FILE *file)
{
   fs_inst *inst = (fs_inst *)be_inst;

   if (inst->predicate) {
      fprintf(file, "(%cf%d.%d) ",
              inst->predicate_inverse ? '-' : '+',
              inst->flag_subreg / 2,
              inst->flag_subreg % 2);
   }

   fprintf(file, "%s", brw_instruction_name(devinfo, inst->opcode));
   if (inst->saturate)
      fprintf(file, ".sat");
   if (inst->conditional_mod) {
      fprintf(file, "%s", conditional_modifier[inst->conditional_mod]);
      if (!inst->predicate &&
          (devinfo->gen < 5 || (inst->opcode != BRW_OPCODE_SEL &&
                                inst->opcode != BRW_OPCODE_CSEL &&
                                inst->opcode != BRW_OPCODE_IF &&
                                inst->opcode != BRW_OPCODE_WHILE))) {
         fprintf(file, ".f%d.%d", inst->flag_subreg / 2,
                 inst->flag_subreg % 2);
      }
   }
   fprintf(file, "(%d) ", inst->exec_size);

   if (inst->mlen) {
      fprintf(file, "(mlen: %d) ", inst->mlen);
   }

   if (inst->eot) {
      fprintf(file, "(EOT) ");
   }

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
   case BAD_FILE:
      fprintf(file, "(null)");
      break;
   case UNIFORM:
      fprintf(file, "***u%d***", inst->dst.nr);
      break;
   case ATTR:
      fprintf(file, "***attr%d***", inst->dst.nr);
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
   case IMM:
      unreachable("not reached");
   }

   if (inst->dst.offset ||
       (inst->dst.file == VGRF &&
        alloc.sizes[inst->dst.nr] * REG_SIZE != inst->size_written)) {
      const unsigned reg_size = (inst->dst.file == UNIFORM ? 4 : REG_SIZE);
      fprintf(file, "+%d.%d", inst->dst.offset / reg_size,
              inst->dst.offset % reg_size);
   }

   if (inst->dst.stride != 1)
      fprintf(file, "<%u>", inst->dst.stride);
   fprintf(file, ":%s, ", brw_reg_type_to_letters(inst->dst.type));

   for (int i = 0; i < inst->sources; i++) {
      if (inst->src[i].negate)
         fprintf(file, "-");
      if (inst->src[i].abs)
         fprintf(file, "|");
      switch (inst->src[i].file) {
      case VGRF:
         fprintf(file, "vgrf%d", inst->src[i].nr);
         break;
      case FIXED_GRF:
         fprintf(file, "g%d", inst->src[i].nr);
         break;
      case MRF:
         fprintf(file, "***m%d***", inst->src[i].nr);
         break;
      case ATTR:
         fprintf(file, "attr%d", inst->src[i].nr);
         break;
      case UNIFORM:
         fprintf(file, "u%d", inst->src[i].nr);
         break;
      case BAD_FILE:
         fprintf(file, "(null)");
         break;
      case IMM:
         switch (inst->src[i].type) {
         case BRW_REGISTER_TYPE_F:
            fprintf(file, "%-gf", inst->src[i].f);
            break;
         case BRW_REGISTER_TYPE_DF:
            fprintf(file, "%fdf", inst->src[i].df);
            break;
         case BRW_REGISTER_TYPE_W:
         case BRW_REGISTER_TYPE_D:
            fprintf(file, "%dd", inst->src[i].d);
            break;
         case BRW_REGISTER_TYPE_UW:
         case BRW_REGISTER_TYPE_UD:
            fprintf(file, "%uu", inst->src[i].ud);
            break;
         case BRW_REGISTER_TYPE_Q:
            fprintf(file, "%" PRId64 "q", inst->src[i].d64);
            break;
         case BRW_REGISTER_TYPE_UQ:
            fprintf(file, "%" PRIu64 "uq", inst->src[i].u64);
            break;
         case BRW_REGISTER_TYPE_VF:
            fprintf(file, "[%-gF, %-gF, %-gF, %-gF]",
                    brw_vf_to_float((inst->src[i].ud >>  0) & 0xff),
                    brw_vf_to_float((inst->src[i].ud >>  8) & 0xff),
                    brw_vf_to_float((inst->src[i].ud >> 16) & 0xff),
                    brw_vf_to_float((inst->src[i].ud >> 24) & 0xff));
            break;
         case BRW_REGISTER_TYPE_V:
         case BRW_REGISTER_TYPE_UV:
            fprintf(file, "%08x%s", inst->src[i].ud,
                    inst->src[i].type == BRW_REGISTER_TYPE_V ? "V" : "UV");
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
      }

      if (inst->src[i].offset ||
          (inst->src[i].file == VGRF &&
           alloc.sizes[inst->src[i].nr] * REG_SIZE != inst->size_read(i))) {
         const unsigned reg_size = (inst->src[i].file == UNIFORM ? 4 : REG_SIZE);
         fprintf(file, "+%d.%d", inst->src[i].offset / reg_size,
                 inst->src[i].offset % reg_size);
      }

      if (inst->src[i].abs)
         fprintf(file, "|");

      if (inst->src[i].file != IMM) {
         unsigned stride;
         if (inst->src[i].file == ARF || inst->src[i].file == FIXED_GRF) {
            unsigned hstride = inst->src[i].hstride;
            stride = (hstride == 0 ? 0 : (1 << (hstride - 1)));
         } else {
            stride = inst->src[i].stride;
         }
         if (stride != 1)
            fprintf(file, "<%u>", stride);

         fprintf(file, ":%s", brw_reg_type_to_letters(inst->src[i].type));
      }

      if (i < inst->sources - 1 && inst->src[i + 1].file != BAD_FILE)
         fprintf(file, ", ");
   }

   fprintf(file, " ");

   if (inst->force_writemask_all)
      fprintf(file, "NoMask ");

   if (inst->exec_size != dispatch_width)
      fprintf(file, "group%d ", inst->group);

   fprintf(file, "\n");
}

void
fs_visitor::setup_fs_payload_gen6()
{
   assert(stage == MESA_SHADER_FRAGMENT);
   struct brw_wm_prog_data *prog_data = brw_wm_prog_data(this->prog_data);
   const unsigned payload_width = MIN2(16, dispatch_width);
   assert(dispatch_width % payload_width == 0);
   assert(devinfo->gen >= 6);

   prog_data->uses_src_depth = prog_data->uses_src_w =
      (nir->info.inputs_read & (1 << VARYING_SLOT_POS)) != 0;

   prog_data->uses_sample_mask =
      (nir->info.system_values_read & SYSTEM_BIT_SAMPLE_MASK_IN) != 0;

   /* From the Ivy Bridge PRM documentation for 3DSTATE_PS:
    *
    *    "MSDISPMODE_PERSAMPLE is required in order to select
    *    POSOFFSET_SAMPLE"
    *
    * So we can only really get sample positions if we are doing real
    * per-sample dispatch.  If we need gl_SamplePosition and we don't have
    * persample dispatch, we hard-code it to 0.5.
    */
   prog_data->uses_pos_offset = prog_data->persample_dispatch &&
      (nir->info.system_values_read & SYSTEM_BIT_SAMPLE_POS);

   /* R0: PS thread payload header. */
   payload.num_regs++;

   for (unsigned j = 0; j < dispatch_width / payload_width; j++) {
      /* R1: masks, pixel X/Y coordinates. */
      payload.subspan_coord_reg[j] = payload.num_regs++;
   }

   for (unsigned j = 0; j < dispatch_width / payload_width; j++) {
      /* R3-26: barycentric interpolation coordinates.  These appear in the
       * same order that they appear in the brw_barycentric_mode enum.  Each
       * set of coordinates occupies 2 registers if dispatch width == 8 and 4
       * registers if dispatch width == 16.  Coordinates only appear if they
       * were enabled using the "Barycentric Interpolation Mode" bits in
       * WM_STATE.
       */
      for (int i = 0; i < BRW_BARYCENTRIC_MODE_COUNT; ++i) {
         if (prog_data->barycentric_interp_modes & (1 << i)) {
            payload.barycentric_coord_reg[i][j] = payload.num_regs;
            payload.num_regs += payload_width / 4;
         }
      }

      /* R27-28: interpolated depth if uses source depth */
      if (prog_data->uses_src_depth) {
         payload.source_depth_reg[j] = payload.num_regs;
         payload.num_regs += payload_width / 8;
      }

      /* R29-30: interpolated W set if GEN6_WM_USES_SOURCE_W. */
      if (prog_data->uses_src_w) {
         payload.source_w_reg[j] = payload.num_regs;
         payload.num_regs += payload_width / 8;
      }

      /* R31: MSAA position offsets. */
      if (prog_data->uses_pos_offset) {
         payload.sample_pos_reg[j] = payload.num_regs;
         payload.num_regs++;
      }

      /* R32-33: MSAA input coverage mask */
      if (prog_data->uses_sample_mask) {
         assert(devinfo->gen >= 7);
         payload.sample_mask_in_reg[j] = payload.num_regs;
         payload.num_regs += payload_width / 8;
      }
   }

   if (nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DEPTH)) {
      source_depth_to_render_target = true;
   }
}

void
fs_visitor::setup_vs_payload()
{
   /* R0: thread header, R1: urb handles */
   payload.num_regs = 2;
}

void
fs_visitor::setup_gs_payload()
{
   assert(stage == MESA_SHADER_GEOMETRY);

   struct brw_gs_prog_data *gs_prog_data = brw_gs_prog_data(prog_data);
   struct brw_vue_prog_data *vue_prog_data = brw_vue_prog_data(prog_data);

   /* R0: thread header, R1: output URB handles */
   payload.num_regs = 2;

   if (gs_prog_data->include_primitive_id) {
      /* R2: Primitive ID 0..7 */
      payload.num_regs++;
   }

   /* Always enable VUE handles so we can safely use pull model if needed.
    *
    * The push model for a GS uses a ton of register space even for trivial
    * scenarios with just a few inputs, so just make things easier and a bit
    * safer by always having pull model available.
    */
   gs_prog_data->base.include_vue_handles = true;

   /* R3..RN: ICP Handles for each incoming vertex (when using pull model) */
   payload.num_regs += nir->info.gs.vertices_in;

   /* Use a maximum of 24 registers for push-model inputs. */
   const unsigned max_push_components = 24;

   /* If pushing our inputs would take too many registers, reduce the URB read
    * length (which is in HWords, or 8 registers), and resort to pulling.
    *
    * Note that the GS reads <URB Read Length> HWords for every vertex - so we
    * have to multiply by VerticesIn to obtain the total storage requirement.
    */
   if (8 * vue_prog_data->urb_read_length * nir->info.gs.vertices_in >
       max_push_components) {
      vue_prog_data->urb_read_length =
         ROUND_DOWN_TO(max_push_components / nir->info.gs.vertices_in, 8) / 8;
   }
}

void
fs_visitor::setup_cs_payload()
{
   assert(devinfo->gen >= 7);
   payload.num_regs = 1;
}

void
fs_visitor::calculate_register_pressure()
{
   invalidate_live_intervals();
   calculate_live_intervals();

   unsigned num_instructions = 0;
   foreach_block(block, cfg)
      num_instructions += block->instructions.length();

   regs_live_at_ip = rzalloc_array(mem_ctx, int, num_instructions);

   for (unsigned reg = 0; reg < alloc.count; reg++) {
      for (int ip = virtual_grf_start[reg]; ip <= virtual_grf_end[reg]; ip++)
         regs_live_at_ip[ip] += alloc.sizes[reg];
   }
}

void
fs_visitor::optimize()
{
   /* Start by validating the shader we currently have. */
   validate();

   /* bld is the common builder object pointing at the end of the program we
    * used to translate it into i965 IR.  For the optimization and lowering
    * passes coming next, any code added after the end of the program without
    * having explicitly called fs_builder::at() clearly points at a mistake.
    * Ideally optimization passes wouldn't be part of the visitor so they
    * wouldn't have access to bld at all, but they do, so just in case some
    * pass forgets to ask for a location explicitly set it to NULL here to
    * make it trip.  The dispatch width is initialized to a bogus value to
    * make sure that optimizations set the execution controls explicitly to
    * match the code they are manipulating instead of relying on the defaults.
    */
   bld = fs_builder(this, 64);

   assign_constant_locations();
   lower_constant_loads();

   validate();

   split_virtual_grfs();
   validate();

#define OPT(pass, args...) ({                                           \
      pass_num++;                                                       \
      bool this_progress = pass(args);                                  \
                                                                        \
      if (unlikely(INTEL_DEBUG & DEBUG_OPTIMIZER) && this_progress) {   \
         char filename[64];                                             \
         snprintf(filename, 64, "%s%d-%s-%02d-%02d-" #pass,              \
                  stage_abbrev, dispatch_width, nir->info.name, iteration, pass_num); \
                                                                        \
         backend_shader::dump_instructions(filename);                   \
      }                                                                 \
                                                                        \
      validate();                                                       \
                                                                        \
      progress = progress || this_progress;                             \
      this_progress;                                                    \
   })

   if (unlikely(INTEL_DEBUG & DEBUG_OPTIMIZER)) {
      char filename[64];
      snprintf(filename, 64, "%s%d-%s-00-00-start",
               stage_abbrev, dispatch_width, nir->info.name);

      backend_shader::dump_instructions(filename);
   }

   bool progress = false;
   int iteration = 0;
   int pass_num = 0;

   OPT(remove_extra_rounding_modes);

   do {
      progress = false;
      pass_num = 0;
      iteration++;

      OPT(remove_duplicate_mrf_writes);

      OPT(opt_algebraic);
      OPT(opt_cse);
      OPT(opt_copy_propagation);
      OPT(opt_predicated_break, this);
      OPT(opt_cmod_propagation);
      OPT(dead_code_eliminate);
      OPT(opt_peephole_sel);
      OPT(dead_control_flow_eliminate, this);
      OPT(opt_register_renaming);
      OPT(opt_saturate_propagation);
      OPT(register_coalesce);
      OPT(compute_to_mrf);
      OPT(eliminate_find_live_channel);

      OPT(compact_virtual_grfs);
   } while (progress);

   /* Do this after cmod propagation has had every possible opportunity to
    * propagate results into SEL instructions.
    */
   if (OPT(opt_peephole_csel))
      OPT(dead_code_eliminate);

   progress = false;
   pass_num = 0;

   if (OPT(lower_pack)) {
      OPT(register_coalesce);
      OPT(dead_code_eliminate);
   }

   OPT(lower_simd_width);

   /* After SIMD lowering just in case we had to unroll the EOT send. */
   OPT(opt_sampler_eot);

   OPT(lower_logical_sends);

   if (progress) {
      OPT(opt_copy_propagation);
      /* Only run after logical send lowering because it's easier to implement
       * in terms of physical sends.
       */
      if (OPT(opt_zero_samples))
         OPT(opt_copy_propagation);
      /* Run after logical send lowering to give it a chance to CSE the
       * LOAD_PAYLOAD instructions created to construct the payloads of
       * e.g. texturing messages in cases where it wasn't possible to CSE the
       * whole logical instruction.
       */
      OPT(opt_cse);
      OPT(register_coalesce);
      OPT(compute_to_mrf);
      OPT(dead_code_eliminate);
      OPT(remove_duplicate_mrf_writes);
      OPT(opt_peephole_sel);
   }

   OPT(opt_redundant_discard_jumps);

   if (OPT(lower_load_payload)) {
      split_virtual_grfs();
      OPT(register_coalesce);
      OPT(lower_simd_width);
      OPT(compute_to_mrf);
      OPT(dead_code_eliminate);
   }

   OPT(opt_combine_constants);
   OPT(lower_integer_multiplication);

   if (devinfo->gen <= 5 && OPT(lower_minmax)) {
      OPT(opt_cmod_propagation);
      OPT(opt_cse);
      OPT(opt_copy_propagation);
      OPT(dead_code_eliminate);
   }

   if (OPT(lower_conversions)) {
      OPT(opt_copy_propagation);
      OPT(dead_code_eliminate);
      OPT(lower_simd_width);
   }

   lower_uniform_pull_constant_loads();

   validate();
}

/**
 * Three source instruction must have a GRF/MRF destination register.
 * ARF NULL is not allowed.  Fix that up by allocating a temporary GRF.
 */
void
fs_visitor::fixup_3src_null_dest()
{
   bool progress = false;

   foreach_block_and_inst_safe (block, fs_inst, inst, cfg) {
      if (inst->is_3src(devinfo) && inst->dst.is_null()) {
         inst->dst = fs_reg(VGRF, alloc.allocate(dispatch_width / 8),
                            inst->dst.type);
         progress = true;
      }
   }

   if (progress)
      invalidate_live_intervals();
}

void
fs_visitor::allocate_registers(unsigned min_dispatch_width, bool allow_spilling)
{
   bool allocated_without_spills;

   static const enum instruction_scheduler_mode pre_modes[] = {
      SCHEDULE_PRE,
      SCHEDULE_PRE_NON_LIFO,
      SCHEDULE_PRE_LIFO,
   };

   bool spill_all = allow_spilling && (INTEL_DEBUG & DEBUG_SPILL_FS);

   /* Try each scheduling heuristic to see if it can successfully register
    * allocate without spilling.  They should be ordered by decreasing
    * performance but increasing likelihood of allocating.
    */
   for (unsigned i = 0; i < ARRAY_SIZE(pre_modes); i++) {
      schedule_instructions(pre_modes[i]);

      if (0) {
         assign_regs_trivial();
         allocated_without_spills = true;
      } else {
         allocated_without_spills = assign_regs(false, spill_all);
      }
      if (allocated_without_spills)
         break;
   }

   if (!allocated_without_spills) {
      if (!allow_spilling)
         fail("Failure to register allocate and spilling is not allowed.");

      /* We assume that any spilling is worse than just dropping back to
       * SIMD8.  There's probably actually some intermediate point where
       * SIMD16 with a couple of spills is still better.
       */
      if (dispatch_width > min_dispatch_width) {
         fail("Failure to register allocate.  Reduce number of "
              "live scalar values to avoid this.");
      } else {
         compiler->shader_perf_log(log_data,
                                   "%s shader triggered register spilling.  "
                                   "Try reducing the number of live scalar "
                                   "values to improve performance.\n",
                                   stage_name);
      }

      /* Since we're out of heuristics, just go spill registers until we
       * get an allocation.
       */
      while (!assign_regs(true, spill_all)) {
         if (failed)
            break;
      }
   }

   /* This must come after all optimization and register allocation, since
    * it inserts dead code that happens to have side effects, and it does
    * so based on the actual physical registers in use.
    */
   insert_gen4_send_dependency_workarounds();

   if (failed)
      return;

   opt_bank_conflicts();

   schedule_instructions(SCHEDULE_POST);

   if (last_scratch > 0) {
      MAYBE_UNUSED unsigned max_scratch_size = 2 * 1024 * 1024;

      prog_data->total_scratch = brw_get_scratch_size(last_scratch);

      if (stage == MESA_SHADER_COMPUTE) {
         if (devinfo->is_haswell) {
            /* According to the MEDIA_VFE_STATE's "Per Thread Scratch Space"
             * field documentation, Haswell supports a minimum of 2kB of
             * scratch space for compute shaders, unlike every other stage
             * and platform.
             */
            prog_data->total_scratch = MAX2(prog_data->total_scratch, 2048);
         } else if (devinfo->gen <= 7) {
            /* According to the MEDIA_VFE_STATE's "Per Thread Scratch Space"
             * field documentation, platforms prior to Haswell measure scratch
             * size linearly with a range of [1kB, 12kB] and 1kB granularity.
             */
            prog_data->total_scratch = ALIGN(last_scratch, 1024);
            max_scratch_size = 12 * 1024;
         }
      }

      /* We currently only support up to 2MB of scratch space.  If we
       * need to support more eventually, the documentation suggests
       * that we could allocate a larger buffer, and partition it out
       * ourselves.  We'd just have to undo the hardware's address
       * calculation by subtracting (FFTID * Per Thread Scratch Space)
       * and then add FFTID * (Larger Per Thread Scratch Space).
       *
       * See 3D-Media-GPGPU Engine > Media GPGPU Pipeline >
       * Thread Group Tracking > Local Memory/Scratch Space.
       */
      assert(prog_data->total_scratch < max_scratch_size);
   }
}

bool
fs_visitor::run_vs()
{
   assert(stage == MESA_SHADER_VERTEX);

   setup_vs_payload();

   if (shader_time_index >= 0)
      emit_shader_time_begin();

   emit_nir_code();

   if (failed)
      return false;

   compute_clip_distance();

   emit_urb_writes();

   if (shader_time_index >= 0)
      emit_shader_time_end();

   calculate_cfg();

   optimize();

   assign_curb_setup();
   assign_vs_urb_setup();

   fixup_3src_null_dest();
   allocate_registers(8, true);

   return !failed;
}

bool
fs_visitor::run_tcs_single_patch()
{
   assert(stage == MESA_SHADER_TESS_CTRL);

   struct brw_tcs_prog_data *tcs_prog_data = brw_tcs_prog_data(prog_data);

   /* r1-r4 contain the ICP handles. */
   payload.num_regs = 5;

   if (shader_time_index >= 0)
      emit_shader_time_begin();

   /* Initialize gl_InvocationID */
   fs_reg channels_uw = bld.vgrf(BRW_REGISTER_TYPE_UW);
   fs_reg channels_ud = bld.vgrf(BRW_REGISTER_TYPE_UD);
   bld.MOV(channels_uw, fs_reg(brw_imm_uv(0x76543210)));
   bld.MOV(channels_ud, channels_uw);

   if (tcs_prog_data->instances == 1) {
      invocation_id = channels_ud;
   } else {
      const unsigned invocation_id_mask = devinfo->gen >= 11 ?
         INTEL_MASK(22, 16) : INTEL_MASK(23, 17);
      const unsigned invocation_id_shift = devinfo->gen >= 11 ? 16 : 17;

      invocation_id = bld.vgrf(BRW_REGISTER_TYPE_UD);

      /* Get instance number from g0.2 bits 23:17, and multiply it by 8. */
      fs_reg t = bld.vgrf(BRW_REGISTER_TYPE_UD);
      fs_reg instance_times_8 = bld.vgrf(BRW_REGISTER_TYPE_UD);
      bld.AND(t, fs_reg(retype(brw_vec1_grf(0, 2), BRW_REGISTER_TYPE_UD)),
              brw_imm_ud(invocation_id_mask));
      bld.SHR(instance_times_8, t, brw_imm_ud(invocation_id_shift - 3));

      bld.ADD(invocation_id, instance_times_8, channels_ud);
   }

   /* Fix the disptach mask */
   if (nir->info.tess.tcs_vertices_out % 8) {
      bld.CMP(bld.null_reg_ud(), invocation_id,
              brw_imm_ud(nir->info.tess.tcs_vertices_out), BRW_CONDITIONAL_L);
      bld.IF(BRW_PREDICATE_NORMAL);
   }

   emit_nir_code();

   if (nir->info.tess.tcs_vertices_out % 8) {
      bld.emit(BRW_OPCODE_ENDIF);
   }

   /* Emit EOT write; set TR DS Cache bit */
   fs_reg srcs[3] = {
      fs_reg(retype(brw_vec1_grf(0, 0), BRW_REGISTER_TYPE_UD)),
      fs_reg(brw_imm_ud(WRITEMASK_X << 16)),
      fs_reg(brw_imm_ud(0)),
   };
   fs_reg payload = bld.vgrf(BRW_REGISTER_TYPE_UD, 3);
   bld.LOAD_PAYLOAD(payload, srcs, 3, 2);

   fs_inst *inst = bld.emit(SHADER_OPCODE_URB_WRITE_SIMD8_MASKED,
                            bld.null_reg_ud(), payload);
   inst->mlen = 3;
   inst->eot = true;

   if (shader_time_index >= 0)
      emit_shader_time_end();

   if (failed)
      return false;

   calculate_cfg();

   optimize();

   assign_curb_setup();
   assign_tcs_single_patch_urb_setup();

   fixup_3src_null_dest();
   allocate_registers(8, true);

   return !failed;
}

bool
fs_visitor::run_tes()
{
   assert(stage == MESA_SHADER_TESS_EVAL);

   /* R0: thread header, R1-3: gl_TessCoord.xyz, R4: URB handles */
   payload.num_regs = 5;

   if (shader_time_index >= 0)
      emit_shader_time_begin();

   emit_nir_code();

   if (failed)
      return false;

   emit_urb_writes();

   if (shader_time_index >= 0)
      emit_shader_time_end();

   calculate_cfg();

   optimize();

   assign_curb_setup();
   assign_tes_urb_setup();

   fixup_3src_null_dest();
   allocate_registers(8, true);

   return !failed;
}

bool
fs_visitor::run_gs()
{
   assert(stage == MESA_SHADER_GEOMETRY);

   setup_gs_payload();

   this->final_gs_vertex_count = vgrf(glsl_type::uint_type);

   if (gs_compile->control_data_header_size_bits > 0) {
      /* Create a VGRF to store accumulated control data bits. */
      this->control_data_bits = vgrf(glsl_type::uint_type);

      /* If we're outputting more than 32 control data bits, then EmitVertex()
       * will set control_data_bits to 0 after emitting the first vertex.
       * Otherwise, we need to initialize it to 0 here.
       */
      if (gs_compile->control_data_header_size_bits <= 32) {
         const fs_builder abld = bld.annotate("initialize control data bits");
         abld.MOV(this->control_data_bits, brw_imm_ud(0u));
      }
   }

   if (shader_time_index >= 0)
      emit_shader_time_begin();

   emit_nir_code();

   emit_gs_thread_end();

   if (shader_time_index >= 0)
      emit_shader_time_end();

   if (failed)
      return false;

   calculate_cfg();

   optimize();

   assign_curb_setup();
   assign_gs_urb_setup();

   fixup_3src_null_dest();
   allocate_registers(8, true);

   return !failed;
}

/* From the SKL PRM, Volume 16, Workarounds:
 *
 *   0877  3D   Pixel Shader Hang possible when pixel shader dispatched with
 *              only header phases (R0-R2)
 *
 *   WA: Enable a non-header phase (e.g. push constant) when dispatch would
 *       have been header only.
 *
 * Instead of enabling push constants one can alternatively enable one of the
 * inputs. Here one simply chooses "layer" which shouldn't impose much
 * overhead.
 */
static void
gen9_ps_header_only_workaround(struct brw_wm_prog_data *wm_prog_data)
{
   if (wm_prog_data->num_varying_inputs)
      return;

   if (wm_prog_data->base.curb_read_length)
      return;

   wm_prog_data->urb_setup[VARYING_SLOT_LAYER] = 0;
   wm_prog_data->num_varying_inputs = 1;
}

bool
fs_visitor::run_fs(bool allow_spilling, bool do_rep_send)
{
   struct brw_wm_prog_data *wm_prog_data = brw_wm_prog_data(this->prog_data);
   brw_wm_prog_key *wm_key = (brw_wm_prog_key *) this->key;

   assert(stage == MESA_SHADER_FRAGMENT);

   if (devinfo->gen >= 6)
      setup_fs_payload_gen6();
   else
      setup_fs_payload_gen4();

   if (0) {
      emit_dummy_fs();
   } else if (do_rep_send) {
      assert(dispatch_width == 16);
      emit_repclear_shader();
   } else {
      if (shader_time_index >= 0)
         emit_shader_time_begin();

      calculate_urb_setup();
      if (nir->info.inputs_read > 0 ||
          (nir->info.outputs_read > 0 && !wm_key->coherent_fb_fetch)) {
         if (devinfo->gen < 6)
            emit_interpolation_setup_gen4();
         else
            emit_interpolation_setup_gen6();
      }

      /* We handle discards by keeping track of the still-live pixels in f0.1.
       * Initialize it with the dispatched pixels.
       */
      if (wm_prog_data->uses_kill) {
         const fs_reg dispatch_mask =
            devinfo->gen >= 6 ? brw_vec1_grf(1, 7) : brw_vec1_grf(0, 0);
         bld.exec_all().group(1, 0)
            .MOV(retype(brw_flag_reg(0, 1), BRW_REGISTER_TYPE_UW),
                 retype(dispatch_mask, BRW_REGISTER_TYPE_UW));
      }

      emit_nir_code();

      if (failed)
	 return false;

      if (wm_prog_data->uses_kill)
         bld.emit(FS_OPCODE_PLACEHOLDER_HALT);

      if (wm_key->alpha_test_func)
         emit_alpha_test();

      emit_fb_writes();

      if (shader_time_index >= 0)
         emit_shader_time_end();

      calculate_cfg();

      optimize();

      assign_curb_setup();

      if (devinfo->gen >= 9)
         gen9_ps_header_only_workaround(wm_prog_data);

      assign_urb_setup();

      fixup_3src_null_dest();
      allocate_registers(8, allow_spilling);

      if (failed)
         return false;
   }

   return !failed;
}

bool
fs_visitor::run_cs(unsigned min_dispatch_width)
{
   assert(stage == MESA_SHADER_COMPUTE);
   assert(dispatch_width >= min_dispatch_width);

   setup_cs_payload();

   if (shader_time_index >= 0)
      emit_shader_time_begin();

   if (devinfo->is_haswell && prog_data->total_shared > 0) {
      /* Move SLM index from g0.0[27:24] to sr0.1[11:8] */
      const fs_builder abld = bld.exec_all().group(1, 0);
      abld.MOV(retype(brw_sr0_reg(1), BRW_REGISTER_TYPE_UW),
               suboffset(retype(brw_vec1_grf(0, 0), BRW_REGISTER_TYPE_UW), 1));
   }

   emit_nir_code();

   if (failed)
      return false;

   emit_cs_terminate();

   if (shader_time_index >= 0)
      emit_shader_time_end();

   calculate_cfg();

   optimize();

   assign_curb_setup();

   fixup_3src_null_dest();
   allocate_registers(min_dispatch_width, true);

   if (failed)
      return false;

   return !failed;
}

/**
 * Return a bitfield where bit n is set if barycentric interpolation mode n
 * (see enum brw_barycentric_mode) is needed by the fragment shader.
 *
 * We examine the load_barycentric intrinsics rather than looking at input
 * variables so that we catch interpolateAtCentroid() messages too, which
 * also need the BRW_BARYCENTRIC_[NON]PERSPECTIVE_CENTROID mode set up.
 */
static unsigned
brw_compute_barycentric_interp_modes(const struct gen_device_info *devinfo,
                                     const nir_shader *shader)
{
   unsigned barycentric_interp_modes = 0;

   nir_foreach_function(f, shader) {
      if (!f->impl)
         continue;

      nir_foreach_block(block, f->impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (intrin->intrinsic != nir_intrinsic_load_interpolated_input)
               continue;

            /* Ignore WPOS; it doesn't require interpolation. */
            if (nir_intrinsic_base(intrin) == VARYING_SLOT_POS)
               continue;

            intrin = nir_instr_as_intrinsic(intrin->src[0].ssa->parent_instr);
            enum glsl_interp_mode interp = (enum glsl_interp_mode)
               nir_intrinsic_interp_mode(intrin);
            nir_intrinsic_op bary_op = intrin->intrinsic;
            enum brw_barycentric_mode bary =
               brw_barycentric_mode(interp, bary_op);

            barycentric_interp_modes |= 1 << bary;

            if (devinfo->needs_unlit_centroid_workaround &&
                bary_op == nir_intrinsic_load_barycentric_centroid)
               barycentric_interp_modes |= 1 << centroid_to_pixel(bary);
         }
      }
   }

   return barycentric_interp_modes;
}

static void
brw_compute_flat_inputs(struct brw_wm_prog_data *prog_data,
                        const nir_shader *shader)
{
   prog_data->flat_inputs = 0;

   nir_foreach_variable(var, &shader->inputs) {
      unsigned slots = glsl_count_attribute_slots(var->type, false);
      for (unsigned s = 0; s < slots; s++) {
         int input_index = prog_data->urb_setup[var->data.location + s];

         if (input_index < 0)
            continue;

         /* flat shading */
         if (var->data.interpolation == INTERP_MODE_FLAT)
            prog_data->flat_inputs |= 1 << input_index;
      }
   }
}

static uint8_t
computed_depth_mode(const nir_shader *shader)
{
   if (shader->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DEPTH)) {
      switch (shader->info.fs.depth_layout) {
      case FRAG_DEPTH_LAYOUT_NONE:
      case FRAG_DEPTH_LAYOUT_ANY:
         return BRW_PSCDEPTH_ON;
      case FRAG_DEPTH_LAYOUT_GREATER:
         return BRW_PSCDEPTH_ON_GE;
      case FRAG_DEPTH_LAYOUT_LESS:
         return BRW_PSCDEPTH_ON_LE;
      case FRAG_DEPTH_LAYOUT_UNCHANGED:
         return BRW_PSCDEPTH_OFF;
      }
   }
   return BRW_PSCDEPTH_OFF;
}

/**
 * Move load_interpolated_input with simple (payload-based) barycentric modes
 * to the top of the program so we don't emit multiple PLNs for the same input.
 *
 * This works around CSE not being able to handle non-dominating cases
 * such as:
 *
 *    if (...) {
 *       interpolate input
 *    } else {
 *       interpolate the same exact input
 *    }
 *
 * This should be replaced by global value numbering someday.
 */
static bool
move_interpolation_to_top(nir_shader *nir)
{
   bool progress = false;

   nir_foreach_function(f, nir) {
      if (!f->impl)
         continue;

      nir_block *top = nir_start_block(f->impl);
      exec_node *cursor_node = NULL;

      nir_foreach_block(block, f->impl) {
         if (block == top)
            continue;

         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (intrin->intrinsic != nir_intrinsic_load_interpolated_input)
               continue;
            nir_intrinsic_instr *bary_intrinsic =
               nir_instr_as_intrinsic(intrin->src[0].ssa->parent_instr);
            nir_intrinsic_op op = bary_intrinsic->intrinsic;

            /* Leave interpolateAtSample/Offset() where they are. */
            if (op == nir_intrinsic_load_barycentric_at_sample ||
                op == nir_intrinsic_load_barycentric_at_offset)
               continue;

            nir_instr *move[3] = {
               &bary_intrinsic->instr,
               intrin->src[1].ssa->parent_instr,
               instr
            };

            for (unsigned i = 0; i < ARRAY_SIZE(move); i++) {
               if (move[i]->block != top) {
                  move[i]->block = top;
                  exec_node_remove(&move[i]->node);
                  if (cursor_node) {
                     exec_node_insert_after(cursor_node, &move[i]->node);
                  } else {
                     exec_list_push_head(&top->instr_list, &move[i]->node);
                  }
                  cursor_node = &move[i]->node;
                  progress = true;
               }
            }
         }
      }
      nir_metadata_preserve(f->impl, (nir_metadata)
                            ((unsigned) nir_metadata_block_index |
                             (unsigned) nir_metadata_dominance));
   }

   return progress;
}

/**
 * Demote per-sample barycentric intrinsics to centroid.
 *
 * Useful when rendering to a non-multisampled buffer.
 */
static bool
demote_sample_qualifiers(nir_shader *nir)
{
   bool progress = true;

   nir_foreach_function(f, nir) {
      if (!f->impl)
         continue;

      nir_builder b;
      nir_builder_init(&b, f->impl);

      nir_foreach_block(block, f->impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(instr);
            if (intrin->intrinsic != nir_intrinsic_load_barycentric_sample &&
                intrin->intrinsic != nir_intrinsic_load_barycentric_at_sample)
               continue;

            b.cursor = nir_before_instr(instr);
            nir_ssa_def *centroid =
               nir_load_barycentric(&b, nir_intrinsic_load_barycentric_centroid,
                                    nir_intrinsic_interp_mode(intrin));
            nir_ssa_def_rewrite_uses(&intrin->dest.ssa,
                                     nir_src_for_ssa(centroid));
            nir_instr_remove(instr);
            progress = true;
         }
      }

      nir_metadata_preserve(f->impl, (nir_metadata)
                            ((unsigned) nir_metadata_block_index |
                             (unsigned) nir_metadata_dominance));
   }

   return progress;
}

/**
 * Pre-gen6, the register file of the EUs was shared between threads,
 * and each thread used some subset allocated on a 16-register block
 * granularity.  The unit states wanted these block counts.
 */
static inline int
brw_register_blocks(int reg_count)
{
   return ALIGN(reg_count, 16) / 16 - 1;
}

const unsigned *
brw_compile_fs(const struct brw_compiler *compiler, void *log_data,
               void *mem_ctx,
               const struct brw_wm_prog_key *key,
               struct brw_wm_prog_data *prog_data,
               nir_shader *shader,
               struct gl_program *prog,
               int shader_time_index8, int shader_time_index16,
               int shader_time_index32, bool allow_spilling,
               bool use_rep_send, struct brw_vue_map *vue_map,
               char **error_str)
{
   const struct gen_device_info *devinfo = compiler->devinfo;

   shader = brw_nir_apply_sampler_key(shader, compiler, &key->tex, true);
   brw_nir_lower_fs_inputs(shader, devinfo, key);
   brw_nir_lower_fs_outputs(shader);

   if (devinfo->gen < 6) {
      brw_setup_vue_interpolation(vue_map, shader, prog_data, devinfo);
   }

   if (!key->multisample_fbo)
      NIR_PASS_V(shader, demote_sample_qualifiers);
   NIR_PASS_V(shader, move_interpolation_to_top);
   shader = brw_postprocess_nir(shader, compiler, true);

   /* key->alpha_test_func means simulating alpha testing via discards,
    * so the shader definitely kills pixels.
    */
   prog_data->uses_kill = shader->info.fs.uses_discard ||
      key->alpha_test_func;
   prog_data->uses_omask = key->multisample_fbo &&
      shader->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK);
   prog_data->computed_depth_mode = computed_depth_mode(shader);
   prog_data->computed_stencil =
      shader->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_STENCIL);

   prog_data->persample_dispatch =
      key->multisample_fbo &&
      (key->persample_interp ||
       (shader->info.system_values_read & (SYSTEM_BIT_SAMPLE_ID |
                                            SYSTEM_BIT_SAMPLE_POS)) ||
       shader->info.fs.uses_sample_qualifier ||
       shader->info.outputs_read);

   prog_data->has_render_target_reads = shader->info.outputs_read != 0ull;

   prog_data->early_fragment_tests = shader->info.fs.early_fragment_tests;
   prog_data->post_depth_coverage = shader->info.fs.post_depth_coverage;
   prog_data->inner_coverage = shader->info.fs.inner_coverage;

   prog_data->barycentric_interp_modes =
      brw_compute_barycentric_interp_modes(compiler->devinfo, shader);

   cfg_t *simd8_cfg = NULL, *simd16_cfg = NULL, *simd32_cfg = NULL;

   fs_visitor v8(compiler, log_data, mem_ctx, key,
                 &prog_data->base, prog, shader, 8,
                 shader_time_index8);
   if (!v8.run_fs(allow_spilling, false /* do_rep_send */)) {
      if (error_str)
         *error_str = ralloc_strdup(mem_ctx, v8.fail_msg);

      return NULL;
   } else if (likely(!(INTEL_DEBUG & DEBUG_NO8))) {
      simd8_cfg = v8.cfg;
      prog_data->base.dispatch_grf_start_reg = v8.payload.num_regs;
      prog_data->reg_blocks_8 = brw_register_blocks(v8.grf_used);
   }

   if (v8.max_dispatch_width >= 16 &&
       likely(!(INTEL_DEBUG & DEBUG_NO16) || use_rep_send)) {
      /* Try a SIMD16 compile */
      fs_visitor v16(compiler, log_data, mem_ctx, key,
                     &prog_data->base, prog, shader, 16,
                     shader_time_index16);
      v16.import_uniforms(&v8);
      if (!v16.run_fs(allow_spilling, use_rep_send)) {
         compiler->shader_perf_log(log_data,
                                   "SIMD16 shader failed to compile: %s",
                                   v16.fail_msg);
      } else {
         simd16_cfg = v16.cfg;
         prog_data->dispatch_grf_start_reg_16 = v16.payload.num_regs;
         prog_data->reg_blocks_16 = brw_register_blocks(v16.grf_used);
      }
   }

   /* Currently, the compiler only supports SIMD32 on SNB+ */
   if (v8.max_dispatch_width >= 32 && !use_rep_send &&
       compiler->devinfo->gen >= 6 &&
       unlikely(INTEL_DEBUG & DEBUG_DO32)) {
      /* Try a SIMD32 compile */
      fs_visitor v32(compiler, log_data, mem_ctx, key,
                     &prog_data->base, prog, shader, 32,
                     shader_time_index32);
      v32.import_uniforms(&v8);
      if (!v32.run_fs(allow_spilling, false)) {
         compiler->shader_perf_log(log_data,
                                   "SIMD32 shader failed to compile: %s",
                                   v32.fail_msg);
      } else {
         simd32_cfg = v32.cfg;
         prog_data->dispatch_grf_start_reg_32 = v32.payload.num_regs;
         prog_data->reg_blocks_32 = brw_register_blocks(v32.grf_used);
      }
   }

   /* When the caller requests a repclear shader, they want SIMD16-only */
   if (use_rep_send)
      simd8_cfg = NULL;

   /* Prior to Iron Lake, the PS had a single shader offset with a jump table
    * at the top to select the shader.  We've never implemented that.
    * Instead, we just give them exactly one shader and we pick the widest one
    * available.
    */
   if (compiler->devinfo->gen < 5) {
      if (simd32_cfg || simd16_cfg)
         simd8_cfg = NULL;
      if (simd32_cfg)
         simd16_cfg = NULL;
   }

   /* If computed depth is enabled SNB only allows SIMD8. */
   if (compiler->devinfo->gen == 6 &&
       prog_data->computed_depth_mode != BRW_PSCDEPTH_OFF)
      assert(simd16_cfg == NULL && simd32_cfg == NULL);

   if (compiler->devinfo->gen <= 5 && !simd8_cfg) {
      /* Iron lake and earlier only have one Dispatch GRF start field.  Make
       * the data available in the base prog data struct for convenience.
       */
      if (simd16_cfg) {
         prog_data->base.dispatch_grf_start_reg =
            prog_data->dispatch_grf_start_reg_16;
      } else if (simd32_cfg) {
         prog_data->base.dispatch_grf_start_reg =
            prog_data->dispatch_grf_start_reg_32;
      }
   }

   if (prog_data->persample_dispatch) {
      /* Starting with SandyBridge (where we first get MSAA), the different
       * pixel dispatch combinations are grouped into classifications A
       * through F (SNB PRM Vol. 2 Part 1 Section 7.7.1).  On all hardware
       * generations, the only configurations supporting persample dispatch
       * are are this in which only one dispatch width is enabled.
       */
      if (simd32_cfg || simd16_cfg)
         simd8_cfg = NULL;
      if (simd32_cfg)
         simd16_cfg = NULL;
   }

   /* We have to compute the flat inputs after the visitor is finished running
    * because it relies on prog_data->urb_setup which is computed in
    * fs_visitor::calculate_urb_setup().
    */
   brw_compute_flat_inputs(prog_data, shader);

   fs_generator g(compiler, log_data, mem_ctx, &prog_data->base,
                  v8.promoted_constants, v8.runtime_check_aads_emit,
                  MESA_SHADER_FRAGMENT);

   if (unlikely(INTEL_DEBUG & DEBUG_WM)) {
      g.enable_debug(ralloc_asprintf(mem_ctx, "%s fragment shader %s",
                                     shader->info.label ?
                                        shader->info.label : "unnamed",
                                     shader->info.name));
   }

   if (simd8_cfg) {
      prog_data->dispatch_8 = true;
      g.generate_code(simd8_cfg, 8);
   }

   if (simd16_cfg) {
      prog_data->dispatch_16 = true;
      prog_data->prog_offset_16 = g.generate_code(simd16_cfg, 16);
   }

   if (simd32_cfg) {
      prog_data->dispatch_32 = true;
      prog_data->prog_offset_32 = g.generate_code(simd32_cfg, 32);
   }

   return g.get_assembly();
}

fs_reg *
fs_visitor::emit_cs_work_group_id_setup()
{
   assert(stage == MESA_SHADER_COMPUTE);

   fs_reg *reg = new(this->mem_ctx) fs_reg(vgrf(glsl_type::uvec3_type));

   struct brw_reg r0_1(retype(brw_vec1_grf(0, 1), BRW_REGISTER_TYPE_UD));
   struct brw_reg r0_6(retype(brw_vec1_grf(0, 6), BRW_REGISTER_TYPE_UD));
   struct brw_reg r0_7(retype(brw_vec1_grf(0, 7), BRW_REGISTER_TYPE_UD));

   bld.MOV(*reg, r0_1);
   bld.MOV(offset(*reg, bld, 1), r0_6);
   bld.MOV(offset(*reg, bld, 2), r0_7);

   return reg;
}

static void
fill_push_const_block_info(struct brw_push_const_block *block, unsigned dwords)
{
   block->dwords = dwords;
   block->regs = DIV_ROUND_UP(dwords, 8);
   block->size = block->regs * 32;
}

static void
cs_fill_push_const_info(const struct gen_device_info *devinfo,
                        struct brw_cs_prog_data *cs_prog_data)
{
   const struct brw_stage_prog_data *prog_data = &cs_prog_data->base;
   int subgroup_id_index = get_subgroup_id_param_index(prog_data);
   bool cross_thread_supported = devinfo->gen > 7 || devinfo->is_haswell;

   /* The thread ID should be stored in the last param dword */
   assert(subgroup_id_index == -1 ||
          subgroup_id_index == (int)prog_data->nr_params - 1);

   unsigned cross_thread_dwords, per_thread_dwords;
   if (!cross_thread_supported) {
      cross_thread_dwords = 0u;
      per_thread_dwords = prog_data->nr_params;
   } else if (subgroup_id_index >= 0) {
      /* Fill all but the last register with cross-thread payload */
      cross_thread_dwords = 8 * (subgroup_id_index / 8);
      per_thread_dwords = prog_data->nr_params - cross_thread_dwords;
      assert(per_thread_dwords > 0 && per_thread_dwords <= 8);
   } else {
      /* Fill all data using cross-thread payload */
      cross_thread_dwords = prog_data->nr_params;
      per_thread_dwords = 0u;
   }

   fill_push_const_block_info(&cs_prog_data->push.cross_thread, cross_thread_dwords);
   fill_push_const_block_info(&cs_prog_data->push.per_thread, per_thread_dwords);

   unsigned total_dwords =
      (cs_prog_data->push.per_thread.size * cs_prog_data->threads +
       cs_prog_data->push.cross_thread.size) / 4;
   fill_push_const_block_info(&cs_prog_data->push.total, total_dwords);

   assert(cs_prog_data->push.cross_thread.dwords % 8 == 0 ||
          cs_prog_data->push.per_thread.size == 0);
   assert(cs_prog_data->push.cross_thread.dwords +
          cs_prog_data->push.per_thread.dwords ==
             prog_data->nr_params);
}

static void
cs_set_simd_size(struct brw_cs_prog_data *cs_prog_data, unsigned size)
{
   cs_prog_data->simd_size = size;
   unsigned group_size = cs_prog_data->local_size[0] *
      cs_prog_data->local_size[1] * cs_prog_data->local_size[2];
   cs_prog_data->threads = (group_size + size - 1) / size;
}

static nir_shader *
compile_cs_to_nir(const struct brw_compiler *compiler,
                  void *mem_ctx,
                  const struct brw_cs_prog_key *key,
                  const nir_shader *src_shader,
                  unsigned dispatch_width)
{
   nir_shader *shader = nir_shader_clone(mem_ctx, src_shader);
   shader = brw_nir_apply_sampler_key(shader, compiler, &key->tex, true);
   brw_nir_lower_cs_intrinsics(shader, dispatch_width);
   return brw_postprocess_nir(shader, compiler, true);
}

const unsigned *
brw_compile_cs(const struct brw_compiler *compiler, void *log_data,
               void *mem_ctx,
               const struct brw_cs_prog_key *key,
               struct brw_cs_prog_data *prog_data,
               const nir_shader *src_shader,
               int shader_time_index,
               char **error_str)
{
   prog_data->local_size[0] = src_shader->info.cs.local_size[0];
   prog_data->local_size[1] = src_shader->info.cs.local_size[1];
   prog_data->local_size[2] = src_shader->info.cs.local_size[2];
   unsigned local_workgroup_size =
      src_shader->info.cs.local_size[0] * src_shader->info.cs.local_size[1] *
      src_shader->info.cs.local_size[2];

   unsigned min_dispatch_width =
      DIV_ROUND_UP(local_workgroup_size, compiler->devinfo->max_cs_threads);
   min_dispatch_width = MAX2(8, min_dispatch_width);
   min_dispatch_width = util_next_power_of_two(min_dispatch_width);
   assert(min_dispatch_width <= 32);

   fs_visitor *v8 = NULL, *v16 = NULL, *v32 = NULL;
   cfg_t *cfg = NULL;
   const char *fail_msg = NULL;
   unsigned promoted_constants = 0;

   /* Now the main event: Visit the shader IR and generate our CS IR for it.
    */
   if (min_dispatch_width <= 8) {
      nir_shader *nir8 = compile_cs_to_nir(compiler, mem_ctx, key,
                                           src_shader, 8);
      v8 = new fs_visitor(compiler, log_data, mem_ctx, key, &prog_data->base,
                          NULL, /* Never used in core profile */
                          nir8, 8, shader_time_index);
      if (!v8->run_cs(min_dispatch_width)) {
         fail_msg = v8->fail_msg;
      } else {
         /* We should always be able to do SIMD32 for compute shaders */
         assert(v8->max_dispatch_width >= 32);

         cfg = v8->cfg;
         cs_set_simd_size(prog_data, 8);
         cs_fill_push_const_info(compiler->devinfo, prog_data);
         promoted_constants = v8->promoted_constants;
      }
   }

   if (likely(!(INTEL_DEBUG & DEBUG_NO16)) &&
       !fail_msg && min_dispatch_width <= 16) {
      /* Try a SIMD16 compile */
      nir_shader *nir16 = compile_cs_to_nir(compiler, mem_ctx, key,
                                            src_shader, 16);
      v16 = new fs_visitor(compiler, log_data, mem_ctx, key, &prog_data->base,
                           NULL, /* Never used in core profile */
                           nir16, 16, shader_time_index);
      if (v8)
         v16->import_uniforms(v8);

      if (!v16->run_cs(min_dispatch_width)) {
         compiler->shader_perf_log(log_data,
                                   "SIMD16 shader failed to compile: %s",
                                   v16->fail_msg);
         if (!cfg) {
            fail_msg =
               "Couldn't generate SIMD16 program and not "
               "enough threads for SIMD8";
         }
      } else {
         /* We should always be able to do SIMD32 for compute shaders */
         assert(v16->max_dispatch_width >= 32);

         cfg = v16->cfg;
         cs_set_simd_size(prog_data, 16);
         cs_fill_push_const_info(compiler->devinfo, prog_data);
         promoted_constants = v16->promoted_constants;
      }
   }

   /* We should always be able to do SIMD32 for compute shaders */
   assert(!v16 || v16->max_dispatch_width >= 32);

   if (!fail_msg && (min_dispatch_width > 16 || (INTEL_DEBUG & DEBUG_DO32))) {
      /* Try a SIMD32 compile */
      nir_shader *nir32 = compile_cs_to_nir(compiler, mem_ctx, key,
                                            src_shader, 32);
      v32 = new fs_visitor(compiler, log_data, mem_ctx, key, &prog_data->base,
                           NULL, /* Never used in core profile */
                           nir32, 32, shader_time_index);
      if (v8)
         v32->import_uniforms(v8);
      else if (v16)
         v32->import_uniforms(v16);

      if (!v32->run_cs(min_dispatch_width)) {
         compiler->shader_perf_log(log_data,
                                   "SIMD32 shader failed to compile: %s",
                                   v16->fail_msg);
         if (!cfg) {
            fail_msg =
               "Couldn't generate SIMD32 program and not "
               "enough threads for SIMD16";
         }
      } else {
         cfg = v32->cfg;
         cs_set_simd_size(prog_data, 32);
         cs_fill_push_const_info(compiler->devinfo, prog_data);
         promoted_constants = v32->promoted_constants;
      }
   }

   const unsigned *ret = NULL;
   if (unlikely(cfg == NULL)) {
      assert(fail_msg);
      if (error_str)
         *error_str = ralloc_strdup(mem_ctx, fail_msg);
   } else {
      fs_generator g(compiler, log_data, mem_ctx, &prog_data->base,
                     promoted_constants, false, MESA_SHADER_COMPUTE);
      if (INTEL_DEBUG & DEBUG_CS) {
         char *name = ralloc_asprintf(mem_ctx, "%s compute shader %s",
                                      src_shader->info.label ?
                                         src_shader->info.label : "unnamed",
                                      src_shader->info.name);
         g.enable_debug(name);
      }

      g.generate_code(cfg, prog_data->simd_size);

      ret = g.get_assembly();
   }

   delete v8;
   delete v16;
   delete v32;

   return ret;
}

/**
 * Test the dispatch mask packing assumptions of
 * brw_stage_has_packed_dispatch().  Call this from e.g. the top of
 * fs_visitor::emit_nir_code() to cause a GPU hang if any shader invocation is
 * executed with an unexpected dispatch mask.
 */
static UNUSED void
brw_fs_test_dispatch_packing(const fs_builder &bld)
{
   const gl_shader_stage stage = bld.shader->stage;

   if (brw_stage_has_packed_dispatch(bld.shader->devinfo, stage,
                                     bld.shader->stage_prog_data)) {
      const fs_builder ubld = bld.exec_all().group(1, 0);
      const fs_reg tmp = component(bld.vgrf(BRW_REGISTER_TYPE_UD), 0);
      const fs_reg mask = (stage == MESA_SHADER_FRAGMENT ? brw_vmask_reg() :
                           brw_dmask_reg());

      ubld.ADD(tmp, mask, brw_imm_ud(1));
      ubld.AND(tmp, mask, tmp);

      /* This will loop forever if the dispatch mask doesn't have the expected
       * form '2^n-1', in which case tmp will be non-zero.
       */
      bld.emit(BRW_OPCODE_DO);
      bld.CMP(bld.null_reg_ud(), tmp, brw_imm_ud(0), BRW_CONDITIONAL_NZ);
      set_predicate(BRW_PREDICATE_NORMAL, bld.emit(BRW_OPCODE_WHILE));
   }
}
