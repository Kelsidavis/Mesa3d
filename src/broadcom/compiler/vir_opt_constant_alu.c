/*
 * Copyright © 2021 Raspberry Pi Ltd
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

/**
 * @file v3d_opt_constant_alu.c
 *
 * Identified sequences of ALU instructions that operate on constant operands
 * and reduces them to a uniform load.
 *
 * This is useful, for example, to optimize the result of removing leading
 * ldunifa instructions in the DCE pass, which can leave a series of constant
 * additions that increment the unifa address by 4 for each leading ldunif
 * removed. It helps turn this:
 *
 * nop t1; ldunif (0x00000004 / 0.000000)
 * nop t2; ldunif (0x00000004 / 0.000000)
 * add t3, t1, t2
 *
 * into:
 *
 * nop t1; ldunif (0x00000004 / 0.000000)
 * nop t2; ldunif (0x00000004 / 0.000000)
 * nop t4; ldunif (0x00000008 / 0.000000)
 * mov t3, t4
 *
 * For best results we want to run copy propagation in between this and
 * the combine constants pass: every time we manage to convert an alu to
 * a uniform load, we move the uniform to the original alu destination. By
 * running copy propagation immediately after we can reuse the uniform as
 * source in more follow-up alu instructions, making them constant and allowing
 * this pass to continue making progress. However, if we run the small
 * immediates optimization before that, that pass can convert some of the movs
 * to use small immediates instead of the uniforms and prevent us from making
 * the best of this pass, as small immediates don't get copy propagated.
 */

#include "v3d_compiler.h"

#include "util/half_float.h"
#include "util/u_math.h"

static bool
opt_constant_add(struct v3d_compile *c, struct qinst *inst, union fi *values)
{
        struct qreg unif = { };
        switch (inst->qpu.alu.add.op) {
        case V3D_QPU_A_ADD:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, values[0].ui + values[1].ui);
                break;

        case V3D_QPU_A_SUB:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, values[0].ui - values[1].ui);
                break;

        case V3D_QPU_A_SHL:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, values[0].ui << (values[1].ui & 0x1f));
                break;

        case V3D_QPU_A_SHR:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, values[0].ui >> (values[1].ui & 0x1f));
                break;

        case V3D_QPU_A_ASR:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, (int32_t)values[0].ui >> (values[1].ui & 0x1f));
                break;

        case V3D_QPU_A_ROR: {
                uint32_t shift = values[1].ui & 0x1f;
                uint32_t result = shift ? ((values[0].ui >> shift) |
                                           (values[0].ui << (32 - shift)))
                                        : values[0].ui;
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, result);
                break;
        }

        case V3D_QPU_A_AND:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, values[0].ui & values[1].ui);
                break;

        case V3D_QPU_A_OR:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, values[0].ui | values[1].ui);
                break;

        case V3D_QPU_A_XOR:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, values[0].ui ^ values[1].ui);
                break;

        case V3D_QPU_A_MIN:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, MIN2((int32_t)values[0].ui,
                                              (int32_t)values[1].ui));
                break;

        case V3D_QPU_A_MAX:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, MAX2((int32_t)values[0].ui,
                                              (int32_t)values[1].ui));
                break;

        case V3D_QPU_A_UMIN:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, MIN2(values[0].ui, values[1].ui));
                break;

        case V3D_QPU_A_UMAX:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, MAX2(values[0].ui, values[1].ui));
                break;

        case V3D_QPU_A_FADD:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_f(c, values[0].f + values[1].f);
                break;

        case V3D_QPU_A_FSUB:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_f(c, values[0].f - values[1].f);
                break;

        case V3D_QPU_A_FMIN:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_f(c, fminf(values[0].f, values[1].f));
                break;

        case V3D_QPU_A_FMAX:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_f(c, fmaxf(values[0].f, values[1].f));
                break;

        case V3D_QPU_A_VFPACK: {
                assert(inst->qpu.alu.add.output_pack == V3D_QPU_PACK_NONE);

                const uint32_t packed =
                        (((uint32_t)_mesa_float_to_half(values[1].f)) << 16) |
                        _mesa_float_to_half(values[0].f);

                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, packed);
                break;
        }

        /* Vectorized half-precision operations (f16x2) */
        case V3D_QPU_A_VADD: {
                /* Unpack f16x2 values, add, and repack */
                float a_lo = _mesa_half_to_float(values[0].ui & 0xffff);
                float a_hi = _mesa_half_to_float(values[0].ui >> 16);
                float b_lo = _mesa_half_to_float(values[1].ui & 0xffff);
                float b_hi = _mesa_half_to_float(values[1].ui >> 16);
                uint32_t result = ((uint32_t)_mesa_float_to_half(a_hi + b_hi) << 16) |
                                  _mesa_float_to_half(a_lo + b_lo);
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, result);
                break;
        }

        case V3D_QPU_A_VSUB: {
                float a_lo = _mesa_half_to_float(values[0].ui & 0xffff);
                float a_hi = _mesa_half_to_float(values[0].ui >> 16);
                float b_lo = _mesa_half_to_float(values[1].ui & 0xffff);
                float b_hi = _mesa_half_to_float(values[1].ui >> 16);
                uint32_t result = ((uint32_t)_mesa_float_to_half(a_hi - b_hi) << 16) |
                                  _mesa_float_to_half(a_lo - b_lo);
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, result);
                break;
        }

        case V3D_QPU_A_VFMIN: {
                float a_lo = _mesa_half_to_float(values[0].ui & 0xffff);
                float a_hi = _mesa_half_to_float(values[0].ui >> 16);
                float b_lo = _mesa_half_to_float(values[1].ui & 0xffff);
                float b_hi = _mesa_half_to_float(values[1].ui >> 16);
                uint32_t result = ((uint32_t)_mesa_float_to_half(fminf(a_hi, b_hi)) << 16) |
                                  _mesa_float_to_half(fminf(a_lo, b_lo));
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, result);
                break;
        }

        case V3D_QPU_A_VFMAX: {
                float a_lo = _mesa_half_to_float(values[0].ui & 0xffff);
                float a_hi = _mesa_half_to_float(values[0].ui >> 16);
                float b_lo = _mesa_half_to_float(values[1].ui & 0xffff);
                float b_hi = _mesa_half_to_float(values[1].ui >> 16);
                uint32_t result = ((uint32_t)_mesa_float_to_half(fmaxf(a_hi, b_hi)) << 16) |
                                  _mesa_float_to_half(fmaxf(a_lo, b_lo));
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, result);
                break;
        }

        /* Unary operations */
        case V3D_QPU_A_NOT:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, ~values[0].ui);
                break;

        case V3D_QPU_A_NEG:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, -values[0].ui);
                break;

        case V3D_QPU_A_FROUND:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_f(c, roundf(values[0].f));
                break;

        case V3D_QPU_A_FTRUNC:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_f(c, truncf(values[0].f));
                break;

        case V3D_QPU_A_FFLOOR:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_f(c, floorf(values[0].f));
                break;

        case V3D_QPU_A_FCEIL:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_f(c, ceilf(values[0].f));
                break;

        case V3D_QPU_A_FTOIN:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, (int32_t)roundf(values[0].f));
                break;

        case V3D_QPU_A_FTOIZ:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, (int32_t)truncf(values[0].f));
                break;

        case V3D_QPU_A_FTOUZ:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, (uint32_t)truncf(values[0].f));
                break;

        case V3D_QPU_A_ITOF:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_f(c, (float)(int32_t)values[0].ui);
                break;

        case V3D_QPU_A_UTOF:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_f(c, (float)values[0].ui);
                break;

        case V3D_QPU_A_CLZ:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, values[0].ui ? __builtin_clz(values[0].ui) : 32);
                break;

        /* Packing operations */
        case V3D_QPU_A_VPACK: {
                /* Pack low 16 bits of each input: result = (a & 0xFFFF) | ((b & 0xFFFF) << 16) */
                uint32_t result = (values[0].ui & 0xffff) | ((values[1].ui & 0xffff) << 16);
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, result);
                break;
        }

        case V3D_QPU_A_V8PACK: {
                /* Pack 4x8 from two inputs containing 2x16 each:
                 * Takes low 8 bits from each 16-bit half of the inputs.
                 * src0 = [a1:a0], src1 = [b1:b0] (each 16-bit)
                 * result = [b1_lo8 : b0_lo8 : a1_lo8 : a0_lo8]
                 */
                uint8_t a0 = (values[0].ui >> 0) & 0xff;
                uint8_t a1 = (values[0].ui >> 16) & 0xff;
                uint8_t b0 = (values[1].ui >> 0) & 0xff;
                uint8_t b1 = (values[1].ui >> 16) & 0xff;
                uint32_t result = a0 | (a1 << 8) | (b0 << 16) | (b1 << 24);
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, result);
                break;
        }

        /* SFU operations - transcendental functions */
        case V3D_QPU_A_RECIP:
                /* 1.0 / x */
                if (values[0].f != 0.0f) {
                        c->cursor = vir_after_inst(inst);
                        unif = vir_uniform_f(c, 1.0f / values[0].f);
                } else {
                        return false;
                }
                break;

        case V3D_QPU_A_RSQRT:
                /* 1.0 / sqrt(x) */
                if (values[0].f > 0.0f) {
                        c->cursor = vir_after_inst(inst);
                        unif = vir_uniform_f(c, 1.0f / sqrtf(values[0].f));
                } else {
                        return false;
                }
                break;

        case V3D_QPU_A_EXP:
                /* 2^x (base 2 exponential) */
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_f(c, exp2f(values[0].f));
                break;

        case V3D_QPU_A_LOG:
                /* log2(x) (base 2 logarithm) */
                if (values[0].f > 0.0f) {
                        c->cursor = vir_after_inst(inst);
                        unif = vir_uniform_f(c, log2f(values[0].f));
                } else {
                        return false;
                }
                break;

        case V3D_QPU_A_SIN:
                /* sin(x * 2 * pi) - input is in revolutions, not radians */
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_f(c, sinf(values[0].f * 2.0f * M_PI));
                break;

        default:
                return false;
        }

        /* Remove the original ALU instruction and replace it with a uniform
         * load. If the original instruction loaded an implicit uniform we
         * need to replicate that in the new instruction.
         */
        struct qreg dst = inst->dst;
        struct qinst *mov = vir_MOV_dest(c, dst, unif);
        mov->uniform = inst->uniform;
        vir_remove_instruction(c, inst);
        if (dst.file == QFILE_TEMP)
                c->defs[dst.index] = mov;
        return true;
}

static bool
opt_constant_mul(struct v3d_compile *c, struct qinst *inst, union fi *values)
{
        struct qreg unif = { };
        switch (inst->qpu.alu.mul.op) {
        case V3D_QPU_M_ADD:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, values[0].ui + values[1].ui);
                break;

        case V3D_QPU_M_SUB:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, values[0].ui - values[1].ui);
                break;

        case V3D_QPU_M_UMUL24:
                /* UMUL24 multiplies the low 24 bits */
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, (values[0].ui & 0xffffff) *
                                         (values[1].ui & 0xffffff));
                break;

        case V3D_QPU_M_SMUL24: {
                /* SMUL24 is signed 24-bit multiply */
                int32_t a = (values[0].ui << 8) >> 8;  /* sign-extend 24-bit */
                int32_t b = (values[1].ui << 8) >> 8;
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, a * b);
                break;
        }

        case V3D_QPU_M_FMUL:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_f(c, values[0].f * values[1].f);
                break;

        case V3D_QPU_M_VFMUL: {
                /* Vectorized f16x2 multiplication */
                float a_lo = _mesa_half_to_float(values[0].ui & 0xffff);
                float a_hi = _mesa_half_to_float(values[0].ui >> 16);
                float b_lo = _mesa_half_to_float(values[1].ui & 0xffff);
                float b_hi = _mesa_half_to_float(values[1].ui >> 16);
                uint32_t result = ((uint32_t)_mesa_float_to_half(a_hi * b_hi) << 16) |
                                  _mesa_float_to_half(a_lo * b_lo);
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, result);
                break;
        }

        /* Unary operations - MOV just copies the constant value */
        case V3D_QPU_M_MOV:
        case V3D_QPU_M_FMOV:
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, values[0].ui);
                break;

        /* Normalization operations */
        case V3D_QPU_M_FTOUNORM16: {
                /* Float [0.0, 1.0] to u16 [0, 65535] */
                float clamped = CLAMP(values[0].f, 0.0f, 1.0f);
                uint16_t result = (uint16_t)(clamped * 65535.0f + 0.5f);
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, result);
                break;
        }

        case V3D_QPU_M_FTOSNORM16: {
                /* Float [-1.0, 1.0] to s16 [-32767, 32767] */
                float clamped = CLAMP(values[0].f, -1.0f, 1.0f);
                int16_t result = (int16_t)(clamped * 32767.0f + (clamped >= 0 ? 0.5f : -0.5f));
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, (uint32_t)(uint16_t)result);
                break;
        }

        case V3D_QPU_M_VFTOUNORM8: {
                /* f16x2 to u8x2: convert each f16 [0,1] to u8 [0,255] */
                float lo = _mesa_half_to_float(values[0].ui & 0xffff);
                float hi = _mesa_half_to_float(values[0].ui >> 16);
                uint8_t lo_u8 = (uint8_t)(CLAMP(lo, 0.0f, 1.0f) * 255.0f + 0.5f);
                uint8_t hi_u8 = (uint8_t)(CLAMP(hi, 0.0f, 1.0f) * 255.0f + 0.5f);
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, lo_u8 | (hi_u8 << 8));
                break;
        }

        case V3D_QPU_M_VFTOSNORM8: {
                /* f16x2 to s8x2: convert each f16 [-1,1] to s8 [-127,127] */
                float lo = _mesa_half_to_float(values[0].ui & 0xffff);
                float hi = _mesa_half_to_float(values[0].ui >> 16);
                int8_t lo_s8 = (int8_t)(CLAMP(lo, -1.0f, 1.0f) * 127.0f +
                                        (lo >= 0 ? 0.5f : -0.5f));
                int8_t hi_s8 = (int8_t)(CLAMP(hi, -1.0f, 1.0f) * 127.0f +
                                        (hi >= 0 ? 0.5f : -0.5f));
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_ui(c, (uint8_t)lo_s8 | ((uint8_t)hi_s8 << 8));
                break;
        }

        case V3D_QPU_M_FUNPACKUNORMLO: {
                /* Unpack low byte as unorm to float: u8 [0,255] to [0.0, 1.0] */
                uint8_t val = values[0].ui & 0xff;
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_f(c, (float)val / 255.0f);
                break;
        }

        case V3D_QPU_M_FUNPACKUNORMHI: {
                /* Unpack high byte (byte 1) as unorm to float */
                uint8_t val = (values[0].ui >> 8) & 0xff;
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_f(c, (float)val / 255.0f);
                break;
        }

        case V3D_QPU_M_FUNPACKSNORMLO: {
                /* Unpack low byte as snorm to float: s8 [-127,127] to [-1.0, 1.0] */
                int8_t val = (int8_t)(values[0].ui & 0xff);
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_f(c, CLAMP((float)val / 127.0f, -1.0f, 1.0f));
                break;
        }

        case V3D_QPU_M_FUNPACKSNORMHI: {
                /* Unpack high byte (byte 1) as snorm to float */
                int8_t val = (int8_t)((values[0].ui >> 8) & 0xff);
                c->cursor = vir_after_inst(inst);
                unif = vir_uniform_f(c, CLAMP((float)val / 127.0f, -1.0f, 1.0f));
                break;
        }

        default:
                return false;
        }

        /* Remove the original ALU instruction and replace it with a uniform
         * load. If the original instruction loaded an implicit uniform we
         * need to replicate that in the new instruction.
         */
        struct qreg dst = inst->dst;
        struct qinst *mov = vir_MOV_dest(c, dst, unif);
        mov->uniform = inst->uniform;
        vir_remove_instruction(c, inst);
        if (dst.file == QFILE_TEMP)
                c->defs[dst.index] = mov;
        return true;
}

static bool
try_opt_constant_alu(struct v3d_compile *c, struct qinst *inst)
{
        if(inst->qpu.type != V3D_QPU_INSTR_TYPE_ALU)
                return false;

        /* If the instruction does anything other than writing the result
         * directly to the destination, skip.
         */
        if (inst->qpu.alu.add.output_pack != V3D_QPU_PACK_NONE ||
            inst->qpu.alu.mul.output_pack != V3D_QPU_PACK_NONE) {
                return false;
        }

        if (inst->qpu.flags.ac != V3D_QPU_COND_NONE ||
            inst->qpu.flags.mc != V3D_QPU_COND_NONE) {
                return false;
        }

        assert(vir_get_nsrc(inst) <= 2);
        union fi values[2];
        for (int i = 0; i < vir_get_nsrc(inst); i++) {
                if (inst->src[i].file == QFILE_SMALL_IMM &&
                    v3d_qpu_small_imm_unpack(c->devinfo,
                                             inst->qpu.raddr_b,
                                             &values[i].ui)) {
                        continue;
                }

                if (inst->src[i].file == QFILE_TEMP) {
                        struct qinst *def = c->defs[inst->src[i].index];
                        if (!def)
                                return false;

                        if ((def->qpu.sig.ldunif || def->qpu.sig.ldunifrf) &&
                            c->uniform_contents[def->uniform] == QUNIFORM_CONSTANT) {
                                values[i].ui = c->uniform_data[def->uniform];
                                continue;
                        }
                }

                return false;
        }

        if (vir_is_add(inst))
                return opt_constant_add(c, inst, values);

        if (vir_is_mul(inst))
                return opt_constant_mul(c, inst, values);

        return false;
}

bool
vir_opt_constant_alu(struct v3d_compile *c)
{
        bool progress = false;
        vir_for_each_block(block, c) {
                c->cur_block = block;
                vir_for_each_inst_safe(inst, block) {
                        progress = try_opt_constant_alu(c, inst) || progress;
                }
        }

        return progress;
}
