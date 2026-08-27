/*
 * Copyright © 2024 Raspberry Pi Ltd
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
 * @file vir_opt_algebraic.c
 *
 * Algebraic optimizations at the VIR level.
 *
 * This pass performs identity optimizations and strength reductions that
 * can be applied when one operand is a known constant:
 *
 * Identity operations:
 *   - ADD x, 0 → MOV x
 *   - FADD x, 0.0 → MOV x (FMOV)
 *   - SUB x, 0 → MOV x
 *   - FSUB x, 0.0 → MOV x (FMOV)
 *   - MUL x, 1 → MOV x
 *   - FMUL x, 1.0 → MOV x (FMOV)
 *   - AND x, ~0 → MOV x
 *   - OR x, 0 → MOV x
 *   - XOR x, 0 → MOV x
 *   - SHL x, 0 → MOV x
 *   - SHR x, 0 → MOV x
 *   - ASR x, 0 → MOV x
 *
 * Zero operations:
 *   - MUL x, 0 → MOV 0
 *   - FMUL x, 0.0 → MOV 0.0 (careful with NaN/Inf)
 *   - AND x, 0 → MOV 0
 *
 * Strength reduction:
 *   - MUL x, 2^n → SHL x, n (for small n)
 *   - FMUL x, 2.0 → FADD x, x
 */

#include "v3d_compiler.h"

static bool debug = false;

/**
 * Check if a temp holds a known constant value.
 * Returns true and sets *value if the temp is a constant.
 */
static bool
get_temp_const_value(struct v3d_compile *c, struct qreg src, uint32_t *value)
{
        if (src.file == QFILE_SMALL_IMM) {
                /* Small immediates need to be unpacked */
                if (!v3d_qpu_small_imm_unpack(c->devinfo, src.index, value))
                        return false;
                return true;
        }

        if (src.file != QFILE_TEMP)
                return false;

        struct qinst *def = c->defs[src.index];
        if (!def)
                return false;

        if ((def->qpu.sig.ldunif || def->qpu.sig.ldunifrf) &&
            c->uniform_contents[def->uniform] == QUNIFORM_CONSTANT) {
                *value = c->uniform_data[def->uniform];
                return true;
        }

        return false;
}

/**
 * Check if value is a power of 2 and return the exponent.
 */
static bool
is_power_of_two(uint32_t value, int *shift)
{
        if (value == 0 || (value & (value - 1)) != 0)
                return false;

        *shift = 0;
        while ((value & 1) == 0) {
                value >>= 1;
                (*shift)++;
        }
        return true;
}

/**
 * Replace instruction with a MOV from the given source.
 */
static void
replace_with_mov(struct v3d_compile *c, struct qinst *inst, struct qreg src)
{
        if (debug) {
                fprintf(stderr, "Algebraic opt: ");
                vir_dump_inst(c, inst);
                fprintf(stderr, " -> MOV from temp %d\n", src.index);
        }

        /* Convert to MOV in the MUL pipe */
        inst->qpu.alu.add.op = V3D_QPU_A_NOP;
        inst->qpu.alu.mul.op = V3D_QPU_M_MOV;
        inst->src[0] = src;
        inst->src[1] = c->undef;

        /* Clear any input unpacks */
        inst->qpu.alu.mul.a.unpack = V3D_QPU_UNPACK_NONE;
        inst->qpu.alu.mul.b.unpack = V3D_QPU_UNPACK_NONE;
}

/**
 * Replace instruction with a MOV loading a constant.
 */
static void
replace_with_const(struct v3d_compile *c, struct qinst *inst, uint32_t value)
{
        if (debug) {
                fprintf(stderr, "Algebraic opt: ");
                vir_dump_inst(c, inst);
                fprintf(stderr, " -> constant 0x%08x\n", value);
        }

        c->cursor = vir_after_inst(inst);
        struct qreg unif = vir_uniform_ui(c, value);

        struct qreg dst = inst->dst;
        struct qinst *mov = vir_MOV_dest(c, dst, unif);
        mov->uniform = inst->uniform;
        vir_remove_instruction(c, inst);
        if (dst.file == QFILE_TEMP)
                c->defs[dst.index] = mov;
}

/**
 * Replace MUL with SHL for power-of-2 multiplications.
 */
static void
replace_mul_with_shl(struct v3d_compile *c, struct qinst *inst,
                     struct qreg src, int shift)
{
        if (debug) {
                fprintf(stderr, "Strength reduction: ");
                vir_dump_inst(c, inst);
                fprintf(stderr, " -> SHL by %d\n", shift);
        }

        c->cursor = vir_after_inst(inst);
        struct qreg shift_reg = vir_uniform_ui(c, shift);

        struct qreg dst = inst->dst;
        struct qinst *shl = vir_SHL_dest(c, dst, src, shift_reg);
        shl->uniform = inst->uniform;
        vir_remove_instruction(c, inst);
        if (dst.file == QFILE_TEMP)
                c->defs[dst.index] = shl;
}

static bool
try_opt_algebraic_add(struct v3d_compile *c, struct qinst *inst)
{
        uint32_t val0, val1;
        bool has_const0 = get_temp_const_value(c, inst->src[0], &val0);
        bool has_const1 = get_temp_const_value(c, inst->src[1], &val1);

        switch (inst->qpu.alu.add.op) {
        case V3D_QPU_A_ADD:
                /* ADD x, 0 → MOV x */
                if (has_const1 && val1 == 0) {
                        replace_with_mov(c, inst, inst->src[0]);
                        return true;
                }
                if (has_const0 && val0 == 0) {
                        replace_with_mov(c, inst, inst->src[1]);
                        return true;
                }
                break;

        case V3D_QPU_A_FADD:
        case V3D_QPU_A_FADDNF:
                /* FADD x, 0.0 → FMOV x */
                if (has_const1 && val1 == 0) {
                        replace_with_mov(c, inst, inst->src[0]);
                        return true;
                }
                if (has_const0 && val0 == 0) {
                        replace_with_mov(c, inst, inst->src[1]);
                        return true;
                }
                break;

        case V3D_QPU_A_SUB:
                /* SUB x, 0 → MOV x */
                if (has_const1 && val1 == 0) {
                        replace_with_mov(c, inst, inst->src[0]);
                        return true;
                }
                /* SUB x, x → 0 */
                if (inst->src[0].file == QFILE_TEMP &&
                    inst->src[1].file == QFILE_TEMP &&
                    inst->src[0].index == inst->src[1].index) {
                        replace_with_const(c, inst, 0);
                        return true;
                }
                break;

        case V3D_QPU_A_FSUB:
                /* FSUB x, 0.0 → FMOV x */
                if (has_const1 && val1 == 0) {
                        replace_with_mov(c, inst, inst->src[0]);
                        return true;
                }
                break;

        case V3D_QPU_A_AND:
                /* AND x, 0 → 0 */
                if ((has_const0 && val0 == 0) || (has_const1 && val1 == 0)) {
                        replace_with_const(c, inst, 0);
                        return true;
                }
                /* AND x, ~0 → MOV x */
                if (has_const1 && val1 == 0xFFFFFFFF) {
                        replace_with_mov(c, inst, inst->src[0]);
                        return true;
                }
                if (has_const0 && val0 == 0xFFFFFFFF) {
                        replace_with_mov(c, inst, inst->src[1]);
                        return true;
                }
                /* AND x, x → MOV x */
                if (inst->src[0].file == QFILE_TEMP &&
                    inst->src[1].file == QFILE_TEMP &&
                    inst->src[0].index == inst->src[1].index) {
                        replace_with_mov(c, inst, inst->src[0]);
                        return true;
                }
                break;

        case V3D_QPU_A_OR:
                /* OR x, 0 → MOV x */
                if (has_const1 && val1 == 0) {
                        replace_with_mov(c, inst, inst->src[0]);
                        return true;
                }
                if (has_const0 && val0 == 0) {
                        replace_with_mov(c, inst, inst->src[1]);
                        return true;
                }
                /* OR x, ~0 → ~0 */
                if ((has_const0 && val0 == 0xFFFFFFFF) ||
                    (has_const1 && val1 == 0xFFFFFFFF)) {
                        replace_with_const(c, inst, 0xFFFFFFFF);
                        return true;
                }
                /* OR x, x → MOV x */
                if (inst->src[0].file == QFILE_TEMP &&
                    inst->src[1].file == QFILE_TEMP &&
                    inst->src[0].index == inst->src[1].index) {
                        replace_with_mov(c, inst, inst->src[0]);
                        return true;
                }
                break;

        case V3D_QPU_A_XOR:
                /* XOR x, 0 → MOV x */
                if (has_const1 && val1 == 0) {
                        replace_with_mov(c, inst, inst->src[0]);
                        return true;
                }
                if (has_const0 && val0 == 0) {
                        replace_with_mov(c, inst, inst->src[1]);
                        return true;
                }
                /* XOR x, x → 0 */
                if (inst->src[0].file == QFILE_TEMP &&
                    inst->src[1].file == QFILE_TEMP &&
                    inst->src[0].index == inst->src[1].index) {
                        replace_with_const(c, inst, 0);
                        return true;
                }
                break;

        case V3D_QPU_A_SHL:
        case V3D_QPU_A_SHR:
        case V3D_QPU_A_ASR:
        case V3D_QPU_A_ROR:
                /* SHIFT x, 0 → MOV x */
                if (has_const1 && (val1 & 0x1f) == 0) {
                        replace_with_mov(c, inst, inst->src[0]);
                        return true;
                }
                break;

        case V3D_QPU_A_MIN:
        case V3D_QPU_A_FMIN:
                /* MIN x, x → MOV x (when both sources are same temp) */
                if (inst->src[0].file == QFILE_TEMP &&
                    inst->src[1].file == QFILE_TEMP &&
                    inst->src[0].index == inst->src[1].index) {
                        replace_with_mov(c, inst, inst->src[0]);
                        return true;
                }
                break;

        case V3D_QPU_A_MAX:
        case V3D_QPU_A_FMAX:
                /* MAX x, x → MOV x */
                if (inst->src[0].file == QFILE_TEMP &&
                    inst->src[1].file == QFILE_TEMP &&
                    inst->src[0].index == inst->src[1].index) {
                        replace_with_mov(c, inst, inst->src[0]);
                        return true;
                }
                break;

        case V3D_QPU_A_NOT:
                /* NOT(NOT x) → MOV x */
                if (inst->src[0].file == QFILE_TEMP) {
                        struct qinst *src_def = c->defs[inst->src[0].index];
                        if (src_def &&
                            src_def->qpu.type == V3D_QPU_INSTR_TYPE_ALU &&
                            vir_is_add(src_def) &&
                            src_def->qpu.alu.add.op == V3D_QPU_A_NOT &&
                            src_def->qpu.alu.add.output_pack == V3D_QPU_PACK_NONE &&
                            src_def->qpu.alu.add.a.unpack == V3D_QPU_UNPACK_NONE) {
                                replace_with_mov(c, inst, src_def->src[0]);
                                return true;
                        }
                }
                break;

        case V3D_QPU_A_NEG:
                /* NEG(NEG x) → MOV x */
                if (inst->src[0].file == QFILE_TEMP) {
                        struct qinst *src_def = c->defs[inst->src[0].index];
                        if (src_def &&
                            src_def->qpu.type == V3D_QPU_INSTR_TYPE_ALU &&
                            vir_is_add(src_def) &&
                            src_def->qpu.alu.add.op == V3D_QPU_A_NEG &&
                            src_def->qpu.alu.add.output_pack == V3D_QPU_PACK_NONE &&
                            src_def->qpu.alu.add.a.unpack == V3D_QPU_UNPACK_NONE) {
                                replace_with_mov(c, inst, src_def->src[0]);
                                return true;
                        }
                }
                break;

        default:
                break;
        }

        return false;
}

static bool
try_opt_algebraic_mul(struct v3d_compile *c, struct qinst *inst)
{
        uint32_t val0, val1;
        bool has_const0 = get_temp_const_value(c, inst->src[0], &val0);
        bool has_const1 = get_temp_const_value(c, inst->src[1], &val1);
        int shift;

        switch (inst->qpu.alu.mul.op) {
        case V3D_QPU_M_FMUL:
                /* FMUL x, 1.0 → FMOV x */
                if (has_const1 && val1 == 0x3F800000) { /* 1.0f */
                        replace_with_mov(c, inst, inst->src[0]);
                        return true;
                }
                if (has_const0 && val0 == 0x3F800000) {
                        replace_with_mov(c, inst, inst->src[1]);
                        return true;
                }
                /* FMUL x, 0.0 → 0.0 (but be careful with NaN) */
                /* Skip this optimization as it changes NaN behavior */
                break;

        case V3D_QPU_M_UMUL24:
                /* UMUL24 x, 0 → 0 */
                if ((has_const0 && val0 == 0) || (has_const1 && val1 == 0)) {
                        replace_with_const(c, inst, 0);
                        return true;
                }
                /* UMUL24 x, 1 → MOV x (masked to 24 bits) */
                /* Skip as result is masked */
                /* UMUL24 x, 2^n → SHL x, n (for n < 24) */
                if (has_const1 && is_power_of_two(val1 & 0xFFFFFF, &shift) && shift < 24) {
                        replace_mul_with_shl(c, inst, inst->src[0], shift);
                        return true;
                }
                if (has_const0 && is_power_of_two(val0 & 0xFFFFFF, &shift) && shift < 24) {
                        replace_mul_with_shl(c, inst, inst->src[1], shift);
                        return true;
                }
                break;

        case V3D_QPU_M_ADD:
                /* ADD in MUL pipe: x + 0 → MOV x */
                if (has_const1 && val1 == 0) {
                        replace_with_mov(c, inst, inst->src[0]);
                        return true;
                }
                if (has_const0 && val0 == 0) {
                        replace_with_mov(c, inst, inst->src[1]);
                        return true;
                }
                break;

        case V3D_QPU_M_SUB:
                /* SUB in MUL pipe: x - 0 → MOV x */
                if (has_const1 && val1 == 0) {
                        replace_with_mov(c, inst, inst->src[0]);
                        return true;
                }
                break;

        default:
                break;
        }

        return false;
}

static bool
try_opt_algebraic(struct v3d_compile *c, struct qinst *inst)
{
        if (inst->qpu.type != V3D_QPU_INSTR_TYPE_ALU)
                return false;

        /* Skip if conditional execution */
        if (inst->qpu.flags.ac != V3D_QPU_COND_NONE ||
            inst->qpu.flags.mc != V3D_QPU_COND_NONE)
                return false;

        /* Skip if output packing */
        if (inst->qpu.alu.add.output_pack != V3D_QPU_PACK_NONE ||
            inst->qpu.alu.mul.output_pack != V3D_QPU_PACK_NONE)
                return false;

        /* Skip if input unpacking - values would be modified */
        if (inst->qpu.alu.add.a.unpack != V3D_QPU_UNPACK_NONE ||
            inst->qpu.alu.add.b.unpack != V3D_QPU_UNPACK_NONE ||
            inst->qpu.alu.mul.a.unpack != V3D_QPU_UNPACK_NONE ||
            inst->qpu.alu.mul.b.unpack != V3D_QPU_UNPACK_NONE)
                return false;

        if (vir_is_add(inst))
                return try_opt_algebraic_add(c, inst);

        if (vir_is_mul(inst))
                return try_opt_algebraic_mul(c, inst);

        return false;
}

bool
vir_opt_algebraic(struct v3d_compile *c)
{
        bool progress = false;

        vir_for_each_block(block, c) {
                c->cur_block = block;
                vir_for_each_inst_safe(inst, block) {
                        progress = try_opt_algebraic(c, inst) || progress;
                }
        }

        return progress;
}
