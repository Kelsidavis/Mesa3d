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
 * @file vir_opt_constant_propagate.c
 *
 * This optimization pass propagates constant values through the program.
 *
 * It tracks which temps are known to hold constant values (either from
 * direct uniform loads or through MOV chains) and replaces uses of those
 * temps with references to the original constant-loading instruction.
 *
 * This enables more opportunities for constant folding, as the constant
 * folder only looks one level deep at instruction definitions.
 *
 * Example:
 *   Before:
 *     ldunif t0  (constant 5)
 *     mov t1, t0
 *     mov t2, t1
 *     add t3, t2, t2
 *
 *   After:
 *     ldunif t0  (constant 5)
 *     mov t1, t0
 *     mov t2, t1
 *     add t3, t0, t0   ; now constant folder can evaluate this
 */

#include "v3d_compiler.h"

static bool debug = false;

/**
 * Information about a temp's constant value.
 */
struct const_value {
        /* True if this temp holds a known constant value */
        bool is_const;
        /* The temp index that originally loaded this constant via ldunif */
        uint32_t source_temp;
};

/**
 * Check if an instruction is a simple MOV with no modifications.
 */
static bool
is_simple_mov(struct qinst *inst)
{
        if (!inst)
                return false;

        if (inst->qpu.type != V3D_QPU_INSTR_TYPE_ALU)
                return false;

        if (inst->qpu.alu.mul.op != V3D_QPU_M_MOV &&
            inst->qpu.alu.mul.op != V3D_QPU_M_FMOV)
                return false;

        /* No packing */
        if (inst->qpu.alu.mul.output_pack != V3D_QPU_PACK_NONE)
                return false;

        /* No unpacking */
        if (inst->qpu.alu.mul.a.unpack != V3D_QPU_UNPACK_NONE)
                return false;

        /* No conditional execution */
        if (inst->qpu.flags.mc != V3D_QPU_COND_NONE)
                return false;

        return true;
}

/**
 * Check if an instruction loads a constant uniform.
 */
static bool
is_const_ldunif(struct v3d_compile *c, struct qinst *inst)
{
        if (!inst)
                return false;

        if (!inst->qpu.sig.ldunif && !inst->qpu.sig.ldunifrf)
                return false;

        if (c->uniform_contents[inst->uniform] != QUNIFORM_CONSTANT)
                return false;

        return true;
}

/**
 * Build a map of which temps hold constant values.
 */
static void
build_const_map(struct v3d_compile *c, struct const_value *const_map)
{
        /* Initialize all temps as non-constant */
        for (uint32_t i = 0; i < c->num_temps; i++) {
                const_map[i].is_const = false;
                const_map[i].source_temp = i;
        }

        /* First pass: find all temps that are direct ldunif of constants */
        for (uint32_t i = 0; i < c->num_temps; i++) {
                struct qinst *def = c->defs[i];
                if (is_const_ldunif(c, def)) {
                        const_map[i].is_const = true;
                        const_map[i].source_temp = i;
                }
        }

        /* Iteratively propagate through MOV chains until no changes */
        bool changed;
        do {
                changed = false;
                for (uint32_t i = 0; i < c->num_temps; i++) {
                        if (const_map[i].is_const)
                                continue;

                        struct qinst *def = c->defs[i];
                        if (!is_simple_mov(def))
                                continue;

                        if (def->src[0].file != QFILE_TEMP)
                                continue;

                        uint32_t src_temp = def->src[0].index;
                        if (const_map[src_temp].is_const) {
                                const_map[i].is_const = true;
                                const_map[i].source_temp = const_map[src_temp].source_temp;
                                changed = true;
                        }
                }
        } while (changed);
}

/**
 * Try to replace a temp source with the original constant source.
 */
static bool
try_propagate_const(struct v3d_compile *c, struct qinst *inst,
                    struct const_value *const_map)
{
        bool progress = false;

        for (int i = 0; i < vir_get_nsrc(inst); i++) {
                if (inst->src[i].file != QFILE_TEMP)
                        continue;

                uint32_t temp_idx = inst->src[i].index;
                if (!const_map[temp_idx].is_const)
                        continue;

                uint32_t source_temp = const_map[temp_idx].source_temp;

                /* Already pointing to the source */
                if (temp_idx == source_temp)
                        continue;

                /* Can't propagate if the instruction has unpack on this source */
                if (vir_is_add(inst)) {
                        if (i == 0 && inst->qpu.alu.add.a.unpack != V3D_QPU_UNPACK_NONE)
                                continue;
                        if (i == 1 && inst->qpu.alu.add.b.unpack != V3D_QPU_UNPACK_NONE)
                                continue;
                } else if (vir_is_mul(inst)) {
                        if (i == 0 && inst->qpu.alu.mul.a.unpack != V3D_QPU_UNPACK_NONE)
                                continue;
                        if (i == 1 && inst->qpu.alu.mul.b.unpack != V3D_QPU_UNPACK_NONE)
                                continue;
                }

                if (debug) {
                        fprintf(stderr, "Constant propagate src[%d] temp %d -> %d: ",
                                i, temp_idx, source_temp);
                        vir_dump_inst(c, inst);
                        fprintf(stderr, "\n");
                }

                inst->src[i].index = source_temp;
                progress = true;
        }

        return progress;
}

bool
vir_opt_constant_propagate(struct v3d_compile *c)
{
        bool progress = false;

        struct const_value *const_map = calloc(c->num_temps,
                                               sizeof(struct const_value));
        if (!const_map)
                return false;

        build_const_map(c, const_map);

        vir_for_each_block(block, c) {
                vir_for_each_inst(inst, block) {
                        if (inst->qpu.type != V3D_QPU_INSTR_TYPE_ALU)
                                continue;

                        progress = try_propagate_const(c, inst, const_map) || progress;
                }
        }

        free(const_map);

        return progress;
}
