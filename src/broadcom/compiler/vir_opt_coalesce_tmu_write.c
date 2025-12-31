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
 * @file vir_opt_coalesce_tmu_write.c
 *
 * This optimization pass merges ALU operations with their TMU write MOVs.
 *
 * When an ALU instruction produces a value that is only used by a MOV to
 * a TMU magic register, we can change the ALU instruction to write directly
 * to the TMU register and remove the MOV:
 *
 * Before:
 *   add r0, r1, r2    ; compute value
 *   mov tmu_s, r0     ; write to TMU
 *
 * After:
 *   add tmu_s, r1, r2 ; compute and write directly to TMU
 *
 * This reduces instruction count and register pressure.
 */

#include "v3d_compiler.h"

static bool debug = false;

/**
 * Check if this instruction is a simple MOV to a magic register (TMU).
 */
static bool
is_tmu_write_mov(struct qinst *inst)
{
        if (inst->qpu.type != V3D_QPU_INSTR_TYPE_ALU)
                return false;

        /* Must be writing to a magic register */
        if (inst->dst.file != QFILE_MAGIC)
                return false;

        /* Must be a MOV in the MUL pipe */
        if (!vir_is_mul(inst))
                return false;

        if (inst->qpu.alu.mul.op != V3D_QPU_M_MOV &&
            inst->qpu.alu.mul.op != V3D_QPU_M_FMOV)
                return false;

        /* Source must be a temp */
        if (inst->src[0].file != QFILE_TEMP)
                return false;

        /* Must not have any unpack operations */
        if (inst->qpu.alu.mul.a.unpack != V3D_QPU_UNPACK_NONE)
                return false;

        /* Must not have conditional execution */
        if (inst->qpu.flags.mc != V3D_QPU_COND_NONE)
                return false;

        return true;
}

/**
 * Check if the source definition can have its destination changed to a
 * magic register.
 */
static bool
can_coalesce_def(struct v3d_compile *c, struct qinst *def, struct qinst *mov)
{
        /* Must be an ALU instruction */
        if (def->qpu.type != V3D_QPU_INSTR_TYPE_ALU)
                return false;

        /* Must write to a temp */
        if (def->dst.file != QFILE_TEMP)
                return false;

        /* SFU operations must write to a physical register, not magic */
        if (v3d_qpu_uses_sfu(&def->qpu))
                return false;

        /* ldunif/ldunifa can't write to magic registers */
        if (def->qpu.sig.ldunif || def->qpu.sig.ldunifa ||
            def->qpu.sig.ldunifrf || def->qpu.sig.ldunifarf)
                return false;

        /* ldvary can't write to magic registers */
        if (def->qpu.sig.ldvary)
                return false;

        /* ldtmu can't write to magic registers */
        if (def->qpu.sig.ldtmu)
                return false;

        /* Must not have output pack (magic registers can't be packed) */
        if (vir_is_add(def)) {
                if (def->qpu.alu.add.output_pack != V3D_QPU_PACK_NONE)
                        return false;
        } else if (vir_is_mul(def)) {
                if (def->qpu.alu.mul.output_pack != V3D_QPU_PACK_NONE)
                        return false;
        }

        /* Must not have conditional execution or flags write that we'd
         * want to preserve
         */
        if (def->qpu.flags.ac != V3D_QPU_COND_NONE ||
            def->qpu.flags.mc != V3D_QPU_COND_NONE)
                return false;

        return true;
}

/**
 * Count uses of a temp register.
 */
static int
count_temp_uses(struct v3d_compile *c, uint32_t temp_index)
{
        int uses = 0;

        vir_for_each_block(block, c) {
                vir_for_each_inst(inst, block) {
                        for (int i = 0; i < vir_get_nsrc(inst); i++) {
                                if (inst->src[i].file == QFILE_TEMP &&
                                    inst->src[i].index == temp_index) {
                                        uses++;
                                }
                        }
                }
        }

        return uses;
}

/**
 * Check that the def and mov are in the same basic block and that
 * there are no intervening writes to the temp between def and mov.
 */
static bool
def_and_mov_in_same_block_no_intervening_writes(struct v3d_compile *c,
                                                 struct qinst *def,
                                                 struct qinst *mov)
{
        /* Find the block containing the mov */
        struct qblock *mov_block = NULL;
        vir_for_each_block(block, c) {
                vir_for_each_inst(inst, block) {
                        if (inst == mov) {
                                mov_block = block;
                                break;
                        }
                }
                if (mov_block)
                        break;
        }

        if (!mov_block)
                return false;

        /* Check that def is in the same block and comes before mov */
        bool found_def = false;
        vir_for_each_inst(inst, mov_block) {
                if (inst == def) {
                        found_def = true;
                        continue;
                }

                if (found_def) {
                        /* Check for intervening writes to the temp */
                        if (inst->dst.file == QFILE_TEMP &&
                            inst->dst.index == def->dst.index) {
                                return false;
                        }

                        if (inst == mov)
                                return true;
                }
        }

        return false;
}

static bool
try_coalesce_tmu_write(struct v3d_compile *c, struct qinst *mov)
{
        if (!is_tmu_write_mov(mov))
                return false;

        uint32_t temp_index = mov->src[0].index;

        /* Get the definition of the source temp */
        struct qinst *def = c->defs[temp_index];
        if (!def)
                return false;

        /* Check if we can coalesce */
        if (!can_coalesce_def(c, def, mov))
                return false;

        /* The temp must only be used by this MOV */
        if (count_temp_uses(c, temp_index) != 1)
                return false;

        /* Def and mov must be in the same block with no intervening writes */
        if (!def_and_mov_in_same_block_no_intervening_writes(c, def, mov))
                return false;

        if (debug) {
                fprintf(stderr, "Coalescing TMU write:\n");
                fprintf(stderr, "  def: ");
                vir_dump_inst(c, def);
                fprintf(stderr, "\n  mov: ");
                vir_dump_inst(c, mov);
                fprintf(stderr, "\n");
        }

        /* Change the definition's destination to the magic register */
        c->defs[temp_index] = NULL;
        def->dst = mov->dst;

        /* Remove the MOV */
        vir_remove_instruction(c, mov);

        return true;
}

bool
vir_opt_coalesce_tmu_write(struct v3d_compile *c)
{
        bool progress = false;

        vir_for_each_block(block, c) {
                vir_for_each_inst_safe(inst, block) {
                        progress = try_coalesce_tmu_write(c, inst) || progress;
                }
        }

        return progress;
}
