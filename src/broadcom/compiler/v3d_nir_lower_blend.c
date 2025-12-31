/*
 * Copyright 2025 Raspberry Pi Ltd
 * SPDX-License-Identifier: MIT
 */

#include "util/format/u_format.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_format_convert.h"
#include "compiler/nir/nir_lower_blend.h"
#include "v3d_compiler.h"

/* For dynamic blend enables, we need to save the original colors before
 * nir_lower_blend modifies them, then wrap the results with bcsel.
 */
struct blend_src_info {
   nir_variable *orig_color[V3D_MAX_DRAW_BUFFERS];
   bool has_blend[V3D_MAX_DRAW_BUFFERS];
};

/* Dynamic blend equation support. When blend equations are dynamic, we need
 * to generate code that computes all possible blend factor values and blend
 * function results, then selects the correct one at runtime based on uniforms.
 */

/**
 * Compute a blend factor value for a single channel, selecting dynamically
 * based on the factor uniform.
 */
static nir_def *
v3d_blend_factor_dynamic(nir_builder *b, nir_def *factor,
                         nir_def *src, nir_def *src1, nir_def *dst,
                         nir_def *bconst, unsigned chan)
{
   nir_def *one = nir_imm_float(b, 1.0f);
   nir_def *zero = nir_imm_float(b, 0.0f);

   /* Compute factor value without inversion (factor & ~0x10) */
   nir_def *base_factor = nir_iand_imm(b, factor, ~PIPE_BLENDFACTOR_INVERT_BIT);

   /* Compute all possible base factor values */
   nir_def *src_c = nir_channel(b, src, chan);
   nir_def *dst_c = nir_channel(b, dst, chan);
   nir_def *src_a = nir_channel(b, src, 3);
   nir_def *dst_a = nir_channel(b, dst, 3);
   nir_def *bconst_c = nir_channel(b, bconst, chan);
   nir_def *bconst_a = nir_channel(b, bconst, 3);

   /* SRC_ALPHA_SATURATE: min(As, 1 - Ad) for RGB, 1 for alpha */
   nir_def *alpha_sat;
   if (chan < 3) {
      alpha_sat = nir_fmin(b, src_a, nir_fsub(b, one, dst_a));
   } else {
      alpha_sat = one;
   }

   /* src1 may be NULL if dual-source blending is not used */
   nir_def *src1_c = src1 ? nir_channel(b, src1, chan) : zero;
   nir_def *src1_a = src1 ? nir_channel(b, src1, 3) : zero;

   /* Build selection chain for base factor (before inversion) */
   nir_def *result = one;  /* Default to ONE */
   result = nir_bcsel(b, nir_ieq_imm(b, base_factor, PIPE_BLENDFACTOR_ONE), one, result);
   result = nir_bcsel(b, nir_ieq_imm(b, base_factor, PIPE_BLENDFACTOR_SRC_COLOR), src_c, result);
   result = nir_bcsel(b, nir_ieq_imm(b, base_factor, PIPE_BLENDFACTOR_SRC_ALPHA), src_a, result);
   result = nir_bcsel(b, nir_ieq_imm(b, base_factor, PIPE_BLENDFACTOR_DST_ALPHA), dst_a, result);
   result = nir_bcsel(b, nir_ieq_imm(b, base_factor, PIPE_BLENDFACTOR_DST_COLOR), dst_c, result);
   result = nir_bcsel(b, nir_ieq_imm(b, base_factor, PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE), alpha_sat, result);
   result = nir_bcsel(b, nir_ieq_imm(b, base_factor, PIPE_BLENDFACTOR_CONST_COLOR), bconst_c, result);
   result = nir_bcsel(b, nir_ieq_imm(b, base_factor, PIPE_BLENDFACTOR_CONST_ALPHA), bconst_a, result);
   result = nir_bcsel(b, nir_ieq_imm(b, base_factor, PIPE_BLENDFACTOR_SRC1_COLOR), src1_c, result);
   result = nir_bcsel(b, nir_ieq_imm(b, base_factor, PIPE_BLENDFACTOR_SRC1_ALPHA), src1_a, result);

   /* Apply inversion if INVERT_BIT is set: result = 1.0 - result */
   nir_def *is_inverted = nir_ine_imm(b, nir_iand_imm(b, factor, PIPE_BLENDFACTOR_INVERT_BIT), 0);
   nir_def *inverted = nir_fsub(b, one, result);
   result = nir_bcsel(b, is_inverted, inverted, result);

   return result;
}

/**
 * Apply blend function dynamically based on function uniform.
 */
static nir_def *
v3d_blend_func_dynamic(nir_builder *b, nir_def *func,
                       nir_def *src, nir_def *dst)
{
   /* Compute all possible function results */
   nir_def *add_result = nir_fadd(b, src, dst);
   nir_def *sub_result = nir_fsub(b, src, dst);
   nir_def *rsub_result = nir_fsub(b, dst, src);
   nir_def *min_result = nir_fmin(b, src, dst);
   nir_def *max_result = nir_fmax(b, src, dst);

   /* Select based on function */
   nir_def *result = add_result;  /* Default to ADD */
   result = nir_bcsel(b, nir_ieq_imm(b, func, PIPE_BLEND_ADD), add_result, result);
   result = nir_bcsel(b, nir_ieq_imm(b, func, PIPE_BLEND_SUBTRACT), sub_result, result);
   result = nir_bcsel(b, nir_ieq_imm(b, func, PIPE_BLEND_REVERSE_SUBTRACT), rsub_result, result);
   result = nir_bcsel(b, nir_ieq_imm(b, func, PIPE_BLEND_MIN), min_result, result);
   result = nir_bcsel(b, nir_ieq_imm(b, func, PIPE_BLEND_MAX), max_result, result);

   return result;
}

/**
 * Check if a blend function uses factors (ADD, SUBTRACT, REVERSE_SUBTRACT do,
 * MIN and MAX don't).
 */
static nir_def *
v3d_blend_factored_dynamic(nir_builder *b, nir_def *func)
{
   /* Factored if func < PIPE_BLEND_MIN (which is 3) */
   return nir_ult_imm(b, func, PIPE_BLEND_MIN);
}

static bool
save_original_colors_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct blend_src_info *info = data;

   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location != FRAG_RESULT_COLOR &&
       (sem.location < FRAG_RESULT_DATA0 ||
        sem.location >= FRAG_RESULT_DATA0 + V3D_MAX_DRAW_BUFFERS)) {
      return false;
   }

   int rt = sem.location == FRAG_RESULT_COLOR ? 0 :
            sem.location - FRAG_RESULT_DATA0;

   if (!info->has_blend[rt])
      return false;

   /* Create a variable to save the original color if not already created */
   if (!info->orig_color[rt]) {
      info->orig_color[rt] = nir_local_variable_create(
         nir_shader_get_entrypoint(b->shader),
         glsl_vec4_type(),
         "orig_blend_src");
   }

   /* Save the original color before blend modifies it */
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *src = intr->src[0].ssa;
   nir_def *padded = nir_pad_vector(b, src, 4);
   nir_store_var(b, info->orig_color[rt], padded, 0xf);

   return true;
}

static bool
apply_dynamic_blend_enable_instr(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   struct blend_src_info *info = data;

   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   if (sem.location != FRAG_RESULT_COLOR &&
       (sem.location < FRAG_RESULT_DATA0 ||
        sem.location >= FRAG_RESULT_DATA0 + V3D_MAX_DRAW_BUFFERS)) {
      return false;
   }

   int rt = sem.location == FRAG_RESULT_COLOR ? 0 :
            sem.location - FRAG_RESULT_DATA0;

   if (!info->has_blend[rt] || !info->orig_color[rt])
      return false;

   /* Load dynamic blend enable for this RT */
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *blend_enable = nir_load_blend_enabled_v3d(b, .base = rt);
   nir_def *enable_bool = nir_ine_imm(b, blend_enable, 0);

   /* Get the blended color (current store value) and original color */
   nir_def *blended = intr->src[0].ssa;
   nir_def *original = nir_load_var(b, info->orig_color[rt]);

   /* Trim original to match blended component count */
   if (blended->num_components < 4) {
      original = nir_trim_vector(b, original, blended->num_components);
   }

   /* Select between blended and original based on dynamic enable */
   nir_def *result = nir_bcsel(b, enable_bool, blended, original);

   /* Replace the store source */
   nir_src_rewrite(&intr->src[0], result);

   return true;
}

/* Context for dynamic blend equation lowering */
struct dynamic_blend_ctx {
   struct v3d_compile *c;
   uint8_t cbufs;
};

/**
 * Lower a store_output to use dynamic blend equations.
 * This performs software blending with runtime-selected blend factors/functions.
 */
static bool
apply_dynamic_blend_equation_instr(nir_builder *b, nir_intrinsic_instr *store, void *data)
{
   struct dynamic_blend_ctx *ctx = data;

   if (store->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(store);
   if (sem.location != FRAG_RESULT_COLOR &&
       (sem.location < FRAG_RESULT_DATA0 ||
        sem.location >= FRAG_RESULT_DATA0 + V3D_MAX_DRAW_BUFFERS)) {
      return false;
   }

   int rt = sem.location == FRAG_RESULT_COLOR ? 0 :
            sem.location - FRAG_RESULT_DATA0;

   /* Skip if this RT is not active */
   if (!(ctx->cbufs & (1 << rt)))
      return false;

   b->cursor = nir_before_instr(&store->instr);

   /* Get the source color (pad to 4 components for blend math) */
   nir_def *src = nir_pad_vector(b, store->src[0].ssa, 4);

   /* Load blend constants */
   nir_def *bconst = nir_vec4(b,
                               nir_load_blend_const_color_r_float(b),
                               nir_load_blend_const_color_g_float(b),
                               nir_load_blend_const_color_b_float(b),
                               nir_load_blend_const_color_a_float(b));

   /* Load destination color via framebuffer fetch */
   b->shader->info.outputs_read |= BITFIELD64_BIT(sem.location);
   b->shader->info.fs.uses_fbfetch_output = true;
   b->shader->info.fs.uses_sample_shading = true;
   sem.fb_fetch_output = true;

   nir_def *dst = nir_load_output(b, 4, nir_src_bit_size(store->src[0]),
                                   nir_imm_int(b, 0),
                                   .dest_type = nir_intrinsic_src_type(store),
                                   .io_semantics = sem);

   /* Clamp source to format range for fixed-point formats */
   enum pipe_format format = ctx->c->fs_key->color_fmt[rt].format;
   if (util_format_is_unorm(format)) {
      src = nir_fsat(b, src);
      bconst = nir_fsat(b, bconst);
   } else if (util_format_is_snorm(format)) {
      src = nir_fclamp(b, src, nir_imm_float(b, -1.0f), nir_imm_float(b, 1.0f));
      bconst = nir_fclamp(b, bconst, nir_imm_float(b, -1.0f), nir_imm_float(b, 1.0f));
   }

   /* Ensure dst alpha reads as 1.0 if format has no alpha */
   const struct util_format_description *desc = util_format_description(format);
   nir_def *one = nir_imm_float(b, 1.0f);
   nir_def *zero = nir_imm_float(b, 0.0f);
   bool has_alpha = desc->nr_channels >= 4 &&
                    desc->channel[3].type != UTIL_FORMAT_TYPE_VOID;
   dst = nir_vec4(b,
                  nir_channel(b, dst, 0),
                  desc->nr_channels > 1 ? nir_channel(b, dst, 1) : zero,
                  desc->nr_channels > 2 ? nir_channel(b, dst, 2) : zero,
                  has_alpha ? nir_channel(b, dst, 3) : one);

   /* Load dynamic blend parameters for this RT */
   nir_def *rgb_func = nir_load_blend_rgb_func_v3d(b, .base = rt);
   nir_def *rgb_src_factor = nir_load_blend_rgb_src_factor_v3d(b, .base = rt);
   nir_def *rgb_dst_factor = nir_load_blend_rgb_dst_factor_v3d(b, .base = rt);
   nir_def *alpha_func = nir_load_blend_alpha_func_v3d(b, .base = rt);
   nir_def *alpha_src_factor = nir_load_blend_alpha_src_factor_v3d(b, .base = rt);
   nir_def *alpha_dst_factor = nir_load_blend_alpha_dst_factor_v3d(b, .base = rt);

   /* Blend each channel */
   nir_def *channels[4];

   for (unsigned c = 0; c < 4; c++) {
      nir_def *func = (c < 3) ? rgb_func : alpha_func;
      nir_def *src_factor = (c < 3) ? rgb_src_factor : alpha_src_factor;
      nir_def *dst_factor = (c < 3) ? rgb_dst_factor : alpha_dst_factor;

      nir_def *psrc = nir_channel(b, src, c);
      nir_def *pdst = nir_channel(b, dst, c);

      /* Check if this function uses factors */
      nir_def *factored = v3d_blend_factored_dynamic(b, func);

      /* Compute factored blend: src * src_factor + dst * dst_factor */
      nir_def *src_fac = v3d_blend_factor_dynamic(b, src_factor, src, NULL, dst, bconst, c);
      nir_def *dst_fac = v3d_blend_factor_dynamic(b, dst_factor, src, NULL, dst, bconst, c);
      nir_def *psrc_factored = nir_fmul(b, psrc, src_fac);
      nir_def *pdst_factored = nir_fmul(b, pdst, dst_fac);

      /* For factored blend functions, use factored src/dst; otherwise use raw */
      nir_def *blend_src = nir_bcsel(b, factored, psrc_factored, psrc);
      nir_def *blend_dst = nir_bcsel(b, factored, pdst_factored, pdst);

      /* Apply the blend function */
      channels[c] = v3d_blend_func_dynamic(b, func, blend_src, blend_dst);
   }

   nir_def *blended = nir_vec(b, channels, 4);

   /* Trim to original component count */
   unsigned num_components = store->src[0].ssa->num_components;
   if (num_components < 4) {
      blended = nir_trim_vector(b, blended, num_components);
   }

   /* Replace the store source */
   nir_src_rewrite(&store->src[0], blended);

   return true;
}

bool
v3d_nir_lower_blend(nir_shader *nir, struct v3d_compile *c)
{
   if (!c->fs_key->software_blend)
      return false;

   /* For dynamic blend equations, we use a fully dynamic implementation
    * that selects blend factors and functions at runtime via uniforms.
    */
   if (c->fs_key->dynamic_blend_equations) {
      struct dynamic_blend_ctx ctx = {
         .c = c,
         .cbufs = c->fs_key->cbufs,
      };
      return nir_shader_intrinsics_pass(nir, apply_dynamic_blend_equation_instr,
                                        nir_metadata_control_flow, &ctx);
   }

   nir_lower_blend_options options = {
      /* logic op is handled elsewhere in the compiler */
      .logicop_enable = false,
      .scalar_blend_const = true,
   };

   struct blend_src_info info = { 0 };
   bool lower_blend = false;

   for (unsigned rt = 0; rt < V3D_MAX_DRAW_BUFFERS; rt++) {
      if (!(c->fs_key->cbufs & (1 << rt))) {
         static const nir_lower_blend_channel replace = {
            .func = PIPE_BLEND_ADD,
            .src_factor = PIPE_BLENDFACTOR_ONE,
            .dst_factor = PIPE_BLENDFACTOR_ZERO,
         };

         options.rt[rt].rgb = replace;
         options.rt[rt].alpha = replace;
         continue;
      }

      /* Check if this RT has non-replace blend mode */
      bool is_replace =
         c->fs_key->blend[rt].rgb_func == PIPE_BLEND_ADD &&
         c->fs_key->blend[rt].rgb_src_factor == PIPE_BLENDFACTOR_ONE &&
         c->fs_key->blend[rt].rgb_dst_factor == PIPE_BLENDFACTOR_ZERO &&
         c->fs_key->blend[rt].alpha_func == PIPE_BLEND_ADD &&
         c->fs_key->blend[rt].alpha_src_factor == PIPE_BLENDFACTOR_ONE &&
         c->fs_key->blend[rt].alpha_dst_factor == PIPE_BLENDFACTOR_ZERO;

      info.has_blend[rt] = !is_replace;
      lower_blend = true;

      /* Colour write mask is handled by the hardware. */
      options.rt[rt].colormask = 0xf;

      options.format[rt] = c->fs_key->color_fmt[rt].format;

      options.rt[rt].rgb.func = c->fs_key->blend[rt].rgb_func;
      options.rt[rt].alpha.func = c->fs_key->blend[rt].alpha_func;
      options.rt[rt].rgb.dst_factor = c->fs_key->blend[rt].rgb_dst_factor;
      options.rt[rt].alpha.dst_factor = c->fs_key->blend[rt].alpha_dst_factor;
      options.rt[rt].rgb.src_factor = c->fs_key->blend[rt].rgb_src_factor;
      options.rt[rt].alpha.src_factor = c->fs_key->blend[rt].alpha_src_factor;
   }

   if (!lower_blend)
      return false;

   /* For dynamic blend enables, save original colors before blend modifies them */
   if (c->fs_key->dynamic_blend_enables) {
      nir_shader_intrinsics_pass(nir, save_original_colors_instr,
                                 nir_metadata_control_flow, &info);
   }

   bool progress = nir_lower_blend(nir, &options);

   /* For dynamic blend enables, wrap blend results with bcsel */
   if (c->fs_key->dynamic_blend_enables && progress) {
      nir_shader_intrinsics_pass(nir, apply_dynamic_blend_enable_instr,
                                 nir_metadata_control_flow, &info);
   }

   return progress;
}
