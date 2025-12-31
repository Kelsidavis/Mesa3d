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

bool
v3d_nir_lower_blend(nir_shader *nir, struct v3d_compile *c)
{
   if (!c->fs_key->software_blend)
      return false;

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
