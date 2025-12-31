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

#include "v3dv_private.h"
#include "broadcom/common/v3d_tiling.h"
#include "util/box.h"
#include "vk_format.h"

static void
copy_linear_image_to_from_memory(void *image_ptr,
                                 uint32_t image_stride,
                                 void *mem_ptr,
                                 uint32_t mem_stride,
                                 uint32_t cpp,
                                 VkOffset3D offset,
                                 VkExtent3D extent,
                                 bool to_image)
{
   for (uint32_t y = 0; y < extent.height; y++) {
      void *img_row = image_ptr + (offset.y + y) * image_stride + offset.x * cpp;
      void *mem_row = mem_ptr + y * mem_stride;

      if (to_image)
         memcpy(img_row, mem_row, extent.width * cpp);
      else
         memcpy(mem_row, img_row, extent.width * cpp);
   }
}

static void
copy_tiled_image_to_from_memory(void *image_ptr,
                                uint32_t image_stride,
                                void *mem_ptr,
                                uint32_t mem_stride,
                                enum v3d_tiling_mode tiling,
                                uint32_t cpp,
                                uint32_t image_height,
                                VkOffset3D offset,
                                VkExtent3D extent,
                                bool to_image)
{
   struct pipe_box box = {
      .x = offset.x,
      .y = offset.y,
      .z = offset.z,
      .width = extent.width,
      .height = extent.height,
      .depth = extent.depth,
   };

   if (to_image) {
      v3d_store_tiled_image(image_ptr, image_stride,
                            mem_ptr, mem_stride,
                            tiling, cpp, image_height, &box);
   } else {
      v3d_load_tiled_image(mem_ptr, mem_stride,
                           image_ptr, image_stride,
                           tiling, cpp, image_height, &box);
   }
}

static void
do_copy_image_to_from_memory(struct v3dv_image *image,
                             void *image_base_ptr,
                             const VkImageSubresourceLayers *subres,
                             VkOffset3D offset,
                             VkExtent3D extent,
                             void *mem_ptr,
                             uint32_t mem_row_stride,
                             uint32_t mem_layer_stride,
                             bool to_image)
{
   /* For now, we only support plane 0 */
   uint32_t plane = 0;
   uint32_t cpp = image->planes[plane].cpp;
   const struct v3d_resource_slice *slice =
      &image->planes[plane].slices[subres->mipLevel];

   uint32_t layer_count = subres->layerCount;
   if (layer_count == VK_REMAINING_ARRAY_LAYERS)
      layer_count = image->vk.array_layers - subres->baseArrayLayer;

   void *slice_ptr = image_base_ptr + slice->offset;

   for (uint32_t layer = 0; layer < layer_count; layer++) {
      uint32_t img_layer = subres->baseArrayLayer + layer;

      /* Calculate layer offset in image memory */
      void *layer_ptr;
      if (image->vk.image_type == VK_IMAGE_TYPE_3D) {
         layer_ptr = slice_ptr;
      } else {
         layer_ptr = slice_ptr + img_layer * image->planes[plane].cube_map_stride;
      }

      /* Memory pointer for this layer */
      void *mem_layer_ptr = mem_ptr + layer * mem_layer_stride;

      for (uint32_t z = 0; z < extent.depth; z++) {
         void *depth_ptr;
         if (image->vk.image_type == VK_IMAGE_TYPE_3D) {
            depth_ptr = layer_ptr + (offset.z + z) * slice->size;
         } else {
            depth_ptr = layer_ptr;
         }

         void *mem_depth_ptr = mem_layer_ptr + z * mem_layer_stride;

         VkOffset3D slice_offset = {
            .x = offset.x,
            .y = offset.y,
            .z = 0,
         };
         VkExtent3D slice_extent = {
            .width = extent.width,
            .height = extent.height,
            .depth = 1,
         };

         if (slice->tiling == V3D_TILING_RASTER) {
            copy_linear_image_to_from_memory(depth_ptr,
                                             slice->stride,
                                             mem_depth_ptr,
                                             mem_row_stride,
                                             cpp,
                                             slice_offset,
                                             slice_extent,
                                             to_image);
         } else {
            copy_tiled_image_to_from_memory(depth_ptr,
                                            slice->stride,
                                            mem_depth_ptr,
                                            mem_row_stride,
                                            slice->tiling,
                                            cpp,
                                            slice->padded_height,
                                            slice_offset,
                                            slice_extent,
                                            to_image);
         }
      }
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
v3dv_CopyMemoryToImage(VkDevice _device,
                       const VkCopyMemoryToImageInfo *info)
{
   V3DV_FROM_HANDLE(v3dv_device, device, _device);
   V3DV_FROM_HANDLE(v3dv_image, image, info->dstImage);

   /* We need the image memory to be mapped */
   struct v3dv_device_memory *mem = image->planes[0].mem;
   if (!mem || !mem->bo)
      return VK_ERROR_MEMORY_MAP_FAILED;

   /* Map the image memory if not already mapped */
   void *image_ptr = mem->bo->map;
   bool needs_unmap = false;
   if (!image_ptr) {
      if (!v3dv_bo_map(device, mem->bo, mem->bo->size))
         return VK_ERROR_MEMORY_MAP_FAILED;
      image_ptr = mem->bo->map;
      needs_unmap = true;
   }

   /* Adjust for image's offset in the memory allocation */
   image_ptr += image->planes[0].mem_offset;

   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkMemoryToImageCopy *region = &info->pRegions[i];

      struct vk_image_buffer_layout mem_layout =
         vk_memory_to_image_copy_layout(&image->vk, region);

      do_copy_image_to_from_memory(image,
                                   image_ptr,
                                   &region->imageSubresource,
                                   region->imageOffset,
                                   region->imageExtent,
                                   (void *)region->pHostPointer,
                                   mem_layout.row_stride_B,
                                   mem_layout.image_stride_B,
                                   true /* to_image */);
   }

   if (needs_unmap)
      v3dv_bo_unmap(device, mem->bo);

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
v3dv_CopyImageToMemory(VkDevice _device,
                       const VkCopyImageToMemoryInfo *info)
{
   V3DV_FROM_HANDLE(v3dv_device, device, _device);
   V3DV_FROM_HANDLE(v3dv_image, image, info->srcImage);

   /* We need the image memory to be mapped */
   struct v3dv_device_memory *mem = image->planes[0].mem;
   if (!mem || !mem->bo)
      return VK_ERROR_MEMORY_MAP_FAILED;

   /* Map the image memory if not already mapped */
   void *image_ptr = mem->bo->map;
   bool needs_unmap = false;
   if (!image_ptr) {
      if (!v3dv_bo_map(device, mem->bo, mem->bo->size))
         return VK_ERROR_MEMORY_MAP_FAILED;
      image_ptr = mem->bo->map;
      needs_unmap = true;
   }

   /* Adjust for image's offset in the memory allocation */
   image_ptr += image->planes[0].mem_offset;

   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkImageToMemoryCopy *region = &info->pRegions[i];

      struct vk_image_buffer_layout mem_layout =
         vk_image_to_memory_copy_layout(&image->vk, region);

      do_copy_image_to_from_memory(image,
                                   image_ptr,
                                   &region->imageSubresource,
                                   region->imageOffset,
                                   region->imageExtent,
                                   region->pHostPointer,
                                   mem_layout.row_stride_B,
                                   mem_layout.image_stride_B,
                                   false /* to_image */);
   }

   if (needs_unmap)
      v3dv_bo_unmap(device, mem->bo);

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
v3dv_CopyImageToImage(VkDevice _device,
                      const VkCopyImageToImageInfo *info)
{
   V3DV_FROM_HANDLE(v3dv_device, device, _device);
   V3DV_FROM_HANDLE(v3dv_image, src_image, info->srcImage);
   V3DV_FROM_HANDLE(v3dv_image, dst_image, info->dstImage);

   /* Map source image */
   struct v3dv_device_memory *src_mem = src_image->planes[0].mem;
   if (!src_mem || !src_mem->bo)
      return VK_ERROR_MEMORY_MAP_FAILED;

   void *src_ptr = src_mem->bo->map;
   bool src_needs_unmap = false;
   if (!src_ptr) {
      if (!v3dv_bo_map(device, src_mem->bo, src_mem->bo->size))
         return VK_ERROR_MEMORY_MAP_FAILED;
      src_ptr = src_mem->bo->map;
      src_needs_unmap = true;
   }
   src_ptr += src_image->planes[0].mem_offset;

   /* Map destination image */
   struct v3dv_device_memory *dst_mem = dst_image->planes[0].mem;
   if (!dst_mem || !dst_mem->bo) {
      if (src_needs_unmap)
         v3dv_bo_unmap(device, src_mem->bo);
      return VK_ERROR_MEMORY_MAP_FAILED;
   }

   void *dst_ptr = dst_mem->bo->map;
   bool dst_needs_unmap = false;
   if (!dst_ptr) {
      if (!v3dv_bo_map(device, dst_mem->bo, dst_mem->bo->size)) {
         if (src_needs_unmap)
            v3dv_bo_unmap(device, src_mem->bo);
         return VK_ERROR_MEMORY_MAP_FAILED;
      }
      dst_ptr = dst_mem->bo->map;
      dst_needs_unmap = true;
   }
   dst_ptr += dst_image->planes[0].mem_offset;

   /* For image-to-image copy, we use a temporary linear buffer */
   for (uint32_t i = 0; i < info->regionCount; i++) {
      const VkImageCopy2 *region = &info->pRegions[i];

      uint32_t src_cpp = src_image->planes[0].cpp;
      uint32_t row_size = region->extent.width * src_cpp;
      uint32_t layer_size = row_size * region->extent.height;
      uint32_t total_size = layer_size * region->extent.depth;

      /* Allocate temporary buffer */
      void *tmp = malloc(total_size);
      if (!tmp) {
         if (src_needs_unmap)
            v3dv_bo_unmap(device, src_mem->bo);
         if (dst_needs_unmap)
            v3dv_bo_unmap(device, dst_mem->bo);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }

      /* Copy from source to tmp */
      do_copy_image_to_from_memory(src_image,
                                   src_ptr,
                                   &region->srcSubresource,
                                   region->srcOffset,
                                   region->extent,
                                   tmp,
                                   row_size,
                                   layer_size,
                                   false /* to_image */);

      /* Copy from tmp to destination */
      do_copy_image_to_from_memory(dst_image,
                                   dst_ptr,
                                   &region->dstSubresource,
                                   region->dstOffset,
                                   region->extent,
                                   tmp,
                                   row_size,
                                   layer_size,
                                   true /* to_image */);

      free(tmp);
   }

   if (src_needs_unmap)
      v3dv_bo_unmap(device, src_mem->bo);
   if (dst_needs_unmap)
      v3dv_bo_unmap(device, dst_mem->bo);

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
v3dv_TransitionImageLayout(VkDevice _device,
                           uint32_t transitionCount,
                           const VkHostImageLayoutTransitionInfo *pTransitions)
{
   /* V3D doesn't use image layouts for memory access ordering.
    * The tiling format is fixed at image creation time, so this is a no-op.
    */
   return VK_SUCCESS;
}
