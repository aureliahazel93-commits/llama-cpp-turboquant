// Phase 15: Vulkan Shader Dispatch Infrastructure
//
// Based on ncnn convolution_vulkan.cpp pattern:
// - create_pipeline() at model load: compile SPIR-V, create descriptor set layouts
// - upload_model() at model load: stage weights to GPU memory (UMA = zero-copy)
// - forward() per compute: bind buffers, set push constants, record dispatch
//
// Key ncnn patterns replicated:
// 1. Storage buffers only (no UBOs) — bindings indexed by position
// 2. Push constants carry resolved strides (M, N, K, row_offset)
// 3. No explicit pipeline barriers between dispatches in same command buffer
// 4. Specialization constants for subgroup_size (32 or 64)
// 5. Dispatch dims: global = (M - row_offset) × N, local = subgroup_size

#include "ggml.h"
#include <vulkan/vulkan.h>
#include <vector>
#include <unordered_map>
#include <string>

#ifndef GGML_HYBRID_DEQUANT_VULKAN_H
#define GGML_HYBRID_DEQUANT_VULKAN_H

struct vulkan_shader {
    VkShaderModule module = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool = VK_NULL_HANDLE;
    uint32_t subgroup_size = 32;
    bool initialized = false;
};

struct vulkan_buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void * mapped = nullptr;  // UMA: permanently mapped (host_visible + device_local)
};

struct vulkan_queue {
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool cmd_pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd_buf = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties2 props2 = {};

    // Compiled shaders per quant type
    std::unordered_map<ggml_type, vulkan_shader> shaders;

    // Descriptor sets (reused per dispatch)
    VkDescriptorSet desc_sets[GGML_TYPE_COUNT] = {};

    bool initialized = false;
};

// ============= SPIR-V embedded shaders =============
// Compiled at build time via CMake custom command (glslangValidator -V --vn)
// Each quant type has its own .comp shader compiled to a C header with the SPIR-V bytes

// These are included from the CMake-generated headers:
// #include "gemm_q4_0.spv.h"  → extern const uint32_t gemm_q4_0_spv[];
// #include "gemm_q8_0.spv.h"  → extern const uint32_t gemm_q8_0_spv[];
// etc.

// ============= Init: compile shaders, create pipelines =============

static VkShaderModule create_shader_module(VkDevice device, const uint32_t * spv, size_t size) {
    VkShaderModuleCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = size;
    ci.pCode = spv;
    VkShaderModule mod;
    vkCreateShaderModule(device, &ci, nullptr, &mod);
    return mod;
}

static bool create_shader_pipeline(
    VkDevice device,
    vulkan_shader * sh,
    const uint32_t * spv_data,
    size_t spv_size,
    uint32_t subgroup_size)
{
    sh->subgroup_size = subgroup_size;
    sh->module = create_shader_module(device, spv_data, spv_size);

    // Descriptor set layout: 3 storage buffers (weights, activations, output)
    VkDescriptorSetLayoutBinding bindings[3] = {};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo dslci = {};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 3;
    dslci.pBindings = bindings;
    vkCreateDescriptorSetLayout(device, &dslci, nullptr, &sh->desc_layout);

    // Push constant range: 24 bytes (6 uint32_t: M, N, K, row_offset, w_stride, x_stride)
    VkPushConstantRange pc_range = {};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset = 0;
    pc_range.size = 24;

    // Pipeline layout
    VkPipelineLayoutCreateInfo plci = {};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &sh->desc_layout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pc_range;
    vkCreatePipelineLayout(device, &plci, nullptr, &sh->layout);

    // Specialization: subgroup_size
    VkSpecializationMapEntry spe = {};
    spe.constantID = 0;
    spe.offset = 0;
    spe.size = sizeof(uint32_t);

    VkSpecializationInfo spec_info = {};
    spec_info.mapEntryCount = 1;
    spec_info.pMapEntries = &spe;
    spec_info.dataSize = sizeof(uint32_t);
    spec_info.pData = &subgroup_size;

    // Compute pipeline
    VkComputePipelineCreateInfo cpci = {};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sh->module;
    cpci.stage.pName = "main";
    cpci.stage.pSpecializationInfo = &spec_info;
    cpci.layout = sh->layout;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &sh->pipeline);

    sh->initialized = true;
    return true;
}

// ============= UMA Buffer Allocation =============
//
// Allocates a single buffer accessible by both CPU (via mapped pointer) and iGPU (via VkBuffer).
// On UMA hardware: HOST_VISIBLE + DEVICE_LOCAL → true zero-copy (same physical pages).
// On non-UMA: HOST_VISIBLE only → works but with PCIe copy overhead (not zero-copy).

static vulkan_buffer allocate_uma_buffer(VkDevice device, VkPhysicalDevice phys_dev, VkDeviceSize size) {
    vulkan_buffer buf = {};

    // Find memory type that is HOST_VISIBLE and preferably DEVICE_LOCAL (UMA)
    VkPhysicalDeviceMemoryProperties mem_props;
    vkGetPhysicalDeviceMemoryProperties(phys_dev, &mem_props);

    uint32_t mem_type_index = UINT32_MAX;
    bool is_uma = false;

    // Try to find HOST_VISIBLE + DEVICE_LOCAL (true UMA)
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
        if ((mem_props.memoryTypes[i].propertyFlags &
             (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            mem_type_index = i;
            is_uma = true;
            break;
        }
    }

    // Fallback: HOST_VISIBLE only (non-UMA, still works with copy)
    if (mem_type_index == UINT32_MAX) {
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++) {
            if (mem_props.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
                mem_type_index = i;
                break;
            }
        }
    }

    if (mem_type_index == UINT32_MAX) return buf;  // no suitable memory type

    // Create buffer
    VkBufferCreateInfo bci = {};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device, &bci, nullptr, &buf.buffer);

    // Allocate memory
    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(device, buf.buffer, &reqs);

    VkMemoryAllocateInfo mai = {};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = reqs.size;
    mai.memoryTypeIndex = mem_type_index;
    vkAllocateMemory(device, &mai, nullptr, &buf.memory);
    vkBindBufferMemory(device, buf.buffer, buf.memory, 0);

    // Persistently map (UMA: pointer and VkBuffer share the same physical pages)
    vkMapMemory(device, buf.memory, 0, size, 0, &buf.mapped);
    buf.size = size;

    return buf;
}

// ============= GEMM Dispatch =============

void vulkan_gemm_dispatch(
    vulkan_queue * q,
    ggml_tensor * dst,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    int row_start,
    int row_end)
{
    if (!q || !q->initialized) return;

    const uint32_t M = (uint32_t)src0->ne[1];
    const uint32_t N = (uint32_t)src1->ne[1];
    const uint32_t K = (uint32_t)src0->ne[0];
    const uint32_t igpu_rows = row_end - row_start;

    if (igpu_rows == 0) return;

    // Look up shader for this quant type
    auto it = q->shaders.find(src0->type);
    if (it == q->shaders.end() || !it->second.initialized) return;  // no shader for this type

    const vulkan_shader & sh = it->second;

    // Begin command buffer (ncnn pattern: no explicit barriers between dispatches)
    vkBeginCommandBuffer(q->cmd_buf, nullptr);

    // Bind pipeline
    vkCmdBindPipeline(q->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, sh.pipeline);

    // Update descriptor sets: bindings[0]=weights [1]=activations [2]=output
    // On UMA, the VkBuffer IS the same memory as the CPU pointer — no staging copy needed.
    VkDescriptorBufferInfo buf_infos[3];
    // ... (set buf_infos from src0, src1, dst VkBuffers)

    VkWriteDescriptorSet writes[3] = {};
    for (int i = 0; i < 3; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = q->desc_sets[src0->type];
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buf_infos[i];
    }
    vkUpdateDescriptorSets(q->device, 3, writes, 0, nullptr);

    VkDescriptorSet desc_set = q->desc_sets[src0->type];
    vkCmdBindDescriptorSets(q->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            sh.layout, 0, 1, &desc_set, 0, nullptr);

    // Push constants (ncnn pattern: resolved strides, not raw dims)
    struct {
        uint32_t M;
        uint32_t N;
        uint32_t K;
        uint32_t row_offset;
        uint32_t w_stride;
        uint32_t x_stride;
    } pc_data = { M, N, K, (uint32_t)row_start, (uint32_t)src0->nb[1], (uint32_t)src1->nb[1] };

    vkCmdPushConstants(q->cmd_buf, sh.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc_data), &pc_data);

    // Dispatch: one workgroup per (row, col) pair in the iGPU slice
    // global = (igpu_rows, N, 1), local = (subgroup_size, 1, 1)
    uint32_t groups_x = igpu_rows;
    uint32_t groups_y = N;
    vkCmdDispatch(q->cmd_buf, groups_x, groups_y, 1);

    vkEndCommandBuffer(q->cmd_buf);

    // Submit (ncnn pattern: no explicit barriers — queue submission order is implicit)
    VkSubmitInfo submit = {};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &q->cmd_buf;
    vkQueueSubmit(q->queue, 1, &submit, VK_NULL_HANDLE);

    // Note: vkQueueWaitIdle is called by the hybrid dispatcher after both CPU + iGPU finish
}

#endif // GGML_HYBRID_DEQUANT_VULKAN_H
