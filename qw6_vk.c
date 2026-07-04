#ifdef QW6_VULKAN

#include "qw6_vk.h"

#include <vulkan/vulkan.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    VkBuffer buffer;
    VkDeviceMemory memory;
    void *mapped;
    VkDeviceSize size;
} qw6_vk_buffer_t;

typedef struct {
    VkInstance instance;
    VkPhysicalDevice physical;
    VkDevice device;
    VkQueue queue;
    uint32_t queue_family;
    VkCommandPool command_pool;
} qw6_vk_t;

#define QW6_VK_CHECK(expr) do { \
    VkResult _rc = (expr); \
    if (_rc != VK_SUCCESS) { \
        fprintf(stderr, "qw6_vk: Vulkan call failed %d at %s:%d\n", (int)_rc, __FILE__, __LINE__); \
        return -1; \
    } \
} while (0)

static int qw6_vk_read_file(const char *path, uint32_t **out_words, size_t *out_bytes) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "qw6_vk: cannot open shader %s\n", path);
        return -1;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return -1;
    }
    long n = ftell(f);
    if (n <= 0 || (n % 4) != 0) {
        fclose(f);
        return -1;
    }
    rewind(f);
    uint32_t *buf = malloc((size_t)n);
    if (!buf) {
        fclose(f);
        return -1;
    }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) {
        free(buf);
        fclose(f);
        return -1;
    }
    fclose(f);
    *out_words = buf;
    *out_bytes = (size_t)n;
    return 0;
}

static uint32_t qw6_vk_find_memory_type(qw6_vk_t *vk, uint32_t bits,
                                        VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(vk->physical, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if ((bits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & flags) == flags)
            return i;
    }
    return UINT32_MAX;
}

static int qw6_vk_buffer_create(qw6_vk_t *vk, qw6_vk_buffer_t *b,
                                VkDeviceSize size, VkBufferUsageFlags usage) {
    memset(b, 0, sizeof(*b));
    b->size = size;
    VkBufferCreateInfo bi = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    QW6_VK_CHECK(vkCreateBuffer(vk->device, &bi, NULL, &b->buffer));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(vk->device, b->buffer, &req);
    uint32_t type = qw6_vk_find_memory_type(vk, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (type == UINT32_MAX) return -1;
    VkMemoryAllocateInfo ai = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = req.size,
        .memoryTypeIndex = type,
    };
    QW6_VK_CHECK(vkAllocateMemory(vk->device, &ai, NULL, &b->memory));
    QW6_VK_CHECK(vkBindBufferMemory(vk->device, b->buffer, b->memory, 0));
    QW6_VK_CHECK(vkMapMemory(vk->device, b->memory, 0, size, 0, &b->mapped));
    return 0;
}

static void qw6_vk_buffer_destroy(qw6_vk_t *vk, qw6_vk_buffer_t *b) {
    if (b->mapped) vkUnmapMemory(vk->device, b->memory);
    if (b->buffer) vkDestroyBuffer(vk->device, b->buffer, NULL);
    if (b->memory) vkFreeMemory(vk->device, b->memory, NULL);
    memset(b, 0, sizeof(*b));
}

static int qw6_vk_init(qw6_vk_t *vk) {
    memset(vk, 0, sizeof(*vk));
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "qw6",
        .applicationVersion = 1,
        .pEngineName = "qw6",
        .engineVersion = 1,
        .apiVersion = VK_API_VERSION_1_1,
    };
    VkInstanceCreateInfo ici = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    QW6_VK_CHECK(vkCreateInstance(&ici, NULL, &vk->instance));

    uint32_t ndev = 0;
    QW6_VK_CHECK(vkEnumeratePhysicalDevices(vk->instance, &ndev, NULL));
    if (ndev == 0) return -1;
    VkPhysicalDevice *devs = calloc(ndev, sizeof(*devs));
    if (!devs) return -1;
    QW6_VK_CHECK(vkEnumeratePhysicalDevices(vk->instance, &ndev, devs));
    vk->physical = devs[0];
    for (uint32_t i = 0; i < ndev; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devs[i], &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
            props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            vk->physical = devs[i];
            break;
        }
    }
    free(devs);

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vk->physical, &nq, NULL);
    VkQueueFamilyProperties *qprops = calloc(nq, sizeof(*qprops));
    if (!qprops) return -1;
    vkGetPhysicalDeviceQueueFamilyProperties(vk->physical, &nq, qprops);
    vk->queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++) {
        if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            vk->queue_family = i;
            break;
        }
    }
    free(qprops);
    if (vk->queue_family == UINT32_MAX) return -1;

    float priority = 1.0f;
    VkDeviceQueueCreateInfo qci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = vk->queue_family,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    VkDeviceCreateInfo dci = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qci,
    };
    QW6_VK_CHECK(vkCreateDevice(vk->physical, &dci, NULL, &vk->device));
    vkGetDeviceQueue(vk->device, vk->queue_family, 0, &vk->queue);

    VkCommandPoolCreateInfo cp = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = vk->queue_family,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
    };
    QW6_VK_CHECK(vkCreateCommandPool(vk->device, &cp, NULL, &vk->command_pool));

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(vk->physical, &props);
    fprintf(stderr, "qw6_vk: device: %s\n", props.deviceName);
    return 0;
}

static void qw6_vk_free(qw6_vk_t *vk) {
    if (vk->device) vkDeviceWaitIdle(vk->device);
    if (vk->command_pool) vkDestroyCommandPool(vk->device, vk->command_pool, NULL);
    if (vk->device) vkDestroyDevice(vk->device, NULL);
    if (vk->instance) vkDestroyInstance(vk->instance, NULL);
    memset(vk, 0, sizeof(*vk));
}

static int qw6_vk_dispatch(qw6_vk_t *vk, const char *shader_path,
                           qw6_vk_buffer_t **buffers, uint32_t n_buffers,
                           const void *push, uint32_t push_size,
                           uint32_t gx, uint32_t gy, uint32_t gz) {
    VkDescriptorSetLayoutBinding *bindings = calloc(n_buffers, sizeof(*bindings));
    if (!bindings) return -1;
    for (uint32_t i = 0; i < n_buffers; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = n_buffers,
        .pBindings = bindings,
    };
    VkDescriptorSetLayout dsl;
    QW6_VK_CHECK(vkCreateDescriptorSetLayout(vk->device, &dlci, NULL, &dsl));
    free(bindings);

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = push_size,
    };
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &dsl,
        .pushConstantRangeCount = push_size ? 1u : 0u,
        .pPushConstantRanges = push_size ? &pcr : NULL,
    };
    VkPipelineLayout pl;
    QW6_VK_CHECK(vkCreatePipelineLayout(vk->device, &plci, NULL, &pl));

    uint32_t *spv = NULL;
    size_t spv_bytes = 0;
    if (qw6_vk_read_file(shader_path, &spv, &spv_bytes) != 0) return -1;
    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spv_bytes,
        .pCode = spv,
    };
    VkShaderModule sm;
    QW6_VK_CHECK(vkCreateShaderModule(vk->device, &smci, NULL, &sm));
    free(spv);

    VkComputePipelineCreateInfo pci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = sm,
            .pName = "main",
        },
        .layout = pl,
    };
    VkPipeline pipe;
    QW6_VK_CHECK(vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &pci, NULL, &pipe));

    VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = n_buffers,
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    VkDescriptorPool pool;
    QW6_VK_CHECK(vkCreateDescriptorPool(vk->device, &dpci, NULL, &pool));
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &dsl,
    };
    VkDescriptorSet ds;
    QW6_VK_CHECK(vkAllocateDescriptorSets(vk->device, &dsai, &ds));

    VkDescriptorBufferInfo *infos = calloc(n_buffers, sizeof(*infos));
    VkWriteDescriptorSet *writes = calloc(n_buffers, sizeof(*writes));
    if (!infos || !writes) return -1;
    for (uint32_t i = 0; i < n_buffers; i++) {
        infos[i].buffer = buffers[i]->buffer;
        infos[i].offset = 0;
        infos[i].range = buffers[i]->size;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = ds;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(vk->device, n_buffers, writes, 0, NULL);
    free(infos);
    free(writes);

    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = vk->command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cb;
    QW6_VK_CHECK(vkAllocateCommandBuffers(vk->device, &cbai, &cb));
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    QW6_VK_CHECK(vkBeginCommandBuffer(cb, &begin));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    if (push_size) vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push);
    vkCmdDispatch(cb, gx, gy, gz);
    QW6_VK_CHECK(vkEndCommandBuffer(cb));

    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    QW6_VK_CHECK(vkCreateFence(vk->device, &fci, NULL, &fence));
    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
    };
    QW6_VK_CHECK(vkQueueSubmit(vk->queue, 1, &submit, fence));
    QW6_VK_CHECK(vkWaitForFences(vk->device, 1, &fence, VK_TRUE, UINT64_MAX));

    vkDestroyFence(vk->device, fence, NULL);
    vkFreeCommandBuffers(vk->device, vk->command_pool, 1, &cb);
    vkDestroyDescriptorPool(vk->device, pool, NULL);
    vkDestroyPipeline(vk->device, pipe, NULL);
    vkDestroyShaderModule(vk->device, sm, NULL);
    vkDestroyPipelineLayout(vk->device, pl, NULL);
    vkDestroyDescriptorSetLayout(vk->device, dsl, NULL);
    return 0;
}

typedef struct {
    uint32_t n;
    float eps;
} qw6_vk_rmsnorm_push_t;

static int qw6_vk_rmsnorm(qw6_vk_t *vk, const float *x, const float *w,
                          float *y, uint32_t n, float eps) {
    qw6_vk_buffer_t bx, bw, by;
    if (qw6_vk_buffer_create(vk, &bx, n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &bw, n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &by, n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    memcpy(bx.mapped, x, n * sizeof(float));
    memcpy(bw.mapped, w, n * sizeof(float));
    qw6_vk_buffer_t *bufs[3] = {&bx, &bw, &by};
    qw6_vk_rmsnorm_push_t push = {.n = n, .eps = eps};
    int rc = qw6_vk_dispatch(vk, "vulkan/rmsnorm_full.spv", bufs, 3,
                             &push, sizeof(push), 1, 1, 1);
    if (rc == 0) memcpy(y, by.mapped, n * sizeof(float));
    qw6_vk_buffer_destroy(vk, &by);
    qw6_vk_buffer_destroy(vk, &bw);
    qw6_vk_buffer_destroy(vk, &bx);
    return rc;
}

typedef struct {
    uint32_t rows;
    uint32_t cols;
} qw6_vk_matvec_push_t;

static int qw6_vk_matvec_f32(qw6_vk_t *vk, const float *w, const float *x,
                             float *y, uint32_t rows, uint32_t cols) {
    qw6_vk_buffer_t bw, bx, by;
    if (qw6_vk_buffer_create(vk, &bw, (VkDeviceSize)rows * cols * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &bx, cols * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &by, rows * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    memcpy(bw.mapped, w, (size_t)rows * cols * sizeof(float));
    memcpy(bx.mapped, x, cols * sizeof(float));
    qw6_vk_buffer_t *bufs[3] = {&bw, &bx, &by};
    qw6_vk_matvec_push_t push = {.rows = rows, .cols = cols};
    int rc = qw6_vk_dispatch(vk, "vulkan/matvec_f32.spv", bufs, 3,
                             &push, sizeof(push), rows, 1, 1);
    if (rc == 0) memcpy(y, by.mapped, rows * sizeof(float));
    qw6_vk_buffer_destroy(vk, &by);
    qw6_vk_buffer_destroy(vk, &bx);
    qw6_vk_buffer_destroy(vk, &bw);
    return rc;
}

int qw6_vk_matvec_f32_host(const float *w, const float *x, float *y,
                           uint32_t rows, uint32_t cols) {
    qw6_vk_t vk;
    if (qw6_vk_init(&vk) != 0) return -1;
    int rc = qw6_vk_matvec_f32(&vk, w, x, y, rows, cols);
    qw6_vk_free(&vk);
    return rc;
}

int qw6_vk_rmsnorm_host(const float *x, const float *w, float *y,
                        uint32_t n, float eps) {
    qw6_vk_t vk;
    if (qw6_vk_init(&vk) != 0) return -1;
    int rc = qw6_vk_rmsnorm(&vk, x, w, y, n, eps);
    qw6_vk_free(&vk);
    return rc;
}

static int qw6_vk_vec_add(qw6_vk_t *vk, const float *a, const float *b,
                          float *c, uint32_t n) {
    qw6_vk_buffer_t ba, bb, bc;
    if (qw6_vk_buffer_create(vk, &ba, n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &bb, n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &bc, n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    memcpy(ba.mapped, a, n * sizeof(float));
    memcpy(bb.mapped, b, n * sizeof(float));

    VkDescriptorSetLayoutBinding bindings[3];
    memset(bindings, 0, sizeof(bindings));
    for (uint32_t i = 0; i < 3; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings,
    };
    VkDescriptorSetLayout dsl;
    QW6_VK_CHECK(vkCreateDescriptorSetLayout(vk->device, &dlci, NULL, &dsl));

    VkPushConstantRange pcr = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(uint32_t),
    };
    VkPipelineLayoutCreateInfo plci = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &dsl,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcr,
    };
    VkPipelineLayout pl;
    QW6_VK_CHECK(vkCreatePipelineLayout(vk->device, &plci, NULL, &pl));

    uint32_t *spv = NULL;
    size_t spv_bytes = 0;
    if (qw6_vk_read_file("vulkan/vec_add.spv", &spv, &spv_bytes) != 0) return -1;
    VkShaderModuleCreateInfo smci = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spv_bytes,
        .pCode = spv,
    };
    VkShaderModule sm;
    QW6_VK_CHECK(vkCreateShaderModule(vk->device, &smci, NULL, &sm));
    free(spv);

    VkComputePipelineCreateInfo pci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = sm,
            .pName = "main",
        },
        .layout = pl,
    };
    VkPipeline pipe;
    QW6_VK_CHECK(vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &pci, NULL, &pipe));

    VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 3,
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    VkDescriptorPool pool;
    QW6_VK_CHECK(vkCreateDescriptorPool(vk->device, &dpci, NULL, &pool));
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &dsl,
    };
    VkDescriptorSet ds;
    QW6_VK_CHECK(vkAllocateDescriptorSets(vk->device, &dsai, &ds));

    VkDescriptorBufferInfo infos[3] = {
        {.buffer = ba.buffer, .offset = 0, .range = ba.size},
        {.buffer = bb.buffer, .offset = 0, .range = bb.size},
        {.buffer = bc.buffer, .offset = 0, .range = bc.size},
    };
    VkWriteDescriptorSet writes[3];
    memset(writes, 0, sizeof(writes));
    for (uint32_t i = 0; i < 3; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = ds;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(vk->device, 3, writes, 0, NULL);

    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = vk->command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cb;
    QW6_VK_CHECK(vkAllocateCommandBuffers(vk->device, &cbai, &cb));
    VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    QW6_VK_CHECK(vkBeginCommandBuffer(cb, &begin));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, NULL);
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &n);
    vkCmdDispatch(cb, (n + 255) / 256, 1, 1);
    QW6_VK_CHECK(vkEndCommandBuffer(cb));

    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    QW6_VK_CHECK(vkCreateFence(vk->device, &fci, NULL, &fence));
    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
    };
    QW6_VK_CHECK(vkQueueSubmit(vk->queue, 1, &submit, fence));
    QW6_VK_CHECK(vkWaitForFences(vk->device, 1, &fence, VK_TRUE, UINT64_MAX));

    memcpy(c, bc.mapped, n * sizeof(float));

    vkDestroyFence(vk->device, fence, NULL);
    vkFreeCommandBuffers(vk->device, vk->command_pool, 1, &cb);
    vkDestroyDescriptorPool(vk->device, pool, NULL);
    vkDestroyPipeline(vk->device, pipe, NULL);
    vkDestroyShaderModule(vk->device, sm, NULL);
    vkDestroyPipelineLayout(vk->device, pl, NULL);
    vkDestroyDescriptorSetLayout(vk->device, dsl, NULL);
    qw6_vk_buffer_destroy(vk, &bc);
    qw6_vk_buffer_destroy(vk, &bb);
    qw6_vk_buffer_destroy(vk, &ba);
    return 0;
}

int qw6_vk_selftest(void) {
    qw6_vk_t vk;
    if (qw6_vk_init(&vk) != 0) return 1;

    const uint32_t n = 1024;
    float *a = calloc(n, sizeof(float));
    float *b = calloc(n, sizeof(float));
    float *c = calloc(n, sizeof(float));
    if (!a || !b || !c) return 1;
    for (uint32_t i = 0; i < n; i++) {
        a[i] = (float)i * 0.25f;
        b[i] = 2.0f - (float)i * 0.125f;
    }
    int fail = qw6_vk_vec_add(&vk, a, b, c, n) != 0;
    for (uint32_t i = 0; i < n && !fail; i++) {
        float expected = a[i] + b[i];
        if (fabsf(c[i] - expected) > 1e-6f) {
            fprintf(stderr, "qw6_vk: vec_add mismatch at %u got %.6f expected %.6f\n",
                    i, c[i], expected);
            fail = 1;
        }
    }

    const uint32_t rn = 2048;
    float *rx = calloc(rn, sizeof(float));
    float *rw = calloc(rn, sizeof(float));
    float *ry = calloc(rn, sizeof(float));
    if (!rx || !rw || !ry) fail = 1;
    if (!fail) {
        double ss = 0.0;
        for (uint32_t i = 0; i < rn; i++) {
            rx[i] = sinf((float)i * 0.013f);
            rw[i] = 0.75f + 0.0001f * (float)(i % 17);
            ss += (double)rx[i] * rx[i];
        }
        fail = qw6_vk_rmsnorm(&vk, rx, rw, ry, rn, 1e-6f) != 0;
        float scale = 1.0f / sqrtf((float)(ss / rn) + 1e-6f);
        for (uint32_t i = 0; i < rn && !fail; i++) {
            float expected = rx[i] * scale * rw[i];
            if (fabsf(ry[i] - expected) > 2e-5f) {
                fprintf(stderr, "qw6_vk: rmsnorm mismatch at %u got %.6f expected %.6f\n",
                        i, ry[i], expected);
                fail = 1;
            }
        }
    }

    const uint32_t rows = 7, cols = 19;
    float *mw = calloc(rows * cols, sizeof(float));
    float *mx = calloc(cols, sizeof(float));
    float *my = calloc(rows, sizeof(float));
    if (!mw || !mx || !my) fail = 1;
    if (!fail) {
        for (uint32_t r = 0; r < rows; r++)
            for (uint32_t col = 0; col < cols; col++)
                mw[r * cols + col] = 0.01f * (float)((int)r - (int)col);
        for (uint32_t col = 0; col < cols; col++)
            mx[col] = cosf((float)col * 0.2f);
        fail = qw6_vk_matvec_f32(&vk, mw, mx, my, rows, cols) != 0;
        for (uint32_t r = 0; r < rows && !fail; r++) {
            float expected = 0.0f;
            for (uint32_t col = 0; col < cols; col++)
                expected += mw[r * cols + col] * mx[col];
            if (fabsf(my[r] - expected) > 2e-5f) {
                fprintf(stderr, "qw6_vk: matvec_f32 mismatch row %u got %.6f expected %.6f\n",
                        r, my[r], expected);
                fail = 1;
            }
        }
    }

    free(a);
    free(b);
    free(c);
    free(rx);
    free(rw);
    free(ry);
    free(mw);
    free(mx);
    free(my);
    qw6_vk_free(&vk);
    if (fail) {
        fprintf(stderr, "qw6_vk: self-test failed\n");
        return 1;
    }
    fprintf(stderr, "qw6_vk: self-test passed\n");
    return 0;
}

#endif
