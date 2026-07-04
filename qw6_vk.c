#ifdef QW6_VULKAN

#include "qw6.h"
#include "qw6_vk.h"
#include "qw6_iq_tables.h"

#include <vulkan/vulkan.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/time.h>

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
    double timestamp_period_ns;
    bool timestamp_supported;
} qw6_vk_t;

#define QW6_VK_CHECK(expr) do { \
    VkResult _rc = (expr); \
    if (_rc != VK_SUCCESS) { \
        fprintf(stderr, "qw6_vk: Vulkan call failed %d at %s:%d\n", (int)_rc, __FILE__, __LINE__); \
        return -1; \
    } \
} while (0)

/* Forward declarations for pipeline types (defined later) */
#define QW6_VK_CACHE_MAX 32
typedef struct {
    char shader_path[128];
    uint32_t n_buffers;
    uint32_t push_size;
    VkPipelineLayout layout;
    VkDescriptorSetLayout dsl;
    VkPipeline pipeline;
    VkShaderModule sm;
    uint64_t calls;
    double cpu_setup_ms;
    double gpu_ms;
    double total_ms;
    uint64_t read_bytes;
    uint64_t write_bytes;
    int valid;
} vk_pipe_cache_t;
struct qw6_vk_pipe_s;
typedef struct qw6_vk_pipe_s qw6_vk_pipe_t;

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

static double qw6_vk_now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}

static int qw6_vk_device_override_matches(uint32_t idx, const VkPhysicalDeviceProperties *props,
                                          const char *override) {
    if (!override || !*override) return 0;
    char *end = NULL;
    long want_idx = strtol(override, &end, 10);
    if (end && *end == '\0') return (want_idx >= 0 && (uint32_t)want_idx == idx);
    if (strstr(override, props->deviceName) != NULL) return 1;
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "%04x:%04x", props->vendorID, props->deviceID);
    if (strstr(override, pattern) != NULL) return 1;
    return 0;
}

static int qw6_vk_device_score(const VkPhysicalDeviceProperties *props,
                               const VkPhysicalDeviceFeatures *features,
                               const VkQueueFamilyProperties *qprops,
                               uint32_t nq) {
    int score = 0;
    if (props->vendorID == 0x1002) score += 1000;
    if (props->vendorID == 0x1002 && strstr(props->deviceName, "BC-250")) score += 2000;
    if (props->vendorID == 0x1002 && strstr(props->deviceName, "GFX1013")) score += 1500;
    if (props->deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 400;
    if (props->deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score += 200;
    if (features->shaderInt64) score += 1;
    for (uint32_t i = 0; i < nq; i++) {
        if ((qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qprops[i].timestampValidBits > 0) {
            score += 100;
            break;
        }
    }
    return score;
}

static int qw6_vk_memory_report(qw6_vk_t *vk, const qw6_model_t *m) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(vk->physical, &mp);
    VkDeviceSize device_local = 0;
    VkDeviceSize host_visible = 0;
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++) {
        if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            device_local += mp.memoryHeaps[i].size;
        }
        if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_MULTI_INSTANCE_BIT) {
            continue;
        }
    }
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        const VkMemoryType *mt = &mp.memoryTypes[i];
        if (mt->propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
            uint32_t heap = mt->heapIndex;
            if (heap < mp.memoryHeapCount) {
                host_visible += mp.memoryHeaps[heap].size;
            }
        }
    }

    size_t kv_ctx = m->max_context > 0 ? m->max_context : QW6_DEFAULT_CTX;
    if (kv_ctx < 4096) kv_ctx = 4096;
    VkDeviceSize weights = (VkDeviceSize)m->total_weight_bytes + 512ull * 1024ull * 1024ull;
    VkDeviceSize scratch = (VkDeviceSize)(QW6_HIDDEN_SIZE * 6 + QW6_LINEAR_QKV_DIM * 2 +
                                          QW6_VOCAB_SIZE + 1 + QW6_EXPERTS_PER_TOK * 5) * sizeof(float);
    VkDeviceSize kv = (VkDeviceSize)QW6_NUM_FULL_ATTN * kv_ctx * QW6_NUM_KV_HEADS * QW6_HEAD_DIM * sizeof(float) * 2;
    VkDeviceSize dn = (VkDeviceSize)QW6_NUM_LINEAR_ATTN * QW6_NUM_VALUE_HEADS * QW6_VALUE_HEAD_DIM * QW6_VALUE_HEAD_DIM * sizeof(float);
    VkDeviceSize conv = (VkDeviceSize)QW6_NUM_LINEAR_ATTN * QW6_CONV1D_KERNEL * QW6_LINEAR_QKV_DIM * sizeof(float);
    VkDeviceSize estimated = weights + scratch + kv + dn + conv;
    fprintf(stderr,
            "qw6_vk: memory: device-local=%llu MiB host-visible=%llu MiB estimated=%llu MiB\n",
            (unsigned long long)(device_local / (1024ull * 1024ull)),
            (unsigned long long)(host_visible / (1024ull * 1024ull)),
            (unsigned long long)(estimated / (1024ull * 1024ull)));
    fprintf(stderr,
            "qw6_vk: memory: weights=%llu MiB scratch=%llu MiB kv=%llu MiB dn=%llu MiB conv=%llu MiB\n",
            (unsigned long long)(weights / (1024ull * 1024ull)),
            (unsigned long long)(scratch / (1024ull * 1024ull)),
            (unsigned long long)(kv / (1024ull * 1024ull)),
            (unsigned long long)(dn / (1024ull * 1024ull)),
            (unsigned long long)(conv / (1024ull * 1024ull)));
    if (getenv("QW6_VK_REQUIRE_DEVICE_LOCAL")) {
        if (device_local == 0 || estimated > device_local) {
            fprintf(stderr,
                    "qw6_vk: selected GPU lacks enough device-local memory for the current footprint\n");
            fprintf(stderr,
                    "qw6_vk: set QW6_VK_REQUIRE_DEVICE_LOCAL=0 to continue with the host-visible fallback layout\n");
            return -1;
        }
    }
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
    const char *device_override = getenv("QW6_VK_DEVICE");
    VkPhysicalDevice chosen = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties chosen_props;
    uint32_t chosen_queue_family = UINT32_MAX;
    uint32_t chosen_timestamp_bits = 0;
    int chosen_score = INT_MIN;
    if (device_override && *device_override) {
        fprintf(stderr, "qw6_vk: device override requested: %s\n", device_override);
    }
    for (uint32_t i = 0; i < ndev; i++) {
        VkPhysicalDeviceProperties props;
        VkPhysicalDeviceFeatures features;
        vkGetPhysicalDeviceProperties(devs[i], &props);
        vkGetPhysicalDeviceFeatures(devs[i], &features);
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nq, NULL);
        uint32_t queue_family = UINT32_MAX;
        uint32_t timestamp_bits = 0;
        VkQueueFamilyProperties *qprops = NULL;
        if (nq > 0) {
            qprops = calloc(nq, sizeof(*qprops));
            if (!qprops) {
                free(devs);
                return -1;
            }
            vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nq, qprops);
            for (uint32_t q = 0; q < nq; q++) {
                if (!(qprops[q].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
                if (queue_family == UINT32_MAX) {
                    queue_family = q;
                    timestamp_bits = qprops[q].timestampValidBits;
                }
                if (qprops[q].timestampValidBits > timestamp_bits) {
                    queue_family = q;
                    timestamp_bits = qprops[q].timestampValidBits;
                }
            }
        }
        int score = (queue_family == UINT32_MAX) ? INT_MIN / 2 :
            qw6_vk_device_score(&props, &features, qprops, nq);
        fprintf(stderr, "qw6_vk: device[%u]: %s vendor=%04x device=%04x type=%u compute=%s timestamp_bits=%u score=%d\n",
                i, props.deviceName, props.vendorID, props.deviceID, props.deviceType,
                (queue_family == UINT32_MAX) ? "no" : "yes", timestamp_bits, score);
        if (queue_family != UINT32_MAX &&
            qw6_vk_device_override_matches(i, &props, device_override)) {
            chosen = devs[i];
            chosen_props = props;
            chosen_queue_family = queue_family;
            chosen_timestamp_bits = timestamp_bits;
            chosen_score = INT_MAX;
            free(qprops);
            break;
        }
        if (score > chosen_score) {
            chosen = devs[i];
            chosen_props = props;
            chosen_queue_family = queue_family;
            chosen_timestamp_bits = timestamp_bits;
            chosen_score = score;
        }
        free(qprops);
    }
    if (chosen == VK_NULL_HANDLE || chosen_queue_family == UINT32_MAX) {
        free(devs);
        return -1;
    }
    vk->physical = chosen;
    vk->queue_family = chosen_queue_family;
    free(devs);

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

    vk->timestamp_period_ns = chosen_props.limits.timestampPeriod;
    vk->timestamp_supported = chosen_props.limits.timestampComputeAndGraphics &&
                              chosen_timestamp_bits > 0 &&
                              vk->timestamp_period_ns > 0.0;
    fprintf(stderr, "qw6_vk: selected device: %s vendor=%04x device=%04x queue_family=%u timestamp=%s\n",
            chosen_props.deviceName, chosen_props.vendorID, chosen_props.deviceID,
            vk->queue_family, vk->timestamp_supported ? "on" : "off");
    return 0;
}

static void qw6_vk_free(qw6_vk_t *vk) {
    if (vk->device) vkDeviceWaitIdle(vk->device);
    if (vk->command_pool) vkDestroyCommandPool(vk->device, vk->command_pool, NULL);
    if (vk->device) vkDestroyDevice(vk->device, NULL);
    if (vk->instance) vkDestroyInstance(vk->instance, NULL);
    memset(vk, 0, sizeof(*vk));
}

/* Uncached dispatch (legacy, still used by selftest/probes) */
static int qw6_vk_dispatch(qw6_vk_t *vk, const char *shader_path,
                           qw6_vk_buffer_t **buffers, const size_t *offsets,
                           uint32_t n_buffers,
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
        infos[i].offset = offsets ? offsets[i] : 0;
        infos[i].range = buffers[i]->size - infos[i].offset;
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

typedef struct {
    uint32_t n;
    uint32_t n_groups;
    float eps;
} qw6_vk_rmsnorm_apply_push_t;

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
        int rc = qw6_vk_dispatch(vk, "vulkan/rmsnorm_full.spv", bufs, NULL, 3,
                             &push, sizeof(push), 1, 1, 1);
    if (rc == 0) memcpy(y, by.mapped, n * sizeof(float));
    qw6_vk_buffer_destroy(vk, &by);
    qw6_vk_buffer_destroy(vk, &bw);
    qw6_vk_buffer_destroy(vk, &bx);
    return rc;
}

static int qw6_vk_rmsnorm_chunked(qw6_vk_t *vk, const float *x, const float *w,
                                  float *y, uint32_t n, float eps,
                                  uint32_t n_groups) {
    if (n == 0 || n_groups == 0) return -1;
    qw6_vk_buffer_t bx, bw, by, bs;
    if (qw6_vk_buffer_create(vk, &bx, n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &bw, n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bx);
        return -1;
    }
    if (qw6_vk_buffer_create(vk, &by, n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bw);
        qw6_vk_buffer_destroy(vk, &bx);
        return -1;
    }
    if (qw6_vk_buffer_create(vk, &bs, n_groups * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &by);
        qw6_vk_buffer_destroy(vk, &bw);
        qw6_vk_buffer_destroy(vk, &bx);
        return -1;
    }
    memcpy(bx.mapped, x, n * sizeof(float));
    memcpy(bw.mapped, w, n * sizeof(float));

    qw6_vk_buffer_t *bufs[4] = {&bx, &bw, &by, &bs};
    qw6_vk_rmsnorm_push_t reduce_push = {.n = n, .eps = eps};
        int rc = qw6_vk_dispatch(vk, "vulkan/rmsnorm.spv", bufs, NULL, 4,
                             &reduce_push, sizeof(reduce_push), n_groups, 1, 1);
    if (rc == 0) {
        qw6_vk_rmsnorm_apply_push_t apply_push = {
            .n = n,
            .n_groups = n_groups,
            .eps = eps,
        };
        rc = qw6_vk_dispatch(vk, "vulkan/rmsnorm_apply.spv", bufs, NULL, 4,
                             &apply_push, sizeof(apply_push), n_groups, 1, 1);
    }
    if (rc == 0) memcpy(y, by.mapped, n * sizeof(float));
    qw6_vk_buffer_destroy(vk, &bs);
    qw6_vk_buffer_destroy(vk, &by);
    qw6_vk_buffer_destroy(vk, &bw);
    qw6_vk_buffer_destroy(vk, &bx);
    return rc;
}

typedef struct {
    uint32_t rows;
    uint32_t cols;
} qw6_vk_matvec_push_t;

typedef struct {
    uint32_t idx;
    uint32_t val_bits;
} qw6_vk_argmax_out_t;

typedef struct {
    uint32_t n;
} qw6_vk_n_push_t;

typedef struct {
    uint32_t n_experts;
    uint32_t top_k;
} qw6_vk_moe_route_push_t;

typedef struct {
    uint32_t dim;
    uint32_t top_k;
    uint32_t has_shared;
    float shared_weight;
} qw6_vk_moe_gather_push_t;

typedef struct {
    uint32_t seq_len;
    uint32_t n_q_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
} qw6_vk_attention_gqa_push_t;

typedef struct {
    uint32_t dim;
    uint32_t kernel_size;
} qw6_vk_conv1d_push_t;

typedef struct {
    uint32_t key_heads;
    uint32_t key_dim;
    uint32_t val_heads;
    uint32_t val_dim;
    uint32_t has_beta;
} qw6_vk_deltanet_push_t;

typedef struct {
    uint32_t key_heads;
    uint32_t val_heads;
    uint32_t dim;
    float scale;
} qw6_vk_deltanet_gated_push_t;

typedef struct {
    uint32_t q_dim;
    uint32_t kv_dim;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t position;
    uint32_t q_head_dim;
    uint32_t k_head_dim;
    uint32_t q_rot;
    uint32_t k_rot;
    float theta_base;
} qw6_vk_mrope_push_t;

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
        int rc = qw6_vk_dispatch(vk, "vulkan/matvec_f32.spv", bufs, NULL, 3,
                             &push, sizeof(push), rows, 1, 1);
    if (rc == 0) memcpy(y, by.mapped, rows * sizeof(float));
    qw6_vk_buffer_destroy(vk, &by);
    qw6_vk_buffer_destroy(vk, &bx);
    qw6_vk_buffer_destroy(vk, &bw);
    return rc;
}

static int qw6_vk_matmul_q4k(qw6_vk_t *vk, const void *w, const float *x,
                             float *y, uint32_t rows, uint32_t cols) {
    if (rows == 0 || cols == 0 || (cols % 256) != 0) return -1;
    const uint32_t block_size = 144;
    VkDeviceSize w_size = (VkDeviceSize)rows * (cols / 256) * block_size;
    qw6_vk_buffer_t bw, bx, by;
    if (qw6_vk_buffer_create(vk, &bw, w_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &bx, cols * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bw);
        return -1;
    }
    if (qw6_vk_buffer_create(vk, &by, rows * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bx);
        qw6_vk_buffer_destroy(vk, &bw);
        return -1;
    }
    memcpy(bw.mapped, w, (size_t)w_size);
    memcpy(bx.mapped, x, cols * sizeof(float));
    qw6_vk_buffer_t *bufs[3] = {&bw, &bx, &by};
    qw6_vk_matvec_push_t push = {.rows = rows, .cols = cols};
        int rc = qw6_vk_dispatch(vk, "vulkan/matmul_q4k.spv", bufs, NULL, 3,
                             &push, sizeof(push), rows, 1, 1);
    if (rc == 0) memcpy(y, by.mapped, rows * sizeof(float));
    qw6_vk_buffer_destroy(vk, &by);
    qw6_vk_buffer_destroy(vk, &bx);
    qw6_vk_buffer_destroy(vk, &bw);
    return rc;
}

static int qw6_vk_matmul_q5k(qw6_vk_t *vk, const void *w, const float *x,
                             float *y, uint32_t rows, uint32_t cols) {
    if (rows == 0 || cols == 0 || (cols % 256) != 0) return -1;
    const uint32_t block_size = 176;
    VkDeviceSize w_size = (VkDeviceSize)rows * (cols / 256) * block_size;
    qw6_vk_buffer_t bw, bx, by;
    if (qw6_vk_buffer_create(vk, &bw, w_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &bx, cols * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bw);
        return -1;
    }
    if (qw6_vk_buffer_create(vk, &by, rows * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bx);
        qw6_vk_buffer_destroy(vk, &bw);
        return -1;
    }
    memcpy(bw.mapped, w, (size_t)w_size);
    memcpy(bx.mapped, x, cols * sizeof(float));
    qw6_vk_buffer_t *bufs[3] = {&bw, &bx, &by};
    qw6_vk_matvec_push_t push = {.rows = rows, .cols = cols};
        int rc = qw6_vk_dispatch(vk, "vulkan/matmul_q5k.spv", bufs, NULL, 3,
                             &push, sizeof(push), rows, 1, 1);
    if (rc == 0) memcpy(y, by.mapped, rows * sizeof(float));
    qw6_vk_buffer_destroy(vk, &by);
    qw6_vk_buffer_destroy(vk, &bx);
    qw6_vk_buffer_destroy(vk, &bw);
    return rc;
}

static int qw6_vk_matmul_q6k(qw6_vk_t *vk, const void *w, const float *x,
                             float *y, uint32_t rows, uint32_t cols) {
    if (rows == 0 || cols == 0 || (cols % 256) != 0) return -1;
    const uint32_t block_size = 210;
    VkDeviceSize w_size = (VkDeviceSize)rows * (cols / 256) * block_size;
    qw6_vk_buffer_t bw, bx, by;
    if (qw6_vk_buffer_create(vk, &bw, w_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &bx, cols * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bw);
        return -1;
    }
    if (qw6_vk_buffer_create(vk, &by, rows * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bx);
        qw6_vk_buffer_destroy(vk, &bw);
        return -1;
    }
    memcpy(bw.mapped, w, (size_t)w_size);
    memcpy(bx.mapped, x, cols * sizeof(float));
    qw6_vk_buffer_t *bufs[3] = {&bw, &bx, &by};
    qw6_vk_matvec_push_t push = {.rows = rows, .cols = cols};
        int rc = qw6_vk_dispatch(vk, "vulkan/matmul_q6k.spv", bufs, NULL, 3,
                             &push, sizeof(push), rows, 1, 1);
    if (rc == 0) memcpy(y, by.mapped, rows * sizeof(float));
    qw6_vk_buffer_destroy(vk, &by);
    qw6_vk_buffer_destroy(vk, &bx);
    qw6_vk_buffer_destroy(vk, &bw);
    return rc;
}

static int qw6_vk_matmul_iq2xxs(qw6_vk_t *vk, const void *w, const float *x,
                                float *y, uint32_t rows, uint32_t cols) {
    if (rows == 0 || cols == 0 || (cols % 256) != 0) return -1;
    const uint32_t block_size = 66;
    VkDeviceSize w_size = (VkDeviceSize)rows * (cols / 256) * block_size;
    qw6_vk_buffer_t bw, bx, by;
    if (qw6_vk_buffer_create(vk, &bw, w_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &bx, cols * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bw);
        return -1;
    }
    if (qw6_vk_buffer_create(vk, &by, rows * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bx);
        qw6_vk_buffer_destroy(vk, &bw);
        return -1;
    }
    memcpy(bw.mapped, w, (size_t)w_size);
    memcpy(bx.mapped, x, cols * sizeof(float));
    qw6_vk_buffer_t *bufs[3] = {&bw, &bx, &by};
    qw6_vk_matvec_push_t push = {.rows = rows, .cols = cols};
        int rc = qw6_vk_dispatch(vk, "vulkan/matmul_iq2xxs.spv", bufs, NULL, 3,
                             &push, sizeof(push), rows, 1, 1);
    if (rc == 0) memcpy(y, by.mapped, rows * sizeof(float));
    qw6_vk_buffer_destroy(vk, &by);
    qw6_vk_buffer_destroy(vk, &bx);
    qw6_vk_buffer_destroy(vk, &bw);
    return rc;
}

static int qw6_vk_silu_mul(qw6_vk_t *vk, const float *gate, const float *up,
                           float *out, uint32_t n) {
    qw6_vk_buffer_t bg, bu, bo;
    if (qw6_vk_buffer_create(vk, &bg, n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &bu, n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &bo, n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    memcpy(bg.mapped, gate, n * sizeof(float));
    memcpy(bu.mapped, up, n * sizeof(float));
    qw6_vk_buffer_t *bufs[3] = {&bg, &bu, &bo};
    qw6_vk_n_push_t push = {.n = n};
        int rc = qw6_vk_dispatch(vk, "vulkan/silu_mul.spv", bufs, NULL, 3,
                             &push, sizeof(push), (n + 255) / 256, 1, 1);
    if (rc == 0) memcpy(out, bo.mapped, n * sizeof(float));
    qw6_vk_buffer_destroy(vk, &bo);
    qw6_vk_buffer_destroy(vk, &bu);
    qw6_vk_buffer_destroy(vk, &bg);
    return rc;
}

static int qw6_vk_argmax(qw6_vk_t *vk, const float *x, uint32_t n,
                         uint32_t *out_idx, float *out_val) {
    qw6_vk_buffer_t bx, bout, bscratch;
    if (qw6_vk_buffer_create(vk, &bx, n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &bout, sizeof(qw6_vk_argmax_out_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &bscratch, sizeof(qw6_vk_argmax_out_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    memcpy(bx.mapped, x, n * sizeof(float));
    memset(bout.mapped, 0, sizeof(qw6_vk_argmax_out_t));
    memset(bscratch.mapped, 0, sizeof(qw6_vk_argmax_out_t));
    qw6_vk_buffer_t *bufs[3] = {&bx, &bout, &bscratch};
    qw6_vk_n_push_t push = {.n = n};
        int rc = qw6_vk_dispatch(vk, "vulkan/argmax.spv", bufs, NULL, 3,
                             &push, sizeof(push), 1, 1, 1);
    if (rc == 0) {
        const qw6_vk_argmax_out_t *o = (const qw6_vk_argmax_out_t *)bout.mapped;
        *out_idx = o->idx;
        memcpy(out_val, &o->val_bits, sizeof(*out_val));
    }
    qw6_vk_buffer_destroy(vk, &bscratch);
    qw6_vk_buffer_destroy(vk, &bout);
    qw6_vk_buffer_destroy(vk, &bx);
    return rc;
}

static int qw6_vk_sampling_greedy(qw6_vk_t *vk, const float *logits,
                                  uint32_t n, uint32_t *token) {
    if (n == 0) return -1;
    qw6_vk_buffer_t bl, bo;
    if (qw6_vk_buffer_create(vk, &bl, n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &bo, sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bl);
        return -1;
    }
    memcpy(bl.mapped, logits, n * sizeof(float));
    memset(bo.mapped, 0, sizeof(uint32_t));
    qw6_vk_buffer_t *bufs[2] = {&bl, &bo};
    qw6_vk_n_push_t push = {.n = n};
        int rc = qw6_vk_dispatch(vk, "vulkan/sampling.spv", bufs, NULL, 2,
                             &push, sizeof(push), 1, 1, 1);
    if (rc == 0) memcpy(token, bo.mapped, sizeof(uint32_t));
    qw6_vk_buffer_destroy(vk, &bo);
    qw6_vk_buffer_destroy(vk, &bl);
    return rc;
}

static int qw6_vk_attention_gqa(qw6_vk_t *vk, const float *q,
                                const float *k_cache, const float *v_cache,
                                float *out, uint32_t seq_len,
                                uint32_t n_q_heads, uint32_t n_kv_heads,
                                uint32_t head_dim) {
    if (seq_len == 0 || seq_len > 256 || n_q_heads == 0 ||
        n_kv_heads == 0 || head_dim == 0 || n_q_heads % n_kv_heads != 0)
        return -1;
    size_t q_n = (size_t)n_q_heads * head_dim;
    size_t kv_n = (size_t)seq_len * n_kv_heads * head_dim;
    qw6_vk_buffer_t bq, bk, bv, bo;
    if (qw6_vk_buffer_create(vk, &bq, q_n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &bk, kv_n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bq);
        return -1;
    }
    if (qw6_vk_buffer_create(vk, &bv, kv_n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bk);
        qw6_vk_buffer_destroy(vk, &bq);
        return -1;
    }
    if (qw6_vk_buffer_create(vk, &bo, q_n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bv);
        qw6_vk_buffer_destroy(vk, &bk);
        qw6_vk_buffer_destroy(vk, &bq);
        return -1;
    }
    memcpy(bq.mapped, q, q_n * sizeof(float));
    memcpy(bk.mapped, k_cache, kv_n * sizeof(float));
    memcpy(bv.mapped, v_cache, kv_n * sizeof(float));
    qw6_vk_buffer_t *bufs[4] = {&bq, &bk, &bv, &bo};
    qw6_vk_attention_gqa_push_t push = {
        .seq_len = seq_len,
        .n_q_heads = n_q_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
    };
        int rc = qw6_vk_dispatch(vk, "vulkan/attention_gqa.spv", bufs, NULL, 4,
                             &push, sizeof(push), n_q_heads, 1, 1);
    if (rc == 0) memcpy(out, bo.mapped, q_n * sizeof(float));
    qw6_vk_buffer_destroy(vk, &bo);
    qw6_vk_buffer_destroy(vk, &bv);
    qw6_vk_buffer_destroy(vk, &bk);
    qw6_vk_buffer_destroy(vk, &bq);
    return rc;
}

static int qw6_vk_moe_route(qw6_vk_t *vk, const float *logits,
                            uint32_t *indices, float *weights,
                            uint32_t n_experts, uint32_t top_k) {
    if (n_experts == 0 || n_experts > 256 || top_k == 0 || top_k > n_experts || top_k > 8) return -1;

    qw6_vk_buffer_t blogits, bindices, bweights;
    if (qw6_vk_buffer_create(vk, &blogits, n_experts * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &bindices, top_k * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &bweights, top_k * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    memcpy(blogits.mapped, logits, n_experts * sizeof(float));
    memset(bindices.mapped, 0, top_k * sizeof(uint32_t));
    memset(bweights.mapped, 0, top_k * sizeof(float));
    qw6_vk_buffer_t *bufs[3] = {&blogits, &bindices, &bweights};
    qw6_vk_moe_route_push_t push = {.n_experts = n_experts, .top_k = top_k};
        int rc = qw6_vk_dispatch(vk, "vulkan/moe_route.spv", bufs, NULL, 3,
                             &push, sizeof(push), 1, 1, 1);
    if (rc == 0) {
        memcpy(indices, bindices.mapped, top_k * sizeof(uint32_t));
        memcpy(weights, bweights.mapped, top_k * sizeof(float));
    }
    qw6_vk_buffer_destroy(vk, &bweights);
    qw6_vk_buffer_destroy(vk, &bindices);
    qw6_vk_buffer_destroy(vk, &blogits);
    return rc;
}

static int qw6_vk_moe_gather(qw6_vk_t *vk, const float *expert_out,
                             const float *expert_weights,
                             const float *shared_out, float shared_weight,
                             float *out, uint32_t dim, uint32_t top_k) {
    if (dim == 0 || top_k == 0 || top_k > 8) return -1;
    qw6_vk_buffer_t be, bw, bs, bo;
    if (qw6_vk_buffer_create(vk, &be, (VkDeviceSize)top_k * dim * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &bw, top_k * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &be);
        return -1;
    }
    if (qw6_vk_buffer_create(vk, &bs, dim * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bw);
        qw6_vk_buffer_destroy(vk, &be);
        return -1;
    }
    if (qw6_vk_buffer_create(vk, &bo, dim * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bs);
        qw6_vk_buffer_destroy(vk, &bw);
        qw6_vk_buffer_destroy(vk, &be);
        return -1;
    }
    memcpy(be.mapped, expert_out, (size_t)top_k * dim * sizeof(float));
    memcpy(bw.mapped, expert_weights, top_k * sizeof(float));
    if (shared_out) memcpy(bs.mapped, shared_out, dim * sizeof(float));
    else memset(bs.mapped, 0, dim * sizeof(float));
    qw6_vk_buffer_t *bufs[4] = {&be, &bw, &bs, &bo};
    qw6_vk_moe_gather_push_t push = {
        .dim = dim,
        .top_k = top_k,
        .has_shared = shared_out ? 1u : 0u,
        .shared_weight = shared_weight,
    };
        int rc = qw6_vk_dispatch(vk, "vulkan/moe_gather.spv", bufs, NULL, 4,
                             &push, sizeof(push), (dim + 255) / 256, 1, 1);
    if (rc == 0) memcpy(out, bo.mapped, dim * sizeof(float));
    qw6_vk_buffer_destroy(vk, &bo);
    qw6_vk_buffer_destroy(vk, &bs);
    qw6_vk_buffer_destroy(vk, &bw);
    qw6_vk_buffer_destroy(vk, &be);
    return rc;
}

static int qw6_vk_deltanet_conv1d(qw6_vk_t *vk, const float *x,
                                  const uint16_t *w, float *out,
                                  uint32_t dim, uint32_t kernel_size) {
    if (dim == 0 || kernel_size == 0) return -1;
    uint32_t n_weights = dim * kernel_size;
    uint32_t n_words = (n_weights + 1) / 2;
    uint32_t *packed = calloc(n_words, sizeof(uint32_t));
    if (!packed) return -1;
    for (uint32_t i = 0; i < n_weights; i++) {
        uint32_t shift = (i & 1) ? 16u : 0u;
        packed[i >> 1] |= (uint32_t)w[i] << shift;
    }

    qw6_vk_buffer_t bx, bw, bo;
    if (qw6_vk_buffer_create(vk, &bx, dim * kernel_size * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        free(packed);
        return -1;
    }
    if (qw6_vk_buffer_create(vk, &bw, n_words * sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bx);
        free(packed);
        return -1;
    }
    if (qw6_vk_buffer_create(vk, &bo, dim * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bw);
        qw6_vk_buffer_destroy(vk, &bx);
        free(packed);
        return -1;
    }
    memcpy(bx.mapped, x, dim * kernel_size * sizeof(float));
    memcpy(bw.mapped, packed, n_words * sizeof(uint32_t));
    free(packed);
    qw6_vk_buffer_t *bufs[3] = {&bx, &bw, &bo};
    qw6_vk_conv1d_push_t push = {.dim = dim, .kernel_size = kernel_size};
        int rc = qw6_vk_dispatch(vk, "vulkan/deltanet_conv1d.spv", bufs, NULL, 3,
                             &push, sizeof(push), (dim + 255) / 256, 1, 1);
    if (rc == 0) memcpy(out, bo.mapped, dim * sizeof(float));
    qw6_vk_buffer_destroy(vk, &bo);
    qw6_vk_buffer_destroy(vk, &bw);
    qw6_vk_buffer_destroy(vk, &bx);
    return rc;
}

static int qw6_vk_deltanet_retrieve(qw6_vk_t *vk, const float *state,
                                    const float *query, float *out,
                                    uint32_t key_heads, uint32_t key_dim,
                                    uint32_t val_heads, uint32_t val_dim) {
    if (key_heads == 0 || key_dim == 0 || val_heads == 0 || val_dim == 0) return -1;
    size_t state_n = (size_t)key_heads * key_dim * val_heads * val_dim;
    size_t query_n = (size_t)key_heads * key_dim;
    size_t out_n = (size_t)val_heads * val_dim;

    qw6_vk_buffer_t bs, bq, bo;
    if (qw6_vk_buffer_create(vk, &bs, state_n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &bq, query_n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bs);
        return -1;
    }
    if (qw6_vk_buffer_create(vk, &bo, out_n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bq);
        qw6_vk_buffer_destroy(vk, &bs);
        return -1;
    }
    memcpy(bs.mapped, state, state_n * sizeof(float));
    memcpy(bq.mapped, query, query_n * sizeof(float));
    qw6_vk_buffer_t *bufs[3] = {&bs, &bq, &bo};
    qw6_vk_deltanet_push_t push = {
        .key_heads = key_heads,
        .key_dim = key_dim,
        .val_heads = val_heads,
        .val_dim = val_dim,
        .has_beta = 0,
    };
        int rc = qw6_vk_dispatch(vk, "vulkan/deltanet_retrieve.spv", bufs, NULL, 3,
                             &push, sizeof(push), (uint32_t)((out_n + 255) / 256), 1, 1);
    if (rc == 0) memcpy(out, bo.mapped, out_n * sizeof(float));
    qw6_vk_buffer_destroy(vk, &bo);
    qw6_vk_buffer_destroy(vk, &bq);
    qw6_vk_buffer_destroy(vk, &bs);
    return rc;
}

static int qw6_vk_deltanet_update(qw6_vk_t *vk, float *state,
                                  const float *key, const float *value,
                                  const float *query, const float *beta,
                                  uint32_t key_heads, uint32_t key_dim,
                                  uint32_t val_heads, uint32_t val_dim) {
    if (key_heads == 0 || key_dim == 0 || val_heads == 0 || val_dim == 0) return -1;
    size_t state_n = (size_t)key_heads * key_dim * val_heads * val_dim;
    size_t key_n = (size_t)key_heads * key_dim;
    size_t value_n = (size_t)val_heads * val_dim;

    qw6_vk_buffer_t bs, bk, bv, bq, bbeta;
    if (qw6_vk_buffer_create(vk, &bs, state_n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &bk, key_n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bs);
        return -1;
    }
    if (qw6_vk_buffer_create(vk, &bv, value_n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bk);
        qw6_vk_buffer_destroy(vk, &bs);
        return -1;
    }
    if (qw6_vk_buffer_create(vk, &bq, key_n * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bv);
        qw6_vk_buffer_destroy(vk, &bk);
        qw6_vk_buffer_destroy(vk, &bs);
        return -1;
    }
    if (qw6_vk_buffer_create(vk, &bbeta, key_heads * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bq);
        qw6_vk_buffer_destroy(vk, &bv);
        qw6_vk_buffer_destroy(vk, &bk);
        qw6_vk_buffer_destroy(vk, &bs);
        return -1;
    }
    memcpy(bs.mapped, state, state_n * sizeof(float));
    memcpy(bk.mapped, key, key_n * sizeof(float));
    memcpy(bv.mapped, value, value_n * sizeof(float));
    memcpy(bq.mapped, query, key_n * sizeof(float));
    if (beta) memcpy(bbeta.mapped, beta, key_heads * sizeof(float));
    else memset(bbeta.mapped, 0, key_heads * sizeof(float));
    qw6_vk_buffer_t *bufs[5] = {&bs, &bk, &bv, &bq, &bbeta};
    qw6_vk_deltanet_push_t push = {
        .key_heads = key_heads,
        .key_dim = key_dim,
        .val_heads = val_heads,
        .val_dim = val_dim,
        .has_beta = beta ? 1u : 0u,
    };
        int rc = qw6_vk_dispatch(vk, "vulkan/deltanet_update.spv", bufs, NULL, 5,
                             &push, sizeof(push), (uint32_t)((state_n + 255) / 256), 1, 1);
    if (rc == 0) memcpy(state, bs.mapped, state_n * sizeof(float));
    qw6_vk_buffer_destroy(vk, &bbeta);
    qw6_vk_buffer_destroy(vk, &bq);
    qw6_vk_buffer_destroy(vk, &bv);
    qw6_vk_buffer_destroy(vk, &bk);
    qw6_vk_buffer_destroy(vk, &bs);
    return rc;
}

static int qw6_vk_mrope(qw6_vk_t *vk, float *q, float *k,
                        uint32_t q_dim, uint32_t kv_dim,
                        uint32_t n_heads, uint32_t n_kv_heads,
                        uint32_t position, uint32_t rotary_dim) {
    if (q_dim == 0 || kv_dim == 0 || n_heads == 0 || n_kv_heads == 0) return -1;
    if (q_dim % n_heads != 0 || kv_dim % n_kv_heads != 0) return -1;

    uint32_t q_head_dim = q_dim / n_heads;
    uint32_t k_head_dim = kv_dim / n_kv_heads;
    uint32_t q_rot = rotary_dim < q_head_dim ? rotary_dim : q_head_dim;
    uint32_t k_rot = rotary_dim < k_head_dim ? rotary_dim : k_head_dim;
    uint32_t q_pairs = n_heads * (q_rot / 2);
    uint32_t k_pairs = n_kv_heads * (k_rot / 2);
    uint32_t total_pairs = q_pairs + k_pairs;
    if (total_pairs == 0) return 0;

    qw6_vk_buffer_t bq, bk;
    if (qw6_vk_buffer_create(vk, &bq, q_dim * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &bk, kv_dim * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    memcpy(bq.mapped, q, q_dim * sizeof(float));
    memcpy(bk.mapped, k, kv_dim * sizeof(float));
    qw6_vk_buffer_t *bufs[2] = {&bq, &bk};
    qw6_vk_mrope_push_t push = {
        .q_dim = q_dim,
        .kv_dim = kv_dim,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .position = position,
        .q_head_dim = q_head_dim,
        .k_head_dim = k_head_dim,
        .q_rot = q_rot,
        .k_rot = k_rot,
        .theta_base = QW6_ROPE_THETA,
    };
        int rc = qw6_vk_dispatch(vk, "vulkan/rope_mrope.spv", bufs, NULL, 2,
                             &push, sizeof(push), (total_pairs + 255) / 256, 1, 1);
    if (rc == 0) {
        memcpy(q, bq.mapped, q_dim * sizeof(float));
        memcpy(k, bk.mapped, kv_dim * sizeof(float));
    }
    qw6_vk_buffer_destroy(vk, &bk);
    qw6_vk_buffer_destroy(vk, &bq);
    return rc;
}

static int qw6_vk_mtp_draft(qw6_vk_t *vk, const float *w, const float *hidden,
                            float *logits, uint32_t vocab_size,
                            uint32_t hidden_size) {
    if (vocab_size == 0 || hidden_size == 0) return -1;
    qw6_vk_buffer_t bw, bh, bl;
    if (qw6_vk_buffer_create(vk, &bw, (VkDeviceSize)vocab_size * hidden_size * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) return -1;
    if (qw6_vk_buffer_create(vk, &bh, hidden_size * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bw);
        return -1;
    }
    if (qw6_vk_buffer_create(vk, &bl, vocab_size * sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_buffer_destroy(vk, &bh);
        qw6_vk_buffer_destroy(vk, &bw);
        return -1;
    }
    memcpy(bw.mapped, w, (size_t)vocab_size * hidden_size * sizeof(float));
    memcpy(bh.mapped, hidden, hidden_size * sizeof(float));
    qw6_vk_buffer_t *bufs[3] = {&bw, &bh, &bl};
    qw6_vk_matvec_push_t push = {.rows = vocab_size, .cols = hidden_size};
        int rc = qw6_vk_dispatch(vk, "vulkan/mtp_draft.spv", bufs, NULL, 3,
                             &push, sizeof(push), vocab_size, 1, 1);
    if (rc == 0) memcpy(logits, bl.mapped, vocab_size * sizeof(float));
    qw6_vk_buffer_destroy(vk, &bl);
    qw6_vk_buffer_destroy(vk, &bh);
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

int qw6_vk_matmul_q4k_host(const void *w, const float *x, float *y,
                           uint32_t rows, uint32_t cols) {
    qw6_vk_t vk;
    if (qw6_vk_init(&vk) != 0) return -1;
    int rc = qw6_vk_matmul_q4k(&vk, w, x, y, rows, cols);
    qw6_vk_free(&vk);
    return rc;
}

int qw6_vk_matmul_q5k_host(const void *w, const float *x, float *y,
                           uint32_t rows, uint32_t cols) {
    qw6_vk_t vk;
    if (qw6_vk_init(&vk) != 0) return -1;
    int rc = qw6_vk_matmul_q5k(&vk, w, x, y, rows, cols);
    qw6_vk_free(&vk);
    return rc;
}

int qw6_vk_matmul_q6k_host(const void *w, const float *x, float *y,
                           uint32_t rows, uint32_t cols) {
    qw6_vk_t vk;
    if (qw6_vk_init(&vk) != 0) return -1;
    int rc = qw6_vk_matmul_q6k(&vk, w, x, y, rows, cols);
    qw6_vk_free(&vk);
    return rc;
}

int qw6_vk_matmul_iq2xxs_host(const void *w, const float *x, float *y,
                              uint32_t rows, uint32_t cols) {
    qw6_vk_t vk;
    if (qw6_vk_init(&vk) != 0) return -1;
    int rc = qw6_vk_matmul_iq2xxs(&vk, w, x, y, rows, cols);
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

int qw6_vk_rmsnorm_chunked_host(const float *x, const float *w, float *y,
                                uint32_t n, float eps, uint32_t n_groups) {
    qw6_vk_t vk;
    if (qw6_vk_init(&vk) != 0) return -1;
    int rc = qw6_vk_rmsnorm_chunked(&vk, x, w, y, n, eps, n_groups);
    qw6_vk_free(&vk);
    return rc;
}

int qw6_vk_silu_mul_host(const float *gate, const float *up, float *out,
                         uint32_t n) {
    qw6_vk_t vk;
    if (qw6_vk_init(&vk) != 0) return -1;
    int rc = qw6_vk_silu_mul(&vk, gate, up, out, n);
    qw6_vk_free(&vk);
    return rc;
}

int qw6_vk_argmax_host(const float *x, uint32_t n, uint32_t *out_idx,
                       float *out_val) {
    qw6_vk_t vk;
    if (qw6_vk_init(&vk) != 0) return -1;
    int rc = qw6_vk_argmax(&vk, x, n, out_idx, out_val);
    qw6_vk_free(&vk);
    return rc;
}

int qw6_vk_sampling_greedy_host(const float *logits, uint32_t n,
                                uint32_t *token) {
    qw6_vk_t vk;
    if (qw6_vk_init(&vk) != 0) return -1;
    int rc = qw6_vk_sampling_greedy(&vk, logits, n, token);
    qw6_vk_free(&vk);
    return rc;
}

int qw6_vk_attention_gqa_host(const float *q, const float *k_cache,
                              const float *v_cache, float *out,
                              uint32_t seq_len, uint32_t n_q_heads,
                              uint32_t n_kv_heads, uint32_t head_dim) {
    qw6_vk_t vk;
    if (qw6_vk_init(&vk) != 0) return -1;
    int rc = qw6_vk_attention_gqa(&vk, q, k_cache, v_cache, out, seq_len,
                                  n_q_heads, n_kv_heads, head_dim);
    qw6_vk_free(&vk);
    return rc;
}

int qw6_vk_moe_route_host(const float *logits, uint32_t *indices,
                          float *weights, uint32_t n_experts,
                          uint32_t top_k) {
    qw6_vk_t vk;
    if (qw6_vk_init(&vk) != 0) return -1;
    int rc = qw6_vk_moe_route(&vk, logits, indices, weights, n_experts, top_k);
    qw6_vk_free(&vk);
    return rc;
}

int qw6_vk_moe_gather_host(const float *expert_out,
                           const float *expert_weights,
                           const float *shared_out, float shared_weight,
                           float *out, uint32_t dim, uint32_t top_k) {
    qw6_vk_t vk;
    if (qw6_vk_init(&vk) != 0) return -1;
    int rc = qw6_vk_moe_gather(&vk, expert_out, expert_weights, shared_out,
                               shared_weight, out, dim, top_k);
    qw6_vk_free(&vk);
    return rc;
}

int qw6_vk_deltanet_conv1d_host(const float *x, const uint16_t *w, float *out,
                                uint32_t dim, uint32_t kernel_size) {
    qw6_vk_t vk;
    if (qw6_vk_init(&vk) != 0) return -1;
    int rc = qw6_vk_deltanet_conv1d(&vk, x, w, out, dim, kernel_size);
    qw6_vk_free(&vk);
    return rc;
}

int qw6_vk_deltanet_retrieve_host(const float *state, const float *query,
                                  float *out, uint32_t key_heads,
                                  uint32_t key_dim, uint32_t val_heads,
                                  uint32_t val_dim) {
    qw6_vk_t vk;
    if (qw6_vk_init(&vk) != 0) return -1;
    int rc = qw6_vk_deltanet_retrieve(&vk, state, query, out, key_heads,
                                      key_dim, val_heads, val_dim);
    qw6_vk_free(&vk);
    return rc;
}

int qw6_vk_deltanet_update_host(float *state, const float *key,
                                const float *value, const float *query,
                                const float *beta, uint32_t key_heads,
                                uint32_t key_dim, uint32_t val_heads,
                                uint32_t val_dim) {
    qw6_vk_t vk;
    if (qw6_vk_init(&vk) != 0) return -1;
    int rc = qw6_vk_deltanet_update(&vk, state, key, value, query, beta,
                                    key_heads, key_dim, val_heads, val_dim);
    qw6_vk_free(&vk);
    return rc;
}

int qw6_vk_mrope_host(float *q, float *k, uint32_t q_dim, uint32_t kv_dim,
                      uint32_t n_heads, uint32_t n_kv_heads,
                      uint32_t position, uint32_t rotary_dim) {
    qw6_vk_t vk;
    if (qw6_vk_init(&vk) != 0) return -1;
    int rc = qw6_vk_mrope(&vk, q, k, q_dim, kv_dim, n_heads, n_kv_heads,
                          position, rotary_dim);
    qw6_vk_free(&vk);
    return rc;
}

int qw6_vk_mtp_draft_host(const float *w, const float *hidden, float *logits,
                          uint32_t vocab_size, uint32_t hidden_size) {
    qw6_vk_t vk;
    if (qw6_vk_init(&vk) != 0) return -1;
    int rc = qw6_vk_mtp_draft(&vk, w, hidden, logits, vocab_size, hidden_size);
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

static void qw6_vk_put_u16(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xffu);
    p[1] = (uint8_t)(v >> 8);
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

    const uint32_t crn = 4099, crg = 8;
    float *crx = calloc(crn, sizeof(float));
    float *crw = calloc(crn, sizeof(float));
    float *cry = calloc(crn, sizeof(float));
    if (!crx || !crw || !cry) fail = 1;
    if (!fail) {
        double ss = 0.0;
        for (uint32_t i = 0; i < crn; i++) {
            crx[i] = sinf((float)i * 0.009f) + 0.1f * cosf((float)i * 0.017f);
            crw[i] = 0.5f + 0.0002f * (float)(i % 23);
            ss += (double)crx[i] * crx[i];
        }
        fail = qw6_vk_rmsnorm_chunked(&vk, crx, crw, cry, crn, 1e-6f, crg) != 0;
        float scale = 1.0f / sqrtf((float)(ss / crn) + 1e-6f);
        for (uint32_t i = 0; i < crn && !fail; i++) {
            float expected = crx[i] * scale * crw[i];
            if (fabsf(cry[i] - expected) > 2e-5f) {
                fprintf(stderr, "qw6_vk: rmsnorm_chunked mismatch at %u got %.6f expected %.6f\n",
                        i, cry[i], expected);
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

    const uint32_t q4_rows = 3, q4_cols = 256, q4_block = 144;
    uint8_t *q4w = calloc((size_t)q4_rows * q4_block, 1);
    float *q4x = calloc(q4_cols, sizeof(float));
    float *q4y = calloc(q4_rows, sizeof(float));
    float q4_expected[3] = {0};
    if (!q4w || !q4x || !q4y) fail = 1;
    if (!fail) {
        for (uint32_t c = 0; c < q4_cols; c++)
            q4x[c] = sinf((float)c * 0.031f);
        for (uint32_t r = 0; r < q4_rows; r++) {
            uint8_t *blk = q4w + (size_t)r * q4_block;
            qw6_vk_put_u16(blk + 0, 0x3c00); /* d = 1.0 */
            qw6_vk_put_u16(blk + 2, 0x0000); /* dmin = 0.0 */
            for (uint32_t j = 0; j < 8; j++) {
                uint8_t sc = 1;
                if (j < 4) blk[4 + j] = sc;
                else blk[8 + j] = sc;
            }
            for (uint32_t c = 0; c < q4_cols; c++) {
                uint32_t group64 = c >> 6;
                uint32_t sub = (c >> 5) & 1u;
                uint32_t off = 16 + group64 * 32 + (c & 31u);
                uint8_t qv = (uint8_t)((r * 3 + c * 5 + 7) & 15u);
                if (sub == 0) blk[off] = (uint8_t)((blk[off] & 0xf0u) | qv);
                else blk[off] = (uint8_t)((blk[off] & 0x0fu) | (qv << 4));
                uint32_t sj = group64 * 2 + sub;
                uint8_t sc = sj < 4
                    ? (uint8_t)(blk[4 + sj] & 63u)
                    : (uint8_t)((blk[8 + sj] & 0x0fu) | ((blk[sj] >> 6) << 4));
                q4_expected[r] += (float)sc * (float)qv * q4x[c];
            }
        }
        fail = qw6_vk_matmul_q4k(&vk, q4w, q4x, q4y, q4_rows, q4_cols) != 0;
        for (uint32_t r = 0; r < q4_rows && !fail; r++) {
            if (fabsf(q4y[r] - q4_expected[r]) > 1e-2f) {
                fprintf(stderr, "qw6_vk: matmul_q4k mismatch row %u got %.6f expected %.6f\n",
                        r, q4y[r], q4_expected[r]);
                fail = 1;
            }
        }
    }

    const uint32_t q5_rows = 3, q5_cols = 256, q5_block = 176;
    uint8_t *q5w = calloc((size_t)q5_rows * q5_block, 1);
    float *q5x = calloc(q5_cols, sizeof(float));
    float *q5y = calloc(q5_rows, sizeof(float));
    float q5_expected[3] = {0};
    if (!q5w || !q5x || !q5y) fail = 1;
    if (!fail) {
        for (uint32_t c = 0; c < q5_cols; c++)
            q5x[c] = cosf((float)c * 0.027f);
        for (uint32_t r = 0; r < q5_rows; r++) {
            uint8_t *blk = q5w + (size_t)r * q5_block;
            qw6_vk_put_u16(blk + 0, 0x3c00); /* d = 1.0 */
            qw6_vk_put_u16(blk + 2, 0x0000); /* dmin = 0.0 */
            for (uint32_t j = 0; j < 8; j++) {
                uint8_t sc = 1;
                if (j < 4) blk[4 + j] = sc;
                else blk[8 + j] = sc;
            }
            for (uint32_t c = 0; c < q5_cols; c++) {
                uint32_t group64 = c >> 6;
                uint32_t sub = (c >> 5) & 1u;
                uint32_t lane = c & 31u;
                uint32_t off = 48 + group64 * 32 + lane;
                uint8_t qv = (uint8_t)((r * 7 + c * 3 + 5) & 31u);
                if (sub == 0) blk[off] = (uint8_t)((blk[off] & 0xf0u) | (qv & 15u));
                else blk[off] = (uint8_t)((blk[off] & 0x0fu) | ((qv & 15u) << 4));
                if (qv & 16u) blk[16 + lane] |= (uint8_t)(1u << (group64 * 2 + sub));
            }
            const uint8_t *ql = blk + 48;
            const uint8_t *qh = blk + 16;
            uint8_t u1 = 1, u2 = 2;
            for (uint32_t group64 = 0; group64 < 4; group64++) {
                for (uint32_t lane = 0; lane < 32; lane++) {
                    uint32_t qv = (ql[lane] & 0x0fu) + ((qh[lane] & u1) ? 16u : 0u);
                    q5_expected[r] += (float)qv * q5x[group64 * 64 + lane];
                }
                for (uint32_t lane = 0; lane < 32; lane++) {
                    uint32_t qv = (ql[lane] >> 4) + ((qh[lane] & u2) ? 16u : 0u);
                    q5_expected[r] += (float)qv * q5x[group64 * 64 + 32 + lane];
                }
                ql += 32;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        fail = qw6_vk_matmul_q5k(&vk, q5w, q5x, q5y, q5_rows, q5_cols) != 0;
        for (uint32_t r = 0; r < q5_rows && !fail; r++) {
            if (fabsf(q5y[r] - q5_expected[r]) > 1e-2f) {
                fprintf(stderr, "qw6_vk: matmul_q5k mismatch row %u got %.6f expected %.6f\n",
                        r, q5y[r], q5_expected[r]);
                fail = 1;
            }
        }
    }

    const uint32_t q6_rows = 3, q6_cols = 256, q6_block = 210;
    uint8_t *q6w = calloc((size_t)q6_rows * q6_block, 1);
    float *q6x = calloc(q6_cols, sizeof(float));
    float *q6y = calloc(q6_rows, sizeof(float));
    float q6_expected[3] = {0};
    if (!q6w || !q6x || !q6y) fail = 1;
    if (!fail) {
        for (uint32_t c = 0; c < q6_cols; c++)
            q6x[c] = sinf((float)c * 0.019f) + 0.2f * cosf((float)c * 0.013f);
        for (uint32_t r = 0; r < q6_rows; r++) {
            uint8_t *blk = q6w + (size_t)r * q6_block;
            qw6_vk_put_u16(blk + 208, 0x3c00); /* d = 1.0 */
            for (uint32_t i = 0; i < 16; i++) blk[192 + i] = 1;
            for (uint32_t c = 0; c < q6_cols; c++) {
                uint32_t half = c >> 7;
                uint32_t in_half = c & 127u;
                uint32_t lane = in_half & 31u;
                uint32_t quad = in_half >> 5;
                int qv = (int)((r * 11 + c * 7 + 9) & 63u) - 32;
                uint32_t u = (uint32_t)(qv + 32);
                uint32_t ql_off = half * 64 + lane + ((quad == 1 || quad == 3) ? 32 : 0);
                uint32_t qh_off = 128 + half * 32 + lane;
                if (quad < 2) blk[ql_off] = (uint8_t)((blk[ql_off] & 0xf0u) | (u & 15u));
                else blk[ql_off] = (uint8_t)((blk[ql_off] & 0x0fu) | ((u & 15u) << 4));
                blk[qh_off] |= (uint8_t)(((u >> 4) & 3u) << (quad * 2));
                q6_expected[r] += (float)qv * q6x[c];
            }
        }
        fail = qw6_vk_matmul_q6k(&vk, q6w, q6x, q6y, q6_rows, q6_cols) != 0;
        for (uint32_t r = 0; r < q6_rows && !fail; r++) {
            if (fabsf(q6y[r] - q6_expected[r]) > 1e-2f) {
                fprintf(stderr, "qw6_vk: matmul_q6k mismatch row %u got %.6f expected %.6f\n",
                        r, q6y[r], q6_expected[r]);
                fail = 1;
            }
        }
    }

    /* IQ2_XXS matmul self-test */
    {
        const uint32_t iq_rows = 3, iq_cols = 256, iq_block = 66;
        uint8_t *iqw = calloc((size_t)iq_rows * iq_block, 1);
        float *iqx = calloc(iq_cols, sizeof(float));
        float *iqy = calloc(iq_rows, sizeof(float));
        float iq_expected[3] = {0};
        if (!iqw || !iqx || !iqy) fail = 1;
        if (!fail) {
            for (uint32_t c = 0; c < iq_cols; c++)
                iqx[c] = sinf((float)c * 0.029f) + 0.15f * cosf((float)c * 0.017f);
            for (uint32_t r = 0; r < iq_rows; r++) {
                uint8_t *blk = iqw + (size_t)r * iq_block;
                float d_val = 1.0f + 0.5f * (float)r;
                qw6_vk_put_u16(blk + 0, 0x3c00); /* d = 1.0 (overwrite below for r>0) */
                if (r == 1) qw6_vk_put_u16(blk + 0, 0x3e00); /* d = 1.5 */
                if (r == 2) qw6_vk_put_u16(blk + 0, 0x4000); /* d = 2.0 */
                d_val = (r == 0) ? 1.0f : (r == 1) ? 1.5f : 2.0f;
                /* Fill all 8 ib32 sub-blocks with grid_idx=0, sign_idx=0, scale_extra=0 */
                for (uint32_t ib = 0; ib < 8; ib++) {
                    uint32_t ib_off = 2 + ib * 8;
                    /* aux32[0] = 4 grid indices all 0 */
                    memset(blk + ib_off, 0, 4);
                    /* aux32[1] = 0 (sign idx all 0, scale_extra=0) */
                    memset(blk + ib_off + 4, 0, 4);
                }
                /* Compute expected: each dequantized value = d * 0.125 * 8 = d */
                for (uint32_t c = 0; c < iq_cols; c++)
                    iq_expected[r] += d_val * iqx[c];
            }
            fail = qw6_vk_matmul_iq2xxs(&vk, iqw, iqx, iqy, iq_rows, iq_cols) != 0;
            for (uint32_t r = 0; r < iq_rows && !fail; r++) {
                if (fabsf(iqy[r] - iq_expected[r]) > 2e-2f) {
                    fprintf(stderr,
                            "qw6_vk: matmul_iq2xxs mismatch row %u got %.6f expected %.6f\n",
                            r, iqy[r], iq_expected[r]);
                    fail = 1;
                }
            }
        }
        free(iqy);
        free(iqx);
        free(iqw);
    }

    const uint32_t sn = 513;
    float *sg = calloc(sn, sizeof(float));
    float *su = calloc(sn, sizeof(float));
    float *so = calloc(sn, sizeof(float));
    if (!sg || !su || !so) fail = 1;
    if (!fail) {
        for (uint32_t i = 0; i < sn; i++) {
            sg[i] = ((float)(int)(i % 31) - 15.0f) * 0.125f;
            su[i] = cosf((float)i * 0.031f);
        }
        fail = qw6_vk_silu_mul(&vk, sg, su, so, sn) != 0;
        for (uint32_t i = 0; i < sn && !fail; i++) {
            float expected = (sg[i] / (1.0f + expf(-sg[i]))) * su[i];
            if (fabsf(so[i] - expected) > 2e-5f) {
                fprintf(stderr, "qw6_vk: silu_mul mismatch at %u got %.6f expected %.6f\n",
                        i, so[i], expected);
                fail = 1;
            }
        }
    }

    const uint32_t an = 4097;
    float *ax = calloc(an, sizeof(float));
    if (!ax) fail = 1;
    if (!fail) {
        uint32_t got_idx = 0;
        float got_val = 0.0f;
        for (uint32_t i = 0; i < an; i++) ax[i] = sinf((float)i * 0.019f) - (float)(i % 7) * 0.01f;
        ax[3079] = 42.0f;
        fail = qw6_vk_argmax(&vk, ax, an, &got_idx, &got_val) != 0;
        if (!fail && (got_idx != 3079 || fabsf(got_val - 42.0f) > 1e-6f)) {
            fprintf(stderr, "qw6_vk: argmax mismatch got idx=%u val=%.6f expected idx=3079 val=42.000000\n",
                    got_idx, got_val);
            fail = 1;
        }
    }

    const uint32_t gn = 1029;
    float *gx = calloc(gn, sizeof(float));
    if (!gx) fail = 1;
    if (!fail) {
        uint32_t token = 0;
        for (uint32_t i = 0; i < gn; i++)
            gx[i] = cosf((float)i * 0.023f) - 0.001f * (float)i;
        gx[877] = 11.0f;
        fail = qw6_vk_sampling_greedy(&vk, gx, gn, &token) != 0;
        if (!fail && token != 877) {
            fprintf(stderr, "qw6_vk: sampling_greedy mismatch got %u expected 877\n", token);
            fail = 1;
        }
    }

    const uint32_t q_dim = 64, kv_dim = 16, q_heads = 4, kv_heads = 1, rot = 8;
    float *mq_cpu = calloc(q_dim, sizeof(float));
    float *mk_cpu = calloc(kv_dim, sizeof(float));
    float *mq_gpu = calloc(q_dim, sizeof(float));
    float *mk_gpu = calloc(kv_dim, sizeof(float));
    if (!mq_cpu || !mk_cpu || !mq_gpu || !mk_gpu) fail = 1;
    if (!fail) {
        for (uint32_t i = 0; i < q_dim; i++) mq_cpu[i] = sinf((float)i * 0.07f);
        for (uint32_t i = 0; i < kv_dim; i++) mk_cpu[i] = cosf((float)i * 0.11f);
        memcpy(mq_gpu, mq_cpu, q_dim * sizeof(float));
        memcpy(mk_gpu, mk_cpu, kv_dim * sizeof(float));
        qw6_cpu_mrope(mq_cpu, mk_cpu, (int)q_dim, (int)kv_dim,
                      (int)q_heads, (int)kv_heads, 17, (int)rot);
        fail = qw6_vk_mrope(&vk, mq_gpu, mk_gpu, q_dim, kv_dim,
                            q_heads, kv_heads, 17, rot) != 0;
        for (uint32_t i = 0; i < q_dim && !fail; i++) {
            if (fabsf(mq_gpu[i] - mq_cpu[i]) > 2e-5f) {
                fprintf(stderr, "qw6_vk: mrope q mismatch at %u got %.6f expected %.6f\n",
                        i, mq_gpu[i], mq_cpu[i]);
                fail = 1;
            }
        }
        for (uint32_t i = 0; i < kv_dim && !fail; i++) {
            if (fabsf(mk_gpu[i] - mk_cpu[i]) > 2e-5f) {
                fprintf(stderr, "qw6_vk: mrope k mismatch at %u got %.6f expected %.6f\n",
                        i, mk_gpu[i], mk_cpu[i]);
                fail = 1;
            }
        }
    }

    const uint32_t as = 5, aqh = 4, akvh = 2, ahd = 8;
    float *aq = calloc(aqh * ahd, sizeof(float));
    float *ak = calloc(as * akvh * ahd, sizeof(float));
    float *av = calloc(as * akvh * ahd, sizeof(float));
    float *ao_cpu = calloc(aqh * ahd, sizeof(float));
    float *ao_gpu = calloc(aqh * ahd, sizeof(float));
    if (!aq || !ak || !av || !ao_cpu || !ao_gpu) fail = 1;
    if (!fail) {
        for (uint32_t i = 0; i < aqh * ahd; i++) aq[i] = sinf((float)i * 0.071f);
        for (uint32_t i = 0; i < as * akvh * ahd; i++) {
            ak[i] = cosf((float)i * 0.037f);
            av[i] = sinf((float)i * 0.053f) + 0.25f * cosf((float)i * 0.011f);
        }
        qw6_cpu_attention_gqa(ao_cpu, aq, ak, av, (int)as, (int)aqh,
                              (int)akvh, (int)ahd);
        fail = qw6_vk_attention_gqa(&vk, aq, ak, av, ao_gpu, as, aqh, akvh, ahd) != 0;
        for (uint32_t i = 0; i < aqh * ahd && !fail; i++) {
            if (fabsf(ao_gpu[i] - ao_cpu[i]) > 2e-5f) {
                fprintf(stderr,
                        "qw6_vk: attention_gqa mismatch at %u got %.6f expected %.6f\n",
                        i, ao_gpu[i], ao_cpu[i]);
                fail = 1;
            }
        }
    }

    const uint32_t en = 256, top_k = 8;
    float *elogits = calloc(en, sizeof(float));
    int *eidx_cpu = calloc(top_k, sizeof(int));
    float *ew_cpu = calloc(top_k, sizeof(float));
    uint32_t *eidx_gpu = calloc(top_k, sizeof(uint32_t));
    float *ew_gpu = calloc(top_k, sizeof(float));
    if (!elogits || !eidx_cpu || !ew_cpu || !eidx_gpu || !ew_gpu) fail = 1;
    if (!fail) {
        for (uint32_t i = 0; i < en; i++)
            elogits[i] = sinf((float)i * 0.17f) + cosf((float)i * 0.031f);
        elogits[7] = 5.0f;
        elogits[149] = 4.5f;
        qw6_cpu_moe_route(eidx_cpu, ew_cpu, elogits, (int)en, (int)top_k);
        fail = qw6_vk_moe_route(&vk, elogits, eidx_gpu, ew_gpu, en, top_k) != 0;
        for (uint32_t i = 0; i < top_k && !fail; i++) {
            if (eidx_gpu[i] != (uint32_t)eidx_cpu[i] || fabsf(ew_gpu[i] - ew_cpu[i]) > 2e-5f) {
                fprintf(stderr,
                        "qw6_vk: moe_route mismatch at %u got idx=%u weight=%.6f expected idx=%d weight=%.6f\n",
                        i, eidx_gpu[i], ew_gpu[i], eidx_cpu[i], ew_cpu[i]);
                fail = 1;
            }
        }
    }

    const uint32_t gd = 257, gtop = 8;
    float *gexpert = calloc((size_t)gd * gtop, sizeof(float));
    float *gweights = calloc(gtop, sizeof(float));
    float *gshared = calloc(gd, sizeof(float));
    float *go_cpu = calloc(gd, sizeof(float));
    float *go_gpu = calloc(gd, sizeof(float));
    if (!gexpert || !gweights || !gshared || !go_cpu || !go_gpu) fail = 1;
    if (!fail) {
        float wsum = 0.0f;
        for (uint32_t e = 0; e < gtop; e++) {
            gweights[e] = 0.1f + 0.03f * (float)e;
            wsum += gweights[e];
            for (uint32_t i = 0; i < gd; i++)
                gexpert[(size_t)e * gd + i] = sinf((float)(e * 17 + i) * 0.021f);
        }
        for (uint32_t e = 0; e < gtop; e++) gweights[e] /= wsum;
        for (uint32_t i = 0; i < gd; i++) {
            gshared[i] = cosf((float)i * 0.019f);
            go_cpu[i] = 0.25f * gshared[i];
            for (uint32_t e = 0; e < gtop; e++)
                go_cpu[i] += gweights[e] * gexpert[(size_t)e * gd + i];
        }
        fail = qw6_vk_moe_gather(&vk, gexpert, gweights, gshared, 0.25f,
                                 go_gpu, gd, gtop) != 0;
        for (uint32_t i = 0; i < gd && !fail; i++) {
            if (fabsf(go_gpu[i] - go_cpu[i]) > 2e-5f) {
                fprintf(stderr,
                        "qw6_vk: moe_gather mismatch at %u got %.6f expected %.6f\n",
                        i, go_gpu[i], go_cpu[i]);
                fail = 1;
            }
        }
    }

    const uint32_t cd = 17, cks = 4;
    float *cx = calloc(cd * cks, sizeof(float));
    uint16_t *cw = calloc(cd * cks, sizeof(uint16_t));
    float *co_cpu = calloc(cd, sizeof(float));
    float *co_gpu = calloc(cd, sizeof(float));
    if (!cx || !cw || !co_cpu || !co_gpu) fail = 1;
    if (!fail) {
        const uint16_t fp16_vals[4] = {0x3c00, 0x4000, 0x3800, 0xbc00};
        for (uint32_t i = 0; i < cd * cks; i++) {
            cx[i] = sinf((float)i * 0.09f) + 0.25f * cosf((float)i * 0.07f);
            cw[i] = fp16_vals[i % 4];
        }
        qw6_cpu_conv1d_causal(co_cpu, cx, cw, (int)cd, (int)cks);
        fail = qw6_vk_deltanet_conv1d(&vk, cx, cw, co_gpu, cd, cks) != 0;
        for (uint32_t i = 0; i < cd && !fail; i++) {
            if (fabsf(co_gpu[i] - co_cpu[i]) > 2e-5f) {
                fprintf(stderr,
                        "qw6_vk: deltanet_conv1d mismatch at %u got %.6f expected %.6f\n",
                        i, co_gpu[i], co_cpu[i]);
                fail = 1;
            }
        }
    }

    const uint32_t rkh = 2, rkd = 3, rvh = 2, rvd = 4;
    float *rstate = calloc((size_t)rkh * rkd * rvh * rvd, sizeof(float));
    float *rquery = calloc((size_t)rkh * rkd, sizeof(float));
    float *ro_cpu = calloc((size_t)rvh * rvd, sizeof(float));
    float *ro_gpu = calloc((size_t)rvh * rvd, sizeof(float));
    if (!rstate || !rquery || !ro_cpu || !ro_gpu) fail = 1;
    if (!fail) {
        for (uint32_t i = 0; i < rkh * rkd * rvh * rvd; i++)
            rstate[i] = sinf((float)i * 0.041f) - 0.5f * cosf((float)i * 0.013f);
        for (uint32_t i = 0; i < rkh * rkd; i++)
            rquery[i] = cosf((float)i * 0.19f);
        qw6_cpu_deltanet_retrieve(ro_cpu, rstate, rquery,
                                  (int)rkh, (int)rkd, (int)rvh, (int)rvd);
        fail = qw6_vk_deltanet_retrieve(&vk, rstate, rquery, ro_gpu,
                                        rkh, rkd, rvh, rvd) != 0;
        for (uint32_t i = 0; i < rvh * rvd && !fail; i++) {
            if (fabsf(ro_gpu[i] - ro_cpu[i]) > 2e-5f) {
                fprintf(stderr,
                        "qw6_vk: deltanet_retrieve mismatch at %u got %.6f expected %.6f\n",
                        i, ro_gpu[i], ro_cpu[i]);
                fail = 1;
            }
        }
    }

    const uint32_t ukh = 2, ukd = 3, uvh = 2, uvd = 4;
    float *us_cpu = calloc((size_t)ukh * ukd * uvh * uvd, sizeof(float));
    float *us_gpu = calloc((size_t)ukh * ukd * uvh * uvd, sizeof(float));
    float *ukey = calloc((size_t)ukh * ukd, sizeof(float));
    float *uquery = calloc((size_t)ukh * ukd, sizeof(float));
    float *uvalue = calloc((size_t)uvh * uvd, sizeof(float));
    float *ubeta = calloc(ukh, sizeof(float));
    if (!us_cpu || !us_gpu || !ukey || !uquery || !uvalue || !ubeta) fail = 1;
    if (!fail) {
        for (uint32_t i = 0; i < ukh * ukd * uvh * uvd; i++) {
            us_cpu[i] = 0.01f * (float)((int)(i % 9) - 4);
            us_gpu[i] = us_cpu[i];
        }
        for (uint32_t i = 0; i < ukh * ukd; i++) {
            ukey[i] = sinf((float)i * 0.23f);
            uquery[i] = cosf((float)i * 0.17f);
        }
        for (uint32_t i = 0; i < uvh * uvd; i++)
            uvalue[i] = 0.5f * sinf((float)i * 0.11f);
        for (uint32_t i = 0; i < ukh; i++)
            ubeta[i] = 0.75f + 0.1f * (float)i;
        qw6_cpu_deltanet_update(us_cpu, ukey, uvalue, uquery, ubeta,
                                (int)ukh, (int)ukd, (int)uvh, (int)uvd);
        fail = qw6_vk_deltanet_update(&vk, us_gpu, ukey, uvalue, uquery,
                                      ubeta, ukh, ukd, uvh, uvd) != 0;
        for (uint32_t i = 0; i < ukh * ukd * uvh * uvd && !fail; i++) {
            if (fabsf(us_gpu[i] - us_cpu[i]) > 2e-5f) {
                fprintf(stderr,
                        "qw6_vk: deltanet_update mismatch at %u got %.6f expected %.6f\n",
                        i, us_gpu[i], us_cpu[i]);
                fail = 1;
            }
        }
    }

    const uint32_t mv = 13, mh = 37;
    float *mtp_w = calloc(mv * mh, sizeof(float));
    float *mtp_hidden = calloc(mh, sizeof(float));
    float *mtp_cpu = calloc(mv, sizeof(float));
    float *mtp_gpu = calloc(mv, sizeof(float));
    if (!mtp_w || !mtp_hidden || !mtp_cpu || !mtp_gpu) fail = 1;
    if (!fail) {
        for (uint32_t i = 0; i < mv * mh; i++)
            mtp_w[i] = 0.03f * sinf((float)i * 0.07f);
        for (uint32_t i = 0; i < mh; i++)
            mtp_hidden[i] = cosf((float)i * 0.13f);
        qw6_cpu_mtp_draft(mtp_cpu, mtp_hidden, mtp_w, (int)mv, (int)mh);
        fail = qw6_vk_mtp_draft(&vk, mtp_w, mtp_hidden, mtp_gpu, mv, mh) != 0;
        for (uint32_t i = 0; i < mv && !fail; i++) {
            if (fabsf(mtp_gpu[i] - mtp_cpu[i]) > 2e-5f) {
                fprintf(stderr,
                        "qw6_vk: mtp_draft mismatch at %u got %.6f expected %.6f\n",
                        i, mtp_gpu[i], mtp_cpu[i]);
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
    free(crx);
    free(crw);
    free(cry);
    free(mw);
    free(mx);
    free(my);
    free(q4w);
    free(q4x);
    free(q4y);
    free(q5w);
    free(q5x);
    free(q5y);
    free(q6w);
    free(q6x);
    free(q6y);
    free(sg);
    free(su);
    free(so);
    free(ax);
    free(gx);
    free(mq_cpu);
    free(mk_cpu);
    free(mq_gpu);
    free(mk_gpu);
    free(aq);
    free(ak);
    free(av);
    free(ao_cpu);
    free(ao_gpu);
    free(elogits);
    free(eidx_cpu);
    free(ew_cpu);
    free(eidx_gpu);
    free(ew_gpu);
    free(gexpert);
    free(gweights);
    free(gshared);
    free(go_cpu);
    free(go_gpu);
    free(cx);
    free(cw);
    free(co_cpu);
    free(co_gpu);
    free(rstate);
    free(rquery);
    free(ro_cpu);
    free(ro_gpu);
    free(us_cpu);
    free(us_gpu);
    free(ukey);
    free(uquery);
    free(uvalue);
    free(ubeta);
    free(mtp_w);
    free(mtp_hidden);
    free(mtp_cpu);
    free(mtp_gpu);
    qw6_vk_free(&vk);
    if (fail) {
        fprintf(stderr, "qw6_vk: self-test failed\n");
        return 1;
    }
    fprintf(stderr, "qw6_vk: self-test passed\n");
    return 0;
}

/* ========================================================================
 * Phase 2: Full GPU dispatch pipeline
 * ======================================================================== */



struct qw6_vk_pipe_s {
    qw6_vk_t vk;
    bool strict;
    bool profile;
    VkQueryPool timestamp_queries;
    uint32_t timestamp_query_count;
    uint32_t timestamp_query_next;
    uint64_t dispatch_count;
    uint64_t fallback_count;
    uint64_t read_bytes;
    uint64_t write_bytes;
    vk_pipe_cache_t pipe_cache[QW6_VK_CACHE_MAX];
    int pipe_cache_count;
    VkDescriptorPool descriptor_pool;
    VkCommandBuffer dispatch_cb;
    VkFence dispatch_fence;

    /* Big buffer housing all weight tensors */
    qw6_vk_buffer_t weights;
    size_t weight_capacity;
    qw6_vk_buffer_t iq2s_grid;
    qw6_vk_buffer_t iq3s_grid;

    /* Persistent scratch buffers */
    qw6_vk_buffer_t scr_hidden;    /* [HIDDEN_SIZE] float */
    qw6_vk_buffer_t scr_resid;     /* [HIDDEN_SIZE] float */
    qw6_vk_buffer_t scr_normed;    /* [HIDDEN_SIZE] float */
    qw6_vk_buffer_t scr_norm_w;    /* [HIDDEN_SIZE] float */
    qw6_vk_buffer_t scr_attn;      /* [HIDDEN_SIZE] float */
    qw6_vk_buffer_t scr_ffn;       /* [HIDDEN_SIZE] float */
    qw6_vk_buffer_t scr_s0;        /* [HIDDEN_SIZE*2] float */
    qw6_vk_buffer_t scr_s1;        /* [HIDDEN_SIZE*2] float */
    qw6_vk_buffer_t scr_logits;    /* [VOCAB_SIZE] float */
    qw6_vk_buffer_t scr_sample;    /* [1] uint32_t */
    qw6_vk_buffer_t scr_moe_idx;   /* [QW6_EXPERTS_PER_TOK] uint32_t */
    qw6_vk_buffer_t scr_moe_w;     /* [QW6_EXPERTS_PER_TOK] float */

    /* KV cache for full-attn layers */
    qw6_vk_buffer_t k_cache[QW6_NUM_LAYERS];
    qw6_vk_buffer_t v_cache[QW6_NUM_LAYERS];

    /* DeltaNet state + Conv1D state for linear-attn layers */
    qw6_vk_buffer_t dn_state[QW6_NUM_LAYERS];
    qw6_vk_buffer_t conv_state[QW6_NUM_LAYERS];

    uint32_t max_ctx;
};

/* Cached dispatch: same as qw6_vk_dispatch but reuses pipelines and layouts.
 * Must be defined here (after struct) so it can access the cache. */
static int qw6_vk_pipe_dispatch(struct qw6_vk_pipe_s *p, const char *shader_path,
                                 qw6_vk_buffer_t **buffers, const size_t *offsets,
                                 uint32_t n_buffers,
                                 const void *push, uint32_t push_size,
                                 uint32_t gx, uint32_t gy, uint32_t gz) {
    qw6_vk_t *vk = &p->vk;
    bool profile = p->profile;
    double setup_t0 = profile ? qw6_vk_now_ms() : 0.0;

    /* Find or create cached pipeline */
    int ci = -1;
    for (int i = 0; i < p->pipe_cache_count; i++) {
        if (p->pipe_cache[i].valid &&
            strcmp(p->pipe_cache[i].shader_path, shader_path) == 0 &&
            p->pipe_cache[i].n_buffers == n_buffers &&
            p->pipe_cache[i].push_size == push_size) {
            ci = i;
            break;
        }
    }
    if (ci < 0 && p->pipe_cache_count < QW6_VK_CACHE_MAX) {
        ci = p->pipe_cache_count++;
        vk_pipe_cache_t *c = &p->pipe_cache[ci];
        strncpy(c->shader_path, shader_path, sizeof(c->shader_path) - 1);
        c->n_buffers = n_buffers;
        c->push_size = push_size;
        c->valid = 0;

        VkDescriptorSetLayoutBinding bindings[5];
        memset(bindings, 0, sizeof(bindings));
        VkDescriptorSetLayoutCreateInfo dlci = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = n_buffers,
            .pBindings = bindings,
        };
        for (uint32_t i = 0; i < n_buffers && i < 5; i++) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        if (vkCreateDescriptorSetLayout(vk->device, &dlci, NULL, &c->dsl) != VK_SUCCESS) return -1;

        VkPushConstantRange pcr = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = push_size,
        };
        VkPipelineLayoutCreateInfo plci = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &c->dsl,
            .pushConstantRangeCount = push_size ? 1u : 0u,
            .pPushConstantRanges = push_size ? &pcr : NULL,
        };
        if (vkCreatePipelineLayout(vk->device, &plci, NULL, &c->layout) != VK_SUCCESS) return -1;

        uint32_t *spv = NULL;
        size_t spv_bytes = 0;
        if (qw6_vk_read_file(shader_path, &spv, &spv_bytes) != 0) return -1;
        VkShaderModuleCreateInfo smci = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = spv_bytes,
            .pCode = spv,
        };
        if (vkCreateShaderModule(vk->device, &smci, NULL, &c->sm) != VK_SUCCESS) { free(spv); return -1; }
        free(spv);

        VkComputePipelineCreateInfo pci = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = c->sm,
                .pName = "main",
            },
            .layout = c->layout,
        };
        if (vkCreateComputePipelines(vk->device, VK_NULL_HANDLE, 1, &pci, NULL, &c->pipeline) != VK_SUCCESS) return -1;
        c->valid = 1;
    }
    if (ci < 0)
        return qw6_vk_dispatch(vk, shader_path, buffers, offsets,
                               n_buffers, push, push_size, gx, gy, gz);

    vk_pipe_cache_t *c = &p->pipe_cache[ci];

    vkResetDescriptorPool(vk->device, p->descriptor_pool, 0);
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = p->descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &c->dsl,
    };
    VkDescriptorSet ds;
    if (vkAllocateDescriptorSets(vk->device, &dsai, &ds) != VK_SUCCESS) return -1;

    VkDescriptorBufferInfo infos[5];
    VkWriteDescriptorSet writes[5];
    memset(infos, 0, sizeof(infos));
    memset(writes, 0, sizeof(writes));
    uint64_t read_bytes = 0;
    for (uint32_t i = 0; i < n_buffers && i < 5; i++) {
        infos[i].buffer = buffers[i]->buffer;
        infos[i].offset = offsets ? offsets[i] : 0;
        infos[i].range = buffers[i]->size - infos[i].offset;
        if (i + 1 < n_buffers) {
            read_bytes += infos[i].range;
        }
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = ds;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(vk->device, n_buffers, writes, 0, NULL);

    VkCommandBuffer cb = p->dispatch_cb;
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    if (vkBeginCommandBuffer(cb, &begin) != VK_SUCCESS) return -1;
    uint32_t ts0 = UINT32_MAX, ts1 = UINT32_MAX;
    if (profile && p->timestamp_queries &&
        p->timestamp_query_next + 1 < p->timestamp_query_count) {
        ts0 = p->timestamp_query_next++;
        ts1 = p->timestamp_query_next++;
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            p->timestamp_queries, ts0);
    }
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->layout, 0, 1, &ds, 0, NULL);
    if (push_size) vkCmdPushConstants(cb, c->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push);
    vkCmdDispatch(cb, gx, gy, gz);
    if (ts1 != UINT32_MAX) {
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            p->timestamp_queries, ts1);
    }
    if (vkEndCommandBuffer(cb) != VK_SUCCESS) return -1;

    vkResetFences(vk->device, 1, &p->dispatch_fence);
    VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb };
    double t0 = qw6_vk_now_ms();
    if (vkQueueSubmit(vk->queue, 1, &submit, p->dispatch_fence) != VK_SUCCESS) return -1;
    if (vkWaitForFences(vk->device, 1, &p->dispatch_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS) return -1;
    double wall_ms = qw6_vk_now_ms() - t0;
    c->calls++;
    c->total_ms += wall_ms;
    p->dispatch_count++;
    if (profile) {
        c->cpu_setup_ms += t0 - setup_t0;
        c->read_bytes += read_bytes;
        c->write_bytes += (n_buffers > 0) ? infos[n_buffers - 1].range : 0;
        p->read_bytes += read_bytes;
        p->write_bytes += (n_buffers > 0) ? infos[n_buffers - 1].range : 0;
        if (ts1 != UINT32_MAX) {
            uint64_t stamps[2] = {0, 0};
            VkResult qrc = vkGetQueryPoolResults(vk->device, p->timestamp_queries,
                                                 ts0, 2, sizeof(stamps), stamps,
                                                 sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
            if (qrc == VK_SUCCESS && stamps[1] >= stamps[0]) {
                double gpu_ns = (double)(stamps[1] - stamps[0]) * vk->timestamp_period_ns;
                c->gpu_ms += gpu_ns / 1000000.0;
            }
        }
    }

    return 0;
}

/* ---- Helpers ---- */

/* Upload a single tensor's data to the big weight buffer.
 * Returns the byte offset within the weight buffer, or (size_t)-1 on error. */
static size_t qw6_vk_upload_tensor(struct qw6_vk_pipe_s *p,
                                    const qw6_tensor_t *t) {
    if (!t || !t->data || t->data_size == 0) return (size_t)-1;
    /* Mark as not yet uploaded in case of failure below */
    ((qw6_tensor_t *)t)->vk_offset = (size_t)-1;
    /* Align offset to 256 bytes for GPU-friendly access */
    size_t off = p->weight_capacity;
    off = (off + 255) & ~(size_t)255;
    if (off + t->data_size > p->weights.size) return (size_t)-1;
    memcpy((uint8_t *)p->weights.mapped + off, t->data, t->data_size);
    /* Store offset back to tensor for later dispatch */
    /* (t is const, but vk_offset is mutable via the union / cast) */
    ((qw6_tensor_t *)t)->vk_offset = off;
    p->weight_capacity = off + t->data_size;
    return off;
}

/* Dispatch a quantised MatVec for a tensor, reading from the weight buffer
 * at t->vk_offset, applying against input buffer, writing to output buffer.
 * Returns 0 on success. */
static int qw6_vk_pipe_matvec(struct qw6_vk_pipe_s *p,
                               const qw6_tensor_t *t,
                               qw6_vk_buffer_t *inp, qw6_vk_buffer_t *out,
                               uint32_t rows, uint32_t cols) {
    if (!t->data || t->vk_offset == (size_t)-1) return -1;
    const size_t offs[3] = {t->vk_offset, 0, 0};
    qw6_vk_buffer_t *bufs[3] = {&p->weights, inp, out};
    qw6_vk_matvec_push_t push = {.rows = rows, .cols = cols};


    switch (t->quant) {
    case QW6_Q_FP32:
        return qw6_vk_pipe_dispatch(p, "vulkan/matvec_f32.spv",
                                    bufs, offs, 3, &push, sizeof(push), rows, 1, 1);
    case QW6_Q_Q5_K:
        return qw6_vk_pipe_dispatch(p, "vulkan/matmul_q5k.spv",
                                    bufs, offs, 3, &push, sizeof(push), rows, 1, 1);
    case QW6_Q_Q6_K:
        return qw6_vk_pipe_dispatch(p, "vulkan/matmul_q6k.spv",
                                    bufs, offs, 3, &push, sizeof(push), rows, 1, 1);
    case QW6_Q_Q4_K_M:
    case QW6_Q_Q4_K_S:
        return qw6_vk_pipe_dispatch(p, "vulkan/matmul_q4k.spv",
                                    bufs, offs, 3, &push, sizeof(push), rows, 1, 1);
    case QW6_Q_IQ2_XXS:
    case QW6_Q_IQ2_M:
        return qw6_vk_pipe_dispatch(p, "vulkan/matmul_iq2xxs.spv",
                                    bufs, offs, 3, &push, sizeof(push), rows, 1, 1);
    case QW6_Q_IQ2_S: {
        const size_t offs4[4] = {t->vk_offset, 0, 0, 0};
        qw6_vk_buffer_t *bufs4[4] = {&p->weights, inp, out, &p->iq2s_grid};
        return qw6_vk_pipe_dispatch(p, "vulkan/matmul_iq2s.spv",
                                    bufs4, offs4, 4, &push, sizeof(push), rows, 1, 1);
    }
    case QW6_Q_IQ3_S: {
        const size_t offs4[4] = {t->vk_offset, 0, 0, 0};
        qw6_vk_buffer_t *bufs4[4] = {&p->weights, inp, out, &p->iq3s_grid};
        return qw6_vk_pipe_dispatch(p, "vulkan/matmul_iq3s.spv",
                                    bufs4, offs4, 4, &push, sizeof(push), rows, 1, 1);
    }
    default:
        /* Unsupported on GPU — caller must fall back to CPU */
        return -1;
    }
}

/* GPU dispatch helpers for pipeline ops */

static int qw6_vk_pipe_rmsnorm(struct qw6_vk_pipe_s *p,
                                qw6_vk_buffer_t *x, qw6_vk_buffer_t *w,
                                qw6_vk_buffer_t *y, uint32_t n, float eps) {
    const size_t zero[3] = {0, 0, 0};
    qw6_vk_buffer_t *bufs[3] = {x, w, y};
    qw6_vk_rmsnorm_push_t push = {.n = n, .eps = eps};
    return qw6_vk_pipe_dispatch(p, "vulkan/rmsnorm_full.spv",
                                bufs, zero, 3, &push, sizeof(push), 1, 1, 1);
}

static int qw6_vk_pipe_add(struct qw6_vk_pipe_s *p,
                            qw6_vk_buffer_t *a, qw6_vk_buffer_t *b,
                            qw6_vk_buffer_t *c, uint32_t n) {
    const size_t zero[3] = {0, 0, 0};
    qw6_vk_buffer_t *bufs[3] = {a, b, c};
    return qw6_vk_pipe_dispatch(p, "vulkan/add.spv",
                                bufs, zero, 3, &n, sizeof(n),
                                (n + 255) / 256, 1, 1);
}

static int qw6_vk_pipe_silu_mul(struct qw6_vk_pipe_s *p,
                                 qw6_vk_buffer_t *gate, qw6_vk_buffer_t *up,
                                 qw6_vk_buffer_t *out, uint32_t n) {
    const size_t zero[3] = {0, 0, 0};
    qw6_vk_buffer_t *bufs[3] = {gate, up, out};
    qw6_vk_n_push_t push = {.n = n};
    return qw6_vk_pipe_dispatch(p, "vulkan/silu_mul.spv",
                                bufs, zero, 3, &push, sizeof(push),
                                (n + 255) / 256, 1, 1);
}

static int qw6_vk_pipe_mrope(struct qw6_vk_pipe_s *p,
                              qw6_vk_buffer_t *qbuf, qw6_vk_buffer_t *kbuf,
                              uint32_t q_dim, uint32_t kv_dim,
                              uint32_t n_heads, uint32_t n_kv_heads,
                              uint32_t position, uint32_t rotary_dim) {
    const size_t zero[2] = {0, 0};
    qw6_vk_buffer_t *bufs[2] = {qbuf, kbuf};
    qw6_vk_mrope_push_t push = {
        .q_dim = q_dim, .kv_dim = kv_dim,
        .n_heads = n_heads, .n_kv_heads = n_kv_heads,
        .position = position,
        .q_head_dim = q_dim / n_heads, .k_head_dim = kv_dim / n_kv_heads,
        .q_rot = rotary_dim, .k_rot = rotary_dim,
        .theta_base = QW6_ROPE_THETA,
    };
    uint32_t total_pairs = (q_dim + kv_dim) / 2;
    return qw6_vk_pipe_dispatch(p, "vulkan/rope_mrope.spv",
                                bufs, zero, 2, &push, sizeof(push),
                                (total_pairs + 255) / 256, 1, 1);
}

static int qw6_vk_pipe_attention_gqa(struct qw6_vk_pipe_s *p,
                                      qw6_vk_buffer_t *qbuf,
                                      qw6_vk_buffer_t *k_cache,
                                      qw6_vk_buffer_t *v_cache,
                                      qw6_vk_buffer_t *outbuf,
                                      uint32_t seq_len,
                                      uint32_t n_q_heads,
                                      uint32_t n_kv_heads,
                                      uint32_t head_dim) {
    const size_t zero[4] = {0, 0, 0, 0};
    qw6_vk_buffer_t *bufs[4] = {qbuf, k_cache, v_cache, outbuf};
    qw6_vk_attention_gqa_push_t push = {
        .seq_len = seq_len,
        .n_q_heads = n_q_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
    };
    return qw6_vk_pipe_dispatch(p, "vulkan/attention_gqa.spv",
                                bufs, zero, 4, &push, sizeof(push),
                                n_q_heads, 1, 1);
}

__attribute__((unused)) static int qw6_vk_pipe_conv1d(struct qw6_vk_pipe_s *p,
                               qw6_vk_buffer_t *xbuf, qw6_vk_buffer_t *wbuf,
                               qw6_vk_buffer_t *outbuf,
                               uint32_t dim, uint32_t kernel_size) {
    const size_t zero[3] = {0, 0, 0};
    qw6_vk_buffer_t *bufs[3] = {xbuf, wbuf, outbuf};
    qw6_vk_conv1d_push_t push = {.dim = dim, .kernel_size = kernel_size};
    return qw6_vk_pipe_dispatch(p, "vulkan/deltanet_conv1d.spv",
                                bufs, zero, 3, &push, sizeof(push),
                                (dim + 255) / 256, 1, 1);
}

static int qw6_vk_pipe_conv1d_f32_state(struct qw6_vk_pipe_s *p,
                                        qw6_vk_buffer_t *state,
                                        qw6_vk_buffer_t *x,
                                        const qw6_tensor_t *w,
                                        qw6_vk_buffer_t *out,
                                        uint32_t dim,
                                        uint32_t kernel_size) {
    if (!w || w->quant != QW6_Q_FP32 || w->vk_offset == (size_t)-1) return -1;
    const size_t offs[4] = {0, 0, w->vk_offset, 0};
    qw6_vk_buffer_t *bufs[4] = {state, x, &p->weights, out};
    qw6_vk_conv1d_push_t push = {.dim = dim, .kernel_size = kernel_size};
    return qw6_vk_pipe_dispatch(p, "vulkan/deltanet_conv1d_f32.spv",
                                bufs, offs, 4, &push, sizeof(push),
                                (dim + 255) / 256, 1, 1);
}

__attribute__((unused)) static int qw6_vk_pipe_deltanet_retrieve(struct qw6_vk_pipe_s *p,
                                          qw6_vk_buffer_t *state,
                                          qw6_vk_buffer_t *query,
                                          qw6_vk_buffer_t *out,
                                          uint32_t key_heads,
                                          uint32_t key_dim,
                                          uint32_t val_heads,
                                          uint32_t val_dim) {
    const size_t zero[3] = {0, 0, 0};
    qw6_vk_buffer_t *bufs[3] = {state, query, out};
    qw6_vk_deltanet_push_t push = {
        .key_heads = key_heads, .key_dim = key_dim,
        .val_heads = val_heads, .val_dim = val_dim,
        .has_beta = 0,
    };
    uint32_t out_n = val_heads * val_dim;
    return qw6_vk_pipe_dispatch(p, "vulkan/deltanet_retrieve.spv",
                                bufs, zero, 3, &push, sizeof(push),
                                (out_n + 255) / 256, 1, 1);
}

static int qw6_vk_pipe_deltanet_gated(struct qw6_vk_pipe_s *p,
                                      qw6_vk_buffer_t *state,
                                      qw6_vk_buffer_t *qkv,
                                      qw6_vk_buffer_t *gate,
                                      qw6_vk_buffer_t *beta,
                                      qw6_vk_buffer_t *out,
                                      uint32_t key_heads,
                                      uint32_t val_heads,
                                      uint32_t dim) {
    const size_t zero[5] = {0, 0, 0, 0, 0};
    qw6_vk_buffer_t *bufs[5] = {state, qkv, gate, beta, out};
    qw6_vk_deltanet_gated_push_t push = {
        .key_heads = key_heads,
        .val_heads = val_heads,
        .dim = dim,
        .scale = 1.0f / sqrtf((float)dim),
    };
    uint32_t n = val_heads * dim;
    return qw6_vk_pipe_dispatch(p, "vulkan/deltanet_gated.spv",
                                bufs, zero, 5, &push, sizeof(push),
                                (n + 255) / 256, 1, 1);
}

__attribute__((unused)) static int qw6_vk_pipe_deltanet_update(struct qw6_vk_pipe_s *p,
                                        qw6_vk_buffer_t *state,
                                        qw6_vk_buffer_t *key,
                                        qw6_vk_buffer_t *value,
                                        qw6_vk_buffer_t *query,
                                        qw6_vk_buffer_t *beta,
                                        uint32_t key_heads,
                                        uint32_t key_dim,
                                        uint32_t val_heads,
                                        uint32_t val_dim) {
    const size_t zero[5] = {0, 0, 0, 0, 0};
    qw6_vk_buffer_t *bufs[5] = {state, key, value, query, beta};
    qw6_vk_deltanet_push_t push = {
        .key_heads = key_heads, .key_dim = key_dim,
        .val_heads = val_heads, .val_dim = val_dim,
        .has_beta = 1,
    };
    uint32_t state_n = val_heads * val_dim * val_dim;
    return qw6_vk_pipe_dispatch(p, "vulkan/deltanet_update.spv",
                                bufs, zero, 5, &push, sizeof(push),
                                (state_n + 255) / 256, 1, 1);
}

static int qw6_vk_pipe_moe_route(struct qw6_vk_pipe_s *p,
                                  qw6_vk_buffer_t *logits,
                                  qw6_vk_buffer_t *indices,
                                  qw6_vk_buffer_t *weights,
                                  uint32_t n_experts, uint32_t top_k) {
    const size_t zero[3] = {0, 0, 0};
    qw6_vk_buffer_t *bufs[3] = {logits, indices, weights};
    qw6_vk_moe_route_push_t push = {.n_experts = n_experts, .top_k = top_k};
    return qw6_vk_pipe_dispatch(p, "vulkan/moe_route.spv",
                                bufs, zero, 3, &push, sizeof(push), 1, 1, 1);
}

__attribute__((unused)) static int qw6_vk_pipe_moe_gather(struct qw6_vk_pipe_s *p,
                                   qw6_vk_buffer_t *expert_out,
                                   qw6_vk_buffer_t *expert_weights,
                                   qw6_vk_buffer_t *shared_out,
                                   float shared_weight,
                                   qw6_vk_buffer_t *out,
                                   uint32_t dim, uint32_t top_k) {
    const size_t zero[4] = {0, 0, 0, 0};
    qw6_vk_buffer_t *bufs[4] = {expert_out, expert_weights, shared_out, out};
    qw6_vk_moe_gather_push_t push = {
        .dim = dim, .top_k = top_k,
        .has_shared = shared_out ? 1u : 0u,
        .shared_weight = shared_weight,
    };
    return qw6_vk_pipe_dispatch(p, "vulkan/moe_gather.spv",
                                bufs, zero, 4, &push, sizeof(push),
                                (dim + 255) / 256, 1, 1);
}

typedef struct {
    uint32_t heads;
    uint32_t dim;
} qw6_vk_l2_norm_push_t;

typedef struct {
    uint32_t heads;
    uint32_t dim;
    float eps;
} qw6_vk_rmsnorm_heads_push_t;

typedef struct {
    uint32_t src_offset;  /* in float units */
    uint32_t dst_offset;
    uint32_t count;
} qw6_vk_buf_copy_push_t;

static int qw6_vk_pipe_l2_norm_heads(struct qw6_vk_pipe_s *p,
                                      qw6_vk_buffer_t *buf,
                                      size_t offset,
                                      uint32_t heads, uint32_t dim) {
    const size_t offs[2] = {offset, offset};
    qw6_vk_buffer_t *bufs[2] = {buf, buf};
    qw6_vk_l2_norm_push_t push = {.heads = heads, .dim = dim};
    return qw6_vk_pipe_dispatch(p, "vulkan/l2_norm_heads.spv",
                                bufs, offs, 2, &push, sizeof(push),
                                heads, 1, 1);
}

static int qw6_vk_pipe_rmsnorm_heads(struct qw6_vk_pipe_s *p,
                                     qw6_vk_buffer_t *buf,
                                     const qw6_tensor_t *w,
                                     uint32_t heads, uint32_t dim) {
    if (!w || w->vk_offset == (size_t)-1) return -1;
    const size_t offs[2] = {0, w->vk_offset};
    qw6_vk_buffer_t *bufs[2] = {buf, &p->weights};
    qw6_vk_rmsnorm_heads_push_t push = {.heads = heads, .dim = dim, .eps = QW6_RMS_EPS};
    return qw6_vk_pipe_dispatch(p, "vulkan/rmsnorm_heads.spv",
                                bufs, offs, 2, &push, sizeof(push),
                                heads, 1, 1);
}

static int qw6_vk_pipe_sample_greedy(struct qw6_vk_pipe_s *p,
                                       qw6_vk_buffer_t *logits_buf,
                                       uint32_t *out_token) {
    const size_t zero[2] = {0, 0};
    qw6_vk_buffer_t *bufs[2] = {logits_buf, &p->scr_sample};
    qw6_vk_n_push_t push = {.n = QW6_VOCAB_SIZE};
    if (qw6_vk_pipe_dispatch(p, "vulkan/sampling.spv",
                              bufs, zero, 2, &push, sizeof(push),
                              1, 1, 1) != 0) return -1;
    uint32_t tok;
    memcpy(&tok, p->scr_sample.mapped, sizeof(tok));
    p->read_bytes += sizeof(tok);
    *out_token = tok;
    return 0;
}

static int qw6_vk_pipe_buf_copy(struct qw6_vk_pipe_s *p,
                                 qw6_vk_buffer_t *src, size_t src_off,
                                 qw6_vk_buffer_t *dst, size_t dst_off,
                                 uint32_t count_floats) {
    const size_t offs[2] = {src_off, dst_off};
    qw6_vk_buffer_t *bufs[2] = {src, dst};
    qw6_vk_buf_copy_push_t push = {
        .src_offset = 0,  /* already accounted in buffer offset */
        .dst_offset = 0,
        .count = count_floats,
    };
    return qw6_vk_pipe_dispatch(p, "vulkan/buf_copy.spv",
                                bufs, offs, 2, &push, sizeof(push),
                                (count_floats + 255) / 256, 1, 1);
}

static int qw6_vk_pipe_alpha_beta(struct qw6_vk_pipe_s *p,
                                    qw6_vk_buffer_t *alpha,
                                    qw6_vk_buffer_t *beta,
                                    const qw6_tensor_t *dt,
                                    const qw6_tensor_t *a,
                                    uint32_t n_heads) {
    if (!dt || dt->vk_offset == (size_t)-1) return -1;
    if (!a || a->vk_offset == (size_t)-1) return -1;
    const size_t offs[4] = {0, 0, dt->vk_offset, a->vk_offset};
    qw6_vk_buffer_t *bufs[4] = {alpha, beta, &p->weights, &p->weights};
    qw6_vk_n_push_t push = {.n = n_heads};
    return qw6_vk_pipe_dispatch(p, "vulkan/deltanet_alpha_beta.spv",
                                bufs, offs, 4, &push, sizeof(push),
                                1, 1, 1);
}

static int qw6_vk_pipe_sigmoid_mul(struct qw6_vk_pipe_s *p,
                                     qw6_vk_buffer_t *gate, size_t gate_off,
                                     qw6_vk_buffer_t *x,
                                     qw6_vk_buffer_t *out,
                                     uint32_t n) {
    const size_t offs[3] = {gate_off, 0, 0};
    qw6_vk_buffer_t *bufs[3] = {gate, x, out};
    qw6_vk_n_push_t push = {.n = n};
    return qw6_vk_pipe_dispatch(p, "vulkan/sigmoid_mul.spv",
                                bufs, offs, 3, &push, sizeof(push),
                                (n + 255) / 256, 1, 1);
}

typedef struct {
    uint32_t val_heads;
    uint32_t dim;
    float eps;
} qw6_vk_deltanet_norm_gate_push_t;

static int qw6_vk_pipe_deltanet_norm_gate(struct qw6_vk_pipe_s *p,
                                           qw6_vk_buffer_t *gdn,
                                           const qw6_tensor_t *norm_w,
                                           qw6_vk_buffer_t *z,
                                           qw6_vk_buffer_t *out,
                                           uint32_t val_heads, uint32_t dim) {
    if (!norm_w || norm_w->vk_offset == (size_t)-1) return -1;
    const size_t offs[4] = {0, norm_w->vk_offset, 0, 0};
    qw6_vk_buffer_t *bufs[4] = {gdn, &p->weights, z, out};
    qw6_vk_deltanet_norm_gate_push_t push = {
        .val_heads = val_heads,
        .dim = dim,
        .eps = QW6_RMS_EPS,
    };
    return qw6_vk_pipe_dispatch(p, "vulkan/deltanet_norm_gate.spv",
                                bufs, offs, 4, &push, sizeof(push),
                                val_heads, 1, 1);
}

typedef struct {
    uint32_t n;
    float a;
} qw6_vk_axpy_push_t;

static int qw6_vk_pipe_axpy(struct qw6_vk_pipe_s *p,
                             qw6_vk_buffer_t *y,
                             qw6_vk_buffer_t *x,
                             float a, uint32_t n) {
    const size_t zero[2] = {0, 0};
    qw6_vk_buffer_t *bufs[2] = {y, x};
    qw6_vk_axpy_push_t push = {.n = n, .a = a};
    return qw6_vk_pipe_dispatch(p, "vulkan/vec_axpy.spv",
                                bufs, zero, 2, &push, sizeof(push),
                                (n + 255) / 256, 1, 1);
}

static int qw6_vk_strict_fallback(struct qw6_vk_pipe_s *p, const char *what) {
    p->fallback_count++;
    if (p->strict) {
        fprintf(stderr, "qw6_vk: strict Vulkan mode forbids CPU fallback (%s)\n", what);
        return -1;
    }
    return 0;
}

/* ---- Init ---- */

int qw6_vk_pipe_init(qw6_vk_pipe_t **p, qw6_model_t *m, bool strict) {
    if (!p || !m) return -1;
    qw6_vk_pipe_t *ctx = calloc(1, sizeof(qw6_vk_pipe_t));
    if (!ctx) return -1;
    ctx->strict = strict;
    ctx->profile = getenv("QW6_PROFILE") != NULL;

    if (qw6_vk_init(&ctx->vk) != 0) {
        free(ctx);
        return -1;
    }
    if (qw6_vk_memory_report(&ctx->vk, m) != 0) {
        qw6_vk_free(&ctx->vk);
        free(ctx);
        return -1;
    }

    if (ctx->profile && ctx->vk.timestamp_supported) {
        VkQueryPoolCreateInfo qpci = {
            .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .queryType = VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = 8192,
        };
        if (vkCreateQueryPool(ctx->vk.device, &qpci, NULL, &ctx->timestamp_queries) == VK_SUCCESS) {
            ctx->timestamp_query_count = qpci.queryCount;
        } else {
            ctx->vk.timestamp_supported = false;
        }
    }

    VkDescriptorPoolSize pool_size = {
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 5,
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    if (vkCreateDescriptorPool(ctx->vk.device, &dpci, NULL,
                               &ctx->descriptor_pool) != VK_SUCCESS) {
        qw6_vk_free(&ctx->vk); free(ctx); return -1;
    }

    VkCommandBufferAllocateInfo cbai = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx->vk.command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(ctx->vk.device, &cbai,
                                 &ctx->dispatch_cb) != VK_SUCCESS) {
        qw6_vk_pipe_free(ctx); return -1;
    }

    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(ctx->vk.device, &fci, NULL,
                      &ctx->dispatch_fence) != VK_SUCCESS) {
        qw6_vk_pipe_free(ctx); return -1;
    }

    ctx->max_ctx = m->max_context > 0 ? m->max_context : QW6_DEFAULT_CTX;

    /* ---- Allocate the big weight buffer ---- */
    size_t total_w = m->total_weight_bytes + 512 * 1024 * 1024; /* +512 MB for dequant temps */
    if (qw6_vk_buffer_create(&ctx->vk, &ctx->weights, total_w,
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_free(&ctx->vk); free(ctx); return -1;
    }
    if (qw6_vk_buffer_create(&ctx->vk, &ctx->iq2s_grid, sizeof(iq2s_grid),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_pipe_free(ctx); return -1;
    }
    memcpy(ctx->iq2s_grid.mapped, iq2s_grid, sizeof(iq2s_grid));

    if (qw6_vk_buffer_create(&ctx->vk, &ctx->iq3s_grid, sizeof(iq3s_grid),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
        qw6_vk_pipe_free(ctx); return -1;
    }
    memcpy(ctx->iq3s_grid.mapped, iq3s_grid, sizeof(iq3s_grid));

    /* ---- Allocate scratch buffers ---- */
    /* Hidden-size buffers */
#define VK_BUF_SZ(nm, sz) do { \
    if (qw6_vk_buffer_create(&ctx->vk, &ctx->nm, (sz) * sizeof(float), \
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) \
        { qw6_vk_pipe_free(ctx); return -1; } \
} while(0)
    VK_BUF_SZ(scr_hidden, QW6_HIDDEN_SIZE);
    VK_BUF_SZ(scr_resid, QW6_HIDDEN_SIZE);
    VK_BUF_SZ(scr_normed, QW6_HIDDEN_SIZE);
    VK_BUF_SZ(scr_norm_w, QW6_HIDDEN_SIZE);
    VK_BUF_SZ(scr_attn, QW6_HIDDEN_SIZE);
    VK_BUF_SZ(scr_ffn, QW6_HIDDEN_SIZE);
    VK_BUF_SZ(scr_s0, QW6_LINEAR_QKV_DIM);
    VK_BUF_SZ(scr_s1, QW6_LINEAR_QKV_DIM);
    VK_BUF_SZ(scr_logits, QW6_VOCAB_SIZE);
    /* Small uint32 sample output buffer */
    if (qw6_vk_buffer_create(&ctx->vk, &ctx->scr_sample, sizeof(uint32_t),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0)
        { qw6_vk_pipe_free(ctx); return -1; }
    if (qw6_vk_buffer_create(&ctx->vk, &ctx->scr_moe_idx,
                             QW6_EXPERTS_PER_TOK * sizeof(uint32_t),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0)
        { qw6_vk_pipe_free(ctx); return -1; }
    if (qw6_vk_buffer_create(&ctx->vk, &ctx->scr_moe_w,
                             QW6_EXPERTS_PER_TOK * sizeof(float),
                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0)
        { qw6_vk_pipe_free(ctx); return -1; }
#undef VK_BUF_SZ

    /* ---- Upload embedding and output tensors ---- */
    qw6_vk_upload_tensor(ctx, &m->tok_embeddings);
    qw6_vk_upload_tensor(ctx, &m->output);
    qw6_vk_upload_tensor(ctx, &m->output_norm);

    /* ---- Upload per-layer tensors ---- */
    for (int l = 0; l < QW6_NUM_LAYERS; l++) {
        qw6_vk_upload_tensor(ctx, &m->layers[l].norm);
        qw6_vk_upload_tensor(ctx, &m->layers[l].post_norm);
        qw6_vk_upload_tensor(ctx, &m->layers[l].attn_q);
        qw6_vk_upload_tensor(ctx, &m->layers[l].attn_k);
        qw6_vk_upload_tensor(ctx, &m->layers[l].attn_v);
        qw6_vk_upload_tensor(ctx, &m->layers[l].attn_o);
        qw6_vk_upload_tensor(ctx, &m->layers[l].attn_gate);
        qw6_vk_upload_tensor(ctx, &m->layers[l].attn_q_norm);
        qw6_vk_upload_tensor(ctx, &m->layers[l].attn_k_norm);
        qw6_vk_upload_tensor(ctx, &m->layers[l].conv1d);
        qw6_vk_upload_tensor(ctx, &m->layers[l].dn_key);
        qw6_vk_upload_tensor(ctx, &m->layers[l].dn_value);
        qw6_vk_upload_tensor(ctx, &m->layers[l].dn_query);
        qw6_vk_upload_tensor(ctx, &m->layers[l].dn_out);
        qw6_vk_upload_tensor(ctx, &m->layers[l].dn_gate);
        qw6_vk_upload_tensor(ctx, &m->layers[l].dn_norm);
        qw6_vk_upload_tensor(ctx, &m->layers[l].dn_alpha);
        qw6_vk_upload_tensor(ctx, &m->layers[l].dn_beta);
        qw6_vk_upload_tensor(ctx, &m->layers[l].dn_dt);
        qw6_vk_upload_tensor(ctx, &m->layers[l].dn_a);
        qw6_vk_upload_tensor(ctx, &m->layers[l].moe_router);
        qw6_vk_upload_tensor(ctx, &m->layers[l].shared_gate);
        qw6_vk_upload_tensor(ctx, &m->layers[l].shared_up);
        qw6_vk_upload_tensor(ctx, &m->layers[l].shared_down);
        qw6_vk_upload_tensor(ctx, &m->layers[l].shared_router);
        /* Upload all 256 expert gate/up/down tensors */
        for (int e = 0; e < QW6_NUM_EXPERTS; e++) {
            qw6_vk_upload_tensor(ctx, &m->layers[l].expert_gate[e]);
            qw6_vk_upload_tensor(ctx, &m->layers[l].expert_up[e]);
            qw6_vk_upload_tensor(ctx, &m->layers[l].expert_down[e]);
        }
    }

    /* ---- Allocate KV caches (full-attn layers only) ---- */
    size_t kv_ctx = ctx->max_ctx < 4096 ? 4096 : ctx->max_ctx;
    size_t kv_layer_bytes = (size_t)kv_ctx * QW6_NUM_KV_HEADS * QW6_HEAD_DIM * sizeof(float);
    for (int l = 0; l < QW6_NUM_LAYERS; l++) {
        if (qw6_layer_type(l) != QW6_LAYER_FULL_ATTN) continue;
        if (qw6_vk_buffer_create(&ctx->vk, &ctx->k_cache[l], kv_layer_bytes,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
            qw6_vk_pipe_free(ctx); return -1;
        }
        if (qw6_vk_buffer_create(&ctx->vk, &ctx->v_cache[l], kv_layer_bytes,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
            qw6_vk_pipe_free(ctx); return -1;
        }
    }

    /* ---- Allocate DeltaNet state buffers (linear-attn layers) ---- */
    size_t dn_bytes = (size_t)QW6_NUM_VALUE_HEADS * QW6_VALUE_HEAD_DIM * QW6_VALUE_HEAD_DIM * sizeof(float);
    size_t conv_bytes = (size_t)QW6_CONV1D_KERNEL * QW6_LINEAR_QKV_DIM * sizeof(float);
    for (int l = 0; l < QW6_NUM_LAYERS; l++) {
        if (qw6_layer_type(l) != QW6_LAYER_LINEAR_ATTN) continue;
        if (qw6_vk_buffer_create(&ctx->vk, &ctx->dn_state[l], dn_bytes,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
            qw6_vk_pipe_free(ctx); return -1;
        }
        memset(ctx->dn_state[l].mapped, 0, dn_bytes);
        if (qw6_vk_buffer_create(&ctx->vk, &ctx->conv_state[l], conv_bytes,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0) {
            qw6_vk_pipe_free(ctx); return -1;
        }
        memset(ctx->conv_state[l].mapped, 0, conv_bytes);
    }

    fprintf(stderr, "qw6_vk_pipe: %zu bytes weights uploaded, %u max_ctx\n",
            ctx->weight_capacity, ctx->max_ctx);
    /* Pre-dequantize attn_o weights for GPU (Q5_K shader has a bug) */
    for (int _li = 0; _li < QW6_NUM_LAYERS; _li++) {
        qw6_tensor_t *ao = &m->layers[_li].attn_o;
        if (!ao->data || ao->vk_offset == (size_t)-1 || ao->data_size < 64) continue;
        uint32_t nrows = ao->rows, ncols = ao->cols;
        size_t f32_sz = (size_t)nrows * ncols * sizeof(float);
        /* Extend the weight buffer to hold dequantized version */
        size_t deq_off = ctx->weight_capacity;
        deq_off = (deq_off + 255) & ~(size_t)255;
        if (deq_off + f32_sz > ctx->weights.size) {
            fprintf(stderr, "qw6_vk_pipe: L%d attn_o dequant buffer too small\n", _li);
            continue;
        }
        float *deq_ptr = (float *)((uint8_t *)ctx->weights.mapped + deq_off);
        /* Dequantize each row */
        for (uint32_t _r = 0; _r < nrows; _r++) {
            if (qw6_tensor_dequantize_row(ao, _r, deq_ptr + _r * ncols) != 0)
                break;
        }
        /* Change quant type to FP32 and point to dequantized data */
        ao->quant = QW6_Q_FP32;
        ao->data = deq_ptr;
        ao->vk_offset = deq_off;
        ao->data_size = f32_sz;
        ctx->weight_capacity = deq_off + f32_sz;
        fprintf(stderr, "qw6_vk_pipe: L%d attn_o dequantized to FP32 (%zu MB)\n",
                _li, f32_sz / (1024*1024));
    }
    *p = ctx;
    return 0;
}

/* ---- Forward pass ---- */

int qw6_vk_pipe_forward(qw6_vk_pipe_t *p, qw6_model_t *m,
                        uint32_t token, uint32_t pos,
                        float *logits_out) {
    if (!p || !m || !logits_out) return -1;

    /* Step 1: Token embedding lookup.
     * Dequantize tok_embeddings[token] into hidden buffer.
     * Since embedding is a single row, we copy from mmap'd data directly.
     * For GPU, we upload the row and dispatch matvec with identity input. */
    const qw6_tensor_t *emb = &m->tok_embeddings;
    if (!emb->data || emb->vk_offset == (size_t)-1) {
        fprintf(stderr, "qw6_vk_pipe: tok_embeddings not uploaded "
                "(data=%p vk_offset=%zu)\n",
                (void *)emb->data, emb->vk_offset);
        return -1;
    }

    /* Upload embeddings row (it's Q5_K -- dequant on CPU for now, then copy to GPU) */
    float *hidden_cpu = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    if (!hidden_cpu) return -1;
    if (qw6_tensor_dequantize_row(emb, (int)token, hidden_cpu) != 0) {
        free(hidden_cpu);
        return -1;
    }
    memcpy(p->scr_hidden.mapped, hidden_cpu, QW6_HIDDEN_SIZE * sizeof(float));


    /* Scratch buffers for inter-layer state */
    qw6_vk_buffer_t *hid = &p->scr_hidden;
    qw6_vk_buffer_t *res = &p->scr_resid;
    qw6_vk_buffer_t *nrm = &p->scr_normed;
    qw6_vk_buffer_t *nw  = &p->scr_norm_w;
    qw6_vk_buffer_t *att = &p->scr_attn;
    qw6_vk_buffer_t *ffn = &p->scr_ffn;
    qw6_vk_buffer_t *s0  = &p->scr_s0;
    qw6_vk_buffer_t *s1  = &p->scr_s1;
    qw6_vk_buffer_t *lg  = &p->scr_logits;

    /* Helper: dispatch RMSNorm on GPU using persistent buffers */
#define VK_RMSNORM(x, w, y) \
    qw6_vk_pipe_rmsnorm(p, x, w, y, QW6_HIDDEN_SIZE, QW6_RMS_EPS)

    /* Helper: dispatch a matmul for tensor t, using x as input, y as output.
     * Tries GPU first; if quant not supported, falls back to CPU. */
#define VK_MATMUL(t, x, y, rows, cols) do { \
        if ((t)->vk_offset != (size_t)-1 && \
            qw6_vk_pipe_matvec(p, t, x, y, rows, cols) == 0) { \
            /* GPU dispatch succeeded */ \
        } else if (qw6_vk_strict_fallback(p, "matmul") != 0) { \
            free(hidden_cpu); \
            return -1; \
        } else { \
            /* Fall back to CPU */ \
            float *_x = (float *)(x)->mapped; \
            float *_y = (float *)(y)->mapped; \
            qw6_tensor_matvec(_y, t, _x, rows); \
            memcpy((y)->mapped, _y, (rows) * sizeof(float)); \
        } \
    } while(0)

    /* Helper: GPU element-wise add */
#define VK_ADD(dst, src, n) \
    qw6_vk_pipe_add(p, dst, src, dst, n)

    /* Helper: GPU SiLU multiply */
#define VK_SILU_MUL(gate, up, out, n) \
    qw6_vk_pipe_silu_mul(p, gate, up, out, n)

    /* Helper: read back a GPU buffer to CPU */
#define VK_READ(buf, dst, n) memcpy(dst, (buf)->mapped, (n) * sizeof(float))

    /* Helper: write CPU data to GPU buffer */
#define VK_WRITE(buf, src, n) memcpy((buf)->mapped, src, (n) * sizeof(float))



    /* ---- Layer loop ---- */
    for (int l = 0; l < QW6_NUM_LAYERS; l++) {
        qw6_vk_buffer_t *att_out = att;

        /* Copy hidden -> resid */
        memcpy(res->mapped, hid->mapped, QW6_HIDDEN_SIZE * sizeof(float));

        /* Pre-attention RMSNorm: normed = rmsnorm(hidden, norm_w) */
        /* Upload norm weights to GPU scratch */
        VK_WRITE(nw, ((float *)m->layers[l].norm.data), QW6_HIDDEN_SIZE);
        VK_RMSNORM(hid, nw, nrm);

        if (qw6_layer_type(l) == QW6_LAYER_LINEAR_ATTN) {
            /* ---- Linear Attention (Gated DeltaNet) ---- */
            /* QKV projection: attn_q = [qkv_dim, hidden] */
            VK_MATMUL(&m->layers[l].attn_q, nrm, s0,
                      QW6_LINEAR_QKV_DIM, QW6_HIDDEN_SIZE);

            /* Z (gate) projection */
            VK_MATMUL(&m->layers[l].attn_gate, nrm, s1,
                      QW6_LINEAR_VALUE_DIM, QW6_HIDDEN_SIZE);

            /* Beta, Alpha projections */
            VK_MATMUL(&m->layers[l].dn_beta, nrm, att,  /* reuse attn as tmp */
                      QW6_NUM_VALUE_HEADS, QW6_HIDDEN_SIZE);
            float *beta_cpu = (float *)att->mapped;

            VK_MATMUL(&m->layers[l].dn_alpha, nrm, ffn,  /* reuse ffn as tmp */
                      QW6_NUM_VALUE_HEADS, QW6_HIDDEN_SIZE);
            float *alpha_cpu = (float *)ffn->mapped;

            /* Conv1D state: shift + causal conv + SiLU */
            float *qkv = (float *)s0->mapped;
            float *z   = (float *)s1->mapped;
            float *cs  = (float *)p->conv_state[l].mapped;
            float *conv_out = (float *)s0->mapped; /* reuse s0 */

            if (qw6_vk_pipe_conv1d_f32_state(p, &p->conv_state[l], s0,
                                             &m->layers[l].conv1d, s0,
                                             QW6_LINEAR_QKV_DIM,
                                             QW6_CONV1D_KERNEL) != 0) {
                if (qw6_vk_strict_fallback(p, "conv1d") != 0) { free(hidden_cpu); return -1; }
                memmove(cs, cs + QW6_LINEAR_QKV_DIM,
                        (size_t)(QW6_CONV1D_KERNEL - 1) * QW6_LINEAR_QKV_DIM * sizeof(float));
                memcpy(cs + (size_t)(QW6_CONV1D_KERNEL - 1) * QW6_LINEAR_QKV_DIM,
                       qkv, QW6_LINEAR_QKV_DIM * sizeof(float));

                const float *cw = (const float *)m->layers[l].conv1d.data;
                for (int c = 0; c < QW6_LINEAR_QKV_DIM; c++) {
                    float sum = 0.0f;
                    for (int k = 0; k < QW6_CONV1D_KERNEL; k++)
                        sum += cs[(size_t)k * QW6_LINEAR_QKV_DIM + c] *
                               cw[(size_t)c * QW6_CONV1D_KERNEL + k];
                    conv_out[c] = qw6_silu(sum);
                }
            }

            /* Split QKV, L2-normalize heads (GPU with CPU fallback) */
            float *q = conv_out;
            float *k = conv_out + QW6_NUM_KEY_HEADS * QW6_KEY_HEAD_DIM;
            float *v = conv_out + 2 * QW6_NUM_KEY_HEADS * QW6_KEY_HEAD_DIM;
            size_t k_off = (size_t)QW6_NUM_KEY_HEADS * QW6_KEY_HEAD_DIM * sizeof(float);
            if (qw6_vk_pipe_l2_norm_heads(p, s0, 0,
                                          QW6_NUM_KEY_HEADS, QW6_KEY_HEAD_DIM) != 0) {
                if (qw6_vk_strict_fallback(p, "l2_norm_heads(q)") != 0) { free(hidden_cpu); return -1; }
                qw6_l2_norm_heads(q, QW6_NUM_KEY_HEADS, QW6_KEY_HEAD_DIM);
            }
            if (qw6_vk_pipe_l2_norm_heads(p, s0, k_off,
                                          QW6_NUM_KEY_HEADS, QW6_KEY_HEAD_DIM) != 0) {
                if (qw6_vk_strict_fallback(p, "l2_norm_heads(k)") != 0) { free(hidden_cpu); return -1; }
                qw6_l2_norm_heads(k, QW6_NUM_KEY_HEADS, QW6_KEY_HEAD_DIM);
            }

            /* Apply sigmoid/softplus to beta/alpha (GPU with CPU fallback) */
            if (qw6_vk_pipe_alpha_beta(p, ffn, att,
                                       &m->layers[l].dn_dt,
                                       &m->layers[l].dn_a,
                                       QW6_NUM_VALUE_HEADS) != 0) {
                if (qw6_vk_strict_fallback(p, "alpha_beta") != 0) { free(hidden_cpu); return -1; }
                float *dt_cpu = (float *)m->layers[l].dn_dt.data;
                float *a_cpu  = (float *)m->layers[l].dn_a.data;
                for (int h = 0; h < QW6_NUM_VALUE_HEADS; h++) {
                    beta_cpu[h] = qw6_sigmoid(beta_cpu[h]);
                    alpha_cpu[h] = qw6_softplus(alpha_cpu[h] + dt_cpu[h]) * a_cpu[h];
                }
            }

            /* Fused Gated DeltaNet recurrent update + retrieve. */
            float *gdn = (float *)nrm->mapped; /* reuse nrm for GDN output */
            if (qw6_vk_pipe_deltanet_gated(p, &p->dn_state[l], s0, ffn, att, nrm,
                                           QW6_NUM_KEY_HEADS,
                                           QW6_NUM_VALUE_HEADS,
                                           QW6_VALUE_HEAD_DIM) != 0) {
                if (qw6_vk_strict_fallback(p, "deltanet_gated") != 0) { free(hidden_cpu); return -1; }
                float *state = (float *)p->dn_state[l].mapped;
                qw6_gated_delta_net_single(gdn, state, q, k, v, alpha_cpu, beta_cpu);
                memcpy(p->dn_state[l].mapped, state,
                       (size_t)QW6_NUM_VALUE_HEADS * QW6_VALUE_HEAD_DIM * QW6_VALUE_HEAD_DIM * sizeof(float));
            }

            /* Fused output RMSNorm + SiLU(z) gate (GPU with CPU fallback) */
            if (qw6_vk_pipe_deltanet_norm_gate(p, nrm, &m->layers[l].dn_norm,
                                               s0, s1,
                                               QW6_NUM_VALUE_HEADS,
                                               QW6_VALUE_HEAD_DIM) != 0) {
                if (qw6_vk_strict_fallback(p, "deltanet_norm_gate") != 0) { free(hidden_cpu); return -1; }
                float *dn_norm_w = (float *)m->layers[l].dn_norm.data;
                float *gated = (float *)s1->mapped;
                float *gdn_cpu = (float *)nrm->mapped;
                for (int h = 0; h < QW6_NUM_VALUE_HEADS; h++) {
                    float *dst = gated + h * QW6_VALUE_HEAD_DIM;
                    qw6_cpu_rmsnorm(dst, gdn_cpu + h * QW6_VALUE_HEAD_DIM,
                                    dn_norm_w + h * QW6_VALUE_HEAD_DIM,
                                    QW6_VALUE_HEAD_DIM);
                    for (int i = 0; i < QW6_VALUE_HEAD_DIM; i++)
                        dst[i] *= qw6_silu(z[h * QW6_VALUE_HEAD_DIM + i]);
                }
                memcpy(s1->mapped, gated, QW6_LINEAR_VALUE_DIM * sizeof(float));
            }

            /* Output projection: dn_out * gated -> attn */
            VK_MATMUL(&m->layers[l].dn_out, s1, att,
                      QW6_HIDDEN_SIZE, QW6_LINEAR_VALUE_DIM);
            att_out = att;

        } else {
            /* ---- Full Attention (GQA) ---- */
            /* Q projection (with gate) */
            VK_MATMUL(&m->layers[l].attn_q, nrm, s0,
                      QW6_NUM_Q_HEADS * QW6_HEAD_DIM * 2, QW6_HIDDEN_SIZE);

            /* K and V projections */
            VK_MATMUL(&m->layers[l].attn_k, nrm, s1,
                      QW6_NUM_KV_HEADS * QW6_HEAD_DIM, QW6_HIDDEN_SIZE);
            VK_MATMUL(&m->layers[l].attn_v, nrm, ffn,
                      QW6_NUM_KV_HEADS * QW6_HEAD_DIM, QW6_HIDDEN_SIZE);

            /* Q/K norms */
            float *k_cpu = (float *)s1->mapped;
            float *q_cpu = (float *)s0->mapped;
            float *gate_cpu = (float *)s0->mapped + QW6_NUM_Q_HEADS * QW6_HEAD_DIM;

            /* Apply Q/K RMSNorm per head (GPU with CPU fallback) */
            if (qw6_vk_pipe_rmsnorm_heads(p, s0, &m->layers[l].attn_q_norm,
                                          QW6_NUM_Q_HEADS, QW6_HEAD_DIM) != 0) {
                if (qw6_vk_strict_fallback(p, "attn_q_norm") != 0) { free(hidden_cpu); return -1; }
                float *qnw = (float *)m->layers[l].attn_q_norm.data;
                for (int h = 0; h < QW6_NUM_Q_HEADS; h++)
                    qw6_cpu_rmsnorm(q_cpu + h * QW6_HEAD_DIM,
                                    q_cpu + h * QW6_HEAD_DIM,
                                    qnw, QW6_HEAD_DIM);
            }
            if (qw6_vk_pipe_rmsnorm_heads(p, s1, &m->layers[l].attn_k_norm,
                                          QW6_NUM_KV_HEADS, QW6_HEAD_DIM) != 0) {
                if (qw6_vk_strict_fallback(p, "attn_k_norm") != 0) { free(hidden_cpu); return -1; }
                float *knw = (float *)m->layers[l].attn_k_norm.data;
                for (int h = 0; h < QW6_NUM_KV_HEADS; h++)
                    qw6_cpu_rmsnorm(k_cpu + h * QW6_HEAD_DIM,
                                    k_cpu + h * QW6_HEAD_DIM,
                                    knw, QW6_HEAD_DIM);
            }

            /* MRoPE */
            {
                int rot_dim = (int)(QW6_HEAD_DIM * QW6_PARTIAL_ROTARY);
                if (qw6_vk_pipe_mrope(p, s0, s1,
                                       QW6_NUM_Q_HEADS * QW6_HEAD_DIM,
                                       QW6_NUM_KV_HEADS * QW6_HEAD_DIM,
                                       QW6_NUM_Q_HEADS, QW6_NUM_KV_HEADS,
                                       pos, (uint32_t)rot_dim) != 0) {
                    if (qw6_vk_strict_fallback(p, "mrope") != 0) { free(hidden_cpu); return -1; }
                    qw6_cpu_mrope(q_cpu, k_cpu,
                                  QW6_NUM_Q_HEADS * QW6_HEAD_DIM,
                                  QW6_NUM_KV_HEADS * QW6_HEAD_DIM,
                                  QW6_NUM_Q_HEADS, QW6_NUM_KV_HEADS,
                                  pos, rot_dim);
                }
            }

            /* Write K,V to KV cache (GPU copy with CPU fallback) */
            size_t kv_pos = (size_t)pos * QW6_NUM_KV_HEADS * QW6_HEAD_DIM;
            size_t kv_bytes = (size_t)QW6_NUM_KV_HEADS * QW6_HEAD_DIM * sizeof(float);
            size_t k_cache_off = kv_pos * sizeof(float);
            size_t v_cache_off = kv_pos * sizeof(float);
            if (qw6_vk_pipe_buf_copy(p, s1, 0, &p->k_cache[l], k_cache_off,
                                      QW6_NUM_KV_HEADS * QW6_HEAD_DIM) != 0) {
                if (qw6_vk_strict_fallback(p, "kv_copy_k") != 0) { free(hidden_cpu); return -1; }
                float *k_slot = (float *)p->k_cache[l].mapped + kv_pos;
                memcpy(k_slot, k_cpu, kv_bytes);
            }
            if (qw6_vk_pipe_buf_copy(p, ffn, 0, &p->v_cache[l], v_cache_off,
                                      QW6_NUM_KV_HEADS * QW6_HEAD_DIM) != 0) {
                if (qw6_vk_strict_fallback(p, "kv_copy_v") != 0) { free(hidden_cpu); return -1; }
                float *v_slot = (float *)p->v_cache[l].mapped + kv_pos;
                memcpy(v_slot, (float *)ffn->mapped, kv_bytes);
            }

            /* GQA attention. Current shader uses a 256-entry shared score
             * buffer, so keep CPU fallback for longer contexts. */
            float *attn_cpu = (float *)att->mapped;
            if (pos < 256 &&
                qw6_vk_pipe_attention_gqa(p, s0, &p->k_cache[l], &p->v_cache[l], att,
                                          (uint32_t)pos + 1,
                                          QW6_NUM_Q_HEADS, QW6_NUM_KV_HEADS,
                                          QW6_HEAD_DIM) == 0) {
                /* GPU path wrote attn_cpu through the mapped buffer. */
            } else {
                if (pos >= 256 && qw6_vk_strict_fallback(p, "attention_gqa_context") != 0) { free(hidden_cpu); return -1; }
                qw6_cpu_attention_gqa(attn_cpu, q_cpu,
                                      (float *)p->k_cache[l].mapped,
                                      (float *)p->v_cache[l].mapped,
                                      (int)pos + 1,
                                      QW6_NUM_Q_HEADS, QW6_NUM_KV_HEADS, QW6_HEAD_DIM);
            }

            /* Apply gate: att *= sigmoid(gate) (GPU with CPU fallback) */
            size_t gate_off = (size_t)QW6_NUM_Q_HEADS * QW6_HEAD_DIM * sizeof(float);
            if (qw6_vk_pipe_sigmoid_mul(p, s0, gate_off, att, att,
                                        QW6_NUM_Q_HEADS * QW6_HEAD_DIM) != 0) {
                if (qw6_vk_strict_fallback(p, "sigmoid_mul") != 0) { free(hidden_cpu); return -1; }
                for (int i = 0; i < QW6_NUM_Q_HEADS * QW6_HEAD_DIM; i++)
                    attn_cpu[i] *= qw6_sigmoid(gate_cpu[i]);
                memcpy(att->mapped, attn_cpu,
                       QW6_NUM_Q_HEADS * QW6_HEAD_DIM * sizeof(float));
            }

            /* Output projection. attn_o is pre-dequantized to FP32 at init
             * because the Q5_K shader is still numerically wrong for it. */
            VK_MATMUL(&m->layers[l].attn_o, att, s1,
                      QW6_HIDDEN_SIZE, QW6_NUM_Q_HEADS * QW6_HEAD_DIM);
            att_out = s1;
        }

        /* Residual: hidden = resid + attn */
        VK_ADD(hid, att_out, QW6_HIDDEN_SIZE);


        /* ---- MoE Block ---- */
        memcpy(res->mapped, hid->mapped, QW6_HIDDEN_SIZE * sizeof(float));

        /* Post-attention RMSNorm */
        VK_WRITE(nw, ((float *)m->layers[l].post_norm.data), QW6_HIDDEN_SIZE);
        VK_RMSNORM(hid, nw, nrm);

        /* Router matvec: 256 experts, top-k=8 */
        VK_MATMUL(&m->layers[l].moe_router, nrm, s0,
                  QW6_NUM_EXPERTS, QW6_HIDDEN_SIZE);

        /* GPU MoE routing with CPU fallback */
        int idx[QW6_EXPERTS_PER_TOK] = {0};
        float w[QW6_EXPERTS_PER_TOK] = {0};
        if (qw6_vk_pipe_moe_route(p, s0, &p->scr_moe_idx, &p->scr_moe_w,
                                  QW6_NUM_EXPERTS, QW6_EXPERTS_PER_TOK) != 0) {
            if (qw6_vk_strict_fallback(p, "moe_route") != 0) { free(hidden_cpu); return -1; }
            float *router_logits = (float *)s0->mapped;
            qw6_cpu_moe_route(idx, w, router_logits,
                              QW6_NUM_EXPERTS, QW6_EXPERTS_PER_TOK);
        } else {
            /* Read back GPU routing results (only 8 indices + 8 weights) */
            uint32_t idx32[QW6_EXPERTS_PER_TOK];
            memcpy(idx32, p->scr_moe_idx.mapped, sizeof(idx32));
            memcpy(w, p->scr_moe_w.mapped, sizeof(w));
            for (int e = 0; e < QW6_EXPERTS_PER_TOK; e++) idx[e] = (int)idx32[e];
        }

        /* Zero out FFN output on GPU */
        memset(ffn->mapped, 0, QW6_HIDDEN_SIZE * sizeof(float));

        /* For each selected expert, dispatch gate/up/down matmuls */
        for (int e = 0; e < QW6_EXPERTS_PER_TOK; e++) {
            int expert = idx[e];
            const qw6_tensor_t *eg = &m->layers[l].expert_gate[expert];
            const qw6_tensor_t *eu = &m->layers[l].expert_up[expert];
            const qw6_tensor_t *ed = &m->layers[l].expert_down[expert];

            /* gate = expert_gate * normed, up = expert_up * normed */
            VK_MATMUL(eg, nrm, s1, QW6_MOE_INTER, QW6_HIDDEN_SIZE);
            VK_MATMUL(eu, nrm, att, QW6_MOE_INTER, QW6_HIDDEN_SIZE);

            /* SiLU gate * up -> mid (GPU dispatch) */
            VK_SILU_MUL(s1, att, s0, QW6_MOE_INTER);

            /* down = expert_down * mid  (reuse nrm buffer) */
            VK_MATMUL(ed, s0, nrm, QW6_HIDDEN_SIZE, QW6_MOE_INTER);

            /* Accumulate: ffn += w[e] * nrm (GPU axpy with CPU fallback) */
            if (qw6_vk_pipe_axpy(p, ffn, nrm, w[e],
                                 QW6_HIDDEN_SIZE) != 0) {
                if (qw6_vk_strict_fallback(p, "axpy") != 0) { free(hidden_cpu); return -1; }
                float *ffn_p = (float *)ffn->mapped;
                float *tmp_p = (float *)nrm->mapped;
                for (int i = 0; i < QW6_HIDDEN_SIZE; i++)
                    ffn_p[i] += w[e] * tmp_p[i];
                memcpy(ffn->mapped, ffn_p, QW6_HIDDEN_SIZE * sizeof(float));
            }
        }

        /* Shared expert */
        if (m->layers[l].shared_gate.data &&
            m->layers[l].shared_up.data &&
            m->layers[l].shared_down.data) {
            float shared_weight = 1.0f;
            if (m->layers[l].shared_router.data) {
                float sw = 0;
                VK_MATMUL(&m->layers[l].shared_router, nrm, att, 1, QW6_HIDDEN_SIZE);
                sw = *(float *)att->mapped;
                shared_weight = qw6_sigmoid(sw);
            }
            VK_MATMUL(&m->layers[l].shared_gate, nrm, s0,
                      QW6_SHARED_INTER, QW6_HIDDEN_SIZE);
            VK_MATMUL(&m->layers[l].shared_up, nrm, s1,
                      QW6_SHARED_INTER, QW6_HIDDEN_SIZE);
            /* mid = silu(sg) * su (GPU dispatch) */
            VK_SILU_MUL(s0, s1, s0, QW6_SHARED_INTER);

            VK_MATMUL(&m->layers[l].shared_down, s0, nrm,
                      QW6_HIDDEN_SIZE, QW6_SHARED_INTER);

            /* Accumulate: ffn += shared_weight * nrm (GPU axpy with CPU fallback) */
            if (qw6_vk_pipe_axpy(p, ffn, nrm, shared_weight,
                                 QW6_HIDDEN_SIZE) != 0) {
                if (qw6_vk_strict_fallback(p, "shared_axpy") != 0) { free(hidden_cpu); return -1; }
                float *ffn_p = (float *)ffn->mapped;
                float *tmp_p = (float *)nrm->mapped;
                for (int i = 0; i < QW6_HIDDEN_SIZE; i++)
                    ffn_p[i] += shared_weight * tmp_p[i];
                memcpy(ffn->mapped, ffn_p, QW6_HIDDEN_SIZE * sizeof(float));
            }
        }

        /* Residual: hidden = resid + ffn */
        VK_ADD(hid, ffn, QW6_HIDDEN_SIZE);
    }


    /* ---- Output: Final RMSNorm + output projection ---- */
    {
        VK_WRITE(nw, ((float *)m->output_norm.data), QW6_HIDDEN_SIZE);
        VK_RMSNORM(hid, nw, nrm);

        /* Output projection: [vocab, hidden] matvec. This is 248320 rows - large! */
        VK_MATMUL(&m->output, nrm, lg, QW6_VOCAB_SIZE, QW6_HIDDEN_SIZE);
    }

    /* Read back logits to CPU (NULL = skip, e.g. when only GPU sample is needed) */
    if (logits_out) {
        memcpy(logits_out, lg->mapped, QW6_VOCAB_SIZE * sizeof(float));
        p->read_bytes += (uint64_t)QW6_VOCAB_SIZE * sizeof(float);
    }
    free(hidden_cpu);
    return 0;
}

/* Greedy forward: does full forward pass but uses GPU argmax to avoid
 * reading back all 248k logits. Returns the sampled token ID.
 * If dump_logits_buf is non-NULL, also reads back the full logits for debugging. */
int qw6_vk_pipe_forward_greedy(qw6_vk_pipe_t *p, qw6_model_t *m,
                                uint32_t token, uint32_t pos,
                                uint32_t *out_token, float *dump_logits_buf) {
    /* Do the full forward pass */
    int rc = qw6_vk_pipe_forward(p, m, token, pos,
                                  dump_logits_buf ? dump_logits_buf : NULL);
    if (rc == 0) {
        /* GPU argmax on the scr_logits buffer */
        rc = qw6_vk_pipe_sample_greedy(p, &p->scr_logits, out_token);
    }
    return rc;
}

/* ---- Free ---- */

void qw6_vk_pipe_free(qw6_vk_pipe_t *p) {
    if (!p) return;
    if (p->profile) {
        fprintf(stderr, "qw6_vk profile: dispatches=%llu fallbacks=%llu timestamps=%s read=%llu write=%llu bytes\n",
                (unsigned long long)p->dispatch_count,
                (unsigned long long)p->fallback_count,
                p->vk.timestamp_supported ? "on" : "off",
                (unsigned long long)p->read_bytes,
                (unsigned long long)p->write_bytes);
        for (int i = 0; i < p->pipe_cache_count; i++) {
            vk_pipe_cache_t *c = &p->pipe_cache[i];
            if (!c->valid || c->calls == 0) continue;
            fprintf(stderr, "  %-32s calls=%llu setup=%.3f ms gpu=%.3f ms wall=%.3f ms avg=%.3f ms read=%llu write=%llu\n",
                    c->shader_path,
                    (unsigned long long)c->calls,
                    c->cpu_setup_ms,
                    c->gpu_ms,
                    c->total_ms,
                    c->total_ms / (double)c->calls,
                    (unsigned long long)c->read_bytes,
                    (unsigned long long)c->write_bytes);
        }
    }
    if (p->timestamp_queries) vkDestroyQueryPool(p->vk.device, p->timestamp_queries, NULL);
    for (int i = 0; i < p->pipe_cache_count; i++) {
        vk_pipe_cache_t *c = &p->pipe_cache[i];
        if (!c->valid) continue;
        if (c->pipeline) vkDestroyPipeline(p->vk.device, c->pipeline, NULL);
        if (c->sm) vkDestroyShaderModule(p->vk.device, c->sm, NULL);
        if (c->layout) vkDestroyPipelineLayout(p->vk.device, c->layout, NULL);
        if (c->dsl) vkDestroyDescriptorSetLayout(p->vk.device, c->dsl, NULL);
    }
    if (p->dispatch_fence) vkDestroyFence(p->vk.device, p->dispatch_fence, NULL);
    if (p->dispatch_cb) vkFreeCommandBuffers(p->vk.device, p->vk.command_pool, 1, &p->dispatch_cb);
    if (p->descriptor_pool) vkDestroyDescriptorPool(p->vk.device, p->descriptor_pool, NULL);
    if (p->weights.buffer) qw6_vk_buffer_destroy(&p->vk, &p->weights);
    if (p->iq2s_grid.buffer) qw6_vk_buffer_destroy(&p->vk, &p->iq2s_grid);
    if (p->iq3s_grid.buffer) qw6_vk_buffer_destroy(&p->vk, &p->iq3s_grid);
    if (p->scr_hidden.buffer) qw6_vk_buffer_destroy(&p->vk, &p->scr_hidden);
    if (p->scr_resid.buffer) qw6_vk_buffer_destroy(&p->vk, &p->scr_resid);
    if (p->scr_normed.buffer) qw6_vk_buffer_destroy(&p->vk, &p->scr_normed);
    if (p->scr_norm_w.buffer) qw6_vk_buffer_destroy(&p->vk, &p->scr_norm_w);
    if (p->scr_attn.buffer) qw6_vk_buffer_destroy(&p->vk, &p->scr_attn);
    if (p->scr_ffn.buffer) qw6_vk_buffer_destroy(&p->vk, &p->scr_ffn);
    if (p->scr_s0.buffer) qw6_vk_buffer_destroy(&p->vk, &p->scr_s0);
    if (p->scr_s1.buffer) qw6_vk_buffer_destroy(&p->vk, &p->scr_s1);
    if (p->scr_logits.buffer) qw6_vk_buffer_destroy(&p->vk, &p->scr_logits);
    if (p->scr_sample.buffer) qw6_vk_buffer_destroy(&p->vk, &p->scr_sample);
    if (p->scr_moe_idx.buffer) qw6_vk_buffer_destroy(&p->vk, &p->scr_moe_idx);
    if (p->scr_moe_w.buffer) qw6_vk_buffer_destroy(&p->vk, &p->scr_moe_w);
    for (int i = 0; i < QW6_NUM_LAYERS; i++) {
        if (qw6_layer_type(i) == QW6_LAYER_FULL_ATTN) {
            if (p->k_cache[i].buffer) qw6_vk_buffer_destroy(&p->vk, &p->k_cache[i]);
            if (p->v_cache[i].buffer) qw6_vk_buffer_destroy(&p->vk, &p->v_cache[i]);
        } else {
            if (p->dn_state[i].buffer) qw6_vk_buffer_destroy(&p->vk, &p->dn_state[i]);
            if (p->conv_state[i].buffer) qw6_vk_buffer_destroy(&p->vk, &p->conv_state[i]);
        }
    }
    qw6_vk_free(&p->vk);
    memset(p, 0, sizeof(*p));
    free(p);
}

#endif
