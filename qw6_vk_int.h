#ifndef QW6_VK_INT_H
#define QW6_VK_INT_H

#ifdef QW6_VULKAN

#include <vulkan/vulkan.h>
#include <stdint.h>
#include <stddef.h>

/* Internal types shared between qw6_vk.c and qw6_vk_pipe.c */

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

/* Dispatch a compute shader with given buffers, push constants, and workgroup counts.
 * buffers[0..n_buffers-1] are storage buffers bound at bindings 0..n-1.
 * push points to push constant data of push_size bytes.
 * Returns 0 on success. */
int qw6_vk_dispatch_ext(qw6_vk_t *vk, const char *shader_path,
                        VkPipeline *cached_pipe,
                        qw6_vk_buffer_t **buffers, const size_t *offsets,
                        uint32_t n_buffers,
                        const void *push, uint32_t push_size,
                        uint32_t gx, uint32_t gy, uint32_t gz);

/* Create/destroy buffers */
int qw6_vk_buffer_create_ext(qw6_vk_t *vk, qw6_vk_buffer_t *b,
                             VkDeviceSize size, VkBufferUsageFlags usage);
void qw6_vk_buffer_destroy_ext(qw6_vk_t *vk, qw6_vk_buffer_t *b);

#endif /* QW6_VULKAN */
#endif /* QW6_VK_INT_H */
