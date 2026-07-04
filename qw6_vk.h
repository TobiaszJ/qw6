#ifndef QW6_VK_H
#define QW6_VK_H

#ifdef QW6_VULKAN

#include <stdint.h>

int qw6_vk_selftest(void);
int qw6_vk_matvec_f32_host(const float *w, const float *x, float *y,
                           uint32_t rows, uint32_t cols);
int qw6_vk_rmsnorm_host(const float *x, const float *w, float *y,
                        uint32_t n, float eps);

#endif

#endif
