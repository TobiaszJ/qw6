#ifndef QW6_VK_H
#define QW6_VK_H

#ifdef QW6_VULKAN

#include <stdint.h>
#include <stdbool.h>

int qw6_vk_selftest(void);
int qw6_vk_matvec_f32_host(const float *w, const float *x, float *y,
                           uint32_t rows, uint32_t cols);
int qw6_vk_matmul_q4k_host(const void *w, const float *x, float *y,
                           uint32_t rows, uint32_t cols);
int qw6_vk_matmul_q5k_host(const void *w, const float *x, float *y,
                           uint32_t rows, uint32_t cols);
int qw6_vk_matmul_q6k_host(const void *w, const float *x, float *y,
                           uint32_t rows, uint32_t cols);
int qw6_vk_matmul_iq2xxs_host(const void *w, const float *x, float *y,
                              uint32_t rows, uint32_t cols);
int qw6_vk_rmsnorm_host(const float *x, const float *w, float *y,
                        uint32_t n, float eps);
int qw6_vk_rmsnorm_chunked_host(const float *x, const float *w, float *y,
                                uint32_t n, float eps, uint32_t n_groups);
int qw6_vk_silu_mul_host(const float *gate, const float *up, float *out,
                         uint32_t n);
int qw6_vk_argmax_host(const float *x, uint32_t n, uint32_t *out_idx,
                       float *out_val);
int qw6_vk_sampling_greedy_host(const float *logits, uint32_t n,
                                uint32_t *token);
int qw6_vk_attention_gqa_host(const float *q, const float *k_cache,
                              const float *v_cache, float *out,
                              uint32_t seq_len, uint32_t n_q_heads,
                              uint32_t n_kv_heads, uint32_t head_dim);
int qw6_vk_moe_route_host(const float *logits, uint32_t *indices,
                          float *weights, uint32_t n_experts,
                          uint32_t top_k);
int qw6_vk_moe_gather_host(const float *expert_out,
                           const float *expert_weights,
                           const float *shared_out, float shared_weight,
                           float *out, uint32_t dim, uint32_t top_k);
int qw6_vk_deltanet_conv1d_host(const float *x, const uint16_t *w, float *out,
                                uint32_t dim, uint32_t kernel_size);
int qw6_vk_deltanet_retrieve_host(const float *state, const float *query,
                                  float *out, uint32_t key_heads,
                                  uint32_t key_dim, uint32_t val_heads,
                                  uint32_t val_dim);
int qw6_vk_deltanet_update_host(float *state, const float *key,
                                const float *value, const float *query,
                                const float *beta, uint32_t key_heads,
                                uint32_t key_dim, uint32_t val_heads,
                                uint32_t val_dim);
int qw6_vk_mrope_host(float *q, float *k, uint32_t q_dim, uint32_t kv_dim,
                      uint32_t n_heads, uint32_t n_kv_heads,
                      uint32_t position, uint32_t rotary_dim);
int qw6_vk_mtp_draft_host(const float *w, const float *hidden, float *logits,
                          uint32_t vocab_size, uint32_t hidden_size);

/* ---- GPU Pipeline (Phase 2 dispatch orchestration) ---- */

/* Opaque pipeline context */
typedef struct qw6_vk_pipe_s qw6_vk_pipe_t;

/* Initialize GPU pipeline: detect device, allocate buffers, upload weights */
int qw6_vk_pipe_init(qw6_vk_pipe_t **p, qw6_model_t *m, bool strict);

/* Run one token forward pass entirely on GPU.
 * token: input token ID
 * pos:   position in sequence (for KV cache indexing and RoPE)
 * logits_out: [VOCAB_SIZE] output logits (read back from GPU)
 * Returns 0 on success. */
int qw6_vk_pipe_forward(qw6_vk_pipe_t *p, qw6_model_t *m,
                        uint32_t token, uint32_t pos,
                        float *logits_out);

/* Greedy forward: same as qw6_vk_pipe_forward but uses GPU argmax
 * to read back only the sampled token ID, not all logits.
 * If dump_logits_buf is non-NULL, also reads back full logits for debugging. */
int qw6_vk_pipe_forward_greedy(qw6_vk_pipe_t *p, qw6_model_t *m,
                                uint32_t token, uint32_t pos,
                                uint32_t *out_token, float *dump_logits_buf);

/* Free all GPU resources */
void qw6_vk_pipe_free(qw6_vk_pipe_t *p);

#endif

#endif
