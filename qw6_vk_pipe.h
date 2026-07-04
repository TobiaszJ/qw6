#ifndef QW6_VK_PIPE_H
#define QW6_VK_PIPE_H

#ifdef QW6_VULKAN

#include "qw6.h"
#include "qw6_vk.h"
#include <stdint.h>

/* --- Persistent GPU pipeline context --- */

typedef struct {
    /* Base Vulkan device */
    void *vk; /* points to qw6_vk_t (opaque in this header) */

    /* Model weights in one big GPU-visible buffer */
    void *weight_buf;    /* qw6_vk_buffer_t* */
    size_t weight_size;

    /* Persistent scratch buffers on GPU */
    void *buf_hidden;    /* qw6_vk_buffer_t* [HIDDEN_SIZE] */
    void *buf_resid;     /* qw6_vk_buffer_t* [HIDDEN_SIZE] */
    void *buf_normed;    /* qw6_vk_buffer_t* [HIDDEN_SIZE] */
    void *buf_norm_w;    /* qw6_vk_buffer_t* [HIDDEN_SIZE] */
    void *buf_attn;      /* qw6_vk_buffer_t* [HIDDEN_SIZE] */
    void *buf_ffn;       /* qw6_vk_buffer_t* [HIDDEN_SIZE] */

    /* Intermediate compute buffers */
    void *buf_scratch0;  /* qw6_vk_buffer_t* [max(HIDDEN_SIZE*2, VOCAB_SIZE)] */
    void *buf_scratch1;  /* qw6_vk_buffer_t* [max(HIDDEN_SIZE*2, VOCAB_SIZE)] */
    void *buf_scratch2;  /* qw6_vk_buffer_t* [HIDDEN_SIZE] */
    void *buf_scratch3;  /* qw6_vk_buffer_t* [HIDDEN_SIZE] */
    void *buf_scratch4;  /* qw6_vk_buffer_t* [LINEAR_QKV_DIM] */

    /* Logits buffer (read back to CPU after forward) */
    void *buf_logits;    /* qw6_vk_buffer_t* [VOCAB_SIZE] */

    /* Per-layer KV cache buffers (full-attn layers only) */
    void *k_cache[QW6_NUM_LAYERS];  /* qw6_vk_buffer_t* */
    void *v_cache[QW6_NUM_LAYERS];  /* qw6_vk_buffer_t* */

    /* Per-layer DeltaNet state (linear-attn layers only) */
    void *dn_state[QW6_NUM_LAYERS]; /* qw6_vk_buffer_t* */

    /* Conv1D state (linear-attn layers) */
    void *conv_state[QW6_NUM_LAYERS]; /* qw6_vk_buffer_t* */

    /* Pipeline cache */
    void *pipe_rmsnorm;
    void *pipe_rmsnorm_full;
    void *pipe_matvec_f32;
    void *pipe_matmul_q4k;
    void *pipe_matmul_q5k;
    void *pipe_matmul_q6k;
    void *pipe_matmul_iq2xxs;
    void *pipe_silu_mul;
    void *pipe_argmax;
    void *pipe_rope_mrope;
    void *pipe_attention_gqa;
    void *pipe_conv1d;
    void *pipe_deltanet_retrieve;
    void *pipe_deltanet_update;
    void *pipe_moe_route;
    void *pipe_moe_gather;

    /* Tensor weight offsets within the big weight buffer */
    size_t *tensor_offsets;  /* array indexed by tensor_idx */
    int n_tensors;
} qw6_vk_pipe_t;

/* --- API --- */

/* Initialize GPU pipeline: detect device, allocate buffers, upload weights */
int qw6_vk_pipe_init(qw6_vk_pipe_t *p, qw6_model_t *m);

/* Run one token forward pass entirely on GPU.
 * token: input token ID
 * pos:   position in sequence (for KV cache indexing and RoPE)
 * logits_out: [VOCAB_SIZE] output logits (written by GPU, read back to this buffer)
 * Returns 0 on success. */
int qw6_vk_pipe_forward(qw6_vk_pipe_t *p, qw6_model_t *m,
                        uint32_t token, uint32_t pos,
                        float *logits_out);

/* Free all GPU resources */
void qw6_vk_pipe_free(qw6_vk_pipe_t *p);

#endif /* QW6_VULKAN */
#endif /* QW6_VK_PIPE_H */
