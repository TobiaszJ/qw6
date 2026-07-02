/* qw6.h — Qwen 3.6-35B-A3B inference engine header
 *
 * Model-specific inference engine for Qwen 3.6-35B-A3B on AMD BC-250.
 * Inspired by antirez/ds4 (DwarfStar4).
 *
 * Phase 1: CPU reference path (correctness, not speed).
 * Phase 2: Vulkan compute backend (GFX1013 / RADV).
 *
 * See ARCHITECTURE.md, MODEL_CARD.md, ROADMAP.md for design.
 */

#ifndef QW6_H
#define QW6_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>

/* ---- Compile-time model constants (from config.json) ---- */

#define QW6_NUM_LAYERS          40
#define QW6_NUM_FULL_ATTN       10  /* every 4th layer: 3,7,11,...,39 */
#define QW6_NUM_LINEAR_ATTN     30  /* the rest */
#define QW6_FULL_ATTN_INTERVAL  4

#define QW6_HIDDEN_SIZE         2048
#define QW6_NUM_Q_HEADS         16
#define QW6_NUM_KV_HEADS        2
#define QW6_HEAD_DIM            256

/* Gated DeltaNet (linear attention) */
#define QW6_NUM_KEY_HEADS       16
#define QW6_KEY_HEAD_DIM        128
#define QW6_NUM_VALUE_HEADS     32
#define QW6_VALUE_HEAD_DIM      128
#define QW6_CONV1D_KERNEL       4

/* MoE */
#define QW6_NUM_EXPERTS         256
#define QW6_EXPERTS_PER_TOK     8
#define QW6_MOE_INTER           512
#define QW6_SHARED_INTER        512

/* Vocabulary */
#define QW6_VOCAB_SIZE          248320
#define QW6_BOS_TOKEN_ID        248044
#define QW6_EOS_TOKEN_ID        248044

/* MTP */
#define QW6_MTP_LAYERS         1

/* MRoPE */
#define QW6_PARTIAL_ROTY       0.25f
#define QW6_MROPE_SECTIONS      3
#define QW6_MROPE_SEC_T        11
#define QW6_MROPE_SEC_H        11
#define QW6_MROPE_SEC_W        10
#define QW6_ROPE_THETA         10000000.0f

/* RMSNorm */
#define QW6_RMS_EPS            1e-6f

/* Context */
#define QW6_MAX_CONTEXT        262144
#define QW6_DEFAULT_CTX        65536
#define QW6_PREFILL_CHUNK      2048

/* ---- Quantisation types ---- */

typedef enum {
    QW6_Q_FP32 = 0,
    QW6_Q_FP16,
    QW6_Q_Q8_0,
    QW6_Q_Q4_K_M,
    QW6_Q_Q4_K_S,
    QW6_Q_Q3_K_M,
    QW6_Q_IQ3_XXS,
    QW6_Q_IQ2_M,
    QW6_Q_IQ2_XXS,
} qw6_quant_t;

/* ---- Tensor ---- */

typedef struct {
    char name[128];
    uint32_t rows;
    uint32_t cols;
    qw6_quant_t quant;
    void *data;         /* quantised data ( 格式取决于 quant) */
    size_t data_size;   /* bytes */
} qw6_tensor_t;

/* ---- Layer types ---- */

typedef enum {
    QW6_LAYER_LINEAR_ATTN = 0,
    QW6_LAYER_FULL_ATTN   = 1,
} qw6_layer_type_t;

/* ---- Model ---- */

typedef struct {
    /* Embeddings */
    qw6_tensor_t tok_embeddings;   /* [vocab, hidden] */
    qw6_tensor_t output;           /* [vocab, hidden] (untied) */

    /* Per-layer */
    struct {
        qw6_layer_type_t type;

        /* RMSNorm */
        qw6_tensor_t norm;             /* [hidden] FP32 */

        /* Full attention (type == FULL_ATTN only) */
        qw6_tensor_t attn_q;           /* [num_q*head_dim, hidden] */
        qw6_tensor_t attn_k;           /* [kv*head_dim, hidden] */
        qw6_tensor_t attn_v;           /* [kv*head_dim, hidden] */
        qw6_tensor_t attn_o;           /* [num_q*head_dim, hidden] */
        qw6_tensor_t attn_gate;        /* [hidden] FP16 — output gate */

        /* Linear attention (type == LINEAR_ATTN only) */
        qw6_tensor_t conv1d;           /* [hidden, 4] FP16 */
        qw6_tensor_t dn_key;           /* [key_heads*key_dim, hidden] */
        qw6_tensor_t dn_value;         /* [val_heads*val_dim, hidden] */
        qw6_tensor_t dn_query;         /* [val_heads*val_dim, hidden] */
        qw6_tensor_t dn_out;           /* [hidden, val_heads*val_dim] */
        qw6_tensor_t dn_gate;          /* [hidden] FP16 — output gate */
        qw6_tensor_t dn_norm;          /* [hidden] FP32 — key normalisation */

        /* MoE (all layers) */
        qw6_tensor_t moe_router;       /* [num_experts, hidden] */
        /* Expert weights: gate_proj, up_proj, down_proj per expert */
        /* Stored contiguously for cache-friendly loading */
        qw6_tensor_t expert_gate[QW6_NUM_EXPERTS];   /* [moe_inter, hidden] */
        qw6_tensor_t expert_up[QW6_NUM_EXPERTS];     /* [moe_inter, hidden] */
        qw6_tensor_t expert_down[QW6_NUM_EXPERTS];   /* [hidden, moe_inter] */
        qw6_tensor_t shared_gate;     /* [shared_inter, hidden] */
        qw6_tensor_t shared_up;       /* [shared_inter, hidden] */
        qw6_tensor_t shared_down;     /* [hidden, shared_inter] */
    } layers[QW6_NUM_LAYERS];

    /* MTP */
    struct {
        qw6_tensor_t norm;
        /* One simplified layer for draft */
        qw6_tensor_t embed;    /* [vocab, hidden] or shares with main */
        /* More tensors added when MTP architecture is finalised */
    } mtp;

    /* DeltaNet state (per linear-attn layer) */
    /* state[l] = [num_key_heads, key_head_dim, num_value_heads, value_head_dim] */
    /* FP16 to save memory */
    float *deltanet_state[QW6_NUM_LAYERS]; /* NULL for full-attn layers */

    /* KV cache (per full-attn layer) */
    /* k_cache[l] = [max_ctx, num_kv_heads, head_dim] */
    /* v_cache[l] = [max_ctx, num_kv_heads, head_dim] */
    float *k_cache[QW6_NUM_LAYERS]; /* NULL for linear-attn layers */
    float *v_cache[QW6_NUM_LAYERS]; /* NULL for linear-attn layers */

    /* Size info */
    uint32_t max_context;
    uint32_t current_context;   /* number of tokens in cache */
    size_t total_weight_bytes;
} qw6_model_t;

/* ---- Session ---- */

typedef struct {
    qw6_model_t *model;
    uint32_t *tokens;          /* token IDs */
    uint32_t n_tokens;
    uint32_t capacity;
    float *logits;             /* [vocab] — last token logits */
} qw6_session_t;

/* ---- API ---- */

/* Model loading / freeing */
int qw6_model_load(qw6_model_t *m, const char *gguf_path);
void qw6_model_free(qw6_model_t *m);

/* Session */
int qw6_session_init(qw6_session_t *s, qw6_model_t *m, uint32_t max_tokens);
void qw6_session_free(qw6_session_t *s);

/* Inference */
int qw6_prefill(qw6_session_t *s, const uint32_t *tokens, uint32_t n);
int qw6_generate(qw6_session_t *s, uint32_t n_tokens, float temp, float top_p);
uint32_t qw6_sample(qw6_session_t *s, float temp, float top_p);

/* Tokenizer */
int qw6_token_encode(const char *text, uint32_t **out_tokens, uint32_t *out_n);
int qw6_token_decode(const uint32_t *tokens, uint32_t n, char **out_text);

/* CPU kernels (Phase 1) */
void qw6_cpu_rmsnorm(float *out, const float *x, const float *weight, int dim);
void qw6_cpu_matmul_f16(float *out, const void *w, const float *x,
                        int rows, int cols, qw6_quant_t quant);
void qw6_cpu_matmul_q8(float *out, const void *w, const float *x,
                       int rows, int cols);
void qw6_cpu_matmul_q4km(float *out, const void *w, const float *x,
                         int rows, int cols);
void qw6_cpu_matmul_iq2m(float *out, const void *w, const float *x,
                         int rows, int cols);

/* Gated DeltaNet operations */
void qw6_cpu_conv1d_causal(float *out, const float *x, const void *conv_w,
                           int dim, int kernel_size);
void qw6_cpu_deltanet_update(float *state, const float *key,
                             const float *value, float *query,
                             int key_heads, int key_dim,
                             int val_heads, int val_dim);
void qw6_cpu_deltanet_retrieve(float *out, const float *state, const float *query,
                                int key_heads, int key_dim,
                                int val_heads, int val_dim);

/* MRoPE */
void qw6_cpu_mrope(float *q, float *k, int q_dim, int kv_dim,
                  int n_heads, int n_kv_heads,
                  uint32_t position, int rotary_dim);

/* MoE routing */
void qw6_cpu_moe_route(int *expert_indices, float *expert_weights,
                      const float *router_logits, int n_experts, int top_k);

/* SiLU */
static inline float qw6_silu(float x) {
    return x / (1.0f + expf(-x));
}

/* Softmax */
void qw6_cpu_softmax(float *x, int n);

/* Argmax */
int qw6_cpu_argmax(const float *x, int n);

/* ---- Utility ---- */

static inline qw6_layer_type_t qw6_layer_type(int layer_idx) {
    return (layer_idx % QW6_FULL_ATTN_INTERVAL == QW6_FULL_ATTN_INTERVAL - 1)
        ? QW6_LAYER_FULL_ATTN
        : QW6_LAYER_LINEAR_ATTN;
}

/* GGUF reader (minimal, qw6-specific) */
int qw6_gguf_read_file(const char *path, qw6_model_t *m);

/* Debug */
void qw6_dump_tokens(const uint32_t *tokens, uint32_t n);
void qw6_dump_logits(const float *logits, int n, int top_k);

/* Version */
#define QW6_VERSION "0.0.1-prealpha"
#define QW6_BUILD_PHASE 0  /* 0=research, 1=cpu-ref, 2=vulkan */

#endif /* QW6_H */