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

#define _POSIX_C_SOURCE 200809L

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
#define QW6_LINEAR_QKV_DIM      (QW6_NUM_KEY_HEADS * QW6_KEY_HEAD_DIM * 2 + \
                                 QW6_NUM_VALUE_HEADS * QW6_VALUE_HEAD_DIM)
#define QW6_LINEAR_VALUE_DIM    (QW6_NUM_VALUE_HEADS * QW6_VALUE_HEAD_DIM)

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
#define QW6_PARTIAL_ROTARY      0.25f
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
    QW6_Q_BF16,
    QW6_Q_Q8_0,
    QW6_Q_Q4_K_M,
    QW6_Q_Q4_K_S,
    QW6_Q_Q5_K,
    QW6_Q_Q6_K,
    QW6_Q_Q3_K_M,
    QW6_Q_IQ3_S,
    QW6_Q_IQ3_XXS,
    QW6_Q_IQ2_S,
    QW6_Q_IQ2_M,
    QW6_Q_IQ2_XXS,
} qw6_quant_t;

/* ---- Tensor ---- */

typedef struct {
    char name[128];
    uint32_t rows;
    uint32_t cols;
    uint32_t ne[4];
    uint32_t n_dims;
    qw6_quant_t quant;
    void *data;         /* quantised data ( 格式取决于 quant) */
    size_t data_size;   /* bytes */
    uint64_t file_offset;
#ifdef QW6_VULKAN
    size_t vk_offset;   /* byte offset in GPU weight buffer (0 = not uploaded) */
#endif
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
    qw6_tensor_t output_norm;      /* [hidden] */

    /* Per-layer */
    struct {
        qw6_layer_type_t type;

        /* RMSNorm */
        qw6_tensor_t norm;             /* [hidden] FP32 */
        qw6_tensor_t post_norm;        /* [hidden] FP32 */

        /* Full attention (type == FULL_ATTN only) */
        qw6_tensor_t attn_q;           /* [num_q*head_dim, hidden] */
        qw6_tensor_t attn_k;           /* [kv*head_dim, hidden] */
        qw6_tensor_t attn_v;           /* [kv*head_dim, hidden] */
        qw6_tensor_t attn_o;           /* [num_q*head_dim, hidden] */
        qw6_tensor_t attn_q_norm;
        qw6_tensor_t attn_k_norm;
        qw6_tensor_t attn_gate;        /* [hidden] FP16 — output gate */

        /* Linear attention (type == LINEAR_ATTN only) */
        qw6_tensor_t conv1d;           /* [hidden, 4] FP16 */
        qw6_tensor_t dn_key;           /* [key_heads*key_dim, hidden] */
        qw6_tensor_t dn_value;         /* [val_heads*val_dim, hidden] */
        qw6_tensor_t dn_query;         /* [val_heads*val_dim, hidden] */
        qw6_tensor_t dn_out;           /* [hidden, val_heads*val_dim] */
        qw6_tensor_t dn_gate;          /* [hidden] FP16 — output gate */
        qw6_tensor_t dn_norm;          /* [hidden] FP32 - key normalisation */
        qw6_tensor_t dn_alpha;
        qw6_tensor_t dn_beta;
        qw6_tensor_t dn_dt;
        qw6_tensor_t dn_a;

        /* MoE (all layers) */
        qw6_tensor_t moe_router;       /* [num_experts, hidden] */
        qw6_tensor_t shared_router;
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
    void *weight_map;
    size_t weight_map_size;
    int weight_fd;
} qw6_model_t;

/* ---- Session ---- */

typedef struct {
    qw6_model_t *model;
    uint32_t *tokens;          /* token IDs */
    uint32_t n_tokens;
    uint32_t capacity;
    float *conv_state[QW6_NUM_LAYERS];
    float *logits;             /* [vocab] — last token logits */
#ifdef QW6_VULKAN
    void *vk_pipe;             /* qw6_vk_pipe_t*, NULL = CPU */
#endif
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
/* CPU reference quant layouts used by self-tests and small fixtures:
 * Q8:  [rows float scales][rows*cols int8 weights].
 * Q4K: [rows float scales][ceil(rows*cols/2) packed int4 weights],
 *       low nibble first, signed range [-8,7] via nibble-8.
 * IQ2M: [rows float scales][ceil(rows*cols/4) packed uint2 weights],
 *       two-bit values map to {-1.5,-0.5,0.5,1.5}.
 */
void qw6_cpu_matmul_q8(float *out, const void *w, const float *x,
                       int rows, int cols);
void qw6_cpu_matmul_q4km(float *out, const void *w, const float *x,
                         int rows, int cols);
void qw6_cpu_matmul_iq2m(float *out, const void *w, const float *x,
                         int rows, int cols);
int qw6_tensor_matvec(float *out, const qw6_tensor_t *t, const float *x,
                      uint32_t max_rows);
int qw6_tensor_dequantize_row(const qw6_tensor_t *t, uint32_t row,
                              float *dst);

/* Gated DeltaNet operations */
void qw6_l2_norm_heads(float *x, int heads, int dim);
void qw6_gated_delta_net_single(float *out, float *state,
                                const float *q16, const float *k16,
                                const float *v, const float *gate,
                                const float *beta);
/* x is laid out as [kernel_size, dim] from newest sample to oldest sample. */
void qw6_cpu_conv1d_causal(float *out, const float *x, const void *conv_w,
                           int dim, int kernel_size);
void qw6_cpu_deltanet_update(float *state, const float *key,
                              const float *value, const float *query,
                              const float *beta,
                              int key_heads, int key_dim,
                              int val_heads, int val_dim);
void qw6_cpu_deltanet_retrieve(float *out, const float *state, const float *query,
                                int key_heads, int key_dim,
                                int val_heads, int val_dim);

/* MRoPE */
void qw6_cpu_mrope(float *q, float *k, int q_dim, int kv_dim,
                  int n_heads, int n_kv_heads,
                  uint32_t position, int rotary_dim);
void qw6_cpu_attention_gqa(float *out, const float *q,
                           const float *k_cache, const float *v_cache,
                           int seq_len, int n_q_heads, int n_kv_heads,
                           int head_dim);

/* MoE routing */
void qw6_cpu_moe_route(int *expert_indices, float *expert_weights,
                      const float *router_logits, int n_experts, int top_k);

/* MTP speculative decoding */
void qw6_cpu_mtp_draft(float *logits, const float *hidden,
                       const float *mtp_weight, int vocab_size, int hidden_size);

/* SiLU */
static inline float qw6_silu(float x) {
    return x / (1.0f + expf(-x));
}
static inline float qw6_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}
static inline float qw6_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
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

/* ---- GGUF reader ---- */

#define GGUF_MAGIC       0x46554747u  /* "GGUF" little-endian */
#define GGUF_VERSION     3u
#define GGUF_MAX_TENSORS 4096
#define GGUF_MAX_KV     512
#define GGUF_MAX_DIMS   4
#define GGUF_DEFAULT_ALIGNMENT 32u

/* GGUF value types (match gguf.h enum) */
typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_type_t;

/* GGML tensor data types (subset, match ggml.h) */
typedef enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS  = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S   = 19,
    GGML_TYPE_IQ4_NL  = 20,
    GGML_TYPE_IQ3_S   = 21,
    GGML_TYPE_IQ2_S   = 22,
    GGML_TYPE_IQ4_XS  = 23,
    GGML_TYPE_I8      = 24,
    GGML_TYPE_I16     = 25,
    GGML_TYPE_I32     = 26,
    GGML_TYPE_I64     = 27,
    GGML_TYPE_F64     = 28,
    GGML_TYPE_IQ1_M   = 29,
    GGML_TYPE_BF16    = 30,
    GGML_TYPE_Q4_0_4_4 = 31,
    GGML_TYPE_Q4_0_4_8 = 32,
    GGML_TYPE_Q4_0_8_8 = 33,
    GGML_TYPE_TQ1_0   = 34,
    GGML_TYPE_TQ2_0   = 35,
} ggml_type_t;

/* GGUF metadata KV pair */
typedef struct {
    char key[128];
    gguf_type_t type;
    union {
        uint8_t  u8;
        int8_t   i8;
        uint16_t u16;
        int16_t  i16;
        uint32_t u32;
        int32_t  i32;
        float    f32;
        bool     b;
        uint64_t u64;
        int64_t  i64;
        double   f64;
    } val;
    /* For STRING type: points into raw data buffer */
    char *str;       /* owned, null-terminated */
    uint64_t str_len;

    /* For ARRAY type */
    gguf_type_t arr_type;
    uint64_t arr_count;
    void *arr_data;  /* owned, raw elements */
    char **arr_strs; /* owned, for string arrays */
} gguf_kv_t;

/* GGUF tensor info */
typedef struct {
    char name[128];
    uint32_t n_dims;
    int64_t dims[GGUF_MAX_DIMS];
    ggml_type_t type;
    uint64_t offset;
} gguf_tensor_info_t;

/* GGUF context — result of parsing */
typedef struct {
    uint32_t version;
    uint64_t tensor_count;
    uint64_t kv_count;
    gguf_kv_t kv[GGUF_MAX_KV];
    uint32_t kv_parsed;
    gguf_tensor_info_t tensors[GGUF_MAX_TENSORS];
    uint32_t tensors_parsed;
    uint32_t alignment;
    uint64_t data_offset;
    uint64_t file_size;
} gguf_ctx_t;

/* API */
int qw6_gguf_read_file(const char *path, qw6_model_t *m);
int qw6_gguf_parse(const char *path, gguf_ctx_t *ctx);
void qw6_gguf_free(gguf_ctx_t *ctx);
const gguf_kv_t *qw6_gguf_find_kv(const gguf_ctx_t *ctx, const char *key);
ggml_type_t qw6_gguf_type_to_qw6(ggml_type_t ggml_type, qw6_quant_t *out_quant);

/* Debug */
void qw6_dump_tokens(const uint32_t *tokens, uint32_t n);
void qw6_dump_logits(const float *logits, int n, int top_k);
void qw6_dump_logprobs(const float *logits, int n, int top_k);

/* Version */
#define QW6_VERSION "0.0.1-prealpha"
#ifdef QW6_VULKAN
#define QW6_BUILD_PHASE 2  /* 0=research, 1=cpu-ref, 2=vulkan */
#else
#define QW6_BUILD_PHASE 1
#endif

#endif /* QW6_H */
