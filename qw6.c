/* qw6.c — Qwen 3.6-35B-A3B inference engine (CPU reference path)
 *
 * A self-contained C inference engine for Qwen 3.6-35B-A3B on AMD BC-250.
 * Inspired by antirez/ds4 (DwarfStar4).
 *
 * Phase 1: CPU reference path — correctness, not speed.
 * Phase 2: Vulkan compute backend (GFX1013 / RADV) — planned.
 *
 * Build:  make cpu       (CPU reference)
 *         make vulkan    (Vulkan backend — Phase 2)
 *
 * Usage:  ./qw6 -m model.gguf -p "Hello"
 *         ./qw6 -m model.gguf --dump-tokens -p "Hello"
 *         ./qw6 -m model.gguf -p "Hello" --nothink --ctx 4096 -n 256
 *
 * See ARCHITECTURE.md, MODEL_CARD.md, ROADMAP.md for design.
 */

#include "qw6.h"
#include "qw6_tok.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

/* ---- Assertions ---- */

#define QW6_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "qw6: ASSERT FAILED: %s:%d: %s\n", \
                __FILE__, __LINE__, (msg)); \
        exit(1); \
    } \
} while (0)

#define QW6_ASSERT_PTR(p) QW6_ASSERT((p) != NULL, "null pointer: " #p)

/* ---- GGUF reader (minimal) ---- */

#define GGUF_MAGIC 0x46554747  /* "GGUF" little-endian */

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
} gguf_header_t;

int qw6_gguf_read_file(const char *path, qw6_model_t *m) {
    QW6_ASSERT_PTR(path);
    QW6_ASSERT_PTR(m);

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "qw6: cannot open model file: %s\n", path);
        return -1;
    }

    gguf_header_t hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fprintf(stderr, "qw6: cannot read GGUF header\n");
        fclose(f);
        return -1;
    }

    if (hdr.magic != GGUF_MAGIC) {
        fprintf(stderr, "qw6: not a GGUF file (magic=0x%08x)\n", hdr.magic);
        fclose(f);
        return -1;
    }

    if (hdr.version != 3) {
        fprintf(stderr, "qw6: unsupported GGUF version %u\n", hdr.version);
        fclose(f);
        return -1;
    }

    fprintf(stderr, "qw6: GGUF v%u, %llu tensors, %llu metadata KV pairs\n",
            hdr.version,
            (unsigned long long)hdr.tensor_count,
            (unsigned long long)hdr.metadata_kv_count);

    /* Phase 1 TODO: parse metadata KV pairs and tensor info */
    fclose(f);
    return -1;
}

static int qw6_gguf_inspect_file(const char *path) {
    QW6_ASSERT_PTR(path);

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "qw6: cannot open GGUF file: %s\n", path);
        return -1;
    }

    gguf_header_t hdr;
    size_t nread = fread(&hdr, sizeof(hdr), 1, f);
    fclose(f);
    if (nread != 1) {
        fprintf(stderr, "qw6: cannot read complete GGUF header\n");
        return -1;
    }
    if (hdr.magic != GGUF_MAGIC) {
        fprintf(stderr, "qw6: not a GGUF file (magic=0x%08x)\n", hdr.magic);
        return -1;
    }
    printf("GGUF v%u\n", hdr.version);
    printf("tensors: %llu\n", (unsigned long long)hdr.tensor_count);
    printf("metadata_kv: %llu\n", (unsigned long long)hdr.metadata_kv_count);
    return hdr.version == 3 ? 0 : -1;
}

/* ---- Model lifecycle ---- */

int qw6_model_load(qw6_model_t *m, const char *gguf_path) {
    QW6_ASSERT_PTR(m);
    QW6_ASSERT_PTR(gguf_path);

    memset(m, 0, sizeof(*m));
    m->max_context = QW6_DEFAULT_CTX;
    return qw6_gguf_read_file(gguf_path, m);
}

void qw6_model_free(qw6_model_t *m) {
    QW6_ASSERT_PTR(m);

    if (m->tok_embeddings.data) free(m->tok_embeddings.data);
    if (m->output.data) free(m->output.data);

    for (int i = 0; i < QW6_NUM_LAYERS; i++) {
        if (m->layers[i].norm.data) free(m->layers[i].norm.data);
        if (m->layers[i].attn_q.data) free(m->layers[i].attn_q.data);
        if (m->layers[i].attn_k.data) free(m->layers[i].attn_k.data);
        if (m->layers[i].attn_v.data) free(m->layers[i].attn_v.data);
        if (m->layers[i].attn_o.data) free(m->layers[i].attn_o.data);
        if (m->layers[i].attn_gate.data) free(m->layers[i].attn_gate.data);
        if (m->layers[i].conv1d.data) free(m->layers[i].conv1d.data);
        if (m->layers[i].dn_key.data) free(m->layers[i].dn_key.data);
        if (m->layers[i].dn_value.data) free(m->layers[i].dn_value.data);
        if (m->layers[i].dn_query.data) free(m->layers[i].dn_query.data);
        if (m->layers[i].dn_out.data) free(m->layers[i].dn_out.data);
        if (m->layers[i].dn_gate.data) free(m->layers[i].dn_gate.data);
        if (m->layers[i].dn_norm.data) free(m->layers[i].dn_norm.data);
        if (m->layers[i].moe_router.data) free(m->layers[i].moe_router.data);
        for (int j = 0; j < QW6_NUM_EXPERTS; j++) {
            if (m->layers[i].expert_gate[j].data) free(m->layers[i].expert_gate[j].data);
            if (m->layers[i].expert_up[j].data) free(m->layers[i].expert_up[j].data);
            if (m->layers[i].expert_down[j].data) free(m->layers[i].expert_down[j].data);
        }
        if (m->layers[i].shared_gate.data) free(m->layers[i].shared_gate.data);
        if (m->layers[i].shared_up.data) free(m->layers[i].shared_up.data);
        if (m->layers[i].shared_down.data) free(m->layers[i].shared_down.data);
        if (m->deltanet_state[i]) free(m->deltanet_state[i]);
        if (m->k_cache[i]) free(m->k_cache[i]);
        if (m->v_cache[i]) free(m->v_cache[i]);
    }

    if (m->mtp.norm.data) free(m->mtp.norm.data);
    if (m->mtp.embed.data) free(m->mtp.embed.data);
    memset(m, 0, sizeof(*m));
}

/* ---- Session ---- */

int qw6_session_init(qw6_session_t *s, qw6_model_t *m, uint32_t max_tokens) {
    QW6_ASSERT_PTR(s);
    QW6_ASSERT_PTR(m);

    memset(s, 0, sizeof(*s));
    s->model = m;
    s->capacity = max_tokens;
    s->tokens = calloc(max_tokens, sizeof(uint32_t));
    QW6_ASSERT_PTR(s->tokens);
    s->logits = calloc(QW6_VOCAB_SIZE, sizeof(float));
    QW6_ASSERT_PTR(s->logits);

    for (int i = 0; i < QW6_NUM_LAYERS; i++) {
        if (qw6_layer_type(i) == QW6_LAYER_LINEAR_ATTN) {
            size_t sz = (size_t)QW6_NUM_KEY_HEADS * QW6_KEY_HEAD_DIM *
                        QW6_NUM_VALUE_HEADS * QW6_VALUE_HEAD_DIM;
            m->deltanet_state[i] = calloc(sz, sizeof(float));
            QW6_ASSERT_PTR(m->deltanet_state[i]);
        } else {
            size_t kv_sz = (size_t)m->max_context * QW6_NUM_KV_HEADS * QW6_HEAD_DIM;
            m->k_cache[i] = calloc(kv_sz, sizeof(float));
            m->v_cache[i] = calloc(kv_sz, sizeof(float));
            QW6_ASSERT_PTR(m->k_cache[i]);
            QW6_ASSERT_PTR(m->v_cache[i]);
        }
    }
    return 0;
}

void qw6_session_free(qw6_session_t *s) {
    QW6_ASSERT_PTR(s);
    if (s->tokens) free(s->tokens);
    if (s->logits) free(s->logits);
    memset(s, 0, sizeof(*s));
}

/* ---- CPU Kernels (Phase 1: correctness, not speed) ---- */

void qw6_cpu_rmsnorm(float *out, const float *x, const float *weight, int dim) {
    QW6_ASSERT_PTR(out);
    QW6_ASSERT_PTR(x);
    QW6_ASSERT_PTR(weight);
    QW6_ASSERT(dim > 0, "dim > 0");

    float ss = 0.0f;
    for (int i = 0; i < dim; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / dim + QW6_RMS_EPS);
    for (int i = 0; i < dim; i++) out[i] = x[i] * ss * weight[i];
}

void qw6_cpu_matmul_f16(float *out, const void *w, const float *x,
                        int rows, int cols, qw6_quant_t quant) {
    QW6_ASSERT_PTR(out);
    QW6_ASSERT_PTR(w);
    QW6_ASSERT_PTR(x);
    QW6_ASSERT(rows > 0 && cols > 0, "rows>0 && cols>0");

    if (quant == QW6_Q_FP32) {
        const float *w32 = (const float *)w;
        for (int r = 0; r < rows; r++) {
            float acc = 0.0f;
            for (int c = 0; c < cols; c++) acc += w32[r * cols + c] * x[c];
            out[r] = acc;
        }
    } else if (quant == QW6_Q_FP16) {
        const uint16_t *w16 = (const uint16_t *)w;
        for (int r = 0; r < rows; r++) {
            float acc = 0.0f;
            for (int c = 0; c < cols; c++) {
                uint16_t h = w16[r * cols + c];
                uint32_t sign = (h >> 15) & 1;
                uint32_t exp = (h >> 10) & 0x1f;
                uint32_t mant = h & 0x3ff;
                float val;
                if (exp == 0) {
                    val = (sign ? -1.0f : 1.0f) * (mant / 1048576.0f);
                } else {
                    uint32_t f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                    memcpy(&val, &f, sizeof(float));
                }
                acc += val * x[c];
            }
            out[r] = acc;
        }
    } else if (quant == QW6_Q_Q8_0) {
        qw6_cpu_matmul_q8(out, w, x, rows, cols);
    } else if (quant == QW6_Q_Q4_K_M || quant == QW6_Q_Q4_K_S) {
        qw6_cpu_matmul_q4km(out, w, x, rows, cols);
    } else if (quant == QW6_Q_IQ2_M || quant == QW6_Q_IQ2_XXS) {
        qw6_cpu_matmul_iq2m(out, w, x, rows, cols);
    } else {
        memset(out, 0, (size_t)rows * sizeof(float));
    }
}

void qw6_cpu_matmul_q8(float *out, const void *w, const float *x,
                       int rows, int cols) {
    QW6_ASSERT_PTR(out); QW6_ASSERT_PTR(w); QW6_ASSERT_PTR(x);
    QW6_ASSERT(rows > 0 && cols > 0, "rows>0 && cols>0");

    /* CPU reference layout: [rows float scales][rows*cols int8 weights].
     * This is intentionally simple and not a GGML block layout. */
    const float *scales = (const float *)w;
    const int8_t *q = (const int8_t *)(scales + rows);
    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        for (int c = 0; c < cols; c++) {
            acc += ((float)q[r * cols + c] * scales[r]) * x[c];
        }
        out[r] = acc;
    }
}

void qw6_cpu_matmul_q4km(float *out, const void *w, const float *x,
                         int rows, int cols) {
    QW6_ASSERT_PTR(out); QW6_ASSERT_PTR(w); QW6_ASSERT_PTR(x);
    QW6_ASSERT(rows > 0 && cols > 0, "rows>0 && cols>0");

    const float *scales = (const float *)w;
    const uint8_t *packed = (const uint8_t *)(scales + rows);
    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        for (int c = 0; c < cols; c++) {
            size_t idx = (size_t)r * (size_t)cols + (size_t)c;
            uint8_t byte = packed[idx >> 1];
            int q = (idx & 1) ? (byte >> 4) : (byte & 0x0f);
            q -= 8;
            acc += ((float)q * scales[r]) * x[c];
        }
        out[r] = acc;
    }
}

void qw6_cpu_matmul_iq2m(float *out, const void *w, const float *x,
                         int rows, int cols) {
    QW6_ASSERT_PTR(out); QW6_ASSERT_PTR(w); QW6_ASSERT_PTR(x);
    QW6_ASSERT(rows > 0 && cols > 0, "rows>0 && cols>0");

    const float *scales = (const float *)w;
    const uint8_t *packed = (const uint8_t *)(scales + rows);
    static const float lut[4] = {-1.5f, -0.5f, 0.5f, 1.5f};
    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        for (int c = 0; c < cols; c++) {
            size_t idx = (size_t)r * (size_t)cols + (size_t)c;
            uint8_t code = (packed[idx >> 2] >> ((idx & 3) * 2)) & 0x03;
            acc += (lut[code] * scales[r]) * x[c];
        }
        out[r] = acc;
    }
}

/* ---- Gated DeltaNet ---- */

void qw6_cpu_conv1d_causal(float *out, const float *x, const void *conv_w,
                           int dim, int kernel_size) {
    QW6_ASSERT_PTR(out); QW6_ASSERT_PTR(x); QW6_ASSERT_PTR(conv_w);
    QW6_ASSERT(dim > 0 && kernel_size > 0, "dim>0 && ks>0");

    const uint16_t *w16 = (const uint16_t *)conv_w;
    for (int d = 0; d < dim; d++) {
        float acc = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            uint16_t h = w16[d * kernel_size + k];
            uint32_t sign = (h >> 15) & 1;
            uint32_t exp = (h >> 10) & 0x1f;
            uint32_t mant = h & 0x3ff;
            float wval;
            if (exp == 0) {
                wval = (sign ? -1.0f : 1.0f) * (mant / 1048576.0f);
            } else {
                uint32_t f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
                memcpy(&wval, &f, sizeof(float));
            }
            acc += wval * x[k * dim + d];
        }
        out[d] = acc;
    }
}

void qw6_cpu_deltanet_update(float *state, const float *key,
                             const float *value, float *query,
                             int key_heads, int key_dim,
                             int val_heads, int val_dim) {
    QW6_ASSERT_PTR(state); QW6_ASSERT_PTR(key);
    QW6_ASSERT_PTR(value); QW6_ASSERT_PTR(query);
    QW6_ASSERT(key_heads > 0 && key_dim > 0, "key dims > 0");
    QW6_ASSERT(val_heads > 0 && val_dim > 0, "value dims > 0");

    /* Minimal CPU reference: additive key/value outer-product state.
     * The production DeltaNet rule will add gating/normalisation later. */
    for (int kh = 0; kh < key_heads; kh++) {
        const float *k = key + kh * key_dim;
        for (int vh = 0; vh < val_heads; vh++) {
            const float *v = value + vh * val_dim;
            float *s = state + (size_t)(kh * key_dim * val_heads + vh) * val_dim;
            for (int kd = 0; kd < key_dim; kd++)
                for (int vd = 0; vd < val_dim; vd++)
                    s[kd * val_dim + vd] += k[kd] * v[vd];
        }
    }
    (void)query;
}

void qw6_cpu_deltanet_retrieve(float *out, const float *state, const float *query,
                                int key_heads, int key_dim,
                                int val_heads, int val_dim) {
    QW6_ASSERT_PTR(out); QW6_ASSERT_PTR(state); QW6_ASSERT_PTR(query);

    memset(out, 0, (size_t)val_heads * val_dim * sizeof(float));
    for (int kh = 0; kh < key_heads; kh++) {
        const float *q = query + kh * key_dim;
        for (int vh = 0; vh < val_heads; vh++) {
            const float *s = state + (size_t)(kh * key_dim * val_heads + vh) * val_dim;
            float *o = out + vh * val_dim;
            for (int kd = 0; kd < key_dim; kd++) {
                float qk = q[kd];
                for (int vd = 0; vd < val_dim; vd++) {
                    o[vd] += s[kd * val_dim + vd] * qk;
                }
            }
        }
    }
}

/* ---- MRoPE ---- */

void qw6_cpu_mrope(float *q, float *k, int q_dim, int kv_dim,
                  int n_heads, int n_kv_heads,
                  uint32_t position, int rotary_dim) {
    QW6_ASSERT_PTR(q); QW6_ASSERT_PTR(k);
    QW6_ASSERT(q_dim > 0 && kv_dim > 0, "dims > 0");
    QW6_ASSERT(n_heads > 0 && n_kv_heads > 0, "heads > 0");
    QW6_ASSERT(q_dim % n_heads == 0, "q dim divisible by heads");
    QW6_ASSERT(kv_dim % n_kv_heads == 0, "kv dim divisible by heads");

    int q_head_dim = q_dim / n_heads;
    int k_head_dim = kv_dim / n_kv_heads;
    int q_rot = rotary_dim < q_head_dim ? rotary_dim : q_head_dim;
    int k_rot = rotary_dim < k_head_dim ? rotary_dim : k_head_dim;
    QW6_ASSERT(q_rot >= 0 && k_rot >= 0, "rotary dims >= 0");

    for (int h = 0; h < n_heads; h++) {
        float *qh = q + h * q_head_dim;
        for (int i = 0; i + 1 < q_rot; i += 2) {
            float theta = (float)position / powf(QW6_ROPE_THETA, (float)i / q_rot);
            float c = cosf(theta), s = sinf(theta);
            float a = qh[i], b = qh[i + 1];
            qh[i] = a * c - b * s;
            qh[i + 1] = a * s + b * c;
        }
    }
    for (int h = 0; h < n_kv_heads; h++) {
        float *kh = k + h * k_head_dim;
        for (int i = 0; i + 1 < k_rot; i += 2) {
            float theta = (float)position / powf(QW6_ROPE_THETA, (float)i / k_rot);
            float c = cosf(theta), s = sinf(theta);
            float a = kh[i], b = kh[i + 1];
            kh[i] = a * c - b * s;
            kh[i + 1] = a * s + b * c;
        }
    }
}

/* ---- MoE Routing ---- */

void qw6_cpu_moe_route(int *expert_indices, float *expert_weights,
                      const float *router_logits, int n_experts, int top_k) {
    QW6_ASSERT_PTR(expert_indices);
    QW6_ASSERT_PTR(expert_weights);
    QW6_ASSERT_PTR(router_logits);
    QW6_ASSERT(n_experts > 0 && top_k > 0 && top_k <= n_experts, "moe bounds");
    QW6_ASSERT(n_experts <= 256, "n_experts<=256");

    float logits[256];
    memcpy(logits, router_logits, (size_t)n_experts * sizeof(float));

    for (int k = 0; k < top_k; k++) {
        int best = 0;
        float best_val = logits[0];
        for (int i = 1; i < n_experts; i++) {
            if (logits[i] > best_val) { best_val = logits[i]; best = i; }
        }
        expert_indices[k] = best;
        expert_weights[k] = best_val;
        logits[best] = -INFINITY;
    }

    float max_w = expert_weights[0];
    for (int k = 1; k < top_k; k++)
        if (expert_weights[k] > max_w) max_w = expert_weights[k];
    float sum = 0.0f;
    for (int k = 0; k < top_k; k++) {
        expert_weights[k] = expf(expert_weights[k] - max_w);
        sum += expert_weights[k];
    }
    for (int k = 0; k < top_k; k++) expert_weights[k] /= sum;
}

/* ---- Softmax / Argmax ---- */

void qw6_cpu_softmax(float *x, int n) {
    QW6_ASSERT_PTR(x); QW6_ASSERT(n > 0, "n > 0");
    float max = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

int qw6_cpu_argmax(const float *x, int n) {
    QW6_ASSERT_PTR(x); QW6_ASSERT(n > 0, "n > 0");
    int best = 0;
    for (int i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    return best;
}

/* ---- Inference (Phase 1 stub) ---- */

int qw6_prefill(qw6_session_t *s, const uint32_t *tokens, uint32_t n) {
    QW6_ASSERT_PTR(s); QW6_ASSERT_PTR(tokens); QW6_ASSERT(n > 0, "n > 0");
    (void)tokens; (void)n;
    fprintf(stderr, "qw6: prefill not yet implemented (Phase 1)\n");
    return -1;
}

int qw6_generate(qw6_session_t *s, uint32_t n_tokens, float temp, float top_p) {
    QW6_ASSERT_PTR(s); QW6_ASSERT(n_tokens > 0, "n_tokens > 0");
    (void)n_tokens; (void)temp; (void)top_p;
    fprintf(stderr, "qw6: generate not yet implemented (Phase 1)\n");
    return -1;
}

uint32_t qw6_sample(qw6_session_t *s, float temp, float top_p) {
    QW6_ASSERT_PTR(s);
    (void)top_p;
    if (temp <= 0.0f) return qw6_cpu_argmax(s->logits, QW6_VOCAB_SIZE);
    return qw6_cpu_argmax(s->logits, QW6_VOCAB_SIZE);
}

/* ---- Tokenizer convenience API ---- */

int qw6_token_encode(const char *text, uint32_t **out_tokens, uint32_t *out_n) {
    QW6_ASSERT_PTR(text); QW6_ASSERT_PTR(out_tokens); QW6_ASSERT_PTR(out_n);

    qw6_tokenizer_t tokenizer;
    if (qw6_tok_init(&tokenizer, "tokenizer/tokenizer.json") != 0) return -1;
    int rc = qw6_tok_encode(&tokenizer, text, out_tokens, out_n);
    qw6_tok_free(&tokenizer);
    return rc;
}

int qw6_token_decode(const uint32_t *tokens, uint32_t n, char **out_text) {
    QW6_ASSERT_PTR(tokens); QW6_ASSERT_PTR(out_text);

    qw6_tokenizer_t tokenizer;
    if (qw6_tok_init(&tokenizer, "tokenizer/tokenizer.json") != 0) return -1;
    int rc = qw6_tok_decode(&tokenizer, tokens, n, out_text);
    qw6_tok_free(&tokenizer);
    return rc;
}

/* ---- Debug ---- */

void qw6_dump_tokens(const uint32_t *tokens, uint32_t n) {
    QW6_ASSERT_PTR(tokens);
    printf("[");
    for (uint32_t i = 0; i < n && i < 20; i++) {
        printf("%u", tokens[i]);
        if (i < n - 1 && i < 19) printf(", ");
    }
    if (n > 20) printf(", ...");
    printf("] (%u tokens)\n", n);
}

void qw6_dump_logits(const float *logits, int n, int top_k) {
    QW6_ASSERT_PTR(logits);
    int *idx = malloc((size_t)top_k * sizeof(int));
    float *val = malloc((size_t)top_k * sizeof(float));
    QW6_ASSERT_PTR(idx); QW6_ASSERT_PTR(val);

    for (int k = 0; k < top_k; k++) { idx[k] = -1; val[k] = -INFINITY; }
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < top_k; k++) {
            if (logits[i] > val[k]) {
                for (int j = top_k - 1; j > k; j--) { val[j] = val[j-1]; idx[j] = idx[j-1]; }
                val[k] = logits[i]; idx[k] = i; break;
            }
        }
    }
    printf("Top-%d logits:\n", top_k);
    for (int k = 0; k < top_k; k++) printf("  token %d: %.6f\n", idx[k], val[k]);
    free(idx); free(val);
}

/* ---- Self-test (no model/tokenizer/BC-250 required) ---- */

static int selftest_close(float got, float want, float eps, const char *name) {
    if (fabsf(got - want) <= eps) return 0;
    fprintf(stderr, "self-test: %s got %.7f want %.7f\n", name, got, want);
    return 1;
}

static int selftest_rmsnorm(void) {
    const float x[4] = {1.0f, -2.0f, 3.0f, -4.0f};
    const float w[4] = {1.0f, 0.5f, 2.0f, -1.0f};
    float out[4] = {0};
    qw6_cpu_rmsnorm(out, x, w, 4);

    const float scale = 1.0f / sqrtf(7.5f + QW6_RMS_EPS);
    int fail = 0;
    fail += selftest_close(out[0], x[0] * scale * w[0], 1e-6f, "rmsnorm[0]");
    fail += selftest_close(out[1], x[1] * scale * w[1], 1e-6f, "rmsnorm[1]");
    fail += selftest_close(out[2], x[2] * scale * w[2], 1e-6f, "rmsnorm[2]");
    fail += selftest_close(out[3], x[3] * scale * w[3], 1e-6f, "rmsnorm[3]");
    return fail;
}

static int selftest_matmul(void) {
    const float w32[6] = {1.0f, 2.0f, 3.0f, -1.0f, 0.5f, 4.0f};
    const float x[3] = {2.0f, -1.0f, 0.5f};
    float out[2] = {0};
    qw6_cpu_matmul_f16(out, w32, x, 2, 3, QW6_Q_FP32);

    int fail = 0;
    fail += selftest_close(out[0], 1.5f, 1e-6f, "matmul_fp32[0]");
    fail += selftest_close(out[1], -0.5f, 1e-6f, "matmul_fp32[1]");

    const uint16_t w16[4] = {0x3c00, 0x4000, 0xbc00, 0x3800};
    const float x16[2] = {3.0f, 2.0f};
    qw6_cpu_matmul_f16(out, w16, x16, 2, 2, QW6_Q_FP16);
    fail += selftest_close(out[0], 7.0f, 1e-6f, "matmul_fp16[0]");
    fail += selftest_close(out[1], -2.0f, 1e-6f, "matmul_fp16[1]");

    struct {
        float scales[2];
        int8_t q[6];
    } q8 = {{0.5f, 0.25f}, {2, 4, 6, -4, 2, 8}};
    qw6_cpu_matmul_q8(out, &q8, x, 2, 3);
    fail += selftest_close(out[0], 1.5f, 1e-6f, "matmul_q8[0]");
    fail += selftest_close(out[1], -1.5f, 1e-6f, "matmul_q8[1]");

    struct {
        float scales[2];
        uint8_t q[3];
    } q4 = {{0.5f, 0.25f}, {0xca, 0xf7, 0x4a}};
    qw6_cpu_matmul_q4km(out, &q4, x, 2, 3);
    fail += selftest_close(out[0], -0.25f, 1e-6f, "matmul_q4[0]");
    fail += selftest_close(out[1], 2.5f, 1e-6f, "matmul_q4[1]");

    struct {
        float scales[2];
        uint8_t q[2];
    } iq2 = {{1.0f, 0.5f}, {0xd8, 0x06}};
    qw6_cpu_matmul_iq2m(out, &iq2, x, 2, 3);
    fail += selftest_close(out[0], -3.75f, 1e-6f, "matmul_iq2[0]");
    fail += selftest_close(out[1], 1.125f, 1e-6f, "matmul_iq2[1]");
    return fail;
}

static int selftest_conv1d(void) {
    const float x[6] = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.5f};
    const uint16_t w[6] = {0x3c00, 0x4000, 0x3800, 0x3c00, 0x3c00, 0xbc00};
    float out[2] = {0};
    qw6_cpu_conv1d_causal(out, x, w, 2, 3);

    int fail = 0;
    fail += selftest_close(out[0], 6.5f, 1e-6f, "conv1d[0]");
    fail += selftest_close(out[1], 5.5f, 1e-6f, "conv1d[1]");
    return fail;
}

static int selftest_deltanet(void) {
    float state[4] = {0};
    const float key[2] = {2.0f, -1.0f};
    const float value[2] = {0.5f, 3.0f};
    float query[2] = {1.0f, 2.0f};
    float out[2] = {0};

    qw6_cpu_deltanet_update(state, key, value, query, 1, 2, 1, 2);
    qw6_cpu_deltanet_retrieve(out, state, query, 1, 2, 1, 2);

    int fail = 0;
    fail += selftest_close(out[0], 0.0f, 1e-6f, "deltanet[0]");
    fail += selftest_close(out[1], 0.0f, 1e-6f, "deltanet[1]");
    return fail;
}

static int selftest_mrope(void) {
    float q[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float k[2] = {1.0f, 0.0f};
    qw6_cpu_mrope(q, k, 4, 2, 2, 1, 0, 2);

    int fail = 0;
    fail += selftest_close(q[0], 1.0f, 1e-6f, "mrope_pos0_q0");
    fail += selftest_close(q[3], 1.0f, 1e-6f, "mrope_pos0_q3");
    fail += selftest_close(k[0], 1.0f, 1e-6f, "mrope_pos0_k0");

    qw6_cpu_mrope(q, k, 4, 2, 2, 1, 1, 2);
    fail += selftest_close(q[0], cosf(1.0f), 1e-6f, "mrope_pos1_q0");
    fail += selftest_close(q[1], sinf(1.0f), 1e-6f, "mrope_pos1_q1");
    return fail;
}

static int selftest_softmax_argmax(void) {
    float x[3] = {1.0f, 2.0f, 3.0f};
    qw6_cpu_softmax(x, 3);
    int fail = 0;
    fail += selftest_close(x[0] + x[1] + x[2], 1.0f, 1e-6f, "softmax_sum");
    fail += (qw6_cpu_argmax(x, 3) == 2) ? 0 : 1;
    if (fail) fprintf(stderr, "self-test: softmax/argmax failed\n");
    return fail;
}

static int selftest_moe_route(void) {
    const float logits[4] = {0.0f, 2.0f, 1.0f, -1.0f};
    int idx[2] = {-1, -1};
    float weights[2] = {0};
    qw6_cpu_moe_route(idx, weights, logits, 4, 2);

    int fail = 0;
    if (idx[0] != 1 || idx[1] != 2) fail = 1;
    fail += selftest_close(weights[0] + weights[1], 1.0f, 1e-6f, "moe_weight_sum");
    if (weights[0] <= weights[1]) fail = 1;
    if (fail) fprintf(stderr, "self-test: moe route failed\n");
    return fail;
}

static int qw6_selftest(void) {
    int fail = 0;
    fail += selftest_rmsnorm();
    fail += selftest_matmul();
    fail += selftest_conv1d();
    fail += selftest_deltanet();
    fail += selftest_mrope();
    fail += selftest_softmax_argmax();
    fail += selftest_moe_route();
    if (fail) {
        fprintf(stderr, "qw6: self-test failed (%d checks)\n", fail);
        return 1;
    }
    fprintf(stderr, "qw6: self-test passed\n");
    return 0;
}

/* ---- CLI ---- */

static void usage(void) {
    fprintf(stderr,
        "qw6 %s (Phase %d) — Qwen 3.6-35B-A3B inference engine\n\n"
        "Usage: qw6 -m <model.gguf> [options]\n\n"
        "Options:\n"
        "  -m <path>       Model GGUF file (required)\n"
        "  -p <prompt>     Prompt text\n"
        "  -n <N>          Max tokens to generate (default: 256)\n"
        "  --ctx <N>       Context window (default: %d)\n"
        "  --temp <F>      Temperature (default: 0.0 = greedy)\n"
        "  --nothink       Disable thinking mode\n"
        "  --cpu           CPU backend (default)\n"
        "  --vulkan        Vulkan backend (Phase 2)\n"
        "  --dump-tokens   Tokenise prompt and exit\n"
        "  --inspect-gguf  Inspect GGUF header and exit\n"
        "  --self-test     Run CPU kernel self-tests and exit\n"
        "  --bench         Run benchmark\n"
        "  -h, --help      This help\n\n"
        "Status: pre-alpha. CPU reference path under development.\n",
        QW6_VERSION, QW6_BUILD_PHASE, QW6_DEFAULT_CTX
    );
}
int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *inspect_path = NULL;
    const char *prompt = NULL;
    const char *tok_path = "tokenizer/tokenizer.json";
    int n_tokens = 256;
    int ctx = QW6_DEFAULT_CTX;
    float temp = 0.0f;
    bool nothink = false, dump_tokens = false, bench = false, self_test = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i+1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i+1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "--inspect-gguf") == 0 && i+1 < argc) inspect_path = argv[++i];
        else if (strcmp(argv[i], "--tok") == 0 && i+1 < argc) tok_path = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) n_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ctx") == 0 && i+1 < argc) ctx = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temp") == 0 && i+1 < argc) temp = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--nothink") == 0) nothink = true;
        else if (strcmp(argv[i], "--dump-tokens") == 0) dump_tokens = true;
        else if (strcmp(argv[i], "--self-test") == 0) self_test = true;
        else if (strcmp(argv[i], "--bench") == 0) bench = true;
        else if (strcmp(argv[i], "--cpu") == 0) { /* default */ }
        else if (strcmp(argv[i], "--vulkan") == 0) {
            fprintf(stderr, "qw6: Vulkan backend not yet available (Phase 2)\n");
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(); return 0;
        } else {
            fprintf(stderr, "qw6: unknown option: %s\n", argv[i]);
            usage(); return 1;
        }
    }

    if (self_test) return qw6_selftest();
    if (inspect_path) return qw6_gguf_inspect_file(inspect_path);

    if (!prompt && !bench) { fprintf(stderr, "qw6: -p <prompt> required\n"); return 1; }

    fprintf(stderr, "qw6 %s -- Phase %d (CPU reference)\n", QW6_VERSION, QW6_BUILD_PHASE);
    fprintf(stderr, "Model: Qwen 3.6-35B-A3B (35B total, 3B active, 256 experts)\n");
    fprintf(stderr, "Architecture: hybrid attention (30 Gated DeltaNet + 10 Gated Attention)\n\n");

    /* Load tokenizer */
    qw6_tokenizer_t tokenizer;
    fprintf(stderr, "qw6: loading tokenizer from %s ...\n", tok_path);
    if (qw6_tok_init(&tokenizer, tok_path) != 0) {
        fprintf(stderr, "qw6: failed to load tokenizer\n");
        return 1;
    }
    fprintf(stderr, "qw6: tokenizer loaded (%u vocab, %u merges, %u added)\n\n",
            tokenizer.vocab_count, tokenizer.merge_count, tokenizer.added_count);

    if (dump_tokens && prompt) {
        fprintf(stderr, "qw6: encoding: \"%s\"\n\n", prompt);
        uint32_t *tokens = NULL;
        uint32_t n = 0;
        if (qw6_tok_encode(&tokenizer, prompt, &tokens, &n) == 0) {
            qw6_dump_tokens(tokens, n);
            /* Also show raw byte-level strings for debugging */
            fprintf(stderr, "\nqw6: raw tokens: \"");
            for (uint32_t i = 0; i < n; i++) {
                fprintf(stderr, "%s", qw6_tok_id_to_str(&tokenizer, tokens[i]));
            }
            fprintf(stderr, "\"\n");
            /* Now decode properly with byte-level reversal */
            char *decoded = NULL;
            if (qw6_tok_decode(&tokenizer, tokens, n, &decoded) == 0) {
                fprintf(stderr, "qw6: decoded: \"%s\"\n", decoded);
                free(decoded);
            }
            free(tokens);
        } else {
            fprintf(stderr, "qw6: encoding failed\n");
        }
        qw6_tok_free(&tokenizer);
        return 0;
    }

    /* Model loading (Phase 1 TODO) */
    if (!model_path) {
        fprintf(stderr, "qw6: -m <model.gguf> required for inference (not for --dump-tokens)\n");
        qw6_tok_free(&tokenizer);
        return 1;
    }

    qw6_model_t model;
    if (qw6_model_load(&model, model_path) != 0) {
        fprintf(stderr, "qw6: model loading is Phase 1 TODO (GGUF parser)\n");
        qw6_tok_free(&tokenizer);
        return 1;
    }

    qw6_session_t session;
    if (qw6_session_init(&session, &model, (uint32_t)ctx) != 0) {
        fprintf(stderr, "qw6: session init failed\n");
        qw6_model_free(&model);
        qw6_tok_free(&tokenizer);
        return 1;
    }

    if (prompt) {
        fprintf(stderr, "qw6: prompt: \"%s\"\n", prompt);
        fprintf(stderr, "qw6: thinking mode: %s\n", nothink ? "disabled" : "enabled");
        fprintf(stderr, "qw6: ctx=%d max_tokens=%d temp=%.1f\n\n", ctx, n_tokens, temp);

        uint32_t *tokens = NULL;
        uint32_t n = 0;
        if (qw6_tok_encode(&tokenizer, prompt, &tokens, &n) != 0) {
            fprintf(stderr, "qw6: tokeniser encoding failed\n");
            qw6_session_free(&session);
            qw6_model_free(&model);
            qw6_tok_free(&tokenizer);
            return 1;
        }
        fprintf(stderr, "qw6: tokenised to %u tokens\n", n);

        if (qw6_prefill(&session, tokens, n) != 0) {
            fprintf(stderr, "qw6: prefill is Phase 1 TODO\n");
            free(tokens);
            qw6_session_free(&session);
            qw6_model_free(&model);
            qw6_tok_free(&tokenizer);
            return 1;
        }

        fprintf(stderr, "qw6: generating %d tokens...\n", n_tokens);
        clock_t start = clock();
        int gen = qw6_generate(&session, (uint32_t)n_tokens, temp, 0.9f);
        double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
        fprintf(stderr, "qw6: %d tokens in %.2fs\n", gen, elapsed);
        free(tokens);
    }

    if (bench) fprintf(stderr, "qw6: benchmarks not yet implemented\n");

    qw6_session_free(&session);
    qw6_model_free(&model);
    qw6_tok_free(&tokenizer);
    return 0;
}
