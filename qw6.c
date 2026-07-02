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
    } else {
        /* Phase 1 TODO: Q8_0, Q4_K_M, IQ2_M dequant kernels */
        (void)w; (void)x;
        memset(out, 0, (size_t)rows * sizeof(float));
    }
}

void qw6_cpu_matmul_q8(float *out, const void *w, const float *x,
                       int rows, int cols) {
    QW6_ASSERT_PTR(out); QW6_ASSERT_PTR(w); QW6_ASSERT_PTR(x);
    (void)w; (void)x; (void)rows; (void)cols;
    /* Phase 1 TODO */
}

void qw6_cpu_matmul_q4km(float *out, const void *w, const float *x,
                         int rows, int cols) {
    QW6_ASSERT_PTR(out); QW6_ASSERT_PTR(w); QW6_ASSERT_PTR(x);
    (void)w; (void)x; (void)rows; (void)cols;
    /* Phase 1 TODO */
}

void qw6_cpu_matmul_iq2m(float *out, const void *w, const float *x,
                         int rows, int cols) {
    QW6_ASSERT_PTR(out); QW6_ASSERT_PTR(w); QW6_ASSERT_PTR(x);
    (void)w; (void)x; (void)rows; (void)cols;
    /* Phase 1 TODO */
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
            if (k == 0) acc += wval * x[d];  /* simplified single-token */
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

    /* Phase 1 TODO: proper Gated DeltaNet delta-rule update.
     * Reference: Qwen3-Next / fla library. */
    (void)state; (void)key; (void)value; (void)query;
    (void)key_heads; (void)key_dim; (void)val_heads; (void)val_dim;
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
    (void)q; (void)k; (void)q_dim; (void)kv_dim;
    (void)n_heads; (void)n_kv_heads; (void)position; (void)rotary_dim;
    /* Phase 1 TODO: interleaved MRoPE with [11,11,10] sections */
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

/* ---- Tokenizer (Phase 1 stub) ---- */

int qw6_token_encode(const char *text, uint32_t **out_tokens, uint32_t *out_n) {
    QW6_ASSERT_PTR(text); QW6_ASSERT_PTR(out_tokens); QW6_ASSERT_PTR(out_n);
    (void)text;
    *out_tokens = NULL;
    *out_n = 0;
    return -1;
}

int qw6_token_decode(const uint32_t *tokens, uint32_t n, char **out_text) {
    QW6_ASSERT_PTR(tokens); QW6_ASSERT_PTR(out_text);
    (void)tokens; (void)n;
    *out_text = NULL;
    return -1;
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
        "  --bench         Run benchmark\n"
        "  -h, --help      This help\n\n"
        "Status: pre-alpha. CPU reference path under development.\n",
        QW6_VERSION, QW6_BUILD_PHASE, QW6_DEFAULT_CTX
    );
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt = NULL;
    int n_tokens = 256;
    int ctx = QW6_DEFAULT_CTX;
    float temp = 0.0f;
    bool nothink = false, dump_tokens = false, bench = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i+1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i+1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i+1 < argc) n_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ctx") == 0 && i+1 < argc) ctx = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temp") == 0 && i+1 < argc) temp = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--nothink") == 0) nothink = true;
        else if (strcmp(argv[i], "--dump-tokens") == 0) dump_tokens = true;
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

    if (!model_path) { fprintf(stderr, "qw6: -m <model.gguf> required\n"); return 1; }
    if (!prompt && !bench) { fprintf(stderr, "qw6: -p <prompt> required\n"); return 1; }

    fprintf(stderr, "qw6 %s — Phase %d (CPU reference)\n", QW6_VERSION, QW6_BUILD_PHASE);
    fprintf(stderr, "Model: Qwen 3.6-35B-A3B (35B total, 3B active, 256 experts)\n");
    fprintf(stderr, "Architecture: hybrid attention (30 Gated DeltaNet + 10 Gated Attention)\n\n");

    qw6_model_t model;
    if (qw6_model_load(&model, model_path) != 0) {
        fprintf(stderr, "qw6: model loading is Phase 1 TODO (GGUF parser)\n");
        return 1;
    }

    qw6_session_t session;
    if (qw6_session_init(&session, &model, (uint32_t)ctx) != 0) {
        fprintf(stderr, "qw6: session init failed\n");
        qw6_model_free(&model);
        return 1;
    }

    if (dump_tokens && prompt) {
        uint32_t *tokens = NULL;
        uint32_t n = 0;
        if (qw6_token_encode(prompt, &tokens, &n) == 0) {
            qw6_dump_tokens(tokens, n);
            free(tokens);
        } else {
            fprintf(stderr, "qw6: tokeniser not yet implemented\n");
        }
        qw6_session_free(&session);
        qw6_model_free(&model);
        return 0;
    }

    if (prompt) {
        fprintf(stderr, "qw6: prompt: \"%s\"\n", prompt);
        fprintf(stderr, "qw6: thinking mode: %s\n", nothink ? "disabled" : "enabled");
        fprintf(stderr, "qw6: ctx=%d max_tokens=%d temp=%.1f\n\n", ctx, n_tokens, temp);

        uint32_t *tokens = NULL;
        uint32_t n = 0;
        if (qw6_token_encode(prompt, &tokens, &n) != 0) {
            fprintf(stderr, "qw6: BPE tokeniser is Phase 1 TODO (vocab %d)\n", QW6_VOCAB_SIZE);
            qw6_session_free(&session);
            qw6_model_free(&model);
            return 1;
        }
        fprintf(stderr, "qw6: tokenised to %u tokens\n", n);

        if (qw6_prefill(&session, tokens, n) != 0) {
            fprintf(stderr, "qw6: prefill is Phase 1 TODO\n");
            free(tokens);
            qw6_session_free(&session);
            qw6_model_free(&model);
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
    return 0;
}