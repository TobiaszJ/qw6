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
#include "qw6_iq_tables.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

/* ---- Assertions ---- */

#define QW6_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "qw6: ASSERT FAILED: %s:%d: %s\n", \
                __FILE__, __LINE__, (msg)); \
        exit(1); \
    } \
} while (0)

#define QW6_ASSERT_PTR(p) QW6_ASSERT((p) != NULL, "null pointer: " #p)

#ifndef QW6_QK_K
#define QW6_QK_K 256
#endif

/* ---- GGUF reader ---- */

/* Note: GGUF_MAGIC, types, and structs are declared in qw6.h. */

/* Size of a single GGUF scalar value (excluding strings/arrays) */
static size_t gguf_type_size(gguf_type_t t) {
    switch (t) {
        case GGUF_TYPE_UINT8:   return 1;
        case GGUF_TYPE_INT8:    return 1;
        case GGUF_TYPE_UINT16:  return 2;
        case GGUF_TYPE_INT16:   return 2;
        case GGUF_TYPE_UINT32:  return 4;
        case GGUF_TYPE_INT32:   return 4;
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_BOOL:    return 1;
        case GGUF_TYPE_UINT64:  return 8;
        case GGUF_TYPE_INT64:   return 8;
        case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;  /* STRING and ARRAY are variable-size */
    }
}

/* Read a GGUF string: uint64_t length + chars (no null terminator) */
static int gguf_read_str(FILE *f, char **out_str, uint64_t *out_len) {
    QW6_ASSERT_PTR(f);
    QW6_ASSERT_PTR(out_str);

    uint64_t len;
    if (fread(&len, sizeof(len), 1, f) != 1) {
        fprintf(stderr, "qw6: GGUF string length read failed\n");
        return -1;
    }

    /* Sanity-check: no string in a real GGUF should exceed 1MB */
    if (len > (1u << 20)) {
        fprintf(stderr, "qw6: GGUF string length %llu unreasonably large\n",
                (unsigned long long)len);
        return -1;
    }

    char *s = malloc((size_t)len + 1);
    QW6_ASSERT_PTR(s);
    if (len > 0 && fread(s, 1, (size_t)len, f) != (size_t)len) {
        fprintf(stderr, "qw6: GGUF string data read failed\n");
        free(s);
        return -1;
    }
    s[len] = '\0';

    *out_str = s;
    if (out_len) *out_len = len;
    return 0;
}

/* Read a single GGUF scalar value into a kv union */
static int gguf_read_scalar(FILE *f, gguf_type_t type, gguf_kv_t *kv) {
    QW6_ASSERT_PTR(f);
    QW6_ASSERT_PTR(kv);

    switch (type) {
        case GGUF_TYPE_UINT8: {
            uint8_t v;
            if (fread(&v, 1, 1, f) != 1) return -1;
            kv->val.u8 = v;
            break;
        }
        case GGUF_TYPE_INT8: {
            int8_t v;
            if (fread(&v, 1, 1, f) != 1) return -1;
            kv->val.i8 = v;
            break;
        }
        case GGUF_TYPE_UINT16: {
            uint16_t v;
            if (fread(&v, 2, 1, f) != 1) return -1;
            kv->val.u16 = v;
            break;
        }
        case GGUF_TYPE_INT16: {
            int16_t v;
            if (fread(&v, 2, 1, f) != 1) return -1;
            kv->val.i16 = v;
            break;
        }
        case GGUF_TYPE_UINT32: {
            uint32_t v;
            if (fread(&v, 4, 1, f) != 1) return -1;
            kv->val.u32 = v;
            break;
        }
        case GGUF_TYPE_INT32: {
            int32_t v;
            if (fread(&v, 4, 1, f) != 1) return -1;
            kv->val.i32 = v;
            break;
        }
        case GGUF_TYPE_FLOAT32: {
            float v;
            if (fread(&v, 4, 1, f) != 1) return -1;
            kv->val.f32 = v;
            break;
        }
        case GGUF_TYPE_BOOL: {
            int8_t v;
            if (fread(&v, 1, 1, f) != 1) return -1;
            kv->val.b = (v != 0);
            break;
        }
        case GGUF_TYPE_UINT64: {
            uint64_t v;
            if (fread(&v, 8, 1, f) != 1) return -1;
            kv->val.u64 = v;
            break;
        }
        case GGUF_TYPE_INT64: {
            int64_t v;
            if (fread(&v, 8, 1, f) != 1) return -1;
            kv->val.i64 = v;
            break;
        }
        case GGUF_TYPE_FLOAT64: {
            double v;
            if (fread(&v, 8, 1, f) != 1) return -1;
            kv->val.f64 = v;
            break;
        }
        default:
            return -1;
    }
    return 0;
}

/* Read a GGUF array value */
static int gguf_read_array(FILE *f, gguf_kv_t *kv) {
    QW6_ASSERT_PTR(f);
    QW6_ASSERT_PTR(kv);

    /* Read array element type (int32_t) */
    int32_t arr_type_raw;
    if (fread(&arr_type_raw, 4, 1, f) != 1) return -1;
    if (arr_type_raw < 0 || arr_type_raw > (int32_t)GGUF_TYPE_FLOAT64) {
        fprintf(stderr, "qw6: GGUF array type %d out of range\n", arr_type_raw);
        return -1;
    }
    gguf_type_t elem_type = (gguf_type_t)arr_type_raw;
    kv->arr_type = elem_type;

    /* Read array count (uint64_t) */
    uint64_t count;
    if (fread(&count, 8, 1, f) != 1) return -1;
    kv->arr_count = count;

    /* Sanity-check: no array should exceed 10M elements */
    if (count > (10ull * 1024 * 1024)) {
        fprintf(stderr, "qw6: GGUF array count %llu unreasonably large\n",
                (unsigned long long)count);
        return -1;
    }

    if (elem_type == GGUF_TYPE_STRING) {
        /* Array of strings */
        kv->arr_strs = calloc((size_t)count, sizeof(char *));
        QW6_ASSERT_PTR(kv->arr_strs);
        kv->arr_data = NULL;

        for (uint64_t i = 0; i < count; i++) {
            char *s = NULL;
            if (gguf_read_str(f, &s, NULL) != 0) {
                /* Clean up on failure */
                for (uint64_t j = 0; j < i; j++) free(kv->arr_strs[j]);
                free(kv->arr_strs);
                kv->arr_strs = NULL;
                return -1;
            }
            kv->arr_strs[i] = s;
        }
    } else {
        /* Array of scalars — read raw bytes */
        size_t elem_sz = gguf_type_size(elem_type);
        if (elem_sz == 0) {
            fprintf(stderr, "qw6: GGUF array of non-scalar type %d\n", elem_type);
            return -1;
        }

        size_t total = (size_t)count * elem_sz;
        kv->arr_data = malloc(total);
        QW6_ASSERT_PTR(kv->arr_data);
        kv->arr_strs = NULL;

        if (fread(kv->arr_data, 1, total, f) != total) {
            free(kv->arr_data);
            kv->arr_data = NULL;
            return -1;
        }
    }

    return 0;
}

/* Free a single KV pair's owned data */
static void gguf_kv_free_data(gguf_kv_t *kv) {
    if (kv->str) { free(kv->str); kv->str = NULL; }
    if (kv->arr_data) { free(kv->arr_data); kv->arr_data = NULL; }
    if (kv->arr_strs) {
        for (uint64_t i = 0; i < kv->arr_count; i++)
            if (kv->arr_strs[i]) free(kv->arr_strs[i]);
        free(kv->arr_strs);
        kv->arr_strs = NULL;
    }
}

void qw6_gguf_free(gguf_ctx_t *ctx) {
    QW6_ASSERT_PTR(ctx);
    for (uint32_t i = 0; i < ctx->kv_parsed; i++)
        gguf_kv_free_data(&ctx->kv[i]);
    memset(ctx, 0, sizeof(*ctx));
}

/* Find a KV pair by key name. Returns NULL if not found. */
const gguf_kv_t *qw6_gguf_find_kv(const gguf_ctx_t *ctx, const char *key) {
    QW6_ASSERT_PTR(ctx);
    QW6_ASSERT_PTR(key);
    for (uint32_t i = 0; i < ctx->kv_parsed; i++) {
        if (strcmp(ctx->kv[i].key, key) == 0)
            return &ctx->kv[i];
    }
    return NULL;
}

/* Map GGML tensor type → qw6 quant type. Returns GGML_TYPE count for sanity. */
static const char *ggml_type_name(ggml_type_t t) {
    switch (t) {
        case GGML_TYPE_F32:  return "F32";
        case GGML_TYPE_F16:  return "F16";
        case GGML_TYPE_Q4_0: return "Q4_0";
        case GGML_TYPE_Q4_1: return "Q4_1";
        case GGML_TYPE_Q5_0: return "Q5_0";
        case GGML_TYPE_Q5_1: return "Q5_1";
        case GGML_TYPE_Q8_0: return "Q8_0";
        case GGML_TYPE_Q8_1: return "Q8_1";
        case GGML_TYPE_Q2_K: return "Q2_K";
        case GGML_TYPE_Q3_K: return "Q3_K";
        case GGML_TYPE_Q4_K: return "Q4_K";
        case GGML_TYPE_Q5_K: return "Q5_K";
        case GGML_TYPE_Q6_K: return "Q6_K";
        case GGML_TYPE_Q8_K: return "Q8_K";
        case GGML_TYPE_IQ2_XXS: return "IQ2_XXS";
        case GGML_TYPE_IQ2_XS:  return "IQ2_XS";
        case GGML_TYPE_IQ3_XXS: return "IQ3_XXS";
        case GGML_TYPE_IQ1_S:   return "IQ1_S";
        case GGML_TYPE_IQ4_NL:  return "IQ4_NL";
        case GGML_TYPE_IQ3_S:   return "IQ3_S";
        case GGML_TYPE_IQ2_S:   return "IQ2_S";
        case GGML_TYPE_IQ4_XS:  return "IQ4_XS";
        case GGML_TYPE_I8:      return "I8";
        case GGML_TYPE_I16:     return "I16";
        case GGML_TYPE_I32:     return "I32";
        case GGML_TYPE_I64:     return "I64";
        case GGML_TYPE_F64:     return "F64";
        case GGML_TYPE_IQ1_M:   return "IQ1_M";
        case GGML_TYPE_BF16:    return "BF16";
        case GGML_TYPE_Q4_0_4_4: return "Q4_0_4_4";
        case GGML_TYPE_Q4_0_4_8: return "Q4_0_4_8";
        case GGML_TYPE_Q4_0_8_8: return "Q4_0_8_8";
        case GGML_TYPE_TQ1_0:   return "TQ1_0";
        case GGML_TYPE_TQ2_0:   return "TQ2_0";
        default: return "?";
    }
}

/* Check if GGML type is one qw6 recognises */
static bool qw6_gguf_type_supported(ggml_type_t t) {
    switch (t) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_S:
            return true;
        default:
            return false;
    }
}

static qw6_quant_t qw6_gguf_to_quant(ggml_type_t t);

static uint64_t qw6_file_size(FILE *f) {
    QW6_ASSERT_PTR(f);
    long here = ftell(f);
    if (here < 0) return 0;
    if (fseek(f, 0, SEEK_END) != 0) return 0;
    long end = ftell(f);
    if (fseek(f, here, SEEK_SET) != 0) return 0;
    return end < 0 ? 0 : (uint64_t)end;
}

static uint64_t gguf_tensor_span(const gguf_ctx_t *ctx, uint32_t i,
                                 uint64_t file_size) {
    QW6_ASSERT_PTR(ctx);
    QW6_ASSERT(i < ctx->tensors_parsed, "tensor index in range");

    uint64_t start = ctx->data_offset + ctx->tensors[i].offset;
    uint64_t end = file_size;
    for (uint32_t j = 0; j < ctx->tensors_parsed; j++) {
        uint64_t candidate = ctx->data_offset + ctx->tensors[j].offset;
        if (candidate > start && candidate < end) end = candidate;
    }
    return end > start ? end - start : 0;
}

static int tensor_layer_index(const char *name) {
    QW6_ASSERT_PTR(name);
    if (strncmp(name, "blk.", 4) != 0) return -1;
    char *end = NULL;
    long idx = strtol(name + 4, &end, 10);
    if (!end || *end != '.') return -1;
    if (idx < 0 || idx >= QW6_NUM_LAYERS) return -1;
    return (int)idx;
}

static void qw6_tensor_bind(qw6_tensor_t *dst, const gguf_tensor_info_t *ti,
                            uint8_t *base, uint64_t abs_offset,
                            uint64_t span) {
    QW6_ASSERT_PTR(dst);
    QW6_ASSERT_PTR(ti);
    snprintf(dst->name, sizeof(dst->name), "%s", ti->name);
    dst->n_dims = ti->n_dims;
    for (uint32_t i = 0; i < GGUF_MAX_DIMS; i++)
        dst->ne[i] = i < ti->n_dims ? (uint32_t)ti->dims[i] : 1u;
    dst->cols = ti->n_dims > 0 ? (uint32_t)ti->dims[0] : 1u;
    dst->rows = ti->n_dims > 1 ? (uint32_t)ti->dims[1] : 1u;
    dst->file_offset = abs_offset;
    dst->data_size = (size_t)span;
    dst->data = base ? (void *)(base + abs_offset) : NULL;
    dst->quant = qw6_gguf_to_quant(ti->type);
}

static bool name_has_suffix(const char *name, const char *suffix) {
    QW6_ASSERT_PTR(name);
    QW6_ASSERT_PTR(suffix);
    size_t n = strlen(name), s = strlen(suffix);
    return n >= s && strcmp(name + n - s, suffix) == 0;
}

static size_t qw6_quant_block_size(qw6_quant_t q) {
    switch (q) {
        case QW6_Q_FP32: return sizeof(float);
        case QW6_Q_FP16:
        case QW6_Q_BF16: return sizeof(uint16_t);
        case QW6_Q_Q4_K_M:
        case QW6_Q_Q4_K_S: return 144;
        case QW6_Q_Q5_K: return 176;
        case QW6_Q_Q6_K: return 210;
        case QW6_Q_IQ2_XXS: return 66;
        case QW6_Q_IQ2_S: return 82;
        case QW6_Q_IQ3_S: return 110;
        default: return 0;
    }
}

static size_t qw6_tensor_row_size_for(qw6_quant_t q, uint32_t cols) {
    size_t bs = qw6_quant_block_size(q);
    if (bs == 0) return 0;
    if (q == QW6_Q_FP32 || q == QW6_Q_FP16 || q == QW6_Q_BF16)
        return (size_t)cols * bs;
    if (cols % QW6_QK_K != 0) return 0;
    return (size_t)(cols / QW6_QK_K) * bs;
}

static void bind_expert_pack(qw6_tensor_t *dst, const gguf_tensor_info_t *ti,
                             uint8_t *base, uint64_t abs_offset,
                             uint64_t span) {
    QW6_ASSERT_PTR(dst); QW6_ASSERT_PTR(ti);
    if (ti->n_dims != 3 || ti->dims[2] != QW6_NUM_EXPERTS) return;

    qw6_quant_t q = qw6_gguf_to_quant(ti->type);
    uint32_t cols = (uint32_t)ti->dims[0];
    uint32_t rows = (uint32_t)ti->dims[1];
    size_t row_size = qw6_tensor_row_size_for(q, cols);
    size_t expert_size = row_size * rows;
    if (row_size == 0 || expert_size == 0 ||
        expert_size * QW6_NUM_EXPERTS > span) return;

    for (uint32_t e = 0; e < QW6_NUM_EXPERTS; e++) {
        snprintf(dst[e].name, sizeof(dst[e].name), "%.100s#%03u", ti->name, e);
        dst[e].cols = cols;
        dst[e].rows = rows;
        dst[e].n_dims = 2;
        dst[e].ne[0] = cols;
        dst[e].ne[1] = rows;
        dst[e].ne[2] = 1;
        dst[e].ne[3] = 1;
        dst[e].quant = q;
        dst[e].file_offset = abs_offset + (uint64_t)e * expert_size;
        dst[e].data_size = expert_size;
        dst[e].data = base ? (void *)(base + dst[e].file_offset) : NULL;
    }
}

static int bind_layer_tensor(qw6_model_t *m, int layer,
                             const gguf_tensor_info_t *ti,
                             uint8_t *base, uint64_t abs_offset,
                             uint64_t span) {
    QW6_ASSERT_PTR(m);
    QW6_ASSERT_PTR(ti);
    QW6_ASSERT(layer >= 0 && layer < QW6_NUM_LAYERS, "layer in range");

    char prefix[32];
    snprintf(prefix, sizeof(prefix), "blk.%d.", layer);
    const char *local = ti->name + strlen(prefix);
    qw6_tensor_t *dst = NULL;

    if (strcmp(local, "attn_norm.weight") == 0) dst = &m->layers[layer].norm;
    else if (strcmp(local, "post_attention_norm.weight") == 0) dst = &m->layers[layer].post_norm;
    else if (strcmp(local, "attn_qkv.weight") == 0) dst = &m->layers[layer].attn_q;
    else if (strcmp(local, "attn_q.weight") == 0) dst = &m->layers[layer].attn_q;
    else if (strcmp(local, "attn_k.weight") == 0) dst = &m->layers[layer].attn_k;
    else if (strcmp(local, "attn_v.weight") == 0) dst = &m->layers[layer].attn_v;
    else if (strcmp(local, "attn_q_norm.weight") == 0) dst = &m->layers[layer].attn_q_norm;
    else if (strcmp(local, "attn_k_norm.weight") == 0) dst = &m->layers[layer].attn_k_norm;
    else if (strcmp(local, "attn_output.weight") == 0) dst = &m->layers[layer].attn_o;
    else if (strcmp(local, "attn_gate.weight") == 0) dst = &m->layers[layer].attn_gate;
    else if (strcmp(local, "ssm_conv1d.weight") == 0) dst = &m->layers[layer].conv1d;
    else if (strcmp(local, "ssm_in.weight") == 0) dst = &m->layers[layer].dn_key;
    else if (strcmp(local, "ssm_out.weight") == 0) dst = &m->layers[layer].dn_out;
    else if (strcmp(local, "ssm_norm.weight") == 0) dst = &m->layers[layer].dn_norm;
    else if (strcmp(local, "ssm_alpha.weight") == 0) dst = &m->layers[layer].dn_alpha;
    else if (strcmp(local, "ssm_beta.weight") == 0) dst = &m->layers[layer].dn_beta;
    else if (strcmp(local, "ssm_dt.bias") == 0) dst = &m->layers[layer].dn_dt;
    else if (strcmp(local, "ssm_a") == 0) dst = &m->layers[layer].dn_a;
    else if (strcmp(local, "ffn_gate_inp.weight") == 0) dst = &m->layers[layer].moe_router;
    else if (strcmp(local, "ffn_gate_inp_shexp.weight") == 0) dst = &m->layers[layer].shared_router;
    else if (strcmp(local, "ffn_gate_exps.weight") == 0) {
        bind_expert_pack(m->layers[layer].expert_gate, ti, base, abs_offset, span);
        return 1;
    } else if (strcmp(local, "ffn_up_exps.weight") == 0) {
        bind_expert_pack(m->layers[layer].expert_up, ti, base, abs_offset, span);
        return 1;
    } else if (strcmp(local, "ffn_down_exps.weight") == 0) {
        bind_expert_pack(m->layers[layer].expert_down, ti, base, abs_offset, span);
        return 1;
    }
    else if (strcmp(local, "ffn_gate_shexp.weight") == 0) dst = &m->layers[layer].shared_gate;
    else if (strcmp(local, "ffn_up_shexp.weight") == 0) dst = &m->layers[layer].shared_up;
    else if (strcmp(local, "ffn_down_shexp.weight") == 0) dst = &m->layers[layer].shared_down;
    else if (name_has_suffix(local, ".weight") &&
             (strstr(local, "ssm_") || strstr(local, "ffn_") ||
              strstr(local, "attn_"))) {
        return 0;
    }

    if (!dst) return 0;
    qw6_tensor_bind(dst, ti, base, abs_offset, span);
    return 1;
}

static int qw6_validate_qwen_metadata(const gguf_ctx_t *ctx) {
    QW6_ASSERT_PTR(ctx);
    const gguf_kv_t *arch = qw6_gguf_find_kv(ctx, "general.architecture");
    const gguf_kv_t *layers = qw6_gguf_find_kv(ctx, "qwen3_5_moe.block_count");
    const gguf_kv_t *hidden = qw6_gguf_find_kv(ctx, "qwen3_5_moe.embedding_length");

    if (!arch || arch->type != GGUF_TYPE_STRING) {
        fprintf(stderr, "qw6: missing general.architecture\n");
        return -1;
    }
    if (strcmp(arch->str, "qwen3_5_moe") != 0 &&
        strcmp(arch->str, "qwen3.5moe") != 0 &&
        strcmp(arch->str, "qwen35moe") != 0) {
        fprintf(stderr, "qw6: expected qwen3_5_moe architecture, got %s\n",
                arch->str);
        return -1;
    }
    if (!layers) layers = qw6_gguf_find_kv(ctx, "qwen35moe.block_count");
    if (!hidden) hidden = qw6_gguf_find_kv(ctx, "qwen35moe.embedding_length");
    if (layers && layers->type == GGUF_TYPE_UINT32 &&
        layers->val.u32 != QW6_NUM_LAYERS) {
        fprintf(stderr, "qw6: expected %d layers, got %u\n",
                QW6_NUM_LAYERS, layers->val.u32);
        return -1;
    }
    if (hidden && hidden->type == GGUF_TYPE_UINT32 &&
        hidden->val.u32 != QW6_HIDDEN_SIZE) {
        fprintf(stderr, "qw6: expected hidden size %d, got %u\n",
                QW6_HIDDEN_SIZE, hidden->val.u32);
        return -1;
    }
    return 0;
}

/* Parse a GGUF file: header + metadata KV + tensor info table.
 * Does NOT load tensor data — just records offsets. */
int qw6_gguf_parse(const char *path, gguf_ctx_t *ctx) {
    QW6_ASSERT_PTR(path);
    QW6_ASSERT_PTR(ctx);
    memset(ctx, 0, sizeof(*ctx));
    ctx->alignment = GGUF_DEFAULT_ALIGNMENT;

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "qw6: cannot open GGUF file: %s\n", path);
        return -1;
    }
    ctx->file_size = qw6_file_size(f);

    /* --- Header --- */
    uint32_t magic, version;
    uint64_t tensor_count, kv_count;

    if (fread(&magic, 4, 1, f) != 1) { fclose(f); return -1; }
    if (fread(&version, 4, 1, f) != 1) { fclose(f); return -1; }
    if (fread(&tensor_count, 8, 1, f) != 1) { fclose(f); return -1; }
    if (fread(&kv_count, 8, 1, f) != 1) { fclose(f); return -1; }

    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "qw6: not a GGUF file (magic=0x%08x)\n", magic);
        fclose(f);
        return -1;
    }
    if (version != GGUF_VERSION) {
        fprintf(stderr, "qw6: unsupported GGUF version %u (expected %u)\n",
                version, GGUF_VERSION);
        fclose(f);
        return -1;
    }

    ctx->version = version;
    ctx->tensor_count = tensor_count;
    ctx->kv_count = kv_count;

    fprintf(stderr, "qw6: GGUF v%u, %llu tensors, %llu KV pairs\n",
            version, (unsigned long long)tensor_count,
            (unsigned long long)kv_count);

    if (kv_count > GGUF_MAX_KV) {
        fprintf(stderr, "qw6: too many KV pairs (%llu > %u)\n",
                (unsigned long long)kv_count, GGUF_MAX_KV);
        fclose(f);
        return -1;
    }
    if (tensor_count > GGUF_MAX_TENSORS) {
        fprintf(stderr, "qw6: too many tensors (%llu > %u)\n",
                (unsigned long long)tensor_count, GGUF_MAX_TENSORS);
        fclose(f);
        return -1;
    }

    /* --- Metadata KV pairs --- */
    for (uint64_t i = 0; i < kv_count; i++) {
        gguf_kv_t *kv = &ctx->kv[ctx->kv_parsed];

        /* Key string */
        char *key_str = NULL;
        if (gguf_read_str(f, &key_str, NULL) != 0) {
            fprintf(stderr, "qw6: GGUF KV key read failed at pair %llu\n",
                    (unsigned long long)i);
            qw6_gguf_free(ctx);
            fclose(f);
            return -1;
        }
        snprintf(kv->key, sizeof(kv->key), "%s", key_str);
        free(key_str);

        /* Value type (int32_t) */
        int32_t vtype_raw;
        if (fread(&vtype_raw, 4, 1, f) != 1) {
            fprintf(stderr, "qw6: GGUF KV type read failed at pair %llu\n",
                    (unsigned long long)i);
            qw6_gguf_free(ctx);
            fclose(f);
            return -1;
        }
        if (vtype_raw < 0 || vtype_raw > (int32_t)GGUF_TYPE_FLOAT64) {
            fprintf(stderr, "qw6: GGUF KV type %d out of range at pair %llu\n",
                    vtype_raw, (unsigned long long)i);
            qw6_gguf_free(ctx);
            fclose(f);
            return -1;
        }
        gguf_type_t vtype = (gguf_type_t)vtype_raw;
        kv->type = vtype;

        /* Value data */
        if (vtype == GGUF_TYPE_STRING) {
            if (gguf_read_str(f, &kv->str, &kv->str_len) != 0) {
                qw6_gguf_free(ctx);
                fclose(f);
                return -1;
            }
        } else if (vtype == GGUF_TYPE_ARRAY) {
            if (gguf_read_array(f, kv) != 0) {
                qw6_gguf_free(ctx);
                fclose(f);
                return -1;
            }
        } else {
            if (gguf_read_scalar(f, vtype, kv) != 0) {
                qw6_gguf_free(ctx);
                fclose(f);
                return -1;
            }
        }

        /* Check for general.alignment override */
        if (strcmp(kv->key, "general.alignment") == 0 && vtype == GGUF_TYPE_UINT32) {
            ctx->alignment = kv->val.u32;
        }

        ctx->kv_parsed++;
    }

    /* --- Tensor info table --- */
    for (uint64_t i = 0; i < tensor_count; i++) {
        gguf_tensor_info_t *ti = &ctx->tensors[ctx->tensors_parsed];

        /* Tensor name (string) */
        char *name_str = NULL;
        if (gguf_read_str(f, &name_str, NULL) != 0) {
            fprintf(stderr, "qw6: tensor name read failed at tensor %llu\n",
                    (unsigned long long)i);
            qw6_gguf_free(ctx);
            fclose(f);
            return -1;
        }
        snprintf(ti->name, sizeof(ti->name), "%s", name_str);
        free(name_str);

        /* Number of dimensions (uint32_t) */
        uint32_t n_dims;
        if (fread(&n_dims, 4, 1, f) != 1) {
            qw6_gguf_free(ctx);
            fclose(f);
            return -1;
        }
        if (n_dims > GGUF_MAX_DIMS) {
            fprintf(stderr, "qw6: tensor '%s' has %u dims (max %u)\n",
                    ti->name, n_dims, GGUF_MAX_DIMS);
            qw6_gguf_free(ctx);
            fclose(f);
            return -1;
        }
        ti->n_dims = n_dims;

        /* Dimension sizes (int64_t each) */
        for (uint32_t d = 0; d < n_dims; d++) {
            if (fread(&ti->dims[d], 8, 1, f) != 1) {
                qw6_gguf_free(ctx);
                fclose(f);
                return -1;
            }
        }

        /* Tensor data type (int32_t, maps to ggml_type_t) */
        int32_t ttype_raw;
        if (fread(&ttype_raw, 4, 1, f) != 1) {
            qw6_gguf_free(ctx);
            fclose(f);
            return -1;
        }
        ti->type = (ggml_type_t)ttype_raw;

        /* Tensor data offset (uint64_t, relative to data start) */
        if (fread(&ti->offset, 8, 1, f) != 1) {
            qw6_gguf_free(ctx);
            fclose(f);
            return -1;
        }

        ctx->tensors_parsed++;
    }

    /* --- Compute data offset --- */
    /* The data section starts immediately after the metadata, aligned */
    long meta_end = ftell(f);
    if (meta_end < 0) {
        fprintf(stderr, "qw6: ftell failed\n");
        qw6_gguf_free(ctx);
        fclose(f);
        return -1;
    }

    uint64_t aligned_offset = (uint64_t)meta_end;
    uint32_t align = ctx->alignment;
    if (align == 0) align = GGUF_DEFAULT_ALIGNMENT;
    size_t remainder = (size_t)(aligned_offset % align);
    if (remainder > 0)
        aligned_offset += (align - remainder);

    ctx->data_offset = aligned_offset;

    fprintf(stderr, "qw6: parsed %u KV pairs, %u tensors, data at offset %llu (align=%u)\n",
            ctx->kv_parsed, ctx->tensors_parsed,
            (unsigned long long)ctx->data_offset, ctx->alignment);

    fclose(f);
    return 0;
}

/* Map GGML tensor type → qw6 quant enum (used by tensor data loader — Phase 1) */
__attribute__((unused))
static qw6_quant_t qw6_gguf_to_quant(ggml_type_t t) {
    switch (t) {
        case GGML_TYPE_F32:     return QW6_Q_FP32;
        case GGML_TYPE_F16:    return QW6_Q_FP16;
        case GGML_TYPE_BF16:   return QW6_Q_BF16;
        case GGML_TYPE_Q8_0:   return QW6_Q_Q8_0;
        case GGML_TYPE_Q5_K:   return QW6_Q_Q5_K;
        case GGML_TYPE_Q6_K:   return QW6_Q_Q6_K;
        case GGML_TYPE_Q4_K:   return QW6_Q_Q4_K_M;
        case GGML_TYPE_IQ2_S:  return QW6_Q_IQ2_S;
        case GGML_TYPE_IQ2_XXS: return QW6_Q_IQ2_XXS;
        case GGML_TYPE_IQ3_S:  return QW6_Q_IQ3_S;
        default:                return QW6_Q_FP32;   /* safe fallback */
    }
}

/* Load GGUF metadata into qw6 model struct (tensor data NOT loaded yet) */
int qw6_gguf_read_file(const char *path, qw6_model_t *m) {
    QW6_ASSERT_PTR(path);
    QW6_ASSERT_PTR(m);

    gguf_ctx_t ctx;
    if (qw6_gguf_parse(path, &ctx) != 0) return -1;
    if (qw6_validate_qwen_metadata(&ctx) != 0) {
        qw6_gguf_free(&ctx);
        return -1;
    }

    /* Extract key metadata */
    const gguf_kv_t *kv;

    kv = qw6_gguf_find_kv(&ctx, "general.architecture");
    if (kv && kv->type == GGUF_TYPE_STRING) {
        fprintf(stderr, "qw6: architecture: %s\n", kv->str);
    }

    /* Print some metadata for debugging */
    kv = qw6_gguf_find_kv(&ctx, "general.name");
    if (kv && kv->type == GGUF_TYPE_STRING) {
        fprintf(stderr, "qw6: model name: %s\n", kv->str);
    }

    /* Count tensor types */
    int type_counts[30] = {0};
    for (uint32_t i = 0; i < ctx.tensors_parsed; i++) {
        int t = (int)ctx.tensors[i].type;
        if (t >= 0 && t < 30) type_counts[t]++;
    }
    fprintf(stderr, "qw6: tensor type breakdown:\n");
    for (int t = 0; t < 30; t++) {
        if (type_counts[t] > 0)
            fprintf(stderr, "  %s: %d tensors\n",
                    ggml_type_name((ggml_type_t)t), type_counts[t]);
    }

    /* Check all tensor types are supported */
    int unsupported = 0;
    for (uint32_t i = 0; i < ctx.tensors_parsed; i++) {
        if (!qw6_gguf_type_supported(ctx.tensors[i].type)) {
            fprintf(stderr, "qw6: WARNING: tensor '%s' uses unsupported type %s\n",
                    ctx.tensors[i].name, ggml_type_name(ctx.tensors[i].type));
            unsupported++;
        }
    }

    /* Build a lightweight tensor index. Actual tensor bytes are loaded lazily
     * by the future forward path; this keeps Phase 1 metadata checks cheap. */
    m->max_context = QW6_DEFAULT_CTX;
    m->total_weight_bytes = 0;
    m->weight_fd = open(path, O_RDONLY);
    if (m->weight_fd < 0) {
        fprintf(stderr, "qw6: cannot open GGUF for mmap: %s\n", path);
        qw6_gguf_free(&ctx);
        return -1;
    }
    m->weight_map_size = (size_t)ctx.file_size;
    m->weight_map = mmap(NULL, m->weight_map_size, PROT_READ, MAP_PRIVATE,
                         m->weight_fd, 0);
    if (m->weight_map == MAP_FAILED) {
        fprintf(stderr, "qw6: mmap failed for GGUF weights\n");
        close(m->weight_fd);
        m->weight_fd = -1;
        m->weight_map = NULL;
        qw6_gguf_free(&ctx);
        return -1;
    }
    uint8_t *weight_base = (uint8_t *)m->weight_map;

    uint32_t layer_seen[QW6_NUM_LAYERS] = {0};
    uint32_t ssm_tensors = 0, attn_tensors = 0, moe_tensors = 0;
    uint32_t token_embd = 0, output = 0, output_norm = 0;

    for (uint32_t i = 0; i < ctx.tensors_parsed; i++) {
        const char *name = ctx.tensors[i].name;
        uint64_t span = gguf_tensor_span(&ctx, i, ctx.file_size);
        if (span > (uint64_t)SIZE_MAX - m->total_weight_bytes) {
            fprintf(stderr, "qw6: tensor byte count overflow\n");
            qw6_gguf_free(&ctx);
            return -1;
        }
        m->total_weight_bytes += (size_t)span;

        if (strcmp(name, "token_embd.weight") == 0) token_embd++;
        if (strcmp(name, "output.weight") == 0) output++;
        if (strcmp(name, "output_norm.weight") == 0) output_norm++;
        if (strstr(name, ".ssm_")) ssm_tensors++;
        if (strstr(name, ".attn_")) attn_tensors++;
        if (strstr(name, ".ffn_")) moe_tensors++;

        int layer = tensor_layer_index(name);
        if (layer >= 0) layer_seen[layer]++;

        uint64_t abs_offset = ctx.data_offset + ctx.tensors[i].offset;
        if (strcmp(name, "token_embd.weight") == 0)
            qw6_tensor_bind(&m->tok_embeddings, &ctx.tensors[i],
                            weight_base, abs_offset, span);
        else if (strcmp(name, "output.weight") == 0)
            qw6_tensor_bind(&m->output, &ctx.tensors[i],
                            weight_base, abs_offset, span);
        else if (strcmp(name, "output_norm.weight") == 0)
            qw6_tensor_bind(&m->output_norm, &ctx.tensors[i],
                            weight_base, abs_offset, span);
        else if (layer >= 0)
            (void)bind_layer_tensor(m, layer, &ctx.tensors[i],
                                    weight_base, abs_offset, span);
    }

    uint32_t layers_with_tensors = 0;
    for (uint32_t i = 0; i < QW6_NUM_LAYERS; i++)
        if (layer_seen[i] > 0) layers_with_tensors++;

    fprintf(stderr, "qw6: indexed %.2f GiB tensor data\n",
            (double)m->total_weight_bytes / (1024.0 * 1024.0 * 1024.0));
    fprintf(stderr, "qw6: tensor groups: emb=%u output=%u norm=%u layers=%u/%u attn=%u ssm=%u moe=%u\n",
            token_embd, output, output_norm, layers_with_tensors,
            QW6_NUM_LAYERS, attn_tensors, ssm_tensors, moe_tensors);

    if (unsupported > 0 || token_embd != 1 || output != 1 ||
        layers_with_tensors != QW6_NUM_LAYERS || ssm_tensors == 0 ||
        attn_tensors == 0 || moe_tensors == 0) {
        fprintf(stderr, "qw6: GGUF does not match required Qwen3.6 tensor layout\n");
        qw6_gguf_free(&ctx);
        return -1;
    }

    qw6_gguf_free(&ctx);
    fprintf(stderr, "qw6: GGUF tensor index ready (data loading is next Phase 1 step)\n");
    return 0;
}

/* Inspect a GGUF file — print header + metadata + tensor table */
static int qw6_gguf_inspect_file(const char *path) {
    QW6_ASSERT_PTR(path);

    gguf_ctx_t ctx;
    if (qw6_gguf_parse(path, &ctx) != 0) return -1;

    printf("GGUF v%u\n", ctx.version);
    printf("tensors: %u\n", ctx.tensors_parsed);
    printf("metadata_kv: %u\n", ctx.kv_parsed);
    printf("alignment: %u\n", ctx.alignment);
    printf("data_offset: %llu\n\n", (unsigned long long)ctx.data_offset);

    /* Print key metadata */
    printf("--- Metadata ---\n");
    for (uint32_t i = 0; i < ctx.kv_parsed; i++) {
        const gguf_kv_t *kv = &ctx.kv[i];
        printf("  %s = ", kv->key);
        switch (kv->type) {
            case GGUF_TYPE_STRING:
                printf("\"%s\" (string)", kv->str);
                break;
            case GGUF_TYPE_UINT32:
                printf("%u (uint32)", kv->val.u32);
                break;
            case GGUF_TYPE_INT32:
                printf("%d (int32)", kv->val.i32);
                break;
            case GGUF_TYPE_FLOAT32:
                printf("%f (float32)", kv->val.f32);
                break;
            case GGUF_TYPE_BOOL:
                printf("%s (bool)", kv->val.b ? "true" : "false");
                break;
            case GGUF_TYPE_UINT64:
                printf("%llu (uint64)", (unsigned long long)kv->val.u64);
                break;
            case GGUF_TYPE_INT64:
                printf("%lld (int64)", (long long)kv->val.i64);
                break;
            case GGUF_TYPE_ARRAY:
                printf("[array of %u elements, type %d]",
                       (uint32_t)kv->arr_count, (int)kv->arr_type);
                break;
            case GGUF_TYPE_UINT8:
                printf("%u (uint8)", kv->val.u8);
                break;
            case GGUF_TYPE_INT8:
                printf("%d (int8)", kv->val.i8);
                break;
            case GGUF_TYPE_UINT16:
                printf("%u (uint16)", kv->val.u16);
                break;
            case GGUF_TYPE_INT16:
                printf("%d (int16)", kv->val.i16);
                break;
            case GGUF_TYPE_FLOAT64:
                printf("%f (float64)", kv->val.f64);
                break;
            default:
                printf("(type %d)", (int)kv->type);
        }
        printf("\n");
    }

    /* Print first 20 tensors */
    printf("\n--- Tensors (first 20 of %u) ---\n", ctx.tensors_parsed);
    for (uint32_t i = 0; i < ctx.tensors_parsed && i < 20; i++) {
        const gguf_tensor_info_t *ti = &ctx.tensors[i];
        printf("  [%3u] %-40s %s [", i, ti->name, ggml_type_name(ti->type));
        for (uint32_t d = 0; d < ti->n_dims; d++) {
            printf("%lld%s", (long long)ti->dims[d],
                   d < ti->n_dims - 1 ? "," : "");
        }
        printf("] offset=%llu\n", (unsigned long long)ti->offset);
    }
    if (ctx.tensors_parsed > 20)
        printf("  ... (%u more)\n", ctx.tensors_parsed - 20);

    qw6_gguf_free(&ctx);
    return 0;
}

/* ---- Model lifecycle ---- */

int qw6_model_load(qw6_model_t *m, const char *gguf_path) {
    QW6_ASSERT_PTR(m);
    QW6_ASSERT_PTR(gguf_path);

    memset(m, 0, sizeof(*m));
    m->weight_fd = -1;
    m->max_context = QW6_DEFAULT_CTX;
    return qw6_gguf_read_file(gguf_path, m);
}

void qw6_model_free(qw6_model_t *m) {
    QW6_ASSERT_PTR(m);

    for (int i = 0; i < QW6_NUM_LAYERS; i++) {
        if (m->deltanet_state[i]) free(m->deltanet_state[i]);
        if (m->k_cache[i]) free(m->k_cache[i]);
        if (m->v_cache[i]) free(m->v_cache[i]);
    }

    if (m->weight_map && m->weight_map != MAP_FAILED)
        munmap(m->weight_map, m->weight_map_size);
    if (m->weight_fd >= 0) close(m->weight_fd);
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
            m->deltanet_state[i] = calloc((size_t)QW6_NUM_VALUE_HEADS *
                                          QW6_VALUE_HEAD_DIM * QW6_VALUE_HEAD_DIM,
                                          sizeof(float));
            s->conv_state[i] = calloc((size_t)QW6_CONV1D_KERNEL *
                                      QW6_LINEAR_QKV_DIM, sizeof(float));
            QW6_ASSERT_PTR(m->deltanet_state[i]);
            QW6_ASSERT_PTR(s->conv_state[i]);
        } else {
            m->k_cache[i] = calloc((size_t)max_tokens * QW6_NUM_KV_HEADS *
                                   QW6_HEAD_DIM, sizeof(float));
            m->v_cache[i] = calloc((size_t)max_tokens * QW6_NUM_KV_HEADS *
                                   QW6_HEAD_DIM, sizeof(float));
            QW6_ASSERT_PTR(m->k_cache[i]);
            QW6_ASSERT_PTR(m->v_cache[i]);
        }
    }
    return 0;
}

void qw6_session_free(qw6_session_t *s) {
    QW6_ASSERT_PTR(s);
    for (int i = 0; i < QW6_NUM_LAYERS; i++)
        if (s->conv_state[i]) free(s->conv_state[i]);
    if (s->tokens) free(s->tokens);
    if (s->logits) free(s->logits);
    memset(s, 0, sizeof(*s));
}

/* ---- Native GGML block dequantization ---- */

#ifndef QW6_QK_K
#define QW6_QK_K 256
#endif
#define QW6_K_SCALE_SIZE 12

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[QW6_K_SCALE_SIZE];
    uint8_t qs[QW6_QK_K / 2];
} qw6_block_q4_k_t;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[QW6_K_SCALE_SIZE];
    uint8_t qh[QW6_QK_K / 8];
    uint8_t qs[QW6_QK_K / 2];
} qw6_block_q5_k_t;

typedef struct {
    uint8_t ql[QW6_QK_K / 2];
    uint8_t qh[QW6_QK_K / 4];
    int8_t scales[QW6_QK_K / 16];
    uint16_t d;
} qw6_block_q6_k_t;

typedef struct {
    uint16_t d;
    uint16_t qs[QW6_QK_K / 8];
} qw6_block_iq2_xxs_t;

typedef struct {
    uint16_t d;
    uint8_t qs[QW6_QK_K / 4];
    uint8_t qh[QW6_QK_K / 32];
    uint8_t scales[QW6_QK_K / 32];
} qw6_block_iq2_s_t;

typedef struct {
    uint16_t d;
    uint8_t qs[QW6_QK_K / 4];
    uint8_t qh[QW6_QK_K / 32];
    uint8_t signs[QW6_QK_K / 8];
    uint8_t scales[QW6_QK_K / 64];
} qw6_block_iq3_s_t;

static float qw6_fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    uint32_t out;

    if (exp == 0) {
        if (mant == 0) out = sign;
        else {
            exp = 1;
            while ((mant & 0x0400) == 0) { mant <<= 1; exp--; }
            mant &= 0x03ff;
            out = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + 112) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &out, sizeof(f));
    return f;
}

static float qw6_bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static void qw6_get_scale_min_k4(int j, const uint8_t *q, uint8_t *d,
                                 uint8_t *m) {
    QW6_ASSERT_PTR(q); QW6_ASSERT_PTR(d); QW6_ASSERT_PTR(m);
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

static int qw6_dequantize_row_q4_k(const void *src, float *dst, int k) {
    QW6_ASSERT_PTR(src); QW6_ASSERT_PTR(dst);
    QW6_ASSERT(k > 0 && k % QW6_QK_K == 0, "q4_k row multiple");
    const qw6_block_q4_k_t *x = (const qw6_block_q4_k_t *)src;
    int nb = k / QW6_QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t *q = x[i].qs;
        const float d = qw6_fp16_to_f32(x[i].d);
        const float min = qw6_fp16_to_f32(x[i].dmin);
        int is = 0;
        for (int j = 0; j < QW6_QK_K; j += 64) {
            uint8_t sc, m;
            qw6_get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            float d1 = d * sc, m1 = min * m;
            qw6_get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; l++) *dst++ = d1 * (q[l] & 0x0f) - m1;
            for (int l = 0; l < 32; l++) *dst++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
    return 0;
}

static int qw6_dequantize_row_q5_k(const void *src, float *dst, int k) {
    QW6_ASSERT_PTR(src); QW6_ASSERT_PTR(dst);
    QW6_ASSERT(k > 0 && k % QW6_QK_K == 0, "q5_k row multiple");
    const qw6_block_q5_k_t *x = (const qw6_block_q5_k_t *)src;
    int nb = k / QW6_QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t *ql = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const float d = qw6_fp16_to_f32(x[i].d);
        const float min = qw6_fp16_to_f32(x[i].dmin);
        uint8_t u1 = 1, u2 = 2;
        int is = 0;
        for (int j = 0; j < QW6_QK_K; j += 64) {
            uint8_t sc, m;
            qw6_get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            float d1 = d * sc, m1 = min * m;
            qw6_get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; l++)
                *dst++ = d1 * ((ql[l] & 0x0f) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; l++)
                *dst++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            u1 <<= 2;
            u2 <<= 2;
            is += 2;
        }
    }
    return 0;
}

static int qw6_dequantize_row_q6_k(const void *src, float *dst, int k) {
    QW6_ASSERT_PTR(src); QW6_ASSERT_PTR(dst);
    QW6_ASSERT(k > 0 && k % QW6_QK_K == 0, "q6_k row multiple");
    const qw6_block_q6_k_t *x = (const qw6_block_q6_k_t *)src;
    int nb = k / QW6_QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = qw6_fp16_to_f32(x[i].d);
        const uint8_t *ql = x[i].ql;
        const uint8_t *qh = x[i].qh;
        const int8_t *sc = x[i].scales;
        for (int n = 0; n < QW6_QK_K; n += 128) {
            for (int l = 0; l < 32; l++) {
                int q1 = ((ql[l +  0] & 0x0f) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = ((ql[l + 32] & 0x0f) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = ((ql[l +  0] >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = ((ql[l + 32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32;
                dst[n + l +  0] = d * sc[(n / 16) + 0] * q1;
                dst[n + l + 32] = d * sc[(n / 16) + 2] * q2;
                dst[n + l + 64] = d * sc[(n / 16) + 4] * q3;
                dst[n + l + 96] = d * sc[(n / 16) + 6] * q4;
            }
            ql += 64;
            qh += 32;
        }
        dst += QW6_QK_K;
    }
    return 0;
}

static int qw6_dequantize_row_iq2_xxs(const void *src, float *dst, int k) {
    QW6_ASSERT_PTR(src); QW6_ASSERT_PTR(dst);
    QW6_ASSERT(k > 0 && k % QW6_QK_K == 0, "iq2_xxs row multiple");
    const qw6_block_iq2_xxs_t *x = (const qw6_block_iq2_xxs_t *)src;
    int nb = k / QW6_QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = qw6_fp16_to_f32(x[i].d);
        for (int ib32 = 0; ib32 < QW6_QK_K / 32; ib32++) {
            uint32_t aux32[2];
            memcpy(aux32, x[i].qs + 4 * ib32, sizeof(aux32));
            const uint8_t *aux8 = (const uint8_t *)aux32;
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                const uint8_t *grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                uint8_t signs = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
                for (int j = 0; j < 8; j++)
                    *dst++ = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.0f : 1.0f);
            }
        }
    }
    return 0;
}

static int qw6_dequantize_row_iq2_s(const void *src, float *dst, int k) {
    QW6_ASSERT_PTR(src); QW6_ASSERT_PTR(dst);
    QW6_ASSERT(k > 0 && k % QW6_QK_K == 0, "iq2_s row multiple");
    const qw6_block_iq2_s_t *x = (const qw6_block_iq2_s_t *)src;
    int nb = k / QW6_QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = qw6_fp16_to_f32(x[i].d);
        const uint8_t *qs = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const uint8_t *signs = qs + QW6_QK_K / 8;
        for (int ib32 = 0; ib32 < QW6_QK_K / 32; ib32++) {
            float db0 = d * (0.5f + (x[i].scales[ib32] & 0x0f)) * 0.25f;
            float db1 = d * (0.5f + (x[i].scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; l++) {
                float dl = l < 2 ? db0 : db1;
                uint32_t grid_idx = qs[l] | ((qh[ib32] << (8 - 2 * l)) & 0x300);
                const uint8_t *grid = (const uint8_t *)(iq2s_grid + grid_idx);
                for (int j = 0; j < 8; j++)
                    *dst++ = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.0f : 1.0f);
            }
            qs += 4;
            signs += 4;
        }
    }
    return 0;
}

static int qw6_dequantize_row_iq3_s(const void *src, float *dst, int k) {
    QW6_ASSERT_PTR(src); QW6_ASSERT_PTR(dst);
    QW6_ASSERT(k > 0 && k % QW6_QK_K == 0, "iq3_s row multiple");
    const qw6_block_iq3_s_t *x = (const qw6_block_iq3_s_t *)src;
    int nb = k / QW6_QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = qw6_fp16_to_f32(x[i].d);
        const uint8_t *qs = x[i].qs;
        const uint8_t *qh = x[i].qh;
        const uint8_t *signs = x[i].signs;
        for (int ib32 = 0; ib32 < QW6_QK_K / 32; ib32 += 2) {
            float db1 = d * (1 + 2 * (x[i].scales[ib32 / 2] & 0x0f));
            float db2 = d * (1 + 2 * (x[i].scales[ib32 / 2] >> 4));
            for (int l = 0; l < 4; l++) {
                const uint8_t *g1 = (const uint8_t *)(iq3s_grid + (qs[2*l] | ((qh[0] << (8 - 2*l)) & 256)));
                const uint8_t *g2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[0] << (7 - 2*l)) & 256)));
                for (int j = 0; j < 4; j++) {
                    dst[j] = db1 * g1[j] * (signs[l] & kmask_iq2xs[j] ? -1.0f : 1.0f);
                    dst[j + 4] = db1 * g2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.0f : 1.0f);
                }
                dst += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; l++) {
                const uint8_t *g1 = (const uint8_t *)(iq3s_grid + (qs[2*l] | ((qh[1] << (8 - 2*l)) & 256)));
                const uint8_t *g2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[1] << (7 - 2*l)) & 256)));
                for (int j = 0; j < 4; j++) {
                    dst[j] = db2 * g1[j] * (signs[l] & kmask_iq2xs[j] ? -1.0f : 1.0f);
                    dst[j + 4] = db2 * g2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.0f : 1.0f);
                }
                dst += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
    return 0;
}

static int qw6_tensor_dequantize_row(const qw6_tensor_t *t, uint32_t row,
                                     float *dst) {
    QW6_ASSERT_PTR(t); QW6_ASSERT_PTR(dst);
    QW6_ASSERT(t->data != NULL, "tensor has data");
    QW6_ASSERT(row < t->rows, "row in tensor");

    uint32_t cols = t->cols;
    if (t->quant == QW6_Q_FP32) {
        const float *src = (const float *)t->data + (size_t)row * cols;
        memcpy(dst, src, (size_t)cols * sizeof(float));
        return 0;
    }
    if (t->quant == QW6_Q_FP16 || t->quant == QW6_Q_BF16) {
        const uint16_t *src = (const uint16_t *)t->data + (size_t)row * cols;
        for (uint32_t i = 0; i < cols; i++)
            dst[i] = t->quant == QW6_Q_FP16 ? qw6_fp16_to_f32(src[i])
                                            : qw6_bf16_to_f32(src[i]);
        return 0;
    }

    size_t block_size = 0;
    if (t->quant == QW6_Q_Q4_K_M || t->quant == QW6_Q_Q4_K_S)
        block_size = sizeof(qw6_block_q4_k_t);
    else if (t->quant == QW6_Q_Q5_K)
        block_size = sizeof(qw6_block_q5_k_t);
    else if (t->quant == QW6_Q_Q6_K)
        block_size = sizeof(qw6_block_q6_k_t);
    else if (t->quant == QW6_Q_IQ2_XXS)
        block_size = sizeof(qw6_block_iq2_xxs_t);
    else if (t->quant == QW6_Q_IQ2_S)
        block_size = sizeof(qw6_block_iq2_s_t);
    else if (t->quant == QW6_Q_IQ3_S)
        block_size = sizeof(qw6_block_iq3_s_t);
    else
        return -1;

    QW6_ASSERT(cols % QW6_QK_K == 0, "k quant row multiple");
    size_t row_size = (size_t)(cols / QW6_QK_K) * block_size;
    const uint8_t *src = (const uint8_t *)t->data + (size_t)row * row_size;
    if (t->quant == QW6_Q_Q4_K_M || t->quant == QW6_Q_Q4_K_S)
        return qw6_dequantize_row_q4_k(src, dst, (int)cols);
    if (t->quant == QW6_Q_Q5_K)
        return qw6_dequantize_row_q5_k(src, dst, (int)cols);
    if (t->quant == QW6_Q_Q6_K)
        return qw6_dequantize_row_q6_k(src, dst, (int)cols);
    if (t->quant == QW6_Q_IQ2_XXS)
        return qw6_dequantize_row_iq2_xxs(src, dst, (int)cols);
    if (t->quant == QW6_Q_IQ2_S)
        return qw6_dequantize_row_iq2_s(src, dst, (int)cols);
    return qw6_dequantize_row_iq3_s(src, dst, (int)cols);
}

static int qw6_probe_tensor_row(const char *label, const qw6_tensor_t *t) {
    QW6_ASSERT_PTR(label);
    QW6_ASSERT_PTR(t);
    if (!t->data || t->cols == 0 || t->rows == 0) {
        fprintf(stderr, "qw6: probe %-18s unavailable\n", label);
        return -1;
    }

    float *row = calloc(t->cols, sizeof(float));
    QW6_ASSERT_PTR(row);
    int rc = qw6_tensor_dequantize_row(t, 0, row);
    if (rc != 0) {
        fprintf(stderr, "qw6: probe %-18s unsupported quant=%d tensor=%s\n",
                label, (int)t->quant, t->name);
        free(row);
        return -1;
    }

    double sum = 0.0, abs_sum = 0.0, max_abs = 0.0;
    for (uint32_t i = 0; i < t->cols; i++) {
        double v = row[i];
        double av = fabs(v);
        sum += v;
        abs_sum += av;
        if (av > max_abs) max_abs = av;
    }
    fprintf(stderr,
            "qw6: probe %-18s quant=%d shape=[%u,%u] sum=%.6f abs=%.6f max=%.6f first=%.6f\n",
            label, (int)t->quant, t->cols, t->rows,
            sum, abs_sum, max_abs, row[0]);
    free(row);
    return 0;
}

static int qw6_model_probe_dequant(qw6_model_t *m) {
    QW6_ASSERT_PTR(m);
    int fail = 0;
    fail += qw6_probe_tensor_row("output.weight", &m->output) != 0;
    fail += qw6_probe_tensor_row("shared_gate", &m->layers[0].shared_gate) != 0;
    fail += qw6_probe_tensor_row("shared_down", &m->layers[0].shared_down) != 0;
    return fail == 0 ? 0 : -1;
}

int qw6_tensor_matvec(float *out, const qw6_tensor_t *t, const float *x,
                      uint32_t max_rows) {
    QW6_ASSERT_PTR(out); QW6_ASSERT_PTR(t); QW6_ASSERT_PTR(x);
    QW6_ASSERT(t->cols > 0 && t->rows > 0, "tensor shape valid");

    uint32_t rows = t->rows;
    if (max_rows > 0 && max_rows < rows) rows = max_rows;
    float *row = malloc((size_t)t->cols * sizeof(float));
    QW6_ASSERT_PTR(row);

    for (uint32_t r = 0; r < rows; r++) {
        if (qw6_tensor_dequantize_row(t, r, row) != 0) {
            free(row);
            return -1;
        }
        float acc = 0.0f;
        for (uint32_t c = 0; c < t->cols; c++) acc += row[c] * x[c];
        out[r] = acc;
    }

    free(row);
    return 0;
}

static int qw6_probe_matvec(qw6_model_t *m) {
    QW6_ASSERT_PTR(m);
    if (!m->tok_embeddings.data || !m->output.data) return -1;

    float *hidden = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    float logits[8] = {0};
    QW6_ASSERT_PTR(hidden);

    if (qw6_tensor_dequantize_row(&m->tok_embeddings, 0, hidden) != 0) {
        free(hidden);
        return -1;
    }
    if (qw6_tensor_matvec(logits, &m->output, hidden, 8) != 0) {
        free(hidden);
        return -1;
    }

    fprintf(stderr,
            "qw6: probe native matvec logits[0..7]=%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
            logits[0], logits[1], logits[2], logits[3],
            logits[4], logits[5], logits[6], logits[7]);
    free(hidden);
    return 0;
}

static int qw6_probe_layer0_router(qw6_model_t *m) {
    QW6_ASSERT_PTR(m);
    qw6_tensor_t *norm_t = &m->layers[0].norm;
    qw6_tensor_t *router_t = &m->layers[0].moe_router;
    if (!norm_t->data || !router_t->data) return -1;

    float *hidden = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    float *norm_w = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    float *normed = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    float router_logits[QW6_NUM_EXPERTS] = {0};
    int expert_idx[QW6_EXPERTS_PER_TOK] = {0};
    float expert_w[QW6_EXPERTS_PER_TOK] = {0};
    QW6_ASSERT_PTR(hidden); QW6_ASSERT_PTR(norm_w); QW6_ASSERT_PTR(normed);

    int rc = qw6_tensor_dequantize_row(&m->tok_embeddings, 0, hidden);
    if (rc == 0) rc = qw6_tensor_dequantize_row(norm_t, 0, norm_w);
    if (rc == 0) {
        qw6_cpu_rmsnorm(normed, hidden, norm_w, QW6_HIDDEN_SIZE);
        rc = qw6_tensor_matvec(router_logits, router_t, normed, QW6_NUM_EXPERTS);
    }
    if (rc == 0) {
        qw6_cpu_moe_route(expert_idx, expert_w, router_logits,
                          QW6_NUM_EXPERTS, QW6_EXPERTS_PER_TOK);
        fprintf(stderr, "qw6: probe layer0 router top8:");
        for (int i = 0; i < QW6_EXPERTS_PER_TOK; i++)
            fprintf(stderr, " %d:%.4f", expert_idx[i], expert_w[i]);
        fprintf(stderr, "\n");
        const qw6_tensor_t *eg = &m->layers[0].expert_gate[expert_idx[0]];
        fprintf(stderr,
                "qw6: probe top expert gate tensor=%s quant=%d shape=[%u,%u] bytes=%zu\n",
                eg->name, (int)eg->quant, eg->cols, eg->rows, eg->data_size);
    }

    free(hidden); free(norm_w); free(normed);
    return rc;
}

static int qw6_probe_layer0_shared_ffn(qw6_model_t *m) {
    QW6_ASSERT_PTR(m);
    qw6_tensor_t *gate_t = &m->layers[0].shared_gate;
    qw6_tensor_t *up_t = &m->layers[0].shared_up;
    qw6_tensor_t *down_t = &m->layers[0].shared_down;
    if (!gate_t->data || !up_t->data || !down_t->data) return -1;

    float *hidden = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    float *norm_w = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    float *normed = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    float *gate = calloc(QW6_SHARED_INTER, sizeof(float));
    float *up = calloc(QW6_SHARED_INTER, sizeof(float));
    float *mid = calloc(QW6_SHARED_INTER, sizeof(float));
    float *out = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    QW6_ASSERT_PTR(hidden); QW6_ASSERT_PTR(norm_w); QW6_ASSERT_PTR(normed);
    QW6_ASSERT_PTR(gate); QW6_ASSERT_PTR(up); QW6_ASSERT_PTR(mid); QW6_ASSERT_PTR(out);

    int rc = qw6_tensor_dequantize_row(&m->tok_embeddings, 0, hidden);
    if (rc == 0) rc = qw6_tensor_dequantize_row(&m->layers[0].norm, 0, norm_w);
    if (rc == 0) {
        qw6_cpu_rmsnorm(normed, hidden, norm_w, QW6_HIDDEN_SIZE);
        rc = qw6_tensor_matvec(gate, gate_t, normed, QW6_SHARED_INTER);
    }
    if (rc == 0) rc = qw6_tensor_matvec(up, up_t, normed, QW6_SHARED_INTER);
    if (rc == 0) {
        for (int i = 0; i < QW6_SHARED_INTER; i++) mid[i] = qw6_silu(gate[i]) * up[i];
        rc = qw6_tensor_matvec(out, down_t, mid, QW6_HIDDEN_SIZE);
    }
    if (rc == 0) {
        double sum = 0.0, abs_sum = 0.0, max_abs = 0.0;
        for (int i = 0; i < QW6_HIDDEN_SIZE; i++) {
            double v = out[i], av = fabs(v);
            sum += v; abs_sum += av; if (av > max_abs) max_abs = av;
        }
        fprintf(stderr,
                "qw6: probe layer0 shared ffn sum=%.6f abs=%.6f max=%.6f first=%.6f\n",
                sum, abs_sum, max_abs, out[0]);
    }

    free(hidden); free(norm_w); free(normed); free(gate);
    free(up); free(mid); free(out);
    return rc;
}

static int qw6_probe_layer0_routed_ffn(qw6_model_t *m) {
    QW6_ASSERT_PTR(m);
    float *hidden = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    float *norm_w = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    float *normed = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    float *router_logits = calloc(QW6_NUM_EXPERTS, sizeof(float));
    float *gate = calloc(QW6_MOE_INTER, sizeof(float));
    float *up = calloc(QW6_MOE_INTER, sizeof(float));
    float *mid = calloc(QW6_MOE_INTER, sizeof(float));
    float *tmp = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    float *out = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    int idx[QW6_EXPERTS_PER_TOK] = {0};
    float w[QW6_EXPERTS_PER_TOK] = {0};
    QW6_ASSERT_PTR(hidden); QW6_ASSERT_PTR(norm_w); QW6_ASSERT_PTR(normed);
    QW6_ASSERT_PTR(router_logits); QW6_ASSERT_PTR(gate); QW6_ASSERT_PTR(up);
    QW6_ASSERT_PTR(mid); QW6_ASSERT_PTR(tmp); QW6_ASSERT_PTR(out);

    int rc = qw6_tensor_dequantize_row(&m->tok_embeddings, 0, hidden);
    if (rc == 0) rc = qw6_tensor_dequantize_row(&m->layers[0].norm, 0, norm_w);
    if (rc == 0) {
        qw6_cpu_rmsnorm(normed, hidden, norm_w, QW6_HIDDEN_SIZE);
        rc = qw6_tensor_matvec(router_logits, &m->layers[0].moe_router,
                               normed, QW6_NUM_EXPERTS);
    }
    if (rc == 0) {
        qw6_cpu_moe_route(idx, w, router_logits,
                          QW6_NUM_EXPERTS, QW6_EXPERTS_PER_TOK);
        for (int e = 0; e < QW6_EXPERTS_PER_TOK && rc == 0; e++) {
            const int expert = idx[e];
            rc = qw6_tensor_matvec(gate, &m->layers[0].expert_gate[expert],
                                   normed, QW6_MOE_INTER);
            if (rc == 0) rc = qw6_tensor_matvec(up, &m->layers[0].expert_up[expert],
                                                normed, QW6_MOE_INTER);
            if (rc == 0) {
                for (int i = 0; i < QW6_MOE_INTER; i++) mid[i] = qw6_silu(gate[i]) * up[i];
                rc = qw6_tensor_matvec(tmp, &m->layers[0].expert_down[expert],
                                       mid, QW6_HIDDEN_SIZE);
            }
            if (rc == 0)
                for (int i = 0; i < QW6_HIDDEN_SIZE; i++) out[i] += w[e] * tmp[i];
        }
    }

    if (rc == 0) {
        double sum = 0.0, abs_sum = 0.0, max_abs = 0.0;
        for (int i = 0; i < QW6_HIDDEN_SIZE; i++) {
            double v = out[i], av = fabs(v);
            sum += v; abs_sum += av; if (av > max_abs) max_abs = av;
        }
        fprintf(stderr,
                "qw6: probe layer0 routed ffn sum=%.6f abs=%.6f max=%.6f first=%.6f\n",
                sum, abs_sum, max_abs, out[0]);
    }

    free(hidden); free(norm_w); free(normed); free(router_logits);
    free(gate); free(up); free(mid); free(tmp); free(out);
    return rc;
}

static float qw6_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static float qw6_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

static void qw6_l2_norm_heads(float *x, int heads, int dim) {
    for (int h = 0; h < heads; h++) {
        float ss = 0.0f;
        float *row = x + h * dim;
        for (int i = 0; i < dim; i++) ss += row[i] * row[i];
        float scale = 1.0f / sqrtf(ss + QW6_RMS_EPS);
        for (int i = 0; i < dim; i++) row[i] *= scale;
    }
}

static int qw6_ssm_conv1d_single(float *out, const qw6_tensor_t *conv,
                                 const float *current) {
    QW6_ASSERT_PTR(out); QW6_ASSERT_PTR(conv); QW6_ASSERT_PTR(current);
    if (!conv->data || conv->quant != QW6_Q_FP32 ||
        conv->rows != QW6_LINEAR_QKV_DIM || conv->cols != QW6_CONV1D_KERNEL)
        return -1;

    const float *w = (const float *)conv->data;
    for (int c = 0; c < QW6_LINEAR_QKV_DIM; c++) {
        out[c] = current[c] * w[c * QW6_CONV1D_KERNEL + (QW6_CONV1D_KERNEL - 1)];
    }
    return 0;
}

static void qw6_gated_delta_net_single(float *out, float *state,
                                       const float *q16, const float *k16,
                                       const float *v, const float *gate,
                                       const float *beta) {
    const float scale = 1.0f / sqrtf((float)QW6_VALUE_HEAD_DIM);
    for (int vh = 0; vh < QW6_NUM_VALUE_HEADS; vh++) {
        const int kh = vh % QW6_NUM_KEY_HEADS;
        const float *q = q16 + kh * QW6_KEY_HEAD_DIM;
        const float *k = k16 + kh * QW6_KEY_HEAD_DIM;
        const float *vv = v + vh * QW6_VALUE_HEAD_DIM;
        float *oo = out + vh * QW6_VALUE_HEAD_DIM;
        float *s = state + (size_t)vh * QW6_VALUE_HEAD_DIM * QW6_VALUE_HEAD_DIM;
        float decay = expf(gate[vh]);
        float b = beta[vh];

        for (int j = 0; j < QW6_VALUE_HEAD_DIM; j++) {
            float *row = s + j * QW6_VALUE_HEAD_DIM;
            for (int i = 0; i < QW6_VALUE_HEAD_DIM; i++) row[i] *= decay;

            float pred = 0.0f;
            for (int i = 0; i < QW6_VALUE_HEAD_DIM; i++) pred += row[i] * k[i];
            float delta = (vv[j] - pred) * b;
            for (int i = 0; i < QW6_VALUE_HEAD_DIM; i++) row[i] += delta * k[i];

            float sum = 0.0f;
            for (int i = 0; i < QW6_VALUE_HEAD_DIM; i++) sum += row[i] * q[i];
            oo[j] = sum * scale;
        }
    }
}

static int qw6_probe_layer0_deltanet_forward(qw6_model_t *m) {
    QW6_ASSERT_PTR(m);
    qw6_tensor_t *qkv_t = &m->layers[0].attn_q;
    qw6_tensor_t *z_t = &m->layers[0].attn_gate;
    qw6_tensor_t *alpha_t = &m->layers[0].dn_alpha;
    qw6_tensor_t *beta_t = &m->layers[0].dn_beta;
    qw6_tensor_t *out_t = &m->layers[0].dn_out;
    if (!qkv_t->data || !z_t->data || !alpha_t->data || !beta_t->data ||
        !m->layers[0].conv1d.data || !m->layers[0].dn_norm.data ||
        !m->layers[0].dn_dt.data || !m->layers[0].dn_a.data || !out_t->data)
        return -1;

    float *hidden = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    float *norm_w = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    float *normed = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    float *qkv = calloc(QW6_LINEAR_QKV_DIM, sizeof(float));
    float *conv = calloc(QW6_LINEAR_QKV_DIM, sizeof(float));
    float *z = calloc(QW6_LINEAR_VALUE_DIM, sizeof(float));
    float *beta = calloc(QW6_NUM_VALUE_HEADS, sizeof(float));
    float *alpha = calloc(QW6_NUM_VALUE_HEADS, sizeof(float));
    float *dt = calloc(QW6_NUM_VALUE_HEADS, sizeof(float));
    float *a = calloc(QW6_NUM_VALUE_HEADS, sizeof(float));
    float *dn_norm = calloc(QW6_VALUE_HEAD_DIM, sizeof(float));
    float *gdn = calloc(QW6_LINEAR_VALUE_DIM, sizeof(float));
    float *gated = calloc(QW6_LINEAR_VALUE_DIM, sizeof(float));
    float *state = calloc((size_t)QW6_NUM_VALUE_HEADS * QW6_VALUE_HEAD_DIM * QW6_VALUE_HEAD_DIM,
                          sizeof(float));
    float *out = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    QW6_ASSERT_PTR(hidden); QW6_ASSERT_PTR(norm_w); QW6_ASSERT_PTR(normed);
    QW6_ASSERT_PTR(qkv); QW6_ASSERT_PTR(conv); QW6_ASSERT_PTR(z);
    QW6_ASSERT_PTR(beta); QW6_ASSERT_PTR(alpha); QW6_ASSERT_PTR(dt); QW6_ASSERT_PTR(a);
    QW6_ASSERT_PTR(dn_norm); QW6_ASSERT_PTR(gdn); QW6_ASSERT_PTR(gated);
    QW6_ASSERT_PTR(state); QW6_ASSERT_PTR(out);

    int rc = qw6_tensor_dequantize_row(&m->tok_embeddings, 0, hidden);
    if (rc == 0) rc = qw6_tensor_dequantize_row(&m->layers[0].norm, 0, norm_w);
    if (rc == 0) {
        qw6_cpu_rmsnorm(normed, hidden, norm_w, QW6_HIDDEN_SIZE);
        rc = qw6_tensor_matvec(qkv, qkv_t, normed, QW6_LINEAR_QKV_DIM);
    }
    if (rc == 0) rc = qw6_tensor_matvec(z, z_t, normed, QW6_LINEAR_VALUE_DIM);
    if (rc == 0) rc = qw6_tensor_matvec(beta, beta_t, normed, QW6_NUM_VALUE_HEADS);
    if (rc == 0) rc = qw6_tensor_matvec(alpha, alpha_t, normed, QW6_NUM_VALUE_HEADS);
    if (rc == 0) rc = qw6_tensor_dequantize_row(&m->layers[0].dn_dt, 0, dt);
    if (rc == 0) rc = qw6_tensor_dequantize_row(&m->layers[0].dn_a, 0, a);
    if (rc == 0) rc = qw6_tensor_dequantize_row(&m->layers[0].dn_norm, 0, dn_norm);
    if (rc == 0) rc = qw6_ssm_conv1d_single(conv, &m->layers[0].conv1d, qkv);
    if (rc == 0) {
        for (int i = 0; i < QW6_LINEAR_QKV_DIM; i++) conv[i] = qw6_silu(conv[i]);
        float *q = conv;
        float *k = conv + QW6_NUM_KEY_HEADS * QW6_KEY_HEAD_DIM;
        float *v = conv + 2 * QW6_NUM_KEY_HEADS * QW6_KEY_HEAD_DIM;
        qw6_l2_norm_heads(q, QW6_NUM_KEY_HEADS, QW6_KEY_HEAD_DIM);
        qw6_l2_norm_heads(k, QW6_NUM_KEY_HEADS, QW6_KEY_HEAD_DIM);
        for (int h = 0; h < QW6_NUM_VALUE_HEADS; h++) {
            beta[h] = qw6_sigmoid(beta[h]);
            alpha[h] = qw6_softplus(alpha[h] + dt[h]) * a[h];
        }
        qw6_gated_delta_net_single(gdn, state, q, k, v, alpha, beta);
        for (int h = 0; h < QW6_NUM_VALUE_HEADS; h++) {
            float *dst = gated + h * QW6_VALUE_HEAD_DIM;
            qw6_cpu_rmsnorm(dst, gdn + h * QW6_VALUE_HEAD_DIM, dn_norm, QW6_VALUE_HEAD_DIM);
            for (int i = 0; i < QW6_VALUE_HEAD_DIM; i++)
                dst[i] *= qw6_silu(z[h * QW6_VALUE_HEAD_DIM + i]);
        }
        rc = qw6_tensor_matvec(out, out_t, gated, QW6_HIDDEN_SIZE);
    }
    if (rc == 0) {
        double sum = 0.0, abs_sum = 0.0, max_abs = 0.0;
        for (int i = 0; i < QW6_HIDDEN_SIZE; i++) {
            double v = out[i], av = fabs(v);
            sum += v; abs_sum += av; if (av > max_abs) max_abs = av;
        }
        fprintf(stderr,
                "qw6: probe layer0 deltanet forward sum=%.6f abs=%.6f max=%.6f first=%.6f\n",
                sum, abs_sum, max_abs, out[0]);
    }

    free(hidden); free(norm_w); free(normed); free(qkv); free(conv); free(z);
    free(beta); free(alpha); free(dt); free(a); free(dn_norm);
    free(gdn); free(gated); free(state); free(out);
    return rc;
}

static int qw6_probe_layer0_attn_qkv(qw6_model_t *m) {
    QW6_ASSERT_PTR(m);
    qw6_tensor_t *qkv_t = &m->layers[0].attn_q;
    if (!qkv_t->data) return -1;

    float *hidden = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    float *norm_w = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    float *normed = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    float qkv[8] = {0};
    QW6_ASSERT_PTR(hidden); QW6_ASSERT_PTR(norm_w); QW6_ASSERT_PTR(normed);

    int rc = qw6_tensor_dequantize_row(&m->tok_embeddings, 0, hidden);
    if (rc == 0) rc = qw6_tensor_dequantize_row(&m->layers[0].norm, 0, norm_w);
    if (rc == 0) {
        qw6_cpu_rmsnorm(normed, hidden, norm_w, QW6_HIDDEN_SIZE);
        rc = qw6_tensor_matvec(qkv, qkv_t, normed, 8);
    }
    if (rc == 0) {
        fprintf(stderr,
                "qw6: probe layer0 attn_qkv[0..7]=%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f\n",
                qkv[0], qkv[1], qkv[2], qkv[3],
                qkv[4], qkv[5], qkv[6], qkv[7]);
    }

    free(hidden); free(norm_w); free(normed);
    return rc;
}

static void qw6_add_inplace(float *dst, const float *src, int n) {
    for (int i = 0; i < n; i++) dst[i] += src[i];
}

static int qw6_apply_layer_moe(float *out, qw6_model_t *m, int layer,
                               const float *x) {
    QW6_ASSERT_PTR(out); QW6_ASSERT_PTR(m); QW6_ASSERT_PTR(x);
    const qw6_tensor_t *router_t = &m->layers[layer].moe_router;
    if (!router_t->data) return -1;

    float *router_logits = calloc(QW6_NUM_EXPERTS, sizeof(float));
    float *gate = calloc(QW6_MOE_INTER, sizeof(float));
    float *up = calloc(QW6_MOE_INTER, sizeof(float));
    float *mid = calloc(QW6_MOE_INTER, sizeof(float));
    float *tmp = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    float *sgate = calloc(QW6_SHARED_INTER, sizeof(float));
    float *sup = calloc(QW6_SHARED_INTER, sizeof(float));
    float *smid = calloc(QW6_SHARED_INTER, sizeof(float));
    int idx[QW6_EXPERTS_PER_TOK] = {0};
    float w[QW6_EXPERTS_PER_TOK] = {0};
    QW6_ASSERT_PTR(router_logits); QW6_ASSERT_PTR(gate); QW6_ASSERT_PTR(up);
    QW6_ASSERT_PTR(mid); QW6_ASSERT_PTR(tmp); QW6_ASSERT_PTR(sgate);
    QW6_ASSERT_PTR(sup); QW6_ASSERT_PTR(smid);

    memset(out, 0, QW6_HIDDEN_SIZE * sizeof(float));
    int rc = qw6_tensor_matvec(router_logits, router_t, x, QW6_NUM_EXPERTS);
    if (rc == 0) {
        qw6_cpu_moe_route(idx, w, router_logits,
                          QW6_NUM_EXPERTS, QW6_EXPERTS_PER_TOK);
        for (int e = 0; e < QW6_EXPERTS_PER_TOK && rc == 0; e++) {
            const int expert = idx[e];
            rc = qw6_tensor_matvec(gate, &m->layers[layer].expert_gate[expert],
                                   x, QW6_MOE_INTER);
            if (rc == 0) rc = qw6_tensor_matvec(up, &m->layers[layer].expert_up[expert],
                                                x, QW6_MOE_INTER);
            if (rc == 0) {
                for (int i = 0; i < QW6_MOE_INTER; i++) mid[i] = qw6_silu(gate[i]) * up[i];
                rc = qw6_tensor_matvec(tmp, &m->layers[layer].expert_down[expert],
                                       mid, QW6_HIDDEN_SIZE);
            }
            if (rc == 0)
                for (int i = 0; i < QW6_HIDDEN_SIZE; i++) out[i] += w[e] * tmp[i];
        }
    }

    if (rc == 0 && m->layers[layer].shared_gate.data &&
        m->layers[layer].shared_up.data && m->layers[layer].shared_down.data) {
        float shared_weight = 1.0f;
        if (m->layers[layer].shared_router.data) {
            rc = qw6_tensor_matvec(&shared_weight, &m->layers[layer].shared_router, x, 1);
            shared_weight = qw6_sigmoid(shared_weight);
        }
        if (rc == 0) rc = qw6_tensor_matvec(sgate, &m->layers[layer].shared_gate,
                                            x, QW6_SHARED_INTER);
        if (rc == 0) rc = qw6_tensor_matvec(sup, &m->layers[layer].shared_up,
                                            x, QW6_SHARED_INTER);
        if (rc == 0) {
            for (int i = 0; i < QW6_SHARED_INTER; i++) smid[i] = qw6_silu(sgate[i]) * sup[i];
            rc = qw6_tensor_matvec(tmp, &m->layers[layer].shared_down,
                                   smid, QW6_HIDDEN_SIZE);
        }
        if (rc == 0)
            for (int i = 0; i < QW6_HIDDEN_SIZE; i++) out[i] += shared_weight * tmp[i];
    }

    free(router_logits); free(gate); free(up); free(mid); free(tmp);
    free(sgate); free(sup); free(smid);
    return rc;
}

static int qw6_apply_linear_attn(float *out, qw6_session_t *s, int layer,
                                 const float *x) {
    qw6_model_t *m = s->model;
    float *qkv = calloc(QW6_LINEAR_QKV_DIM, sizeof(float));
    float *conv = calloc(QW6_LINEAR_QKV_DIM, sizeof(float));
    float *z = calloc(QW6_LINEAR_VALUE_DIM, sizeof(float));
    float *beta = calloc(QW6_NUM_VALUE_HEADS, sizeof(float));
    float *alpha = calloc(QW6_NUM_VALUE_HEADS, sizeof(float));
    float *dt = calloc(QW6_NUM_VALUE_HEADS, sizeof(float));
    float *a = calloc(QW6_NUM_VALUE_HEADS, sizeof(float));
    float *dn_norm = calloc(QW6_VALUE_HEAD_DIM, sizeof(float));
    float *gdn = calloc(QW6_LINEAR_VALUE_DIM, sizeof(float));
    float *gated = calloc(QW6_LINEAR_VALUE_DIM, sizeof(float));
    QW6_ASSERT_PTR(qkv); QW6_ASSERT_PTR(conv); QW6_ASSERT_PTR(z);
    QW6_ASSERT_PTR(beta); QW6_ASSERT_PTR(alpha); QW6_ASSERT_PTR(dt);
    QW6_ASSERT_PTR(a); QW6_ASSERT_PTR(dn_norm); QW6_ASSERT_PTR(gdn);
    QW6_ASSERT_PTR(gated);

    int rc = qw6_tensor_matvec(qkv, &m->layers[layer].attn_q, x, QW6_LINEAR_QKV_DIM);
    if (rc == 0) rc = qw6_tensor_matvec(z, &m->layers[layer].attn_gate,
                                        x, QW6_LINEAR_VALUE_DIM);
    if (rc == 0) rc = qw6_tensor_matvec(beta, &m->layers[layer].dn_beta,
                                        x, QW6_NUM_VALUE_HEADS);
    if (rc == 0) rc = qw6_tensor_matvec(alpha, &m->layers[layer].dn_alpha,
                                        x, QW6_NUM_VALUE_HEADS);
    if (rc == 0) rc = qw6_tensor_dequantize_row(&m->layers[layer].dn_dt, 0, dt);
    if (rc == 0) rc = qw6_tensor_dequantize_row(&m->layers[layer].dn_a, 0, a);
    if (rc == 0) rc = qw6_tensor_dequantize_row(&m->layers[layer].dn_norm, 0, dn_norm);
    if (rc == 0) {
        float *cs = s->conv_state[layer];
        memmove(cs, cs + QW6_LINEAR_QKV_DIM,
                (size_t)(QW6_CONV1D_KERNEL - 1) * QW6_LINEAR_QKV_DIM * sizeof(float));
        memcpy(cs + (size_t)(QW6_CONV1D_KERNEL - 1) * QW6_LINEAR_QKV_DIM,
               qkv, QW6_LINEAR_QKV_DIM * sizeof(float));

        const float *cw = (const float *)m->layers[layer].conv1d.data;
        for (int c = 0; c < QW6_LINEAR_QKV_DIM; c++) {
            float sum = 0.0f;
            for (int k = 0; k < QW6_CONV1D_KERNEL; k++)
                sum += cs[(size_t)k * QW6_LINEAR_QKV_DIM + c] *
                       cw[(size_t)c * QW6_CONV1D_KERNEL + k];
            conv[c] = qw6_silu(sum);
        }

        float *q = conv;
        float *k = conv + QW6_NUM_KEY_HEADS * QW6_KEY_HEAD_DIM;
        float *v = conv + 2 * QW6_NUM_KEY_HEADS * QW6_KEY_HEAD_DIM;
        qw6_l2_norm_heads(q, QW6_NUM_KEY_HEADS, QW6_KEY_HEAD_DIM);
        qw6_l2_norm_heads(k, QW6_NUM_KEY_HEADS, QW6_KEY_HEAD_DIM);
        for (int h = 0; h < QW6_NUM_VALUE_HEADS; h++) {
            beta[h] = qw6_sigmoid(beta[h]);
            alpha[h] = qw6_softplus(alpha[h] + dt[h]) * a[h];
        }
        qw6_gated_delta_net_single(gdn, m->deltanet_state[layer],
                                   q, k, v, alpha, beta);
        for (int h = 0; h < QW6_NUM_VALUE_HEADS; h++) {
            float *dst = gated + h * QW6_VALUE_HEAD_DIM;
            qw6_cpu_rmsnorm(dst, gdn + h * QW6_VALUE_HEAD_DIM,
                            dn_norm, QW6_VALUE_HEAD_DIM);
            for (int i = 0; i < QW6_VALUE_HEAD_DIM; i++)
                dst[i] *= qw6_silu(z[h * QW6_VALUE_HEAD_DIM + i]);
        }
        rc = qw6_tensor_matvec(out, &m->layers[layer].dn_out, gated, QW6_HIDDEN_SIZE);
    }

    free(qkv); free(conv); free(z); free(beta); free(alpha); free(dt);
    free(a); free(dn_norm); free(gdn); free(gated);
    return rc;
}

static void qw6_rmsnorm_heads_weighted(float *x, const float *w,
                                       int heads, int dim) {
    for (int h = 0; h < heads; h++)
        qw6_cpu_rmsnorm(x + h * dim, x + h * dim, w, dim);
}

static int qw6_apply_full_attn(float *out, qw6_session_t *s, int layer,
                               const float *x, uint32_t pos) {
    qw6_model_t *m = s->model;
    float *qfull = calloc(QW6_NUM_Q_HEADS * QW6_HEAD_DIM * 2, sizeof(float));
    float *q = calloc(QW6_NUM_Q_HEADS * QW6_HEAD_DIM, sizeof(float));
    float *gate = calloc(QW6_NUM_Q_HEADS * QW6_HEAD_DIM, sizeof(float));
    float *k = calloc(QW6_NUM_KV_HEADS * QW6_HEAD_DIM, sizeof(float));
    float *v = calloc(QW6_NUM_KV_HEADS * QW6_HEAD_DIM, sizeof(float));
    float *qw = calloc(QW6_HEAD_DIM, sizeof(float));
    float *kw = calloc(QW6_HEAD_DIM, sizeof(float));
    float *attn = calloc(QW6_NUM_Q_HEADS * QW6_HEAD_DIM, sizeof(float));
    QW6_ASSERT_PTR(qfull); QW6_ASSERT_PTR(q); QW6_ASSERT_PTR(gate);
    QW6_ASSERT_PTR(k); QW6_ASSERT_PTR(v); QW6_ASSERT_PTR(qw);
    QW6_ASSERT_PTR(kw); QW6_ASSERT_PTR(attn);

    int rc = qw6_tensor_matvec(qfull, &m->layers[layer].attn_q, x,
                               QW6_NUM_Q_HEADS * QW6_HEAD_DIM * 2);
    if (rc == 0) rc = qw6_tensor_matvec(k, &m->layers[layer].attn_k, x,
                                        QW6_NUM_KV_HEADS * QW6_HEAD_DIM);
    if (rc == 0) rc = qw6_tensor_matvec(v, &m->layers[layer].attn_v, x,
                                        QW6_NUM_KV_HEADS * QW6_HEAD_DIM);
    if (rc == 0) rc = qw6_tensor_dequantize_row(&m->layers[layer].attn_q_norm, 0, qw);
    if (rc == 0) rc = qw6_tensor_dequantize_row(&m->layers[layer].attn_k_norm, 0, kw);
    if (rc == 0) {
        for (int h = 0; h < QW6_NUM_Q_HEADS; h++) {
            memcpy(q + h * QW6_HEAD_DIM,
                   qfull + h * QW6_HEAD_DIM * 2,
                   QW6_HEAD_DIM * sizeof(float));
            memcpy(gate + h * QW6_HEAD_DIM,
                   qfull + h * QW6_HEAD_DIM * 2 + QW6_HEAD_DIM,
                   QW6_HEAD_DIM * sizeof(float));
        }
        qw6_rmsnorm_heads_weighted(q, qw, QW6_NUM_Q_HEADS, QW6_HEAD_DIM);
        qw6_rmsnorm_heads_weighted(k, kw, QW6_NUM_KV_HEADS, QW6_HEAD_DIM);
        qw6_cpu_mrope(q, k, QW6_NUM_Q_HEADS * QW6_HEAD_DIM,
                      QW6_NUM_KV_HEADS * QW6_HEAD_DIM,
                      QW6_NUM_Q_HEADS, QW6_NUM_KV_HEADS,
                      pos, (int)(QW6_HEAD_DIM * QW6_PARTIAL_ROTARY));

        float *k_slot = m->k_cache[layer] +
            (size_t)pos * QW6_NUM_KV_HEADS * QW6_HEAD_DIM;
        float *v_slot = m->v_cache[layer] +
            (size_t)pos * QW6_NUM_KV_HEADS * QW6_HEAD_DIM;
        memcpy(k_slot, k, QW6_NUM_KV_HEADS * QW6_HEAD_DIM * sizeof(float));
        memcpy(v_slot, v, QW6_NUM_KV_HEADS * QW6_HEAD_DIM * sizeof(float));

        qw6_cpu_attention_gqa(attn, q, m->k_cache[layer], m->v_cache[layer],
                              (int)pos + 1, QW6_NUM_Q_HEADS, QW6_NUM_KV_HEADS,
                              QW6_HEAD_DIM);
        for (int i = 0; i < QW6_NUM_Q_HEADS * QW6_HEAD_DIM; i++)
            attn[i] *= qw6_sigmoid(gate[i]);
        rc = qw6_tensor_matvec(out, &m->layers[layer].attn_o, attn, QW6_HIDDEN_SIZE);
    }

    free(qfull); free(q); free(gate); free(k); free(v); free(qw); free(kw); free(attn);
    return rc;
}

static int qw6_forward_token(qw6_session_t *s, uint32_t token, uint32_t pos) {
    qw6_model_t *m = s->model;
    float *hidden = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    float *resid = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    float *norm_w = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    float *normed = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    float *attn = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    float *ffn = calloc(QW6_HIDDEN_SIZE, sizeof(float));
    QW6_ASSERT_PTR(hidden); QW6_ASSERT_PTR(resid); QW6_ASSERT_PTR(norm_w);
    QW6_ASSERT_PTR(normed); QW6_ASSERT_PTR(attn); QW6_ASSERT_PTR(ffn);

    int rc = qw6_tensor_dequantize_row(&m->tok_embeddings, token, hidden);
    for (int l = 0; l < QW6_NUM_LAYERS && rc == 0; l++) {
        memcpy(resid, hidden, QW6_HIDDEN_SIZE * sizeof(float));
        rc = qw6_tensor_dequantize_row(&m->layers[l].norm, 0, norm_w);
        if (rc == 0) qw6_cpu_rmsnorm(normed, hidden, norm_w, QW6_HIDDEN_SIZE);
        if (rc == 0 && qw6_layer_type(l) == QW6_LAYER_LINEAR_ATTN)
            rc = qw6_apply_linear_attn(attn, s, l, normed);
        else if (rc == 0)
            rc = qw6_apply_full_attn(attn, s, l, normed, pos);
        if (rc == 0) {
            memcpy(hidden, resid, QW6_HIDDEN_SIZE * sizeof(float));
            qw6_add_inplace(hidden, attn, QW6_HIDDEN_SIZE);
        }

        memcpy(resid, hidden, QW6_HIDDEN_SIZE * sizeof(float));
        if (rc == 0) rc = qw6_tensor_dequantize_row(&m->layers[l].post_norm, 0, norm_w);
        if (rc == 0) {
            qw6_cpu_rmsnorm(normed, hidden, norm_w, QW6_HIDDEN_SIZE);
            rc = qw6_apply_layer_moe(ffn, m, l, normed);
        }
        if (rc == 0) {
            memcpy(hidden, resid, QW6_HIDDEN_SIZE * sizeof(float));
            qw6_add_inplace(hidden, ffn, QW6_HIDDEN_SIZE);
        }
    }
    if (rc == 0) rc = qw6_tensor_dequantize_row(&m->output_norm, 0, norm_w);
    if (rc == 0) {
        qw6_cpu_rmsnorm(normed, hidden, norm_w, QW6_HIDDEN_SIZE);
        rc = qw6_tensor_matvec(s->logits, &m->output, normed, QW6_VOCAB_SIZE);
    }

    free(hidden); free(resid); free(norm_w); free(normed); free(attn); free(ffn);
    return rc;
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
                              const float *value, const float *query,
                              const float *beta,
                              int key_heads, int key_dim,
                              int val_heads, int val_dim) {
    QW6_ASSERT_PTR(state); QW6_ASSERT_PTR(key);
    QW6_ASSERT_PTR(value); QW6_ASSERT_PTR(query);
    QW6_ASSERT(key_heads > 0 && key_dim > 0, "key dims > 0");
    QW6_ASSERT(val_heads > 0 && val_dim > 0, "val dims > 0");

    /* Gated DeltaNet update (O(1) per token):
     *   dot   = key[head] · query[head]
     *   gate  = 1 / (1 + |beta[head] * dot|)
     *   delta = value * gate
     *   state += outer(key, delta)
     * If beta is NULL, falls back to uniform beta=1.0 per head. */
    for (int kh = 0; kh < key_heads; kh++) {
        const float *k = key + kh * key_dim;
        const float *q = query + kh * key_dim;
        float b = beta ? beta[kh] : 1.0f;

        /* dot(key, query) */
        float dot = 0.0f;
        for (int kd = 0; kd < key_dim; kd++)
            dot += k[kd] * q[kd];

        /* gating */
        float gate = 1.0f / (1.0f + fabsf(b * dot));

        for (int vh = 0; vh < val_heads; vh++) {
            const float *v = value + vh * val_dim;
            float *s = state + (size_t)(kh * key_dim * val_heads + vh) * val_dim;

            for (int kd = 0; kd < key_dim; kd++) {
                float key_contrib = k[kd];
                for (int vd = 0; vd < val_dim; vd++) {
                    s[kd * val_dim + vd] += key_contrib * (v[vd] * gate);
                }
            }
        }
    }
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

/* ---- Gated Attention / GQA ---- */

void qw6_cpu_attention_gqa(float *out, const float *q,
                           const float *k_cache, const float *v_cache,
                           int seq_len, int n_q_heads, int n_kv_heads,
                           int head_dim) {
    QW6_ASSERT_PTR(out); QW6_ASSERT_PTR(q);
    QW6_ASSERT_PTR(k_cache); QW6_ASSERT_PTR(v_cache);
    QW6_ASSERT(seq_len > 0 && head_dim > 0, "attention dims > 0");
    QW6_ASSERT(n_q_heads > 0 && n_kv_heads > 0, "attention heads > 0");
    QW6_ASSERT(n_q_heads % n_kv_heads == 0, "GQA ratio integral");

    int group = n_q_heads / n_kv_heads;
    float *scores = malloc((size_t)seq_len * sizeof(float));
    QW6_ASSERT_PTR(scores);

    for (int h = 0; h < n_q_heads; h++) {
        int kv_h = h / group;
        const float *qh = q + (size_t)h * head_dim;
        float max_score = -INFINITY;

        for (int t = 0; t < seq_len; t++) {
            const float *kh = k_cache + ((size_t)t * n_kv_heads + kv_h) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += qh[d] * kh[d];
            scores[t] = dot / sqrtf((float)head_dim);
            if (scores[t] > max_score) max_score = scores[t];
        }

        float sum = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum += scores[t];
        }
        for (int d = 0; d < head_dim; d++) out[(size_t)h * head_dim + d] = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            const float w = scores[t] / sum;
            const float *vh = v_cache + ((size_t)t * n_kv_heads + kv_h) * head_dim;
            for (int d = 0; d < head_dim; d++)
                out[(size_t)h * head_dim + d] += w * vh[d];
        }
    }

    free(scores);
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

/* ---- MTP Speculative Decoding ---- */

void qw6_cpu_mtp_draft(float *logits, const float *hidden,
                       const float *mtp_weight, int vocab_size, int hidden_size) {
    QW6_ASSERT_PTR(logits); QW6_ASSERT_PTR(hidden); QW6_ASSERT_PTR(mtp_weight);
    QW6_ASSERT(vocab_size > 0 && hidden_size > 0, "dims > 0");

    /* CPU reference MTP draft: single-layer linear projection
     * hidden -> logits via matmul. This matches Qwen3.6's MTP head. */
    for (int v = 0; v < vocab_size; v++) {
        const float *row = mtp_weight + (size_t)v * hidden_size;
        float acc = 0.0f;
        for (int h = 0; h < hidden_size; h++)
            acc += row[h] * hidden[h];
        logits[v] = acc;
    }
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
    if (s->n_tokens + n > s->capacity) {
        fprintf(stderr, "qw6: prompt exceeds context capacity\n");
        return -1;
    }
    for (uint32_t i = 0; i < n; i++) {
        uint32_t pos = s->n_tokens;
        s->tokens[s->n_tokens++] = tokens[i];
        if (qw6_forward_token(s, tokens[i], pos) != 0) {
            fprintf(stderr, "qw6: native forward failed at prompt token %u\n", i);
            return -1;
        }
    }
    return 0;
}

int qw6_generate(qw6_session_t *s, uint32_t n_tokens, float temp, float top_p) {
    QW6_ASSERT_PTR(s); QW6_ASSERT(n_tokens > 0, "n_tokens > 0");
    int generated = 0;
    for (uint32_t i = 0; i < n_tokens; i++) {
        if (s->n_tokens >= s->capacity) break;
        uint32_t tok = qw6_sample(s, temp, top_p);
        fprintf(stdout, "%u%s", tok, i + 1 == n_tokens ? "\n" : " ");
        fflush(stdout);
        s->tokens[s->n_tokens++] = tok;
        generated++;
        if (tok == QW6_EOS_TOKEN_ID) break;
        if (qw6_forward_token(s, tok, s->n_tokens - 1) != 0) {
            fprintf(stderr, "qw6: native forward failed at generated token %u\n", i);
            return generated > 0 ? generated : -1;
        }
    }
    return generated;
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
    const float query[2] = {1.0f, 2.0f};
    float out[2] = {0};

    /* key=[2,-1], query=[1,2] -> dot=2*1+(-1)*2 = 0 ...noch ein
     * gating(beta=NULL -> b=1.0) -> gate = 1.0, no suppression
     * state = outer(key, value) = [[1,6],[-0.5,-3]]
     * retrieve: out[0]=1*1+(-0.5)*2=0, out[1]=6*1+(-3)*2=0 */
    qw6_cpu_deltanet_update(state, key, value, query, NULL, 1, 2, 1, 2);
    qw6_cpu_deltanet_retrieve(out, state, query, 1, 2, 1, 2);

    int fail = 0;
    fail += selftest_close(out[0], 0.0f, 1e-6f, "deltanet[0]");
    fail += selftest_close(out[1], 0.0f, 1e-6f, "deltanet[1]");
    if (fail) fprintf(stderr, "self-test: deltanet failed\n");
    return fail;
}

static int selftest_mrope(void) {
    float q[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float k[2] = {1.0f, 0.0f};

    /* pos=0: identity rotation — everything stays the same */
    qw6_cpu_mrope(q, k, 4, 2, 2, 1, 0, 2);
    int fail = 0;
    fail += selftest_close(q[0], 1.0f, 1e-6f, "mrope_pos0_q0");
    fail += selftest_close(q[3], 1.0f, 1e-6f, "mrope_pos0_q3");
    fail += selftest_close(k[0], 1.0f, 1e-6f, "mrope_pos0_k0");

    /* pos=1: TAIL rotation (q_from=2 because head_dim=4, rotary=2,
     * partial_rotary puts rotary section at the tail).
     * q[2,3] rotate with pos=1: freq for i=0 is 1/(1e7^0) ≈ 1, but theta
     * is 1/theta_base here, qh[2] rotates by cos(1); k[0,1] rotate. */
    q[0] = 0.0f; q[1] = 0.0f; q[2] = 0.0f; q[3] = 1.0f;
    k[0] = 1.0f; k[1] = 0.0f;
    qw6_cpu_mrope(q, k, 4, 2, 2, 1, 1, 2);
    fail += selftest_close(q[0], 0.0f, 1e-6f, "mrope_pos1_q0");
    fail += selftest_close(q[1], 0.0f, 1e-6f, "mrope_pos1_q1");
    fail += selftest_close(k[0], cosf(1.0f), 1e-6f, "mrope_pos1_k0");
    fail += selftest_close(k[1], sinf(1.0f), 1e-6f, "mrope_pos1_k1");
    if (fail) fprintf(stderr, "self-test: mrope failed\n");
    return fail;
}

static int selftest_attention_gqa(void) {
    const float q[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float k[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float v[4] = {2.0f, 4.0f, 6.0f, 8.0f};
    float out[4] = {0};

    qw6_cpu_attention_gqa(out, q, k, v, 2, 2, 1, 2);
    float a = expf(1.0f / sqrtf(2.0f));
    float w0 = a / (a + 1.0f);
    float w1 = 1.0f / (a + 1.0f);

    int fail = 0;
    fail += selftest_close(out[0], w0 * 2.0f + w1 * 6.0f, 1e-6f, "gqa_h0_0");
    fail += selftest_close(out[1], w0 * 4.0f + w1 * 8.0f, 1e-6f, "gqa_h0_1");
    fail += selftest_close(out[2], w1 * 2.0f + w0 * 6.0f, 1e-6f, "gqa_h1_0");
    fail += selftest_close(out[3], w1 * 4.0f + w0 * 8.0f, 1e-6f, "gqa_h1_1");
    if (fail) fprintf(stderr, "self-test: attention GQA failed\n");
    return fail;
}

static int selftest_mtp_draft(void) {
    const float hidden[3] = {1.0f, -2.0f, 0.5f};
    const float w[6] = {1.0f, 0.0f, 2.0f, -1.0f, 0.5f, 1.0f};
    float logits[2] = {0};
    qw6_cpu_mtp_draft(logits, hidden, w, 2, 3);

    int fail = 0;
    fail += selftest_close(logits[0], 2.0f, 1e-6f, "mtp[0]");
    fail += selftest_close(logits[1], -1.5f, 1e-6f, "mtp[1]");
    if (fail) fprintf(stderr, "self-test: mtp draft failed\n");
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
    fail += selftest_attention_gqa();
    fail += selftest_mtp_draft();
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
        "  --load-only     Load and validate model metadata, then exit\n"
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
    bool load_only = false;

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
        else if (strcmp(argv[i], "--load-only") == 0) load_only = true;
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

    if (!prompt && !bench && !load_only) {
        fprintf(stderr, "qw6: -p <prompt> required\n");
        return 1;
    }

    fprintf(stderr, "qw6 %s -- Phase %d (CPU reference)\n", QW6_VERSION, QW6_BUILD_PHASE);
    fprintf(stderr, "Model: Qwen 3.6-35B-A3B (35B total, 3B active, 256 experts)\n");
    fprintf(stderr, "Architecture: hybrid attention (30 Gated DeltaNet + 10 Gated Attention)\n\n");

    if (load_only) {
        if (!model_path) {
            fprintf(stderr, "qw6: -m <model.gguf> required for --load-only\n");
            return 1;
        }
        qw6_model_t model;
        if (qw6_model_load(&model, model_path) != 0) {
            fprintf(stderr, "qw6: model metadata validation failed\n");
            return 1;
        }
        if (qw6_model_probe_dequant(&model) != 0) {
            fprintf(stderr, "qw6: model tensor dequant probe failed\n");
            qw6_model_free(&model);
            return 1;
        }
        if (qw6_probe_matvec(&model) != 0) {
            fprintf(stderr, "qw6: native matvec probe failed\n");
            qw6_model_free(&model);
            return 1;
        }
        if (qw6_probe_layer0_router(&model) != 0) {
            fprintf(stderr, "qw6: native layer0 router probe failed\n");
            qw6_model_free(&model);
            return 1;
        }
        if (qw6_probe_layer0_shared_ffn(&model) != 0) {
            fprintf(stderr, "qw6: native layer0 shared FFN probe failed\n");
            qw6_model_free(&model);
            return 1;
        }
        if (qw6_probe_layer0_routed_ffn(&model) != 0) {
            fprintf(stderr, "qw6: native layer0 routed FFN probe failed\n");
            qw6_model_free(&model);
            return 1;
        }
        if (qw6_probe_layer0_deltanet_forward(&model) != 0) {
            fprintf(stderr, "qw6: native layer0 DeltaNet forward probe failed\n");
            qw6_model_free(&model);
            return 1;
        }
        if (qw6_probe_layer0_attn_qkv(&model) != 0) {
            fprintf(stderr, "qw6: native layer0 attn_qkv probe failed\n");
            qw6_model_free(&model);
            return 1;
        }
        fprintf(stderr, "qw6: model metadata validation passed\n");
        qw6_model_free(&model);
        return 0;
    }

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
        fprintf(stderr, "qw6: model loading failed\n");
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
            fprintf(stderr, "qw6: native prefill failed\n");
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
        if (gen > 0) {
            char *decoded = NULL;
            if (qw6_tok_decode(&tokenizer, session.tokens + n, (uint32_t)gen, &decoded) == 0) {
                fprintf(stderr, "qw6: generated text: \"%s\"\n", decoded);
                free(decoded);
            }
        }
        fprintf(stderr, "qw6: %d tokens in %.2fs\n", gen, elapsed);
        free(tokens);
    }

    if (bench) fprintf(stderr, "qw6: benchmarks not yet implemented\n");

    qw6_session_free(&session);
    qw6_model_free(&model);
    qw6_tok_free(&tokenizer);
    return 0;
}
