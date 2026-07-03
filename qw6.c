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
        default: return "?";
    }
}

/* Check if GGML type is one qw6 recognises */
static bool qw6_gguf_type_supported(ggml_type_t t) {
    switch (t) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
            return true;
        default:
            return false;
    }
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
        case GGML_TYPE_Q8_0:   return QW6_Q_Q8_0;
        case GGML_TYPE_Q4_K:   return QW6_Q_Q4_K_M;
        case GGML_TYPE_Q6_K:   return QW6_Q_Q4_K_M;  /* placeholder—add Q6_K to enum */
        case GGML_TYPE_IQ2_XXS: return QW6_Q_IQ2_XXS;
        default:                return QW6_Q_FP32;   /* safe fallback */
    }
}

/* Load GGUF metadata into qw6 model struct (tensor data NOT loaded yet) */
int qw6_gguf_read_file(const char *path, qw6_model_t *m) {
    QW6_ASSERT_PTR(path);
    QW6_ASSERT_PTR(m);

    gguf_ctx_t ctx;
    if (qw6_gguf_parse(path, &ctx) != 0) return -1;

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
    for (uint32_t i = 0; i < ctx.tensors_parsed; i++) {
        if (!qw6_gguf_type_supported(ctx.tensors[i].type)) {
            fprintf(stderr, "qw6: WARNING: tensor '%s' uses unsupported type %s\n",
                    ctx.tensors[i].name, ggml_type_name(ctx.tensors[i].type));
        }
    }

    /* For Phase 1: we only parse and report. Tensor data loading is next. */
    m->max_context = QW6_DEFAULT_CTX;
    m->total_weight_bytes = 0;

    /* Calculate total tensor data size */
    for (uint32_t i = 0; i < ctx.tensors_parsed; i++) {
        uint64_t elements = 1;
        for (uint32_t d = 0; d < ctx.tensors[i].n_dims; d++)
            elements *= (uint64_t)ctx.tensors[i].dims[d];

        /* Approximate bytes per element by type */
        size_t bpe = 0;
        switch (ctx.tensors[i].type) {
            case GGML_TYPE_F32:  bpe = 4; break;
            case GGML_TYPE_F16:  bpe = 2; break;
            case GGML_TYPE_Q8_0: bpe = 1; break;  /* 34 bytes per 32 elements */
            case GGML_TYPE_Q4_K: bpe = 1; break;  /* approx */
            case GGML_TYPE_Q6_K: bpe = 1; break;  /* approx */
            default: bpe = 1; break;
        }
        m->total_weight_bytes += (size_t)elements * bpe;
    }

    fprintf(stderr, "qw6: ~%zu MB tensor data (approximate)\n",
            m->total_weight_bytes / (1024 * 1024));

    qw6_gguf_free(&ctx);

    /* Phase 1: metadata parsed, tensor data loading still TODO */
    fprintf(stderr, "qw6: GGUF metadata parsed. Tensor data loading not yet implemented.\n");
    return -1;  /* return -1 until tensor data is actually loaded */
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
