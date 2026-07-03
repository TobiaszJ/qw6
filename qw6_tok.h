/* qw6_tok.h — BPE Tokenizer for Qwen 3.6-35B-A3B
 *
 * ByteLevel BPE tokenizer (Qwen2Tokenizer compatible).
 * Vocab: 248,044 BPE tokens + 26 added (special) tokens = 248,320 total.
 */

#ifndef QW6_TOK_H
#define QW6_TOK_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define QW6_TOK_VOCAB_SIZE     248320
#define QW6_TOK_BPE_VOCAB      248044
#define QW6_TOK_NUM_ADDED      26
#define QW6_TOK_MAX_MERGES     250000

/* Special token IDs */
#define QW6_TOK_BOS            248044
#define QW6_TOK_IM_START       248045
#define QW6_TOK_IM_END         248046
#define QW6_TOK_VISION_START   248053
#define QW6_TOK_VISION_END     248054
#define QW6_TOK_IMAGE_PAD      248056
#define QW6_TOK_VIDEO_PAD      248057

typedef struct {
    /* Heap-allocated to avoid stack overflow (248K pointers = ~2MB) */
    char **vocab;           /* [QW6_TOK_VOCAB_SIZE] */
    uint32_t vocab_count;

    struct {
        uint32_t a;
        uint32_t b;
        uint32_t result;
    } *merges;              /* [QW6_TOK_MAX_MERGES] */
    uint32_t merge_count;

    struct {
        char content[64];
        uint32_t id;
        bool special;
    } added[QW6_TOK_NUM_ADDED];
    uint32_t added_count;

    /* Byte-level: byte → unicode string */
    char byte_to_char[256][5];
} qw6_tokenizer_t;

/* API */
int qw6_tok_init(qw6_tokenizer_t *t, const char *tokenizer_json_path);
void qw6_tok_free(qw6_tokenizer_t *t);
int qw6_tok_encode(qw6_tokenizer_t *t, const char *text,
                   uint32_t **out_tokens, uint32_t *out_count);
int qw6_tok_decode(qw6_tokenizer_t *t, const uint32_t *tokens,
                   uint32_t count, char **out_text);
int qw6_tok_load_json(qw6_tokenizer_t *t, const char *path);
const char *qw6_tok_id_to_str(qw6_tokenizer_t *t, uint32_t id);

#endif /* QW6_TOK_H */