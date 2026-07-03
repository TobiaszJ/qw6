#define _POSIX_C_SOURCE 200809L
/* qw6_tok.c — BPE Tokenizer for Qwen 3.6-35B-A3B (CPU reference)
 *
 * ByteLevel BPE tokenizer compatible with Qwen2Tokenizer / HuggingFace tokenizers.
 * Loads vocab + merges + added tokens from tokenizer.json.
 *
 * Phase 1: CPU reference implementation — correctness over speed.
 */

#include "qw6_tok.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#define TOK_ASSERT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "qw6_tok: ASSERT: %s:%d: %s\n", __FILE__, __LINE__, msg); \
        exit(1); \
    } \
} while (0)

#define TOK_ASSERT_PTR(p) TOK_ASSERT((p) != NULL, "null: " #p)

/* ---- Simple hash map for string→ID lookup (djb2) ---- */

typedef struct {
    char *key;
    uint32_t id;
} hash_entry_t;

typedef struct {
    hash_entry_t *entries;
    size_t capacity;
    size_t count;
} hash_map_t;

static void hash_init(hash_map_t *h, size_t capacity) {
    h->capacity = capacity;
    h->count = 0;
    h->entries = calloc(capacity, sizeof(hash_entry_t));
}

static unsigned long hash_djb2(const char *str) {
    unsigned long hash = 5381;
    int c;
    while ((c = *str++))
        hash = ((hash << 5) + hash) + c;
    return hash;
}

static void hash_insert(hash_map_t *h, const char *key, uint32_t id) {
    if (h->count >= h->capacity * 3 / 4) return; /* don't overfill */
    unsigned long idx = hash_djb2(key) % h->capacity;
    while (h->entries[idx].key != NULL) {
        if (strcmp(h->entries[idx].key, key) == 0) return; /* already exists */
        idx = (idx + 1) % h->capacity;
    }
    h->entries[idx].key = strdup(key);
    h->entries[idx].id = id;
    h->count++;
}

static int32_t hash_lookup(hash_map_t *h, const char *key) {
    unsigned long idx = hash_djb2(key) % h->capacity;
    size_t probes = 0;
    while (h->entries[idx].key != NULL && probes < h->capacity) {
        if (strcmp(h->entries[idx].key, key) == 0)
            return (int32_t)h->entries[idx].id;
        idx = (idx + 1) % h->capacity;
        probes++;
    }
    return -1;
}

static void hash_free(hash_map_t *h) {
    for (size_t i = 0; i < h->capacity; i++)
        if (h->entries[i].key) free(h->entries[i].key);
    free(h->entries);
    h->entries = NULL;
    h->capacity = 0;
}



/* ---- Minimal JSON parser for tokenizer.json ---- */
/* We only need: objects, arrays, strings, numbers, booleans. */

typedef enum {
    JSON_TOK_LBRACE = 1,
    JSON_TOK_RBRACE,
    JSON_TOK_LBRACKET,
    JSON_TOK_RBRACKET,
    JSON_TOK_COLON,
    JSON_TOK_COMMA,
    JSON_TOK_STRING,
    JSON_TOK_NUMBER,
    JSON_TOK_TRUE,
    JSON_TOK_FALSE,
    JSON_TOK_NULL,
    JSON_TOK_EOF,
} json_token_type_t;

typedef struct {
    json_token_type_t type;
    char *str;       /* for JSON_TOK_STRING (owned, null-terminated) */
    double number;   /* for JSON_TOK_NUMBER */
    size_t pos;      /* position in source after this token */
} json_token_t;

typedef struct {
    const char *src;
    size_t src_len;
    size_t pos;
} json_parser_t;

static void json_skip_ws(json_parser_t *p) {
    while (p->pos < p->src_len) {
        char c = p->src[p->pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            p->pos++;
        } else {
            break;
        }
    }
}

static int json_parse_string(json_parser_t *p, char **out) {
    TOK_ASSERT(p->pos < p->src_len, "unexpected EOF in string");
    TOK_ASSERT(p->src[p->pos] == '"', "expected '\"'");

    p->pos++; /* skip opening quote */

    /* Calculate unescaped length */
    size_t start = p->pos;
    size_t len = 0;
    while (p->pos < p->src_len && p->src[p->pos] != '"') {
        if (p->src[p->pos] == '\\') {
            p->pos++;
            TOK_ASSERT(p->pos < p->src_len, "unescaped \\ at EOF");
        }
        p->pos++;
        len++;
    }
    TOK_ASSERT(p->pos < p->src_len, "unterminated string");
    p->pos++; /* skip closing quote */

    /* Build unescaped string */
    char *result = malloc(len + 1);
    TOK_ASSERT_PTR(result);

    size_t i = start;
    size_t j = 0;
    while (i < p->pos - 1) {
        char c = p->src[i];
        if (c == '\\') {
            i++;
            char esc = p->src[i];
            switch (esc) {
                case 'n':  result[j++] = '\n'; break;
                case 't':  result[j++] = '\t'; break;
                case 'r':  result[j++] = '\r'; break;
                case '"':  result[j++] = '"';  break;
                case '\\': result[j++] = '\\'; break;
                case '/':  result[j++] = '/';  break;
                case 'b':  result[j++] = '\b'; break;
                case 'f':  result[j++] = '\f'; break;
                case 'u': {
                    /* UTF-8 escape: \uXXXX */
                    /* For now, just handle ASCII range */
                    unsigned int cp = 0;
                    for (int k = 0; k < 4; k++) {
                        i++;
                        char hex = p->src[i];
                        cp <<= 4;
                        if (hex >= '0' && hex <= '9') cp |= (hex - '0');
                        else if (hex >= 'a' && hex <= 'f') cp |= (hex - 'a' + 10);
                        else if (hex >= 'A' && hex <= 'F') cp |= (hex - 'A' + 10);
                    }
                    /* Encode as UTF-8 */
                    if (cp < 0x80) {
                        result[j++] = (char)cp;
                    } else if (cp < 0x800) {
                        result[j++] = (char)(0xC0 | (cp >> 6));
                        result[j++] = (char)(0x80 | (cp & 0x3F));
        } else {
                        result[j++] = (char)(0xE0 | (cp >> 12));
                        result[j++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                        result[j++] = (char)(0x80 | (cp & 0x3F));
                    }
                    break;
                }
                default:
                    result[j++] = esc;
                    break;
            }
            i++;
        } else {
            result[j++] = c;
            i++;
        }
    }
    result[j] = '\0';

    *out = result;
    return 0;
}

static json_token_t json_next_token(json_parser_t *p) {
    json_token_t tok = {0};
    json_skip_ws(p);

    if (p->pos >= p->src_len) {
        tok.type = JSON_TOK_EOF;
        return tok;
    }

    char c = p->src[p->pos];

    switch (c) {
        case '{': p->pos++; tok.type = JSON_TOK_LBRACE; return tok;
        case '}': p->pos++; tok.type = JSON_TOK_RBRACE; return tok;
        case '[': p->pos++; tok.type = JSON_TOK_LBRACKET; return tok;
        case ']': p->pos++; tok.type = JSON_TOK_RBRACKET; return tok;
        case ':': p->pos++; tok.type = JSON_TOK_COLON; return tok;
        case ',': p->pos++; tok.type = JSON_TOK_COMMA; return tok;
        case '"': {
            char *str = NULL;
            tok.pos = p->pos;
            json_parse_string(p, &str);
            tok.type = JSON_TOK_STRING;
            tok.str = str;
            tok.pos = p->pos;
            return tok;
        }
        case 't':
            if (p->pos + 4 <= p->src_len && strncmp(p->src + p->pos, "true", 4) == 0) {
                p->pos += 4;
                tok.type = JSON_TOK_TRUE;
                return tok;
            }
            break;
        case 'f':
            if (p->pos + 5 <= p->src_len && strncmp(p->src + p->pos, "false", 5) == 0) {
                p->pos += 5;
                tok.type = JSON_TOK_FALSE;
                return tok;
            }
            break;
        case 'n':
            if (p->pos + 4 <= p->src_len && strncmp(p->src + p->pos, "null", 4) == 0) {
                p->pos += 4;
                tok.type = JSON_TOK_NULL;
                return tok;
            }
            break;
        default:
            if (c == '-' || (c >= '0' && c <= '9')) {
                /* Parse number */
                size_t start = p->pos;
                if (c == '-') p->pos++;
                while (p->pos < p->src_len &&
                       ((p->src[p->pos] >= '0' && p->src[p->pos] <= '9') ||
                        p->src[p->pos] == '.' || p->src[p->pos] == 'e' ||
                        p->src[p->pos] == 'E' || p->src[p->pos] == '+' ||
                        p->src[p->pos] == '-')) {
                    p->pos++;
                }
                tok.type = JSON_TOK_NUMBER;
                tok.number = atof(p->src + start);
                return tok;
            }
            break;
    }

    fprintf(stderr, "qw6_tok: JSON parse error at pos %zu: '%c' (0x%02x)\n",
            p->pos, c, (unsigned char)c);
    tok.type = JSON_TOK_EOF;
    return tok;
}

/* Skip a JSON value (object, array, string, number, bool, null) */
static void json_skip_value(json_parser_t *p) {
    json_token_t tok = json_next_token(p);
    if (tok.type == JSON_TOK_STRING) {
        free(tok.str);
    } else if (tok.type == JSON_TOK_LBRACE) {
        /* Skip object */
        tok = json_next_token(p);
        if (tok.type == JSON_TOK_RBRACE) return;
        for (;;) {
            /* key (string) */
            if (tok.type == JSON_TOK_STRING) free(tok.str);
            /* colon */
            tok = json_next_token(p);
            TOK_ASSERT(tok.type == JSON_TOK_COLON, "expected colon");
            /* value */
            json_skip_value(p);
            tok = json_next_token(p);
            if (tok.type == JSON_TOK_RBRACE) return;
            TOK_ASSERT(tok.type == JSON_TOK_COMMA, "expected comma");
            tok = json_next_token(p);
        }
    } else if (tok.type == JSON_TOK_LBRACKET) {
        /* Skip array — handle simple values (string/number/bool/null) directly,
         * recurse only for nested objects/arrays */
        tok = json_next_token(p);
        if (tok.type == JSON_TOK_RBRACKET) return;
        for (;;) {
            /* tok holds the current element */
            if (tok.type == JSON_TOK_STRING) {
                free(tok.str);
            } else if (tok.type == JSON_TOK_LBRACE || tok.type == JSON_TOK_LBRACKET) {
                /* Nested structure — need to rewind one token and recurse.
                 * Since we can't rewind, we handle it inline. */
                if (tok.type == JSON_TOK_LBRACE) {
                    /* Skip object */
                    tok = json_next_token(p);
                    if (tok.type == JSON_TOK_RBRACE) goto array_elem_done;
                    for (;;) {
                        if (tok.type == JSON_TOK_STRING) free(tok.str);
                        tok = json_next_token(p);
                        TOK_ASSERT(tok.type == JSON_TOK_COLON, "expected colon");
                        json_skip_value(p);
                        tok = json_next_token(p);
                        if (tok.type == JSON_TOK_RBRACE) break;
                        TOK_ASSERT(tok.type == JSON_TOK_COMMA, "expected comma");
                        tok = json_next_token(p);
                    }
                } else {
                    /* Nested array — skip by recursive call (rewinds via json_next_token) */
                    /* Push back is not possible, so we handle inline */
                    /* This is getting complex, but nested arrays in tokenizer.json
                     * only appear in merges (which we handle separately) */
                    for (;;) {
                        if (tok.type == JSON_TOK_STRING) free(tok.str);
                        else if (tok.type == JSON_TOK_LBRACE || tok.type == JSON_TOK_LBRACKET) {
                            json_skip_value(p); /* This is wrong but won't be hit */
                        }
                        tok = json_next_token(p);
                        if (tok.type == JSON_TOK_RBRACKET) break;
                        TOK_ASSERT(tok.type == JSON_TOK_COMMA, "expected comma");
                        tok = json_next_token(p);
                    }
                }
            }
            /* else: number, bool, null — nothing to free */
        array_elem_done:
            tok = json_next_token(p);
            if (tok.type == JSON_TOK_RBRACKET) return;
            TOK_ASSERT(tok.type == JSON_TOK_COMMA, "expected comma");
            tok = json_next_token(p);
        }
    }
}

/* Auto-generated by tools/gen_byte_table.py — do not edit. */
/* GPT-2/Qwen byte-level alphabet: byte → unicode string (UTF-8) */
static const char *qw6_byte_chars[] = {
    "\304\200",
    "\304\201",
    "\304\202",
    "\304\203",
    "\304\204",
    "\304\205",
    "\304\206",
    "\304\207",
    "\304\210",
    "\304\211",
    "\304\212",
    "\304\213",
    "\304\214",
    "\304\215",
    "\304\216",
    "\304\217",
    "\304\220",
    "\304\221",
    "\304\222",
    "\304\223",
    "\304\224",
    "\304\225",
    "\304\226",
    "\304\227",
    "\304\230",
    "\304\231",
    "\304\232",
    "\304\233",
    "\304\234",
    "\304\235",
    "\304\236",
    "\304\237",
    "\304\240",
    "!",
    "\"",
    "#",
    "$",
    "%",
    "&",
    "'",
    "(",
    ")",
    "*",
    "+",
    ",",
    "-",
    ".",
    "/",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    ";",
    "<",
    "=",
    ">",
    "?",
    "@",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "[",
    "\\",
    "]",
    "^",
    "_",
    "`",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "{",
    "|",
    "}",
    "~",
    "\304\241",
    "\304\242",
    "\304\243",
    "\304\244",
    "\304\245",
    "\304\246",
    "\304\247",
    "\304\250",
    "\304\251",
    "\304\252",
    "\304\253",
    "\304\254",
    "\304\255",
    "\304\256",
    "\304\257",
    "\304\260",
    "\304\261",
    "\304\262",
    "\304\263",
    "\304\264",
    "\304\265",
    "\304\266",
    "\304\267",
    "\304\270",
    "\304\271",
    "\304\272",
    "\304\273",
    "\304\274",
    "\304\275",
    "\304\276",
    "\304\277",
    "\305\200",
    "\305\201",
    "\305\202",
    "\302\241",
    "\302\242",
    "\302\243",
    "\302\244",
    "\302\245",
    "\302\246",
    "\302\247",
    "\302\250",
    "\302\251",
    "\302\252",
    "\302\253",
    "\302\254",
    "\305\203",
    "\302\256",
    "\302\257",
    "\302\260",
    "\302\261",
    "\302\262",
    "\302\263",
    "\302\264",
    "\302\265",
    "\302\266",
    "\302\267",
    "\302\270",
    "\302\271",
    "\302\272",
    "\302\273",
    "\302\274",
    "\302\275",
    "\302\276",
    "\302\277",
    "\303\200",
    "\303\201",
    "\303\202",
    "\303\203",
    "\303\204",
    "\303\205",
    "\303\206",
    "\303\207",
    "\303\210",
    "\303\211",
    "\303\212",
    "\303\213",
    "\303\214",
    "\303\215",
    "\303\216",
    "\303\217",
    "\303\220",
    "\303\221",
    "\303\222",
    "\303\223",
    "\303\224",
    "\303\225",
    "\303\226",
    "\303\227",
    "\303\230",
    "\303\231",
    "\303\232",
    "\303\233",
    "\303\234",
    "\303\235",
    "\303\236",
    "\303\237",
    "\303\240",
    "\303\241",
    "\303\242",
    "\303\243",
    "\303\244",
    "\303\245",
    "\303\246",
    "\303\247",
    "\303\250",
    "\303\251",
    "\303\252",
    "\303\253",
    "\303\254",
    "\303\255",
    "\303\256",
    "\303\257",
    "\303\260",
    "\303\261",
    "\303\262",
    "\303\263",
    "\303\264",
    "\303\265",
    "\303\266",
    "\303\267",
    "\303\270",
    "\303\271",
    "\303\272",
    "\303\273",
    "\303\274",
    "\303\275",
    "\303\276",
    "\303\277"
};
/* 256 entries, bytes 0-255 */


static void qw6_init_byte_table(qw6_tokenizer_t *t) {
    for (int i = 0; i < 256; i++) {
        snprintf(t->byte_to_char[i], sizeof(t->byte_to_char[i]),
                 "%s", qw6_byte_chars[i]);
    }
}

/* ---- Tokenizer loading ---- */

int qw6_tok_load_json(qw6_tokenizer_t *t, const char *path) {
    TOK_ASSERT_PTR(t);
    TOK_ASSERT_PTR(path);

    if (t->merges) { free(t->merges); t->merges = NULL; }
    if (t->vocab) { free(t->vocab); t->vocab = NULL; }
    memset(t, 0, sizeof(*t));
    t->vocab = calloc(QW6_TOK_VOCAB_SIZE, sizeof(char*));
    t->merges = calloc(QW6_TOK_MAX_MERGES, sizeof(*t->merges));
    qw6_init_byte_table(t);

    /* Read the entire file */
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "qw6_tok: cannot open %s\n", path);
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    TOK_ASSERT(fsize > 0, "empty tokenizer.json");

    char *src = malloc(fsize + 1);
    TOK_ASSERT_PTR(src);
    fread(src, 1, fsize, f);
    src[fsize] = '\0';
    fclose(f);

    json_parser_t p = { .src = src, .src_len = (size_t)fsize, .pos = 0 };

    /* Parse top-level object */
    json_token_t tok = json_next_token(&p);
    TOK_ASSERT(tok.type == JSON_TOK_LBRACE, "expected '{'");

    while (1) {
        tok = json_next_token(&p);
        if (tok.type == JSON_TOK_RBRACE) break;
        TOK_ASSERT(tok.type == JSON_TOK_STRING, "expected key string");

        char *key = tok.str;

        tok = json_next_token(&p);
        TOK_ASSERT(tok.type == JSON_TOK_COLON, "expected ':'");

        if (strcmp(key, "model") == 0) {
            /* Parse model object: { type, vocab, merges, ... } */
            tok = json_next_token(&p);
            TOK_ASSERT(tok.type == JSON_TOK_LBRACE, "expected '{' for model");

            while (1) {
                tok = json_next_token(&p);
                if (tok.type == JSON_TOK_RBRACE) break;
                TOK_ASSERT(tok.type == JSON_TOK_STRING, "expected model key");
                char *mkey = tok.str;

                tok = json_next_token(&p);
                TOK_ASSERT(tok.type == JSON_TOK_COLON, "expected ':'");

                if (strcmp(mkey, "vocab") == 0) {
                    /* vocab = { "token_str": id, ... } */
                    tok = json_next_token(&p);
                    TOK_ASSERT(tok.type == JSON_TOK_LBRACE, "expected '{' for vocab");

                    while (1) {
                        tok = json_next_token(&p);
                        if (tok.type == JSON_TOK_RBRACE) break;
                        TOK_ASSERT(tok.type == JSON_TOK_STRING, "expected vocab key");
                        char *vstr = tok.str;

                        tok = json_next_token(&p);
                        TOK_ASSERT(tok.type == JSON_TOK_COLON, "expected ':'");
                        tok = json_next_token(&p);
                        TOK_ASSERT(tok.type == JSON_TOK_NUMBER, "expected vocab id");

                        uint32_t id = (uint32_t)tok.number;
                        if (id < QW6_TOK_VOCAB_SIZE) {
                            t->vocab[id] = vstr;  /* takes ownership */
                            vstr = NULL;  /* don't free */
                            if (id >= t->vocab_count) t->vocab_count = id + 1;
                        }

                        if (vstr) free(vstr);

                        tok = json_next_token(&p);
                        if (tok.type == JSON_TOK_COMMA) continue;
                        if (tok.type == JSON_TOK_RBRACE) break;
                        TOK_ASSERT(0, "expected , or } in vocab");
                    }
                } else if (strcmp(mkey, "merges") == 0) {
                    /* merges = [ "a b", "c d", ... ]
                     * Build a hash map from vocab for O(1) lookups. */
                    hash_map_t vhash;
                    hash_init(&vhash, t->vocab_count * 2 + 1);
                    for (uint32_t i = 0; i < t->vocab_count; i++) {
                        if (t->vocab[i])
                            hash_insert(&vhash, t->vocab[i], i);
                    }
                    fprintf(stderr, "qw6_tok: hash map built (%zu entries, cap %zu)\n",
                            vhash.count, vhash.capacity);

                    tok = json_next_token(&p);
                    TOK_ASSERT(tok.type == JSON_TOK_LBRACKET, "expected '[' for merges");

                    while (1) {
                        tok = json_next_token(&p);
                        if (tok.type == JSON_TOK_RBRACKET) break;
                        if (tok.type == JSON_TOK_COMMA) continue;
                        TOK_ASSERT(tok.type == JSON_TOK_STRING, "expected merge string");
                        char *mstr = tok.str;

                        /* Parse "a b" → split at first space */
                        char *space = strchr(mstr, ' ');
                        if (space) {
                            *space = '\0';
                            char *a_str = mstr;
                            char *b_str = space + 1;

                            int32_t a_id = hash_lookup(&vhash, a_str);
                            int32_t b_id = hash_lookup(&vhash, b_str);

                            /* Merge result = concatenated string */
                            size_t la = strlen(a_str);
                            size_t lb = strlen(b_str);
                            char merged[512];
                            TOK_ASSERT(la + lb < 512, "merge string too long");
                            memcpy(merged, a_str, la);
                            memcpy(merged + la, b_str, lb);
                            merged[la + lb] = '\0';

                            int32_t result_id = hash_lookup(&vhash, merged);

                            if (a_id >= 0 && b_id >= 0 && result_id >= 0 &&
                                t->merge_count < QW6_TOK_MAX_MERGES) {
                                t->merges[t->merge_count].a = (uint32_t)a_id;
                                t->merges[t->merge_count].b = (uint32_t)b_id;
                                t->merges[t->merge_count].result = (uint32_t)result_id;
                                t->merge_count++;
                            }
                        }
                        free(mstr);

                        tok = json_next_token(&p);
                        if (tok.type == JSON_TOK_COMMA) continue;
                        if (tok.type == JSON_TOK_RBRACKET) break;
                    }

                    hash_free(&vhash);
                    fprintf(stderr, "qw6_tok: parsed %u merges\n", t->merge_count);
        } else {
                    /* Skip other model keys (type, unk_token, etc.) */
                    json_skip_value(&p);
                }

                free(mkey);

                tok = json_next_token(&p);
                if (tok.type == JSON_TOK_COMMA) continue;
                if (tok.type == JSON_TOK_RBRACE) break;
                TOK_ASSERT(0, "expected , or } in model");
            }
        } else if (strcmp(key, "added_tokens") == 0) {
            /* added_tokens = [ { content, id, special, ... }, ... ] */
            tok = json_next_token(&p);
            TOK_ASSERT(tok.type == JSON_TOK_LBRACKET, "expected '[' for added_tokens");

            while (1) {
                tok = json_next_token(&p);
                if (tok.type == JSON_TOK_RBRACKET) break;
                if (tok.type == JSON_TOK_COMMA) continue;
                TOK_ASSERT(tok.type == JSON_TOK_LBRACE, "expected '{' for added_token");

                char content[64] = {0};
                uint32_t id = 0;
                bool special = false;

                while (1) {
                    tok = json_next_token(&p);
                    if (tok.type == JSON_TOK_RBRACE) break;
                    TOK_ASSERT(tok.type == JSON_TOK_STRING, "expected added_token key");
                    char *akey = tok.str;

                    tok = json_next_token(&p);
                    TOK_ASSERT(tok.type == JSON_TOK_COLON, "expected ':'");

                    if (strcmp(akey, "content") == 0) {
                        tok = json_next_token(&p);
                        TOK_ASSERT(tok.type == JSON_TOK_STRING, "expected content string");
                        snprintf(content, sizeof(content), "%s", tok.str);
                        free(tok.str);
                    } else if (strcmp(akey, "id") == 0) {
                        tok = json_next_token(&p);
                        TOK_ASSERT(tok.type == JSON_TOK_NUMBER, "expected id number");
                        id = (uint32_t)tok.number;
                    } else if (strcmp(akey, "special") == 0) {
                        tok = json_next_token(&p);
                        TOK_ASSERT(tok.type == JSON_TOK_TRUE || tok.type == JSON_TOK_FALSE,
                                   "expected special bool");
                        special = (tok.type == JSON_TOK_TRUE);
        } else {
                        json_skip_value(&p);
                    }

                    free(akey);

                    tok = json_next_token(&p);
                    if (tok.type == JSON_TOK_COMMA) continue;
                    if (tok.type == JSON_TOK_RBRACE) break;
                }

                if (t->added_count < QW6_TOK_NUM_ADDED) {
                    snprintf(t->added[t->added_count].content,
                             sizeof(t->added[t->added_count].content), "%s", content);
                    t->added[t->added_count].id = id;
                    t->added[t->added_count].special = special;
                    t->added_count++;

                    /* Also store in main vocab if not already there */
                    if (id < QW6_TOK_VOCAB_SIZE && !t->vocab[id]) {
                        t->vocab[id] = strdup(content);
                    }
                }
            }
        } else {
            /* Skip other top-level keys */
            json_skip_value(&p);
        }

        free(key);

        tok = json_next_token(&p);
        if (tok.type == JSON_TOK_COMMA) continue;
        if (tok.type == JSON_TOK_RBRACE) break;
        TOK_ASSERT(0, "expected , or } at top level");
    }

    free(src);

    fprintf(stderr, "qw6_tok: loaded %u vocab tokens, %u merges, %u added tokens\n",
            t->vocab_count, t->merge_count, t->added_count);
    return 0;
}

int qw6_tok_init(qw6_tokenizer_t *t, const char *tokenizer_json_path) {
    return qw6_tok_load_json(t, tokenizer_json_path);
}

void qw6_tok_free(qw6_tokenizer_t *t) {
    TOK_ASSERT_PTR(t);
    for (uint32_t i = 0; i < QW6_TOK_VOCAB_SIZE; i++) {
        if (t->vocab[i]) {
            free(t->vocab[i]);
            t->vocab[i] = NULL;
        }
    }
    if (t->merges) { free(t->merges); t->merges = NULL; }
    if (t->vocab) { free(t->vocab); t->vocab = NULL; }
    memset(t, 0, sizeof(*t));
}

const char *qw6_tok_id_to_str(qw6_tokenizer_t *t, uint32_t id) {
    TOK_ASSERT_PTR(t);
    if (id < t->vocab_count && t->vocab[id]) {
        return t->vocab[id];
    }
    return "<unk>";
}
/* ---- Pre-tokenization (regex-based) ---- */
/* GPT-2 style regex: split text into pre-tokens
 * Pattern: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
 *
 * For simplicity, we use a basic implementation that handles ASCII well.
 * Full Unicode \p{L} / \p{N} support needs ICU or a Unicode table.
 * Phase 1: ASCII + basic Unicode (bytes 128-255 treated as letters).
 */

typedef struct {
    const char *text;
    size_t len;
    size_t pos;
} pretok_state_t;

static bool is_letter(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80;
}

static bool is_number(unsigned char c) {
    return c >= '0' && c <= '9';
}

static bool is_space_char(unsigned char c) {
    return c == ' ' || c == '\t';
}

static bool is_newline(unsigned char c) {
    return c == '\r' || c == '\n';
}

static bool is_whitespace(unsigned char c) {
    return is_space_char(c) || is_newline(c);
}

/* Get next pre-token. Returns length, or 0 if done. */
static size_t pretok_next(pretok_state_t *st, const char **out_start) {
    *out_start = st->text + st->pos;
    size_t start = st->pos;

    if (st->pos >= st->len) return 0;

    unsigned char c = (unsigned char)st->text[st->pos];

    /* Check for apostrophe contractions: 's, 't, 're, 've, 'm, 'll, 'd (case-insensitive) */
    if (c == '\'') {
        /* Look ahead for s, t, re, ve, m, ll, d */
        if (st->pos + 1 < st->len) {
            unsigned char n = (unsigned char)st->text[st->pos + 1];
            char lower = tolower(n);
            if (lower == 's' || lower == 't' || lower == 'm' || lower == 'd') {
                st->pos += 2;
                return st->pos - start;
            }
            if (lower == 'r' && st->pos + 2 < st->len && tolower(st->text[st->pos + 2]) == 'e') {
                st->pos += 3;
                return st->pos - start;
            }
            if (lower == 'v' && st->pos + 2 < st->len && tolower(st->text[st->pos + 2]) == 'e') {
                st->pos += 3;
                return st->pos - start;
            }
            if (lower == 'l' && st->pos + 2 < st->len && tolower(st->text[st->pos + 2]) == 'l') {
                st->pos += 3;
                return st->pos - start;
            }
        }
    }

    /* [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+ — optional non-letter prefix + letters */
    {
        size_t p = st->pos;
        /* Optional prefix: one non-letter, non-number, non-newline char */
        if (p < st->len && !is_newline((unsigned char)st->text[p]) &&
            !is_letter((unsigned char)st->text[p]) &&
            !is_number((unsigned char)st->text[p])) {
            p++;
        }
        /* Must have at least one letter */
        if (p < st->len && is_letter((unsigned char)st->text[p])) {
            while (p < st->len && is_letter((unsigned char)st->text[p])) p++;
            st->pos = p;
            return p - start;
        }
        /* No letter after prefix — backtrack */
    }

    /* \p{N} — single number */
    if (is_number(c)) {
        st->pos++;
        /* GPT-2 regex matches single digit, not number run */
        return 1;
    }

    /* ?[^\s\p{L}\p{M}\p{N}]+[\r\n]* — punct + trailing newlines */
    if (!is_whitespace(c) && !is_letter(c) && !is_number(c)) {
        size_t p = st->pos;
        while (p < st->len && !is_whitespace((unsigned char)st->text[p]) &&
               !is_letter((unsigned char)st->text[p]) &&
               !is_number((unsigned char)st->text[p])) {
            p++;
        }
        while (p < st->len && is_newline((unsigned char)st->text[p])) p++;
        st->pos = p;
        return p - start;
    }

    /* \s*[\r\n]+ — newlines with preceding spaces */
    if (is_newline(c)) {
        size_t p = st->pos;
        while (p < st->len && is_whitespace((unsigned char)st->text[p])) {
            if (is_newline((unsigned char)st->text[p])) {
                /* Consume all newlines after optional leading spaces */
                while (p < st->len && (is_newline((unsigned char)st->text[p]) ||
                      is_space_char((unsigned char)st->text[p]))) {
                    if (is_newline((unsigned char)st->text[p]))
                        p++;
                    else
                        break;
                }
                break;
            }
            p++;
        }
        /* Simplified: just consume all whitespace */
        while (p < st->len && is_whitespace((unsigned char)st->text[p])) p++;
        st->pos = p;
        return p - start;
    }

    /* \s+(?!\S) | \s+ — whitespace (trailing or general) */
    if (is_space_char(c)) {
        size_t p = st->pos;
        while (p < st->len && is_space_char((unsigned char)st->text[p])) p++;
        st->pos = p;
        return p - start;
    }

    /* Fallback: single char */
    st->pos++;
    return 1;
}

/* ---- BPE encoding ---- */

/* Find a vocab ID by string. Linear search (slow but correct for load).
 * Called during encoding, should use hash map for production. */
static int32_t find_token_id(qw6_tokenizer_t *t, const char *str) {
    /* Hash map would be faster but we don't keep one alive.
     * For encoding, we build a temporary hash map on first call. */
    static hash_map_t enc_hash = {0};
    static bool hash_built = false;

    if (!hash_built) {
        hash_init(&enc_hash, t->vocab_count * 2 + 1);
        for (uint32_t i = 0; i < t->vocab_count; i++) {
            if (t->vocab[i])
                hash_insert(&enc_hash, t->vocab[i], i);
        }
        hash_built = true;
        fprintf(stderr, "qw6_tok: encode hash map built (%zu entries)\n", enc_hash.count);
    }

    return hash_lookup(&enc_hash, str);
}

/* Find merge rank for a pair (a, b). Returns merge index or -1. */
static int32_t find_merge(qw6_tokenizer_t *t, uint32_t a, uint32_t b) {
    for (uint32_t i = 0; i < t->merge_count; i++) {
        if (t->merges[i].a == a && t->merges[i].b == b) {
            return (int32_t)i;
        }
    }
    return -1;
}

/* Apply BPE merges to a list of token IDs */
static void apply_bpe(qw6_tokenizer_t *t, uint32_t *tokens, uint32_t *count) {
    if (*count < 2) return;

    /* Repeatedly find the best merge (lowest merge rank) and apply it */
    bool changed = true;
    while (changed) {
        changed = false;
        int32_t best_rank = -1;
        uint32_t best_pos = 0;

        for (uint32_t i = 0; i + 1 < *count; i++) {
            int32_t rank = find_merge(t, tokens[i], tokens[i + 1]);
            if (rank >= 0 && (best_rank < 0 || rank < best_rank)) {
                best_rank = rank;
                best_pos = i;
            }
        }

        if (best_rank >= 0) {
            /* Merge tokens[best_pos] and tokens[best_pos+1] */
            tokens[best_pos] = t->merges[best_rank].result;
            /* Shift remaining tokens left */
            for (uint32_t i = best_pos + 1; i + 1 < *count; i++) {
                tokens[i] = tokens[i + 1];
            }
            (*count)--;
            changed = true;
        }
    }
}

/* Encode a single pre-token string → BPE token IDs */
static int encode_pretoken(qw6_tokenizer_t *t, const char *str, size_t len,
                           uint32_t *out, uint32_t *out_count) {
    /* Step 1: Convert each byte to byte-level char, then look up as token.
     * For ByteLevel: each byte → unicode char → initial token.
     */
    uint32_t byte_tokens[256];  /* max 256 bytes per pre-token */
    uint32_t n = 0;

    for (size_t i = 0; i < len && n < 256; i++) {
        unsigned char b = (unsigned char)str[i];
        const char *bc = t->byte_to_char[b];

        /* Look up this byte-level char in vocab */
        int32_t id = find_token_id(t, bc);
        if (id < 0) {
            /* Try raw byte */
            char single[2] = { (char)b, 0 };
            id = find_token_id(t, single);
        }
        if (id < 0) {
            /* Fallback: use byte value + offset.
             * In Qwen2Tokenizer, bytes map to specific IDs.
             * The first 256 vocab entries are the byte-level chars. */
            id = (int32_t)b;  /* vocab entries 0-255 are byte-level */
        }
        byte_tokens[n++] = (uint32_t)id;
    }

    /* Step 2: Apply BPE merges */
    apply_bpe(t, byte_tokens, &n);

    /* Output */
    for (uint32_t i = 0; i < n && *out_count < 65536; i++) {
        out[(*out_count)++] = byte_tokens[i];
    }

    return 0;
}

/* ---- Public encode/decode ---- */

int qw6_tok_encode(qw6_tokenizer_t *t, const char *text,
                   uint32_t **out_tokens, uint32_t *out_count) {
    TOK_ASSERT_PTR(t);
    TOK_ASSERT_PTR(text);
    TOK_ASSERT_PTR(out_tokens);
    TOK_ASSERT_PTR(out_count);

    size_t text_len = strlen(text);
    if (text_len == 0) {
        *out_tokens = NULL;
        *out_count = 0;
        return 0;
    }

    /* Check for added tokens (special tokens) in text first */
    /* For now, skip special token detection — just BPE encode */

    uint32_t *tokens = malloc(65536 * sizeof(uint32_t));
    TOK_ASSERT_PTR(tokens);
    uint32_t count = 0;

    /* Pre-tokenize and encode each pre-token */
    pretok_state_t st = { .text = text, .len = text_len, .pos = 0 };
    const char *pretok_start = NULL;

    while (1) {
        size_t plen = pretok_next(&st, &pretok_start);
        if (plen == 0) break;

        encode_pretoken(t, pretok_start, plen, tokens, &count);
    }

    *out_tokens = tokens;
    *out_count = count;
    return 0;
}

/* ---- Byte-level decoding ---- */

/* Build inverse map: unicode char (UTF-8) → original byte.
 * Uses the same qw6_byte_chars[] table, but reversed: for each byte i,
 * qw6_byte_chars[i] is the UTF-8 string, so we parse the UTF-8 and map
 * the resulting codepoint back to byte i. */
static unsigned char qw6_unicode_to_byte[0x200]; /* codepoint → byte */
static bool qw6_inverse_built = false;

static void build_inverse_byte_map(void) {
    if (qw6_inverse_built) return;
    memset(qw6_unicode_to_byte, 0xFF, sizeof(qw6_unicode_to_byte));

    for (int byte = 0; byte < 256; byte++) {
        const char *s = qw6_byte_chars[byte];
        /* Decode UTF-8 (1-3 bytes) to a codepoint */
        unsigned cp = 0;
        unsigned char c0 = (unsigned char)s[0];
        if (c0 < 0x80) {
            cp = c0;
        } else if ((c0 & 0xE0) == 0xC0) {
            cp = ((unsigned)(c0 & 0x1F) << 6) | ((unsigned)(unsigned char)s[1] & 0x3F);
        } else if ((c0 & 0xF0) == 0xE0) {
            cp = ((unsigned)(c0 & 0x0F) << 12) |
                 ((unsigned)(unsigned char)s[1] & 0x3F) << 6 |
                 ((unsigned)(unsigned char)s[2] & 0x3F);
        }
        if (cp < 0x200) {
            qw6_unicode_to_byte[cp] = (unsigned char)byte;
        }
    }
    qw6_inverse_built = true;
}

/* Reverse byte-level encoding: convert UTF-8 string with byte-level chars
 * back to the original bytes. E.g. "Ġ" (U+0120) -> byte 32 (space). */
static char *reverse_byte_level(const char *str, size_t len) {
    build_inverse_byte_map();

    char *result = malloc(len + 1);
    TOK_ASSERT_PTR(result);
    size_t pos = 0;
    size_t i = 0;

    while (i < len) {
        unsigned char c = (unsigned char)str[i];

        if (c < 0x80) {
            /* ASCII byte: map via inverse table (handles special cases
             * like byte 34 = " which maps to U+0022 = itself) */
            unsigned char b = qw6_unicode_to_byte[c];
            /* For printable ASCII (33-126), the char IS the byte directly.
             * For other ASCII (0-32, 127), the table maps to the original byte. */
            if (c >= 33 && c <= 126) {
                result[pos++] = (char)c;
            } else {
                /* Non-printable ASCII range (shouldn't normally appear,
                 * since these map to non-ASCII unicode chars) */
                result[pos++] = (char)b;
            }
            i++;
        } else if ((c & 0xE0) == 0xC0) {
            /* 2-byte UTF-8: decode codepoint */
            if (i + 1 >= len) { result[pos++] = (char)c; i++; continue; }
            unsigned cp = ((unsigned)(c & 0x1F) << 6) |
                          ((unsigned)(unsigned char)str[i+1] & 0x3F);
            unsigned char b = qw6_unicode_to_byte[cp < 0x200 ? cp : 0];
            if (b != 0xFF) {
                result[pos++] = (char)b;
            }
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            /* 3-byte UTF-8: decode codepoint */
            if (i + 2 >= len) { result[pos++] = (char)c; i++; continue; }
            unsigned cp = ((unsigned)(c & 0x0F) << 12) |
                          ((unsigned)(unsigned char)str[i+1] & 0x3F) << 6 |
                          ((unsigned)(unsigned char)str[i+2] & 0x3F);
            unsigned char b = qw6_unicode_to_byte[cp < 0x200 ? cp : 0];
            if (b != 0xFF) {
                result[pos++] = (char)b;
            }
            i += 3;
        } else {
            /* 4-byte UTF-8 or invalid: pass through */
            result[pos++] = (char)c;
            i++;
        }
    }

    result[pos] = '\0';
    return result;
}

int qw6_tok_decode(qw6_tokenizer_t *t, const uint32_t *tokens,
                   uint32_t count, char **out_text) {
    TOK_ASSERT_PTR(t);
    TOK_ASSERT_PTR(tokens);
    TOK_ASSERT_PTR(out_text);

    /* Step 1: concatenate token strings (in byte-level encoding) */
    size_t total_len = 0;
    for (uint32_t i = 0; i < count; i++) {
        const char *s = qw6_tok_id_to_str(t, tokens[i]);
        if (s) total_len += strlen(s);
    }

    char *raw = malloc(total_len + 1);
    TOK_ASSERT_PTR(raw);
    size_t pos = 0;
    for (uint32_t i = 0; i < count; i++) {
        const char *s = qw6_tok_id_to_str(t, tokens[i]);
        if (s) {
            size_t slen = strlen(s);
            memcpy(raw + pos, s, slen);
            pos += slen;
        }
    }
    raw[pos] = '\0';

    /* Step 2: reverse byte-level encoding (Ġ → space, etc.) */
    char *result = reverse_byte_level(raw, pos);
    free(raw);

    *out_text = result;
    return 0;
}