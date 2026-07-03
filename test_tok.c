
/* test_tok.c — Tokenizer regression test */
#include "qw6_tok.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const uint32_t ids_hello[] = {9419, 1814};
static const uint32_t ids_hello_comma[] = {9419, 11, 1814, 0};
static const uint32_t ids_fox[] = {760, 3841, 13477, 37550};
static const uint32_t ids_digits[] = {16, 17, 18, 19, 20};

struct test_case {
    const char *text;
    const uint32_t *expected;
    uint32_t expected_count;
};

static const struct test_case tests[] = {
    {"Hello world",       ids_hello,      2},
    {"Hello, world!",     ids_hello_comma, 4},
    {"The quick brown fox", ids_fox,       4},
    {"12345",             ids_digits,     5},
};

#define NUM_TESTS (int)(sizeof(tests) / sizeof(tests[0]))

int main(void) {
    qw6_tokenizer_t tok;
    int rc = qw6_tok_init(&tok, "tokenizer/tokenizer.json");
    if (rc != 0) { fprintf(stderr, "FAIL: init\n"); return 1; }

    int pass = 0, fail = 0;

    for (int i = 0; i < NUM_TESTS; i++) {
        uint32_t *tokens = NULL;
        uint32_t n = 0;

        if (qw6_tok_encode(&tok, tests[i].text, &tokens, &n) != 0) {
            fprintf(stderr, "FAIL [%d] encode error\n", i);
            fail++;
            continue;
        }

        char *decoded = NULL;
        bool roundtrip_ok = false;
        if (qw6_tok_decode(&tok, tokens, n, &decoded) == 0) {
            roundtrip_ok = (strcmp(decoded, tests[i].text) == 0);
            if (!roundtrip_ok)
                fprintf(stderr, "FAIL [%d] roundtrip: \"%s\" -> \"%s\"\n",
                        i, tests[i].text, decoded);
            free(decoded);
        }

        bool ids_ok = (n == tests[i].expected_count);
        if (ids_ok) {
            for (uint32_t j = 0; j < n; j++) {
                if (tokens[j] != tests[i].expected[j]) { ids_ok = false; break; }
            }
        }

        if (ids_ok && roundtrip_ok) {
            fprintf(stderr, "PASS [%d] \"%s\" -> [%u tokens]\n", i, tests[i].text, n);
            pass++;
        } else { fail++; }
        free(tokens);
    }

    qw6_tok_free(&tok);
    fprintf(stderr, "\n%d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
