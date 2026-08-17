/* Shared structured-mutation engine for the parser fuzz harnesses
 * (fuzz_json, fuzz_grammar, fuzz_st_header, fuzz_tok).
 *
 * Same philosophy as fuzz_rans.c: seeded xorshift, bounded case counts,
 * mutants placed in exactly-sized heap buffers so ASan catches any read
 * past the documented end. This header only mutates bytes; what "exercise
 * one mutant" means is each harness's own business.
 */
#ifndef FUZZ_MUT_H
#define FUZZ_MUT_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint64_t fm_state = 0x9E3779B97F4A7C15ULL;
static uint32_t fm_rnd(void) {
    fm_state ^= fm_state << 13; fm_state ^= fm_state >> 7;
    fm_state ^= fm_state << 17; return (uint32_t)(fm_state >> 32);
}

/* Interesting bytes to splice in: structural delimiters, escapes, NUL,
 * high bytes, and digits (to grow numbers in place). */
static const char FM_SPICE[] = "{}[]\",:\\\0\x01\x7f\x80\xc0\xe0\xf0\xff" "0123456789eE+-.";

/* Produce a mutant of base[0..blen) in a fresh heap buffer of exactly
 * *out_len + 1 bytes (always NUL-terminated one past the reported length,
 * for string-consuming parsers; byte-consuming parsers just use *out_len).
 * Applies 1..4 random mutations: byte flip, splice from FM_SPICE,
 * truncation, span duplication, span overwrite. Caller frees. */
static uint8_t *fm_mutate(const uint8_t *base, size_t blen, size_t *out_len) {
    size_t cap = blen * 2 + 64;
    uint8_t *m = (uint8_t *)malloc(cap);
    if (!m) { *out_len = 0; return NULL; }
    memcpy(m, base, blen);
    size_t len = blen;
    int nmut = 1 + (int)(fm_rnd() % 4);
    for (int i = 0; i < nmut && len > 0; i++) {
        switch (fm_rnd() % 5) {
        case 0:                                    /* flip one byte */
            m[fm_rnd() % len] ^= (uint8_t)(1u << (fm_rnd() % 8));
            break;
        case 1:                                    /* splice an interesting byte */
            m[fm_rnd() % len] = (uint8_t)FM_SPICE[fm_rnd() % (sizeof(FM_SPICE) - 1)];
            break;
        case 2:                                    /* truncate */
            len = fm_rnd() % len;
            break;
        case 3: {                                  /* duplicate a short span */
            size_t at = fm_rnd() % len, n = 1 + fm_rnd() % 16;
            if (n > len - at) n = len - at;
            if (len + n < cap) {
                memmove(m + at + n, m + at, len - at);
                len += n;
            }
            break;
        }
        default: {                                 /* overwrite a span with noise */
            size_t at = fm_rnd() % len, n = 1 + fm_rnd() % 8;
            for (size_t j = 0; j < n && at + j < len; j++)
                m[at + j] = (uint8_t)fm_rnd();
            break;
        }
        }
    }
    /* shrink to exact size: the mutant plus one NUL, nothing else, so any
     * parser read past the end is an ASan report, not silent slack. */
    uint8_t *exact = (uint8_t *)malloc(len + 1);
    if (!exact) { free(m); *out_len = 0; return NULL; }
    memcpy(exact, m, len);
    exact[len] = 0;
    free(m);
    *out_len = len;
    return exact;
}

#endif
