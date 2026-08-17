/* guard_wire.h -- CGUARD1: the gateway's guarded-prompt payload (#8).
 *
 * The gateway renders each architecture's chat template server-side and,
 * historically, sent one flat string the engine tokenized as-is -- so user
 * content spliced into the template could contain added/special-token text
 * ("<|system|>", "<|end_message|>", the OLMoE boundary marker) and tokenize
 * into REAL control tokens: role spoofing from inside a message. CGUARD1
 * keeps the rendering exactly where it was and adds one thing: the byte
 * ranges of the payload that are untrusted content. The engine feeds those
 * ranges to tok_encode_guarded, which suppresses added-token matches that
 * start inside them -- and nothing else changes: one-pass tokenization of
 * the same flat string, byte-identical ids for benign content.
 *
 * Layout (all ASCII decimal, '\n'-terminated header lines):
 *
 *   CGUARD1\n
 *   <k>\n                 number of ranges, 0 <= k <= GW_MAX_SPANS
 *   <start> <end>\n       k times; byte offsets into the template text,
 *                         sorted, non-overlapping, 0 <= start <= end <= tlen
 *   <template bytes>      the rest of the payload, verbatim
 *
 * A payload that does not begin with "CGUARD1\n" is a legacy flat prompt
 * (gw_parse returns 1): engines keep tokenizing it unguarded, which is what
 * standalone clients of the wire protocol still send.
 */
#ifndef GUARD_WIRE_H
#define GUARD_WIRE_H

#include <stdlib.h>
#include <string.h>

#define GW_MAGIC     "CGUARD1\n"
#define GW_MAGIC_LEN 8
#define GW_MAX_SPANS 4096

/* Parse a payload. Returns:
 *   1  not a CGUARD1 payload: *text_out / *tlen_out are the whole payload,
 *      *spans_out NULL, *nspans_out 0 (legacy flat prompt)
 *   0  parsed: *text_out / *tlen_out point INTO pl (no copy), *spans_out is a
 *      malloc'd [2*k] array (NULL when k==0) the caller frees
 *  -1  malformed CGUARD1 (bad counts, offsets out of order or out of range):
 *      the caller must refuse the request -- never fall back to unguarded
 *      tokenization of a payload that CLAIMED to carry guard ranges. */
static int gw_parse(const char *pl, int plen, const char **text_out, int *tlen_out,
                    int **spans_out, int *nspans_out) {
    *spans_out = NULL; *nspans_out = 0;
    if (plen < GW_MAGIC_LEN || memcmp(pl, GW_MAGIC, GW_MAGIC_LEN) != 0) {
        *text_out = pl; *tlen_out = plen;
        return 1;
    }
    const char *p = pl + GW_MAGIC_LEN, *end = pl + plen;
    int *spans = NULL;
    /* one bounded decimal number: value into v, or bail to fail */
    #define GW_NUM(v) do { \
        long _a = -1, _n = 0; \
        while (p < end && *p >= '0' && *p <= '9' && _n < 10) { \
            _a = (_a < 0 ? 0 : _a) * 10 + (*p - '0'); p++; _n++; } \
        if (_a < 0 || p >= end) goto fail; \
        (v) = _a; \
    } while (0)
    long k = 0;
    GW_NUM(k);
    if (p >= end || *p != '\n') goto fail;
    p++;
    if (k > GW_MAX_SPANS) goto fail;
    if (k > 0) {
        spans = (int *)malloc((size_t)k * 2 * sizeof(int));
        if (!spans) goto fail;
    }
    {
        long prev = 0;
        for (long i = 0; i < k; i++) {
            long a = 0, b = 0;
            GW_NUM(a);
            if (p >= end || *p != ' ') goto fail;
            p++;
            GW_NUM(b);
            if (p >= end || *p != '\n') goto fail;
            p++;
            if (a < prev || b < a) goto fail;
            spans[2 * i] = (int)a; spans[2 * i + 1] = (int)b;
            prev = b;
        }
        {
            int tlen = (int)(end - p);
            if (prev > tlen) goto fail;             /* ranges beyond the text */
            *text_out = p; *tlen_out = tlen;
            *spans_out = spans; *nspans_out = (int)k;
            return 0;
        }
    }
    #undef GW_NUM
fail:
    free(spans);
    return -1;
}

#endif /* GUARD_WIRE_H */
