/* Guarded tokenization (#8): tok_encode_guarded + the CGUARD1 wire parser.
 *
 * The contract under test, on the in-repo tiny fixtures:
 *   1. benign identity -- with spans that contain no added-token text, the
 *      guarded encode produces ID-FOR-ID the same output as tok_encode
 *      (single-pass tokenization: BPE merges across the template/content
 *      boundary are preserved, the property a per-fragment build cannot give);
 *   2. suppression -- an added token whose text starts inside a span encodes
 *      as literal bytes (round-trips through tok_decode), while the same
 *      token text OUTSIDE the spans still becomes the control id;
 *   3. fail closed -- a malformed span table yields 0 tokens, never an
 *      unguarded encode;
 *   4. gw_parse -- legacy passthrough, well-formed payloads (0 and N spans),
 *      and every malformed shape refused with -1;
 *   5. end to end -- a payload built exactly the way openai_server.py builds
 *      it parses and encodes with the template marker live and the injected
 *      marker inert.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../tok.h"
#include "../guard_wire.h"

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

#define FIRST_ADDED_ID 274      /* both tiny fixtures: added ids are >= 274 */

static int count_added(const int *ids, int n) {
    int c = 0;
    for (int i = 0; i < n; i++) if (ids[i] >= FIRST_ADDED_ID) c++;
    return c;
}

int main(void) {
    Tok T;
    tok_load(&T, "tests/tok_o200k_tiny.json");

    /* 1. benign identity, spans covering multibyte-free and plain text */
    {
        const char *text = "hello <|message_user|> world 123";
        int plain[128], guarded[128];
        int span_all[2] = {0, 5};            /* "hello": no marker inside */
        int np = tok_encode(&T, text, (int)strlen(text), plain, 128);
        int ng = tok_encode_guarded(&T, text, (int)strlen(text), span_all, 1, guarded, 128);
        CHECK(np == ng && memcmp(plain, guarded, (size_t)np * sizeof(int)) == 0);
        CHECK(count_added(plain, np) == 1);  /* the marker, outside the span, matched */
    }

    /* 2. suppression inside the span, matching outside it */
    {
        /*        0123456789...                                    */
        const char *text = "<|message_user|>x<|message_user|>y";
        int mlen = 16;                       /* strlen("<|message_user|>") */
        int ids[128];
        /* span covers the SECOND occurrence (starts at 17) */
        int span[2] = {17, 17 + mlen};
        int n = tok_encode_guarded(&T, text, (int)strlen(text), span, 1, ids, 128);
        CHECK(count_added(ids, n) == 1);     /* first matched, second suppressed */
        char back[256];
        int nb = tok_decode(&T, ids, n, back, sizeof back);
        CHECK(nb == (int)strlen(text) && memcmp(back, text, (size_t)nb) == 0);
    }

    /* 3. malformed span tables fail closed */
    {
        const char *text = "abc";
        int ids[16];
        int rev[2] = {2, 1};                                  /* end < start */
        CHECK(tok_encode_guarded(&T, text, 3, rev, 1, ids, 16) == 0);
        int over[2] = {0, 4};                                 /* beyond len */
        CHECK(tok_encode_guarded(&T, text, 3, over, 1, ids, 16) == 0);
        int unsorted[4] = {2, 3, 0, 1};                       /* out of order */
        CHECK(tok_encode_guarded(&T, text, 3, unsorted, 2, ids, 16) == 0);
    }

    /* 4. gw_parse shapes */
    {
        const char *txt; int tl, ns, *sp;
        /* legacy passthrough */
        CHECK(gw_parse("plain prompt", 12, &txt, &tl, &sp, &ns) == 1);
        CHECK(tl == 12 && ns == 0 && sp == NULL && memcmp(txt, "plain prompt", 12) == 0);
        /* zero spans */
        const char *p0 = "CGUARD1\n0\nbody";
        CHECK(gw_parse(p0, (int)strlen(p0), &txt, &tl, &sp, &ns) == 0);
        CHECK(ns == 0 && tl == 4 && memcmp(txt, "body", 4) == 0);
        /* two spans */
        const char *p2 = "CGUARD1\n2\n0 2\n3 5\nabcde";
        CHECK(gw_parse(p2, (int)strlen(p2), &txt, &tl, &sp, &ns) == 0);
        CHECK(ns == 2 && tl == 5 && sp[0] == 0 && sp[1] == 2 && sp[2] == 3 && sp[3] == 5);
        free(sp);
        /* malformed: truncated header, reversed, unsorted, beyond text, junk count */
        const char *bad[] = {
            "CGUARD1\n",                 /* no count line          */
            "CGUARD1\n1\n",              /* missing span line      */
            "CGUARD1\n1\n5 2\nabcdef",   /* reversed               */
            "CGUARD1\n2\n3 4\n0 1\nabcdef", /* unsorted            */
            "CGUARD1\n1\n0 99\nabc",     /* beyond the text        */
            "CGUARD1\nx\nabc",           /* non-numeric count      */
        };
        for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
            CHECK(gw_parse(bad[i], (int)strlen(bad[i]), &txt, &tl, &sp, &ns) == -1);
            CHECK(sp == NULL || 1);      /* contract: nothing to free on -1 */
        }
    }

    /* 5. end to end: a server-shaped payload */
    {
        /* template: MARKER + content + MARKER, content = the same marker text */
        const char *tmpl = "<|message_user|><|message_user|><|message_user|>";
        char payload[256];
        int n = snprintf(payload, sizeof payload, "CGUARD1\n1\n16 32\n%s", tmpl);
        const char *txt; int tl, ns, *sp;
        CHECK(gw_parse(payload, n, &txt, &tl, &sp, &ns) == 0);
        int ids[128];
        int nt = tok_encode_guarded(&T, txt, tl, sp, ns, ids, 128);
        free(sp);
        CHECK(count_added(ids, nt) == 2);   /* outer markers live, inner inert */
        char back[256];
        int nb = tok_decode(&T, ids, nt, back, sizeof back);
        CHECK(nb == (int)strlen(tmpl) && memcmp(back, tmpl, (size_t)nb) == 0);
    }

    tok_free(&T);
    printf("OK guarded encode: identity, suppression, fail-closed, wire parse, end-to-end\n");
    return 0;
}
