/* Structured-mutation fuzz for json.h against hostile documents.
 *
 * json.h parses two kinds of untrusted input: safetensors headers from
 * model mirrors and tokenizer.json/config files. Unlike rans.h it has no
 * error channel -- malformed input yields a partial tree -- so the only
 * automatically checkable properties are memory ones, which is exactly
 * what ASan+UBSan assert here: no OOB read/write while parsing, no
 * invalid access while walking the resulting tree, no double free /
 * leak-adjacent crash in json_free. Every mutant lives in an exactly
 * sized buffer (fuzz_mut.h) so one byte read past the terminating NUL
 * is a report.
 *
 * Seeded and bounded (7 base documents x 900 structured mutants). Build
 * and run via `make fuzz-json` (ASan+UBSan, non-recoverable). NOT part
 * of TEST_BINS: the sanitizer flags differ from the normal test build.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fuzz_mut.h"
#include "../json.h"

/* Shapes drawn from the real callers: a safetensors header, a config,
 * a tokenizer with merges + added_tokens, ref.json, escape/number edge
 * soup, and a nesting bomb near J_MAX_DEPTH built at runtime. */
static const char *BASES[] = {
    "{\"__metadata__\":{\"colibri.fmt\":\"4\"},"
    "\"model.layers.0.mlp.experts.7.w1\":{\"dtype\":\"F32\",\"shape\":[64,32],"
    "\"data_offsets\":[0,8192]},"
    "\"model.layers.0.mlp.experts.7.w2\":{\"dtype\":\"BF16\",\"shape\":[32,64],"
    "\"data_offsets\":[8192,12288]}}",

    "{\"num_hidden_layers\":75,\"num_experts\":256,\"rope_theta\":1e6,"
    "\"rms_norm_eps\":1e-05,\"tie_word_embeddings\":false,\"quant\":null}",

    "{\"model\":{\"vocab\":{\"a\":0,\"b\":1,\"ab\":2,\"<s>\":3},"
    "\"merges\":[\"a b\",[\"a\",\"b\"]]},"
    "\"added_tokens\":[{\"id\":3,\"content\":\"<s>\",\"special\":true}]}",

    "{\"prompt_ids\":[510,5347,273],\"full_ids\":[510,5347,273,6181,310]}",

    "{\"esc\":\"a\\n\\t\\\"\\\\\\u0041\\ud83d\\ude00\\ud800trunc\\u12\","
    "\"nums\":[0,-0,1e308,-1e308,9223372036854775807,0.5e-7,1e999]}",

    "[[[{\"k\":[true,false,null]}]],\"\",{},[]]",

    "{\"dup\":1,\"dup\":2,\"dup\":{\"dup\":[3,3,3]}}",
};

static long walked;

/* Touch every reachable field the way real callers do (json_get by key,
 * strlen on strings) so a dangling or misparsed pointer becomes a
 * sanitizer report instead of dormant state. */
static void walk(jval *v, int depth) {
    if (!v || depth > J_MAX_DEPTH + 8) return;
    walked++;
    switch (v->t) {
    case J_STR: if (v->str) walked += (long)strlen(v->str); break;
    case J_OBJ:
        for (int i = 0; i < v->len; i++) {
            if (v->keys[i]) walked += (long)strlen(v->keys[i]);
            if (json_get(v, v->keys[i]) == NULL) walked--;   /* dup keys: first wins */
            walk(v->kids[i], depth + 1);
        }
        break;
    case J_ARR:
        for (int i = 0; i < v->len; i++) walk(v->kids[i], depth + 1);
        break;
    default: break;
    }
}

int main(void) {
    long parsed = 0, cases = 0;

    /* runtime base 8: a nesting bomb that crosses J_MAX_DEPTH, so the
     * depth bound itself is under mutation too */
    size_t bomb_n = (size_t)J_MAX_DEPTH + 40;
    char *bomb = (char *)malloc(bomb_n * 2 + 2);
    for (size_t i = 0; i < bomb_n; i++) { bomb[i] = '['; bomb[bomb_n + i] = ']'; }
    bomb[bomb_n * 2] = 0;

    const size_t nbases = sizeof(BASES) / sizeof(BASES[0]) + 1;
    for (size_t b = 0; b < nbases; b++) {
        const char *base = b < nbases - 1 ? BASES[b] : bomb;
        size_t blen = strlen(base);
        for (int it = 0; it < 900; it++) {
            size_t mlen;
            uint8_t *m = fm_mutate((const uint8_t *)base, blen, &mlen);
            if (!m) continue;
            cases++;
            char *arena = NULL;
            /* alternate the two arena_out modes real callers use */
            jval *root = json_parse((const char *)m, (it & 1) ? &arena : NULL);
            if (root) { parsed++; walk(root, 0); json_free(root); }
            free(arena);
            free(m);
        }
    }
    free(bomb);
    printf("fuzz_json: %ld cases, %ld parsed, %ld nodes+bytes walked -- clean\n",
           cases, parsed, walked);
    return 0;
}
