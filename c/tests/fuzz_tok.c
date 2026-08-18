/* Structured-mutation fuzz for tok.h: the tokenizer.json loader (tok_load)
 * and the encode paths (tok_encode_raw, tok_encode_guarded) -- the piece of
 * #5's scope deferred at the time (json.h/grammar.h/st.h landed in PR #1)
 * because the token-framing work (#8: CGUARD1, tok_encode_guarded's span
 * suppression) was about to change these exact paths. That work has since
 * landed, so this now fuzzes the paths as they actually ship.
 *
 * Two modes, because the two surfaces fail differently:
 *
 *   LOADER (tok_load): parses a tokenizer.json from an untrusted model
 *   mirror and, by design, exit(1)s on anything malformed -- the same
 *   DISCOVERY-TIME ABORT SURFACE contract st.h's header parse uses (see the
 *   negative-id / OOB-write / NULL-deref comments at tok.h's vocab and
 *   added_tokens loops). A clean refusal is an exit, not a return, so this
 *   forks a child per mutant exactly like fuzz_st_header.c: the child writes
 *   the mutant and calls tok_load; the parent classifies the outcome.
 *   POSIX only (fork); skipped on Windows, same as fuzz_st_header.
 *
 *   ENCODE (tok_encode_raw / tok_encode_guarded): the SEC #8 surface --
 *   tok_encode_raw is what untrusted user content is supposed to go
 *   through (no added-token recognition at all), and tok_encode_guarded is
 *   the span-suppression mechanism CGUARD1 relies on. Neither exits on bad
 *   input (a malformed span table returns 0, by contract), so this runs
 *   in-process against ONE real tokenizer loaded once (the in-repo tiny
 *   fixture also used by test_tok_guarded.c), fuzzing mutated UTF-8-ish
 *   text and, separately, adversarial span tables (in-bounds, out-of-order,
 *   out-of-range) -- exactly the shape tok_encode_guarded's own validation
 *   (`a<prev||b<a||b>len`) exists to reject cleanly instead of OOB-reading
 *   the guard mask.
 *
 * Seeded and bounded. Build and run via `make fuzz-tok` (ASan+UBSan,
 * non-recoverable). NOT part of TEST_BINS: the sanitizer flags differ from
 * the normal test build.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#endif
#include "fuzz_mut.h"
#include "../tok.h"

/* ================= loader fuzz (fork-per-mutant) ================= */

/* Small, hand-shaped tokenizer.json bodies covering the branches tok_load
 * actually takes: plain merges-list BPE (cl100k-shaped), the o200k/kimi
 * pretokenizer-family detection, a rankbpe tokenizer (merges omitted --
 * T->rankbpe=1), and a pair-array merge form alongside the "left right"
 * string form. All exercise the added_tokens id-bounds validation. */
static const char *LOADER_BASES[] = {
    "{\"model\":{\"vocab\":{\"a\":0,\"b\":1,\"ab\":2,\"<s>\":3,\" \":4},"
    "\"merges\":[\"a b\",[\"a\",\"b\"]]},"
    "\"added_tokens\":[{\"id\":3,\"content\":\"<s>\",\"special\":true},"
    "{\"id\":4,\"content\":\" \",\"special\":false}]}",

    "{\"model\":{\"vocab\":{\"x\":0,\"y\":1,\"xy\":2}},"
    "\"added_tokens\":[{\"id\":2,\"content\":\"xy\",\"special\":false}]}",

    "{\"model\":{\"vocab\":{\"h\":0,\"i\":1,\"hi\":2,\"<|m|>\":3},"
    "\"merges\":[\"h i\"]},"
    "\"added_tokens\":[{\"id\":3,\"content\":\"<|m|>\",\"special\":true}],"
    "\"pre_tokenizer\":{\"pretokenizers\":[{\"pattern\":{\"Regex\":"
    "\"[A-Z]|\\\\p{Lu}\\\\p{Ll}*\"}}]}}",

    "{\"model\":{\"vocab\":{\"c\":0,\"j\":1,\"cj\":2},\"merges\":[\"c j\"]},"
    "\"added_tokens\":[{\"id\":2,\"content\":\"cj\",\"special\":false}],"
    "\"pre_tokenizer\":{\"pretokenizers\":[{\"pattern\":{\"Regex\":"
    "\"\\\\p{Han}+\"}}]}}",
};

static int write_container(const char *path, const uint8_t *bytes, size_t n) {
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    int bad = n && fwrite(bytes, 1, n, f) != n;
    if (fclose(f) != 0) bad = 1;
    return bad ? -1 : 0;
}

#ifndef _WIN32
static long fuzz_loader(const char *dir) {
    char path[512];
    snprintf(path, sizeof path, "%s/tokenizer.json", dir);

    long cases = 0, crashed = 0;
    const size_t nbases = sizeof(LOADER_BASES) / sizeof(LOADER_BASES[0]);
    for (size_t b = 0; b < nbases; b++) {
        size_t blen = strlen(LOADER_BASES[b]);
        for (int it = 0; it < 800; it++) {
            size_t mlen;
            uint8_t *m = fm_mutate((const uint8_t *)LOADER_BASES[b], blen, &mlen);
            if (!m) continue;
            cases++;
            if (write_container(path, m, mlen) != 0) { free(m); continue; }
            free(m);

            pid_t pid = fork();
            if (pid == 0) {
                if (!freopen("/dev/null", "w", stderr)) { /* keep stderr */ }
                Tok T;
                tok_load(&T, path);   /* accepted or exit(1): either is fine here */
                _exit(0);
            }
            int status = 0;
            waitpid(pid, &status, 0);
            if (WIFSIGNALED(status)) {
                int sig = WTERMSIG(status);
                fprintf(stderr,
                        "fuzz_tok(loader): child killed by signal %d "
                        "(base %zu iter %d) -- sanitizer report or crash\n",
                        sig, b, it);
                crashed++;
                if (crashed >= 3) {
                    fprintf(stderr, "too many loader crashes; stopping\n");
                    return -crashed;
                }
            }
        }
    }
    remove(path);
    return crashed ? -crashed : cases;
}
#endif

/* ================= encode fuzz (in-process) ================= */

/* Corpus for the text mutants: plain ASCII, the fixture's own added-token
 * marker (so mutants sometimes still contain it, intact or partially so --
 * exactly what the guard span logic must classify correctly), multibyte
 * UTF-8, and a deliberately truncated/invalid UTF-8 tail (u8_next's decoder
 * is a manual byte walk; a lone continuation or lead byte at the end of the
 * buffer is precisely the kind of off-by-one an ASan run catches). */
static const char *TEXT_BASES[] = {
    "hello world 123",
    "<|message_user|>hello<|message_user|>",
    "caf\xc3\xa9 na\xc3\xafve \xe4\xbd\xa0\xe5\xa5\xbd \xf0\x9f\x98\x80",
    "trailing lead byte \xe2\x82",         /* truncated 3-byte sequence */
    "lone continuation \x80\x80\x80 tail",
    "",
};

#define FIRST_ADDED_ID 274   /* tests/tok_o200k_tiny.json: added ids start here */

/* Mostly valid, in-order span tables (the shape real callers -- guard()'s
 * STX/ETX-derived spans -- actually produce), with a deliberate minority
 * that violates tok_encode_guarded's own contract (a<prev, b<a, b>len) so
 * its refusal path -- not just its accept path -- still runs under the
 * sanitizer, without turning every iteration into a logged refusal (the
 * refusal path itself fprintf's by design: fail loud, never silently
 * degrade to an unguarded encode -- that diagnostic is correct production
 * behavior, not fuzz noise to suppress). */
static void random_spans(int len, int *spans, int *nspans) {
    if (fm_rnd() % 5 == 0) { *nspans = 0; return; }   /* 20%: no spans at all */
    int n = 1 + (int)(fm_rnd() % 2);                   /* 1 or 2 spans */
    int prev = 0;
    for (int i = 0; i < n; i++) {
        int a, b;
        if (fm_rnd() % 40 == 0) {                       /* 2.5%: deliberately adversarial */
            a = (int)(fm_rnd() % (unsigned)(len + 8)) - 4;
            b = a + (int)(fm_rnd() % (unsigned)(len + 8)) - 2;
        } else if (prev <= len) {                       /* valid: in [prev,len], ordered */
            a = prev + (int)(fm_rnd() % (unsigned)(len - prev + 1));
            b = a + (int)(fm_rnd() % (unsigned)(len - a + 1));
        } else { a = prev; b = prev; }                  /* out of room: degenerate but valid */
        spans[2 * i] = a;
        spans[2 * i + 1] = b;
        prev = b > prev ? b : prev;
    }
    *nspans = n;
}

static long fuzz_encode(Tok *T) {
    long cases = 0;
    int out[4096];
    const size_t nbases = sizeof(TEXT_BASES) / sizeof(TEXT_BASES[0]);
    for (size_t b = 0; b < nbases; b++) {
        size_t blen = strlen(TEXT_BASES[b]);
        for (int it = 0; it < 2000; it++) {
            size_t mlen;
            uint8_t *m = fm_mutate((const uint8_t *)TEXT_BASES[b], blen, &mlen);
            if (!m) continue;
            cases++;

            int n_raw = tok_encode_raw(T, (const char *)m, (int)mlen, out, 4096);
            if (n_raw < 0 || n_raw > 4096) {
                fprintf(stderr, "fuzz_tok(encode): tok_encode_raw returned %d "
                                "(base %zu iter %d)\n", n_raw, b, it);
                free(m); return -cases;
            }
            for (int i = 0; i < n_raw; i++)   /* raw never emits an added-token id */
                if (out[i] >= FIRST_ADDED_ID) {
                    fprintf(stderr, "fuzz_tok(encode): tok_encode_raw emitted "
                                    "added id %d (base %zu iter %d)\n",
                            out[i], b, it);
                    free(m); return -cases;
                }

            int spans[8], nspans;
            random_spans((int)mlen, spans, &nspans);
            int n_g = tok_encode_guarded(T, (const char *)m, (int)mlen,
                                         spans, nspans, out, 4096);
            if (n_g < 0 || n_g > 4096) {
                fprintf(stderr, "fuzz_tok(encode): tok_encode_guarded returned %d "
                                "(base %zu iter %d)\n", n_g, b, it);
                free(m); return -cases;
            }
            /* nspans==0 must be byte-identical to plain tok_encode (tok.h's
             * own documented contract: tok_encode == tok_encode_guarded with
             * an empty span table). */
            if (nspans == 0) {
                int n_plain = tok_encode(T, (const char *)m, (int)mlen, out, 4096);
                if (n_plain != n_g) {
                    fprintf(stderr, "fuzz_tok(encode): tok_encode/guarded(0 spans) "
                                    "diverged: %d vs %d (base %zu iter %d)\n",
                            n_plain, n_g, b, it);
                    free(m); return -cases;
                }
            }
            free(m);
        }
    }
    return cases;
}

int main(void) {
    long enc_cases = 0, loader_cases = 0;

    Tok T;
    tok_load(&T, "tests/tok_o200k_tiny.json");
    enc_cases = fuzz_encode(&T);
    tok_free(&T);
    if (enc_cases < 0) return 2;

#ifndef _WIN32
    char dir[] = "/tmp/coli_fuzz_tok_XXXXXX";
    if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }
    loader_cases = fuzz_loader(dir);
    rmdir(dir);
    if (loader_cases < 0) return 2;
#else
    printf("fuzz_tok: loader half skipped (needs fork; encode half ran)\n");
#endif

    printf("fuzz_tok: %ld encode cases, %ld loader cases -- clean\n",
           enc_cases, loader_cases);
    return 0;
}
