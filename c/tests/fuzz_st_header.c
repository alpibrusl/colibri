/* Structured-mutation fuzz for st.h's safetensors header parse.
 *
 * st_init() parses the header of an untrusted model container and, by
 * design, exit(1)s on anything malformed (the DISCOVERY-TIME ABORT
 * SURFACE the header documents). A clean refusal is therefore an exit,
 * not a return -- so this harness, like test_st_shape.c, forks a child
 * per mutant: the child writes the mutated container and calls st_init;
 * the parent classifies the outcome.
 *
 *   - normal exit (any code)      -> accepted or cleanly refused: fine
 *   - killed by SIGSEGV/SIGABRT   -> a sanitizer report or a real crash
 *                                    on hostile input: FAIL loudly
 *
 * The child runs under the same ASan+UBSan the Makefile target builds
 * with, so an OOB read in the header walk becomes SIGABRT in the child
 * and a nonzero exit here. POSIX only (fork); Windows is covered by the
 * subprocess pattern in test_st_shape.c instead.
 *
 * Seeded and bounded (5 base headers x 800 mutants). Build and run via
 * `make fuzz-st-header`. NOT part of TEST_BINS.
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
#include "../st.h"

static const char *BASES[] = {
    "{\"t\":{\"dtype\":\"F32\",\"shape\":[64,32],\"data_offsets\":[0,8192]}}",
    "{\"__metadata__\":{\"colibri.fmt\":\"4\"},"
    "\"w\":{\"dtype\":\"U8\",\"shape\":[128],\"data_offsets\":[0,128]}}",
    "{\"a\":{\"dtype\":\"BF16\",\"shape\":[2,3,4],\"data_offsets\":[0,48]},"
    "\"b\":{\"dtype\":\"I8\",\"shape\":[16],\"data_offsets\":[48,64]}}",
    "{\"x\":{\"dtype\":\"F16\",\"shape\":[0,4096],\"data_offsets\":[0,0]}}",
    "{\"q\":{\"dtype\":\"U8\",\"shape\":[8,8],\"data_offsets\":[0,64]}}",
};

/* write a container: 8-byte little-endian header length, header bytes,
 * then payload_bytes of zero data so a valid data_offsets range has
 * something to point at */
static int write_container(const char *dir, const uint8_t *hdr, size_t hlen,
                           size_t payload_bytes) {
    char path[512];
    snprintf(path, sizeof path, "%s/model.safetensors", dir);
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint64_t h = (uint64_t)hlen;
    int bad = fwrite(&h, 8, 1, f) != 1 ||
              fwrite(hdr, 1, hlen, f) != hlen;
    for (size_t i = 0; i < payload_bytes && !bad; i++)
        if (fputc(0, f) == EOF) bad = 1;
    if (fclose(f) != 0) bad = 1;
    return bad ? -1 : 0;
}

#ifndef _WIN32
int main(void) {
    char dir[] = "/tmp/coli_fuzz_st_XXXXXX";
    if (!mkdtemp(dir)) { perror("mkdtemp"); return 1; }

    long cases = 0, crashed = 0;
    const size_t nbases = sizeof(BASES) / sizeof(BASES[0]);
    for (size_t b = 0; b < nbases; b++) {
        size_t blen = strlen(BASES[b]);
        for (int it = 0; it < 800; it++) {
            size_t mlen;
            uint8_t *m = fm_mutate((const uint8_t *)BASES[b], blen, &mlen);
            if (!m) continue;
            cases++;
            /* generous payload so a legitimately-parsed offset range is
             * in-bounds; mutated offsets are st.h's job to reject */
            if (write_container(dir, m, mlen, 16384) != 0) { free(m); continue; }
            free(m);

            pid_t pid = fork();
            if (pid == 0) {
                /* child: silence the abort-surface diagnostics, run parse */
                if (!freopen("/dev/null", "w", stderr)) { /* keep stderr */ }
                shards S;
                st_init(&S, dir);
                _exit(0);              /* accepted: fine */
            }
            int status = 0;
            waitpid(pid, &status, 0);
            if (WIFSIGNALED(status)) {
                int sig = WTERMSIG(status);
                fprintf(stderr,
                        "fuzz_st_header: child killed by signal %d "
                        "(base %zu iter %d) -- sanitizer report or crash\n",
                        sig, b, it);
                crashed++;
                if (crashed >= 3) {   /* fail fast once the pattern is clear */
                    fprintf(stderr, "too many crashes; stopping\n");
                    return 2;
                }
            }
        }
    }
    /* clean up the container file (dir left for post-mortem is unnecessary) */
    char path[512];
    snprintf(path, sizeof path, "%s/model.safetensors", dir);
    remove(path);
    rmdir(dir);

    if (crashed) { fprintf(stderr, "fuzz_st_header: %ld crashes\n", crashed); return 2; }
    printf("fuzz_st_header: %ld cases, no child crashed -- clean\n", cases);
    return 0;
}
#else
int main(void) {
    printf("fuzz_st_header: skipped (needs fork; use test_st_shape.c on Windows)\n");
    return 0;
}
#endif
