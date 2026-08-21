/* The phase counters are written from loader threads (#62 gate 2).
 *
 * coli_v4_block_profile_add() is called from profiled_expert_load_start and
 * _finish, which run on the dual-expert-loader and persistent-loader pthreads,
 * not just the main thread. The obvious implementation -- `double total +=
 * seconds` -- is a data race: it loses samples under contention and reports a
 * total that is simply too small, with nothing to indicate it. A profiler that
 * under-reports is worse than no profiler, because the missing time looks like
 * time some other phase did not spend.
 *
 * This drives the counters from many threads at once and checks the total is
 * EXACT, so the race would show up as a shortfall rather than as a plausible
 * number. Runs under ThreadSanitizer in test-tsan.
 */
#include <pthread.h>
#include <stdio.h>
#include <string.h>

#include "../deepseek_v4_internal.h"

#define THREADS 8
#define ADDS    20000
#define STEP_NS 1000            /* 1 us per add, exactly representable */

static void *hammer(void *arg) {
    (void)arg;
    for (int i = 0; i < ADDS; i++)
        coli_v4_block_profile_add(COLI_V4_BLOCK_PROFILE_LOADER_WAIT,
                                  (double)STEP_NS * 1e-9);
    return NULL;
}

static int failures;

static void check(const char *what, int ok) {
    if (!ok) { printf("  FAIL: %s\n", what); failures++; }
}

int main(void) {
    /* every kind starts at zero and is independent */
    for (int k = 0; k < COLI_V4_BLOCK_PROFILE_KINDS; k++)
        check("counters start at zero", coli_v4_block_profile_seconds(k) == 0.0);

    /* out-of-range and non-positive inputs are ignored, not recorded:
     * a negative span means the clock went backwards, and folding that into a
     * total silently cancels real time that was actually spent */
    coli_v4_block_profile_add(-1, 1.0);
    coli_v4_block_profile_add(COLI_V4_BLOCK_PROFILE_KINDS, 1.0);
    coli_v4_block_profile_add(COLI_V4_BLOCK_PROFILE_MOE_TOTAL, -1.0);
    coli_v4_block_profile_add(COLI_V4_BLOCK_PROFILE_MOE_TOTAL, 0.0);
    check("bad kind and non-positive spans are ignored",
          coli_v4_block_profile_seconds(COLI_V4_BLOCK_PROFILE_MOE_TOTAL) == 0.0);
    check("reading a bad kind returns zero, not garbage",
          coli_v4_block_profile_seconds(-1) == 0.0 &&
          coli_v4_block_profile_seconds(COLI_V4_BLOCK_PROFILE_KINDS) == 0.0);

    pthread_t t[THREADS];
    for (int i = 0; i < THREADS; i++)
        if (pthread_create(&t[i], NULL, hammer, NULL) != 0) {
            printf("  FAIL: cannot start thread %d\n", i);
            return 1;
        }
    for (int i = 0; i < THREADS; i++) pthread_join(t[i], NULL);

    /* exact, not approximate: THREADS * ADDS * 1us. A lost update under the
     * race would land BELOW this, which is the whole point of asserting
     * equality rather than a tolerance. */
    double expected = (double)THREADS * ADDS * STEP_NS * 1e-9;
    double got = coli_v4_block_profile_seconds(COLI_V4_BLOCK_PROFILE_LOADER_WAIT);
    if (got != expected) {
        printf("  FAIL: concurrent adds lost time: got %.9f, expected %.9f "
               "(%.0f of %d adds)\n", got, expected,
               got / (STEP_NS * 1e-9), THREADS * ADDS);
        failures++;
    }
    check("an unrelated kind is untouched by the hammering",
          coli_v4_block_profile_seconds(COLI_V4_BLOCK_PROFILE_ATTENTION) == 0.0);

    /* the clock is monotonic and usable for spans */
    double a = coli_v4_block_profile_now();
    double b = coli_v4_block_profile_now();
    check("profile clock does not go backwards", b >= a);

    if (failures) { printf("V4 block profile tests: %d FAILED\n", failures); return 1; }
    printf("V4 block profile tests: ok\n");
    return 0;
}
