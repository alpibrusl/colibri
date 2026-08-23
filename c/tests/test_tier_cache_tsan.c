/* tier_cache.h under concurrency (#10): the reservation protocol and the
 * cross-thread accessor discipline (#9), driven WITHOUT the engine.
 *
 * First test of this machinery that does not `#include "../colibri.c"` --
 * the extraction is what makes that possible. QT is opaque to the module
 * (never dereferenced), so a stub stands in for the engine's weight views.
 *
 * The harness reproduces the two real access patterns:
 *   - N "pilot" threads: probe residency, pick a victim, RESERVE
 *     (eid=-(e+2), used=-1) under the lock, "load" outside it, PUBLISH
 *     (eid=e, fresh clock stamp) under the lock -- pilot_realload's shape.
 *   - one "demand" thread: UNLOCKED hit-scans stamping used on hits, plus
 *     heat/recency bumps through the u32 accessors -- moe()'s shape, the
 *     side that historically raced.
 *
 * Asserted: after every publish, exactly ONE slot in the layer carries that
 * eid (dedup + visible-reservation both worked); at the end, all resident
 * eids are unique and every slot is either free or a valid publication.
 * Under `make test-tsan` the same binary is the race gate: any torn or
 * invented access on the shared fields is a ThreadSanitizer report.
 */
typedef struct { int opaque; } QT;   /* engine weight views: opaque to the module */
#include "../tier.h"
#include "../tier_cache.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

#define NEXPERT 64
#define CAP      8
#define PILOTS   4
#define ITERS    20000

static TierCache tc;
static uint8_t slab_bytes[4];                   /* stand-in for real expert weights */
static ESlot cache_slots[CAP];
static ESlot *cache_layers[1] = { cache_slots };
static int ecn_arr[1];
static ESlot *pin_layers[1];
static int npin_arr[1] = { 0 };
static uint32_t heat[NEXPERT], last[NEXPERT];
static uint32_t *heat_layers[1] = { heat }, *last_layers[1] = { last };
static pthread_mutex_t mx = PTHREAD_MUTEX_INITIALIZER;
static _Atomic long published = 0, dropped = 0, deduped = 0;

static unsigned rng_next(unsigned *st) {
    *st ^= *st << 13; *st ^= *st >> 17; *st ^= *st << 5; return *st;
}

/* under mx: the invariant a broken reservation protocol violates first */
static void assert_unique(int eid) {
    int seen = 0;
    for (int z = 0; z < __atomic_load_n(&tc.ecn[0], __ATOMIC_RELAXED); z++)
        if (sl_eid(&cache_slots[z]) == eid) seen++;
    CHECK(seen == 1);
}

static void *pilot(void *arg) {
    unsigned st = 0x9E3779B9u ^ (unsigned)(size_t)arg;
    for (int i = 0; i < ITERS; i++) {
        int eid = (int)(rng_next(&st) % NEXPERT);
        pthread_mutex_lock(&mx);
        if (expert_resident_or_reserved(&tc, 0, eid)) {
            pthread_mutex_unlock(&mx);
            deduped++;
            continue;
        }
        int nn = __atomic_load_n(&tc.ecn[0], __ATOMIC_RELAXED);
        int slot;
        if (nn < tc.ecap) { slot = nn; __atomic_store_n(&tc.ecn[0], nn + 1, __ATOMIC_RELAXED); }
        else slot = pilot_pick_slot(&tc, 0, eid, cache_slots, nn, 1);
        if (slot < 0) { pthread_mutex_unlock(&mx); dropped++; continue; }
        ESlot *dst = &cache_slots[slot];
        sl_eid_set(dst, -(eid + 2));            /* visible reservation */
        sl_used_set(dst, (uint64_t)-1);         /* never an LRU victim while loading */
        pthread_mutex_unlock(&mx);

        /* the pread would happen here, outside the lock */
        dst->slab = slab_bytes;                 /* a resident owns a slab (#1034) */

        pthread_mutex_lock(&mx);
        sl_eid_set(dst, eid);                   /* publish */
        sl_used_set(dst, __atomic_add_fetch(&tc.eclock, 1, __ATOMIC_RELAXED));
        assert_unique(eid);
        pthread_mutex_unlock(&mx);
        published++;
    }
    return NULL;
}

static void *demand(void *arg) {
    (void)arg;
    unsigned st = 0xC0FFEEu;
    for (int i = 0; i < ITERS * 2; i++) {
        int eid = (int)(rng_next(&st) % NEXPERT);
        /* moe()'s unlocked hit path: scan, stamp used on hit */
        int nn = __atomic_load_n(&tc.ecn[0], __ATOMIC_RELAXED);
        for (int z = 0; z < nn; z++)
            if (sl_eid(&cache_slots[z]) == eid) {
                sl_used_set(&cache_slots[z],
                            __atomic_add_fetch(&tc.eclock, 1, __ATOMIC_RELAXED));
                break;
            }
        /* routing's heat/recency bumps, accessor-mediated */
        uint32_t h = u32_ld(&tc.eheat[0][eid]);
        if (h < UINT32_MAX) u32_st(&tc.eheat[0][eid], h + 1);
        u32_st(&tc.elast[0][eid],
               __atomic_add_fetch(&tc.eaccess_clock, 1, __ATOMIC_RELAXED));
    }
    return NULL;
}

int main(void) {
    tc.ecache = cache_layers; tc.ecn = ecn_arr; tc.ecap = CAP;
    tc.pin = pin_layers; tc.npin = npin_arr;
    tc.eheat = heat_layers; tc.elast = last_layers;
    tc.ext_served = NULL;
    for (int z = 0; z < CAP; z++) cache_slots[z].eid = -1;

    pthread_t pt[PILOTS], dt;
    for (long i = 0; i < PILOTS; i++)
        CHECK(pthread_create(&pt[i], NULL, pilot, (void *)(i + 1)) == 0);
    CHECK(pthread_create(&dt, NULL, demand, NULL) == 0);
    for (int i = 0; i < PILOTS; i++) pthread_join(pt[i], NULL);
    pthread_join(dt, NULL);

    /* end state: every slot free or a unique valid publication */
    int nn = tc.ecn[0];
    CHECK(nn <= CAP);
    for (int a = 0; a < nn; a++) {
        int ea = sl_eid(&cache_slots[a]);
        CHECK(ea == -1 || (ea >= 0 && ea < NEXPERT));
        if (ea < 0) continue;
        for (int b = a + 1; b < nn; b++) CHECK(sl_eid(&cache_slots[b]) != ea);
    }
    CHECK(published > 0 && deduped > 0);

    /* tier_promote (moe()'s end-of-block step): staged ws[] slots enter the
     * LRU by SWAP -- the promoted eids become resident exactly once, and the
     * evicted residents surface in the staging slots (ownership moves, no
     * slot is dropped). Single-threaded here by design: promotion runs on
     * the demand thread with the layer pilot-quiesced. */
    ESlot staged[2]; memset(staged, 0, sizeof staged);
    staged[0].eid = NEXPERT;        /* ids outside the run's range: countable */
    staged[1].eid = NEXPERT + 1;
    staged[0].slab = staged[1].slab = slab_bytes;   /* loaded experts own slabs */
    tier_promote(&tc, 0, staged, 2);
    nn = tc.ecn[0];
    int found0 = 0, found1 = 0;
    for (int z = 0; z < nn; z++) {
        int e = sl_eid(&cache_slots[z]);
        if (e == NEXPERT) found0++;
        if (e == NEXPERT + 1) found1++;
    }
    CHECK(found0 == 1 && found1 == 1);
    CHECK(sl_eid(&staged[0]) != NEXPERT && sl_eid(&staged[1]) != NEXPERT + 1);

    printf("test_tier_cache: ok — %ld published, %ld deduped, %ld dropped "
           "across %d pilots vs 1 demand thread; residents unique; "
           "tier_promote swap verified\n",
           (long)published, (long)deduped, (long)dropped, PILOTS);
    return 0;
}
