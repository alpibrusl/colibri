/* test_couple_drift.c -- the COUPLE table's live hit rate, and when it reports
 * itself stale (#32).
 *
 * couple_drift_tick() is the whole drift detector: it closes a window every
 * CP_DRIFT_WIN enqueues, folds that window's attributed-hit rate into an EWMA,
 * and warns once when the EWMA decays below COUPLE_DRIFT_MIN. This pins the
 * three properties the report depends on being honest:
 *
 *   - a table that is landing its hints never reports stale;
 *   - a table that is not does, and says so exactly once, not every window;
 *   - a window that reports more hits than it issued (the enqueue->attribution
 *     lag carrying across a boundary) is clamped, not allowed to push the EWMA
 *     above 1.0 and mask a later decay.
 *
 * Includes colibri.c for the tick and the counters themselves (same pattern as
 * test_tier_origin.c / test_pilot_ring.c): both are static, so this is the only
 * way to exercise them without a real model, a real table and a real trace.
 */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static int fail(const char *s){ fprintf(stderr,"FAIL: %s\n",s); return 1; }

/* Reset every piece of drift state, so cases do not inherit each other. */
static void reset(double dmin, double alpha){
    g_cp_ewma=-1.0; g_cp_win_enq=0; g_cp_win_base=0; g_cp_windows=0; g_cp_stale=0;
    g_cp_drift_min=dmin; g_cp_drift_a=alpha;
    atomic_store(&g_couple_hits,0);
    g_pilot_real=1;   /* the regime in which an origin is stamped at all -- case 6
                       * covers the other one */
}

/* Drive one full window: `hits` of its CP_DRIFT_WIN hints get attributed.
 * The hits are credited BEFORE the tick that closes the window, which is the
 * real ordering -- moe_resolve_block attributes on the demand path, and the
 * enqueue site ticks afterwards. */
static void window(int hits){
    for(int i=0;i<CP_DRIFT_WIN;i++){
        if(i<hits) atomic_fetch_add(&g_couple_hits,1);
        couple_drift_tick();
    }
}

int main(void){
    int bad=0;

    /* Case 1: nothing is reported before a window closes. A run too short to
     * measure must not produce a verdict -- the same discipline the offline
     * tools follow when a trace is too small. */
    {
        reset(0.15,0.25);
        for(int i=0;i<CP_DRIFT_WIN-1;i++) couple_drift_tick();
        if(g_cp_windows!=0) bad|=fail("case1: window closed early");
        if(g_cp_ewma>=0.0)  bad|=fail("case1: EWMA seeded before any evidence");
        if(g_cp_stale)      bad|=fail("case1: stale reported with no window closed");
    }

    /* Case 2: a table that lands most of its hints is healthy and stays silent. */
    {
        reset(0.15,0.25);
        for(int w=0;w<8;w++) window((int)(CP_DRIFT_WIN*0.6));
        if(g_cp_windows!=8) bad|=fail("case2: wrong window count");
        if(g_cp_stale)      bad|=fail("case2: a landing table reported stale");
        if(!(g_cp_ewma>0.5&&g_cp_ewma<0.7)) bad|=fail("case2: EWMA did not track a 60% rate");
    }

    /* Case 3: a table that lands almost nothing decays below the threshold and
     * reports -- exactly once, however many windows follow. */
    {
        reset(0.15,0.25);
        window((int)(CP_DRIFT_WIN*0.9));      /* seed high, so the decay is the EWMA's */
        if(g_cp_stale) bad|=fail("case3: stale on the seeding window");
        int first=-1;
        for(int w=0;w<40;w++){
            window(0);
            if(g_cp_stale && first<0) first=w;
        }
        if(first<0)     bad|=fail("case3: a dead table never reported stale");
        if(!g_cp_stale) bad|=fail("case3: stale flag did not latch");
        if(!(g_cp_ewma<0.15)) bad|=fail("case3: EWMA did not decay below the threshold");
    }

    /* Case 4: recovery re-arms, but only with margin -- a table hovering exactly
     * at the threshold must not alternate stale/healthy every window. */
    {
        reset(0.15,0.25);
        window(0); for(int w=0;w<8;w++) window(0);
        if(!g_cp_stale) bad|=fail("case4: setup did not reach stale");
        for(int w=0;w<20;w++) window((int)(CP_DRIFT_WIN*0.8));
        if(g_cp_stale) bad|=fail("case4: a recovered table stayed latched stale");
        if(!(g_cp_ewma>0.15*1.5)) bad|=fail("case4: recovery did not clear the hysteresis margin");
    }

    /* Case 5: a window credited with MORE hits than it issued (lag across the
     * boundary) is clamped to 1.0. Without the clamp the EWMA could be driven
     * above 1 and would then need extra windows to fall back through the
     * threshold -- a real decay would be reported late. */
    {
        reset(0.15,1.0);                      /* alpha=1: the EWMA is that window's rate */
        window(CP_DRIFT_WIN*2);
        if(g_cp_ewma>1.0) bad|=fail("case5: EWMA exceeded 1.0 on an over-credited window");
        if(!(g_cp_ewma>0.99)) bad|=fail("case5: clamp did not land at 1.0");
    }

    /* Case 6: COUPLE without PILOT_REAL. The worker takes the fadvise-only
     * expert_prefetch path, which never touches an ESlot, so no origin is ever
     * stamped and g_couple_hits is structurally pinned at 0. The detector must
     * stay silent rather than read that as a dead table -- otherwise every
     * fadvise-mode run reports a perfectly good table as stale. */
    {
        reset(0.15,0.25);
        g_pilot_real=0;
        for(int w=0;w<40;w++) window(0);
        if(g_cp_windows!=0) bad|=fail("case6: window closed with no origin-stamping path");
        if(g_cp_stale)      bad|=fail("case6: stale reported where hits cannot be attributed");
        if(g_cp_ewma>=0.0)  bad|=fail("case6: EWMA seeded from unattributable enqueues");
        g_pilot_real=1;
    }

    if(!bad) printf("test_couple_drift: OK\n");
    return bad;
}
