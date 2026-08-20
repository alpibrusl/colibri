/* test_tier_origin.c -- demand-hit attribution against ESlot.origin (#32).
 *
 * tier_publish/tier_promote stamp origin; this pins the OTHER half of the
 * contract, moe_resolve_block's ecache-hit branch: a demand hit against a
 * PILOT- or COUPLE-published slot must count exactly once and CLEAR the
 * origin back to DEMAND, so a slot that stays resident across many demand
 * hits is credited on the first one only -- not on every access, which
 * would make g_pilot_hits/g_couple_hits count residency instead of the
 * fetch paying off.
 *
 * Includes colibri.c for moe_resolve_block itself (same pattern as
 * test_uring.c / test_pilot_ring.c): it is static, so this is the only way
 * to exercise it without a real model and checkpoint.
 */
#define main coli_glm_main_unused
#include "../colibri.c"
#undef main

static int fail(const char *s){ fprintf(stderr,"FAIL: %s\n",s); return 1; }

/* One layer, one slot, no VULKAN (portable build: moe_resolve_block takes no
 * vk_active/vk_hit params). */
static void setup(Model *m, ESlot *slots, int n){
    memset(m,0,sizeof(*m));
    m->tc.pin=calloc(1,sizeof(ESlot*)); m->tc.npin=calloc(1,sizeof(int));
    m->tc.ecache=calloc(1,sizeof(ESlot*)); m->tc.ecn=calloc(1,sizeof(int));
    m->tc.ecache[0]=slots; m->tc.ecn[0]=n; m->tc.ecap=n;
}

static void teardown(Model *m){
    free(m->tc.pin); free(m->tc.npin); free(m->tc.ecache); free(m->tc.ecn);
}

int main(void){
    int bad=0;

    /* Case 1: a COUPLE-published slot is hit once -- attributed and cleared. */
    {
        Model m; ESlot slots[1]={0};
        setup(&m,slots,1);
        sl_eid_set(&slots[0],7); sl_origin_set(&slots[0],TIER_ORIGIN_COUPLE);
        atomic_store(&g_couple_hits,0); atomic_store(&g_pilot_hits,0);

        int uniq[1]={7}; ESlot *use[1]; int missk[1],qof[1];
        int nmiss=moe_resolve_block(&m,0,uniq,0,1,use,missk,qof);

        if(nmiss!=0) bad|=fail("case1: expected a hit (nmiss==0)");
        if(use[0]!=&slots[0]) bad|=fail("case1: use[] did not point at the resident slot");
        if(atomic_load(&g_couple_hits)!=1) bad|=fail("case1: COUPLE hit not attributed");
        if(atomic_load(&g_pilot_hits)!=0) bad|=fail("case1: PILOT counter moved on a COUPLE slot");
        if(sl_origin(&slots[0])!=TIER_ORIGIN_DEMAND) bad|=fail("case1: origin not cleared after attribution");
        teardown(&m);
    }

    /* Case 2: same slot, hit a SECOND time (origin already cleared by case 1's
     * logic, replayed here fresh) -- must NOT double-count. */
    {
        Model m; ESlot slots[1]={0};
        setup(&m,slots,1);
        sl_eid_set(&slots[0],7); sl_origin_set(&slots[0],TIER_ORIGIN_DEMAND); /* already claimed */
        atomic_store(&g_couple_hits,0); atomic_store(&g_pilot_hits,0);

        int uniq[1]={7}; ESlot *use[1]; int missk[1],qof[1];
        moe_resolve_block(&m,0,uniq,0,1,use,missk,qof);

        if(atomic_load(&g_couple_hits)!=0) bad|=fail("case2: a cleared-origin slot re-attributed on a later hit");
        teardown(&m);
    }

    /* Case 3: PILOT origin attributes to g_pilot_hits, not g_couple_hits. */
    {
        Model m; ESlot slots[1]={0};
        setup(&m,slots,1);
        sl_eid_set(&slots[0],3); sl_origin_set(&slots[0],TIER_ORIGIN_PILOT);
        atomic_store(&g_couple_hits,0); atomic_store(&g_pilot_hits,0);

        int uniq[1]={3}; ESlot *use[1]; int missk[1],qof[1];
        moe_resolve_block(&m,0,uniq,0,1,use,missk,qof);

        if(atomic_load(&g_pilot_hits)!=1) bad|=fail("case3: PILOT hit not attributed");
        if(atomic_load(&g_couple_hits)!=0) bad|=fail("case3: COUPLE counter moved on a PILOT slot");
        if(sl_origin(&slots[0])!=TIER_ORIGIN_DEMAND) bad|=fail("case3: origin not cleared after attribution");
        teardown(&m);
    }

    /* Case 4: a genuinely miss experts still misses cleanly, and a resident
     * DEMAND-origin slot is a hit that moves neither counter (the ordinary
     * case: LRU keeping something warm, nothing ever speculated it). */
    {
        Model m; ESlot slots[1]={0};
        setup(&m,slots,1);
        sl_eid_set(&slots[0],5); sl_origin_set(&slots[0],TIER_ORIGIN_DEMAND);
        atomic_store(&g_couple_hits,0); atomic_store(&g_pilot_hits,0);

        int uniq[2]={5,9}; ESlot *use[2]; int missk[2],qof[2];
        m.ws[0]=(ESlot){0};
        int nmiss=moe_resolve_block(&m,0,uniq,0,2,use,missk,qof);

        if(nmiss!=1) bad|=fail("case4: expected exactly one miss (eid 9)");
        if(missk[0]!=1) bad|=fail("case4: wrong slot flagged as the miss");
        if(atomic_load(&g_pilot_hits)!=0 || atomic_load(&g_couple_hits)!=0)
            bad|=fail("case4: a plain DEMAND-origin hit moved an attribution counter");
        teardown(&m);
    }

    if(!bad) printf("test_tier_origin: ok -- origin attributed once and cleared on first "
                     "demand hit, PILOT/COUPLE counted separately, plain hits untouched\n");
    return bad;
}
