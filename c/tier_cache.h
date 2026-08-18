/* tier_cache.h -- lo stato di placement condiviso main/pilot, estratto dal
 * monolite (#10, primo taglio).
 *
 * Questo modulo possiede il VOCABOLARIO del tier RAM: lo slot di un expert
 * (ESlot), il suo protocollo di prenotazione/pubblicazione, la disciplina
 * atomica dei campi cross-thread (#9), lo stato di placement (LRU ecache,
 * hot-store pin, contatori heat/recency) e le due decisioni condivise --
 * "questo expert e' gia' servito?" (expert_resident_or_reserved) e "quale
 * slot sacrifico?" (pilot_pick_slot). I MECCANISMI che leggono e scrivono
 * questo stato (moe(), i pilot worker, rss_guard, repin) restano per ora in
 * colibri.c e vi accedono tramite Model.tc; migreranno qui a fette nelle
 * prossime tappe di #10.
 *
 * Contratto di inclusione (stile kv_persist.h): includere DOPO quant.h (QT)
 * e tier.h (tier_lfru_score). Nessun lock e' preso qui dentro: il locking e'
 * dei chiamanti (g_pilot_mx sui percorsi pilota; l'invariante
 * g_cur_moe_layer sul percorso demand di moe()).
 */
#ifndef TIER_CACHE_H
#define TIER_CACHE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* slot di un expert: pesi quantizzati + scale. Nel container pre-quantizzato g/u/d sono
 * VISTE dentro `slab` (una sola pread coalescente); nel fallback hanno buffer propri.
 * slab_cap/fslab_cap: capienza allocata — gli slot ws[] sono riusati TRA layer e gli
 * expert non hanno tutti la stessa taglia (layer MTP int8 = 2x i layer int4). */
typedef struct { int eid; QT g,u,d; uint8_t *slab; float *fslab;
                 int64_t slab_cap, fslab_cap; uint64_t used;
                 unsigned in_flight; /* async GPU readers borrowing this slot */
                 /* pin-arena backing (#419): when set, slab/fslab are interior
                  * slices of a per-layer arena and must never be free()d —
                  * expert_host_release detaches them, expert_host_ensure
                  * re-attaches. NULL for every individually-allocated slot. */
                 uint8_t *aslab; float *afslab; } ESlot;

static void eslot_acquire(ESlot *s){ __atomic_add_fetch(&s->in_flight,1,__ATOMIC_ACQ_REL); }
static void eslot_release(ESlot *s){
    unsigned old=__atomic_fetch_sub(&s->in_flight,1,__ATOMIC_ACQ_REL);
    if(!old){ fprintf(stderr,"[CUDA] ESlot reference underflow\n"); abort(); }
}
static int eslot_busy(const ESlot *s){ return __atomic_load_n(&s->in_flight,__ATOMIC_ACQUIRE)!=0; }
static void eslots_acquire(ESlot **slots,int n){ for(int i=0;i<n;i++) eslot_acquire(slots[i]); }
static void eslots_release(ESlot **slots,int n){ for(int i=0;i<n;i++) eslot_release(slots[i]); }

/* Disciplina dei campi cross-thread (#9). I pilot worker toccano gli slot del
 * layer L+1 sotto g_pilot_mx mentre moe() percorre quelli del layer L senza
 * lock -- layer disgiunti per l'invariante g_cur_moe_layer, ma la macchina
 * astratta C non lo sa: una load semplice che il compilatore puo' spezzare o
 * rileggere e' formalmente in gara con la store dell'altro lato non appena una
 * finestra (rss_guard, repin fra i turni) si sovrappone. Ogni accesso
 * cross-thread a eid/used/ecn e ai contatori heat/recency passa da questi
 * accessor RELAXED (disciplina READ_ONCE/WRITE_ONCE, la stessa dei cursori
 * del pool PIPE); l'ORDINAMENTO fra campi resta garantito da g_pilot_mx e dal
 * protocollo di prenotazione pubblicato, non da qui. I campi toccati solo dal
 * thread proprietario (gli slot ws[] di staging prima della pubblicazione)
 * restano accessi semplici. */
static inline int      sl_eid(const ESlot *s){ return __atomic_load_n(&s->eid,__ATOMIC_RELAXED); }
static inline void     sl_eid_set(ESlot *s,int v){ __atomic_store_n(&s->eid,v,__ATOMIC_RELAXED); }
static inline uint64_t sl_used(const ESlot *s){ return __atomic_load_n(&s->used,__ATOMIC_RELAXED); }
static inline void     sl_used_set(ESlot *s,uint64_t v){ __atomic_store_n(&s->used,v,__ATOMIC_RELAXED); }
static inline uint32_t u32_ld(const uint32_t *p){ return __atomic_load_n(p,__ATOMIC_RELAXED); }
static inline void     u32_st(uint32_t *p,uint32_t v){ __atomic_store_n(p,v,__ATOMIC_RELAXED); }

static int eslot_lru_victim(ESlot *slots,int n){
    int lru=-1;
    for(int i=0;i<n;i++) if(!eslot_busy(&slots[i])&&(lru<0||sl_used(&slots[i])<sl_used(&slots[lru]))) lru=i;
    return lru;
}

/* Lo stato di placement, per-modello. Vive dentro Model (campo `tc`) e ne
 * segue l'allocazione; questo struct e' il confine che i meccanismi devono
 * attraversare per toccarlo. */
typedef struct {
    ESlot **ecache; int *ecn; int ecap;          /* LRU expert per-layer */
    ESlot **pin; int *npin;                      /* HOT-STORE: expert pinnati in RAM (mai evicted) */
    uint32_t **eusage;                           /* contatori persistenti (per STATS/PIN) */
    uint32_t **eheat;                            /* calore recente per promotion/demotion live */
    uint32_t **elast; uint32_t eaccess_clock;    /* recency per LFRU session-local */
    /* DISK-CLASS: PRIVATE recency state, read only by expert_classify(). Private --
     * not the real elast/eaccess_clock -- kept fully separate so DISK-CLASS's bookkeeping
     * can never read from or write into stock eviction state: every DISK-CLASS write lives
     * inside its own need_classify/dc_on gate, so "byte-identical with PROF=0" is provable
     * by construction instead of by argument. (Historical note: when this was first written,
     * the Metal pre-routed FASE A path (g_pre_idx) never bumped the real elast/eaccess_clock
     * -- on Metal decode the real clock froze at end of prefill, so REPIN's LRU tie-breaker
     * ran on stale recency for the rest of the run. That was an upstream defect; it has since
     * been reported and fixed (#417, cfcc742) -- FASE A now bumps the real clock too. The
     * private clock is retained anyway: separation from stock state is the stronger property,
     * independent of whether the real clock is correct.) elast_dc/eaccess_clock_dc tick in
     * BOTH FASE A paths, under the same need_classify gate, at the same rate the real clock
     * ticks on the CPU path (one per selected (position,expert)) -- so the
     * COLI_DISKCLASS_WINDOW window keeps its meaning in every mode. elast_pre snapshots
     * elast_dc just BEFORE this call's own bump (see the touched[] guard in FASE A) --
     * classifying against the live array would read the bump routing just made a few lines
     * above the load that needed it, so a giant cold prefill burst would score every expert
     * "just accessed" and get called warm. Recency alone (not eheat's access COUNT): a count
     * never decays, so an expert hot early in a long session would keep reading "warm" long
     * after it dropped out of the working set. Same shape/allocation as elast; NULL for dense
     * layers. */
    uint32_t **elast_dc, **elast_pre; uint32_t eaccess_clock_dc;
    uint64_t eclock;                             /* LRU clock (used stamps) */
    /* sonda opzionale di un tier piu' veloce (registry VRAM Vulkan): "questo
     * expert verra' servito altrove, non caricarlo". NULL = nessun tier. */
    int (*ext_served)(int layer,int eid);
} TierCache;

/* Sonda di residenza (#9): eid e' pinnato, residente, o prenotato in volo su
 * `layer`? UNA definizione per lo scan che era ricopiato a mano in ogni punto
 * d'ingresso del prefetch (pilot_realload, pilot_uring_batch, couple_prefetch,
 * pilot_prefetch, repin) -- lo "scan di residenza sicuro per convenzione su ~6
 * siti" segnalato dall'audit. Il locking e' del CHIAMANTE: ogni chiamante
 * attuale tiene g_pilot_mx. */
static int expert_resident_or_reserved(TierCache *tc,int layer,int eid){
    if(tc->ext_served && tc->ext_served(layer,eid)) return 1;   /* VRAM-tier-served: no load */
    ESlot *P=tc->pin[layer];
    for(int z=0;z<tc->npin[layer];z++) if(sl_eid(&P[z])==eid) return 1;
    ESlot *Sl=tc->ecache[layer]; int nn=__atomic_load_n(&tc->ecn[layer],__ATOMIC_RELAXED);
    for(int z=0;z<nn;z++){ int e=sl_eid(&Sl[z]); if(e==eid||e==-(eid+2)) return 1; }
    return 0;
}

/* Scelta della vittima + LFRU eviction guard (#441, ristretto da #497): UNA
 * definizione per le due copie identiche di pilot_realload e pilot_uring_batch.
 * Ritorna l'indice dello slot, o -1 (tutti in volo, oppure il guard ha protetto
 * la vittima: il chiamante conta un drop in entrambi i casi). evict_guard e' il
 * flag PILOT_EVICT_GUARD del chiamante (0 = LRU pura, per gli A/B). Il
 * chiamante tiene g_pilot_mx. */
static int pilot_pick_slot(TierCache *tc,int layer,int eid,ESlot *Sl,int nn,int evict_guard){
    int slot=-1;
    for(int z=0;z<nn;z++){
        if(eslot_busy(&Sl[z])) continue;                 /* borrowed by an async GPU read */
        int e=sl_eid(&Sl[z]);
        if(e==-1){ slot=z; break; }                      /* riusa uno slot libero/fallito */
        if(e<-1) continue;                               /* prenotazione di un altro worker */
        if(slot<0 || sl_used(&Sl[z])<sl_used(&Sl[slot])) slot=z;
    }
    if(slot<0) return -1;
    /* proteggi la vittima solo se davvero CALDA (>=2 accessi demand) e chiaramente
     * piu' calda della speculazione, con l'isteresi 25%+4-freq di tier_pick_lfru */
    if(evict_guard && tc->eheat && tc->elast && sl_eid(&Sl[slot])>=0){
        int vid=sl_eid(&Sl[slot]); uint32_t vh=u32_ld(&tc->eheat[layer][vid]);
        if(vh>=2){
            uint32_t clk=__atomic_load_n(&tc->eaccess_clock,__ATOMIC_RELAXED);
            uint64_t vs=tier_lfru_score(vh,u32_ld(&tc->elast[layer][vid]),clk);
            uint64_t cs=tier_lfru_score(u32_ld(&tc->eheat[layer][eid]),u32_ld(&tc->elast[layer][eid]),clk);
            if(vs+(vs>>2)+(4u<<8)>cs) return -1;
        }
    }
    return slot;
}

#endif /* TIER_CACHE_H */
