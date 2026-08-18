/* gpu_ops.h -- coli_gpu_ops: ONE backend boundary (#11, slice 1).
 *
 * The audit's GPU findings (#1/#2/#4): ~120 `#ifdef COLI_CUDA/COLI_VULKAN/
 * COLI_METAL` blocks weave three shape-incompatible vendor APIs through the
 * engine, and every new backend or engine multiplies that N x M editing
 * burden. This header is the seam: a small vtable of the operations the
 * ENGINE actually needs, in engine vocabulary (resident tensor handles,
 * expert groups), that each backend fills.
 *
 * Deliberately NOT here (and why): the CUDA PIPE device-resident decode
 * pipeline and the Metal fused layer command buffer are single-backend fast
 * paths -- forcing them through a common signature would either bloat this
 * table into a union of three vendor APIs (the audit's finding #1, merely
 * relocated) or lobotomize the fast paths. They stay native behind their
 * build flags and shrink as later slices find genuinely common shapes.
 *
 * Handles are `void *` on purpose: a resident tensor is backend property
 * (ColiVkTensor*, ColiCudaTensor*, ...); the engine only stores and returns
 * them. All ops return 0 on failure/unsupported so call sites keep their
 * historical "0 -> CPU fallback" shape.
 *
 * The `dev` parameter is the multi-device story: 0 is the primary device,
 * 1 the optional second tier device. For Vulkan this FOLDS the dev2 clone
 * family (`coli_vk_*2`, ~250 verbatim lines that had already drifted --
 * upload_tensor accepts fmt 7, upload_tensor_d2 doesn't) into one call
 * surface; deleting the clone bodies inside backend_vulkan.c is issue #11
 * bullet 4, unblocked by this seam.
 *
 * Inclusion contract (single-TU style, like tier_cache.h): include AFTER
 * the backend header(s) whose adapter you want compiled -- under
 * COLI_VULKAN this header defines `coli_vk_gpu_ops` and needs
 * backend_vulkan.h's prototypes in scope.
 */
#ifndef COLI_GPU_OPS_H
#define COLI_GPU_OPS_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct ColiGpuOps {
    const char *name;                     /* "vulkan", "cuda", "metal" */

    /* ---- lifecycle / capacity ---- */
    /* Is device `dev` initialized and usable? (0 = primary, 1 = tier dev2.) */
    int (*dev_available)(int dev);
    /* Device-local memory usage/budget in GB; 0 if the backend can't say. */
    int (*mem_budget)(int dev, double *used_gb, double *budget_gb);

    /* ---- resident tensor ---- */
    /* Upload-or-reuse a quantized tensor [O,I] (fmt/gs as in QT) on `dev`.
     * *t is the backend handle slot (NULL on first call). Returns 0 on
     * failure/unsupported fmt -> caller keeps its CPU copy authoritative. */
    int    (*tensor_ensure)(void **t, const void *weights, const float *scales,
                            int fmt, int I, int O, int gs, int dev);
    void   (*tensor_free)(void *t);
    size_t (*tensor_bytes)(const void *t);
    int    (*tensor_dev)(const void *t);  /* which device owns this handle */

    /* ---- dense matmul on a resident tensor ----
     * y[S,O] = x[S,I] @ dequant(W)^T; uploads on first call via *t. `dev`
     * was missing from slice 1's signature (Vulkan's single dense device
     * hid the gap) -- added in slice 5 once CUDA's per-tensor cuda_device
     * showed every OTHER family already carries it. Vulkan ignores it. */
    int (*matmul)(void **t, float *y, const float *x,
                  const void *weights, const float *scales,
                  int fmt, int S, int I, int O, int gs, int dev);

    /* ---- batched expert MLP (the moe() hot path) ----
     * For each expert c of n: y_c = down_c(silu(gate_c(x_c)) * up_c(x_c)),
     * x/y packed [sum(rows)*D]; g/u/d are resident handles on `dev`.
     * Sync form computes in place; async form is issue (one in flight per
     * device) + take (joins, reads back packed y). */
    int (*expert_group)(void *const *g, void *const *u, void *const *d,
                        const int *rows, int n, float *y, const float *x, int dev);
    int (*expert_group_issue)(void *const *g, void *const *u, void *const *d,
                              const int *rows, int n, const float *x, int dev);
    int (*expert_group_take)(float *y, int dev);

    /* ---- dense matmul pair ----
     * Two resident matmuls sharing one input x in ONE submit (the q_a + kv_a
     * attention prologue). Same fmt/gs/I on both. 0 -> per-matmul fallback. */
    int (*matmul_pair)(void **t1, float *y1, const void *w1, const float *s1, int O1,
                       void **t2, float *y2, const void *w2, const float *s2, int O2,
                       int fmt, const float *x, int S, int I, int gs);

    /* ---- decode-attention family (MLA absorb core, slice 2) ----
     * Device KV mirror: per-layer persistent latent/rope caches. ensure
     * allocates a layer's cache at max_rows (0 -> unsupported here); row
     * mirrors one host row at an absolute position; reset drops all layers
     * (the caller keeps a valid-watermark and re-mirrors). */
    int  (*kv_ensure)(int layer, int max_rows, int K, int Rd);
    int  (*kv_row)(int layer, int pos, const float *L, const float *R);
    void (*kv_reset)(void);
    /* Absorb attention over cache rows [st0,T): q [S,H*(Qn+R)] roped, kv_b
     * resident via *kvb, ctx out [S,H*V]. The _project variant fuses the
     * o-projection ([Dout,H*V], resident via *ot) so ctx never leaves the
     * device. 0 -> CPU path. */
    int (*attn_absorb)(void **kvb, const void *w, const float *sc, int fmt, int gs,
                       float *ctx, const float *q, int layer, int S, int H,
                       int Qn, int R, int V, int K, int st0, int T, float scale);
    int (*attn_absorb_project)(void **kvb, const void *w, const float *sc, int fmt, int gs,
                               void **ot, const void *ow, const float *osc, int ofmt, int ogs,
                               float *out, const float *q, int layer, int S, int H,
                               int Qn, int R, int V, int K, int st0, int T, float scale, int Dout);
    /* q-prep chain: [q_a+kv_a pair] -> rmsnorm(q latent) -> q_b in one
     * submit; lnw = q-latent RMS-norm weights [Oqa]; lat_out (NULLable)
     * receives the normed latent for the DSA indexer. 0 -> split path. */
    int (*attn_qprep)(int layer,
                      void **qa,  const void *wqa,  const float *sqa,  int Oqa,
                      void **kva, const void *wkva, const float *skva, int Okva,
                      void **qb,  const void *wqb,  const float *sqb,  int Oqb,
                      int fmt, int gs, const float *lnw, float eps,
                      const float *x, int S, int I,
                      float *q_out, float *kv_out, float *lat_out);
} ColiGpuOps;

/* ================= which table does a call site reach? =================
 * (#11, slice 6 -- the decision slice 5 deliberately left open.)
 *
 * Slice 5 stopped at a real fork in the road: `g_gops` was ONE pointer, bound
 * at whichever backend's init() ran. Vulkan bound it; CUDA binding it too
 * would let whichever backend initialized LAST silently replace the table the
 * OTHER backend's already-gated call sites still call through -- and because a
 * backend fills only the families whose shape it shares (CUDA leaves the
 * kv_ and attn_ family NULL by design, see below), "silently replace" means a
 * NULL function-pointer call, not a slow path.
 *
 * The decision, and the evidence for it: this engine never asks a runtime
 * "which backend is active?" question, so the seam must not invent one. Every
 * engine call site through this vtable sits inside `#ifdef COLI_VULKAN` (all
 * 36 of them when this was written) behind g_vulkan; the CUDA sites sit inside
 * `#ifdef COLI_CUDA` behind g_cuda_enabled. The backend is STATICALLY known
 * wherever the engine calls -- the preprocessor picked it, and the resident
 * handle passed in lives in that backend's own field (QT::vk vs QT::cuda). One
 * shared pointer answered a question nobody asked, and answered it wrong the
 * moment two backends were live at once.
 *
 * So: ONE SLOT PER BACKEND, bound by that backend at its own init, read by
 * that backend's own call sites. What this vtable buys is the op SHAPE -- one
 * signature per operation instead of three vendor APIs woven through the
 * engine -- not a single mutable binding. Genuinely backend-agnostic code
 * (bullet 5: the small engines reaching GPU through this seam) takes a
 * `const ColiGpuOps *` as a PARAMETER from a caller that knows which backend
 * it means; it never reads a global to find out.
 *
 * Two invariants bind time enforces, so a later edit cannot quietly
 * reintroduce the failure modes above:
 *   1. A table binds only into its OWN slot -- checked against ops->name, so
 *      coli_gops_bind(COLI_GPU_VULKAN, &coli_cuda_gpu_ops) is a rejected call
 *      instead of a corrupted engine.
 *   2. A bound slot is never silently REPLACED by a different table (binding
 *      the same table again is the idempotent no-op an init retry wants).
 * Plus a floor on sparseness (coli_gops_complete): lifecycle + the whole
 * resident-tensor family. Above that floor an unfilled op is NULL by contract
 * -- that is how CUDA declines the families whose shape it does not share --
 * and generic code asks COLI_GOPS_HAS before calling.
 *
 * The engine reads its slots through GOPS_VK / GOPS_CUDA (colibri.c): the call
 * site names its backend in the same breath its #ifdef already does.
 */
enum { COLI_GPU_VULKAN = 0, COLI_GPU_CUDA = 1, COLI_GPU_METAL = 2, COLI_GPU_BACKENDS = 3 };

/* Slot i belongs to the backend named here; ops->name must match to bind. */
static const char *const coli_gops_slot_name[COLI_GPU_BACKENDS] = { "vulkan", "cuda", "metal" };

static const ColiGpuOps *coli_gops_tbl[COLI_GPU_BACKENDS];

/* Non-NULL only for a backend that actually initialized. */
static inline const ColiGpuOps *coli_gops(int id){
    return (id >= 0 && id < COLI_GPU_BACKENDS) ? coli_gops_tbl[id] : NULL;
}

/* An op left NULL means "this backend does not do that" -- the question to ask
 * of any `const ColiGpuOps *` whose backend you do not know statically. */
#define COLI_GOPS_HAS(ops, fn) ((ops) != NULL && (ops)->fn != NULL)

/* The floor. A table missing any of these is not a sparse backend, it is an
 * incomplete one: every consumer of this seam assumes it can at least ask "is
 * this device up" and place, size, locate and free a resident tensor. */
static inline int coli_gops_complete(const ColiGpuOps *o){
    return o && o->name && o->dev_available && o->tensor_ensure &&
           o->tensor_free && o->tensor_bytes && o->tensor_dev;
}

/* 1 = the slot now holds `ops`. 0 = rejected with the slot untouched: bad id,
 * incomplete table, wrong slot for this backend, or a different table already
 * bound there. Call at backend init and treat 0 as fatal -- it can only mean
 * the caller wired up the wrong table, a build-time mistake wearing a
 * run-time disguise. */
static inline int coli_gops_bind(int id, const ColiGpuOps *ops){
    if(id < 0 || id >= COLI_GPU_BACKENDS) return 0;
    if(!coli_gops_complete(ops)) return 0;
    if(strcmp(ops->name, coli_gops_slot_name[id]) != 0) return 0;
    if(coli_gops_tbl[id] && coli_gops_tbl[id] != ops) return 0;
    coli_gops_tbl[id] = ops;
    return 1;
}

/* ======================= Vulkan adapter ======================= */
#ifdef COLI_VULKAN

static int vkops_dev_available(int dev){
    return dev ? coli_vk_dev2_available() : coli_vk_available();
}
static int vkops_mem_budget(int dev, double *u, double *b){
    return dev ? coli_vk_mem_budget2(u,b) : coli_vk_mem_budget(u,b);
}
static int vkops_tensor_ensure(void **t, const void *w, const float *s,
                               int fmt, int I, int O, int gs, int dev){
    return dev ? coli_vk_tensor_ensure2((ColiVkTensor**)t,w,s,fmt,I,O,gs)
               : coli_vk_tensor_ensure ((ColiVkTensor**)t,w,s,fmt,I,O,gs);
}
static void   vkops_tensor_free (void *t){ coli_vk_tensor_free((ColiVkTensor*)t); }
static size_t vkops_tensor_bytes(const void *t){ return coli_vk_tensor_bytes((const ColiVkTensor*)t); }
static int    vkops_tensor_dev  (const void *t){ return coli_vk_tensor_dev((const ColiVkTensor*)t); }
static int vkops_matmul(void **t, float *y, const float *x,
                        const void *w, const float *s,
                        int fmt, int S, int I, int O, int gs, int dev){
    (void)dev;   /* one dense device (G); dev2 is expert-tier only */
    return coli_vk_matmul((ColiVkTensor**)t,y,x,w,s,fmt,S,I,O,gs);
}
static int vkops_expert_group(void *const *g, void *const *u, void *const *d,
                              const int *rows, int n, float *y, const float *x, int dev){
    return dev ? coli_vk_expert_group2((ColiVkTensor *const*)g,(ColiVkTensor *const*)u,
                                       (ColiVkTensor *const*)d,rows,n,y,x)
               : coli_vk_expert_group ((ColiVkTensor *const*)g,(ColiVkTensor *const*)u,
                                       (ColiVkTensor *const*)d,rows,n,y,x);
}
static int vkops_expert_group_issue(void *const *g, void *const *u, void *const *d,
                                    const int *rows, int n, const float *x, int dev){
    return dev ? coli_vk_expert_group_issue2((ColiVkTensor *const*)g,(ColiVkTensor *const*)u,
                                             (ColiVkTensor *const*)d,rows,n,x)
               : coli_vk_expert_group_issue ((ColiVkTensor *const*)g,(ColiVkTensor *const*)u,
                                             (ColiVkTensor *const*)d,rows,n,x);
}
static int vkops_expert_group_take(float *y, int dev){
    return dev ? coli_vk_expert_group_take2(y) : coli_vk_expert_group_take(y);
}
static int vkops_matmul_pair(void **t1, float *y1, const void *w1, const float *s1, int O1,
                             void **t2, float *y2, const void *w2, const float *s2, int O2,
                             int fmt, const float *x, int S, int I, int gs){
    return coli_vk_matmul_pair((ColiVkTensor**)t1,y1,w1,s1,O1,
                               (ColiVkTensor**)t2,y2,w2,s2,O2,fmt,x,S,I,gs);
}
static int  vkops_kv_ensure(int layer,int max_rows,int K,int Rd){ return coli_vk_kv_ensure(layer,max_rows,K,Rd); }
static int  vkops_kv_row(int layer,int pos,const float *L,const float *R){ return coli_vk_kv_row(layer,pos,L,R); }
static void vkops_kv_reset(void){ coli_vk_kv_reset(); }
static int vkops_attn_absorb(void **kvb, const void *w, const float *sc, int fmt, int gs,
                             float *ctx, const float *q, int layer, int S, int H,
                             int Qn, int R, int V, int K, int st0, int T, float scale){
    return coli_vk_attention_absorb((ColiVkTensor**)kvb,w,sc,fmt,gs,ctx,q,layer,S,H,Qn,R,V,K,st0,T,scale);
}
static int vkops_attn_absorb_project(void **kvb, const void *w, const float *sc, int fmt, int gs,
                                     void **ot, const void *ow, const float *osc, int ofmt, int ogs,
                                     float *out, const float *q, int layer, int S, int H,
                                     int Qn, int R, int V, int K, int st0, int T, float scale, int Dout){
    return coli_vk_attention_absorb_project((ColiVkTensor**)kvb,w,sc,fmt,gs,
                                            (ColiVkTensor**)ot,ow,osc,ofmt,ogs,
                                            out,q,layer,S,H,Qn,R,V,K,st0,T,scale,Dout);
}
static int vkops_attn_qprep(int layer,
                            void **qa,  const void *wqa,  const float *sqa,  int Oqa,
                            void **kva, const void *wkva, const float *skva, int Okva,
                            void **qb,  const void *wqb,  const float *sqb,  int Oqb,
                            int fmt, int gs, const float *lnw, float eps,
                            const float *x, int S, int I,
                            float *q_out, float *kv_out, float *lat_out){
    return coli_vk_attn_qprep(layer,(ColiVkTensor**)qa,wqa,sqa,Oqa,
                              (ColiVkTensor**)kva,wkva,skva,Okva,
                              (ColiVkTensor**)qb,wqb,sqb,Oqb,
                              fmt,gs,lnw,eps,x,S,I,q_out,kv_out,lat_out);
}

static const ColiGpuOps coli_vk_gpu_ops = {
    .name               = "vulkan",
    .dev_available      = vkops_dev_available,
    .mem_budget         = vkops_mem_budget,
    .tensor_ensure      = vkops_tensor_ensure,
    .tensor_free        = vkops_tensor_free,
    .tensor_bytes       = vkops_tensor_bytes,
    .tensor_dev         = vkops_tensor_dev,
    .matmul             = vkops_matmul,
    .expert_group       = vkops_expert_group,
    .expert_group_issue = vkops_expert_group_issue,
    .expert_group_take  = vkops_expert_group_take,
    .matmul_pair        = vkops_matmul_pair,
    .kv_ensure          = vkops_kv_ensure,
    .kv_row             = vkops_kv_row,
    .kv_reset           = vkops_kv_reset,
    .attn_absorb        = vkops_attn_absorb,
    .attn_absorb_project= vkops_attn_absorb_project,
    .attn_qprep         = vkops_attn_qprep,
};

#endif /* COLI_VULKAN */

/* ======================= CUDA adapter (#11, slice 5) ======================= */
/* Fills the families whose shape genuinely matches CUDA's own API one-to-one:
 * lifecycle, resident tensor, dense matmul. `dev` throughout this adapter is
 * a CUDA ORDINAL (backend_cuda.h: "Devices are CUDA ordinals, not positions
 * in the input list") -- unlike Vulkan's dev in {0,1}, this is the same
 * ordinal callers already pass to qt_cuda_upload/coli_cuda_matmul via
 * QT.cuda_device, so no translation happens at this boundary.
 *
 * Deliberately UNFILLED here, and why (leaving these NULL is the honest
 * choice, not a placeholder to paper over later):
 *
 *   - expert_group / expert_group_issue / matmul_pair: CUDA's own multi-
 *     device group dispatch (colibri.c's dev_nc[]/dev_off[]/dev_total[]
 *     arrays in moe()) already fans a SINGLE routed batch out across up to
 *     COLI_CUDA_MAX_DEVICES devices with its own packing/sync -- that shape
 *     has no single-call equivalent in this vtable (built for Vulkan's
 *     one-or-two-device model) without reintroducing per-device loops
 *     *inside* the adapter, which would just relocate colibri.c's own
 *     dispatch logic rather than collapse it.
 *   - expert_group_take: a harder mismatch than "just wire it" -- Vulkan's
 *     take(y,dev) memcpy's into a caller buffer; coli_cuda_expert_group_take
 *     RETURNS a pointer to pinned device-visible memory (zero-copy, valid
 *     until the next issue on that device) with no byte count the adapter
 *     could safely memcpy without also being told rows*D. Bridging that
 *     needs a signature change (e.g. a length parameter) accepted
 *     deliberately, not smuggled into this slice.
 *   - kv_ensure/kv_row/kv_reset, attn_absorb(_project), attn_qprep: no CUDA
 *     equivalent shape exists AT ALL. CUDA's decode attention is a
 *     device-pointer, device-resident pipeline (coli_cuda_attention_absorb_*
 *     variants operating on m->kv_dev_R/latent/rope device buffers PIPE
 *     already manages) -- architecturally distinct from Vulkan's persistent
 *     per-layer mirror-and-fuse design, not a narrower version of it.
 *
 * WIRED into colibri.c as of slice 6, once the "which global(s)" question
 * slice 5 raised got its answer (see the registry comment above): CUDA binds
 * its own COLI_GPU_CUDA slot at coli_cuda_init success and the CUDA-gated call
 * sites read GOPS_CUDA, so the two backends can be live together without
 * either one's table standing where the other's call sites look. The families
 * routed through this table are exactly the filled ones -- the resident-tensor
 * family (including qt_cuda_upload's fmt==4 dispatch, which lived in colibri.c
 * AND here until this slice collapsed it to one copy), dense matmul, and
 * tensor_dev.
 *
 * Three CUDA sites stay native on purpose, and the reason is the same each
 * time -- their shape is not this vtable's shape, so routing them would change
 * behavior or lose information rather than collapse an #ifdef:
 *   - coli_cuda_mem_info in the expert-tier planner: mem_budget reports
 *     used/budget GB as doubles; the planner does byte arithmetic against
 *     g_cuda_dense_projected and CUDA_RESERVE_GB and needs FREE bytes. A
 *     round-trip through GB doubles to recover them would be a worse call than
 *     leaving one line native.
 *   - coli_cuda_tensor_upload_compressed (COLI_ANS) and coli_cuda_tensor_update:
 *     no member exists for either; an in-place refresh of an already-resident
 *     tensor has no Vulkan counterpart to share a signature with yet.
 *   - layer_cuda_shard_kvb's upload: it calls the GROUPED upload for every
 *     format, while qt_cuda_upload (and therefore this adapter) uses it only
 *     for fmt==4. Two live dispatch policies, one of which is presumably
 *     wrong; routing this site would silently pick a side on hardware nobody
 *     here can run. Named, not "fixed" blind -- see the issue. */
#ifdef COLI_CUDA

static int cuops_dev_available(int dev){
    int n = coli_cuda_device_count();
    for(int i=0;i<n;i++) if(coli_cuda_device_at(i)==dev) return 1;
    return 0;
}
static int cuops_mem_budget(int dev, double *used_gb, double *budget_gb){
    size_t free_b=0, total_b=0;
    if(!coli_cuda_mem_info(dev,&free_b,&total_b) || !total_b) return 0;
    if(used_gb) *used_gb = (double)(total_b-free_b)/1e9;
    if(budget_gb) *budget_gb = (double)total_b/1e9;
    return 1;
}
/* Mirrors qt_cuda_upload's own dispatch exactly (colibri.c:583): fmt==4 needs
 * the grouped-scale upload or its [O,ceil(I/gs)] scales get truncated to O
 * floats and the group kernels read garbage; every other format uses the
 * plain upload. Idempotent on repeat calls, like Vulkan's tensor_ensure (the
 * CUDA backend's own *tensor slot is the cache; a non-NULL *t is reused). */
static int cuops_tensor_ensure(void **t, const void *w, const float *s,
                               int fmt, int I, int O, int gs, int dev){
    ColiCudaTensor **ct = (ColiCudaTensor**)t;
    if(fmt==4) return coli_cuda_tensor_upload_g(ct,w,s,fmt,I,O,dev,gs);
    return coli_cuda_tensor_upload(ct,w,s,fmt,I,O,dev);
}
static void   cuops_tensor_free (void *t){ coli_cuda_tensor_free((ColiCudaTensor*)t); }
static size_t cuops_tensor_bytes(const void *t){ return coli_cuda_tensor_bytes((const ColiCudaTensor*)t); }
static int    cuops_tensor_dev  (const void *t){ return coli_cuda_tensor_device((const ColiCudaTensor*)t); }
static int cuops_matmul(void **t, float *y, const float *x,
                        const void *w, const float *s,
                        int fmt, int S, int I, int O, int gs, int dev){
    return coli_cuda_matmul((ColiCudaTensor**)t,y,x,w,s,fmt,S,I,O,dev,gs);
}

static const ColiGpuOps coli_cuda_gpu_ops = {
    .name               = "cuda",
    .dev_available      = cuops_dev_available,
    .mem_budget         = cuops_mem_budget,
    .tensor_ensure      = cuops_tensor_ensure,
    .tensor_free        = cuops_tensor_free,
    .tensor_bytes       = cuops_tensor_bytes,
    .tensor_dev         = cuops_tensor_dev,
    .matmul             = cuops_matmul,
    /* expert_group*, matmul_pair, kv_*, attn_* -- see the comment above:
     * left NULL deliberately, not resolved by a stub that would silently
     * segfault the first time colibri.c called through it. */
};

#endif /* COLI_CUDA */

#endif /* COLI_GPU_OPS_H */
