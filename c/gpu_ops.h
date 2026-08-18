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
     * y[S,O] = x[S,I] @ dequant(W)^T; uploads on first call via *t. */
    int (*matmul)(void **t, float *y, const float *x,
                  const void *weights, const float *scales,
                  int fmt, int S, int I, int O, int gs);

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
} ColiGpuOps;

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
                        int fmt, int S, int I, int O, int gs){
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
};

#endif /* COLI_VULKAN */

#endif /* COLI_GPU_OPS_H */
