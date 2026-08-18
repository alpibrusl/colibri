/* test_gpu_ops_registry.c -- the backend-binding contract (#11, slice 6).
 *
 * The regression this pins: `g_gops` was one pointer bound by whichever
 * backend initialized last, so bringing CUDA up beside Vulkan would have left
 * Vulkan-gated call sites reading CUDA's table -- whose kv_ and attn_ members
 * are NULL by design. Here two tables are bound at once and each slot is
 * checked to still hold its own, which is the property that made wiring CUDA
 * into colibri.c safe.
 *
 * Engine-free on purpose (the tier_cache.h/test_tier_cache_tsan pattern):
 * gpu_ops.h compiles with no backend defined, so this runs in the portable
 * gate on every host -- no Vulkan loader, no CUDA runtime, no GPU.
 */
#include "../gpu_ops.h"

#include <stdio.h>

/* Two fakes shaped like the real tables: one complete (Vulkan's shape), one
 * sparse (CUDA's -- lifecycle + the resident-tensor family, the rest NULL). */
static int   fake_dev_available(int dev){ (void)dev; return 1; }
static int   fake_tensor_ensure(void **t, const void *w, const float *s,
                                int fmt, int I, int O, int gs, int dev){
    (void)w;(void)s;(void)fmt;(void)I;(void)O;(void)gs;(void)dev;
    if(t) *t=(void*)&fake_dev_available;
    return 1;
}
static void   fake_tensor_free (void *t){ (void)t; }
static size_t fake_tensor_bytes(const void *t){ (void)t; return 64; }
static int    fake_tensor_dev  (const void *t){ (void)t; return 0; }
static void   fake_kv_reset(void){ }

static const ColiGpuOps fake_vk = {
    .name          = "vulkan",
    .dev_available = fake_dev_available,
    .tensor_ensure = fake_tensor_ensure,
    .tensor_free   = fake_tensor_free,
    .tensor_bytes  = fake_tensor_bytes,
    .tensor_dev    = fake_tensor_dev,
    .kv_reset      = fake_kv_reset,   /* the family CUDA leaves NULL */
};
static const ColiGpuOps fake_cuda = {
    .name          = "cuda",
    .dev_available = fake_dev_available,
    .tensor_ensure = fake_tensor_ensure,
    .tensor_free   = fake_tensor_free,
    .tensor_bytes  = fake_tensor_bytes,
    .tensor_dev    = fake_tensor_dev,
};
/* A second, distinct table claiming the same backend -- stands in for a later
 * edit that binds a different table into an already-occupied slot. */
static const ColiGpuOps fake_cuda_other = {
    .name          = "cuda",
    .dev_available = fake_dev_available,
    .tensor_ensure = fake_tensor_ensure,
    .tensor_free   = fake_tensor_free,
    .tensor_bytes  = fake_tensor_bytes,
    .tensor_dev    = fake_tensor_dev,
};
/* Sparse BELOW the floor: no resident-tensor family at all. */
static const ColiGpuOps fake_incomplete = {
    .name          = "metal",
    .dev_available = fake_dev_available,
};

#define CHECK(cond) do{ if(!(cond)){ \
    fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#cond); return 1; } }while(0)

int main(void){
    /* Nothing is bound before a backend initializes, and an out-of-range id
     * is a NULL read rather than a walk off the end of the table. */
    CHECK(coli_gops(COLI_GPU_VULKAN)==NULL);
    CHECK(coli_gops(COLI_GPU_CUDA)==NULL);
    CHECK(coli_gops(COLI_GPU_METAL)==NULL);
    CHECK(coli_gops(-1)==NULL);
    CHECK(coli_gops(COLI_GPU_BACKENDS)==NULL);
    CHECK(!coli_gops_bind(-1,&fake_vk));
    CHECK(!coli_gops_bind(COLI_GPU_BACKENDS,&fake_vk));
    CHECK(!coli_gops_bind(COLI_GPU_VULKAN,NULL));

    /* A table binds into its own slot, and only its own: the wrong-slot call
     * is refused instead of quietly corrupting the engine. */
    CHECK(!coli_gops_bind(COLI_GPU_CUDA,&fake_vk));
    CHECK(!coli_gops_bind(COLI_GPU_VULKAN,&fake_cuda));
    CHECK(!coli_gops_bind(COLI_GPU_METAL,&fake_vk));
    CHECK(coli_gops(COLI_GPU_VULKAN)==NULL);   /* every refusal left it untouched */
    CHECK(coli_gops(COLI_GPU_CUDA)==NULL);

    /* A table that cannot even place a resident tensor is incomplete, not
     * sparse -- refused whatever slot it names. */
    CHECK(!coli_gops_bind(COLI_GPU_METAL,&fake_incomplete));
    CHECK(coli_gops(COLI_GPU_METAL)==NULL);

    /* THE REGRESSION: both backends live at once, neither slot disturbed by
     * the other's bind, in either order. */
    CHECK(coli_gops_bind(COLI_GPU_VULKAN,&fake_vk));
    CHECK(coli_gops(COLI_GPU_VULKAN)==&fake_vk);
    CHECK(coli_gops_bind(COLI_GPU_CUDA,&fake_cuda));
    CHECK(coli_gops(COLI_GPU_CUDA)==&fake_cuda);
    CHECK(coli_gops(COLI_GPU_VULKAN)==&fake_vk);   /* CUDA did not replace it */
    CHECK(coli_gops(COLI_GPU_METAL)==NULL);

    /* Re-binding the SAME table is the no-op an init retry needs; a DIFFERENT
     * table into an occupied slot is refused, slot unchanged. */
    CHECK(coli_gops_bind(COLI_GPU_CUDA,&fake_cuda));
    CHECK(!coli_gops_bind(COLI_GPU_CUDA,&fake_cuda_other));
    CHECK(coli_gops(COLI_GPU_CUDA)==&fake_cuda);

    /* The sparse contract: an unfilled op is NULL, and COLI_GOPS_HAS is how
     * backend-agnostic code asks before calling (bullet 5's small engines). */
    CHECK(COLI_GOPS_HAS(coli_gops(COLI_GPU_VULKAN),kv_reset));
    CHECK(!COLI_GOPS_HAS(coli_gops(COLI_GPU_CUDA),kv_reset));
    CHECK(!COLI_GOPS_HAS(coli_gops(COLI_GPU_METAL),tensor_ensure));  /* NULL table */
    CHECK(COLI_GOPS_HAS(coli_gops(COLI_GPU_CUDA),tensor_ensure));

    /* The floor really is callable on anything that bound. */
    for(int id=0;id<COLI_GPU_BACKENDS;id++){
        const ColiGpuOps *o=coli_gops(id);
        if(!o) continue;
        void *h=NULL;
        CHECK(o->dev_available(0));
        CHECK(o->tensor_ensure(&h,NULL,NULL,1,8,8,0,0));
        CHECK(h!=NULL);
        CHECK(o->tensor_bytes(h)==64);
        CHECK(o->tensor_dev(h)==0);
        o->tensor_free(h);
    }

    puts("gpu ops registry tests: ok");
    return 0;
}
