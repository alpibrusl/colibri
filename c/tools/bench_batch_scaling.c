/* bench_batch_scaling — does batching amortize the fp8 dequantization?
 *
 * The post-merge baseline puts V4 decode at 80% compute, 20% I/O stall, and
 * the compute is instruction-bound: every fp8 byte needs decoding before it can
 * multiply, and at batch 1 each decoded weight is used exactly once. Arithmetic
 * intensity is ~0.5 FLOP/byte, the worst possible ratio.
 *
 * Batching should fix that: with S rows in flight, one decode of a weight feeds
 * S multiply-accumulates. This measures whether the real kernel actually does
 * that, on V4's own dense shape (I=4096, O=2048), by sweeping S and reporting
 * time PER TOKEN. Flat per-token cost would mean the kernel re-decodes per row
 * and batching buys nothing; falling cost is the amortization.
 *
 * MEASURE WITH OMP_WAIT_POLICY=active. This host pays ~65 us of fork/join per
 * parallel region under the default policy (see tests/bench_omp_grain), which
 * at S=1 is a quarter of the whole call. Batching amortizes that fixed cost and
 * it looks exactly like a batching win -- it reads as 2.5x by S=16 and is
 * entirely an artifact. Under a spin policy the real answer appears: 1.02x,
 * because the batch loop sits INSIDE the decode and e4m3_decode runs once per
 * (output row, batch row, element). See docs/experiments/v4-batching-gate.
 *
 *   make -C c bench-batch-scaling                 # default policy: misleading
 *   OMP_WAIT_POLICY=active make -C c bench-batch-scaling
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <math.h>
#include "../quant.h"

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
                         return t.tv_sec + t.tv_nsec*1e-9; }

int main(int argc, char **argv){
    const int I = argc>1 ? atoi(argv[1]) : 4096;
    const int O = argc>2 ? atoi(argv[2]) : 2048;
    const int reps = argc>3 ? atoi(argv[3]) : 12;
    const int nblk = ((O+127)/128) * (I/128);

    uint8_t *W  = malloc((size_t)I*O);
    float *scl  = malloc((size_t)nblk*sizeof(float));
    if(!W || !scl){ fprintf(stderr,"OOM\n"); return 1; }
    for(size_t i=0;i<(size_t)I*O;i++) W[i] = (uint8_t)((i*97u+13u) & 0xFF);
    for(int i=0;i<nblk;i++) scl[i] = 0.5f + (float)(i%7)*0.01f;

    printf("bench_batch_scaling: I=%d O=%d, %d reps, best-of\n", I, O, reps);
    printf("  %4s %11s %13s %9s %10s\n", "S", "total ms", "ms/token", "vs S=1", "GB/s");
    double base = 0;
    for(int S=1; S<=64; S*=2){
        float *x = malloc((size_t)S*I*sizeof(float));
        float *y = malloc((size_t)S*O*sizeof(float));
        if(!x||!y){ fprintf(stderr,"OOM at S=%d\n",S); return 1; }
        for(size_t i=0;i<(size_t)S*I;i++) x[i] = (float)((i%13)-6) * 0.125f;
        double best = 1e9;
        for(int r=0;r<reps;r++){
            double t0 = now();
            matmul_fp8(y, x, W, scl, S, I, O);
            double e = now()-t0;
            if(e<best) best = e;
        }
        double per_tok = best/S;
        if(S==1) base = per_tok;
        /* the weight matrix is read once per call regardless of S */
        double gbs = (double)I*O / best / 1e9;
        printf("  %4d %11.2f %13.3f %8.2fx %9.1f\n",
               S, best*1e3, per_tok*1e3, base/per_tok, gbs);
        free(x); free(y);
    }
    free(W); free(scl);
    return 0;
}
