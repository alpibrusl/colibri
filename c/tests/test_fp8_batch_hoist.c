/* matmul_fp8's hoisted-dequant path must be BIT-identical to the narrow path.
 *
 * #37 moved the dequantization out of the batch loop for S>=FP8_HOIST_MIN: a
 * block of four weight rows is decoded once and then the batch runs against it,
 * instead of e4m3_decode being re-evaluated for every (row, batch row, element).
 * Each output's accumulation order -- over i within a block, then over blocks --
 * is deliberately unchanged, so the results must match to the bit, not to a
 * tolerance.
 *
 * The oracle needs no second implementation: S below the threshold still runs
 * the original loop, so row s of a batched result must equal the S=1 result for
 * that same input row. Any reassociation shows up immediately.
 *
 * Shapes cover the block-remainder case (I not a multiple of FP8_BLOCK) and the
 * O-remainder arm (O not a multiple of 4), plus a batch that straddles
 * FP8_S_TILE so the tiling itself is exercised.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "../quant.h"

static int check(int I, int O, int S, unsigned seed) {
    int nblk = ((O + FP8_BLOCK - 1)/FP8_BLOCK) * (int)fp8_nblk(I);
    uint8_t *W = malloc((size_t)I*O);
    float *scl = malloc((size_t)nblk*sizeof(float));
    float *x   = malloc((size_t)S*I*sizeof(float));
    float *yb  = malloc((size_t)S*O*sizeof(float));   /* batched */
    float *y1  = malloc((size_t)O*sizeof(float));     /* one row at a time */
    if(!W||!scl||!x||!yb||!y1){ printf("OOM\n"); return 1; }

    unsigned st = seed;
    #define NEXT (st = st*1664525u + 1013904223u)
    for(size_t i=0;i<(size_t)I*O;i++) W[i] = (uint8_t)((NEXT>>16) & 0xFF);
    for(int i=0;i<nblk;i++)           scl[i] = 0.25f + (float)((NEXT>>20)&0x3F)*0.01f;
    /* full-entropy activations: dyadic values would make every sum exact and
     * the comparison vacuous, which is a trap this repo has hit before. */
    for(size_t i=0;i<(size_t)S*I;i++) x[i] = ((float)(NEXT>>8) / 8388608.0f) - 1.0f;

    matmul_fp8(yb, x, W, scl, S, I, O);

    int bad = 0;
    for(int s=0;s<S && bad<4;s++){
        matmul_fp8(y1, x + (size_t)s*I, W, scl, 1, I, O);
        for(int o=0;o<O;o++){
            if(memcmp(&yb[(size_t)s*O+o], &y1[o], sizeof(float)) != 0){
                printf("  MISMATCH I=%d O=%d S=%d row %d col %d: batched %.9g vs single %.9g\n",
                       I, O, S, s, o, yb[(size_t)s*O+o], y1[o]);
                if(++bad >= 4) break;
            }
        }
    }
    free(W); free(scl); free(x); free(yb); free(y1);
    return bad;
}

int main(void){
    struct { int I, O, S; } cases[] = {
        {4096, 2048,  4}, {4096, 2048,  8}, {4096, 2048, 16},
        {4096, 2048, 17},                       /* straddles FP8_S_TILE */
        {4096, 2048, 64},
        { 384,  256,  8},                       /* small, exact blocks */
        { 320,  256,  8},                       /* I not a multiple of 128 */
        { 384,  254,  8},                       /* O not a multiple of 4 */
        { 320,  254, 19},                       /* both remainders + tile straddle */
    };
    int fail = 0;
    for(unsigned c=0;c<sizeof cases/sizeof *cases;c++){
        int bad = check(cases[c].I, cases[c].O, cases[c].S, 0x9E3779B9u + c*7919u);
        printf("  I=%-5d O=%-5d S=%-3d %s\n", cases[c].I, cases[c].O, cases[c].S,
               bad ? "FAIL" : "bit-identical");
        fail |= bad;
    }
    puts(fail ? "test_fp8_batch_hoist: FAIL" : "test_fp8_batch_hoist: ok");
    return fail ? 1 : 0;
}
