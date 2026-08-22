/* Store the RESIDENT fp8 weights as bf16, row-interleaved, and the LUT goes away.
 *
 *   cc -O3 -ffp-contract=off tools/bench_bf16_dense.c -o bench_bf16_dense
 *
 * Build with the same -ffp-contract=off the engine pins (#76), or the scalar
 * baseline fuses and the vector arm no longer matches it.
 *
 * matmul_fp8 costs two loads per MAC: the weight byte, and a dependent gather
 * into a 256-entry float table. NEON has no gather for that table, which is why
 * the interleaved-layout prototype only reached 1.3x -- it vectorised the
 * arithmetic and left the four table loads in place.
 *
 * e4m3 carries 3 mantissa bits; bf16 carries 7. So every e4m3 value is EXACTLY
 * representable in bf16, and bf16 -> fp32 is a 16-bit shift. Convert once at
 * load and the decode becomes one instruction on four values at a time, with
 * the identical fp32 operand reaching the multiply. Cost: 2 bytes per parameter
 * instead of 1.
 *
 * Contraction is pinned off tree-wide (#76), so the vector arm uses separate
 * multiply and add to match the scalar loop. Bit-identity is checked, not
 * assumed.
 */
#include <arm_neon.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+t.tv_nsec*1e-9;}
static float LUT[256];
#define BLOCK 128

/* shipped: fp8 bytes, four row streams, scalar LUT */
static void fp8_rows4(float *y,const float *x,const unsigned char *W,
                      const float *scl,int I,int O,int ng){
    for(int o=0;o+4<=O;o+=4){
        const unsigned char *w0=W+(size_t)o*I,*w1=w0+I,*w2=w1+I,*w3=w2+I;
        double a0=0,a1=0,a2=0,a3=0;
        for(int g=0;g<ng;g++){
            int b=g*BLOCK; float sc=scl[g],c0=0,c1=0,c2=0,c3=0;
            for(int i=b;i<b+BLOCK;i++){ float xi=x[i];
                c0+=LUT[w0[i]]*xi; c1+=LUT[w1[i]]*xi;
                c2+=LUT[w2[i]]*xi; c3+=LUT[w3[i]]*xi; }
            a0+=(double)c0*sc;a1+=(double)c1*sc;a2+=(double)c2*sc;a3+=(double)c3*sc;
        }
        y[o]=a0;y[o+1]=a1;y[o+2]=a2;y[o+3]=a3;
    }
}
/* bf16, four rows interleaved: one load, one shift, one mul, one add per column */
static void bf16_lane4(float *y,const float *x,const unsigned short *B,
                       const float *scl,int I,int O,int ng){
    for(int o=0;o+4<=O;o+=4){
        const unsigned short *p=B+(size_t)(o/4)*I*4;
        double a[4]={0,0,0,0};
        for(int g=0;g<ng;g++){
            int b=g*BLOCK; float sc=scl[g];
            float32x4_t c=vdupq_n_f32(0.0f);
            for(int i=b;i<b+BLOCK;i++){
                uint16x4_t raw=vld1_u16(p+(size_t)i*4);
                float32x4_t v=vreinterpretq_f32_u32(vshll_n_u16(raw,16));
                c=vaddq_f32(c,vmulq_n_f32(v,x[i]));   /* mul+add: contraction is
                                                         pinned off tree-wide */
            }
            float t[4]; vst1q_f32(t,c);
            for(int r=0;r<4;r++) a[r]+=(double)t[r]*sc;
        }
        for(int r=0;r<4;r++) y[o+r]=(float)a[r];
    }
}
int main(void){
    const int I=4096,O=2048,ng=I/BLOCK;
    for(int c=0;c<256;c++){ int e=(c>>3)&15,m=c&7; float v;
        if(e==0) v=(float)m/8.f*0.015625f;
        else v=(1.f+(float)m/8.f)*(float)(1<<(e>1?e-1:0))/64.f;
        LUT[c]=(c&128)?-v:v; }
    size_t wn=(size_t)I*O;
    unsigned char *W=malloc(wn);
    unsigned short *B=malloc(wn*2);
    float *x=malloc(I*4),*scl=malloc(ng*4),*y0=malloc(O*4),*y=malloc(O*4);
    unsigned s=17;
    for(size_t i=0;i<wn;i++){ s=s*1103515245u+12345u; W[i]=(unsigned char)(s>>16); }
    /* convert once, exactly: e4m3 -> fp32 -> top 16 bits */
    int inexact=0;
    for(int o=0;o<O;o++) for(int i=0;i<I;i++){
        float f=LUT[W[(size_t)o*I+i]];
        unsigned bits; memcpy(&bits,&f,4);
        if(bits & 0xFFFFu) inexact++;                 /* would lose mantissa bits */
        B[(size_t)(o/4)*I*4+(size_t)i*4+(o%4)] = (unsigned short)(bits>>16);
    }
    for(int i=0;i<I;i++){ s=s*1103515245u+12345u; unsigned b=(s&0x7FFFFFu)|0x3F000000u;
        float f; memcpy(&f,&b,4); x[i]=(s&0x80000000u)?-f:f; }
    for(int g=0;g<ng;g++) scl[g]=0.5f+g*0.001f;

    printf("e4m3 values needing mantissa bits below bf16: %d of %zu\n", inexact, wn);
    double t1=1e30,t2=1e30;
    for(int r=0;r<20;r++){ double a=now(); fp8_rows4(y0,x,W,scl,I,O,ng); double e=now()-a; if(e<t1)t1=e; }
    for(int r=0;r<20;r++){ double a=now(); bf16_lane4(y,x,B,scl,I,O,ng); double e=now()-a; if(e<t2)t2=e; }
    int diff=0; for(int i=0;i<O;i++) if(y[i]!=y0[i]) diff++;
    printf("\n%dx%d projection, single thread, best of 20\n",O,I);
    printf("  fp8 + LUT, 4 row streams (shipped)  %7.2f ms   1.00x\n",t1*1e3);
    printf("  bf16 interleaved, NEON lanes        %7.2f ms   %.2fx   rows differing: %d\n",
           t2*1e3,t1/t2,diff);
    printf("\n  memory: 1 byte/param -> 2. V4's resident dense set 6.27 -> 12.5 GiB.\n");
    return 0;
}
