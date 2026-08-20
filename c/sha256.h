/* SHA-256 (FIPS 180-4), self-contained.
 *
 * #13 wants blob checksums verified at load, and this engine's default CPU path
 * is dependency-free -- so the primitive lives here rather than in a library.
 * It is deliberately the plain reference construction: no SIMD, no unrolling
 * beyond the compression loop the spec itself writes that way. The bytes it
 * guards are read at NVMe speed and hashed at roughly a fifth of it, so the
 * interesting engineering is WHEN to verify (see coli_verify_tensor in st.h),
 * not how fast this is.
 *
 * Streaming API so a multi-GB tensor is hashed as it is read, never buffered
 * whole:
 *     Sha256 s; sha256_init(&s);
 *     sha256_update(&s, p, n);  ...
 *     unsigned char out[32]; sha256_final(&s, out);
 *
 * sha256_hex writes 64 lowercase hex chars + NUL.
 *
 * Validated against the FIPS 180-4 / NIST CAVS vectors in tests/test_sha256.c,
 * including the one-million-'a' case that catches length-encoding and
 * multi-block-carry mistakes a short vector cannot.
 */
#ifndef COLI_SHA256_H
#define COLI_SHA256_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>

typedef struct {
    uint32_t h[8];
    uint64_t len;            /* total message bytes, for the 64-bit length field */
    unsigned char buf[64];
    size_t n;                /* bytes currently held in buf */
} Sha256;

static const uint32_t SHA256_K[64] = {
    0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
    0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
    0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
    0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
    0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
    0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
    0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
    0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
};

#define SHA256_ROR(x,n) (((x) >> (n)) | ((x) << (32 - (n))))

static void sha256_block(Sha256 *s, const unsigned char *p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)p[i*4] << 24) | ((uint32_t)p[i*4+1] << 16) |
               ((uint32_t)p[i*4+2] << 8) | (uint32_t)p[i*4+3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = SHA256_ROR(w[i-15],7) ^ SHA256_ROR(w[i-15],18) ^ (w[i-15] >> 3);
        uint32_t s1 = SHA256_ROR(w[i-2],17) ^ SHA256_ROR(w[i-2],19)  ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=s->h[0],b=s->h[1],c=s->h[2],d=s->h[3],e=s->h[4],f=s->h[5],g=s->h[6],hh=s->h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = SHA256_ROR(e,6) ^ SHA256_ROR(e,11) ^ SHA256_ROR(e,25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = hh + S1 + ch + SHA256_K[i] + w[i];
        uint32_t S0 = SHA256_ROR(a,2) ^ SHA256_ROR(a,13) ^ SHA256_ROR(a,22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    s->h[0]+=a; s->h[1]+=b; s->h[2]+=c; s->h[3]+=d;
    s->h[4]+=e; s->h[5]+=f; s->h[6]+=g; s->h[7]+=hh;
}

static void sha256_init(Sha256 *s) {
    s->h[0]=0x6a09e667u; s->h[1]=0xbb67ae85u; s->h[2]=0x3c6ef372u; s->h[3]=0xa54ff53au;
    s->h[4]=0x510e527fu; s->h[5]=0x9b05688cu; s->h[6]=0x1f83d9abu; s->h[7]=0x5be0cd19u;
    s->len = 0; s->n = 0;
}

static void sha256_update(Sha256 *s, const void *data, size_t n) {
    const unsigned char *p = (const unsigned char *)data;
    s->len += (uint64_t)n;
    if (s->n) {                                  /* top up a partial block first */
        size_t take = 64 - s->n; if (take > n) take = n;
        memcpy(s->buf + s->n, p, take);
        s->n += take; p += take; n -= take;
        if (s->n == 64) { sha256_block(s, s->buf); s->n = 0; }
    }
    while (n >= 64) { sha256_block(s, p); p += 64; n -= 64; }
    if (n) { memcpy(s->buf, p, n); s->n = n; }
}

static void sha256_final(Sha256 *s, unsigned char out[32]) {
    uint64_t bits = s->len * 8;
    unsigned char pad = 0x80;
    sha256_update(s, &pad, 1);
    /* sha256_update bumped len; the padding zeros run until 56 mod 64 so the
     * 8-byte big-endian bit length lands at the end of a block. */
    unsigned char z = 0;
    while (s->n != 56) sha256_update(s, &z, 1);
    unsigned char lenbe[8];
    for (int i = 0; i < 8; i++) lenbe[i] = (unsigned char)(bits >> (56 - 8*i));
    sha256_update(s, lenbe, 8);
    for (int i = 0; i < 8; i++) {
        out[i*4]   = (unsigned char)(s->h[i] >> 24);
        out[i*4+1] = (unsigned char)(s->h[i] >> 16);
        out[i*4+2] = (unsigned char)(s->h[i] >> 8);
        out[i*4+3] = (unsigned char)(s->h[i]);
    }
}

static void sha256_hex(const unsigned char digest[32], char out[65]) {
    static const char hexd[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        out[i*2]   = hexd[digest[i] >> 4];
        out[i*2+1] = hexd[digest[i] & 15];
    }
    out[64] = 0;
}
#endif
