/* test_sha256.c -- the SHA-256 primitive against FIPS 180-4 / NIST CAVS vectors.
 *
 * #13's blob checksums are only worth anything if the hash is right, and a
 * hand-written crypto primitive is exactly the kind of code that looks correct
 * and is not. These are the published vectors, including the ones that catch
 * the two mistakes a short test cannot:
 *
 *   - the one-million-'a' case exercises multi-block carry and the 64-bit
 *     length field (a 32-bit length silently truncates past 512 MB);
 *   - the 56-and-64-byte cases sit exactly on the padding boundary, where an
 *     off-by-one appends a whole extra block or none at all.
 *
 * Also checked: streaming in awkward chunk sizes must equal one-shot hashing,
 * since the engine hashes tensors as they are read, never whole.
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "../sha256.h"

static int fails = 0;

static void hash_of(const void *p, size_t n, char hex[65]) {
    Sha256 s; sha256_init(&s);
    sha256_update(&s, p, n);
    unsigned char d[32]; sha256_final(&s, d);
    sha256_hex(d, hex);
}

static void expect(const char *msg, size_t n, const char *want, const char *tag) {
    char got[65]; hash_of(msg, n, got);
    if (strcmp(got, want)) {
        printf("FAIL %s:\n  got  %s\n  want %s\n", tag, got, want); fails++;
    }
}

int main(void) {
    /* FIPS 180-4 Appendix B + NIST CAVS */
    expect("", 0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "empty");
    expect("abc", 3,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "abc");
    expect("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", 56,
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1", "two-block (56B)");
    expect("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno"
           "ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu", 112,
        "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1", "112B");

    /* Padding-boundary lengths: 55 fits with the length field, 56 forces an
     * extra block, 64 is exactly one block. */
    { char a[64]; memset(a, 'a', sizeof a);
      expect(a, 55, "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318", "55 x 'a'");
      expect(a, 56, "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a", "56 x 'a'");
      expect(a, 64, "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb", "64 x 'a'"); }

    /* One million 'a': multi-block carry + the 64-bit length field. */
    { size_t n = 1000000; char *big = malloc(n);
      if (!big) { printf("FAIL: OOM for the 1M vector\n"); return 1; }
      memset(big, 'a', n);
      expect(big, n, "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "1M x 'a'");
      free(big); }

    /* Streaming must equal one-shot, at chunk sizes that straddle the 64-byte
     * block in every awkward way -- this is how the engine actually calls it. */
    { size_t n = 100000; unsigned char *b = malloc(n);
      if (!b) { printf("FAIL: OOM for the streaming case\n"); return 1; }
      for (size_t i = 0; i < n; i++) b[i] = (unsigned char)((i * 31 + 7) & 0xff);
      char one[65]; hash_of(b, n, one);
      const size_t chunks[] = {1, 3, 63, 64, 65, 127, 128, 1000, 4096};
      for (size_t c = 0; c < sizeof chunks / sizeof *chunks; c++) {
          Sha256 s; sha256_init(&s);
          for (size_t off = 0; off < n; off += chunks[c]) {
              size_t take = chunks[c]; if (off + take > n) take = n - off;
              sha256_update(&s, b + off, take);
          }
          unsigned char d[32]; sha256_final(&s, d);
          char got[65]; sha256_hex(d, got);
          if (strcmp(got, one)) {
              printf("FAIL streaming at chunk %zu:\n  got  %s\n  want %s\n", chunks[c], got, one);
              fails++;
          }
      }
      free(b); }

    /* A single flipped bit must change the digest -- the property the whole
     * feature rests on. */
    { unsigned char a[128], b2[128];
      for (int i = 0; i < 128; i++) a[i] = b2[i] = (unsigned char)i;
      b2[77] ^= 0x01;
      char ha[65], hb[65]; hash_of(a, sizeof a, ha); hash_of(b2, sizeof b2, hb);
      if (!strcmp(ha, hb)) { printf("FAIL: a flipped bit did not change the digest\n"); fails++; } }

    if (!fails) printf("test_sha256: OK (NIST vectors, padding boundaries, streaming, bit flip)\n");
    return fails ? 1 : 0;
}
