/* cfse_pack bounds validation: parse_shard and do_cert against hostile
 * data_offsets.
 *
 * cfse_pack shares st.h's threat model -- it parses untrusted safetensors
 * containers -- but historically had none of st.h's guards: offsets were
 * raw double->int64 casts (UB on NaN/inf/>=2^63), b<a wrapped rn=b-a to a
 * huge size_t, and out-of-file offsets indexed past the heap buffer holding
 * the file (in+ds+a). This gates the off_pair_ok fix.
 *
 * Same white-box pattern as the engine tests: include the tool with its
 * main renamed and drive the static functions directly.
 */
#define main coli_cfse_main_unused
#include "../cfse_pack.c"
#undef main
#ifndef _WIN32
#include <unistd.h>
#else
#include <direct.h>
#define rmdir _rmdir
#endif

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

/* build an in-memory container: 8-byte LE header length + header + payload */
static size_t build(char *out, size_t cap, const char *offs, size_t payload) {
    char header[512];
    int hn = snprintf(header, sizeof(header),
                      "{\"t\":{\"dtype\":\"U8\",\"shape\":[16],\"data_offsets\":%s}}",
                      offs);
    if (hn < 0 || (size_t)hn >= sizeof(header)) return 0;
    size_t total = 8 + (size_t)hn + payload;
    if (total > cap) return 0;
    uint64_t hlen = (uint64_t)hn;
    memcpy(out, &hlen, 8);
    memcpy(out + 8, header, (size_t)hn);
    memset(out + 8 + hn, 0, payload);
    return total;
}

static int parse_ok(const char *offs, size_t payload) {
    static char buf[4096];
    size_t n = build(buf, sizeof(buf), offs, payload);
    if (!n) return -2;
    jval *root; char *arena; TEnt *E; int NE; size_t ds;
    /* parse_shard mutates nothing in buf, but takes char* */
    return parse_shard(buf, n, &root, &arena, &E, &NE, &ds) == 0 ? 1 : 0;
}

int main(void) {
    /* accept: well-formed offsets covering the payload exactly */
    CHECK(parse_ok("[0,16]", 16) == 1);
    /* refuse: every shape of hostile data_offsets */
    CHECK(parse_ok("[16,0]", 16) == 0);                    /* b < a: rn would wrap */
    CHECK(parse_ok("[0,17]", 16) == 0);                    /* past EOF */
    CHECK(parse_ok("[-8,8]", 16) == 0);                    /* negative */
    CHECK(parse_ok("[0,nan]", 16) == 0);                   /* cast would be UB */
    CHECK(parse_ok("[0,1e300]", 16) == 0);                 /* far beyond int64 */
    CHECK(parse_ok("[0,9223372036854775807]", 16) == 0);   /* >= 2^63 as double */
    CHECK(parse_ok("[0,15.5]", 16) == 0);                  /* fractional */
    CHECK(parse_ok("[0,\"16\"]", 16) == 0);                /* wrong type */

    /* do_cert on a crafted file must refuse (return 1), not read OOB */
    char dir[] = "test_cfse_bounds_XXXXXX";
    CHECK(mkdtemp(dir) != NULL);
    char path[512];
    snprintf(path, sizeof(path), "%s/evil.safetensors", dir);
    static char buf[4096];
    size_t n = build(buf, sizeof(buf), "[16,0]", 16);
    CHECK(n > 0);
    FILE *f = fopen(path, "wb");
    CHECK(f && fwrite(buf, 1, n, f) == n && fclose(f) == 0);
    CHECK(do_cert(path) == 1);
    /* and a header length beyond EOF must refuse too */
    uint64_t evil_hlen = (uint64_t)1 << 62;
    f = fopen(path, "wb");
    CHECK(f && fwrite(&evil_hlen, 8, 1, f) == 1 && fclose(f) == 0);
    CHECK(do_cert(path) == 1);
    remove(path);
    rmdir(dir);

    printf("OK cfse_pack bounds: hostile data_offsets refused in parse_shard and do_cert\n");
    return 0;
}
