/* test_castore.c -- the content-addressed container's C read path (#14).
 *
 * The store this reads is produced by alpibrusl/lex-moe's `moe-store` (Rust).
 * The point of the decision recorded on #14 is that colibri reimplements the
 * READ path in C rather than linking that crate, which means the two
 * implementations must agree about a format neither owns alone. These cases
 * pin the parts where they could silently drift, on a fixture built to the
 * layout moe-store writes:
 *
 *   <root>/manifests/<manifest-sha256>.json
 *   <root>/blobs/<first 2 hex>/<remaining 62 hex>
 *
 * Integrity needs no side table here, which is the format's whole argument: a
 * blob's NAME is the hash of its contents, so a corrupt blob fails to match the
 * address it was fetched from. #13 had to add colibri.sha256 to safetensors to
 * get the same property.
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include "../castore.h"

static int fails = 0;
#define CHECK(c) do{ if(!(c)){ printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); fails++; } }while(0)

static void mkdirp(const char *p){
#ifdef _WIN32
    mkdir(p);
#else
    mkdir(p, 0755);
#endif
}

static void hex_of(const void *p, size_t n, char out[65]){
    Sha256 s; sha256_init(&s); sha256_update(&s, p, n);
    unsigned char d[32]; sha256_final(&s, d); sha256_hex(d, out);
}

static void write_all(const char *path, const void *p, size_t n){
    FILE *f = fopen(path, "wb");
    if(!f){ printf("FAIL: cannot create %s (run from c/)\n", path); fails++; return; }
    fwrite(p, 1, n, f); fclose(f);
}

/* Build a store the way moe-store lays one out. Returns the manifest hash. */
static void build_store(const char *root, char manifest_hex[65], char blob_hex[65]){
    char p[1024];
    mkdirp(root);
    snprintf(p, sizeof p, "%s/blobs", root);     mkdirp(p);
    snprintf(p, sizeof p, "%s/manifests", root); mkdirp(p);

    /* 16 x 16 F32 = 1024 bytes: the manifest's shape and the blob's size must
     * agree, which st_init_ca checks (case 8) and this fixture must satisfy.
     * It did not before that check existed -- a 257-byte blob under a [16,16]
     * F32 shape -- and nothing noticed, which is the argument for the check. */
    static unsigned char payload[1024];
    for(size_t i=0;i<sizeof payload;i++) payload[i] = (unsigned char)((i*7+3) & 0xff);
    hex_of(payload, sizeof payload, blob_hex);

    snprintf(p, sizeof p, "%s/blobs/%.2s", root, blob_hex); mkdirp(p);
    snprintf(p, sizeof p, "%s/blobs/%.2s/%s", root, blob_hex, blob_hex + 2);
    write_all(p, payload, sizeof payload);

    char man[2048];
    int mn = snprintf(man, sizeof man,
        "{\"arch\":\"olmoe\",\"tensors\":{"
        "\"model.layers.1.mlp.experts.2.gate_proj.weight\":"
        "{\"blob\":\"%s\",\"dtype\":\"F32\",\"shape\":[16,16],"
        "\"role\":\"routed_expert\",\"layer\":1,\"expert\":2}}}", blob_hex);
    hex_of(man, (size_t)mn, manifest_hex);
    snprintf(p, sizeof p, "%s/manifests/%s.json", root, manifest_hex);
    write_all(p, man, (size_t)mn);
}

int main(void){
    const char *root = "tests/tmp_castore";
    char manifest_hex[65], blob_hex[65];
    build_store(root, manifest_hex, blob_hex);
    const char *NAME = "model.layers.1.mlp.experts.2.gate_proj.weight";

    /* 1. the manifest parses, and the fields the engine will need survive it */
    { ca_store S;
      CHECK(ca_open(&S, root, manifest_hex) == 0);
      CHECK(S.n == 1);
      CHECK(!strcmp(S.arch, "olmoe"));
      const ca_entry *e = ca_find(&S, NAME);
      CHECK(e != NULL);
      if(e){
          CHECK(!strcmp(e->blob, blob_hex));
          CHECK(!strcmp(e->dtype, "F32"));
          CHECK(e->rank == 2 && e->shape[0] == 16 && e->shape[1] == 16);
          CHECK(e->numel == 256);
          CHECK(e->layer == 1 && e->expert == 2);   /* optional fields, present here */
      }
      CHECK(ca_find(&S, "no.such.tensor") == NULL);
      CHECK(ca_nbytes(&S, NAME) == 1024);
      ca_close(&S); }

    /* 2. a blob reads back byte-exact, and the hash check passes */
    { ca_store S;
      CHECK(ca_open(&S, root, manifest_hex) == 0);
      static unsigned char buf[2048];
      int64_t n = ca_read(&S, NAME, buf, sizeof buf);
      CHECK(n == 1024);
      for(int i=0;i<1024;i++) if(buf[i] != (unsigned char)((i*7+3) & 0xff)){
          printf("FAIL: blob byte %d differs\n", i); fails++; break; }
      /* the destination bound is enforced, not assumed */
      CHECK(ca_read(&S, NAME, buf, 16) == -1);
      ca_close(&S); }

    /* 3. THE PROPERTY THE FORMAT EXISTS FOR: flip one bit in a blob and the
     *    address no longer describes it. No side table, no checksum map. */
    { char p[1024];
      snprintf(p, sizeof p, "%s/blobs/%.2s/%s", root, blob_hex, blob_hex + 2);
      FILE *f = fopen(p, "r+b");
      if(f){ fseek(f, 100, SEEK_SET); int c = fgetc(f); fseek(f, 100, SEEK_SET); fputc(c ^ 1, f); fclose(f); }

      ca_store S;
      CHECK(ca_open(&S, root, manifest_hex) == 0);
      static unsigned char buf[2048];
      CHECK(ca_read(&S, NAME, buf, sizeof buf) == -1);   /* refused */
      ca_close(&S);

      /* and the escape hatch lets a benchmark past it, as st.h's does.
       * The field is set directly rather than through the environment: setting
       * it that way does not reliably reach getenv in the same process on the
       * Windows CRT, which broke tests/test_blob_checksum there -- and then
       * broke this test the same way. What matters is the flag's BEHAVIOUR. */
      ca_store S2;
      CHECK(ca_open(&S2, root, manifest_hex) == 0);
      S2.no_verify = 1;
      CHECK(ca_read(&S2, NAME, buf, sizeof buf) == 1024);
      ca_close(&S2);

      if(f){ f = fopen(p, "r+b");                        /* restore */
             fseek(f, 100, SEEK_SET); int c = fgetc(f); fseek(f, 100, SEEK_SET); fputc(c ^ 1, f); fclose(f); } }

    /* 4. a TAMPERED MANIFEST is caught before any tensor is looked up -- the
     *    manifest is content-addressed too, so its own name is its checksum */
    { char p[1024], bad[2048];
      snprintf(p, sizeof p, "%s/manifests/%s.json", root, manifest_hex);
      int n = snprintf(bad, sizeof bad,
          "{\"arch\":\"evil\",\"tensors\":{\"%s\":"
          "{\"blob\":\"%s\",\"dtype\":\"F32\",\"shape\":[16,16],\"role\":\"dense\"}}}",
          NAME, blob_hex);
      char keep[2048]; int64_t klen = 0;
      char *orig = ca_slurp(p, &klen);
      if(orig && klen < (int64_t)sizeof keep){ memcpy(keep, orig, (size_t)klen); }
      write_all(p, bad, (size_t)n);

      ca_store S;
      CHECK(ca_open(&S, root, manifest_hex) == -1);      /* refused: hash disagrees */
      if(orig){ write_all(p, keep, (size_t)klen); free(orig); } }

    /* 5. malformed manifests are refused rather than half-loaded.
     *
     * Each is written under ITS OWN hash so the content-address check passes and
     * SHAPE validation is what refuses -- isolating the two without needing the
     * verify hatch, which is also what keeps this portable. */
    { const char *bads[] = {
          "{\"arch\":\"x\"}",                                           /* no tensors */
          "{\"arch\":\"x\",\"tensors\":{\"a\":{\"dtype\":\"F32\"}}}",   /* no blob */
          "{\"arch\":\"x\",\"tensors\":{\"a\":{\"blob\":\"nothex\"}}}", /* bad hash */
          "{\"arch\":\"x\",\"tensors\":{\"a\":{\"blob\":\"0123\"}}}",   /* short hash */
          "not json at all",
      };
      for(size_t i=0;i<sizeof bads/sizeof *bads;i++){
          char h[65], bp[1024];
          hex_of(bads[i], strlen(bads[i]), h);
          snprintf(bp, sizeof bp, "%s/manifests/%s.json", root, h);
          write_all(bp, bads[i], strlen(bads[i]));
          ca_store Sb;
          if(ca_open(&Sb, root, h) != -1){
              printf("FAIL: malformed manifest %zu was accepted\n", i); fails++; ca_close(&Sb); }
          unlink(bp);
      } }

    /* 7. st_init_ca: a CA manifest populates `shards`, and st.h's ordinary
     *    readers work against it unchanged. That is the whole point of the
     *    mapping -- one blob per tensor is st_tensor with off=0, nbytes = the
     *    file size, and no descriptor. */
    { shards S;
      CHECK(st_init_ca(&S, root, manifest_hex) == 0);
      CHECK(S.n == 1);
      CHECK(S.ca_root != NULL);
      st_tensor *t = st_find(&S, NAME);          /* the ordinary lookup, unchanged */
      CHECK(t != NULL);
      if(t){
          CHECK(t->ca_blob != NULL && !strcmp(t->ca_blob, blob_hex));
          CHECK(t->fd == -1);                    /* no descriptor is held */
          CHECK(t->off == 0);                    /* a blob IS the tensor */
          CHECK(t->nbytes == 1024);
          CHECK(st_nbytes(&S, NAME) == 1024);
      }
      CHECK(st_find(&S, "no.such.tensor") == NULL);

      /* st_read_raw goes through st_pread_tensor, which opens the blob for the
       * read and closes it after -- the caller never sees the difference */
      static unsigned char buf[2048];
      st_read_raw(&S, NAME, buf, 0);
      for(int i=0;i<1024;i++) if(buf[i] != (unsigned char)((i*7+3) & 0xff)){
          printf("FAIL: st_read_raw returned wrong bytes at %d\n", i); fails++; break; }

      /* drop=1 must not posix_fadvise a -1 descriptor */
      st_read_raw(&S, NAME, buf, 1);
      CHECK(buf[0] == (unsigned char)3);
    }

    /* 8. a manifest whose declared shape disagrees with the blob it points at is
     *    refused: numel comes from the manifest and nbytes from the file, and a
     *    caller sizing a buffer from the config would otherwise overrun it. */
    { char man[2048], h[65], p2[1024];
      int mn = snprintf(man, sizeof man,
          "{\"arch\":\"olmoe\",\"tensors\":{\"%s\":"
          "{\"blob\":\"%s\",\"dtype\":\"F32\",\"shape\":[999,999],\"role\":\"dense\"}}}",
          NAME, blob_hex);
      hex_of(man, (size_t)mn, h);
      snprintf(p2, sizeof p2, "%s/manifests/%s.json", root, h);
      write_all(p2, man, (size_t)mn);
      shards S;
      CHECK(st_init_ca(&S, root, h) == -1);
      unlink(p2); }

    /* 6. a hash that is not 64 lowercase hex is refused up front */
    { ca_store S;
      CHECK(ca_open(&S, root, "NOTAHASH") == -1);
      CHECK(ca_open(&S, root, "0123456789abcdef") == -1); }

    { char p[1024];
      snprintf(p, sizeof p, "%s/manifests/%s.json", root, manifest_hex); unlink(p);
      snprintf(p, sizeof p, "%s/blobs/%.2s/%s", root, blob_hex, blob_hex + 2); unlink(p);
      snprintf(p, sizeof p, "%s/blobs/%.2s", root, blob_hex); rmdir(p);
      snprintf(p, sizeof p, "%s/blobs", root); rmdir(p);
      snprintf(p, sizeof p, "%s/manifests", root); rmdir(p);
      rmdir(root); }

    if(!fails) printf("test_castore: OK\n");
    return fails ? 1 : 0;
}
