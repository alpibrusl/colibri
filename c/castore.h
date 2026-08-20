/* Content-addressed expert container: the C read path (#14).
 *
 * An alternative container backend beside safetensors shards. Tensors are
 * stored as sha256-addressed, expert-granular blobs; a manifest maps tensor
 * name -> blob hash plus dtype/shape/role. The reference implementation is
 * alpibrusl/lex-moe's `moe-store` crate, and this reads what it writes.
 *
 * WHY C AND NOT FFI (decided on #14): "pure C with zero engine dependencies" is
 * a property this engine is built around -- README's first line, CONTRIBUTING's
 * ask of every change, and a checkbox on every PR in this fork. Linking the Rust
 * crate, even behind a build flag, would make CA containers readable only where
 * a Rust toolchain exists: a format the engine can only sometimes read is not
 * really a container backend. The cost is two implementations of one format,
 * kept in sync by discipline; moe-store stays the reference, and any divergence
 * is a bug here rather than a fork of the format.
 *
 * Only the READ path lives in C. Ingest, GC, delta encoding and the network tier
 * stay out-of-process in Rust: they are offline, off the hot path, and nothing
 * about them needs to be in the engine.
 *
 * ## On-disk layout (moe-store)
 *
 *   <root>/manifests/<manifest-sha256>.json
 *   <root>/blobs/<first 2 hex>/<remaining 62 hex>
 *
 * The manifest is JSON:
 *
 *   { "arch": "olmoe",
 *     "tensors": { "<name>": { "blob": "<64 hex>", "dtype": "F32",
 *                              "shape": [256, 64], "role": "dense",
 *                              "layer": 0, "expert": 3 }, ... } }
 *
 * `layer` and `expert` are omitted where they do not apply, `config`,
 * `tokenizer` and `chat_template` are optional top-level blob hashes.
 *
 * ## Integrity is free here
 *
 * #13 had to add a checksum map to safetensors containers because nothing in
 * that format binds bytes to a name. Here the blob's NAME is the hash of its
 * contents, so verification needs no side table: read the blob, hash it, compare
 * to the path it came from. ca_read does this by default; CA_NO_VERIFY (or the
 * shared COLI_NO_VERIFY) skips it for the same benchmarking reason st.h's does.
 *
 * ## Not in this cut
 *
 * Populating `shards` so the existing loaders read a CA container directly. That
 * is the obvious next step -- one blob per tensor maps onto st_tensor as
 * (fd, off=0, nbytes) almost exactly -- but it needs an answer for file
 * descriptors first: `shards` holds a fixed fds[512], and a GLM-scale model is
 * ~120k tensors, so it needs a bounded fd pool whose eviction cannot close a
 * descriptor a caller still holds. That is a design question, not a line of
 * plumbing, and it does not belong bolted onto the format reader.
 *
 * Also out of scope, as this issue states: the network protocol, and GC.
 */
#ifndef COLI_CASTORE_H
#define COLI_CASTORE_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include "json.h"
#include "sha256.h"
#include "compat.h"
#include "st.h"

#define CA_HEX 64
#define CA_MAX_TENSORS (1 << 20)     /* same ceiling as ST_FMT_STAMP_MAX: far
                                      * beyond any real model, bounds a hostile
                                      * manifest's persistent allocation */

typedef struct {
    char *name;
    char  blob[CA_HEX + 1];          /* lowercase hex, NUL-terminated */
    char *dtype;                     /* "F32", "BF16", "U8", ... as the manifest spells it */
    int64_t shape[8];
    int     rank;
    int64_t numel;
    int     layer, expert;           /* -1 when the manifest omits them */
} ca_entry;

typedef struct {
    char     *root;
    char     *arch;
    ca_entry *t;
    int       n, cap;
    int      *hidx;                  /* name -> index, open addressing */
    int       hcap;
    int       no_verify;
} ca_store;

static uint64_t ca_hash(const char *s) {
    uint64_t h = 1469598103934665603ULL;
    while (*s) { h ^= (unsigned char)*s++; h *= 1099511628211ULL; }
    return h;
}

static int ca_is_hex64(const char *v) {
    int n = 0;
    for (; v[n]; n++) {
        char c = v[n];
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) return 0;
    }
    return n == CA_HEX;
}

/* <root>/blobs/<aa>/<rest> into `out`. Returns 0 on success. */
static int ca_blob_path(const ca_store *S, const char *hex, char *out, size_t cap) {
    if (!ca_is_hex64(hex)) return -1;
    int n = snprintf(out, cap, "%s/blobs/%.2s/%s", S->root, hex, hex + 2);
    return (n > 0 && (size_t)n < cap) ? 0 : -1;
}

static int ca_idx(const ca_store *S, const char *name) {
    if (!S->hidx) return -1;
    uint64_t h = ca_hash(name) & (uint64_t)(S->hcap - 1);
    while (S->hidx[h] >= 0) {
        int i = S->hidx[h];
        if (!strcmp(S->t[i].name, name)) return i;
        h = (h + 1) & (uint64_t)(S->hcap - 1);
    }
    return -1;
}

static void ca_index_build(ca_store *S) {
    S->hcap = 1; while (S->hcap < S->n * 2) S->hcap <<= 1;
    S->hidx = (int *)malloc((size_t)S->hcap * sizeof(int));
    if (!S->hidx) { fprintf(stderr, "OOM indexing %d manifest entries\n", S->n); exit(1); }
    for (int i = 0; i < S->hcap; i++) S->hidx[i] = -1;
    for (int i = 0; i < S->n; i++) {
        uint64_t h = ca_hash(S->t[i].name) & (uint64_t)(S->hcap - 1);
        while (S->hidx[h] >= 0) h = (h + 1) & (uint64_t)(S->hcap - 1);
        S->hidx[h] = i;
    }
}

static const ca_entry *ca_find(const ca_store *S, const char *name) {
    int i = ca_idx(S, name);
    return i >= 0 ? &S->t[i] : NULL;
}

/* Whole file into a malloc'd buffer. *len gets the size. NULL on failure. */
static char *ca_slurp(const char *path, int64_t *len) {
    int fd = open(path, COMPAT_O_RDONLY);
    if (fd < 0) return NULL;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0) { close(fd); return NULL; }
    int64_t n = (int64_t)st.st_size;
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { close(fd); return NULL; }
    int64_t got = 0;
    while (got < n) {
        ssize_t r = pread(fd, buf + got, (size_t)(n - got), got);
        if (r <= 0) { free(buf); close(fd); return NULL; }
        got += r;
    }
    close(fd);
    buf[n] = 0;
    if (len) *len = n;
    return buf;
}

/* Parse <root>/manifests/<hex>.json. Returns 0 on success, -1 on failure
 * (message on stderr). TRUST-VERIFY-REFUSE: a manifest that is malformed, or
 * names a blob that is not a 64-char lowercase hex digest, is refused rather
 * than partially accepted -- a container the engine cannot fully understand must
 * not be half-loaded. */
static int ca_open(ca_store *S, const char *root, const char *manifest_hex) {
    memset(S, 0, sizeof(*S));
    if (!ca_is_hex64(manifest_hex)) {
        fprintf(stderr, "castore: '%s' is not a 64-char lowercase hex manifest hash\n", manifest_hex);
        return -1;
    }
    { const char *nv = getenv("CA_NO_VERIFY");
      if (!nv) nv = getenv("COLI_NO_VERIFY");
      S->no_verify = (nv && atoi(nv)) ? 1 : 0; }

    char path[4096];
    if (snprintf(path, sizeof path, "%s/manifests/%s.json", root, manifest_hex) >= (int)sizeof path) {
        fprintf(stderr, "castore: store path too long\n"); return -1; }
    int64_t mlen = 0;
    char *text = ca_slurp(path, &mlen);
    if (!text) { fprintf(stderr, "castore: cannot read manifest %s\n", path); return -1; }

    /* The manifest is content-addressed like everything else: its name is the
     * hash of its bytes, so a tampered manifest is caught before a single tensor
     * is looked up. This is the whole argument for the format in one check. */
    if (!S->no_verify) {
        Sha256 sh; sha256_init(&sh);
        sha256_update(&sh, text, (size_t)mlen);
        unsigned char d[32]; sha256_final(&sh, d);
        char got[65]; sha256_hex(d, got);
        if (strcmp(got, manifest_hex)) {
            fprintf(stderr, "castore: manifest %s hashes to %s -- refusing (tampered or corrupt manifest)\n",
                    manifest_hex, got);
            free(text); return -1;
        }
    }

    char *arena = NULL;
    jval *root_v = json_parse(text, &arena);
    if (!root_v || root_v->t != J_OBJ) {
        fprintf(stderr, "castore: %s does not parse as a JSON object\n", path);
        free(text); return -1; }
    jval *arch = json_get(root_v, "arch");
    jval *tensors = json_get(root_v, "tensors");
    if (!tensors || tensors->t != J_OBJ) {
        fprintf(stderr, "castore: %s has no 'tensors' object\n", path);
        free(text); return -1; }
    if (tensors->len < 0 || tensors->len > CA_MAX_TENSORS) {
        fprintf(stderr, "castore: manifest declares %d tensors -- refusing (untrusted manifest)\n",
                tensors->len);
        free(text); return -1; }

    S->root = strdup(root);
    S->arch = strdup(arch && arch->t == J_STR ? arch->str : "");
    S->cap = tensors->len > 0 ? tensors->len : 1;
    S->t = (ca_entry *)calloc((size_t)S->cap, sizeof(ca_entry));
    if (!S->root || !S->arch || !S->t) { fprintf(stderr, "castore: OOM\n"); free(text); return -1; }

    for (int i = 0; i < tensors->len; i++) {
        jval *e = tensors->kids[i];
        if (e->t != J_OBJ) {
            fprintf(stderr, "castore: tensor '%s' is not an object\n", tensors->keys[i]);
            free(text); return -1; }
        jval *b = json_get(e, "blob");
        if (!b || b->t != J_STR || !ca_is_hex64(b->str)) {
            fprintf(stderr, "castore: tensor '%s' has no valid 64-char hex blob hash\n", tensors->keys[i]);
            free(text); return -1; }
        ca_entry *t = &S->t[S->n];
        t->name = strdup(tensors->keys[i]);
        memcpy(t->blob, b->str, CA_HEX); t->blob[CA_HEX] = 0;
        jval *dt = json_get(e, "dtype");
        t->dtype = strdup(dt && dt->t == J_STR ? dt->str : "");
        t->layer = t->expert = -1;
        jval *ly = json_get(e, "layer");   if (ly && ly->t == J_NUM) t->layer  = (int)ly->num;
        jval *ex = json_get(e, "expert");  if (ex && ex->t == J_NUM) t->expert = (int)ex->num;
        jval *sh = json_get(e, "shape");
        t->numel = 1; t->rank = 0;
        if (sh && sh->t == J_ARR) {
            if (sh->len > 8) {
                fprintf(stderr, "castore: tensor '%s' has rank %d, max 8\n", tensors->keys[i], sh->len);
                free(text); return -1; }
            for (int k = 0; k < sh->len; k++) {
                jval *d = sh->kids[k];
                if (d->t != J_NUM || d->num < 0) {
                    fprintf(stderr, "castore: tensor '%s' has a non-integer dimension\n", tensors->keys[i]);
                    free(text); return -1; }
                t->shape[t->rank++] = (int64_t)d->num;
                t->numel *= (int64_t)d->num;
            }
        }
        if (!t->name || !t->dtype) { fprintf(stderr, "castore: OOM\n"); free(text); return -1; }
        S->n++;
    }
    free(text);              /* the jval tree is intentionally leaked, as st.h's parses are */
    ca_index_build(S);
    return 0;
}

/* Read a tensor's bytes into `out` (capacity `cap`), verifying that what came
 * off disk hashes to the name it was stored under. Returns the byte count, or
 * -1 on any failure. */
static int64_t ca_read(const ca_store *S, const char *name, void *out, int64_t cap) {
    const ca_entry *t = ca_find(S, name);
    if (!t) { fprintf(stderr, "castore: no tensor '%s' in the manifest\n", name); return -1; }
    char path[4096];
    if (ca_blob_path(S, t->blob, path, sizeof path) != 0) {
        fprintf(stderr, "castore: '%s' has an unusable blob hash\n", name); return -1; }
    int64_t n = 0;
    char *buf = ca_slurp(path, &n);
    if (!buf) { fprintf(stderr, "castore: cannot read blob %s for '%s'\n", path, name); return -1; }
    if (n > cap) {
        fprintf(stderr, "castore: '%s' is %lld bytes, destination holds %lld -- refusing\n",
                name, (long long)n, (long long)cap);
        free(buf); return -1; }
    if (!S->no_verify) {
        Sha256 sh; sha256_init(&sh);
        sha256_update(&sh, buf, (size_t)n);
        unsigned char d[32]; sha256_final(&sh, d);
        char got[65]; sha256_hex(d, got);
        if (strcmp(got, t->blob)) {
            fprintf(stderr, "%s: BLOB HASH MISMATCH\n  stored as %s\n  hashes to %s\n"
                    "  the bytes are not what this address promises -- refusing "
                    "(corrupt or tampered blob; CA_NO_VERIFY=1 skips this check)\n",
                    name, t->blob, got);
            free(buf); return -1;
        }
    }
    memcpy(out, buf, (size_t)n);
    free(buf);
    return n;
}

/* Byte size of a tensor's blob without reading it. -1 if unknown. */
static int64_t ca_nbytes(const ca_store *S, const char *name) {
    const ca_entry *t = ca_find(S, name);
    if (!t) return -1;
    char path[4096];
    if (ca_blob_path(S, t->blob, path, sizeof path) != 0) return -1;
    struct stat st;
    return stat(path, &st) == 0 ? (int64_t)st.st_size : -1;
}

static void ca_close(ca_store *S);   /* defined below; st_init_ca unwinds through it */

/* Populate a `shards` from a CA manifest, so every existing reader in st.h
 * works against a content-addressed container unchanged (#14).
 *
 * One blob per tensor maps onto st_tensor almost exactly: the blob IS the whole
 * tensor, so off=0 and nbytes is the file size. fd stays -1 and ca_blob carries
 * the address; st_pread_tensor dispatches on that, and the posix_fadvise and
 * mirror-routing sites skip a negative fd on their own.
 *
 * What this does NOT give you is the coalesced expert read. colibri.c's
 * expert_load_impl loads three weight tensors in ONE pread because they are
 * adjacent inside a shard; in a CA store every tensor is its own file, so that
 * adjacency does not exist. #36 spent an entire issue proving how much those
 * scattered reads cost (67% more of them, measured), so the expert path
 * deliberately refuses a CA container rather than silently taking the
 * regression -- see coli_expert_path_supports_ca in colibri.c. Resolving that
 * (pack blobs, or accept the reads and measure) is the next decision, and it
 * should be made on numbers.
 *
 * Returns 0 on success. The store's own verification applies: a manifest whose
 * bytes do not match its name is refused before a tensor is indexed.
 */
static int st_init_ca(shards *S, const char *root, const char *manifest_hex) {
    ca_store cs;
    if (ca_open(&cs, root, manifest_hex) != 0) return -1;
    memset(S, 0, sizeof(*S));
    S->ca_root = strdup(root);
    S->cap = cs.n > 0 ? cs.n : 1;
    S->t = (st_tensor *)calloc((size_t)S->cap, sizeof(st_tensor));
    if (!S->ca_root || !S->t) { fprintf(stderr, "castore: OOM building shards\n"); ca_close(&cs); return -1; }

    for (int i = 0; i < cs.n; i++) {
        const ca_entry *e = &cs.t[i];
        int64_t nb = ca_nbytes(&cs, e->name);
        if (nb < 0) {
            fprintf(stderr, "castore: blob for '%s' is missing from the store\n", e->name);
            ca_close(&cs); return -1; }
        st_tensor *t = &S->t[S->n++];
        t->name    = strdup(e->name);
        t->ca_blob = strdup(e->blob);
        t->fd      = -1;                 /* no descriptor: see st_tensor.ca_blob */
        t->off     = 0;                  /* a blob IS the tensor */
        t->nbytes  = nb;
        t->dtype   = st_dtype_code(e->dtype);
        t->numel   = e->numel;
        t->rank    = e->rank;
        for (int k = 0; k < e->rank && k < ST_MAX_RANK; k++) t->shape[k] = e->shape[k];
        if (!t->name || !t->ca_blob) { fprintf(stderr, "castore: OOM\n"); ca_close(&cs); return -1; }
        /* Same shape/bytes agreement st_init_multi enforces on a shard header:
         * numel comes from the manifest and nbytes from the file, two
         * independent facts, and a disagreement would overrun a caller's buffer
         * sized from the config. U8/I8 are read by byte count, so their numel is
         * legitimately unused. */
        { int esz = st_dtype_esz(t->dtype);
          if (t->dtype != 3 && t->numel * (int64_t)esz != t->nbytes) {
              fprintf(stderr, "castore: '%s' declares %lld elements but its blob is %lld bytes (dtype %d)"
                      " -- refusing (manifest disagrees with the store)\n",
                      t->name, (long long)t->numel, (long long)t->nbytes, t->dtype);
              ca_close(&cs); return -1; } }
    }

    S->hcap = 1; while (S->hcap < S->n * 2) S->hcap <<= 1;
    S->hidx = (int *)malloc((size_t)S->hcap * sizeof(int));
    if (!S->hidx) { fprintf(stderr, "castore: OOM indexing\n"); ca_close(&cs); return -1; }
    for (int i = 0; i < S->hcap; i++) S->hidx[i] = -1;
    for (int i = 0; i < S->n; i++) {
        uint64_t h = st_hash(S->t[i].name) & (uint64_t)(S->hcap - 1);
        while (S->hidx[h] >= 0) h = (h + 1) & (uint64_t)(S->hcap - 1);
        S->hidx[h] = i;
    }
    ca_close(&cs);
    return 0;
}

static void ca_close(ca_store *S) {
    for (int i = 0; i < S->n; i++) { free(S->t[i].name); free(S->t[i].dtype); }
    free(S->t); free(S->hidx); free(S->root); free(S->arch);
    memset(S, 0, sizeof(*S));
}
#endif
