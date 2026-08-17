/* kv_persist write-error discipline: a failed append must NOT advance the
 * header's record count.
 *
 * The crash-safety design writes records BEFORE the count, so a torn append
 * is invisible on reload -- but only if write errors are actually checked.
 * Unchecked, an ENOSPC still advanced disk_nrec and rewrote the header,
 * persisting a count that claims records whose bytes never reached the disk;
 * kv_disk_load would then resume a corrupt conversation. This gates the fix:
 * on any write/flush failure the append bails before the header update,
 * disables persistence for the run (g_kvsave=0), and keeps disk_nrec at the
 * last consistent prefix.
 *
 * No model file needed: like test_kv_alloc, the fake Model only feeds
 * n_layers/kv_lora/qk_rope through kv_alloc and the record codec. The
 * failure injection uses /dev/full (writes buffer fine, fflush fails with
 * ENOSPC -- exactly the deferred-failure shape the fix must catch), so that
 * part is Linux-only and skips elsewhere.
 */
#define main coli_glm_main_unused
#include "../colibri.c"
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

int main(void) {
    static Model m;
    m.c.n_layers = 2; m.c.kv_lora = 8; m.c.qk_rope = 4; m.c.vocab = 64;
    m.kv = calloc(1, sizeof(KVState));
    CHECK(m.kv != NULL);
    kv_alloc(&m, 32);

    /* distinctive per-position values so the reload check is real */
    int hist[3] = {7, 11, 13};
    for (int p = 0; p < 3; p++)
        for (int i = 0; i < m.c.n_layers; i++) {
            for (int j = 0; j < m.c.kv_lora; j++)
                m.Lc[i][(int64_t)p * m.c.kv_lora + j] = (float)(100 * p + 10 * i + j);
            for (int j = 0; j < m.c.qk_rope; j++)
                m.Rc[i][(int64_t)p * m.c.qk_rope + j] = (float)(200 * p + 10 * i + j);
        }

    /* --- happy path: append then reload round-trips --- */
    char dir[] = "test_kv_persist_XXXXXX";
    CHECK(mkdtemp(dir) != NULL);
    snprintf(m.kv->disk_path, sizeof(m.kv->disk_path), "%s/.coli_kv", dir);
    kv_disk_append(&m, hist, 3);
    CHECK(m.kv->disk_nrec == 3);
    CHECK(g_kvsave == 1);
    if (m.kv->disk_fp) { fclose(m.kv->disk_fp); m.kv->disk_fp = NULL; }

    /* wipe live state, reload from disk, compare */
    int hist2[32] = {0};
    for (int i = 0; i < m.c.n_layers; i++) {
        memset(m.Lc[i], 0, (size_t)32 * m.c.kv_lora * 4);
        memset(m.Rc[i], 0, (size_t)32 * m.c.qk_rope * 4);
    }
    m.kv->disk_nrec = 0;
    CHECK(kv_disk_load(&m, hist2, 32) == 3);
    CHECK(hist2[0] == 7 && hist2[1] == 11 && hist2[2] == 13);
    CHECK(m.Lc[1][(int64_t)2 * m.c.kv_lora + 3] == (float)(100 * 2 + 10 * 1 + 3));
    CHECK(m.Rc[0][(int64_t)1 * m.c.qk_rope + 2] == (float)(200 * 1 + 10 * 0 + 2));

    char path[600];
    snprintf(path, sizeof(path), "%s/.coli_kv", dir);
    remove(path);
    rmdir(dir);

#ifdef __linux__
    /* --- failure injection: /dev/full defers the error to fflush --- */
    if (access("/dev/full", W_OK) == 0) {
        if (m.kv->disk_fp) { fclose(m.kv->disk_fp); m.kv->disk_fp = NULL; }
        snprintf(m.kv->disk_path, sizeof(m.kv->disk_path), "/dev/full");
        m.kv->disk_nrec = 0;
        g_kvsave = 1;
        kv_disk_append(&m, hist, 3);
        CHECK(m.kv->disk_nrec == 0);      /* count must NOT advance over lost bytes */
        CHECK(g_kvsave == 0);             /* persistence disabled for the run */
        CHECK(m.kv->disk_fp == NULL);     /* handle closed, no half-open state */
        printf("OK kv_persist errors: round-trip + ENOSPC append leaves the count untouched\n");
    } else
#endif
    {
        printf("OK kv_persist errors: round-trip (failure injection skipped: no /dev/full)\n");
    }
    return 0;
}
