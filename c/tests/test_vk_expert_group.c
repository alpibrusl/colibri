/* backend_vulkan.c's expert-group and matmul paths, driven WITHOUT the engine,
 * against a real Vulkan implementation (Lavapipe in CI -- a full software
 * compute device, so the shaders actually execute).
 *
 * Why this exists (#11 slice 3): until now the only CI over backend_vulkan.c
 * was compile + coli_vk_init() -- the upload, matmul, and expert-group code
 * paths (including the dev2 clone family) had NO behavioral gate anywhere.
 * This test is that gate: int8 tensors with known contents go up, the fused
 * gate_up->down expert group runs on-device, and the result is checked
 * against a scalar CPU reference to tight f32 tolerance.
 *
 * dev2 coverage without second hardware: coli_vk_init_dev2(path, 0) forces
 * the SAME physical device as a second logical device (the backend's
 * documented pre-hardware test mode), so the *2 entry points -- the clone
 * family being de-duplicated -- run for real on Lavapipe too.
 *
 * No Vulkan device (loader missing / no ICD): prints "skip" and exits 0 --
 * the test never fails a machine for lacking a GPU stack; the Vulkan CI job
 * is where it must actually run (Lavapipe is guaranteed there).
 */
#include "../backend_vulkan.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #condition); \
        exit(1); \
    } \
} while (0)

#define D    64          /* hidden dim (gate/up input, down output) */
#define IM   48          /* moe_inter (gate/up output, down input) */
#define NEXP 3           /* experts in the group */

static unsigned rng_state = 0x2545F491u;
static unsigned rng_next(void) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5;
    return rng_state;
}
static float rng_float(void) {          /* [-1, 1), deterministic */
    return (float)((double)(int32_t)rng_next() / 2147483648.0);
}
static int8_t rng_q8(void) { return (int8_t)((int)(rng_next() % 255) - 127); }

static float silu(float v) { return v / (1.0f + expf(-v)); }

/* y[S,O] = x[S,I] @ dequant(q8[O,I] * s[O])^T -- the scalar truth */
static void ref_matmul_q8(float *y, const float *x, const int8_t *q, const float *s,
                          int S, int I, int O) {
    for (int r = 0; r < S; r++)
        for (int o = 0; o < O; o++) {
            float acc = 0;
            for (int i = 0; i < I; i++) acc += x[(size_t)r * I + i] * (float)q[(size_t)o * I + i];
            y[(size_t)r * O + o] = acc * s[o];
        }
}

/* one expert, one row: down(silu(gate(x)) * up(x)) */
static void ref_expert_row(float *y, const float *x,
                           const int8_t *qg, const float *sg,
                           const int8_t *qu, const float *su,
                           const int8_t *qd, const float *sd) {
    float g[IM], u[IM], h[IM];
    ref_matmul_q8(g, x, qg, sg, 1, D, IM);
    ref_matmul_q8(u, x, qu, su, 1, D, IM);
    for (int i = 0; i < IM; i++) h[i] = silu(g[i]) * u[i];
    ref_matmul_q8(y, h, qd, sd, 1, IM, D);
}

static double max_rel_err(const float *got, const float *want, int n) {
    double worst = 0;
    for (int i = 0; i < n; i++) {
        double denom = fabs(want[i]) > 1e-3 ? fabs(want[i]) : 1e-3;
        double e = fabs((double)got[i] - want[i]) / denom;
        if (e > worst) worst = e;
    }
    return worst;
}

/* per-expert weights + their resident handles (dev0 and dev2 slots) */
static int8_t qg[NEXP][IM * D], qu[NEXP][IM * D], qd[NEXP][D * IM];
static float  sg[NEXP][IM],     su[NEXP][IM],     sd[NEXP][D];
static ColiVkTensor *tg[NEXP], *tu[NEXP], *td[NEXP];
static ColiVkTensor *tg2[NEXP], *tu2[NEXP], *td2[NEXP];

int main(int argc, char **argv) {
    const char *spv = argc > 1 ? argv[1] : "shaders/qmatmul.spv";
    if (!coli_vk_init(spv)) {
        printf("test_vk_expert_group: skip (no Vulkan device / shaders at %s)\n", spv);
        return 0;
    }

    for (int c = 0; c < NEXP; c++) {
        for (int i = 0; i < IM * D; i++) { qg[c][i] = rng_q8(); qu[c][i] = rng_q8(); }
        for (int i = 0; i < D * IM; i++) qd[c][i] = rng_q8();
        for (int i = 0; i < IM; i++) { sg[c][i] = 0.01f + 0.001f * (float)(i % 7); su[c][i] = 0.02f; }
        for (int i = 0; i < D;  i++) sd[c][i] = 0.015f;
        CHECK(coli_vk_tensor_ensure(&tg[c], qg[c], sg[c], 1, D, IM, 0));
        CHECK(coli_vk_tensor_ensure(&tu[c], qu[c], su[c], 1, D, IM, 0));
        CHECK(coli_vk_tensor_ensure(&td[c], qd[c], sd[c], 1, IM, D, 0));
        CHECK(coli_vk_tensor_bytes(tg[c]) > 0);
        CHECK(coli_vk_tensor_dev(tg[c]) == 0);
    }

    /* ---- dense matmul against the reference ---- */
    {
        enum { S = 4 };
        float x[S * D], y[S * IM], want[S * IM];
        for (int i = 0; i < S * D; i++) x[i] = rng_float();
        ColiVkTensor *t = NULL;
        CHECK(coli_vk_matmul(&t, y, x, qg[0], sg[0], 1, S, D, IM, 0));
        ref_matmul_q8(want, x, qg[0], sg[0], S, D, IM);
        double e = max_rel_err(y, want, S * IM);
        CHECK(e < 1e-4);
        coli_vk_tensor_free(t);
    }

    /* ---- expert group: sync form, ragged rows ---- */
    int rows[NEXP] = { 1, 3, 2 };
    int R = rows[0] + rows[1] + rows[2];
    float *x = malloc((size_t)R * D * sizeof(float));
    float *y = malloc((size_t)R * D * sizeof(float));
    float *want = malloc((size_t)R * D * sizeof(float));
    CHECK(x && y && want);
    for (int i = 0; i < R * D; i++) x[i] = rng_float();
    { int off = 0;
      for (int c = 0; c < NEXP; c++)
          for (int r = 0; r < rows[c]; r++, off++)
              ref_expert_row(want + (size_t)off * D, x + (size_t)off * D,
                             qg[c], sg[c], qu[c], su[c], qd[c], sd[c]); }
    CHECK(coli_vk_expert_group(tg, tu, td, rows, NEXP, y, x));
    double e_sync = max_rel_err(y, want, R * D);
    CHECK(e_sync < 1e-3);

    /* ---- expert group: async issue + take, same inputs ---- */
    memset(y, 0, (size_t)R * D * sizeof(float));
    CHECK(coli_vk_expert_group_issue(tg, tu, td, rows, NEXP, x));
    CHECK(coli_vk_expert_group_take(y));
    double e_async = max_rel_err(y, want, R * D);
    CHECK(e_async < 1e-3);

    /* ---- dev2 (same-device test mode): the clone family, for real ---- */
    int dev2 = coli_vk_init_dev2(spv, 0);
    double e2 = -1;
    if (dev2) {
        CHECK(coli_vk_dev2_available());
        for (int c = 0; c < NEXP; c++) {
            CHECK(coli_vk_tensor_ensure2(&tg2[c], qg[c], sg[c], 1, D, IM, 0));
            CHECK(coli_vk_tensor_ensure2(&tu2[c], qu[c], su[c], 1, D, IM, 0));
            CHECK(coli_vk_tensor_ensure2(&td2[c], qd[c], sd[c], 1, IM, D, 0));
            CHECK(coli_vk_tensor_dev(tg2[c]) == 1);
        }
        memset(y, 0, (size_t)R * D * sizeof(float));
        CHECK(coli_vk_expert_group_issue2(tg2, tu2, td2, rows, NEXP, x));
        CHECK(coli_vk_expert_group_take2(y));
        e2 = max_rel_err(y, want, R * D);
        CHECK(e2 < 1e-3);
        memset(y, 0, (size_t)R * D * sizeof(float));
        CHECK(coli_vk_expert_group2(tg2, tu2, td2, rows, NEXP, y, x));
        CHECK(max_rel_err(y, want, R * D) < 1e-3);
    }

    printf("test_vk_expert_group: ok — matmul + expert group (sync %.2e / async %.2e) "
           "over %d experts, %d rows; dev2 %s\n",
           e_sync, e_async, NEXP, R,
           dev2 ? "exercised (same-device mode)" : "unavailable (skipped)");
    free(x); free(y); free(want);
    return 0;
}
