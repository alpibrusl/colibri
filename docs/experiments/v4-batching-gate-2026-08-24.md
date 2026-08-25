# Does batching buy bytes? The #37 gate, measured offline

**Date** 2026-08-24 · **Issue** [#37](https://github.com/alpibrusl/colibri/issues/37) ·
**Model** DeepSeek V4 Flash (43 MoE layers, top-6 of 256, ~13.7 MB/expert) ·
**Tools** `c/tools/batch_union.py`, `c/tools/batch_expert_major.py` ·
**Input** six ROUTE_TRACE captures, one per prompt fixture, treated as independent sessions

#37 argues that in the **streaming** regime — unlike the full-resident GLM experiment that
preceded it — batching amortizes a cold expert fetch across every queued token routing to
it, and that scheduling **expert-major** (run all tokens needing expert *e* while *e* is
resident) is what unlocks it. This is the cheap offline test of that claim, before any
engine code.

## Result 1: the union bends, but only gently

Distinct experts a step must fetch for S concurrent sessions, against S × the single-session
count:

| S | union/step | no sharing | ratio | bytes/token |
|---|---|---|---|---|
| 1 | 258.0 | 258.0 | 1.000 | 1.00× |
| 2 | 498.3 | 516.0 | 0.966 | 0.97× |
| 3 | 726.6 | 774.0 | 0.939 | 0.94× |
| 4 | 907.6 | 1032.0 | 0.879 | 0.88× |
| 5 | 1099.6 | 1290.0 | 0.852 | 0.85× |

Sublinear, and slightly better than uniform-random draws would give (routing is somewhat
concentrated) — but 15% at S=5, not the 1/S the issue hoped for. With top-6 of 256, two
independent tokens rarely want the same expert.

## Result 2: expert-major ordering buys nothing

Same bounded per-layer LRU, same traces, only the intra-step order differs:

| cap | S | token-major | expert-major | gain |
|---|---|---|---|---|
| 32 | 5 | 118.7 | 119.5 | 0.99× |
| 64 | 5 | 70.3 | 70.3 | 1.00× |
| 128 | 5 | 42.1 | 42.2 | 1.00× |

**1.00× everywhere.** The reordering #37 proposes does not change the miss count at any
cache size tested, because the per-step working set (~26 experts/layer at S=5) already fits
in any realistic cache — nothing evicts anything *within* a step, so the order it happens in
is irrelevant.

## Result 3: what actually decides it is cache capacity

Misses per token produced, so S=1 and S=5 are directly comparable:

| per-layer cap | S=1 | S=5 | |
|---|---|---|---|
| 32 | 74.5 | 118.7 | **1.60× worse** |
| 48 | 67.3 | 89.4 | 1.35× worse |
| 64 | 65.9 | 70.3 | 1.07× ≈ neutral |
| 96 | 65.7 | 49.9 | 0.76× better |
| 128 | 65.7 | 42.1 | 0.64× better |
| 192 | 65.7 | 38.5 | 0.59× better |

There is a **crossover near cap ≈ 64**. Below it, S sessions sharing one cache contend
harder than they share, and batching is actively worse. Above it, cross-session temporal
reuse wins.

## Where we actually run

At 13.7 MB/expert across 43 layers:

| `--ram` | experts/layer |
|---|---|
| 24 GB | ~42 |
| **32 GB** | **~56** |
| 48 GB | ~83 |
| 64 GB | ~111 |

The real 128-token run measures 72.7% hit, which brackets to cap ≈ 50–60 — consistent.

So **at the budget colibri targets, batching lands on the wrong side of the crossover**:
neutral at best, 1.35–1.60× worse if memory is tighter. Reaching the S=5 win needs cap ≥ 96,
which is ~55 GB of expert cache *alone* — a 96–128 GB host, at which point you are close to
the full-residency regime where the GLM experiment already found batching does not raise
aggregate throughput.

## Verdict on #37, and the part this does NOT settle

The stated mechanism — expert-major ordering amortizing fetches — **does not survive**: the
ordering is worth 1.00×, and the S-scaling is neutral-to-negative at our memory budget.

But this experiment measures **bytes only**, and bytes are not where the time goes. The
post-merge baseline puts decode at **80% compute, 20% I/O stall**. Batching's other effect —
turning a batch-1 GEMV into a batch-N GEMM, so one fp8 dequantization serves N tokens — is
untouched here and attacks the term that actually dominates. The engine already has the
kernels (`coli_fp8_matmul_batch_ref` / `_pre`, batch up to 128); what is missing is a
scheduler to feed them.

**So batching is still worth pursuing — for the compute reason, not the bytes reason.**
#37 as written targets the smaller term.

## Result 4: the batch kernel does not batch

`c/tools/bench_batch_scaling` runs the real `matmul_fp8` on V4's dense shape
(I=4096, O=2048) and sweeps the batch, reporting time **per token**.

The first reading looked like a win — 2.49x by S=16. It was an artifact, and the artifact is
instructive. With the default OpenMP wait policy this host pays ~65 us of fork/join per
parallel region (see `bench_omp_grain`), which at S=1 is a quarter of the whole call.
Batching spread that fixed cost over more tokens, and that is all the "speedup" was.

Pinning `OMP_WAIT_POLICY=active` (fork/join ~2 us) removes it:

| S | ms/token, default policy | ms/token, spin | vs S=1 (spin) |
|---|---|---|---|
| 1 | 0.271 | 0.156 | 1.00x |
| 2 | 0.226 | 0.154 | 1.02x |
| 8 | 0.168 | 0.153 | 1.02x |
| 16 | 0.155 | 0.153 | 1.02x |
| 64 | 0.154 | 0.153 | 1.02x |

**1.02x. Batching the kernel is worth nothing**, and total time scales exactly linearly with
S (0.16, 0.31, 0.61, 1.22, 2.45, 4.89, 9.78 ms).

The reason is in the loop nest:

```c
for (o …)                       /* output rows, 4 at a time, parallel */
    for (s = 0; s < S; s++)     /* batch INSIDE */
        for (i …)
            c0 += e4m3_decode(w0[i]) * xi;
```

`e4m3_decode` is evaluated once per *(output row, batch row, input element)*. The weight rows
stay in L1 across the `s` loop, so this is not a bandwidth problem — the **decode work is
simply repeated S times**. The kernel is S matvecs sharing a loop nest, not a matmul.

So the amortization is available and unclaimed. Hoisting the decode out of the `s` loop —
decode a 128-element block of 4 weight rows once, then run S dot products against it — keeps
each row's accumulation order over blocks intact and should therefore stay bit-identical.
Given that speed on this kernel tracks instruction count, and decode is the dominant
instruction, the headroom is large. **This is the finding of the experiment.**

It also plausibly explains a separate anomaly: prefill measured ~2.3 tok/s on a 595-token
prompt, roughly 8x off its own roofline. Prefill is exactly where S is large, and exactly
where this kernel re-decodes every weight for every row.

## Result 5: even so, the routed experts never reach a useful batch

Batching splits into two paths that behave in opposite ways. Tokens served per *activated*
expert, measured from the traces and extrapolated (top-6 of 256, independent routing):

| S | tokens per activated expert |
|---|---|
| 2 | 1.04 (measured) |
| 5 | 1.17 (measured) |
| 16 | 1.19 |
| 64 | 1.92 |
| 128 | 3.15 |
| **256** | **6.01** |

At S=16 concurrent sessions a given expert still serves ~1.2 tokens. Reaching a batch of 6
needs roughly **256 concurrent sessions**.

So a decode-side fix helps:

- **dense / attention** (~37% of decode time): sees the full batch S — but only in prefill,
  since decode is inherently S=1 per session
- **routed experts** (~35% of decode time, and all of the I/O): effectively batch 1 until S
  reaches the hundreds

**The sparsity that makes streaming possible is the same property that defeats batching.**
Top-6 of 256 is what keeps per-token traffic at 3.45 GB instead of 167 GB — and it is exactly
why S independent tokens scatter across nearly S x 6 distinct experts. The two properties are
in tension by construction.

## What this means for the roadmap

1. **Fix the kernel** (Result 4). It is a real, bounded, probably bit-identical optimization,
   and its main beneficiary is **prefill**, which is 8x off roofline and which nobody has
   looked at. This does not need a scheduler.
2. **Do not build the #37 scheduler for the bytes** (Results 1-3): expert-major ordering is
   worth 1.00x, and at our memory budget concurrency makes misses *worse*.
3. **For single-stream decode, batching is not the lever at all** (Result 5). Bytes and
   instructions per byte are: int4 dense is -33% traffic *and* fewer decode operations, and
   it hits the dominant term twice.

## Method## Method

    ROUTE_TRACE=trace.txt ./deepseek_v4 <model> --prompt-file <p> --max-tokens 24
    python3 c/tools/batch_expert_major.py trace1.txt trace2.txt ...

Caveats worth keeping: 5 sessions × 23 decode steps is a small sample; sessions are modelled
as decoding in lockstep at equal rates; prefill forwards are excluded (batching prefill is a
different question); and the LRU here is plain, where the engine's real policy is LFRU with
an eviction guard.
