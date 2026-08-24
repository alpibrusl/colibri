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

## Method

    ROUTE_TRACE=trace.txt ./deepseek_v4 <model> --prompt-file <p> --max-tokens 24
    python3 c/tools/batch_expert_major.py trace1.txt trace2.txt ...

Caveats worth keeping: 5 sessions × 23 decode steps is a small sample; sessions are modelled
as decoding in lockstep at equal rates; prefill forwards are excluded (batching prefill is a
different question); and the LRU here is plain, where the engine's real policy is LFRU with
an eviction guard.
