# DeepSeek V4 Flash: the baseline, and the one number that matters (#62)

**Date** 2026-08-21 · **Issues** [#62](https://github.com/alpibrusl/colibri/issues/62) ·
**Model** DeepSeek V4 Flash, 284B MoE, 167 GB on disk, 74 shards verified against the repo manifest ·
**Tool** `c/tools/v4_baseline.py`

The ask was a baseline for V4 so improvement can be measured rather than
asserted. The baseline is below, and the single most important thing in it is
that **the headline metric depends almost entirely on how many tokens you
generate**. A hit rate quoted without a sequence length is not a measurement.

## The baseline

Prompt `What is the capital of France?` (9 generated tokens), one workstation:

| | cold, `--ram 32` | warm, `--ram 32` | warm, `--ram 48` |
|---|---|---|---|
| hit rate | 42.04% | 43.58% | 43.72% |
| expert requests | 3504 | 3504 | 3504 |
| misses | 2031 | 1977 | 1972 |
| bytes streamed | 27.15 GB | 26.43 GB | 26.36 GB |
| bytes / token | 3017 MB | 2937 MB | 2929 MB |
| TTFT | 55.5 s | 49.7 s | 50.6 s |
| decode | 97.0 s | 88.1 s | 91.5 s |
| tok/s | 0.093 | 0.102 | 0.098 |

Cold ≠ warm because the engine persists `.coli_usage` into the model directory
and reads it back, so consecutive runs are **not independent** — the harness
takes `--cold` to remove it. Cold reproduces an independently-captured earlier
run to the digit (42.038%, 27153137664 bytes). Routing is deterministic: hit
rate is identical across repeats, wall-clock varies ~2%.

## Sequence length is the dominant variable

All cold, `--ram 32`. The 48/128-token rows use a prompt that actually generates
(`--ngen` is a **cap**, not a target — generation stops at EOS, so the France
prompt yields 9 tokens whatever `--ngen` says, and a sweep over `--ngen` alone
varies nothing):

| generated | requests | distinct | hit rate | ceiling | **gap** | capacity misses | bytes/token | tok/s |
|---|---|---|---|---|---|---|---|---|
| 8 | 3,246 | 1,906 | 39.93% | 41.28% | 1.36 pp | 44 | 3259 MB | 0.085 |
| 9 | 3,504 | 1,972 | 42.01% | 43.72% | 1.71 pp | 60 | 3019 MB | 0.091 |
| 48 | 14,717 | 4,131 | 63.27% | 71.93% | **8.66 pp** | 1,274 | 1505 MB | 0.124 |
| 128 | 35,357 | 5,438 | 72.41% | 84.62% | **12.21 pp** | 4,317 | 1019 MB | 0.161 |

*Ceiling* is what a demand-fetch cache can reach at best: it must miss once per
distinct expert, so the compulsory-miss floor is `distinct` and the ceiling is
`(requests − distinct)/requests`. *Gap* is what better caching could still
recover. *Capacity misses* = `misses − distinct`, the misses a bigger or smarter
cache could have avoided.

Three things move together as generation lengthens, and all three matter:

- **The ceiling rises**, 43.7% → 84.6%. Distinct experts grow far more slowly
  than selections (5,438 distinct across 35,357 selections), so reuse dominates.
- **The gap widens**, 1.7 pp → 12.2 pp. Capacity misses go from 60 to 4,317.
- **Cost per token falls 3×**, 3019 → 1019 MB, and throughput nearly doubles,
  0.091 → 0.161 tok/s. Long generations are much cheaper per token.

### The correction this forced

The 9-token measurement says `--ram 48` buys **+0.14 pp** over `--ram 32` and
sits at *exactly zero* capacity misses — the infinite-cache ceiling, hit
precisely. Read alone, that says the cache is finished and cache work is
pointless.

That conclusion is an artifact of a 9-token toy workload, where ~100% of misses
are compulsory first touches and no cache of any size can help. At 128 tokens
there are 4,317 capacity misses and 12.21 pp on the table. **Caching is the
largest identified lever, not a closed one** — the opposite of what the short
prompt showed. Any V4 result quoted at 9 tokens should be assumed to say nothing
about V4 in use.

### The prediction, and what it exposed

The length story makes a falsifiable prediction: if the 9-token result was an
artifact, then more cache — worthless at 9 tokens — must pay at 128. Tested cold:

| `--ngen 128` | `--ram 32` | `--ram 48` | Δ |
|---|---|---|---|
| hit rate | 72.41% | 75.46% | **+3.05 pp** |
| capacity misses | 4,317 | 3,238 | −1,079 |
| bytes streamed | 130.42 GB | 115.99 GB | **−11.1%** |
| bytes / token | 1019 MB | 906 MB | −11.1% |
| wall clock | 948.9 s | 956.5 s | **+0.8%** |

The prediction holds: +3.05 pp against +0.14 pp at 9 tokens, a 22× larger
effect from the same 16 GiB.

The second row of that table is the more valuable result. **11% fewer bytes
bought no time at all** — 0.8% slower, within run-to-run noise. Bytes streamed
and wall clock are decoupled in this engine, which independently confirms the
device measurement below from the opposite direction: if I/O were on the
critical path, an 11% byte reduction could not have cost nothing.

So the two levers do different jobs, and should not be conflated:

- **Cache work reduces bytes**, which is real — disk wear, energy, and any
  shared or networked storage — but on this hardware it does not reduce latency.
- **Only the invisible 97% can reduce latency**, and it is invisible (#62).

## What the baseline does rule out: it is not I/O bound

The 128-token run streams 130.42 GB in 948.9 s: **137 MB/s effective**. The
device, measured at the engine's own access pattern — random reads of 12.6 MB
(the observed bytes-per-miss), `F_NOCACHE`, fresh offsets per trial:

| trial | throughput | per read |
|---|---|---|
| seed 101 | 5.92 GB/s | 2.13 ms |
| seed 202 | 4.69 GB/s | 2.69 ms |
| seed 303 | 5.22 GB/s | 2.41 ms |

*(Offsets must be fresh per trial: a first attempt reused one seed and re-read
what the previous probe had just cached, reporting 21 GB/s — faster than the
device can physically go, which is how the mistake announced itself.)*

At 2.4 ms per read, the 128-token run's 9,755 misses need **~23 s of device time
out of 948.9 s — about 2.5%**, and the device has ~35× more throughput than the
engine draws. Compression, smaller experts and cheaper container formats — the
levers that pay when streaming is bandwidth-bound — cannot buy much here. The
bytes are not the problem.

So: **~97% of the wall clock is somewhere we cannot see** — compute, or failure
to overlap fetch with compute, or scheduling.

## Why we cannot see it

`deepseek_v4.c` emits no `route_trace.h` records (#62). Every measurement tool in
this tree — PILOT, COUPLE, the replay record, `expert_io_replay.py`,
`session_overlap.py`, `admission_gate.py` — reads that trace, so all of them are
blind to the one model large enough to make their claims interesting.

This is the defect `route_trace.h:373` names as the tree's recurring one: a fix
lands in one engine and not its siblings. Here it is load-bearing.

## Gates for the next change

Stated now, before any work, so the answer can be a clean negative:

1. **Instrumentation (#62).** `deepseek_v4.c` emits `ROUTE_TRACE` rows in the
   same format as the other four engines, and `replay_diff.py` runs against V4
   unmodified. `coli_v4_route` has **three duplicated call sites**
   (`deepseek_v4.c:3172`, `:3702`, `:4009`) — an instance of #12; hooking two of
   three yields a trace that is wrong rather than absent, which is worse.
2. **Attribution.** With the trace, per-phase timing accounts for ≥90% of the
   run, naming where the ~97% that is not device time goes. Until this passes,
   every speedup claim is a guess.
3. **Capacity (the 12.21 pp).** At 128 tokens, a cache change recovers ≥4 pp of
   the gap without raising bytes streamed — beating the +3.05 pp that simply
   buying 16 GiB of RAM already achieves. This is the lever the length sweep
   found; it did not exist at 9 tokens. Judge it on **bytes**, not on wall
   clock: `--ram 48` shows an 11% byte reduction can be latency-neutral, so a
   cache change that claims a speedup is measuring something else.
4. **Prefetch (the compulsory floor).** PILOT on V4 makes ≥15% of first touches
   resident at first use, at ≤5% added bytes. This is the only lever that can
   move the ceiling itself.

**Measure at ≥128 generated tokens.** Every gate above is void at 9.

## Reproducing

```
python3 c/tools/v4_baseline.py --selftest
python3 c/tools/v4_baseline.py --model ~/models/DeepSeek-V4-Flash --ram 32 --runs 3
python3 c/tools/v4_baseline.py --model ~/models/DeepSeek-V4-Flash --ram 32 --cold --runs 1 \
    --ngen 128 --prompt "Write a detailed essay on the history of the Roman Empire, covering its founding, its expansion, and the causes of its decline."
```

The harness parses the engine's counters **by name, scoped to the line that
emits them**. An earlier positional parser printed a tidy table of zeros when the
engine simply was not built; a later unscoped `bytes=` matched the
`v4_dense_resident ... bytes=6.267GiB` line and reported 0.00 GB streamed. A
baseline that can print a plausible wrong number is worse than no baseline, so
the selftest carries both decoys. The ceiling is reported only for `--cold` runs,
because `distinct` is otherwise cumulative over `.coli_usage` history and a
previous run with a different prompt would silently understate it.
