# Where V4's time actually goes — and the 7.5× that was sitting on the floor (#62 gate 2)

**Date** 2026-08-21 · **Issues** [#62](https://github.com/alpibrusl/colibri/issues/62) ·
**Follows** `deepseek-v4-baseline-2026-08-21.md` (#63) ·
**Model** DeepSeek V4 Flash, 284B MoE, 167 GB on disk

The baseline could say what a run was *not* bound by — device time ~2.5%, the
expert cache within 0.14 pp of its ceiling — and could not say what it *was*
bound by. It named that gap gate 2. This closes it, and the answer was not in
the engine.

## The instrumentation already existed and did not work

`COLI_V4_EXPERIMENTAL_BLOCK_OTHER_PROFILE` had four call sites placed in
`moe_token_pipeline` and the loader wrappers. But `coli_v4_block_profile_now`
and `coli_v4_block_profile_add` were **declared and never defined anywhere in
the tree**: the flag compiles and then fails to link. `v4_prof_emit` printed
four hardcoded `0.000` fields. Coverage was also partial in a way that mattered
— `MOE_TOTAL` reached only the pipelined token path, so `moe_token` and the
prefill batch union went unattributed, and prefill is where TTFT goes; attention
was never timed at all.

Now always-on, like the expert-store disk/matmul accounting it sits beside: a
profiler you have to rebuild for cannot answer where a given run's time went.
The cost was measured rather than assumed — cold runs land at 95.1 s and 97.6 s
against a pre-change cold spread of 95.3–98.8 s, counters identical to the digit.

## The finding: the build was single-threaded

This host had no `libomp`. The Makefile's Darwin branch warns and falls back, so
**every engine here was building single-threaded on a 16-core machine** (12
performance cores). Installing it:

| | single-threaded | OpenMP | |
|---|---|---|---|
| 9 tokens, wall | 95.1 s | 20.7 s | **4.6×** |
| 128 tokens, wall | 948.9 s | 125.9 s | **7.5×** |
| 128 tokens, tok/s | 0.161 | 1.017 | 6.3× |

Counters are byte-identical across the pair (42.009% hit, 2032 misses,
27166507008 bytes) and the output text is unchanged, so this is pure compute
parallelism — not a different computation that happens to be faster. It accounts
for most of the "~97% of wall clock unaccounted for" the baseline reported.

## The phase split, and how it inverts

```
v4_phase wall=... moe=... attention=... other=...
v4_phase_moe_inner gate_decode=... loader_start=... loader_wait=...
                   expert_disk=... expert_matmul=...  (thread-seconds)
```

9 generated tokens, cold, `--ram 32`:

| phase | single-threaded | OpenMP | parallel speedup |
|---|---|---|---|
| attention | 47.0 s (49%) | 7.1 s (34%) | **6.6×** |
| moe | 45.2 s (47%) | 12.2 s (59%) | **3.7×** |
| other | 3.5 s (4%) | 1.5 s (7%) | 2.3× |
| `loader_wait` (inside moe) | 0.35 s | 1.64 s | **4.7× worse** |

Attention parallelises well, MoE less well, and **disk does not parallelise at
all** — so the phase that grows is the one spent waiting for it.

At 128 tokens under OpenMP: wall 125.9 s = moe 63.0 + attention 54.4 + other 8.5,
with `loader_wait` 13.0 s, `expert_disk` 48.6 and `expert_matmul` 24.7
thread-seconds.

## The trap in reading that, and the test that avoids it

`expert_disk` = 48.6 against a 63.0 s MoE span reads as "I/O is now the leading
term". **That reading is wrong.** Those are *thread-seconds* summed across the
loader threads and overlapped with compute; the part that actually stalls MoE is
`loader_wait`, 13.0 s. A sum of thread-seconds can exceed the wall-clock span it
sits inside, and comparing it to one is a category error.

So it was tested instead of argued. At 128 tokens under OpenMP, comparing two
cache sizes both far below memory pressure:

| | `--ram 24` | `--ram 32` | Δ |
|---|---|---|---|
| hit rate | 66.22% | 72.40% | +6.2 pp |
| bytes | 159.67 GB | 130.44 GB | **−18.3%** |
| moe | 63.47 s | 63.29 s | **−0.3%** |
| wall | 129.31 s | 127.71 s | **−1.2%** |

**22% more bytes costs 1.2% more wall clock.** The baseline's "bytes and latency
are decoupled" survives OpenMP. I/O is far more *exposed* than it was
(`loader_wait` ~10% of wall, up from ~0.4%) and still not binding.

### `--ram 48` is not a usable instrument on this host

It raises the hit rate to 77.5% and cuts bytes 18.5% while running **65%
slower**. That is not a cache result: 48 GiB of expert cache on a 64 GiB machine
under 12 threads is memory pressure. The tell is `expert_matmul` rising **217%**
for identical matmul work, which no I/O effect can explain, while `loader_wait`
*falls* 26%. The 24-vs-32 pair above is the clean version of the same test.

## What is still unattributed

Of MoE's 63 s at 128 tokens, `loader_wait` accounts for 13 s and `expert_matmul`
for 24.7 thread-seconds. A large remainder is unnamed. The **bf16 router path is
the obvious suspect and is currently untimed**: `GATE_DECODE` covers only the fp8
fallback, because its block sits inside `COLI_V4_DISABLE_BF16_ROUTE` and does not
run in a default build. It is reported as `0.000` rather than dropped — a phase
that is genuinely zero is information, and a phase silently absent is not.

`other` is deliberately a **subtraction**, not a sum of named parts, so a phase
nobody has timed shows up as an unexplained remainder instead of vanishing from a
total that still adds to 100%.

## Gates for the next change

1. **Name MoE's remainder.** Time the bf16 router path and the shared-expert
   forward, so the named phases account for ≥90% of the MoE span. Until then
   "optimise MoE" has no target.
2. **Check the other engines.** V4 was single-threaded here; `colibri`, `olmoe`,
   `inkling` and `kimi_k3` build from the same warned fallback. Whether their
   published numbers carry the same 5–7× is unmeasured.
3. **Re-run the #63 sequence-length sweep under OpenMP.** The ceilings and hit
   rates hold, but every wall-clock figure in that document is single-threaded.

## Reproducing

```
brew install libomp && make -C c deepseek-v4-clean && make -C c deepseek-v4
python3 c/tools/v4_baseline.py --model ~/models/DeepSeek-V4-Flash --ram 32 --runs 3 --cold
```

`v4_phase` and `v4_phase_moe_inner` go to stderr next to `v4_tokens`.

The counters are nanoseconds in `uint64_t` accumulated with `__atomic_fetch_add`,
not seconds in a `double`: `coli_v4_block_profile_add` is called from the
dual-expert-loader and persistent-loader pthreads, so `total += seconds` is a
data race that loses samples while still reporting a plausible number.
`test_v4_block_profile` drives it from 8 threads and asserts the total is exact —
mutation-tested, since a plain `+=` loses 46,812 of 160,000 adds and
ThreadSanitizer names the line.
