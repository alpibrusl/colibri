# DeepSeek V4 Flash: the baseline after the upstream merge (#87)

**Date** 2026-08-24 · **Issues** [#87](https://github.com/alpibrusl/colibri/issues/87),
[#89](https://github.com/alpibrusl/colibri/issues/89) ·
**Tree** `next` at `9f0df11` (post #86 merge, post #91 fix) vs pre-merge `fa4921b` ·
**Model** DeepSeek V4 Flash, 284B MoE, 167 GB on disk ·
**Host** M4 Max, 64 GiB, 12 P-cores + 4 E-cores

This supersedes the *timings* in `deepseek-v4-baseline-2026-08-21.md` and
`deepseek-v4-phase-attribution-2026-08-21.md`, which were measured before the
upstream merge (#86). Their hit rates, byte counts and the sequence-length
finding still stand — those are properties of routing and the cache.

Both arms measured on the same host in the same session, cold every run
(`.coli_usage` removed), alternating between configs so drift hits all of them
equally. `--ram 32`, `--ngen 128`, `V4_ROWS16=0`, 30-token essay prompt, median
of 3. **The host was made genuinely quiet first** — a Colima VM (9.2 GB RSS,
~48% CPU) and an Android emulator were stopped; see "Method" below, this is not
optional.

## Headline

| config | decode | tok/s | TTFT | moe | attn | loader_wait | expert_disk | hit% |
|---|---|---|---|---|---|---|---|---|
| pre-merge `fa4921b`, 3 lanes | 75.1 s | 1.70 | 12.83 s | 40.7 | 28.9 | 13.0 | 41.0 | 72.7 |
| `next`, 3 lanes | 78.7 s | 1.63 | 12.74 s | 44.6 | 28.5 | 17.8 | 41.2 | 72.7 |
| **`next`, defaults (9 lanes)** | **77.4 s** | **1.65** | **12.52 s** | 43.3 | 28.6 | 16.0 | 67.5 | 72.7 |

Roofline is 5.55 tok/s (NVMe-bound at 72.4% hit), so `next` sits at **29.8%** of
it; pre-merge was 30.7%.

## The merge cost 1.55x, and it was one OpenMP thread

Before #91, `next` decoded in **121.6 s** against pre-merge's 75.1 s. The entire
gap was thread count. V4 sized its team as `logical - loaders` = 16 − 3 = 13 on a
12 P-core host, putting one thread on an efficiency core; every `schedule(static)`
region then ends when that thread ends.

| threads | decode |
|---|---|
| 9 | 87.9 s |
| **12** (P-core count) | **78.7 s** |
| 13 (the old default) | 121.6 s |
| 16 | 125.7 s |

A cliff, not a slope — the diagnostic signature worth remembering. It showed in
**all** compute (attention 28.7 → 42.6 s, MoE 40.4 → 72.4 s) while `hit_rate`,
`bytes` and `expert_disk` were unchanged, which is what ruled out I/O immediately.

Fixed in #91 by capping the team with `omp_tune.h`'s `coli_physical_cores()`.
Note the trap that hid it: the runtime banner reports the **loader reservation**
(3), not the thread pool, so reading the banner to check the pool tells you
nothing.

## Residual: ~3-5%, and it is stall, not compute

With threads fixed and lanes matched, `next` is **1.048×** slower than pre-merge
(1.030× at defaults). It is not slower compute:

- attention is unchanged (28.9 → 28.5 s) — the merge did not touch that path
- `expert_disk` is unchanged (41.0 → 41.2 thread-s) and hit rate is identical to
  the digit (72.7%), so the loaders do the same work
- `loader_wait` rises 13.0 → 17.8 s, which more than accounts for the +3.9 s MoE
  delta

So compute waits longer on the same I/O: an overlap/scheduling change, not a
kernel regression. Raising lanes to 9 recovers part of it (`loader_wait` 17.8 →
16.0). Small enough to leave; recorded so it is not rediscovered.

## `routed_matmul` is no longer comparable across the merge

`routed_matmul` reads 19.4 s pre-merge and 4.5 s post. **This is an accounting
change, not a 4× speedup.** Upstream restructured the expert path so most matmul
time no longer flows through `coli_v4_expert_store_matmul_sec`, which is what that
field samples; MoE *total* went up over the same interval. Do not quote the
pre/post `routed_matmul` delta as a result — the unattributed remainder inside
`moe` absorbed it.

## Loader lanes: 9, not 3

#85 measured upstream's 9-lane default as worse on this host and planned to keep 3.
That does not reproduce on the merged tree. At 12 threads: 9 lanes 77.2 s
(`loader_wait` 15.98) vs 3 lanes 79.0 s (`loader_wait` 18.09). Disk thread-seconds
rise as expected, but the stall *falls* — the extra queue depth hides latency here
rather than contending. **Keep upstream's default.**

## Prefill

At the 30-token prompt above, TTFT is flat (0.993× lanes-matched). That prompt is
too short to test a prefill optimization. On the 595-token `p7_longctx` fixture,
where upstream's hoisted `wq_a`/`wkv` activation quantization can actually pay:

| round | pre-merge `fa4921b` | `next` | delta |
|---|---|---|---|
| 1 | 269.092 s | 254.990 s | −5.24% |
| 2 | 269.086 s | 252.435 s | −6.19% |

**~5.7% faster**, both rounds agreeing in direction and magnitude — real, modest,
and in the direction upstream claimed for the hoist.

Worth noting how tight the pre-merge arm is: 269.092 vs 269.086 s, 6 ms apart on a
4.5-minute run. That reproducibility is itself evidence the host was quiet enough
for the comparison to mean something.

Also worth noting on its own: 595 prompt tokens in ~255 s is ~2.3 tok/s of
prefill, which is slow enough to be worth its own investigation.

## Method

Host hygiene is part of the measurement, not a preamble to it. The first attempt
at this was aborted because a VM and an Android emulator were running: on a 64 GiB
host measuring a 155 GB model at `--ram 32`, ~12 GB of foreign RSS lands directly
on the page cache backing the expert hit rate, and sustained foreign CPU steals
from the compute team. It would have produced plausible, wrong numbers.

    ps -Ao rss,%cpu,comm -m | head -8
    pgrep -fl 'VirtualMachine|qemu-system'

Reproduce:

    python3 c/tools/v4_baseline.py --model ~/models/DeepSeek-V4-Flash --ram 32 --runs 3 --cold

For an A/B against another tree, run both on the same host in the same session and
alternate — absolute timings drift between days, so a comparison against a
different day's numbers is only trustworthy if the shared arm reproduces its own.
