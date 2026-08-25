# Where V4's prefill time goes — and why raising the chunk cap is not the lever

**Date** 2026-08-25 · **Tree** `next` at `2971608` (post decode-hoist #98) ·
**Model** DeepSeek V4 Flash · **Host** M4 Max, 64 GiB, 12 P-cores ·
**Prompt** `tests/prompts/p7_longctx.txt`, 595 tokens, `--memory-gb 32`, `V4_ROWS16=0`

Prefill had never been profiled. It was flagged as "~8x off roofline" in
`v4-batching-gate-2026-08-24.md`, which turns out to be the wrong frame.

## The profile

```
wall 252.0s   attention 108.0 (43%)   moe 131.4 (52%)   other 12.6 (5%)
                                      routed_matmul 96.1  shared_expert 18.9
                                      loader_wait 2.8
v4_tokens  prompt=595  hit_rate=5.829  bytes=331.8 GB
```

**`loader_wait` is 2.8 s of 252 s — 1.1%.** Prefill is not I/O-bound; the loaders keep up
completely. It is ~95% compute, split **43% attention / 38% routed matmul / 7% shared
expert**. The earlier "off its I/O roofline" framing was wrong.

Two things are startling and neither is the bottleneck:

- **hit rate 5.8%** — a 595-token prefill touches nearly every expert, so the cache is
  useless here by construction.
- **331.8 GB streamed, twice the 167 GB model** — the cache thrashes and reloads. It costs
  almost nothing in wall time because the disk keeps up and the reads overlap compute.

## A hypothesis, and its refutation

The decode hoist (#98) makes `matmul_fp8` amortize dequantization across the batch, worth
1.35x at S>=16. The in-code note at `deepseek_v4.c:11333` records prefill chunk 64 -> 128 as
worth only 3%, measured *before* that fix. It seemed to follow that the flat curve **was**
the defect, and that with the hoist in place a wider chunk should now pay.

Measured post-hoist, same host and prompt:

| `V4_PREFILL_CHUNK` | TTFT | attention | routed_matmul |
|---|---|---|---|
| 128 | 250.5 s | 108.2 | 95.4 |
| 64 | 258.3 s | 109.6 | 96.7 |
| 32 | 269.6 s | 110.9 | 96.9 |

**64 -> 128 is 3.1% — the same as before the hoist.** The curve did not steepen.

And the full sweep says something sharper: **`routed_matmul` is flat**, 96.9 -> 96.7 -> 95.4
across a 4x change in chunk width. The expert matmuls do not batch at *any* width the cap
permits. What little TTFT does improve (269.6 -> 250.5, 7.6%) comes from attention moving
slightly and from per-chunk fixed overhead being amortized over fewer chunks -- not from the
expert path at all.

The reason is a result from the batching gate that should have been applied *before* running
this: with top-6 of 256, tokens per activated expert is

| chunk | tokens per activated expert |
|---|---|
| 32 | 1.41 |
| 64 | 1.92 |
| 128 | 3.15 |
| **171** | **4.08** — first to reach the hoist threshold |
| 512 | 12.00 |

Both tested points sit **below** `FP8_HOIST_MIN`. The experiment compared two places on the
same flat stretch and could not have discriminated. The expert matmuls do not batch at any
chunk width the current cap permits.

## What raising the cap is actually worth

`V4_PREFILL_CHUNK` clamps to [1, 128] because the batch kernels' contract is 128 (six sites
validate `batch > 128`). Lifting it to 512 would put experts at S=12 and engage the hoist on
`routed_matmul`:

    routed_matmul 95.4s -> 70.7s     TTFT 250.5s -> 225.8s    1.11x

**~11%**, for a change that touches six validators plus every batch-scaled buffer, and grows
prefill activation memory 4x. Real, bounded, and not obviously worth it first.

## Inside attention: the largest stage is single-threaded

`DSV4_ATTN_PROF=1` already instruments every stage; nothing needed adding. Totals across the
whole 595-token prefill (107.7 s, matching `v4_phase`'s 108.0 s):

| stage | time | % of attention | % of prefill |
|---|---|---|---|
| **attn** | 49.74 s | **46.2%** | **19.8%** |
| **wo** | 30.67 s | 28.5% | 12.2% |
| qb | 12.21 s | 11.3% | 4.9% |
| idx | 6.13 s | 5.7% | 2.4% |
| comp | 5.75 s | 5.3% | 2.3% |
| qa | 2.21 s | 2.1% | 0.9% |
| kv | 0.82 s | 0.8% | 0.3% |
| rope | 0.13 s | 0.1% | — |

`wo` and `qb` are fp8 batch matmuls at chunk width 128, so they already take the #98 hoist —
30.7 + 12.2 s at 1.35x accounts almost exactly for the ~13 s the hoist saved on prefill,
which is a useful independent check that it engaged.

**`attn` is 49.7 s — 20% of the entire prefill — and it runs single-threaded.**
`coli_v4_sparse_attention_ref` (`deepseek_v4.c:4131`) contains no OpenMP pragma, and its
caller is a plain `for (item = 0; item < batch; item++)` loop. Eleven of twelve threads are
idle for a fifth of prefill.

It is not free money, and the two reasons are worth stating because they shape the fix:

1. **Shared scratch.** `v4_attn_scratch` hands out slots from a `static void *arena[24]` —
   process-global, not per-thread. The item loop takes slots 21 and 22, so threading it as-is
   would race.
2. **A sequential KV ring.** Each item memcpys its own kv into `state->kv` at
   `position % window_size` and then attends over that ring, so item *i+1* reads what item *i*
   just wrote. Causal, within the batch.

The second is the real one, and it has a standard answer: write every item's kv into the ring
first, then attend in parallel with per-item masking. The masking already exists — the loop
builds `indices[i] = i <= position ? i : -1` — so the information needed is present; it is the
ordering that has to change, not the arithmetic. Each item's own accumulation order would be
untouched, so bit-identity is a reasonable target rather than a hope.

Ceiling: even at a conservative 8x from 12 threads, 49.7 s -> ~6 s saves ~44 s of 250 s, about
**1.2x on prefill** — and it composes with the chunk cap's 1.11x.

## What is worth doing instead

**Parallelize the `attn` stage.** It is 20% of prefill running on one core, the ceiling is
~1.2x, and the blockers are per-thread scratch plus a KV-write/attend split — both ordinary
work, neither requiring new numerics. That beats the chunk cap's 1.11x, which costs six
validator changes and 4x the activation memory.

The two compose: attn parallelized *and* the cap lifted would put prefill near
250 s -> ~185 s. Neither touches decode, where a single session is S=1 by construction.

Reproduce:

    V4_ROWS16=0 V4_PREFILL_CHUNK=<n> ./deepseek_v4 <model> \
        --prompt-file tests/prompts/p7_longctx.txt --max-tokens 2 --memory-gb 32
