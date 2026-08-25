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

| `V4_PREFILL_CHUNK` | TTFT | routed_matmul |
|---|---|---|
| 128 | 250.5 s | 95.4 |
| 64 | 258.3 s | 96.7 |

**3.1% — the same as before the hoist.** The curve did not steepen.

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

## What is worth doing instead

**Attention is 43% of prefill and untouched by any of the above.** It is the largest single
term, it grows with sequence length, and nobody has profiled inside it. That is where the
next look belongs — the chunk cap is a known 11% with known cost, and attention is an
unknown fraction of a larger number.

Reproduce:

    V4_ROWS16=0 V4_PREFILL_CHUNK=<n> ./deepseek_v4 <model> \
        --prompt-file tests/prompts/p7_longctx.txt --max-tokens 2 --memory-gb 32
