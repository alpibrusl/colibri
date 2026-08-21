# Does the matmul_fp8 speedup generalise? (#68)

**Date** 2026-08-21 · **Follows** [#68](https://github.com/alpibrusl/colibri/pull/68) ·
**Finds** [#69](https://github.com/alpibrusl/colibri/issues/69) ·
**Model** DeepSeek V4 Flash, 284B MoE, real 167 GB checkpoint

#68 was measured on two prompts. Two prompts is an anecdote: a kernel change can
easily favour one routing pattern, one prefill/decode balance, or one cache
occupancy. This sweeps content type, prompt length and generation length, with
the two binaries **interleaved per case** — this host thermally throttles under
sustained load, and an all-BEFORE block followed by an all-AFTER block would be
measuring the cooling curve rather than the kernel.

Every run is cold (`.coli_usage` removed first), `--memory-gb 32`,
`--no-dspark`, and the generated text is hashed with the timing line stripped.

## A · content type, 48 generated tokens

| prompt | old | new | speedup | text |
|---|---|---|---|---|
| factual QA (stops at EOS, 9 tok) | 16.78 s | 13.06 s | 1.28× | identical |
| essay | 68.32 s | 49.23 s | **1.39×** | identical |
| code | 64.46 s | 50.35 s | 1.28× | identical |
| maths | 75.49 s | 58.30 s | 1.29× | **differs — see below** |
| list / table | 57.19 s | 45.38 s | 1.26× | identical |
| Spanish | 73.72 s | 57.55 s | 1.28× | identical |
| 450-word context | 530.24 s | 412.58 s | 1.29× | identical |

## B · generation length

| prompt | tokens | old | new | speedup | text |
|---|---|---|---|---|---|
| essay | 16 | 40.22 s | 28.41 s | **1.42×** | identical |
| code | 16 | 41.42 s | 29.38 s | 1.41× | identical |
| Spanish | 16 | 44.47 s | 36.84 s | 1.21× | identical |
| essay | 128 | 132.99 s | 109.28 s | 1.22× | identical |
| code | 128 | 140.32 s | 109.25 s | 1.28× | identical |
| Spanish | 128 | 143.94 s | 112.55 s | 1.28× | identical |

**13 comparisons: min 1.21×, median 1.28×, max 1.42×.** The gain does not depend
on content, on prompt length (6 to 450 words), or on generation length (9 to 128
tokens). It is a property of the kernel, not of the prompt the kernel was tuned
against.

The spread that does exist has an explanation: `matmul_fp8` serves the fp8 dense
weights, so cases where dense work is a larger share of the run gain more. The
450-word-context case is slowest in absolute terms *and* still gains 1.29×,
because its long prefill drops the expert hit rate to 23% and adds streaming
time that the kernel cannot touch.

## The maths prompt, and what it turned out to be

One case in thirteen produced different text between the two binaries. The
obvious reading — "the change is not bit-identical after all" — is wrong, and
the check that settles it is to stop varying the kernel:

```
same binary (old), same prompt, same flags, cold each time
  run A   text sha256 764140e45b25f524
  run B   text sha256 f5a843f66f5ca23e
  run C   text sha256 f5a843f66f5ca23e
```

**The old binary disagrees with itself.** `coli_v4_expert_forward_ref` dispatches
on cache layout — an expert whose slot got rows16-packed runs the packed int4
kernel, one that did not runs the v17 fallback on a different layout — and the
number of packed slots varies run to run (844 / 836 / 846 on the three runs
above). A performance decision silently selects which arithmetic computes each
expert. Filed as **#69**; it is pre-existing and unrelated to #68.

Worth stating plainly, because it weakens something the tree relies on: the
token-exactness oracles can pass or fail by luck, and `replay_diff` (#15) cannot
currently distinguish a real regression from a packing coin-flip.

## Reproducing

`tests/prompts/` holds the seven prompts. Build both binaries, then interleave:

```
git show <before>:c/quant.h > /tmp/old.h   # keep a copy of the current one first
# build each into its own binary, then per prompt:
rm -f $MODEL/.coli_usage
./v4_<variant> $MODEL --prompt-file tests/prompts/p3_code.txt \
    --memory-gb 32 --max-tokens 48 --no-dspark
```

Hash the output with the `TUNE decode` line stripped — it carries the timing and
would differ on every run regardless.
