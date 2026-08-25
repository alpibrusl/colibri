# The premise check: 284B streamed vs 32B resident, same machine

**Date** 2026-08-25 · **Host** M4 Max, 64 GiB, 12 P-cores, quiet (VM and emulator stopped) ·
**A** colibri `next` @`2971608` + DeepSeek V4 Flash — 284B MoE, 167 GB, streamed from NVMe ·
**B** ollama + `muse-glimmer:30b-mlx` — 32.3B dense, nvfp4, 21 GB, fully RAM-resident

Every optimization in this tree is a few percent against a ceiling. This measures the
ceiling's premise instead: on *this* hardware, is a huge model streamed from disk a better
use of the machine than a smaller one that fits in memory?

## Speed

| | colibri, 284B streamed | dense 32B resident | ratio |
|---|---|---|---|
| decode | 1.68 tok/s | 47.3 tok/s | **28×** |
| prefill | 2.48 tok/s | 399 tok/s | **161×** |

For scale: colibri's own NVMe-bound roofline is 5.55 tok/s. **Perfect execution of every
remaining idea in this repo would still leave the dense model 9× ahead on decode.** The gap
is not an engineering gap.

## Quality, on four ordinary prompts

The 32B answers are complete and correct. On `p3_code` it produced a two-pointer merge with
a docstring, worked examples, both empty-input edge cases, and a complexity note — a good
answer by any standard. On `p1_factual` both models get Paris.

**Four prompts is not a quality benchmark and this section should not be read as one.**
Simple factual, code, math and essay tasks do not discriminate between a good 32B and a
frontier MoE. The tasks where 284B should win — hard multi-step reasoning, niche knowledge,
long-context synthesis — were not tested. Two further asymmetries: the dense model here is a
*reasoning* model that emits a thinking trace, which is a different product category, and the
two use different quantizations (nvfp4 vs fp8).

What the four prompts do establish is narrower and still useful: **on ordinary work the gap
is not visible**, so the 28× is not buying anything on this class of task.

## What this means

To prefer the streamed 284B you have to believe the quality gap is worth a **28× throughput
penalty** — and that it shows up on the work you actually do. That is a high bar, and nothing
measured here clears it.

This does not make colibri pointless. It relocates the point:

- **Where it is genuinely the only option**: models that do not fit at any quantization —
  Kimi K3 at ~1.6 TB has no resident alternative on a workstation. The value is *feasibility*,
  not throughput.
- **Where it is a research vehicle**: the tier hierarchy, routing telemetry and the offline
  gates in `docs/experiments/` are real knowledge about streaming MoE inference. This session
  alone produced five clean negatives.
- **Where it is not the answer**: "a fast local LLM on a laptop." A resident smaller model
  wins by more than an order of magnitude and no amount of kernel work closes that.

## The decode lever, in this light

`int4` dense weights are −33% traffic *and* fewer decode instructions, hitting the dominant
term twice — optimistically ~2.2–2.5 tok/s. Against 47 tok/s from a resident model, that
changes nothing strategically. It is worth doing only if the goal is the feasibility case
above, where the alternative is not running the model at all.

## Reproduce

    ollama run muse-glimmer:30b-mlx --verbose "$(cat c/tests/prompts/p3_code.txt)"
    cd c && V4_ROWS16=0 python3 ./coli run --model ~/models/DeepSeek-V4-Flash \
        --ram 32 --ngen 128 "<same prompt>"

Run them sequentially, never concurrently — two multi-threaded engines on 12 P-cores produce
nonsense in both directions.
