# OLMoE routing structure: layout clusters, weights do not (#31, #32, #36)

**Date** 2026-08-20 · **Issues** [#31](https://github.com/alpibrusl/colibri/issues/31), [#32](https://github.com/alpibrusl/colibri/issues/32), [#36](https://github.com/alpibrusl/colibri/issues/36)

Three research issues each opened with a falsifiable gate and the instruction to
measure before writing engine code. All three gates need a real *trained* MoE —
`trace_health.py` exists because a random-weight checkpoint scored over threshold
on two of them. This is the first run of all three against one.

**Results in one line each:**

| Issue | Question | Verdict |
|---|---|---|
| #36 | Would a co-activation-clustered disk layout cut reads? | **PASS** — 31.4% fewer reads, held out, and it survives a domain change |
| #32 | Does token *t*'s routing predict token *t+1*'s? | **PASS** — +0.129 recall over marginal heat, with a caveat below |
| #31 | Are experts redundant in weight space? | **FAIL, decisively** — 0/48 groups. Close it. |

## Environment

| | |
|---|---|
| Host | Apple M4 Max, 16 CPU cores, 64 GB RAM, macOS 25.2 |
| Model | `allenai/OLMoE-1B-7B-0125-Instruct` (16 layers, 64 experts/layer, top-8) |
| Engine snapshot | `tools/convert_olmoe_merged.py` → int8 merged, 6.9 GB |
| Engine | `c/olmoe`, `next` @ `aa4f7c8`, `CACHE=64 BITS=8 CTX=2048 TEMP=0 PILOT=0` |
| Weights for #31 | the unconverted HF checkpoint (13.8 GB bf16) |

`convert_olmoe_merged.py --repo` downloads and converts one shard at a time,
deleting each source shard after extraction, so the whole pipeline peaked about
14 GB above the converted output rather than needing 14 GB *plus* the checkpoint.

## Corpora

Two deliberately different domains, because #36 names "one layout cannot serve
two workloads" as a thing that could kill it.

| Corpus | Domain | Prompts | Trace rows | Forwards |
|---|---|---|---|---|
| A | general knowledge, one continuous session | 15 | 20,016 | 915 |
| B | code and mathematics, `/reset` between prompts | 10 | 14,352 | 610 |

Both cleared `trace_health.py` before any tool was allowed a verdict:

| Corpus | mean normalized gate entropy | mean top gate | verdict |
|---|---|---|---|
| A | 0.9521 | 0.2307 | discriminating |
| B | 0.9491 | 0.2323 | discriminating |
| *(random-weight fixture, for contrast)* | *0.9971–0.9999* | *0.1283–0.1467* | *DEGENERATE, no verdict* |

The contrast row is not decoration. The same `make_glm_bench_model.py` fixture
used elsewhere in this run is rejected outright, which is the check doing its job.

## #36 — layout: PASS

`tools/expert_layout.py`, fitted on the first half of A's forwards and scored on
the second half it never saw:

| layout | reads | experts/read |
|---|---|---|
| identity (today's container order) | 53,180 | 1.187 |
| **clustered** | **36,477** | **1.731** |
| random control | 52,673 | 1.199 |

**+0.314 vs identity, +0.307 vs the random control.**

The random control barely moves (52,673 against identity's 53,180), which is the
result that matters: the metric is not responding to *any* permutation, so the
clustered number is measuring co-activation and not the act of shuffling.

### Cross-domain

Fitting on one domain and scoring on the other, the explicit test for #36's
stated risk:

| fit → score | vs identity | vs random | verdict |
|---|---|---|---|
| A (general) → B (code/math) | **+0.314** | +0.304 | holds |
| B (code/math) → A (general) | **+0.278** | +0.271 | holds |
| B → B (in-sample, for scale) | +0.397 | +0.394 | — |

A layout fitted on one domain keeps roughly 70–80% of its in-sample gain on a
domain it was never fitted to. The workload-specificity objection does not hold
at this scale: co-activation structure here is largely a property of the model.

**Recommendation: #36 has earned its engine slice** — the container rewrite plus
the loader's `expert_id -> slot` mapping. The permutation is emitted already.

## #32 — sequence-level prediction: PASS, with a caveat

`tools/route_temporal.py` on corpus A, budget = true-set size ×1, 14,624
(transition, layer) scores:

| predictor | recall@K |
|---|---|
| marginal heat | 0.2568 |
| previous-token identity | 0.3658 |
| **learned conditional** | **0.3862** |

**Temporal gain over marginal: +0.1294**, over the tool's threshold.

**The caveat, stated plainly:** the degenerate random-weight trace that motivated
`trace_health.py` scored **+0.135** on this same metric — larger than the +0.1294
measured here on a real model. This trace passes the health check and that one did
not, so the number here is meaningful and that one was an artifact; but the
magnitudes being comparable means this gain is *modest*, and it should not be
quoted as though structureless initialization could not produce something similar
looking. A second corpus and a per-layer breakdown would firm this up.

Also worth noting for whoever picks this up: corpus A is a single continuous chat
session, so the tool saw **1 sequence**. Multi-sequence capture (as corpus B does
with `/reset`) is the better shape for this measurement.

## #31 — cross-expert weight redundancy: FAIL

`tools/expert_redundancy.py` over the raw HF checkpoint, **all 16 layers × 3
projections = 48 groups**. Gate: 95% energy at rank `r ≤ min(O, I)/4`.

**0 of 48 groups pass.** Representative rows:

| layer | proj | r95(stack) | gate threshold | r95(vec) | cos_max |
|---|---|---|---|---|---|
| 0 | gate | 1670 | 256 | 60/64 | 0.112 |
| 0 | up | 1699 | 256 | 60/64 | 0.003 |
| 0 | down | 920 | 256 | 60/64 | 0.002 |
| 15 | gate | 1729 | 256 | 60/64 | 0.012 |
| 15 | up | 1731 | 256 | 60/64 | 0.005 |

This is not a near miss. `r95(stack)` for the gate/up projections *exceeds
min(O, I)* itself — there is no shared low-dimensional basis at all, let alone one
at `inter/4`. `r95(vec) = 60/64` says you would need 60 of the 64 experts' worth
of directions to reconstruct the set. Maximum post-alignment cosine across the
most similar pairs tops out at 0.112 and is usually below 0.01: after correcting
for the permutation and GLU sign-scale gauge that #31 correctly identified,
trained experts in this checkpoint are **very close to mutually orthogonal**.

The tool's own selftest was run first and passes, so the machinery distinguishes
iid (fail), shared-basis (pass), and permuted near-copies (cosine −0.072 → 1.000
after alignment). It is measuring what it claims to.

**Recommendation: close #31 with this data.** A clean negative was named as an
acceptable outcome, and this is one. Orthogonality is arguably what a
well-trained router *should* produce — redundant experts would be wasted capacity.

Scope of the claim: one checkpoint, one family. If a differently-trained MoE
(heavier load-balancing loss, far more experts, upcycled-from-dense) is ever a
target, the gate is cheap to re-run — 2m43s for all 48 groups here.

## Reproducing

```sh
python3 tools/convert_olmoe_merged.py --repo allenai/OLMoE-1B-7B-0125-Instruct --out ~/models/olmoe_merged
SNAP=~/models/olmoe_merged CHAT=1 MAX_NEW=60 CTX=2048 PILOT=0 TEMP=0 \
    ROUTE_TRACE=trace_a.txt ./olmoe 64 8 < prompts_a.txt

python3 tools/trace_health.py      trace_a.txt          # gate every verdict on this
python3 tools/expert_layout.py     trace_a.txt --out=layout.json --perm=perm.json
python3 tools/route_temporal.py    trace_a.txt --out=temporal.json
python3 tools/expert_redundancy.py ~/models/olmoe-hf    --out=redundancy.json
```

Both `--layers=0,4,8` and the space-separated `--layers 0,4,8` work. The
space form — the one this tool's own usage line shows — used to raise
`IndexError`; fixed in the same change as this document.
