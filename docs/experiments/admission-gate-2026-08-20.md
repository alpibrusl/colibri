# Residency-aware admission is not worth building (#37)

**Date** 2026-08-20 · **Issue** [#37](https://github.com/alpibrusl/colibri/issues/37) · **Follows** #45, #46, #58

#58 established that the amortization #37 was opened for already happens: a
shared `step_decode_batch` deduplicates the routed set into `uniq[]`, so each
expert is fetched once per block whatever order the work is processed in. That
left **admission** — which sessions share a step when more are waiting than a
step can carry — as the only lever batching does not already pull.

The re-scope put a gate on it before any engine work. **The gate fails.**

## CHECK 2 — the union mostly already fits

Fraction of (step, layer) decisions where the routed union exceeds a tier of
capacity C, over 16 simulated sessions from corpus A:

| S | C=4 | C=8 | C=16 | C=32 | C=64 |
|---|---|---|---|---|---|
| 2 | 6% | 6% | 0% | 0% | **0%** |
| 4 | 6% | 6% | 6% | 1% | **0%** |
| 8 | 6% | 6% | 6% | 6% | **0%** |
| 16 | 6% | 6% | 6% | 6% | **0%** |

At full residency (C = 64 of 64 experts) the union **never** exceeds the tier,
so admission cannot change anything — the same blind spot the continuous-batching
experiment hit at 99% hits. Even at a tiny C = 4 it only exceeds 6% of the time.
There is far less pressure here than the proposal assumed.

## CHECK 3 — the choice does not pay

Bytes per token, residency-aware admission against arbitrary, across the (S, C)
grid:

| corpus | median gain | reaching the 10% gate | best | worst |
|---|---|---|---|---|
| A | **+1.4%** | 0 of 15 | +7.0% (S=2, C=16) | −11.7% |
| B | **−1.6%** | 1 of 15 | +12.8% (S=4, C=16) | −9.7% |

**Residency-aware admission is frequently *worse* than arbitrary.** That is not
noise in one direction: admission must be decided *before* the forward that
produces the routing, so it can only use the session's previous-token routed set
— the predictor #32 measured at 0.366 recall. A confident choice made on a weak
prediction underperforms an arbitrary one, because it systematically defers the
sessions whose experts are cold without removing the need to fetch them.

This is kill-reason 3 from the re-scope, confirmed: *the information needed to
schedule well does not exist early enough.*

## A flaw in the first version of this gate

Worth recording, because it produced a wrong verdict before it was caught.

The gate was originally written as **"best gain ≥ 10%"**. Taking the maximum over
a grid of ~15 noisy measurements clears any threshold by chance, and it duly did:
corpus B reported PASS from a single +12.8% outlier surrounded by negatives — at
a *different* operating point than corpus A's best. Two corpora whose best points
disagree about where the win is are not describing a win.

The verdict is now the **median** across the grid. A gain worth building for has
to appear at most operating points, not at one. Under that statistic both corpora
fail, and they agree.

## What was not checked

**Kill-reason 1, "no choice to make":** whether N ≈ S in practice needs a real
serve trace under load, which does not exist here. It is a further reason the
item may be moot, not a reason it survives — if deployments rarely have more
waiting sessions than slots, admission has no freedom at all and CHECK 3's
result is academic.

## Verdict

**Close #37's remaining scope item, and the issue with it.** Its four other items
delivered — the origin tag (#39), the overlap gate (#45), bytes-per-token
accounting (#46) and the S-sweep (#58) — and the fifth is answered rather than
abandoned: the batching lever the full-resident experiment could not see turned
out to be pulled already, and the residual scheduling freedom is not worth
spending engine complexity on.

What the issue leaves behind is real: bytes/token is now a first-class live
metric, the amortization is measured at **−53.5%** live and **−41.9%** on real
traces, and the reason not to build a scheduler is written down with the data
rather than left as an open TODO.

## Reproducing

```sh
python3 tools/admission_gate.py --trace trace.txt
python3 tools/admission_gate.py --selftest
```
