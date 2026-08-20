# The bytes/token curve bends — without a scheduler (#37)

**Date** 2026-08-20 · **Issue** [#37](https://github.com/alpibrusl/colibri/issues/37) · **Follows** #39, #45, #46

#37's last scope item: *"the S-sweep at constrained `--ram`, as its own experiment
doc"*. This is it, and it changes what the remaining engine work has to justify.

| S | tokens | bytes | MB/token | aggregate tok/s |
|---|---|---|---|---|
| 1 | 40 | 9.74 GB | **243.5** | 17.11 |
| 2 | 80 | 16.67 GB | 208.3 | 17.78 |
| 4 | 160 | 26.02 GB | 162.6 | 18.84 |
| 8 | 320 | 36.25 GB | **113.3** | 20.77 |

**Bytes per token falls 53.5% from S=1 to S=8**, and aggregate throughput rises
with it. The curve bends, which is exactly what the issue predicted the
full-resident experiment could not see:

> the bytes/token curve should bend with S in a way the full-resident
> measurement had no way to show, because at 99% hits there are no fetches to
> amortize.

## The finding that matters: no scheduler was involved

This is stock `run_serve_mux`. Sessions are admitted in slot-index order, exactly
as before, and every row goes into one `step_decode_batch`. **The amortization
#37 proposes to build a scheduler for is already happening**, because a shared
batch fetches each expert once for every row that routes to it.

So the residency-ordered scheduling pass in #37's scope now has to justify itself
against *this* baseline rather than against FIFO-with-no-batching. Its remaining
value is in the cases batching does not cover — sessions that arrive too late to
join a batch, and admission order when the tier cannot hold the union — not in
the amortization itself.

## Read this number as an upper bound

The fixture is `make_glm_bench_model.py`'s, and `trace_health.py` rejects its
routing outright:

```
DEGENERATE TRACE: mean normalized gate entropy 0.9971 >= 0.99
```

Random weights mean every session routes to nearly the *same* experts, which is
the best possible case for amortization. The real magnitude is the offline
measurement in #45, on a trained OLMoE:

| | bytes/token, S=1 → S=8 |
|---|---|
| this sweep (degenerate fixture) | **−53.5%** |
| #45, real trace | **−41.9%** |

The fixture overstates by roughly 12 points. Both agree on direction and both are
substantial; the trustworthy magnitude is #45's, and this run's contribution is
that the effect is **live and measurable in a running server**, not only
projected from traces.

## Method

`RAM_GB=0.35` forces the tier to miss — full residency is the one configuration
where this measurement is guaranteed to show nothing, which is the trap the
continuous-batching experiment fell into. Each session emits 40 tokens at
`TEMP=1.0`; the metric is the `BYTES` line's engine-wide `all_per_token` from
#46, which exists precisely because a shared fetch cannot be billed to one
session.

`TEMP=0` is unusable here for a reason worth recording: with random weights the
greedy argmax is a fixed token, and on this fixture that token is the EOS id, so
every request emits zero tokens and every counter reads zero. Sampling avoids it.

## What this does not claim

**Not tok/s as a headline.** Aggregate throughput does rise (17.11 → 20.77), but
#37 is explicit that tok/s mixes in the multi-row matmul bandwidth problem the
continuous-batching experiment found, which is real and separate. Bytes per token
is the metric on purpose.

**Not a real workload.** One fixture, one prompt shape, 40 tokens per session,
degenerate routing. It demonstrates the mechanism and the instrumentation; #45
carries the magnitude.

## Suggested resolution for #37

- [x] the origin tag on the cache slot — #39
- [x] sessions overlap enough to amortize — #45
- [x] bytes-fetched-per-token accounting — #46
- [x] the S-sweep at constrained `--ram` — this document
- [ ] an expert-residency-ordered scheduling pass — **re-scope or close**

The scheduling pass is the only item left, and the sweep has moved the goalposts
under it: batching already delivers the amortization the issue was opened for. A
scheduler should now be proposed against a specific gap batching leaves, with its
own falsifiable gate, rather than inheriting this issue's original motivation.

## Reproducing

```sh
# S sessions, one slot each, 40 tokens apiece, tier deliberately missing
SNAP=<fixture> SERVE=1 SERVE_BATCH=1 KV_SLOTS=$S RAM_GB=0.35 ./colibri 2 16 16 < submits.txt
# read the last BYTES line: <id> <turn_bytes> <turn/tok> <all_bytes> <all/tok>
```
