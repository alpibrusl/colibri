# Do concurrent sessions overlap enough to amortize a fetch? (#37)

**Date** 2026-08-20 · **Issue** [#37](https://github.com/alpibrusl/colibri/issues/37) · **Prerequisite** #39 (the origin tag)

#37 proposes scheduling by expert residency rather than arrival order, so a
~19 MB fetch is paid once and consumed by every queued token routing to it. It
names its own kill condition and the cheap way to test it:

> **Sessions may not overlap.** If concurrent requests route to disjoint
> experts, there is nothing to amortize. Measurable from existing traces before
> any engine work: take N traces, compute the expected union size per step as a
> function of S.

**The kill condition does not hold.** Bytes fetched per token falls ~62–66%
from S=1 to S=16, and the observed union sits *below* what independent routing
predicts — so there is exploitable correlation, not merely pigeonhole
arithmetic.

## The baseline is the whole difficulty

"The union grows sublinearly in S" is not by itself evidence of anything. With
E experts per layer and K routed per step, the union *must* grow sublinearly
once S·K approaches E. That is the pigeonhole principle. Reporting it alone
would be restating arithmetic as a finding.

So three curves are reported together:

| curve | meaning |
|---|---|
| `disjoint` | S·K — no overlap at all, nothing to amortize |
| `independent` | E·(1−(1−K/E)^S) — uniform independent routing, i.e. the pigeonhole effect and nothing more |
| `observed` | measured from the trace |

`observed` below `independent` means sessions concentrate on shared popular
experts. `observed` at `disjoint` would kill the idea.

## Results

Corpus A and B from `olmoe-routing-structure-2026-08-20.md`; OLMoE-1B-7B,
16 sparse layers, 64 experts, mean top-k 8.7; expert = 6.29 MB.

### A — general knowledge

| S | disjoint | independent | observed | experts/session | MB/token |
|---|---|---|---|---|---|
| 1 | 139 | 139 | 139 | 139.2 | 875.6 |
| 2 | 277 | 259 | 252 | 126.2 | 793.9 |
| 4 | 555 | 452 | 427 | 106.7 | 671.2 |
| 8 | 1024 | 704 | 650 | 81.3 | 511.2 |
| 16 | 1024 | 924 | 847 | 52.9 | 333.1 |

**−62.0% bytes/token**, observed union **8.3% below** independent.

### B — code and mathematics

| S | disjoint | independent | observed | experts/session | MB/token |
|---|---|---|---|---|---|
| 1 | 138 | 138 | 139 | 138.9 | 873.7 |
| 16 | 1024 | 924 | 746 | 46.6 | 293.4 |

**−66.4% bytes/token**, observed union **19.2% below** independent.

### Mixed — the pessimistic case

Sessions drawn from *both* domains, i.e. concurrent requests from unrelated
workloads, which is the shape #37 worries about:

| S | disjoint | independent | observed | experts/session | MB/token |
|---|---|---|---|---|---|
| 1 | 139 | 139 | 140 | 139.7 | 878.7 |
| 2 | 277 | 258 | 253 | 126.7 | 797.3 |
| 4 | 554 | 452 | 422 | 105.6 | 664.3 |
| 8 | 1024 | 704 | 637 | 79.6 | 501.0 |
| 16 | 1024 | 924 | 835 | 52.2 | 328.3 |

**−62.6% bytes/token**, observed union **9.6% below** independent.

Mixing domains costs essentially nothing: −62.6% against −62.0% same-domain.
The overlap being exploited is the model's popular-expert structure, not a
shared workload — the same conclusion #36 reached about layout, and the same
one COUPLE reached about prediction.

## Two methodological traps, both hit

Recorded because both produced plausible numbers before being caught.

**Sampling with replacement inflates overlap.** Drawing S sessions *with*
replacement lets one call stand in for two sessions. On routing constructed to
be perfectly disjoint, S=8 drawn from 16 calls reported **~19% amortization** —
entirely an artifact of the sampler. S concurrent sessions occupy S different
positions, so the draw must be without replacement. A CI case pins this.

**Call ids collide across traces.** Every ROUTE_TRACE numbers its calls from 0,
so merging two traces on the bare id fuses trace A's call 7 with trace B's. That
silently merged unrelated routing decisions into oversized sessions and inflated
mean top-k from 8.7 to 13.2, making the mixed-domain case look *better* than
same-domain (72.5% against 62.0%). Namespacing the id per file corrects it to
62.6%. A CI case pins this too.

## What this does and does not establish

**Does:** the union of experts S concurrent sessions need grows substantially
slower than S, on real routing, and more slowly than independent routing
predicts. There is real work for a residency-ordered scheduler to do, and it
survives mixing workloads.

**Does not:** model a tier cache. This measures the experts a step *needs*, not
the subset that would *miss*. With a warm cache the absolute fetch counts fall
for everyone; what decides whether the scheduler has anything to work with is
the *shape* of the curve, which is what is reported.

**Does not:** predict tok/s. #37 is explicit that tok/s mixes in the multi-row
matmul bandwidth problem the continuous-batching experiment already found, which
is real and separate. Bytes per token is the metric on purpose.

## Remaining scope on #37

- [x] the origin tag on the cache slot — #39
- [x] sessions overlap enough to amortize — this document
- [ ] bytes-fetched-per-token accounting per session and in aggregate
- [ ] an expert-residency-ordered scheduling pass, behind a flag, with a fairness bound
- [ ] the S-sweep at constrained `--ram`, as its own experiment doc

The next slice is the accounting: the origin tag makes a fetch attributable, and
per-session byte counters turn "the fetch was amortized" into a measurable claim
in a live run rather than an offline projection.

## Reproducing

```sh
python3 tools/session_overlap.py trace_a.txt trace_b.txt --max-s 16 --trials 600
python3 tools/session_overlap.py --selftest
```
