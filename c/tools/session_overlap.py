#!/usr/bin/env python3
"""Is there anything for expert-major scheduling to amortize? (#37)

#37 proposes scheduling by expert residency instead of arrival order, so a
~19 MB expert fetch is paid once and consumed by every queued token that routes
to it. It also names the thing that would kill it, and the cheap way to check:

> **Sessions may not overlap.** If concurrent requests route to disjoint
> experts, there is nothing to amortize. Measurable from existing traces before
> any engine work: take N traces, compute the expected union size per step as a
> function of S.

This is that measurement. No engine work, no checkpoint, no drive.

## The baseline is the whole difficulty

"The union grows sublinearly in S" is not by itself evidence of anything. With
only E experts in a layer and K routed per step, the union MUST grow sublinearly
once S*K approaches E -- that is the pigeonhole principle, not a property of the
model. Reporting sublinearity alone would restate arithmetic as a finding.

So three curves are reported together:

    disjoint      S*K            no overlap at all -- nothing to amortize
    independent   E*(1-(1-K/E)^S)  what uniform, independent routing gives you
                                   (the pigeonhole effect, and nothing more)
    observed      measured from the trace

`observed` below `independent` means sessions concentrate on shared popular
experts -- real, exploitable correlation on top of the arithmetic. `observed` at
`independent` still amortizes, but by exactly as much as a coin-flip model would
predict. `observed` at `disjoint` kills the idea.

The primary metric is #37's own: **bytes fetched per token**, i.e. union size
divided by S. tok/s is deliberately not modelled -- the issue is explicit that
tok/s mixes in a separate kernel-bandwidth problem.

## What this does not model

A tier cache. This measures the union of experts a step NEEDS, not the subset
that would actually miss. With a warm cache the absolute fetch counts are lower
for everyone; the SHAPE of the curve -- whether S sessions need proportionally
fewer distinct experts than S separate ones -- is what decides whether
residency-ordered scheduling has anything to work with, and that is what this
reports.

Usage:
  python3 tools/session_overlap.py trace.txt [trace2.txt ...] [--max-s 16] [--trials 400]
  python3 tools/session_overlap.py --selftest
"""
import os
import random
import sys
from collections import defaultdict

# 6.29 MB: one OLMoE int8 merged expert, the reference checkpoint in
# docs/experiments/olmoe-routing-structure-2026-08-20.md. Override with
# --expert-bytes for a different container.
DEFAULT_EXPERT_BYTES = 6_291_456

# Below this, the observed curve is close enough to S*K that there is nothing
# worth reordering a scheduler for.
AMORTIZATION_THRESHOLD = 0.10


def parse(paths):
    """{layer: [routed set per call]} — one entry per (call, layer) in the trace."""
    per = defaultdict(dict)
    for fidx, path in enumerate(paths):
        with open(path) as fh:
            for line in fh:
                p = line.split()
                if len(p) < 4:
                    continue
                try:
                    call, layer = int(p[0]), int(p[2])
                except ValueError:
                    continue
                # Namespace the call id per FILE. Every ROUTE_TRACE numbers its
                # calls from 0, so keying on the bare id fuses trace A's call 7
                # with trace B's -- which silently merges two unrelated routing
                # decisions into one oversized "session" and inflates mean top-k.
                s = per[layer].setdefault((fidx, call), set())
                for tok in p[3:]:
                    try:
                        s.add(int(tok.split(":", 1)[0]))
                    except ValueError:
                        pass
    return {layer: list(calls.values()) for layer, calls in per.items()}


def independent_union(experts, k, s):
    """E*(1-(1-K/E)^S): the union under uniform independent routing."""
    if experts <= 0 or k <= 0:
        return 0.0
    return experts * (1.0 - (1.0 - k / experts) ** s)


def observed_union(sets, s, trials, rng):
    """Mean |union| of S sets drawn from the trace, as concurrent sessions.

    Drawn WITHOUT replacement. S concurrent sessions occupy S different
    positions, so sampling the same call twice would model one session as two
    -- and it inflates the answer: with replacement, S=8 drawn from 16 calls
    collides often enough to report ~19% amortization on routing constructed to
    have exactly none. That is a property of the sampler, not of the model.

    When the trace has fewer distinct calls than S there is no honest draw, so
    the whole trace is the union and the caller is warned (see `thin`).
    """
    if not sets:
        return 0.0
    if len(sets) <= s:
        u = set()
        for x in sets:
            u |= x
        return float(len(u))
    total = 0
    for _ in range(trials):
        u = set()
        for idx in rng.sample(range(len(sets)), s):
            u |= sets[idx]
        total += len(u)
    return total / trials


def thin(by_layer, max_s):
    """Layers with too few distinct calls to draw max_s sessions from."""
    return [layer for layer, sets in by_layer.items() if len(sets) <= max_s]


def analyse(by_layer, max_s, trials, seed=0):
    """Per S: mean union, the two baselines, and experts per session-step."""
    rng = random.Random(seed)
    experts = max((max(s) for sets in by_layer.values() for s in sets if s), default=0) + 1
    k = 0
    n = 0
    for sets in by_layer.values():
        for s in sets:
            k += len(s)
            n += 1
    k = k / n if n else 0

    rows = []
    s_val = 1
    while s_val <= max_s:
        obs = dis = ind = 0.0
        for layer, sets in by_layer.items():
            if not sets:
                continue
            obs += observed_union(sets, s_val, trials, rng)
            dis += min(s_val * (sum(len(x) for x in sets) / len(sets)), experts)
            ind += independent_union(experts, sum(len(x) for x in sets) / len(sets), s_val)
        rows.append({"s": s_val, "observed": obs, "disjoint": dis, "independent": ind,
                     "per_session": obs / s_val})
        s_val *= 2
    return {"experts": experts, "mean_k": k, "layers": len(by_layer), "rows": rows}


def verdict(result):
    """Does bytes/token actually fall with S, and does it beat the coin flip?"""
    rows = result["rows"]
    if len(rows) < 2:
        return 0.0, 0.0, "not enough points"
    base, last = rows[0]["per_session"], rows[-1]["per_session"]
    drop = 1.0 - (last / base) if base else 0.0
    # how far below the independent-routing prediction the observed union sits
    corr = 0.0
    if last and rows[-1]["independent"]:
        corr = 1.0 - (rows[-1]["observed"] / rows[-1]["independent"])
    if drop <= AMORTIZATION_THRESHOLD:
        note = "sessions do not overlap enough — nothing to amortize"
    elif corr > 0.05:
        note = "amortizes, and by MORE than independent routing predicts"
    else:
        note = "amortizes, at about what independent routing predicts"
    return drop, corr, note


def report(result, expert_bytes, label="", thin_layers=()):
    print(f"{label}{result['layers']} sparse layers, {result['experts']} experts, "
          f"mean top-k {result['mean_k']:.1f}")
    print(f"  {'S':>3}  {'disjoint':>9} {'independent':>11} {'observed':>9}   "
          f"{'experts/sess':>12} {'MB/token':>9}")
    for r in result["rows"]:
        print(f"  {r['s']:>3}  {r['disjoint']:>9.0f} {r['independent']:>11.0f} "
              f"{r['observed']:>9.0f}   {r['per_session']:>12.1f} "
              f"{r['per_session']*expert_bytes/1e6:>9.1f}")
    if thin_layers:
        print(f"  NOTE: {len(thin_layers)} layer(s) have <= S distinct calls; their union is the"
              f" whole trace, which OVERSTATES overlap. Capture a longer trace.")
    drop, corr, note = verdict(result)
    print(f"  bytes/token falls {drop:.1%} from S=1 to S={result['rows'][-1]['s']}; "
          f"observed union is {corr:+.1%} vs independent routing")
    print(f"  -> {note}")
    return drop, corr


# ---------------------------------------------------------------- selftest
def selftest():
    ok = True
    rng = random.Random(1)

    # 1. disjoint sessions: every call routes to its own private block, so the
    #    union must grow exactly linearly and the verdict must be "nothing".
    disjoint = {0: [set(range(i * 4, i * 4 + 4)) for i in range(256)]}
    res = analyse(disjoint, max_s=8, trials=400)
    drop, _, note = verdict(res)
    if drop > AMORTIZATION_THRESHOLD:
        print(f"selftest: disjoint routing reported {drop:.1%} amortization, expected ~0")
        ok = False
    if "nothing to amortize" not in note:
        print(f"selftest: disjoint verdict was {note!r}")
        ok = False

    # 2. identical sessions: every call routes to the same set, so S sessions
    #    need exactly what 1 does and per-session cost falls like 1/S.
    same = {0: [set(range(4)) for _ in range(256)]}
    res = analyse(same, max_s=8, trials=200)
    drop, _, _ = verdict(res)
    if drop < 0.8:
        print(f"selftest: identical routing reported only {drop:.1%} amortization")
        ok = False

    # 3. the independent baseline must match a direct simulation, or the curve
    #    everything is judged against is wrong.
    experts, k, s = 64, 8, 4
    sets = [set(rng.sample(range(experts), k)) for _ in range(500)]
    sim = observed_union(sets, s, 2000, random.Random(2))
    pred = independent_union(experts, k, s)
    if abs(sim - pred) / pred > 0.03:
        print(f"selftest: independent baseline {pred:.1f} vs simulated {sim:.1f}")
        ok = False

    # 4. uniform random routing must land AT the independent line, not below it
    res = analyse({0: sets}, max_s=8, trials=400)
    _, corr, _ = verdict(res)
    if abs(corr) > 0.05:
        print(f"selftest: uniform routing sits {corr:+.1%} off the independent line")
        ok = False

    print("selftest: ok" if ok else "selftest: FAILED")
    return 0 if ok else 1


def main():
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        raise SystemExit(selftest())
    traces, max_s, trials = [], 16, 400
    expert_bytes = DEFAULT_EXPERT_BYTES
    i = 0

    def value(flag):
        nonlocal i
        a = args[i]
        if "=" in a:
            i += 1
            return a.split("=", 1)[1]
        if i + 1 >= len(args):
            raise SystemExit(f"{flag}: expected a value")
        i += 2
        return args[i - 1]

    while i < len(args):
        a = args[i]
        if a.startswith("--max-s"):
            max_s = int(value("--max-s"))
        elif a.startswith("--trials"):
            trials = int(value("--trials"))
        elif a.startswith("--expert-bytes"):
            expert_bytes = int(value("--expert-bytes"))
        elif a.startswith("--"):
            raise SystemExit(f"unknown argument: {a}")
        else:
            traces.append(a)
            i += 1
    if not traces:
        raise SystemExit(__doc__)

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import trace_health
        gate = trace_health.gate_stats(traces)
        warn = trace_health.warning(gate)
    except Exception:
        warn = None

    # Same-domain: sessions drawn from one trace.
    for t in traces:
        bl = parse([t])
        report(analyse(bl, max_s, trials), expert_bytes,
               label=f"{os.path.basename(t)}: ", thin_layers=thin(bl, max_s))
        print()

    # Mixed: sessions drawn from ALL traces, the pessimistic case for #37 --
    # concurrent requests from different workloads.
    if len(traces) > 1:
        bl = parse(traces)
        report(analyse(bl, max_s, trials), expert_bytes,
               label=f"mixed ({len(traces)} domains): ", thin_layers=thin(bl, max_s))
        print()

    if warn:
        print(f"  {warn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
