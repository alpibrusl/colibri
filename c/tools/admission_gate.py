#!/usr/bin/env python3
"""Is residency-aware admission worth building? (#37, re-scoped)

#58 showed the amortization #37 was opened for already happens: a shared
step_decode_batch deduplicates the routed set, so each expert is fetched once
per block whatever order the work is processed in. What batching does NOT decide
is ADMISSION -- which sessions share a step when more are waiting than a step can
carry. That choice is free today (slot-index order) and could be made on
residency instead.

This runs the two cheap kill-checks from the re-scope, in order:

  CHECK 2  "the union already fits"
           If the routed union of S sessions is smaller than the tier at
           realistic cache sizes, everything is resident and admission cannot
           matter. This is the trap the continuous-batching experiment fell into
           at 99% hits, and it kills the item outright if it holds.

  CHECK 3  "the predictor is too weak"
           Admission must be decided BEFORE the forward that produces the
           routing, so it can only use PREDICTED routing. The only signal
           available is the session's previous token's routed set, which #32
           measured at 0.366 recall. If a residency-aware choice made on that
           is indistinguishable from an arbitrary one, the information needed to
           schedule well does not exist early enough.

The gate, fixed before measuring, as #31/#36 did:

    PASS if residency-aware admission reduces bytes/token by >= 10% against
    arbitrary admission at realistic (N, S, C) -- otherwise the choice is not
    worth making and slot-index order is fine.

The verdict is the MEDIAN across the (S, C) grid, not the best point. Taking the
maximum over ~15 noisy measurements clears any threshold by chance: the first
version of this gate did exactly that and reported PASS on one corpus from a
lone outlier surrounded by negative gains, at a different operating point than
the other corpus's outlier. A gain worth building for has to show up at most
operating points, not at one.

Usage:
  python3 tools/admission_gate.py --trace T [--experts-per-layer 64]
  python3 tools/admission_gate.py --selftest
"""
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GAIN_GATE = 0.10          # bytes/token reduction that would justify building it
CACHE_SIZES = (4, 8, 16, 32, 64)
SESSION_COUNTS = (2, 4, 8, 16)


def sessions_from_trace(path, n_sessions, steps):
    """Carve one trace into N independent session timelines.

    A ROUTE_TRACE is one session's history, so N concurrent sessions are
    simulated by taking N disjoint contiguous windows of it. Contiguous, not
    sampled: the predictor under test is "this session's PREVIOUS token", which
    only means anything if consecutive steps really are consecutive.
    """
    per = defaultdict(dict)
    with open(path) as fh:
        for line in fh:
            p = line.split()
            if len(p) < 4:
                continue
            try:
                call, layer = int(p[0]), int(p[2])
            except ValueError:
                continue
            s = per[layer].setdefault(call, set())
            for tok in p[3:]:
                try:
                    s.add(int(tok.split(":", 1)[0]))
                except ValueError:
                    pass
    layers = sorted(per)
    if not layers:
        return [], []
    calls = sorted(per[layers[0]])
    need = n_sessions * steps
    if len(calls) < need:
        return [], layers
    out = []
    for i in range(n_sessions):
        window = calls[i * steps:(i + 1) * steps]
        out.append([{l: per[l].get(c, set()) for l in layers} for c in window])
    return out, layers


def union_fits(sessions, layers, cache_sizes, s_counts):
    """CHECK 2: how often does the routed union exceed the tier?"""
    rows = []
    for S in s_counts:
        if len(sessions) < S:
            continue
        for C in cache_sizes:
            over = total = 0
            steps = min(len(x) for x in sessions)
            for t in range(steps):
                for l in layers:
                    u = set()
                    for i in range(S):
                        u |= sessions[i][t][l]
                    total += 1
                    if len(u) > C:
                        over += 1
            rows.append((S, C, over / total if total else 0.0))
    return rows


def simulate(sessions, layers, S, C, expert_bytes, residency_aware, seed=0):
    """Bytes fetched and tokens emitted, admitting S of the waiting sessions.

    The tier is per-layer LRU of capacity C. A session's PREDICTED routing is its
    previous step's routed set -- the only thing available before the forward.
    Residency-aware admission ranks sessions by how much of that prediction is
    already resident; arbitrary admission takes the first S.
    """
    rng = random.Random(seed)
    resident = {l: [] for l in layers}          # LRU, most-recent last
    bytes_fetched = 0
    tokens = 0
    steps = min(len(x) for x in sessions)
    N = len(sessions)
    for t in range(steps):
        if residency_aware and t > 0:
            def score(i):
                res = 0
                for l in layers:
                    pred = sessions[i][t - 1][l]
                    res += len(pred & set(resident[l]))
                return res
            order = sorted(range(N), key=lambda i: (-score(i), i))
        else:
            order = list(range(N))
            if not residency_aware:
                rng.shuffle(order)              # arbitrary, not adversarial
        admitted = order[:S]
        tokens += len(admitted)
        for l in layers:
            need = set()
            for i in admitted:
                need |= sessions[i][t][l]
            have = set(resident[l])
            miss = need - have
            bytes_fetched += len(miss) * expert_bytes
            for e in sorted(miss):
                resident[l].append(e)
            for e in need:                      # touch: LRU recency
                if e in resident[l]:
                    resident[l].remove(e)
                    resident[l].append(e)
            while len(resident[l]) > C:
                resident[l].pop(0)
    return bytes_fetched, tokens


def selftest():
    ok = True
    layers = [0]
    # Two sessions, disjoint experts, cache too small for both: admitting the one
    # whose experts are resident must fetch strictly less than admitting either.
    a = [{0: {0, 1}} for _ in range(6)]
    b = [{0: {8, 9}} for _ in range(6)]
    fb_aware, tok_a = simulate([a, b], layers, 1, 2, 1000, True)
    fb_arb, tok_b = simulate([a, b], layers, 1, 2, 1000, False)
    if tok_a != tok_b:
        print("selftest: token counts must match between policies"); ok = False
    if not fb_aware <= fb_arb:
        print(f"selftest: residency-aware fetched more ({fb_aware} > {fb_arb})"); ok = False
    # A cache that holds everything makes the policies identical -- CHECK 2's point
    big_aware, _ = simulate([a, b], layers, 2, 64, 1000, True)
    big_arb, _ = simulate([a, b], layers, 2, 64, 1000, False)
    if big_aware != big_arb:
        print(f"selftest: with a big enough cache the policies must agree "
              f"({big_aware} vs {big_arb})"); ok = False
    # union_fits must report 0 when the cache is larger than any union
    rows = union_fits([a, b], layers, (64,), (2,))
    if rows and rows[0][2] != 0.0:
        print(f"selftest: union_fits reported {rows[0][2]} with a huge cache"); ok = False
    print("selftest: ok" if ok else "selftest: FAILED")
    return 0 if ok else 1


def main():
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        raise SystemExit(selftest())
    trace = None
    n_experts = 64
    expert_bytes = 6_291_456
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
        if a.startswith("--trace"):
            trace = value("--trace")
        elif a.startswith("--experts-per-layer"):
            n_experts = int(value("--experts-per-layer"))
        elif a.startswith("--expert-bytes"):
            expert_bytes = int(value("--expert-bytes"))
        else:
            raise SystemExit(f"unknown argument: {a}")
    if not trace:
        raise SystemExit(__doc__)

    max_s = max(SESSION_COUNTS)
    sessions, layers = sessions_from_trace(trace, max_s, steps=24)
    if not sessions:
        raise SystemExit(f"{trace}: too short to carve {max_s} sessions of 24 steps")
    print(f"{os.path.basename(trace)}: {len(sessions)} simulated sessions x "
          f"{len(sessions[0])} steps, {len(layers)} layers, {n_experts} experts/layer\n")

    print("CHECK 2 -- does the routed union exceed the tier? (0% kills the item)")
    print(f"  {'S':>3}" + "".join(f"{'C='+str(C):>9}" for C in CACHE_SIZES))
    rows = union_fits(sessions, layers, CACHE_SIZES, SESSION_COUNTS)
    by_s = defaultdict(dict)
    for S, C, frac in rows:
        by_s[S][C] = frac
    for S in sorted(by_s):
        print(f"  {S:>3}" + "".join(f"{by_s[S][C]*100:>8.0f}%" for C in CACHE_SIZES))
    print()

    print(f"CHECK 3 -- residency-aware vs arbitrary admission (gate: >= {GAIN_GATE:.0%})")
    print(f"  {'S':>3} {'C':>4} {'arbitrary':>12} {'aware':>12} {'gain':>8}")
    gains = []
    for S in SESSION_COUNTS:
        if len(sessions) <= S:
            continue                      # no choice to make: N must exceed S
        for C in CACHE_SIZES:
            fa, ta = simulate(sessions, layers, S, C, expert_bytes, False)
            fb, tb = simulate(sessions, layers, S, C, expert_bytes, True)
            if not ta or not tb:
                continue
            ba, bb = fa / ta, fb / tb
            gain = 1 - bb / ba if ba else 0.0
            gains.append(((S, C), gain))
            print(f"  {S:>3} {C:>4} {ba/1e6:>10.1f}MB {bb/1e6:>10.1f}MB {gain:>7.1%}")
    print()
    vals = sorted(g for _, g in gains)
    med = vals[len(vals) // 2] if vals else 0.0
    wins = sum(1 for g in vals if g >= GAIN_GATE)
    best_at, best = max(gains, key=lambda kv: kv[1]) if gains else ((0, 0), 0.0)
    print(f"  median gain {med:+.1%} over {len(vals)} operating points; "
          f"{wins} of {len(vals)} reach the {GAIN_GATE:.0%} gate")
    print(f"  best {best:+.1%} at S={best_at[0]} C={best_at[1]}, "
          f"worst {vals[0]:+.1%}" if vals else "")
    # The verdict is the MEDIAN, not the maximum. A max over a grid of noisy
    # measurements clears any threshold by chance -- the first version of this
    # gate said "best gain >= 10%" and duly reported PASS on one corpus from a
    # single outlier surrounded by negatives, at a different (S,C) than the other
    # corpus's outlier. That is the multiple-comparisons trap, not a finding.
    print(f"  -> " + ("PASS, worth building" if med >= GAIN_GATE
                      else "FAILS the gate: no consistent gain from admission order"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
