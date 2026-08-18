#!/usr/bin/env python3
"""Measure token-level routing predictability from ROUTE_TRACE dumps (#32).

The engine's existing predictors all condition on the CURRENT token (PILOT
runs the next layer's router on this token's hidden state; COUPLE maps this
token's layer-L set to its layer-L+dL candidates). The open question in #32
is whether token t's routing profile predicts token t+1's — the temporal
axis none of them use. This tool answers it OFFLINE, from traces the engines
already emit, before anyone writes engine code: the same
measure-first discipline that justified COUPLE (route_pairs.py) originally.

Input: ROUTE_TRACE files ("<call> <row> <layer> <id>:<gate> ...", one line
per (moe call, batch row); route_trace.h documents the format). Forwards are
recovered with route_pairs.py's heuristic (layer number dropping = new
forward); each forward is treated as one sequence step (during decode, one
forward == one token; prefill forwards are kept as steps too, using row 0).

Predictors evaluated per token transition t -> t+1, per sparse layer, all
budgeted to the same top-K (K = the layer's observed set size, times
--budget-mult):
  identity     predict t+1's layer-L set = t's layer-L set
               (the LOOKA kind-0 baseline, cost-free in the engine)
  heat         top-K most frequent experts for layer L so far
               (the .coli_usage / marginal baseline)
  conditional  rank layer-L candidates f by sum of counts[(L',e) -> (L,f)]
               over ALL (L',e) active at token t — the full-profile
               cross-token table #32 proposes, learned online (counts update
               AFTER each transition is scored, so every score is causal)

Metric: recall@K = |predicted top-K ∩ true set| / |true set|, averaged over
transitions. The gap between `conditional` (or `identity`) and `heat` is the
achievable prefetch-hit improvement a temporal predictor could buy; if the
gap is ~0 on real traces, #32 item 1 dies here, cheaply.

Usage:
  python3 tools/route_temporal.py trace1.txt [trace2.txt ...]
  python3 tools/route_temporal.py --selftest
"""
import sys
import random
from collections import defaultdict


def parse_traces(paths):
    """-> list of sequences; each sequence is a list of {layer: [ids]} steps."""
    sequences = []
    for path in paths:
        steps, cur, prev_layer = [], {}, -1
        with open(path) as f:
            lines = f.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) < 4:
                continue
            row, layer = int(parts[1]), int(parts[2])
            if row != 0:
                continue                       # one representative row per forward
            if layer < prev_layer and cur:     # layer dropped: new forward
                steps.append(cur)
                cur = {}
            prev_layer = layer
            cur[layer] = [int(t.split(":")[0]) for t in parts[3:]]
        if cur:
            steps.append(cur)
        if len(steps) >= 2:
            sequences.append(steps)
    return sequences


def evaluate(sequences, budget_mult=1):
    heat = defaultdict(lambda: defaultdict(int))     # L -> {e: count}
    cond = defaultdict(lambda: defaultdict(int))     # (L', e, L) -> {f: count}
    hits = defaultdict(float)
    total = 0

    def topk(scored, k):
        return set(e for e, _ in sorted(scored.items(),
                                        key=lambda kv: (-kv[1], kv[0]))[:k])

    for steps in sequences:
        for t in range(len(steps) - 1):
            prof, nxt = steps[t], steps[t + 1]
            for L, true_ids in sorted(nxt.items()):
                true = set(true_ids)
                if not true:
                    continue
                k = max(1, len(true) * budget_mult)
                pred_id = set(prof.get(L, [])) if L in prof else set()
                pred_heat = topk(heat[L], k) if heat[L] else set()
                scores = defaultdict(int)
                for Lp, es in prof.items():
                    for e in es:
                        for f, c in cond[(Lp, e, L)].items():
                            scores[f] += c
                pred_cond = topk(scores, k) if scores else pred_id
                total += 1
                hits["identity"] += len(pred_id & true) / len(true)
                hits["heat"] += len(pred_heat & true) / len(true)
                hits["conditional"] += len(pred_cond & true) / len(true)
            # counts update AFTER scoring: every prediction above was causal
            for L, es in nxt.items():
                for e in es:
                    heat[L][e] += 1
            for Lp, es in prof.items():
                for e in es:
                    for L, fs in nxt.items():
                        d = cond[(Lp, e, L)]
                        for f in fs:
                            d[f] += 1
    if not total:
        raise SystemExit("no token transitions found in the trace(s)")
    return {name: s / total for name, s in hits.items()}, total


# ---------------- selftest: synthetic traces with known structure ----------------

def synth(rng, tokens, layers, nexpert, k, persist):
    """One sequence. Each token keeps each routed expert with prob `persist`,
    else resamples uniformly — persist=0 is an iid trace (no temporal signal),
    persist high is a strongly predictable one."""
    steps = []
    prev = {L: rng.sample(range(nexpert), k) for L in range(layers)}
    steps.append({L: list(v) for L, v in prev.items()})
    for _ in range(tokens - 1):
        cur = {}
        for L in range(layers):
            ids = []
            for e in prev[L]:
                if rng.random() < persist and e not in ids:
                    ids.append(e)
            while len(ids) < k:
                c = rng.randrange(nexpert)
                if c not in ids:
                    ids.append(c)
            cur[L] = ids
        steps.append(cur)
        prev = cur
    return steps


def selftest():
    rng = random.Random(0)
    nexpert, k = 64, 8
    uniform_recall = k / nexpert

    persistent = [synth(rng, 400, 4, nexpert, k, persist=0.7) for _ in range(3)]
    r_p, n_p = evaluate(persistent)
    iid = [synth(rng, 400, 4, nexpert, k, persist=0.0) for _ in range(3)]
    r_i, n_i = evaluate(iid)

    print(f"selftest persistent ({n_p} transitions): " +
          " ".join(f"{k2}={v:.3f}" for k2, v in sorted(r_p.items())))
    print(f"selftest iid        ({n_i} transitions): " +
          " ".join(f"{k2}={v:.3f}" for k2, v in sorted(r_i.items())))

    ok = True
    # persistent trace: temporal predictors must clearly beat the marginal
    if not (r_p["identity"] > r_p["heat"] + 0.15):
        print("FAIL: identity did not beat heat on the persistent trace"); ok = False
    if not (r_p["conditional"] > r_p["heat"] + 0.15):
        print("FAIL: conditional did not beat heat on the persistent trace"); ok = False
    if not (r_p["identity"] > 0.6):
        print("FAIL: identity recall implausibly low for persist=0.7"); ok = False
    # iid trace: nothing should beat uniform chance by much
    for name, v in r_i.items():
        if v > uniform_recall + 0.1:
            print(f"FAIL: {name}={v:.3f} beats chance on an iid trace"); ok = False
    print("selftest:", "ok" if ok else "FAILED")
    return 0 if ok else 1


def main():
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        raise SystemExit(selftest())
    budget_mult = 1
    if args and args[0].startswith("--budget-mult="):
        budget_mult = int(args.pop(0).split("=")[1])
    if not args:
        raise SystemExit(__doc__)
    sequences = parse_traces(args)
    recalls, total = evaluate(sequences, budget_mult)
    print(f"{len(sequences)} sequence(s), {total} (transition, layer) scores, "
          f"budget = true-set size x{budget_mult}")
    for name in ("heat", "identity", "conditional"):
        print(f"  recall@K {name:12s} {recalls[name]:.4f}")
    gap = max(recalls["identity"], recalls["conditional"]) - recalls["heat"]
    print(f"  temporal gain over marginal: {gap:+.4f}"
          f"  ({'worth an engine slice' if gap > 0.05 else 'no temporal signal worth chasing'})")


if __name__ == "__main__":
    main()
