#!/usr/bin/env python3
"""#37 gate: does the per-step routed-expert union grow sublinearly in S?

Consumes ROUTE_TRACE files (`call row layer expert:gate ...`), one per
independent session. Models S sessions decoding concurrently: at wall-clock
step t, session i is at its own decode token t. The union of experts needed
across those S sessions, per layer, is what one expert-major step must fetch.

Reports the SHARING RATIO = |union| / (S * mean per-session count).
  1.0  -> no sharing at all; batching buys nothing on bytes (#37 refuted)
  <1.0 -> every overlap is a fetch paid once instead of S times

This is the upper bound on the bytes/token win, independent of cache policy:
it counts distinct experts, assuming a fetch is shared perfectly within a step.
A cache can only do worse than this, never better.
"""
import sys, statistics
from collections import defaultdict
from pathlib import Path

def parse(path):
    """-> list of decode steps; each step is {layer: frozenset(experts)}.

    One trace line is one MoE call = ONE LAYER of one forward pass, with `row`
    indexing the tokens in that call's batch. A decode STEP is a whole forward
    pass, so calls must be grouped across layers: the k-th time layer L appears
    belongs to forward k. (Grouping by `call` instead gives one layer per step,
    which silently makes every union look 43x too small.)

    Prefill forwards carry many rows per call and are skipped -- batching
    prefill is a separate question."""
    seen = defaultdict(int)                    # layer -> how many forwards so far
    forwards = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
    for line in Path(path).read_text(errors="replace").splitlines():
        f = line.split()
        if len(f) < 4: continue
        try: call, row, layer = int(f[0]), int(f[1]), int(f[2])
        except ValueError: continue
        if row == 0:
            k = seen[layer]; seen[layer] += 1
        else:
            k = seen[layer] - 1                # same call, later row
        for tok in f[3:]:
            eid = tok.split(":")[0]
            if eid.lstrip("-").isdigit():
                forwards[k][layer][row].add(int(eid))
    steps = []
    for k in sorted(forwards):
        layers = forwards[k]
        if max(len(rows) for rows in layers.values()) != 1:
            continue                           # prefill batch, not a decode step
        steps.append({L: frozenset(next(iter(rows.values()))) for L, rows in layers.items()})
    return steps

def main(paths):
    sessions = [parse(p) for p in paths]
    sessions = [s for s in sessions if s]
    print(f"sessions: {len(sessions)}   decode steps each: {[len(s) for s in sessions]}")
    if not sessions: return 1
    T = min(len(s) for s in sessions)
    layers = sorted({L for s in sessions for st in s for L in st})
    print(f"steps used: {T}   layers: {len(layers)}\n")
    print(f"{'S':>3} {'union/step':>11} {'no-share':>10} {'ratio':>7} {'bytes/token':>13}")
    print("-"*49)
    base = None
    for S in range(1, len(sessions)+1):
        u_tot = n_tot = 0
        for t in range(T):
            for L in layers:
                per = [sessions[i][t].get(L, frozenset()) for i in range(S)]
                u_tot += len(set().union(*per)) if per else 0
                n_tot += sum(len(p) for p in per)
        union_per_step = u_tot / T
        noshare_per_step = n_tot / T
        ratio = u_tot / n_tot if n_tot else 0
        # bytes/token: distinct fetches per step, divided by S tokens produced
        per_tok = union_per_step / S
        if base is None: base = per_tok
        print(f"{S:>3} {union_per_step:>11.1f} {noshare_per_step:>10.1f} {ratio:>7.3f} "
              f"{per_tok:>9.1f} ({per_tok/base:>5.2f}x)")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
