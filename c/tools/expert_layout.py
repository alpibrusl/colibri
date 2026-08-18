#!/usr/bin/env python3
"""Measure whether a co-activation-clustered expert layout would help (#36).

Each expert already loads as ONE coalesced ~19 MB pread, so within an expert
the I/O is sequential and near-optimal. Across the K experts a layer routes to
it is K reads at scattered offsets — because the container stores experts in
(layer, expert_id) order and expert *id* has nothing to do with which experts
fire together.

We already know which fire together: route_pairs.py builds that table for
COUPLE, where conditioning on it measured +3.6..+9.4pp prefetch recall over
marginal heat. This tool asks the other question that statistic answers for
free: if experts that co-activate were stored ADJACENT, how many fewer reads
would a forward issue?

Nothing here changes routing, quantization or output. Same bytes, different
order in the file. The permutation is emitted so a later engine slice can map
expert_id -> slot on the way in; this tool exists to decide whether that slice
is worth writing.

## The model

For I/O, the unit that matters is not a position's top-K, it is the UNION over
positions of a (forward, layer)'s routed experts: that is the set the engine
actually fetches before the layer can run. Given a permutation of expert ids to
slots, that set occupies some number of maximal CONTIGUOUS runs of slots, and
each run is one read. So:

    reads(forward, layer) = number of maximal runs of consecutive slots

which is |S| when nothing is adjacent (today's worst case) and 1 when the whole
set is contiguous. Everything below reports that number, before and after.

## Reading the verdict

Three layouts are always scored together, and the third is the one that keeps
this honest:

    identity   today's container order
    clustered  the layout this tool computes
    random     a shuffled control

A clustered layout that beats identity but not the random control is measuring
nothing — it means the metric moves under any permutation. The gate is the gain
over BOTH.

And the permutation is FITTED ON ONE SPLIT AND SCORED ON ANOTHER, for the same
reason placement_crossval.py splits: a seriation heuristic will happily find
structure in sampling noise. Scored in-sample, this tool reported a ~9% read
reduction on routing generated to have no structure at all — a number that
would have sent someone to write a container rewrite for nothing. Held out, a
layout only scores if the co-activation it was fitted to recurs.

Usage:
  python3 tools/expert_layout.py trace1.txt [trace2.txt ...]
  python3 tools/expert_layout.py --out=layout.json --perm=perm.json trace.txt
  python3 tools/expert_layout.py --selftest
"""
import json
import os
import random
import sys
from collections import defaultdict

try:                          # imported as tools.expert_layout
    from . import ledger
except ImportError:           # run as a script, or loaded by file path
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ledger

# A gain smaller than this is noise dressed as a finding: the run-count metric
# moves a little under any permutation, which is what the random control below
# measures. Chosen to match the "worth an engine slice" bar route_temporal.py
# sets for its own decision (5 points).
GAIN_THRESHOLD = 0.05


# ---------------------------------------------------------------- parsing
def parse_forwards(paths):
    """-> list of {layer: set(expert_ids)}, one entry per forward.

    ROUTE_TRACE is "<call> <row> <layer> <id>:<gate> ...", layer-major within a
    forward (route_trace.h). Forward boundaries are recovered the way
    route_pairs.py and route_temporal.py recover them — the layer number
    dropping — so all three tools segment a trace identically.

    Rows are UNIONED rather than sampled: during prefill many positions share
    one forward, and the engine fetches the union before the layer can run.
    That is the set whose layout we care about.
    """
    forwards = []
    for path in paths:
        cur, prev_layer = defaultdict(set), -1
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    layer = int(parts[2])
                except ValueError:
                    continue
                if layer < prev_layer and cur:
                    forwards.append({k: set(v) for k, v in cur.items()})
                    cur = defaultdict(set)
                prev_layer = layer
                for tok in parts[3:]:
                    eid = tok.split(":")[0]
                    try:
                        cur[layer].add(int(eid))
                    except ValueError:
                        pass
        if cur:
            forwards.append({k: set(v) for k, v in cur.items()})
    return forwards


# ------------------------------------------------------- co-activation graph
def coactivation(forwards):
    """-> {layer: {(e, f): count}} for e < f, counted once per forward.

    Once per forward, not once per position: two experts that co-occur in one
    fetch are adjacent-worthy exactly once, however many rows wanted them.
    """
    graph = defaultdict(lambda: defaultdict(int))
    for fwd in forwards:
        for layer, ids in fwd.items():
            ordered = sorted(ids)
            for i, e in enumerate(ordered):
                for f in ordered[i + 1:]:
                    graph[layer][(e, f)] += 1
    return graph


def experts_of(forwards):
    """-> {layer: sorted list of expert ids seen}."""
    seen = defaultdict(set)
    for fwd in forwards:
        for layer, ids in fwd.items():
            seen[layer] |= ids
    return {layer: sorted(ids) for layer, ids in seen.items()}


# ------------------------------------------------------------- the ordering
def greedy_chain(ids, weights):
    """Seriation by greedy nearest-neighbour on the co-activation graph.

    Start at the heaviest edge, then repeatedly append whichever unplaced
    expert is most coupled to the current tail. O(E^2) and stdlib-only, which
    matters because this has to run wherever a checkpoint lives.

    It is a heuristic for an NP-hard linear arrangement, and deliberately not
    dressed up as more: the metric below scores whatever order it produces, so
    a better ordering can be swapped in without changing how it is judged.
    """
    if len(ids) < 2:
        return list(ids)

    def w(a, b):
        return weights.get((a, b) if a < b else (b, a), 0)

    remaining = set(ids)
    # Heaviest edge as the seed; ties broken by id so the output is
    # deterministic (a layout that changes between runs is not a layout).
    best_pair, best_w = None, -1
    for (a, b), c in sorted(weights.items()):
        if a in remaining and b in remaining and c > best_w:
            best_pair, best_w = (a, b), c
    if best_pair is None:                      # no edges at all
        return sorted(remaining)
    chain = [best_pair[0], best_pair[1]]
    remaining -= set(chain)
    while remaining:
        tail = chain[-1]
        nxt = max(sorted(remaining), key=lambda x: (w(tail, x), -x))
        chain.append(nxt)
        remaining.discard(nxt)
    return chain


def slots_of(order):
    """A placement list -> {expert_id: slot index}."""
    return {eid: i for i, eid in enumerate(order)}


# ----------------------------------------------------------------- scoring
def runs_of(slots, ids):
    """Maximal runs of consecutive slots — i.e. reads — for one fetch set."""
    if not ids:
        return 0
    placed = sorted(slots[i] for i in ids if i in slots)
    if not placed:
        return 0
    runs = 1
    for a, b in zip(placed, placed[1:]):
        if b != a + 1:
            runs += 1
    return runs


def score(forwards, placement):
    """-> (reads, fetched, spans) totalled over every (forward, layer)."""
    reads = fetched = 0
    for fwd in forwards:
        for layer, ids in fwd.items():
            slots = placement.get(layer)
            if not slots:
                continue
            reads += runs_of(slots, ids)
            fetched += len(ids)
    return reads, fetched


def identity_placement(by_layer):
    return {layer: slots_of(ids) for layer, ids in by_layer.items()}


def random_placement(by_layer, seed=0):
    rng = random.Random(seed)
    out = {}
    for layer, ids in by_layer.items():
        shuffled = list(ids)
        rng.shuffle(shuffled)
        out[layer] = slots_of(shuffled)
    return out


def clustered_placement(by_layer, graph):
    return {layer: slots_of(greedy_chain(ids, graph.get(layer, {})))
            for layer, ids in by_layer.items()}


def evaluate(forwards, holdout=0.5):
    """Fit the layout on the first split, score every layout on the second.

    `identity` and `random` do not look at the fit split at all, so scoring all
    three on the same held-out forwards is the apples-to-apples comparison. The
    fit/score boundary is what makes the clustered number mean "this layout
    generalizes" rather than "this heuristic memorized these forwards".
    """
    cut = max(1, int(len(forwards) * (1.0 - holdout)))
    fit, held = forwards[:cut], forwards[cut:] or forwards[:cut]

    # The slot space is every expert the trace ever mentions, so a layout is
    # defined for experts that appear only in the held-out half too.
    by_layer = experts_of(forwards)
    graph = coactivation(fit)
    placements = {
        "identity": identity_placement(by_layer),
        "clustered": clustered_placement(by_layer, graph),
        "random": random_placement(by_layer),
    }
    out = {}
    for name, placement in placements.items():
        reads, fetched = score(held, placement)
        out[name] = {
            "reads": reads,
            "experts_fetched": fetched,
            "experts_per_read": (fetched / reads) if reads else 0.0,
        }
    return out, placements, by_layer


def gains(scored):
    """Reduction in reads vs identity, and vs the random control."""
    base = scored["identity"]["reads"]
    rnd = scored["random"]["reads"]
    clu = scored["clustered"]["reads"]
    return {
        "vs_identity": (base - clu) / base if base else 0.0,
        "vs_random": (rnd - clu) / rnd if rnd else 0.0,
    }


# ------------------------------------------------------------------ report
def report_of(paths, forwards, scored, g):
    """Ledger-shaped claims (tools/ledger.py): flat, integer, comparable."""
    worth = g["vs_identity"] > GAIN_THRESHOLD and g["vs_random"] > GAIN_THRESHOLD
    claims = {
        "forwards": len(forwards),
        "read_reduction_vs_identity_micro": ledger.micro(g["vs_identity"]),
        "read_reduction_vs_random_micro": ledger.micro(g["vs_random"]),
        "worth_engine_slice": ledger.bit(worth),
    }
    for name, s in sorted(scored.items()):
        claims[f"{name}.reads"] = s["reads"]
        claims[f"{name}.experts_per_read_micro"] = ledger.micro(s["experts_per_read"])
    verdict = ("clustered layout worth an engine slice" if worth
               else "no layout gain beyond the control — close #36 with this data")
    return ledger.claims_report("expert_layout.py", paths, verdict,
                                {"scored": scored, "gains": g,
                                 "threshold": GAIN_THRESHOLD}, claims)


def permutation_doc(placements, by_layer):
    """The layout itself: {layer: [expert_id in slot order]}.

    Emitted separately from the report because it is an ARTIFACT a container
    rewrite would consume, not a measurement. Slot order, not id order: slot i
    holds `order[i]`.
    """
    slots = placements["clustered"]
    return {"version": 1,
            "layers": {str(layer): sorted(by_layer[layer], key=lambda e: slots[layer][e])
                       for layer in sorted(by_layer)}}


# ---------------------------------------------------------------- selftest
def synth_clustered(n_forwards, n_layers, n_experts, group, k, rng):
    """Experts drawn from a few co-firing groups — the structure #36 bets on.

    The groups are drawn over a SHUFFLED id space on purpose. Co-firing groups
    that happened to be contiguous ranges of expert id would already be
    optimally laid out by the identity order, and the tool would look useless
    on exactly the data it is supposed to win on. Real expert ids carry no such
    favour, which is the whole premise of #36.
    """
    forwards = []
    ids = list(range(n_experts))
    rng.shuffle(ids)
    groups = [ids[g:g + group] for g in range(0, n_experts, group)]
    for _ in range(n_forwards):
        fwd = {}
        for layer in range(n_layers):
            pool = rng.choice(groups)
            picks = pool if len(pool) <= k else rng.sample(pool, k)
            fwd[layer] = set(picks)
        forwards.append(fwd)
    return forwards


def synth_uniform(n_forwards, n_layers, n_experts, k, rng):
    """Routing with no clustering at all: no permutation can help."""
    return [{layer: set(rng.sample(range(n_experts), k))
             for layer in range(n_layers)}
            for _ in range(n_forwards)]


def selftest():
    ok = True
    rng = random.Random(11)

    clustered = synth_clustered(400, 4, 64, group=8, k=4, rng=rng)
    scored_c, _, _ = evaluate(clustered)
    gc = gains(scored_c)
    print(f"selftest clustered: reads identity={scored_c['identity']['reads']} "
          f"random={scored_c['random']['reads']} clustered={scored_c['clustered']['reads']} "
          f"-> {gc['vs_identity']:+.3f} vs identity, {gc['vs_random']:+.3f} vs random")
    if gc["vs_identity"] <= GAIN_THRESHOLD:
        print("FAIL: no gain on a trace built from co-firing groups"); ok = False
    if gc["vs_random"] <= GAIN_THRESHOLD:
        print("FAIL: clustered layout did not beat the random control"); ok = False

    uniform = synth_uniform(400, 4, 64, k=4, rng=rng)
    scored_u, _, _ = evaluate(uniform)
    gu = gains(scored_u)
    print(f"selftest uniform:   reads identity={scored_u['identity']['reads']} "
          f"clustered={scored_u['clustered']['reads']} "
          f"-> {gu['vs_identity']:+.3f} vs identity")
    # The honesty check: on structureless routing this tool must report NO win.
    # A seriation heuristic that manufactures a gain here would manufacture one
    # on a real trace too, and #36 would be decided by an artifact.
    if gu["vs_identity"] > GAIN_THRESHOLD:
        print("FAIL: reported a layout gain on structureless routing"); ok = False

    # A fetch set that is already contiguous is exactly one read, and a
    # scattered one is |S| reads: the metric's two endpoints.
    slots = slots_of([0, 1, 2, 3, 4, 5])
    if runs_of(slots, {1, 2, 3}) != 1:
        print("FAIL: a contiguous set is not one read"); ok = False
    if runs_of(slots, {0, 2, 4}) != 3:
        print("FAIL: a scattered set is not |S| reads"); ok = False
    if runs_of(slots, set()) != 0:
        print("FAIL: an empty set is not zero reads"); ok = False

    print("selftest:", "ok" if ok else "FAILED")
    return 0 if ok else 1


def main():
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        raise SystemExit(selftest())
    out = perm_out = None
    traces = []
    for a in args:
        if a.startswith("--out="):
            out = a.split("=", 1)[1]
        elif a.startswith("--perm="):
            perm_out = a.split("=", 1)[1]
        else:
            traces.append(a)
    if not traces:
        raise SystemExit(__doc__)

    forwards = parse_forwards(traces)
    if not forwards:
        sys.exit("no forwards parsed — is this a ROUTE_TRACE dump? (route_trace.h)")
    scored, placements, by_layer = evaluate(forwards)
    g = gains(scored)

    print(f"{len(forwards)} forward(s), {len(by_layer)} sparse layer(s)")
    for name in ("identity", "clustered", "random"):
        s = scored[name]
        print(f"  {name:10s} reads {s['reads']:8d}   experts/read {s['experts_per_read']:.3f}")
    print(f"  read reduction: {g['vs_identity']:+.3f} vs identity, "
          f"{g['vs_random']:+.3f} vs the random control")
    print(f"  -> {'worth an engine slice' if g['vs_identity'] > GAIN_THRESHOLD and g['vs_random'] > GAIN_THRESHOLD else 'no layout gain beyond the control'}")

    if out:
        with open(out, "w") as f:
            json.dump(report_of(traces, forwards, scored, g), f, indent=1)
        print(f"  report -> {out}  (record it: python3 tools/ledger.py "
              f"--report={out} --series=expert-layout --attempt=1 --out=entry.json)")
    if perm_out:
        with open(perm_out, "w") as f:
            json.dump(permutation_doc(placements, by_layer), f, indent=1)
        print(f"  permutation -> {perm_out}")


if __name__ == "__main__":
    main()
