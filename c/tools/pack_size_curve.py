#!/usr/bin/env python3
"""Is there a pack size that fixes CA's read cost without wrecking delta? (lex-moe#50)

`castore-io-cost-2026-08-20.md` measured per-tensor content addressing at 3.6x
the reads of a co-activation-clustered container. The proposed fix is to
content-address PACKS of co-activating experts rather than individual tensors.
That trades two things against each other, and lex-moe#50 states the gate up
front so the answer can be a clean negative:

    a pack size exists that gets within ~1.2x of the clustered layout's read
    count while keeping a realistic fine-tune's delta under ~2x the per-tensor
    baseline -- or no such point exists, and CA is for distribution rather than
    streaming.

## Three quantities, not two

Packing has a consequence the proposal did not spell out: **a pack is one
content-addressed object, so verifying it means reading all of it.** A reader
that wants the address to still be the checksum must read whole packs, which
inflates BYTES even as it reduces READS. A reader that preads a sub-range gets
the bytes back but can no longer check that range against the pack's hash
without reading the rest.

So this reports:

  reads(P)        sub-range reads -- what the engine issues if it preads inside
                  a pack. This is the number that competes with the clustered
                  layout.
  byte_amp(P)     whole-pack reads / bytes actually needed. What integrity costs
                  if every read must be verifiable against the pack hash.
  delta(P)        bytes a fine-tune must re-store, over the per-tensor baseline.
                  A changed expert dirties its whole pack.

A pack size is only interesting if all three are acceptable at once.

Usage:
  python3 tools/pack_size_curve.py --trace T --perm perm.json --container DIR
  python3 tools/pack_size_curve.py --selftest
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expert_relayout as er

PACK_SIZES = (1, 2, 4, 8, 16, 32, 64)
FINE_TUNE_FRACTIONS = (0.01, 0.03, 0.10)
READ_GATE = 1.2          # within 1.2x of the clustered layout
DELTA_GATE = 2.0         # under 2x the per-tensor delta baseline


def pack_of(perm, layer, expert, p):
    """Which pack an expert lands in, given the co-activation permutation.

    Slot order comes from expert_layout.py: experts that fire together are
    adjacent, so consecutive slots are exactly what a pack should hold.
    """
    order = perm.get("layers", {}).get(str(layer))
    if not order:
        return expert // p
    try:
        return order.index(expert) // p
    except ValueError:
        return (len(order) + expert) // p


def build_slots(perm, by_layer):
    """{layer: {expert: slot}} from the permutation, stable for absent experts."""
    slots = {}
    for layer, experts in by_layer.items():
        order = [e for e in perm.get("layers", {}).get(str(layer), []) if e in experts]
        tail = sorted(e for e in experts if e not in order)
        slots[layer] = {e: i for i, e in enumerate(order + tail)}
    return slots


def measure(per, sizes, slots, expert_bytes):
    """reads / byte-amplification per pack size, from the trace's routed sets."""
    out = {}
    for p in sizes:
        reads = need_bytes = whole_bytes = 0
        for (_call, layer), experts in per.items():
            if layer not in slots:
                continue
            packs = {}
            for e in experts:
                s = slots[layer].get(e)
                if s is None:
                    continue
                packs.setdefault(s // p, []).append(s)
            for members in packs.values():
                members.sort()
                # sub-range reads: consecutive slots inside a pack are adjacent
                runs = 1
                for a, b in zip(members, members[1:]):
                    if b != a + 1:
                        runs += 1
                reads += runs
                need_bytes += len(members) * expert_bytes
                whole_bytes += p * expert_bytes       # verifying costs the whole pack
        out[p] = {"reads": reads,
                  "byte_amp": (whole_bytes / need_bytes) if need_bytes else 0.0}
    return out


def delta_cost(sizes, slots, fractions, seed=0):
    """Bytes a fine-tune re-stores, over the per-tensor baseline.

    A changed expert dirties its whole pack, so the cost is
    (dirty packs x pack size) / (changed experts). Averaged over layers, with
    the changed set drawn uniformly -- a real fine-tune may concentrate, which
    would only help, so this is the pessimistic reading.
    """
    rng = random.Random(seed)
    out = {}
    for p in sizes:
        out[p] = {}
        for f in fractions:
            dirty = changed = 0
            for layer, m in slots.items():
                n = len(m)
                k = max(1, int(round(n * f)))
                picked = rng.sample(sorted(m.values()), k)
                dirty += len({s // p for s in picked}) * p
                changed += k
            out[p][f] = dirty / changed if changed else 0.0
    return out


def report(per, slots, expert_bytes, clustered_reads, tensor_reads):
    m = measure(per, PACK_SIZES, slots, expert_bytes)
    d = delta_cost(PACK_SIZES, slots, FINE_TUNE_FRACTIONS)
    print(f"  clustered layout: {clustered_reads} reads   per-tensor CA: {tensor_reads} reads")
    print(f"  {'pack':>5} {'reads':>9} {'vs clustered':>13} {'byte amp':>9}   "
          + "  ".join(f"delta@{int(f*100)}%" for f in FINE_TUNE_FRACTIONS))
    winners = []
    for p in PACK_SIZES:
        r = m[p]["reads"]
        ratio = r / clustered_reads if clustered_reads else 0.0
        deltas = [d[p][f] for f in FINE_TUNE_FRACTIONS]
        print(f"  {p:>5} {r:>9} {ratio:>12.2f}x {m[p]['byte_amp']:>8.2f}x   "
              + "  ".join(f"{x:>8.2f}x" for x in deltas))
        if ratio <= READ_GATE and max(deltas) <= DELTA_GATE:
            winners.append(p)
    print()
    if winners:
        print(f"  -> pack sizes passing BOTH gates (reads <= {READ_GATE}x, delta <= {DELTA_GATE}x): {winners}")
    else:
        print(f"  -> NO pack size satisfies both gates on this data")
    return winners, m, d


# ---------------------------------------------------------------- selftest
def selftest():
    ok = True
    # Perfectly clustered routing: every forward routes to one contiguous block,
    # so a pack the size of that block should read it in one, and a bigger pack
    # should not help further.
    per = {(c, 0): {0, 1, 2, 3} for c in range(50)}
    slots = {0: {e: e for e in range(64)}}
    m = measure(per, (1, 4, 8), slots, 1000)
    if not (m[1]["reads"] == 200 and m[4]["reads"] == 50 and m[8]["reads"] == 50):
        print(f"selftest: contiguous reads wrong: {[(p, m[p]['reads']) for p in (1,4,8)]}")
        ok = False
    # byte amplification is exactly the pack size when one expert of P is needed
    per1 = {(0, 0): {0}}
    m1 = measure(per1, (1, 8), slots, 1000)
    if abs(m1[8]["byte_amp"] - 8.0) > 1e-9 or abs(m1[1]["byte_amp"] - 1.0) > 1e-9:
        print(f"selftest: byte amplification wrong: {m1}")
        ok = False
    # delta at pack size 1 is always exactly 1x -- the per-tensor baseline
    d = delta_cost((1, 8), slots, (0.03,))
    if abs(d[1][0.03] - 1.0) > 1e-9:
        print(f"selftest: delta at P=1 should be 1.0, got {d[1][0.03]}")
        ok = False
    if not d[8][0.03] >= 1.0:
        print("selftest: delta must not improve with bigger packs")
        ok = False
    print("selftest: ok" if ok else "selftest: FAILED")
    return 0 if ok else 1


def main():
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        raise SystemExit(selftest())
    trace = perm_path = container = None
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
        elif a.startswith("--perm"):
            perm_path = value("--perm")
        elif a.startswith("--container"):
            container = value("--container")
        else:
            raise SystemExit(f"unknown argument: {a}")
    if not (trace and perm_path and container):
        raise SystemExit(__doc__)

    with open(perm_path) as fh:
        perm = json.load(fh)
    per = er.routed_sets(trace)
    rng = er.byte_ranges(container)

    by_layer = {}
    for (layer, expert) in rng:
        by_layer.setdefault(layer, set()).add(expert)
    slots = build_slots(perm, by_layer)

    sizes = [sum(b - a for _s, a, b in segs) for segs in rng.values()]
    expert_bytes = sum(sizes) / len(sizes)

    tensor_reads, _ = er.count_reads(per, er.byte_ranges_castore(container))
    print(f"{os.path.basename(trace)}: {len(per)} moe() calls, "
          f"{len(by_layer)} layers, mean expert {expert_bytes/1e6:.2f} MB\n")
    # the clustered baseline is what a relaid container achieves
    clustered_reads = measure(per, (10**9,), slots, expert_bytes)[10**9]["reads"]
    report(per, slots, expert_bytes, clustered_reads, tensor_reads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
