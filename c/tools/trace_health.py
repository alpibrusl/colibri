#!/usr/bin/env python3
"""Is this ROUTE_TRACE worth drawing a conclusion from? (#32, #36)

Every routing tool here takes a trace and returns a verdict. None of them ask
whether the trace means anything — and they should, because a trace from a
model that is not routing looks, to all of them, exactly like a beautifully
predictable one.

## The run that motivated this

An OLMoE-shaped checkpoint with RANDOM weights, through the real engine, real
ROUTE_TRACE, 200 forwards. `route_temporal.py` reported **+0.135 temporal gain
over marginal** and `expert_layout.py` reported a **21.6% read reduction** —
both comfortably over their "worth an engine slice" thresholds, both from a
model that had never been trained.

Neither number was a bug. With random weights the router's logits are a fixed
random function, so the top-K selection is decided by tiny persistent biases;
that persistence is real structure, and it is structure in the initialization
rather than anything a trained model would share. The tools were right that
the trace was predictable. They had no way to say that the predictability was
worthless.

## The check

Measured on that trace: the mean normalized entropy of the applied gates was
**0.9999** — indistinguishable from uniform — and the mean top gate was 0.1283
against 0.125 for a perfectly flat top-8. A router that is actually
discriminating puts mass somewhere; one that is not spreads it evenly and lets
rounding noise pick the winners.

So: normalized gate entropy, averaged over traced rows. 1.0 is a coin flip
among the routed experts, and a trace at that end cannot support a routing
claim of any kind.

## What the threshold is and is not

`UNIFORM_ENTROPY` is calibrated against the DEGENERATE end only, which is the
end this container could actually produce: 0.9999 measured on random weights.
The trained end is unmeasured here — nobody has run a real checkpoint through
this yet — so treat the number as a floor that catches "this model is not
routing", not as a quality bar that certifies a trace is good. A trace that
passes is not thereby endorsed; it is merely not obviously meaningless.
"""
import math
import os
import sys

# Above this, the applied gates are within rounding of uniform. See the note
# above: a floor, not a quality bar.
UNIFORM_ENTROPY = 0.99


def gate_stats(paths):
    """-> {rows, entropy, top_gate, spread} averaged over traced rows.

    Gates are renormalized per row before the entropy, so this measures the
    SHAPE of the distribution the layer applies and not whether the engine
    happened to normalize top-K (`norm_topk_prob`) before writing it out.
    """
    rows = 0
    ent = top = spread = 0.0
    for path in paths:
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 4:
                    continue
                gates = []
                for tok in parts[3:]:
                    bits = tok.split(":")
                    if len(bits) != 2:
                        continue
                    try:
                        gates.append(float(bits[1]))
                    except ValueError:
                        pass
                total = sum(gates)
                if len(gates) < 2 or total <= 0:
                    continue
                q = [g / total for g in gates]
                ent += -sum(x * math.log(x) for x in q if x > 0) / math.log(len(q))
                top += max(q)
                spread += (max(q) - min(q)) / max(q) if max(q) > 0 else 0.0
                rows += 1
    if not rows:
        return {"rows": 0, "entropy": 0.0, "top_gate": 0.0, "spread": 0.0}
    return {"rows": rows, "entropy": ent / rows,
            "top_gate": top / rows, "spread": spread / rows}


def is_degenerate(stats):
    """True when the gates say this model is not routing in any useful sense."""
    return stats["rows"] > 0 and stats["entropy"] >= UNIFORM_ENTROPY


def warning(stats):
    """The line a tool prints instead of a verdict, or None."""
    if not is_degenerate(stats):
        return None
    return (f"DEGENERATE TRACE: mean normalized gate entropy {stats['entropy']:.4f} "
            f">= {UNIFORM_ENTROPY} (mean top gate {stats['top_gate']:.4f}). The router is "
            f"not discriminating between experts — any predictability here is a "
            f"property of the weights' initialization, not of routing. No verdict "
            f"is issued; check the checkpoint and the conversion.")


def claims(stats):
    """Flat integer claims, for the ledger-shaped reports (tools/ledger.py)."""
    try:                          # imported as tools.trace_health
        from . import ledger
    except ImportError:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import ledger
    return {
        "gate.rows": stats["rows"],
        "gate.entropy_micro": ledger.micro(stats["entropy"]),
        "gate.top_micro": ledger.micro(stats["top_gate"]),
        "gate.spread_micro": ledger.micro(stats["spread"]),
        "gate.degenerate": ledger.bit(is_degenerate(stats)),
    }


def selftest():
    import tempfile
    ok = True

    def write(rows):
        fh = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
        for call, (layer, gates) in enumerate(rows):
            fh.write(f"{call} 0 {layer} " +
                     " ".join(f"{i}:{g:.4f}" for i, g in enumerate(gates)) + "\n")
        fh.close()
        return fh.name

    def check(name, got, want):
        nonlocal ok
        if got != want:
            print(f"FAIL {name}: got {got!r}, want {want!r}")
            ok = False

    # Uniform gates: the case measured on a random-weight checkpoint.
    flat = write([(0, [0.125] * 8)] * 20)
    s = gate_stats([flat])
    check("flat rows", s["rows"], 20)
    check("flat is degenerate", is_degenerate(s), True)
    if abs(s["entropy"] - 1.0) > 1e-9:
        print(f"FAIL flat entropy: {s['entropy']}"); ok = False
    if warning(s) is None:
        print("FAIL: no warning for a degenerate trace"); ok = False

    # A router that actually puts mass somewhere.
    peaked = write([(0, [0.60, 0.20, 0.10, 0.04, 0.03, 0.01, 0.01, 0.01])] * 20)
    s = gate_stats([peaked])
    check("peaked is not degenerate", is_degenerate(s), False)
    check("peaked has no warning", warning(s), None)
    if s["top_gate"] < 0.5:
        print(f"FAIL peaked top gate: {s['top_gate']}"); ok = False

    # The near-uniform end: 4% spread still reads as degenerate, which is the
    # actual measurement from the random-weight run (spread 0.041).
    nearly = write([(0, [0.1284, 0.1268, 0.1251, 0.1242,
                         0.1241, 0.1238, 0.1238, 0.1238])] * 20)
    s = gate_stats([nearly])
    check("near-uniform is degenerate", is_degenerate(s), True)

    check("empty trace has no rows", gate_stats([write([])])["rows"], 0)
    check("empty trace is not flagged", is_degenerate(gate_stats([write([])])), False)

    c = claims(gate_stats([flat]))
    check("claims are integers", all(isinstance(v, int) for v in c.values()), True)
    check("degenerate claim", c["gate.degenerate"], 1)

    for p in (flat, peaked, nearly):
        os.unlink(p)
    print("selftest:", "ok" if ok else "FAILED")
    return 0 if ok else 1


def main():
    args = sys.argv[1:]
    if not args or args[0] == "--selftest":
        raise SystemExit(selftest())
    stats = gate_stats(args)
    print(f"rows {stats['rows']}")
    print(f"  mean normalized gate entropy {stats['entropy']:.4f}  "
          f"(>= {UNIFORM_ENTROPY} is uniform)")
    print(f"  mean top gate                {stats['top_gate']:.4f}")
    print(f"  mean within-row spread       {stats['spread']:.4f}")
    w = warning(stats)
    print(f"  {w}" if w else "  routing looks discriminating enough to draw conclusions from")
    raise SystemExit(1 if is_degenerate(stats) else 0)


if __name__ == "__main__":
    main()
