#!/usr/bin/env python3
"""Where did two runs first diverge? (#15)

A ROUTE_TRACE says which experts each call routed to. With the replay-record
lines route_trace.h now emits -- a `#` header describing the run's
configuration, and `T <call> <token>` for each emitted token -- a trace becomes
a self-contained record of a session, and two of them can be compared.

That is the point of the feature: when the same prompt gives different output on
a different backend, quantization or machine, "the output differs" is not
actionable. **Which call, which layer, which expert** is.

## Configuration difference is not divergence

Two runs at different seeds or temperatures are not expected to agree, and
calling their disagreement a divergence would be worse than useless. The header
is compared first and a mismatch is reported as a DIFFERENT EXPERIMENT, not a
bug -- with the fields that differ named, so the reader can decide.

## What "first" means

Routing and output are ordered against each other by call index: `T <call> ...`
is written after the routing lines of that call, so a routing divergence at call
N precedes a token divergence at call N. This matters because routing is the
cause and the token is the symptom; reporting the token first would point at the
wrong place.

Usage:
  python3 tools/replay_diff.py record_a.txt record_b.txt
  python3 tools/replay_diff.py --selftest
"""
import os
import sys


def parse(path):
    """(header dict, {(call,row,layer): [ids]}, {call: token}, n_routing_lines)"""
    header, routes, tokens, n = {}, {}, {}, 0
    with open(path) as fh:
        for line in fh:
            if line.startswith("# colibri-replay"):
                for field in line.split()[2:]:
                    if "=" in field:
                        k, v = field.split("=", 1)
                        header[k] = v
                continue
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 3:
                continue
            if p[0] == "T":
                try:
                    tokens[int(p[1])] = int(p[2])
                except ValueError:
                    pass
                continue
            if len(p) < 4:
                continue
            try:
                call, row, layer = int(p[0]), int(p[1]), int(p[2])
            except ValueError:
                continue
            ids = []
            for tok in p[3:]:
                try:
                    ids.append(int(tok.split(":", 1)[0]))
                except ValueError:
                    pass
            routes[(call, row, layer)] = ids
            n += 1
    return header, routes, tokens, n


def header_diff(a, b):
    """Fields that differ, ignoring ones absent from both."""
    keys = sorted(set(a) | set(b))
    return [(k, a.get(k, "<absent>"), b.get(k, "<absent>")) for k in keys
            if a.get(k) != b.get(k)]


def first_routing_divergence(ra, rb):
    """Earliest (call,row,layer) whose routed set differs, or None."""
    for key in sorted(set(ra) | set(rb)):
        ia, ib = ra.get(key), rb.get(key)
        if ia != ib:
            return key, ia, ib
    return None


def first_token_divergence(ta, tb):
    for call in sorted(set(ta) | set(tb)):
        va, vb = ta.get(call), tb.get(call)
        if va != vb:
            return call, va, vb
    return None


def compare(pa, pb):
    ha, ra, ta, na = parse(pa)
    hb, rb, tb, nb = parse(pb)
    out = []
    hd = header_diff(ha, hb)
    if hd:
        out.append("DIFFERENT EXPERIMENT -- the records disagree about the run itself:")
        for k, va, vb in hd:
            out.append(f"    {k}: {va}  vs  {vb}")
        out.append("  Two runs configured differently are not expected to agree;")
        out.append("  this is not a divergence. Re-run them with matching settings.")
        return 2, out

    out.append(f"  A: {na} routing lines, {len(ta)} tokens")
    out.append(f"  B: {nb} routing lines, {len(tb)} tokens")

    rd = first_routing_divergence(ra, rb)
    td = first_token_divergence(ta, tb)

    if rd is None and td is None:
        out.append("  IDENTICAL -- same routing at every call, same tokens")
        return 0, out

    # Routing is the cause, the token is the symptom: report the earlier one, and
    # on a tie report routing, because a token that differs at the same call as
    # its routing did so BECAUSE of it.
    if rd is not None and (td is None or rd[0][0] <= td[0]):
        (call, row, layer), ia, ib = rd
        out.append(f"  FIRST DIVERGENCE: routing, call {call} row {row} layer {layer}")
        out.append(f"    A routed {ia}")
        out.append(f"    B routed {ib}")
        if ia is not None and ib is not None:
            only_a, only_b = sorted(set(ia) - set(ib)), sorted(set(ib) - set(ia))
            out.append(f"    only in A: {only_a}   only in B: {only_b}")
        if td is not None:
            out.append(f"  (output first differs later, at call {td[0]}: {td[1]} vs {td[2]})")
    else:
        call, va, vb = td
        out.append(f"  FIRST DIVERGENCE: output, call {call} -- token {va} vs {vb}")
        out.append("    routing agreed everywhere, so the difference is in sampling,")
        out.append("    not in which experts ran")
    return 1, out


# ---------------------------------------------------------------- selftest
def selftest():
    import tempfile
    ok = True
    hdr = "# colibri-replay 1 engine=x layers=2 experts=8 seed=7 temp=0.700000 top_p=0.900000\n"

    def write(tmp, name, body, header=hdr):
        p = os.path.join(tmp, name)
        with open(p, "w") as fh:
            fh.write(header + body)
        return p

    base = "0 0 0 1:0.5 2:0.5\n0 0 1 3:0.5 4:0.5\nT 0 42\n1 0 0 1:0.5 2:0.5\nT 1 43\n"
    with tempfile.TemporaryDirectory() as tmp:
        a = write(tmp, "a.txt", base)
        b = write(tmp, "b.txt", base)
        rc, _ = compare(a, b)
        if rc != 0:
            print(f"selftest: identical records reported rc={rc}"); ok = False

        # a routing difference at call 1
        b2 = write(tmp, "b2.txt", base.replace("1 0 0 1:0.5 2:0.5", "1 0 0 1:0.5 5:0.5"))
        rc, lines = compare(a, b2)
        joined = "\n".join(lines)
        if rc != 1 or "routing, call 1" not in joined:
            print(f"selftest: routing divergence not located: {joined}"); ok = False
        if "only in B: [5]" not in joined:
            print(f"selftest: differing expert not named: {joined}"); ok = False

        # a token difference with routing identical -> reported as sampling
        b3 = write(tmp, "b3.txt", base.replace("T 1 43", "T 1 99"))
        rc, lines = compare(a, b3)
        joined = "\n".join(lines)
        if rc != 1 or "output, call 1" not in joined or "sampling" not in joined:
            print(f"selftest: token-only divergence misreported: {joined}"); ok = False

        # routing wins a tie: both differ at call 1, routing must be reported
        b4 = write(tmp, "b4.txt", base.replace("1 0 0 1:0.5 2:0.5", "1 0 0 1:0.5 5:0.5")
                                      .replace("T 1 43", "T 1 99"))
        rc, lines = compare(a, b4)
        joined = "\n".join(lines)
        if "FIRST DIVERGENCE: routing" not in joined:
            print(f"selftest: routing must win a tie against output: {joined}"); ok = False

        # a configuration difference is NOT a divergence
        b5 = write(tmp, "b5.txt", base,
                   header=hdr.replace("seed=7", "seed=8"))
        rc, lines = compare(a, b5)
        joined = "\n".join(lines)
        if rc != 2 or "DIFFERENT EXPERIMENT" not in joined or "seed" not in joined:
            print(f"selftest: config difference misreported: {joined}"); ok = False

    print("selftest: ok" if ok else "selftest: FAILED")
    return 0 if ok else 1


def main():
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        raise SystemExit(selftest())
    if len(args) != 2:
        raise SystemExit(__doc__)
    rc, lines = compare(args[0], args[1])
    print(f"{os.path.basename(args[0])}  vs  {os.path.basename(args[1])}")
    for l in lines:
        print(l)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
