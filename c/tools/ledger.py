#!/usr/bin/env python3
"""Ledger-shaped output for the offline research tools (#31, #32).

Both measurement tools print a verdict and, between them, exactly one JSON
report — and neither shape is something a third party can hold onto. The
numbers that decide #31 and #32 currently land in a terminal and, if anyone
remembers, in an issue comment. That is the same failure mode lex-robot hit:
its early attempts wrote their trails to /tmp and the runs became unciteable
the moment the container died.

This module is the small piece that stops that: one scaling convention and
one record shape, so a measurement can be appended to a ledger
(https://github.com/alpibrusl/lex-notebooklab) instead of being retyped.
Pure stdlib — `urllib` and `decimal` ship with Python — because these tools
run wherever a real MoE checkpoint happens to live, and a tool that needs
`pip install` before it can log is a tool that will not be run.

## Integers, and why

Ledger claims are integers. Floats invite a rounding-policy argument between
whoever wrote the number and whoever checks it, and there is no way to settle
that argument after the fact. So a rate of 0.4137 travels as `413700`
millionths and everyone compares the same integer.

`micro()` is exact-then-rounded via `decimal`, NOT `int(x * 1e6)`: the naive
form inherits binary floating-point error, so 0.145 scales to 144999 on some
values and 145000 on others depending on which literal produced it. Rounding
is half away from zero, applied once, at the end.

This deliberately matches lex-notebooklab's `micro_of` (src/derive/moe.lex),
which scales the decimal literal the same way. Nothing in the ecosystem
verifies a colibri report today — see the honesty note in `entry()` — but
when something does, the two sides will already agree bit for bit rather than
discovering a half-ulp disagreement at the worst moment.

## Usage

Both tools call `claims_report(...)` and write the result with `--out`. To
turn a report into a ledger record:

    python3 tools/ledger.py --report redundancy_report.json \\
        --series expert-redundancy --attempt olmoe-1b-7b \\
        --out entry.json                      # then: notebooklab record entry.json
    python3 tools/ledger.py --report ... --post http://localhost:8137
"""
import json
import sys
from decimal import Decimal, ROUND_HALF_UP

SCALE_PLACES = 6


def micro(x):
    """Scale a real number to millionths, exactly, rounded half away from zero.

    `repr(x)` is the shortest string that round-trips the float, which is the
    same decimal literal any other reader of this value would see — scaling
    THAT, rather than the binary value, is what makes two implementations
    agree.
    """
    d = Decimal(repr(float(x))).scaleb(SCALE_PLACES)
    return int(d.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def bit(b):
    """Booleans are claims too, and a claim has to be an integer."""
    return 1 if b else 0


def flatten(prefix, value, out):
    """Flatten a nested report into dotted integer claims.

    Claim maps are FLAT by contract (lex-notebooklab's record schema says so,
    and its importer had to learn the same lesson the hard way: a nested
    sub-object silently shadowed the run's real numbers). Floats become
    `<key>_micro`, bools become 0/1, and anything else — a string, a list, a
    None — is dropped rather than coerced, because a claim nobody can compare
    is worse than an absent one.
    """
    if isinstance(value, dict):
        for k, v in value.items():
            flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, bool):
        out[prefix] = bit(value)
    elif isinstance(value, int):
        out[prefix] = value
    elif isinstance(value, float):
        out[f"{prefix}_micro"] = micro(value)
    return out


def claims_report(tool, inputs, verdict, detail, claims):
    """The report both tools write with `--out`.

    `claims` is the ledger-facing part; `detail` keeps whatever full-fidelity
    structure the tool wants to preserve (spectra, per-group tables) and is
    NOT flattened, so nothing is lost by making the summary comparable.
    """
    return {
        "tool": tool,
        "inputs": list(inputs),
        "verdict": verdict,
        "claims": dict(sorted(claims.items())),
        "detail": detail,
    }


def entry(report, series, attempt, trainer=None, notes="", created_at=0,
          evidence=None):
    """A lex-notebooklab run record built from a report.

    Honesty note, and it belongs in the code rather than only in a PR
    description: notebooklab can only RE-DERIVE a claim from evidence it has a
    deriver for — a hash-chained trail, or a signed statement. A colibri
    measurement report is neither, so a record built here binds the report as
    an opaque artifact (digest only) and every claim reads UNVERIFIABLE
    forever. That is the honest outcome, not a shortfall: it means "we kept
    the numbers and what produced them, and nothing here pretends they were
    recomputed". Making them verifiable needs colibri to emit evidence with
    its own integrity — which is the same engine work #32 already needs for
    hit attribution, not a reporting change.
    """
    return {
        "attempt": attempt,
        "series": series,
        "trainer": trainer or report.get("tool", "colibri"),
        "config": {"inputs": report.get("inputs", [])},
        "results": report.get("claims", {}),
        "evidence": list(evidence or []),
        "notes": notes or report.get("verdict", ""),
        "supersedes": "",
        "created_at": int(created_at),
    }


def post(record, base="http://localhost:8137"):
    """POST a record to a notebooklab HTTP door. Returns its run_id."""
    import urllib.request

    req = urllib.request.Request(
        f"{base.rstrip('/')}/runs",
        data=json.dumps(record).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["run_id"]


def selftest():
    ok = True

    def check(name, got, want):
        nonlocal ok
        if got != want:
            print(f"FAIL {name}: got {got!r}, want {want!r}")
            ok = False

    # Exactness: the naive int(x * 1e6) is wrong on the first two of these.
    check("micro(0.145)", micro(0.145), 145000)
    check("micro(0.007)", micro(0.007), 7000)
    check("micro(0.05)", micro(0.05), 50000)
    check("micro(0)", micro(0.0), 0)
    check("micro(1)", micro(1.0), 1000000)
    check("micro(0.4137)", micro(0.4137), 413700)
    check("micro(1e-7)", micro(1e-7), 0)
    check("micro(5e-7)", micro(5e-7), 1)
    check("micro(-0.05)", micro(-0.05), -50000)
    check("micro(0.000012556884)", micro(0.000012556884), 13)

    flat = flatten("", {"a": 1, "b": 0.5, "c": True, "d": "x", "e": None,
                        "g": {"h": 2, "i": 0.25}}, {})
    check("flatten", flat,
          {"a": 1, "b_micro": 500000, "c": 1, "g.h": 2, "g.i_micro": 250000})

    rep = claims_report("t", ["in"], "v", {"raw": [1, 2]}, {"b": 2, "a": 1})
    check("claims sorted", list(rep["claims"]), ["a", "b"])
    rec = entry(rep, "series", "attempt", created_at=1)
    check("results are the claims", rec["results"], {"a": 1, "b": 2})
    check("no evidence by default", rec["evidence"], [])
    check("record is JSON", json.loads(json.dumps(rec))["series"], "series")

    print("selftest:", "ok" if ok else "FAILED")
    return 0 if ok else 1


def main():
    args = sys.argv[1:]
    if not args or args[0] == "--selftest":
        raise SystemExit(selftest())
    opts = {}
    for a in args:
        if a.startswith("--") and "=" in a:
            k, v = a[2:].split("=", 1)
            opts[k] = v
    if "report" not in opts:
        raise SystemExit(__doc__)
    with open(opts["report"]) as f:
        report = json.load(f)
    record = entry(
        report,
        opts.get("series", report.get("tool", "colibri")),
        opts.get("attempt", "1"),
        notes=opts.get("notes", ""),
        created_at=int(opts.get("created-at", 0)),
    )
    if "out" in opts:
        with open(opts["out"], "w") as f:
            json.dump(record, f, indent=1)
        print(f"record -> {opts['out']}")
    if "post" in opts:
        print(f"run_id {post(record, opts['post'])}")
    if "out" not in opts and "post" not in opts:
        json.dump(record, sys.stdout, indent=1)
        print()


if __name__ == "__main__":
    main()
