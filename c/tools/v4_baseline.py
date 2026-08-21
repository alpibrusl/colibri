#!/usr/bin/env python3
"""A repeatable DeepSeek V4 baseline, so improvements can be measured (#62).

V4 Flash is the only big model that fits on an ordinary workstation (167 GB
against GLM's 372, Inkling's 469, Kimi's 1.6 TB), which makes it the one place
this engine's streaming claims can be checked at scale. It is also the engine
with no routing telemetry (#62), so the aggregate counters it *does* emit --
`v4_tokens` and `timing` -- are currently the only measurement available.

This pins them as a baseline: same prompt, same budget, N runs, median reported.
Fields are read BY NAME out of the engine's own output, never by position: an
earlier version of this parsed with a positional sed, and when the engine was
not built it reported a clean table of zeros rather than an error. A baseline
that can silently print zeros is worse than no baseline.

Usage:
  python3 tools/v4_baseline.py --model DIR [--ram 32] [--runs 3] [--cold] [--prompt "..."]
  python3 tools/v4_baseline.py --selftest

--ngen is a CAP, not a target: generation stops at EOS. "What is the capital of
France?" ends after 9 tokens whatever --ngen says, so a sweep over --ngen with a
short prompt varies nothing. Vary sequence length with a prompt that generates.
"""
import os
import re
import statistics
import subprocess
import sys

# name -> (line prefix the field belongs to, regex, cast)
#
# Each field is scoped to the LINE that owns it. A bare `bytes=(\d+)` looks
# right and is not: the engine also prints `v4_dense_resident layers=43
# bytes=6.267GiB` earlier in the run, so an unscoped search returns 6 and the
# baseline reports 0.00 GB streamed -- a clean, plausible, wrong number, which
# is the failure this tool exists to prevent. The selftest below carries that
# decoy line so the mistake cannot come back.
FIELDS = {
    "hit_rate":        ("v4_tokens", r"hit_rate=([0-9.]+)", float),
    "expert_requests": ("v4_tokens", r"expert_requests=(\d+)", int),
    "hits":            ("v4_tokens", r"\bhits=(\d+)", int),
    "misses":          ("v4_tokens", r"misses=(\d+)", int),
    "bytes":           ("v4_tokens", r"\bbytes=(\d+)", int),
    "generated":       ("v4_tokens", r"generated=(\d+)", int),
    # `distinct` is the compulsory-miss floor: a demand-fetch cache must miss at
    # least once per distinct expert, so it sets the ceiling on any hit rate a
    # pure cache can reach. Reported because it is what makes the hit rate
    # interpretable -- 43% looks poor until you see the ceiling is 43.7%.
    "distinct":        ("v4_autopin saved", r"distinct=(\d+)", int),
    "ttft_s":          ("timing",    r"time_to_first_token=([0-9.]+)s", float),
    "decode_s":        ("TUNE",      r"decode: \d+ tokens in ([0-9.]+)s", float),
}


def parse(text):
    """Every field by name. Missing fields are an error, not a zero."""
    out, missing = {}, []
    lines = text.splitlines()
    for name, (prefix, rx, cast) in FIELDS.items():
        hit = None
        for line in lines:
            if prefix in line:
                m = re.search(rx, line)
                if m:
                    hit = cast(m.group(1))
                    break
        if hit is None:
            missing.append(name)
        else:
            out[name] = hit
    return out, missing


def run_once(binary_dir, model, ram, prompt, cold=False, ngen=None):
    # The engine writes `.coli_usage` (autopin) into the model dir and reads it
    # back on the next run, so runs are NOT independent: a warm run inherits the
    # previous run's learned pins and scores higher. Both numbers are worth
    # having -- cold is what a fresh deployment sees, warm is what a served
    # model sees -- but they must not be mixed in one median, so cold runs
    # remove the file first and each cold run is genuinely cold.
    if cold:
        usage = os.path.join(model, ".coli_usage")
        if os.path.exists(usage):
            os.remove(usage)
    cmd = [sys.executable, "./coli", "run", "--model", model, "--ram", str(ram)]
    if ngen:
        cmd += ["--ngen", str(ngen)]
    p = subprocess.run(cmd + [prompt],
                       cwd=binary_dir, capture_output=True, text=True, timeout=3600)
    return p.stdout + p.stderr


def selftest():
    ok = True
    sample = ("v4_dense_resident layers=43 bytes=6.267GiB\n"      # the decoy
              "v4_autopin saved=/m/.coli_usage selections=3504 distinct=1972\n"
              "v4_tokens prompt=11 generated=9 total=20 expert_requests=3504 "
              "hits=1527 misses=1977 hit_rate=43.579 bytes=26431193088 target_only=1\n"
              "TUNE decode: 9 tokens in 88.463s\n"
              "timing time_to_first_token=50.380s after_first=38.081s\n")
    got, missing = parse(sample)
    if missing:
        print(f"selftest: fields not parsed: {missing}"); ok = False
    for k, v in (("hit_rate", 43.579), ("expert_requests", 3504), ("hits", 1527),
                 ("misses", 1977), ("bytes", 26431193088), ("generated", 9),
                 ("ttft_s", 50.380), ("decode_s", 88.463), ("distinct", 1972)):
        if got.get(k) != v:
            print(f"selftest: {k} parsed as {got.get(k)}, expected {v}"); ok = False
    # the case that motivated parsing by name: the engine not built must be an
    # ERROR, never a table of zeros
    got2, missing2 = parse("deepseek-v4 engine is not built. Run: make -C c deepseek-v4\n")
    if not missing2 or got2:
        print(f"selftest: a non-run must report missing fields, got {got2} / {missing2}")
        ok = False
    print("selftest: ok" if ok else "selftest: FAILED")
    return 0 if ok else 1


def main():
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        raise SystemExit(selftest())
    model = ram = None
    runs, prompt, cold, ngen = 3, "What is the capital of France?", False, None
    i = 0

    def value(flag):
        nonlocal i
        a = args[i]
        if "=" in a and a.startswith("--"):
            i += 1
            return a.split("=", 1)[1]
        if i + 1 >= len(args):
            raise SystemExit(f"{flag}: expected a value")
        i += 2
        return args[i - 1]

    while i < len(args):
        a = args[i]
        if a.startswith("--model"):
            model = value("--model")
        elif a.startswith("--ram"):
            ram = value("--ram")
        elif a.startswith("--runs"):
            runs = int(value("--runs"))
        elif a.startswith("--prompt"):
            prompt = value("--prompt")
        elif a.startswith("--ngen"):
            ngen = int(value("--ngen"))
        elif a == "--cold":
            cold = True
            i += 1
        else:
            raise SystemExit(f"unknown argument: {a}")
    if not model:
        raise SystemExit(__doc__)
    ram = ram or "32"
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    rows = []
    for n in range(runs):
        got, missing = parse(run_once(here, os.path.expanduser(model), ram, prompt, cold, ngen))
        if missing:
            print(f"run {n+1}: FAILED -- fields missing: {missing}")
            print("  the engine did not produce its counters; not reporting a number")
            return 1
        rows.append(got)
        print(f"  run {n+1}: hit {got['hit_rate']:.2f}%  "
              f"{got['bytes']/1e9:.2f} GB  ttft {got['ttft_s']:.1f}s  "
              f"decode {got['decode_s']:.1f}s")

    def med(k):
        return statistics.median(r[k] for r in rows)

    print(f"\nBASELINE  --ram {ram}  {'cold' if cold else 'warm'}  "
          f"x{runs} runs, median")
    total, distinct = med("expert_requests"), med("distinct")
    # The ceiling is only sound when `distinct` describes THIS run. The engine
    # accumulates `.coli_usage` across runs, so on a warm run `distinct` counts
    # every expert any previous prompt touched. Same prompt + deterministic
    # routing makes it coincide; a different earlier prompt inflates it and
    # silently understates the ceiling. Only claim it where it is guaranteed.
    if cold:
        ceiling = 100.0 * (total - distinct) / total
        print(f"  hit rate        {med('hit_rate'):.2f}%   "
              f"(ceiling {ceiling:.2f}%, gap {ceiling - med('hit_rate'):.2f} pp)")
        print(f"  distinct experts {int(distinct)}  -> compulsory misses; "
              f"capacity misses {int(med('misses') - distinct)}")
    else:
        print(f"  hit rate        {med('hit_rate'):.2f}%")
        print(f"  distinct experts {int(distinct)} (cumulative over .coli_usage "
              f"history -- rerun with --cold for a ceiling)")
    print(f"  expert requests {int(med('expert_requests'))} "
          f"({int(med('hits'))} hits / {int(med('misses'))} misses)")
    print(f"  bytes streamed  {med('bytes')/1e9:.2f} GB")
    print(f"  bytes/token     {med('bytes')/med('generated')/1e6:.1f} MB")
    print(f"  TTFT            {med('ttft_s'):.1f} s")
    print(f"  decode          {med('decode_s'):.1f} s "
          f"({med('generated')/med('decode_s'):.3f} tok/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
