#!/usr/bin/env python3
"""Measure cross-expert weight redundancy in a trained MoE checkpoint (#31).

The feasibility gate for the shared-basis-plus-residual idea: do the experts
of a layer live in a low-dimensional subspace, so that one resident basis
plus small streamed residuals could replace E independently-streamed
matrices? This tool MEASURES that on a real checkpoint before anyone builds
an encoding. Random-init models (the tiny CI oracles) provably have no such
structure — run this only on trained weights.

Three views per (layer, projection):

  vec     experts flattened to vectors, stacked [E, O*I], mean-centered,
          SVD -> how many directions of expert-space carry 95% of the
          variance (are experts near-copies of few prototypes?)
  stack   expert matrices stacked along rows [E*O, I], SVD -> the rank of a
          shared INPUT-side basis B [I, r]: the decomposition the prototype
          in #31 would actually use (basis resident, coefficients streamed)
  align   naive pairwise cosine understates redundancy because expert
          weights carry exact symmetries (hidden-unit permutation; gate/up
          sign-scale freedom of the GLU pair). For the most similar pairs,
          greedily match hidden units by |correlation| (sign-corrected) and
          report cosine before/after — the exploitable "gauge".

Decision gate (stated in #31, applied by this tool): PASS for a
(layer, proj) if the stacked view reaches 95% energy at rank
r <= min(O, I) / 4. The summary reports the pass fraction across the
checkpoint; the JSON report carries every spectrum for closer reading.

Usage:
  python3 tools/expert_redundancy.py <hf_model_dir> [--layers 0,4,8]
      [--projs gate,up,down] [--max-rows 8192] [--out report.json]
  python3 tools/expert_redundancy.py --selftest      (numpy only, no weights)

Real-checkpoint mode needs torch + safetensors (oracle-requirements.txt);
--selftest needs only numpy, and validates the verdicts on synthetic expert
families with known structure (iid -> FAIL, shared-basis -> PASS, permuted
near-copies -> alignment reveals what naive cosine misses).
"""
import json
import re
import os
import sys

try:                          # imported as tools.expert_redundancy
    from . import ledger
except ImportError:           # run as a script, or loaded by file path
    # Both are real: the documented invocation is `python3 tools/<tool>.py`,
    # and tests/test_research_tools.py loads these modules by path, where
    # neither the package nor the tools directory is importable on its own.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ledger

try:
    import numpy as np
except ImportError:
    sys.exit("expert_redundancy.py needs numpy (pip install numpy)")

RE_EXPERT = re.compile(
    r"layers\.(\d+)\..*experts\.(\d+)\.(gate_proj|up_proj|down_proj|w1|w2|w3)\.weight$")
PROJ_ALIAS = {"w1": "gate", "w2": "down", "w3": "up",
              "gate_proj": "gate", "up_proj": "up", "down_proj": "down"}


def energy_rank(sv, frac=0.95):
    e = np.cumsum(sv.astype(np.float64) ** 2)
    if e[-1] <= 0:
        return len(sv)
    return int(np.searchsorted(e / e[-1], frac) + 1)


def analyze_group(mats, max_rows, rng):
    """mats: list of [O, I] float32 arrays (one per expert), same shape."""
    E = len(mats)
    O, I = mats[0].shape

    X = np.stack([m.reshape(-1) for m in mats])          # [E, O*I]
    Xc = X - X.mean(axis=0, keepdims=True)
    sv_vec = np.linalg.svd(Xc, compute_uv=False)
    r95_vec = energy_rank(sv_vec)

    Y = np.concatenate(mats, axis=0)                     # [E*O, I]
    if Y.shape[0] > max_rows:
        Y = Y[rng.choice(Y.shape[0], max_rows, replace=False)]
    sv_stack = np.linalg.svd(Y, compute_uv=False)
    r95_stack = energy_rank(sv_stack)
    gate_rank = min(O, I) / 4.0
    gate_pass = r95_stack <= gate_rank

    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    C = Xn @ Xn.T
    np.fill_diagonal(C, 0.0)
    cos_max = float(np.abs(C).max()) if E > 1 else 0.0
    cos_mean = float(np.abs(C).sum() / max(1, E * (E - 1)))

    aligned = []
    if E > 1:
        flat = np.abs(C)
        pairs = np.dstack(np.unravel_index(np.argsort(flat, axis=None)[::-1],
                                           C.shape))[0]
        seen = set()
        for a, b in pairs:
            a, b = int(a), int(b)
            if a >= b or (a, b) in seen:
                continue
            seen.add((a, b))
            before = float(C[a, b])
            A, B = mats[a], mats[b].copy()
            An = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
            Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
            M = An @ Bn.T                                # [O, O] row correlations
            perm = np.full(O, -1)
            used = np.zeros(O, bool)
            for i in np.argsort(np.abs(M).max(axis=1))[::-1]:
                j = int(np.argmax(np.where(used, -1.0, np.abs(M[i]))))
                perm[i] = j
                used[j] = True
            sign = np.sign(M[np.arange(O), perm])
            sign[sign == 0] = 1.0
            B2 = B[perm] * sign[:, None]
            after = float(np.dot(A.reshape(-1), B2.reshape(-1)) /
                          (np.linalg.norm(A) * np.linalg.norm(B2) + 1e-12))
            aligned.append({"pair": [a, b], "cos_before": before,
                            "cos_after": after})
            if len(aligned) >= 3:
                break

    return {"experts": E, "shape": [O, I],
            "r95_vec": r95_vec, "vec_dim": E,
            "r95_stack": r95_stack, "stack_dim": min(O, I),
            "gate_rank": gate_rank, "gate_pass": bool(gate_pass),
            "cos_offdiag_max": cos_max, "cos_offdiag_mean": cos_mean,
            "sv_stack_top32": [float(s) for s in sv_stack[:32]],
            "aligned_pairs": aligned}


# ---------------- real-checkpoint loading (torch + safetensors) ----------------

def load_checkpoint(model_dir, layers, projs):
    try:
        import torch  # noqa: F401
        from safetensors.torch import load_file
    except ImportError:
        sys.exit("real-checkpoint mode needs torch + safetensors "
                 "(pip install -r tools/oracle-requirements.txt)")
    import glob
    import os
    groups = {}                                          # (layer, proj) -> {eid: array}
    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shards:
        sys.exit(f"no .safetensors shards in {model_dir}")
    for shard in shards:
        tensors = load_file(shard)
        for name, t in tensors.items():
            m = RE_EXPERT.search(name)
            if not m:
                continue
            layer, eid = int(m.group(1)), int(m.group(2))
            proj = PROJ_ALIAS[m.group(3)]
            if layers and layer not in layers:
                continue
            if projs and proj not in projs:
                continue
            groups.setdefault((layer, proj), {})[eid] = \
                t.to(dtype=__import__("torch").float32).numpy()
        del tensors
    return groups


# ---------------- selftest: synthetic families with known structure ----------------

def selftest():
    rng = np.random.default_rng(0)
    O, I, E = 48, 64, 16
    max_rows = 4096

    iid = [rng.standard_normal((O, I)).astype(np.float32) for _ in range(E)]
    r_iid = analyze_group(iid, max_rows, rng)

    r = 8                                                # shared basis rank << I/4
    B = rng.standard_normal((r, I)).astype(np.float32)
    shared = [(rng.standard_normal((O, r)) @ B +
               0.02 * rng.standard_normal((O, I))).astype(np.float32)
              for _ in range(E)]
    r_shared = analyze_group(shared, max_rows, rng)

    base = rng.standard_normal((O, I)).astype(np.float32)
    permuted = [base]
    for _ in range(E - 1):
        p = rng.permutation(O)
        s = rng.choice([-1.0, 1.0], size=O).astype(np.float32)
        permuted.append((base[p] * s[:, None] +
                         0.01 * rng.standard_normal((O, I))).astype(np.float32))
    r_perm = analyze_group(permuted, max_rows, rng)

    print(f"selftest iid:      gate_pass={r_iid['gate_pass']} "
          f"r95_stack={r_iid['r95_stack']}/{r_iid['stack_dim']} "
          f"cos_max={r_iid['cos_offdiag_max']:.3f}")
    print(f"selftest shared:   gate_pass={r_shared['gate_pass']} "
          f"r95_stack={r_shared['r95_stack']}/{r_shared['stack_dim']}")
    ap = r_perm["aligned_pairs"][0]
    print(f"selftest permuted: cos_before={ap['cos_before']:.3f} "
          f"cos_after={ap['cos_after']:.3f}")

    ok = True
    if r_iid["gate_pass"]:
        print("FAIL: iid experts passed the low-rank gate"); ok = False
    if not r_shared["gate_pass"]:
        print("FAIL: shared-basis experts did not pass the gate"); ok = False
    if not (r_shared["r95_stack"] <= r + 4):
        print("FAIL: recovered rank far above the planted rank"); ok = False
    if not (abs(ap["cos_before"]) < 0.5 and ap["cos_after"] > 0.9):
        print("FAIL: alignment did not reveal the permuted near-copy"); ok = False
    print("selftest:", "ok" if ok else "FAILED")
    return 0 if ok else 1


def report_of(model_dir, report, passed):
    """The measurement as ledger claims (#34).

    The spectra and per-group tables are kept verbatim under `detail` — the
    whole point of `--out` is that nothing is lost — while `claims` is the
    flat integer summary a ledger can hold and compare. Split out of main()
    so it is testable without torch: the loader needs a real checkpoint, the
    reporting does not, and the reporting is the part that has to agree with
    every other tool's convention.
    """
    n = len(report)
    verdict = ("prototype worth building" if passed > n // 2
               else "close #31 with this data")
    claims = ledger.flatten("", report, {})
    claims["groups"] = n
    claims["groups_pass"] = passed
    return ledger.claims_report("expert_redundancy.py", [model_dir], verdict,
                                report, claims)


def main():
    args = sys.argv[1:]
    if args and args[0] == "--selftest":
        raise SystemExit(selftest())
    if not args:
        raise SystemExit(__doc__)
    model_dir, layers, projs, max_rows, out = args[0], None, None, 8192, None
    rest = args[1:]
    i = 0

    def value(flag):
        """--flag=v and the '--flag v' form this file's own usage line shows.

        The previous parser accepted only the first and reached for split("=")[1]
        unconditionally, so the documented spelling died with an IndexError
        instead of a message.
        """
        nonlocal i
        a = rest[i]
        if "=" in a:
            i += 1
            return a.split("=", 1)[1]
        if i + 1 >= len(rest):
            raise SystemExit(f"{flag}: expected a value")
        i += 2
        return rest[i - 1]

    while i < len(rest):
        a = rest[i]
        if a.startswith("--layers"):
            layers = {int(x) for x in value("--layers").split(",")}
        elif a.startswith("--projs"):
            projs = set(value("--projs").split(","))
        elif a.startswith("--max-rows"):
            max_rows = int(value("--max-rows"))
        elif a.startswith("--out"):
            out = value("--out")
        else:
            raise SystemExit(f"unknown argument: {a}")
    groups = load_checkpoint(model_dir, layers, projs)
    if not groups:
        sys.exit("no expert tensors matched — check --layers/--projs and the "
                 "naming pattern (see RE_EXPERT in this file)")
    rng = np.random.default_rng(0)
    report, passed = {}, 0
    for (layer, proj) in sorted(groups):
        by_eid = groups[(layer, proj)]
        mats = [by_eid[e] for e in sorted(by_eid)]
        res = analyze_group(mats, max_rows, rng)
        report[f"L{layer}.{proj}"] = res
        passed += res["gate_pass"]
        print(f"L{layer:3d} {proj:5s} E={res['experts']:3d} "
              f"r95(stack)={res['r95_stack']:4d}/{res['stack_dim']} "
              f"r95(vec)={res['r95_vec']:3d}/{res['vec_dim']} "
              f"cos_max={res['cos_offdiag_max']:.3f} "
              f"{'PASS' if res['gate_pass'] else 'fail'}")
    n = len(report)
    print(f"\ngate (r95_stack <= min(O,I)/4): {passed}/{n} groups pass "
          f"-> {'prototype worth building' if passed > n // 2 else 'close #31 with this data'}")
    if out:
        json.dump(report_of(model_dir, report, passed), open(out, "w"), indent=1)
        print(f"full report -> {out}  "
              f"(record it: python3 tools/ledger.py --report={out} "
              f"--series=expert-redundancy --attempt=1 --out=entry.json)")


if __name__ == "__main__":
    main()
