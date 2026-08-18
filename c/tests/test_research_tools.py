"""The #31/#32 offline measurement tools stay runnable, and stay recordable.

Both tools are experiment PREREQUISITES: when a machine with a real trained
checkpoint (or a captured ROUTE_TRACE corpus) finally runs them, they must
just work — so their synthetic selftests run here, in CI, on every PR.

route_temporal.py is dependency-free (pure stdlib counting) and always runs.
expert_redundancy.py needs numpy for the SVD views; the CI python job
installs no packages, so that test skips cleanly when numpy is absent and
validates for real wherever numpy exists (dev boxes, the oracle jobs' venvs).

The ledger cases below cover the other half of "just work": a measurement
that costs a real checkpoint has to survive the terminal it ran in, so both
tools emit a flat integer claim map (tools/ledger.py) that a run ledger can
hold. The scaling convention is the part worth pinning — an off-by-one in the
last digit is exactly the kind of thing nobody notices until two sides
disagree about a number neither can re-measure.
"""
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"


def _run(tool):
    return subprocess.run(
        [sys.executable, str(TOOLS / tool), "--selftest"],
        capture_output=True, text=True, timeout=300)


class ResearchToolSelftests(unittest.TestCase):
    def test_route_temporal_selftest(self):
        proc = _run("route_temporal.py")
        self.assertEqual(proc.returncode, 0,
                         f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("selftest: ok", proc.stdout)

    def test_expert_redundancy_selftest(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed (CI python job is bare); "
                          "runs wherever the oracle deps exist")
        proc = _run("expert_redundancy.py")
        self.assertEqual(proc.returncode, 0,
                         f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("selftest: ok", proc.stdout)


class LedgerShape(unittest.TestCase):
    """Claims are integers, flat, and scaled the same way everywhere."""

    def _ledger(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ledger", TOOLS / "ledger.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_ledger_selftest(self):
        proc = _run("ledger.py")
        self.assertEqual(proc.returncode, 0,
                         f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("selftest: ok", proc.stdout)

    def test_micro_is_exact_not_float_multiplication(self):
        # int(0.145 * 1e6) == 144999: the naive scaling inherits binary
        # floating-point error, and the error lands in the digit a reader
        # would quote. This is the whole reason ledger.micro exists.
        led = self._ledger()
        self.assertEqual(led.micro(0.145), 145000)
        self.assertEqual(led.micro(0.007), 7000)
        self.assertEqual(led.micro(-0.05), -50000)
        self.assertEqual(led.micro(5e-7), 1)      # half rounds away from zero
        self.assertEqual(led.micro(4e-7), 0)

    def test_route_temporal_report_is_flat_integers(self):
        import importlib.util
        import json
        spec = importlib.util.spec_from_file_location(
            "route_temporal", TOOLS / "route_temporal.py")
        rt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rt)

        trace = HERE / "_tmp_ledger_trace.txt"
        trace.write_text(
            "0 0 1 1:0.5 2:0.5\n"
            "1 0 2 3:0.5 4:0.5\n"
            "2 0 1 1:0.5 2:0.5\n"
            "3 0 2 3:0.5 4:0.5\n"
            "4 0 1 9:0.5 8:0.5\n"
            "5 0 2 3:0.5 7:0.5\n")
        out = HERE / "_tmp_ledger_report.json"
        try:
            seqs = rt.parse_traces([str(trace)])
            recalls, total = rt.evaluate(seqs)
            report = rt.report_of([str(trace)], seqs, recalls, total, 1)
            claims = report["claims"]
            for key, value in claims.items():
                self.assertIsInstance(value, int, f"claim {key} is not an integer")
            self.assertEqual(claims["scores"], total)
            self.assertEqual(claims["recall.identity_micro"], 625000)
            # The verdict bit and the printed sentence come from one rule, so
            # they cannot drift: gain here is well under the threshold.
            gain = max(recalls["identity"], recalls["conditional"]) - recalls["heat"]
            self.assertEqual(claims["worth_engine_slice"],
                             1 if gain > rt.GAIN_THRESHOLD else 0)
            self.assertEqual(claims["temporal_gain_micro"], self._ledger().micro(gain))
            # The full-fidelity numbers survive alongside the claims.
            self.assertIn("recalls", report["detail"])
            out.write_text(json.dumps(report))
            record = self._ledger().entry(json.loads(out.read_text()),
                                          "route-temporal", "1")
            self.assertEqual(record["results"], claims)
            self.assertEqual(record["evidence"], [])
            json.dumps(record)          # a record is JSON, not just a dict
        finally:
            trace.unlink(missing_ok=True)
            out.unlink(missing_ok=True)

    def test_expert_redundancy_report_is_flat_integers(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed (CI python job is bare)")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "expert_redundancy", TOOLS / "expert_redundancy.py")
        er = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(er)

        # The reporting half needs no checkpoint, which is why it is a
        # function: the loader needs torch, the claim shape does not.
        groups = {
            "L0.gate": {"experts": 4, "r95_stack": 7, "stack_dim": 48,
                        "cos_offdiag_max": 0.05, "gate_pass": True},
            "L1.gate": {"experts": 4, "r95_stack": 59, "stack_dim": 48,
                        "cos_offdiag_max": 0.5, "gate_pass": False},
        }
        report = er.report_of("/models/x", groups, passed=1)
        claims = report["claims"]
        for key, value in claims.items():
            self.assertIsInstance(value, int, f"claim {key} is not an integer")
        self.assertEqual(claims["groups"], 2)
        self.assertEqual(claims["groups_pass"], 1)
        self.assertEqual(claims["L0.gate.gate_pass"], 1)
        self.assertEqual(claims["L1.gate.cos_offdiag_max_micro"], 500000)
        self.assertIs(report["detail"], groups)   # nothing is lost


class RouteTemporalParsing(unittest.TestCase):
    """The trace parser agrees with route_trace.h's documented format."""

    def test_forward_grouping_and_recall(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "route_temporal", TOOLS / "route_temporal.py")
        rt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rt)

        # two forwards (tokens), 2 sparse layers each; layer drop = boundary.
        # token 0 routes {1,2}/{3,4}; token 1 keeps them exactly (identity
        # recall 1.0) while heat has only token 0's counts (also full recall
        # here since sets repeat -- so use a third token that changes).
        trace = HERE / "_tmp_route_temporal_trace.txt"
        trace.write_text(
            "0 0 1 1:0.5 2:0.5\n"
            "1 0 2 3:0.5 4:0.5\n"
            "2 0 1 1:0.5 2:0.5\n"
            "3 0 2 3:0.5 4:0.5\n"
            "4 0 1 9:0.5 8:0.5\n"
            "5 0 2 3:0.5 7:0.5\n")
        try:
            seqs = rt.parse_traces([str(trace)])
            self.assertEqual(len(seqs), 1)
            self.assertEqual(len(seqs[0]), 3)            # three forwards
            self.assertEqual(seqs[0][0], {1: [1, 2], 2: [3, 4]})
            recalls, total = rt.evaluate(seqs)
            self.assertEqual(total, 4)                   # 2 transitions x 2 layers
            # transition 0->1 is an exact repeat (identity recall 1.0 on both
            # layers); 1->2 shares only expert 3 on layer 2 (0.5) and nothing
            # on layer 1 (0.0) -> identity = (1+1+0+0.5)/4
            self.assertAlmostEqual(recalls["identity"], 0.625, places=6)
        finally:
            trace.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
