"""The #31/#32 offline measurement tools stay runnable.

Both tools are experiment PREREQUISITES: when a machine with a real trained
checkpoint (or a captured ROUTE_TRACE corpus) finally runs them, they must
just work — so their synthetic selftests run here, in CI, on every PR.

route_temporal.py is dependency-free (pure stdlib counting) and always runs.
expert_redundancy.py needs numpy for the SVD views; the CI python job
installs no packages, so that test skips cleanly when numpy is absent and
validates for real wherever numpy exists (dev boxes, the oracle jobs' venvs).
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
