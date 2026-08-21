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


class TraceHealth(unittest.TestCase):
    """A trace that cannot support a conclusion must not receive a verdict."""

    def _mod(self, name):
        import importlib.util
        spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _trace(self, gates, rows=40, layers=3):
        path = HERE / "_tmp_health_trace.txt"
        with open(path, "w") as f:
            call = 0
            for _ in range(rows):
                for layer in range(layers):
                    f.write(f"{call} 0 {layer} " +
                            " ".join(f"{i}:{g:.4f}" for i, g in enumerate(gates)) + "\n")
                    call += 1
        return path

    def test_selftest(self):
        proc = _run("trace_health.py")
        self.assertEqual(proc.returncode, 0,
                         f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("selftest: ok", proc.stdout)

    def test_uniform_gates_are_degenerate(self):
        th = self._mod("trace_health")
        # The measurement from a random-weight OLMoE checkpoint run through the
        # real engine: gates within 4% of each other, entropy 0.9999.
        path = self._trace([0.1284, 0.1268, 0.1251, 0.1242,
                            0.1241, 0.1238, 0.1238, 0.1238])
        try:
            stats = th.gate_stats([str(path)])
            self.assertTrue(th.is_degenerate(stats))
            self.assertIsNotNone(th.warning(stats))
            self.assertEqual(th.claims(stats)["gate.degenerate"], 1)
        finally:
            path.unlink(missing_ok=True)

    def test_peaked_gates_are_not_degenerate(self):
        th = self._mod("trace_health")
        path = self._trace([0.60, 0.20, 0.10, 0.04, 0.03, 0.01, 0.01, 0.01])
        try:
            stats = th.gate_stats([str(path)])
            self.assertFalse(th.is_degenerate(stats))
            self.assertIsNone(th.warning(stats))
        finally:
            path.unlink(missing_ok=True)

    def test_degenerate_trace_suppresses_both_verdicts(self):
        # The regression this exists for. On a random-weight trace the two
        # tools reported +0.135 temporal gain and a 21.6% read reduction —
        # both over their thresholds, both meaningless. The gain may still be
        # reported; the VERDICT must not be.
        th = self._mod("trace_health")
        rt = self._mod("route_temporal")
        el = self._mod("expert_layout")
        path = self._trace([0.125] * 8)
        try:
            gate = th.gate_stats([str(path)])
            self.assertTrue(th.is_degenerate(gate))

            seqs = rt.parse_traces([str(path)])
            recalls, total = rt.evaluate(seqs)
            report = rt.report_of([str(path)], seqs, recalls, total, 1, gate)
            self.assertEqual(report["claims"]["worth_engine_slice"], 0)
            self.assertIn("degenerate", report["verdict"])
            self.assertEqual(report["claims"]["gate.degenerate"], 1)

            forwards = el.parse_forwards([str(path)])
            scored, _, _ = el.evaluate(forwards)
            report = el.report_of([str(path)], forwards, scored, el.gains(scored), gate)
            self.assertEqual(report["claims"]["worth_engine_slice"], 0)
            self.assertIn("degenerate", report["verdict"])
        finally:
            path.unlink(missing_ok=True)


class ExpertLayout(unittest.TestCase):
    """#36: the layout tool only reports a win when there is one to report."""

    def _mod(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "expert_layout", TOOLS / "expert_layout.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_selftest(self):
        proc = _run("expert_layout.py")
        self.assertEqual(proc.returncode, 0,
                         f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("selftest: ok", proc.stdout)

    def test_reads_metric_endpoints(self):
        el = self._mod()
        slots = el.slots_of([0, 1, 2, 3, 4, 5])
        self.assertEqual(el.runs_of(slots, {1, 2, 3}), 1)   # contiguous: one read
        self.assertEqual(el.runs_of(slots, {0, 2, 4}), 3)   # scattered: |S| reads
        self.assertEqual(el.runs_of(slots, set()), 0)

    def test_clustered_routing_is_detected_out_of_sample(self):
        import random
        el = self._mod()
        rng = random.Random(3)
        forwards = el.synth_clustered(300, 3, 48, group=6, k=3, rng=rng)
        scored, _, _ = el.evaluate(forwards)
        gains = el.gains(scored)
        # Beating BOTH baselines is the gate; beating only the random control
        # would mean the metric moves under any permutation.
        self.assertGreater(gains["vs_identity"], el.GAIN_THRESHOLD)
        self.assertGreater(gains["vs_random"], el.GAIN_THRESHOLD)

    def test_structureless_routing_reports_no_win(self):
        # The regression that matters. Scored in-sample this tool claimed a ~9%
        # read reduction on routing generated with no structure at all —
        # a fitted-on-the-eval-set artifact that would have justified a
        # container rewrite for nothing. Held out, it must report ~zero.
        import random
        el = self._mod()
        rng = random.Random(5)
        forwards = el.synth_uniform(300, 3, 48, k=3, rng=rng)
        scored, _, _ = el.evaluate(forwards)
        self.assertLessEqual(el.gains(scored)["vs_identity"], el.GAIN_THRESHOLD)

    def test_report_is_flat_integers_and_verdict_follows_the_rule(self):
        import random
        el = self._mod()
        rng = random.Random(7)
        forwards = el.synth_clustered(200, 2, 32, group=4, k=2, rng=rng)
        scored, _, _ = el.evaluate(forwards)
        gains = el.gains(scored)
        report = el.report_of(["synthetic"], forwards, scored, gains)
        for key, value in report["claims"].items():
            self.assertIsInstance(value, int, f"claim {key} is not an integer")
        worth = (gains["vs_identity"] > el.GAIN_THRESHOLD
                 and gains["vs_random"] > el.GAIN_THRESHOLD)
        self.assertEqual(report["claims"]["worth_engine_slice"], 1 if worth else 0)
        self.assertEqual(report["claims"]["forwards"], len(forwards))

    def test_emitted_layout_is_a_permutation(self):
        # A container rewrite consumes this directly, so a dropped or duplicated
        # expert would corrupt the model rather than merely slow it down.
        import random
        el = self._mod()
        rng = random.Random(9)
        forwards = el.synth_clustered(120, 3, 24, group=4, k=3, rng=rng)
        _, placements, by_layer = el.evaluate(forwards)
        doc = el.permutation_doc(placements, by_layer)
        self.assertEqual(doc["version"], 1)
        for layer, order in doc["layers"].items():
            expected = by_layer[int(layer)]
            self.assertEqual(sorted(order), sorted(expected),
                             f"layer {layer}: layout is not a permutation")
            self.assertEqual(len(order), len(set(order)),
                             f"layer {layer}: duplicate slot assignment")


class ExpertRelayout(unittest.TestCase):
    """#36's container half: reorder the bytes, change nothing else."""

    def _mod(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "expert_relayout", TOOLS / "expert_relayout.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_selftest(self):
        proc = _run("expert_relayout.py")
        self.assertEqual(proc.returncode, 0,
                         f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("selftest: ok", proc.stdout)

    def _container(self, root, layers=2, experts=6, kinds=(("qs", 16), ("w", 96))):
        """A container with the real pathology: all scales, then all weights."""
        import json as _json
        import struct as _struct
        root.mkdir(parents=True, exist_ok=True)
        names = []
        for kind, size in kinds:
            for layer in range(layers):
                for eid in sorted(range(experts), key=str):   # lexicographic, as the real one is
                    names.append((f"model.layers.{layer}.mlp.experts.{eid}.{kind}", size))
        blobs = {n: bytes((i * 31 + j) % 251 for j in range(sz))
                 for i, (n, sz) in enumerate(names)}
        header, cursor = {}, 0
        for n, sz in names:
            header[n] = {"dtype": "U8", "shape": [sz], "data_offsets": [cursor, cursor + sz]}
            cursor += sz
        blob = _json.dumps(header, separators=(",", ":")).encode()
        with open(root / "model-00000.safetensors", "wb") as fh:
            fh.write(_struct.pack("<Q", len(blob)))
            fh.write(blob)
            for n, _ in names:
                fh.write(blobs[n])
        return len(names)

    def test_rewrite_preserves_every_byte(self):
        # The entire safety argument. A rewrite that changed one byte would
        # change the model, and the failure would look like a quality
        # regression rather than a corrupt file.
        import tempfile
        from pathlib import Path
        er = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "src", Path(tmp) / "dst"
            n = self._container(src)
            tensors, order, meta = er.index_container(str(src))
            self.assertEqual(len(tensors), n)
            perm = {"version": 1, "layers": {"0": [5, 4, 3, 2, 1, 0],
                                             "1": [0, 2, 4, 1, 3, 5]}}
            new, dropped = er.emission_order(tensors, order, perm)
            self.assertEqual(dropped, 0)
            er.write_container(new, str(dst), shard_bytes=1 << 20, metadata=meta)
            self.assertEqual(er.verify(tensors, str(dst)), [])

    def test_expert_tensors_become_adjacent(self):
        # The within-expert win: the real merged container puts every .qs in
        # one region and every weight in another, so the two reads olmoe.c
        # issues per expert land far apart.
        import tempfile
        from pathlib import Path
        er = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            self._container(src)
            tensors, order, _ = er.index_container(str(src))
            perm = {"version": 1, "layers": {"0": list(range(6)), "1": list(range(6))}}
            new, _ = er.emission_order(tensors, order, perm)
            self.assertGreater(er.locality([tensors[n] for n in order]), 0)
            self.assertEqual(er.locality(new), 0)

    def test_expert_absent_from_permutation_survives(self):
        # A permutation is fitted on a trace, which only names experts that
        # trace routed to. The container must still carry all of them.
        import tempfile
        from pathlib import Path
        er = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src"
            n = self._container(src)
            tensors, order, _ = er.index_container(str(src))
            new, dropped = er.emission_order(
                tensors, order, {"version": 1, "layers": {"0": [4, 2]}})
            self.assertEqual(len(new), n, "rewrite lost tensors")
            self.assertEqual(sorted(e["name"] for e in new), sorted(tensors))
            self.assertGreater(dropped, 0)

    def test_measured_reads_drop_for_a_clustered_layout(self):
        # End to end against real header offsets: fewer, longer reads at
        # identical bytes moved, which is #36's actual claim.
        import tempfile
        from pathlib import Path
        er = self._mod()
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "src", Path(tmp) / "dst"
            self._container(src, layers=1, experts=6)
            tensors, order, meta = er.index_container(str(src))
            # a trace whose every forward routes to {0,1,2} -- adjacency should
            # collapse three scattered reads into one
            trace = Path(tmp) / "t.txt"
            trace.write_text("".join(f"{c} 0 0 0:0.4 1:0.3 2:0.3\n" for c in range(8)))
            perm = {"version": 1, "layers": {"0": [0, 1, 2, 3, 4, 5]}}
            new, _ = er.emission_order(tensors, order, perm)
            er.write_container(new, str(dst), shard_bytes=1 << 20, metadata=meta)

            per = er.routed_sets(str(trace))
            before, b_bytes = er.count_reads(per, er.byte_ranges(str(src)))
            after, a_bytes = er.count_reads(per, er.byte_ranges(str(dst)))
            self.assertLess(after, before, "clustering did not reduce reads")
            self.assertEqual(a_bytes, b_bytes, "bytes moved must not change")


class SessionOverlap(unittest.TestCase):
    """#37's kill condition: do concurrent sessions route to disjoint experts?"""

    def _mod(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "session_overlap", TOOLS / "session_overlap.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_selftest(self):
        proc = _run("session_overlap.py")
        self.assertEqual(proc.returncode, 0,
                         f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("selftest: ok", proc.stdout)

    def test_sessions_are_drawn_without_replacement(self):
        # With replacement, S=8 drawn from 16 calls collides often enough to
        # report ~19% amortization on routing built to have exactly none. S
        # concurrent sessions occupy S DIFFERENT positions.
        so = self._mod()
        disjoint = {0: [set(range(i * 4, i * 4 + 4)) for i in range(256)]}
        res = so.analyse(disjoint, max_s=8, trials=300)
        drop, _, note = so.verdict(res)
        self.assertLessEqual(drop, so.AMORTIZATION_THRESHOLD)
        self.assertIn("nothing to amortize", note)

    def test_call_ids_are_namespaced_per_file(self):
        # Every ROUTE_TRACE numbers its calls from 0. Keying on the bare id
        # fuses trace A's call 7 with trace B's, merging two unrelated routing
        # decisions into one oversized session and inflating mean top-k.
        so = self._mod()
        a = HERE / "_tmp_overlap_a.txt"
        b = HERE / "_tmp_overlap_b.txt"
        a.write_text("0 0 0 1:0.5 2:0.5\n1 0 0 1:0.5 2:0.5\n")
        b.write_text("0 0 0 30:0.5 31:0.5\n1 0 0 30:0.5 31:0.5\n")
        try:
            merged = so.parse([str(a), str(b)])
            self.assertEqual(len(merged[0]), 4, "calls from two files were fused")
            for routed in merged[0]:
                self.assertEqual(len(routed), 2, "a call absorbed another file's experts")
        finally:
            a.unlink(missing_ok=True)
            b.unlink(missing_ok=True)

    def test_identical_sessions_amortize_and_uniform_sits_on_the_baseline(self):
        import random
        so = self._mod()
        same = {0: [set(range(4)) for _ in range(256)]}
        drop, _, _ = so.verdict(so.analyse(same, max_s=8, trials=200))
        self.assertGreater(drop, 0.8, "identical routing must amortize almost fully")

        # uniform independent routing must land ON the analytic line, since that
        # line is what every real measurement is judged against
        rng = random.Random(4)
        sets = [set(rng.sample(range(64), 8)) for _ in range(500)]
        _, corr, _ = so.verdict(so.analyse({0: sets}, max_s=8, trials=400))
        self.assertLess(abs(corr), 0.05)


class ExpertIoReplay(unittest.TestCase):
    """#36's last open risk: are fewer, longer reads actually cheaper?"""

    def test_selftest(self):
        proc = _run("expert_io_replay.py")
        self.assertEqual(proc.returncode, 0,
                         f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("selftest: ok", proc.stdout)

    def test_plan_matches_the_counted_reads(self):
        # The benchmark must EXECUTE exactly the reads the measurement COUNTS,
        # or the two report different things under the same name.
        import importlib.util
        import json
        import struct
        import tempfile
        from pathlib import Path

        def load(name):
            spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

        sys.path.insert(0, str(TOOLS))
        try:
            rep, er = load("expert_io_replay"), load("expert_relayout")
        finally:
            sys.path.remove(str(TOOLS))

        with tempfile.TemporaryDirectory() as tmp:
            names, size = [], 2048
            for kind in ("qs", "w"):
                for eid in range(4):
                    names.append(f"model.layers.0.mlp.experts.{eid}.{kind}")
            header, cursor = {}, 0
            for n in names:
                header[n] = {"dtype": "U8", "shape": [size],
                             "data_offsets": [cursor, cursor + size]}
                cursor += size
            blob = json.dumps(header, separators=(",", ":")).encode()
            with open(Path(tmp) / "model-00000.safetensors", "wb") as fh:
                fh.write(struct.pack("<Q", len(blob)))
                fh.write(blob)
                for i, _ in enumerate(names):
                    fh.write(bytes([i % 251]) * size)
            trace = Path(tmp) / "t.txt"
            trace.write_text("".join(f"{c} 0 0 0:0.5 2:0.5\n" for c in range(5)))

            plan = rep.plan(str(trace), tmp)
            counted, counted_bytes = er.count_reads(
                er.routed_sets(str(trace)), er.byte_ranges(tmp))
            self.assertEqual(len(plan), counted)
            self.assertEqual(sum(l for _, _, l in plan), counted_bytes)

            result = rep.bench(str(trace), tmp, cold=False, repeat=1)
            self.assertEqual(result["bytes"], counted_bytes,
                             "benchmark moved different bytes than it planned")

            # Both read implementations must agree. os.pread does not exist on
            # Windows and the seek-then-read fallback only runs there, so
            # without this it ships untested on every other platform.
            saved = rep._PREAD
            try:
                for impl in (rep._pread_seek, saved):
                    rep._PREAD = impl
                    r = rep.bench(str(trace), tmp, cold=False, repeat=1)
                    self.assertEqual(r["bytes"], counted_bytes,
                                     f"{impl.__name__} moved the wrong byte count")
            finally:
                rep._PREAD = saved


class PackSizeCurve(unittest.TestCase):
    """lex-moe#50's gate: does any pack size fix CA's reads without wrecking delta?"""

    def test_selftest(self):
        proc = _run("pack_size_curve.py")
        self.assertEqual(proc.returncode, 0,
                         f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("selftest: ok", proc.stdout)


class AdmissionGate(unittest.TestCase):
    """#37's re-scoped gate: is residency-aware admission worth building?"""

    def test_selftest(self):
        proc = _run("admission_gate.py")
        self.assertEqual(proc.returncode, 0,
                         f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("selftest: ok", proc.stdout)


class ReplayDiff(unittest.TestCase):
    """#15: two runs, and where they first stopped agreeing."""

    def test_selftest(self):
        proc = _run("replay_diff.py")
        self.assertEqual(proc.returncode, 0,
                         f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("selftest: ok", proc.stdout)

    def test_record_lines_stay_out_of_route_trace(self):
        """The record is a SEPARATE stream, and this is why.

        The first design put `# colibri-replay ...` and `T <call> <token>` lines
        into ROUTE_TRACE. Two of the five tools that parse that stream crash on
        the header (int(parts[1]) on 'colibri-replay'), and a `T` line has three
        integer-parseable fields, so a tool that does not also check the line
        length would read it as a routing line and be silently wrong. This pins
        that the parsers are not expected to tolerate either.
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "replay_diff", TOOLS / "replay_diff.py")
        rd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rd)

        path = HERE / "_tmp_replay_record.txt"
        path.write_text(
            "# colibri-replay 1 engine=x layers=2 experts=8 "
            "seed=7 temp=0.700000 top_p=0.900000\n"
            "0 0 0 1:0.5 2:0.5\n"
            "T 0 42\n")
        try:
            header, routes, tokens, n = rd.parse(str(path))
            self.assertEqual(header["seed"], "7")
            self.assertEqual(header["engine"], "x")
            self.assertEqual(n, 1)                       # the T line is not routing
            self.assertEqual(routes[(0, 0, 0)], [1, 2])
            self.assertEqual(tokens[0], 42)
        finally:
            path.unlink(missing_ok=True)


class ReplayRecordCoverage(unittest.TestCase):
    """Every engine that can trace routing must also record a replay (#15).

    The record shipped covering colibri.c alone. Four other engines call
    rt_trace, and on any of them COLI_REPLAY_RECORD silently produced nothing --
    exactly the "a fix lands in one engine and not its siblings" shape
    route_trace.h:373 names as this tree's recurring defect, and the reason #12
    exists. This makes the next omission a test failure instead of a silent one.

    deepseek_v4 is excluded on purpose: it does not include route_trace.h at
    all, so it has no routing telemetry to extend. That is a larger gap, tracked
    separately, not an oversight here.
    """

    ENGINES = ("colibri", "olmoe", "inkling", "kimi_k3")

    def test_every_tracing_engine_records(self):
        src = HERE.parent
        for e in self.ENGINES:
            text = (src / f"{e}.c").read_text(errors="replace")
            self.assertIn("rt_record_header", text,
                          f"{e}.c traces routing but never stamps a replay header")
            self.assertIn("rt_record_token", text,
                          f"{e}.c traces routing but never records a token")

    def test_deepseek_v4_has_no_routing_telemetry_yet(self):
        """Pins the known gap so it is a documented fact, not a surprise."""
        text = (HERE.parent / "deepseek_v4.c").read_text(errors="replace")
        self.assertNotIn("route_trace.h", text)
        self.assertNotIn("rt_trace", text)


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
