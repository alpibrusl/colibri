"""deepseek_v4.c's routing telemetry is wired at EVERY site, and owned once (#62).

Two failures are specific to this engine and neither shows up as a build error
or a crash, which is why they are worth a test rather than a code review.

1. A partly-hooked router. `coli_v4_route` and `coli_v4_route_bf16` are called
   from three separate MoE paths -- the reference token path, the pipelined
   token path, and the batched prefill union. Hooking two of the three produces
   a trace that is WRONG rather than absent: it is well-formed, every consumer
   parses it, and it silently under-reports routing on whichever path was
   missed. A trace that is merely absent announces itself; this one does not.

2. Two owners of route_trace.h. The header keeps its stream, call counter and
   counters in file-scope statics. deepseek_v4.c is compiled once per
   COLI_V4_UNIT_*, so a second `#include "route_trace.h"` in another unit gives
   that object its own rt_fp and its own rt_call -- two independent call
   sequences written to one path, and token records landing in a different
   FILE* from the routing lines they exist to correlate with.

Both are checked against the source because both survive compilation.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = (ROOT / "deepseek_v4.c").read_text(encoding="utf-8")

# the MoE paths that route, and the call_end each owes at its exit
ROUTING_FUNCTIONS = ("moe_token", "moe_token_pipeline", "v4_moe_batch_union")


def unit_body(name):
    """The text between #ifdef COLI_V4_UNIT_<name> and its #endif."""
    start = SOURCE.index(f"#ifdef COLI_V4_UNIT_{name}\n")
    end = SOURCE.index(f"#endif /* COLI_V4_UNIT_{name} */", start)
    return SOURCE[start:end]


def function_body(name):
    """A function's text, from its definition to the line that closes it."""
    match = re.search(rf"^static int {re.escape(name)}\(", SOURCE, re.M)
    if not match:
        raise AssertionError(f"{name}() not found in deepseek_v4.c")
    end = SOURCE.index("\n}\n", match.start())
    return SOURCE[match.start():end]


class V4RouteTraceWiringTest(unittest.TestCase):
    def test_every_routing_call_site_is_traced(self):
        """Each coli_v4_route[_bf16] call is followed by a trace hook.

        Scanned over the whole file rather than a hand-listed set of line
        numbers, so a NEW routing path added later fails this test instead of
        quietly becoming the one the trace does not cover.
        """
        for function in ROUTING_FUNCTIONS:
            body = function_body(function)
            calls = len(re.findall(r"\bcoli_v4_route(?:_bf16)?\s*\(", body))
            self.assertGreater(calls, 0,
                               f"{function}() no longer routes -- update this test")
            self.assertIn("coli_v4_rt_route(", body,
                          f"{function}() routes but never calls coli_v4_rt_route(): "
                          "its selections would be missing from every capture")

    def test_every_routing_function_ends_its_call(self):
        """rt_call must advance once per moe() invocation, as in the other engines.

        Miss one and that path's rows merge into the previous call's id, so the
        <call> field stops counting what colibri.c and olmoe.c count and the
        shared tools segment a V4 trace wrongly.
        """
        for function in ROUTING_FUNCTIONS:
            self.assertIn("coli_v4_rt_call_end()", function_body(function),
                          f"{function}() does not advance the call counter")

    def test_batch_path_traces_the_row_not_a_constant(self):
        """The prefill union routes `batch` rows; each needs its own row index.

        Passing a constant would collapse every position in a prefill batch onto
        row 0, which parses cleanly and loses the batch.
        """
        body = function_body("v4_moe_batch_union")
        self.assertRegex(
            body, r"coli_v4_rt_route\(\s*weights->plan\.layer,\s*item\b",
            "the batched path must trace `item` as its row index")

    def test_route_trace_header_has_exactly_one_owner(self):
        """Only COLI_V4_UNIT_ROUTE_TRACE may include route_trace.h."""
        includes = SOURCE.count('#include "route_trace.h"')
        self.assertEqual(includes, 1,
                         f'route_trace.h is included {includes} times in '
                         'deepseek_v4.c; its statics make a second inclusion a '
                         'second, independent trace')
        self.assertIn('#include "route_trace.h"', unit_body("ROUTE_TRACE"),
                      "the one inclusion is not inside COLI_V4_UNIT_ROUTE_TRACE")

    def test_owner_unit_is_built_and_linked(self):
        """An owner that is not in V4_TARGET_UNITS is a link error, not a silent
        loss -- but the check is cheap and names the fix."""
        units = (ROOT / "Makefile.deepseek-v4.units").read_text(encoding="utf-8")
        self.assertIn("COLI_V4_UNIT_ROUTE_TRACE", units,
                      "the trace owner unit is not in V4_TARGET_UNITS")

    def test_engine_identifies_itself_to_the_format(self):
        """rt_engine_of() must name deepseek_v4 rather than report a raw hash."""
        header = (ROOT / "route_trace.h").read_text(encoding="utf-8")
        names = re.search(r"rt_engine_names\[\]\s*=\s*\{(.*?)\}", header, re.S)
        self.assertIsNotNone(names, "rt_engine_names[] not found")
        self.assertIn('"deepseek_v4"', names.group(1))

    def test_tokens_are_recorded_where_they_are_committed(self):
        """Only committed tokens belong in the replay record.

        V4 drafts speculatively; recording at a draft site would put tokens in
        the record that verification rejected and the run never emitted.
        session_emit_token() is the single commit point.
        """
        body = function_body("session_emit_token")
        self.assertIn("coli_v4_rt_token(", body,
                      "session_emit_token() does not record the token it commits")
        for drafter in ("v4_ngram_draft", "v4_dspark_draft"):
            match = re.search(rf"^static int {drafter}\(", SOURCE, re.M)
            if not match:
                continue
            self.assertNotIn("coli_v4_rt_token(", function_body(drafter),
                             f"{drafter}() records tokens that may be rejected")


if __name__ == "__main__":
    unittest.main()
