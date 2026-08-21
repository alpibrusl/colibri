"""A kernel dispatcher must not call itself, and must reach every arm it claims.

This exists because of a real, merged-branch failure. `matmul_mxfp4_dispatch`
was written to send x86 to the AVX2 kernel and everyone else to a new one:

    #if defined(__AVX2__)
        matmul_mxfp4(...);          <- the original, which contains the AVX2 arm
    #else
        matmul_mxfp4_rows4(...);
    #endif

A later edit meant to route *other* callers through the dispatcher matched this
line too, turning the AVX2 arm into `matmul_mxfp4_dispatch(...)` — unbounded
self-recursion. On aarch64 the `#else` arm compiles and the recursion is
unreachable, so every local test passed and the whole suite was green. Only x86
CI hit it, as a 180-second timeout in the V4 oracle rather than as an error that
named itself.

Two things make it worth a test rather than care:
  * the failure is invisible on the developer's own architecture, and
  * `-Winfinite-recursion` does not fire on it (verified: clang targeting
    x86-64-v3 compiles the recursive version silently).
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def function_bodies(text):
    """-> {name: body} for every static function definition in the file."""
    out = {}
    for m in re.finditer(r"^static\s+[\w \*]+?\b(\w+)\s*\([^;]*?\)\s*\{", text, re.M):
        name, start = m.group(1), m.end()
        depth, i = 1, start
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        out[name] = text[start:i]
    return out


class KernelDispatchTest(unittest.TestCase):
    SOURCES = ("quant.h", "deepseek_v4.c")

    def test_no_dispatcher_calls_itself(self):
        for src in self.SOURCES:
            text = (ROOT / src).read_text(encoding="utf-8", errors="replace")
            for name, body in function_bodies(text).items():
                if not name.endswith("_dispatch"):
                    continue
                self.assertNotRegex(
                    body, rf"\b{re.escape(name)}\s*\(",
                    f"{src}: {name}() calls itself — unbounded recursion on "
                    "whichever branch that arm belongs to. This is invisible on "
                    "an architecture that compiles the other arm.")

    def test_mxfp4_dispatch_reaches_both_arms(self):
        """Every arm must name a DIFFERENT function, or one target is dead."""
        text = (ROOT / "quant.h").read_text(encoding="utf-8")
        body = function_bodies(text)["matmul_mxfp4_dispatch"]
        called = set(re.findall(r"\b(matmul_mxfp4\w*)\s*\(", body))
        self.assertEqual(
            len(called), 2,
            f"matmul_mxfp4_dispatch should call exactly two distinct kernels, "
            f"one per arm; it calls {sorted(called) or 'none'}")
        self.assertNotIn("matmul_mxfp4_dispatch", called)

    def test_no_stale_callers_of_the_renamed_kernel(self):
        """`matmul_mxfp4(` no longer exists; a caller using it must not compile.

        The original was renamed to matmul_mxfp4_scalar_or_avx2 precisely so a
        stale or accidentally-rewritten call is a compile error rather than a
        silent slow path or a silent loop.
        """
        for src in self.SOURCES:
            text = (ROOT / src).read_text(encoding="utf-8", errors="replace")
            self.assertNotRegex(
                text, r"(?<!coli_cuda_)\bmatmul_mxfp4\s*\(",
                f"{src}: calls matmul_mxfp4() — that name was retired; use "
                "matmul_mxfp4_dispatch()")


if __name__ == "__main__":
    unittest.main()
