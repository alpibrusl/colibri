"""The build must pin floating-point contraction (#76).

`deepseek_v4.c` documents that the rows16 kernel's NEON, AVX2, AVX-512 and
scalar arms share one operation sequence and produce bit-identical rows, with
"No fused multiply-add". The vector arms honour that by construction. The scalar
arm cannot: `s += a[i]*b[i]` is contracted into a single fmadd by default, and
the default is per-compiler -- clang within a statement, gcc across statements,
MSVC not at all.

So the invariant is only true while the flag is set, and a flag is exactly the
kind of thing that disappears in a Makefile edit without anyone noticing,
because nothing fails. It would resurface later as x86 and aarch64 quietly
disagreeing on token 30 of some prompt.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FLAG = "-ffp-contract=off"


class FpContractPinnedTest(unittest.TestCase):
    MAKEFILES = ("Makefile", "Makefile.deepseek-v4")

    def test_every_cflags_definition_pins_contraction(self):
        """Not "the file mentions it" -- every CFLAGS the build can select."""
        for name in self.MAKEFILES:
            text = (ROOT / name).read_text(encoding="utf-8")
            # skip continuation lines and comments; a definition is CFLAGS = / +=
            defs = [l for l in text.splitlines()
                    if re.match(r"^CFLAGS\s*(=|\+=)", l) and "-O3" in l]
            self.assertTrue(defs, f"{name}: no CFLAGS definition with -O3 found — "
                                  "has the Makefile been restructured?")
            for line in defs:
                self.assertIn(FLAG, line,
                              f"{name}: this CFLAGS definition does not pin "
                              f"{FLAG}, so whether the scalar arms fuse is left "
                              f"to the compiler:\n    {line.strip()}")

    def test_the_invariant_comment_names_its_dependency(self):
        """A reader of the promise should find out it has a precondition."""
        text = (ROOT / "deepseek_v4.c").read_text(encoding="utf-8", errors="replace")
        idx = text.find("No fused multiply-add")
        self.assertNotEqual(idx, -1, "the bit-identity comment has moved or gone")
        nearby = text[idx:idx + 900]
        self.assertIn("ffp-contract", nearby,
                      "the 'No fused multiply-add' comment does not mention that "
                      "it depends on -ffp-contract being pinned")


if __name__ == "__main__":
    unittest.main()
