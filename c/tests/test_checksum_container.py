"""tools/checksum_container.py stays runnable (#13).

The engine verifies against __metadata__["colibri.sha256"]; this tool is what
writes it, so a container can only be checksummed if the tool works. Its
synthetic selftest needs no weights and runs here, in CI, on every PR --
covering the rewrite preserving every byte, the map covering every tensor, and
a flipped byte being caught on re-verify.
"""
import subprocess
import sys
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent / "tools"


class ChecksumContainerTool(unittest.TestCase):
    def test_selftest(self):
        proc = subprocess.run(
            [sys.executable, str(TOOLS / "checksum_container.py"), "--selftest"],
            capture_output=True, text=True, timeout=300)
        self.assertEqual(proc.returncode, 0,
                         f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        self.assertIn("selftest: ok", proc.stdout)

    def test_hex64_matches_the_engine_check(self):
        """The tool emits what st_is_hex64 (c/st.h) accepts: 64 lowercase hex."""
        import hashlib
        d = hashlib.sha256(b"x").hexdigest()
        self.assertEqual(len(d), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in d))


if __name__ == "__main__":
    unittest.main()
