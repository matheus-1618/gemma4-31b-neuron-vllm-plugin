# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/03_serve.sh"


class ServeSafetyTest(unittest.TestCase):
    def test_mns_above_one_fails_closed(self):
        env = os.environ | {"MNS": "4", "TP": "4", "ALLOW_UNSAFE_MNS": "0"}
        result = subprocess.run(["bash", str(SCRIPT)], env=env, text=True,
                                capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 2)
        self.assertIn("MNS>1 is blocked", result.stderr)

    def test_default_is_single_request_server(self):
        self.assertIn('MNS="${MNS:-1}"', SCRIPT.read_text())


if __name__ == "__main__":
    unittest.main()
