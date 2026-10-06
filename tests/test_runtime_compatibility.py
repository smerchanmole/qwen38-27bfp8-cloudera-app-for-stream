"""Regression checks for Cloudera H100 startup with an older NVCC toolchain."""

import contextlib
import io
import os
from pathlib import Path
import runpy
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class RuntimeCompatibilityTests(unittest.TestCase):
    def test_both_profiles_avoid_nvcc_jit_and_isolate_python(self):
        for filename in ("app.py", "apph100.py"):
            with self.subTest(script=filename):
                app = runpy.run_path(str(ROOT / filename), run_name="runtime_test")
                with patch.dict(os.environ, {
                    "PATH": "/usr/bin", "PYTHONPATH": "/runtime/packages",
                    "PYTHONHOME": "/runtime/python", "CDSW_APP_PORT": "8100",
                }, clear=True), contextlib.redirect_stdout(io.StringIO()):
                    env = app["serving_environment"](Path("/project/venv"))
                    command = app["server_command"](Path("/project/venv/bin/python"), 8100)
                self.assertEqual(env["VLLM_USE_DEEP_GEMM"], "0")
                self.assertEqual(env["VLLM_USE_FLASHINFER_SAMPLER"], "0")
                self.assertNotIn("PYTHONPATH", env)
                self.assertNotIn("PYTHONHOME", env)
                self.assertEqual(env["CDSW_APP_PORT"], "8100")
                self.assertEqual(command[command.index("--gdn-prefill-backend") + 1], "triton")

    def test_compatible_toolchain_can_explicitly_opt_in(self):
        for filename in ("app.py", "apph100.py"):
            with self.subTest(script=filename):
                app = runpy.run_path(str(ROOT / filename), run_name="runtime_test")
                with patch.dict(os.environ, {
                    "VLLM_USE_DEEP_GEMM": "1", "QWEN_GDN_PREFILL_BACKEND": "flashinfer",
                }, clear=True), contextlib.redirect_stdout(io.StringIO()):
                    env = app["serving_environment"](Path("/project/venv"))
                    command = app["server_command"](Path("/project/venv/bin/python"), 8100)
                self.assertEqual(env["VLLM_USE_DEEP_GEMM"], "1")
                self.assertEqual(command[command.index("--gdn-prefill-backend") + 1], "flashinfer")


if __name__ == "__main__":
    unittest.main()
