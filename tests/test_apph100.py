"""CPU checks for the H100 launch contract; GPU inference is tested in Cloudera."""

import contextlib
import io
import os
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "apph100.py"
app = runpy.run_path(str(SCRIPT), run_name="test_apph100_module")


class H100LaunchTests(unittest.TestCase):
    def test_optimized_command_keeps_cloudera_routing_and_media(self):
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()):
            command = app["server_command"](Path("/venv/bin/python"), 8080)
        for flag, expected in (
            ("--host", "127.0.0.1"), ("--port", "8080"),
            ("--tensor-parallel-size", "1"), ("--attention-backend", "FLASH_ATTN"),
            ("--optimization-level", "3"), ("--performance-mode", "throughput"),
        ):
            self.assertEqual(command[command.index(flag) + 1], expected)
        self.assertNotIn("--enforce-eager", command)
        self.assertIn("--async-scheduling", command)
        self.assertIn("qwen_home.HomePageMiddleware", command)
        self.assertIn('{"image":1,"video":0}', command)

    def test_diagnostic_overrides_and_media_domains_remain_valid(self):
        env = {
            "QWEN_ENFORCE_EAGER": "true", "QWEN_ASYNC_SCHEDULING": "false",
            "QWEN_PERFORMANCE_MODE": "interactivity", "QWEN_MAX_NUM_SEQS": "4",
            "VLLM_ALLOWED_MEDIA_DOMAINS": "example.com,images.example.com",
        }
        with patch.dict(os.environ, env, clear=True), contextlib.redirect_stdout(io.StringIO()):
            command = app["server_command"](Path("/venv/bin/python"), 9000)
        self.assertIn("--enforce-eager", command)
        self.assertNotIn("--optimization-level", command)
        self.assertIn("--no-async-scheduling", command)
        self.assertEqual(command[-3:], ["--allowed-media-domains", "example.com", "images.example.com"])

    def test_gpu_guard_accepts_h100_and_rejects_wrong_allocations(self):
        def run_check(cuda_available=True, count=1, name="NVIDIA H100 80GB HBM3", major=9, memory_gib=80):
            gpu = SimpleNamespace(name=name, major=major, minor=0, total_memory=memory_gib * 1024**3)
            cuda = SimpleNamespace(
                is_available=lambda: cuda_available, device_count=lambda: count,
                get_device_properties=lambda _: gpu,
                mem_get_info=lambda _: (gpu.total_memory, gpu.total_memory),
            )

            def child_run(command, **kwargs):
                self.assertEqual(command[:3], ["/venv/bin/python", "-I", "-c"])
                self.assertTrue(kwargs["check"])
                with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}):
                    exec(command[3], {})

            with patch("subprocess.run", side_effect=child_run), contextlib.redirect_stdout(io.StringIO()):
                app["validate_h100"](Path("/venv/bin/python"), {})

        run_check()
        for kwargs in (
            {"cuda_available": False}, {"count": 2},
            {"name": "NVIDIA A100", "major": 8}, {"memory_gib": 20},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                run_check(**kwargs)

    def test_notebook_keeps_kernel_alive_and_validates_gpu_each_start(self):
        namespace = {"__name__": "notebook_test"}
        exec(compile(SCRIPT.read_text(), str(SCRIPT), "exec"), namespace)
        self.assertTrue(namespace["RUNNING_IN_NOTEBOOK"])
        self.assertEqual(namespace["APP_DIR"], Path.cwd().resolve())
        validate_calls = []
        namespace["install_environment"] = lambda: (Path("/venv/bin/python"), {})
        namespace["validate_h100"] = lambda *args: validate_calls.append(args)
        with patch.dict(os.environ, {"CDSW_APP_PORT": "8080"}, clear=True), \
                patch("subprocess.run") as child, patch("os.execve") as replace, \
                contextlib.redirect_stdout(io.StringIO()):
            namespace["main"]()
        self.assertEqual(len(validate_calls), 1)
        child.assert_called_once()
        replace.assert_not_called()


if __name__ == "__main__":
    unittest.main()
