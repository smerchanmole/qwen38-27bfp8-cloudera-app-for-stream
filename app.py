#!/usr/bin/env python3
"""Cloudera AI Workbench Application serving Qwen3.8 through OpenAI + SSE.

Run this file directly as the Cloudera Application script.  The bootstrap
process deliberately imports only Python's standard library.  It creates an
isolated virtual environment, installs the CUDA-specific serving stack there,
validates it, and finally replaces itself with vLLM's OpenAI API server.

Cloudera Applications must listen on 127.0.0.1 and the port supplied through
CDSW_APP_PORT.  Both values are enforced rather than exposed as options.
"""

from __future__ import annotations

import fcntl
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import venv
from pathlib import Path
from typing import Any


# Cloudera can execute the Application script as cells in a Jupyter kernel,
# where __file__ does not exist. In that case the project is the working dir.
RUNNING_IN_NOTEBOOK = "__file__" not in globals()
APP_DIR = Path.cwd().resolve() if RUNNING_IN_NOTEBOOK else Path(__file__).resolve().parent
PYTHON_VERSION = f"{sys.version_info.major}{sys.version_info.minor}"
VLLM_VERSION = os.getenv("VLLM_VERSION", "0.29.0")
CUDA_VARIANT = os.getenv("VLLM_CUDA_VARIANT", "129")
TRANSFORMERS_VERSION = os.getenv("TRANSFORMERS_VERSION", "5.15.0")


def log(message: str) -> None:
    """Write immediately to the Cloudera Application log."""
    print(f"[qwen-app] {message}", flush=True)


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "true" if default else "false").strip().lower()
    if raw not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
        raise ValueError(f"{name} must be a boolean value")
    return raw in {"1", "true", "yes", "on"}


def env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    value = int(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def required_app_port() -> int:
    """Read the only port on which a Cloudera Application may listen."""
    raw = os.getenv("CDSW_APP_PORT")
    if not raw:
        raise RuntimeError(
            "CDSW_APP_PORT is not defined. Run app.py as a Cloudera AI "
            "Application, not as a Workbench Model Deployment."
        )
    try:
        port = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid CDSW_APP_PORT={raw!r}") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError(f"CDSW_APP_PORT is outside the valid range: {port}")
    return port


def architecture() -> str:
    machine = platform.machine().lower()
    aliases = {"amd64": "x86_64", "arm64": "aarch64"}
    machine = aliases.get(machine, machine)
    if machine not in {"x86_64", "aarch64"}:
        raise RuntimeError(f"Unsupported vLLM wheel architecture: {machine}")
    return machine


def install_environment() -> tuple[Path, dict[str, str]]:
    """Create and validate the isolated serving environment.

    A file lock protects a persistent project volume from two simultaneous
    starts.  The stamp includes every dependency selector that affects the
    environment.  The environment is upgraded in place only when that stamp
    changes or QWEN_FORCE_REINSTALL=true is requested.
    """
    arch = architecture()
    venv_dir = APP_DIR / (
        f".venv-qwen38-py{PYTHON_VERSION}-vllm{VLLM_VERSION}-cu{CUDA_VARIANT}"
    )
    venv_python = venv_dir / "bin" / "python"
    lock_path = APP_DIR / ".qwen-venv-install.lock"
    stamp_path = venv_dir / ".qwen-install.json"

    default_wheel = (
        "https://github.com/vllm-project/vllm/releases/download/"
        f"v{VLLM_VERSION}/vllm-{VLLM_VERSION}%2Bcu{CUDA_VARIANT}-"
        f"cp38-abi3-manylinux_2_28_{arch}.whl"
    )
    wheel_url = os.getenv("VLLM_WHEEL_URL", default_wheel)
    pytorch_index = os.getenv(
        "PYTORCH_INDEX_URL",
        f"https://download.pytorch.org/whl/cu{CUDA_VARIANT}",
    )
    expected: dict[str, Any] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "architecture": arch,
        "vllm": VLLM_VERSION,
        "cuda_variant": CUDA_VARIANT,
        "transformers": TRANSFORMERS_VERSION,
        "wheel_url": wheel_url,
        "pytorch_index": pytorch_index,
    }

    install_env = os.environ.copy()
    install_env["PIP_USER"] = "false"
    for name in ("PIP_PREFIX", "PIP_TARGET", "PYTHONUSERBASE"):
        install_env.pop(name, None)

    with lock_path.open("a+") as lock_file:
        log(f"Waiting for dependency lock {lock_path.name}")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        if not venv_python.is_file():
            log(f"Creating isolated virtual environment: {venv_dir.name}")
            venv.EnvBuilder(with_pip=True, clear=False, symlinks=True).create(venv_dir)

        current: dict[str, Any] | None = None
        if stamp_path.is_file():
            try:
                current = json.loads(stamp_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                current = None

        reinstall = env_bool("QWEN_FORCE_REINSTALL", False)
        if current != expected or reinstall:
            reason = "forced" if reinstall else "missing or changed dependency stamp"
            log(f"Installing dependencies ({reason})")
            commands = [
                [
                    str(venv_python),
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "pip>=24.2,<26.0",
                    "packaging>=24.0,<26.0",
                ],
                [
                    str(venv_python),
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "--upgrade",
                    wheel_url,
                    "--extra-index-url",
                    pytorch_index,
                ],
                [
                    str(venv_python),
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "--upgrade",
                    f"transformers=={TRANSFORMERS_VERSION}",
                ],
                [str(venv_python), "-m", "pip", "check"],
            ]
            for command in commands:
                log("Running: " + shlex.join(command))
                subprocess.run(command, env=install_env, cwd=APP_DIR, check=True)

            verify_code = f"""
import torch
import transformers
import vllm
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

assert vllm.__version__ == {VLLM_VERSION!r} or vllm.__version__.startswith({(VLLM_VERSION + '+')!r}), vllm.__version__
assert transformers.__version__ == {TRANSFORMERS_VERSION!r}, transformers.__version__
assert (torch.version.cuda or '').startswith({(CUDA_VARIANT[:-1] + '.' + CUDA_VARIANT[-1])!r}), torch.version.cuda
print('vLLM', vllm.__version__)
print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda)
print('Transformers', transformers.__version__)
print('Architecture', Qwen3_5ForConditionalGeneration.__name__)
"""
            subprocess.run(
                [str(venv_python), "-c", verify_code],
                env=install_env,
                cwd=APP_DIR,
                check=True,
            )
            stamp_path.write_text(
                json.dumps(expected, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            log("Dependency installation and validation completed")
        else:
            log(f"Reusing validated environment: {venv_dir.name}")

    server_env = os.environ.copy()
    server_env["PATH"] = f"{venv_dir / 'bin'}:{server_env.get('PATH', '')}"
    server_env["VIRTUAL_ENV"] = str(venv_dir)
    server_env["PYTHONUNBUFFERED"] = "1"
    server_env.setdefault("TOKENIZERS_PARALLELISM", "false")
    server_env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # The validated Cloudera Runtime had neither ninja nor a complete visible
    # NVCC toolchain.  Disable only FlashInfer sampling JIT; Triton attention
    # and Marlin FP8 weight kernels remain enabled.
    server_env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    # Prevent an allowed public URL from redirecting the media loader to an
    # internal address and bypassing VLLM_ALLOWED_MEDIA_DOMAINS.
    server_env.setdefault("VLLM_MEDIA_URL_ALLOW_REDIRECTS", "0")
    return venv_python, server_env


def server_command(venv_python: Path, port: int) -> list[str]:
    """Translate safe environment variables into vLLM OpenAI CLI options."""
    model_id = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen3.8-27B-FP8")
    served_name = os.getenv("QWEN_SERVED_MODEL_NAME", "qwen3.8-27b-fp8")
    max_model_len = env_int("QWEN_MAX_MODEL_LEN", 262144, 2048, 262144)
    max_num_seqs = env_int("QWEN_MAX_NUM_SEQS", 1, 1, 64)
    max_batched = env_int(
        "QWEN_MAX_NUM_BATCHED_TOKENS", 8192, 2048, 131072
    )
    gpu_util = env_float("QWEN_GPU_MEMORY_UTILIZATION", 0.90, 0.50, 0.95)
    max_images = env_int("QWEN_MAX_IMAGES_PER_PROMPT", 1, 0, 4)
    kv_dtype = os.getenv("QWEN_KV_CACHE_DTYPE", "bfloat16")
    attention_backend = os.getenv("QWEN_ATTENTION_BACKEND", "TRITON_ATTN")
    enforce_eager = env_bool("QWEN_ENFORCE_EAGER", True)

    # The OpenAI server is intentionally bound only to the loopback address
    # and the Cloudera-provided port.  Never replace this with 0.0.0.0.
    command = [
        str(venv_python),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--model",
        model_id,
        "--served-model-name",
        served_name,
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "auto",
        "--kv-cache-dtype",
        kv_dtype,
        "--attention-backend",
        attention_backend,
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_util),
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-num-batched-tokens",
        str(max_batched),
        "--enable-chunked-prefill",
        "--enable-prefix-caching",
        "--limit-mm-per-prompt",
        json.dumps({"image": max_images, "video": 0}, separators=(",", ":")),
    ]
    if enforce_eager:
        command.append("--enforce-eager")

    cpu_offload = env_float("QWEN_CPU_OFFLOAD_GB", 0.0, 0.0, 128.0)
    if cpu_offload:
        command.extend(["--cpu-offload-gb", str(cpu_offload)])

    api_key = os.getenv("QWEN_API_KEY")
    if api_key:
        command.extend(["--api-key", api_key])

    reasoning_parser = os.getenv("QWEN_REASONING_PARSER", "").strip()
    if reasoning_parser:
        command.extend(["--reasoning-parser", reasoning_parser])

    tool_parser = os.getenv("QWEN_TOOL_CALL_PARSER", "").strip()
    if tool_parser:
        command.extend(["--enable-auto-tool-choice", "--tool-call-parser", tool_parser])

    domains = [
        item.strip()
        for item in os.getenv("VLLM_ALLOWED_MEDIA_DOMAINS", "").split(",")
        if item.strip()
    ]
    if domains:
        # Keep this final because argparse consumes one or more domain values.
        command.extend(["--allowed-media-domains", *domains])

    log(
        "Server configuration: "
        f"model={model_id} served_name={served_name} context={max_model_len} "
        f"max_num_seqs={max_num_seqs} prefill={max_batched} "
        f"gpu_utilization={gpu_util} kv_cache={kv_dtype} "
        f"attention={attention_backend} eager={enforce_eager} images={max_images}"
    )
    return command


def main() -> None:
    started = time.monotonic()
    port = required_app_port()
    if not (3, 10) <= sys.version_info[:2] < (3, 14):
        raise RuntimeError("Python 3.10 through 3.13 is required")

    log(
        f"Bootstrapping on Python {platform.python_version()}, "
        f"architecture={architecture()}, CDSW_APP_PORT={port}"
    )
    venv_python, server_env = install_environment()
    command = server_command(venv_python, port)
    log(f"Bootstrap completed in {time.monotonic() - started:.1f}s")
    log("Starting the OpenAI-compatible vLLM server on 127.0.0.1 only")

    if RUNNING_IN_NOTEBOOK:
        # Replacing the kernel would disconnect the notebook runner that owns
        # this Application. Keep its cell active while vLLM serves requests.
        subprocess.run(command, env=server_env, cwd=APP_DIR, check=True)
    else:
        # In a regular script, let vLLM receive signals and exit codes directly.
        os.chdir(APP_DIR)
        os.execve(str(venv_python), command, server_env)


if __name__ == "__main__":
    main()
