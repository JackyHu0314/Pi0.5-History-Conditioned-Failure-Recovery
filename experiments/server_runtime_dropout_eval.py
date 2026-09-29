"""Run the observation-dropout evaluation tree with the installed Blackwell runtime."""

import json
import os
from pathlib import Path
import sys


def main() -> None:
    root = Path(__file__).resolve().parent
    repo = root / "openpi_dropout_eval"
    runtime_root = Path(os.environ["PI05_RUNTIME_ROOT"])
    python = runtime_root / "openpi/.venv/bin/python"
    runtime = runtime_root / "blackwell_runtime"
    manifest = json.loads((runtime / "manifest.json").read_text())
    target = Path(sys.argv[1]).resolve()
    env = os.environ.copy()
    libraries = list(manifest["library_directories"])
    if env.get("LD_LIBRARY_PATH"):
        libraries.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(libraries)
    preload = [
        runtime / name
        for name in (
            "cuda_runtime/lib/libcudart.so.12",
            "cublas/lib/libcublasLt.so.12",
            "cublas/lib/libcublas.so.12",
            "cudnn/lib/libcudnn.so.9",
            "nccl/lib/libnccl.so.2",
        )
    ]
    env["LD_PRELOAD"] = ":".join(map(str, preload))
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        map(
            str,
            (
                root / "sim_deps",
                root / "vendor/LIBERO",
                repo / "src",
                repo / "packages/openpi-client/src",
                repo,
            ),
        )
    )
    env["LIBERO_CONFIG_PATH"] = str(root / "libero_config")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("PYNPUT_BACKEND", "dummy")
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_CUMEM_HOST_ENABLE", "0")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("CUDA_CACHE_MAXSIZE", str(4 * 1024**3))
    env["PYTHONUNBUFFERED"] = "1"
    os.chdir(repo)
    os.execve(str(python), [str(python), "-u", str(target), *sys.argv[2:]], env)


if __name__ == "__main__":
    main()
