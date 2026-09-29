"""Copy this project's OpenPI overlay into a pinned upstream checkout."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


UPSTREAM_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"


def copy_tree(source: Path, target: Path) -> int:
    copied = 0
    for path in source.rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi", type=Path, required=True)
    parser.add_argument("--mode", choices=("training", "evaluation"), required=True)
    args = parser.parse_args()

    checkout = args.openpi.resolve()
    actual_commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != UPSTREAM_COMMIT:
        raise SystemExit(f"expected OpenPI {UPSTREAM_COMMIT}, found {actual_commit}")

    overlays = Path(__file__).resolve().parents[1] / "openpi_overlays"
    copied = copy_tree(overlays / "training", checkout)
    if args.mode == "evaluation":
        copied += copy_tree(overlays / "evaluation_delta", checkout)
    print(f"applied {copied} files to {checkout} ({args.mode})")


if __name__ == "__main__":
    main()
