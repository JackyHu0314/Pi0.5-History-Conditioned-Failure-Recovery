#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


SEEDS = (21, 22, 23)
OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
DATASET_REVISION = "551d7d86f25edd0ffeda8b60053c15438cfd1d6a"


def link_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)


def sha256(path: Path, cache: dict[tuple[int, int], str]) -> str:
    stat = path.stat()
    inode = (stat.st_dev, stat.st_ino)
    if inode not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        cache[inode] = digest.hexdigest()
    return cache[inode]


def portable_config(seed: int) -> dict:
    return {
        "schema_version": "pi05-history-recovery-hf-v1",
        "checkpoint": {
            "name": f"RD-low-s{seed}",
            "seed": seed,
            "step": 1499,
            "completed_updates": 1500,
            "format": "orbax-zarr",
        },
        "source": {
            "openpi_commit": OPENPI_COMMIT,
            "code_repository": "https://github.com/JackyHu0314/Pi0.5-History-Conditioned-Failure-Recovery",
            "dataset": "lerobot/libero_10",
            "dataset_revision": DATASET_REVISION,
        },
        "model": {
            "variant": "H5",
            "total_parameters": 3_359_000_000,
            "trainable_parameters": 436_000_000,
            "dtype": "bfloat16",
            "action_horizon": 10,
            "action_dim": 32,
            "libero_action_dim": 7,
            "max_text_tokens": 200,
            "history_injection": "prefix",
            "history_visual_source": "past",
            "history_selection": "early2_recent6",
            "history_max_records": 8,
            "history_queries_per_record": 2,
            "history_output_tokens": 16,
            "history_width": 512,
            "history_heads": 8,
            "history_mlp_hidden": 2048,
            "freeze_vision": True,
            "train_scope": "history_and_action_expert",
        },
        "training": {
            "steps": 1500,
            "batch_size": 8,
            "fsdp_devices": 8,
            "warmup_steps": 100,
            "base_peak_lr": 3e-6,
            "base_decay_lr": 3e-7,
            "history_peak_lr": 3e-5,
            "history_decay_lr": 3e-6,
            "current_image_dropout_probability": 0.25,
            "current_image_dropout_seed": 2026092900 + seed,
            "recovery_pairs": 26,
            "recovery_windows": 52,
            "recovery_repeat": 20,
            "recovery_episodes": 1040,
            "mixed_training_samples": 67293,
        },
        "frozen_evaluation": {
            "libero_task": 5,
            "initial_state_indices": list(range(10)),
            "control_success": {21: 0.7, 22: 0.7, 23: 0.9}[seed],
            "slip_success": {21: 0.8, 22: 0.8, 23: 0.9}[seed],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True)
    hf_files = args.repository / "huggingface"
    for name in (
        "README.md",
        "NOTICE",
        "MODEL_LICENSE.md",
        "LICENSE",
        "GEMMA_TERMS_OF_USE.txt",
        "GEMMA_PROHIBITED_USE_POLICY.txt",
    ):
        shutil.copy2(hf_files / name, args.output / name)
    shutil.copy2(args.repository / "third_party/openpi-APACHE-2.0.txt", args.output / "LICENSE_APACHE-2.0.txt")

    checkpoint_root = args.campaign / "checkpoints/pi05_libero_history_h5"
    manifest = {
        "schema_version": "pi05-history-recovery-release-v1",
        "openpi_commit": OPENPI_COMMIT,
        "dataset_revision": DATASET_REVISION,
        "code_repository": "https://github.com/JackyHu0314/Pi0.5-History-Conditioned-Failure-Recovery",
        "checkpoints": [],
    }
    for seed in SEEDS:
        name = f"rd-low-s{seed}"
        source = checkpoint_root / f"RD-low-s{seed}/1499"
        target = args.output / "checkpoints" / name
        link_tree(source / "params", target / "params")
        link_tree(source / "assets", target / "assets")
        config = portable_config(seed)
        config_path = args.output / "configs" / f"{name}.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        files = [path for path in target.rglob("*") if path.is_file()]
        manifest["checkpoints"].append({
            "name": name,
            "seed": seed,
            "step": 1499,
            "file_count": len(files),
            "logical_bytes": sum(path.stat().st_size for path in files),
            "control_success": config["frozen_evaluation"]["control_success"],
            "slip_success": config["frozen_evaluation"]["slip_success"],
        })

    results = args.repository / "results/recovery_distill_final"
    shutil.copytree(results, args.output / "results")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    cache: dict[tuple[int, int], str] = {}
    checksum_lines = []
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksum_lines.append(f"{sha256(path, cache)}  {path.relative_to(args.output)}")
    (args.output / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")


if __name__ == "__main__":
    main()
