"""Run same-condition repeats for the completed controlled-slip shards."""

from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import run_controlled_slip_matrix as matrix


def run_repeat(config_name: str, shard_index: int, gpu: int) -> dict:
    config = matrix.CONFIGS[config_name]
    checkpoint, manifest, schedule = matrix.paths(config)
    pair_dir = matrix.WORK / config_name / f"shard{shard_index}"
    output_dir = pair_dir / "repeat-control"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    command = [
        str(matrix.PYTHON),
        str(matrix.RUNTIME),
        str(matrix.EVALUATOR),
        "--variant", config["variant"],
        "--checkpoint-dir", str(checkpoint),
        "--output", str(output_dir / "result.json"),
        "--run-manifest", str(manifest),
        "--task-ids", str(matrix.TASK_ID),
        "--episode-indices", *map(str, matrix.SHARDS[shard_index]),
        "--seed", "7",
        "--observation-intervention", config["observation"],
        "--execution-intervention", "control",
        "--intervention-schedule", str(schedule),
        "--blackout-queries", str(matrix.BLACKOUT_QUERIES),
        "--trace-dir", str(output_dir / "traces"),
        "--replay-prefix-trace-dir", str(pair_dir / "control/traces"),
    ]
    started = time.time()
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1")
    with (output_dir / "run.log").open("w", encoding="utf-8", buffering=1) as handle:
        process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=environment)
    return {
        "config": config_name,
        "shard": shard_index,
        "gpu": gpu,
        "returncode": process.returncode,
        "seconds": time.time() - started,
        "command": command,
    }


def main() -> None:
    available = []
    for gpu, (config_name, shard_index) in enumerate(
        (config_name, shard_index)
        for config_name in matrix.CONFIGS
        for shard_index in range(len(matrix.SHARDS))
    ):
        pair_dir = matrix.WORK / config_name / f"shard{shard_index}"
        if (pair_dir / "control/result.json").is_file() and not (
            pair_dir / "repeat-control/result.json"
        ).is_file():
            available.append((config_name, shard_index, gpu))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(available) or 1) as executor:
        runs = list(executor.map(lambda item: run_repeat(*item), available))
    root = matrix.CAMPAIGN / "control-repeat-calibration"
    root.mkdir(exist_ok=True)
    output = root / f"runs-{int(time.time())}.json"
    output.write_text(json.dumps({"runs": runs}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": runs}, ensure_ascii=False, indent=2))
    if any(run["returncode"] for run in runs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
