"""Collect paired successful B2-LVF recovery windows on all eight GPUs."""

from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parent
SCREEN = ROOT / "observation_dropout_training_20260928"
WORK = ROOT / "recovery_distill_20260929"
PYTHON = Path(os.environ.get("PI05_PYTHON", sys.executable))
RUNTIME = ROOT / "server_runtime_dropout_eval.py"
EVALUATOR = ROOT / "openpi_dropout_eval/scripts/eval_libero_controlled_slip.py"
CHECKPOINT = SCREEN / "checkpoints/pi05_libero_history_h0/B2-drop25-u30000-seed11"
MANIFEST = CHECKPOINT.with_suffix(".run.json")
SCHEDULE = SCREEN / "development/B2-drop25-u30000-seed11/clean/result.json"
LOCK = threading.Lock()
STATUS: dict = {"status": "RUNNING", "jobs": {}}


def write_status() -> None:
    path = WORK / "collection_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(STATUS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def update(episode: int, **values) -> None:
    with LOCK:
        STATUS["jobs"].setdefault(str(episode), {}).update(values)
        write_status()


def run_branch(episode: int, branch: str, gpu: int, replay: Path | None) -> dict:
    branch_dir = WORK / "teacher_rollouts" / f"episode{episode:02d}" / branch
    branch_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        str(RUNTIME),
        str(EVALUATOR),
        "--variant", "H0",
        "--checkpoint-dir", str(CHECKPOINT),
        "--output", str(branch_dir / "result.json"),
        "--run-manifest", str(MANIFEST),
        "--task-ids", "5",
        "--episode-indices", str(episode),
        "--seed", "7",
        "--observation-intervention", "scheduled_dual_camera_last_valid",
        "--execution-intervention", branch,
        "--intervention-schedule", str(SCHEDULE),
        "--blackout-queries", "3",
        "--trace-dir", str(branch_dir / "traces"),
        "--recovery-dataset-dir", str(WORK / "teacher_windows"),
        "--recovery-pre-steps", "8",
        "--recovery-post-steps", "25",
        "--recovery-encode-batch-size", "16",
    ]
    if replay is not None:
        command.extend(["--replay-prefix-trace-dir", str(replay)])
    started = time.time()
    with (branch_dir / "run.log").open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1"),
        )
    return {
        "returncode": process.returncode,
        "seconds": time.time() - started,
        "result": str(branch_dir / "result.json"),
        "log": str(branch_dir / "run.log"),
    }


def run_pair(episode: int, gpu: int) -> None:
    update(episode, status="control_running", gpu=gpu, started_unix=time.time())
    control = run_branch(episode, "control", gpu, None)
    update(episode, control=control)
    if control["returncode"]:
        update(episode, status="failed_control", finished_unix=time.time())
        return
    trace_dir = WORK / "teacher_rollouts" / f"episode{episode:02d}" / "control/traces"
    update(episode, status="slip_running")
    slip = run_branch(episode, "slip", gpu, trace_dir)
    update(episode, slip=slip)
    status = "completed" if not slip["returncode"] else "failed_slip"
    update(episode, status=status, finished_unix=time.time())


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    STATUS["started_unix"] = time.time()
    write_status()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_pair, episode, episode % 8) for episode in range(10)]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    failed = [key for key, value in STATUS["jobs"].items() if value["status"] != "completed"]
    STATUS["status"] = "COMPLETED" if not failed else "FAILED"
    STATUS["failed_jobs"] = failed
    STATUS["finished_unix"] = time.time()
    write_status()
    if failed:
        raise SystemExit(f"collection failed: {failed}")


if __name__ == "__main__":
    main()
