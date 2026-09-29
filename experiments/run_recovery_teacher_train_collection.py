"""Collect held-out-from-evaluation controlled-slip teacher windows on init states 10--39."""

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
WORK = ROOT / "recovery_distill_train_20260929"
PYTHON = Path(os.environ.get("PI05_PYTHON", sys.executable))
RUNTIME = ROOT / "server_runtime_dropout_eval.py"
CLEAN_EVALUATOR = ROOT / "openpi_dropout_eval/scripts/eval_libero_history.py"
SLIP_EVALUATOR = ROOT / "openpi_dropout_eval/scripts/eval_libero_controlled_slip.py"
CHECKPOINT = SCREEN / "checkpoints/pi05_libero_history_h0/B2-drop25-u30000-seed11"
MANIFEST = CHECKPOINT.with_suffix(".run.json")
EPISODES = tuple(range(10, 50))
GPU_EPISODES = tuple(tuple(EPISODES[gpu::8]) for gpu in range(8))
LOCK = threading.Lock()
STATUS: dict = {"status": "RUNNING", "clean": {}, "jobs": {}}


def write_status() -> None:
    path = WORK / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(STATUS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def update(section: str, key: str, **values) -> None:
    with LOCK:
        STATUS[section].setdefault(key, {}).update(values)
        write_status()


def run(command: list[str], *, gpu: int, log: Path) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log.open("w", encoding="utf-8", buffering=1) as handle:
        process = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1"),
        )
    return {
        "returncode": process.returncode,
        "seconds": time.time() - started,
        "log": str(log),
    }


def collect_clean(gpu: int, episodes: tuple[int, ...]) -> None:
    key = f"gpu{gpu}"
    output = WORK / "clean_shards" / key / "result.json"
    command = [
        str(PYTHON), str(RUNTIME), str(CLEAN_EVALUATOR),
        "--variant", "H0",
        "--checkpoint-dir", str(CHECKPOINT),
        "--output", str(output),
        "--run-manifest", str(MANIFEST),
        "--task-ids", "5",
        "--episode-indices", *map(str, episodes),
        "--seed", "7",
    ]
    update("clean", key, status="running", gpu=gpu, episodes=episodes, started_unix=time.time())
    result = run(command, gpu=gpu, log=output.parent / "run.log")
    update(
        "clean",
        key,
        status="completed" if not result["returncode"] else "failed",
        result=result,
        output=str(output),
        finished_unix=time.time(),
    )


def merge_clean() -> Path:
    episodes = []
    for gpu in range(8):
        path = WORK / "clean_shards" / f"gpu{gpu}" / "result.json"
        episodes.extend(json.loads(path.read_text(encoding="utf-8"))["episodes"])
    episodes.sort(key=lambda row: int(row["episode_index"]))
    output = WORK / "clean_schedule/result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"episodes": episodes}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def run_branch(episode: int, branch: str, gpu: int, schedule: Path, replay: Path | None) -> dict:
    branch_dir = WORK / "teacher_rollouts" / f"episode{episode:02d}" / branch
    command = [
        str(PYTHON), str(RUNTIME), str(SLIP_EVALUATOR),
        "--variant", "H0",
        "--checkpoint-dir", str(CHECKPOINT),
        "--output", str(branch_dir / "result.json"),
        "--run-manifest", str(MANIFEST),
        "--task-ids", "5",
        "--episode-indices", str(episode),
        "--seed", "7",
        "--observation-intervention", "scheduled_dual_camera_last_valid",
        "--execution-intervention", branch,
        "--intervention-schedule", str(schedule),
        "--blackout-queries", "3",
        "--trace-dir", str(branch_dir / "traces"),
        "--recovery-dataset-dir", str(WORK / "teacher_windows"),
        "--recovery-pre-steps", "8",
        "--recovery-post-steps", "25",
        "--recovery-encode-batch-size", "16",
    ]
    if replay is not None:
        command.extend(["--replay-prefix-trace-dir", str(replay)])
    result = run(command, gpu=gpu, log=branch_dir / "run.log")
    result["output"] = str(branch_dir / "result.json")
    return result


def collect_gpu_queue(gpu: int, episodes: tuple[int, ...], schedule: Path) -> None:
    for episode in episodes:
        key = str(episode)
        update("jobs", key, status="control_running", gpu=gpu, started_unix=time.time())
        control = run_branch(episode, "control", gpu, schedule, None)
        update("jobs", key, control=control)
        if control["returncode"]:
            update("jobs", key, status="failed_control", finished_unix=time.time())
            continue
        trace_dir = WORK / "teacher_rollouts" / f"episode{episode:02d}" / "control/traces"
        update("jobs", key, status="slip_running")
        slip = run_branch(episode, "slip", gpu, schedule, trace_dir)
        update("jobs", key, slip=slip)
        update(
            "jobs",
            key,
            status="completed" if not slip["returncode"] else "failed_slip",
            finished_unix=time.time(),
        )


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    STATUS["started_unix"] = time.time()
    write_status()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(collect_clean, gpu, GPU_EPISODES[gpu]) for gpu in range(8)]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    clean_failed = [key for key, value in STATUS["clean"].items() if value["status"] != "completed"]
    if clean_failed:
        STATUS["status"] = "FAILED_CLEAN"
        STATUS["failed_clean"] = clean_failed
        write_status()
        raise SystemExit(f"clean collection failed: {clean_failed}")
    schedule = merge_clean()
    STATUS["clean_schedule"] = str(schedule)
    write_status()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(collect_gpu_queue, gpu, GPU_EPISODES[gpu], schedule)
            for gpu in range(8)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    failed = [key for key, value in STATUS["jobs"].items() if value["status"] != "completed"]
    STATUS["status"] = "COMPLETED" if not failed else "FAILED"
    STATUS["failed_jobs"] = failed
    STATUS["finished_unix"] = time.time()
    write_status()
    if failed:
        raise SystemExit(f"teacher collection failed: {failed}")


if __name__ == "__main__":
    main()
