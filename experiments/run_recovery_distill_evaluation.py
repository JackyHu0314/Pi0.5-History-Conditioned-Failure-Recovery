"""Run paired controlled-slip evaluation for a JSON-defined checkpoint matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parent
PYTHON = Path(os.environ.get("PI05_PYTHON", sys.executable))
RUNTIME = ROOT / "server_runtime_dropout_eval.py"
EVALUATOR = ROOT / "openpi_dropout_eval/scripts/eval_libero_controlled_slip.py"
LOCK = threading.Lock()


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_branch(
    *,
    config_name: str,
    config: dict,
    shard_index: int,
    episodes: tuple[int, ...],
    branch: str,
    gpu: int,
    schedule: Path,
    work: Path,
    replay: Path | None,
) -> dict:
    branch_dir = work / config_name / f"shard{shard_index}" / branch
    command = [
        str(PYTHON), str(RUNTIME), str(EVALUATOR),
        "--variant", config["variant"],
        "--checkpoint-dir", config["checkpoint_dir"],
        "--output", str(branch_dir / "result.json"),
        "--run-manifest", config["run_manifest"],
        "--task-ids", "5",
        "--episode-indices", *map(str, episodes),
        "--seed", "7",
        "--observation-intervention", config["observation_intervention"],
        "--execution-intervention", branch,
        "--intervention-schedule", str(schedule),
        "--blackout-queries", "3",
        "--trace-dir", str(branch_dir / "traces"),
    ]
    if replay is not None:
        command.extend(["--replay-prefix-trace-dir", str(replay)])
    branch_dir.mkdir(parents=True, exist_ok=True)
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
        "output": str(branch_dir / "result.json"),
        "log": str(branch_dir / "run.log"),
    }


def run_pair(job: dict, *, schedule: Path, work: Path, status: dict) -> None:
    name = job["config_name"]
    shard = job["shard_index"]
    key = f"{name}-shard{shard}"
    with LOCK:
        status["jobs"][key] = {"status": "control_running", **job, "started_unix": time.time()}
        write(work / "status.json", status)
    control = run_branch(
        config_name=name,
        config=job["config"],
        shard_index=shard,
        episodes=tuple(job["episodes"]),
        branch="control",
        gpu=job["gpu"],
        schedule=schedule,
        work=work,
        replay=None,
    )
    with LOCK:
        status["jobs"][key]["control"] = control
        status["jobs"][key]["status"] = "slip_running" if not control["returncode"] else "failed_control"
        write(work / "status.json", status)
    if control["returncode"]:
        return
    trace_dir = work / name / f"shard{shard}" / "control/traces"
    slip = run_branch(
        config_name=name,
        config=job["config"],
        shard_index=shard,
        episodes=tuple(job["episodes"]),
        branch="slip",
        gpu=job["gpu"],
        schedule=schedule,
        work=work,
        replay=trace_dir,
    )
    with LOCK:
        status["jobs"][key]["slip"] = slip
        status["jobs"][key]["status"] = "completed" if not slip["returncode"] else "failed_slip"
        status["jobs"][key]["finished_unix"] = time.time()
        write(work / "status.json", status)


def worker(queue: list[dict], *, schedule: Path, work: Path, status: dict) -> None:
    for job in queue:
        run_pair(job, schedule=schedule, work=work, status=status)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = read(args.plan.resolve())
    work = Path(plan["work_dir"]).resolve()
    schedule = Path(plan["schedule"]).resolve()
    episodes = tuple(int(value) for value in plan["episode_indices"])
    midpoint = (len(episodes) + 1) // 2
    shards = (episodes[:midpoint], episodes[midpoint:])
    jobs = []
    for config_name, config in plan["configs"].items():
        for shard_index, shard_episodes in enumerate(shards):
            if shard_episodes:
                jobs.append(
                    {
                        "config_name": config_name,
                        "config": config,
                        "shard_index": shard_index,
                        "episodes": list(shard_episodes),
                    }
                )
    queues: list[list[dict]] = [[] for _ in range(8)]
    for index, job in enumerate(jobs):
        job["gpu"] = index % 8
        queues[index % 8].append(job)
    status = {
        "status": "RUNNING",
        "plan": plan,
        "jobs": {},
        "started_unix": time.time(),
    }
    write(work / "status.json", status)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, queue, schedule=schedule, work=work, status=status) for queue in queues]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    failed = [key for key, value in status["jobs"].items() if value["status"] != "completed"]
    if failed:
        status["status"] = "FAILED"
        status["failed_jobs"] = failed
        status["finished_unix"] = time.time()
        write(work / "status.json", status)
        raise SystemExit(f"evaluation failed: {failed}")

    summary = {"schema_version": "pi05-recovery-distill-eval-v1", "episodes": list(episodes), "configs": {}}
    for config_name in plan["configs"]:
        branch_episodes = {"control": [], "slip": []}
        for shard_index in range(2):
            for branch in branch_episodes:
                path = work / config_name / f"shard{shard_index}" / branch / "result.json"
                if path.is_file():
                    branch_episodes[branch].extend(read(path)["episodes"])
        branch_maps = {
            branch: {int(row["episode_index"]): int(bool(row["success"])) for row in rows}
            for branch, rows in branch_episodes.items()
        }
        summary["configs"][config_name] = {
            branch: {
                "episodes": len(values),
                "successes": sum(values.values()),
                "success_rate": sum(values.values()) / len(values),
                "by_episode": values,
            }
            for branch, values in branch_maps.items()
        }
        summary["configs"][config_name]["slip_minus_control"] = (
            summary["configs"][config_name]["slip"]["success_rate"]
            - summary["configs"][config_name]["control"]["success_rate"]
        )
    write(work / "summary.json", summary)
    status["status"] = "COMPLETED"
    status["summary"] = str(work / "summary.json")
    status["finished_unix"] = time.time()
    write(work / "status.json", status)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
