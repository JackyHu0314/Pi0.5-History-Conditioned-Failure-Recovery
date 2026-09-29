"""Run the eight-GPU controlled-slip development matrix on LIBERO task 5."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import threading
import time

import numpy as np


ROOT = Path(__file__).resolve().parent
SCREEN = ROOT / "observation_dropout_training_20260928"
CAMPAIGN = ROOT / "controlled_slip_20260929"
WORK = CAMPAIGN / "development"
PYTHON = Path(os.environ.get("PI05_PYTHON", sys.executable))
RUNTIME = ROOT / "server_runtime_dropout_eval.py"
EVALUATOR = ROOT / "openpi_dropout_eval/scripts/eval_libero_controlled_slip.py"
TASK_ID = 5
BLACKOUT_QUERIES = 3
SHARDS = ((0, 1, 2, 3, 4), (5, 6, 7, 8, 9))
CONFIGS = {
    "B2-zero": {
        "run": "B2-drop25-u30000-seed11",
        "variant": "H0",
        "checkpoint_group": "pi05_libero_history_h0",
        "observation": "scheduled_dual_camera_blackout",
        "expected_input_relation": "same",
        "expected_history_relation": "none",
    },
    "B2-LVF": {
        "run": "B2-drop25-u30000-seed11",
        "variant": "H0",
        "checkpoint_group": "pi05_libero_history_h0",
        "observation": "scheduled_dual_camera_last_valid",
        "expected_input_relation": "different",
        "expected_history_relation": "none",
    },
    "C1": {
        "run": "C1-drop25-u30000-seed11",
        "variant": "H5",
        "checkpoint_group": "pi05_libero_history_h5",
        "observation": "scheduled_dual_camera_blackout",
        "expected_input_relation": "same",
        "expected_history_relation": "same",
    },
    "R7": {
        "run": "R7-drop25-u30000-seed11",
        "variant": "H5",
        "checkpoint_group": "pi05_libero_history_h5",
        "observation": "scheduled_dual_camera_blackout",
        "expected_input_relation": "different",
        "expected_history_relation": "different",
    },
}
LOCK = threading.Lock()
STOP = threading.Event()
STATUS: dict = {}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def update(job: str, **values) -> None:
    with LOCK:
        STATUS["jobs"][job].update(values)
        write_json(WORK / "status.json", STATUS)


def paths(config: dict) -> tuple[Path, Path, Path]:
    checkpoint = SCREEN / "checkpoints" / config["checkpoint_group"] / config["run"]
    schedule = SCREEN / "development" / config["run"] / "clean/result.json"
    return checkpoint, checkpoint.with_suffix(".run.json"), schedule


def run_branch(
    *,
    config_name: str,
    config: dict,
    shard_index: int,
    episodes: tuple[int, ...],
    branch: str,
    gpu: int,
    replay_trace_dir: Path | None,
) -> dict:
    checkpoint, manifest, schedule = paths(config)
    branch_dir = WORK / config_name / f"shard{shard_index}" / branch
    output = branch_dir / "result.json"
    log = branch_dir / "run.log"
    command = [
        str(PYTHON),
        str(RUNTIME),
        str(EVALUATOR),
        "--variant", config["variant"],
        "--checkpoint-dir", str(checkpoint),
        "--output", str(output),
        "--run-manifest", str(manifest),
        "--task-ids", str(TASK_ID),
        "--episode-indices", *map(str, episodes),
        "--seed", "7",
        "--observation-intervention", config["observation"],
        "--execution-intervention", branch,
        "--intervention-schedule", str(schedule),
        "--blackout-queries", str(BLACKOUT_QUERIES),
        "--trace-dir", str(branch_dir / "traces"),
    ]
    if replay_trace_dir is not None:
        command.extend(["--replay-prefix-trace-dir", str(replay_trace_dir)])
    branch_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="1")
    started = time.time()
    with log.open("w", encoding="utf-8", buffering=1) as handle:
        process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=environment)
    return {
        "branch": branch,
        "returncode": process.returncode,
        "seconds": time.time() - started,
        "output": str(output),
        "log": str(log),
        "command": command,
    }


def run_pair(config_name: str, shard_index: int, gpu: int) -> None:
    config = CONFIGS[config_name]
    episodes = SHARDS[shard_index]
    job = f"{config_name}-shard{shard_index}"
    update(job, status="control_running", gpu=gpu, started_unix=time.time())
    control = run_branch(
        config_name=config_name,
        config=config,
        shard_index=shard_index,
        episodes=episodes,
        branch="control",
        gpu=gpu,
        replay_trace_dir=None,
    )
    update(job, control=control)
    if control["returncode"]:
        update(job, status="failed_control", finished_unix=time.time())
        return
    control_trace_dir = WORK / config_name / f"shard{shard_index}" / "control/traces"
    update(job, status="slip_running")
    slip = run_branch(
        config_name=config_name,
        config=config,
        shard_index=shard_index,
        episodes=episodes,
        branch="slip",
        gpu=gpu,
        replay_trace_dir=control_trace_dir,
    )
    update(job, slip=slip)
    if slip["returncode"]:
        update(job, status="failed_slip", finished_unix=time.time())
        return
    update(job, status="completed", finished_unix=time.time())


def episode_map(payload: dict) -> dict[int, dict]:
    return {int(row["initial_state_index"]): row for row in payload["episodes"]}


def load_episode_map(config_name: str, shard_index: int, branch: str) -> dict[int, dict]:
    root = WORK / config_name / f"shard{shard_index}"
    episodes = episode_map(read_json(root / branch / "result.json"))
    repair_root = CAMPAIGN / "repairs" / config_name
    if repair_root.is_dir():
        for repair in sorted(repair_root.glob("episode*/" + branch + "/result.json")):
            for episode_index, episode in episode_map(read_json(repair)).items():
                if episode_index in episodes:
                    episodes[episode_index] = episode
    return episodes


def blind_query(episode: dict) -> dict:
    first_blind = episode["execution_intervention"]["first_blind_policy_query_step"]
    return next(row for row in episode["selected_query_audit"] if row["query_step"] == first_blind)


def blind_actions(episode: dict, query_step: int) -> np.ndarray:
    trace = read_json(Path(episode["trace"]))
    query = next(row for row in trace["policy_queries"] if row["query_step"] == query_step)
    return np.asarray(query["actions"][:5], dtype=np.float64)


def audit_episode(config_name: str, control: dict, slip: dict) -> dict:
    config = CONFIGS[config_name]
    control_query, slip_query = blind_query(control), blind_query(slip)
    control_event = control["execution_intervention"]
    slip_event = slip["execution_intervention"]
    state_delta = float(
        np.max(
            np.abs(
                np.asarray(control_query["current_state"], dtype=np.float64)
                - np.asarray(slip_query["current_state"], dtype=np.float64)
            )
        )
    )
    action_delta = np.abs(
        blind_actions(control, control_query["query_step"])
        - blind_actions(slip, slip_query["query_step"])
    )
    current_images_equal = (
        control_query["current_images_sha256"] == slip_query["current_images_sha256"]
    )
    input_equal = control_query["current_input_sha256"] == slip_query["current_input_sha256"]
    history_equal = control_query["history_input_sha256"] == slip_query["history_input_sha256"]
    checks = {
        "same_episode_seed": control["episode_seed"] == slip["episode_seed"],
        "same_first_blind_query": control_query["query_step"] == slip_query["query_step"],
        "same_action_prefix": (
            control_query["action_prefix_sha256"] == slip_query["action_prefix_sha256"]
        ),
        "same_current_state_le_1e_6": state_delta <= 1e-6,
        "same_intervened_object": control_event["object"] == slip_event["object"],
        "control_retains_grasp": control_event["object"] in control_event["oracle_grasped_objects_after"],
        "slip_breaks_grasp": slip_event["object"] not in slip_event["oracle_grasped_objects_after"],
        "slip_reaches_initial_qpos": slip_event["object_qpos_target_max_abs_error"] <= 1e-9,
        "different_live_camera_after_slip": (
            control_query["clean_current_images_sha256"]
            != slip_query["clean_current_images_sha256"]
        ),
        "different_memory_capture": (
            control_event["memory_capture_images_sha256"]
            != slip_event["memory_capture_images_sha256"]
        ),
        "expected_current_image_relation": (
            current_images_equal
            if config["observation"] == "scheduled_dual_camera_blackout"
            else not current_images_equal
        ),
        "expected_full_input_relation": (
            input_equal if config["expected_input_relation"] == "same" else not input_equal
        ),
        "expected_history_relation": (
            True
            if config["expected_history_relation"] == "none"
            else history_equal
            if config["expected_history_relation"] == "same"
            else not history_equal
        ),
    }
    return {
        "episode_index": control["initial_state_index"],
        "passed": all(checks.values()),
        "checks": checks,
        "control_success": bool(control["success"]),
        "slip_success": bool(slip["success"]),
        "paired_current_state_max_abs_delta": state_delta,
        "first_blind_action_delta": {
            "max_abs": float(np.max(action_delta)),
            "mean_abs": float(np.mean(action_delta)),
        },
        "first_blind_query": control_query["query_step"],
        "object": control_event["object"],
    }


def summarize() -> dict:
    rows = {}
    for config_name in CONFIGS:
        control_episodes: dict[int, dict] = {}
        slip_episodes: dict[int, dict] = {}
        for shard_index in range(len(SHARDS)):
            root = WORK / config_name / f"shard{shard_index}"
            control_episodes.update(load_episode_map(config_name, shard_index, "control"))
            slip_episodes.update(load_episode_map(config_name, shard_index, "slip"))
        audits = [
            audit_episode(config_name, control_episodes[index], slip_episodes[index])
            for index in sorted(control_episodes)
        ]
        outcomes = {
            "both_success": sum(row["control_success"] and row["slip_success"] for row in audits),
            "control_only": sum(row["control_success"] and not row["slip_success"] for row in audits),
            "slip_only": sum(not row["control_success"] and row["slip_success"] for row in audits),
            "both_failure": sum(not row["control_success"] and not row["slip_success"] for row in audits),
        }
        action_max = [row["first_blind_action_delta"]["max_abs"] for row in audits]
        action_mean = [row["first_blind_action_delta"]["mean_abs"] for row in audits]
        control_successes = sum(row["control_success"] for row in audits)
        slip_successes = sum(row["slip_success"] for row in audits)
        rows[config_name] = {
            "episodes": len(audits),
            "protocol_passed": all(row["passed"] for row in audits),
            "protocol_failures": [row["episode_index"] for row in audits if not row["passed"]],
            "control_successes": control_successes,
            "slip_successes": slip_successes,
            "slip_minus_control_pp": 100.0 * (slip_successes - control_successes) / len(audits),
            "paired_outcomes": outcomes,
            "first_blind_action_delta": {
                "max_of_max_abs": max(action_max),
                "median_max_abs": statistics.median(action_max),
                "mean_abs_across_episodes": statistics.fmean(action_mean),
            },
            "episode_audits": audits,
        }
    return {
        "schema_version": "pi05-controlled-slip-development-summary-v1",
        "status": "COMPLETED-DEV" if all(row["protocol_passed"] for row in rows.values()) else "PROTOCOL-FAILED",
        "scope": "post-screen exploratory development intervention on LIBERO-10 task 5",
        "formal_claim": False,
        "task_id": TASK_ID,
        "initial_state_indices": list(range(10)),
        "blackout_queries": BLACKOUT_QUERIES,
        "rows": rows,
    }


def write_report(summary: dict) -> None:
    lines = [
        "# π0.5 受控滑落开发实验",
        "",
        f"状态：`{summary['status']}`。这是主筛选后的探索性开发实验，仅覆盖 LIBERO-10 task 5 的 10 个固定初始状态。",
        "",
        "| 配置 | Control | Slip | Slip-Control | 配对结果 C/S/S/C/F/F | 首次盲区动作差 max(median) | 协议审计 |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for config_name, row in summary["rows"].items():
        paired = row["paired_outcomes"]
        action = row["first_blind_action_delta"]
        lines.append(
            f"| {config_name} | {row['control_successes']}/{row['episodes']} | "
            f"{row['slip_successes']}/{row['episodes']} | {row['slip_minus_control_pp']:+.1f} pp | "
            f"{paired['both_success']}/{paired['control_only']}/{paired['slip_only']}/{paired['both_failure']} | "
            f"{action['max_of_max_abs']:.6f} ({action['median_max_abs']:.6f}) | "
            f"{'PASS' if row['protocol_passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "配对结果顺序为：两者成功 / 仅 control 成功 / 仅 slip 成功 / 两者失败。",
        "",
        "B2-zero 与 C1 在第一次盲区查询应具有相同完整输入；B2-LVF 与 R7 应分别通过 last-valid 当前图像和历史帧看到滑落差异。所有 slip 分支都重放对应 control 的精确动作前缀。",
        "",
        "该结果用于判断干预是否可识别以及哪类上下文产生动作响应；样本量、任务覆盖和事后设计均不支持正式泛化或 SOTA 声明。",
    ]
    (WORK / "结果报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def record_resources() -> None:
    while not STOP.wait(5):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        row = {
            "time_unix": time.time(),
            "disk_free_bytes": shutil.disk_usage(ROOT).free,
            "gpu_rows": [[part.strip() for part in line.split(",")] for line in result.stdout.splitlines()],
        }
        with (WORK / "resources.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    global STATUS
    WORK.mkdir(parents=True, exist_ok=False)
    smoke_audit = CAMPAIGN / "smoke/audit.json"
    if not read_json(smoke_audit)["passed"]:
        raise SystemExit("controlled-slip smoke audit has not passed")
    plan = {
        "schema_version": "pi05-controlled-slip-development-plan-v1",
        "created_unix": time.time(),
        "scope": "exploratory development task after the observation-dropout screen",
        "task_id": TASK_ID,
        "episode_shards": [list(shard) for shard in SHARDS],
        "blackout_queries": BLACKOUT_QUERIES,
        "configs": CONFIGS,
        "evaluator": str(EVALUATOR),
        "evaluator_sha256": sha256(EVALUATOR),
        "smoke_audit": str(smoke_audit),
        "smoke_audit_sha256": sha256(smoke_audit),
        "same_input_action_numerical_tolerance": 5e-3,
        "formal_claim": False,
    }
    write_json(WORK / "plan.json", plan)
    jobs = [
        (config_name, shard_index, gpu)
        for gpu, (config_name, shard_index) in enumerate(
            (config_name, shard_index)
            for config_name in CONFIGS
            for shard_index in range(len(SHARDS))
        )
    ]
    STATUS = {
        "schema_version": "pi05-controlled-slip-development-status-v1",
        "phase": "running",
        "started_unix": time.time(),
        "jobs": {
            f"{config_name}-shard{shard_index}": {
                "status": "planned",
                "gpu": gpu,
                "episodes": list(SHARDS[shard_index]),
            }
            for config_name, shard_index, gpu in jobs
        },
    }
    write_json(WORK / "status.json", STATUS)
    recorder = threading.Thread(target=record_resources, daemon=True)
    recorder.start()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda item: run_pair(*item), jobs))
    failed = [key for key, value in STATUS["jobs"].items() if value["status"] != "completed"]
    if failed:
        STATUS["phase"] = "failed"
        STATUS["failed_jobs"] = failed
    else:
        summary = summarize()
        write_json(WORK / "summary.json", summary)
        write_report(summary)
        STATUS["phase"] = "completed" if summary["status"] == "COMPLETED-DEV" else "protocol_failed"
        STATUS["summary"] = str(WORK / "summary.json")
    STATUS["finished_unix"] = time.time()
    write_json(WORK / "status.json", STATUS)
    STOP.set()
    recorder.join(timeout=10)
    print(json.dumps({"phase": STATUS["phase"], "failed_jobs": failed}, indent=2))
    if failed or STATUS["phase"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
