"""End-to-end recovery distillation campaign with train/dev/test separation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
SCREEN = ROOT / "observation_dropout_training_20260928"
TRAIN_COLLECTION = ROOT / "recovery_distill_train_20260929"
TEST_COLLECTION = ROOT / "recovery_distill_20260929"
WORK = ROOT / "recovery_distill_campaign_20260929"
PYTHON = Path(os.environ.get("PI05_PYTHON", sys.executable))
TRAIN_SCRIPT = ROOT / "openpi_dropout_train/scripts/train_libero_history.py"
BASE_INDEX = ROOT / "strong_search_20260926/cache/index.json"
R7_ROOT = SCREEN / "checkpoints/pi05_libero_history_h5/R7-drop25-u30000-seed11"
R7_PARAMS = R7_ROOT / "29999/params"
R7_MANIFEST = R7_ROOT.with_suffix(".run.json")
CHECKPOINTS = WORK / "checkpoints"
STEPS = 1500
STATE: dict = {"status": "RUNNING", "stages": {}, "started_unix": time.time()}


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def update(stage: str, **values) -> None:
    STATE["stages"].setdefault(stage, {}).update(values)
    STATE["updated_unix"] = time.time()
    write(WORK / "status.json", STATE)


def run(stage: str, command: list[str], *, env: dict | None = None) -> None:
    log = WORK / "logs" / f"{stage}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    update(stage, status="running", command=command, started_unix=time.time(), log=str(log))
    with log.open("w", encoding="utf-8", buffering=1) as handle:
        process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
    update(
        stage,
        status="completed" if not process.returncode else "failed",
        returncode=process.returncode,
        finished_unix=time.time(),
    )
    if process.returncode:
        STATE["status"] = "FAILED"
        write(WORK / "status.json", STATE)
        raise SystemExit(f"stage failed: {stage}")


def wait_for_collection() -> None:
    stage = "wait_for_train_collection"
    update(stage, status="waiting", source=str(TRAIN_COLLECTION / "status.json"))
    while True:
        path = TRAIN_COLLECTION / "status.json"
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if value["status"] == "COMPLETED":
                update(stage, status="completed", finished_unix=time.time())
                return
            if value["status"].startswith("FAILED"):
                raise SystemExit(f"teacher collection failed: {value['status']}")
        time.sleep(20)


def train_command(name: str, seed: int, *, peak_lr: float, history_peak_lr: float) -> list[str]:
    mixed = WORK / "mixed_cache"
    return [
        str(PYTHON), str(ROOT / "server_runtime_dropout_train.py"), str(TRAIN_SCRIPT),
        "--variant", "H5",
        "--cache-index", str(mixed / "index.json"),
        "--base-params-dir", str(R7_PARAMS),
        "--base-checkpoint-ref", str(R7_PARAMS),
        "--norm-stats-ref", str(mixed / "norm_stats/norm_stats.json"),
        "--seed", str(seed),
        "--num-train-steps", str(STEPS),
        "--schedule-total-steps", str(STEPS),
        "--warmup-steps", "100",
        "--peak-lr", str(peak_lr),
        "--decay-lr", str(peak_lr / 10),
        "--history-peak-lr", str(history_peak_lr),
        "--history-decay-lr", str(history_peak_lr / 10),
        "--batch-size", "8",
        "--fsdp-devices", "8",
        "--train-scope", "history_and_action_expert",
        "--history-injection", "prefix",
        "--history-visual-source", "past",
        "--current-image-dropout-probability", "0.25",
        "--current-image-dropout-seed", str(2026092900 + seed),
        "--log-interval", "50",
        "--save-interval", str(STEPS),
        "--checkpoint-base-dir", str(CHECKPOINTS),
        "--checkpoint-format", "zarr",
        "--exp-name", name,
    ]


def checkpoint_config(name: str) -> dict:
    root = CHECKPOINTS / "pi05_libero_history_h5" / name
    return {
        "variant": "H5",
        "checkpoint_dir": str(root),
        "run_manifest": str(root.with_suffix(".run.json")),
        "observation_intervention": "scheduled_dual_camera_blackout",
    }


def original_configs() -> dict:
    b2 = SCREEN / "checkpoints/pi05_libero_history_h0/B2-drop25-u30000-seed11"
    c1 = SCREEN / "checkpoints/pi05_libero_history_h5/C1-drop25-u30000-seed11"
    return {
        "R7-original": {
            "variant": "H5",
            "checkpoint_dir": str(R7_ROOT),
            "run_manifest": str(R7_MANIFEST),
            "observation_intervention": "scheduled_dual_camera_blackout",
        },
        "B2-zero": {
            "variant": "H0",
            "checkpoint_dir": str(b2),
            "run_manifest": str(b2.with_suffix(".run.json")),
            "observation_intervention": "scheduled_dual_camera_blackout",
        },
        "B2-LVF": {
            "variant": "H0",
            "checkpoint_dir": str(b2),
            "run_manifest": str(b2.with_suffix(".run.json")),
            "observation_intervention": "scheduled_dual_camera_last_valid",
        },
        "C1": {
            "variant": "H5",
            "checkpoint_dir": str(c1),
            "run_manifest": str(c1.with_suffix(".run.json")),
            "observation_intervention": "scheduled_dual_camera_blackout",
        },
    }


def evaluation(stage: str, *, configs: dict, episodes: range, schedule: Path) -> dict:
    plan = {
        "work_dir": str(WORK / stage),
        "schedule": str(schedule),
        "episode_indices": list(episodes),
        "configs": configs,
    }
    plan_path = WORK / f"{stage}_plan.json"
    write(plan_path, plan)
    run(
        stage,
        [str(PYTHON), str(ROOT / "run_recovery_distill_evaluation.py"), "--plan", str(plan_path)],
    )
    return json.loads((WORK / stage / "summary.json").read_text(encoding="utf-8"))


def deduplicate(stage: str) -> None:
    command = [
        str(PYTHON), str(ROOT / "deduplicate_finished_checkpoints.py"),
        "--checkpoint-root", str(CHECKPOINTS),
        "--checkpoint-dir", str(R7_ROOT / "29999"),
        "--index-path", str(WORK / "checkpoint_content_index.json"),
        "--storage-format", "zarr",
    ]
    run(stage, command)


def main() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    write(WORK / "status.json", STATE)
    wait_for_collection()

    run(
        "build_mixed_cache",
        [
            sys.executable, str(ROOT / "build_recovery_distill_cache.py"),
            "--base-index", str(BASE_INDEX),
            "--recovery-root", str(TRAIN_COLLECTION / "teacher_windows"),
            "--output-dir", str(WORK / "mixed_cache"),
            "--base-params-dir", str(R7_PARAMS),
            "--repeat", "20",
            "--episode-min", "10",
            "--episode-max-exclusive", "40",
        ],
    )

    audit_env = dict(os.environ)
    audit_env["PYTHONPATH"] = str(ROOT / "openpi_dropout_train/src")
    run(
        "audit_mixed_cache",
        [
            str(PYTHON), str(ROOT / "audit_recovery_distill_cache.py"),
            "--index", str(WORK / "mixed_cache/index.json"),
            "--output", str(WORK / "mixed_cache/audit.json"),
        ],
        env=audit_env,
    )

    probe_env = dict(os.environ)
    probe_env["CUDA_VISIBLE_DEVICES"] = "0"
    probe_env["PYTHONPATH"] = str(ROOT / "openpi_dropout_train/src")
    run(
        "frozen_r7_probe",
        [
            str(PYTHON), str(ROOT / "server_runtime_dropout_train.py"),
            str(ROOT / "run_recovery_history_probe.py"),
            "--run-manifest", str(R7_MANIFEST),
            "--checkpoint-dir", str(R7_ROOT / "29999"),
            "--train-root", str(TRAIN_COLLECTION / "teacher_windows"),
            "--dev-root", str(TRAIN_COLLECTION / "teacher_windows"),
            "--test-root", str(TEST_COLLECTION / "teacher_windows"),
            "--output", str(WORK / "probe/result.json"),
        ],
        env=probe_env,
    )

    candidates = {
        "RD-low-s21": {"peak_lr": 3e-6, "history_peak_lr": 3e-5},
        "RD-high-s21": {"peak_lr": 1e-5, "history_peak_lr": 1e-4},
    }
    for name, hyper in candidates.items():
        run(
            f"train_{name}",
            train_command(name, 21, **hyper),
            env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="1"),
        )
        deduplicate(f"dedup_{name}")

    dev_configs = {"R7-original": original_configs()["R7-original"], "B2-LVF": original_configs()["B2-LVF"]}
    dev_configs.update({name: checkpoint_config(name) for name in candidates})
    dev = evaluation(
        "dev_evaluation",
        configs=dev_configs,
        episodes=range(40, 50),
        schedule=TRAIN_COLLECTION / "clean_schedule/result.json",
    )
    selected = max(
        candidates,
        key=lambda name: (
            dev["configs"][name]["slip"]["successes"],
            dev["configs"][name]["control"]["successes"],
            -candidates[name]["peak_lr"],
        ),
    )
    update(
        "selection",
        status="completed",
        selected=selected,
        rule="max dev slip successes, then control successes, then lower LR",
        dev_summary=str(WORK / "dev_evaluation/summary.json"),
    )

    selected_hyper = candidates[selected]
    seed_names = [selected]
    for seed in (22, 23):
        name = selected.rsplit("-s", 1)[0] + f"-s{seed}"
        run(
            f"train_{name}",
            train_command(name, seed, **selected_hyper),
            env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="1"),
        )
        deduplicate(f"dedup_{name}")
        seed_names.append(name)

    final_configs = original_configs()
    final_configs.update({name: checkpoint_config(name) for name in seed_names})
    final = evaluation(
        "final_evaluation",
        configs=final_configs,
        episodes=range(0, 10),
        schedule=SCREEN / "development/B2-drop25-u30000-seed11/clean/result.json",
    )
    seed_control = [final["configs"][name]["control"]["success_rate"] for name in seed_names]
    seed_slip = [final["configs"][name]["slip"]["success_rate"] for name in seed_names]
    probe = json.loads((WORK / "probe/result.json").read_text(encoding="utf-8"))
    report = {
        "schema_version": "pi05-recovery-distill-campaign-v1",
        "status": "COMPLETED-DEV",
        "train_initial_states": list(range(10, 40)),
        "selection_initial_states": list(range(40, 50)),
        "final_initial_states": list(range(0, 10)),
        "selected_hyperparameter": selected,
        "selected_seed_runs": seed_names,
        "probe": probe["reports"],
        "development": dev,
        "final": final,
        "selected_three_seed_mean": {
            "control": statistics.fmean(seed_control),
            "slip": statistics.fmean(seed_slip),
            "slip_minus_control": statistics.fmean(seed_slip) - statistics.fmean(seed_control),
        },
        "claim_boundary": [
            "one LIBERO-10 task",
            "simulator teleport slip",
            "ten final initial states per seed",
            "teacher is B2 last-valid-frame policy",
            "development result, not SOTA or real-robot evidence",
        ],
    }
    write(WORK / "report.json", report)
    lines = [
        "# π0.5 受控滑落恢复蒸馏结果",
        "",
        "状态：`COMPLETED-DEV`。训练初始状态 10–39，配置选择 40–49，最终评测 0–9。",
        "",
        "| 配置 | Control | Slip | Slip-Control |",
        "|---|---:|---:|---:|",
    ]
    for name, row in final["configs"].items():
        lines.append(
            f"| {name} | {row['control']['successes']}/10 | {row['slip']['successes']}/10 | "
            f"{100 * row['slip_minus_control']:+.1f} pp |"
        )
    lines += [
        "",
        f"选中方法三种子 Control 均值：{100 * statistics.fmean(seed_control):.1f}%。",
        f"选中方法三种子 Slip 均值：{100 * statistics.fmean(seed_slip):.1f}%。",
        "",
        "该实验只支持单任务、仿真受控滑落下的机制判断。",
    ]
    (WORK / "结果报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    STATE["status"] = "COMPLETED"
    STATE["report"] = str(WORK / "report.json")
    STATE["finished_unix"] = time.time()
    write(WORK / "status.json", STATE)


if __name__ == "__main__":
    main()
