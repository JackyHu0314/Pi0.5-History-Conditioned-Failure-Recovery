"""Audit, summarize, and package the controlled-slip development campaign."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import time

import numpy as np

import run_controlled_slip_matrix as matrix


def episode_map(path: Path) -> dict[int, dict]:
    payload = matrix.read_json(path)
    return {int(row["initial_state_index"]): row for row in payload["episodes"]}


def audit_repeat(control: dict, repeat: dict) -> dict:
    control_query = matrix.blind_query(control)
    repeat_query = matrix.blind_query(repeat)
    control_event = control["execution_intervention"]
    repeat_event = repeat["execution_intervention"]
    state_delta = float(
        np.max(
            np.abs(
                np.asarray(control_query["current_state"], dtype=np.float64)
                - np.asarray(repeat_query["current_state"], dtype=np.float64)
            )
        )
    )
    action_delta = np.abs(
        matrix.blind_actions(control, control_query["query_step"])
        - matrix.blind_actions(repeat, repeat_query["query_step"])
    )
    checks = {
        "same_episode_seed": control["episode_seed"] == repeat["episode_seed"],
        "same_first_blind_query": control_query["query_step"] == repeat_query["query_step"],
        "same_action_prefix": (
            control_query["action_prefix_sha256"] == repeat_query["action_prefix_sha256"]
        ),
        "same_current_state_le_1e_6": state_delta <= 1e-6,
        "same_current_images": (
            control_query["current_images_sha256"] == repeat_query["current_images_sha256"]
        ),
        "same_full_policy_input": (
            control_query["current_input_sha256"] == repeat_query["current_input_sha256"]
        ),
        "same_history_input": (
            control_query["history_input_sha256"] == repeat_query["history_input_sha256"]
        ),
        "same_intervened_object": control_event["object"] == repeat_event["object"],
        "both_retain_grasp": (
            control_event["object"] in control_event["oracle_grasped_objects_after"]
            and repeat_event["object"] in repeat_event["oracle_grasped_objects_after"]
        ),
        "same_memory_capture": (
            control_event["memory_capture_images_sha256"]
            == repeat_event["memory_capture_images_sha256"]
        ),
    }
    return {
        "episode_index": control["initial_state_index"],
        "passed": all(checks.values()),
        "checks": checks,
        "control_success": bool(control["success"]),
        "repeat_success": bool(repeat["success"]),
        "outcome_equal": bool(control["success"]) == bool(repeat["success"]),
        "paired_current_state_max_abs_delta": state_delta,
        "first_blind_action_delta": {
            "max_abs": float(np.max(action_delta)),
            "mean_abs": float(np.mean(action_delta)),
        },
    }


def calibration_summary() -> dict:
    rows = {}
    for config_name in matrix.CONFIGS:
        audits = []
        for shard_index in range(len(matrix.SHARDS)):
            control = matrix.load_episode_map(config_name, shard_index, "control")
            repeat = matrix.load_episode_map(config_name, shard_index, "repeat-control")
            audits.extend(audit_repeat(control[index], repeat[index]) for index in sorted(control))
        max_abs = [row["first_blind_action_delta"]["max_abs"] for row in audits]
        mean_abs = [row["first_blind_action_delta"]["mean_abs"] for row in audits]
        rows[config_name] = {
            "episodes": len(audits),
            "protocol_passed": all(row["passed"] for row in audits),
            "protocol_failures": [row["episode_index"] for row in audits if not row["passed"]],
            "outcome_disagreements": sum(not row["outcome_equal"] for row in audits),
            "first_blind_action_delta": {
                "max_of_max_abs": max(max_abs),
                "median_max_abs": statistics.median(max_abs),
                "mean_abs_across_episodes": statistics.fmean(mean_abs),
            },
            "episode_audits": audits,
        }
    return {
        "schema_version": "pi05-control-repeat-calibration-summary-v1",
        "purpose": "estimate independent-inference numerical and rollout variability under identical inputs",
        "rows": rows,
    }


def write_report(summary: dict) -> None:
    calibration = summary["control_repeat_calibration"]["rows"]
    lines = [
        "# π0.5 受控滑落开发实验",
        "",
        f"状态：`{summary['status']}`。范围是 LIBERO-10 task 5 的 10 个固定初始状态；这是主筛选后的探索性开发实验。",
        "",
        "| 配置 | Control | Slip | Slip-Control | 两者成功/仅C/仅S/均失败 | Slip 动作差 median | Control-repeat 噪声 median | 重复结果翻转 | 协议 |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for config_name, row in summary["rows"].items():
        paired = row["paired_outcomes"]
        action = row["first_blind_action_delta"]
        noise = calibration[config_name]
        lines.append(
            f"| {config_name} | {row['control_successes']}/{row['episodes']} | "
            f"{row['slip_successes']}/{row['episodes']} | {row['slip_minus_control_pp']:+.1f} pp | "
            f"{paired['both_success']}/{paired['control_only']}/{paired['slip_only']}/{paired['both_failure']} | "
            f"{action['median_max_abs']:.6f} | "
            f"{noise['first_blind_action_delta']['median_max_abs']:.6f} | "
            f"{noise['outcome_disagreements']}/10 | "
            f"{'PASS' if row['protocol_passed'] and noise['protocol_passed'] else 'FAIL'} |"
        )
    lines += [
        "",
        "Slip 动作差是在第一次盲区查询、相同动作前缀后比较 control 与 slip 的 5 步动作块；control-repeat 在同一条件下再次推理，用于估计数值和 rollout 波动。",
        "",
        "B2-zero 与 C1 在第一次盲区查询应具有相同完整输入；B2-LVF 通过 last-valid 当前图像看到滑落，R7 通过过去视觉历史看到滑落。每个 slip 和 repeat 都重放对应 control 的逐步动作前缀。",
        "",
        "若成功率差落在 control-repeat 的结果翻转量级内，不能归因于记忆模块。该实验没有新训练种子、跨任务覆盖或冻结 held-out，因此不能支持泛化、恢复能力或 SOTA 声明。",
    ]
    (matrix.WORK / "结果报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary = matrix.summarize()
    calibration = calibration_summary()
    summary["control_repeat_calibration"] = calibration
    summary["status"] = (
        "COMPLETED-DEV"
        if summary["status"] == "COMPLETED-DEV"
        and all(row["protocol_passed"] for row in calibration["rows"].values())
        else "PROTOCOL-FAILED"
    )
    summary["finished_unix"] = time.time()
    matrix.write_json(matrix.WORK / "summary.json", summary)
    write_report(summary)
    status_path = matrix.WORK / "status.json"
    status = matrix.read_json(status_path)
    status["phase"] = "completed" if summary["status"] == "COMPLETED-DEV" else "protocol_failed"
    status["failed_jobs"] = []
    status["recovered_jobs"] = ["R7-shard0"]
    status["summary"] = str(matrix.WORK / "summary.json")
    status["finished_unix"] = time.time()
    matrix.write_json(status_path, status)
    print(json.dumps({"status": summary["status"], "rows": summary["rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
