"""Resume the recovery-distillation campaign after the seed-23 ENOSPC failure.

The failed run completed optimization but could not atomically save the final
checkpoint.  This script reruns only seed 23 with the identical configuration,
then continues the frozen final evaluation and report generation.
"""

from __future__ import annotations

import json
import os
import statistics
import time
import traceback

import run_recovery_distill_campaign as campaign


def finish_campaign(*, selected: str, seed_names: list[str]) -> None:
    final_configs = campaign.original_configs()
    final_configs.update({name: campaign.checkpoint_config(name) for name in seed_names})
    final = campaign.evaluation(
        "final_evaluation",
        configs=final_configs,
        episodes=range(0, 10),
        schedule=campaign.SCREEN / "development/B2-drop25-u30000-seed11/clean/result.json",
    )
    seed_control = [final["configs"][name]["control"]["success_rate"] for name in seed_names]
    seed_slip = [final["configs"][name]["slip"]["success_rate"] for name in seed_names]
    probe = json.loads((campaign.WORK / "probe/result.json").read_text(encoding="utf-8"))
    development = json.loads((campaign.WORK / "dev_evaluation/summary.json").read_text(encoding="utf-8"))
    report = {
        "schema_version": "pi05-recovery-distill-campaign-v1",
        "status": "COMPLETED-DEV",
        "train_initial_states": list(range(10, 40)),
        "selection_initial_states": list(range(40, 50)),
        "final_initial_states": list(range(0, 10)),
        "selected_hyperparameter": selected,
        "selected_seed_runs": seed_names,
        "probe": probe["reports"],
        "development": development,
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
    campaign.write(campaign.WORK / "report.json", report)
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
    (campaign.WORK / "结果报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    status_path = campaign.WORK / "status.json"
    campaign.STATE = json.loads(status_path.read_text(encoding="utf-8"))
    campaign.STATE["status"] = "RUNNING"
    recovery = campaign.STATE.setdefault("recovery", {})
    recovery.update(
        {
            "cause": "ENOSPC while atomically saving RD-low-s23 step 1499",
            "protocol_changed": False,
            "action": "rerun only seed 23 with the identical seed, data, schedule, and hyperparameters",
            "resumed_unix": time.time(),
        }
    )
    campaign.write(status_path, campaign.STATE)
    selected = campaign.STATE["stages"]["selection"]["selected"]
    if selected != "RD-low-s21":
        raise RuntimeError(f"unexpected selected configuration: {selected}")
    seed_names = [selected, "RD-low-s22", "RD-low-s23"]
    campaign.run(
        "train_RD-low-s23",
        campaign.train_command("RD-low-s23", 23, peak_lr=3e-6, history_peak_lr=3e-5),
        env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="1"),
    )
    campaign.deduplicate("dedup_RD-low-s23")
    finish_campaign(selected=selected, seed_names=seed_names)
    recovery["completed_unix"] = time.time()
    campaign.STATE["status"] = "COMPLETED"
    campaign.STATE["report"] = str(campaign.WORK / "report.json")
    campaign.STATE["finished_unix"] = time.time()
    campaign.write(status_path, campaign.STATE)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        try:
            campaign.STATE["status"] = "FAILED"
            campaign.STATE.setdefault("recovery", {})["exception"] = traceback.format_exc()
            campaign.write(campaign.WORK / "status.json", campaign.STATE)
        finally:
            raise
