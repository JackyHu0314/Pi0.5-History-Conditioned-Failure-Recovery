"""Create the audited mechanism report after the recovery campaign and post-training probes."""

from __future__ import annotations

import argparse
import glob
import json
import math
import pathlib
import re
import statistics

import numpy as np


STEP_PATTERN = re.compile(r"Step (\d+): grad_norm=([0-9.e+-]+), loss=([0-9.e+-]+)")


def read(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def wilson(successes: int, count: int) -> list[float]:
    z = 1.959963984540054
    rate = successes / count
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    return [center - radius, center + radius]


def mcnemar_exact(candidate: dict[int, int], baseline: dict[int, int]) -> dict:
    candidate_only = sum(candidate[key] == 1 and baseline[key] == 0 for key in candidate)
    baseline_only = sum(candidate[key] == 0 and baseline[key] == 1 for key in candidate)
    discordant = candidate_only + baseline_only
    if not discordant:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(candidate_only, baseline_only) + 1)) / 2**discordant
        p_value = min(1.0, 2 * tail)
    return {
        "candidate_only_success": candidate_only,
        "baseline_only_success": baseline_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def traces(root: pathlib.Path, config: str, branch: str) -> dict[int, dict]:
    result = {}
    pattern = root / "final_evaluation" / config / "shard*" / branch / "traces" / "*.json"
    for value in glob.glob(str(pattern)):
        payload = read(pathlib.Path(value))
        result[int(payload["episode_index"])] = payload
    return result


def mechanism(root: pathlib.Path, config: str) -> dict:
    control = traces(root, config, "control")
    slip = traces(root, config, "slip")
    action_deltas = []
    regrasp_steps = []
    within_blind = 0
    for episode_index in sorted(slip):
        control_trace = control[episode_index]
        slip_trace = slip[episode_index]
        event = slip_trace["execution_intervention"]
        first_blind = int(event["first_blind_policy_query_step"])
        target = event["object"]
        control_query = next(row for row in control_trace["policy_queries"] if row["query_step"] == first_blind)
        slip_query = next(row for row in slip_trace["policy_queries"] if row["query_step"] == first_blind)
        action_deltas.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(control_query["actions"][:5], dtype=np.float64)
                        - np.asarray(slip_query["actions"][:5], dtype=np.float64)
                    )
                )
            )
        )
        first_regrasp = next(
            (
                int(row["step"]) + 1
                for row in slip_trace["steps"]
                if int(row["step"]) + 1 >= first_blind
                and target in row["oracle_grasped_objects_after"]
            ),
            None,
        )
        if first_regrasp is not None:
            delay = first_regrasp - first_blind
            regrasp_steps.append(delay)
            within_blind += delay <= 15
    return {
        "episodes": len(slip),
        "first_blind_action_max_abs_delta_median": statistics.median(action_deltas),
        "first_blind_action_max_abs_delta_values": action_deltas,
        "regrasp_ever": len(regrasp_steps),
        "regrasp_within_15_steps": within_blind,
        "regrasp_delay_median_steps": statistics.median(regrasp_steps) if regrasp_steps else None,
        "regrasp_delay_steps": regrasp_steps,
    }


def training_curve(path: pathlib.Path) -> dict:
    rows = [
        {"step": int(step), "grad_norm": float(grad), "loss": float(loss)}
        for step, grad, loss in STEP_PATTERN.findall(path.read_text(encoding="utf-8", errors="replace"))
    ]
    return {
        "points": rows,
        "first": rows[0] if rows else None,
        "last": rows[-1] if rows else None,
        "min_loss": min((row["loss"] for row in rows), default=None),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.campaign_root.resolve()
    report = read(root / "report.json")
    final = report["final"]
    configs = final["configs"]
    selected_names = report["selected_seed_runs"]
    baseline_slip = {int(key): value for key, value in configs["R7-original"]["slip"]["by_episode"].items()}
    analysis = {}
    for name, row in configs.items():
        successes = int(row["slip"]["successes"])
        values = {int(key): value for key, value in row["slip"]["by_episode"].items()}
        analysis[name] = {
            "control": row["control"],
            "slip": row["slip"],
            "slip_wilson95": wilson(successes, int(row["slip"]["episodes"])),
            "slip_vs_r7_mcnemar": None if name == "R7-original" else mcnemar_exact(values, baseline_slip),
            "mechanism": mechanism(root, name),
        }
    probes = {"R7-original": read(root / "probe/result.json")["reports"]}
    for name in selected_names:
        path = root / "post_probes" / f"{name}.json"
        if path.is_file():
            probes[name] = read(path)["reports"]
    curves = {
        name: training_curve(root / "logs" / f"train_{name}.log")
        for name in selected_names
    }
    result = {
        "schema_version": "pi05-recovery-distill-audited-results-v1",
        "status": "COMPLETED-DEV",
        "split_contract": {
            "train": list(range(10, 40)),
            "hyperparameter_selection": list(range(40, 50)),
            "final": list(range(0, 10)),
        },
        "selected_runs": selected_names,
        "analysis": analysis,
        "probes": probes,
        "training_curves": curves,
        "evidence_boundary": report["claim_boundary"],
    }
    (root / "audited_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# π0.5 历史条件恢复：审计结果",
        "",
        "状态：`COMPLETED-DEV`。训练、配置选择和最终评测使用互不重叠的固定初始状态。",
        "",
        "| 配置 | Control | Slip | 首个盲区动作差 median | 15 步内重抓 | 最终 Slip 95% Wilson CI |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, row in analysis.items():
        mech = row["mechanism"]
        interval = row["slip_wilson95"]
        lines.append(
            f"| {name} | {row['control']['successes']}/10 | {row['slip']['successes']}/10 | "
            f"{mech['first_blind_action_max_abs_delta_median']:.6f} | "
            f"{mech['regrasp_within_15_steps']}/10 | "
            f"[{100 * interval[0]:.1f}%, {100 * interval[1]:.1f}%] |"
        )
    lines += ["", "## 冻结表示 probe", "", "| 配置 | 首个盲区 test | 三个盲区 query test |", "|---|---:|---:|"]
    for name, values in probes.items():
        lines.append(
            f"| {name} | {100 * values['first_blind']['test_accuracy']:.1f}% | "
            f"{100 * values['all_three_blind_queries']['test_accuracy']:.1f}% |"
        )
    lines += [
        "",
        "本结果只覆盖一个 LIBERO-10 任务、teleport 受控滑落和每种子 10 个最终初始状态，不能表述为 SOTA、通用恢复或真实机器人结果。",
    ]
    (root / "审计结果报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
