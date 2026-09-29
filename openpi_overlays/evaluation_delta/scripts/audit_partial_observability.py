"""Audit whether clean rollouts can support the post-grasp blackout experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float | None:
    recalls = []
    for value in (0, 1):
        mask = labels == value
        if np.any(mask):
            recalls.append(float(np.mean(predictions[mask] == labels[mask])))
    return float(np.mean(recalls)) if len(recalls) == 2 else None


def _fit_logistic(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    mean = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    scale[scale < 1e-6] = 1.0
    normalized_train = (train_x - mean) / scale
    normalized_test = (test_x - mean) / scale
    design = np.concatenate([normalized_train, np.ones((len(normalized_train), 1))], axis=1)
    test_design = np.concatenate([normalized_test, np.ones((len(normalized_test), 1))], axis=1)
    weights = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(1500):
        logits = np.clip(design @ weights, -20.0, 20.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        regularization = np.concatenate([weights[:-1], np.zeros(1)]) * 1e-2
        gradient = design.T @ (probabilities - train_y) / len(train_y) + regularization
        weights -= 0.05 * gradient
    return (test_design @ weights >= 0.0).astype(np.int32)


def _state_probe(rows: list[dict]) -> dict:
    labels = np.asarray([row["grasp_success"] for row in rows], dtype=np.int32)
    features = np.asarray([row["blackout_query_state"] for row in rows], dtype=np.float64)
    tasks = np.asarray([row["task_id"] for row in rows], dtype=np.int32)
    predictions = np.full(len(rows), -1, dtype=np.int32)
    folds = []
    for task_id in sorted(set(tasks.tolist())):
        test = tasks == task_id
        train = ~test
        if len(set(labels[train].tolist())) < 2:
            continue
        predictions[test] = _fit_logistic(features[train], labels[train], features[test])
        folds.append(int(task_id))
    evaluated = predictions >= 0
    score = _balanced_accuracy(labels[evaluated], predictions[evaluated]) if np.any(evaluated) else None
    return {
        "method": "leave_one_task_out_l2_logistic",
        "features": "current_robot_state_at_candidate_blackout_query",
        "fold_tasks": folds,
        "evaluated_rows": int(np.sum(evaluated)),
        "balanced_accuracy": score,
    }


def _audit_result(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["observation_intervention"] != "none":
        raise ValueError(f"audit input must be a clean result: {path}")
    rows = []
    scheduled = 0
    complete = 0
    for episode in payload["episodes"]:
        schedule = episode["candidate_blackout_schedule"]
        if schedule is None:
            continue
        scheduled += 1
        query_map = {int(row["query_step"]): row for row in episode["selected_query_audit"]}
        clean_step = int(schedule["clean_outcome_query_step"])
        black_step = int(schedule["blackout_policy_query_steps"][0])
        if clean_step not in query_map or black_step not in query_map:
            continue
        complete += 1
        rows.append(
            {
                "result": str(path.resolve()),
                "task_id": int(episode["task_id"]),
                "episode_index": int(episode["episode_index"]),
                "grasp_success": bool(query_map[clean_step]["oracle_grasped_objects"]),
                "grasped_objects": query_map[clean_step]["oracle_grasped_objects"],
                "blackout_query_state": query_map[black_step]["current_state"],
                "clean_outcome_query_step": clean_step,
                "candidate_blackout_query_step": black_step,
            }
        )
    total = len(payload["episodes"])
    successes = sum(row["grasp_success"] for row in rows)
    return (
        {
            "path": str(path.resolve()),
            "variant": payload["variant"],
            "history_visual_source": payload["history_visual_source"],
            "episodes": total,
            "schedule_coverage": scheduled / total,
            "complete_audit_coverage": complete / total,
            "grasp_successes": successes,
            "grasp_failures": len(rows) - successes,
            "grasp_success_rate": successes / len(rows) if rows else None,
        },
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result_audits = []
    rows = []
    for path in args.result:
        audit, result_rows = _audit_result(path)
        result_audits.append(audit)
        rows.extend(result_rows)

    labels = [row["grasp_success"] for row in rows]
    success_rate = sum(labels) / len(labels) if labels else None
    probe = _state_probe(rows) if labels and len(set(labels)) == 2 else {
        "method": "leave_one_task_out_l2_logistic",
        "features": "current_robot_state_at_candidate_blackout_query",
        "fold_tasks": [],
        "evaluated_rows": 0,
        "balanced_accuracy": None,
    }
    gates = {
        "schedule_coverage_at_least_95_percent": all(
            row["schedule_coverage"] >= 0.95 for row in result_audits
        ),
        "complete_audit_coverage_at_least_95_percent": all(
            row["complete_audit_coverage"] >= 0.95 for row in result_audits
        ),
        "both_natural_outcomes_at_least_20_percent": (
            success_rate is not None and 0.20 <= success_rate <= 0.80
        ),
        "current_state_probe_at_most_65_percent": (
            probe["balanced_accuracy"] is not None and probe["balanced_accuracy"] <= 0.65
        ),
    }
    payload = {
        "schema_version": "pi05-partial-observability-audit-v1",
        "results": result_audits,
        "pooled_rows": len(rows),
        "pooled_grasp_success_rate": success_rate,
        "current_state_probe": probe,
        "gates": gates,
        "approved_for_natural_outcome_blackout": all(gates.values()),
        "next_step": (
            "run_paired_blackout"
            if all(gates.values())
            else "design_controlled_slip_before_model_comparison"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
