"""Audit the mixed cache's same-current-input / different-history contract."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from openpi.training.libero_history_dataset import LiberoHistoryDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    index = json.loads(args.index.read_text(encoding="utf-8"))
    dataset = LiberoHistoryDataset(
        str(args.index),
        split="train",
        variant="H5",
        action_horizon=10,
        current_image_dropout_probability=0.0,
    )
    candidates = {}
    for position, episode_id in enumerate(dataset.episode_ids):
        entry = index["episodes"][str(episode_id)]
        if entry.get("source_kind") != "controlled_slip_teacher":
            continue
        if entry.get("recovery_repeat_index") != 0:
            continue
        key = int(entry["source_episode_index"])
        candidates.setdefault(key, {})[entry["branch"]] = (position, entry)
    episode_index = next(key for key, value in candidates.items() if set(value) == {"control", "slip"})
    samples = {}
    query_steps = {}
    for branch, (position, entry) in candidates[episode_index].items():
        dropout = np.load(entry["arrays"]["current_image_dropout"])
        query_step = int(np.flatnonzero(dropout)[0])
        query_steps[branch] = query_step
        samples[branch] = dataset[dataset.offsets[position] + query_step]
    checks = {
        "same_local_query_step": query_steps["control"] == query_steps["slip"],
        "control_current_base_zero": not np.any(samples["control"]["observation/image"]),
        "slip_current_base_zero": not np.any(samples["slip"]["observation/image"]),
        "control_current_wrist_zero": not np.any(samples["control"]["observation/wrist_image"]),
        "slip_current_wrist_zero": not np.any(samples["slip"]["observation/wrist_image"]),
        "same_current_state_le_1e6": float(
            np.max(
                np.abs(
                    samples["control"]["observation/state"]
                    - samples["slip"]["observation/state"]
                )
            )
        ) <= 1e-6,
        "different_history_base": not np.array_equal(
            samples["control"]["history"]["image_features"]["base_0_rgb"],
            samples["slip"]["history"]["image_features"]["base_0_rgb"],
        ),
        "query_outcome_preserved": bool(
            np.any(samples["slip"]["history"]["image_features"]["base_0_rgb"][7, 1])
        ),
        "oracle_labels_not_model_inputs": all(
            key not in samples["slip"]
            for key in ("object_in_gripper", "slip_observed", "should_regrasp")
        ),
    }
    result = {
        "schema_version": "pi05-recovery-distill-cache-audit-v1",
        "episode_index": episode_index,
        "query_steps": query_steps,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
