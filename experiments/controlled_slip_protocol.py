"""Machine-checkable contract for a paired grasp-slip and camera-loss evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    condition: str
    task_id: int
    initial_state_index: int
    episode_seed: int
    observed_grasp_query_step: int
    intervention_step: int
    first_blind_query_step: int
    action_prefix_sha256: str
    robot_state_at_first_blind_query: tuple[float, ...]
    object_displacement_xyz: tuple[float, float, float]


def audit_pair(control: PairRecord, slip: PairRecord, *, state_tolerance: float) -> dict:
    state_delta = max(
        abs(left - right)
        for left, right in zip(
            control.robot_state_at_first_blind_query,
            slip.robot_state_at_first_blind_query,
            strict=True,
        )
    )
    checks = {
        "condition_labels": control.condition == "control" and slip.condition == "slip",
        "same_pair_id": control.pair_id == slip.pair_id,
        "same_task_and_initial_state": (
            control.task_id,
            control.initial_state_index,
            control.episode_seed,
        )
        == (slip.task_id, slip.initial_state_index, slip.episode_seed),
        "same_action_prefix": control.action_prefix_sha256 == slip.action_prefix_sha256,
        "grasp_observed_before_intervention": (
            control.observed_grasp_query_step < control.intervention_step
            and slip.observed_grasp_query_step < slip.intervention_step
        ),
        "intervention_before_camera_loss": (
            control.intervention_step < control.first_blind_query_step
            and slip.intervention_step < slip.first_blind_query_step
        ),
        "control_has_no_object_displacement": control.object_displacement_xyz == (0.0, 0.0, 0.0),
        "slip_has_object_displacement": slip.object_displacement_xyz != (0.0, 0.0, 0.0),
        "matched_robot_state_at_first_blind_query": state_delta <= state_tolerance,
    }
    return {
        "schema_version": "pi05-controlled-slip-identifiability-audit-v1",
        "pair_id": control.pair_id,
        "state_tolerance": state_tolerance,
        "maximum_absolute_robot_state_delta": state_delta,
        "checks": checks,
        "identifiable": all(checks.values()),
        "control": asdict(control),
        "slip": asdict(slip),
    }


def smoke_record() -> tuple[PairRecord, PairRecord]:
    shared = dict(
        pair_id="smoke-task05-init00",
        task_id=5,
        initial_state_index=0,
        episode_seed=1002604896,
        observed_grasp_query_step=75,
        intervention_step=76,
        first_blind_query_step=80,
        action_prefix_sha256="0" * 64,
        robot_state_at_first_blind_query=(0.1, 0.2, 0.3, 3.0, 0.1, -0.2, 0.01, -0.01),
    )
    return (
        PairRecord(condition="control", object_displacement_xyz=(0.0, 0.0, 0.0), **shared),
        PairRecord(condition="slip", object_displacement_xyz=(0.04, 0.0, -0.02), **shared),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-output", type=Path, required=True)
    args = parser.parse_args()
    control, slip = smoke_record()
    result = audit_pair(control, slip, state_tolerance=1e-6)
    args.smoke_output.parent.mkdir(parents=True, exist_ok=True)
    args.smoke_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not result["identifiable"]:
        raise SystemExit(2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
