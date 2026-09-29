"""Timing protocol for paired post-grasp observation interventions."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses


@dataclasses.dataclass(frozen=True)
class DualCameraBlackoutSchedule:
    """Black out sensor observations after one clean post-close policy query.

    Observation steps index the frame available after that many environment
    actions. Policy queries occur every ``replan_steps`` control steps.
    """

    first_close_plan_query_step: int
    first_post_close_query_step: int
    clean_outcome_query_step: int
    blackout_observation_start_step: int
    blackout_observation_end_step_exclusive: int
    blackout_policy_query_steps: tuple[int, ...]
    replan_steps: int
    blackout_queries: int

    @classmethod
    def from_first_close(
        cls,
        first_close_plan_query_step: int,
        *,
        replan_steps: int,
        blackout_queries: int,
    ) -> "DualCameraBlackoutSchedule":
        if first_close_plan_query_step < 0:
            raise ValueError("first_close_plan_query_step must be non-negative")
        if replan_steps < 1:
            raise ValueError("replan_steps must be positive")
        if blackout_queries < 1:
            raise ValueError("blackout_queries must be positive")
        if first_close_plan_query_step % replan_steps:
            raise ValueError("first close must belong to a policy-query boundary")

        first_post_close_query = first_close_plan_query_step + replan_steps
        clean_outcome_query = first_post_close_query + replan_steps
        first_black_query = clean_outcome_query + replan_steps
        black_queries = tuple(
            first_black_query + index * replan_steps for index in range(blackout_queries)
        )
        return cls(
            first_close_plan_query_step=first_close_plan_query_step,
            first_post_close_query_step=first_post_close_query,
            clean_outcome_query_step=clean_outcome_query,
            blackout_observation_start_step=clean_outcome_query + 1,
            blackout_observation_end_step_exclusive=black_queries[-1] + 1,
            blackout_policy_query_steps=black_queries,
            replan_steps=replan_steps,
            blackout_queries=blackout_queries,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "DualCameraBlackoutSchedule":
        return cls(
            first_close_plan_query_step=int(value["first_close_plan_query_step"]),
            first_post_close_query_step=int(value["first_post_close_query_step"]),
            clean_outcome_query_step=int(value["clean_outcome_query_step"]),
            blackout_observation_start_step=int(value["blackout_observation_start_step"]),
            blackout_observation_end_step_exclusive=int(
                value["blackout_observation_end_step_exclusive"]
            ),
            blackout_policy_query_steps=tuple(int(step) for step in value["blackout_policy_query_steps"]),
            replan_steps=int(value["replan_steps"]),
            blackout_queries=int(value["blackout_queries"]),
        )

    def is_blackout_observation(self, observation_step: int) -> bool:
        return (
            self.blackout_observation_start_step
            <= observation_step
            < self.blackout_observation_end_step_exclusive
        )

    def is_blackout_query(self, query_step: int) -> bool:
        return query_step in self.blackout_policy_query_steps

    def to_mapping(self) -> dict[str, int | list[int]]:
        return {
            "first_close_plan_query_step": self.first_close_plan_query_step,
            "first_post_close_query_step": self.first_post_close_query_step,
            "clean_outcome_query_step": self.clean_outcome_query_step,
            "blackout_observation_start_step": self.blackout_observation_start_step,
            "blackout_observation_end_step_exclusive": self.blackout_observation_end_step_exclusive,
            "blackout_policy_query_steps": list(self.blackout_policy_query_steps),
            "replan_steps": self.replan_steps,
            "blackout_queries": self.blackout_queries,
        }
