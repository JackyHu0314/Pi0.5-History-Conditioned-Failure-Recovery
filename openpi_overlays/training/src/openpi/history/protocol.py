"""Auditable data protocol for the pi0.5 H0--H5 history ablation.

The validation in this module deliberately avoids JAX so manifests can be audited
on data-preparation hosts.  Array conversion remains the responsibility of the
official openpi data loader and model boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import json
import pathlib
from typing import Any, Literal


SCHEMA_VERSION = "pi05-history-v1"
CONDITIONS = ("T0E0", "T1E0", "T0E1", "T1E1")
FORBIDDEN_MODEL_KEYS = frozenset(
    {
        "hidden_task_id",
        "correct_target",
        "response_mode",
        "success",
        "success_label",
        "scene_mode",
    }
)


@dataclasses.dataclass(frozen=True)
class CacheMetadata:
    vision_checkpoint: str
    preprocessing: str
    normalization: str
    feature_dim: int
    patches_per_image: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CacheMetadata":
        required = {field.name for field in dataclasses.fields(cls)}
        missing = required - value.keys()
        if missing:
            raise ValueError(f"cached history metadata missing fields: {sorted(missing)}")
        return cls(**{key: value[key] for key in required})

    def assert_matches(self, expected: "CacheMetadata") -> None:
        if self != expected:
            differences = {
                field.name: (getattr(expected, field.name), getattr(self, field.name))
                for field in dataclasses.fields(self)
                if getattr(self, field.name) != getattr(expected, field.name)
            }
            raise ValueError(f"history cache metadata mismatch (expected, actual): {differences}")


@dataclasses.dataclass(frozen=True)
class Transition:
    """One executed control transition; payload objects are intentionally generic."""

    episode_id: str
    start_step: int
    end_step: int
    observation_before: Any
    state_before: Any
    executed_action: Any
    observation_after: Any
    state_after: Any

    def validate_for_query(self, *, episode_id: str, query_step: int) -> None:
        if self.episode_id != episode_id:
            raise ValueError("history transition crosses episode boundary")
        if self.end_step != self.start_step + 1:
            raise ValueError(
                "history transition must be one real control step: "
                f"got ({self.start_step}, {self.end_step})"
            )
        if self.end_step > query_step:
            raise ValueError(
                f"future leakage: transition end {self.end_step} exceeds query step {query_step}"
            )


class HistorySelection:
    EARLY2_RECENT6 = "early2_recent6"
    RECENT8 = "recent8"

    @staticmethod
    def indices(num_records: int, mode: str, *, max_records: int = 8) -> tuple[int, ...]:
        if num_records < 0:
            raise ValueError("num_records must be non-negative")
        if max_records != 8:
            raise ValueError("H0--H5 v1 fixes max_records=8")
        if mode == HistorySelection.EARLY2_RECENT6:
            chosen = set(range(min(2, num_records)))
            chosen.update(range(max(0, num_records - 6), num_records))
            return tuple(sorted(chosen))
        if mode == HistorySelection.RECENT8:
            return tuple(range(max(0, num_records - 8), num_records))
        raise ValueError(f"unknown history selection mode: {mode}")

    @staticmethod
    def pad(indices: Sequence[int], *, max_records: int = 8) -> tuple[tuple[int, ...], tuple[bool, ...]]:
        if len(indices) > max_records:
            raise ValueError(f"selected {len(indices)} records, maximum is {max_records}")
        padded = tuple(indices) + (0,) * (max_records - len(indices))
        mask = (True,) * len(indices) + (False,) * (max_records - len(indices))
        return padded, mask


class HistoryBuffer:
    """Episode-scoped buffer used by an online environment adapter.

    `append` only accepts a complete observed transition.  The adapter must compute
    representation-aware state deltas before creating model arrays; this class does
    not subtract arbitrary states.
    """

    def __init__(self, selection: str = HistorySelection.EARLY2_RECENT6):
        self._selection = selection
        self._episode_id: str | None = None
        self._records: list[Transition] = []

    @property
    def episode_id(self) -> str | None:
        return self._episode_id

    def reset(self, episode_id: str) -> None:
        if not episode_id:
            raise ValueError("episode_id must be non-empty")
        self._episode_id = episode_id
        self._records.clear()

    def append(self, transition: Transition) -> None:
        if self._episode_id is None:
            raise RuntimeError("reset(episode_id) is required before append")
        transition.validate_for_query(episode_id=self._episode_id, query_step=transition.end_step)
        if self._records and transition.start_step < self._records[-1].end_step:
            raise ValueError("history transitions must be appended in non-overlapping time order")
        self._records.append(transition)

    def selected(self, *, query_step: int) -> tuple[Transition, ...]:
        if self._episode_id is None:
            raise RuntimeError("reset(episode_id) is required before reading history")
        for record in self._records:
            record.validate_for_query(episode_id=self._episode_id, query_step=query_step)
        indices = HistorySelection.indices(len(self._records), self._selection)
        return tuple(self._records[index] for index in indices)


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_manifest(path: str | pathlib.Path, *, expected_split: str | None = None) -> dict[str, Any]:
    path = pathlib.Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"history manifest does not exist: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest, expected_split=expected_split)
    manifest["_manifest_path"] = str(path.resolve())
    manifest["_manifest_sha256"] = stable_json_hash({k: v for k, v in manifest.items() if not k.startswith("_")})
    return manifest


def validate_manifest(manifest: Mapping[str, Any], *, expected_split: str | None = None) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"expected schema_version={SCHEMA_VERSION!r}")
    if manifest.get("validates_task_performance") is False and manifest.get("purpose") != "smoke":
        raise ValueError("non-smoke manifest cannot declare validates_task_performance=false")
    if manifest.get("purpose") == "smoke" and expected_split in {"train", "validation", "test"}:
        raise ValueError("smoke data is forbidden for a formal train/validation/test run")

    storage = manifest.get("history_storage")
    if not isinstance(storage, Mapping) or storage.get("mode") not in {"raw", "cached"}:
        raise ValueError("history_storage.mode must be explicitly 'raw' or 'cached'")
    if storage["mode"] == "cached":
        CacheMetadata.from_mapping(storage.get("cache_metadata", {}))
    elif storage.get("cache_metadata") is not None:
        raise ValueError("raw history must not carry cache_metadata")

    numeric_adapter = manifest.get("numeric_adapter")
    required_numeric = {"state_normalization", "action_normalization", "state_delta_adapter"}
    if not isinstance(numeric_adapter, Mapping) or not required_numeric.issubset(numeric_adapter):
        raise ValueError(
            "manifest numeric_adapter must name state/action normalization and the representation-aware state delta"
        )

    samples = manifest.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("manifest must contain a non-empty samples list")

    episode_split: dict[str, str] = {}
    scene_split: dict[str, str] = {}
    condition_counts = {condition: 0 for condition in CONDITIONS}
    for sample in samples:
        _validate_sample_metadata(sample, expected_split=expected_split)
        condition_counts[sample["condition"]] += 1
        for key, seen in (("episode_id", episode_split), ("scene_group_id", scene_split)):
            identifier, split = str(sample[key]), str(sample["split"])
            previous = seen.setdefault(identifier, split)
            if previous != split:
                raise ValueError(f"{key}={identifier!r} appears in both {previous!r} and {split!r}")

    positive_counts = {count for count in condition_counts.values()}
    if len(positive_counts) != 1:
        raise ValueError(f"four task conditions must be equally represented, got {condition_counts}")


def _validate_sample_metadata(sample: Mapping[str, Any], *, expected_split: str | None) -> None:
    required = {
        "sample_id",
        "file",
        "split",
        "episode_id",
        "scene_group_id",
        "condition",
        "query_step",
        "transitions",
    }
    missing = required - sample.keys()
    if missing:
        raise ValueError(f"sample metadata missing fields: {sorted(missing)}")
    if sample["condition"] not in CONDITIONS:
        raise ValueError(f"invalid task condition: {sample['condition']!r}")
    if expected_split is not None and sample["split"] != expected_split:
        raise ValueError(f"sample {sample['sample_id']} is not in requested split {expected_split!r}")
    query_step = int(sample["query_step"])
    previous_end = -1
    for transition in sample["transitions"]:
        start = int(transition["start_step"])
        end = int(transition["end_step"])
        if end != start + 1:
            raise ValueError(f"sample {sample['sample_id']} has a non-single-step transition")
        if end > query_step:
            raise ValueError(f"sample {sample['sample_id']} leaks a transition after query time")
        if start < previous_end:
            raise ValueError(f"sample {sample['sample_id']} transitions are not time ordered")
        previous_end = end


def assert_model_keys_safe(keys: Sequence[str]) -> None:
    normalized = {key.rsplit("/", 1)[-1].rsplit("__", 1)[-1] for key in keys}
    forbidden = normalized & FORBIDDEN_MODEL_KEYS
    if forbidden:
        raise ValueError(f"hidden/scoring fields cannot enter model arrays: {sorted(forbidden)}")


def validate_intervention(intervention: Mapping[str, Any]) -> None:
    kind = intervention.get("kind", "normal")
    allowed: set[str] = {
        "normal",
        "drop_all",
        "replace_early",
        "replace_recent",
        "permute_storage_only",
        "shuffle_time_semantics_ood",
    }
    if kind not in allowed:
        raise ValueError(f"unsupported history intervention: {kind!r}")
    if kind in {"replace_early", "replace_recent"} and not intervention.get("consistent_transition_source"):
        raise ValueError("counterfactual replacement requires a complete, internally consistent transition source")
    if kind == "shuffle_time_semantics_ood" and intervention.get("distribution") != "OOD":
        raise ValueError("changed time semantics must be explicitly labelled OOD")
