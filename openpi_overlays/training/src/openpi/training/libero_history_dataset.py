"""LIBERO-10 pilot dataset backed by per-episode frame and SigLIP caches."""

from __future__ import annotations

import bisect
import collections
import json
import pathlib

import numpy as np
from scipy.spatial.transform import Rotation

from openpi.history.visual_control import broadcast_current_features
from openpi.history.protocol import HistorySelection


CACHE_SCHEMA = "libero-history-cache-v1"


def libero_state_delta(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Delta for xyz + rotation-vector + two gripper coordinates."""
    delta = np.empty_like(before)
    delta[..., :3] = after[..., :3] - before[..., :3]
    before_rotation = Rotation.from_rotvec(before[..., 3:6])
    after_rotation = Rotation.from_rotvec(after[..., 3:6])
    delta[..., 3:6] = (after_rotation * before_rotation.inv()).as_rotvec()
    delta[..., 6:8] = after[..., 6:8] - before[..., 6:8]
    return delta


class LiberoHistoryDataset:
    def __init__(
        self,
        cache_index: str,
        *,
        split: str,
        variant: str,
        action_horizon: int,
        history_visual_source: str = "past",
        current_image_dropout_probability: float = 0.0,
        current_image_dropout_seed: int = 0,
    ):
        self.root = pathlib.Path(cache_index).resolve().parent
        self.index = json.loads(pathlib.Path(cache_index).read_text(encoding="utf-8"))
        if self.index["schema_version"] != CACHE_SCHEMA:
            raise ValueError(f"expected {CACHE_SCHEMA}")
        self.variant = variant
        self.action_horizon = action_horizon
        if history_visual_source not in {"past", "current"}:
            raise ValueError(f"unknown history visual source: {history_visual_source}")
        self.history_visual_source = history_visual_source
        if not 0.0 <= current_image_dropout_probability <= 1.0:
            raise ValueError("current_image_dropout_probability must be in [0, 1]")
        self.current_image_dropout_probability = current_image_dropout_probability
        self.current_image_dropout_seed = current_image_dropout_seed
        self.episode_ids = self.index["splits"][split]
        self.episodes = self.index["episodes"]
        lengths = [self.episodes[str(episode)]["length"] for episode in self.episode_ids]
        self.offsets = np.cumsum([0, *lengths]).tolist()
        self._arrays: collections.OrderedDict[tuple[tuple[str, str], ...], dict[str, np.ndarray]] = (
            collections.OrderedDict()
        )
        self._array_cache_size = 64

    def __len__(self) -> int:
        return self.offsets[-1]

    def _episode_step(self, index: int) -> tuple[int, int]:
        position = bisect.bisect_right(self.offsets, index) - 1
        return self.episode_ids[position], index - self.offsets[position]

    def _load_episode(self, episode_id: int) -> dict[str, np.ndarray]:
        entry = self.episodes[str(episode_id)]
        cache_key = tuple(sorted((str(key), str(path)) for key, path in entry["arrays"].items()))
        if cache_key not in self._arrays:
            self._arrays[cache_key] = {
                key: np.load(self.root / path, mmap_mode="r")
                for key, path in entry["arrays"].items()
                if self.variant != "H0" or not key.endswith("features")
            }
            if len(self._arrays) > self._array_cache_size:
                self._arrays.popitem(last=False)
        self._arrays.move_to_end(cache_key)
        return self._arrays[cache_key]

    def __getitem__(self, index: int) -> dict:
        episode_id, step = self._episode_step(index)
        entry = self.episodes[str(episode_id)]
        arrays = self._load_episode(episode_id)
        actions = np.asarray(arrays["action"][step : step + self.action_horizon])
        if len(actions) < self.action_horizon:
            actions = np.concatenate(
                [actions, np.repeat(actions[-1:], self.action_horizon - len(actions), axis=0)], axis=0
            )
        dropout_uniform = np.random.default_rng(
            np.random.SeedSequence([self.current_image_dropout_seed, episode_id, step])
        ).random()
        forced_dropout = (
            bool(arrays["current_image_dropout"][step])
            if "current_image_dropout" in arrays
            else False
        )
        drop_current_image = forced_dropout or dropout_uniform < self.current_image_dropout_probability
        base_image = np.asarray(arrays["image"][step])
        wrist_image = np.asarray(arrays["wrist_image"][step])
        sample = {
            "observation/image": np.zeros_like(base_image) if drop_current_image else base_image,
            "observation/wrist_image": np.zeros_like(wrist_image) if drop_current_image else wrist_image,
            "observation/state": np.asarray(arrays["state"][step]),
            "actions": actions,
            "prompt": entry["task"],
        }
        if self.variant != "H0":
            sample["history"] = self._history(
                arrays,
                step,
                mask_query_visual=drop_current_image,
                preserve_query_outcome=bool(entry.get("preserve_query_outcome_in_history", False)),
            )
        return sample

    def _history(
        self,
        arrays: dict[str, np.ndarray],
        query_step: int,
        *,
        mask_query_visual: bool,
        preserve_query_outcome: bool,
    ) -> dict:
        mode = HistorySelection.RECENT8 if self.variant == "H4" else HistorySelection.EARLY2_RECENT6
        selected = HistorySelection.indices(query_step, mode)
        count = len(selected)
        record_mask = np.zeros(8, dtype=np.bool_)
        record_mask[:count] = True
        times = np.zeros(8, dtype=np.int32)
        times[:count] = selected

        def paired_features(name: str) -> np.ndarray:
            source = arrays[name]
            result = np.zeros((8, 2, *source.shape[1:]), dtype=source.dtype)
            if count:
                selected_array = np.asarray(selected)
                result[:count, 0] = source[selected_array]
                result[:count, 1] = source[selected_array + 1]
                if mask_query_visual and not preserve_query_outcome:
                    result[np.flatnonzero(selected_array == query_step), 0] = 0
                    result[np.flatnonzero(selected_array + 1 == query_step), 1] = 0
            return result

        def selected_values(name: str) -> np.ndarray:
            source = arrays[name]
            result = np.zeros((8, *source.shape[1:]), dtype=source.dtype)
            if count:
                result[:count] = source[np.asarray(selected)]
            return result

        state = selected_values("state")
        action = selected_values("action")
        state_delta = np.zeros_like(state)
        if count:
            selected_array = np.asarray(selected)
            state_delta[:count] = libero_state_delta(
                arrays["state"][selected_array], arrays["state"][selected_array + 1]
            )
        image_mask = np.repeat(record_mask[:, None], 2, axis=1)
        if self.history_visual_source == "past":
            base_features = paired_features("image_features")
            wrist_features = paired_features("wrist_features")
        else:
            base_query_features = arrays["image_features"][query_step]
            wrist_query_features = arrays["wrist_features"][query_step]
            if mask_query_visual:
                base_query_features = np.zeros_like(base_query_features)
                wrist_query_features = np.zeros_like(wrist_query_features)
            current = broadcast_current_features(
                {
                    "base_0_rgb": base_query_features,
                    "left_wrist_0_rgb": wrist_query_features,
                },
                record_mask,
            )
            base_features = current["base_0_rgb"]
            wrist_features = current["left_wrist_0_rgb"]
        return {
            "image": {},
            "image_features": {
                "base_0_rgb": base_features,
                "left_wrist_0_rgb": wrist_features,
            },
            "image_mask": {
                "base_0_rgb": image_mask,
                "left_wrist_0_rgb": image_mask,
            },
            "record_mask": record_mask,
            "state": state,
            "action": action,
            "state_delta": state_delta,
            "numeric_mask": np.repeat(record_mask[:, None], 3, axis=1),
            "state_dim_mask": np.repeat(record_mask[:, None], state.shape[-1], axis=1),
            "action_dim_mask": np.repeat(record_mask[:, None], action.shape[-1], axis=1),
            "time": times,
            "query_time": np.asarray(query_step, dtype=np.int32),
        }
