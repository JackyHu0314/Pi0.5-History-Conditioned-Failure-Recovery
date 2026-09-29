from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from openpi.training.libero_history_dataset import LiberoHistoryDataset


def _cache(tmp_path):
    length = 12
    arrays = {
        "image": np.broadcast_to(
            (1 + np.arange(length, dtype=np.uint8))[:, None, None, None], (length, 4, 4, 3)
        ).copy(),
        "wrist_image": np.broadcast_to(
            (101 + np.arange(length, dtype=np.uint8))[:, None, None, None], (length, 4, 4, 3)
        ).copy(),
        "state": np.zeros((length, 8), dtype=np.float32),
        "action": np.zeros((length, 7), dtype=np.float32),
        "image_features": np.broadcast_to(
            np.arange(length, dtype=np.float16)[:, None, None], (length, 3, 4)
        ).copy(),
        "wrist_features": np.broadcast_to(
            (100 + np.arange(length, dtype=np.float16))[:, None, None], (length, 3, 4)
        ).copy(),
    }
    paths = {}
    for name, value in arrays.items():
        path = tmp_path / f"{name}.npy"
        np.save(path, value)
        paths[name] = path.name
    index = {
        "schema_version": "libero-history-cache-v1",
        "splits": {"train": [0]},
        "episodes": {"0": {"length": length, "task": "test", "arrays": paths}},
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return index_path


def test_past_default_is_unchanged(tmp_path):
    index = _cache(tmp_path)
    default = LiberoHistoryDataset(str(index), split="train", variant="H5", action_horizon=10)
    explicit = LiberoHistoryDataset(
        str(index), split="train", variant="H5", action_horizon=10, history_visual_source="past"
    )
    for key, value in default[5]["history"].items():
        if isinstance(value, dict):
            for camera in value:
                np.testing.assert_array_equal(value[camera], explicit[5]["history"][key][camera])
        else:
            np.testing.assert_array_equal(value, explicit[5]["history"][key])


def test_current_uses_query_only_and_preserves_nonvisual_history(tmp_path):
    index = _cache(tmp_path)
    past = LiberoHistoryDataset(
        str(index), split="train", variant="H5", action_horizon=10, history_visual_source="past"
    )[5]["history"]
    current = LiberoHistoryDataset(
        str(index), split="train", variant="H5", action_horizon=10, history_visual_source="current"
    )[5]["history"]
    count = int(current["record_mask"].sum())
    np.testing.assert_array_equal(current["image_features"]["base_0_rgb"][:count], 5)
    np.testing.assert_array_equal(current["image_features"]["left_wrist_0_rgb"][:count], 105)
    assert not np.any(current["image_features"]["base_0_rgb"] == 6)
    for key in ("image_mask", "record_mask", "state", "action", "state_delta", "numeric_mask", "time", "query_time"):
        if isinstance(past[key], dict):
            for camera in past[key]:
                np.testing.assert_array_equal(past[key][camera], current[key][camera])
        else:
            np.testing.assert_array_equal(past[key], current[key])
    np.testing.assert_array_equal(current["image_features"]["base_0_rgb"][count:], 0)


def test_current_empty_history_stays_zero(tmp_path):
    index = _cache(tmp_path)
    history = LiberoHistoryDataset(
        str(index), split="train", variant="H5", action_horizon=10, history_visual_source="current"
    )[0]["history"]
    assert not history["record_mask"].any()
    assert not history["image_features"]["base_0_rgb"].any()


def test_full_dropout_masks_raw_and_query_features_for_past_history(tmp_path):
    index = _cache(tmp_path)
    sample = LiberoHistoryDataset(
        str(index),
        split="train",
        variant="H5",
        action_horizon=10,
        history_visual_source="past",
        current_image_dropout_probability=1.0,
        current_image_dropout_seed=20260928,
    )[5]
    assert not sample["observation/image"].any()
    assert not sample["observation/wrist_image"].any()
    history = sample["history"]
    count = int(history["record_mask"].sum())
    assert count > 1
    np.testing.assert_array_equal(history["image_features"]["base_0_rgb"][count - 1, 1], 0)
    np.testing.assert_array_equal(history["image_features"]["left_wrist_0_rgb"][count - 1, 1], 0)
    assert history["image_features"]["base_0_rgb"][count - 2].any()
    assert history["image_mask"]["base_0_rgb"][:count].all()


def test_full_dropout_masks_broadcast_current_features(tmp_path):
    index = _cache(tmp_path)
    sample = LiberoHistoryDataset(
        str(index),
        split="train",
        variant="H5",
        action_horizon=10,
        history_visual_source="current",
        current_image_dropout_probability=1.0,
        current_image_dropout_seed=20260928,
    )[5]
    history = sample["history"]
    assert not history["image_features"]["base_0_rgb"].any()
    assert not history["image_features"]["left_wrist_0_rgb"].any()
    assert history["image_mask"]["base_0_rgb"][history["record_mask"]].all()


def test_zero_dropout_preserves_current_observation(tmp_path):
    index = _cache(tmp_path)
    sample = LiberoHistoryDataset(
        str(index),
        split="train",
        variant="H0",
        action_horizon=10,
        current_image_dropout_probability=0.0,
        current_image_dropout_seed=20260928,
    )[5]
    np.testing.assert_array_equal(sample["observation/image"], 6)
    np.testing.assert_array_equal(sample["observation/wrist_image"], 106)


def test_dropout_sets_are_deterministic_and_nested(tmp_path):
    index = _cache(tmp_path)
    quarter = LiberoHistoryDataset(
        str(index),
        split="train",
        variant="H0",
        action_horizon=10,
        current_image_dropout_probability=0.25,
        current_image_dropout_seed=20260928,
    )
    half = LiberoHistoryDataset(
        str(index),
        split="train",
        variant="H0",
        action_horizon=10,
        current_image_dropout_probability=0.5,
        current_image_dropout_seed=20260928,
    )
    quarter_mask = np.asarray([not quarter[index]["observation/image"].any() for index in range(len(quarter))])
    repeated_mask = np.asarray([not quarter[index]["observation/image"].any() for index in range(len(quarter))])
    half_mask = np.asarray([not half[index]["observation/image"].any() for index in range(len(half))])
    np.testing.assert_array_equal(quarter_mask, repeated_mask)
    assert np.all(~quarter_mask | half_mask)
    assert quarter_mask.any()
    assert half_mask.any()


if __name__ == "__main__":
    for test in (
        test_past_default_is_unchanged,
        test_current_uses_query_only_and_preserves_nonvisual_history,
        test_current_empty_history_stays_zero,
        test_full_dropout_masks_raw_and_query_features_for_past_history,
        test_full_dropout_masks_broadcast_current_features,
        test_zero_dropout_preserves_current_observation,
        test_dropout_sets_are_deterministic_and_nested,
    ):
        with tempfile.TemporaryDirectory() as directory:
            test(Path(directory))
    print("7 current-frame and dropout tests passed")
