"""Evaluate whether frozen R7 history tokens linearly encode the controlled slip."""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config as training_config
from openpi.training.libero_history_dataset import libero_state_delta


def history_payload(root: pathlib.Path, metadata: dict, query_step: int) -> dict:
    arrays = {key: np.load(root / value, mmap_mode="r") for key, value in metadata["arrays"].items()}
    selected = tuple(range(max(0, query_step - 6), query_step))
    selected = tuple(sorted({*range(min(2, query_step)), *selected}))
    count = len(selected)
    record_mask = np.zeros(8, dtype=np.bool_)
    record_mask[:count] = True
    image_features = {}
    image_masks = {}
    for output_name, source_name in (
        ("base_0_rgb", "image_features"),
        ("left_wrist_0_rgb", "wrist_features"),
    ):
        source = arrays[source_name]
        value = np.zeros((8, 2, *source.shape[1:]), dtype=source.dtype)
        if count:
            indices = np.asarray(selected)
            value[:count, 0] = source[indices]
            value[:count, 1] = source[indices + 1]
        image_features[output_name] = value
        image_masks[output_name] = np.repeat(record_mask[:, None], 2, axis=1)
    states = np.zeros((8, 8), dtype=np.float32)
    actions = np.zeros((8, 7), dtype=np.float32)
    state_deltas = np.zeros((8, 8), dtype=np.float32)
    times = np.zeros(8, dtype=np.int32)
    if count:
        indices = np.asarray(selected)
        states[:count] = arrays["state"][indices]
        actions[:count] = arrays["action"][indices]
        state_deltas[:count] = libero_state_delta(arrays["state"][indices], arrays["state"][indices + 1])
        times[:count] = indices
    return {
        "image": {},
        "image_features": image_features,
        "image_mask": image_masks,
        "record_mask": record_mask,
        "state": states,
        "action": actions,
        "state_delta": state_deltas,
        "numeric_mask": np.repeat(record_mask[:, None], 3, axis=1),
        "state_dim_mask": np.repeat(record_mask[:, None], 8, axis=1),
        "action_dim_mask": np.repeat(record_mask[:, None], 7, axis=1),
        "time": times,
        "query_time": np.asarray(query_step, dtype=np.int32),
    }


def sample(root: pathlib.Path, metadata: dict, query_step: int) -> dict:
    arrays = {key: np.load(root / value, mmap_mode="r") for key, value in metadata["arrays"].items()}
    return {
        "observation/image": np.zeros_like(arrays["image"][query_step]),
        "observation/wrist_image": np.zeros_like(arrays["wrist_image"][query_step]),
        "observation/state": np.asarray(arrays["state"][query_step]),
        "prompt": metadata["task"],
        "history": history_payload(root, metadata, query_step),
    }


def ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, regularization: float) -> np.ndarray:
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-6] = 1.0
    x = (train_x - mean) / scale
    z = (test_x - mean) / scale
    x = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    z = np.concatenate([z, np.ones((len(z), 1))], axis=1)
    labels = train_y.astype(np.float64) * 2.0 - 1.0
    dual = np.linalg.solve(x @ x.T + regularization * np.eye(len(x)), labels)
    return (z @ x.T @ dual >= 0.0).astype(np.int32)


def accuracy(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(prediction == target))


def wilson(successes: int, count: int) -> tuple[float, float]:
    z = 1.959963984540054
    rate = successes / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    radius = z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / denominator
    return center - radius, center + radius


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-manifest", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--train-root", type=pathlib.Path, required=True)
    parser.add_argument("--dev-root", type=pathlib.Path, required=True)
    parser.add_argument("--test-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    config = training_config.make_libero_history_train_config_from_run_manifest(args.run_manifest)
    policy = policy_config.create_trained_policy(config, args.checkpoint_dir)
    embed_history = nnx_utils.module_jit(policy._model._embed_history)  # noqa: SLF001

    rows = []
    for split, dataset_root in (
        ("train", args.train_root.resolve()),
        ("dev", args.dev_root.resolve()),
        ("test", args.test_root.resolve()),
    ):
        for metadata_path in sorted(dataset_root.glob("*/metadata.json")):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            episode_index = int(metadata["episode_index"])
            if split == "train" and not 10 <= episode_index < 40:
                continue
            if split == "dev" and not 40 <= episode_index < 50:
                continue
            if split == "test" and not 0 <= episode_index < 10:
                continue
            first_blind = min(metadata["blackout_local_steps"])
            for query_offset in (0, 5, 10):
                query_step = first_blind + query_offset
                inputs = policy._input_transform(sample(metadata_path.parent, metadata, query_step))  # noqa: SLF001
                inputs = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], inputs)
                observation = model_lib.Observation.from_dict(inputs)
                observation = model_lib.preprocess_observation(None, observation, train=False)
                tokens, mask = embed_history(observation)
                tokens = np.asarray(tokens[0], dtype=np.float32)
                mask = np.asarray(mask[0], dtype=np.bool_)
                valid = tokens[mask]
                feature = np.concatenate([valid.mean(axis=0), valid.std(axis=0)])
                rows.append(
                    {
                        "split": split,
                        "episode_index": episode_index,
                        "branch": metadata["branch"],
                        "query_offset": query_offset,
                        "label": int(metadata["branch"] == "slip"),
                        "feature": feature,
                    }
                )

    regularizations = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]
    reports = {}
    for condition, selected_rows in (
        ("first_blind", [row for row in rows if row["query_offset"] == 0]),
        ("all_three_blind_queries", rows),
    ):
        by_split = {split: [row for row in selected_rows if row["split"] == split] for split in ("train", "dev", "test")}
        arrays = {}
        for split, values in by_split.items():
            arrays[split] = (
                np.stack([row["feature"] for row in values]),
                np.asarray([row["label"] for row in values], dtype=np.int32),
            )
        train_x, train_y = arrays["train"]
        dev_x, dev_y = arrays["dev"]
        selected_regularization = max(
            regularizations,
            key=lambda value: (accuracy(ridge_predict(train_x, train_y, dev_x, value), dev_y), -value),
        )
        combined_x = np.concatenate([train_x, dev_x])
        combined_y = np.concatenate([train_y, dev_y])
        test_x, test_y = arrays["test"]
        test_prediction = ridge_predict(combined_x, combined_y, test_x, selected_regularization)
        successes = int(np.sum(test_prediction == test_y))
        low, high = wilson(successes, len(test_y))
        reports[condition] = {
            "selected_regularization": selected_regularization,
            "train_samples": len(train_y),
            "dev_samples": len(dev_y),
            "test_samples": len(test_y),
            "dev_accuracy": accuracy(
                ridge_predict(train_x, train_y, dev_x, selected_regularization), dev_y
            ),
            "test_accuracy": successes / len(test_y),
            "test_correct": successes,
            "test_wilson95": [low, high],
            "test_predictions": test_prediction.tolist(),
            "test_labels": test_y.tolist(),
        }
    output = {
        "schema_version": "pi05-r7-controlled-slip-linear-probe-v1",
        "checkpoint": str(args.checkpoint_dir.resolve()),
        "representation": "mean_and_std_pool_of_valid_16x2048_history_tokens",
        "split_contract": {
            "train_initial_states": sorted({row["episode_index"] for row in rows if row["split"] == "train"}),
            "dev_initial_states": sorted({row["episode_index"] for row in rows if row["split"] == "dev"}),
            "test_initial_states": sorted({row["episode_index"] for row in rows if row["split"] == "test"}),
        },
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
