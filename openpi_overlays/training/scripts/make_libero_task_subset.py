"""Create a zero-copy single-task LIBERO cache index for overfit diagnosis."""

from __future__ import annotations

import argparse
import copy
import json
import pathlib

import numpy as np


def statistics(arrays: list[np.ndarray]) -> dict[str, list[float]]:
    values = np.concatenate(arrays, axis=0).astype(np.float64)
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index", type=pathlib.Path, required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    source_index_path = args.source_index.resolve()
    source_root = source_index_path.parent
    source = json.loads(source_index_path.read_text(encoding="utf-8"))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    splits = {
        split: [
            episode_id
            for episode_id in source["splits"][split]
            if source["episodes"][str(episode_id)]["task_index"] == args.task_index
        ]
        for split in ("train", "validation")
    }
    selected = splits["train"] + splits["validation"]
    episodes = {}
    for episode_id in selected:
        entry = copy.deepcopy(source["episodes"][str(episode_id)])
        entry["arrays"] = {
            name: str((source_root / path).resolve()) for name, path in entry["arrays"].items()
        }
        episodes[str(episode_id)] = entry

    train_states = [np.load(episodes[str(episode_id)]["arrays"]["state"]) for episode_id in splits["train"]]
    train_actions = [np.load(episodes[str(episode_id)]["arrays"]["action"]) for episode_id in splits["train"]]
    norm_stats_dir = output_dir / "norm_stats"
    norm_stats_dir.mkdir()
    (norm_stats_dir / "norm_stats.json").write_text(
        json.dumps(
            {
                "norm_stats": {
                    "state": statistics(train_states),
                    "actions": statistics(train_actions),
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    task = episodes[str(splits["train"][0])]["task"]
    subset = {
        "schema_version": source["schema_version"],
        "dataset": source["dataset"],
        "split": {
            **source["split"],
            "task_index": args.task_index,
            "task": task,
            "source_index": str(source_index_path),
        },
        "splits": splits,
        "norm_stats_dir": "norm_stats",
        "vision_cache": source["vision_cache"],
        "episodes": episodes,
    }
    (output_dir / "index.json").write_text(
        json.dumps(subset, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"task {args.task_index}: {task}; "
        f"{len(splits['train'])} train and {len(splits['validation'])} validation episodes"
    )


if __name__ == "__main__":
    main()
