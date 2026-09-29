"""Build a mixed LIBERO cache with paired controlled-slip recovery windows."""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-index", type=pathlib.Path, required=True)
    parser.add_argument("--recovery-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--base-params-dir", type=pathlib.Path, required=True)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--episode-min", type=int, default=10)
    parser.add_argument("--episode-max-exclusive", type=int, default=40)
    args = parser.parse_args()

    base_index_path = args.base_index.resolve()
    base_root = base_index_path.parent
    base = json.loads(base_index_path.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "norm_stats").mkdir()
    shutil.copy2(base_root / base["norm_stats_dir"] / "norm_stats.json", output / "norm_stats/norm_stats.json")

    episodes: dict[str, dict] = {}
    train_ids: list[int] = []
    validation_ids: list[int] = []
    next_id = 0

    def add_entry(entry: dict, split: str) -> int:
        nonlocal next_id
        episode_id = next_id
        next_id += 1
        episodes[str(episode_id)] = entry
        (train_ids if split == "train" else validation_ids).append(episode_id)
        return episode_id

    for split in ("train", "validation"):
        for source_id in base["splits"][split]:
            source = base["episodes"][str(source_id)]
            entry = {key: value for key, value in source.items() if key != "arrays"}
            entry["source_episode_id"] = source_id
            entry["source_kind"] = "libero_demonstration"
            entry["arrays"] = {
                key: str((base_root / relative).resolve()) for key, relative in source["arrays"].items()
            }
            add_entry(entry, split)

    windows: dict[tuple[int, str], tuple[pathlib.Path, dict]] = {}
    for metadata_path in sorted(args.recovery_root.resolve().glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        windows[(int(metadata["episode_index"]), str(metadata["branch"]))] = (
            metadata_path.parent,
            metadata,
        )
    paired_indices = sorted(
        episode_index
        for episode_index in {key[0] for key in windows}
        if args.episode_min <= episode_index < args.episode_max_exclusive
        if (episode_index, "control") in windows
        and (episode_index, "slip") in windows
        and windows[(episode_index, "control")][1]["success"]
        and windows[(episode_index, "slip")][1]["success"]
    )
    if not paired_indices:
        raise ValueError("no paired successful control/slip recovery windows")

    recovery_template_entries: list[dict] = []
    for episode_index in paired_indices:
        for branch in ("control", "slip"):
            episode_dir, metadata = windows[(episode_index, branch)]
            entry = {
                "length": int(metadata["length"]),
                "task_index": int(metadata["task_id"]),
                "task": metadata["task"],
                "source_kind": "controlled_slip_teacher",
                "source_episode_index": episode_index,
                "branch": branch,
                "teacher_success": True,
                "preserve_query_outcome_in_history": True,
                "arrays": {
                    key: str((episode_dir / relative).resolve())
                    for key, relative in metadata["arrays"].items()
                },
            }
            recovery_template_entries.append(entry)

    for repeat_index in range(args.repeat):
        for template in recovery_template_entries:
            entry = dict(template)
            entry["recovery_repeat_index"] = repeat_index
            entry["arrays"] = dict(template["arrays"])
            add_entry(entry, "train")

    result = {
        "schema_version": "libero-history-cache-v1",
        "dataset": {
            **base["dataset"],
            "mixture": "official_libero10_plus_controlled_slip_teacher",
            "recovery_pairs": len(paired_indices),
            "recovery_repeat": args.repeat,
        },
        "split": {
            **base["split"],
            "recovery_pair_indices": paired_indices,
            "recovery_repeat": args.repeat,
        },
        "splits": {"train": train_ids, "validation": validation_ids},
        "norm_stats_dir": "norm_stats",
        "vision_cache": {
            **base["vision_cache"],
            "base_params_dir": str(args.base_params_dir.resolve()),
        },
        "episodes": episodes,
    }
    index_path = output / "index.json"
    index_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "index": str(index_path),
        "base_train_episodes": len(base["splits"]["train"]),
        "base_validation_episodes": len(base["splits"]["validation"]),
        "recovery_pair_indices": paired_indices,
        "recovery_physical_windows": len(recovery_template_entries),
        "recovery_virtual_episodes": len(recovery_template_entries) * args.repeat,
        "train_samples": sum(episodes[str(index)]["length"] for index in train_ids),
        "validation_samples": sum(episodes[str(index)]["length"] for index in validation_ids),
    }
    (output / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
