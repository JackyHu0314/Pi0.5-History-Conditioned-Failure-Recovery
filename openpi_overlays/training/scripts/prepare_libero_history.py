"""Prepare the fixed LIBERO-10 history pilot and encode each frame once."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


DATASET_REVISION = "551d7d86f25edd0ffeda8b60053c15438cfd1d6a"
CACHE_SCHEMA = "libero-history-cache-v1"
CAMERAS = {
    "image": "observation.images.image",
    "wrist_image": "observation.images.wrist_image",
}


def _episode_metadata(dataset_root: pathlib.Path):
    import pandas as pd

    files = sorted((dataset_root / "meta" / "episodes").rglob("*.parquet"))
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def _task_texts(dataset_root: pathlib.Path) -> dict[int, str]:
    import pandas as pd

    tasks = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
    return {int(task_index): str(text) for text, task_index in zip(tasks.index, tasks["task_index"], strict=True)}


def _episode_tasks(dataset_root: pathlib.Path, episodes) -> dict[int, int]:
    import pandas as pd

    result = {}
    for (chunk_index, file_index), rows in episodes.groupby(["data/chunk_index", "data/file_index"]):
        path = dataset_root / "data" / f"chunk-{int(chunk_index):03d}" / f"file-{int(file_index):03d}.parquet"
        frame_data = pd.read_parquet(path, columns=["episode_index", "task_index"])
        for episode_id in rows["episode_index"]:
            values = frame_data.loc[frame_data["episode_index"] == episode_id, "task_index"]
            result[int(episode_id)] = int(values.iloc[0])
    return result


def _split_episodes(
    episode_tasks: dict[int, int], *, train_per_task: int, validation_per_task: int, split_seed: int
) -> tuple[list[int], list[int]]:
    train = []
    validation = []
    for task_index in sorted(set(episode_tasks.values())):
        task_episodes = np.asarray(
            sorted(episode for episode, task in episode_tasks.items() if task == task_index), dtype=np.int64
        )
        rng = np.random.default_rng(np.random.SeedSequence([split_seed, task_index]))
        shuffled = rng.permutation(task_episodes)
        train.extend(int(value) for value in shuffled[:train_per_task])
        validation.extend(
            int(value) for value in shuffled[train_per_task : train_per_task + validation_per_task]
        )
    return train, validation


def _save_numeric_arrays(dataset_root: pathlib.Path, output_dir: pathlib.Path, selected_rows, entries: dict) -> None:
    import pandas as pd

    for (chunk_index, file_index), rows in selected_rows.groupby(["data/chunk_index", "data/file_index"]):
        path = dataset_root / "data" / f"chunk-{int(chunk_index):03d}" / f"file-{int(file_index):03d}.parquet"
        frame_data = pd.read_parquet(path)
        for episode_id in rows["episode_index"]:
            episode_id = int(episode_id)
            episode_frames = frame_data.loc[frame_data["episode_index"] == episode_id].sort_values("frame_index")
            episode_dir = output_dir / "episodes" / f"{episode_id:06d}"
            state = np.stack(episode_frames["observation.state"].to_numpy()).astype(np.float32)
            action = np.stack(episode_frames["action"].to_numpy()).astype(np.float32)
            np.save(episode_dir / "state.npy", state)
            np.save(episode_dir / "action.npy", action)
            np.lib.format.open_memmap(
                episode_dir / "image.npy", mode="w+", dtype=np.uint8, shape=(len(state), 224, 224, 3)
            ).flush()
            np.lib.format.open_memmap(
                episode_dir / "wrist_image.npy", mode="w+", dtype=np.uint8, shape=(len(state), 224, 224, 3)
            ).flush()
            entries[str(episode_id)]["arrays"] = {
                "state": str((episode_dir / "state.npy").relative_to(output_dir)),
                "action": str((episode_dir / "action.npy").relative_to(output_dir)),
                "image": str((episode_dir / "image.npy").relative_to(output_dir)),
                "wrist_image": str((episode_dir / "wrist_image.npy").relative_to(output_dir)),
                "image_features": str((episode_dir / "image_features.npy").relative_to(output_dir)),
                "wrist_features": str((episode_dir / "wrist_features.npy").relative_to(output_dir)),
            }


def _decode_selected_video(
    dataset_root: pathlib.Path,
    output_dir: pathlib.Path,
    selected_rows,
    *,
    array_name: str,
    video_key: str,
) -> None:
    import av
    from openpi_client import image_tools

    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = int(info["fps"])
    intervals_by_file: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for row_dict in selected_rows.to_dict("records"):
        chunk = int(row_dict[f"videos/{video_key}/chunk_index"])
        file_index = int(row_dict[f"videos/{video_key}/file_index"])
        start = round(float(row_dict[f"videos/{video_key}/from_timestamp"]) * fps)
        episode_id = int(row_dict["episode_index"])
        length = int(row_dict["length"])
        intervals_by_file.setdefault((chunk, file_index), []).append((start, start + length, episode_id))

    for (chunk, file_index), intervals in intervals_by_file.items():
        intervals.sort()
        arrays = {
            episode_id: np.load(output_dir / "episodes" / f"{episode_id:06d}" / f"{array_name}.npy", mmap_mode="r+")
            for _, _, episode_id in intervals
        }
        decoded = {episode_id: 0 for _, _, episode_id in intervals}
        video_path = (
            dataset_root / "videos" / video_key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
        )
        cursor = 0
        with av.open(str(video_path)) as container:
            for frame_index, frame in enumerate(container.decode(video=0)):
                while cursor < len(intervals) and frame_index >= intervals[cursor][1]:
                    cursor += 1
                if cursor == len(intervals):
                    break
                start, end, episode_id = intervals[cursor]
                if start <= frame_index < end:
                    rgb = frame.to_ndarray(format="rgb24")
                    arrays[episode_id][frame_index - start] = image_tools.resize_with_pad(rgb, 224, 224)
                    decoded[episode_id] += 1
        for array in arrays.values():
            array.flush()
        for start, end, episode_id in intervals:
            if decoded[episode_id] != end - start:
                raise ValueError(
                    f"decoded {decoded[episode_id]} frames for episode {episode_id}, expected {end - start}"
                )


def _write_norm_stats(output_dir: pathlib.Path, train_episodes: list[int]) -> None:
    states = []
    actions = []
    for episode_id in train_episodes:
        episode_dir = output_dir / "episodes" / f"{episode_id:06d}"
        states.append(np.load(episode_dir / "state.npy"))
        actions.append(np.load(episode_dir / "action.npy"))

    def statistics(values: list[np.ndarray]) -> dict[str, list[float]]:
        array = np.concatenate(values, axis=0).astype(np.float64)
        return {
            "mean": np.mean(array, axis=0).tolist(),
            "std": np.std(array, axis=0).tolist(),
            "q01": np.quantile(array, 0.01, axis=0).tolist(),
            "q99": np.quantile(array, 0.99, axis=0).tolist(),
        }

    norm_stats_dir = output_dir / "norm_stats"
    norm_stats_dir.mkdir()
    (norm_stats_dir / "norm_stats.json").write_text(
        json.dumps(
            {"norm_stats": {"state": statistics(states), "actions": statistics(actions)}},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def prepare(args: argparse.Namespace) -> None:
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    episodes = _episode_metadata(dataset_root)
    tasks = _episode_tasks(dataset_root, episodes)
    task_text = _task_texts(dataset_root)
    train, validation = _split_episodes(
        tasks,
        train_per_task=args.train_per_task,
        validation_per_task=args.validation_per_task,
        split_seed=args.split_seed,
    )
    selected = train + validation
    selected_rows = episodes.loc[episodes["episode_index"].isin(selected)].copy()
    entries = {}
    for row in selected_rows.to_dict("records"):
        episode_id = int(row["episode_index"])
        episode_dir = output_dir / "episodes" / f"{episode_id:06d}"
        episode_dir.mkdir(parents=True)
        entries[str(episode_id)] = {
            "length": int(row["length"]),
            "task_index": tasks[episode_id],
            "task": task_text[tasks[episode_id]],
            "dataset_from_index": int(row["dataset_from_index"]),
            "dataset_to_index": int(row["dataset_to_index"]),
        }

    _save_numeric_arrays(dataset_root, output_dir, selected_rows, entries)
    for array_name, video_key in CAMERAS.items():
        _decode_selected_video(
            dataset_root,
            output_dir,
            selected_rows,
            array_name=array_name,
            video_key=video_key,
        )
    _write_norm_stats(output_dir, train)
    index = {
        "schema_version": CACHE_SCHEMA,
        "dataset": {
            "repo_id": "lerobot/libero_10",
            "revision": DATASET_REVISION,
            "codebase_version": "v3.0",
            "fps": 10,
            "state_semantics": "xyz3_axis_angle3_gripper_qpos2",
            "action_semantics": "libero_delta_xyz3_delta_axis_angle3_gripper1",
        },
        "split": {
            "seed": args.split_seed,
            "train_per_task": args.train_per_task,
            "validation_per_task": args.validation_per_task,
        },
        "splits": {"train": train, "validation": validation},
        "norm_stats_dir": "norm_stats",
        "vision_cache": {"status": "pending"},
        "episodes": entries,
    }
    (output_dir / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"prepared {len(train)} train and {len(validation)} validation episodes at {output_dir}")


def encode(args: argparse.Namespace) -> None:
    from flax import nnx
    import jax
    import jax.numpy as jnp

    from openpi.models import model as model_lib
    from openpi.models import pi0_config

    output_dir = args.output_dir.resolve()
    index_path = output_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    params_path = args.base_params_dir.resolve()
    if params_path.name != "params":
        params_path = params_path / "params"
    model_config = pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False)
    model = model_config.load(model_lib.restore_params(params_path, dtype=jnp.bfloat16))
    model.eval()
    graphdef, vision_state = nnx.split(model.PaliGemma.img)

    def encode_device_batch(state, images):
        vision_model = nnx.merge(graphdef, state)
        tokens, _ = vision_model(images, train=False)
        return tokens

    encode_batch = jax.pmap(encode_device_batch, in_axes=(None, 0))
    device_count = jax.local_device_count()
    global_batch_size = device_count * args.encode_batch_size

    for episode_id in [*index["splits"]["train"], *index["splits"]["validation"]]:
        entry = index["episodes"][str(episode_id)]
        for image_name, feature_name in (("image", "image_features"), ("wrist_image", "wrist_features")):
            images = np.load(output_dir / entry["arrays"][image_name], mmap_mode="r")
            features = np.lib.format.open_memmap(
                output_dir / entry["arrays"][feature_name],
                mode="w+",
                dtype=np.float16,
                shape=(len(images), 256, 2048),
            )
            for start in range(0, len(images), global_batch_size):
                batch = np.asarray(images[start : start + global_batch_size])
                valid = len(batch)
                if valid < global_batch_size:
                    batch = np.concatenate([batch, np.repeat(batch[-1:], global_batch_size - valid, axis=0)])
                batch = batch.astype(np.float32) / 127.5 - 1.0
                batch = batch.reshape(device_count, args.encode_batch_size, *batch.shape[1:])
                encoded = encode_batch(vision_state, jnp.asarray(batch))
                encoded.block_until_ready()
                encoded = encoded.reshape(global_batch_size, *encoded.shape[2:])
                features[start : start + valid] = np.asarray(encoded[:valid], dtype=np.float16)
            features.flush()

    index["vision_cache"] = {
        "status": "complete",
        "base_params_dir": str(params_path),
        "preprocessing": "rgb_resize_with_pad_224_uint8_then_minus1_plus1",
        "patches_per_image": 256,
        "feature_dim": 2048,
        "dtype": "float16",
        "cameras": ["base_0_rgb", "left_wrist_0_rgb"],
    }
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"encoded shared SigLIP frame cache at {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="stage", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--dataset-root", type=pathlib.Path, required=True)
    prepare_parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    prepare_parser.add_argument("--train-per-task", type=int, default=12)
    prepare_parser.add_argument("--validation-per-task", type=int, default=3)
    prepare_parser.add_argument("--split-seed", type=int, default=20260925)
    encode_parser = subparsers.add_parser("encode")
    encode_parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    encode_parser.add_argument("--base-params-dir", type=pathlib.Path, required=True)
    encode_parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=8,
        help="images per local JAX device; total batch is this value times the device count",
    )
    args = parser.parse_args()
    prepare(args) if args.stage == "prepare" else encode(args)


if __name__ == "__main__":
    main()
