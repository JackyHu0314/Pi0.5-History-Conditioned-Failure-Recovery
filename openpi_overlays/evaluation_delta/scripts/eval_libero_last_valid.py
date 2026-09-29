"""Closed-loop LIBERO evaluation with a last-valid-frame observation control."""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import json
import math
import pathlib
import statistics
import time
from typing import Literal

import jax
import jax.numpy as jnp
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
import torch
import tyro

from openpi.history import protocol as history_protocol
from openpi.history import observation_intervention as observation_protocol
from openpi.history.visual_control import broadcast_current_features
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config as training_config


LIBERO_DUMMY_ACTION = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
LIBERO_ENV_RESOLUTION = 256
LIBERO_CONTROL_HZ = 20
HISTORY_CAMERA_KEYS = ("base_0_rgb", "left_wrist_0_rgb")
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass(frozen=True)
class Args:
    variant: Literal["H0", "H1", "H2", "H3", "H4", "H5"]
    checkpoint_dir: str
    output: pathlib.Path
    cache_index: pathlib.Path | None = None
    config_name: str | None = None
    run_manifest: pathlib.Path | None = None
    task_suite_name: str = "libero_10"
    episodes_per_task: int = 3
    task_ids: tuple[int, ...] | None = None
    episode_indices: tuple[int, ...] | None = None
    seed: int = 7
    history_intervention: Literal["normal", "drop_all", "mask_numeric", "shuffle_time_ood"] = "normal"
    observation_intervention: Literal[
        "none",
        "scheduled_dual_camera_blackout",
        "scheduled_dual_camera_last_valid",
    ] = "none"
    intervention_schedule: pathlib.Path | None = None
    blackout_queries: int = 1
    gripper_close_threshold: float = 0.5
    resize_size: int = 224
    replan_steps: int = 5
    num_steps_wait: int = 10
    video_dir: pathlib.Path | None = None
    trace_dir: pathlib.Path | None = None
    video_fps: int = LIBERO_CONTROL_HZ


@dataclasses.dataclass(frozen=True)
class PreparedObservation:
    images: dict[str, np.ndarray]
    image_masks: dict[str, np.bool_]
    state: np.ndarray
    eef_quat_xyzw: np.ndarray
    image_features: dict[str, np.ndarray] | None = None


def _blackout_observation(prepared: PreparedObservation) -> PreparedObservation:
    return dataclasses.replace(
        prepared,
        images={key: np.zeros_like(image) for key, image in prepared.images.items()},
        image_features=None,
    )


def _apply_observation_intervention(
    current: PreparedObservation,
    last_valid: PreparedObservation,
    *,
    intervention: str,
    active: bool,
) -> PreparedObservation:
    if not active:
        return current
    if intervention == "scheduled_dual_camera_blackout":
        return _blackout_observation(current)
    if intervention == "scheduled_dual_camera_last_valid":
        return dataclasses.replace(
            current,
            images={key: image.copy() for key, image in last_valid.images.items()},
            image_features=None,
        )
    raise ValueError(f"active observation schedule has no intervention for {intervention}")


def _video_frame(clean: PreparedObservation, observed: PreparedObservation, *, blackout: bool) -> np.ndarray:
    if not blackout:
        return clean.images["base_0_rgb"].copy()
    return np.concatenate(
        [clean.images["base_0_rgb"], observed.images["base_0_rgb"]],
        axis=1,
    )


def _image_sha256(images: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(images):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(images[key]).tobytes())
    return digest.hexdigest()


def _grasped_objects(env) -> tuple[str, ...]:
    return tuple(
        sorted(
            name
            for name, obj in env.env.objects_dict.items()
            if env.env._check_grasp(env.env.robots[0].gripper, obj)
        )
    )


def _load_intervention_schedules(
    path: pathlib.Path,
    *,
    replan_steps: int,
    blackout_queries: int,
) -> dict[tuple[int, int], observation_protocol.DualCameraBlackoutSchedule]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schedules = {}
    for episode in payload["episodes"]:
        first_close = episode["first_close_plan_query_step"]
        if first_close is None:
            continue
        key = (int(episode["task_id"]), int(episode["episode_index"]))
        schedules[key] = observation_protocol.DualCameraBlackoutSchedule.from_first_close(
            int(first_close),
            replan_steps=replan_steps,
            blackout_queries=blackout_queries,
        )
    return schedules


class OnlineHistoryBuffer:
    """Keep the exact selected single-step transitions without retaining a full episode of features."""

    def __init__(self, selection: str):
        self.selection = selection
        self.episode_id: str | None = None
        self.early: list[history_protocol.Transition] = []
        recent_size = 8 if selection == history_protocol.HistorySelection.RECENT8 else 6
        self.recent: collections.deque[history_protocol.Transition] = collections.deque(maxlen=recent_size)

    def reset(self, episode_id: str) -> None:
        self.episode_id = episode_id
        self.early.clear()
        self.recent.clear()

    def append(self, transition: history_protocol.Transition) -> None:
        if self.episode_id is None:
            raise RuntimeError("reset(episode_id) is required before append")
        transition.validate_for_query(episode_id=self.episode_id, query_step=transition.end_step)
        if self.recent and transition.start_step < self.recent[-1].end_step:
            raise ValueError("history transitions must be appended in non-overlapping time order")
        if self.selection == history_protocol.HistorySelection.EARLY2_RECENT6 and len(self.early) < 2:
            self.early.append(transition)
        self.recent.append(transition)

    def selected(self, *, query_step: int) -> tuple[history_protocol.Transition, ...]:
        if self.episode_id is None:
            raise RuntimeError("reset(episode_id) is required before reading history")
        records = {record.start_step: record for record in (*self.early, *self.recent)}
        selected = tuple(records[index] for index in sorted(records))
        for record in selected:
            record.validate_for_query(episode_id=self.episode_id, query_step=query_step)
        return selected


def _quat_to_axis_angle(quat_xyzw: np.ndarray) -> np.ndarray:
    """Match the state conversion used by the official openpi LIBERO evaluator."""
    quat = np.asarray(quat_xyzw, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return np.asarray(quat[:3] * (2.0 * math.acos(quat[3]) / denominator), dtype=np.float32)


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.asarray(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def _relative_rotation_vector(before_xyzw: np.ndarray, after_xyzw: np.ndarray) -> np.ndarray:
    before = np.asarray(before_xyzw, dtype=np.float64)
    after = np.asarray(after_xyzw, dtype=np.float64)
    before /= np.linalg.norm(before)
    after /= np.linalg.norm(after)
    relative = _quat_multiply_xyzw(after, np.asarray([-before[0], -before[1], -before[2], before[3]]))
    relative /= np.linalg.norm(relative)
    if relative[3] < 0.0:
        relative = -relative
    vector_norm = np.linalg.norm(relative[:3])
    if vector_norm < 1e-8:
        return np.asarray(2.0 * relative[:3], dtype=np.float32)
    angle = 2.0 * math.atan2(vector_norm, relative[3])
    return np.asarray(relative[:3] * (angle / vector_norm), dtype=np.float32)


def _state_delta(before: PreparedObservation, after: PreparedObservation) -> np.ndarray:
    return np.concatenate(
        [
            after.state[:3] - before.state[:3],
            _relative_rotation_vector(before.eef_quat_xyzw, after.eef_quat_xyzw),
            after.state[6:] - before.state[6:],
        ]
    ).astype(np.float32)


def _prepare_observation(obs: dict, resize_size: int) -> PreparedObservation:
    base_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_image = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    base_image = image_tools.convert_to_uint8(image_tools.resize_with_pad(base_image, resize_size, resize_size))
    wrist_image = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_image, resize_size, resize_size))
    state = np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"]),
            _quat_to_axis_angle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"]),
        ]
    ).astype(np.float32)
    return PreparedObservation(
        images={
            "base_0_rgb": base_image,
            "left_wrist_0_rgb": wrist_image,
        },
        image_masks={
            "base_0_rgb": np.bool_(True),
            "left_wrist_0_rgb": np.bool_(True),
        },
        state=state,
        eef_quat_xyzw=np.asarray(obs["robot0_eef_quat"], dtype=np.float64).copy(),
    )


def _encode_history_observation(prepared: PreparedObservation, image_encoder) -> tuple[PreparedObservation, float]:
    images = np.stack([prepared.images[key] for key in HISTORY_CAMERA_KEYS]).astype(np.float32)
    images = images / 255.0 * 2.0 - 1.0
    started = time.perf_counter()
    features, _ = image_encoder(jnp.asarray(images), train=False)
    features = np.asarray(features, dtype=np.float16)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return (
        dataclasses.replace(
            prepared,
            image_features={key: features[index] for index, key in enumerate(HISTORY_CAMERA_KEYS)},
        ),
        elapsed_ms,
    )


def _history_payload(
    buffer: OnlineHistoryBuffer,
    *,
    query_step: int,
    feature_shape: tuple[int, ...],
    intervention: str,
    history_visual_source: str = "past",
    current_features: dict[str, np.ndarray] | None = None,
) -> dict:
    records = buffer.selected(query_step=query_step)
    count = len(records)
    image_features = {
        key: np.zeros((8, 2, *feature_shape), dtype=np.float16) for key in HISTORY_CAMERA_KEYS
    }
    image_masks = {key: np.zeros((8, 2), dtype=np.bool_) for key in HISTORY_CAMERA_KEYS}
    states = np.zeros((8, 8), dtype=np.float32)
    actions = np.zeros((8, 7), dtype=np.float32)
    state_deltas = np.zeros((8, 8), dtype=np.float32)
    numeric_masks = np.zeros((8, 3), dtype=np.bool_)
    state_dim_mask = np.zeros((8, 8), dtype=np.bool_)
    action_dim_mask = np.zeros((8, 7), dtype=np.bool_)
    times = np.zeros(8, dtype=np.int32)

    for slot, transition in enumerate(records):
        before = transition.observation_before
        after = transition.observation_after
        if before.image_features is None or after.image_features is None:
            raise ValueError("selected transition is missing cached SigLIP features")
        for key in HISTORY_CAMERA_KEYS:
            image_features[key][slot, 0] = before.image_features[key]
            image_features[key][slot, 1] = after.image_features[key]
            image_masks[key][slot] = [before.image_masks[key], after.image_masks[key]]
        states[slot] = transition.state_before
        actions[slot] = transition.executed_action
        state_deltas[slot] = _state_delta(before, after)
        numeric_masks[slot] = True
        state_dim_mask[slot] = True
        action_dim_mask[slot] = True
        times[slot] = transition.start_step

    record_mask = np.zeros(8, dtype=np.bool_)
    record_mask[:count] = True
    if history_visual_source == "current":
        if current_features is None:
            raise ValueError("current visual control requires query observation features")
        image_features = broadcast_current_features(current_features, record_mask)
    elif history_visual_source != "past":
        raise ValueError(f"unknown history visual source: {history_visual_source}")
    if intervention == "drop_all":
        record_mask[:] = False
        for mask in image_masks.values():
            mask[:] = False
        numeric_masks[:] = False
    elif intervention == "mask_numeric":
        numeric_masks[:] = False
    elif intervention == "shuffle_time_ood":
        times[:count] = times[:count][::-1]
    elif intervention != "normal":
        raise ValueError(f"unknown history intervention: {intervention}")

    return {
        "image": {},
        "image_features": image_features,
        "image_mask": image_masks,
        "record_mask": record_mask,
        "state": states,
        "action": actions,
        "state_delta": state_deltas,
        "numeric_mask": numeric_masks,
        "state_dim_mask": state_dim_mask,
        "action_dim_mask": action_dim_mask,
        "time": times,
        "query_time": np.asarray(query_step, dtype=np.int32),
    }


def _latest_checkpoint(checkpoint_dir: str) -> pathlib.Path | str:
    if "://" in checkpoint_dir:
        return checkpoint_dir
    checkpoint_dir = pathlib.Path(checkpoint_dir).resolve()
    if (checkpoint_dir / "params").is_dir():
        return checkpoint_dir
    steps = sorted(
        (path for path in checkpoint_dir.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if not steps:
        raise FileNotFoundError(f"no numeric checkpoint steps under {checkpoint_dir}")
    latest = steps[-1]
    if not (latest / "params").is_dir():
        raise FileNotFoundError(f"latest checkpoint has no params item: {latest}")
    return latest


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_run_manifest(path: pathlib.Path, *, variant: str, checkpoint_dir: str) -> dict:
    path = path.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["schema_version"] != "pi05-libero-history-run-v1":
        raise ValueError(f"unsupported run manifest schema: {manifest['schema_version']}")
    factory_args = manifest["factory_args"]
    if factory_args["variant"] != variant:
        raise ValueError(f"run manifest variant {factory_args['variant']} does not match --variant {variant}")
    if factory_args["history_input_mode"] != "cached":
        raise ValueError("closed-loop evaluator requires a cached-feature history run manifest")
    manifest_checkpoint = pathlib.Path(manifest["checkpoint_dir"]).resolve()
    if "://" not in checkpoint_dir and manifest_checkpoint != pathlib.Path(checkpoint_dir).resolve():
        requested_checkpoint = pathlib.Path(checkpoint_dir).resolve()
        raise ValueError(
            f"run manifest checkpoint directory {manifest_checkpoint} does not match {requested_checkpoint}"
        )
    data = manifest["data"]
    cache_index = pathlib.Path(data["cache_index"]).resolve()
    norm_stats = pathlib.Path(data["norm_stats_path"]).resolve()
    if _sha256(cache_index) != data["cache_index_sha256"]:
        raise ValueError(f"cache index SHA256 does not match run manifest: {cache_index}")
    if _sha256(norm_stats) != data["norm_stats_sha256"]:
        raise ValueError(f"norm stats SHA256 does not match run manifest: {norm_stats}")
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "schema_version": manifest["schema_version"],
        "checkpoint_dir": str(manifest_checkpoint),
        "factory_args": factory_args,
        "data": {
            "cache_index": str(cache_index),
            "cache_index_sha256": data["cache_index_sha256"],
            "norm_stats_path": str(norm_stats),
            "norm_stats_ref": data["norm_stats_ref"],
            "norm_stats_sha256": data["norm_stats_sha256"],
            "vision_cache": data["vision_cache"],
            "use_quantile_norm": data["use_quantile_norm"],
        },
        "model": manifest["model"],
        "optimization": manifest["optimization"],
    }


def _episode_seed(seed: int, task_id: int, episode_index: int) -> int:
    return int(np.random.SeedSequence([seed, task_id, episode_index]).generate_state(1, dtype=np.uint32)[0])


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    ordered = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(ordered)),
        "p50": float(np.quantile(ordered, 0.50)),
        "p95": float(np.quantile(ordered, 0.95)),
        "max": float(np.max(ordered)),
    }


def _action_summary(actions: list[np.ndarray]) -> dict[str, int | list[float]]:
    if not actions:
        return {"count": 0, "mean": [], "std": [], "min": [], "max": []}
    array = np.stack(actions).astype(np.float64)
    return {
        "count": len(array),
        "mean": np.mean(array, axis=0).tolist(),
        "std": np.std(array, axis=0).tolist(),
        "min": np.min(array, axis=0).tolist(),
        "max": np.max(array, axis=0).tolist(),
    }


def _write_video(path: pathlib.Path, frames: list[np.ndarray], *, fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps)


def _write_trace(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _make_env(task, *, resolution: int, seed: int):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
        control_freq=LIBERO_CONTROL_HZ,
    )
    env.seed(seed)
    return env


def _get_fixed_initial_states(task):
    path = pathlib.Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    return torch.load(path, weights_only=False)


def _load_policy(args: Args, checkpoint: pathlib.Path | str):
    run_manifest_audit = None
    if args.config_name is not None:
        config = training_config.get_config(args.config_name)
    elif args.run_manifest is not None:
        run_manifest_audit = _validate_run_manifest(
            args.run_manifest,
            variant=args.variant,
            checkpoint_dir=args.checkpoint_dir,
        )
        config = training_config.make_libero_history_train_config_from_run_manifest(args.run_manifest)
    else:
        if args.cache_index is None:
            raise ValueError("--cache-index is required for a history experiment checkpoint")
        checkpoint_path = pathlib.Path(args.checkpoint_dir).resolve()
        config = training_config.make_libero_history_train_config(
            variant=args.variant,
            cache_index=args.cache_index,
            seed=args.seed,
            checkpoint_base_dir=str(checkpoint_path.parent),
            exp_name=checkpoint_path.name,
            num_train_steps=1,
            schedule_total_steps=1,
            history_input_mode="cached",
        )
    policy = policy_config.create_trained_policy(config, checkpoint)
    image_encoder = (
        nnx_utils.module_jit(
            policy._model.PaliGemma.img.__call__,  # noqa: SLF001
            static_argnames=("train",),
        )
        if args.variant != "H0"
        else None
    )
    return policy, image_encoder, config.name, run_manifest_audit


def evaluate(args: Args) -> dict:
    if args.episodes_per_task < 1:
        raise ValueError("episodes_per_task must be positive")
    if args.replan_steps < 1:
        raise ValueError("replan_steps must be positive")
    if args.blackout_queries < 1:
        raise ValueError("blackout_queries must be positive")
    if args.task_suite_name not in MAX_STEPS:
        raise ValueError(f"unknown task suite: {args.task_suite_name}")
    if args.variant == "H0" and args.history_intervention != "normal":
        raise ValueError("H0 has no history input, so only the normal intervention is defined")
    if args.config_name is not None and args.variant != "H0":
        raise ValueError("a registered non-history config can only be evaluated with --variant H0")
    if args.config_name is not None and args.run_manifest is not None:
        raise ValueError("--config-name and --run-manifest are mutually exclusive")
    if args.video_fps < 1:
        raise ValueError("video_fps must be positive")
    if args.observation_intervention in {
        "scheduled_dual_camera_blackout",
        "scheduled_dual_camera_last_valid",
    }:
        if args.intervention_schedule is None:
            raise ValueError("scheduled intervention requires --intervention-schedule from a paired clean run")
    elif args.intervention_schedule is not None:
        raise ValueError("--intervention-schedule requires a scheduled observation intervention")

    np.random.seed(args.seed)
    checkpoint = _latest_checkpoint(args.checkpoint_dir)
    policy, image_encoder, loaded_config_name, run_manifest_audit = _load_policy(args, checkpoint)
    history_visual_source = (
        "past"
        if run_manifest_audit is None
        else run_manifest_audit["factory_args"].get("history_visual_source", "past")
    )
    intervention_schedules = (
        _load_intervention_schedules(
            args.intervention_schedule,
            replan_steps=args.replan_steps,
            blackout_queries=args.blackout_queries,
        )
        if args.intervention_schedule is not None
        else {}
    )
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task_ids = list(range(task_suite.n_tasks)) if args.task_ids is None else list(args.task_ids)
    episode_indices = (
        list(range(args.episodes_per_task)) if args.episode_indices is None else list(args.episode_indices)
    )
    for task_id in task_ids:
        if not 0 <= task_id < task_suite.n_tasks:
            raise ValueError(f"task id {task_id} is outside [0, {task_suite.n_tasks})")
    episodes: list[dict] = []
    all_wall_latency_ms: list[float] = []
    all_policy_latency_ms: list[float] = []
    all_history_encoding_ms: list[float] = []
    steady_wall_latency_ms: list[float] = []
    steady_policy_latency_ms: list[float] = []
    steady_history_encoding_ms: list[float] = []
    cold_wall_latency_ms: float | None = None
    cold_policy_latency_ms: float | None = None
    cold_history_encoding_ms: float | None = None

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = _get_fixed_initial_states(task)
        for episode_index in episode_indices:
            if not 0 <= episode_index < len(initial_states):
                raise ValueError(
                    f"initial-state index {episode_index} is outside [0, {len(initial_states)}) for task {task_id}"
                )
        env = _make_env(task, resolution=LIBERO_ENV_RESOLUTION, seed=args.seed)
        try:
            for episode_index in episode_indices:
                episode_seed = _episode_seed(args.seed, task_id, episode_index)
                env.seed(episode_seed)
                env.reset()
                obs = env.set_init_state(initial_states[episode_index])
                policy._rng = jax.random.key(episode_seed)  # noqa: SLF001
                action_plan: collections.deque[np.ndarray] = collections.deque()
                selection = (
                    history_protocol.HistorySelection.RECENT8
                    if args.variant == "H4"
                    else history_protocol.HistorySelection.EARLY2_RECENT6
                )
                history_buffer = OnlineHistoryBuffer(selection)
                episode_id = f"task{task_id:02d}-episode{episode_index:02d}"
                history_buffer.reset(episode_id)
                episode_schedule = intervention_schedules.get((task_id, episode_index))
                sim_steps = 0
                control_steps = 0
                success = False
                wall_latency_ms: list[float] = []
                policy_latency_ms: list[float] = []
                history_encoding_ms: list[float] = []
                episode_steady_wall_latency_ms: list[float] = []
                episode_steady_policy_latency_ms: list[float] = []
                episode_steady_history_encoding_ms: list[float] = []
                episode_cold_wall_latency_ms: float | None = None
                episode_cold_policy_latency_ms: float | None = None
                episode_cold_history_encoding_ms: float | None = None
                executed_actions: list[np.ndarray] = []
                trace_steps: list[dict] = []
                policy_queries: list[dict] = []
                query_audit: dict[int, dict] = {}
                video_frames: list[np.ndarray] = []
                plan_query_step = -1
                first_close_plan_query_step: int | None = None

                for _ in range(args.num_steps_wait):
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION.tolist())
                    sim_steps += 1

                clean_prepared = _prepare_observation(obs, args.resize_size)
                last_valid_prepared = clean_prepared
                blackout = bool(
                    episode_schedule is not None
                    and episode_schedule.is_blackout_observation(control_steps)
                )
                prepared = _apply_observation_intervention(
                    clean_prepared,
                    last_valid_prepared,
                    intervention=args.observation_intervention,
                    active=blackout,
                )
                if args.video_dir is not None:
                    video_frames.append(_video_frame(clean_prepared, prepared, blackout=blackout))
                if args.variant != "H0":
                    assert image_encoder is not None
                    prepared, encode_ms = _encode_history_observation(prepared, image_encoder)
                    history_encoding_ms.append(encode_ms)
                    all_history_encoding_ms.append(encode_ms)
                    if cold_history_encoding_ms is None:
                        cold_history_encoding_ms = encode_ms
                        episode_cold_history_encoding_ms = encode_ms
                    else:
                        steady_history_encoding_ms.append(encode_ms)
                        episode_steady_history_encoding_ms.append(encode_ms)
                while not success and control_steps < MAX_STEPS[args.task_suite_name]:
                    if not action_plan:
                        grasped_objects = _grasped_objects(env)
                        policy_input = {
                            "observation/image": prepared.images["base_0_rgb"],
                            "observation/wrist_image": prepared.images["left_wrist_0_rgb"],
                            "observation/state": prepared.state,
                            "prompt": str(task.language),
                        }
                        if args.variant != "H0":
                            assert prepared.image_features is not None
                            policy_input["history"] = _history_payload(
                                history_buffer,
                                query_step=control_steps,
                                feature_shape=prepared.image_features["base_0_rgb"].shape,
                                intervention=args.history_intervention,
                                history_visual_source=history_visual_source,
                                current_features=prepared.image_features,
                            )
                        started = time.perf_counter()
                        policy_output = policy.infer(policy_input)
                        wall_ms = (time.perf_counter() - started) * 1000.0
                        wall_latency_ms.append(wall_ms)
                        all_wall_latency_ms.append(wall_ms)
                        is_cold_inference = cold_wall_latency_ms is None
                        if is_cold_inference:
                            cold_wall_latency_ms = wall_ms
                            episode_cold_wall_latency_ms = wall_ms
                        else:
                            steady_wall_latency_ms.append(wall_ms)
                            episode_steady_wall_latency_ms.append(wall_ms)
                        if "policy_timing" in policy_output:
                            measured = float(policy_output["policy_timing"]["infer_ms"])
                            policy_latency_ms.append(measured)
                            all_policy_latency_ms.append(measured)
                            if is_cold_inference:
                                cold_policy_latency_ms = measured
                                episode_cold_policy_latency_ms = measured
                            else:
                                steady_policy_latency_ms.append(measured)
                                episode_steady_policy_latency_ms.append(measured)
                        action_chunk = np.asarray(policy_output["actions"], dtype=np.float32)
                        if action_chunk.ndim != 2 or action_chunk.shape[1] != 7:
                            raise ValueError(f"policy returned actions with shape {action_chunk.shape}, expected [T,7]")
                        if len(action_chunk) < args.replan_steps:
                            raise ValueError(
                                f"policy returned {len(action_chunk)} actions, "
                                f"shorter than replan_steps={args.replan_steps}"
                            )
                        plan_query_step = control_steps
                        query_record = {
                            "query_step": control_steps,
                            "observation_blackout": blackout,
                            "observation_substitution": (
                                "last_valid"
                                if blackout
                                and args.observation_intervention == "scheduled_dual_camera_last_valid"
                                else "zeros"
                                if blackout
                                else "current"
                            ),
                            "current_state": prepared.state.tolist(),
                            "current_images_sha256": _image_sha256(prepared.images),
                            "clean_current_images_sha256": _image_sha256(clean_prepared.images),
                            "oracle_grasped_objects": list(grasped_objects),
                            "action_chunk_sha256": hashlib.sha256(
                                np.ascontiguousarray(action_chunk[: args.replan_steps]).tobytes()
                            ).hexdigest(),
                        }
                        query_audit[control_steps] = query_record
                        if args.trace_dir is not None:
                            policy_queries.append(
                                {
                                    **query_record,
                                    "wall_inference_ms": wall_ms,
                                    "actions": action_chunk.tolist(),
                                }
                            )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    if first_close_plan_query_step is None and float(action[-1]) >= args.gripper_close_threshold:
                        first_close_plan_query_step = plan_query_step
                    before = prepared
                    blackout_before = blackout
                    next_obs, reward, done, _ = env.step(action.tolist())
                    sim_steps += 1
                    next_control_step = control_steps + 1
                    clean_after = _prepare_observation(next_obs, args.resize_size)
                    blackout_after = bool(
                        episode_schedule is not None
                        and episode_schedule.is_blackout_observation(next_control_step)
                    )
                    if not blackout_after:
                        last_valid_prepared = clean_after
                    after = _apply_observation_intervention(
                        clean_after,
                        last_valid_prepared,
                        intervention=args.observation_intervention,
                        active=blackout_after,
                    )
                    if args.video_dir is not None:
                        video_frames.append(_video_frame(clean_after, after, blackout=blackout_after))
                    if args.variant != "H0":
                        assert image_encoder is not None
                        after, encode_ms = _encode_history_observation(after, image_encoder)
                        history_encoding_ms.append(encode_ms)
                        all_history_encoding_ms.append(encode_ms)
                        if cold_history_encoding_ms is None:
                            cold_history_encoding_ms = encode_ms
                            episode_cold_history_encoding_ms = encode_ms
                        else:
                            steady_history_encoding_ms.append(encode_ms)
                            episode_steady_history_encoding_ms.append(encode_ms)
                    history_buffer.append(
                        history_protocol.Transition(
                            episode_id=episode_id,
                            start_step=control_steps,
                            end_step=control_steps + 1,
                            observation_before=before,
                            state_before=before.state,
                            executed_action=action,
                            observation_after=after,
                            state_after=after.state,
                        )
                    )
                    executed_actions.append(action.copy())
                    if args.trace_dir is not None:
                        trace_steps.append(
                            {
                                "step": control_steps,
                                "plan_query_step": plan_query_step,
                                "state_before": before.state.tolist(),
                                "action": action.tolist(),
                                "reward": float(reward),
                                "done": bool(done),
                                "state_after": after.state.tolist(),
                                "observation_blackout_before": blackout_before,
                                "observation_blackout_after": blackout_after,
                                "oracle_grasped_objects_after": list(_grasped_objects(env)),
                            }
                        )
                    control_steps += 1
                    prepared = after
                    clean_prepared = clean_after
                    blackout = blackout_after
                    success = bool(done)

                candidate_schedule = (
                    observation_protocol.DualCameraBlackoutSchedule.from_first_close(
                        first_close_plan_query_step,
                        replan_steps=args.replan_steps,
                        blackout_queries=args.blackout_queries,
                    )
                    if first_close_plan_query_step is not None
                    else None
                )
                audit_schedule = episode_schedule or candidate_schedule
                audit_query_steps = (
                    {
                        audit_schedule.first_close_plan_query_step,
                        audit_schedule.first_post_close_query_step,
                        audit_schedule.clean_outcome_query_step,
                        *audit_schedule.blackout_policy_query_steps,
                    }
                    if audit_schedule is not None
                    else set()
                )
                selected_query_audit = [
                    query_audit[step] for step in sorted(audit_query_steps) if step in query_audit
                ]
                outcome = "success" if success else "failure"
                artifact_stem = f"task{task_id:02d}-episode{episode_index:02d}-{outcome}"
                video_path = None
                if args.video_dir is not None:
                    video_path = args.video_dir / f"{artifact_stem}.mp4"
                    _write_video(video_path, video_frames, fps=args.video_fps)
                trace_path = None
                if args.trace_dir is not None:
                    trace_path = args.trace_dir / f"{artifact_stem}.json"
                    _write_trace(
                        trace_path,
                        {
                            "task_id": task_id,
                            "task": str(task.language),
                            "episode_index": episode_index,
                            "episode_seed": episode_seed,
                            "success": success,
                            "checkpoint": str(checkpoint),
                            "config_name": loaded_config_name,
                            "first_close_plan_query_step": first_close_plan_query_step,
                            "candidate_blackout_schedule": (
                                candidate_schedule.to_mapping() if candidate_schedule is not None else None
                            ),
                            "applied_blackout_schedule": (
                                episode_schedule.to_mapping() if episode_schedule is not None else None
                            ),
                            "selected_query_audit": selected_query_audit,
                            "policy_queries": policy_queries,
                            "steps": trace_steps,
                        },
                    )
                episode_result = {
                    "task_id": task_id,
                    "task": str(task.language),
                    "episode_index": episode_index,
                    "initial_state_index": episode_index,
                    "episode_seed": episode_seed,
                    "success": success,
                    "first_close_plan_query_step": first_close_plan_query_step,
                    "candidate_blackout_schedule": (
                        candidate_schedule.to_mapping() if candidate_schedule is not None else None
                    ),
                    "applied_blackout_schedule": (
                        episode_schedule.to_mapping() if episode_schedule is not None else None
                    ),
                    "selected_query_audit": selected_query_audit,
                    "blackout_query_reached": any(
                        record["observation_blackout"] for record in selected_query_audit
                    ),
                    "control_steps": control_steps,
                    "sim_steps_including_settle": sim_steps,
                    "history_records": 0 if args.variant == "H0" else control_steps,
                    "wall_inference_ms": _latency_summary(wall_latency_ms),
                    "policy_reported_inference_ms": _latency_summary(policy_latency_ms),
                    "history_frame_encoding_ms": _latency_summary(history_encoding_ms),
                    "cold_start_ms": {
                        "wall_inference": episode_cold_wall_latency_ms,
                        "policy_reported_inference": episode_cold_policy_latency_ms,
                        "history_frame_encoding": episode_cold_history_encoding_ms,
                    },
                    "steady_state_wall_inference_ms": _latency_summary(episode_steady_wall_latency_ms),
                    "steady_state_policy_reported_inference_ms": _latency_summary(
                        episode_steady_policy_latency_ms
                    ),
                    "steady_state_history_frame_encoding_ms": _latency_summary(
                        episode_steady_history_encoding_ms
                    ),
                    "executed_action": _action_summary(executed_actions),
                    "total_model_compute_ms_per_control_step": (
                        (sum(wall_latency_ms) + sum(history_encoding_ms)) / control_steps
                        if control_steps
                        else None
                    ),
                    "video": str(video_path.resolve()) if video_path is not None else None,
                    "trace": str(trace_path.resolve()) if trace_path is not None else None,
                }
                episodes.append(episode_result)
                print(json.dumps({"episode": episode_result}, ensure_ascii=False), flush=True)
        finally:
            env.close()

    tasks = []
    for task_id in task_ids:
        task_episodes = [episode for episode in episodes if episode["task_id"] == task_id]
        tasks.append(
            {
                "task_id": task_id,
                "task": task_episodes[0]["task"],
                "episodes": len(task_episodes),
                "successes": sum(bool(episode["success"]) for episode in task_episodes),
                "success_rate": statistics.fmean(bool(episode["success"]) for episode in task_episodes),
            }
        )
    return {
        "schema_version": "pi05-libero-last-valid-control-v1",
        "variant": args.variant,
        "config_name": loaded_config_name,
        "run_manifest": str(args.run_manifest.resolve()) if args.run_manifest is not None else None,
        "run_manifest_audit": run_manifest_audit,
        "history_intervention": args.history_intervention,
        "history_visual_source": history_visual_source,
        "observation_intervention": args.observation_intervention,
        "intervention_schedule": (
            str(args.intervention_schedule.resolve()) if args.intervention_schedule is not None else None
        ),
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "cache_index": str(args.cache_index.resolve()) if args.cache_index is not None else None,
        "protocol": {
            "task_suite": args.task_suite_name,
            "task_ids": task_ids,
            "episodes_per_task": len(episode_indices),
            "fixed_initial_state_indices": episode_indices,
            "max_control_steps": MAX_STEPS[args.task_suite_name],
            "sim_control_hz": LIBERO_CONTROL_HZ,
            "settle_steps": args.num_steps_wait,
            "replan_steps": args.replan_steps,
            "blackout_queries": args.blackout_queries,
            "gripper_close_threshold": args.gripper_close_threshold,
            "blackout_camera_keys": (
                list(HISTORY_CAMERA_KEYS)
                if args.observation_intervention
                in {"scheduled_dual_camera_blackout", "scheduled_dual_camera_last_valid"}
                else []
            ),
            "blackout_image_masks_remain_valid": True,
            "missing_observation_fill": (
                "last_valid_camera_frames_with_current_proprioception"
                if args.observation_intervention == "scheduled_dual_camera_last_valid"
                else "zero_camera_frames_with_current_proprioception"
                if args.observation_intervention == "scheduled_dual_camera_blackout"
                else None
            ),
            "blackout_schedule_rule": (
                "first close chunk; two subsequent clean queries; then N blackout queries"
            ),
            "history_selection": (
                None if args.variant == "H0" else "recent8" if args.variant == "H4" else "early2_recent6"
            ),
            "history_transition": None if args.variant == "H0" else "one actually executed environment action",
            "history_visual_cache": (
                None
                if args.variant == "H0"
                else "encode_each_observed_frame_once_with_checkpoint_frozen_siglip_float16"
            ),
            "history_cameras": None if args.variant == "H0" else list(HISTORY_CAMERA_KEYS),
            "camera_preprocessing": "rotate_180_resize_with_pad_uint8",
            "state": "eef_xyz3+eef_axis_angle3+gripper_qpos2",
            "action": "LIBERO delta xyz3+delta axis_angle3+gripper1",
            "state_delta": "delta_xyz3+relative_rotation_rotvec3+delta_gripper_qpos2",
        },
        "overall": {
            "episodes": len(episodes),
            "successes": sum(bool(episode["success"]) for episode in episodes),
            "success_rate": statistics.fmean(bool(episode["success"]) for episode in episodes),
            "wall_inference_ms": _latency_summary(all_wall_latency_ms),
            "policy_reported_inference_ms": _latency_summary(all_policy_latency_ms),
            "history_frame_encoding_ms": _latency_summary(all_history_encoding_ms),
            "cold_start_ms": {
                "wall_inference": cold_wall_latency_ms,
                "policy_reported_inference": cold_policy_latency_ms,
                "history_frame_encoding": cold_history_encoding_ms,
            },
            "steady_state_wall_inference_ms": _latency_summary(steady_wall_latency_ms),
            "steady_state_policy_reported_inference_ms": _latency_summary(steady_policy_latency_ms),
            "steady_state_history_frame_encoding_ms": _latency_summary(steady_history_encoding_ms),
            "total_model_compute_ms_per_control_step": (
                (sum(all_wall_latency_ms) + sum(all_history_encoding_ms))
                / sum(int(episode["control_steps"]) for episode in episodes)
                if sum(int(episode["control_steps"]) for episode in episodes)
                else None
            ),
        },
        "latency_samples_ms": {
            "wall_inference": all_wall_latency_ms,
            "policy_reported_inference": all_policy_latency_ms,
            "history_frame_encoding": all_history_encoding_ms,
            "steady_state_wall_inference": steady_wall_latency_ms,
            "steady_state_policy_reported_inference": steady_policy_latency_ms,
            "steady_state_history_frame_encoding": steady_history_encoding_ms,
        },
        "tasks": tasks,
        "episodes": episodes,
    }


def main(args: Args) -> None:
    result = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(tyro.cli(Args))
