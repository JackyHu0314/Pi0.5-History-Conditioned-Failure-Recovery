"""Train one official JAX pi0.5 history variant on the LIBERO-10 pilot cache."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pathlib


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_or_uri(value: str) -> str:
    return value if "://" in value else str(pathlib.Path(value).resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=[f"H{i}" for i in range(6)], required=True)
    parser.add_argument("--cache-index", type=pathlib.Path, required=True)
    parser.add_argument("--base-params-dir", required=True)
    parser.add_argument("--base-checkpoint-ref")
    parser.add_argument("--norm-stats-ref")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-train-steps", type=int, required=True)
    parser.add_argument("--schedule-total-steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--peak-lr", type=float, default=1e-5)
    parser.add_argument("--decay-lr", type=float, default=1e-6)
    parser.add_argument("--history-peak-lr", type=float, default=1e-4)
    parser.add_argument("--history-decay-lr", type=float, default=1e-5)
    parser.add_argument("--ema-decay", default="none")
    parser.add_argument("--gradient-accumulation-steps", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fsdp-devices", type=int, default=8)
    parser.add_argument(
        "--train-scope",
        choices=("full_except_vision", "history_only", "action_expert", "history_and_action_expert"),
        default="full_except_vision",
    )
    parser.add_argument("--history-injection", choices=("prefix", "residual"), default="prefix")
    parser.add_argument("--history-visual-source", choices=("past", "current"), default="past")
    parser.add_argument("--current-image-dropout-probability", type=float, default=0.0)
    parser.add_argument("--current-image-dropout-seed", type=int, default=0)
    parser.add_argument("--discrete-state-input", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--keep-period", type=int)
    parser.add_argument("--checkpoint-base-dir", required=True)
    parser.add_argument("--checkpoint-format", choices=("ocdbt", "zarr"), default="ocdbt")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--run-manifest", type=pathlib.Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cache_index = args.cache_index.resolve()
    cache = json.loads(cache_index.read_text(encoding="utf-8"))
    if cache["vision_cache"]["status"] != "complete":
        raise ValueError("the shared SigLIP frame cache is not complete")
    base_params_dir = _local_or_uri(args.base_params_dir)
    if args.variant != "H0" and _local_or_uri(cache["vision_cache"]["base_params_dir"]) != base_params_dir:
        raise ValueError("the history feature cache and initialization checkpoint use different vision weights")
    os.environ.setdefault(
        "JAX_COMPILATION_CACHE_DIR",
        str(cache_index.parent.parent / "jax_compilation_cache"),
    )

    from openpi.training import config as training_config

    ema_decay = None if args.ema_decay == "none" else float(args.ema_decay)
    factory_args = {
        "variant": args.variant,
        "cache_index": str(cache_index),
        "seed": args.seed,
        "checkpoint_base_dir": str(pathlib.Path(args.checkpoint_base_dir).resolve()),
        "checkpoint_format": args.checkpoint_format,
        "exp_name": args.exp_name,
        "num_train_steps": args.num_train_steps,
        "schedule_total_steps": args.schedule_total_steps,
        "base_params_dir": base_params_dir,
        "warmup_steps": args.warmup_steps,
        "peak_lr": args.peak_lr,
        "decay_lr": args.decay_lr,
        "history_peak_lr": args.history_peak_lr,
        "history_decay_lr": args.history_decay_lr,
        "ema_decay": ema_decay,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "batch_size": args.batch_size,
        "fsdp_devices": args.fsdp_devices,
        "train_scope": args.train_scope,
        "history_injection": args.history_injection,
        "history_visual_source": args.history_visual_source,
        "current_image_dropout_probability": args.current_image_dropout_probability,
        "current_image_dropout_seed": args.current_image_dropout_seed,
        "discrete_state_input": args.discrete_state_input,
        "log_interval": args.log_interval,
        "save_interval": args.save_interval,
        "keep_period": args.keep_period,
        "history_input_mode": "cached",
        "resume": args.resume,
    }
    config = training_config.make_libero_history_train_config(
        **factory_args,
    )
    norm_stats_path = cache_index.parent / cache["norm_stats_dir"] / "norm_stats.json"
    run_manifest = {
        "schema_version": "pi05-libero-history-run-v1",
        "factory_args": factory_args,
        "checkpoint_dir": str(config.checkpoint_dir),
        "initialization_checkpoint": {
            "ref": args.base_checkpoint_ref or factory_args["base_params_dir"],
            "params_dir": factory_args["base_params_dir"],
        },
        "data": {
            "cache_index": str(cache_index),
            "cache_index_sha256": _sha256(cache_index),
            "norm_stats_path": str(norm_stats_path),
            "norm_stats_ref": args.norm_stats_ref or str(norm_stats_path),
            "norm_stats_sha256": _sha256(norm_stats_path),
            "dataset": cache["dataset"],
            "split": cache["split"],
            "vision_cache": cache["vision_cache"],
            "use_quantile_norm": True,
            "augmentation": {
                "current_image_dropout_probability": args.current_image_dropout_probability,
                "current_image_dropout_seed": args.current_image_dropout_seed,
                "current_image_dropout_cameras": ["base_0_rgb", "left_wrist_0_rgb"],
                "image_masks_remain_valid": True,
                "query_visual_removed_from_history": True,
            },
        },
        "model": {
            "variant": args.variant,
            "history_injection": args.history_injection,
            "history_visual_source": args.history_visual_source,
            "history_input_mode": "cached",
            "freeze_vision": True,
            "train_scope": args.train_scope,
            "discrete_state_input": args.discrete_state_input,
        },
        "optimization": {
            "micro_batch_size": args.batch_size,
            "effective_batch_size": config.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "base_peak_lr": args.peak_lr,
            "base_decay_lr": args.decay_lr,
            "history_peak_lr": args.history_peak_lr,
            "history_decay_lr": args.history_decay_lr,
            "ema_decay": ema_decay,
            "log_interval": args.log_interval,
            "save_interval": config.save_interval,
        },
        "checkpoint": {
            "format": args.checkpoint_format,
        },
        "config": dataclasses.asdict(config),
    }
    run_manifest_path = args.run_manifest or config.checkpoint_dir.with_suffix(".run.json")
    run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run_manifest_path.write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "run_manifest": str(run_manifest_path),
        **run_manifest,
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2, default=str))
    if args.dry_run:
        return

    import train

    train.main(config)


if __name__ == "__main__":
    main()
