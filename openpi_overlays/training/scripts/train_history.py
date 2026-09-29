"""Fail-closed launcher for one official JAX pi0.5 H0--H5 run."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=[f"H{i}" for i in range(6)], required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-train-steps", type=int, required=True)
    parser.add_argument("--schedule-total-steps", type=int, default=2000)
    parser.add_argument("--checkpoint-base-dir", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Delay openpi/JAX imports so manifest auditing and dry-run fail before any
    # checkpoint download or accelerator initialization.
    from openpi.history import protocol

    manifest = protocol.load_manifest(args.manifest, expected_split="train")
    if manifest.get("purpose") == "smoke" or manifest.get("validates_task_performance") is not True:
        raise ValueError("formal training requires a real manifest with validates_task_performance=true")
    evaluator = pathlib.Path(args.evaluator)
    if not evaluator.is_file():
        raise FileNotFoundError(f"closed-loop evaluator is missing: {evaluator}")
    interface = manifest.get("model_interface", {})
    required = {"action_dim", "action_horizon"}
    if not required.issubset(interface):
        raise ValueError(f"manifest model_interface missing {sorted(required - interface.keys())}")
    storage = manifest["history_storage"]
    input_mode = storage["mode"]
    checkpoint_id = (
        storage["cache_metadata"]["vision_checkpoint"]
        if input_mode == "cached"
        else manifest.get("vision_checkpoint", "")
    )
    if not checkpoint_id:
        raise ValueError("manifest must identify the frozen vision checkpoint")

    from openpi.training import config as training_config

    config = training_config.make_history_train_config(
        variant=args.variant,
        manifest_path=str(pathlib.Path(args.manifest).resolve()),
        seed=args.seed,
        action_dim=int(interface["action_dim"]),
        action_horizon=int(interface["action_horizon"]),
        history_input_mode=input_mode,
        expected_vision_checkpoint=checkpoint_id,
        num_train_steps=args.num_train_steps,
        schedule_total_steps=args.schedule_total_steps,
        checkpoint_base_dir=str(pathlib.Path(args.checkpoint_base_dir).resolve()),
        exp_name=args.exp_name,
        resume=args.resume,
    )
    run_metadata = {
        "variant": args.variant,
        "seed": args.seed,
        "named_rng_streams": {
            "base_init": args.seed,
            "history_init": args.seed,
            "data_sampling": args.seed,
            "training_noise_and_time": args.seed,
            "evaluation_action_noise": args.seed,
        },
        "manifest_sha256": manifest["_manifest_sha256"],
        "evaluator": str(evaluator.resolve()),
        "num_train_steps": args.num_train_steps,
        "schedule_total_steps": args.schedule_total_steps,
        "config": dataclasses.asdict(config),
    }
    print(json.dumps(run_metadata, ensure_ascii=False, indent=2, default=str))
    if args.dry_run:
        return

    # `train_history.py` and upstream `train.py` share this directory, so this
    # import works when invoked exactly as documented from the repository root.
    import train

    train.main(config)


if __name__ == "__main__":
    main()
