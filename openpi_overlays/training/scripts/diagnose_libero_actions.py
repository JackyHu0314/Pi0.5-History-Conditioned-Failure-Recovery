"""Compare physical action predictions on fixed demonstration frames (not closed-loop success)."""
import argparse
import json
from pathlib import Path

import jax
import numpy as np
from openpi.policies import policy_config
from openpi.training import config
from openpi.training.libero_history_dataset import LiberoHistoryDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache-index", type=Path, required=True)
    p.add_argument("--official", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    cfg = config.get_config("pi05_libero") if args.official else config.make_libero_history_train_config(
        variant="H0", cache_index=args.cache_index, seed=11,
        checkpoint_base_dir=str(args.checkpoint.parent), exp_name="diagnostic",
        num_train_steps=1, schedule_total_steps=1,
    )
    policy = policy_config.create_trained_policy(cfg, args.checkpoint)
    records = []
    for split in ("train", "validation"):
        dataset = LiberoHistoryDataset(str(args.cache_index), split=split, variant="H0", action_horizon=10)
        seen = set()
        for eid, offset in zip(dataset.episode_ids, dataset.offsets):
            entry = dataset.episodes[str(eid)]
            if entry["task"] in seen:
                continue
            seen.add(entry["task"])
            for fraction in (0.1, 0.5, 0.8):
                step = int(entry["length"] * fraction)
                sample = dataset[offset + step]
                truth = sample.pop("actions")
                policy._rng = jax.random.key(11)
                predicted = np.asarray(policy.infer(sample)["actions"])
                records.append({
                    "split": split, "episode": eid, "step": step, "task": entry["task"],
                    "prediction": predicted.tolist(), "truth": truth.tolist(),
                    "mse_per_dim": ((predicted - truth)**2).mean(axis=0).tolist(),
                    "zero_action_mse_per_dim": (truth**2).mean(axis=0).tolist(),
                    "gripper_sign_accuracy": float(np.mean((predicted[:, 6] > 0) == (truth[:, 6] > 0))),
                })
    result = {"checkpoint": str(args.checkpoint), "official": args.official, "records": records, "summary": {}}
    for split in ("train", "validation"):
        subset = [x for x in records if x["split"] == split]
        result["summary"][split] = {
            "windows": len(subset),
            "mse_per_dim": np.mean([x["mse_per_dim"] for x in subset], axis=0).tolist(),
            "zero_action_mse_per_dim": np.mean([x["zero_action_mse_per_dim"] for x in subset], axis=0).tolist(),
            "gripper_sign_accuracy": float(np.mean([x["gripper_sign_accuracy"] for x in subset])),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
