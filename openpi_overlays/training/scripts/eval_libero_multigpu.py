"""Shard clean LIBERO-10 evaluation across GPUs and compare paired results."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

import numpy as np


TASK_IDS = tuple(range(10))
SPLIT_EPISODES = {
    "dev": tuple(range(10)),
    "final": tuple(range(10, 50)),
}


def _wait_for_training_barrier(output_dir: pathlib.Path) -> None:
    """Keep this campaign's evaluation isolated from all of its training jobs."""
    work = next(
        (parent for parent in (output_dir, *output_dir.parents) if parent.name == "current_frame_control_20260928"),
        None,
    )
    if work is None or not (work / "training_barrier.enabled").is_file():
        return
    campaign = work / "campaign.json"
    waiting_statuses = {"planned", "training", "deduplicating"}
    while True:
        state = json.loads(campaign.read_text(encoding="utf-8"))
        waiting = sorted(
            key for key, row in state["jobs"].items() if row["status"] in waiting_statuses
        )
        approval = work / "convergence_approved.json"
        if not waiting and approval.is_file():
            print("training barrier released; starting evaluation", flush=True)
            return
        if waiting:
            print(f"waiting for all C1 training checkpoints: {waiting}", flush=True)
        else:
            print("all checkpoints saved; waiting for convergence approval", flush=True)
        time.sleep(15)


def _write_json(path: pathlib.Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_manifest_audit(path: pathlib.Path) -> dict:
    path = path.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "schema_version": manifest["schema_version"],
        "checkpoint_dir": manifest["checkpoint_dir"],
        "factory_args": manifest["factory_args"],
        "data": manifest["data"],
        "model": manifest["model"],
        "optimization": manifest["optimization"],
    }


def _configuration_factory_args(run_manifest: dict) -> dict:
    ignored = {"seed", "checkpoint_base_dir", "exp_name", "resume", "log_interval", "keep_period"}
    return {key: value for key, value in run_manifest["factory_args"].items() if key not in ignored}


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def _episode_map(result: dict) -> dict[tuple[int, int], dict]:
    episodes = result["episodes"]
    keys = [(int(episode["task_id"]), int(episode["episode_index"])) for episode in episodes]
    duplicates = [key for key, count in collections.Counter(keys).items() if count != 1]
    if duplicates:
        raise ValueError(f"duplicate episode keys: {duplicates}")
    return dict(zip(keys, episodes, strict=True))


def _bootstrap_task_stratified(
    differences: dict[int, list[float]], *, replicates: int, seed: int
) -> dict[str, float | int | str]:
    task_ids = sorted(differences)
    arrays = [np.asarray(differences[task_id], dtype=np.float64) for task_id in task_ids]
    observed = statistics.fmean(float(np.mean(array)) for array in arrays)
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        task_means = []
        for array in arrays:
            indices = rng.integers(0, len(array), size=len(array))
            task_means.append(float(np.mean(array[indices])))
        samples[replicate] = statistics.fmean(task_means)
    return {
        "estimate": observed,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "replicates": replicates,
        "resampling_unit": "fixed_initial_state_within_task",
        "unique_episode_units": sum(len(array) for array in arrays),
    }


def _paired_comparison(candidate: dict, baseline: dict, *, replicates: int, seed: int) -> dict:
    candidate_map = _episode_map(candidate)
    baseline_map = _episode_map(baseline)
    if candidate_map.keys() != baseline_map.keys():
        missing = sorted(baseline_map.keys() - candidate_map.keys())
        extra = sorted(candidate_map.keys() - baseline_map.keys())
        raise ValueError(f"paired episode mismatch: missing={missing}, extra={extra}")
    seed_mismatches = [
        key
        for key in candidate_map
        if int(candidate_map[key]["episode_seed"]) != int(baseline_map[key]["episode_seed"])
    ]
    if seed_mismatches:
        raise ValueError(f"paired episode seed mismatch: {seed_mismatches}")

    by_task: dict[int, list[float]] = collections.defaultdict(list)
    task_rows = []
    totals = {"both_success": 0, "candidate_only": 0, "baseline_only": 0, "both_failure": 0}
    for task_id in sorted({key[0] for key in candidate_map}):
        keys = sorted(key for key in candidate_map if key[0] == task_id)
        candidate_values = [bool(candidate_map[key]["success"]) for key in keys]
        baseline_values = [bool(baseline_map[key]["success"]) for key in keys]
        differences = [
            float(current) - float(reference) for current, reference in zip(candidate_values, baseline_values)
        ]
        by_task[task_id].extend(differences)
        row_counts = {
            "both_success": sum(current and reference for current, reference in zip(candidate_values, baseline_values)),
            "candidate_only": sum(
                current and not reference for current, reference in zip(candidate_values, baseline_values)
            ),
            "baseline_only": sum(
                not current and reference for current, reference in zip(candidate_values, baseline_values)
            ),
            "both_failure": sum(
                not current and not reference for current, reference in zip(candidate_values, baseline_values)
            ),
        }
        for name, value in row_counts.items():
            totals[name] += value
        task_rows.append(
            {
                "task_id": task_id,
                "episodes": len(keys),
                "candidate_success_rate": statistics.fmean(candidate_values),
                "baseline_success_rate": statistics.fmean(baseline_values),
                "success_rate_delta": statistics.fmean(differences),
                **row_counts,
            }
        )

    return {
        "candidate_macro_success_rate": statistics.fmean(row["candidate_success_rate"] for row in task_rows),
        "baseline_macro_success_rate": statistics.fmean(row["baseline_success_rate"] for row in task_rows),
        "paired_macro_success_rate_delta": _bootstrap_task_stratified(by_task, replicates=replicates, seed=seed),
        "paired_outcomes": totals,
        "tasks": task_rows,
    }


def _aggregate_shards(args: argparse.Namespace, shard_specs: list[dict]) -> dict:
    expected_keys = {(task_id, episode) for task_id in TASK_IDS for episode in SPLIT_EPISODES[args.split]}
    episodes = []
    latency_samples: dict[str, list[float]] = collections.defaultdict(list)
    cold_starts = []
    checkpoints = set()
    loaded_configs = set()
    result_seeds = set()
    result_variants = set()
    result_manifest_hashes = set()
    result_interventions = set()
    for shard in shard_specs:
        result = json.loads(pathlib.Path(shard["output"]).read_text(encoding="utf-8"))
        shard_map = _episode_map(result)
        shard_expected = {
            (task_id, episode) for task_id in shard["task_ids"] for episode in SPLIT_EPISODES[args.split]
        }
        if set(shard_map) != shard_expected:
            raise ValueError(
                f"shard {shard['gpu']} episode mismatch: "
                f"missing={sorted(shard_expected - set(shard_map))}, extra={sorted(set(shard_map) - shard_expected)}"
            )
        episodes.extend(result["episodes"])
        for name, values in result["latency_samples_ms"].items():
            latency_samples[name].extend(float(value) for value in values)
        cold_starts.append(
            {
                "gpu": shard["gpu"],
                "task_ids": shard["task_ids"],
                **result["overall"]["cold_start_ms"],
            }
        )
        checkpoints.add(result["checkpoint"])
        loaded_configs.add(result["config_name"])
        result_seeds.add(int(result["seed"]))
        result_variants.add(result["variant"])
        result_interventions.add(result["history_intervention"])
        result_manifest_hashes.add(
            None if result["run_manifest_audit"] is None else result["run_manifest_audit"]["sha256"]
        )

    combined_map = _episode_map({"episodes": episodes})
    if set(combined_map) != expected_keys:
        raise ValueError(
            f"combined episode mismatch: "
            f"missing={sorted(expected_keys - set(combined_map))}, extra={sorted(set(combined_map) - expected_keys)}"
        )
    if len(checkpoints) != 1 or len(loaded_configs) != 1:
        raise ValueError(f"shards loaded different checkpoints/configs: {checkpoints=}, {loaded_configs=}")
    if result_seeds != {args.eval_seed} or result_variants != {args.variant}:
        raise ValueError(f"shards used unexpected protocol: {result_seeds=}, {result_variants=}")
    if result_interventions != {args.history_intervention}:
        raise ValueError(f"shards used unexpected history intervention: {result_interventions}")
    manifest_audit = _run_manifest_audit(args.run_manifest) if args.run_manifest is not None else None
    expected_manifest_hashes = {None if manifest_audit is None else manifest_audit["sha256"]}
    if result_manifest_hashes != expected_manifest_hashes:
        raise ValueError(
            f"shards used an unexpected run manifest: {result_manifest_hashes=}, {expected_manifest_hashes=}"
        )

    episodes.sort(key=lambda episode: (episode["task_id"], episode["episode_index"]))
    tasks = []
    for task_id in TASK_IDS:
        rows = [episode for episode in episodes if episode["task_id"] == task_id]
        tasks.append(
            {
                "task_id": task_id,
                "task": rows[0]["task"],
                "episodes": len(rows),
                "successes": sum(bool(row["success"]) for row in rows),
                "success_rate": statistics.fmean(bool(row["success"]) for row in rows),
            }
        )

    overall_success_rate = statistics.fmean(bool(episode["success"]) for episode in episodes)
    macro_success_rate = statistics.fmean(task["success_rate"] for task in tasks)
    return {
        "schema_version": "pi05-libero-multigpu-eval-v1",
        "run": {
            "run_id": args.run_id,
            "variant": args.variant,
            "training_seed": args.training_seed,
            "eval_seed": args.eval_seed,
            "split": args.split,
            "episode_indices": list(SPLIT_EPISODES[args.split]),
            "checkpoint": checkpoints.pop(),
            "config_name": loaded_configs.pop(),
            "run_manifest": str(args.run_manifest.resolve()) if args.run_manifest is not None else None,
            "training_run_manifest": manifest_audit,
            "gpu_ids": args.gpus,
            "frozen_config_id": args.frozen_config_id,
            "history_intervention": args.history_intervention,
        },
        "overall": {
            "episodes": len(episodes),
            "successes": sum(bool(episode["success"]) for episode in episodes),
            "success_rate": overall_success_rate,
            "task_macro_success_rate": macro_success_rate,
        },
        "latency": {
            "cold_start_per_worker_ms": cold_starts,
            "wall_inference_ms_including_compile": _latency_summary(latency_samples["wall_inference"]),
            "wall_inference_ms_excluding_compile": _latency_summary(
                latency_samples["steady_state_wall_inference"]
            ),
            "policy_reported_inference_ms_excluding_compile": _latency_summary(
                latency_samples["steady_state_policy_reported_inference"]
            ),
            "history_frame_encoding_ms_excluding_compile": _latency_summary(
                latency_samples["steady_state_history_frame_encoding"]
            ),
        },
        "tasks": tasks,
        "episodes": episodes,
        "shards": shard_specs,
    }


def run(args: argparse.Namespace) -> None:
    if len(set(args.gpus)) != len(args.gpus) or len(args.gpus) > len(TASK_IDS):
        raise ValueError("--gpus must contain between one and ten unique GPU ids")
    if args.split == "final" and args.frozen_config_id is None:
        raise ValueError("--split final requires --frozen-config-id after candidate selection is frozen")
    if args.run_manifest is not None and args.training_seed is None:
        raise ValueError("--training-seed is required with --run-manifest")
    if args.run_manifest is not None and args.frozen_config_id is None:
        raise ValueError("--frozen-config-id is required with --run-manifest")
    if args.run_manifest is not None:
        manifest_seed = int(_run_manifest_audit(args.run_manifest)["factory_args"]["seed"])
        if manifest_seed != args.training_seed:
            raise ValueError(f"run manifest seed {manifest_seed} does not match --training-seed {args.training_seed}")
    output_dir = args.output_dir.resolve()
    _wait_for_training_barrier(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    shard_dir = output_dir / "shards"
    log_dir = output_dir / "logs"
    shard_dir.mkdir()
    log_dir.mkdir()
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    evaluator = repo_root / "scripts/eval_libero_history.py"
    task_partitions = [list(TASK_IDS[index:: len(args.gpus)]) for index in range(len(args.gpus))]
    shard_specs = []
    processes = []
    log_files = []
    manifest = {
        "schema_version": "pi05-libero-multigpu-run-v1",
        "status": "running",
        "run_id": args.run_id,
        "split": args.split,
        "history_intervention": args.history_intervention,
        "episode_indices": list(SPLIT_EPISODES[args.split]),
        "workers": [],
    }

    for gpu, task_ids in zip(args.gpus, task_partitions, strict=True):
        output = shard_dir / f"gpu{gpu}.json"
        log = log_dir / f"gpu{gpu}.log"
        command = [
            sys.executable,
            str(evaluator),
            "--variant",
            args.variant,
            "--checkpoint-dir",
            args.checkpoint_dir,
            "--output",
            str(output),
            "--task-ids",
            *(str(task_id) for task_id in task_ids),
            "--episode-indices",
            *(str(index) for index in SPLIT_EPISODES[args.split]),
            "--seed",
            str(args.eval_seed),
            "--history-intervention",
            args.history_intervention,
        ]
        if args.config_name is not None:
            command.extend(["--config-name", args.config_name])
        if args.run_manifest is not None:
            command.extend(["--run-manifest", str(args.run_manifest.resolve())])
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_file = log.open("w", encoding="utf-8")
        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=repo_root,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        spec = {
            "gpu": gpu,
            "task_ids": task_ids,
            "output": str(output),
            "log": str(log),
            "pid": process.pid,
            "status": "running",
        }
        shard_specs.append(spec)
        processes.append(process)
        log_files.append(log_file)
        manifest["workers"].append(spec)
    _write_json(output_dir / "run.json", manifest)

    failure = None
    while any(process.poll() is None for process in processes):
        for process, spec in zip(processes, shard_specs, strict=True):
            return_code = process.poll()
            if return_code is not None and spec["status"] == "running":
                spec["return_code"] = return_code
                spec["status"] = "completed" if return_code == 0 else "failed"
                if return_code != 0 and failure is None:
                    failure = spec
        if failure is not None:
            for process, spec in zip(processes, shard_specs, strict=True):
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                    spec["return_code"] = process.returncode
                    spec["status"] = "terminated_after_peer_failure"
            break
        time.sleep(0.5)

    for process, spec in zip(processes, shard_specs, strict=True):
        return_code = process.wait()
        if spec["status"] == "running":
            spec["return_code"] = return_code
            spec["status"] = "completed" if return_code == 0 else "failed"
            if return_code != 0 and failure is None:
                failure = spec
    for log_file in log_files:
        log_file.close()

    if failure is not None:
        manifest["status"] = "failed"
        manifest["failure"] = failure
        _write_json(output_dir / "run.json", manifest)
        raise RuntimeError(f"evaluation worker failed: {failure}")

    try:
        result = _aggregate_shards(args, shard_specs)
        if args.baseline_result is not None:
            baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))
            result["paired_to_baseline"] = _paired_comparison(
                result,
                baseline,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed,
            )
        _write_json(output_dir / "result.json", result)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["failure"] = {
            "stage": "aggregation",
            "type": type(error).__name__,
            "message": str(error),
        }
        _write_json(output_dir / "run.json", manifest)
        raise
    manifest["status"] = "completed"
    manifest["result"] = str(output_dir / "result.json")
    _write_json(output_dir / "run.json", manifest)
    print(json.dumps(result["overall"], ensure_ascii=False, indent=2))


def compare(args: argparse.Namespace) -> None:
    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))
    baseline_map = _episode_map(baseline)
    candidates = [json.loads(path.read_text(encoding="utf-8")) for path in args.candidate_results]
    budgets = {
        (
            candidate["run"]["training_run_manifest"]["factory_args"]["num_train_steps"],
            candidate["run"]["training_run_manifest"]["factory_args"]["schedule_total_steps"],
        )
        for candidate in candidates
    }
    if len(budgets) != 1:
        raise ValueError(f"candidate training budgets differ: {sorted(budgets)}")
    num_train_steps, schedule_total_steps = budgets.pop()
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for candidate in candidates:
        config_id = candidate["run"]["frozen_config_id"]
        if config_id is None:
            raise ValueError(f"candidate {candidate['run']['run_id']} has no frozen_config_id")
        groups[config_id].append(candidate)

    configurations = {}
    for config_id, runs in sorted(groups.items()):
        variants = {run["run"]["variant"] for run in runs}
        if len(variants) != 1:
            raise ValueError(f"configuration {config_id} spans multiple variants: {sorted(variants)}")
        variant = variants.pop()
        interventions = {run["run"].get("history_intervention", "normal") for run in runs}
        if len(interventions) != 1:
            raise ValueError(f"configuration {config_id} mixes history interventions: {sorted(interventions)}")
        intervention = interventions.pop()
        configuration_args = {
            json.dumps(
                _configuration_factory_args(run["run"]["training_run_manifest"]),
                sort_keys=True,
            )
            for run in runs
        }
        if len(configuration_args) != 1:
            raise ValueError(f"configuration {config_id} has inconsistent factory arguments across seeds")
        factory_args = json.loads(configuration_args.pop())
        seeds = sorted(run["run"]["training_seed"] for run in runs)
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"duplicate training seeds for {config_id}: {seeds}")
        missing_seeds = sorted(set(args.required_training_seeds) - set(seeds))
        if missing_seeds:
            raise ValueError(f"{config_id} is missing required training seeds: {missing_seeds}")
        per_seed = {}
        candidate_maps = []
        for offset, run_result in enumerate(sorted(runs, key=lambda result: result["run"]["training_seed"])):
            seed = run_result["run"]["training_seed"]
            comparison = _paired_comparison(
                run_result,
                baseline,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed + offset,
            )
            comparison["task_macro_success_rate"] = run_result["overall"]["task_macro_success_rate"]
            per_seed[str(seed)] = comparison
            candidate_map = _episode_map(run_result)
            if candidate_map.keys() != baseline_map.keys():
                raise ValueError(f"candidate {config_id} seed {seed} does not match baseline episodes")
            candidate_maps.append(candidate_map)

        averaged_differences: dict[int, list[float]] = collections.defaultdict(list)
        averaged_task_rates = []
        for task_id in TASK_IDS:
            keys = sorted(key for key in baseline_map if key[0] == task_id)
            averaged_success = [
                statistics.fmean(bool(candidate_map[key]["success"]) for candidate_map in candidate_maps)
                for key in keys
            ]
            averaged_task_rates.append(statistics.fmean(averaged_success))
            averaged_differences[task_id].extend(
                current - float(bool(baseline_map[key]["success"]))
                for current, key in zip(averaged_success, keys, strict=True)
            )
        configurations[config_id] = {
            "variant": variant,
            "history_intervention": intervention,
            "configuration_factory_args": factory_args,
            "training_seeds": seeds,
            "per_training_seed": per_seed,
            "seed_average": {
                "task_macro_success_rate": statistics.fmean(averaged_task_rates),
                "paired_macro_success_rate_delta": _bootstrap_task_stratified(
                    averaged_differences,
                    replicates=args.bootstrap_replicates,
                    seed=args.bootstrap_seed,
                ),
                "independence_note": (
                    "Training seeds are averaged within each task/initial-state cell before bootstrap; "
                    "repeated initial states are not counted as additional independent episodes."
                ),
            },
        }

    result = {
        "schema_version": "pi05-libero-paired-comparison-v1",
        "baseline": {
            "run_id": baseline["run"]["run_id"],
            "frozen_config_id": baseline["run"]["frozen_config_id"],
            "variant": baseline["run"]["variant"],
            "split": baseline["run"]["split"],
            "eval_seed": baseline["run"]["eval_seed"],
            "history_intervention": baseline["run"].get("history_intervention", "normal"),
            "episodes": len(baseline["episodes"]),
            "task_macro_success_rate": baseline["overall"]["task_macro_success_rate"],
        },
        "shared_training_budget": {
            "num_train_steps": num_train_steps,
            "schedule_total_steps": schedule_total_steps,
        },
        "required_training_seeds": sorted(args.required_training_seeds),
        "configurations": configurations,
    }
    _write_json(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--variant", choices=("H0", "H1", "H2", "H3", "H4", "H5"), required=True)
    run_parser.add_argument("--checkpoint-dir", required=True)
    config_source = run_parser.add_mutually_exclusive_group(required=True)
    config_source.add_argument("--config-name")
    config_source.add_argument("--run-manifest", type=pathlib.Path)
    run_parser.add_argument("--training-seed", type=int)
    run_parser.add_argument("--eval-seed", type=int, default=7)
    run_parser.add_argument("--split", choices=tuple(SPLIT_EPISODES), default="dev")
    run_parser.add_argument(
        "--history-intervention",
        choices=("normal", "drop_all", "mask_numeric", "shuffle_time_ood"),
        default="normal",
    )
    run_parser.add_argument("--frozen-config-id")
    run_parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    run_parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    run_parser.add_argument("--baseline-result", type=pathlib.Path)
    run_parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    run_parser.add_argument("--bootstrap-seed", type=int, default=20260926)
    run_parser.set_defaults(handler=run)

    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--baseline-result", type=pathlib.Path, required=True)
    compare_parser.add_argument("--candidate-results", type=pathlib.Path, nargs="+", required=True)
    compare_parser.add_argument("--output", type=pathlib.Path, required=True)
    compare_parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    compare_parser.add_argument("--bootstrap-seed", type=int, default=20260926)
    compare_parser.add_argument("--required-training-seeds", type=int, nargs="+", default=[11, 23, 47])
    compare_parser.set_defaults(handler=compare)
    return parser


def main() -> None:
    args = _parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
