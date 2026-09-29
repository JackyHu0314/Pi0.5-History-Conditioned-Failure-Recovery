# Paired multi-GPU LIBERO-10 evaluation

`scripts/eval_libero_multigpu.py` runs the clean closed-loop protocol and keeps
one policy process alive for every GPU worker. With one GPU, that worker runs
all ten tasks. With eight GPUs, the ten tasks are divided round-robin, so six
workers run one task and two workers run two tasks. A checkpoint is loaded
exactly once per worker.

The fixed protocol is:

- `dev`: initial-state indices 0 through 9, 100 episodes total.
- `final`: initial-state indices 10 through 49, 400 episodes total.
- Environment and policy evaluation seed: 7 for the official baseline and all
  candidates.
- Clean `normal` history condition only.
- Candidate training seeds: 11, 23, and 47, reported separately and then
  averaged within each task/initial-state cell.
- Every trained candidate has an explicit `--frozen-config-id`, such as `B1`,
  `R2`, or `R3`. The model variant (`H0`--`H5`) is only its module family.

Every LIBERO-10 task in the pinned simulator has exactly 50 fixed initial
states. A final run requires an explicit frozen configuration identifier. Do
not assign that identifier until model selection on `dev` is complete.

Run the official baseline on eight GPUs:

```bash
python scripts/eval_libero_multigpu.py run \
  --run-id official-pi05-libero-dev \
  --variant H0 \
  --config-name pi05_libero \
  --checkpoint-dir /ABS/weights/pi05_libero \
  --frozen-config-id B0 \
  --split dev \
  --eval-seed 7 \
  --gpus 0 1 2 3 4 5 6 7 \
  --output-dir /ABS/evals/official-pi05-libero-dev
```

Run one history checkpoint on one GPU. The sibling run manifest is named
`<checkpoint_run_dir>.run.json`; it reconstructs the exact model injection,
training scope, normalization, and cache identity used during training.

```bash
python scripts/eval_libero_multigpu.py run \
  --run-id R2-seed11-dev \
  --variant H2 \
  --run-manifest /ABS/checkpoints/H2-seed11.run.json \
  --checkpoint-dir /ABS/checkpoints/H2-seed11 \
  --training-seed 11 \
  --frozen-config-id R2 \
  --split dev \
  --eval-seed 7 \
  --gpus 3 \
  --baseline-result /ABS/evals/official-pi05-libero-dev/result.json \
  --output-dir /ABS/evals/R2-seed11-dev
```

The same command with `--gpus 0 1 2 3 4 5 6 7` evaluates one candidate with
eight task-sharded workers. To evaluate eight different candidates
concurrently, launch eight commands and give each command one distinct GPU.

Compare all three training seeds for every candidate configuration:

```bash
python scripts/eval_libero_multigpu.py compare \
  --baseline-result /ABS/evals/official-pi05-libero-dev/result.json \
  --candidate-results \
    /ABS/evals/B1-seed11-dev/result.json \
    /ABS/evals/B1-seed23-dev/result.json \
    /ABS/evals/B1-seed47-dev/result.json \
    /ABS/evals/R2-seed11-dev/result.json \
    /ABS/evals/R2-seed23-dev/result.json \
    /ABS/evals/R2-seed47-dev/result.json \
  --output /ABS/evals/dev-comparison.json
```

The comparison groups runs by `frozen_config_id`, never by `variant`. It rejects
different factory arguments under one configuration ID, missing or repeated
task/initial-state pairs, mismatched episode seeds, duplicate training seeds,
missing required training seeds, and different training-step budgets. It
reports the variant as the module family, each training seed separately, the
mean across seeds, paired candidate-only/baseline-only outcomes, and a 95%
bootstrap interval. Bootstrap resampling occurs within each task over the ten
or forty unique initial states. Repeated evaluation of the same initial state
under multiple training seeds is averaged before resampling and is never
counted as an additional independent episode.

After freezing one configuration, run `final` with the same identifier for the
campaign:

```bash
python scripts/eval_libero_multigpu.py run \
  --run-id R2-seed11-final \
  --variant H2 \
  --run-manifest /ABS/checkpoints/H2-seed11.run.json \
  --checkpoint-dir /ABS/checkpoints/H2-seed11 \
  --training-seed 11 \
  --split final \
  --frozen-config-id R2 \
  --eval-seed 7 \
  --gpus 0 1 2 3 4 5 6 7 \
  --output-dir /ABS/evals/R2-seed11-final
```

Each output directory contains `run.json`, one log and result shard per worker,
and the strict aggregate `result.json`. Any worker failure terminates the
remaining workers and records the failing worker in `run.json`. Aggregation
failures are also recorded. Latency includes the exact first compile call for
each worker in a separate field and reports steady-state distributions with
that call removed.

After model selection and the clean `final` run are complete, evaluate the
selected history configuration and training seed with each of the three
diagnostics by adding exactly one of:

```text
--history-intervention drop_all
--history-intervention mask_numeric
--history-intervention shuffle_time_ood
```

Keep `--split final`, the same evaluation seed, checkpoint, configuration ID,
and initial-state indices. Compare each diagnostic result to that model's
`normal` final result with `compare --required-training-seeds 11` (or the
selected training seed). These diagnostics measure dependence on history and
are not inputs to model selection.
