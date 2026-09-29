# pi0.5 history evaluation on LIBERO-10

The simulator source is fixed to the submodule revision used by the openpi base
tree (`215abfb217dbac7d5f1273282331b9b1866c0479`):

```text
LIBERO f78abd68ee283de9f9be3c8f7e2a9ad60246e95c
```

Use the existing Python 3.11 openpi environment.  Do not install LIBERO's old
`requirements.txt`: it pins Python 3.8-era NumPy and Torch.  Install only the
simulator packages that are absent, without dependency resolution changing the
existing JAX, Torch, NumPy, or CUDA stack:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/libero
git -C third_party/libero checkout f78abd68ee283de9f9be3c8f7e2a9ad60246e95c

python -m pip install --no-deps -e third_party/libero
python -m pip install --no-deps \
  robosuite==1.4.1 mujoco==3.2.3 bddl==1.0.1 gym==0.25.2 \
  easydict==1.9 cloudpickle==2.1.0 future==0.18.2 \
  pynput==1.7.7 termcolor==2.4.0 pyyaml==6.0.2
```

Before the first `import libero`, create `~/.libero/config.yaml`; otherwise this
pinned revision asks an interactive setup question.  If `OPENPI_ROOT` is the
absolute repository root, the file is:

```yaml
benchmark_root: OPENPI_ROOT/third_party/libero/libero/libero
bddl_files: OPENPI_ROOT/third_party/libero/libero/libero/bddl_files
init_states: OPENPI_ROOT/third_party/libero/libero/libero/init_files
datasets: OPENPI_ROOT/third_party/libero/libero/datasets
assets: OPENPI_ROOT/third_party/libero/libero/libero/assets
```

Replace `OPENPI_ROOT` with the actual absolute path.  LIBERO-10 BDDL files,
initial states, and simulator assets are already present in the pinned source.

The environment imports Matplotlib directly, while robosuite imports NumPy,
SciPy, Numba, Pillow, and OpenCV.  Reuse the versions already present in the
openpi environment.  On Python 3.11, add a current Matplotlib and a compatible
Numba (0.60 or 0.61) only if the import probe reports either missing.  MuJoCo
may additionally need `absl-py`, `etils[epath]`, `glfw`, and `PyOpenGL`;
install those only when absent.  The evaluator calls `torch.load(...,
weights_only=False)` for the pinned official LIBERO initial-state files, so
Torch 2.6+ does not require a package downgrade or a source patch.

Probe the simulator without loading a policy:

```bash
MUJOCO_GL=egl python - <<'PY'
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
import mujoco
import robosuite

suite = benchmark.get_benchmark_dict()["libero_10"]()
print(mujoco.__version__, robosuite.__version__, suite.n_tasks)
PY
```

Run one normal evaluation on the latest numeric checkpoint step. New training
runs should use their sibling run manifest so the evaluator reconstructs the
exact history injection and training scope and verifies the cache and norm-stat
hashes:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python scripts/eval_libero_history.py \
  --variant H2 \
  --checkpoint-dir /ABS/checkpoints/pi05_libero_history_h2/H2-seed11 \
  --run-manifest /ABS/checkpoints/pi05_libero_history_h2/H2-seed11.run.json \
  --episodes-per-task 3 \
  --seed 11 \
  --history-intervention normal \
  --output /ABS/results/H2-seed11-normal.json
```

Run the paired history-dropout diagnostic with the same checkpoint, seed, and
fixed initial-state indices by changing only these arguments:

```bash
--history-intervention drop_all \
--output /ABS/results/H2-seed11-drop_all.json
```

Load the untouched official LIBERO policy config and checkpoint, restrict the
run to three task/initial-state pairs, and retain diagnostic artifacts:

```bash
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl python scripts/eval_libero_history.py \
  --variant H0 \
  --config-name pi05_libero \
  --checkpoint-dir /ABS/weights/pi05_libero \
  --task-ids 0 1 2 \
  --episode-indices 0 \
  --seed 7 \
  --video-dir /ABS/results/official-pi05-libero/videos \
  --trace-dir /ABS/results/official-pi05-libero/traces \
  --output /ABS/results/official-pi05-libero/result.json
```

`--config-name` selects a registered upstream config through
`training.config.get_config`; `--run-manifest` selects an H0--H5 experiment. An
explicit `--episode-indices` list takes the place of `--episodes-per-task`.
Each trace JSON contains every executed action with its before/after state,
reward, terminal flag, originating replan step, and every raw action chunk
returned by the policy.  Videos contain the 180-degree-rotated base camera at
the 20 Hz control rate.  Both artifact outputs are disabled unless their
directories are supplied.

H0 has no history input and is evaluated only with `normal`.  The evaluator
uses standard LIBERO success, never the synthetic T/E condition labels.  It
keeps the official evaluator's 20 Hz simulator control rate; the LeRobot
container's 10 FPS metadata describes the stored pilot data and does not change
closed-loop action execution.  It
records every episode's success, control steps, fixed initial-state index,
policy wall latency, and the extra latency for encoding each newly observed
frame once with the checkpoint's frozen SigLIP.  It prints one JSON episode row
immediately, then atomically writes the final aggregate JSON.

Use `scripts/eval_libero_multigpu.py` for the locked dev/final protocol and
paired statistics described in `examples/libero/MULTIGPU_EVAL.md`.
