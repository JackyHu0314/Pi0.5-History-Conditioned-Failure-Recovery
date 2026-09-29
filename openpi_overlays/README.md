# OpenPI overlays

This directory contains only files added to or changed from OpenPI commit
`215abfb217dbac7d5f1273282331b9b1866c0479`.

- `training/`: history tokens, history compressor, LIBERO cache/dataset and
  multi-GPU training changes.
- `evaluation_delta/`: changes applied on top of `training/` for partial
  observability, last-valid-frame and controlled-slip evaluation.

Create separate training and evaluation checkouts:

```bash
git clone https://github.com/Physical-Intelligence/openpi.git openpi-train
git -C openpi-train checkout 215abfb217dbac7d5f1273282331b9b1866c0479
python tools/apply_openpi_overlay.py --openpi openpi-train --mode training

git clone https://github.com/Physical-Intelligence/openpi.git openpi-eval
git -C openpi-eval checkout 215abfb217dbac7d5f1273282331b9b1866c0479
python tools/apply_openpi_overlay.py --openpi openpi-eval --mode evaluation
```
