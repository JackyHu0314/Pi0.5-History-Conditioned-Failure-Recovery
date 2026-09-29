"""Wait for the recovery campaign, probe trained checkpoints, and finalize the report."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT / "recovery_distill_campaign_20260929"
TRAIN_WINDOWS = ROOT / "recovery_distill_train_20260929/teacher_windows"
TEST_WINDOWS = ROOT / "recovery_distill_20260929/teacher_windows"
PYTHON = Path(os.environ.get("PI05_PYTHON", sys.executable))
STATUS = {"status": "WAITING", "started_unix": time.time(), "jobs": {}}


def write() -> None:
    path = CAMPAIGN / "post_analysis_status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(STATUS, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    write()
    while True:
        campaign_status = json.loads((CAMPAIGN / "status.json").read_text(encoding="utf-8"))
        if campaign_status["status"] == "COMPLETED":
            break
        if campaign_status["status"] == "FAILED":
            STATUS["status"] = "BLOCKED_BY_CAMPAIGN_FAILURE"
            write()
            raise SystemExit(1)
        time.sleep(30)
    report = json.loads((CAMPAIGN / "report.json").read_text(encoding="utf-8"))
    STATUS["status"] = "RUNNING"
    write()
    for name in report["selected_seed_runs"]:
        checkpoint_root = CAMPAIGN / "checkpoints/pi05_libero_history_h5" / name
        output = CAMPAIGN / "post_probes" / f"{name}.json"
        log = CAMPAIGN / "logs" / f"probe_{name}.log"
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(PYTHON), str(ROOT / "server_runtime_dropout_train.py"),
            str(ROOT / "run_recovery_history_probe.py"),
            "--run-manifest", str(checkpoint_root.with_suffix(".run.json")),
            "--checkpoint-dir", str(checkpoint_root / "1499"),
            "--train-root", str(TRAIN_WINDOWS),
            "--dev-root", str(TRAIN_WINDOWS),
            "--test-root", str(TEST_WINDOWS),
            "--output", str(output),
        ]
        STATUS["jobs"][name] = {"status": "running", "command": command, "started_unix": time.time()}
        write()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="0", XLA_PYTHON_CLIENT_PREALLOCATE="false")
        with log.open("w", encoding="utf-8", buffering=1) as handle:
            process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
        STATUS["jobs"][name].update(
            status="completed" if not process.returncode else "failed",
            returncode=process.returncode,
            finished_unix=time.time(),
            output=str(output),
            log=str(log),
        )
        write()
        if process.returncode:
            STATUS["status"] = "FAILED"
            write()
            raise SystemExit(1)
    command = [
        str(PYTHON), str(ROOT / "finalize_recovery_distill_results.py"),
        "--campaign-root", str(CAMPAIGN),
    ]
    log = CAMPAIGN / "logs/finalize_post_analysis.log"
    with log.open("w", encoding="utf-8", buffering=1) as handle:
        process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    STATUS["finalizer"] = {
        "returncode": process.returncode,
        "command": command,
        "log": str(log),
    }
    STATUS["status"] = "COMPLETED" if not process.returncode else "FAILED"
    STATUS["finished_unix"] = time.time()
    write()
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
