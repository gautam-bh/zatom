"""T0-8: the production run survives preemption and time limits under the overrun QOS.

Overrun jobs can be preempted after 2 hours and may be granted less than `--time`. Three
things have to hold for a run to carry on from where it stopped instead of restarting from
the pretrained weights:
  1. step checkpoints (the only kind `train_fm.py` resumes from) are written often enough;
  2. the job is requeued on preemption *and* on its time limit;
  3. the requeued job finds the same run dir, checkpoint dir and W&B run.
"""

import datetime
import os
import re
import signal
import subprocess
import time

import hydra

from tests.predeploy.launcher import LAUNCH_DIR, PROD_LAUNCHER, SMOKE_LAUNCHER, sbatch_directives
from tests.predeploy.test_t0_config import _launcher_cfg

REQUEUE_LIB = LAUNCH_DIR / "lora_bandgap_requeue.sh"

# The smoke run measured 1.14 batches/s per rank at batch 8, i.e. ~157 optimizer steps
# (2520 batches / accumulate 16) per ~37 min epoch. A step-count interval would have to be
# tiny to matter; job 61064951 shows 5000 steps is more than the whole 11-hour run.
MAX_LOST_TRAINING = datetime.timedelta(minutes=30)


def _instantiate(cb_cfg):
    return hydra.utils.instantiate(cb_cfg)


def test_step_checkpoints_are_time_based_and_frequent():
    """A preemption loses at most MAX_LOST_TRAINING of training."""
    cfg = _launcher_cfg(PROD_LAUNCHER)
    ckpt = _instantiate(cfg.callbacks.last_model_checkpoint)
    assert ckpt._train_time_interval is not None, "periodic checkpoints must be time-based"
    assert ckpt._train_time_interval <= MAX_LOST_TRAINING
    assert ckpt.save_top_k != 0
    # The file names must stay resumable: `find_latest_step_checkpoint` only picks `{epoch}-{step}`
    assert cfg.callbacks.last_model_checkpoint.filename is None


def test_smoke_launcher_checkpoint_overrides_are_valid():
    """The smoke test's step-count override must not clash with the time interval."""
    cfg = _launcher_cfg(SMOKE_LAUNCHER)
    ckpt = _instantiate(cfg.callbacks.last_model_checkpoint)  # raises on a clash
    assert ckpt._every_n_train_steps == 2


def test_launcher_requests_requeue_and_a_timeout_warning():
    text = PROD_LAUNCHER.read_text()
    directives = sbatch_directives(PROD_LAUNCHER)
    assert re.search(r"^#SBATCH --requeue\b", text, re.M), "preempted jobs must be requeued"
    # Requeued jobs keep their job id; without append the rerun truncates the first logs
    assert directives.get("open-mode") == "append"
    signal_spec = re.fullmatch(r"B:USR1@(\d+)", directives.get("signal", ""))
    assert signal_spec, "the batch shell (B:) needs USR1 before the time limit"
    assert int(signal_spec.group(1)) >= 120
    assert "time-min" in directives, "overrun requires --time-min"
    assert directives["qos"] == "overrun"


def test_launcher_uses_requeue_helpers_before_paths_are_built():
    text = PROD_LAUNCHER.read_text()
    source = text.index("lora_bandgap_requeue.sh")
    identity = text.index("load_or_save_run_identity")
    assert source < identity < text.index("RUN_DIR_NAME=")
    assert re.search(r"run_with_timeout_requeue bash -c \"", text)


def _bash(script, env=None, **kwargs):
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        **kwargs,
    )


def test_requeued_start_reuses_the_first_starts_identity(tmp_path):
    state = tmp_path / "state" / "job.env"
    show = f'source {REQUEUE_LIB}; load_or_save_run_identity {state}; echo "$RUN_ID $RUN_DATE $ARCHITECTURE"'
    first = _bash(show, {"RUN_ID": "aaaa1111", "RUN_DATE": "2026-01-01_00-00-00", "ARCHITECTURE": "tft_160M"})
    # The rerun's defaults are fresh (new random id, new date), as in the launcher
    rerun = _bash(show, {"RUN_ID": "bbbb2222", "RUN_DATE": "2026-01-02_09-09-09", "ARCHITECTURE": "tft_160M"})
    assert first.stdout.split()[-3:] == ["aaaa1111", "2026-01-01_00-00-00", "tft_160M"]
    assert rerun.stdout.split()[-3:] == ["aaaa1111", "2026-01-01_00-00-00", "tft_160M"]
    assert "reusing run identity" in rerun.stdout


def test_timeout_warning_requeues_and_still_waits_for_training(tmp_path):
    shim = tmp_path / "bin"
    shim.mkdir()
    calls = tmp_path / "scontrol_calls"
    (shim / "scontrol").write_text(f'#!/bin/bash\necho "$@" >> {calls}\n')
    (shim / "scontrol").chmod(0o755)
    script = (
        f"source {REQUEUE_LIB}; "
        "run_with_timeout_requeue bash -c 'sleep 3; exit 7'; "
        'echo "status=$TRAIN_STATUS"'
    )
    proc = subprocess.Popen(
        ["bash", "-c", script],
        stdout=subprocess.PIPE,
        text=True,
        env={**os.environ, "PATH": f"{shim}:{os.environ['PATH']}", "SLURM_JOB_ID": "12345"},
    )
    time.sleep(1)
    os.kill(proc.pid, signal.SIGUSR1)
    out, _ = proc.communicate(timeout=30)
    assert calls.read_text().split() == ["requeue", "12345"]
    assert "status=7" in out, out  # the training command's status, not the trap's 138
