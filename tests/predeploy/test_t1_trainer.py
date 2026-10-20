"""T1-8 / T1-10: real Lightning `fit` runs on CPU with the LoRA band gap config.

T1-10: one process, a couple of batches and one validation batch; the checkpoint monitor
metric must exist afterwards, or the real run's best-checkpoint selection never fires and
`trainer.test()` has nothing to load.

T1-8: two processes under DDP (gloo). Catches the unused-parameter crash of job 56923536,
and checks a dead rank takes the job down within the collective timeout (I-4) instead of
hanging for 30 minutes as in job 56395971.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from lightning import Trainer

from tests.predeploy.conftest import REPO_ROOT, build_model, compose_cfg
from zatom.train_fm import best_checkpoint_path
from zatom.utils import instantiate_callbacks

DDP_SCRIPT = Path(__file__).with_name("ddp_cpu_run.py")


def test_t1_10_monitor_metric_is_logged(datamodule, tmp_path):
    cfg = compose_cfg(
        [
            f"callbacks.model_checkpoint.dirpath={tmp_path}",
            f"callbacks.last_model_checkpoint.dirpath={tmp_path}",
            f"model.sampling.save_dir={tmp_path}",
            "model.sampling.num_samples=2",
            "model.sampling.batch_size=2",
            "model.sampling.steps=2",
            "~callbacks.epoch_progress",
            "~callbacks.learning_rate_monitor",  # needs a logger; checkpointing doesn't
        ]
    )
    model = build_model(cfg)
    model.strict_loading = False
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        callbacks=instantiate_callbacks(cfg.callbacks, using_logger=True),
        default_root_dir=str(tmp_path),
        enable_progress_bar=False,
    )
    trainer.fit(model=model, datamodule=datamodule)

    monitor = cfg.callbacks.model_checkpoint.monitor
    assert monitor in trainer.callback_metrics, sorted(trainer.callback_metrics)
    assert list(tmp_path.glob("lora-bandgap-epoch@*.ckpt")), list(tmp_path.iterdir())
    # `trainer.test()` must load model_checkpoint's best, not last_model_checkpoint's latest step
    best = best_checkpoint_path(trainer, monitor)
    assert Path(best).name.startswith("lora-bandgap-epoch@"), best


def _run_ddp(tmp_path, **env):
    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(DDP_SCRIPT)],
        cwd=REPO_ROOT,
        env={**os.environ, "DDP_OUT_DIR": str(tmp_path), "PROJECT_ROOT": str(REPO_ROOT), **env},
        capture_output=True,
        text=True,
        timeout=900,
    )
    return proc, time.monotonic() - started


@pytest.mark.slow
def test_t1_8_two_rank_ddp_trains(datamodule, tmp_path):
    proc, _ = _run_ddp(tmp_path)
    tail = (proc.stdout + proc.stderr)[-4000:]
    assert proc.returncode == 0, tail
    assert "DDP_RUN_OK" in proc.stdout, tail
    assert "Expected to have finished reduction" not in proc.stderr


@pytest.mark.slow
def test_t1_8_dead_rank_fails_fast(datamodule, tmp_path):
    proc, elapsed = _run_ddp(tmp_path, DDP_CRASH_RANK0="1", DDP_TIMEOUT_S="30")
    assert proc.returncode != 0
    assert elapsed < 300, f"job took {elapsed:.0f}s to die after rank 0 failed"
