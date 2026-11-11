"""T2-4: a preempted run resumes mid-epoch from its last timed checkpoint and keeps training.

Emulates what overrun does to the production run on one GPU (a login node is enough):
  1. start `train_fm.py` with the production config, but periodic checkpoints every minute;
  2. once a step checkpoint exists, SIGTERM it (SLURM's preemption signal; SIGKILL after 30 s);
  3. rerun the identical command, as a requeued job would (same run dir and checkpoint dir).
Passes if the rerun restores the newest step checkpoint, trains past that step, and carries
the Lightning `Timer` forward, so `trainer.max_time` caps the run's total training time
rather than restarting with each requeue.

Usage (repo root, inside the job image, GPU visible):
    shifter --image=registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1 --module=gpu \
        --env=PROJECT_ROOT=$PWD python tests/predeploy/gpu/t2_4_preempt_resume.py <pretrained.ckpt>
"""

import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, ".")
from zatom.train_fm import STEP_CKPT_PATTERN, find_latest_step_checkpoint  # noqa: E402

PRETRAINED = sys.argv[1]
RUN_DATE = "t2_4_preempt"
NAME = "t2_4_preempt_resume"
CKPT_DIR = Path(f"/pscratch/sd/{os.environ['USER'][0]}/{os.environ['USER']}/zatom_checkpoints/t2_4/checkpoints")
LOG_DIR = Path("scripts/perlmutter/regular/logs")
RUN_DIR = Path(f"logs/finetune_fm/runs/{NAME}_{RUN_DATE}")


def command():
    return [
        "python", "zatom/train_fm.py",
        "experiment=lora_bandgap",
        "logger=csv",
        "data.datamodule.batch_size.train=2",
        "data.datamodule.batch_size.val=2",
        "data.datamodule.batch_size.test=2",
        "trainer.accumulate_grad_batches=4",
        f"pretrained_ckpt_path={PRETRAINED}",
        f"ckpt_path={CKPT_DIR}",
        f"callbacks.model_checkpoint.dirpath={CKPT_DIR}",
        f"callbacks.last_model_checkpoint.dirpath={CKPT_DIR}",
        "callbacks.last_model_checkpoint.train_time_interval.minutes=1",
        "model/architecture=tft_160M",
        f"date={RUN_DATE}",
        f"name={NAME}",
        "task_name=finetune_fm",
        "trainer.num_nodes=1",
        "trainer.devices=1",
        "trainer.max_epochs=1",
        "trainer.max_time=00:00:40:00",
        "++trainer.limit_train_batches=1200",
        "++trainer.limit_val_batches=2",
        "++trainer.limit_test_batches=1",
    ]


def timer_elapsed(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    timer = next(v for k, v in state["callbacks"].items() if k.startswith("Timer"))
    return state["global_step"], timer["time_elapsed"]["train"]


def main():
    shutil.rmtree(CKPT_DIR, ignore_errors=True)
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    CKPT_DIR.mkdir(parents=True)
    env = {**os.environ, "HYDRA_FULL_ERROR": "1", "PYTHONUNBUFFERED": "1"}

    # 1. First allocation: train until a timed step checkpoint has landed, then "preempt"
    log1 = open(LOG_DIR / "t2_4_run1.log", "w")
    run1 = subprocess.Popen(command(), stdout=log1, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    deadline = time.time() + 30 * 60
    while time.time() < deadline and run1.poll() is None:
        steps = [f for f in os.listdir(CKPT_DIR) if STEP_CKPT_PATTERN.match(f)]
        if len(steps) >= 2:  # >= 2 timed checkpoints, so the time interval demonstrably repeats
            break
        time.sleep(10)
    assert run1.poll() is None, f"run 1 exited ({run1.returncode}) before 2 step checkpoints; see {log1.name}"
    time.sleep(20)  # be mid-way between checkpoints, like a real preemption
    os.killpg(run1.pid, signal.SIGTERM)
    try:
        run1.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(run1.pid, signal.SIGKILL)
        run1.wait()
    latest = find_latest_step_checkpoint(str(CKPT_DIR))
    step1, elapsed1 = timer_elapsed(latest)
    print(f"run 1 preempted; newest step checkpoint {Path(latest).name}: step {step1}, timer {elapsed1:.0f} s")
    exception_ckpts = [f for f in os.listdir(CKPT_DIR) if "-v" in f]
    assert not exception_ckpts, f"preemption wrote versioned/exception checkpoints: {exception_ckpts}"

    # 2. Requeued allocation: the identical command
    log2_path = LOG_DIR / "t2_4_run2.log"
    with open(log2_path, "w") as log2:
        status = subprocess.run(command(), stdout=log2, stderr=subprocess.STDOUT, env=env).returncode
    text = log2_path.read_text(errors="replace")
    assert status == 0, f"run 2 failed ({status}); see {log2_path}"
    restored = re.search(r"Restored all states from the checkpoint at (\S+)", text)
    assert restored, "run 2 did not restore a checkpoint"
    assert Path(restored.group(1)).name == Path(latest).name, (restored.group(1), latest)

    final = find_latest_step_checkpoint(str(CKPT_DIR))
    last = CKPT_DIR / "last.ckpt"
    step2, elapsed2 = timer_elapsed(last if last.exists() else final)
    print(f"run 2 finished at step {step2}, timer {elapsed2:.0f} s")
    assert step2 > step1, "run 2 restored but trained no new steps"
    # At run 1's pace, run 2's own steps take ~elapsed1 * (step2 - step1) / step1. A Timer that
    # restarted at zero would report about that; a carried-over one adds elapsed1 on top.
    run2_alone = elapsed1 * (step2 - step1) / step1
    assert elapsed2 > run2_alone + 0.5 * elapsed1, (
        f"Timer restarted from zero ({elapsed2:.0f} s ~ {run2_alone:.0f} s for run 2 alone): "
        "max_time would not bound total training"
    )
    # Known and harmless: the restored checkpoint's own file is missing from the top-k list
    # it carries, so each preemption leaves one extra step checkpoint; resume picks the newest.
    step_ckpts = [f for f in os.listdir(CKPT_DIR) if STEP_CKPT_PATTERN.match(f)]
    assert len(step_ckpts) <= 3, f"step checkpoints are not being pruned: {sorted(step_ckpts)}"
    print("T2-4 PASS: preempted run resumed mid-epoch from its last timed checkpoint and kept training")


if __name__ == "__main__":
    main()
