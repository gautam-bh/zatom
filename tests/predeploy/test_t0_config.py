"""T0: static checks on the launchers and the config they compose to. No model build.

Every test here maps to a failed Perlmutter job or to a silent misconfiguration; see §1 and
§3 of docs/plans/2026-10-19-lora-bandgap-predeploy-tests.md.
"""

import datetime
import os
import re
import subprocess
from pathlib import Path

import hydra
import pytest
import torch

from tests.predeploy.conftest import compose_cfg
from tests.predeploy.launcher import (
    LORA_LAUNCHERS,
    PROD_LAUNCHER,
    SMOKE_LAUNCHER,
    launcher_env,
    sbatch_directives,
    train_overrides,
)

LAUNCHERS = (PROD_LAUNCHER, SMOKE_LAUNCHER)


def _launcher_cfg(path):
    return compose_cfg(train_overrides(path), tiny=False, cpu=False)


@pytest.mark.parametrize("path", LAUNCHERS, ids=lambda p: p.name)
def test_t0_1_launcher_overrides_compose(path):
    """T0-1: the exact override list each launcher passes resolves with no missing keys."""
    from omegaconf import OmegaConf

    cfg = _launcher_cfg(path)
    OmegaConf.to_container(cfg.model, resolve=True)
    OmegaConf.to_container(cfg.data, resolve=True)
    OmegaConf.to_container(cfg.callbacks, resolve=True)
    OmegaConf.to_container(cfg.trainer, resolve=True)


@pytest.mark.parametrize("path", LAUNCHERS, ids=lambda p: p.name)
def test_t0_2_resolved_values_match_launcher(path):
    """T0-2: batch size, accumulation, topology and time limits are what we think they are."""
    cfg = _launcher_cfg(path)
    directives = sbatch_directives(path)
    dm = cfg.data.datamodule
    assert dm.batch_size.train == 8
    assert cfg.trainer.accumulate_grad_batches == 16
    assert cfg.model.augmentations.multiplicity == 8
    assert cfg.trainer.devices == int(directives["ntasks-per-node"])
    assert cfg.trainer.num_nodes == int(directives["nodes"])
    assert int(directives["gpus-per-node"]) == cfg.trainer.devices
    assert cfg.trainer.check_val_every_n_epoch == 1
    assert cfg.model.finetune_loss == "generative"

    def seconds(dhms):
        parts = [int(x) for x in str(dhms).split(":")]
        return sum(v * s for v, s in zip(reversed(parts), (1, 60, 3600, 86400)))

    assert seconds(cfg.trainer.max_time) < seconds(directives["time"]), (
        cfg.trainer.max_time,
        directives["time"],
    )


@pytest.mark.parametrize("path", LORA_LAUNCHERS, ids=lambda p: p.name)
def test_t0_3_launchers_parse(path):
    """T0-3: every launcher is valid bash."""
    subprocess.run(["bash", "-n", str(path)], check=True)


@pytest.mark.parametrize("path", LAUNCHERS, ids=lambda p: p.name)
def test_t0_3_run_id_generation(path):
    """T0-3 / job 55419435: the run-id line yields a non-empty W&B-safe id without host python."""
    line = next(
        line for line in path.read_text().splitlines() if line.startswith("DEFAULT_RUN_ID=")
    )
    assert "python" not in line
    cmd = line.split("=", 1)[1].split("  #")[0].strip()
    # macOS `tr` rejects non-UTF-8 bytes unless LC_ALL=C; Linux (Perlmutter) does not care
    out = subprocess.run(
        ["bash", "-c", f"printf '%s' {cmd}"],
        capture_output=True,
        text=True,
        env={**os.environ, "LC_ALL": "C"},
        check=True,
    ).stdout
    assert re.fullmatch(r"(smoke)?[a-z0-9]{6,8}", out), out


def _have_wandb_creds():
    netrc = Path.home() / ".netrc"
    return bool(os.environ.get("WANDB_API_KEY")) or (
        netrc.exists() and "api.wandb.ai" in netrc.read_text()
    )


@pytest.mark.skipif(not _have_wandb_creds(), reason="no W&B credentials on this machine")
def test_t0_4_wandb_preflight():
    """T0-4 / jobs 55777952, 56395971: the same authenticated round trip the launchers make."""
    import wandb

    assert wandb.Api().viewer.username


@pytest.mark.parametrize("path", LAUNCHERS, ids=lambda p: p.name)
def test_t0_5_run_dir_and_ckpt_path(path):
    """T0-5: Hydra's run dir is the launcher's RUN_DIR_NAME, and checkpoints go to pscratch."""
    env = launcher_env(path)
    cfg = _launcher_cfg(path)
    assert Path(cfg.hydra.run.dir).name == env["RUN_DIR_NAME"]
    assert env["CKPT_PATH"].startswith("/pscratch/")
    assert env["RUN_DIR_NAME"] in env["CKPT_PATH"]
    assert cfg.callbacks.model_checkpoint.dirpath == env["CKPT_PATH"]
    assert cfg.callbacks.last_model_checkpoint.dirpath == env["CKPT_PATH"]


def test_t0_6_strategy_fails_loudly_and_fast():
    """T0-6: unused params stay an error (I-1) and a dead rank can't hang the job 30 min (I-4)."""
    cfg = _launcher_cfg(PROD_LAUNCHER)
    assert cfg.strategy.find_unused_parameters is False
    timeout = hydra.utils.instantiate(cfg.strategy.timeout)
    assert isinstance(timeout, datetime.timedelta)
    assert timeout <= datetime.timedelta(minutes=10)


def test_t0_6_monitor_metric_is_a_generation_metric():
    """I-6: best-checkpoint selection must not use the aux band gap prediction loss."""
    cfg = _launcher_cfg(PROD_LAUNCHER)
    assert cfg.callbacks.model_checkpoint.monitor == "val_matbench/loss"
    assert cfg.callbacks.model_checkpoint.mode == "min"
    smoke_monitor = re.search(r'^MONITOR_METRIC="([^"]+)"', SMOKE_LAUNCHER.read_text(), re.M)
    assert smoke_monitor.group(1) == cfg.callbacks.model_checkpoint.monitor


def test_t0_6_no_exception_checkpoints():
    """I-2: a crash must not write a checkpoint that the next submission resumes from."""
    cfg = _launcher_cfg(PROD_LAUNCHER)
    assert cfg.callbacks.last_model_checkpoint.save_on_exception is False
    assert cfg.callbacks.model_checkpoint.save_on_exception is False


PRETRAINED = os.environ.get("PRETRAINED_CKPT_PATH")


@pytest.mark.skipif(not PRETRAINED, reason="set PRETRAINED_CKPT_PATH to check the real checkpoint")
def test_t0_7_pretrained_checkpoint_matches_architecture():
    """T0-7: the checkpoint exists and its width matches `model/architecture`."""
    assert Path(PRETRAINED).is_file(), PRETRAINED
    cfg = compose_cfg(train_overrides(PROD_LAUNCHER, PRETRAINED), tiny=False, cpu=False)
    state_dict = torch.load(PRETRAINED, map_location="cpu", weights_only=False, mmap=True)[
        "state_dict"
    ]
    q_proj = next(v for k, v in state_dict.items() if k.endswith("q_proj.weight"))
    assert q_proj.shape[-1] == cfg.model.architecture.hidden_size
