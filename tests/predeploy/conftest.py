"""Shared fixtures for the LoRA band gap pre-deploy suite.

See docs/plans/2026-10-19-lora-bandgap-predeploy-tests.md for what each test guards against.
Configs are composed exactly as `zatom/train_fm.py` composes them (`experiment=lora_bandgap`),
with a tiny-model override so CPU forward/backward passes take seconds.
"""

import copy
import os
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("PROJECT_ROOT", str(REPO_ROOT))

# `joint_datamodule` imports OMol25, which imports fairchem at module level. fairchem is a
# GitHub-only dependency; when it is not installed (e.g. a laptop without GitHub access),
# stub it out. OMol25 is never instantiated here (its proportion is 0.0).
try:
    import fairchem.core.datasets  # noqa: F401
except ImportError:
    for name in ("fairchem", "fairchem.core", "fairchem.core.datasets"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["fairchem.core.datasets"].AseDBDataset = object

# `jvp_flash_attention` needs triton (CUDA-only). The LoRA model runs with `jvp_attn: false`,
# so on a machine without triton (macOS) stub the import; on Perlmutter the real one loads.
try:
    import jvp_flash_attention.jvp_attention  # noqa: F401
except ImportError:
    for name in ("jvp_flash_attention", "jvp_flash_attention.jvp_attention"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["jvp_flash_attention.jvp_attention"].JVPAttn = None

import hydra  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig, open_dict  # noqa: E402

from zatom import register_custom_omegaconf_resolvers  # noqa: E402

CONFIG_DIR = str(REPO_ROOT / "configs")

# Small enough for CPU, same code path as the 160M model
TINY_MODEL = [
    "model.architecture.hidden_size=64",
    "model.architecture.num_layers=2",
    "model.architecture.aux_layer=1",
    "model.architecture.token_num_heads=4",
]

# Overrides every CPU test needs on top of `experiment=lora_bandgap`
CPU_BASE = [
    "experiment=lora_bandgap",
    "~logger",
    "model.compile=false",
    "data.datamodule.batch_size.train=2",
    "data.datamodule.batch_size.val=2",
    "data.datamodule.batch_size.test=2",
    "data.datamodule.num_workers.train=0",
    "data.datamodule.num_workers.val=0",
    "data.datamodule.num_workers.test=0",
    "model.augmentations.multiplicity=2",
    "model.sampling.save=false",
    "model.sampling.visualize=false",
    "trainer.accelerator=cpu",
    "trainer.devices=1",
]

_resolvers_registered = False


def compose_cfg(overrides=(), tiny: bool = True, cpu: bool = True) -> DictConfig:
    """Compose `train_fm.yaml` the way `train_fm.py` does, plus `overrides`.

    Args:
        overrides: Extra Hydra overrides, applied last.
        tiny: Shrink the model to `TINY_MODEL`.
        cpu: Prepend `CPU_BASE`. Pass False to compose from `overrides` alone (e.g. a launcher's).
    """
    global _resolvers_registered
    if not _resolvers_registered:
        register_custom_omegaconf_resolvers()
        _resolvers_registered = True
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(
            config_name="train_fm.yaml",
            return_hydra_config=True,
            overrides=[*(CPU_BASE if cpu else []), *(TINY_MODEL if tiny else []), *overrides],
        )
    HydraConfig.instance().set_config(cfg)
    return cfg


def build_model(cfg: DictConfig):
    """Instantiate the LightningModule exactly as `train_fm.py` does."""
    cfg = copy.deepcopy(cfg)
    with open_dict(cfg):
        for dataset in cfg.model.datasets:
            cfg.model.datasets[dataset].pop("_target_", None)
    return hydra.utils.instantiate(cfg.model)


def attach_stub_trainer(model, datamodule=None, **kwargs):
    """Give `model` the few trainer attributes its hooks read, without a real Trainer."""
    attrs = dict(
        datamodule=datamodule,
        training=True,
        sanity_checking=False,
        overfit_batches=0,
        global_step=0,
        ckpt_path=None,
        pretrained_ckpt_path=None,
    )
    attrs.update(kwargs)
    model._trainer = types.SimpleNamespace(**attrs)
    return model


def prepare_batch(model, batch):
    """Set the periodicity masks `training_step` normally sets before `forward`."""
    sample_is_periodic = torch.isin(batch.dataset_idx, model.periodic_datasets)
    batch.sample_is_periodic = sample_is_periodic
    batch.node_is_periodic = sample_is_periodic[batch.batch]
    return batch


def ensure_matbench_metadata(datamodule) -> None:
    """Write `num_nodes.pt` / `spacegroups_bincount.pt` if this checkout lacks them.

    The model reads these at construction time to size padding and sample structure sizes.
    They live untracked on Perlmutter; this recreates them the way the (commented-out)
    block in `JointDataModule.setup` does.
    """
    root = Path(datamodule.hparams.datasets.matbench.root)
    if (root / "num_nodes.pt").exists() and (root / "spacegroups_bincount.pt").exists():
        return
    datasets = [
        datamodule.matbench_train_dataset,
        datamodule.matbench_val_dataset,
        datamodule.matbench_test_dataset,
    ]
    # Per-structure atom counts from the collated storage's slices (no per-item `get`)
    idx = [torch.as_tensor(list(ds.indices()), dtype=torch.long) for ds in datasets]
    num_nodes = torch.cat([ds.slices["atom_types"].diff()[i] for ds, i in zip(datasets, idx)])
    spacegroups = torch.cat([ds._data.spacegroup.view(-1)[i] for ds, i in zip(datasets, idx)])
    torch.save(num_nodes, root / "num_nodes.pt")
    torch.save(torch.bincount(spacegroups), root / "spacegroups_bincount.pt")


@pytest.fixture(scope="session")
def tiny_cfg() -> DictConfig:
    return compose_cfg()


@pytest.fixture(scope="session")
def datamodule(tiny_cfg):
    """Real matbench datamodule. First run downloads + processes matbench_mp_gap (slow)."""
    dm = hydra.utils.instantiate(tiny_cfg.data.datamodule, _recursive_=False)
    dm.setup("fit")
    ensure_matbench_metadata(dm)
    return dm


@pytest.fixture()
def tiny_model(tiny_cfg, datamodule):
    """Tiny LoRA band gap model with the fine-tuning freeze applied, attached to a stub trainer."""
    from zatom.callbacks.lora_finetuning import LoRABandGapFinetuning

    torch.manual_seed(0)
    model = build_model(tiny_cfg)
    LoRABandGapFinetuning().freeze_before_training(model)
    attach_stub_trainer(model, datamodule)
    return model


@pytest.fixture()
def matbench_batch(datamodule):
    return next(iter(datamodule.train_dataloader()))
