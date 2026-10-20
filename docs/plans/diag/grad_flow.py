"""CPU check: which trainable params get no grad in one LoRA-bandgap forward/backward.

Run from the repo root (job 56923536 diagnosis, see ../2026-10-19-lora-bandgap-predeploy-PROGRESS.md):
    shifter --image=registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1 --env=PROJECT_ROOT=$PWD \
        python docs/plans/diag/grad_flow.py
"""

import collections
import os
import sys
import types

sys.path.insert(0, os.getcwd())
import hydra
import torch
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from zatom import register_custom_omegaconf_resolvers

register_custom_omegaconf_resolvers()
with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base="1.3"):
    cfg = compose(
        config_name="train_fm.yaml",
        return_hydra_config=True,
        overrides=[
            "experiment=lora_bandgap",
            "~logger",
            "data.datamodule.batch_size.train=2",
            "data.datamodule.num_workers.train=0",
            "model.augmentations.multiplicity=1",
        ],
    )
HydraConfig.instance().set_config(cfg)

dm = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)
dm.setup("fit")
with open_dict(cfg):
    for d in cfg.model.datasets:
        cfg.model.datasets[d].pop("_target_")
model = hydra.utils.instantiate(cfg.model)

from zatom.callbacks.lora_finetuning import LoRABandGapFinetuning

LoRABandGapFinetuning().freeze_before_training(model)
params = dict(model.named_parameters())
names = [n for n, p in params.items() if p.requires_grad]
print("trainable tensors:", len(names), flush=True)

# `forward` reads these off the trainer; `training_step` normally sets the periodic masks
model._trainer = types.SimpleNamespace(
    datamodule=dm, training=True, sanity_checking=False, overfit_batches=0, global_step=0
)
model.train()
batch = next(iter(dm.train_dataloader()))
sp = torch.isin(batch.dataset_idx, model.periodic_datasets)
batch.sample_is_periodic = sp
batch.node_is_periodic = sp[batch.batch]

loss_dict = model.forward(batch)
print("loss:", float(loss_dict["loss"]), flush=True)
print("loss terms:", {k: float(v.sum()) for k, v in loss_dict.items() if "loss" in k}, flush=True)
loss_dict["loss"].backward()

none = [(i, n) for i, n in enumerate(names) if params[n].grad is None]
zero = [
    (i, n)
    for i, n in enumerate(names)
    if params[n].grad is not None and params[n].grad.abs().sum() == 0
]
print("NO-GRAD params:", len(none))
for i, n in none[:6] + none[-6:]:
    print("  ", i, n)
print("NO-GRAD groups:", collections.Counter(n.rsplit(".", 4)[0] for _, n in none))
print("ZERO-GRAD params (lora_A is expected here at init since lora_B=0):", len(zero))
print("ZERO-GRAD groups:", collections.Counter(n.rsplit(".", 4)[0] for _, n in zero))
