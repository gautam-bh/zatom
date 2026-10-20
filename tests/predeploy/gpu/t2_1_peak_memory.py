"""T2-1: peak GPU memory of one LoRA band gap training step at production settings.

Runs a single `training_step` + backward of the full 160M model on one GPU with the
launcher's batch size (8) and multiplicity (8), and fails if peak allocated memory exceeds
the budget. Every batch is padded to the dataset's largest structure, so one step is the
run's worst case. Unlike job 56923536's 47.8 GiB, this includes the backward pass through
the 5 decoder blocks that the generative objective (I-1 fix) now trains.

Usage (Perlmutter, from the repo root, inside an interactive GPU allocation):
    salloc -q interactive -C "gpu&hbm80g" -N 1 -G 1 -t 30 -A m5008
    shifter --image=registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1 --env=PROJECT_ROOT=$PWD \
        python tests/predeploy/gpu/t2_1_peak_memory.py [--budget-gib 70]
"""

import argparse
import sys

import torch

sys.path.insert(0, ".")
from tests.predeploy.conftest import (  # noqa: E402
    attach_stub_trainer,
    build_model,
    compose_cfg,
)
from tests.predeploy.launcher import PROD_LAUNCHER, train_overrides  # noqa: E402

GIB = 1024**3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-gib", type=float, default=70.0)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "T2-1 needs a GPU"

    import hydra

    from zatom.callbacks.lora_finetuning import LoRABandGapFinetuning

    overrides = [
        o
        for o in train_overrides(PROD_LAUNCHER)
        if o.split("=")[0]
        in (
            "experiment",
            "data.datamodule.batch_size.train",
            "trainer.accumulate_grad_batches",
            "model/architecture",
        )
    ]
    cfg = compose_cfg(
        [
            *overrides,
            "~logger",
            "data.datamodule.num_workers.train=0",
            "model.compile=false",
        ],
        tiny=False,
        cpu=False,
    )
    print(
        f"batch_size={cfg.data.datamodule.batch_size.train} "
        f"multiplicity={cfg.model.augmentations.multiplicity} "
        f"finetune_loss={cfg.model.finetune_loss}"
    )

    datamodule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)
    datamodule.setup("fit")
    model = build_model(cfg)
    LoRABandGapFinetuning().freeze_before_training(model)
    attach_stub_trainer(model, datamodule)
    model.log = lambda *a, **k: None  # no Trainer to log to
    model = model.cuda().train()

    batch = next(iter(datamodule.train_dataloader())).cuda()
    torch.cuda.reset_peak_memory_stats()
    loss = model.training_step(batch, 0)
    loss.backward()
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated() / GIB
    reserved = torch.cuda.max_memory_reserved() / GIB
    print(f"pad length={model.max_num_nodes}  loss={loss.item():.4f}")
    print(
        f"peak allocated={peak:.1f} GiB  peak reserved={reserved:.1f} GiB  "
        f"budget={args.budget_gib} GiB"
    )
    if peak > args.budget_gib:
        print("T2-1 FAIL: over budget; halve batch_size and double accumulate_grad_batches")
        return 1
    print("T2-1 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
