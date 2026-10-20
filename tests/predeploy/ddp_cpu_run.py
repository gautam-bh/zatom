"""2-rank DDP fine-tuning run on CPU (gloo), driven by test_t1_trainer.py (T1-8).

DDP's unused-parameter check fires on CPU exactly as it did on the A100s in job 56923536,
so this is the cheapest reproduction of that crash. Lightning re-launches this script once
per extra rank, which is why it is a script rather than a test function.

Environment:
    DDP_OUT_DIR: checkpoint / output directory (required).
    DDP_TIMEOUT_S: override the strategy's collective timeout (seconds).
    DDP_CRASH_RANK0: if "1", rank 0 raises mid-epoch while rank 1 keeps training, to check
        the job dies within the timeout instead of hanging (I-4).
"""

import datetime
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import hydra  # noqa: E402
import torch  # noqa: E402
from lightning import Callback, Trainer  # noqa: E402
from lightning.pytorch.strategies import DDPStrategy  # noqa: E402

from tests.predeploy.conftest import build_model, compose_cfg  # noqa: E402
from zatom.utils import instantiate_callbacks  # noqa: E402


class CrashRank0(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_rank == 0 and batch_idx == 1:
            raise RuntimeError("injected rank-0 failure")


def main():
    out_dir = os.environ["DDP_OUT_DIR"]
    torch.set_num_threads(1)
    cfg = compose_cfg(
        [
            f"callbacks.model_checkpoint.dirpath={out_dir}",
            f"callbacks.last_model_checkpoint.dirpath={out_dir}",
            f"model.sampling.save_dir={out_dir}",
            "model.sampling.num_samples=2",
            "model.sampling.batch_size=2",
            "model.sampling.steps=2",
            "~callbacks.epoch_progress",
            "~callbacks.learning_rate_monitor",  # needs a logger; checkpointing doesn't
        ]
    )
    datamodule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)
    model = build_model(cfg)
    callbacks = instantiate_callbacks(cfg.callbacks, using_logger=True)
    if os.environ.get("DDP_CRASH_RANK0") == "1":
        callbacks.append(CrashRank0())

    timeout = hydra.utils.instantiate(cfg.strategy.timeout)
    if os.environ.get("DDP_TIMEOUT_S"):
        timeout = datetime.timedelta(seconds=int(os.environ["DDP_TIMEOUT_S"]))
    strategy = DDPStrategy(
        process_group_backend="gloo",
        find_unused_parameters=cfg.strategy.find_unused_parameters,
        static_graph=cfg.strategy.static_graph,
        timeout=timeout,
    )
    trainer = Trainer(
        accelerator="cpu",
        devices=2,
        strategy=strategy,
        max_epochs=1,
        limit_train_batches=4,
        accumulate_grad_batches=2,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        callbacks=callbacks,
        default_root_dir=out_dir,
        enable_progress_bar=False,
    )
    model.strict_loading = False
    trainer.fit(model=model, datamodule=datamodule)
    if trainer.global_rank == 0:
        print("DDP_RUN_OK", sorted(trainer.callback_metrics))


if __name__ == "__main__":
    main()
