from lightning import Callback, LightningModule, Trainer

from zatom.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class EpochProgressLogger(Callback):
    """Log epoch-end metrics to the Python logger (and thus the log file).

    Lets you follow training progress via `tail -f logs/.../finetune_fm.log`
    on a remote machine or when running non-interactively (SLURM, background).
    """

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        metrics = trainer.callback_metrics
        train_metrics = {k: f"{v:.4f}" for k, v in metrics.items() if "train" in k}
        if train_metrics:
            log.info(
                f"Epoch {trainer.current_epoch} | step {trainer.global_step} | "
                + " | ".join(f"{k}: {v}" for k, v in train_metrics.items())
            )

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        val_metrics = {k: f"{v:.4f}" for k, v in metrics.items() if "val" in k}
        if val_metrics:
            log.info(
                f"Epoch {trainer.current_epoch} | step {trainer.global_step} | "
                + " | ".join(f"{k}: {v}" for k, v in val_metrics.items())
            )
