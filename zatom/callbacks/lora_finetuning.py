from lightning import LightningModule
from lightning.pytorch.callbacks.finetuning import BaseFinetuning
from torch.optim import Optimizer

from zatom.models.architectures.transformer.band_gap_embedder import BandGapEmbedder
from zatom.models.architectures.transformer.lora import LoRALinear


class LoRABandGapFinetuning(BaseFinetuning):
    """Freeze the entire pretrained model; unfreeze only LoRA adapters and BandGapEmbedder.

    Usage: load a pretrained checkpoint, inject LoRA via `inject_lora`, then attach this
    callback. Only ~2M parameters (LoRA A/B matrices + embedder) will be trained.
    """

    def freeze_before_training(self, pl_module: LightningModule) -> None:
        """Freeze all parameters, then selectively unfreeze LoRA and band gap modules."""
        if not hasattr(pl_module, "model"):
            raise AttributeError("LightningModule must have a 'model' attribute.")

        # Freeze everything
        self.freeze(pl_module.model)

        # Unfreeze LoRA adapters
        lora_modules = [
            m for m in pl_module.model.modules() if isinstance(m, LoRALinear)
        ]
        for m in lora_modules:
            self.make_trainable(m.lora_A)
            self.make_trainable(m.lora_B)

        # Unfreeze BandGapEmbedder
        bg_modules = [
            m for m in pl_module.model.modules() if isinstance(m, BandGapEmbedder)
        ]
        for m in bg_modules:
            self.make_trainable(m)

    def finetune_function(
        self, pl_module: LightningModule, epoch: int, optimizer: Optimizer
    ) -> None:
        pass
