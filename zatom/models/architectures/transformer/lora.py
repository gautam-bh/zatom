import math
from typing import Optional, Set

import torch.nn as nn
from torch import Tensor


class LoRALinear(nn.Module):
    """Frozen nn.Linear augmented with trainable low-rank adapters.

    Forward: out = W(x) + B(A(x)) * (alpha / rank)
    W is frozen; only A and B are trained.

    Args:
        linear: The pre-trained linear layer to wrap (its weights are frozen in-place).
        rank: LoRA rank r.
        alpha: Scaling factor; effective scale = alpha / rank (default alpha=rank → scale=1).
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.rank = rank
        self.scale = alpha / rank

        in_features = linear.in_features
        out_features = linear.out_features

        self.linear = linear
        for param in self.linear.parameters():
            param.requires_grad = False

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self._lora_enabled = True

    def enable_lora(self) -> None:
        self._lora_enabled = True

    def disable_lora(self) -> None:
        self._lora_enabled = False

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(x)
        if self._lora_enabled:
            out = out + self.lora_B(self.lora_A(x)) * self.scale
        return out


def inject_lora(
    module: nn.Module,
    rank: int,
    alpha: float,
    target_modules: Optional[Set[str]] = None,
) -> None:
    """Replace matching nn.Linear layers with LoRALinear wrappers, in-place.

    Args:
        module: Root module to walk.
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        target_modules: Attribute names to wrap (e.g. {"q_proj", "k_proj"}).
                        If None, wraps every nn.Linear.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and (
            target_modules is None or name in target_modules
        ):
            setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
        else:
            inject_lora(child, rank=rank, alpha=alpha, target_modules=target_modules)
