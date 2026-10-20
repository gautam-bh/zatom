"""T1-3: after the LoRA freeze, exactly the LoRA adapters and the band gap embedder train.

Builds the real 160M architecture (no data needed) so the counts match job 56923536's model
summary: 212 trainable tensors, ~1.9M trainable parameters.
"""

import types

import hydra

from tests.predeploy.conftest import compose_cfg
from zatom.callbacks.lora_finetuning import LoRABandGapFinetuning


def test_freeze_leaves_only_lora_and_embedder_trainable():
    cfg = compose_cfg(tiny=False)
    arch = hydra.utils.instantiate(cfg.model.architecture)
    LoRABandGapFinetuning().freeze_before_training(types.SimpleNamespace(model=arch))

    trainable = {n: p for n, p in arch.named_parameters() if p.requires_grad}
    unexpected = [
        n
        for n in trainable
        if not (".lora_A." in n or ".lora_B." in n or "band_gap_embedder." in n)
    ]
    assert not unexpected, f"non-LoRA params left trainable: {unexpected[:8]}"

    # 16 trunk layers x 4 proj x 2 (A, B) + 5 decoder blocks x 2 attn x 4 proj x 2 + 4 embedder
    assert len(trainable) == 212
    numel = sum(p.numel() for p in trainable.values())
    assert 1.8e6 < numel < 2.0e6, numel

    frozen_lora_base = [
        n for n, p in arch.named_parameters() if ".linear." in n and p.requires_grad
    ]
    assert not frozen_lora_base
