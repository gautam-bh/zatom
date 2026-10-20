"""T1-1: pretrained base weights actually land in the LoRA-wrapped model (I-5).

`train_fm.py` sets `model.strict_loading = False`, so if the `{name}.weight ->
{name}.linear.weight` remap in `on_load_checkpoint` misses a key, the base weights would
stay random with no error. `on_load_checkpoint` now refuses such a checkpoint; these tests
cover both directions on a tiny model, plus the real checkpoint when one is available.
"""

import os
from pathlib import Path

import pytest
import torch

from tests.predeploy.conftest import attach_stub_trainer, build_model, compose_cfg

LORA_OR_NEW = (".lora_A.", ".lora_B.", "band_gap_embedder.")


def _plain_state_dict(lora_model):
    """What a pretrained (non-LoRA, no band gap embedder) checkpoint of this model holds."""
    out = {}
    for key, val in lora_model.state_dict().items():
        if any(s in key for s in LORA_OR_NEW):
            continue
        out[key.replace(".linear.weight", ".weight").replace(".linear.bias", ".bias")] = (
            torch.randn_like(val) if val.is_floating_point() else val.clone()
        )
    return out


def _init_run(model, datamodule, tmp_path):
    ckpt = tmp_path / "pretrained.ckpt"
    ckpt.touch()
    return attach_stub_trainer(
        model, datamodule, ckpt_path=str(ckpt), pretrained_ckpt_path=str(ckpt)
    )


def test_pretrained_weights_load_into_lora_model(tiny_cfg, datamodule, tmp_path):
    model = _init_run(build_model(tiny_cfg), datamodule, tmp_path)
    state_dict = _plain_state_dict(model)
    checkpoint = {"state_dict": dict(state_dict), "loops": {}, "optimizer_states": []}

    model.on_load_checkpoint(checkpoint)
    result = model.load_state_dict(checkpoint["state_dict"], strict=False)

    assert not result.unexpected_keys, result.unexpected_keys[:8]
    assert all(any(s in k for s in LORA_OR_NEW) for k in result.missing_keys), [
        k for k in result.missing_keys if not any(s in k for s in LORA_OR_NEW)
    ][:8]
    loaded = model.state_dict()
    for key, val in state_dict.items():
        remapped = key
        if key not in loaded:
            remapped = key.replace(".weight", ".linear.weight").replace(".bias", ".linear.bias")
        torch.testing.assert_close(loaded[remapped], val, msg=key)


def test_missing_base_weight_is_an_error(tiny_cfg, datamodule, tmp_path):
    model = _init_run(build_model(tiny_cfg), datamodule, tmp_path)
    state_dict = _plain_state_dict(model)
    dropped = next(k for k in state_dict if k.endswith("q_proj.weight"))
    del state_dict[dropped]
    with pytest.raises(RuntimeError, match="q_proj"):
        model.on_load_checkpoint({"state_dict": state_dict, "loops": {}, "optimizer_states": []})


PRETRAINED = os.environ.get("PRETRAINED_CKPT_PATH")


@pytest.mark.slow
@pytest.mark.skipif(not PRETRAINED, reason="set PRETRAINED_CKPT_PATH to check the real checkpoint")
def test_real_pretrained_checkpoint_loads_completely(datamodule, tmp_path):
    """The real 160M checkpoint: every non-LoRA weight is present and copied over exactly."""
    cfg = compose_cfg(tiny=False)
    model = build_model(cfg)
    attach_stub_trainer(model, datamodule, ckpt_path=PRETRAINED, pretrained_ckpt_path=PRETRAINED)
    checkpoint = torch.load(PRETRAINED, map_location="cpu", weights_only=False)
    original = dict(checkpoint["state_dict"])

    model.on_load_checkpoint(checkpoint)
    result = model.load_state_dict(checkpoint["state_dict"], strict=False)
    assert not result.unexpected_keys, result.unexpected_keys[:8]
    assert all(any(s in k for s in LORA_OR_NEW) for k in result.missing_keys)

    loaded = model.state_dict()
    for key, val in original.items():
        if key.endswith((".weight", ".bias")) and key not in loaded:
            key_remapped = key.replace(".weight", ".linear.weight").replace(
                ".bias", ".linear.bias"
            )
        else:
            key_remapped = key
        if key_remapped in loaded:
            assert torch.equal(loaded[key_remapped], val), key
    assert Path(PRETRAINED).is_file()
