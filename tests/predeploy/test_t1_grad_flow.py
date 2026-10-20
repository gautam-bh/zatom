"""T1-2: every trainable parameter is reachable from the loss, and the loss is generative.

Job 56923536 died in DDP with 80 of 212 trainable tensors receiving no gradient: the
cross-attention decoder blocks' LoRA adapters. Root cause (I-1): `task_name=finetune_fm`
replaced the objective with the auxiliary band gap *prediction* loss, which never touches
the decoder. DDP's unused-parameter check is the symptom; this test checks both the symptom
(`grad is None`) and the cause (the flow-matching terms must be in the objective).
"""

import pytest

from tests.predeploy.conftest import prepare_batch

GENERATIVE_LOSS_KEYS = (
    "atom_types_loss",
    "pos_loss",
    "frac_coords_loss",
    "lengths_scaled_loss",
    "angles_radians_loss",
)


@pytest.mark.parametrize("cfg_dropout_prob", [0.0, 1.0])
def test_every_trainable_param_gets_grad(tiny_model, matbench_batch, cfg_dropout_prob):
    tiny_model.model.band_gap_embedder.dropout_prob = cfg_dropout_prob
    tiny_model.train()
    loss_dict = tiny_model.forward(prepare_batch(tiny_model, matbench_batch))
    loss_dict["loss"].backward()

    trainable = {n: p for n, p in tiny_model.named_parameters() if p.requires_grad}
    assert trainable, "freeze left nothing trainable"
    # `grad is None`, not `grad == 0`: lora_A legitimately gets a zero grad at step 0
    # because lora_B is zero-initialized
    no_grad = sorted(n for n, p in trainable.items() if p.grad is None)
    assert (
        not no_grad
    ), f"{len(no_grad)} trainable tensors unreachable from the loss: {no_grad[:8]}"


def test_objective_includes_generative_losses(tiny_model, matbench_batch):
    tiny_model.train()
    loss_dict = tiny_model.forward(prepare_batch(tiny_model, matbench_batch))
    generative = sum(loss_dict[k] for k in GENERATIVE_LOSS_KEYS)
    assert generative.item() > 0
    # The objective must contain the flow-matching terms, not just the aux prediction loss
    assert loss_dict["loss"].item() >= generative.item() - 1e-5, (
        f"loss={loss_dict['loss'].item():.4f} < generative={generative.item():.4f}; "
        f"aux_global_property_loss={loss_dict['aux_global_property_loss'].item():.4f}"
    )
