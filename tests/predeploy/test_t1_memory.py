"""T1-6: attention memory at the launcher's batch size stays under the A100 budget (I-3).

`Zatom.forward` pads every batch to `max_num_nodes` = the largest structure in the dataset
(444 atoms for matbench_mp_gap), not to the batch's own max. So attention memory per step
is fixed by the dataset and the launcher's batch size; there is no unlucky batch.

This is a lower bound (score + softmax tensors only). T2-1 measures the real peak on a GPU.
"""

from tests.predeploy.conftest import compose_cfg, prepare_batch
from tests.predeploy.launcher import PROD_LAUNCHER, train_overrides

GIB = 1024**3
BUDGET_GIB = 60  # of 80; leave room for weights, other activations, and allocator slack


def test_padding_is_dataset_max(tiny_model, matbench_batch, datamodule):
    import torch

    num_nodes = torch.load(f"{datamodule.hparams.datasets.matbench.root}/num_nodes.pt")
    with torch.no_grad():
        tiny_model.forward(prepare_batch(tiny_model, matbench_batch))
    # The batch holds 2 small structures, yet forward pads them to the dataset maximum
    assert int(torch.bincount(matbench_batch.batch).max()) < int(num_nodes.max())
    assert tiny_model.max_num_nodes == int(num_nodes.max())


def test_attention_memory_lower_bound_fits(datamodule):
    import torch

    cfg = compose_cfg(train_overrides(PROD_LAUNCHER), tiny=False, cpu=False)
    num_nodes = torch.load(f"{datamodule.hparams.datasets.matbench.root}/num_nodes.pt")
    pad_len = int(num_nodes.max())
    p99 = int(num_nodes.float().quantile(0.99))

    arch = cfg.model.architecture
    graphs = cfg.data.datamodule.batch_size.train * cfg.model.augmentations.multiplicity
    # trunk layers + 5 decoder blocks x (self + cross) attention, each keeping fp32 scores
    # and softmax for backward
    attn_layers = arch.num_layers + 5 * 2
    bytes_needed = graphs * arch.token_num_heads * pad_len**2 * 4 * attn_layers * 2
    print(
        f"matbench num_atoms p99={p99} max(pad)={pad_len}; B_eff={graphs}; "
        f"attention lower bound {bytes_needed / GIB:.1f} GiB"
    )
    assert bytes_needed / GIB < BUDGET_GIB
