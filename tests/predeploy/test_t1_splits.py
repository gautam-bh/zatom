"""I-13: matbench validation must be held out from training.

Matbench ships only train/test splits, so `JointDataModule` used to build "val" from the
train split itself; at `proportion: 1.0` val was the whole training set and
`val_matbench/loss` (the checkpoint monitor) could not see overfitting. With
`val_holdout_fraction > 0` a fixed, seeded slice of the train split becomes val and is
removed from train.
"""

import torch

from tests.predeploy.conftest import compose_cfg
from zatom.data.joint_datamodule import holdout_split


def test_holdout_split_is_deterministic_and_disjoint():
    train_a, val_a = holdout_split(1000, 0.05, seed=0)
    train_b, val_b = holdout_split(1000, 0.05, seed=0)
    assert torch.equal(train_a, train_b) and torch.equal(val_a, val_b)
    assert len(val_a) == 50 and len(train_a) == 950
    assert not set(train_a.tolist()) & set(val_a.tolist())
    assert sorted(train_a.tolist() + val_a.tolist()) == list(range(1000))


def test_lora_bandgap_holds_out_validation():
    cfg = compose_cfg()
    assert cfg.data.datamodule.datasets.matbench.val_holdout_fraction > 0


def test_matbench_val_is_disjoint_from_train(datamodule):
    train = datamodule.matbench_train_dataset
    val = datamodule.matbench_val_dataset
    assert len(val) > 0
    # Both are index views of the same processed train split
    train_idx, val_idx = set(train.indices()), set(val.indices())
    assert not train_idx & val_idx, f"{len(train_idx & val_idx)} val structures are also trained on"
    assert train.split == val.split == "train"
    assert datamodule.matbench_test_dataset.split == "test"
