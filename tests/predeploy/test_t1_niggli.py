"""Matbench structures must follow the lattice convention the pretrained model learned.

The model was pretrained on Niggli-reduced MP20 cells (99.9% have a <= b <= c); raw
matbench cells break that in ~25% of structures, and those carried ~74% of the weighted
`lengths_scaled_loss` (docs/plans/diag/lengths_loss*). `datasets.matbench.niggli` reduces
them the same way MP20 does.
"""

import torch
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure

from tests.predeploy.conftest import compose_cfg
from zatom.data.components.preprocessing_utils import canonicalize_crystal


def test_niggli_canonicalization_keeps_the_crystal():
    # A skewed, unsorted supercell-like cell: b < c < a and a non-reduced angle
    lattice = Lattice.from_parameters(7.0, 2.5, 4.0, 90, 90, 120)
    raw = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    reduced = canonicalize_crystal(raw, niggli=True)
    a, b, c = reduced.lattice.abc
    assert a <= b <= c
    assert StructureMatcher().fit(raw, reduced)
    assert len(reduced) == len(raw)


def test_lora_bandgap_uses_niggli_reduced_matbench():
    cfg = compose_cfg()
    assert cfg.data.datamodule.datasets.matbench.niggli is True


def test_loaded_matbench_cells_follow_the_pretraining_convention(datamodule):
    for ds in (datamodule.matbench_train_dataset, datamodule.matbench_test_dataset):
        lengths = ds._data.lengths_scaled.reshape(-1, 3)
        is_sorted = (lengths[:, 0] <= lengths[:, 1] + 1e-4) & (lengths[:, 1] <= lengths[:, 2] + 1e-4)
        assert is_sorted.float().mean() > 0.99, f"{ds.split}: {is_sorted.float().mean():.1%} sorted"
        assert torch.isfinite(lengths).all()
