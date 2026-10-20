"""T1-9: generation metrics have the same keys whether samples are valid or not (I-7).

Each rank logs its own `get_metrics` keys with `sync_dist=True`. If one rank's key set
differed (e.g. a metric skipped when that rank produced 0 valid samples), the collectives
would not match and NCCL would hang, as job 56395971's ranks did for 30 minutes.
"""

import numpy as np

from zatom.eval.crystal_generation import CrystalGenerationEvaluator


def _rock_salt():
    return {
        "atom_types": np.array([11, 11, 11, 11, 17, 17, 17, 17]),
        "frac_coords": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
                [0.5, 0.5, 0.5],
            ]
        ),
        "lengths": np.array([5.64, 5.64, 5.64]),
        "angles": np.array([90.0, 90.0, 90.0]),
    }


def _collapsed():
    # Every atom on the same site: structurally invalid
    return {
        "atom_types": np.array([1, 1, 1]),
        "frac_coords": np.zeros((3, 3)),
        "lengths": np.array([1.0, 1.0, 1.0]),
        "angles": np.array([90.0, 90.0, 90.0]),
    }


def _metric_keys(make_sample, n=3):
    evaluator = CrystalGenerationEvaluator(dataset_cif_list=None)
    for i in range(n):
        sample = make_sample()
        sample["pos"] = sample["frac_coords"] * sample["lengths"]
        sample["sample_idx"] = i
        evaluator.append_pred_array(sample)
    metrics = evaluator.get_metrics(save=False, n_jobs=1)
    return metrics


def test_metric_keys_match_for_valid_and_invalid_samples():
    valid = _metric_keys(_rock_salt)
    invalid = _metric_keys(_collapsed)
    assert valid["valid_rate"].float().mean() == 1.0
    assert invalid["valid_rate"].float().mean() == 0.0
    assert sorted(valid) == sorted(invalid)
