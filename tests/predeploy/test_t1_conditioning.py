"""T1-4 / T1-5: band gap conditioning is wired in, in eV, and CFG batches are laid out right.

Guards f31258c (condition on raw eV, not z-scored targets) and 6a9f57c (CFG doubles the
batch as [cond | uncond] blocks, not interleaved).
"""

import torch

from tests.predeploy.conftest import prepare_batch


def _capture_embedder_inputs(model):
    seen = []

    def hook(_module, args, kwargs):
        band_gap = args[0] if args else kwargs["band_gap"]
        seen.append(band_gap.detach().clone())

    handle = model.model.band_gap_embedder.register_forward_pre_hook(hook, with_kwargs=True)
    return seen, handle


def test_band_gap_reaches_embedder_in_ev(tiny_model, matbench_batch, datamodule):
    tiny_model.eval()
    seen, handle = _capture_embedder_inputs(tiny_model)
    try:
        with torch.no_grad():
            tiny_model.forward(prepare_batch(tiny_model, matbench_batch))
    finally:
        handle.remove()
    assert seen, "band gap embedder was never called during forward"
    band_gap = seen[0]

    # The dataset hands out z-scored targets; forward must undo that before conditioning
    from zatom.data.components.matbench_dataset import Matbench

    idx = Matbench.avail_tasks["matbench_mp_gap"]
    ds = datamodule.matbench_train_dataset
    expected = matbench_batch.y[:, idx] * ds.scale[0, idx] + ds.shift[0, idx]
    torch.testing.assert_close(band_gap, expected.float())
    assert band_gap.min() >= -1e-4 and band_gap.max() < 20.0, band_gap  # eV, never negative


def test_different_band_gaps_change_the_prediction(tiny_model, matbench_batch):
    """Same noised input, 0 eV vs 4 eV: the output must differ (conditioning is not a no-op)."""
    tiny_model.eval()
    batch = prepare_batch(tiny_model, matbench_batch)

    outputs = {}
    for gap in (0.0, 4.0, float("nan")):
        captured = {}

        def hook(_module, _args, out):
            captured["out"] = out.detach().clone()

        handle = tiny_model.model.band_gap_embedder.register_forward_hook(hook)
        torch.manual_seed(0)
        original = tiny_model.model.band_gap_embedder.forward

        def fixed_gap(band_gap, *a, _gap=gap, **kw):
            return original(torch.full_like(band_gap, _gap), *a, **kw)

        tiny_model.model.band_gap_embedder.forward = fixed_gap
        try:
            with torch.no_grad():
                loss = tiny_model.forward(batch)["loss"]
        finally:
            handle.remove()
            tiny_model.model.band_gap_embedder.forward = original
        outputs[gap] = (loss.item(), captured["out"])

    assert outputs[0.0][0] != outputs[4.0][0], "band gap does not affect the model output"
    # NaN means "unconditional": zero embedding, identical to force_uncond
    assert torch.all(outputs[float("nan")][1] == 0)


def test_cfg_sampling_doubles_batch_as_cond_then_uncond(tiny_model):
    tiny_model.eval()
    seen, handle = _capture_embedder_inputs(tiny_model)
    target = torch.tensor([1.0, 2.5, 4.0])
    try:
        torch.manual_seed(0)
        with torch.no_grad():
            out_a, _ = tiny_model.sample_and_decode(
                num_nodes_bincount=torch.tensor([0, 0, 0, 1, 1]),
                spacegroups_bincount=None,
                batch_size=3,
                cfg_scale=2.0,
                dataset_index=tiny_model.dataset_to_index["matbench"],
                dataset_idx=tiny_model.dataset_to_idx["matbench"],
                steps=2,
                band_gap=target,
            )
    finally:
        handle.remove()
    assert seen, "embedder not called during sampling"
    for band_gap in seen:
        assert band_gap.shape == (6,)
        torch.testing.assert_close(band_gap[:3], target)
        assert torch.isnan(band_gap[3:]).all()

    torch.manual_seed(0)
    with torch.no_grad():
        out_b, _ = tiny_model.sample_and_decode(
            num_nodes_bincount=torch.tensor([0, 0, 0, 1, 1]),
            spacegroups_bincount=None,
            batch_size=3,
            cfg_scale=2.0,
            dataset_index=tiny_model.dataset_to_index["matbench"],
            dataset_idx=tiny_model.dataset_to_idx["matbench"],
            steps=2,
            band_gap=target,
        )
    for key in out_a:
        torch.testing.assert_close(out_a[key], out_b[key])
