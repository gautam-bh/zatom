"""Tests for CFG inference with band gap conditioning."""

import unittest

import torch

from zatom.models.architectures.transformer.band_gap_embedder import BandGapEmbedder
from zatom.models.architectures.transformer.lora import LoRALinear, inject_lora


class TestBandGapEmbedderCFG(unittest.TestCase):
    """Test BandGapEmbedder CFG behaviour."""

    def setUp(self):
        self.hidden = 64
        self.embedder = BandGapEmbedder(hidden_size=self.hidden, dropout_prob=0.1)

    def test_force_uncond_returns_zeros(self):
        bg = torch.tensor([1.5, 2.0, 0.0])
        out = self.embedder(bg, train=False, force_uncond=True)
        self.assertTrue(torch.all(out == 0), "force_uncond=True must return all zeros")

    def test_nan_maps_to_zero_embedding(self):
        nan_bg = torch.full((4,), float("nan"))
        out = self.embedder(nan_bg, train=False)
        self.assertTrue(torch.all(out == 0), "NaN band gap must produce zero embedding")

    def test_valid_bg_produces_nonzero_embedding(self):
        bg = torch.tensor([1.5, 3.0])
        out = self.embedder(bg, train=False)
        self.assertFalse(torch.all(out == 0), "Valid band gap must produce non-zero embedding")

    def test_cond_and_uncond_embeddings_differ(self):
        """CFG requires cond ≠ uncond for non-zero band gap."""
        bg = torch.tensor([2.5])
        cond = self.embedder(bg, train=False, force_uncond=False)
        uncond = self.embedder(bg, train=False, force_uncond=True)
        self.assertFalse(
            torch.allclose(cond, uncond),
            "Conditioned and unconditional embeddings must differ for CFG to work",
        )

    def test_cfg_guidance_formula(self):
        """v_guided = v_uncond + scale * (v_cond - v_uncond) differs from v_cond and v_uncond."""
        bg = torch.tensor([1.5])
        v_cond = self.embedder(bg, train=False)
        v_uncond = torch.zeros_like(v_cond)  # unconditional = zeros

        for scale in [1.5, 3.0, 5.0]:
            v_guided = v_uncond + scale * (v_cond - v_uncond)
            self.assertFalse(
                torch.allclose(v_guided, v_cond),
                f"Guided output must differ from conditioned at scale={scale}",
            )
            self.assertFalse(
                torch.allclose(v_guided, v_uncond),
                f"Guided output must differ from unconditional at scale={scale}",
            )

    def test_use_cfg_triggered_by_dropout_prob(self):
        """use_cfg should be True when band_gap_embedder.dropout_prob > 0."""
        embedder_with_dropout = BandGapEmbedder(hidden_size=32, dropout_prob=0.1)
        embedder_no_dropout = BandGapEmbedder(hidden_size=32, dropout_prob=0.0)

        use_cfg_on = getattr(embedder_with_dropout, "dropout_prob", 0) > 0
        use_cfg_off = getattr(embedder_no_dropout, "dropout_prob", 0) > 0

        self.assertTrue(use_cfg_on, "dropout_prob=0.1 should enable CFG")
        self.assertFalse(use_cfg_off, "dropout_prob=0.0 should not enable CFG")


class TestLoRALinear(unittest.TestCase):
    """Sanity checks for LoRALinear used during CFG fine-tuning."""

    def test_base_weights_frozen(self):
        linear = torch.nn.Linear(32, 32)
        lora = LoRALinear(linear, rank=4, alpha=4.0)
        for p in lora.linear.parameters():
            self.assertFalse(p.requires_grad, "Base linear weights must be frozen")

    def test_adapter_weights_trainable(self):
        lora = LoRALinear(torch.nn.Linear(32, 32), rank=4, alpha=4.0)
        self.assertTrue(lora.lora_A.weight.requires_grad)
        self.assertTrue(lora.lora_B.weight.requires_grad)

    def test_lora_b_init_zero(self):
        lora = LoRALinear(torch.nn.Linear(32, 32), rank=4, alpha=4.0)
        self.assertTrue(
            torch.all(lora.lora_B.weight == 0),
            "lora_B must be zero-init so LoRA starts as identity",
        )

    def test_output_equals_base_at_init(self):
        """At init lora_B=0, so output should equal base linear."""
        torch.manual_seed(0)
        linear = torch.nn.Linear(16, 16, bias=False)
        lora = LoRALinear(linear, rank=4, alpha=4.0)
        x = torch.randn(8, 16)
        with torch.no_grad():
            self.assertTrue(
                torch.allclose(lora(x), linear(x)),
                "LoRA output must equal base output at init (lora_B=0)",
            )


if __name__ == "__main__":
    unittest.main()
