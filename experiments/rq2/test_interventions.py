"""CPU checks; no policy weights, simulator, or old artifacts are modified."""

import unittest

import torch

from drvla.sae import SAEConfig, TopKSAE
from experiments.rq2.common import ROOT, output_path
from experiments.rq2.interventions import apply_pooled_delta, feature_delta, pooled_scale


class InterventionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.sae = TopKSAE(SAEConfig(input_dim=8, num_features=16, k=4)).eval()
        self.hidden = torch.randn(1, 5, 8)
        self.x = self.hidden.mean(1)
        self.f = self.sae.encode(self.x)
        self.pair = {
            "old": [0, 1], "new": [2, 3], "new_mean": [0.2] * 16,
            "random_old": [4, 5], "random_new": [4, 5],
            "general_old": [6, 7], "general_new": [6, 7],
            "random_pool": list(range(4, 16)), "general_pool": list(range(4, 16)),
        }

    def test_identity_bit_exact_even_bfloat16(self):
        for dtype in (torch.float32, torch.bfloat16):
            hidden = self.hidden.to(dtype)
            edited, _ = apply_pooled_delta(hidden, hidden.mean(1).float(), self.sae, torch.zeros_like(self.f))
            self.assertIs(edited, hidden)

    def test_preserves_recipient_error_and_token_differences(self):
        diff = torch.randn_like(self.f) * 0.02
        edited, delta = apply_pooled_delta(self.hidden, self.x, self.sae, diff)
        scale = pooled_scale(self.sae, self.x)
        mean = (self.x - self.sae.pre_bias).mean(-1, keepdim=True)
        reconstruction = self.sae.decoder(self.f) * scale + mean + self.sae.pre_bias
        error = self.x - reconstruction
        expected = self.sae.decoder(self.f + diff) * scale + mean + self.sae.pre_bias + error
        torch.testing.assert_close(edited.mean(1), expected)
        torch.testing.assert_close(edited[:, 0] - edited[:, 1], self.hidden[:, 0] - self.hidden[:, 1])
        torch.testing.assert_close(delta, expected - self.x)

    def test_suppression_and_patch_nonnegative(self):
        f = torch.rand_like(self.f)
        for mode in ("suppress_old", "random_suppress", "general_suppress", "patch_old"):
            difference, ids = feature_delta(self.sae, f, self.pair, [0.2] * 16, mode, 1.0, torch.rand_like(f))
            self.assertGreaterEqual(float((f + difference).min()), -1e-6)
            self.assertEqual(len(ids), 2)
            self.assertEqual(set(torch.nonzero(difference[0]).flatten().tolist()), set(ids))

    def test_boost_controls_match_decoded_norm(self):
        for mode in ("random_boost", "general_boost"):
            target, _ = feature_delta(self.sae, self.f, self.pair, [0.2] * 16, "boost_new")
            control, _ = feature_delta(self.sae, self.f, self.pair, [0.2] * 16, mode)
            torch.testing.assert_close(self.sae.decoder(target).norm(), self.sae.decoder(control).norm())

    def test_output_guard(self):
        with self.assertRaises(ValueError):
            output_path(ROOT / "workspace/rq1/new.json")
        with self.assertRaises(ValueError):
            output_path(ROOT / "workspace/rq2")
        self.assertEqual(output_path(ROOT / "workspace/rq2/a.json"), ROOT / "workspace/rq2/a.json")


if __name__ == "__main__":
    unittest.main()
