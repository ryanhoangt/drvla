import unittest

import torch

from experiments.exp1.common import ROOT, output_path
from experiments.exp1.patches import patch_hidden


class PatchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.hidden = torch.randn(1, 5, 8)
        self.donor = torch.randn_like(self.hidden)
        self.mask = torch.tensor([[True, True, True, False, False]])

    def test_dense_exact_and_padding_untouched(self):
        y, _ = patch_hidden(self.hidden, self.donor, self.mask, "dense")
        torch.testing.assert_close(y[:, :3], self.donor[:, :3], rtol=0, atol=0)
        torch.testing.assert_close(y[:, 3:], self.hidden[:, 3:], rtol=0, atol=0)

    def test_dense_and_pooled_preserve_same_mean_shift(self):
        dense, _ = patch_hidden(self.hidden, self.donor, self.mask, "dense")
        pooled, _ = patch_hidden(self.hidden, self.donor, self.mask, "pooled")
        torch.testing.assert_close(dense.mean(1), pooled.mean(1))
        delta = pooled - self.hidden
        torch.testing.assert_close(delta[:, 0], delta[:, 2])
        torch.testing.assert_close(delta[:, 3:], torch.zeros_like(delta[:, 3:]), rtol=0, atol=0)

    def test_norm_matching(self):
        _, d = patch_hidden(self.hidden, self.donor, self.mask, "pooled_norm")
        self.assertAlmostEqual(d["realized_vs_dense_norm"], 1.0, places=6)

    def test_self_and_zero_edit_bit_exact_bfloat16(self):
        x = self.hidden.to(torch.bfloat16)
        for donor, mode in [(self.donor.to(torch.bfloat16), "self"), (x, "dense"), (x, "pooled_norm")]:
            y, _ = patch_hidden(x, donor, self.mask, mode)
            self.assertIs(y, x)

    def test_output_isolation(self):
        for name in ("workspace/rq1/a.json", "workspace/rq2/a.json", "workspace/exp1"):
            with self.assertRaises(ValueError):
                output_path(ROOT / name)


if __name__ == "__main__":
    unittest.main()
