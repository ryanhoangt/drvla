"""Production-width QR checks, including rank-deficient decoder directions."""

import unittest

import torch

from experiments.exp2.arm1.selection import orthogonal


class OrthogonalTests(unittest.TestCase):
    def test_correlated_production_width_dictionary(self):
        generator = torch.Generator().manual_seed(60966)
        decoder = torch.randn(2048, 76, generator=generator)
        decoder[:, 1:] += .8*decoder[:, :1]
        decoder /= decoder.norm(dim=0)
        basis, _ = orthogonal(decoder, list(range(76)))
        # Selected decoder directions must be removed by the resulting span.
        residual = decoder.double()-basis.double()@(basis.double().T@decoder.double())
        self.assertLess(float(residual.norm()/decoder.double().norm()), 2e-6)

    def test_duplicate_decoder_columns_fail_rank_gate(self):
        decoder = torch.randn(32, 8)
        decoder[:, 7] = decoder[:, 0]
        with self.assertRaisesRegex(ValueError, "rank"):
            orthogonal(decoder, list(range(8)))

    @unittest.skipUnless(torch.cuda.is_available(), "GPU precision regression runs in Slurm")
    def test_gpu_high_precision_setting_is_preserved(self):
        previous = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("high")
            generator = torch.Generator(device="cuda").manual_seed(60966)
            decoder = torch.randn(2048, 76, generator=generator, device="cuda")
            decoder[:, 1:] += .8*decoder[:, :1]
            decoder /= decoder.norm(dim=0)
            basis, _ = orthogonal(decoder, list(range(76)))
            self.assertEqual(torch.get_float32_matmul_precision(), "high")
            matrix = decoder.cpu().double()
            residual = matrix-basis.double()@(basis.double().T@matrix)
            self.assertLess(float(residual.norm()/matrix.norm()), 2e-6)
        finally:
            torch.set_float32_matmul_precision(previous)


if __name__ == "__main__":
    unittest.main()
