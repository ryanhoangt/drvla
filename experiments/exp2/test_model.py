"""Checks for the forward/backward intervention semantics, independent of CUDA."""

import unittest

import torch
from torch import nn

from experiments.exp2.model import instrument_joint_forward, project


class TinyJoint(nn.Module):
    def forward(self, value):
        def compute_layer_complete(layer_idx, inputs_embeds):
            outputs_embeds = []
            for i, hidden_states in enumerate(inputs_embeds):
                # A residual skip which a layernorm-only hook cannot overwrite.
                out_emb = hidden_states + 2*hidden_states
                outputs_embeds.append(out_emb)
            return outputs_embeds
        return compute_layer_complete(0, [value])[0]


class ProjectionTests(unittest.TestCase):
    def test_joint_instrumentation_edits_entire_residual_and_its_gradient(self):
        model = TinyJoint()
        x = torch.tensor([[1., 2., 3.]], requires_grad=True)
        reference = model(x)
        original, _ = instrument_joint_forward(model, lambda stream, layer, h: h)
        self.assertTrue(torch.equal(model(x), reference))
        basis = torch.tensor([[1.], [0.], [0.]])
        model._exp2_transform = lambda stream, layer, h: project(h, basis, torch.tensor([True]))
        output = model(x)
        torch.testing.assert_close(output, torch.tensor([[0., 6., 9.]]))
        output.sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([[0., 3., 3.]]))
        model.forward = original

    def test_visual_projection_blocks_subspace_gradient_and_preserves_language(self):
        torch.manual_seed(1)
        x = torch.randn(2, 6, 12, requires_grad=True)
        basis = torch.linalg.qr(torch.randn(12, 3)).Q
        eligible = torch.tensor([[True, True, False, False, False, False]]).expand(2, -1)
        output = project(x, basis, eligible)
        torch.testing.assert_close(output[eligible] @ basis, torch.zeros(4, 3), atol=1e-6, rtol=0)
        self.assertTrue(torch.equal(output[~eligible], x[~eligible]))
        upstream = torch.randn_like(output)
        (output * upstream).sum().backward()
        torch.testing.assert_close(x.grad[eligible] @ basis, torch.zeros(4, 3), atol=1e-6, rtol=0)
        self.assertTrue(torch.equal(x.grad[~eligible], upstream[~eligible]))

    def test_bf16_preserves_untouched_tokens_exactly(self):
        x = torch.randn(1, 6, 12).bfloat16()
        basis = torch.linalg.qr(torch.randn(12, 3)).Q
        mask = torch.tensor([[True, True, False, False, False, False]])
        result = project(x, basis, mask)
        self.assertEqual(result.dtype, x.dtype)
        self.assertTrue(torch.equal(result[~mask], x[~mask]))


if __name__ == "__main__":
    unittest.main()
