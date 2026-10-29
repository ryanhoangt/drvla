"""CPU integration checks using the actual OpenPI FM/joint code with tiny Gemmas.

No policy checkpoint, dataset download, GPU, or full vision backbone is loaded.
The 16-dimensional two-layer models exercise the native RMSNorm/attention,
flow-matching objective, checkpointed backward and our residual instrumentation.
"""

import copy
import types
import unittest

import numpy as np
import torch
from torch import nn
from transformers import GemmaConfig, GemmaForCausalLM

from openpi import transforms
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.shared.normalize import NormStats

from experiments.exp2.model import Boundary, tensors


class TinyJoint(nn.Module):
    forward = PaliGemmaWithExpertModel.forward

    def __init__(self, num_layers=2):
        super().__init__()
        cfg = GemmaConfig(hidden_size=16, intermediate_size=32, num_attention_heads=8,
                          num_key_value_heads=1, head_dim=2, num_hidden_layers=num_layers, vocab_size=32,
                          hidden_activation="gelu_pytorch_tanh", use_adarms=False, adarms_cond_dim=None)
        self.paligemma = nn.Module()
        self.paligemma.config = types.SimpleNamespace(text_config=cfg)
        self.paligemma.model = nn.Module()
        self.paligemma.model.language_model = GemmaForCausalLM(cfg).model
        self.paligemma.language_model = self.paligemma.model.language_model
        self.paligemma.vision_tower = nn.Identity()
        expert_cfg = copy.deepcopy(cfg)
        expert_cfg.use_adarms = True
        expert_cfg.adarms_cond_dim = 16
        self.gemma_expert = GemmaForCausalLM(expert_cfg)
        self.gemma_expert.model.embed_tokens = None


def tiny_model(num_layers=2, action_dim=4, action_horizon=2):
    # Avoid constructing the production 3.6B model; retain its actual methods.
    model = PI0Pytorch.__new__(PI0Pytorch)
    nn.Module.__init__(model)
    model.config = types.SimpleNamespace(action_horizon=action_horizon, action_dim=action_dim)
    model.pi05 = True
    model.gradient_checkpointing_enabled = False
    model.paligemma_with_expert = TinyJoint(num_layers)
    model.action_in_proj, model.action_out_proj = nn.Linear(action_dim, 16), nn.Linear(16, action_dim)
    model.time_mlp_in, model.time_mlp_out = nn.Linear(16, 16), nn.Linear(16, 16)
    model.vision_stub, model.language_stub = nn.Linear(3, 16), nn.Embedding(32, 16)
    model._preprocess_observation = lambda obs, train=True: (
        list(obs.images.values()), list(obs.image_masks.values()), obs.tokenized_prompt,
        obs.tokenized_prompt_mask, obs.state)

    def embed_prefix(images, masks, tokens, token_mask):
        embedded = [model.vision_stub(image.mean((-2, -1)))[:, None] for image in images]
        embedded.append(model.language_stub(tokens))
        valid = torch.cat([mask[:, None] for mask in masks]+[token_mask], dim=1)
        return torch.cat(embedded, dim=1), valid, torch.zeros_like(valid)

    model.embed_prefix = embed_prefix
    return model.eval()


class NativeForwardTests(unittest.TestCase):
    def test_normalized_double_actions_reach_native_fm_as_float32(self):
        model = tiny_model()
        stats = {"actions": NormStats(mean=np.zeros(4, dtype=np.float64), std=np.ones(4, dtype=np.float64))}
        normalizer = transforms.Normalize(stats)

        def transform(raw):
            return normalizer({"actions": raw["actions"], "state": np.zeros(4, np.float32),
                "image": {key: np.zeros((2, 2, 3), np.uint8) for key in ("base", "wrist", "masked")},
                "image_mask": {"base": np.True_, "wrist": np.True_, "masked": np.False_},
                "tokenized_prompt": np.array([1, 2], np.int32), "tokenized_prompt_mask": np.ones(2, bool)})

        raw = {"image": None, "wrist": None, "state": None, "actions": np.ones((2, 4), np.float32)}
        self.assertEqual(transform(raw)["actions"].dtype, np.float64)
        observation, actions = tensors(types.SimpleNamespace(_input_transform=transform), raw, "task", device="cpu")
        self.assertEqual(actions.dtype, torch.float32)
        noise, time = torch.zeros_like(actions), torch.tensor([0.5])
        original_loss = model(observation, actions, noise=noise, time=time)
        fixed_noise = torch.randn_like(actions)
        original_action = model.sample_actions(torch.device("cpu"), observation, noise=fixed_noise, num_steps=2)
        boundary = Boundary(model, layers=(0, 1))
        boundary.projection_layer = 0
        loss = model(observation, actions, noise=noise, time=time)
        self.assertTrue(torch.equal(loss, original_loss))
        action = model.sample_actions(torch.device("cpu"), observation, noise=fixed_noise, num_steps=2)
        self.assertTrue(torch.equal(action, original_action))
        # Exact passive behavior is checked on BOTH native joint FM and cached inference.
        model.requires_grad_(False)
        boundary.basis = torch.eye(16)[:, :2]
        boundary.capture = boundary.gradients = True
        boundary.clear()
        model(observation, actions, noise=noise, time=time).mean().backward()
        gradient = boundary.cache[0].grad
        self.assertIsNotNone(gradient)
        torch.testing.assert_close(gradient[boundary.visual] @ boundary.basis,
                                   torch.zeros(int(boundary.visual.sum()), 2), atol=1e-7, rtol=0)
        # The same intervention must execute during checkpointed full-SFT backward.
        boundary.capture = boundary.gradients = False
        boundary.clear()
        model.requires_grad_(True)
        model.gradient_checkpointing_enable()
        model.train()
        model(observation, actions, noise=noise, time=time).mean().backward()
        self.assertGreater(float(model.action_out_proj.weight.grad.norm()), 0)
        self.assertGreaterEqual(boundary.calls.get(0, 0), 1)
        boundary.close()


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
