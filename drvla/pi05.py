"""Residual-stream activations from Pi0.5, using openpi's PyTorch implementation.

Each timestep is one forward pass of the policy (``PI0Pytorch.sample_actions``)
on that timestep's observation. Forward hooks on the selected decoder blocks
record the block output averaged over all token positions:

* PaliGemma layers see the prefix: 3 x 256 image tokens (base camera, wrist
  camera, and the masked-out right-wrist slot) followed by the padded prompt,
  which in Pi0.5 contains the task and the discretized robot state.
* Action-expert layers see the action tokens. They run once per flow-matching
  step; the recorded value is from the final denoising step.
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from drvla.layers import get_layer_module, parse_layer

IMAGE_SIZE = 224


def resize_with_pad(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Resize to ``size`` x ``size`` keeping aspect ratio, zero-padding the short side."""
    pil = Image.fromarray(image)
    if pil.size == (size, size):
        return image
    ratio = max(pil.width / size, pil.height / size)
    resized = pil.resize((int(pil.width / ratio), int(pil.height / ratio)), resample=Image.BILINEAR)
    canvas = Image.new(resized.mode, (size, size), 0)
    canvas.paste(resized, (max(0, int((size - resized.width) / 2)), max(0, int((size - resized.height) / 2))))
    return np.asarray(canvas)


class Pi05ActivationExtractor:
    """Runs Pi0.5 on batches of observations and returns mean-pooled activations per layer.

    Args:
        checkpoint_dir: PyTorch checkpoint directory (``config.json``, ``model.safetensors``, ``assets/``).
        asset_id: Sub-directory of ``assets/`` holding the ``norm_stats.json`` used to normalize
            the robot state before it is tokenized (``physical-intelligence/libero`` for
            pi05_libero, ``droid`` for pi05_droid).
        layers: Layer names from :mod:`drvla.layers`.
        num_denoising_steps: Flow-matching steps per forward pass.
        seed: Seeds the flow-matching noise, which the action-expert activations depend on.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        asset_id: str,
        layers: list[str],
        device: str = "cuda",
        num_denoising_steps: int = 10,
        seed: int = 0,
    ):
        import safetensors.torch
        from openpi.models import pi0_config
        from openpi.models import tokenizer as _tokenizer
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from openpi.shared import normalize

        checkpoint_dir = Path(checkpoint_dir)
        config = json.loads((checkpoint_dir / "config.json").read_text())
        self.model_config = pi0_config.Pi0Config(
            action_dim=config["action_dim"],
            action_horizon=config["action_horizon"],
            dtype=config["precision"],
            paligemma_variant=config["paligemma_variant"],
            action_expert_variant=config["action_expert_variant"],
            pi05=True,
            pytorch_compile_mode=None,
        )
        model = PI0Pytorch(self.model_config)
        safetensors.torch.load_model(model, checkpoint_dir / "model.safetensors")
        model.paligemma_with_expert.to_bfloat16_for_selected_params(config["precision"])
        self.model = model.to(device).eval()
        self.device = device
        self.num_denoising_steps = num_denoising_steps
        self.generator = torch.Generator(device=device).manual_seed(seed)

        state_stats = normalize.load(checkpoint_dir / "assets" / asset_id)["state"]
        self.state_q01 = np.asarray(state_stats.q01)
        self.state_q99 = np.asarray(state_stats.q99)
        self.tokenizer = _tokenizer.PaligemmaTokenizer(max_len=self.model_config.max_token_len)

        self.layers = list(layers)
        for layer in self.layers:
            parse_layer(layer)
        self._captured: dict[str, torch.Tensor] = {}
        self._handles = [get_layer_module(self.model, layer).register_forward_hook(self._hook(layer)) for layer in self.layers]

    def _hook(self, layer: str):
        def record(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Mean over tokens in the model's dtype, then float32 (as for the paper's activations).
            self._captured[layer] = hidden.detach().mean(dim=1).float().cpu()

        return record

    def close(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Quantile-normalize the state to [-1, 1] before it is discretized into the prompt.

        openpi may pad the statistics to the model's action dimension (32 for pi05_droid),
        so they can be longer than the state but never shorter.
        """
        dims = len(state)
        if dims > len(self.state_q01):
            raise ValueError(
                f"State has {dims} dims but the norm stats cover only {len(self.state_q01)}; "
                "check that --asset-id matches the dataset and checkpoint."
            )
        normalized = np.array(state, copy=True)  # keeps the state's dtype, as in the paper's pipeline
        normalized[:] = (state - self.state_q01[:dims]) / (self.state_q99[:dims] - self.state_q01[:dims] + 1e-6) * 2.0 - 1.0
        return normalized

    def _image_batch(self, images: np.ndarray) -> torch.Tensor:
        if images.dtype != np.uint8:
            raise TypeError(f"Expected uint8 images, got {images.dtype}.")
        batch = np.stack([resize_with_pad(image) for image in images]).astype(np.float32)
        batch = (batch / 255.0 - 0.5) * 2.0
        return torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)

    @torch.no_grad()
    def __call__(self, images: np.ndarray, wrist_images: np.ndarray, states: np.ndarray, prompt: str) -> dict[str, np.ndarray]:
        """Activations for a batch of B timesteps of one episode.

        Args:
            images: (B, H, W, 3) uint8 base-camera frames.
            wrist_images: (B, H, W, 3) uint8 wrist-camera frames.
            states: (B, state_dim) raw robot state.
            prompt: Task instruction.

        Returns:
            ``{layer: (B, d) float32 array}``.
        """
        from openpi.models import model as _model

        batch_size = len(images)
        base = self._image_batch(images)
        tokens, masks = zip(*(self.tokenizer.tokenize(prompt, state=self._normalize_state(s)) for s in states))
        ones = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        observation = _model.Observation(
            images={
                "base_0_rgb": base,
                "left_wrist_0_rgb": self._image_batch(wrist_images),
                "right_wrist_0_rgb": torch.zeros_like(base),
            },
            image_masks={"base_0_rgb": ones, "left_wrist_0_rgb": ones, "right_wrist_0_rgb": ~ones},
            state=torch.as_tensor(states, dtype=torch.float32, device=self.device),
            tokenized_prompt=torch.as_tensor(np.stack(tokens), dtype=torch.long, device=self.device),
            tokenized_prompt_mask=torch.as_tensor(np.stack(masks), dtype=torch.bool, device=self.device),
            token_ar_mask=torch.zeros(batch_size, len(tokens[0]), dtype=torch.long, device=self.device),
            token_loss_mask=torch.zeros(batch_size, len(tokens[0]), dtype=torch.bool, device=self.device),
        )
        noise = torch.randn(
            (batch_size, self.model_config.action_horizon, self.model_config.action_dim),
            generator=self.generator,
            device=self.device,
        )
        self._captured.clear()
        self.model.sample_actions(self.device, observation, noise=noise, num_steps=self.num_denoising_steps)
        missing = [layer for layer in self.layers if layer not in self._captured]
        if missing:
            raise RuntimeError(f"Hooks did not fire for {missing}.")
        return {layer: self._captured[layer].numpy() for layer in self.layers}
