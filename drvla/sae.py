"""TopK sparse autoencoder with per-sample input normalization and AuxK loss.

Follows Gao et al., "Scaling and evaluating sparse autoencoders" (2024). Given an
activation ``x``:

1. ``x_c = x - b_pre`` where ``b_pre`` is learned and initialized to the
   geometric median of training activations.
2. ``x_n = (x_c - mean(x_c)) / ||x_c - mean(x_c)||`` (scalar mean over the model
   dimension, per sample).
3. ``z = ReLU(TopK(W_enc x_n))``.
4. ``x_hat = (W_dec z) * ||x_c - mean(x_c)|| + mean(x_c) + b_pre``.

Decoder columns are kept at unit norm, so each feature's contribution is set by
its scalar activation and ``W_dec[:, j]`` is the steering direction for feature
``j``. Neither the encoder nor the decoder has a bias.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_FORMAT = "drvla-sae-v1"


@dataclass
class SAEConfig:
    input_dim: int
    num_features: int
    k: int
    auxk: int = 512
    auxk_coef: float = 1 / 32
    dead_steps_threshold: int = 500

    def __post_init__(self):
        if not 0 < self.k < self.num_features:
            raise ValueError(f"k={self.k} must be in (0, num_features={self.num_features}).")


@dataclass
class SAEOutput:
    features: torch.Tensor  # (B, num_features), non-negative and k-sparse
    reconstruction: torch.Tensor  # (B, input_dim)
    pre_activation: torch.Tensor  # (B, num_features), encoder output before TopK
    topk_indices: torch.Tensor  # (B, k)
    x_normalized: torch.Tensor  # (B, input_dim)
    recon_normalized: torch.Tensor  # (B, input_dim)


class TopKSAE(nn.Module):
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.k = config.k
        # AuxK uses at most as many dead latents as there are latents outside the TopK.
        self.auxk = min(config.auxk, config.num_features - config.k)

        self.pre_bias = nn.Parameter(torch.zeros(config.input_dim))
        self.encoder = nn.Linear(config.input_dim, config.num_features, bias=False)
        self.decoder = nn.Linear(config.num_features, config.input_dim, bias=False)
        # Optimization steps since each latent was last in the TopK (for AuxK).
        self.register_buffer("steps_since_activation", torch.zeros(config.num_features, dtype=torch.long))

        nn.init.xavier_uniform_(self.decoder.weight)
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True) + 1e-8)
            # Encoder starts as the scaled transpose of the decoder.
            self.encoder.weight.copy_(self.decoder.weight.T * (self.k / config.num_features) ** 0.5)

    @property
    def decoder_directions(self) -> torch.Tensor:
        """Unit-norm decoder directions, shape (num_features, input_dim)."""
        return self.decoder.weight.T

    def forward(self, x: torch.Tensor) -> SAEOutput:
        x_centered = x - self.pre_bias
        x_mean = x_centered.mean(dim=-1, keepdim=True)
        x_shifted = x_centered - x_mean
        x_norm = x_shifted.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_normalized = x_shifted / x_norm

        pre_activation = self.encoder(x_normalized)
        topk_values, topk_indices = torch.topk(pre_activation, self.k, dim=-1)
        features = torch.zeros_like(pre_activation).scatter_(-1, topk_indices, F.relu(topk_values))

        if self.training:
            self.steps_since_activation += 1
            self.steps_since_activation[topk_indices.unique()] = 0

        recon_normalized = self.decoder(features)
        reconstruction = recon_normalized * x_norm + x_mean + self.pre_bias
        return SAEOutput(features, reconstruction, pre_activation, topk_indices, x_normalized, recon_normalized)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Sparse feature activations for a batch of inputs (inference only)."""
        was_training = self.training
        self.eval()
        features = self.forward(x).features
        self.train(was_training)
        return features

    def dead_mask(self) -> torch.Tensor:
        return self.steps_since_activation > self.config.dead_steps_threshold

    def auxk_loss(self, out: SAEOutput) -> torch.Tensor:
        """MSE between the normalized residual and its reconstruction from the top dead latents."""
        dead = torch.where(self.dead_mask())[0]
        num_aux = min(self.auxk, len(dead))
        if num_aux == 0:
            return out.x_normalized.new_zeros(())
        residual = out.x_normalized - out.recon_normalized
        aux_values, local_idx = torch.topk(out.pre_activation[:, dead], num_aux, dim=-1)
        aux_values = F.relu(aux_values)
        if (aux_values == 0).all():
            return out.x_normalized.new_zeros(())
        aux_features = torch.zeros_like(out.pre_activation).scatter_(-1, dead[local_idx], aux_values)
        return F.mse_loss(self.decoder(aux_features), residual)

    @torch.no_grad()
    def init_pre_bias(self, samples: torch.Tensor, max_iter: int = 100, tol: float = 1e-5):
        """Set ``b_pre`` to the geometric median of ``samples`` (Weiszfeld's algorithm)."""
        median = samples.mean(dim=0)
        for _ in range(max_iter):
            weights = 1.0 / (samples - median).norm(dim=1, keepdim=True).clamp(min=1e-8)
            new_median = (weights * samples).sum(dim=0) / weights.sum()
            if (new_median - median).norm() < tol:
                break
            median = new_median
        self.pre_bias.copy_(median)

    @torch.no_grad()
    def normalize_decoder_(self):
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True) + 1e-8)

    @torch.no_grad()
    def project_decoder_grads_(self):
        """Remove the gradient component parallel to each decoder column (unit-norm constraint)."""
        grad = self.decoder.weight.grad
        if grad is not None:
            weight = self.decoder.weight
            grad.sub_((grad * weight).sum(dim=0, keepdim=True) * weight)

    def save(self, path: str | Path, **extra):
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "config": asdict(self.config),
                "state_dict": self.state_dict(),
                **extra,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "TopKSAE":
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(
                f"{path} is not a {CHECKPOINT_FORMAT} checkpoint (format={checkpoint.get('format')!r})."
            )
        sae = cls(SAEConfig(**checkpoint["config"]))
        sae.load_state_dict(checkpoint["state_dict"])
        return sae.to(device).eval()


SAE_FILENAME = "sae.pt"


def sae_path(sae_dir: str | Path, layer: str) -> Path:
    """Location of the SAE for ``layer`` inside an SAE directory (``<sae_dir>/<layer>/sae.pt``)."""
    return Path(sae_dir) / layer / SAE_FILENAME


def load_sae(sae_dir: str | Path, layer: str, device: str | torch.device = "cpu") -> TopKSAE:
    path = sae_path(sae_dir, layer)
    if not path.exists():
        raise FileNotFoundError(f"No SAE for layer {layer} at {path}.")
    return TopKSAE.load(path, device=device)


@torch.no_grad()
def encode_batched(sae: TopKSAE, activations, batch_size: int = 4096) -> torch.Tensor:
    """Encode a (T, d) array or tensor in batches; returns features on the CPU as float32."""
    device = sae.pre_bias.device
    x = torch.tensor(np.asarray(activations), dtype=torch.float32)
    chunks = [sae.encode(x[i : i + batch_size].to(device)).cpu() for i in range(0, len(x), batch_size)]
    return torch.cat(chunks) if chunks else torch.zeros(0, sae.config.num_features)
