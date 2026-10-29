"""Two residual projections and a strictly PG0..PG16 training mask."""

import hashlib

import torch

from experiments.exp2.model import Boundary, project


class TaskBoundary(Boundary):
    def __init__(self, model):
        self.bases = {}
        self.pre_cache = {}
        super().__init__(model, (5, 11, 17))

    def clear(self):
        super().clear()
        self.pre_cache = {}

    def _transform(self, layer, hidden):
        self.calls[layer] = self.calls.get(layer, 0) + 1
        if self.gradients and not hidden.requires_grad:
            hidden = hidden.detach().requires_grad_(True)
        if self.capture:
            self.pre_cache[layer] = hidden
            if hidden.requires_grad:
                hidden.retain_grad()
        if layer in self.bases:
            before = hidden
            hidden = project(hidden, self.bases[layer], self.visual)
            if not self.model.training:
                visual = self.visual[..., None]
                self.realized.append({"layer": layer,
                    "removed_energy_fraction": float(((hidden.float()-before.float())**2 * visual).sum()
                        / (before.float().square()*visual).sum().clamp_min(1e-20)),
                    "rank": self.bases[layer].shape[1], "visual_tokens": int(self.visual.sum())})
        if self.edit is not None and layer == self.projection_layer:
            hidden = self.edit(hidden)
        if self.capture:
            self.cache[layer] = hidden
        return hidden


def pg_only(model):
    model.requires_grad_(False)
    blocks = model.paligemma_with_expert.paligemma.language_model.layers
    if len(blocks) != 18:
        raise ValueError("Expected 18 PaliGemma blocks")
    for block in blocks[:17]:
        block.requires_grad_(True)
    allowed = {id(p) for block in blocks[:17] for p in block.parameters()}
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert {id(p) for p in model.parameters() if p.requires_grad} == allowed
    assert not any(p.requires_grad for p in model.paligemma_with_expert.gemma_expert.parameters())
    return names


def frozen_digest(model):
    """Exact hash, including AE and PG17; no tolerance for weight changes."""
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if not value.requires_grad:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
