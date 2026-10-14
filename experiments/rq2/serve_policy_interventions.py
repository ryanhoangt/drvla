"""One policy/SAE server for paired RQ2 rollout and counterfactual requests."""

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import numpy as np
import torch

from drvla.layers import get_layer_module
from drvla.sae import load_sae, sae_path
from experiments.rq2.common import ROOT, sha256
from experiments.rq2.interventions import MODES, apply_pooled_delta, feature_delta


class InterventionPolicy:
    def __init__(self, policy, bundle, sae_dir, device="cuda", features_hash=None):
        self.policy = policy
        self.bundle = bundle
        self.features_hash = features_hash
        self.layer = bundle["manifest"]["layer"]
        if sha256(sae_path(sae_dir, self.layer)) != bundle["manifest"]["sae_sha256"]:
            raise ValueError("Feature bundle and SAE hashes differ")
        self.sae = load_sae(sae_dir, self.layer, device=device)
        self.sae.requires_grad_(False)
        self._mode = "baseline"
        self._pair = None
        self._alpha = 1.0
        self._active = False
        self._donor = None
        self._captured = {}
        self._bypass = False
        self._handle = get_layer_module(policy._model, self.layer).register_forward_hook(self._hook)

    @property
    def metadata(self):
        return {
            **self.policy.metadata, "rq2": "pooled-direction-v1", "layer": self.layer,
            "sae_sha256": self.bundle["manifest"]["sae_sha256"], "modes": [*MODES, "unhooked"],
            "features_sha256": self.features_hash,
            "prompt_style": self.bundle["manifest"].get("prompt_style", "raw-bddl"),
        }

    def _hook(self, module, inputs, output):
        if self._bypass:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        pooled = hidden.detach().mean(dim=1).float()
        features = self.sae.encode(pooled)
        difference = torch.zeros_like(features)
        ids = []
        if self._active:
            difference, ids = feature_delta(
                self.sae, features, self._pair, self.bundle["global_mean"],
                self._mode, self._alpha, self._donor,
            )
        edited, delta = apply_pooled_delta(hidden, pooled, self.sae, difference)
        actual_delta = (edited.float() - hidden.float()).mean(dim=1)
        after_features = self.sae.encode(edited.detach().mean(dim=1).float())
        target_norm = 0.0
        target_realized_norm = 0.0
        if self._active and self._mode.startswith(("random_", "general_")):
            target_mode = "suppress_old" if self._mode.endswith("suppress") else "boost_new"
            target_diff, _ = feature_delta(
                self.sae, features, self._pair, self.bundle["global_mean"], target_mode, self._alpha,
            )
            target_edited, target_delta = apply_pooled_delta(hidden, pooled, self.sae, target_diff)
            target_norm = float(target_delta.norm())
            target_realized_norm = float((target_edited.float() - hidden.float()).mean(dim=1).norm())
        self._captured = {
            "pooled": pooled[0].cpu().numpy(), "features": features[0].cpu().numpy(),
            "features_after": after_features[0].cpu().numpy(),
            "edited_features": (features + difference)[0].cpu().numpy(),
            "feature_delta": difference[0].cpu().numpy(),
            "diagnostics": {
                "applied": bool(torch.count_nonzero(delta)), "n_features": len(ids),
                "feature_ids": ids, "delta_norm": float(delta.norm()),
                "realized_delta_norm": float(actual_delta.norm()),
                "relative_delta_norm": float(delta.norm() / pooled.norm().clamp(min=1e-12)),
                "control_target_norm": target_norm,
                "control_norm_ratio": float(delta.norm() / target_norm) if target_norm > 0 else 1.0,
                "control_realized_target_norm": target_realized_norm,
                "control_realized_norm_ratio": float(actual_delta.norm() / target_realized_norm)
                if target_realized_norm > 0 else 1.0,
                "min_edited_feature": float((features + difference).min()),
            },
        }
        if edited is hidden:
            return output
        if isinstance(output, tuple):
            return (edited, *output[1:])
        return edited

    def _run(self, obs, noise):
        self._captured = {}
        result = self.policy.infer(obs, noise=noise)
        if not self._captured:
            raise RuntimeError(f"Hook did not run: {self.layer}")
        return result

    @torch.no_grad()
    def infer(self, observation):
        obs = dict(observation)
        request = dict(obs.pop("rq2"))
        pair_key = request["pair"]
        if pair_key not in self.bundle["pairs"]:
            raise KeyError(pair_key)
        mode = request.get("mode", "baseline")
        if mode not in (*MODES, "unhooked"):
            raise ValueError(mode)
        self._pair = self.bundle["pairs"][pair_key]
        self._mode = mode
        self._alpha = float(request.get("alpha", 1.0))
        if mode == "patch_old" and self._alpha > 1:
            raise ValueError("patch_old alpha > 1 could create negative feature coordinates")
        seed = int(request["noise_seed"])
        generator = np.random.default_rng(seed)
        config = self.policy._model.config
        noise = generator.standard_normal((config.action_horizon, config.action_dim), dtype=np.float32)
        # The caller controls which inference calls receive the edit.
        active = bool(request.get("active", True))
        if mode == "unhooked":
            self._bypass = True
            try:
                result = self.policy.infer(obs, noise=noise)
            finally:
                self._bypass = False
            result["rq2"] = {"noise_seed": seed, "noise": noise, "mode": mode}
            return result
        self._active = False
        self._donor = None
        if mode == "patch_old" and active:
            donor_obs = {**obs, "prompt": self._pair["old_prompt"]}
            self._run(donor_obs, noise)
            self._donor = torch.as_tensor(
                self._captured["features"], device=self.sae.pre_bias.device,
            )[None, :]
        self._active = active
        result = self._run(obs, noise)
        captured = self._captured
        result["rq2"] = {
            **captured, "noise_seed": seed, "noise": noise, "mode": mode, "active": active,
        }
        return result

    def reset(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/pi05_libero_pytorch")
    parser.add_argument("--saes", type=Path, default=ROOT / "saes/pi05_libero_server")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=8200)
    args = parser.parse_args()
    torch.set_num_threads(2)
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as training_config

    config = training_config.get_config("pi05_libero")
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, pytorch_compile_mode=None))
    base_policy = policy_config.create_trained_policy(config, args.checkpoint, pytorch_device=args.device)
    policy = InterventionPolicy(base_policy, json.loads(args.features.read_text()), args.saes, args.device,
                                features_hash=sha256(args.features))
    logging.info("RQ2 policy ready: %s port %d", policy.layer, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy, host="127.0.0.1", port=args.port, metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
