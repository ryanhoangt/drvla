"""Exp1 dense/pooled localization server; no SAEs, gradients, or policy training."""

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import numpy as np
import torch

from drvla.layers import get_layer_module
from experiments.exp1.common import ROOT, SITES, sha256
from experiments.exp1.patches import patch_hidden


class LocalizationPolicy:
    def __init__(self, policy, config, config_hash):
        self.policy, self.config, self.config_hash = policy, config, config_hash
        self.phase = "passive"
        self.site, self.mode = None, None
        self.capture, self.donor, self.counts, self.diagnostics = {}, {}, {}, []
        self.mask, self.donor_mask = None, None
        self.handles = [get_layer_module(policy._model, name).register_forward_hook(self._hook(site))
                        for site, name in SITES.items()]
        original_embed = policy._model.embed_prefix

        def embed_prefix(*args, **kwargs):
            result = original_embed(*args, **kwargs)
            self.mask = result[1].detach().clone()
            return result

        policy._model.embed_prefix = embed_prefix

    @property
    def metadata(self):
        return {**self.policy.metadata, "protocol": "exp1-core-v1", "config_sha256": self.config_hash,
                "sites": SITES}

    def _hook(self, site):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if self.phase == "capture":
                self.capture.setdefault(site, []).append(hidden.detach().clone())
                return output
            if self.phase != "patch" or site != self.site:
                return output
            index = self.counts.get(site, 0)
            if index >= len(self.donor[site]):
                raise ValueError("Recipient has more denoising calls than the donor")
            self.counts[site] = index + 1
            if site.startswith("pg"):
                eligible = self.mask & self.donor_mask
            else:
                eligible = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.bool)
            edited, diagnostics = patch_hidden(hidden, self.donor[site][index], eligible, self.mode)
            diagnostics["flow_call_index"] = index if site.startswith("ae") else None
            self.diagnostics.append(diagnostics)
            if edited is hidden:
                return output
            return (edited, *output[1:]) if isinstance(output, tuple) else edited
        return hook

    @torch.no_grad()
    def infer(self, observation):
        obs = dict(observation)
        request = dict(obs.pop("exp1"))
        mode = request["mode"]
        site = request.get("site", "pg11")
        if mode not in ("baseline", "unhooked", "dense", "pooled", "pooled_norm", "self") or site not in SITES:
            raise ValueError("Invalid patch request")
        model_config = self.policy._model.config
        noise = np.random.default_rng(int(request["noise_seed"])).standard_normal(
            (model_config.action_horizon, model_config.action_dim), dtype=np.float32)
        self.capture, self.donor, self.counts, self.diagnostics = {}, {}, {}, []
        self.phase = "passive"
        if mode not in ("baseline", "unhooked"):
            self.phase = "capture"
            donor_obs = {**obs, "prompt": obs["prompt"] if mode == "self" else self.config["old_prompt"]}
            self.policy.infer(donor_obs, noise=noise)
            self.donor = self.capture
            self.donor_mask = self.mask.clone()
            self.site, self.mode, self.phase = site, mode, "patch"
        if mode == "unhooked":
            # Physically disable registered decoder hooks for the equality gate.
            for handle in self.handles:
                handle.remove()
            try:
                result = self.policy.infer(obs, noise=noise)
            finally:
                self.handles = [get_layer_module(self.policy._model, name).register_forward_hook(self._hook(s))
                                for s, name in SITES.items()]
        else:
            result = self.policy.infer(obs, noise=noise)
        if mode not in ("baseline", "unhooked"):
            if self.counts.get(site, 0) != len(self.donor.get(site, [])) or not self.diagnostics:
                raise ValueError("Donor/recipient hook call counts differ, or the hook did not run")
        info = {"site": site, "mode": mode, "noise": noise, "diagnostics": self.diagnostics,
                "recipient_valid_prefix_tokens": int(self.mask.sum())}
        if mode not in ("baseline", "unhooked"):
            info["donor_valid_prefix_tokens"] = int(self.donor_mask.sum())
        result["exp1"] = info
        self.phase = "passive"
        self.capture, self.donor = {}, {}
        return result

    def reset(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as training_config

    config = training_config.get_config("pi05_libero")
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, pytorch_compile_mode=None))
    base = policy_config.create_trained_policy(config, ROOT / "checkpoints/pi05_libero_pytorch", pytorch_device="cuda")
    policy = LocalizationPolicy(base, json.loads(args.config.read_text()), sha256(args.config))
    websocket_policy_server.WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port,
                                                 metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
