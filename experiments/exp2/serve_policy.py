"""Evaluate full-SFT Exp2 checkpoints with all training projections removed."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from experiments.exp2.common import CHECKPOINT, sha256
from experiments.exp2.model import Boundary, load_policy


class RectifiedPolicy:
    def __init__(self, run):
        self.run = run
        self.policy = load_policy()
        self.checkpoints = {"frozen": CHECKPOINT / "model.safetensors"}
        for arm in ("sft", "sae_m", "random_sae", "pca_promptdiff"):
            self.checkpoints[arm] = run / f"checkpoint_{arm}/model.safetensors"
            if not self.checkpoints[arm].is_file():
                raise FileNotFoundError(self.checkpoints[arm])
        self.current = "frozen"
        self.boundary = Boundary(self.policy._model, (5, 11, 17))
        self.boundary.capture = True
        self.probes = {}
        for layer in (5, 11, 17):
            with np.load(run / f"probe_pg{layer}.npz") as z:
                self.probes[layer] = {key: z[key] for key in z.files}

    @property
    def metadata(self):
        return {**self.policy.metadata, "protocol": "exp2-cift-v1", "projection_removed": True,
                "manifest_sha256": sha256(self.run / "manifest.json"), "arms": list(self.checkpoints)}

    @torch.no_grad()
    def infer(self, observation):
        obs = dict(observation)
        request = obs.pop("exp2")
        arm = request["arm"]
        if arm not in self.checkpoints:
            raise ValueError(arm)
        if arm != self.current:
            import safetensors.torch
            safetensors.torch.load_model(self.policy._model, self.checkpoints[arm], device="cpu")
            self.current = arm
        self.policy._model.eval()
        self.boundary.basis = None
        self.boundary.clear()
        noise = np.random.default_rng(int(request["noise_seed"])).standard_normal((10, 32), dtype=np.float32)
        result = self.policy.infer(obs, noise=noise)
        info = {"arm": arm, "noise": noise, "projection_removed": True}
        for layer, probe in self.probes.items():
            pooled = self.boundary.cache[layer].float().mean(1)[0].cpu().numpy()
            logits = ((pooled-probe["mean"])/probe["scale"]) @ probe["coef"].T + probe["intercept"]
            probability = np.exp(logits-logits.max())
            info[f"pooled_pg{layer}"] = pooled
            info[f"probe_pg{layer}"] = (probability/probability.sum()).astype(np.float32)
        result["exp2"] = info
        return result

    def reset(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    from openpi.serving import websocket_policy_server

    policy = RectifiedPolicy(args.run_dir)
    print(json.dumps(policy.metadata), flush=True)
    websocket_policy_server.WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port,
                                                 metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
