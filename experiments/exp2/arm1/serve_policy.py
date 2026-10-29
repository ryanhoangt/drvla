"""Checkpoint switching and explicit per-request projection mode."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import safetensors.torch
import torch

from experiments.exp2.common import CHECKPOINT, sha256
from experiments.exp2.model import load_policy
from experiments.exp2.arm1.common import REFERENCE
from experiments.exp2.arm1.model import TaskBoundary


class RectifiedPolicy:
    def __init__(self, run, device="cuda"):
        self.run = run
        self.policy = load_policy()
        self.checkpoints = {"frozen": CHECKPOINT / "model.safetensors"}
        for name in ("sft_pg_only", "acttask", "random_acttask"):
            self.checkpoints[name] = run / f"checkpoint_{name}/model.safetensors"
            if not self.checkpoints[name].is_file():
                raise FileNotFoundError(self.checkpoints[name])
        self.bases = torch.load(run / "bases.pt", map_location=device, weights_only=True)
        self.current = "frozen"
        self.boundary = TaskBoundary(self.policy._model)
        self.boundary.capture = True
        self.probes = {}
        for layer in (5, 11, 17):
            with np.load(REFERENCE / f"probe_pg{layer}.npz") as z:
                self.probes[layer] = {key: z[key] for key in z.files}

    @property
    def metadata(self):
        return {**self.policy.metadata, "protocol": "exp2-arm1-v1", "explicit_hook_modes": True,
                "manifest_sha256": sha256(self.run / "manifest.json")}

    @torch.no_grad()
    def infer(self, observation):
        obs = dict(observation)
        request = obs.pop("exp2")
        label = request["arm"]
        if label == "frozen":
            arm, mode = "frozen", "off"
        else:
            arm, mode = label.rsplit("_", 1)
        if arm not in self.checkpoints or mode not in ("off", "on"):
            raise ValueError(label)
        if arm != self.current:
            safetensors.torch.load_model(self.policy._model, self.checkpoints[arm], device="cpu")
            self.current = arm
        self.policy._model.eval()
        self.boundary.bases = self.bases[arm] if mode == "on" else {}
        self.boundary.clear()
        noise = np.random.default_rng(int(request["noise_seed"])).standard_normal((10, 32), dtype=np.float32)
        result = self.policy.infer(obs, noise=noise)
        info = {"arm": label, "noise": noise, "projection_removed": mode == "off"}
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
