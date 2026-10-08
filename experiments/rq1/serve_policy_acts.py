"""openpi policy server that also returns mean-pooled residual-stream activations.

Same model and preprocessing as ``openpi/scripts/serve_policy.py``; forward hooks on the
drvla layers (the same hooks as ``drvla.pi05``) add ``activations: {layer: (d,) float32}``
to every response. PaliGemma layers run once per inference (prefix); action-expert
layers run once per denoising step and the last (final) step is kept.

Run from the openpi directory:
    uv run ../experiments/rq1/serve_policy_acts.py --port 8000 \
        --config pi05_libero --dir ../checkpoints/pi05_libero_pytorch
"""

import dataclasses
import logging

import numpy as np
import torch
import tyro
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

from drvla.layers import PAPER_LAYERS, get_layer_module


class ActivationPolicy:
    def __init__(self, policy, layers):
        self._policy = policy
        self.layers = list(layers)
        self._captured = {}
        model = policy._model
        for layer in self.layers:
            get_layer_module(model, layer).register_forward_hook(self._hook(layer))

    def _hook(self, layer):
        def record(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._captured[layer] = hidden.detach().mean(dim=1).float().cpu()

        return record

    @property
    def metadata(self):
        return self._policy.metadata

    def infer(self, obs):
        self._captured.clear()
        with torch.no_grad():
            out = self._policy.infer(obs)
        missing = [l for l in self.layers if l not in self._captured]
        if missing:
            raise RuntimeError(f"Hooks did not fire for {missing}")
        out["activations"] = {l: self._captured[l][0].numpy().astype(np.float32) for l in self.layers}
        return out

    def reset(self):
        pass


@dataclasses.dataclass
class Args:
    config: str = "pi05_libero"
    dir: str = "../checkpoints/pi05_libero_pytorch"
    port: int = 8000


def main(args: Args):
    config = _config.get_config(args.config)
    # No torch.compile: hooks must run eagerly, as in drvla's activation collection.
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, pytorch_compile_mode=None))
    policy = _policy_config.create_trained_policy(config, args.dir)
    policy = ActivationPolicy(policy, PAPER_LAYERS)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
