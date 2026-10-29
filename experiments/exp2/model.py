"""Deployment transforms and residual-stream instrumentation for full SFT.

OpenPI's joint FM forward bypasses decoder-block __call__. We instrument an
in-memory copy of its joint forward immediately after the second residual add.
Inference uses ordinary block-output hooks at the same boundary. Hooking only
the next layernorm input would leave the residual skip path unprojected.
No modifications to OpenPI or its environment are necessary.
"""

import ast
import dataclasses
import inspect
import textwrap
import types

import numpy as np
import torch

from experiments.exp2.common import CHECKPOINT


def load_policy():
    from openpi.policies import policy_config
    from openpi.training import config

    cfg = config.get_config("pi05_libero")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    policy = policy_config.create_trained_policy(cfg, CHECKPOINT, pytorch_device="cuda")
    model = policy._model
    model.requires_grad_(False)
    original = model._preprocess_observation
    # Deterministic paired observations; no independently sampled image augmentations.
    model._preprocess_observation = lambda obs, train=True: original(obs, train=False)
    return policy


def tensors(policy, raw, prompt, device="cuda"):
    from openpi.models.model import Observation
    import jax

    inputs = policy._input_transform({"observation/image": raw["image"],
                                     "observation/wrist_image": raw["wrist"],
                                     "observation/state": raw["state"], "prompt": prompt,
                                     "actions": raw["actions"]})
    inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x).copy()).to(device)[None], inputs)
    # Normalize uses NumPy float64 statistics. The native OpenPI trainer casts
    # the resulting actions to float32 before constructing the FM noisy input.
    actions = inputs.pop("actions").to(dtype=torch.float32)
    return Observation.from_dict(inputs), actions


def instrument_joint_forward(module, transform):
    """Insert one explicit residual callback into a copy of the installed source.

    Fail closed if the expected source structure changes. Keeping the remaining
    implementation identical avoids inventing a second flow-matching backbone.
    """
    original = module.forward
    source = ast.parse(textwrap.dedent(inspect.getsource(original.__func__)))
    functions = [node for node in ast.walk(source)
                 if isinstance(node, ast.FunctionDef) and node.name == "compute_layer_complete"]
    if len(functions) != 1:
        raise RuntimeError("OpenPI joint layer function changed; do not silently use incorrect hooks")

    class InsertResidualCallback(ast.NodeTransformer):
        count = 0

        def visit_Expr(self, node):
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == "outputs_embeds"
                    and call.func.attr == "append" and len(call.args) == 1
                    and isinstance(call.args[0], ast.Name) and call.args[0].id == "out_emb"):
                self.count += 1
                assignment = ast.parse("out_emb = self._exp2_transform(i, layer_idx, out_emb)").body[0]
                return [ast.copy_location(assignment, node), node]
            return node

    insertion = InsertResidualCallback()
    insertion.visit(functions[0])
    if insertion.count != 1:
        raise RuntimeError(f"Expected exactly one joint residual callback site, found {insertion.count}")
    ast.fix_missing_locations(source)
    namespace = dict(original.__func__.__globals__)
    exec(compile(source, "<exp2-instrumented-openpi-joint-forward>", "exec"), namespace)
    module._exp2_transform = transform
    module.forward = types.MethodType(namespace[original.__func__.__name__], module)
    return original, ast.unparse(source)


class Boundary:
    """Capture or project the output of PG5/11 on joint FM AND cached inference."""

    def __init__(self, model, layers=(5, 11)):
        self.model = model
        self.capture = False
        self.gradients = False
        self.basis = None
        self.projection_layer = 11
        self.edit = None
        self.cache, self.calls, self.realized = {}, {}, []
        self.valid = None
        self.visual = None
        self.original_embed = model.embed_prefix
        self.layers = set(layers)

        def embed(*args, **kwargs):
            result = self.original_embed(*args, **kwargs)
            self.valid = result[1].detach()
            lang = args[2] if len(args) > 2 else kwargs["lang_tokens"]
            self.visual = self.valid.clone()
            self.visual[:, result[0].shape[1] - lang.shape[1]:] = False
            return result

        model.embed_prefix = embed
        self.handles = []
        blocks = model.paligemma_with_expert.paligemma.language_model.layers
        for layer in layers:
            self.handles.append(blocks[layer].register_forward_hook(self._hook(layer)))
        self.original_joint, self.instrumented_source = instrument_joint_forward(
            model.paligemma_with_expert, self._joint_transform)

    def clear(self):
        self.cache, self.calls, self.realized = {}, {}, []

    def _joint_transform(self, stream, layer, hidden):
        return self._transform(layer, hidden) if stream == 0 and layer in self.layers else hidden

    def _transform(self, layer, hidden):
        self.calls[layer] = self.calls.get(layer, 0) + 1
        if self.gradients and not hidden.requires_grad:
            hidden = hidden.detach().requires_grad_(True)
        if self.capture:
            self.cache[layer] = hidden
            if hidden.requires_grad:
                hidden.retain_grad()
        if self.basis is not None and layer == self.projection_layer:
            projected = project(hidden, self.basis, self.visual)
            if not self.model.training:
                delta = projected.float() - hidden.float()
                self.realized.append({"delta_norm": float(delta.norm()),
                                      "hidden_norm": float(hidden.float().norm()),
                                      "visual_tokens": int(self.visual.sum())})
            hidden = projected
        if self.edit is not None and layer == self.projection_layer:
            hidden = self.edit(hidden)
        return hidden

    def _hook(self, layer):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            edited = self._transform(layer, hidden)
            return (edited, *output[1:]) if isinstance(output, tuple) else edited
        return hook

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.model.embed_prefix = self.original_embed
        self.model.paligemma_with_expert.forward = self.original_joint
        del self.model.paligemma_with_expert._exp2_transform


def project(hidden, basis, eligible):
    x = hidden.float()
    delta = (x @ basis.float()) @ basis.float().T
    return torch.where(eligible[..., None], (x - delta).to(hidden.dtype), hidden)
