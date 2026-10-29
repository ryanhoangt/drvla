"""ID-only scoring, attribution gates and paired rectification training on Slurm."""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch

from drvla.sae import load_sae, sae_path
from experiments.exp2.artifacts import save_scores
from experiments.exp2.common import CHECKPOINT, ROOT, SAES, destination, inventory, sha256, write_json
from experiments.exp2.model import Boundary, load_policy, tensors


class Samples:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.arrays = {}
        for filename in sorted({r["file"] for r in self.manifest["samples"]}):
            with np.load(self.directory / filename) as z:
                self.arrays[filename] = {key: z[key] for key in z.files}

    def rows(self, split):
        return [r for r in self.manifest["samples"] if r["split"] == split]

    def raw(self, row):
        return {key: value[row["row"]] for key, value in self.arrays[row["file"]].items()}


def flow_input(policy, data, row, seed, device="cuda"):
    observation, action = tensors(policy, data.raw(row), row["prompt"], device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(action.shape, generator=generator, device=device)
    # Beta(1.5, 1) via its inverse CDF, with the original OpenPI endpoint scaling.
    # Both noise and time are paired across prompts, arms, and actual ablations.
    timepoint = 0.001 + 0.999 * torch.rand(1, generator=generator, device=device).pow(2/3)
    return observation, action, noise, timepoint


def fm_loss(model, obs, action, noise, timepoint):
    # Match the original OpenPI objective, including its padded motor dimensions.
    return model(obs, action, noise=noise, time=timepoint).mean()


def score(policy, boundary, data, run, limit=0, device="cuda"):
    model = policy._model
    saes = {layer: load_sae(SAES, f"paligemma.layer_{layer}.output", device=device)
            for layer in (5, 11, 17)}
    for sae in saes.values():
        sae.requires_grad_(False)
    rows = data.rows("score")
    # Round-robin by task so a reduced smoke test is still suite-balanced.
    rows = sorted(rows, key=lambda r: (r["row"], r["task"]))
    if limit:
        rows = rows[:limit]
    records = {layer: [] for layer in saes}
    validations = []
    rng = np.random.default_rng(41)
    started = time.monotonic()
    for index, row in enumerate(rows):
        boundary.capture, boundary.gradients = True, True
        boundary.clear()
        obs, actions, noise, timepoint = flow_input(policy, data, row, 1000 + index, device=device)
        loss = fm_loss(model, obs, actions, noise, timepoint)
        loss.backward()
        originals = {}
        for layer, sae in saes.items():
            h = boundary.cache[layer]
            x = h.detach().float().mean(1)
            f = sae.encode(x)
            centered = x - sae.pre_bias
            scale = (centered - centered.mean(-1, keepdim=True)).norm(dim=-1, keepdim=True)
            if h.grad is None:
                if layer != 17:
                    raise RuntimeError(f"Missing action-loss gradient at PG{layer}")
                effect = torch.zeros_like(f)
            else:
                # Derivative of a broadcast coefficient intervention, holding the
                # recipient scale and error fixed. No encoder Jacobian included.
                effect = (h.grad.float().sum(1) @ sae.decoder.weight) * scale * f
            originals[layer] = (f.detach(), effect.detach(), x.detach())
        original_calls = dict(boundary.calls)
        if any(original_calls.get(layer) != 1 for layer in saes):
            raise RuntimeError(f"Unexpected FM boundary call counts: {original_calls}")
        boundary.capture, boundary.gradients = True, False
        candidates = data.manifest["counterfactual_candidates"][row["prompt"]]
        counterfactuals = rng.choice(candidates, size=2, replace=False).tolist()
        cf_features, differences = {layer: [] for layer in saes}, []
        with torch.no_grad():
            for prompt in counterfactuals:
                cf, _ = tensors(policy, data.raw(row), prompt, device=device)
                boundary.clear()
                fm_loss(model, cf, actions, noise, timepoint)
                for layer, sae in saes.items():
                    x = boundary.cache[layer].float().mean(1)
                    cf_features[layer].append(sae.encode(x))
                    if layer == 11:
                        # Pool valid visual tokens for a modality-matched PCA control.
                        h_cf = boundary.cache[layer].float()
                        differences.append((h_cf * boundary.visual[..., None]).sum(1)
                                           / boundary.visual.sum(1, keepdim=True))
        for layer, (f, effect, x) in originals.items():
            cf = torch.stack(cf_features[layer]).mean(0)
            sensitivity = ((f[None] - torch.stack(cf_features[layer])).clamp(min=0)
                           / (f[None] + 1e-8)).mean(0)
            records[layer].append({"f": f[0].cpu().numpy(), "cf": cf[0].cpu().numpy(),
                                   "effect": effect[0].cpu().numpy(),
                                   "sensitivity": sensitivity[0].cpu().numpy(), "pooled": x[0].cpu().numpy(),
                                   "task": row["task"], "episode": row["episode"],
                                   "frame": row["frame"]})
        # Check gradient attribution against actual small and full coefficient edits.
        if index < min(10, len(rows)):
            f, effect, _ = originals[11]
            j = int(effect.abs().argmax())
            sae = saes[11]
            boundary.capture = False
            for strength in (0.1, 1.0):
                def ablate(hidden, strength=strength):
                    pooled = hidden.float().mean(1)
                    centered = pooled - sae.pre_bias
                    scale = (centered - centered.mean(-1, keepdim=True)).norm(dim=-1, keepdim=True)
                    delta = -strength * scale * f[:, j:j+1] * sae.decoder.weight[:, j][None]
                    return hidden + delta[:, None].to(hidden.dtype)
                boundary.edit = ablate
                with torch.no_grad():
                    observed = float(fm_loss(model, obs, actions, noise, timepoint)) - float(loss.detach())
                validations.append({"sample": index, "feature": j, "strength": strength,
                                    "predicted_loss_delta": float(-strength * effect[0, j]),
                                    "actual_loss_delta": observed})
            boundary.edit = None
        # Recompute original visual mean without keeping an entire backward graph.
        boundary.capture, boundary.gradients = True, False
        with torch.no_grad():
            boundary.clear()
            fm_loss(model, obs, actions, noise, timepoint)
            h = boundary.cache[11].float()
            visual_mean = (h * boundary.visual[..., None]).sum(1) / boundary.visual.sum(1, keepdim=True)
            for cf_visual in differences:
                validations.append({"pca_difference": (visual_mean - cf_visual)[0].cpu().tolist()})
        del loss, originals, h
        boundary.clear()
        if index % 10 == 0 or index + 1 == len(rows):
            print(json.dumps({"phase": "score", "done": index+1, "total": len(rows),
                              "seconds": round(time.monotonic()-started, 1)}), flush=True)
    summary = {}
    bases = {}
    for layer, examples in records.items():
        values = {key: np.stack([r[key] for r in examples])
                  for key in ("f", "cf", "effect", "sensitivity", "pooled")}
        attribution = np.abs(values["effect"]).mean(0)
        sensitivity = values["sensitivity"].mean(0)
        literal = attribution * (1-sensitivity)
        # Conditioning avoids mistaking inactive examples for invariant features.
        active = values["f"] > 0
        conditional = (values["sensitivity"] * active).sum(0) / active.sum(0).clip(1)
        joint = (np.abs(values["effect"]) * (1-values["sensitivity"])).mean(0)
        selected = np.argsort(-literal, kind="stable")[:16] if attribution.any() else np.array([], dtype=int)
        summary[str(layer)] = {"feature_ids": selected.tolist(), "attribution": attribution[selected].tolist(),
                               "literal_sensitivity": sensitivity[selected].tolist(),
                               "active_conditional_sensitivity": conditional[selected].tolist(),
                               "score": literal[selected].tolist(),
                               "joint_score_top16": np.argsort(-joint)[:16].tolist(),
                               "zero_gradient": bool(np.all(attribution == 0))}
        save_scores(run / f"score_pg{layer}.npz", values, examples, attribution,
                    sensitivity, literal, joint)
        if layer == 11:
            if len(selected) != 16 or np.count_nonzero(attribution) < 32:
                raise RuntimeError("Too few action-relevant latents for a 16-direction controlled pilot")
            decoder = saes[layer].decoder.weight.detach().float()
            bases["sae_m"] = torch.linalg.qr(decoder[:, selected]).Q.cpu()
            pool = np.setdiff1d(np.flatnonzero(attribution > 0), selected)
            # Activity/frequency matching without access to OOD outcomes.
            descriptors = np.stack([np.log(values["f"].mean(0)+1e-8),
                                    np.log(active.mean(0)+1e-8)], axis=1)
            descriptors /= descriptors.std(0).clip(1e-8)
            random_ids = []
            for j in selected:
                distance = ((descriptors[pool]-descriptors[j])**2).sum(1)
                nearest = np.argsort(distance)[:min(8, len(pool))]
                k = int(rng.choice(nearest))
                random_ids.append(int(pool[k]))
                pool = np.delete(pool, k)
            bases["random_sae"] = torch.linalg.qr(decoder[:, random_ids]).Q.cpu()
            summary[str(layer)]["random_control_ids"] = random_ids
    differences = np.array([v["pca_difference"] for v in validations if "pca_difference" in v])
    # Centered PCA of paired visual-token prompt differences. It is explicitly
    # an instruction-sensitive comparator, not an identified shortcut subspace.
    _, singular, vh = np.linalg.svd(differences - differences.mean(0), full_matrices=False)
    bases["pca_promptdiff"] = torch.tensor(vh[:16].T.copy(), dtype=torch.float32)
    for name, basis in bases.items():
        torch.testing.assert_close(basis.T @ basis, torch.eye(basis.shape[1]), atol=2e-5, rtol=2e-5)
    torch.save(bases, run / "bases.pt")
    summary.update({"examples": len(rows), "counterfactuals_per_example": 2,
                    "pca_singular_values": singular[:20].tolist(),
                    "sae_sha256": {str(layer): sha256(sae_path(SAES, f"paligemma.layer_{layer}.output"))
                                   for layer in saes},
                    "attribution_checks": [v for v in validations if "pca_difference" not in v],
                    "estimator": "Pooled coefficient derivative: scale * f_j * sum_token(grad_h) dot d_j; "
                                 "fixed recipient normalization and reconstruction error.",
                    "selection": "Literal requested M=A*(1-mean_S); conditional/joint alternatives diagnostic only.",
                    "warning": "PG17 output has zero action-loss gradient by architecture. "
                               "Prompt-invariant action-relevant is a candidate, not identified memorization."})
    write_json(run / "scoring.json", summary)
    boundary.clear()
    boundary.capture = boundary.gradients = False
    gc.collect()
    torch.cuda.empty_cache()
    return bases


def fit_probes(run):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    for layer in (5, 11, 17):
        with np.load(run / f"score_pg{layer}.npz") as z:
            x, y = z["pooled"], z["task"]
        scale = StandardScaler().fit(x)
        probe = LogisticRegression(C=1.0, max_iter=1500).fit(scale.transform(x), y)
        np.savez(run / f"probe_pg{layer}.npz", mean=scale.mean_, scale=scale.scale_,
                 coef=probe.coef_, intercept=probe.intercept_, classes=probe.classes_)


@torch.no_grad()
def validation(policy, boundary, data, rows, run, name, device="cuda"):
    model = policy._model
    model.eval()
    total, pooled = [], []
    boundary.capture = True
    for i, row in enumerate(rows):
        inputs = flow_input(policy, data, row, 9000+i, device=device)
        boundary.clear()
        total.append(float(fm_loss(model, *inputs)))
        pooled.append(boundary.cache[11].float().mean(1)[0].cpu().numpy())
    pooled = np.stack(pooled)
    np.savez_compressed(run / f"validation_{name}.npz", loss=np.array(total), pooled=pooled,
                        task=np.array([r["task"] for r in rows]),
                        episode=np.array([r["episode"] for r in rows]))
    with np.load(run / "probe_pg11.npz") as probe:
        logits = ((pooled-probe["mean"])/probe["scale"]) @ probe["coef"].T + probe["intercept"]
        accuracy = float((probe["classes"][logits.argmax(1)] == np.array([r["task"] for r in rows])).mean())
    boundary.capture = False
    return {"mean_loss": float(np.mean(total)), "per_example": total, "frozen_probe_accuracy": accuracy}


def train(policy, boundary, data, run, bases, steps, validation_limit, arms, device="cuda"):
    import safetensors.torch
    from openpi.training.config import get_config

    model = policy._model
    first = data.rows("score")[0]
    raw = data.raw(first)
    observation = {"observation/image": raw["image"], "observation/wrist_image": raw["wrist"],
                   "observation/state": raw["state"], "prompt": first["prompt"]}
    fixed_noise = np.random.default_rng(0).standard_normal((10, 32), dtype=np.float32)
    original_action = policy.infer(observation, noise=fixed_noise)["actions"]
    model.requires_grad_(True)
    parameters = list(model.parameters())
    cfg = get_config("pi05_libero")
    warmup = min(25, steps)
    write_json(run / "full_sft.json", {"method": "full-parameter SFT; no adapters",
               "trainable_parameters": sum(p.numel() for p in parameters),
               "precision": cfg.pytorch_training_precision, "betas": [cfg.optimizer.b1, cfg.optimizer.b2],
               "eps": cfg.optimizer.eps, "weight_decay": cfg.optimizer.weight_decay,
               "clip_norm": cfg.optimizer.clip_gradient_norm, "learning_rate": 5e-5,
               "warmup_steps": warmup, "effective_batch": 2,
               "native_reference": "openpi/scripts/train_pytorch.py; native PI0Pytorch.forward and AdamW. "
                                   "Exp2 uses its episode-balanced ID subset, paired random streams, "
                                   "shortened rectification warmup and explicit projection instrumentation."})
    rows = data.rows("train")
    valid = sorted(data.rows("validation"), key=lambda r: (r["row"], r["task"]))
    if validation_limit:
        valid = valid[:validation_limit]
    boundary.basis = None
    baseline = validation(policy, boundary, data, valid, run, "frozen", device=device)
    results = {"frozen": baseline}
    write_json(run / "validation.json", results)
    rng = np.random.default_rng(2026)
    indices = np.concatenate([rng.permutation(len(rows)) for _ in range(1+2*steps//len(rows))])[:2*steps]
    model.gradient_checkpointing_enable()
    reset_gates = {}
    for arm in arms:
        boundary.basis = None
        safetensors.torch.load_model(model, CHECKPOINT / "model.safetensors", device="cpu")
        model.eval()
        restored = policy.infer(observation, noise=fixed_noise)["actions"]
        np.testing.assert_array_equal(original_action, restored, err_msg="Full-SFT checkpoint reset: "+arm)
        reset_gates[arm] = {"reset_action_max_abs_delta": 0.0}
        write_json(run / "full_sft_reset_gates.json", reset_gates)
        optimizer = torch.optim.AdamW(parameters, lr=5e-5, betas=(cfg.optimizer.b1, cfg.optimizer.b2),
                                      eps=cfg.optimizer.eps, weight_decay=cfg.optimizer.weight_decay)
        boundary.basis = bases[arm].to(device) if arm in bases else None
        model.train()
        history = []
        started = time.monotonic()
        for step in range(steps):
            lr = 5e-5 * min(1.0, (step+1)/max(1, warmup))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for micro in range(2):
                row = rows[int(indices[2*step+micro])]
                torch.manual_seed(300000 + 2*step+micro)
                inputs = flow_input(policy, data, row, 300000 + 2*step+micro, device=device)
                boundary.clear()
                loss = fm_loss(model, *inputs)
                if not bool(torch.isfinite(loss)):
                    raise ValueError(f"Nonfinite {arm} loss at step {step}")
                (loss/2).backward()
                losses.append(float(loss.detach()))
                # Residual callbacks run during recomputation too: an invocation
                # is required, unlike old decoder-block hooks bypassed in training.
                if not boundary.calls.get(11):
                    raise RuntimeError("Projection boundary did not execute during training")
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if not bool(torch.isfinite(norm)):
                raise ValueError("Nonfinite full-parameter gradient")
            optimizer.step()
            history.append({"step": step+1, "loss": float(np.mean(losses)), "grad_norm": float(norm), "lr": lr})
            if step % 25 == 0 or step+1 == steps:
                print(json.dumps({"phase": "train", "arm": arm, **history[-1],
                                  "seconds": round(time.monotonic()-started, 1),
                                  "peak_gpu_gib": torch.cuda.max_memory_allocated()/1024**3}), flush=True)
                write_json(run / f"training_{arm}.json", history)
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
        # Primary evaluation always removes the projection.
        boundary.basis = None
        results[arm] = {"without_projection": validation(policy, boundary, data, valid, run, arm, device=device)}
        if arm in bases:
            boundary.basis = bases[arm].to(device)
            results[arm]["with_projection_diagnostic"] = validation(policy, boundary, data, valid, run,
                                                                  arm+"_projected", device=device)
            boundary.basis = None
        checkpoint = run / f"checkpoint_{arm}"
        checkpoint.mkdir()
        safetensors.torch.save_model(model, checkpoint / "model.safetensors")
        write_json(checkpoint / "metadata.json", {"method": "full_sft", "arm": arm, "steps": steps,
                   "projection_layer": 11, "source_manifest_sha256": sha256(run / "manifest.json")})
        write_json(run / "validation.json", results)
        gc.collect()
        torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--score-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=30)
    parser.add_argument("--arms", nargs="+", choices=("sft", "sae_m", "random_sae", "pca_promptdiff"),
                        default=["sft", "sae_m", "random_sae", "pca_promptdiff"])
    parser.add_argument("--smoke-run", type=Path)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Run only inside a new Slurm allocation with exactly one assigned GPU")
    if os.getenv("SLURM_JOB_PARTITION") != "main":
        raise RuntimeError("User requested main only; do not run Exp2 on MIG")
    if args.smoke_run:
        smoke = json.loads((args.smoke_run / "status.json").read_text())
        full_sft = json.loads((args.smoke_run / "full_sft.json").read_text())
        audit = json.loads((args.smoke_run / "protected_artifacts_audit.json").read_text())
        if smoke["status"] != "training_complete" or not audit["unchanged"] or "full-parameter" not in full_sft["method"]:
            raise RuntimeError("Full-SFT smoke gates did not complete")
    torch.set_num_threads(4)
    print(json.dumps({"phase": "startup", "job_id": os.environ["SLURM_JOB_ID"],
                      "gpu": torch.cuda.get_device_name(), "steps": args.steps, "arms": args.arms}), flush=True)
    run = destination(args.run_dir)
    run.mkdir(parents=True, exist_ok=False)
    snapshot = run / "source"
    snapshot.mkdir()
    for path in sorted((ROOT / "experiments/exp2").glob("*")):
        if path.suffix in (".py", ".md", ".slurm"):
            shutil.copyfile(path, snapshot / path.name)
    write_json(run / "manifest.json", {"arguments": vars(args) | {"run_dir": str(args.run_dir),
               "data_dir": str(args.data_dir), "smoke_run": str(args.smoke_run) if args.smoke_run else None},
               "job_id": os.environ["SLURM_JOB_ID"],
               "partition": os.getenv("SLURM_JOB_PARTITION"), "host": socket.gethostname(),
               "gpu": torch.cuda.get_device_name(),
               "gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory/1024**3,
               "cpu_affinity": sorted(os.sched_getaffinity(0)),
               "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
               "data_manifest_sha256": sha256(args.data_dir / "manifest.json"),
               "source_sha256": {p.name: sha256(p) for p in snapshot.iterdir()}})
    before = inventory()
    write_json(run / "protected_before.json", before)
    try:
        data = Samples(args.data_dir)
        policy = load_policy()
        first = data.rows("score")[0]
        raw = data.raw(first)
        obs = {"observation/image": raw["image"], "observation/wrist_image": raw["wrist"],
               "observation/state": raw["state"], "prompt": first["prompt"]}
        noise = np.random.default_rng(0).standard_normal((10, 32), dtype=np.float32)
        unhooked = policy.infer(obs, noise=noise)["actions"]
        fm_inputs = flow_input(policy, data, first, 1000)
        print(json.dumps({"phase": "native_fm_gate", "action_dtype": str(fm_inputs[1].dtype),
                          "noise_dtype": str(fm_inputs[2].dtype), "time_dtype": str(fm_inputs[3].dtype)}), flush=True)
        with torch.no_grad():
            original_loss = fm_loss(policy._model, *fm_inputs)
        boundary = Boundary(policy._model, (5, 11, 17))
        (snapshot / "instrumented_joint_forward.py").write_text(boundary.instrumented_source + "\n")
        with torch.no_grad():
            instrumented_loss = fm_loss(policy._model, *fm_inputs)
        torch.testing.assert_close(original_loss, instrumented_loss, atol=0, rtol=0,
                                   msg="Passive instrumentation changed native OpenPI FM loss")
        hooked = policy.infer(obs, noise=noise)["actions"]
        repeated = policy.infer(obs, noise=noise)["actions"]
        np.testing.assert_array_equal(unhooked, hooked, err_msg="Passive boundary hook changed actions")
        np.testing.assert_array_equal(hooked, repeated, err_msg="Repeated fixed-noise inference differs")
        write_json(run / "identity_gates.json", {"passive_hook_max_abs": 0.0, "repeat_max_abs": 0.0,
                    "native_vs_instrumented_FM_loss_delta": 0.0,
                    "instrumented_joint_forward_sha256": sha256(snapshot / "instrumented_joint_forward.py")})
        print(json.dumps({"phase": "identity_gates", "status": "passed"}), flush=True)
        bases = score(policy, boundary, data, run, args.score_limit)
        fit_probes(run)
        results = train(policy, boundary, data, run, bases, args.steps, args.validation_limit, args.arms)
        write_json(run / "status.json", {"status": "training_complete", "steps": args.steps,
                                         "validation": results, "rollouts_complete": False})
        print(f"Training complete: {run}", flush=True)
    finally:
        after = inventory()
        changed = [p for p in set(before)|set(after) if before.get(p) != after.get(p)]
        write_json(run / "protected_artifacts_audit.json", {"checked_files": len(before), "changed": changed,
                                                          "unchanged": not changed})
        if changed:
            raise RuntimeError(f"Protected artifacts changed: {changed[:10]}")


if __name__ == "__main__":
    main()
