"""Arm 1: paired PG-only SFT, action-filtered projection and matched control."""

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import safetensors.torch
import torch

from drvla.sae import load_sae
from experiments.exp2.common import CHECKPOINT, ROOT, SAES, destination, sha256, write_json
from experiments.exp2.model import load_policy, tensors
from experiments.exp2.pilot import Samples, flow_input, fm_loss
from experiments.exp2.arm1.common import REFERENCE, protected
from experiments.exp2.arm1.model import TaskBoundary, frozen_digest, pg_only
from experiments.exp2.arm1.selection import build


@torch.no_grad()
def validation(policy, boundary, data, run, name, rows, selection, device="cuda", reference=REFERENCE):
    policy._model.eval()
    boundary.capture, boundary.gradients = True, False
    saes = {layer: load_sae(SAES, f"paligemma.layer_{layer}.output", device=device).requires_grad_(False)
            for layer in (5, 11)}
    with np.load(reference / "probe_pg11.npz") as z:
        probe = {k: z[k] for k in z.files}

    def prediction(x):
        logits = ((x-probe["mean"])/probe["scale"]) @ probe["coef"].T + probe["intercept"]
        prob = np.exp(logits-logits.max())
        return int(probe["classes"][logits.argmax()]), prob/prob.sum()

    samples = []
    arrays = {"loss": [], "pooled": [], "cf_pooled": [], "task": [], "cf_task": []}
    for i, row in enumerate(rows):
        inputs = flow_input(policy, data, row, 9000+i, device=device)
        boundary.clear()
        loss = float(fm_loss(policy._model, *inputs))
        pooled = boundary.cache[11].float().mean(1)[0].cpu().numpy()
        original, p_id = prediction(pooled)
        energies = list(boundary.realized)
        id_f = {layer: sae.encode(boundary.cache[layer].float().mean(1))[0].cpu().numpy()
                for layer, sae in saes.items()}
        candidates = data.manifest["counterfactual_candidates"][row["prompt"]]
        prompt = candidates[i % len(candidates)]
        cf_task = data.manifest["prompts"].index(prompt)
        cf, _ = tensors(policy, data.raw(row), prompt, device=device)
        boundary.clear()
        fm_loss(policy._model, cf, *inputs[1:])  # Representation extraction; never train on this loss.
        cf_pooled = boundary.cache[11].float().mean(1)[0].cpu().numpy()
        predicted, p_cf = prediction(cf_pooled)
        persistence = {}
        for layer, sae in saes.items():
            cf_f = sae.encode(boundary.cache[layer].float().mean(1))[0].cpu().numpy()
            ids = selection["layers"][str(layer)]["tasks"][row["task"]]["selected"]
            with np.load(run / f"selection_pg{layer}.npz") as scores:
                denominator = scores["f"][scores["task"] == row["task"]][:, ids].sum(1).mean()
            persistence[str(layer)] = {"cf_over_frozen_train_ID": float(cf_f[ids].sum()/max(denominator, 1e-8)),
                                      "ID_over_frozen_train_ID": float(id_f[layer][ids].sum()/max(denominator, 1e-8))}
        sample = {"task": row["task"], "cf_task": cf_task, "episode": row["episode"], "frame": row["frame"],
                  "loss": loss, "id_probe_correct": original == row["task"],
                  "cf_probe_new_correct": predicted == cf_task, "cf_probe_old_correct": predicted == row["task"],
                  "cf_p_new": float(p_cf[cf_task]), "cf_p_old": float(p_cf[row["task"]]),
                  "id_p_old": float(p_id[row["task"]]), "persistence": persistence, "ID_projection": energies}
        samples.append(sample)
        for key, value in {"loss": loss, "pooled": pooled, "cf_pooled": cf_pooled,
                           "task": row["task"], "cf_task": cf_task}.items():
            arrays[key].append(value)
    for value in arrays.values():
        if not np.isfinite(np.array(value)).all():
            raise ValueError("Nonfinite validation")
    np.savez_compressed(run / f"validation_{name}.npz", **{k: np.array(v) for k, v in arrays.items()})
    summary = {"n": len(rows), "ID_FM": float(np.mean(arrays["loss"])),
        "ID_PG11_probe_accuracy": float(np.mean([s["id_probe_correct"] for s in samples])),
        "CF_PG11_new_task_accuracy": float(np.mean([s["cf_probe_new_correct"] for s in samples])),
        "CF_PG11_old_task_accuracy": float(np.mean([s["cf_probe_old_correct"] for s in samples])),
        "CF_S_orig": {str(layer): float(np.mean([s["persistence"][str(layer)]["cf_over_frozen_train_ID"]
                                                 for s in samples])) for layer in saes},
        "projection_energy": {str(layer): float(np.mean([e["removed_energy_fraction"] for s in samples
            for e in s["ID_projection"] if e["layer"] == layer])) for layer in boundary.bases}}
    write_json(run / f"validation_details_{name}.json", samples)
    boundary.clear()
    boundary.capture = False
    return summary


def train(policy, boundary, data, run, bases, steps=500, validation_limit=120, device="cuda", reference=REFERENCE):
    model = policy._model
    names = pg_only(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    initial_frozen = frozen_digest(model)
    write_json(run / "training_config.json", {"method": "PG0..PG16 SFT, no adapters or auxiliary loss",
        "trainable_names": names, "trainable_parameters": sum(p.numel() for p in parameters),
        "frozen_sha256_before": initial_frozen, "layers_projected": [5, 11],
        "steps": steps, "lr": 5e-5, "warmup": min(25, steps), "microbatch": 1, "accumulation": 2,
        "betas": [.9, .95], "eps": 1e-8, "weight_decay": 1e-10, "clip_norm": 1.,
        "same_data_order_and_FM_noise_all_arms": True, "no_grad_on_frozen_modules": False})
    selection = json.loads((run / "selection.json").read_text())
    rows = data.rows("train")
    valid = sorted(data.rows("validation"), key=lambda r: (r["row"], r["task"]))[:validation_limit]
    results = {}

    def check(name):
        results[name] = validation(policy, boundary, data, run, name, valid, selection, device, reference)
        write_json(run / "validation.json", results)
        print(json.dumps({"phase": "validation", "name": name, **results[name]}), flush=True)

    boundary.bases = {}
    check("frozen_off")
    for arm in bases:
        boundary.bases = {layer: value.to(device) for layer, value in bases[arm].items()}
        check("frozen_"+arm+"_on")
    boundary.bases = {}
    rng = np.random.default_rng(2026)
    indices = np.concatenate([rng.permutation(len(rows)) for _ in range(1+2*steps//len(rows))])[:2*steps]
    model.gradient_checkpointing_enable()
    gates = {}
    for arm in ("sft_pg_only", "acttask", "random_acttask"):
        boundary.bases = {}
        safetensors.torch.load_model(model, CHECKPOINT / "model.safetensors", device="cpu")
        assert frozen_digest(model) == initial_frozen
        boundary.bases = {layer: value.to(device) for layer, value in bases.get(arm, {}).items()}
        optimizer = torch.optim.AdamW(parameters, lr=5e-5, betas=(.9, .95), eps=1e-8, weight_decay=1e-10)
        model.train()
        history = []
        started = time.monotonic()
        gradient_gate = False
        for step in range(steps):
            lr = 5e-5*min(1., (step+1)/max(1, min(25, steps)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for micro in range(2):
                seed = 300000+2*step+micro
                torch.manual_seed(seed)
                row = rows[int(indices[2*step+micro])]
                boundary.clear()
                loss = fm_loss(model, *flow_input(policy, data, row, seed, device=device))
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite {arm} step {step}")
                (loss/2).backward()
                losses.append(float(loss.detach()))
                if not all(boundary.calls.get(layer, 0) >= 1 for layer in (5, 11)):
                    raise RuntimeError("PG projection boundary bypassed")
            if step == 0:
                blocks = model.paligemma_with_expert.paligemma.language_model.layers
                if not any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in blocks[0].parameters()):
                    raise RuntimeError("Action loss did not backpropagate through frozen modules to PG0")
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                    raise RuntimeError("Frozen weights received gradients")
                gradient_gate = True
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not torch.isfinite(norm):
                raise ValueError("Nonfinite gradient")
            optimizer.step()
            history.append({"step": step+1, "loss": float(np.mean(losses)), "grad_norm": float(norm), "lr": lr})
            if step % 25 == 0 or step+1 == steps:
                write_json(run / f"training_{arm}.json", history)
                print(json.dumps({"phase": "train", "arm": arm, **history[-1],
                                  "seconds": round(time.monotonic()-started, 1)}), flush=True)
        optimizer.zero_grad(set_to_none=True)
        del optimizer, loss
        gc.collect()
        torch.cuda.empty_cache()
        after = frozen_digest(model)
        if after != initial_frozen:
            raise RuntimeError("AE/PG17/vision/embedding/motor weights changed")
        gates[arm] = {"frozen_weights_exact": True, "frozen_sha256_after": after,
                      "action_gradient_reaches_PG0": gradient_gate}
        write_json(run / "frozen_weight_gates.json", gates)
        boundary.bases = {}
        check(arm+"_off")
        if arm in bases:
            boundary.bases = {layer: value.to(device) for layer, value in bases[arm].items()}
            check(arm+"_on")
            boundary.bases = {}
        checkpoint = run / f"checkpoint_{arm}"
        checkpoint.mkdir()
        safetensors.torch.save_model(model, checkpoint / "model.safetensors")
        write_json(checkpoint / "metadata.json", {"arm": arm, "steps": steps, "trainable": "PG0..PG16",
                                                 "source_manifest_sha256": sha256(run / "manifest.json")})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "workspace/exp2/data_v1")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--validation-limit", type=int, default=120)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID") or os.getenv("SLURM_JOB_PARTITION") != "main":
        raise RuntimeError("Use a new Slurm main GPU job")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one assigned GPU")
    torch.set_num_threads(4)
    run = destination(args.run_dir)
    run.mkdir(parents=True, exist_ok=False)
    snapshot = run / "source"
    shutil.copytree(ROOT / "experiments/exp2/arm1", snapshot, ignore=shutil.ignore_patterns("__pycache__"))
    write_json(run / "manifest.json", {"arguments": {"run_dir": str(run), "data_dir": str(args.data_dir),
               "steps": args.steps, "validation_limit": args.validation_limit}, "job": os.environ["SLURM_JOB_ID"],
               "partition": os.environ["SLURM_JOB_PARTITION"], "gpu": torch.cuda.get_device_name(),
               "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"), "reference": str(REFERENCE),
               "data_sha256": sha256(args.data_dir / "manifest.json"),
               "source_sha256": {p.name: sha256(p) for p in snapshot.iterdir() if p.is_file()}})
    before = protected()
    write_json(run / "protected_before.json", before)
    try:
        data, policy = Samples(args.data_dir), load_policy()
        inputs = flow_input(policy, data, data.rows("score")[0], 1000)
        with torch.no_grad():
            native = fm_loss(policy._model, *inputs)
        boundary = TaskBoundary(policy._model)
        with torch.no_grad():
            passive = fm_loss(policy._model, *inputs)
        torch.testing.assert_close(native, passive, rtol=0, atol=0)
        write_json(run / "identity_gates.json", {"native_vs_passive_FM_exact": True})
        bases = build(policy, boundary, data, run)
        results = train(policy, boundary, data, run, bases, args.steps, args.validation_limit)
        boundary.close()
        write_json(run / "status.json", {"status": "training_complete", "validation": results})
    except Exception as error:
        write_json(run / "status.json", {"status": "failed", "error_type": type(error).__name__,
                                         "error": str(error), "rollouts_complete": False})
        raise
    finally:
        after = protected()
        changed = [p for p in set(before)|set(after) if before.get(p) != after.get(p)]
        write_json(run / "protected_artifacts_audit.json", {"unchanged": not changed, "changed": changed,
                                                           "checked_files": len(before)})
        if changed:
            raise RuntimeError(f"Previous artifacts changed: {changed[:10]}")


if __name__ == "__main__":
    main()
