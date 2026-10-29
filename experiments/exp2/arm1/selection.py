"""RQ1 task sets filtered by training-only, visual-token action attribution."""

import json

import numpy as np
import torch

from drvla.sae import load_sae, sae_path
from drvla.store import ActivationStore
from experiments.exp2.common import ROOT, SAES, sha256, write_json
from experiments.exp2.pilot import flow_input, fm_loss


@torch.no_grad()
def task_statistics(data, saes, device):
    """Reproduce RQ1's episode-level Cohen d within libero_object only.

    These are policy-training demonstrations, including the ID validation
    episodes used in Exp2. Validation is held out from gradient updates and
    action-attribution scoring, not from the historical RQ1 task dictionary.
    """
    store = ActivationStore(ROOT / "activations/pi05_libero_server")
    prompts = data.manifest["prompts"]
    means = {layer: [] for layer in saes}
    labels, episodes = [], []
    for episode in store.episode_ids:
        task = store.episode(episode)["task"]
        if task not in prompts:
            continue
        labels.append(prompts.index(task))
        episodes.append(episode)
        for layer, sae in saes.items():
            x = np.array(store.activations(f"paligemma.layer_{layer}.output", episode), copy=True)
            features = sae.encode(torch.as_tensor(x, device=device, dtype=torch.float32))
            means[layer].append(features.mean(0).cpu().numpy())
    labels = np.array(labels)
    result = {}
    for layer in saes:
        x = np.stack(means[layer])
        d = []
        for k in range(len(prompts)):
            a, b = x[labels == k], x[labels != k]
            if min(len(a), len(b)) < 2:
                raise ValueError("Insufficient task episodes")
            d.append((a.mean(0)-b.mean(0))/np.sqrt(.5*(a.var(0)+b.var(0))+1e-8))
        result[layer] = np.stack(d)
    return result, episodes


def orthogonal(decoder, indices):
    matrix = decoder[:, indices].float()
    singular = torch.linalg.svdvals(matrix)
    rank = int((singular > singular[0]*1e-6).sum())
    if rank != len(indices):
        raise ValueError(f"Decoder union rank {rank} differs from count {len(indices)}")
    basis = torch.linalg.qr(matrix).Q
    # OpenPI sets float32 matmul precision="high". On H100, the diagnostic
    # Q.T@Q then uses TF32 even though cuSOLVER produced a valid FP32 QR.
    # Verify the stored FP32 vectors with a FP64 Gram matrix; keep the same
    # tolerance and leave native policy/projection precision unchanged.
    exact_gram = basis.double().T @ basis.double()
    torch.testing.assert_close(exact_gram, torch.eye(rank, device=matrix.device, dtype=torch.float64),
                               atol=2e-5, rtol=2e-5)
    print(json.dumps({"phase": "QR_gate", "rank": rank,
                      "FP64_gram_max_error": float((exact_gram-torch.eye(rank, device=matrix.device)).abs().max()),
                      "native_FP32_gram_max_error": float((basis.T@basis-torch.eye(rank, device=matrix.device)).abs().max()),
                      "native_matmul_precision": torch.get_float32_matmul_precision()}), flush=True)
    return basis.cpu(), singular.cpu().tolist()


def select(d, f, effect, task, decoder, rng):
    """Top8 by d*A among positive-d, active members of each RQ1 top16."""
    per_task, details = [], []
    for k in range(d.shape[0]):
        a = np.abs(effect[task == k]).mean(0)
        top = np.argsort(-d[k], kind="stable")[:16]
        eligible = top[(d[k, top] > 0) & (a[top] > 0)]
        chosen = eligible[np.argsort(-(d[k, eligible]*a[eligible]), kind="stable")[:8]]
        if len(chosen) != 8:
            raise ValueError(f"Task {k} has only {len(chosen)} active positive-d features")
        per_task.append(chosen.tolist())
        details.append({"task": k, "rq1_top16": top.tolist(), "selected": chosen.tolist(),
                        "overlap_with_rq1": len(set(chosen)&set(top)),
                        "d": d[k, chosen].tolist(), "A_visual": a[chosen].tolist(),
                        "d_times_A": (d[k, chosen]*a[chosen]).tolist()})
    union = sorted({j for ids in per_task for j in ids})
    global_a = np.abs(effect).mean(0)
    pool = np.setdiff1d(np.flatnonzero(global_a > 0), union)
    # A bijective replacement preserves the exact task-set overlap structure,
    # union count and rank. Match A, activation magnitude, and active frequency.
    descriptors = np.stack([np.log(global_a+1e-12), np.log(f.mean(0)+1e-8),
                            np.log((f > 0).mean(0)+1e-8)], 1)
    descriptors /= descriptors.std(0).clip(1e-8)
    mapping = {}
    for j in union:
        distances = ((descriptors[pool]-descriptors[j])**2).sum(1)
        nearest = np.argsort(distances, kind="stable")[:min(8, len(pool))]
        if not len(nearest):
            raise ValueError("Insufficient active control features")
        index = int(rng.choice(nearest))
        mapping[j] = int(pool[index])
        pool = np.delete(pool, index)
    random_ids = [mapping[j] for j in union]
    basis, singular = orthogonal(decoder, union)
    control, random_singular = orthogonal(decoder, random_ids)
    return basis, control, {"tasks": details, "union": union, "rank": basis.shape[1],
        "random_union": random_ids, "random_task_sets": [[mapping[j] for j in ids] for ids in per_task],
        "singular_values": singular, "random_singular_values": random_singular,
        "mean_attribution_target": float(global_a[union].mean()),
        "mean_attribution_random": float(global_a[random_ids].mean()),
        "mean_active_frequency_target": float((f[:, union] > 0).mean()),
        "mean_active_frequency_random": float((f[:, random_ids] > 0).mean()),
        "mean_activation_target": float(f[:, union].mean()),
        "mean_activation_random": float(f[:, random_ids].mean())}


def build(policy, boundary, data, run, device="cuda", statistics=None, score_limit=0):
    saes = {layer: load_sae(SAES, f"paligemma.layer_{layer}.output", device=device).requires_grad_(False)
            for layer in (5, 11)}
    if statistics is None:
        statistics, episodes = task_statistics(data, saes, device)
    else:
        episodes = []  # Synthetic software tests only.
    rows = sorted(data.rows("score"), key=lambda r: (r["row"], r["task"]))
    if score_limit:
        rows = rows[:score_limit]
    records = {layer: [] for layer in saes}
    checks = []
    for index, row in enumerate(rows):
        boundary.capture = boundary.gradients = True
        boundary.clear()
        inputs = flow_input(policy, data, row, 1000+index, device=device)
        loss = fm_loss(policy._model, *inputs)
        loss.backward()
        for layer, sae in saes.items():
            h = boundary.pre_cache[layer]
            if h.grad is None:
                raise RuntimeError(f"No action gradient at PG{layer}")
            x = h.detach().float().mean(1)
            f = sae.encode(x)
            centered = x-sae.pre_bias
            scale = (centered-centered.mean(-1, keepdim=True)).norm(dim=-1, keepdim=True)
            visual_grad = (h.grad.float()*boundary.visual[..., None]).sum(1)
            effect = (visual_grad @ sae.decoder.weight) * scale * f
            records[layer].append({"f": f[0].cpu().numpy(), "effect": effect.detach()[0].cpu().numpy()})
            if index < 10:
                j = int(effect.abs().argmax())
                delta = (-.1 * scale * f[:, j:j+1] * sae.decoder.weight[:, j][None]).detach()
                visual = boundary.visual.clone()
                boundary.projection_layer = layer
                boundary.edit = lambda hidden, delta=delta, visual=visual: torch.where(
                    visual[..., None], hidden+delta[:, None].to(hidden.dtype), hidden)
                boundary.capture = boundary.gradients = False
                with torch.no_grad():
                    actual = float(fm_loss(policy._model, *inputs))-float(loss.detach())
                checks.append({"sample": index, "layer": layer, "feature": j,
                               "predicted_delta": float(-.1*effect[0, j]), "actual_delta": actual})
                boundary.edit = None
        del loss
        boundary.clear()
        if index % 10 == 0 or index+1 == len(rows):
            print(json.dumps({"phase": "visual_attribution", "done": index+1, "total": len(rows)}), flush=True)
    task = np.array([r["task"] for r in rows])
    bases = {"acttask": {}, "random_acttask": {}}
    summary = {"method": "positive Cohen d * within-task abs visual-token coefficient action attribution",
        "attribution": "Pooled SAE coefficient, fixed recipient scale/error; broadcast edit only on valid visual tokens.",
        "rq1_episodes": episodes, "score_examples": len(rows), "finite_difference_checks": checks,
        "validation_caveat": "RQ1 d uses policy-training ID episodes, including Exp2 ID validation episodes. "
                             "Those are excluded from action scoring and parameter updates.",
        "layers": {}}
    rng = np.random.default_rng(20261019)
    for layer, samples in records.items():
        f = np.stack([s["f"] for s in samples])
        effect = np.stack([s["effect"] for s in samples])
        # Persist costly training-only scores before later numerical gates.
        np.savez_compressed(run / f"selection_pg{layer}.npz", d=statistics[layer], f=f, effect=effect, task=task,
                            episode=np.array([r["episode"] for r in rows]))
        basis, random, info = select(statistics[layer], f, effect, task, saes[layer].decoder.weight.detach(), rng)
        bases["acttask"][layer], bases["random_acttask"][layer] = basis, random
        info["sae_sha256"] = sha256(sae_path(SAES, f"paligemma.layer_{layer}.output"))
        summary["layers"][str(layer)] = info
    write_json(run / "selection.json", summary)
    torch.save(bases, run / "bases.pt")
    boundary.capture = boundary.gradients = False
    boundary.clear()
    return bases
