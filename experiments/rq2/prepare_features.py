"""Freeze training-only task feature sets and matched control sets for RQ2.

Reuses the RQ1 Cohen's-d selection, but saves explicit feature IDs and input
hashes. Existing activations, SAEs, index, and RQ1 outputs are read-only inputs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from drvla.classifier import GeneralityClassifier
from drvla.index import FeatureIndex
from drvla.sae import load_sae, sae_path
from drvla.store import ActivationStore
from experiments.rq1.analyze_rq1 import BASE_SUITES, instructed_task, task_language
from experiments.rq1.task_features import select_features
from experiments.rq1.task_meta import BDDL_ROOT, suite_meta
from experiments.rq2.common import ROOT, manifest, output_path, sha256, write_json


def matched_features(target, candidates, mean, frequency, rng):
    """Match ID mean activation and frequency, without looking at rollout outcomes."""
    candidates = np.asarray(candidates, dtype=int)
    if len(candidates) < len(target):
        raise ValueError(f"Need {len(target)} control features, have {len(candidates)}")
    descriptors = np.stack([np.log(mean + 1e-8), np.log(frequency + 1e-8)], axis=1)
    descriptors /= descriptors.std(axis=0).clip(1e-8)
    available = set(candidates.tolist())
    selected = []
    for feature in sorted(target, key=lambda j: mean[j], reverse=True):
        pool = np.array(sorted(available))
        distance = ((descriptors[pool] - descriptors[feature]) ** 2).sum(axis=1)
        nearest = pool[np.argsort(distance)[: min(4, len(pool))]]
        chosen = int(rng.choice(nearest))
        selected.append(chosen)
        available.remove(chosen)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", default="paligemma.layer_5.output")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = output_path(args.out)
    act_dir = ROOT / "activations/pi05_libero_server"
    sae_dir = ROOT / "saes/pi05_libero_server"
    index_dir = ROOT / "index/pi05_libero_server"
    classifier_path = ROOT / "data/classifiers/libero.json"
    signature = {
        "format": "rq2-pooled-features-v1", "layer": args.layer, "k": args.k, "seed": args.seed,
        "activations": str(act_dir), "saes": str(sae_dir), "index": str(index_dir),
        "sae_sha256": sha256(sae_path(sae_dir, args.layer)),
        "collection_sha256": sha256(act_dir / "collection.json"),
        "index_sha256": sha256(index_dir / f"{args.layer}.npz"),
        "classifier_sha256": sha256(classifier_path),
        "bddl_sha256": {str(p.relative_to(BDDL_ROOT)): sha256(p)
                        for base in BASE_SUITES for suite in (base, base + "_task")
                        for p in sorted((BDDL_ROOT / suite).glob("*.bddl"))},
    }
    manifest(out / "manifest.json", signature)
    if (out / "features.json").exists():
        print(f"Already prepared: {out}", flush=True)
        return
    torch.set_num_threads(2)
    store = ActivationStore(act_dir)
    sae = load_sae(sae_dir, args.layer, device=args.device)
    tasks = sorted({store.episode(e)["task"] for e in store.episode_ids})
    base_metas = {b: suite_meta(b) for b in BASE_SUITES}
    suite_of = {task_language(n): b for b in BASE_SUITES for n in base_metas[b]}
    print(f"Selecting K={args.k} features from {len(store)} training episodes", flush=True)
    sets, mu, effect = select_features(store, sae, args.layer, tasks, suite_of, args.k)
    metrics = FeatureIndex(index_dir).metrics(args.layer)
    probability = GeneralityClassifier.load(classifier_path).predict_proba(metrics)
    frequency = metrics.activation_frequency.to_numpy()
    # Weight task means equally for matching; all selection uses ID demonstrations.
    global_mean = mu.mean(axis=0)
    alive = np.flatnonzero(global_mean > 0)
    general = np.flatnonzero((probability >= 0.5) & (global_mean > 0))
    rng = np.random.default_rng(args.seed)
    pairs = {}
    for base in BASE_SUITES:
        for filename, meta in suite_meta(base + "_task").items():
            original = task_language(filename)
            instruction = instructed_task(meta, base_metas[base])
            if instruction not in tasks:
                continue
            ko, ki = tasks.index(original), tasks.index(instruction)
            shared = np.intersect1d(sets[ko], sets[ki])
            old = np.setdiff1d(sets[ko], shared).tolist()
            new = np.setdiff1d(sets[ki], shared).tolist()
            if not old or not new:
                raise ValueError(f"No exclusive task features: {filename}")
            excluded = set(sets[ko].tolist()) | set(sets[ki].tolist())
            random_pool = [int(j) for j in alive if int(j) not in excluded]
            general_pool = [int(j) for j in general if int(j) not in excluded]
            entry = {
                "original": original, "instructed": instruction,
                "old_prompt": meta["orig_language"], "new_prompt": meta["language"],
                "old_goal": meta["orig_goal"], "new_goal": meta["goal"],
                "old": old, "new": new, "shared": shared.tolist(),
                "old_mean": mu[ko].tolist(), "new_mean": mu[ki].tolist(),
                "random_old": matched_features(old, random_pool, global_mean, frequency, rng),
                "random_new": matched_features(new, random_pool, global_mean, frequency, rng),
                "general_old": matched_features(old, general_pool, global_mean, frequency, rng),
                "general_new": matched_features(new, general_pool, global_mean, frequency, rng),
                "random_pool": random_pool, "general_pool": general_pool,
                "old_effect": effect[ko, old].tolist(), "new_effect": effect[ki, new].tolist(),
            }
            pairs[base + "/" + filename.removesuffix(".bddl")] = entry
    write_json(out / "features.json", {
        "manifest": signature, "num_features": sae.config.num_features,
        "num_general_controls": len(general), "global_mean": global_mean.tolist(), "pairs": pairs,
    })
    print(json.dumps({"pairs": len(pairs), "general_control_pool": len(general), "out": str(out)}), flush=True)


if __name__ == "__main__":
    main()
