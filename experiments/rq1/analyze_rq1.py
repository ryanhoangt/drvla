"""Step 3 (RQ1): do failing rollouts rely more on memorized features?

For every logged rollout and inference call this computes, per layer:

* ``r_mem``  = sum_j (1 - P(general)_j) f_j / sum_j f_j   (activation-weighted memorization)
* ``r_spec`` = sum_j s_j f_j / sum_j f_j, where s_j is the share of feature j's training activation
  mass that falls on its single most active task (task-specificity, in [1/40, 1])
* ``dec_orig`` / ``dec_instr``: cosine similarity of the feature vector to the training centroid of
  the scene's original task / of the instructed task (task suites), and the decoded (argmax) task
* ``fvu``: SAE reconstruction error relative to the training variance (OOD-ness of the activation)

and compares successes with failures (episode = unit) per suite, overall, and within task.

    python experiments/rq1/analyze_rq1.py --rollouts workspace/rq1/rollouts \
        --activations activations/pi05_libero_server --saes saes/pi05_libero_server \
        --index index/pi05_libero_server --out workspace/rq1/analysis
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from drvla.classifier import GeneralityClassifier  # noqa: E402
from drvla.index import FeatureIndex  # noqa: E402
from drvla.sae import load_sae  # noqa: E402
from drvla.store import ActivationStore  # noqa: E402
from task_meta import suite_meta  # noqa: E402

BASE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def task_language(bddl_name):
    """LIBERO task.language from a BDDL/task name (drops the SCENE prefix of libero_10/90)."""
    name = re.sub(r"\.bddl$", "", bddl_name)
    name = re.sub(r"^.*SCENE\d+_", "", name)
    return name.replace("_", " ")


class LayerModel:
    """SAE + per-feature scores + per-task centroids for one layer."""

    def __init__(self, layer, store, sae_dir, index, classifier, device):
        self.layer = layer
        self.sae = load_sae(sae_dir, layer, device=device)
        metrics = index.metrics(layer)
        self.p_gen = classifier.predict_proba(metrics)
        tasks = sorted({store.episode(e)["task"] for e in store.episode_ids})
        self.tasks = tasks
        F = self.sae.config.num_features
        mass = np.zeros((len(tasks), F))
        count = np.zeros(len(tasks))
        sq_sum, x_sum, n = 0.0, None, 0
        for e in store.episode_ids:
            x = np.asarray(store.activations(layer, e), dtype=np.float32)
            f = self.encode(x)
            k = tasks.index(store.episode(e)["task"])
            mass[k] += f.sum(0)
            count[k] += len(f)
            x_sum = x.sum(0) if x_sum is None else x_sum + x.sum(0)
            n += len(x)
        self.centroids = mass / count[:, None]  # (40, F) mean feature vector per task
        share = mass / np.maximum(mass.sum(0, keepdims=True), 1e-12)
        self.spec = share.max(0)  # (F,)
        self.spec_task = share.argmax(0)
        self.x_mean = x_sum / n
        # Training variance for FVU, from a subsample.
        xs = np.concatenate([np.asarray(store.activations(layer, e)) for e in store.episode_ids[::8]])
        self.x_var = float(((xs - self.x_mean) ** 2).sum(1).mean())

    def encode(self, x):
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.sae.pre_bias.device)
        with torch.no_grad():
            return self.sae.encode(x).cpu().numpy()

    def step_scores(self, x):
        xt = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.sae.pre_bias.device)
        with torch.no_grad():
            out = self.sae(xt)
        f = out.features.cpu().numpy()
        recon = out.reconstruction.cpu().numpy()
        total = np.maximum(f.sum(1), 1e-12)
        fn = f / np.linalg.norm(f, axis=1, keepdims=True).clip(1e-12)
        cn = self.centroids / np.linalg.norm(self.centroids, axis=1, keepdims=True).clip(1e-12)
        cos = fn @ cn.T  # (T, 40)
        return {
            "r_mem": (f * (1 - self.p_gen)).sum(1) / total,
            "r_spec": (f * self.spec).sum(1) / total,
            "fvu": ((x - recon) ** 2).sum(1) / self.x_var,
            "cos": cos,
            "l0": (f > 0).sum(1),
        }


def instructed_task(meta, base_metas):
    """Training task whose goal equals the perturbed task's goal (same base suite), if any."""
    for name, m in base_metas.items():
        if m["goal"] == meta["goal"]:
            return task_language(name)
    return None


def auroc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, s))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollouts", required=True, type=Path)
    p.add_argument("--activations", required=True)
    p.add_argument("--saes", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--classifier", default="data/classifiers/libero.json")
    p.add_argument("--layers", nargs="+", default=["paligemma.layer_5.output", "paligemma.layer_11.output",
                                                   "paligemma.layer_17.output", "action_expert.layer_11.output"])
    p.add_argument("--early", type=int, default=6, help="Inference calls in the early window (6 x 5 = 30 steps).")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    store = ActivationStore(args.activations)
    index = FeatureIndex(args.index)
    clf = GeneralityClassifier.load(args.classifier)
    suites = sorted(d.name for d in args.rollouts.iterdir() if (d / "labels.jsonl").exists())
    base_metas = {b: suite_meta(b) for b in BASE_SUITES}

    episodes, steps = [], []
    for layer in args.layers:
        lm = LayerModel(layer, store, args.saes, index, clf, args.device)
        frac_mem = float((lm.p_gen < 0.5).mean())
        print(f"{layer}: {frac_mem:.1%} of features memorized (P(general) < 0.5)")
        for suite in suites:
            base = next(b for b in BASE_SUITES if suite == b or suite.startswith(b + "_"))
            metas = suite_meta(suite) if suite != base else base_metas[base]
            for line in (args.rollouts / suite / "labels.jsonl").read_text().splitlines():
                r = json.loads(line)
                d = np.load(args.rollouts / suite / f"task{r['task_id']:02d}_ep{r['episode']:03d}.npz")
                x = d["act/" + layer]
                sc = lm.step_scores(x)
                orig = task_language(r["task"] + ".bddl")
                k_orig = lm.tasks.index(orig) if orig in lm.tasks else None
                instr = instructed_task(metas.get(r["task"] + ".bddl", {"goal": None}), base_metas[base]) \
                    if suite.endswith("_task") else orig
                k_instr = lm.tasks.index(instr) if instr in lm.tasks else None
                dec = sc["cos"].argmax(1)
                n = len(x)
                tt = np.asarray(d["infer_t"])
                trap_t = r.get("trap_t")
                for i in range(n):
                    steps.append({
                        "layer": layer, "suite": suite, "task_id": r["task_id"], "episode": r["episode"],
                        "success": r["success"], "trap": r.get("trap"), "i": i, "t": int(tt[i]),
                        "t_rel_trap": (int(tt[i]) - trap_t) if trap_t is not None else None,
                        "phase": i / max(n - 1, 1),
                        "r_mem": float(sc["r_mem"][i]), "r_spec": float(sc["r_spec"][i]), "fvu": float(sc["fvu"][i]),
                        "dec_is_orig": bool(k_orig is not None and dec[i] == k_orig),
                        "dec_is_instr": bool(k_instr is not None and dec[i] == k_instr),
                        "cos_orig": float(sc["cos"][i, k_orig]) if k_orig is not None else np.nan,
                        "cos_instr": float(sc["cos"][i, k_instr]) if k_instr is not None else np.nan,
                    })
                e = args.early
                episodes.append({
                    "layer": layer, "suite": suite, "base": base, "pert": suite[len(base) + 1:] or "none",
                    "task_id": r["task_id"], "episode": r["episode"], "success": r["success"],
                    "trap": r.get("trap"), "trap_kind": r.get("trap_kind"), "n_infer": n,
                    "has_instr_task": k_instr is not None,
                    **{f"{m}_mean": float(np.mean(sc[m])) for m in ("r_mem", "r_spec", "fvu")},
                    **{f"{m}_early": float(np.mean(sc[m][:e])) for m in ("r_mem", "r_spec", "fvu")},
                    "dec_orig_early": float(np.mean(dec[:e] == k_orig)) if k_orig is not None else np.nan,
                    "dec_instr_early": float(np.mean(dec[:e] == k_instr)) if k_instr is not None else np.nan,
                    "cos_orig_minus_instr_early": float(np.mean(sc["cos"][:e, k_orig] - sc["cos"][:e, k_instr]))
                    if (k_orig is not None and k_instr is not None) else np.nan,
                })
        del lm
        torch.cuda.empty_cache()

    ep = pd.DataFrame(episodes)
    st = pd.DataFrame(steps)
    ep.to_parquet(args.out / "episodes.parquet")
    st.to_parquet(args.out / "steps.parquet")

    # Success-vs-failure comparisons. Within-task: z-score each score within (layer, suite, task) first,
    # so that tasks with low success and unusual scores do not drive the pooled comparison.
    metrics = ["r_mem_early", "r_spec_early", "fvu_early", "r_mem_mean", "r_spec_mean", "fvu_mean",
               "dec_orig_early", "cos_orig_minus_instr_early"]
    rows = []
    for (layer, suite), g in ep.groupby(["layer", "suite"]):
        fail = ~g["success"].astype(bool)
        for m in metrics:
            v = g[m].astype(float)
            if v.isna().all():
                continue
            z = v.groupby(g["task_id"]).transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
            a, b = v[fail], v[~fail]
            mixed = g.groupby("task_id")["success"].transform(lambda s: 0 < s.mean() < 1).astype(bool)
            rows.append({
                "layer": layer, "suite": suite, "metric": m, "n_fail": int(fail.sum()), "n_succ": int((~fail).sum()),
                "mean_fail": float(a.mean()) if len(a) else np.nan, "mean_succ": float(b.mean()) if len(b) else np.nan,
                "auroc_fail": auroc(fail, v),
                "auroc_fail_within_task": auroc(fail[mixed], z[mixed]) if mixed.sum() else np.nan,
                "n_mixed_tasks": int(g.loc[mixed, "task_id"].nunique()),
                "p_mwu": float(mannwhitneyu(a, b).pvalue) if len(a) > 1 and len(b) > 1 else np.nan,
            })
    comp = pd.DataFrame(rows)
    comp.to_csv(args.out / "success_vs_failure.csv", index=False)

    # Perturbed vs in-distribution (successful ID rollouts as reference).
    rows = []
    for (layer, base), g in ep.groupby(["layer", "base"]):
        ref = g[(g.pert == "none")]
        for pert, h in g[g.pert != "none"].groupby("pert"):
            for m in ["r_mem_early", "r_spec_early", "fvu_early"]:
                rows.append({"layer": layer, "base": base, "pert": pert, "metric": m,
                             "id_mean": float(ref[m].mean()), "pert_mean": float(h[m].mean()),
                             "auroc_pert_vs_id": auroc(np.r_[np.zeros(len(ref)), np.ones(len(h))], np.r_[ref[m], h[m]])})
    pd.DataFrame(rows).to_csv(args.out / "perturbed_vs_id.csv", index=False)

    # Time course: mean score by inference index for success / failure.
    tc = st.groupby(["layer", "suite", "success", "i"])[["r_mem", "r_spec", "fvu", "dec_is_orig", "dec_is_instr"]].mean()
    tc.to_csv(args.out / "timecourse.csv")
    print(comp[comp.metric.isin(["r_mem_early", "r_spec_early", "fvu_early", "dec_orig_early"])]
          .round(3).to_string())


if __name__ == "__main__":
    main()
