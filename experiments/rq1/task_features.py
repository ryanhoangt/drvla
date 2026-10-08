"""Step 2: a task-identity feature set as the new M.

For each LIBERO task k, F_k = the K SAE features whose per-episode mean activation best separates
task k from the other tasks of the same suite (Cohen's d on TRAINING episodes only). On a rollout
step, the activation of a feature set is

    S_k(x) = sum_{j in F_k} f_j(x) / sum_{j in F_k} mu_jk,   mu_jk = mean of f_j over task-k training steps

so S_k ~ 1 means "as active as in task k's own demonstrations". For each rollout:
S_orig (task of the scene / BDDL file) and, for ``_task`` suites, S_instr (training task whose goal
matches the new instruction, when one exists). Controls: random feature sets of the same size,
normalized the same way.

    python experiments/rq1/task_features.py --layer paligemma.layer_5.output
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from analyze_rq1 import BASE_SUITES, instructed_task, task_language  # noqa: E402
from drvla.sae import load_sae  # noqa: E402
from drvla.store import ActivationStore  # noqa: E402
from task_meta import suite_meta  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def select_features(store, sae, layer, tasks, suite_of, K):
    ep_mean, ep_task, step_sum, step_n = [], [], np.zeros((len(tasks), sae.config.num_features)), np.zeros(len(tasks))
    for e in store.episode_ids:
        f = encode(sae, store.activations(layer, e))
        k = tasks.index(store.episode(e)["task"])
        ep_mean.append(f.mean(0))
        ep_task.append(k)
        step_sum[k] += f.sum(0)
        step_n[k] += len(f)
    E, T = np.stack(ep_mean), np.array(ep_task)
    mu = step_sum / step_n[:, None]
    D = np.zeros_like(mu)
    for k, t in enumerate(tasks):
        same = np.array([suite_of[tasks[x]] == suite_of[t] for x in T])
        a, b = E[T == k], E[same & (T != k)]
        D[k] = (a.mean(0) - b.mean(0)) / np.sqrt(0.5 * (a.var(0) + b.var(0)) + 1e-8)
    sets = np.argsort(-D, 1)[:, :K]
    return sets, mu, D


def encode(sae, x):
    x = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=sae.pre_bias.device)
    with torch.no_grad():
        return sae.encode(x).cpu().numpy()


def score(f, idx, mu_k):
    return f[:, idx].sum(1) / max(mu_k[idx].sum(), 1e-9)


def auroc(y, s):
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", default="paligemma.layer_5.output")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--n-random", type=int, default=50)
    p.add_argument("--early", type=int, default=6)
    p.add_argument("--rollouts", default=str(ROOT / "workspace/rq1/rollouts"))
    p.add_argument("--out", default=str(ROOT / "workspace/rq1/analysis"))
    args = p.parse_args()
    rng = np.random.default_rng(0)

    store = ActivationStore(ROOT / "activations/pi05_libero_server")
    sae = load_sae(ROOT / "saes/pi05_libero_server", args.layer, device="cuda")
    tasks = sorted({store.episode(e)["task"] for e in store.episode_ids})
    base_metas = {b: suite_meta(b) for b in BASE_SUITES}
    suite_of = {task_language(n): b for b in BASE_SUITES for n in base_metas[b]}
    sets, mu, D = select_features(store, sae, args.layer, tasks, suite_of, args.k)
    F = sae.config.num_features
    mu_all = mu.mean(0)  # random sets are normalized by their mean over all tasks (task means can be ~0)
    alive = np.nonzero(mu_all > 0)[0]
    rand_sets = [rng.choice(alive, args.k, replace=False) for _ in range(args.n_random)]

    rows = []
    for suite_dir in sorted(Path(args.rollouts).glob("libero_*")):
        suite = suite_dir.name
        base = next(b for b in BASE_SUITES if suite == b or suite.startswith(b + "_"))
        metas = suite_meta(suite) if suite != base else base_metas[base]
        for line in (suite_dir / "labels.jsonl").read_text().splitlines():
            r = json.loads(line)
            f = encode(sae, np.load(suite_dir / f"task{r['task_id']:02d}_ep{r['episode']:03d}.npz")["act/" + args.layer])
            ko = tasks.index(task_language(r["task"] + ".bddl"))
            instr = instructed_task(metas[r["task"] + ".bddl"], base_metas[base]) if suite.endswith("_task") else None
            ki = tasks.index(instr) if instr in tasks else None
            e = slice(0, args.early)
            row = {"suite": suite, "base": base, "pert": suite[len(base) + 1:] or "none", "task": r["task"],
                   "task_id": r["task_id"], "episode": r["episode"], "success": r["success"],
                   "trap": bool(r.get("trap")), "s_orig_early": score(f[e], sets[ko], mu[ko]).mean(),
                   "s_orig_mean": score(f, sets[ko], mu[ko]).mean(),
                   "s_orig_t0": score(f[:1], sets[ko], mu[ko])[0],
                   "s_instr_t0": score(f[:1], sets[ki], mu[ki])[0] if ki is not None else np.nan,
                   "s_instr_early": score(f[e], sets[ki], mu[ki]).mean() if ki is not None else np.nan}
            for i, rs in enumerate(rand_sets):
                row[f"rand{i}_early"] = score(f[e], rs, mu_all).mean()
                row[f"rand{i}_t0"] = score(f[:1], rs, mu_all)[0]
            rows.append(row)
    df = pd.DataFrame(rows)
    tag = args.layer.replace("paligemma.layer_", "PG").replace("action_expert.layer_", "AE").replace(".output", "")
    df.to_parquet(Path(args.out) / f"task_features_{tag}_k{args.k}.parquet")
    df["grp"] = np.where(df.success, "success", np.where(df.trap, "trap", "other"))
    print(f"== {args.layer}: K={args.k}, min d' of selected = {np.sort(D, 1)[:, -args.k].min():.2f}")
    for window in ["early", "t0"]:
        report(df, window, args, tag)
    ot = df[df.suite == "libero_object_task"].groupby("task_id").agg(
        trap=("trap", "mean"), succ=("success", "mean"), s_orig_t0=("s_orig_t0", "mean"), s_instr_t0=("s_instr_t0", "mean"),
        s_orig_early=("s_orig_early", "mean"), s_instr_early=("s_instr_early", "mean"))
    print("(iii) object_task per task:")
    print(ot.round(2).to_string())


def report(df, window, args, tag):
    rand_cols = [c for c in df if c.startswith("rand") and c.endswith("_" + window)]
    col = f"s_orig_{window}"
    print(f"---- window: {window}")

    # (i) within task, trap vs success, mixed tasks only
    pert = df[df.pert != "none"].copy()
    mixed = pert.groupby(["suite", "task_id"]).grp.transform(lambda g: (g == "success").any() and (g == "trap").any())
    h = pert[mixed & (pert.grp != "other")].copy()

    for pt in ["swap", "task", "all"]:
        hh = h if pt == "all" else h[h.pert == pt]
        if hh.empty:
            continue
        sub = h.index.isin(hh.index)
        a = auroc(h.grp[sub] == "trap", h.groupby(["suite", "task_id"])[col]
                  .transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))[sub])
        r = [auroc(h.grp[sub] == "trap", h.groupby(["suite", "task_id"])[c]
                   .transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))[sub]) for c in rand_cols]
        print(f"(i) within-task trap vs success [{pt}] n={sub.sum()} tasks={hh.groupby(['suite','task_id']).ngroups}: "
              f"S_orig AUROC={a:.3f}; random sets {np.nanmean(r):.3f} [{np.nanpercentile(r, 2.5):.3f}, {np.nanpercentile(r, 97.5):.3f}]")

    # (ii) task level: S_orig in perturbed relative to the same task in-distribution vs trap rate
    idm = df[df.pert == "none"].groupby("task")[[col] + rand_cols].mean()
    tl = pert.groupby(["suite", "pert", "task"]).agg(trap=("trap", "mean"), succ=("success", "mean"),
                                                     **{c: (c, "mean") for c in [col] + rand_cols}).reset_index()
    for c in [col] + rand_cols:
        tl[c] = tl[c] / tl.task.map(idm[c])
    for pt in ["swap", "task", "all"]:
        t = tl if pt == "all" else tl[tl.pert == pt]
        rho = spearmanr(t[col], t.trap)
        rr = [spearmanr(t[c], t.trap, nan_policy="omit")[0] for c in rand_cols]
        print(f"(ii) task-level S_orig(pert)/S_orig(ID) vs trap rate [{pt}] n={len(t)}: rho={rho[0]:.3f} (p={rho[1]:.1e}); "
              f"random sets {np.mean(rr):.3f} [{np.percentile(rr, 2.5):.3f}, {np.percentile(rr, 97.5):.3f}]")
    tl.to_csv(Path(args.out) / f"task_features_tasklevel_{tag}_k{args.k}_{window}.csv", index=False)


if __name__ == "__main__":
    main()
