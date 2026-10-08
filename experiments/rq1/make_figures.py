"""Slide-ready figures for the RQ1 report (PNG, light surface).

    python experiments/rq1/make_figures.py   # writes workspace/rq1/figures/*.png
"""

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RQ1 = ROOT / "workspace/rq1"
OUT = RQ1 / "figures"
OUT.mkdir(exist_ok=True)

# Reference palette (dataviz skill): categorical slots 1-2 + a neutral for "other".
SURFACE, TEXT, TEXT2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, OTHER = "#2a78d6", "#eb6834", "#1baf7a", "#b5b4ad"
GROUP_COLORS = {"success": BLUE, "trap": ORANGE, "other": OTHER}
GROUP_LABELS = {"success": "Success", "trap": "Fail: memory trap", "other": "Fail: other"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelcolor": TEXT2, "text.color": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2,
})
PERT_SUITES = ["libero_spatial_swap", "libero_spatial_task", "libero_goal_swap", "libero_goal_task",
               "libero_object_swap", "libero_object_task", "libero_10_swap", "libero_10_task"]


def nice(suite):
    return suite.replace("libero_", "").replace("_", "-")


def subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=TEXT2, fontsize=11, va="bottom")


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def load_labels():
    rows = []
    for f in glob.glob(str(RQ1 / "rollouts/*/labels.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    d = pd.DataFrame(rows)
    d["grp"] = np.where(d.success, "success", np.where(d.trap.fillna(False).astype(bool), "trap", "other"))
    return d


def short_layer(s):
    return s.str.replace("paligemma.layer_", "PG").str.replace("action_expert.layer_", "AE").str.replace(".output", "")


# ---------------------------------------------------------------- F1: outcomes per suite
def fig_outcomes(d):
    t = d[d.suite.isin(PERT_SUITES)].groupby("suite").grp.value_counts(normalize=True).unstack().fillna(0)
    t = t.loc[t.success.sort_values().index]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    left = np.zeros(len(t))
    for g in ["success", "trap", "other"]:
        v = t[g].values * 100
        ax.barh([nice(s) for s in t.index], v, left=left, color=GROUP_COLORS[g], height=0.62,
                edgecolor=SURFACE, linewidth=2, label=GROUP_LABELS[g])
        for y, (x0, w) in enumerate(zip(left, v)):
            if w >= 9:
                ax.text(x0 + w / 2, y, f"{w:.0f}%", ha="center", va="center", fontsize=10,
                        color="white" if g != "other" else TEXT)
        left += v
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of rollouts (200 per suite)")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.14))
    ax.set_title("Most failures are memory traps", pad=28)
    subtitle(ax, "pi0.5-LIBERO on LIBERO-PRO; trap = picks the object at the old location, or does the original task")
    save(fig, "f1_outcomes_by_suite.png")


# ---------------------------------------------------------------- F2: per-task outcomes
def fig_per_task(d):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharey=True)
    for ax, suite in zip(axes.flat, PERT_SUITES):
        g = d[d.suite == suite]
        c = g.groupby("task_id").grp.value_counts().unstack().reindex(columns=["success", "trap", "other"]).fillna(0)
        y = g.success.astype(float)
        r2 = 1 - g.groupby("task_id").success.transform(lambda s: s - s.mean()).pow(2).sum() / ((y - y.mean()) ** 2).sum()
        bottom = np.zeros(len(c))
        for k in ["success", "trap", "other"]:
            ax.bar(c.index, c[k], bottom=bottom, color=GROUP_COLORS[k], width=0.75, edgecolor=SURFACE, linewidth=1.5,
                   label=GROUP_LABELS[k])
            bottom += c[k].values
        ax.set_title(f"{nice(suite)}  ({r2:.0%})", fontsize=12)
        ax.set_xticks(range(10))
        ax.grid(axis="x", visible=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("rollouts (of 20)")
    for ax in axes[1]:
        ax.set_xlabel("task id")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower left", bbox_to_anchor=(0.06, -0.06))
    fig.suptitle("Outcome is decided per task: most tasks give the same result in 20/20 rollouts",
                 x=0.06, y=1.0, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.06, 0.945, "20 rollouts per task; (xx%) = share of the variance in success explained by task identity",
             color=TEXT2, fontsize=11)
    fig.subplots_adjust(hspace=0.45, top=0.86)
    save(fig, "f2_outcomes_per_task.png")


# ---------------------------------------------------------------- F3: RQ1 null (AUROC)
def boot_auc(y, s, rng, n=1000):
    y, s = np.asarray(y), np.asarray(s)
    bs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            bs.append(roc_auc_score(y[i], s[i]))
    return roc_auc_score(y, s), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def fig_auroc():
    ep = pd.read_parquet(RQ1 / "analysis/episodes.parquet")
    ep["L"] = short_layer(ep.layer)
    ep = ep[ep.pert != "none"].copy()
    ep["grp"] = np.where(ep.success, "success", np.where(ep.trap.fillna(False).astype(bool), "trap", "other"))
    rng = np.random.default_rng(0)
    layers = ["PG5", "PG11", "PG17", "AE11"]
    res = {"fail": [], "trap": []}
    for L in layers:
        g = ep[ep.L == L].copy()
        g["z"] = g.groupby(["suite", "task_id"]).r_mem_early.transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
        mixed = g.groupby(["suite", "task_id"]).success.transform(lambda s: 0 < s.mean() < 1).astype(bool)
        h = g[mixed]
        res["fail"].append(boot_auc(~h.success.astype(bool), h.z, rng))
        t = h[h.grp != "other"]
        res["trap"].append(boot_auc(t.grp == "trap", t.z, rng))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(layers))
    for k, (key, color, label) in enumerate([("fail", BLUE, "Fail vs success"), ("trap", ORANGE, "Memory trap vs success")]):
        a = np.array(res[key])
        xs = x + (k - 0.5) * 0.22
        ax.errorbar(xs, a[:, 0], yerr=[a[:, 0] - a[:, 1], a[:, 2] - a[:, 0]], fmt="o", color=color, ms=8,
                    elinewidth=2, capsize=0, label=label)
    ax.axhline(0.5, color=MUTED, linewidth=1.5)
    ax.text(len(layers) - 0.55, 0.505, "chance", color=MUTED, fontsize=10, va="bottom", ha="right")
    ax.set_xticks(x, layers)
    ax.set_ylim(0.3, 0.7)
    ax.set_ylabel("AUROC (95% bootstrap CI)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("RQ1: memorized-feature share does not predict failure", pad=28)
    subtitle(ax, "R_mem in the first 30 steps, z-scored within task; 600 rollouts from tasks with mixed outcomes")
    save(fig, "f3_rq1_auroc.png")


# ---------------------------------------------------------------- F4: R_mem time course
def fig_timecourse():
    st = pd.read_parquet(RQ1 / "analysis/steps.parquet",
                         columns=["layer", "suite", "task_id", "episode", "success", "trap", "t_rel_trap", "r_mem"])
    st = st[(st.layer == "paligemma.layer_5.output") & st.suite.isin(PERT_SUITES) & st.t_rel_trap.notna()].copy()
    st["grp"] = np.where(st.success, "success", np.where(st.trap.fillna(False).astype(bool), "trap", "other"))
    mixed = st.groupby(["suite", "task_id"]).success.transform(lambda s: 0 < s.mean() < 1).astype(bool)
    st = st[mixed].copy()
    st["z"] = st.groupby(["suite", "task_id"]).r_mem.transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
    st = st[(st.t_rel_trap >= -100) & (st.t_rel_trap < 100)]
    st["bin"] = (st.t_rel_trap // 10) * 10 + 5
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for g in ["success", "trap", "other"]:
        e = st[st.grp == g].groupby(["bin", "suite", "task_id", "episode"]).z.mean().reset_index()
        m = e.groupby("bin").z.agg(["mean", "sem"])
        ax.plot(m.index, m["mean"], color=GROUP_COLORS[g], label=GROUP_LABELS[g])
        ax.fill_between(m.index, m["mean"] - 1.96 * m["sem"], m["mean"] + 1.96 * m["sem"], color=GROUP_COLORS[g],
                        alpha=0.15, linewidth=0)
    ax.axvline(0, color=MUTED, linewidth=1.5)
    ax.text(2, ax.get_ylim()[1] * 0.92, "first object lifted", color=MUTED, fontsize=10)
    ax.set_xlabel("env steps relative to first lift")
    ax.set_ylabel("R_mem (z within task)")
    ax.legend(loc="lower right")
    ax.set_title("R_mem follows the grasp phase; outcome groups largely overlap", pad=28)
    subtitle(ax, "PaliGemma layer 5; tasks with mixed outcomes only; lines = mean, bands = 95% CI over rollouts")
    save(fig, "f4_rmem_timecourse.png")


# ---------------------------------------------------------------- F5: FVU shift vs trap rate
def fig_fvu_shift():
    ep = pd.read_parquet(RQ1 / "analysis/episodes.parquet")
    ep = ep[ep.layer == "paligemma.layer_5.output"].copy()
    lab = load_labels()[["suite", "task_id", "episode", "task"]]
    ep = ep.merge(lab, on=["suite", "task_id", "episode"])
    ep["trap"] = ep.trap.fillna(False).astype(bool)
    idm = ep[ep.pert == "none"].groupby(["base", "task"]).fvu_early.mean().rename("fvu_id")
    pt = ep[ep.pert != "none"].groupby(["base", "pert", "task"]).agg(fvu=("fvu_early", "mean"), trap=("trap", "mean"))
    pt = pt.reset_index().join(idm, on=["base", "task"])
    pt["d"] = pt.fvu - pt.fvu_id
    rho, p = spearmanr(pt.d, pt.trap)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for pert, color, label in [("swap", BLUE, "Position swap"), ("task", ORANGE, "Task change")]:
        q = pt[pt.pert == pert]
        ax.scatter(q.d, q.trap * 100, s=64, color=color, edgecolor=SURFACE, linewidth=1.5, label=label, alpha=0.9)
    ax.text(0.98, 0.95, f"Spearman ρ = {rho:.2f}  (p = {p:.1e}, n = {len(pt)} tasks)", transform=ax.transAxes,
            ha="right", va="top", fontsize=11, color=TEXT)
    ax.set_xlabel("Δ SAE reconstruction error vs. same task in-distribution (PG5)")
    ax.set_ylabel("% rollouts that are memory traps")
    ax.legend(loc="center right")
    ax.set_title("Tasks trap more when the scene still 'looks like' training", pad=28)
    subtitle(ax, "Each dot = one perturbed task; x > 0 means the representation moved away from the training data")
    save(fig, "f5_fvu_shift_vs_trap.png")
    return rho, p


# ---------------------------------------------------------------- F6: task probe, object-task
def fig_probe():
    d = pd.read_parquet(RQ1 / "analysis/task_probe.parquet")
    d = d[(d.suite == "libero_object_task") & (d.layer == "paligemma.layer_5.output")]
    d = d[d.i <= 30]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    series = [(5, ORANGE, "task 5 (trap 20/20)", 0.0), (4, AQUA, "task 4 (trap 20/20)", 0.0),
              (2, BLUE, "task 2 (trap 20/20)", 0.05), ("succ", OTHER, "successful rollouts", -0.05)]
    for tid, color, label, dy in series:
        m = (d[d.success] if tid == "succ" else d[d.task_id == tid]).groupby("i").p_orig.mean()
        ax.plot(m.index * 5, m.values, color=color)
        ax.text(m.index[-1] * 5 + 3, m.values[-1] + dy, label, color=TEXT2, fontsize=10, va="center")
    ax.set_ylim(-0.08, 1.05)
    ax.set_xlim(0, 150)
    ax.set_xlabel("env step")
    ax.set_ylabel("P(original task) from linear probe")
    ax.set_title("Two kinds of trap: misread task vs. ignored instruction", pad=44)
    subtitle(ax, "object-task suite, PaliGemma layer 5 (probe accuracy 99.7%).\n"
             "Tasks 4/5: backbone encodes the old task. Task 2: it encodes the new one, yet the robot does the old task")
    save(fig, "f6_task_probe.png")


# ---------------------------------------------------------------- F7: input mismatch
def fig_mismatch():
    import torch

    from drvla.sae import load_sae
    from drvla.store import ActivationStore

    layer = "paligemma.layer_5.output"
    roll = np.concatenate([np.load(f)["act/" + layer] for s in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
                           for f in sorted(glob.glob(str(RQ1 / f"rollouts/{s}/task*_ep00[0-4].npz")))])
    vals = {}
    for name, act, sae_dir in [("Original SAE", "activations/pi05_libero", "saes/pi05_libero"),
                               ("Re-trained SAE", "activations/pi05_libero_server", "saes/pi05_libero_server")]:
        store = ActivationStore(ROOT / act)
        tr = np.concatenate([np.asarray(store.activations(layer, e)) for e in store.episode_ids[::10]])
        sae = load_sae(ROOT / sae_dir, layer)
        mu, var = tr.mean(0), ((tr - tr.mean(0)) ** 2).sum(1).mean()

        def fvu(x):
            with torch.no_grad():
                xh = sae(torch.tensor(x, dtype=torch.float32)).reconstruction.numpy()
            return float(((x - xh) ** 2).sum(1).mean() / var)

        vals[name] = (fvu(tr), fvu(roll))
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(2)
    for k, (name, color) in enumerate([("Original SAE", ORANGE), ("Re-trained SAE", BLUE)]):
        v = vals[name]
        bars = ax.bar(x + (k - 0.5) * 0.36, v, width=0.34, color=color, label=name, edgecolor=SURFACE, linewidth=2)
        for b, val in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.003, f"{val:.3f}", ha="center", va="bottom", fontsize=10,
                    color=TEXT2)
    ax.set_xticks(x, ["its training data", "real policy rollouts (in-distribution)"])
    ax.set_ylabel("SAE reconstruction error (FVU)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    ax.set_title("Side finding: original LIBERO SAEs saw the wrong inputs", pad=28)
    subtitle(ax, "PG5. Dr. VLA collected activations with the state in the prompt; the deployed policy has none")
    save(fig, "f7_input_mismatch.png")
    return vals


# ---------------------------------------------------------------- F8: task-identity features (step 2)
def fig_task_features():
    layers = ["PG0", "PG5", "PG11", "PG17", "AE11"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    for i, L in enumerate(layers):
        t = pd.read_csv(RQ1 / f"analysis/task_features_tasklevel_{L}_k16_t0.csv")
        t = t[t.pert == "task"]
        rand = [spearmanr(t[c], t.trap, nan_policy="omit")[0] for c in t if c.startswith("rand")]
        lo, hi = np.percentile(rand, [2.5, 97.5])
        a1.plot([i, i], [lo, hi], color=OTHER, linewidth=8, solid_capstyle="round",
                label="random feature sets (95%)" if i == 0 else None)
        rho = spearmanr(t.s_orig_t0, t.trap)[0]
        a1.scatter([i], [rho], s=80, color=ORANGE, zorder=3, label="original-task features" if i == 0 else None)
        a1.text(i + 0.12, rho, f"{rho:.2f}", va="center", fontsize=10, color=TEXT2)
    a1.axhline(0, color=MUTED, linewidth=1.2)
    a1.set_xticks(range(len(layers)), layers)
    a1.set_xlim(-0.5, len(layers) - 0.3)
    a1.set_ylim(-0.6, 0.85)
    a1.set_ylabel("Spearman ρ with trap rate (40 tasks)")
    a1.grid(axis="x", visible=False)
    a1.legend(loc="lower right", fontsize=10)
    a1.set_title("Signal appears from PG5 on, not in PG0", fontsize=13, pad=10)

    t = pd.read_csv(RQ1 / "analysis/task_features_tasklevel_PG5_k16_t0.csv")
    t = t[t.pert == "task"]
    a2.scatter(t.s_orig_t0, t.trap * 100, s=64, color=ORANGE, edgecolor=SURFACE, linewidth=1.5)
    rho, p = spearmanr(t.s_orig_t0, t.trap)
    a2.text(0.03, 0.40, f"ρ = {rho:.2f} (p = {p:.3f})\npartial ρ = 0.42\n(controlling for word\noverlap of instructions)",
            transform=a2.transAxes, ha="left", va="center", fontsize=10, color=TEXT)
    a2.set_xlabel("original-task feature activity at t = 0, relative to in-distribution")
    a2.set_ylabel("% rollouts that are memory traps")
    a2.set_title("PG5, each dot = one task-change task", fontsize=13, pad=10)
    fig.suptitle("Before the robot moves: if the new instruction fails to switch off the old task's features, it traps",
                 x=0.06, y=1.04, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.06, 0.965, "Task-change suites. Features per task picked on training data only (top-16 by Cohen's d vs. "
             "other tasks in the suite); measured at the first inference step", color=TEXT2, fontsize=11)
    fig.subplots_adjust(wspace=0.28, top=0.86)
    save(fig, "f8_task_features.png")


if __name__ == "__main__":
    labels = load_labels()
    fig_outcomes(labels)
    fig_per_task(labels)
    fig_auroc()
    fig_timecourse()
    print("fvu shift rho/p:", fig_fvu_shift())
    fig_probe()
    print("mismatch FVU:", fig_mismatch())
