"""Reproduce the RQ2 report figures from completed, read-only pilot artifacts.

    env openpi/.venv/bin/python -m experiments.rq2.make_figures

No policy/simulator is started. All outputs, including the Matplotlib cache,
stay under workspace/rq2. figure_data.json contains the plotted measurements;
figure_manifest.json records SHA256 hashes and checks that inputs did not change.
"""

import argparse
import json
import os
from pathlib import Path
import re

from experiments.rq2.common import ROOT, RQ2_ROOT, output_path, sha256, write_json

os.environ["MPLCONFIGDIR"] = str(RQ2_ROOT / "figures" / ".mplcache")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, TEXT, TEXT2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, OTHER = "#2a78d6", "#eb6834", "#1baf7a", "#b5b4ad"
CONDITIONS = [
    "baseline", "suppress_old", "boost_new", "patch_old", "random_suppress",
    "general_suppress", "random_boost", "general_boost",
]
LABELS = {
    "baseline": "Baseline", "suppress_old": "Suppress old", "boost_new": "Boost new",
    "patch_old": "Patch old", "random_suppress": "Random suppress",
    "general_suppress": "General suppress", "random_boost": "Random boost",
    "general_boost": "General boost", "old_prompt": "Old prompt",
}
COLORS = {c: BLUE if c in ("suppress_old", "boost_new", "patch_old") else OTHER for c in CONDITIONS}
COLORS["baseline"] = MUTED

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelcolor": TEXT2, "text.color": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2,
})


class Inputs:
    def __init__(self):
        self.hashes = {}

    def track(self, path):
        path = Path(path).resolve()
        if path not in self.hashes:
            self.hashes[path] = sha256(path)
        return path

    def json(self, path):
        return json.loads(self.track(path).read_text())

    def npz(self, path):
        return np.load(self.track(path), allow_pickle=False)

    def unchanged(self):
        changed = [str(p) for p, h in self.hashes.items() if sha256(p) != h]
        if changed:
            raise ValueError(f"Plot inputs changed during rendering: {changed}")


class Run:
    def __init__(self, root, inputs):
        self.root, self.inputs = Path(root).resolve(), inputs
        self.config = inputs.json(self.root / "run_manifest.json")
        self.bundle = inputs.json(ROOT / self.config["features"])
        self.summary = inputs.json(self.root / "summary.json")
        self.audit = inputs.json(self.root / "protected_artifacts_audit.json")
        self.rows = []
        for path in sorted((self.root / "rollouts").glob("*/results.jsonl")):
            self.rows.extend(json.loads(s) for s in inputs.track(path).read_text().splitlines() if s.strip())
        self.by_key = {}
        for row in self.rows:
            key = (row["task_id"], row["episode"], row["noise_base"], row["condition"])
            if key in self.by_key:
                raise ValueError(f"Duplicate rollout {key}")
            self.by_key[key] = row
        expected = {(t, e, n, c) for t in self.config["tasks"] for e in self.config["episodes"]
                    for n in self.config["noise_seeds"] for c in (*CONDITIONS, "old_prompt")}
        if set(self.by_key) != expected:
            raise ValueError("This figure set requires a complete nine-condition pilot")
        self.verification = {}
        for path in sorted((self.root / "verification").glob("*.json")):
            match = re.fullmatch(r"task(\d+)_ep(\d+)_seed(-?\d+)", path.stem)
            if match is None:
                raise ValueError(path)
            key = tuple(map(int, match.groups()))
            report = inputs.json(path)
            if any(v != 0 for v in report["identity_checks_max_abs"].values()):
                raise ValueError(f"Identity gate failed in {path}")
            self.verification[key] = (report, path.with_suffix(".npz"))
        blocks = {key[:3] for key in self.by_key}
        if set(self.verification) != blocks:
            raise ValueError("Missing or unexpected verification blocks")

    def row(self, task, condition):
        # The primary confirmation has one initial state and one noise base per task.
        if len(self.config["episodes"]) != 1 or len(self.config["noise_seeds"]) != 1:
            raise ValueError("Primary plots require one paired block per task")
        return self.by_key[(task, self.config["episodes"][0], self.config["noise_seeds"][0], condition)]

    def artifact(self, row):
        path = (self.root / row["artifact"]).resolve()
        if self.root not in path.parents:
            raise ValueError("Artifact escapes its run directory")
        return self.inputs.npz(path)


def heading(fig, title, subtitle):
    fig.suptitle(title, x=0.06, y=0.99, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.06, 0.91, subtitle, color=TEXT2, fontsize=10)


def save(fig, out, filename):
    fig.savefig(output_path(out / filename), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out / filename}", flush=True)


def outcomes(run, out):
    n = len(run.config["tasks"])
    values = {}
    for c in CONDITIONS:
        rows = [r for r in run.rows if r["condition"] == c]
        values[c] = {"success": sum(r["success"] for r in rows),
                     "old_object_failure": sum(r["trap_failure"] for r in rows), "n": len(rows)}
        official = run.summary["conditions"][c]
        if values[c]["success"] != official["successes"] or values[c]["old_object_failure"] != official["trap_failures"]:
            raise ValueError("Figure measurements disagree with completed-run summary")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for ax, key, title in zip(axes, ("success", "old_object_failure"),
                              ("New-goal successes", "Old-object failures (event + final failure)")):
        counts = [values[c][key] for c in CONDITIONS]
        ax.barh(np.arange(len(CONDITIONS)), counts,
                color=[COLORS[c] if key == "success" else ORANGE for c in CONDITIONS], height=0.64)
        for i, count in enumerate(counts):
            ax.text(count + 0.07, i, f"{count}/{n}", va="center", fontsize=11)
        ax.set_xlim(0, n + 0.6)
        ax.set_xticks(range(n + 1))
        ax.set_xlabel("Rollouts")
        ax.set_title(title, fontsize=12, pad=13)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(CONDITIONS)), [LABELS[c] for c in CONDITIONS])
    axes[0].invert_yaxis()
    heading(fig, "The selected SAE edits do not rescue old-object failures",
            "Env-canonical prompt; four paired tasks; one initial state/noise per task; no suite-level SR claim")
    fig.subplots_adjust(left=0.19, right=0.97, top=0.79, bottom=0.12, wspace=0.18)
    save(fig, out, "f1_paired_outcomes.png")
    return values


def task_outcomes(run, out):
    tasks = run.config["tasks"]
    categories = np.zeros((len(tasks), len(CONDITIONS)), dtype=int)
    labels, data = [], {}
    for i, task in enumerate(tasks):
        row = run.row(task, "baseline")
        pair = run.bundle["pairs"]["libero_object/" + row["task"]]
        old, new = pair["old_goal"][0][1], pair["new_goal"][0][1]
        def nice(name):
            return name.removesuffix("_1").replace("_", " ")
        labels.append(f"Task {task}: {nice(old)} → {nice(new)}")
        data[str(task)] = {}
        for j, c in enumerate(CONDITIONS):
            r = run.row(task, c)
            category = 2 if r["success"] else 0 if r["trap_failure"] else 1
            categories[i, j] = category
            data[str(task)][c] = {"success": r["success"], "old_task_event": r["old_task_event"],
                                  "first_lift": r["first_lift"], "steps": r["steps"]}
    fig, ax = plt.subplots(figsize=(12, 4.7))
    ax.imshow(categories, cmap=ListedColormap([ORANGE, OTHER, BLUE]), vmin=-0.5, vmax=2.5, aspect="auto")
    for i in range(len(tasks)):
        for j in range(len(CONDITIONS)):
            ax.text(j, i, ("Old-object\nfail", "Other\nfail", "Success")[categories[i, j]],
                    ha="center", va="center", color="white" if categories[i, j] != 1 else TEXT, fontsize=10)
    ax.set_yticks(range(len(tasks)), labels)
    ax.set_xticks(range(len(CONDITIONS)), [LABELS[c].replace(" ", "\n", 1) for c in CONDITIONS])
    ax.set_xticks(np.arange(-0.5, len(CONDITIONS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tasks), 1), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color=SURFACE, linewidth=3)
    ax.tick_params(which="both", length=0)
    heading(fig, "All edits rescue task 1; none changes the old-object choices",
            "One rollout per cell; task 1 already lifts the instructed object in its failed baseline")
    fig.subplots_adjust(left=0.32, right=0.98, top=0.78, bottom=0.19)
    save(fig, out, "f2_outcomes_per_task.png")
    return data


def suppression_fidelity(run, out):
    measurements = []
    calls = run.config["intervene_calls"]
    for task in run.config["tasks"]:
        row = run.row(task, "suppress_old")
        ids = run.bundle["pairs"]["libero_object/" + row["task"]]["old"]
        with run.artifact(row) as x:
            before = float(x["features"][:calls, ids].sum())
            after = float(x["features_after"][:calls, ids].sum())
            intended = float(x["edited_features"][:calls, ids].sum())
            if before <= 0:
                raise ValueError("Cannot measure suppression without active old-task coordinates")
            relative = x["edit/realized_delta_norm"][:calls] / np.linalg.norm(x["pooled"][:calls], axis=-1)
            measurements.append({"task_id": task, "old_before_sum": before, "old_after_sum": after,
                                 "remaining_ratio": after / before, "intended_remaining_ratio": intended / before,
                                 "mean_realized_relative_delta": float(relative.mean())})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    labels = [f"Task {m['task_id']}" for m in measurements]
    for ax, key, title, ylabel in [
        (axes[0], "remaining_ratio", "Old-task coordinates are strongly reduced", "Re-encoded activation remaining (%)"),
        (axes[1], "mean_realized_relative_delta", "The residual-stream edit is small", "Mean realized Δ / pooled hidden norm (%)"),
    ]:
        values = np.array([m[key] for m in measurements]) * 100
        ax.bar(labels, values, color=BLUE if key == "remaining_ratio" else AQUA, width=0.6)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.1f}%" if key == "remaining_ratio" else f"{value:.3f}%",
                    ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, max(values) * 1.25)
        ax.set_title(title, fontsize=12, pad=14)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
    heading(fig, "The null wrong-object result is not an identity edit",
            "Canonical suppress-old; coefficients summed over target IDs and the first six inference calls")
    fig.text(0.06, 0.015, "Intended old coefficients are zero. Re-encoding can restore activity through nonorthogonal directions/TopK.",
             color=TEXT2, fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.16, wspace=0.38)
    save(fig, out, "f3_suppression_fidelity.png")
    return measurements


def action_and_control_fidelity(run, out):
    modes = CONDITIONS[1:] + ["old_prompt"]
    action_ratios = {c: [] for c in modes}
    for task in run.config["tasks"]:
        report, _ = run.verification[(task, run.config["episodes"][0], run.config["noise_seeds"][0])]
        denominator = report["effects"]["old_prompt"]["action_xyz_delta_norm"]
        if denominator <= 0:
            raise ValueError("Old-prompt action contrast is zero")
        for c in modes:
            action_ratios[c].append({"task_id": task, "ratio_to_old_prompt":
                                    report["effects"][c]["action_xyz_delta_norm"] / denominator})
    controls = ["random_suppress", "general_suppress", "random_boost", "general_boost"]
    norm_ratios = {}
    for c in controls:
        nominal, realized = [], []
        for task in run.config["tasks"]:
            with run.artifact(run.row(task, c)) as x:
                mask = x["edit/control_target_norm"] > 1e-8
                nominal.extend(x["edit/control_norm_ratio"][mask].tolist())
                mask = x["edit/control_realized_target_norm"] > 1e-8
                realized.extend(x["edit/control_realized_norm_ratio"][mask].tolist())
        norm_ratios[c] = {"nominal": nominal, "realized": realized,
                          "within_5pct": float(np.mean(np.abs(np.asarray(realized) - 1) < 0.05))}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    for i, c in enumerate(modes):
        values = np.array([r["ratio_to_old_prompt"] for r in action_ratios[c]])
        if np.any(values <= 0):
            raise ValueError("Log-scale action plot requires nonzero action changes")
        axes[0].scatter(i + np.linspace(-0.16, 0.16, len(values)), values, s=33,
                        color=COLORS.get(c, ORANGE), zorder=3)
        axes[0].plot([i - 0.24, i + 0.24], [values.mean()] * 2, color=TEXT, linewidth=1.7)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=1))
    axes[0].set_xticks(range(len(modes)), [LABELS[c].replace(" ", "\n", 1) for c in modes], fontsize=9)
    axes[0].set_ylabel("XYZ action-change norm / old-prompt contrast")
    axes[0].set_title("Nonzero action effects; small versus full prompt change", fontsize=11, pad=14)
    axes[0].grid(axis="x", visible=False)
    axes[0].text(0, -0.3, "Each dot: one task at t=0, same observation/noise. Black ticks: means.",
                 transform=axes[0].transAxes, color=TEXT2, fontsize=9)
    axes[1].axhspan(0.95, 1.05, color=GRID, alpha=0.65)
    axes[1].axhline(1, color=MUTED, linestyle="--", linewidth=1)
    for i, c in enumerate(controls):
        values = np.asarray(norm_ratios[c]["realized"])
        axes[1].scatter(i + np.linspace(-0.19, 0.19, len(values)), values, s=17, color=ORANGE, alpha=0.65, zorder=3)
        axes[1].scatter(i, np.mean(norm_ratios[c]["nominal"]), marker="D", color=BLUE, s=45, zorder=4)
        axes[1].plot([i - 0.25, i + 0.25], [values.mean()] * 2, color=TEXT, linewidth=1.7)
    axes[1].set_xticks(range(len(controls)), [LABELS[c].replace(" ", "\n", 1) for c in controls], fontsize=10)
    axes[1].set_ylabel("Control / same-recipient target edit norm")
    axes[1].set_title("BF16 weakens suppression norm matching", fontsize=11, pad=14)
    axes[1].grid(axis="x", visible=False)
    handles = [plt.Line2D([], [], color=ORANGE, marker="o", linestyle="", label="Realized (per call)"),
               plt.Line2D([], [], color=BLUE, marker="D", linestyle="", label="Intended (mean)")]
    axes[1].legend(handles=handles, loc="upper left", fontsize=9)
    axes[1].text(0, -0.3, "Shaded band: ±5%. Only calls with nonzero target norm are shown.",
                 transform=axes[1].transAxes, color=TEXT2, fontsize=9)
    heading(fig, "Action changes are real, but controls are not perfectly matched after rounding",
            "First five XYZ commands in the same-observation probe; control norms measured at each recipient")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.77, bottom=0.25, wspace=0.3)
    save(fig, out, "f4_action_and_control_fidelity.png")
    return {"action_ratios": action_ratios, "control_norm_ratios": norm_ratios}


def prompt_protocol(raw, canonical, out):
    blocks = sorted(canonical.verification)
    counts = {"raw": {}, "canonical": {}}
    case_effects = []
    for c in CONDITIONS:
        counts["raw"][c] = sum(raw.by_key[(*b, c)]["success"] for b in blocks)
        counts["canonical"][c] = sum(canonical.by_key[(*b, c)]["success"] for b in blocks)
    for block in blocks:
        a, a_path = raw.verification[block]
        b, b_path = canonical.verification[block]
        if a["observation_sha256"] != b["observation_sha256"] or a["noise_seed"] != b["noise_seed"]:
            raise ValueError("Prompt-case runs do not have matched physical inputs/noise")
        with raw.inputs.npz(a_path) as x, canonical.inputs.npz(b_path) as y:
            np.testing.assert_array_equal(x["noise"], y["noise"])
            for key in ("image", "wrist", "state"):
                np.testing.assert_array_equal(x[key], y[key])
            delta = y["action/baseline"][:5, :3] - x["action/baseline"][:5, :3]
            case_effects.append({"task_id": block[0], "xyz_action_delta_norm": float(np.linalg.norm(delta))})
    for key, pair in raw.bundle["pairs"].items():
        other = canonical.bundle["pairs"][key]
        for field in ("old", "new", "old_mean", "new_mean", "random_old", "random_new", "general_old", "general_new"):
            if pair[field] != other[field]:
                raise ValueError(f"Feature selection changed between prompt-case runs: {key}/{field}")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), gridspec_kw={"width_ratios": [1.9, 1]})
    xs = np.arange(len(CONDITIONS))
    for shift, name, color, label in [(-0.18, "raw", BLUE, "Raw BDDL: Pick…"),
                                      (0.18, "canonical", ORANGE, "Env prompt: pick…")]:
        values = [counts[name][c] for c in CONDITIONS]
        axes[0].bar(xs + shift, values, width=0.34, color=color, label=label)
    axes[0].set_xticks(xs, [LABELS[c].replace(" ", "\n", 1) for c in CONDITIONS], fontsize=9)
    axes[0].set_ylim(0, len(blocks))
    axes[0].set_yticks(range(len(blocks) + 1))
    axes[0].set_ylabel(f"New-goal successes / {len(blocks)} matched tasks")
    axes[0].grid(axis="x", visible=False)
    axes[0].legend(loc="upper right", fontsize=10)
    axes[0].set_title("Case sensitivity changes one baseline completion", fontsize=11, pad=14)
    values = [r["xyz_action_delta_norm"] for r in case_effects]
    axes[1].bar([f"Task {r['task_id']}" for r in case_effects], values, color=BLUE, width=0.6)
    for i, value in enumerate(values):
        axes[1].text(i, value, f"{value:.3f}", va="bottom", ha="center", fontsize=10)
    axes[1].set_ylim(0, max(values) * 1.25)
    axes[1].set_ylabel("‖Δ(first five XYZ commands)‖₂")
    axes[1].set_title("Capitalization changes the initial actions", fontsize=11, pad=14)
    axes[1].grid(axis="x", visible=False)
    heading(fig, "Prompt normalization is part of the evaluation protocol",
            "Matched initial state 0 only (n=4 per condition); same images/state/noise and feature sets, not pooled runs")
    fig.text(0.06, 0.015, "Right: policy command units, not end-effector displacement. No additional policy inference was run for this plot.",
             color=TEXT2, fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.19, wspace=0.3)
    save(fig, out, "f5_prompt_protocol.png")
    return {"matched_counts": counts, "case_action_effects": case_effects, "matched_blocks": blocks}


def task_one_trajectory(run, out):
    task = 1
    selected = [("baseline", MUTED), ("boost_new", BLUE), ("random_boost", AQUA)]
    trajectories = {}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for c, color in selected:
        row = run.row(task, c)
        pair = run.bundle["pairs"]["libero_object/" + row["task"]]
        with run.artifact(row) as x:
            names = x["obj_names"].tolist()
            target, basket = names.index(pair["new_goal"][0][1]), names.index("basket_1")
            target_xyz = x["obj_pos"][:, target]
            height = target_xyz[:, 2] - x["obj_pos"][:, basket, 2]
            distance = np.linalg.norm(target_xyz - x["eef_pos_after"], axis=-1)
            steps = np.arange(1, len(height) + 1)
            goal = x["new_goal_ok"].all(axis=-1)
            if bool(goal.any()) != row["success"]:
                raise ValueError("Trajectory predicates disagree with the outcome record")
            trajectories[c] = {"steps": row["steps"], "success": row["success"],
                               "first_lift": row["first_lift"], "old_task_event": row["old_task_event"],
                               "final_height_over_basket_m": float(height[-1]),
                               "final_object_eef_distance_m": float(distance[-1]),
                               "first_new_goal_t": row["first_new_goal_t"]}
            for ax, values in zip(axes, (height, distance)):
                ax.plot(steps, values, color=color, label=LABELS[c])
                ax.scatter(steps[-1], values[-1], color=color, marker="*" if row["success"] else "x", s=95, zorder=4)
            axes[0].annotate(f"goal @ {row['steps']}" if row["success"] else f"no goal @ {row['steps']}",
                             (steps[-1], height[-1]), xytext=(-5, -18 if c == "random_boost" else 8),
                             textcoords="offset points", ha="right", color=color, fontsize=9)
    for ax in axes:
        ax.axvspan(0, 30, color=GRID, alpha=0.65)
        ax.axvline(30, color=MUTED, linestyle=":", linewidth=1)
        ax.set_xlabel("Executed action step")
        ax.set_xlim(0, run.config["max_steps"] + 10)
    axes[0].set_title("Alphabet soup height relative to basket body origin", fontsize=11, pad=14)
    axes[0].set_ylabel("Height difference (m)")
    axes[1].set_title("Alphabet soup distance from end-effector", fontsize=11, pad=14)
    axes[1].set_ylabel("Post-action distance (m)")
    axes[1].legend(loc="upper right", fontsize=10)
    heading(fig, "The rescued rollout already chose the instructed object",
            "Task 1, canonical prompt; target = alphabet soup; shaded interval = first 30 actions (edit window)")
    fig.text(0.06, 0.015, "Stars: first observed goal / rollout termination. Positions support a placement/release interpretation, not a contact-state proof.",
             color=TEXT2, fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.76, bottom=0.16, wspace=0.3)
    save(fig, out, "f6_task1_trajectory.png")
    return trajectories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-run", type=Path, default=RQ2_ROOT / "pilot_pg5_k16_a1_first6_20261009")
    parser.add_argument("--canonical-run", type=Path, default=RQ2_ROOT / "pilot_pg5_k16_a1_first6_envcase_20261009")
    parser.add_argument("--out", type=Path, default=RQ2_ROOT / "figures")
    args = parser.parse_args()
    out = output_path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = Inputs()
    raw, canonical = Run(args.raw_run, inputs), Run(args.canonical_run, inputs)
    if canonical.bundle["manifest"].get("prompt_style") != "env-lowercase":
        raise ValueError("Primary report must use env-canonical prompts")
    if canonical.config["alpha"] != 1 or canonical.config["intervene_calls"] != 6:
        raise ValueError("These plot captions describe alpha=1 and a six-call edit window")
    measurements = {
        "outcomes": outcomes(canonical, out),
        "per_task_outcomes": task_outcomes(canonical, out),
        "suppression_fidelity": suppression_fidelity(canonical, out),
        "action_and_control_fidelity": action_and_control_fidelity(canonical, out),
        "prompt_protocol": prompt_protocol(raw, canonical, out),
        "task1_trajectories": task_one_trajectory(canonical, out),
    }
    inputs.unchanged()
    write_json(out / "figure_data.json", measurements)
    write_json(out / "figure_manifest.json", {
        "script_sha256": sha256(Path(__file__)), "matplotlib_version": matplotlib.__version__,
        "numpy_version": np.__version__, "raw_run": str(raw.root), "canonical_run": str(canonical.root),
        "input_sha256": {str(p.relative_to(ROOT)): h for p, h in sorted(inputs.hashes.items())},
        "inputs_unchanged_after_rendering": True,
        "output_sha256": {p.name: sha256(p) for p in sorted(out.glob("f*.png"))},
    })


if __name__ == "__main__":
    main()
