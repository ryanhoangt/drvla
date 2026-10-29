"""Reproducible final interpretation/figures; no policy or GPU imports."""

import argparse
import json
import math
from pathlib import Path

from experiments.exp2.common import ROOT, destination, write_json


LABELS = {"frozen": "Frozen", "sft_pg_only_off": "SFT PG-only", "acttask_off": "Arm1 off",
          "acttask_on": "Arm1 on", "random_acttask_off": "Random off", "random_acttask_on": "Random on"}


def rank(values):
    return [sum(w < v for w in values)+(sum(w == v for w in values)-1)/2 for v in values]


def correlation(a, b):
    a = [x-sum(a)/len(a) for x in a]
    b = [x-sum(b)/len(b) for x in b]
    return sum(x*y for x, y in zip(a, b))/math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))


def readout(training, output):
    output = destination(output)
    summary = json.loads((output / "summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("Final readout requires a completed evaluation")
    evaluation = output / "evaluation"
    results = json.loads((evaluation / "results.json").read_text())
    previous = json.loads((ROOT / "workspace/exp2/cift_job_60924/evaluation/results.json").read_text())
    frozen = [r for r in previous if r["arm"] == "frozen" and r["suite"] in ("libero_object", "libero_object_task")]
    rows = frozen+results
    selection = json.loads((training / "selection.json").read_text())
    fd = {}
    for layer in (5, 11):
        checks = [r for r in selection["finite_difference_checks"] if r["layer"] == layer]
        predicted = [r["predicted_delta"] for r in checks]
        actual = [r["actual_delta"] for r in checks]
        fd[str(layer)] = {"n": len(checks), "sign_agreement": sum(p*a > 0 for p, a in zip(predicted, actual)),
                          "spearman": correlation(rank(predicted), rank(actual))}
    arms = list(LABELS)
    counts = {}
    for arm in arms:
        counts[arm] = {}
        for suite in ("libero_object", "libero_object_task"):
            group = [r for r in rows if r["arm"] == arm and r["suite"] == suite]
            if len(group) != 10:
                raise ValueError("Incomplete paired results")
            counts[arm][suite] = {k: sum(bool(r[k]) for r in group)
                                 for k in ("success", "correct_first_lift", "trap_event", "trap_failure")}
    table = ["| Configuration | ID success | OOD success | OOD correct first lift | OOD trap + failure |",
             "|---|---:|---:|---:|---:|"]
    for arm in arms:
        ident, ood = counts[arm]["libero_object"], counts[arm]["libero_object_task"]
        table.append(f"| {LABELS[arm]} | {ident['success']}/10 | {ood['success']}/10 | "
                     f"{ood['correct_first_lift']}/10 | {ood['trap_failure']}/10 |")
    task_table = ["| Task ID | Original → requested object | Frozen first lift | Arm1 off first lift | Arm1 on first lift |",
                  "|---|---|---|---|---|"]
    for task in range(10):
        case = {arm: next(r for r in rows if r["arm"] == arm and r["suite"] == "libero_object_task"
                          and r["task_id"] == task) for arm in arms}
        base = case["frozen"]
        task_table.append(f"| {task} | {base['old_object']} → {base['new_object']} | "
                          f"{base['first_lift']} | {case['acttask_off']['first_lift']} | {case['acttask_on']['first_lift']} |")
    text = """# Exp2 Arm 1 final readout — training61066 / evaluation61126

Job61126 completed with exit0 after53m41s. The combined evaluation has100 unique
rollouts:33 committed trials reused from cancelled job61066 plus67 newly completed
trials. All artifacts exist. The evaluation audit checked55,315 files and found
no protected modifications. All three models completed500 steps; frozen AE,
PG17, vision, embeddings and motor weights remained byte-exact.

**This pilot does not show a behavioral benefit specific to action-filtered task
CIFT.** Hook-off preserves10/10 ID success but has1/10 OOD success, the same as
matched PG-only SFT and random projection. Both target inference modes rescue
zero of the four baseline traps by correct first lift or goal completion.

""" + "\n".join(table) + """

The one OOD success shared by every fine-tuned configuration is task1, cream
cheese → alphabet soup. Frozen already lifts alphabet soup; completion improves
after training. This is a placement/completion rescue, not a new object-choice
rescue. SFT alone also lifts the requested butter in task7 but fails completion;
Arm1 off instead lifts cream cheese there.

The four actual baseline traps are task0 alphabet soup → cream cheese, task2
salad dressing → tomato sauce, task4 ketchup → milk, and task5 tomato sauce → BBQ
sauce. Arm1 off retains all four old-object first lifts. Hook-on replaces the
task0 trap with no lift, keeps the other three, and creates two new traps on
tasks7 and8. Hence its net trap count increases4→5. A disappearance of one trap
event without lifting the requested object is not a successful intervention.
ID hook-on also regresses on task9 (orange juice; no lift).

""" + "\n".join(task_table) + """

Offline, on120 ID validation frames paired with feasible counterfactual prompts:

- Arm1 off ID FM is0.003955 vs frozen0.003587 and SFT0.004172; ID rollout success
  stays10/10 in this small sample. AE freezing did not make FM loss identical.
- CF new-task frozen PG11 probe accuracy rises47.5%→55.8%, but random off is57.5%
  and matched SFT is50.0%. There is no SAE-specific probe advantage over random.
- Arm1 off PG11 original-feature persistence falls0.1938→0.1143 (about41%),
  while the four old-object traps remain. This feature metric is not a sufficient
  behavioral surrogate in this pilot.
- Arm1 off has t=0 PG11 new-task probability0.746,0.986,0.988 on old-object
  failures0,2,5 respectively. Instruction identity is decodable in these cases;
  this does not establish that the action policy uses it or identify an AE cause.
- Hook-on PG11 ID probe accuracy is10.8%, although ID success is9/10. Directly
  projecting the probe's input shifts its distribution. A collapsed frozen
  probe is not proof that old-task representations were selectively erased.

Selection/causal limits:

- Each task uses8 positive-d action-active features from its RQ1 top16. Overlap
  8/16 is guaranteed by construction, not evidence that a causal bug was fixed.
  Union ranks are76 atPG5 and78 atPG11, unlike the original16-direction PG11 edit.
- Mean absolute attribution matches random within about6%, but projection energy
  does not: frozen PG11 removal is14.4% target versus53.3% random; after training,
  15.2% versus37.7%. Decoder geometry is a remaining dosage difference.
- The saved0.1-strength visual coefficient-ablation checks agree in sign with
  attribution only5/10 times per layer. Spearman is0.418 (PG5) and0.139 (PG11).
  This weakens the claim that the action filter identifies causal features.
  Small edits in BF16 and local nonlinear effects are possible explanations,
  not established causes. These checks do not directly validate the multi-feature
  raw residual projection used for training.
- Attribution is the derivative of a pooled coefficient broadcast to visual
  tokens; it differs from orthogonally projecting token-specific residuals.
  High action relevance alone does not distinguish geometry from shortcuts.
- Ten tasks with one initial state each are exploratory. The null result applies
  to this selection/projection/update budget, not every possible CIFT variant.

The immediate useful follow-up is to calibrate the action filter with paired
finite-strength edits (noise/time fixed, comparing predicted and realized action
or FM changes), then validate feature sets with token-aligned causal contrasts.
The current result does not justify claiming the full diagnostic→training-rescue
story, or assuming a margin loss will fix it.

Artifacts: `ARM1_report.md` contains the complete metric tables; `outcomes.png`
shows behavior and task-paired first lifts; `attribution_checks.png` shows the
saved causal sanity checks. No new GPU experiment was launched for this readout.
"""
    (output / "READOUT.md").write_text(text)
    write_json(output / "readout_data.json", {"counts": counts, "attribution_checks": fd})
    plot(rows, counts, selection, fd, output)


def plot(rows, counts, selection, fd, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    import numpy as np

    arms = list(LABELS)
    colors = ["#64748b", "#16a34a", "#2563eb", "#60a5fa", "#d97706", "#fbbf24"]
    fig = plt.figure(figsize=(14, 10), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.55])
    panels = [("libero_object", "success", "ID goal completions"),
              ("libero_object_task", "success", "OOD goal completions"),
              ("libero_object_task", "trap_failure", "OOD trap + failure")]
    for i, (suite, metric, title) in enumerate(panels):
        ax = fig.add_subplot(grid[0, i])
        values = [counts[arm][suite][metric] for arm in arms]
        bars = ax.bar(range(len(arms)), values, color=colors)
        ax.bar_label(bars, padding=3)
        ax.set_xticks(range(len(arms)), [LABELS[a] for a in arms], rotation=35, ha="right")
        ax.set_ylim(0, 11)
        ax.set_yticks([0, 2, 4, 6, 8, 10])
        ax.set_title(title+" (n=10)")
    ax = fig.add_subplot(grid[1, :])
    matrix = np.zeros((10, len(arms)))
    annotations = []
    task_labels = []
    for task in range(10):
        annotation = []
        for i, arm in enumerate(arms):
            r = next(r for r in rows if r["arm"] == arm and r["suite"] == "libero_object_task" and r["task_id"] == task)
            matrix[task, i] = 0 if r["success"] else (1 if r["correct_first_lift"] else (
                2 if r["trap_event"] else (4 if r["first_lift"] is None else 3)))
            annotation.append((r["first_lift"] or "no lift").removesuffix("_1").replace("_", " "))
            if i == 0:
                task_labels.append(f"{task}: {r['old_object'].removesuffix('_1')} → {r['new_object'].removesuffix('_1')}")
        annotations.append(annotation)
    cmap = ListedColormap(["#bbf7d0", "#e2f8aa", "#fecaca", "#e2e8f0", "#e9d5ff"])
    ax.imshow(matrix, cmap=cmap, vmin=-.5, vmax=4.5, aspect="auto")
    ax.set_yticks(range(10), task_labels)
    ax.set_xticks(range(len(arms)), [LABELS[a] for a in arms])
    for task in range(10):
        for i in range(len(arms)):
            ax.text(i, task, annotations[task][i], ha="center", va="center", fontsize=8)
    ax.set_title("OOD first lifted object — paired task initial states")
    ax.legend(handles=[Patch(color=cmap(i), label=label) for i, label in enumerate(
        ["Goal complete", "Correct lift, goal failed", "Trap + failure", "Other wrong object", "No lift"])],
        loc="upper center", bbox_to_anchor=(.5, -.12), ncol=5, fontsize=8)
    fig.savefig(output / "outcomes.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, layer in zip(axes, (5, 11)):
        checks = [r for r in selection["finite_difference_checks"] if r["layer"] == layer]
        predicted = np.array([r["predicted_delta"] for r in checks])
        actual = np.array([r["actual_delta"] for r in checks])
        lim = max(abs(predicted).max(), abs(actual).max())*1.1
        ax.scatter(predicted, actual, color="#2563eb")
        ax.plot([-lim, lim], [-lim, lim], color="gray", linestyle="--")
        ax.axvline(0, color="gray", linewidth=.5)
        ax.axhline(0, color="gray", linewidth=.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Attribution-predicted Δ FM loss")
        ax.set_ylabel("Realized Δ FM loss (0.1-strength edit)")
        ax.set_title(f"PG{layer}: sign {fd[str(layer)]['sign_agreement']}/10; ρ={fd[str(layer)]['spearman']:.3f}")
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
    fig.savefig(output / "attribution_checks.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-run", type=Path, default=ROOT / "workspace/exp2/arm1/job_61066")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "workspace/exp2/arm1/resume_61066_job_61126")
    args = parser.parse_args()
    readout(args.training_run, args.output_dir)
