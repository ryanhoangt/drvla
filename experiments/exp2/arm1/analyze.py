"""Read-only baseline reuse and a report that also supports partial runs."""

import argparse
import collections
import csv
import json
from pathlib import Path

from experiments.exp2.common import destination, write_json
from experiments.exp2.arm1.common import REFERENCE


def analyze(run, evaluation=None, output=None):
    evaluation = Path(evaluation) if evaluation is not None else run / "evaluation"
    output = destination(output) if output is not None else run
    output.mkdir(parents=True, exist_ok=True)
    validation = json.loads((run / "validation.json").read_text()) if (run / "validation.json").exists() else {}
    results = json.loads((evaluation / "results.json").read_text()) if (evaluation / "results.json").exists() else []
    old = json.loads((REFERENCE / "evaluation/results.json").read_text())
    baseline = [dict(r, arm=r["arm"]+"_60924") for r in old
                if r["arm"] in ("frozen", "sft") and r["suite"] in ("libero_object", "libero_object_task")]
    grouped = collections.defaultdict(list)
    for row in baseline+results:
        grouped[row["arm"], row["suite"]].append(row)
    counts = []
    for (arm, suite), rows in grouped.items():
        counts.append({"arm": arm, "suite": suite, "n": len(rows), **{
            key: sum(bool(r[key]) for r in rows)
            for key in ("success", "correct_first_lift", "trap_event", "trap_failure")}})
    table = ["| Arm | Suite | n | New-goal successes | Correct first lift | Trap | Trap + failure |",
             "|---|---|---:|---:|---:|---:|---:|"]
    table += [f'| {r["arm"]} | {r["suite"]} | {r["n"]} | {r["success"]} | {r["correct_first_lift"]} | '
              f'{r["trap_event"]} | {r["trap_failure"]} |' for r in counts]
    offline = ["| Configuration | ID FM | ID PG11 accuracy | CF new-task accuracy | CF old-task accuracy | CF S_orig PG5 / PG11 |",
               "|---|---:|---:|---:|---:|---:|"]
    for name, r in validation.items():
        offline.append(f'| {name} | {r["ID_FM"]:.6f} | {r["ID_PG11_probe_accuracy"]:.3f} | '
                       f'{r["CF_PG11_new_task_accuracy"]:.3f} | {r["CF_PG11_old_task_accuracy"]:.3f} | '
                       f'{r["CF_S_orig"]["5"]:.3f} / {r["CF_S_orig"]["11"]:.3f} |')
    complete = (evaluation / "status.json").exists() and len(results) == 100
    status = json.loads((run / "status.json").read_text()) if (run / "status.json").exists() else {}
    text = "# Exp2 Arm 1: action-filtered task CIFT\n\n"
    state = "failed" if status.get("status") == "failed" else (
        "complete" if complete else "partial; do not interpret missing rollouts as failures")
    text += f"Status: {state}. New rollouts: {len(results)}/100.\n\n"
    if evaluation != run / "evaluation":
        text += f"Training source: `{run}`. Evaluation: `{evaluation}`. Original artifacts remain unchanged.\n\n"
    if state == "failed":
        text += f"Technical failure: {status['error_type']}: {status['error']}\n\n"
    text += "Three separately reset 500-step PG0–PG16 models: SFT, task projection, and matched random projection. "
    text += "PG5+PG11 visual-token projection; no auxiliary loss or LoRA. AE, PG17, embeddings and vision weights stay frozen. "
    text += "Primary CIFT inference removes hooks; hook-on is a combined train/test intervention diagnostic.\n\n"
    if (run / "selection.json").exists():
        selection = json.loads((run / "selection.json").read_text())
        text += "Selection: positive d × within-task visual-token action attribution, top8 within RQ1 top16. "
        text += "Overlap 8/16 is by construction and is not evidence of causal specificity. "
        text += "Global decoder union ranks: " + ", ".join(f"PG{layer}: {r['rank']}" for layer, r in selection["layers"].items()) + ".\n\n"
    text += "RQ1 task dictionaries include policy-training ID episodes also used for ID validation. "
    text += "ID validation is excluded from rectification and action-attribution fitting. Frozen PG11 probes are reused from job60924; "
    text += "probe drift and CF persistence are diagnostics, not behavioral rescue. Freezing AE does not guarantee preserved ID behavior.\n\n"
    text += "\n".join(offline)+"\n\n"+"\n".join(table)+"\n\n"
    text += "Frozen and full-SFT historical references are read from job60924. Only the new PG-only SFT is a matched training-mask baseline. "
    text += "Initial observations and frozen fixed-noise actions are checked exactly against the reused rollout artifacts. "
    text += "Ten initial states per suite are exploratory; compare paired tasks rather than treating aggregate gains as conclusive.\n"
    (output / "ARM1_report.md").write_text(text)
    write_json(output / "summary.json", {"complete": complete, "new_rollouts": len(results), "counts": counts,
                                      "validation": validation})
    with (output / "behavior.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["arm", "suite", "n", "success", "correct_first_lift", "trap_event", "trap_failure"])
        writer.writeheader()
        writer.writerows(counts)
    if validation:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
        names = list(validation)
        axes[0].barh(names, [validation[n]["ID_FM"] for n in names])
        axes[0].set_xlabel("ID FM validation loss")
        axes[1].barh(names, [validation[n]["CF_PG11_new_task_accuracy"] for n in names])
        axes[1].set_xlabel("CF new-task frozen PG11 probe accuracy")
        axes[1].set_xlim(0, 1)
        fig.savefig(output / "offline.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    analyze(args.run_dir, args.evaluation_dir, args.output_dir)
