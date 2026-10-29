"""Generate a self-contained report from completed Exp2 artifacts, without policy calls."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from experiments.exp2.artifacts import load_scores
from experiments.exp2.common import ROOT, destination, sha256, write_json

ARMS = ("frozen", "sft", "sae_m", "random_sae", "pca_promptdiff")
LABELS = ("Frozen", "SFT", "CIFT SAE-M", "Random SAE", "Prompt-diff PCA")
COLORS = ("#777777", "#3498db", "#e87524", "#9b59b6", "#2eae8a")


def load_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def analyze(run):
    run = destination(run)
    manifest = load_json(run / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(run / "manifest.json")
    score = load_json(run / "scoring.json")
    valid = load_json(run / "validation.json")
    results = load_json(run / "evaluation/results.json", [])
    status = load_json(run / "evaluation/status.json", {})
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    measurements = {"training_job": manifest["job_id"], "completed_rollouts": len(results),
                    "evaluation_complete": status.get("status") == "complete", "scoring": {}}
    report = ["# Exp2: ID-only counterfactual SAE scoring and CIFT rectification", "",
              "**Status:** " + ("evaluation complete" if measurements["evaluation_complete"] else
                                "partial artifacts; do not interpret as a completed behavioral experiment"), "",
              "## Scope and predeclared comparisons", "",
              f"Slurm job {manifest['job_id']}, partition {manifest['partition']}, GPU {manifest['gpu']}. "
              f"Four paired training arms, {manifest['arguments']['steps']} optimizer steps each; "
              "effective batch 2, full-parameter SFT with native OpenPI BF16 precision and AdamW defaults, "
              "peak learning rate 5e-5 and a 25-step rectification warmup. "
              "The same ID examples, flow noise/time, optimizer and initial checkpoint are used in every arm. "
              "Only CIFT/projection condition differs. All primary inference removes the projection.", "",
              "Data: 24 ID demonstrations per each of ten object tasks, six frames per episode (1,440 train examples); "
              "four of those episodes per task, two early frames each (80 scoring examples); "
              "four additional episodes per task, three frames each (120 validation examples). "
              "Validation is held out of Exp2 rectification, not necessarily original policy SFT. "
              "Counterfactual prompts use other ID task names whose target objects exist in the ID scene. "
              "There are no counterfactual action labels and no OOD examples in the score or training objective.", "",
              "## What earlier experiments establish", "",
              "RQ1 found generic rarity labels uninformative for within-task failure prediction and task-feature "
              "persistence correlated with trap rates across tasks. These are associations. "
              "A low SAE reconstruction shift is not a calibrated density/subspace-membership estimate. "
              "RQ2's PG5 edits did not change wrong-object choices; the placement rescue also occurred with controls. "
              "Exp1 job 58962 showed dense PG11 donor patching redirects approach, with no old-object lift; "
              "pooled versions still lifted the new object. Pooled broadcast edits preserve token differences "
              "algebraically; these results do not establish destruction of spatial attention or inability of all "
              "test-time interventions to work.", ""]
    if score is not None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, layer in zip(axes[:2], (5, 11), strict=True):
            z = load_scores(run / f"score_pg{layer}.npz")
            attribution, sensitivity, rank_score = z["attribution"], z["sensitivity"], z["score"]
            features, effects, per_s = z["f"], z["effect"], z["sample_sensitivity"]
            episodes, tasks = z["episode"], z["task"]
            selected = np.array(score[str(layer)]["feature_ids"])
            alive = attribution > 0
            ax.scatter(sensitivity[alive], attribution[alive], s=9, alpha=.3, color="#777777")
            ax.scatter(sensitivity[selected], attribution[selected], s=23, color="#e87524", label="selected 16")
            ax.set(xlabel="Unconditional language sensitivity", ylabel="Absolute FM attribution", title=f"PG{layer}")
            ax.set_yscale("log")
            ax.legend(fontsize=8)
            active = features > 0
            conditional = (per_s*active).sum(0)/active.sum(0).clip(1)
            first_half = np.zeros(len(episodes), dtype=bool)
            for task in np.unique(tasks):
                group = np.unique(episodes[tasks == task])
                first_half |= (tasks == task) & np.isin(episodes, group[:max(1, len(group)//2)])
            overlap = None
            if first_half.any() and (~first_half).any():
                selections = []
                for mask in (first_half, ~first_half):
                    s = np.abs(effects[mask]).mean(0) * (1-per_s[mask].mean(0))
                    selections.append(set(np.argsort(-s)[:16].tolist()))
                overlap = len(selections[0] & selections[1])/16
            measurements["scoring"][str(layer)] = {
                "selected": selected.tolist(), "mean_unconditional_sensitivity": float(sensitivity[selected].mean()),
                "mean_active_conditional_sensitivity": float(conditional[selected].mean()),
                "split_episode_top16_overlap_fraction": overlap,
                "score_vs_attribution_spearman": float(spearmanr(rank_score[alive], attribution[alive]).statistic)}
        checks = score["attribution_checks"]
        for strength, color in ((.1, "#3498db"), (1., "#e87524")):
            chosen = [c for c in checks if c["strength"] == strength]
            x = np.array([c["predicted_loss_delta"] for c in chosen])
            y = np.array([c["actual_loss_delta"] for c in chosen])
            axes[2].scatter(x, y, s=25, color=color, label=f"{strength:g}× ablation")
            correlation = float(spearmanr(x, y).statistic) if len(x) > 2 and np.std(y) > 0 and np.std(x) > 0 else None
            measurements[f"attribution_ablation_{strength:g}"] = {"spearman": correlation,
                "sign_agreement": float((np.sign(x) == np.sign(y)).mean()), "n": len(x)}
        axes[2].axhline(0, color="#aaaaaa", linewidth=.5)
        axes[2].axvline(0, color="#aaaaaa", linewidth=.5)
        axes[2].set(xlabel="Predicted loss change", ylabel="Actual loss change", title="Attribution sanity check")
        axes[2].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "f1_scoring.png", dpi=160)
        plt.close(fig)
        report += ["## Candidate scoring and attribution checks", "", "![](figures/f1_scoring.png)", "",
                   f"Scored {score['examples']} ID frames, two counterfactual instructions each. "
                   "Primary ranking uses the requested literal M=A(1−mean S). "
                   "The pooled SAE normalization scale and reconstruction error are fixed when differentiating "
                   "a hypothetical decoder-coordinate edit. Attribution is compared with actual 0.1× and 1× "
                   "coefficient ablations on the first ten balanced scoring examples. "
                   "PG17 output has zero action-loss gradient by architecture and is excluded from CIFT.", "",
                   "| Layer | Mean S, unconditional | Mean S, active-only | Episode-half top-16 overlap |",
                   "|---|---:|---:|---:|"]
        for layer in (5, 11):
            m = measurements["scoring"][str(layer)]
            overlap = m["split_episode_top16_overlap_fraction"]
            report.append(f"| PG{layer} | {m['mean_unconditional_sensitivity']:.3f} | "
                          f"{m['mean_active_conditional_sensitivity']:.3f} | " +
                          (f"{overlap:.2f}" if overlap is not None else "not estimable") + " |")
        # Post-selection read-only comparison with previously frozen ID-only PG5
        # Cohen-d sets. Reuniting old + shared recovers the original task's 16
        # training-selected IDs, rather than its OOD-pair-specific exclusive set.
        historical_path = ROOT / "workspace/rq2/features_pg5_k16_env/features.json"
        if historical_path.exists():
            historical = load_json(historical_path)
            selected = set(score["5"]["feature_ids"])
            comparison = []
            for key, pair in historical["pairs"].items():
                if key.startswith("libero_object/"):
                    task_set = set(pair["old"]) | set(pair["shared"])
                    comparison.append({"task": pair["original"], "task_feature_count": len(task_set),
                                       "overlap": sorted(selected & task_set)})
            measurements["historical_PG5_task_feature_overlap"] = comparison
            measurements["historical_PG5_bundle_sha256"] = sha256(historical_path)
            report += ["", "**Post-selection PG5 comparison to the RQ1/RQ2 ID task-feature sets:**", "",
                       "| ID task | Shared IDs with the global top-16 M set |", "|---|---|"]
            report += [f"| {r['task']} | {r['overlap']} |" for r in comparison]
            report += ["", "These are previously frozen training-selected coordinates, not demonstrated "
                       "trap-causing features. Overlap is descriptive and was not used for Exp2 selection. "
                       "A global M set and ten task-specific sets answer different questions."]
        report += ["", "The unconditional sensitivity assigns zero to inactive examples; compare the active-only "
                   "column before interpreting apparent invariance. An action-relevant, prompt-invariant feature "
                   "could encode indispensable visual geometry. The score does not identify a causal shortcut by itself.", ""]
    if valid is not None:
        names = [a for a in ARMS if a in valid]
        means = [valid[a]["mean_loss"] if a == "frozen" else valid[a]["without_projection"]["mean_loss"] for a in names]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(range(len(names)), means, color=COLORS[:len(names)])
        ax.set_xticks(range(len(names)), [LABELS[ARMS.index(a)] for a in names], rotation=15)
        ax.set(ylabel="Held-out ID flow-matching loss", title="Primary validation: projection removed")
        fig.tight_layout()
        fig.savefig(figures / "f2_validation.png", dpi=160)
        plt.close(fig)
        report += ["## ID validation with projections removed", "", "![](figures/f2_validation.png)", "",
                   "| Model | FM loss | Frozen PG11 probe ID accuracy |", "|---|---:|---:|"]
        for arm, value in zip(names, means, strict=True):
            v = valid[arm] if arm == "frozen" else valid[arm]["without_projection"]
            report.append(f"| {LABELS[ARMS.index(arm)]} | {value:.6f} | {v['frozen_probe_accuracy']:.3f} |")
        report += ["", "FM loss is an offline diagnostic, not ID rollout success. Frozen probe drift must be "
                   "checked before interpreting an OOD probability change as better task reading.", ""]
    if results:
        suites = sorted({r["suite"] for r in results})
        fig, axes = plt.subplots(len(suites), 3, figsize=(13, 3.2*len(suites)), squeeze=False)
        report += ["## Paired behavioral evaluation", "", "![](figures/f3_behavior.png)", "",
                   "| Suite | Model | n | New-goal successes | Correct first lifts | Trap events | Trap + failure |",
                   "|---|---|---:|---:|---:|---:|---:|"]
        summaries = []
        for row, suite in enumerate(suites):
            for arm in ARMS:
                group = [r for r in results if r["suite"] == suite and r["arm"] == arm]
                if not group:
                    continue
                m = {key: sum(bool(r[key]) for r in group)
                     for key in ("success", "correct_first_lift", "trap_event", "trap_failure")}
                summaries.append({"suite": suite, "arm": arm, "n": len(group), **m})
                report.append(f"| {suite} | {LABELS[ARMS.index(arm)]} | {len(group)} | "
                              f"{m['success']} | {m['correct_first_lift']} | {m['trap_event']} | {m['trap_failure']} |")
            for col, (key, label) in enumerate((("success", "New-goal success"),
                                               ("correct_first_lift", "Correct first lift"),
                                               ("trap_event", "Trap event"))):
                group = [r for r in summaries if r["suite"] == suite]
                ax = axes[row, col]
                ax.bar(range(len(group)), [r[key]/r["n"] for r in group], color=COLORS[:len(group)])
                ax.set_xticks(range(len(group)), [LABELS[ARMS.index(r["arm"])] for r in group], rotation=25, fontsize=8)
                ax.set(ylim=(0, 1.12), title=suite+": "+label)
                for i, r in enumerate(group):
                    ax.text(i, r[key]/r["n"]+.03, f"{r[key]}/{r['n']}", ha="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / "f3_behavior.png", dpi=160)
        plt.close(fig)
        measurements["behavior"] = summaries
        frozen = {(r["suite"], r["task_id"], r["episode"]): r for r in results if r["arm"] == "frozen"}
        paired = {}
        for arm in ARMS[1:]:
            group = [r for r in results if r["arm"] == arm and r["suite"].endswith("_task")]
            pairs = [(frozen[(r["suite"], r["task_id"], r["episode"])], r) for r in group]
            paired[arm] = {"n": len(pairs),
                           "completion_rescues": sum(not a["success"] and b["success"] for a, b in pairs),
                           "completion_regressions": sum(a["success"] and not b["success"] for a, b in pairs),
                           "object_choice_rescues": sum(not a["correct_first_lift"] and b["correct_first_lift"] for a, b in pairs),
                           "object_choice_regressions": sum(a["correct_first_lift"] and not b["correct_first_lift"] for a, b in pairs)}
        measurements["paired_task_changes"] = paired
        report += ["", "All conditions share the simulator initial state and per-inference noise schedule. "
                   "First lift means >4 cm above the initial height and <12 cm from the gripper, excluding fixtures. "
                   "An unresolved/no-lift rollout is not counted as correct choice. Trap event and trap+failure "
                   "are separate so that placement improvements cannot masquerade as corrected object selection. "
                   "One state/noise per task is a pilot; the ten tasks are the independent task units.", ""]
    report += ["## Interpretation limits", "",
               "The SAE was trained on timestep-pooled activations. Its decoder directions can be applied to "
               "individual visual tokens, but they have not been validated as a token-level SAE dictionary. "
               "This pilot projects at PG11 on valid visual tokens only. This deployment has no discrete state "
               "tokens. It does not yet test AE projection, wider SAEs, a layer sweep, or base-vs-SFT PCA.", "",
               "Coefficient ablation preserves the fixed recipient SAE residual. Full raw-residual orthogonal "
               "projection during CIFT also removes any error/bias component in the selected span. These are "
               "different interventions. Blocking one span does not guarantee language use: scene information "
               "can reroute, and removing hooks at inference can reactivate existing pathways.", "",
               "Prompt-difference PCA explicitly selects instruction-sensitive directions; it is a diagnostic "
               "comparator, not an established shortcut estimator. Prior OOD experiments motivated PG11 and "
               "this development suite, so this is training-only score estimation, not an OOD-blind research process. "
               "A positive pilot requires independent states/noise and task validation before a generalization claim.", "",
               "Method reference: [Casademunt et al., CAFT](https://arxiv.org/html/2507.16795v1), "
               "which uses train-time projections and removes them at inference; its direction selection involves "
               "interpretation, and it does not establish automatic identification by the new M score. "
               "Implementation guidance: repository `SAE_Interp_Tutorial.md` (ARENA), attribution/ablation "
               "comparison and SAE error-term semantics.", "",
               "## Provenance", "",
               "Source snapshots and hashes are in `source/` and `manifest.json`; selected directions in "
               "`scoring.json` and `bases.pt`; paired losses in `validation_*.npz`; raw simulator/model traces "
               "under `evaluation/rollouts/`. Protected-artifact audits cover RQ1, RQ2, Exp1, activations, "
               "SAEs, indices and policy weights.", ""]
    inputs = [p for p in (run / "scoring.json", run / "validation.json", run / "evaluation/results.json",
                          run / "evaluation/config.json", run / "manifest.json") if p.exists()]
    measurements["input_sha256"] = {str(p.relative_to(run)): sha256(p) for p in inputs}
    write_json(run / "figure_data.json", measurements)
    (run / "EXP2_report.md").write_text("\n".join(report))
    print(f"Report: {run / 'EXP2_report.md'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    analyze(parser.parse_args().run_dir)
