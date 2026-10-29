"""One short English report and one plot for the Exp1 core pilot."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from experiments.exp1.common import output_path, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = output_path(args.run_dir)
    config = json.loads((run / "config.json").read_text())
    preflight = json.loads((run / "preflight.json").read_text())
    status = json.loads((run / "status.json").read_text())
    records = json.loads((run / "results.json").read_text())
    by_condition = {r["condition"]: r for r in records}
    if len(by_condition) != len(records) or len(records) > 6:
        raise ValueError("Exp1 is limited to six unique rollouts")
    input_paths = [run / name for name in ("config.json", "preflight.json", "status.json", "results.json")]
    approach = {}
    # Secondary, post-hoc diagnostic: SAME physical step, before any first lift.
    # It does not redefine the registered first-lift primary outcome.
    for r in records:
        path = run / r["artifact"]
        input_paths.extend([path, run / "rollouts" / (r["condition"] + ".diagnostics.json")])
        with np.load(path, allow_pickle=False) as x:
            if len(x["eef_pos_after"]) < 50:
                continue
            names = x["obj_names"].tolist()
            old, new = names.index(config["old_object"]), names.index(config["new_object"])
            eef = x["eef_pos_after"][49, :2]
            old_distance = float(np.linalg.norm(eef - x["obj_pos"][49, old, :2]))
            new_distance = float(np.linalg.norm(eef - x["obj_pos"][49, new, :2]))
            approach[r["condition"]] = {"step": 50, "old_xy_distance_m": old_distance,
                                       "new_xy_distance_m": new_distance,
                                       "new_minus_old_xy_distance_m": new_distance - old_distance}
    input_hashes = {str(p.relative_to(run)): sha256(p) for p in input_paths}
    if status["positive_controls_valid"] and len(records) == 6:
        dense, pooled, matched = [by_condition[c]["selected_original"] for c in ("dense", "pooled", "pooled_norm")]
        if not dense:
            interpretation = "Dense patching did not produce an original-object lift within the horizon; " \
                             "a no-lift outcome does not establish that the policy kept the new-object choice."
            dense_approach = approach.get("dense", {}).get("new_minus_old_xy_distance_m", 0)
            new_approach = approach.get("baseline_new", {}).get("new_minus_old_xy_distance_m", 0)
            if dense_approach > 0 and new_approach < 0:
                interpretation = "Dense PG11 patching redirected the approach toward the old object, " \
                                 "but produced no lift by the horizon. Pooled variants still lifted the new object. " \
                                 "This is partial evidence about approach mediation, not a successful old-object lift or memorization."
        elif pooled:
            interpretation = "Dense and pooled patches both induced the old-object choice: the mean direction is sufficient in this paired case."
        elif matched:
            interpretation = "Dense and norm-matched pooled patches induced the old-object choice, but raw pooled did not: edit magnitude is a viable explanation."
        else:
            interpretation = "Only dense patching induced the old-object choice: token-resolved information is a candidate explanation, subject to the recorded norm fidelity."
    else:
        interpretation = "The behavioral/site contrast gate was not met; the pilot stopped without a larger experiment."
    norm_fidelity = {}
    for r in records:
        path = run / "rollouts" / (r["condition"] + ".diagnostics.json")
        diagnostics = [d for call in json.loads(path.read_text()) for d in call]
        if r["condition"] == "self" and any(d["dense_delta_norm"] != 0 for d in diagnostics):
            raise ValueError("Self donor/recipient hidden coordinates were not actually identical")
        active = [d["realized_vs_dense_norm"] for d in diagnostics if d["dense_delta_norm"] > 1e-8]
        norm_fidelity[r["condition"]] = {
            "mean_realized_vs_dense_norm": float(np.mean(active)) if active else None,
            "n_active_hook_calls": len(active),
        }
    write_json(run / "summary.json", {"status": status, "interpretation": interpretation,
                                       "results": records, "norm_fidelity": norm_fidelity,
                                       "secondary_posthoc_approach": approach})
    os.environ["MPLCONFIGDIR"] = str(run / ".mplcache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, blue, orange, teal = "#fcfcfb", "#2a78d6", "#eb6834", "#1baf7a"
    plt.rcParams.update({"figure.facecolor": surface, "axes.facecolor": surface,
                         "savefig.facecolor": surface, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    xs = np.arange(2)
    for offset, mode, color in [(-0.25, "dense", blue), (0, "pooled", orange), (0.25, "pooled_norm", teal)]:
        values = [preflight["probes"][s + "_" + mode]["old_action_projection"] for s in ("pg11", "ae11")]
        axes[0].bar(xs + offset, values, width=0.24, color=color, label=mode.replace("_", " "))
    axes[0].axhline(0, color="#898781", linewidth=1)
    axes[0].axhline(1, color="#898781", linestyle=":", linewidth=1)
    axes[0].set_xticks(xs, ["PG11", "AE11"])
    axes[0].set_ylabel("Projection onto old−new XYZ action contrast")
    axes[0].set_title("Fixed-observation site probe")
    axes[0].legend(fontsize=9)
    axes[0].text(0, -0.22, "0 = new-prompt baseline; 1 = old-prompt baseline", transform=axes[0].transAxes, fontsize=9)
    palette = {"baseline_new": "#898781", "baseline_old": "#b5b4ad", "self": "#7057a3",
               "dense": blue, "pooled": orange, "pooled_norm": teal}
    for r in records:
        with np.load(run / r["artifact"], allow_pickle=False) as x:
            eef = x["eef_pos_after"]
            axes[1].plot(eef[:, 0], eef[:, 1], color=palette[r["condition"]], label=r["condition"].replace("_", " "))
    with np.load(run / records[0]["artifact"], allow_pickle=False) as x:
        names = x["obj_names"].tolist()
        for name, label in [(config["old_object"], "Old: cream cheese"), (config["new_object"], "New: alphabet soup")]:
            position = x["initial_object_positions"][names.index(name)]
            axes[1].scatter(position[0], position[1], color="#0b0b0b", marker="X", s=65, zorder=5)
            axes[1].annotate(label, position[:2], xytext=(4, 6), textcoords="offset points", fontsize=9)
    axes[1].set_xlabel("World x (m)")
    axes[1].set_ylabel("World y (m)")
    axes[1].set_title("EEF paths until first lift / horizon")
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].set_aspect("equal", adjustable="datalim")
    fig.suptitle("Exp1 core: can an old-prompt patch change object choice?", x=0.06, ha="left", fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.2, wspace=0.32)
    fig.savefig(output_path(run / "core_result.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    resources = json.loads((run / "manifest.json").read_text())["resources"]
    lines = ["# Exp1: causal object-choice localization (core pilot)", "",
             f"**Bottom line:** {interpretation}", "",
             f"**Scope:** one task (task 1), one initial state, one explicit noise base, {len(records)} rollouts. "
             "No SAE training, attribution sweep, task sweep, or task-completion SR measurement.", "",
             f"**Allocation:** Slurm job {resources['job_id']}, partition `{resources['partition']}`, "
             f"{resources['gpu_name']}, {resources['gpu_memory_gib']:.1f} GiB VRAM; requested 8 CPUs / 64 GB RAM.", "",
             "## Core result", "", "![Fixed-state probes and pre-grasp trajectories](core_result.png)", "",
             "| Condition | First lifted object | Stop step | Old-object lift observed? | XY gap at step 50 (cm) |",
             "|---|---|---:|---|---:|",
             *[f"| {r['condition']} | {r['first_lift'] or 'none'} | {r['steps']} | "
               f"{'yes' if r['selected_original'] else 'no'} | "
               f"{100*approach[r['condition']]['new_minus_old_xy_distance_m']:+.1f} |"
               if r['condition'] in approach else
               f"| {r['condition']} | {r['first_lift'] or 'none'} | {r['steps']} | "
               f"{'yes' if r['selected_original'] else 'no'} | n/a |" for r in records], "",
             "**Primary endpoint:** first lifted object. `none` means unresolved/no lift within the horizon, "
             "not proof of choosing either object. **Secondary (post hoc):** XY gap is distance to NEW "
             "object minus distance to OLD object at the same physical step 50, before every recorded "
             "first lift here. Positive means EEF is closer to the old object; it is not counted as a lift success.", "",
             f"**Selected site:** {status['selected_site'] or 'none'}. Selection was fixed before intervention "
             "rollouts: greatest dense-patch projection onto the initial old−new action contrast, minimum 0.05. "
             "The score is a command contrast, not a proof of object choice.", "",
             "## What was tested", "",
             "- Recipient requests **alphabet soup**; donor requests **cream cheese**, using the SAME images, "
             "robot state, and explicit diffusion noise at every inference call.",
             "- PG11 and AE11 are compared only in the initial probe; if gates pass, four intervention "
             "rollouts use the ONE selected site.",
             "- Dense patch replaces donor coordinates at matching valid token positions. PG patches use the "
             "intersection of donor/recipient valid masks; padding/unmatched positions are untouched.",
             "- Pooled patch broadcasts the mean donor−recipient delta over the SAME eligible positions. "
             "Its total mean shift matches dense, but its norm can be smaller.",
             "- Pooled-norm patch rescales that direction to the dense delta's Frobenius norm at the same "
             "recipient. BF16 realized norm fidelity is logged in `summary.json`.",
             "- AE patches align each donor/recipient denoising-call index; no last-call donor is reused at all times.",
             f"- Patching continues through the decision window; each rollout stops at first lift or {config['max_steps']} actions. "
             "Goal-completion fields at that early stop are NOT success rates.", "",
             "## Gates and limits", "",
             f"- Recorded initial equality gates (max absolute action delta): `{preflight['identity_max_abs']}`.",
             f"- PG17 output negative control: max action delta "
             f"`{preflight['probes']['pg17_dense']['full_action_max_abs_delta']}` (required zero).",
             f"- Behavioral positive controls valid: `{status['positive_controls_valid']}`. "
             "Invalid controls/site contrast stop the study after at most two baseline rollouts.",
             "- Closed-loop self-patch equality was asserted if all six rollouts completed.",
             "- This tests prompt-conditioned object-choice mediation, NOT training-example memorization "
             "or improvement on failing tasks. First lift is a geometry proxy; raw images/poses are saved.",
             "- One state/noise, exploratory site selection, no independent replication; no generalization claim.", "",
             "**Norm check:** pooled-norm realized/dense norm ratios are measured against the hypothetical "
             "dense patch at each control's OWN recipient state, not against another rollout's different "
             "trajectory. Consult `summary.json` for realized BF16 fidelity.", "",
             "**Protocol correction:** the initial 120-action baseline gate truncated observation before "
             "the historical old-prompt lift (step 133). The horizon was fixed at 180 before intervention "
             "rollouts, with no dose/task/horizon sweep. Earlier cancelled/truncated jobs are not pooled "
             "with this result.", "",
             "**Next:** review this one result before authorizing SAE-feature ranking or any wider experiment. "
             "See `protected_artifacts_audit.json` for the RQ1/RQ2 input/source/output metadata audit. "
             "Configuration, source snapshots and numerical inputs are saved in this job directory.", ""]
    output_path(run / "EXP1_report.md").write_text("\n".join(lines))
    if any(sha256(run / p) != h for p, h in input_hashes.items()):
        raise ValueError("Raw Exp1 inputs changed during analysis")
    write_json(run / "analysis_revision.json", {
        "script_sha256": sha256(Path(__file__)), "input_sha256": input_hashes,
        "primary_outcomes_unchanged": True,
        "secondary_diagnostic": "post hoc same-step-50 XY approach; not a primary choice success",
        "previous_auto_analysis": "analysis_v1/; execution source remains archived in source/",
    })
    print(interpretation, flush=True)


if __name__ == "__main__":
    main()
