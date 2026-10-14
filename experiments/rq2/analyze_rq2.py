"""Paired pilot analysis: task outcomes, intervention fidelity, and identity gates."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from experiments.rq2.common import output_path, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = output_path(args.run_dir)
    records = [json.loads(line) for path in sorted((run / "rollouts").glob("*/results.jsonl"))
               for line in path.read_text().splitlines() if line.strip()]
    by_condition = defaultdict(list)
    paired = defaultdict(dict)
    for record in records:
        by_condition[record["condition"]].append(record)
        key = (record["task_id"], record["episode"], record["noise_base"])
        if record["condition"] in paired[key]:
            raise ValueError("Duplicate paired condition")
        paired[key][record["condition"]] = record
    if not records:
        raise ValueError("No completed rollouts")
    expected = set(next(iter(paired.values())))
    if "baseline" not in expected or any(set(p) != expected for p in paired.values()):
        raise ValueError("Incomplete paired pilot; finish missing conditions before summarizing")
    summary = {}
    for condition, group in sorted(by_condition.items()):
        outcome = {
            "n": len(group), "successes": sum(r["success"] for r in group),
            "sr": float(np.mean([r["success"] for r in group])),
            "old_task_events": sum(r["old_task_event"] for r in group),
            "trap_failures": sum(r["trap_failure"] for r in group),
            "first_lift_old": sum(r["lifted_old_target"] for r in group),
            "first_lift_new": sum(r["lifted_new_target"] for r in group),
        }
        if condition != "old_prompt":
            pairs = [(p["baseline"], p[condition]) for p in paired.values()]
            outcome["rescues"] = sum(not a["success"] and b["success"] for a, b in pairs)
            outcome["regressions"] = sum(a["success"] and not b["success"] for a, b in pairs)
            outcome["paired_sr_delta"] = float(np.mean([int(b["success"]) - int(a["success"]) for a, b in pairs]))
        summary[condition] = outcome
    verification = [json.loads(p.read_text()) for p in sorted((run / "verification").glob("*.json"))]
    identity_max = max(v for report in verification for v in report["identity_checks_max_abs"].values())
    if identity_max != 0:
        raise ValueError("Identity verification failed")
    control_fidelity = {}
    for condition in ("random_suppress", "general_suppress", "random_boost", "general_boost"):
        ratios, realized_ratios, total = [], [], 0
        for r in by_condition[condition]:
            with np.load(run / r["artifact"]) as data:
                mask = data["edit/control_target_norm"] > 1e-8
                ratios.extend(data["edit/control_norm_ratio"][mask].tolist())
                realized_mask = data["edit/control_realized_target_norm"] > 1e-8
                realized_ratios.extend(data["edit/control_realized_norm_ratio"][realized_mask].tolist())
                total += int(mask.sum())
        control_fidelity[condition] = {
            "nonzero_target_calls": total,
            "mean_norm_ratio": float(np.mean(ratios)) if ratios else None,
            "min_norm_ratio": float(min(ratios)) if ratios else None,
            "fraction_norm_matched_5pct": float(np.mean(np.abs(np.asarray(ratios) - 1) < 0.05)) if ratios else None,
            "mean_realized_norm_ratio": float(np.mean(realized_ratios)) if realized_ratios else None,
            "fraction_realized_norm_matched_5pct": float(np.mean(np.abs(np.asarray(realized_ratios) - 1) < 0.05))
            if realized_ratios else None,
        }
    result = {"n_rollouts": len(records), "n_pairs": len(paired), "conditions": summary,
              "identity_max_abs": identity_max, "control_fidelity": control_fidelity,
              "verification": verification}
    write_json(run / "summary.json", result)
    lines = ["# RQ2 pooled PG5 pilot", "",
             f"{len(records)} rollouts; {len(paired)} paired initial-state/noise blocks. "
             "This is an exploratory task-perturbation pilot, not a benchmark-wide SR estimate.", "",
             "All passive-hook, zero-edit, and deterministic-repeat gates: exact action equality.", "",
             "| Condition | Success | SR | Old-task event | Trap failure | Rescues / regressions |",
             "|---|---:|---:|---:|---:|---:|"]
    for condition in ("baseline", "suppress_old", "boost_new", "patch_old", "random_suppress",
                      "general_suppress", "random_boost", "general_boost", "old_prompt"):
        if condition not in summary:
            continue
        s = summary[condition]
        change = f"{s['rescues']} / {s['regressions']}" if "rescues" in s else "positive control"
        lines.append(f"| {condition} | {s['successes']}/{s['n']} | {100*s['sr']:.1f}% | "
                     f"{s['old_task_events']} | {s['trap_failures']} | {change} |")
    lines += ["", "`old_prompt` evaluates the ORIGINAL goal and is not LIBERO-PRO success. "
              "Old-task event and final failure are separate; a rollout can recover after an old-task event.",
              "", "## Per-task outcomes", "",
              "| Task | Condition | Success | First lifted object(s) |", "|---|---|---:|---|"]
    for task in sorted({r["task_id"] for r in records}):
        for condition in sorted(by_condition):
            group = [r for r in by_condition[condition] if r["task_id"] == task]
            if group:
                lifted = ", ".join(sorted({str(r["first_lift"]) for r in group}))
                lines.append(f"| {task} | {condition} | {sum(r['success'] for r in group)}/{len(group)} | {lifted} |")
    lines += ["", "## Control fidelity", "",
              "Suppression preserves nonnegative coordinates; insufficient control energy is capped, "
              "not compensated by making latents negative. Inspect these ratios before causal claims.", "",
              "| Control | Nonzero target calls | Mean decoded-norm ratio | Within 5% | Mean realized BF16 ratio |",
              "|---|---:|---:|---:|---:|"]
    for condition, c in control_fidelity.items():
        mean = f"{c['mean_norm_ratio']:.3f}" if c["mean_norm_ratio"] is not None else "n/a"
        fraction = f"{100*c['fraction_norm_matched_5pct']:.1f}%" if c["fraction_norm_matched_5pct"] is not None else "n/a"
        realized = f"{c['mean_realized_norm_ratio']:.3f}" if c["mean_realized_norm_ratio"] is not None else "n/a"
        lines.append(f"| {condition} | {c['nonzero_target_calls']} | {mean} | {fraction} | {realized} |")
    lines += ["", "## Scope and limitations", "",
              "- Frozen pi0.5 and existing deployment-compatible SAE; task-selective feature sets "
              "are candidates, not independently validated memorization features.",
              "- Pooled PG5 coefficients are edited with recipient scale/error retained; the same direction "
              "is broadcast over tokens. This is not a token-specific SAE or DrVLA's projection ablation.",
              "- Early six inference calls (30 simulator actions), unless overridden in the manifest; "
              "new-task boost uses training-demo mean coefficients. No tuning on this pilot's outcomes.",
              "- `patch_old` uses the SAME perturbed image/state/noise with the original prompt. "
              "It tests prompt-conditioned feature mediation, not a donor from an ID layout.",
              "- Task IDs were chosen from prior exploratory RQ1 evidence. Repeated episodes within a task "
              "are not independent tasks; no significance or generalized causal claim from this small pilot.",
              "- Nonorthogonal SAE directions and BF16 rounding can move re-encoded features beyond the "
              "target set. Intended coefficients, realized coefficients, and actual delta norms are saved.", ""]
    (run / "RQ2_report.md").write_text("\n".join(lines))
    print(json.dumps({"run": str(run), "n_rollouts": len(records), "conditions": summary}, indent=2))


if __name__ == "__main__":
    main()
