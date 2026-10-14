# RQ2: Do old-task SAE features causally drive memory-trap failures?

*π0.5-LIBERO on LIBERO-PRO, deployment-compatible Dr. VLA SAE. First intervention pilot, run 09/10/2026; report revised 13/10/2026.*

**Setup:** 108 logged rollouts in two separate runs: 72 with raw BDDL capitalization and 36 with env-canonical prompts. The **primary results** use the 36-rollout canonical run: four selected tasks, one initial state and one noise base per task, nine conditions. These are exploratory paired results, **not benchmark-wide success rates**.

**Memory-trap candidate:** the robot operates on the original task's object despite the new instruction. An old-task event and terminal failure are recorded separately; this behavioral label does not itself establish memorization.

**Intervention:** frozen π0.5 and pooled PG5 SAE (1× expansion, SAE k = 100). Select top-16 task-discriminative features per training task using Cohen's d, then remove IDs shared by the old/new sets. Suppress old coefficients, boost new coefficients to an ID-task mean floor, or patch old coefficients from a **same-image/state/noise, old-prompt donor**. Edit only the first six inference calls / 30 executed actions; replan every five actions, horizon 280. Controls use random or classifier-general features. No outcome-driven retuning.

## TL;DR

| | |
|---|---|
| ❌ | **None of the three old-object failures is rescued** by the selected PG5 interventions (Fig. 1, 2). |
| ⚠️ | Success rises from 0/4 to 1/4, but **every random/general control rescues the same rollout**; this is not a feature-specific memorization benefit (Fig. 1, 6). |
| ✅ | The hook works: identity gates are exact, and suppress-old reduces re-encoded target activity by approximately 79–94% (Fig. 3). |
| 💡 | Targeted edits change initial actions, but much less than changing the full task prompt (Fig. 4). |
| ⚠️ | Intended control norms match; BF16 rounding weakens realized suppression norm matching (Fig. 4). |
| ⚠️ | **Prompt capitalization matters** for baseline completion. Only matched states are compared across protocols (Fig. 5). |

## 1. The selected edits do not rescue old-object failures

![Figure 1. Canonical paired new-goal successes and old-object failures by condition.](../../workspace/rq2/figures/f1_paired_outcomes.png)

- Baseline achieves **0/4** new-goal successes; suppress-old, boost-new, patch-old and each random/general control achieve **1/4**.
- Old-object failures remain **3/4 in every condition**. The same three tasks fail; aggregate SR alone would conceal this null result.
- The old-prompt positive control, omitted from the plot, achieves **3/4 original-goal successes**. That is not LIBERO-PRO success.
- With four deliberately selected tasks and one canonical rollout per task/condition, the 25-point difference is descriptive, not evidence of a transferable improvement.

## 2. Every edit rescues the same task, not a different object choice

![Figure 2. Paired task-level outcomes and original-to-instructed object mapping.](../../workspace/rq2/figures/f2_outcomes_per_task.png)

- **Task 1:** cream cheese → alphabet soup. Baseline already lifts alphabet soup, but does not complete placement. Every targeted edit and control completes the new goal.
- **Tasks 2, 4 and 5:** salad dressing → tomato sauce, ketchup → milk, and tomato sauce → BBQ sauce. The first lifted object remains the original object in every new-prompt condition.
- Thus the rescue is **not a correction of old-object selection**. This pilot supplies no positive causal evidence that removing these old-task coordinates removes a memory trap.

## 3. The null behavioral result is not an identity edit

![Figure 3. Re-encoded old-task activity and realized hidden-state edit size in canonical suppress-old rollouts.](../../workspace/rq2/figures/f3_suppression_fidelity.png)

- All **12 paired blocks across both runs** pass exact action-equality gates: unhooked versus passive baseline, identity versus baseline, and repeated baseline with the same diffusion noise.
- Intended old coefficients are zero. After re-encoding, **5.5–21.2%** of summed old-target activity remains: approximately **79–94% reduction**, measured over the first six edited calls.
- Realized edits average only **0.077–0.761% of pooled hidden norm**. The null result therefore does not rule out insufficient strength/window, downstream recovery, or the wrong causal layer/feature set.
- The edit preserves the recipient SAE reconstruction error and normalization in the ideal arithmetic, broadcasting the same decoder delta over tokens: $h'_i=h_i+s(x)W_{\mathrm{dec}}(f'-f)$, where $x$ is the pooled recipient activation. BF16 rounding and nonorthogonal directions/TopK mean re-encoded features need not equal the intended coefficients.

## 4. Action changes are real, but control fidelity is imperfect

![Figure 4. Same-observation action changes relative to the old-prompt contrast, and intended versus realized control norm matching.](../../workspace/rq2/figures/f4_action_and_control_fidelity.png)

- At the initial fixed observation/noise, targeted edits produce nonzero action changes of **1.5–14.6% of the full old-prompt contrast**. The metric is the L2 change in the first five XYZ action commands—not robot displacement or an attribution-based share of action responsibility.
- Controls match the **intended decoded perturbation norm at each recipient**. If a fixed suppression set cannot supply enough energy, the code selects top-active features from a predeclared candidate pool and logs the actual IDs. These are adaptive controls, not fixed random sets throughout the trajectory.
- After BF16 rounding, random/general suppression mean norm ratios are **0.941/0.965**; only **48%/42%** of calls fall within ±5% of the corresponding target norm. Boost controls are closer (**0.996/0.999**).
- Consequently, nominal norm matching must not be described as perfect physical matching. No feature-specific causal benefit is claimed.

## 5. Prompt normalization changes baseline completion

![Figure 5. Matched-state raw-case versus canonical success counts, and capitalization-induced initial action changes.](../../workspace/rq2/figures/f5_prompt_protocol.png)

- LIBERO lowercases BDDL before extracting language. The first bundle used raw text, **“Pick …”**; the confirmation uses the actual env form, **“pick …”**, for recipient and donor.
- Compare **initial state 0 only**, with the same images/state, diffusion noise and feature IDs/means: raw-case baseline succeeds on **1/4** tasks; canonical baseline on **0/4**. The difference is task 1's completion, not object choice.
- The full raw-case run achieves **2/8** successes for baseline and every edit/control, with zero rescues/regressions; its old-prompt control achieves **7/8 original-goal successes**.
- Do not pool the runs as 108 independent samples, or compare 2/8 directly with 0/4 as a matched protocol effect. The raw run/bundle were preserved; canonical prompts were derived into a new bundle without changing feature selection.

## 6. The apparent SR gain is consistent with a placement/release rescue

![Figure 6. Task-1 target-object height and post-action distance from the end-effector; baseline, boost-new and random boost.](../../workspace/rq2/figures/f6_task1_trajectory.png)

- All three traces concern **alphabet soup**, the instructed object; no old-task event occurs.
- At step **280**, baseline still has no new-goal success. The object is approximately **12 mm** from the end-effector and **180 mm above the basket body origin**.
- Boost-new reaches the goal at step **190**; random boost at step **158**. The object moves farther from the end-effector as the goal becomes true.
- This supports a **placement/release interpretation**, not a contact-state diagnosis. The same rescue with random controls prevents attributing it specifically to reduced memorization.

## Next steps

1. **Select action-relevant candidates before another rollout sweep.** Combine training-only task selectivity with actual same-observation ablation/patching effects; probe PG17 or the action expert, retaining the identity/error-preservation gates.
2. **Separate layer, dose and window tests.** Compare the first-30-action window with longer edits without simultaneously changing layer, K and strength; improve realized-norm control matching.
3. **Require task-selection improvement beyond controls.** If old-object errors decrease, replicate on additional initial states/noise seeds and held-out tasks, then check ID success rates for regressions.
4. **Test the stronger memorization hypothesis separately.** Add held-out feature scoring and matched position-shift / ID-layout donor experiments before claiming memorized-feature causation or moving to RQ3.

*Limitations:* four selected tasks, one canonical initial state/noise base per task; task-discriminative features are not independently validated episodic-memory features; only pooled PG5 and a 30-action edit window; no position-shift or ID-regression test; prompt-conditioned patching is not an ID-layout insertion and lacks matched donor-patch controls. The classifier's “memorized” label alone does not resolve these limitations.

*Verification:* six CPU unit tests and Ruff passed during implementation. Both runs' source snapshots match their execution-manifest hashes. The two protected-artifact metadata audits each checked **54,453** pre-existing input/RQ1 files without size/mtime changes; selected scientific inputs also have SHA256 hashes. Figure generation is read-only and verifies that plotted inputs are unchanged.

*Code:* [experiments/rq2/](README.md); figures: [make_figures.py](make_figures.py). Reproduce with `env openpi/.venv/bin/python -m experiments.rq2.make_figures`.

*Data:* [canonical run](../../workspace/rq2/pilot_pg5_k16_a1_first6_envcase_20261009/summary.json), [raw-case run](../../workspace/rq2/pilot_pg5_k16_a1_first6_20261009/summary.json), and [raw-case protocol note](../../workspace/rq2/pilot_pg5_k16_a1_first6_20261009/PROTOCOL_NOTE.md). Plotted measurements: [figure_data.json](../../workspace/rq2/figures/figure_data.json); input/output hashes: [figure_manifest.json](../../workspace/rq2/figures/figure_manifest.json). Per-run reports/logs/NPZ files are unchanged. The previous report is [archived](../../workspace/rq2/report_archive/20261013/RQ2_report_before_english.md).
