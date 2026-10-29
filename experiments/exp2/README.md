# Exp2 — ID-only counterfactual scoring and CIFT

This experiment tests whether the proposed shortcut score and post-SFT projection
method work. A useful hypothesis is not assumed to be true before the experiment.
Code lives here; every generated artifact goes below `workspace/exp2/`. Existing
RQ1/RQ2/Exp1 sources, reports, datasets, policy weights, activations, SAEs and
indices are protected read-only inputs, checked with metadata audits.

## What we learned from the three reports

- RQ1: rarity-based feature labels do not predict failure within tasks. Original-task
  feature persistence correlates with trap rate across tasks, but does not identify
  a causal memorization mechanism. SAE reconstruction shift is not a density score.
- RQ2: PG5 coefficient editing failed to fix old-object selection. Its completion
  rescue also occurred with controls. This rejects neither all test-time editing
  nor all SAE interventions.
- Exp1 job 58962: dense PG11 prompt-donor patching changed approach, but yielded no
  lift; pooled variants lifted the new object. Broadcasting a delta preserves
  token differences algebraically. It has not been established that pooling
  destroys localization attention or that a correct circuit cannot be recovered.

## Scientific checks on the proposed method

1. **M is a candidate score, not an identified shortcut.** Action-relevant visual
   geometry may appropriately remain prompt-invariant. Absolute FM loss
   attribution is local, signed effects are lost, and small gradients can occur
   when a model already fits a demonstration. We compare predicted and actual
   coefficient ablations at two strengths.
   Existing pooled logs contain no FM-loss gradients; computing A requires new
   forward/backward passes, so the suggested check is not literally zero GPU work.
2. **Unconditional sensitivity is biased by feature inactivity.** The literal
   proposed score is the primary ranking. Active-conditional sensitivity and a
   jointly averaged alternative are saved as diagnostics, without outcome tuning.
3. **Pooled SAE directions are not a validated token dictionary.** We apply their
   span selectively to valid visual tokens, but do not claim token-level SAE
   decomposition. The deployed pi05_libero config has no discrete state tokens.
4. **Projection and coefficient ablation differ.** Coefficient edits use fixed
   recipient scale and reconstruction error. Raw-residual orthogonal projection
   can remove reconstruction-error and bias components in the span too.
5. **Blocking a span does not force instruction following.** The model may route
   scene shortcuts elsewhere; removing training hooks can restore old pathways.
   Compare against SFT and matched random SAE projection, then evaluate with all
   hooks removed.
6. **Prompt-difference PCA is instruction-sensitive.** It is a diagnostic comparator,
   not an established shortcut estimator. Base-vs-SFT PCA is not run here.
7. **A linear probe measures decodability.** A lower P(original) could reflect
   representation drift. We evaluate frozen probes on held-out ID episodes and
   use physical object choice plus completion as primary endpoints.
8. **Prior OOD knowledge motivated this pilot.** Scoring/training use ID only;
   PG11 and the object suite were motivated by earlier OOD development results.
   Do not describe the overall research process as OOD-blind.

CAFT reference: [Casademunt et al.](https://arxiv.org/html/2507.16795v1).
CAFT uses projection during training and removes it at inference; its success
does not establish automatic shortcut identification by M. The local
`SAE_Interp_Tutorial.md` (ARENA) guides fixed-error attribution and actual ablation
checks. RQ1/RQ2/Exp1 report figures were reviewed before this implementation.

## Pilot design

Only the ID `physical-intelligence/libero` cache is read. All ten object tasks are
included, independent of which failed in RQ2. For each task: first 24 demos × 6
frames for rectification; first 4 of those demos × 2 early frames for scoring;
last 4 demos × 3 frames for validation. These are episode-disjoint Exp2 validation
sets, not demonstrations held out of the original policy's SFT. No downloads.
Counterfactuals are two other ID instructions whose target exists in the ID scene;
no counterfactual action labels or OOD mappings are used for scoring.

The technical smoke uses 20 balanced scoring samples, two optimizer steps and
two branches (SFT and SAE-M), solely to check GPU memory and implementation.
It is not an efficacy result. The full pilot uses all 80 scoring samples and four
500-step **full-parameter SFT** branches:

| Branch | PG11 visual-token projection during training |
|---|---|
| sft | None |
| sae_m | Orthogonal span of top-16 literal M-score decoder directions |
| random_sae | 16 non-target SAE directions, matched for ID activity/frequency |
| pca_promptdiff | Top-16 centered PCs of ID visual-token prompt differences |

The native OpenPI PyTorch FM forward, checkpointing, BF16 parameter policy and
AdamW hyperparameters are retained. Exp2 has a dedicated loop for paired random
streams, a suite-balanced subset, projection callbacks, effective batch 2, and a
short 25-step warmup to peak LR 5e-5. This is a rectification pilot, not a reproduction
of the original batch-256/30k-step SFT schedule. LoRA was initially considered from
the methodology's example, but the custom adapter implementation was removed
before any GPU training. All branches restore exactly the same original weights.

Initial closed-loop evaluation: all ten tasks × ID/task/swap × one initial
state/noise × five models including frozen = **150 paired rollouts**. Horizon 280,
replan every 5 actions, simulator-canonical lowercase prompt. Report completion,
correct first lift, trap event and trap+terminal-failure separately. No lift is
unresolved, not correct object selection. There is no seed sweep or benchmark-wide
claim; promising effects require independent confirmation states/noise/tasks.

## Why special instrumentation is necessary

OpenPI's joint FM forward bypasses decoder-block `__call__`, so the old RQ2 hooks
do not execute in that path. A next-layernorm pre-hook is also insufficient:
it would leave the residual skip path untouched. `model.py` creates an in-memory
copy of the installed joint forward, inserting exactly one callback immediately
after the second residual addition. The source structure is asserted and the
instrumented code is saved. Cached inference uses ordinary block-output hooks.
The OpenPI checkout and installed packages are unchanged.

CPU tests check full-residual instrumentation and its gradient, forward/backward
projection, and exact preservation of unedited BF16 tokens. GPU smoke gates
require exact passive-hook action equality, repeated-noise equality, and exact
native-versus-instrumented FM loss equality. Every branch must also reset to exact
baseline actions before optimization. PG17 output is an architectural zero-gradient
control; it is not selected for intervention.

Scoring files store per-example `sample_sensitivity` (samples × features) separately
from mean `sensitivity` (features). Both producer and consumer validate the schema.
`test_pipeline.py` exercises scoring, serialization, probe fitting, all four
training branches, checkpoint reloads, and report/figure generation on CPU using
temporary synthetic inputs and tiny models. This is a software integration test,
not CIFT efficacy evidence. It was added after job 60876 passed native identity
gates and scored 20 frames but failed on duplicate sensitivity field names.

`test_native_forward.py` additionally constructs two-layer, 16-dimensional Gemmas
and runs the actual OpenPI FM forward, cached inference, and checkpointed backward
on CPU. It reproduces NumPy normalization's float64 action output and verifies
the explicit float32 cast used by OpenPI's trainer. This regression check was added
after smoke job 60773 failed before scoring on an action dtype mismatch; its
dependent full job 60777 never ran. No scientific result is inferred from that job.

## Jobs and outputs

Use **main only**, one H100 80 GB GPU, 8 CPUs, 64 GB host RAM. Leave Slurm's
`CUDA_VISIBLE_DEVICES` untouched. MIG CUDA failed during setup and is not used.

```bash
env sbatch --parsable --partition=main --ntasks=1 --cpus-per-task=8 --mem=64G \
  --gres=gpu:1 --time=00:10:00 experiments/exp2/smoke.slurm
# After substituting the actual successful smoke job number:
env sbatch --parsable --partition=main --ntasks=1 --cpus-per-task=8 --mem=64G \
  --gres=gpu:1 --time=04:00:00 --dependency=afterok:SMOKE_JOB \
  --kill-on-invalid-dep=yes \
  --export=ALL,EXP2_SMOKE_RUN=workspace/exp2/smoke_job_SMOKE_JOB experiments/exp2/full.slurm
```

Primary report: `workspace/exp2/cift_job_JOBID/EXP2_report.md`. The full job runs
scoring → full SFT → paired rollouts → figure/report generation automatically,
and refuses to start without a completed full-SFT technical smoke. Sources,
configuration, feature IDs, checkpoints, paired losses, simulator traces and
protected-input audits are saved under that run. Reports label incomplete
artifacts as partial rather than reporting them as a completed experiment.

```bash
env PYTHONDONTWRITEBYTECODE=1 openpi/.venv/bin/python -m unittest experiments.exp2.test_model -v
env PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  openpi/.venv/bin/python -m unittest experiments.exp2.test_native_forward -v
env PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  openpi/.venv/bin/python -m unittest experiments.exp2.test_pipeline -v
env PYTHONDONTWRITEBYTECODE=1 openpi/.venv/bin/ruff check experiments/exp2
```
