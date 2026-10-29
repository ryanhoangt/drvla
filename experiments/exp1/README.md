# Exp1 — one core object-choice localization pilot

This is deliberately small and separate from RQ1/RQ2. It is NOT a layer/task/dose
sweep and does not train SAEs or run attribution patching yet.

Question: **Can an old-prompt activation patch change the selected object, and
does its pooled mean direction retain that effect?**

Only LIBERO-PRO object-task 1, initial state 0, noise base 7. Recipient instruction:
alphabet soup; donor instruction: cream cheese. Same images/state/noise at each
paired inference. Both prompts use the exact lowercase simulator language.

1. Initial equality gates and fixed-observation PG11/AE11 dense/pooled/norm-matched
   probes. PG17 output is a negative connectivity control (required zero action effect).
2. Choose ONE site by greatest dense old−new XYZ command projection, with a fixed
   minimum 0.05. This is exploratory localization, not a claim of object-choice causation.
3. Two baseline rollouts verify the actual first-lift contrast. If invalid, STOP.
4. Four rollouts at that one site: self-patch, dense donor patch, pooled donor
   delta, and pooled delta matched to dense norm. Maximum SIX rollouts in total.

Each rollout stops at first lift or 180 executed actions, with replanning every
five actions. Patching covers the decision window, not just t=0 / 30 actions.
The primary outcome is the first lifted object; **no task-completion SR** is estimated.
The initial 120-action gate was too short (prior RQ2 old-prompt lift was step 133).
The decision horizon was corrected to 180 before any of the four intervention
rollouts were run. This is not a horizon sweep; old job outputs remain separate.
AE donors are aligned by denoising-call index; prefix patches exclude padding and
unmatched valid-mask positions. Norm fidelity after BF16 rounding is recorded.

## Submit (MIG ONLY)

```bash
env mkdir -p workspace/exp1/slurm
env sbatch --parsable --partition=mig --ntasks=1 --cpus-per-task=8 --mem=64G \
  --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1 --time=00:25:00 experiments/exp1/core.slurm
```

Requests one typed H100 `3g.40gb` MIG, 8 CPUs, 64 GB RAM, 25-minute maximum.
The job validates the visible CUDA device has 35–43 GiB VRAM and partition=mig.
Resource flags are repeated on the command line because inherited `SBATCH_*`
environment variables override script directives (this session had 120 GB set).
It never replaces Slurm's `CUDA_VISIBLE_DEVICES`. `srun` inside sbatch binds the
one job step to its allocated cores. No GPU is used on worker-0.

Read only **one report**:
`workspace/exp1/core_task1/job_<JOBID>/EXP1_report.md` (one plot, one outcome table).
Raw logs/NPZ/diagnostics are available but optional to read.

Latest completed core: **job 58962**, six main rollouts. Read
[the single report](../../workspace/exp1/core_task1/job_58962/EXP1_report.md).
Dense PG11 patch changed the approach toward the old object but produced no lift;
pooled and norm-matched pooled patches still lifted the new object. This is
partial approach mediation, not memorization evidence or task-completion SR.
The step-50 approach diagnostic was added post hoc without changing primary
first-lift outcomes; original auto-analysis and execution sources are archived.
Administrative / short-window jobs are documented in
[submission notes](../../workspace/exp1/SUBMISSION_NOTES.md) and are not pooled.

No writes to RQ1/RQ2: their preprocessing helpers are reused read-only, Exp1 has
its own output guard, config/log/tmp directories, source snapshots and protected
artifact metadata audit. Scientific inputs and policy weights remain frozen.
Source changes require a fresh job directory; no automatic overwrite/resume.

CPU checks (no GPU allocation):

```bash
env PYTHONDONTWRITEBYTECODE=1 openpi/.venv/bin/python -m unittest experiments.exp1.test_patches -v
openpi/.venv/bin/ruff check experiments/exp1
```
