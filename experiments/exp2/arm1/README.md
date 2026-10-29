# Exp2 Arm 1 follow-up

All outputs live under `workspace/exp2/arm1/job_<id>`. Original code and artifacts,
including job60924 and its Frozen/full-SFT baselines, remain read-only.

Select top8 positive `Cohen d * A_visual` within each task's RQ1 top16 at PG5
and PG11. Recompute exact RQ1 episode-level statistics on the original
libero_object policy-training activation store. Recompute coefficient action
attribution on the same 80 Exp2 score frames, editing only valid visual tokens,
with the frozen SAE normalization/error held fixed. The SAE is trained on pooled
activations; tokenwise latent semantics are not assumed. No OOD fitting or
auxiliary loss is used.

Orthogonalize each decoder union, and match a random active union on count,
effective rank and approximately log attribution, log activation magnitude and
active frequency. A bijective feature replacement preserves per-task set overlap.
Report actual matching errors and projection energy; exact energy matching is
not claimed. More projected dimensions/two layers differ from the original Exp2
pilot, so an improved result alone does not isolate the feature-selection effect.
The training CAFT projection acts on the raw residual, including any component
of the SAE reconstruction error in that span. It is distinct from the
error-preserving coefficient edit used to estimate attribution.

Three independent 500-step rectifications start from the same frozen checkpoint:
PG-only SFT, action-filtered projection, matched-random projection. Only PG decoder
blocks0..16 train; PG17, AE, embeddings, image encoder/projection and action/time
projections are frozen. Frozen modules remain in the differentiable graph.
Native FM objective/AdamW, lr5e-5, 25 warmup, accumulation2, identical data/noise
order. No LoRA. Check exact hashes of frozen weights before and after every arm.

Offline: 120 original validation frames paired with one feasible training-task
counterfactual each. Frozen off and each projection on; trained SFT off and both
projection models off/on. Frozen probes reused from60924. Cohen-d dictionaries
use all RQ1 policy-training episodes including these ID validation episodes;
validation is held out from attribution fitting and gradient updates, not the
historical RQ1 dictionary. Overlap8/16 is tautological. All these metrics are
diagnostic; projection does not establish a shortcut's causal specificity.

Rollouts: 10 ID `libero_object` and10 `libero_object_task`, episode0, same
initial states/FM noise as60924. Five configurations: SFT off, target off/on,
random off/on =100 new rollouts. Primary CIFT is hook_off; hook_on is a combined
train/test edit diagnostic. Reuse20 Frozen and20 historical full-SFT rollouts
without rerunning them; verify exact initial observation hashes and first frozen
action chunks before each case. No swap suite or margin-loss Arm2.

CPU software check (synthetic tiny weights; not empirical evidence):

```bash
env PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  openpi/.venv/bin/python -m unittest experiments.exp2.arm1.test_pipeline -v
```

Run on user-requested main partition only, one GPU, 8 CPUs, 64GB:

```bash
sbatch experiments/exp2/arm1/full.slurm
```

Technical gates: native/passive FM equality; correct visual mask and both residual
callbacks; nonzero PG0 gradient through frozen AE; exact frozen-weight hashes;
orthonormal full-rank target/control bases; finite losses/gradients; inference
checkpoint reset and mode switches. Offline metrics are recorded before training;
there is no tuning on validation/OOD outcomes and no success threshold chosen
after seeing results. Rollout failures are errors, not silently counted failures.

Job60966 stopped before training at the QR validation gate. Native OpenPI enables
`float32_matmul_precision=high`; its FP32 Gram-matrix diagnostic can use TF32 on
H100 and exceed the 2e-5 tolerance. The retry validates the same stored FP32 QR
vectors with an FP64 Gram matrix at the original tolerance. Native policy and
projection precision, feature selection, optimizer and experiment arms are
unchanged. Both Gram errors are printed so the GPU retry can verify this diagnosis.

Job61066 passed the four technical tests and finished all three 500-step models;
AE/PG17/vision/embedding/motor hashes stayed exact. Slurm cancelled it during
evaluation, after33 committed ID rollouts. To continue without retraining:

```bash
sbatch experiments/exp2/arm1/resume.slurm
```

The resume job reuses the three trained checkpoints, verifies the prior protocol,
copies and SHA256-checks only committed NPZs into a new output directory, and
skips their exact (suite, task, episode, arm) keys. It runs67 missing rollouts;
uncommitted partial trials restart from their original initial state and noise.
All original job61066 files stay unchanged. Reports are written to
`workspace/exp2/arm1/resume_61066_job_<newid>/`, not the training directory.
