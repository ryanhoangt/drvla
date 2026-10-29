#!/bin/bash
#SBATCH --job-name=drvla-exp2-cift
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --chdir=/mnt/data/vhoangth2/repos/drvla
#SBATCH --output=workspace/exp2/slurm/%x_%j.out
#SBATCH --error=workspace/exp2/slurm/%x_%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
exp2_run="workspace/exp2/cift_job_${SLURM_JOB_ID}"
trap 'if [ -f "$exp2_run/manifest.json" ]; then openpi/.venv/bin/python -u -m experiments.exp2.analyze --run-dir "$exp2_run" || true; fi' EXIT
srun --cpu-bind=cores openpi/.venv/bin/python -u -m experiments.exp2.pilot \
  --data-dir workspace/exp2/data_v1 --run-dir "$exp2_run" --steps 500 --validation-limit 120 \
  --smoke-run "${EXP2_SMOKE_RUN:?Set EXP2_SMOKE_RUN to the successful technical smoke directory}"
srun --cpu-bind=cores openpi/.venv/bin/python -u -m experiments.exp2.evaluate \
  --run-dir "$exp2_run" --eval-dir "$exp2_run/evaluation"
