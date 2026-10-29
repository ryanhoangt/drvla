#!/bin/bash
#SBATCH --job-name=drvla-exp2-smoke
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --chdir=/mnt/data/vhoangth2/repos/drvla
#SBATCH --output=workspace/exp2/slurm/%x_%j.out
#SBATCH --error=workspace/exp2/slurm/%x_%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
if [ ! -f workspace/exp2/data_v1/manifest.json ]; then
  srun --cpu-bind=cores openpi/.venv/bin/python -m experiments.exp2.prepare --out workspace/exp2/data_v1
fi
srun --cpu-bind=cores openpi/.venv/bin/python -u -m experiments.exp2.pilot \
  --data-dir workspace/exp2/data_v1 --run-dir "workspace/exp2/smoke_job_${SLURM_JOB_ID}" \
  --steps 2 --score-limit 20 --validation-limit 10 --arms sft sae_m
