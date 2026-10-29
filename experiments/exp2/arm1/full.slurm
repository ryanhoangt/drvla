#!/bin/bash
#SBATCH --job-name=drvla-exp2-arm1
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
arm1_run="workspace/exp2/arm1/job_${SLURM_JOB_ID}"
trap 'if [ -f "$arm1_run/manifest.json" ]; then openpi/.venv/bin/python -u -m experiments.exp2.arm1.analyze --run-dir "$arm1_run" || true; fi' EXIT
srun --cpu-bind=cores openpi/.venv/bin/python -u -m unittest \
  experiments.exp2.arm1.test_selection experiments.exp2.arm1.test_pipeline -v
srun --cpu-bind=cores openpi/.venv/bin/python -u -m experiments.exp2.arm1.pilot \
  --run-dir "$arm1_run" --steps 500 --validation-limit 120
srun --cpu-bind=cores openpi/.venv/bin/python -u -m experiments.exp2.arm1.evaluate \
  --run-dir "$arm1_run" --eval-dir "$arm1_run/evaluation"
