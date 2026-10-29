#!/bin/bash
#SBATCH --job-name=drvla-exp2-arm1-resume
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --chdir=/mnt/data/vhoangth2/repos/drvla
#SBATCH --output=workspace/exp2/slurm/%x_%j.out
#SBATCH --error=workspace/exp2/slurm/%x_%j.err
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
arm1_training="${ARM1_TRAINING_RUN:-workspace/exp2/arm1/job_61066}"
arm1_previous="${ARM1_PREVIOUS_EVALUATION:-$arm1_training/evaluation}"
arm1_resume="workspace/exp2/arm1/resume_61066_job_${SLURM_JOB_ID}"
mkdir -p "$arm1_resume"
trap 'openpi/.venv/bin/python -u -m experiments.exp2.arm1.analyze --run-dir "$arm1_training" --evaluation-dir "$arm1_resume/evaluation" --output-dir "$arm1_resume" || true' EXIT
srun --cpu-bind=cores openpi/.venv/bin/python -u -m unittest experiments.exp2.arm1.test_resume -v
srun --cpu-bind=cores openpi/.venv/bin/python -u -m experiments.exp2.arm1.evaluate \
  --run-dir "$arm1_training" --eval-dir "$arm1_resume/evaluation" \
  --resume-evaluation "$arm1_previous"
