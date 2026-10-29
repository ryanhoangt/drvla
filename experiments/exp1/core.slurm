#!/bin/bash
#SBATCH --job-name=drvla-exp1-core
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --chdir=/mnt/data/vhoangth2/repos/drvla
#SBATCH --output=workspace/exp1/slurm/%x_%j.out
#SBATCH --error=workspace/exp1/slurm/%x_%j.err

set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
# CUDA_VISIBLE_DEVICES is deliberately left exactly as Slurm assigned it.
srun --ntasks=1 --cpus-per-task=8 --cpu-bind=cores \
  openpi/.venv/bin/python -u -m experiments.exp1.run_core \
  --run-dir "workspace/exp1/core_task1/job_${SLURM_JOB_ID}"
