#!/bin/bash
# Launch one rollout_logger client per suite, spread over the policy servers.
# Usage: run_rollouts.sh <out_dir> <trials> <port1,port2,...> suite1 suite2 ...
set -u
OUT=$1; TRIALS=$2; IFS=, read -ra PORTS <<< "$3"; shift 3
cd "$(dirname "$0")/../../openpi"
mkdir -p "$OUT/logs"
i=0
for suite in "$@"; do
  port=${PORTS[$((i % ${#PORTS[@]}))]}
  LP_NUM_THREADS=${LP_NUM_THREADS:-2} OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TMPDIR=${TMPDIR:-/tmp} LIBERO_CONFIG_PATH=../workspace/libero_pro_config \
  PYTHONPATH=$PWD/third_party/LIBERO-PRO:$PWD/../experiments/rq1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  nohup nice -n 5 /mnt/data/vhoangth2/miniconda3/envs/libero/bin/python ../experiments/rq1/rollout_logger.py \
    --suite "$suite" --port "$port" --out "$OUT" --trials "$TRIALS" > "$OUT/logs/$suite.log" 2>&1 &
  i=$((i + 1))
done
