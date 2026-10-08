#!/bin/bash
# Re-collect LIBERO training activations with the openpi policy-server inputs, in N parallel shards.
cd "$(dirname "$0")/../.."
N=${1:-4}
for s in $(seq 0 $((N - 1))); do
  nohup openpi/.venv/bin/python scripts/collect_activations.py --dataset libero \
    --checkpoint checkpoints/pi05_libero_pytorch --out activations/pi05_libero_server \
    --openpi-server-inputs --no-frames --shard $s --num-shards $N \
    > workspace/rq1/logs/collect_libero_server_$s.log 2>&1 &
done
