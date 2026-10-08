#!/bin/bash
# Wait for the LIBERO re-collection, then train SAEs and build the index; then wait for the
# rollouts and run the behavioral labels and the RQ1 analysis.
set -u
cd "$(dirname "$0")/../.."
PY=openpi/.venv/bin/python
ACT=activations/pi05_libero_server
SAE=saes/pi05_libero_server
IDX=index/pi05_libero_server
RO=workspace/rq1/rollouts
log() { echo "$(date '+%F %T') $*"; }

until [ "$(ls $ACT/episodes | wc -l)" -ge 1693 ] && ! pgrep -f "^openpi/.venv/bin/python scripts/collect_activations" >/dev/null; do sleep 60; done
log "collection done: $(ls $ACT/episodes | wc -l) episodes"

for layer in paligemma.layer_{0,5,11,17}.output action_expert.layer_{0,5,11,17}.output; do
  if [ ! -f $SAE/$layer/sae.pt ] || [ ! -f $SAE/$layer/sae_epoch_100.pt ]; then
    log "training SAE $layer"
    $PY scripts/train_sae.py --activations $ACT --layer $layer --out $SAE > workspace/rq1/logs/sae_$layer.log 2>&1 || log "SAE $layer FAILED"
  fi
done
log "building index"
$PY scripts/build_index.py --activations $ACT --saes $SAE --out $IDX > workspace/rq1/logs/build_index.log 2>&1 || log "index FAILED"

until ! pgrep -f "rollout_logger.py --suite" >/dev/null; do sleep 60; done
log "rollouts done: $(cat $RO/*/results.jsonl | wc -l) episodes"
$PY experiments/rq1/behavior.py $RO $(ls $RO | grep '^libero') > workspace/rq1/logs/behavior.log 2>&1 || log "behavior FAILED"
$PY experiments/rq1/analyze_rq1.py --rollouts $RO --activations $ACT --saes $SAE --index $IDX \
  --out workspace/rq1/analysis > workspace/rq1/logs/analysis.log 2>&1 || log "analysis FAILED"
log "pipeline finished"
