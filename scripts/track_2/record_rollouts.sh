#!/usr/bin/env bash
# Record the policy rollout of every checkpoint in a checkpoints.json (run INSIDE the FlashCHORD container).
#   bash record_rollouts.sh /localscratch/wangti98/flash_chord_outputs/mix_tier1/checkpoints.json \
#        /localscratch/wangti98/rollouts/mix_tier1 [episode ...]
# Then, outside the container (env B):
#   PYOPENGL_PLATFORM=egl python src/scripts/track_2/run_render_all.py --toolkit video_to_data \
#       --parquet-glob "/localscratch/wangti98/rollouts/mix_tier1/ep_*.parquet" \
#       --out runs/track2/tier1/rollout_mix --workers 4 --extra --views overview top
set -u
CKPTS=$1; OUT=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
EPS=${*:-$(python -c "import json,sys; print(' '.join(sorted(json.load(open('$CKPTS')), key=int)))")}
for ep in $EPS; do
  ck=$(python -c "import json; print(json.load(open('$CKPTS'))['$ep'])")
  f=$(printf "%s/ep_%02d.parquet" "$OUT" "$ep")
  if [ -f "$f" ]; then echo "skip ep $ep (exists)"; continue; fi
  echo "=== ep $ep  $ck"
  python "$HERE/record_rollout.py" --checkpoint "$ck" --out "$f" --world-count 1 2>&1 | tail -4 || echo "FAILED ep $ep"
done
