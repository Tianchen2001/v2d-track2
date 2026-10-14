#!/usr/bin/env bash
# Render MP4 rollouts of submitted policies with the FlashCHORD built-in viewer (run INSIDE the container).
# Same setup as the official evaluator: 1 world, deterministic actor, explicit reset at frame 0, VOC off.
# Reference markers (keypoints / axes / contacts) are drawn by the viewer.
#
#   bash render_policy_videos.sh <archive dir with checkpoints.json + results/> <out dir> [episode ...]
#   bash render_policy_videos.sh /egr/research-zijunlab/wangti98/v2d/submissions/tier1_v2_20261014 \
#        /localscratch/wangti98/videos/tier1_v2_failed 1 2 4 7 9 11 17 21 25
#
# Env overrides: VIEW_BACKEND=mp4 (or viser for an interactive browser view), WIDTH=1280 HEIGHT=720
set -u
ARCH=$1; OUT=$2; shift 2
EPS=${*:-$(python -c "import json; print(' '.join(sorted(json.load(open('$ARCH/checkpoints.json')), key=int)))")}
ROOT=$(python -c "import flash_chord, pathlib; print(pathlib.Path(flash_chord.__file__).resolve().parents[2])")
VIEW=$ROOT/scripts/view_policy.py
[ -f "$VIEW" ] || { echo "view_policy.py not found under $ROOT"; exit 1; }
BACKEND=${VIEW_BACKEND:-mp4}; W=${WIDTH:-1280}; H=${HEIGHT:-720}
mkdir -p "$OUT"

for ep in $EPS; do
  ck=$(python -c "import json; print(json.load(open('$ARCH/checkpoints.json'))['$ep'])")
  pq=$(printf "%s/results/episode_%06d.parquet" "$ARCH" "$ep")
  # one video frame per 50 Hz control step; episode length = rows of the evaluated trajectory (+20 so the end is visible)
  n=$(python -c "import pyarrow.parquet as pq; print(pq.ParquetFile('$pq').metadata.num_rows + 20)" 2>/dev/null || echo 1500)
  mp4=$(printf "%s/ep_%02d.mp4" "$OUT" "$ep")
  if [ "$BACKEND" = mp4 ] && [ -f "$mp4" ]; then echo "skip ep $ep (exists)"; continue; fi
  echo "=== ep $ep  frames=$n  $ck"
  (cd "$ROOT" && python "$VIEW" \
      evaluation.checkpoint="$ck" \
      evaluation.reset_mode=explicit evaluation.start_frame=0 evaluation.deterministic=true \
      env.voc_scale=0.0 \
      viewer.backend=$BACKEND viewer.output_path="$mp4" viewer.num_frames=$n \
      viewer.video_fps=50 viewer.realtime=false viewer.width=$W viewer.height=$H \
      hydra.run.dir="$OUT/hydra/ep_$ep" 2>&1 | grep -v "^Warp\|^Module" | tail -5) || echo "FAILED ep $ep"
done
ls -la "$OUT"/*.mp4 2>/dev/null
