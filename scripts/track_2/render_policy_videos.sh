#!/usr/bin/env bash
# Render MP4 rollouts of submitted policies with the FlashCHORD built-in viewer (run INSIDE the container).
# Same setup as the official evaluator: 1 world, deterministic actor, explicit reset at frame 0, VOC off,
# tracking-failure termination off. Several camera views side by side, metrics drawn on top.
#
#   bash render_policy_videos.sh <selection dir with checkpoints.json + results/> <out dir> [episode ...]
#   VIEWS="overview side top" bash render_policy_videos.sh /localscratch/wangti98/flash_chord_outputs/mix_tier1_v2 \
#        /localscratch/wangti98/flash_chord_outputs/videos/tier1_v2 1 2 4 7
#
# Env overrides:
#   VIEWS="overview side top"   any of: overview front back left right side top (one simulator run per view; the
#            rollout is deterministic, so every view shows the same motion). Output <out>/ep_NN.mp4 = views side by side.
#   VIEW_BACKEND=mp4 (default) | viser (interactive browser view, one episode at a time, first view only)
#   WIDTH / HEIGHT per view (default 1280x720 for one view, 960x540 for several)
#   MARKERS="markers.keypoints=false markers.contacts=false"  (default: point markers off; they crash the GL
#            viewer of this Newton build, "argument 'widths' expects an array"). MARKERS="markers.enabled=false"
#            turns all markers off; MARKERS="" keeps the config defaults.
#   CAM_DIST=2.5      camera distance in m (default: from how far the reference object travels, min 2.0)
#   CAM_DIR="-1 1 0.55"  overrides the direction of the overview camera (from the target to the camera, z up)
#   OFFICIAL=1 (default) run exactly like the official evaluator (failure termination off, so no restart);
#            OFFICIAL=0 keeps training-style termination (the video restarts from frame 0 on failure)
#   OVERLAY=1 (default) draw per-frame pos / rot / keypoint error and SP / MP status (overlay_metrics.py, same
#            folder as this script, uses results/episode_NNNNNN.parquet); raw viewer videos stay in <out>/raw/
#   EXTRA="..." any further Hydra overrides
set -u
export PYGLET_HEADLESS=${PYGLET_HEADLESS:-true}
HERE=$(cd "$(dirname "$0")" && pwd)
SEL=$1; OUT=$2; shift 2
EPS=${*:-$(python -c "import json; print(' '.join(sorted(json.load(open('$SEL/checkpoints.json')), key=int)))")}
ROOT=$(python -c "import flash_chord, pathlib; print(pathlib.Path(flash_chord.__file__).resolve().parents[2])")
VIEW=$ROOT/scripts/view_policy.py
[ -f "$VIEW" ] || { echo "view_policy.py not found under $ROOT"; exit 1; }
[ -f "$SEL/checkpoints.json" ] || { echo "no $SEL/checkpoints.json"; exit 1; }
BACKEND=${VIEW_BACKEND:-mp4}
VIEWS=${VIEWS:-overview side top}
NV=$(echo $VIEWS | wc -w)
if [ "$NV" -gt 1 ]; then W=${WIDTH:-960}; H=${HEIGHT:-540}; else W=${WIDTH:-1280}; H=${HEIGHT:-720}; fi
MARKERS=${MARKERS-markers.keypoints=false markers.contacts=false}
EXTRA=${EXTRA:-}
# Match flash_chord.evaluation.harness (official evaluator): no VOC, no tracking-failure termination (a failed
# world keeps running to the reference end instead of resetting), reset seed 42, always start at frame 0.
DIS=1.0e6
EVAL_MATCH="++env.auto_reset=false ++env.voc_scale=0.0 ++env.voc.max_force=0.0 ++env.voc.max_torque=0.0 \
++env.reset.always_reset_to_first_frame=true ++env.reset.voc_decay_steps=0 ++env.reset.seed=42"
for f in pelvis_position pelvis_orientation palm_position palm_orientation object_position object_orientation; do
  EVAL_MATCH="$EVAL_MATCH ++env.termination.${f}_threshold=$DIS ++termination.${f}_threshold=$DIS"
done
[ "${OFFICIAL:-1}" = 1 ] || EVAL_MATCH="env.voc_scale=0.0"   # OFFICIAL=0: old behaviour (resets on failure)
mkdir -p "$OUT/raw"

cam_for() {  # $1 trajectory parquet, $2 view name -> viewer.camera_position=[..] viewer.camera_target=[..]
  VIEWNAME=$2 python - "$1" <<'EOF' 2>/dev/null
import os, sys
import numpy as np
import pyarrow.parquet as pq
dirs = {"overview": os.environ.get("CAM_DIR", "-1 1 0.55"), "front": "1 0 0.35", "back": "-1 0 0.35",
        "left": "0 1 0.35", "right": "0 -1 0.35", "side": "0 -1 0.35", "top": "0.03 0.0 1"}
view = os.environ["VIEWNAME"]
t = pq.read_table(sys.argv[1], columns=[f"reference_position_{a}" for a in "xyz"])
p = np.stack([t.column(f"reference_position_{a}").to_numpy() for a in "xyz"], 1)
p = p[np.isfinite(p).all(1)]
lo, hi = p.min(0), p.max(0)
tgt = (lo + hi) / 2
tgt[2] = max(0.6, tgt[2])                         # look at roughly torso height when the object is on the floor
extent = float(np.linalg.norm(hi[:2] - lo[:2]))
d = float(os.environ.get("CAM_DIST", 0)) or max(2.0, 1.2 + 0.9 * extent)
if view == "top":
    d *= 1.3
v = np.array([float(x) for x in dirs.get(view, dirs["overview"]).split()])
pos = tgt + d * v / np.linalg.norm(v)
f = lambda a: "[" + ",".join(f"{x:.3f}" for x in a) + "]"
print(f"viewer.camera_position={f(pos)} viewer.camera_target={f(tgt)}")
EOF
}

for ep in $EPS; do
  ck=$(python -c "import json; print(json.load(open('$SEL/checkpoints.json')).get('$ep', ''))")
  [ -n "$ck" ] && [ -f "$ck" ] || { echo "ep $ep: checkpoint not found ($ck)"; continue; }
  pq=$(printf "%s/results/episode_%06d.parquet" "$SEL" "$ep")
  # one video frame per 50 Hz control step; episode length = rows of the evaluated trajectory (stops at the reference end)
  n=$(python -c "import pyarrow.parquet as pq; print(pq.ParquetFile('$pq').metadata.num_rows)" 2>/dev/null || echo 1500)
  mp4=$(printf "%s/ep_%02d.mp4" "$OUT" "$ep")          # all views + metric overlay
  if [ "$BACKEND" = mp4 ] && [ -f "$mp4" ]; then echo "skip ep $ep (exists)"; continue; fi
  echo "=== ep $ep  frames=$n  views: $VIEWS  $ck"
  raws=()
  for v in $VIEWS; do
    raw=$(printf "%s/raw/ep_%02d_%s.mp4" "$OUT" "$ep" "$v")
    old=$(printf "%s/raw/ep_%02d.mp4" "$OUT" "$ep")   # single-view render from an earlier version = overview
    [ "$v" = overview ] && [ ! -f "$raw" ] && [ -f "$old" ] && mv "$old" "$raw"
    if [ "$BACKEND" != mp4 ] || [ ! -f "$raw" ]; then
      cam=$(cam_for "$pq" "$v")
      echo "    $v camera: ${cam:-viewer default}"
      (cd "$ROOT" && python "$VIEW" \
          evaluation.checkpoint="$ck" \
          evaluation.reset_mode=explicit evaluation.start_frame=0 evaluation.deterministic=true \
          $EVAL_MATCH \
          viewer.backend=$BACKEND viewer.output_path="$raw" viewer.num_frames=$n \
          viewer.video_fps=50 viewer.realtime=false viewer.width=$W viewer.height=$H \
          $cam $MARKERS $EXTRA \
          hydra.run.dir="$OUT/hydra/ep_${ep}_$v" 2>&1 | grep -v "^Warp\|^Module" | tail -3)
      [ "$BACKEND" = mp4 ] || break            # viser: interactive, first view only
    fi
    [ -f "$raw" ] && raws+=("$raw") || echo "    view $v FAILED"
  done
  [ "$BACKEND" = mp4 ] || continue
  [ ${#raws[@]} -gt 0 ] || { echo "FAILED ep $ep"; continue; }
  # label: object name and which run the checkpoint came from (selection.json from select_checkpoints.py)
  label=$(python -c "
import json, os
s = json.load(open('$SEL/selection.json')).get('$ep', {}) if os.path.exists('$SEL/selection.json') else {}
print(f\"ep $ep {s.get('object', '')} ({s.get('chosen', '?')})\")" 2>/dev/null || echo "ep $ep")
  names=()
  for r in "${raws[@]}"; do b=$(basename "$r" .mp4); names+=("${b#ep_*_}"); done
  if [ -f "$pq" ] && [ "${OVERLAY:-1}" = 1 ]; then
    python "$HERE/overlay_metrics.py" --video "${raws[@]}" --view-names "${names[@]}" --parquet "$pq" --out "$mp4" --label "$label" \
      || python "$HERE/overlay_metrics.py" --video "${raws[@]}" --view-names "${names[@]}" --out "$mp4" --label "$label"
  else
    python "$HERE/overlay_metrics.py" --video "${raws[@]}" --view-names "${names[@]}" --out "$mp4" --label "$label"
  fi
done
ls -la "$OUT"/*.mp4 2>/dev/null