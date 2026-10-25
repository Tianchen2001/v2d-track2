#!/usr/bin/env python3
"""
Write G1 retargeter.json variants for IK experiments without code changes. The loader requires the config folder name
to equal robot_name ("g1"), so a variant is used by copying it over configs/g1/retargeter.json for one run and
restoring the original afterwards (see the runbook loop). This script only writes the variant files (and, for the
roll-limit variant, a URDF copy next to the original URDF); it never touches configs/g1 itself.

    python ik_variants.py --toolkit $V2D/video_to_data --out /localscratch/wangti98/ik_trials_ep11/configs
    python ik_variants.py --toolkit $V2D/video_to_data --remove      # delete the URDF copy again

Variants (left and right arm alike, everything else as configs/g1):
  tmp_e1        elbow position_cost 0 -> 1 (= wrist), orientation 0.02 -> 0.2
  tmp_e3        elbow position_cost 3, orientation 0.2
  tmp_e3np      tmp_e3 + posture q_prev_cost 0.05 -> 0 (lets the arm leave the branch of the previous frame)
  tmp_q0        posture q0_cost 0 -> 0.02 (weak pull toward the zero pose, elbow unchanged)
  tmp_lim30e1   tmp_e1 + URDF copy with left_shoulder_roll lower limit -30 deg (generic, not from Tier 1)
  tmp_eo03      elbow orientation only: position 0, orientation 0.02 -> 0.3 (branch hint without a position conflict)
  tmp_eo1       elbow orientation only, orientation 1.0
  tmp_ep01      elbow position 0.1, orientation 0.2 (tie-breaker)
  tmp_ep02      elbow position 0.2, orientation 0.2
  tmp_ep01L / tmp_ep005L / tmp_ep02L   elbow position 0.1 / 0.05 / 0.2, orientation 0.2, LEFT arm only (right as g1)
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

VARIANTS = {
    "tmp_e1": dict(elbow=(1.0, 0.2)),
    "tmp_e3": dict(elbow=(3.0, 0.2)),
    "tmp_e3np": dict(elbow=(3.0, 0.2), q_prev=0.0),
    "tmp_q0": dict(q0=0.02),
    "tmp_lim30e1": dict(elbow=(1.0, 0.2), roll_lower_deg=-30.0),
    "tmp_eo03": dict(elbow=(0.0, 0.3)),
    "tmp_eo1": dict(elbow=(0.0, 1.0)),
    "tmp_ep01": dict(elbow=(0.1, 0.2)),
    "tmp_ep02": dict(elbow=(0.2, 0.2)),
    "tmp_ep01L": dict(elbow=(0.1, 0.2), sides=("left",)),
    "tmp_ep005L": dict(elbow=(0.05, 0.2), sides=("left",)),
    "tmp_ep02L": dict(elbow=(0.2, 0.2), sides=("left",)),
}
URDF_COPY = "main_with_hand_tmp_lim30.urdf"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toolkit", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="folder for <variant>.json")
    ap.add_argument("--remove", action="store_true")
    a = ap.parse_args()
    pkg = a.toolkit.resolve() / "robotic_grounding/source/robotic_grounding/robotic_grounding"
    base = pkg / "retarget/configs/g1"
    urdf_dir = pkg / "assets/urdfs/g1"
    if a.remove:
        if (urdf_dir / URDF_COPY).exists():
            (urdf_dir / URDF_COPY).unlink(); print("removed", urdf_dir / URDF_COPY)
        return
    if a.out is None:
        ap.error("--out is required unless --remove")
    a.out.mkdir(parents=True, exist_ok=True)
    rt = json.loads((base / "retargeter.json").read_text())
    for name, v in VARIANTS.items():
        d = json.loads(json.dumps(rt))
        if "elbow" in v:
            for side in v.get("sides", ("left", "right")):
                e = d["ik_map"][f"{side}_elbow_link"]
                e["position_cost"], e["orientation_cost"] = v["elbow"]
        if "q_prev" in v:
            d["posture_task"]["q_prev_cost"] = v["q_prev"]
        if "q0" in v:
            d["posture_task"]["q0_cost"] = v["q0"]
        if "roll_lower_deg" in v:
            src = (base / d["urdf"]).resolve()
            txt = src.read_text()
            pat = re.compile(r'(<joint[^>]*name="left_shoulder_roll_joint".*?<limit[^>]*lower=")([-0-9.eE+]+)(")', re.S)
            m = pat.search(txt)
            assert m, "left_shoulder_roll_joint limit not found"
            new = f"{math.radians(v['roll_lower_deg']):.6f}"
            (src.parent / URDF_COPY).write_text(pat.sub(lambda mm: mm.group(1) + new + mm.group(3), txt, count=1))
            d["urdf"] = str(Path(d["urdf"]).parent / URDF_COPY)       # still relative to configs/g1
            print(f"  {URDF_COPY}: left_shoulder_roll lower {float(m.group(2)):.4f} -> {new} rad")
        (a.out / f"{name}.json").write_text(json.dumps(d, indent=2))
        el = d["ik_map"]["left_elbow_link"]; pt = d["posture_task"]
        er = d["ik_map"]["right_elbow_link"]
        print(f"{name:12s} R elbow {er['position_cost']}/{er['orientation_cost']} | L elbow pos {el['position_cost']} ori {el['orientation_cost']} | q_prev {pt['q_prev_cost']} "
              f"q0 {pt['q0_cost']} | urdf {d['urdf']} -> {a.out / (name + '.json')}")


if __name__ == "__main__":
    main()