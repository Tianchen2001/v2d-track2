#!/usr/bin/env python3
"""
Compare the human (SOMA) reference arm of one episode between retarget motion roots (read only), against the first
root given (e.g. Tier 1). Uses source_payload (soma_joints, soma_joints_wxyz) of the retargeted parquet, so what is
compared is exactly the human motion the IK was fed, plus the resulting G1 arm joints.

Per root vs the reference, at frame 0 and as mean / p90 over the clip:
  lat      lateral offset of elbow / wrist from the chest mid line along the shoulder axis (+ = toward the arm side),
           and its difference to the reference (negative = more inward / across the body)
  pos      elbow / wrist position difference in the chest frame (cm)
  rot      wrist and upper-arm orientation difference relative to the chest (deg), wrist twist about the forearm
G1: shoulder pitch / roll / yaw and elbow (deg) at frame 0, and roll range over the clip.

    python human_arm_compare.py --episode 11 --side left \
        tier1=/localscratch/wangti98/track2_tier1_motion v8=/localscratch/wangti98/track2_tier3_motion_v8 ...
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

IDX = {"Hips": 0, "Chest": 3, "LeftArm": 12, "LeftForeArm": 13, "LeftHand": 14,
       "RightArm": 40, "RightForeArm": 41, "RightHand": 42}


def find(root: Path, ep: int) -> Path:
    d = json.loads((root / "retarget_results.json").read_text())
    for e in (d if isinstance(d, list) else d.values()):
        if isinstance(e, dict) and e.get("episode_index") == ep and e.get("sequence_id"):
            p = root / f"whole_body/soma/sequence_id={e['sequence_id']}/robot_name=g1/data.parquet"
            if p.exists():
                return p
    raise SystemExit(f"episode {ep} not found under {root}")


def load(p: Path):
    t = pq.read_table(p, columns=["source_payload", "robot_joint_names", "robot_joint_positions"]).to_pydict()
    d = pickle.loads(t["source_payload"][0])
    return (np.asarray(d["soma_joints"], float), np.asarray(d["soma_joints_wxyz"], float),
            list(t["robot_joint_names"][0]), np.asarray(t["robot_joint_positions"][0], float))


def twist(rot: R, axis: np.ndarray) -> np.ndarray:
    """signed twist angle (deg) of rot about axis (swing-twist), per frame"""
    q = rot.as_quat()                                     # xyzw
    proj = np.einsum("ij,ij->i", q[:, :3], axis)
    a = np.degrees(2 * np.arctan2(proj, q[:, 3]))
    return (a + 180.0) % 360.0 - 180.0


def stats(x):
    x = np.asarray(x, float)
    return f"{x[0]:+7.1f} {x.mean():+7.1f} {np.percentile(np.abs(x), 90):6.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--side", choices=["left", "right"], default="left")
    ap.add_argument("roots", nargs="+", help="label=motion_root; the first one is the reference")
    a = ap.parse_args()
    S = "Left" if a.side == "left" else "Right"
    sgn = 1.0 if a.side == "left" else -1.0
    runs = []
    for item in a.roots:
        lab, root = item.split("=", 1)
        p = find(Path(root), a.episode)
        runs.append((lab, p, *load(p)))
    T = min(r[2].shape[0] for r in runs)
    print(f"episode {a.episode}, {a.side} arm, {T} frames compared")
    for lab, p, *_ in runs:
        print(f"  {lab:8s} {p}")

    def arm(J, Q):
        J, Q = J[:T], Q[:T]
        Rc = R.from_quat(Q[:, IDX["Chest"]], scalar_first=True)
        mid = 0.5 * (J[:, IDX["LeftArm"]] + J[:, IDX["RightArm"]])
        ax = J[:, IDX["LeftArm"]] - J[:, IDX["RightArm"]]
        ax /= np.linalg.norm(ax, axis=1, keepdims=True)
        loc = lambda k: Rc.inv().apply(J[:, IDX[k]] - mid)          # chest frame, origin at shoulder mid point
        lat = lambda k: sgn * np.einsum("ij,ij->i", J[:, IDX[k]] - mid, ax)
        rel = lambda k: Rc.inv() * R.from_quat(Q[:, IDX[k]], scalar_first=True)
        return dict(elbow=loc(f"{S}ForeArm"), wrist=loc(f"{S}Hand"), lat_e=lat(f"{S}ForeArm"), lat_w=lat(f"{S}Hand"),
                    R_up=rel(f"{S}Arm"), R_wr=rel(f"{S}Hand"))

    ref = arm(runs[0][2], runs[0][3])
    fore = ref["wrist"] - ref["elbow"]
    fore /= np.linalg.norm(fore, axis=1, keepdims=True)
    print(f"\n{'':8s} {'quantity':34s} {'f0':>7s} {'mean':>7s} {'p90|.|':>6s}")
    for lab, p, J, Q, jn, qj in runs:
        A = arm(J, Q)
        print(f"{lab:8s} {'lateral elbow from mid line, cm':34s} {stats(A['lat_e'] * 100)}")
        print(f"{'':8s} {'lateral wrist from mid line, cm':34s} {stats(A['lat_w'] * 100)}")
        if lab == runs[0][0]:
            continue
        print(f"{'':8s} {'  wrist lateral - ref, cm':34s} {stats((A['lat_w'] - ref['lat_w']) * 100)}")
        print(f"{'':8s} {'  elbow pos diff (chest frame), cm':34s} {stats(np.linalg.norm(A['elbow'] - ref['elbow'], axis=1) * 100)}")
        print(f"{'':8s} {'  wrist pos diff (chest frame), cm':34s} {stats(np.linalg.norm(A['wrist'] - ref['wrist'], axis=1) * 100)}")
        d0 = (A["wrist"][0] - ref["wrist"][0]) * 100
        print(f"{'':8s} {'  wrist diff f0 chest xyz, cm':34s} {np.round(d0, 1)}")
        print(f"{'':8s} {'  upper arm rot diff, deg':34s} {stats(np.degrees((A['R_up'] * ref['R_up'].inv()).magnitude()))}")
        dw = A["R_wr"] * ref["R_wr"].inv()
        print(f"{'':8s} {'  wrist rot diff, deg':34s} {stats(np.degrees(dw.magnitude()))}")
        print(f"{'':8s} {'  wrist twist about forearm, deg':34s} {stats(twist(dw, fore))}")
    print(f"\nG1 {a.side} arm (deg): pitch roll yaw elbow at frame 0 | roll min..max over the clip")
    for lab, p, J, Q, jn, qj in runs:
        v = lambda n: np.degrees(qj[:T, jn.index(f"{a.side}_{n}_joint")])
        roll = v("shoulder_roll")
        f0 = " ".join(f"{v(n)[0]:+7.1f}" for n in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow"))
        print(f"  {lab:8s} {f0} | {roll.min():+6.1f}..{roll.max():+6.1f}")


if __name__ == "__main__":
    main()
