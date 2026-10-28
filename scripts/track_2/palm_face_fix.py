#!/usr/bin/env python3
"""
Turn each palm of a retargeted G1 reference toward the object while the hand is near it.

Per frame and hand (FK with pinocchio, main_with_hand.urdf):
  gap, c   distance from the palm centre (<side>_hand_palm_link + --pivot along x) to the closest object point c
  weight   1 at gap <= --touch, 0 at gap >= --near, linear in between, then a moving average over --blend frames
           so the turn fades in and out
  target   palm centre (--pivot ahead of the palm link origin) unchanged; palm rotation = the smallest rotation that takes the palm normal (palm-frame
           axis --axis, default left -y / right +y = the side the fingers curl to) onto the direction to c, scaled
           by the weight, applied to the original palm rotation
  solve    damped least squares on the 7 arm joints of that side (shoulder pitch/roll/yaw, elbow, wrist
           roll/pitch/yaw), warm-started from the previous frame, URDF joint limits
  smooth   the joint CHANGE is smoothed over --smooth frames (original motion detail kept)
Only the robot changes: root, legs, fingers and the object trajectory are untouched. robot_joint_positions,
ee_pose_w and hand_frames_w (frames the URDF knows) are rewritten by FK.

Output: a new motion root with the same layout for the given episodes (parquet + support usda), and
retarget_results.json listing only those episodes, so training / rendering only needs --results / --motion-root:

    python palm_face_fix.py --toolkit video_to_data \
        --motion-root /localscratch/wangti98/track2_tier1_motion \
        --out-root    /localscratch/wangti98/track2_tier1_motion_palm --episodes 4 25

Object mesh / urdf paths in the parquet keep pointing to --motion-root. Runs in the retarget env.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pinocchio as pin
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

SIDES = ("left", "right")
ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
AXES = {"y": {"left": [0, -1, 0], "right": [0, 1, 0]}, "-y": {"left": [0, 1, 0], "right": [0, -1, 0]},
        "x": {"left": [1, 0, 0], "right": [1, 0, 0]}, "-x": {"left": [-1, 0, 0], "right": [-1, 0, 0]},
        "z": {"left": [0, 0, 1], "right": [0, 0, 1]}, "-z": {"left": [0, 0, -1], "right": [0, 0, -1]}}


def box(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    pad = k // 2
    xp = np.pad(x, [(pad, pad)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    c = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:]), xp]), axis=0)
    return (c[k:] - c[:-k])[: len(x)] / k


class G1:
    def __init__(self, urdf: Path):
        self.m = pin.buildModelFromUrdf(str(urdf), pin.JointModelFreeFlyer())
        self.d = self.m.createData()
        self.lo, self.hi = np.array(self.m.lowerPositionLimit), np.array(self.m.upperPositionLimit)

    def jq(self, name):
        return self.m.joints[self.m.getJointId(name)].idx_q

    def jv(self, name):
        return self.m.joints[self.m.getJointId(name)].idx_v

    def q(self, rp, rwxyz, qidx, qj):
        q = pin.neutral(self.m)
        q[:3] = rp
        w, x, y, z = rwxyz
        q[3:7] = [x, y, z, w]
        q[qidx] = qj
        return q

    def frame(self, q, fid):
        pin.framesForwardKinematics(self.m, self.d, q)
        M = self.d.oMf[fid]
        return M.translation.copy(), M.rotation.copy()


def facing(Rp, n_loc, pp, c):
    v = c - pp
    v = v / max(np.linalg.norm(v), 1e-9)
    return float(np.degrees(np.arccos(np.clip((Rp @ n_loc) @ v, -1, 1))))


def process(args, g1: G1, p_in: Path, p_out: Path, tk: Path) -> dict:
    tbl = pq.read_table(p_in)
    row = {k: v[0] for k, v in tbl.to_pydict().items()}
    jn = list(row["robot_joint_names"])
    qj = np.asarray(row["robot_joint_positions"], float)
    rp = np.asarray(row["robot_root_position"], float)
    rq = np.asarray(row["robot_root_wxyz"], float)
    op = np.asarray(row["object_body_position"], float)
    oq = np.asarray(row["object_body_wxyz"], float)
    if op.ndim == 3:
        op, oq = op[:, 0], oq[:, 0]
    mp = Path(row["object_mesh_paths"][0])
    if not mp.is_absolute():
        mp = tk / "robotic_grounding" / mp
    mesh = trimesh.load(str(mp), force="mesh")
    S = np.asarray(trimesh.sample.sample_surface(mesh, 5000, seed=0)[0])
    T = len(qj)
    qidx = np.array([g1.jq(n) for n in jn])
    fid = {s: g1.m.getFrameId(f"{s}_hand_palm_link") for s in SIDES}
    n_loc = {s: np.asarray(AXES[args.axis][s], float) for s in SIDES}
    cols = {s: [jn.index(f"{s}_{a}_joint") for a in ARM] for s in SIDES}
    vidx = {s: [g1.jv(f"{s}_{a}_joint") for a in ARM] for s in SIDES}
    qcols = {s: [g1.jq(f"{s}_{a}_joint") for a in ARM] for s in SIDES}
    c_loc = np.array([args.pivot, 0.0, 0.0])          # pivot point in the palm frame (palm centre, between the finger bases)
    pt = lambda M: M.translation + M.rotation @ c_loc

    # pass 1: palm pose, closest object point, gap, angle on the original reference
    pal = {s: np.zeros((T, 3)) for s in SIDES}
    Rpal = {s: np.zeros((T, 3, 3)) for s in SIDES}
    cpt = {s: np.zeros((T, 3)) for s in SIDES}
    gap = {s: np.zeros(T) for s in SIDES}
    ang0 = {s: np.zeros(T) for s in SIDES}
    for t in range(T):
        q = g1.q(rp[t], rq[t], qidx, qj[t])
        pin.framesForwardKinematics(g1.m, g1.d, q)
        W = S @ R.from_quat(oq[t], scalar_first=True).as_matrix().T + op[t]
        tree = cKDTree(W)
        for s in SIDES:
            M = g1.d.oMf[fid[s]]
            pal[s][t], Rpal[s][t] = pt(M), M.rotation
            gap[s][t], k = tree.query(pal[s][t])
            cpt[s][t] = W[k]
            ang0[s][t] = facing(Rpal[s][t], n_loc[s], pal[s][t], cpt[s][t])

    # weights
    wgt = {}
    for s in SIDES:
        w = np.clip((args.near - gap[s]) / max(args.near - args.touch, 1e-6), 0.0, 1.0)
        wgt[s] = np.clip(box(w, args.blend), 0.0, 1.0)

    # pass 2: IK per side and frame
    qnew = qj.copy()
    stats = {}
    rejected = {s: 0 for s in SIDES}
    for s in SIDES:
        prev_delta = np.zeros(len(ARM))
        lo, hi = g1.lo[qcols[s]], g1.hi[qcols[s]]
        pos_res, rot_res = [], []
        raw = np.zeros((T, len(ARM)))
        bad = np.zeros(T, bool)
        for t in range(T):
            if wgt[s][t] < 1e-3:
                prev_delta[:] = 0.0
                continue
            n_w = Rpal[s][t] @ n_loc[s]
            d = cpt[s][t] - pal[s][t]
            d /= max(np.linalg.norm(d), 1e-9)
            axis = np.cross(n_w, d)
            sa, ca = np.linalg.norm(axis), float(np.clip(n_w @ d, -1, 1))
            angle = min(np.arctan2(sa, ca), np.radians(args.max_turn))
            if sa < 1e-9:
                axis = np.cross(n_w, [1.0, 0, 0]) if abs(n_w[0]) < 0.9 else np.cross(n_w, [0, 1.0, 0])
            axis /= max(np.linalg.norm(axis), 1e-9)
            R_tgt = R.from_rotvec(axis * angle * wgt[s][t]).as_matrix() @ Rpal[s][t]
            p_tgt = pal[s][t]
            qa = qj[t, cols[s]] + prev_delta
            q = g1.q(rp[t], rq[t], qidx, qj[t])
            q[qcols[s]] = np.clip(qa, lo, hi)
            for _ in range(args.iters):
                pin.framesForwardKinematics(g1.m, g1.d, q)
                M = g1.d.oMf[fid[s]]
                e_p = p_tgt - pt(M)
                e_r = pin.log3(R_tgt @ M.rotation.T)
                if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_r) < 1e-3:
                    break
                J = pin.computeFrameJacobian(g1.m, g1.d, q, fid[s], pin.LOCAL_WORLD_ALIGNED)[:, vidx[s]]
                r = M.rotation @ c_loc
                J = np.vstack([J[:3] - pin.skew(r) @ J[3:], J[3:]])   # velocity of the pivot point
                Jw = np.vstack([args.pos_weight * J[:3], J[3:]])
                e = np.concatenate([args.pos_weight * e_p, e_r])
                dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + args.damping ** 2 * np.eye(6), e)
                dq = np.clip(dq, -0.2, 0.2)
                q[qcols[s]] = np.clip(q[qcols[s]] + dq, lo, hi)
            pin.framesForwardKinematics(g1.m, g1.d, q)
            M = g1.d.oMf[fid[s]]
            pr = np.linalg.norm(p_tgt - pt(M))
            rr = np.degrees(np.linalg.norm(pin.log3(R_tgt @ M.rotation.T)))
            if pr > args.max_pos_res or rr > args.max_rot_res:   # IK did not get there: filled in from the neighbours below
                rejected[s] += 1
                bad[t] = True
                continue
            pos_res.append(pr)
            rot_res.append(rr)
            raw[t] = q[qcols[s]] - qj[t, cols[s]]
            prev_delta = raw[t].copy()
        # rejected frames inside a turned stretch: linear interpolation of the joint change between accepted frames
        # (a hard drop to zero there was what made the arm stutter)
        turned = wgt[s] >= 1e-3
        good = turned & ~bad
        if good.any():
            idx_g = np.flatnonzero(good)
            for j in range(len(ARM)):
                raw[bad, j] = np.interp(np.flatnonzero(bad), idx_g, raw[idx_g, j])
        raw[~turned] = 0.0
        delta = box(raw, args.smooth)
        # palm drift: per-frame scale (halving search), then the scale itself is smoothed (min filter + moving
        # average), so a shrink on one frame fades in and out instead of jumping
        scale = np.ones(T)
        for t in np.flatnonzero(np.abs(delta).max(axis=1) > 1e-6):
            sc = 1.0
            for _ in range(7):
                q = g1.q(rp[t], rq[t], qidx, qj[t])
                q[qcols[s]] = np.clip(qj[t, cols[s]] + sc * delta[t], lo, hi)
                pin.framesForwardKinematics(g1.m, g1.d, q)
                if np.linalg.norm(pt(g1.d.oMf[fid[s]]) - pal[s][t]) <= args.max_drift:
                    break
                sc *= 0.5
            else:
                sc = 0.0
            scale[t] = sc
        shrunk = int((scale < 1.0).sum())
        k = max(1, args.smooth)
        pad = np.pad(scale, (k // 2, k // 2), mode="edge")
        scale = np.array([pad[i:i + k].min() for i in range(T)])
        scale = np.clip(box(scale, k), 0.0, 1.0)
        qnew[:, cols[s]] = np.clip(qj[:, cols[s]] + scale[:, None] * delta, lo, hi)
        step0 = np.degrees(np.abs(np.diff(qj[:, cols[s]], axis=0)).max(axis=1))
        step1 = np.degrees(np.abs(np.diff(qnew[:, cols[s]], axis=0)).max(axis=1))
        jerk_stats = {"arm_step_max_deg": [round(float(step0.max()), 2), round(float(step1.max()), 2)],
                      "arm_step_p99_deg": [round(float(np.percentile(step0, 99)), 2), round(float(np.percentile(step1, 99)), 2)]}
        stats[s] = {"frames_turned": int((wgt[s] > 1e-3).sum()) - rejected[s], "ik_rejected": rejected[s],
                    "drift_shrunk": shrunk, **jerk_stats,
                    "ik_pos_res_max_cm": round(float(np.max(pos_res)) * 100, 2) if pos_res else 0.0,
                    "ik_rot_res_p90_deg": round(float(np.percentile(rot_res, 90)), 1) if rot_res else 0.0}

    # after: angle, palm drift; rewrite FK-derived columns
    ee_names = list(row.get("ee_link_names") or [])
    ee = np.asarray(row["ee_pose_w"], float) if row.get("ee_pose_w") is not None and ee_names else None
    hf_names = row.get("hand_frame_names") or []
    hf = [np.asarray(x, float) for x in row["hand_frames_w"]] if row.get("hand_frames_w") is not None and hf_names else None
    ang1 = {s: np.zeros(T) for s in SIDES}
    drift = {s: np.zeros(T) for s in SIDES}

    def pose7(name):
        M = g1.d.oMf[g1.m.getFrameId(name)]
        return np.concatenate([M.translation, R.from_matrix(M.rotation).as_quat(scalar_first=True)])

    for t in range(T):
        q = g1.q(rp[t], rq[t], qidx, qnew[t])
        pin.framesForwardKinematics(g1.m, g1.d, q)
        for s in SIDES:
            M = g1.d.oMf[fid[s]]
            ang1[s][t] = facing(M.rotation, n_loc[s], pt(M), cpt[s][t])
            drift[s][t] = np.linalg.norm(pt(M) - pal[s][t])
        if ee is not None:
            for e, name in enumerate(ee_names):
                if g1.m.existFrame(name):
                    ee[t, e] = pose7(name)
        if hf is not None:
            for i, names in enumerate(hf_names):
                for k, name in enumerate(names or []):
                    if g1.m.existFrame(name):
                        hf[i][t, k] = pose7(name)
    for s in SIDES:
        near, touch = gap[s] < args.near, gap[s] < args.touch
        f = lambda a, m: round(float(np.median(a[m])), 1) if m.any() else None
        stats[s].update({"near_frames": int(near.sum()), "touch_frames": int(touch.sum()),
                         "angle_near_median_deg": [f(ang0[s], near), f(ang1[s], near)],
                         "angle_touch_median_deg": [f(ang0[s], touch), f(ang1[s], touch)],
                         "palm_drift_max_cm": round(float(drift[s].max()) * 100, 2)})
    row["robot_joint_positions"] = qnew.tolist()
    if ee is not None:
        row["ee_pose_w"] = ee.tolist()
    if hf is not None:
        row["hand_frames_w"] = [x.tolist() for x in hf]
    p_out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict({k: [v] for k, v in row.items()}, schema=tbl.schema), p_out)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toolkit", type=Path, required=True)
    ap.add_argument("--motion-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--axis", default="y", choices=list(AXES), help="palm normal in the palm frame")
    ap.add_argument("--near", type=float, default=0.15, help="m: turn starts below this palm-object gap")
    ap.add_argument("--touch", type=float, default=0.05, help="m: full turn below this gap")
    ap.add_argument("--blend", type=int, default=15, help="frames: moving average of the turn weight")
    ap.add_argument("--smooth", type=int, default=9, help="frames: moving average of the joint change (and of the drift scale)")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--damping", type=float, default=0.05)
    ap.add_argument("--pos-weight", type=float, default=5.0, help="palm position vs orientation weight in the IK")
    ap.add_argument("--max-turn", type=float, default=75.0, help="deg: at most this much turn away from the original palm rotation (partial facing beyond it)")
    ap.add_argument("--max-pos-res", type=float, default=0.02, help="m: IK palm position residual above which a frame is left as is")
    ap.add_argument("--max-rot-res", type=float, default=30.0, help="deg: IK rotation residual above which a frame is left as is")
    ap.add_argument("--pivot", type=float, default=0.0, help="m along the palm-frame x (toward the fingers): the point kept fixed and measured from; 0 = palm link origin at the wrist")
    ap.add_argument("--max-drift", type=float, default=0.02, help="m: after smoothing, shrink the change where the palm moved more")
    args = ap.parse_args()

    tk = args.toolkit.resolve()
    urdf = tk / "robotic_grounding/flash_chord/src/flash_chord/assets/urdfs/g1/main_with_hand.urdf"
    if not urdf.exists():
        urdf = tk / "robotic_grounding/source/robotic_grounding/robotic_grounding/assets/urdfs/g1/main_with_hand.urdf"
    g1 = G1(urdf)
    src, dst = args.motion_root.resolve(), args.out_root.resolve()
    if src == dst:
        raise SystemExit("--out-root must differ from --motion-root")
    data = json.loads((src / "retarget_results.json").read_text())
    entries = data if isinstance(data, list) else list(data.values())
    keep, report = [], {}
    for e in entries:
        if not isinstance(e, dict) or e.get("episode_index") not in args.episodes:
            continue
        p_in = Path(e["parquet"])
        rel = p_in.relative_to(src)
        p_out = dst / rel
        sid = e["sequence_id"]
        print(f"ep {e['episode_index']:02d} {e.get('object', '')}  {sid}", flush=True)
        st = process(args, g1, p_in, p_out, tk)
        for s in SIDES:
            x = st[s]
            print(f"   {s:5s} turned {x['frames_turned']:4d} frames (IK rejected {x['ik_rejected']}, drift-shrunk {x['drift_shrunk']}) | angle near {x['angle_near_median_deg'][0]} -> "
                  f"{x['angle_near_median_deg'][1]} deg, touch {x['angle_touch_median_deg'][0]} -> {x['angle_touch_median_deg'][1]} deg"
                  f" | palm drift max {x['palm_drift_max_cm']} cm, IK rot res p90 {x['ik_rot_res_p90_deg']} deg"
                  f" | arm step max {x['arm_step_max_deg'][0]} -> {x['arm_step_max_deg'][1]} deg/frame, p99 {x['arm_step_p99_deg'][0]} -> {x['arm_step_p99_deg'][1]}")
        stage_src, stage_dst = src / "whole_body/reconstructed_stage", dst / "whole_body/reconstructed_stage"
        stage_dst.mkdir(parents=True, exist_ok=True)
        for f in stage_src.glob(f"{sid}*_support.usda"):
            shutil.copy2(f, stage_dst / f.name)
        e2 = dict(e)
        e2["parquet"] = str(p_out)
        keep.append(e2)
        report[sid] = {"episode_index": e["episode_index"], **st}
    if not keep:
        raise SystemExit("no episode matched")
    (dst / "retarget_results.json").write_text(json.dumps(keep, indent=1))
    (dst / "palm_face_results.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {dst / 'retarget_results.json'} ({len(keep)} episodes)")


if __name__ == "__main__":
    main()