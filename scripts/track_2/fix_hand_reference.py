#!/usr/bin/env python3
"""
Post-process a retargeted G1 motion parquet so the Dex3 hands have a usable reference:

  1. wrist push-out : if the palm penetrates the object, move the palm out along the object surface
                      normal (local damped-least-squares IK on the 7 arm joints only, orientation kept),
  2. finger closure : on frames where the hand should be in contact (stored hand_contact_active label, or
                      palm within --contact-gap of the object), close each finger from an initial pose
                      (open, SOMA-mapped, or closed) until its links touch the object, backing off joint
                      by joint while any finger link penetrates; open otherwise,
  3. smoothing      : moving-average + rate limit on the finger targets,
  4. bookkeeping    : rewrite robot_joint_positions, ee_pose_w (FK of the palm frames, which FlashCHORD
                      validates to 1e-4), keep everything else, write a new parquet next to the old one.

Batch mode mirrors the retarget layout, so training only needs a different motion root:

    python fix_hand_reference.py --toolkit video_to_data \
        --motion-root /localscratch/wangti98/track2_tier1_motion \
        --out-root    /localscratch/wangti98/track2_tier1_motion_hand \
        --retarget-in runs/track2/tier1/retarget_in [--episodes 4 7 25] [--fingers procedural|soma|closed]

    <out-root>/whole_body/soma/sequence_id=<sid>/robot_name=g1/data.parquet   (rewritten)
    <out-root>/whole_body/soma/sequence_id=<sid>/object/...                   (copied, paths in the parquet updated)
    <out-root>/whole_body/reconstructed_stage/<sid>_support.usda             (copied)
    <out-root>/retarget_results.json                                         (copied from the source root)
    <out-root>/hand_fix_results.json                                         (per-episode stats)

Single file: --parquet <in> --out <out> [--input <retarget_in folder>].
Then check with render_retarget_check.py / run_render_all.py --motion-root <out-root>.
Runs in the retarget env (pinocchio, trimesh, scipy, pyarrow; py-soma-x + torch only for --fingers soma).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

SIDES = ("left", "right")
FINGER_CHAINS = {  # proximal -> distal joint names per finger (without side prefix)
    "thumb": ["hand_thumb_0_joint", "hand_thumb_1_joint", "hand_thumb_2_joint"],
    "index": ["hand_index_0_joint", "hand_index_1_joint"],
    "middle": ["hand_middle_0_joint", "hand_middle_1_joint"],
}
ARM_JOINTS = ["shoulder_pitch_joint", "shoulder_roll_joint", "shoulder_yaw_joint", "elbow_joint",
              "wrist_roll_joint", "wrist_pitch_joint", "wrist_yaw_joint"]
PTS_PER_LINK = 200


# ----------------------------------------------------------------------------- geometry
class MeshGap:
    """Signed gap (positive outside) from points in the mesh local frame; also returns the outward normal."""
    def __init__(self, mesh: trimesh.Trimesh, spacing: float = 0.003, n_min: int = 30_000, n_max: int = 300_000):
        n = int(np.clip(mesh.area / spacing**2, n_min, n_max))
        pts, fid = trimesh.sample.sample_surface(mesh, n)
        self.pts = np.asarray(pts, dtype=np.float64); self.nrm = np.asarray(mesh.face_normals[fid], dtype=np.float64)
        self.tree = cKDTree(self.pts); self.n = n

    def __call__(self, q: np.ndarray, with_normal: bool = False):
        q = np.asarray(q, dtype=np.float64).reshape(-1, 3)
        d, i = self.tree.query(q, k=min(8, self.n), workers=1)
        dots = np.einsum("ijk,ijk->ij", q[:, None, :] - self.pts[i], self.nrm[i])
        sgn = np.where(dots.max(axis=1) > 1e-4, 1.0, -1.0)
        if not with_normal:
            return sgn * d[:, 0]
        return sgn * d[:, 0], self.nrm[i[:, 0]]


def load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh")
    if not isinstance(m, trimesh.Trimesh):
        m = trimesh.util.concatenate(m.dump())
    return m


# ----------------------------------------------------------------------------- robot
class Robot:
    def __init__(self, urdf: Path):
        import pinocchio as pin
        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf), pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.vis = pin.buildGeomFromUrdf(self.model, str(urdf), pin.GeometryType.VISUAL, package_dirs=[str(urdf.parent)])
        self.vis_data = self.vis.createData()
        self.jnames = [str(n) for n in self.model.names]
        self.q_index = {self.jnames[j]: self.model.joints[j].idx_q for j in range(2, self.model.njoints) if self.model.joints[j].nq == 1}
        self.v_index = {self.jnames[j]: self.model.joints[j].idx_v for j in range(2, self.model.njoints) if self.model.joints[j].nq == 1}
        self.lower = np.array(self.model.lowerPositionLimit); self.upper = np.array(self.model.upperPositionLimit)
        # surface samples per link (visual meshes)
        rng = np.random.default_rng(0)
        self.link_pts: dict[str, list[tuple[int, np.ndarray]]] = {}
        for gi, g in enumerate(self.vis.geometryObjects):
            link = str(self.model.frames[g.parentFrame].name)
            if not g.meshPath or not Path(g.meshPath).exists():
                continue
            m = load_mesh(Path(g.meshPath))
            v = m.vertices
            idx = rng.choice(len(v), size=min(PTS_PER_LINK, len(v)), replace=False)
            self.link_pts.setdefault(link, []).append((gi, v[idx].copy()))

    def fk(self, q):
        pin = self.pin
        pin.forwardKinematics(self.model, self.data, q); pin.updateFramePlacements(self.model, self.data)
        pin.updateGeometryPlacements(self.model, self.data, self.vis, self.vis_data)

    def link_points_w(self, link: str) -> np.ndarray:
        out = []
        for gi, pts in self.link_pts.get(link, []):
            T = np.array(self.vis_data.oMg[gi].homogeneous)
            out.append(pts @ T[:3, :3].T + T[:3, 3])
        return np.concatenate(out) if out else np.zeros((0, 3))

    def frame(self, name: str):
        fid = self.model.getFrameId(name)
        M = self.data.oMf[fid]
        return np.array(M.translation), np.array(M.rotation)

    def frame_jacobian(self, q, name: str):
        pin = self.pin
        fid = self.model.getFrameId(name)
        pin.computeJointJacobians(self.model, self.data, q)
        return pin.getFrameJacobian(self.model, self.data, fid, pin.LOCAL_WORLD_ALIGNED)


def apply_palm_translation(robot, q, qt, col, s, delta_w) -> None:
    """One damped-least-squares step on the 7 arm joints moving the palm by delta_w (world), orientation held."""
    palm = f"{s}_hand_palm_link"; arm = [f"{s}_{a}" for a in ARM_JOINTS]
    J = robot.frame_jacobian(q, palm)
    Ja = J[:, [robot.v_index[a] for a in arm]]
    W = np.diag([1, 1, 1, 0.3, 0.3, 0.3])  # position strictly, orientation softly (let the wrist tilt a little rather than crank the elbow)
    err = np.concatenate([np.asarray(delta_w, dtype=np.float64), np.zeros(3)])
    JW = W @ Ja
    dq = JW.T @ np.linalg.solve(JW @ JW.T + 2e-3 * np.eye(6), W @ err)
    cap = np.radians(4.0)
    big = np.abs(dq).max()
    if big > cap:
        dq = dq * (cap / big)
    for a, d in zip(arm, dq):
        qi = robot.q_index[a]
        qt[col[a]] = float(np.clip(qt[col[a]] + d, robot.lower[qi], robot.upper[qi]))


def clean_segments(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Fill False holes shorter than min_len (between True runs), then drop True runs shorter than min_len."""
    def runs_of(m, value):
        out, start = [], None
        for i, v in enumerate(m):
            if v == value and start is None:
                start = i
            elif v != value and start is not None:
                out.append((start, i)); start = None
        if start is not None:
            out.append((start, len(m)))
        return out
    m = mask.copy()
    for a, b in runs_of(m, False):
        if b - a < min_len and a > 0 and b < len(m):
            m[a:b] = True
    for a, b in runs_of(m, True):
        if b - a < min_len:
            m[a:b] = False
    return m


def back_off(robot, gap, to_local, q_from, t, qt, col, s, args, max_move=None) -> None:
    """Greedy: while any finger link penetrates more than the allowed margin, move the one finger joint
    (any direction, abduction included) that raises that finger's minimum gap the most."""
    for chain in FINGER_CHAINS.values():
        links = [f"{s}_{jn}".replace("_joint", "_link") for jn in chain]
        cand = [f"{s}_{jn}" for jn in chain]
        budget = {n: (max_move if max_move is not None else 1e9) for n in cand}
        for _ in range(120):
            robot.fk(q_from(t, qt))
            cur = float(gap(to_local(np.concatenate([robot.link_points_w(l) for l in links]))).min())
            if cur >= args.finger_margin - 1e-4:
                break
            best = (cur, None, None)
            for n in cand:
                if budget[n] < np.radians(args.step_deg) - 1e-9:
                    continue
                qi = robot.q_index[n]
                for sgn in (-1.0, 1.0):
                    v = float(np.clip(qt[col[n]] + sgn * np.radians(args.step_deg), robot.lower[qi], robot.upper[qi]))
                    if v == qt[col[n]]:
                        continue
                    old = qt[col[n]]; qt[col[n]] = v
                    robot.fk(q_from(t, qt))
                    g2 = float(gap(to_local(np.concatenate([robot.link_points_w(l) for l in links]))).min())
                    qt[col[n]] = old
                    if g2 > best[0] + 1e-5:
                        best = (g2, n, v)
            if best[1] is None:
                break
            qt[col[best[1]]] = best[2]; budget[best[1]] -= np.radians(args.step_deg)


# ----------------------------------------------------------------------------- parquet
def read_row(path: Path) -> tuple[pa.Table, dict]:
    tbl = pq.read_table(path)
    return tbl, {k: v[0] for k, v in tbl.to_pydict().items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toolkit", type=Path, required=True)
    ap.add_argument("--parquet", type=Path, default=None, help="single file mode: input parquet")
    ap.add_argument("--out", type=Path, default=None, help="single file mode: output parquet")
    ap.add_argument("--motion-root", type=Path, default=None, help="batch mode: retarget output root (contains whole_body/soma)")
    ap.add_argument("--out-root", type=Path, default=None, help="batch mode: new motion root, same layout")
    ap.add_argument("--retarget-in", type=Path, default=None, help="batch mode: dir with sequence_id=* input folders (for --fingers soma)")
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="batch mode: only these episode indices (from retarget_results.json)")
    ap.add_argument("--sids", nargs="*", default=None, help="batch mode: only these sequence ids (substring match)")
    ap.add_argument("--force", action="store_true", help="batch mode: redo episodes whose output exists")
    ap.add_argument("--mesh", type=Path, default=None, help="object mesh override")
    ap.add_argument("--input", type=Path, default=None, help="retarget_in folder (needed for --fingers soma)")
    ap.add_argument("--fingers", choices=["procedural", "soma", "closed", "keep"], default="procedural",
                    help="initial finger pose before the touch/back-off pass: open (procedural), SOMA-mapped, fully closed, or the stored angles")
    ap.add_argument("--contact-gap", type=float, default=0.03, help="m: palm closer than this counts as a contact frame even if the label is 0")
    ap.add_argument("--use-labels", action="store_true", help="only treat stored hand_contact_active frames as contact frames")
    ap.add_argument("--palm-margin", type=float, default=0.004, help="m: minimum palm-link surface gap after wrist push-out")
    ap.add_argument("--finger-margin", type=float, default=-0.008, help="m: finger links close until this gap; negative = target penetration, which is what produces grip force through the finger PD")
    ap.add_argument("--close-rate-deg", type=float, default=6.0, help="max finger joint change per frame while closing/opening")
    ap.add_argument("--min-segment", type=int, default=10, help="frames: contact segments shorter than this are dropped, holes shorter are filled")
    ap.add_argument("--push-smooth", type=int, default=9, help="frames of box filter on the wrist push-out vector")
    ap.add_argument("--step-deg", type=float, default=2.0, help="finger closing step")
    ap.add_argument("--smooth", type=int, default=3, help="frames of moving average on finger targets (final pass)")
    ap.add_argument("--rate-limit-deg", type=float, default=8.0, help="max finger joint change per frame (final pass)")
    ap.add_argument("--no-push", action="store_true", help="skip the wrist push-out")
    ap.add_argument("--no-close", action="store_true", help="skip the finger closing pass (keep the initial pose, only back off penetration)")
    ap.add_argument("--no-backoff", action="store_true", help="skip the penetration back-off (raw initial pose)")
    ap.add_argument("--max-push", type=float, default=0.08, help="m: cap on total wrist push-out per frame")
    args = ap.parse_args()
    toolkit = args.toolkit.resolve()
    urdf = toolkit / "robotic_grounding/source/robotic_grounding/robotic_grounding/assets/urdfs/g1/main_with_hand.urdf"
    robot = Robot(urdf)
    if args.motion_root is not None:
        run_batch(args, toolkit, robot)
        return
    if args.parquet is None or args.out is None:
        raise SystemExit("pass --parquet/--out (single file) or --motion-root/--out-root (batch)")
    process_one(args, toolkit, robot, args.parquet, args.out, args.input, path_map=None)


def run_batch(args, toolkit: Path, robot) -> None:
    import json
    import shutil
    src_root = args.motion_root.resolve(); dst_root = args.out_root.resolve()
    index = {}
    manifest = src_root / "retarget_results.json"
    if manifest.exists():
        data = json.loads(manifest.read_text())
        entries = data if isinstance(data, list) else (data.get("episodes") or data.get("results") or list(data.values()))
        for e in entries:
            if not isinstance(e, dict):
                continue
            sid = e.get("sequence_id") or e.get("sid")
            if not sid:
                for v in e.values():
                    if isinstance(v, str) and "sequence_id=" in v:
                        sid = v.split("sequence_id=")[1].split("/")[0]; break
            if sid:
                ep = e.get("episode_index", e.get("episode", e.get("ep")))
                index[str(sid)] = {"episode_index": int(ep) if ep is not None else None, "object": e.get("object") or e.get("object_name") or ""}
    if args.episodes is not None and not any(v["episode_index"] is not None for v in index.values()):
        raise SystemExit(f"--episodes given but no episode indices found in {manifest}; pass the sequence ids via --sids instead")
    parquets = sorted(src_root.glob("whole_body/soma/sequence_id=*/robot_name=g1/data.parquet"))
    if not parquets:
        raise SystemExit(f"no parquets under {src_root}/whole_body/soma")
    # support surfaces: copy the whole directory once
    stage_src = src_root / "whole_body" / "reconstructed_stage"
    if stage_src.exists():
        stage_dst = dst_root / "whole_body" / "reconstructed_stage"; stage_dst.mkdir(parents=True, exist_ok=True)
        for f in stage_src.iterdir():
            if f.is_file() and not (stage_dst / f.name).exists():
                shutil.copy2(f, stage_dst / f.name)
    if manifest.exists():  # copy the manifest with every path re-rooted so run_train_all.py --results finds the new parquets
        dst_root.mkdir(parents=True, exist_ok=True)
        (dst_root / "retarget_results.json").write_text(manifest.read_text().replace(str(src_root), str(dst_root)))
    results_path = dst_root / "hand_fix_results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    n_done = 0
    for pq_in in parquets:
        seq_dir = pq_in.parents[1]; sid = seq_dir.name.removeprefix("sequence_id=")
        meta = index.get(sid, {})
        ep = meta.get("episode_index")
        if args.episodes is not None and ep not in args.episodes:
            continue
        if args.sids is not None and not any(x in sid for x in args.sids):
            continue
        n_done += 1
        pq_out = dst_root / pq_in.relative_to(src_root)
        if pq_out.exists() and not args.force:
            print(f"skip {sid} (exists)"); continue
        # object assets next to the parquet, same relative layout
        for sub in seq_dir.iterdir():
            if sub.is_dir() and sub.name != "robot_name=g1":
                dst = dst_root / sub.relative_to(src_root)
                if not dst.exists():
                    shutil.copytree(sub, dst)
        inp = (args.retarget_in / f"sequence_id={sid}") if args.retarget_in else None
        print(f"\n[{sid}] episode {ep} {meta.get('object', '')}", flush=True)
        stats = process_one(args, toolkit, robot, pq_in, pq_out, inp, path_map=(str(src_root), str(dst_root)))
        results[sid] = {"episode_index": ep, "object": meta.get("object", ""), "parquet": str(pq_out), "fingers": args.fingers, **stats}
        results_path.write_text(json.dumps(results, indent=2))
    if n_done == 0:
        raise SystemExit("no episode matched the --episodes/--sids filter; nothing written")
    print(f"\nwrote {results_path}")


def process_one(args, toolkit: Path, robot, parquet_in: Path, parquet_out: Path, input_dir, path_map) -> dict:
    tbl, row = read_row(parquet_in)
    if path_map is not None:  # re-root asset paths that live under the source motion root
        for key in ("object_mesh_paths", "object_urdf_paths"):
            if row.get(key):
                row[key] = [p.replace(path_map[0], path_map[1]) if p and p.startswith(path_map[0]) else p for p in row[key]]
    args.input = input_dir; args.out = parquet_out
    jnames = list(row["robot_joint_names"])
    root_p = np.asarray(row["robot_root_position"], dtype=np.float64); root_q = np.asarray(row["robot_root_wxyz"], dtype=np.float64)
    qj = np.asarray(row["robot_joint_positions"], dtype=np.float64)
    obj_p = np.asarray(row["object_body_position"], dtype=np.float64); obj_q = np.asarray(row["object_body_wxyz"], dtype=np.float64)
    if obj_p.ndim == 3:
        obj_p, obj_q = obj_p[:, 0], obj_q[:, 0]
    T = len(root_p)
    sides = list(row.get("hand_sides") or SIDES)
    labels = {s: np.asarray((row.get("hand_contact_active") or [[], []])[i], dtype=np.float64).reshape(-1) for i, s in enumerate(sides)}
    mesh_path = args.mesh or Path(row["object_mesh_paths"][0])
    if not mesh_path.is_absolute():
        mesh_path = toolkit / "robotic_grounding" / mesh_path
    if not mesh_path.exists() and path_map is not None:
        mesh_path = Path(str(mesh_path).replace(path_map[1], path_map[0]))
    mesh = load_mesh(mesh_path); gap = MeshGap(mesh)
    print(f"{T} frames; object {mesh_path.name} {len(mesh.faces)} faces; fingers={args.fingers}")

    col = {n: i for i, n in enumerate(jnames)}
    missing = [n for n in jnames if n not in robot.q_index]
    if missing:
        raise SystemExit(f"joints in parquet not in URDF: {missing}")

    # closing direction per finger joint: the limit end that brings the distal link closer to the palm
    def q_from(t, qjt):
        q = robot.pin.neutral(robot.model); q[:3] = root_p[t]; w, x, y, z = root_q[t]; q[3:7] = [x, y, z, w]
        for n, v in zip(jnames, qjt):
            q[robot.q_index[n]] = v
        return q
    close_sign = {}
    q0 = q_from(0, np.zeros(len(jnames)))
    for s in sides:
        palm_name = f"{s}_hand_palm_link"
        for fname, chain in FINGER_CHAINS.items():
            distal = f"{s}_{chain[-1]}".replace("_joint", "_link")
            for jn in chain:
                full = f"{s}_{jn}"; qi = robot.q_index[full]
                best = None
                for end in (robot.lower[qi], robot.upper[qi]):
                    q = q0.copy(); q[qi] = end; robot.fk(q)
                    tip = robot.link_points_w(distal)  # mesh samples: the link origin sits on the joint axis and would not move
                    tip = tip.mean(axis=0) if len(tip) else robot.frame(distal)[0]
                    d = np.linalg.norm(tip - robot.frame(palm_name)[0])
                    if best is None or d < best[0]:
                        best = (d, end)
                close_sign[full] = 1.0 if best[1] > 0 else -1.0
    # thumb_0 is abduction, not flexion: keep it at the stored/SOMA value unless closed mode
    for s in sides:
        close_sign[f"{s}_hand_thumb_0_joint"] = 0.0

    # optional SOMA-mapped initial finger angles
    soma_init = None
    if args.fingers == "soma":
        soma_init = soma_finger_angles(args, toolkit, T, jnames, robot, close_sign)

    new_qj = qj.copy()
    stats = {s: {"pushed": 0, "contact": 0, "mean_push": []} for s in sides}
    obj_Rm = [R.from_quat(obj_q[t], scalar_first=True).as_matrix() for t in range(T)]
    fj_of = {s: [f"{s}_{jn}" for chain in FINGER_CHAINS.values() for jn in chain] for s in sides}
    hand_links_of = {s: [f"{s}_hand_palm_link", f"{s}_wrist_roll_link", f"{s}_wrist_pitch_link", f"{s}_wrist_yaw_link"]
                     + [f"{s}_{jn}".replace("_joint", "_link") for chain in FINGER_CHAINS.values() for jn in chain] for s in sides}

    def init_fingers(t, qt, s):
        fj = fj_of[s]
        if args.fingers == "keep":
            return
        if args.fingers == "soma":
            init = {n: soma_init[t][n] for n in fj}
        elif args.fingers == "closed":
            init = {n: (robot.upper if close_sign[n] > 0 else robot.lower)[robot.q_index[n]] if close_sign[n] else qt[col[n]] for n in fj}
        else:
            init = {n: 0.0 for n in fj}
        for n, v in init.items():
            qt[col[n]] = v

    # ---- pass A: per-frame push-out vector needed with the fingers in their initial pose (not applied yet)
    push_vec = {s: np.zeros((T, 3)) for s in sides}
    for t in range(T):
        qt = new_qj[t].copy(); Rm, tr = obj_Rm[t], obj_p[t]
        for s in sides:
            init_fingers(t, qt, s)
            q_work = qt.copy(); total = np.zeros(3)
            for _ in range(0 if args.no_push else 20):
                q = q_from(t, q_work); robot.fk(q)
                pw = np.concatenate([robot.link_points_w(l) for l in hand_links_of[s]])
                g, n_local = gap((pw - tr) @ Rm, with_normal=True)
                k = int(np.argmin(g))
                if g[k] >= args.palm_margin or np.linalg.norm(total) >= args.max_push:
                    break
                n_w = Rm @ n_local[k]
                step = min(args.palm_margin - g[k] + 0.002, args.max_push - np.linalg.norm(total))
                total = total + step * n_w
                apply_palm_translation(robot, q, q_work, col, s, step * n_w)
            push_vec[s][t] = total
    # ---- pass B: smooth the push vectors in time (box filter), then apply them with the arm IK
    k = max(1, args.push_smooth)
    for s in sides:
        v = push_vec[s]
        if k > 1:
            v = np.stack([np.convolve(np.pad(v[:, j], (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid") for j in range(3)], 1)
        push_vec[s] = v
    for t in range(T):
        qt = new_qj[t]
        for s in sides:
            target = push_vec[s][t]
            if np.linalg.norm(target) < 1e-4:
                continue
            stats[s]["pushed"] += 1; stats[s]["mean_push"].append(float(np.linalg.norm(target)))
            robot.fk(q_from(t, qt)); p0 = robot.frame(f"{s}_hand_palm_link")[0]
            for _ in range(25):  # converge the palm onto p0 + target, orientation (softly) held
                q = q_from(t, qt); robot.fk(q)
                err = p0 + target - robot.frame(f"{s}_hand_palm_link")[0]
                if np.linalg.norm(err) < 1e-3:
                    break
                apply_palm_translation(robot, q, qt, col, s, err)
        new_qj[t] = qt

    # ---- pass C: contact segments (label or palm proximity), cleaned up in time
    contact = {}
    for s in sides:
        raw = np.zeros(T, dtype=bool)
        for t in range(T):
            robot.fk(q_from(t, new_qj[t]))
            pg = float(gap((robot.link_points_w(f"{s}_hand_palm_link") - obj_p[t]) @ obj_Rm[t]).min())
            lab = labels[s][t] > 0.5 if len(labels[s]) > t else False
            raw[t] = lab if args.use_labels else (lab or pg < args.contact_gap)
        contact[s] = clean_segments(raw, args.min_segment)
        stats[s]["contact"] = int(contact[s].sum())

    # ---- pass D: fingers as a slow position controller, warm-started from the previous frame
    rate = np.radians(args.close_rate_deg)
    for s in sides:
        prev = None
        for t in range(T):
            qt = new_qj[t]; Rm, tr = obj_Rm[t], obj_p[t]
            to_local = lambda p, Rm=Rm, tr=tr: (p - tr) @ Rm  # noqa: E731
            if prev is None:
                init_fingers(t, qt, s)
            else:
                for n in fj_of[s]:
                    qt[col[n]] = prev[col[n]]
            if contact[s][t] and not args.no_close:
                for chain in FINGER_CHAINS.values():
                    links = [f"{s}_{jn}".replace("_joint", "_link") for jn in chain]
                    flex = [f"{s}_{jn}" for jn in chain if close_sign[f"{s}_{jn}"] != 0.0]
                    robot.fk(q_from(t, qt))
                    g = float(gap(to_local(np.concatenate([robot.link_points_w(l) for l in links]))).min())
                    if g > args.finger_margin + 0.002:
                        # close: a few small steps, never past the first touch or the per-frame rate
                        moved = 0.0
                        while moved < rate:
                            for n in flex:
                                qi = robot.q_index[n]
                                qt[col[n]] = float(np.clip(qt[col[n]] + close_sign[n] * np.radians(args.step_deg), robot.lower[qi], robot.upper[qi]))
                            moved += np.radians(args.step_deg)
                            robot.fk(q_from(t, qt))
                            if float(gap(to_local(np.concatenate([robot.link_points_w(l) for l in links]))).min()) <= args.finger_margin:
                                break
            else:
                # open toward the initial pose at the rate limit
                qt_init = qt.copy(); init_fingers(t, qt_init, s)
                for n in fj_of[s]:
                    d = qt_init[col[n]] - qt[col[n]]
                    qt[col[n]] += float(np.clip(d, -rate, rate))
            if not args.no_backoff:
                back_off(robot, gap, to_local, q_from, t, qt, col, s, args, max_move=rate)
            new_qj[t] = qt; prev = qt.copy()
    print("  passes done", flush=True)

    # ---- 3. smoothing + rate limit on finger joints
    fj_all = [f"{s}_{jn}" for s in sides for chain in FINGER_CHAINS.values() for jn in chain]
    if args.smooth > 1:
        k = args.smooth; ker = np.ones(k) / k
        for n in fj_all:
            c = col[n]; x = np.pad(new_qj[:, c], (k // 2, k - 1 - k // 2), mode="edge")
            new_qj[:, c] = np.convolve(x, ker, mode="valid")
    lim = np.radians(args.rate_limit_deg)
    for n in fj_all:
        c = col[n]
        for t in range(1, T):
            d = new_qj[t, c] - new_qj[t - 1, c]
            if abs(d) > lim:
                new_qj[t, c] = new_qj[t - 1, c] + np.sign(d) * lim
    for n in fj_all:
        qi = robot.q_index[n]; new_qj[:, col[n]] = np.clip(new_qj[:, col[n]], robot.lower[qi], robot.upper[qi])
    # smoothing can re-introduce penetration: one more back-off pass per frame
    for t in range(T):
        Rm = R.from_quat(obj_q[t], scalar_first=True).as_matrix(); tr = obj_p[t]
        qt = new_qj[t]
        for s in sides:
            if not args.no_backoff:
                back_off(robot, gap, lambda p, Rm=Rm, tr=tr: (p - tr) @ Rm, q_from, t, qt, col, s, args, max_move=np.radians(args.close_rate_deg))
        new_qj[t] = qt

    # ---- 4. recompute ee_pose_w for the stored EE frames and report
    ee_names = list(row.get("ee_link_names") or [])
    ee = np.asarray(row["ee_pose_w"], dtype=np.float64) if row.get("ee_pose_w") is not None else None
    changed = np.abs(new_qj - qj).max(axis=0)
    for t in range(T):
        robot.fk(q_from(t, new_qj[t]))
        if ee is not None:
            for e, name in enumerate(ee_names):
                if robot.model.existFrame(name):
                    p, Rw = robot.frame(name)
                    ee[t, e, :3] = p; ee[t, e, 3:7] = R.from_matrix(Rw).as_quat(scalar_first=True)
    row["robot_joint_positions"] = new_qj.tolist()
    if ee is not None:
        row["ee_pose_w"] = ee.tolist()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict({k: [v] for k, v in row.items()}, schema=tbl.schema), args.out)
    for s in sides:
        st = stats[s]
        print(f"{s}: contact frames {st['contact']}/{T}, wrist pushed on {st['pushed']} frames"
              + (f" (mean {np.mean(st['mean_push'])*100:.1f} cm)" if st["mean_push"] else ""))
    print("max |delta| per changed joint (deg):", {n: round(float(np.degrees(changed[col[n]])), 1) for n in jnames if changed[col[n]] > 1e-6})
    print(f"wrote {args.out}")
    return {s: {"contact_frames": stats[s]["contact"], "wrist_pushed_frames": stats[s]["pushed"],
                "mean_push_cm": float(np.mean(stats[s]["mean_push"]) * 100) if stats[s]["mean_push"] else 0.0} for s in sides} | {"frames": T}


def soma_finger_angles(args, toolkit: Path, T: int, jnames, robot, close_sign):
    """Map SOMA finger flexion (deg) to Dex3 joints: MCP -> *_0, PIP -> *_1 (thumb: MCP -> thumb_1, IP -> thumb_2)."""
    if args.input is None:
        raise SystemExit("--fingers soma needs --input <retarget_in folder>")
    sys.path.insert(0, str(toolkit / "robotic_grounding" / "source" / "robotic_grounding"))
    import torch
    from robotic_grounding.retarget.read_soma import SOMA
    soma = SOMA(identity_model_type="mhr", device=torch.device("cpu"))
    motion = soma.load_motion(args.input / "soma_params.npz", normalize=False)
    J = motion["joints"]; names = list(motion["joint_names"]); ix = {n: names.index(n) for n in names}

    def ang(a, b, c):
        u = a - b; v = c - b
        return np.degrees(np.arccos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9), -1, 1)))
    out = []
    raw = {}  # joint -> list of raw SOMA flexion degrees, for the diagnostic print
    for t in range(min(T, len(J))):
        d = {}
        for s, S in (("left", "Left"), ("right", "Right")):
            for f, F in (("index", "Index"), ("middle", "Middle")):
                mcp = 180 - ang(J[t, ix[f"{S}Hand"]], J[t, ix[f"{S}Hand{F}1"]], J[t, ix[f"{S}Hand{F}2"]])
                pip = 180 - ang(J[t, ix[f"{S}Hand{F}1"]], J[t, ix[f"{S}Hand{F}2"]], J[t, ix[f"{S}Hand{F}3"]])
                for jn, deg in ((f"{s}_hand_{f}_0_joint", mcp), (f"{s}_hand_{f}_1_joint", pip)):
                    qi = robot.q_index[jn]
                    d[jn] = float(np.clip(close_sign[jn] * np.radians(deg), robot.lower[qi], robot.upper[qi]))
                    raw.setdefault(jn, []).append(deg)
            mcp = 180 - ang(J[t, ix[f"{S}HandThumb1"]], J[t, ix[f"{S}HandThumb2"]], J[t, ix[f"{S}HandThumb3"]])
            ip = 180 - ang(J[t, ix[f"{S}HandThumb2"]], J[t, ix[f"{S}HandThumb3"]], J[t, ix[f"{S}HandThumbEnd"]])
            for jn, deg in ((f"{s}_hand_thumb_1_joint", mcp), (f"{s}_hand_thumb_2_joint", ip)):
                qi = robot.q_index[jn]
                d[jn] = float(np.clip(close_sign[jn] * np.radians(deg), robot.lower[qi], robot.upper[qi]))
                raw.setdefault(jn, []).append(deg)
            d[f"{s}_hand_thumb_0_joint"] = 0.0
        out.append(d)
    while len(out) < T:
        out.append(out[-1])
    print("  SOMA finger mapping (deg): raw flexion min/median/max -> Dex3 target min/median/max, limits, close_sign")
    for jn in sorted(raw):
        r = np.array(raw[jn]); m = np.degrees([o[jn] for o in out]); qi = robot.q_index[jn]
        print(f"    {jn:28s} raw {r.min():6.1f} {np.median(r):6.1f} {r.max():6.1f} -> {m.min():6.1f} {np.median(m):6.1f} {m.max():6.1f}"
              f"   limits [{np.degrees(robot.lower[qi]):.0f}, {np.degrees(robot.upper[qi]):.0f}] sign {close_sign[jn]:+.0f}")
    return out


if __name__ == "__main__":
    main()