#!/usr/bin/env python3
"""
Offline render + physical-plausibility report for one G1 motion parquet: a retargeted reference
(any retargeter that writes the motion_v1 schema) or a policy rollout recorded by record_rollout.py
(detected by the reference_object_* columns: the reference object is drawn as a grey ghost and the
object pose error replaces the object height on the right axis).

No simulator, no Viser, no torch. For every frame of the retargeted parquet it
  * runs Pinocchio FK on the G1 (main_with_hand.urdf, the model the retargeter used),
  * poses the object mesh from object_body_position / object_body_wxyz,
  * measures, for each robot link, the gap between that link's surface points and the
    object surface (negative = penetration depth, positive = clearance),
  * measures object and foot height above the ground plane z = 0,
  * renders the scene with pyrender (EGL, headless) and burns the numbers into the frame.

Outputs in --out:
  videos/<sid>.mp4      rendered replay with per-frame overlay
  <sid>.csv             per-frame numbers
  <sid>.json            summary (penetration stats, label agreement, ground checks)
  <sid>_timeseries.png

Env: pinocchio (pip "pin" or conda), trimesh, rtree, scipy, matplotlib, pyarrow, plus for the video
    pip install pyrender imageio imageio-ffmpeg pillow pyglet

    PYOPENGL_PLATFORM=egl python render_retarget_check.py \
        --toolkit video_to_data \
        --parquet runs/track2/tier1/motion/whole_body/soma/sequence_id=<sid>/robot_name=g1/data.parquet \
        --out runs/track2/tier1/plausibility

If EGL is unavailable on the node, try PYOPENGL_PLATFORM=osmesa or --no-video to get only the numbers.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation as R

CONTACT_THRESHOLD = 0.02   # m, label threshold used by soma_to_g1.py
PEN_REPORT = 0.005         # m, count a frame as penetrating beyond this depth
MAX_PTS_PER_LINK = 150     # surface samples per link for the distance query
FOOT_LINKS = ["left_ankle_roll_link", "right_ankle_roll_link"]


# ----------------------------------------------------------------------------- helpers
def load_mesh_any(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh")
    if not isinstance(m, trimesh.Trimesh):
        m = trimesh.util.concatenate(m.dump())
    return m


class MeshGap:
    """Signed gap from points (given in the mesh local frame) to the mesh surface. Positive outside.

    Nearest-neighbour on a dense surface sampling (~3 mm spacing) with the sign from the sampled face
    normal: ~1 mm error, milliseconds per query. (trimesh's exact signed_distance is ~1000x slower.)
    Sign: outside if any of the 8 nearest samples sees the point on its outer side (robust at edges).
    On open / self-intersecting meshes the sign can still flip locally; magnitudes stay right.
    """
    def __init__(self, mesh: trimesh.Trimesh, spacing: float = 0.003, n_min: int = 30_000, n_max: int = 300_000):
        from scipy.spatial import cKDTree
        self.mesh = mesh
        n = int(np.clip(mesh.area / spacing**2, n_min, n_max))
        pts, fid = trimesh.sample.sample_surface(mesh, n)
        self.pts = np.asarray(pts, dtype=np.float64)
        self.nrm = np.asarray(mesh.face_normals[fid], dtype=np.float64)
        self.tree = cKDTree(self.pts)
        self.n_samples = n
        self.watertight = bool(mesh.is_watertight)

    def __call__(self, pts_local: np.ndarray) -> np.ndarray:
        q = np.asarray(pts_local, dtype=np.float64).reshape(-1, 3)
        k = min(8, self.n_samples)
        d, i = self.tree.query(q, k=k, workers=1)
        # outside if any of the k nearest surface samples sees the point on its outer side
        # (robust at edges, where the single nearest sample can sit on the wrong face)
        dots = np.einsum("ijk,ijk->ij", q[:, None, :] - self.pts[i], self.nrm[i])
        sgn = np.where(dots.max(axis=1) > 1e-4, 1.0, -1.0)
        return sgn * d[:, 0]


def look_at(eye, target, up=(0, 0, 1)) -> np.ndarray:
    eye, target, up = (np.asarray(v, dtype=np.float64) for v in (eye, target, up))
    f = target - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    T = np.eye(4); T[:3, 0] = s; T[:3, 1] = u; T[:3, 2] = -f; T[:3, 3] = eye  # OpenGL camera looks down -z
    return T


def support_boxes(usda: Path | None):
    """[(center_xyz, size_xyz)] for Cube prims in a *_support.usda (best-effort regex parse)."""
    if usda is None or not usda.exists() or usda.stat().st_size == 0:
        return []
    txt = usda.read_text(errors="ignore")
    boxes = []
    for block in re.finditer(r'def Cube "[^"]*"\s*\{(.*?)\n\s*\}', txt, re.S):
        body = block.group(1)
        tr = re.search(r"xformOp:translate\s*=\s*\(([^)]*)\)", body)
        c = [float(v) for v in tr.group(1).split(",")] if tr else [0, 0, 0]
        sz = re.search(r"double size\s*=\s*([\d.eE+-]+)", body); s = float(sz.group(1)) if sz else 2.0
        sc = re.search(r"xformOp:scale\s*=\s*\(([^)]*)\)", body)
        scale = [float(v) for v in sc.group(1).split(",")] if sc else [1, 1, 1]
        boxes.append((np.array(c), np.array(scale) * s))
    return boxes


# ----------------------------------------------------------------------------- parquet
def load_parquet(path: Path) -> dict:
    """Read the single-row motion_v1 parquet with pyarrow only (no torch)."""
    tbl = pq.read_table(path)
    row = {k: v[0] for k, v in tbl.to_pydict().items()}
    d = {
        "fps": float(row.get("fps") or 30.0),
        "robot_joint_names": list(row["robot_joint_names"]),
        "robot_root_position": np.asarray(row["robot_root_position"], dtype=np.float64),
        "robot_root_wxyz": np.asarray(row["robot_root_wxyz"], dtype=np.float64),
        "robot_joint_positions": np.asarray(row["robot_joint_positions"], dtype=np.float64),
        "object_mesh_paths": list(row.get("object_mesh_paths") or []),
        "hand_sides": list(row.get("hand_sides") or []),
    }
    obp = np.asarray(row["object_body_position"], dtype=np.float64)
    obq = np.asarray(row["object_body_wxyz"], dtype=np.float64)
    if obp.ndim == 3:
        obp, obq = obp[:, 0], obq[:, 0]
    d["object_position"], d["object_wxyz"] = obp, obq
    if row.get("reference_object_position") is not None and len(row["reference_object_position"]):
        d["ref_position"] = np.asarray(row["reference_object_position"], dtype=np.float64)
        d["ref_wxyz"] = np.asarray(row["reference_object_wxyz"], dtype=np.float64)
    else:
        d["ref_position"] = d["ref_wxyz"] = None
    hca = row.get("hand_contact_active") or []
    d["labels"] = {s: np.asarray(hca[i], dtype=np.float64).reshape(-1) for i, s in enumerate(d["hand_sides"]) if i < len(hca) and len(hca[i])}
    return d


# ----------------------------------------------------------------------------- robot
class RobotModel:
    def __init__(self, urdf: Path):
        import pinocchio as pin
        self.pin = pin
        pkg = str(urdf.parent)  # mesh paths in the URDF are relative to the URDF directory
        # visual model only (no collision meshes needed)
        self.model = pin.buildModelFromUrdf(str(urdf), pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.vis = pin.buildGeomFromUrdf(self.model, str(urdf), pin.GeometryType.VISUAL, package_dirs=[pkg])
        self.vis_data = self.vis.createData()
        self.joint_names = [str(n) for n in self.model.names]  # [universe, root_joint, ...]
        self.geoms = []
        rng = np.random.default_rng(0)
        for g in self.vis.geometryObjects:
            mesh = self._geom_mesh(g)
            if mesh is None:
                continue
            link = str(self.model.frames[g.parentFrame].name) if g.parentFrame < self.model.nframes else g.name
            v = mesh.vertices
            idx = rng.choice(len(v), size=min(MAX_PTS_PER_LINK, len(v)), replace=False)
            pts = v[idx].copy()
            ctr = pts.mean(axis=0)
            self.geoms.append({"name": g.name, "link": link, "mesh": mesh, "pts": pts,
                               "ctr": ctr, "rad": float(np.linalg.norm(pts - ctr, axis=1).max())})
        print(f"  robot: {self.model.nq} dof, {len(self.geoms)} visual geometries")

    def _geom_mesh(self, g):
        import hppfcl
        try:
            geom = g.geometry
            if isinstance(geom, hppfcl.Box):
                m = trimesh.creation.box(extents=np.asarray(geom.halfSide) * 2)
            elif isinstance(geom, hppfcl.Sphere):
                m = trimesh.creation.icosphere(subdivisions=2, radius=geom.radius)
            elif isinstance(geom, hppfcl.Cylinder):
                m = trimesh.creation.cylinder(radius=geom.radius, height=geom.halfLength * 2)
            elif g.meshPath and Path(g.meshPath).exists():
                m = load_mesh_any(Path(g.meshPath))
                scale = np.asarray(g.meshScale, dtype=np.float64).reshape(-1)
                if scale.size == 3 and not np.allclose(scale, 1.0):
                    m.apply_scale(scale)
            else:
                print(f"  skip geometry {g.name}: no mesh at {g.meshPath!r}")
                return None
        except Exception as exc:  # noqa: BLE001
            print(f"  skip geometry {g.name}: {exc}")
            return None
        return m

    def columns(self, jnames: list[str]):
        name_to_q = {}
        for jid in range(2, self.model.njoints):
            if self.model.joints[jid].nq == 1:
                name_to_q[self.joint_names[jid]] = self.model.joints[jid].idx_q
        cols = [(i, name_to_q[n]) for i, n in enumerate(jnames) if n in name_to_q]
        dropped = [n for n in jnames if n not in name_to_q]
        if dropped:
            print(f"  parquet joints not in model (ignored): {dropped}")
        unused = [n for n in name_to_q if n not in jnames]
        if unused:
            print(f"  model joints not in parquet (left at 0): {unused}")
        return cols

    def fk(self, root_p, root_wxyz, qj, cols):
        pin = self.pin
        q = pin.neutral(self.model)
        q[:3] = root_p
        w, x, y, z = root_wxyz; q[3:7] = [x, y, z, w]
        for i, qi in cols:
            q[qi] = qj[i]
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.updateGeometryPlacements(self.model, self.data, self.vis, self.vis_data)
        return [np.array(self.vis_data.oMg[i].homogeneous) for i in range(len(self.vis.geometryObjects))]

    def frame_z(self, name: str) -> float:
        fid = self.model.getFrameId(name)
        return float(self.data.oMf[fid].translation[2])


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toolkit", type=Path, required=True, help="video_to_data checkout (for the G1 URDF)")
    ap.add_argument("--parquet", type=Path, required=True, help="retargeted data.parquet")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mesh", type=Path, default=None, help="object mesh override (default: object_mesh_paths[0] from the parquet)")
    ap.add_argument("--support", type=Path, default=None, help="optional *_support.usda, drawn and used for the support gap")
    ap.add_argument("--sid", default=None, help="name for output files (default: from the parquet path)")
    ap.add_argument("--stride", type=int, default=1, help="render every n-th frame (numbers are computed for all frames)")
    ap.add_argument("--views", nargs="+", default=["overview", "top"], choices=["overview", "side", "top", "close"],
                    help="views on the top row, left to right")
    ap.add_argument("--view-size", type=int, nargs=2, default=(960, 540), help="pixel size of each view")
    ap.add_argument("--cam", type=float, nargs=3, default=(-2.0, 2.0, 1.4), help="overview camera offset from the scene center, m")
    ap.add_argument("--cam-side", type=float, nargs=3, default=(0.2, -3.0, 0.8), help="side camera offset, m")
    ap.add_argument("--cam-top", type=float, default=4.0, help="top-down camera height above the scene center, m")
    ap.add_argument("--cam-close", type=float, nargs=3, default=(0.7, -0.7, 0.45), help="close-up camera offset from the object, m")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    toolkit = args.toolkit.resolve()
    urdf = toolkit / "robotic_grounding/source/robotic_grounding/robotic_grounding/assets/urdfs/g1/main_with_hand.urdf"
    sid = args.sid
    if sid is None:
        m = re.search(r"sequence_id=([^/]+)", str(args.parquet.resolve()))
        sid = m.group(1) if m else args.parquet.parent.name
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- data
    d = load_parquet(args.parquet)
    root_p, root_q, qj = d["robot_root_position"], d["robot_root_wxyz"], d["robot_joint_positions"]
    obj_p, obj_q = d["object_position"], d["object_wxyz"]
    sides = d["hand_sides"] or ["left", "right"]
    labels = d["labels"]
    fps = d["fps"]
    T = len(root_p)
    ref_p, ref_q = d["ref_position"], d["ref_wxyz"]
    rollout = ref_p is not None  # policy rollout: compare achieved object pose with the reference
    if rollout:
        pos_err = np.linalg.norm(obj_p - ref_p, axis=1)
        rel = R.from_quat(obj_q, scalar_first=True) * R.from_quat(ref_q, scalar_first=True).inv()
        rot_err = rel.magnitude()
        print(f"  rollout mode: object position error mean {pos_err.mean()*100:.1f} cm, max {pos_err.max()*100:.1f} cm")

    mesh_path = args.mesh
    if mesh_path is None:
        if not d["object_mesh_paths"]:
            raise SystemExit("parquet has no object_mesh_paths; pass --mesh")
        raw = d["object_mesh_paths"][0]
        mesh_path = Path(raw) if Path(raw).is_absolute() else toolkit / "robotic_grounding" / raw
    if not mesh_path.exists():
        raise SystemExit(f"object mesh not found: {mesh_path} (pass --mesh)")
    obj_mesh = load_mesh_any(mesh_path)
    t0 = time.time(); gap = MeshGap(obj_mesh); print(f"  surface samples: {gap.n_samples} ({time.time() - t0:.1f}s)")
    obj_ctr_local = obj_mesh.bounding_sphere.primitive.center
    obj_rad = float(obj_mesh.bounding_sphere.primitive.radius)
    FAR = 0.25  # m: links whose bounding spheres are further than this from the object skip the exact query
    print(f"[{sid}] {T} frames @ {fps:g} fps; object {mesh_path.name}: {len(obj_mesh.faces)} faces "
          f"watertight={obj_mesh.is_watertight}, extents={np.round(obj_mesh.extents, 3)} m")

    robot = RobotModel(urdf)
    cols = robot.columns(d["robot_joint_names"])
    is_hand = [("hand" in g["link"]) for g in robot.geoms]
    side_of = [("left" if g["link"].startswith("left") else "right" if g["link"].startswith("right") else None) for g in robot.geoms]

    # ---- per-frame numbers
    rows, geom_poses = [], []
    t0 = time.time()
    for t in range(T):
        poses = robot.fk(root_p[t], root_q[t], qj[t], cols)
        geom_poses.append(poses)
        Rm = R.from_quat(obj_q[t], scalar_first=True).as_matrix(); tr = obj_p[t]
        link_gap = []
        obj_ctr_w = Rm @ obj_ctr_local + tr
        for i, g in enumerate(robot.geoms):
            ctr_w = poses[i][:3, :3] @ g["ctr"] + poses[i][:3, 3]
            lower = float(np.linalg.norm(ctr_w - obj_ctr_w) - obj_rad - g["rad"])
            if lower > FAR:
                link_gap.append(lower)  # conservative lower bound, exact value does not matter this far away
                continue
            pw = g["pts"] @ poses[i][:3, :3].T + poses[i][:3, 3]
            link_gap.append(float(gap((pw - tr) @ Rm).min()))
        r = {"frame": t, "time_s": t / fps}
        for s in sides:
            ids = [i for i in range(len(robot.geoms)) if is_hand[i] and side_of[i] == s]
            if ids:
                worst = min(ids, key=lambda i: link_gap[i])
                r[f"{s}_hand_gap"] = link_gap[worst]; r[f"{s}_hand_worst_link"] = robot.geoms[worst]["link"]
            else:
                r[f"{s}_hand_gap"] = np.nan; r[f"{s}_hand_worst_link"] = ""
            r[f"{s}_label"] = int(labels[s][t]) if s in labels and t < len(labels[s]) else -1
        body = [i for i in range(len(robot.geoms)) if not is_hand[i]]
        wb = min(body, key=lambda i: link_gap[i])
        r["body_gap"] = link_gap[wb]; r["body_worst_link"] = robot.geoms[wb]["link"]
        r["obj_min_z"] = float((obj_mesh.vertices @ Rm.T + tr)[:, 2].min())
        if rollout:
            r["obj_pos_err"] = float(pos_err[t]); r["obj_rot_err"] = float(rot_err[t])
        r["foot_min_z"] = min(robot.frame_z(n) for n in FOOT_LINKS)
        rows.append(r)
        if t % 100 == 0:
            print("  frame %d/%d (%.2fs/frame)  " % (t, T, (time.time() - t0) / (t + 1)) + "  ".join(f"{s[0].upper()} {r[f'{s}_hand_gap']*100:+.1f} cm" for s in sides) + f"  body {r['body_gap']*100:+.1f} cm")

    boxes = support_boxes(args.support)
    top = max(c[2] + s[2] / 2 for c, s in boxes) if boxes else None

    with (args.out / f"{sid}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- summary
    summ = {"sequence_id": sid, "frames": T, "fps": fps, "object_mesh": str(mesh_path), "object_watertight": bool(obj_mesh.is_watertight)}
    lab_any = np.zeros(T, dtype=bool)
    for s in sides:
        gp = np.array([r[f"{s}_hand_gap"] for r in rows]); lab = np.array([r[f"{s}_label"] for r in rows])
        on = lab == 1; lab_any |= on
        pen = np.clip(-gp, 0, None)
        summ[f"{s}_label_active_frames"] = int(on.sum())
        summ[f"{s}_pen_frames_gt5mm"] = int((pen > PEN_REPORT).sum())
        summ[f"{s}_pen_max_mm"] = float(np.nanmax(pen) * 1000)
        summ[f"{s}_pen_mean_when_pen_mm"] = float(pen[pen > 0].mean() * 1000) if (pen > 0).any() else 0.0
        summ[f"{s}_gap_median_when_labeled_mm"] = float(np.median(gp[on]) * 1000) if on.any() else None
        summ[f"{s}_gap_p90_when_labeled_mm"] = float(np.percentile(gp[on], 90) * 1000) if on.any() else None
        summ[f"{s}_within_2cm_frames"] = int((gp < CONTACT_THRESHOLD).sum())
        summ[f"{s}_labeled_but_gap_gt2cm_frames"] = int((on & (gp > CONTACT_THRESHOLD)).sum())
        summ[f"{s}_unlabeled_but_penetrating_frames"] = int((~on & (gp < 0)).sum())
    bg = np.array([r["body_gap"] for r in rows])
    summ["body_pen_frames_gt5mm"] = int((bg < -PEN_REPORT).sum()); summ["body_pen_max_mm"] = float(max(0.0, -bg.min()) * 1000)
    summ["body_pen_links"] = sorted({r["body_worst_link"] for r in rows if r["body_gap"] < -PEN_REPORT})
    omz = np.array([r["obj_min_z"] for r in rows]); fz = np.array([r["foot_min_z"] for r in rows])
    summ["object_below_ground_frames"] = int((omz < -PEN_REPORT).sum()); summ["object_min_z_mm"] = float(omz.min() * 1000)
    summ["object_above_3cm_with_no_contact_label_frames"] = int(((omz > 0.03) & ~lab_any).sum())
    summ["foot_below_ground_frames"] = int((fz < -PEN_REPORT).sum()); summ["foot_min_z_mm"] = float(fz.min() * 1000)
    if rollout:
        summ["object_pos_err_mean_cm"] = float(pos_err.mean() * 100); summ["object_pos_err_max_cm"] = float(pos_err.max() * 100)
        summ["object_pos_err_final_cm"] = float(pos_err[-1] * 100)
        summ["object_rot_err_mean_deg"] = float(np.degrees(rot_err).mean())
        summ["frames_within_10cm_0p5rad"] = int(((pos_err < 0.10) & (rot_err < 0.5)).sum())
        summ["frames_within_3cm_30deg"] = int(((pos_err < 0.03) & (rot_err < np.radians(30))).sum())
        summ["first_frame_err_gt10cm"] = int(np.argmax(pos_err > 0.10)) if (pos_err > 0.10).any() else -1
    if top is not None:
        summ["support_top_m"] = top
        summ["object_through_support_frames"] = int((omz - top < -PEN_REPORT).sum())
    (args.out / f"{sid}.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))

    # ---- time series
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tt = np.array([r["time_s"] for r in rows])
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, dpi=130)
    for s, c in zip(sides, ("#2a78d6", "#e07a1f")):
        gp = np.array([r[f"{s}_hand_gap"] for r in rows]) * 100
        lab = np.array([r[f"{s}_label"] for r in rows])
        axes[0].plot(tt, gp, color=c, lw=1.2, label=f"{s} hand")
        axes[0].fill_between(tt, -8, 25, where=lab == 1, color=c, alpha=0.08, lw=0)
    axes[0].plot(tt, bg * 100, color="#52514e", lw=0.9, ls="--", label="rest of body")
    axes[0].axhline(0, color="#000", lw=0.8); axes[0].axhline(2, color="#8a8984", lw=0.6, ls=":")
    axes[0].set_ylim(-8, 25); axes[0].set_ylabel("gap to object, cm"); axes[0].legend(loc="upper right", fontsize=8, ncol=3)
    axes[0].set_title(f"{sid}: G1 link surface vs object surface (negative = penetration; shaded = contact label on)", fontsize=10, loc="left")
    if rollout:
        axes[1].plot(tt, pos_err * 100, color="#c0392b", lw=1.2, label="object position error, cm")
        axes[1].plot(tt, np.degrees(rot_err), color="#8e44ad", lw=1.0, ls="--", label="object orientation error, deg")
        axes[1].axhline(10, color="#8a8984", lw=0.6, ls=":")
    axes[1].plot(tt, omz * 100, color="#2a78d6", lw=1.2, label="object lowest vertex")
    axes[1].plot(tt, fz * 100, color="#e07a1f", lw=1.2, label="lowest ankle link")
    if top is not None:
        axes[1].axhline(top * 100, color="#8a8984", lw=0.8, ls="--", label="support top")
    axes[1].axhline(0, color="#000", lw=0.8); axes[1].set_ylabel("height, cm"); axes[1].set_xlabel("time, s")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(args.out / f"{sid}_timeseries.png"); plt.close(fig)

    if args.no_video:
        return

    # ---- render: three views on top (overview, side, close-up following the object), metric strip below
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import imageio
    import pyrender
    from PIL import Image, ImageDraw, ImageFont

    VW, VH = args.view_size
    NV = len(args.views)
    W, CH = NV * VW, VH            # chart strip: full width, same height as a view

    def mat(rgb):
        return pyrender.MetallicRoughnessMaterial(baseColorFactor=[*rgb, 1.0], metallicFactor=0.1, roughnessFactor=0.8)

    scene = pyrender.Scene(bg_color=[0.96, 0.96, 0.95, 1.0], ambient_light=[0.4, 0.4, 0.4])
    ground = trimesh.creation.box(extents=(8, 8, 0.01)); ground.apply_translation((0, 0, -0.005))
    scene.add(pyrender.Mesh.from_trimesh(ground, material=mat((0.82, 0.82, 0.80))))
    for c, s in boxes:
        b = trimesh.creation.box(extents=s); b.apply_translation(c)
        scene.add(pyrender.Mesh.from_trimesh(b, material=mat((0.70, 0.66, 0.58))))
    obj_vis = trimesh.Trimesh(obj_mesh.vertices, obj_mesh.faces, process=False)  # drop textures
    obj_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.25, 0.60, 0.35, 0.55], alphaMode="BLEND",
                                                 metallicFactor=0.1, roughnessFactor=0.8, doubleSided=True)
    obj_node = scene.add(pyrender.Mesh.from_trimesh(obj_vis, material=obj_mat))  # translucent so penetration is visible
    ref_node = None
    if rollout:
        ref_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=[0.55, 0.55, 0.60, 0.35], alphaMode="BLEND",
                                                     metallicFactor=0.0, roughnessFactor=1.0, doubleSided=True)
        ref_node = scene.add(pyrender.Mesh.from_trimesh(obj_vis, material=ref_mat))  # grey ghost = reference pose
    m_body, m_pen = mat((0.62, 0.64, 0.68)), mat((0.90, 0.10, 0.10))
    m_hand = {"left": mat((0.16, 0.47, 0.84)), "right": mat((0.88, 0.48, 0.12))}
    nodes, mesh_ok, mesh_pen = [], [], []
    for i, g in enumerate(robot.geoms):
        tm = trimesh.Trimesh(g["mesh"].vertices, g["mesh"].faces, process=False)
        ok = pyrender.Mesh.from_trimesh(tm, material=m_hand[side_of[i]] if is_hand[i] and side_of[i] else m_body, smooth=False)
        pen = pyrender.Mesh.from_trimesh(tm, material=m_pen, smooth=False) if is_hand[i] else ok
        nodes.append(scene.add(ok, pose=geom_poses[0][i])); mesh_ok.append(ok); mesh_pen.append(pen)
    center = np.concatenate([root_p, obj_p]).mean(axis=0); center[2] = 0.6
    cam_node = scene.add(pyrender.PerspectiveCamera(yfov=np.deg2rad(45), aspectRatio=VW / VH), pose=look_at(center + np.asarray(args.cam), center))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5), pose=look_at(center + np.array([2.0, -2.0, 3.0]), center))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.5), pose=look_at(center + np.array([-2.0, 2.0, 3.0]), center))
    # smoothed object track for the close-up camera
    k = max(1, int(fps / 3)); kern = np.ones(k) / k
    obj_smooth = np.stack([np.convolve(np.pad(obj_p[:, j], (k // 2, k - 1 - k // 2), mode="edge"), kern, mode="valid") for j in range(3)], 1)
    close_cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(35), aspectRatio=VW / VH)
    wide_cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(45), aspectRatio=VW / VH)

    # chart strip background (drawn once), cursor drawn per frame
    fig = plt.figure(figsize=(W / 100, CH / 100), dpi=100)
    ax = fig.add_axes([0.05, 0.16, 0.91, 0.70]); ax2 = ax.twinx()
    hand_cols = {"left": "#2a78d6", "right": "#e07a1f"}
    ylim = (-8, 25)
    for s in sides:
        gp = np.array([r[f"{s}_hand_gap"] for r in rows]) * 100
        lab = np.array([r[f"{s}_label"] for r in rows])
        ax.plot(tt, gp, color=hand_cols[s], lw=1.4, label=f"{s} hand gap")
        ax.fill_between(tt, ylim[0], ylim[1], where=lab == 1, color=hand_cols[s], alpha=0.08, lw=0)
    ax.plot(tt, bg * 100, color="#52514e", lw=1.0, ls="--", label="body gap")
    if rollout:
        ax2.plot(tt, pos_err * 100, color="#c0392b", lw=1.3, label="object position error")
        ax2.plot(tt, np.degrees(rot_err), color="#8e44ad", lw=1.0, ls="--", label="object orientation error (deg)")
        ax2.axhline(10, color="#c0392b", lw=0.6, ls=":")
    else:
        ax2.plot(tt, omz * 100, color="#2e9e5b", lw=1.2, label="object lowest point")
    ax.axhline(0, color="#000", lw=0.8); ax.axhline(2, color="#8a8984", lw=0.6, ls=":")
    ax.set_ylim(*ylim); ax.set_xlim(tt[0], tt[-1]); ax.set_ylabel("gap to object, cm"); ax.set_xlabel("time, s")
    if rollout:
        ax2.set_ylim(0, max(60, float(pos_err.max() * 100 + 5))); ax2.set_ylabel("object error: cm / deg", color="#c0392b")
    else:
        ax2.set_ylim(-5, max(120, float(omz.max() * 100 + 10))); ax2.set_ylabel("object lowest point above ground, cm", color="#2e9e5b")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9, ncol=4, framealpha=0.9)
    ax.set_title("negative gap = penetration; shaded = contact label on; dotted = 2 cm label threshold", fontsize=9, loc="left")
    fig.canvas.draw()
    chart = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    x0 = ax.transData.transform((tt[0], 0))[0]; x1 = ax.transData.transform((tt[-1], 0))[0]
    y_top = CH - ax.transAxes.transform((0, 1))[1]; y_bot = CH - ax.transAxes.transform((0, 0))[1]
    plt.close(fig)

    renderer = pyrender.OffscreenRenderer(VW, VH)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18); font_s = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = font_s = ImageFont.load_default()

    def draw_view(t, pose, label):
        scene.set_pose(cam_node, pose)
        color, _ = renderer.render(scene)
        img = Image.fromarray(color); ImageDraw.Draw(img).text((8, VH - 24), label, fill=(60, 60, 60), font=font_s)
        return img

    (args.out / "videos").mkdir(exist_ok=True)
    writer = imageio.get_writer(args.out / "videos" / f"{sid}.mp4", fps=max(1, round(fps / args.stride)), codec="libx264", quality=8, macro_block_size=None)
    t0 = time.time()
    for t in range(0, T, args.stride):
        if t % (50 * args.stride) == 0 and t > 0:
            print(f"  render {t}/{T} ({(time.time() - t0) / (t // args.stride):.2f}s/frame)")
        r = rows[t]
        Tobj = np.eye(4); Tobj[:3, :3] = R.from_quat(obj_q[t], scalar_first=True).as_matrix(); Tobj[:3, 3] = obj_p[t]
        scene.set_pose(obj_node, Tobj)
        if ref_node is not None:
            Tref = np.eye(4); Tref[:3, :3] = R.from_quat(ref_q[t], scalar_first=True).as_matrix(); Tref[:3, 3] = ref_p[t]
            scene.set_pose(ref_node, Tref)
        for i, n in enumerate(nodes):
            scene.set_pose(n, geom_poses[t][i])
            if is_hand[i] and side_of[i]:
                n.mesh = mesh_pen[i] if r[f"{side_of[i]}_hand_gap"] < -PEN_REPORT else mesh_ok[i]
        frame = Image.new("RGB", (W, VH + CH), (255, 255, 255))
        for vi, view in enumerate(args.views):
            if view == "overview":
                cam_node.camera = wide_cam
                img = draw_view(t, look_at(center + np.asarray(args.cam), center), "overview")
            elif view == "side":
                cam_node.camera = wide_cam
                img = draw_view(t, look_at(center + np.asarray(args.cam_side), center), "side")
            elif view == "top":
                cam_node.camera = wide_cam
                c0 = center.copy(); c0[2] = 0.0
                img = draw_view(t, look_at(c0 + np.array([0, 0, args.cam_top]), c0, up=(0, 1, 0)), "top-down (+y up in image)")
            else:
                cam_node.camera = close_cam
                tgt = obj_smooth[t]
                img = draw_view(t, look_at(tgt + np.asarray(args.cam_close), tgt), "close-up (follows object)")
            frame.paste(img, (vi * VW, 0))
        frame.paste(Image.fromarray(chart), (0, VH))
        dr = ImageDraw.Draw(frame)
        # cursor on the chart
        xc = x0 + (x1 - x0) * (r["time_s"] - tt[0]) / max(tt[-1] - tt[0], 1e-6)
        dr.line([(xc, VH + y_top), (xc, VH + y_bot)], fill=(200, 20, 20), width=2)
        # numbers
        lines = [f"{sid}   frame {t}/{T}   t = {r['time_s']:.2f} s"]
        for s in sides:
            gv = r[f"{s}_hand_gap"]
            state = f"PENETRATION {-gv*100:.1f} cm" if gv < -PEN_REPORT else f"gap {gv*100:.1f} cm"
            lines.append(f"{s:>5} hand: {state}   ({r[f'{s}_hand_worst_link']})   label {'ON' if r[f'{s}_label'] == 1 else 'off'}")
        lines.append(f" body: gap {r['body_gap']*100:.1f} cm ({r['body_worst_link']})")
        if rollout:
            lines.append(f"object vs reference: {r['obj_pos_err']*100:.1f} cm, {np.degrees(r['obj_rot_err']):.0f} deg   (grey ghost = reference)")
        lines.append(f"object min z {r['obj_min_z']*100:.1f} cm   foot min z {r['foot_min_z']*100:.1f} cm"
                     + (f"   support top {top*100:.1f} cm" if top is not None else ""))
        y = 8
        for k_, line in enumerate(lines):
            red = ("PENETRATION" in line) or (k_ == len(lines) - 1 and r["obj_min_z"] < -PEN_REPORT) \
                or (line.startswith("object vs reference") and r["obj_pos_err"] > 0.10)
            dr.text((10, y), line, fill=(200, 20, 20) if red else (20, 20, 20), font=font); y += 23
        writer.append_data(np.asarray(frame))
    writer.close(); renderer.delete()
    print(f"wrote {args.out / 'videos' / (sid + '.mp4')}")


if __name__ == "__main__":
    main()