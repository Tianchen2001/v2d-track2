#!/usr/bin/env python3
"""
Render the SOURCE reconstruction of Track 2 episodes (human SOMA-X body + object, as given in a tier folder)
next to the exo camera video, to check its quality by eye. Works for any tier folder in the LeRobot layout
(tier_1_multiview_caption, tier_2_synthetic_noise, or a Tier 3 export in the same layout).

Per frame, one video row:
    [ exo camera video | 3D view: body + object + ground grid + trails | top-down orthographic view, 0.5 m grid ]
    header: episode, object, frame, time, tier          footer: task text
    3D panel text: object linear / angular speed, hand vertices inside the object (approximate depth), and
                   "object not visible (filled)" on frames the dataset flags as invisible

Coordinates: body and object come from the tier exactly as the retargeter reads them (SOMA world = OpenCV
world with y and z flipped); the scene is levelled so the dataset ground plane is z = 0.

    python render_source_recon.py --root <...>/track_2/tier_1_multiview_caption --out <dir> [--episodes 1 7]
    python render_source_recon.py --root <...>/track_2/tier_2_synthetic_noise --out <dir> --stride 2

Run in the environment used for retargeting (robotic_grounding + py-soma-x), plus: pyrender, trimesh,
imageio[ffmpeg], pillow. Headless: PYOPENGL_PLATFORM=egl (default here; set osmesa if EGL is missing).
Needs convert_tier_to_retarget.py in the same folder.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_tier_to_retarget import (  # noqa: E402
    COL_IDENTITY, COL_OBJ_POSE, COL_OBJ_VIS, COL_POSE, COL_SCALE, COL_TRANSL, NJ, SOMA_JOINTS_ORDER,
    col_to_np, fill_invisible, pose7_to_mat, read_jsonl,
)

F_CV_TO_SOMA = np.diag([1.0, -1.0, -1.0])
BG = (20, 22, 26)
BODY_RGBA = [0.86, 0.90, 0.95, 1.0]
OBJ_RGBA = [0.85, 0.25, 0.20, 1.0]
GRID_RGB = [0.30, 0.32, 0.36]
PELVIS_TRAIL_RGB = [0.30, 0.80, 0.95]
OBJ_TRAIL_RGB = [0.98, 0.65, 0.15]
TEXT = (235, 235, 235)
TEXT_DIM = (170, 172, 178)
WARN = (255, 170, 60)


# ----------------------------------------------------------------------------------------------- data
def load_episode(root: Path, info: dict, meta: dict, soma, tmp: Path) -> dict:
    ep = int(meta["episode_index"])
    chunks = int(info.get("chunks_size", 1000))
    t = pq.read_table(root / info["data_path"].format(episode_chunk=ep // chunks, episode_index=ep))
    pmeta = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
    sid = meta.get("sequence_id") or pmeta.get("sequence_id")
    obj_name = meta.get("object") or meta.get("object_name")
    pose = col_to_np(t, COL_POSE).astype(np.float32)
    T = pose.shape[0]
    npz = tmp / f"ep{ep}_soma_params.npz"
    np.savez(npz, poses=pose.reshape(T, NJ, 3), transl=col_to_np(t, COL_TRANSL).astype(np.float32),
             identity_coeffs=col_to_np(t, COL_IDENTITY)[:1].astype(np.float32),
             scale_params=col_to_np(t, COL_SCALE)[:1].astype(np.float32),
             joint_names=np.array(SOMA_JOINTS_ORDER), identity_model_type=np.array("mhr"),
             unit=np.array("meters"), keep_root=np.array(True), absolute_pose=np.array(False))
    motion = soma.load_motion(params_path=str(npz), normalize=False)
    obj = col_to_np(t, COL_OBJ_POSE).astype(np.float64)
    vis = col_to_np(t, COL_OBJ_VIS).astype(bool).reshape(-1)
    pos, quat, _ = fill_invisible(obj[:, :3], obj[:, 3:7], vis)
    obj_cv = pose7_to_mat(pos, quat)                                   # (T,4,4) object -> OpenCV world
    obj_w = obj_cv.copy()
    obj_w[:, :3, :] = np.einsum("ij,tjk->tik", F_CV_TO_SOMA, obj_cv[:, :3, :])   # -> SOMA world
    ts = col_to_np(t, "timestamp").astype(np.float64) if "timestamp" in t.column_names else np.arange(T) / info.get("fps", 30)
    task_idx = int(col_to_np(t, "task_index")[0]) if "task_index" in t.column_names else None
    task = next((x["task"] for x in info.get("tasks", []) if x["task_index"] == task_idx), "")
    gp = root / info.get("ground_plane_path", "ground_plane/{sequence_id}.json").format(sequence_id=sid)
    plane = np.asarray(json.loads(gp.read_text())["plane"], dtype=np.float64) if gp.exists() else None
    mesh = root / info.get("mesh_path", "mesh/{object_name}/{object_name}.glb").format(object_name=obj_name)
    vkey = next((k for k, v in info["features"].items() if v.get("dtype") == "video"), None)
    video = root / info["video_path"].format(episode_chunk=ep // chunks, video_key=vkey, episode_index=ep) if vkey else None
    return dict(ep=ep, sid=sid, obj=obj_name, task=task, T=T, ts=ts, vis=vis, joints=np.asarray(motion["joints"]),
                verts=np.asarray(motion["vertices"]), names=list(motion["joint_names"]), obj_w=obj_w, plane=plane,
                mesh=mesh, video=video if video and video.exists() else None,
                fps=float(info.get("fps", 30)), tier=info.get("trajectory_variant", root.name))


def levelling(plane_cv: np.ndarray | None, joints: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """4x4 transform SOMA world -> render world: ground plane -> z = 0, up = +z."""
    head, foot = joints[:, SOMA_JOINTS_ORDER.index("Head")], joints[:, SOMA_JOINTS_ORDER.index("LeftFoot")]
    up_guess = (head - foot).mean(0); up_guess /= np.linalg.norm(up_guess)
    if plane_cv is not None:
        n = F_CV_TO_SOMA @ plane_cv[:3]; d = plane_cv[3]
        s = np.linalg.norm(n); n, d = n / s, d / s
        if n @ up_guess < 0:
            n, d = -n, -d
        p0 = -d * n
    else:                                                               # no plane: lowest body point of frame 0
        n = up_guess
        p0 = verts[0][np.argmin(verts[0] @ n)]
    z = np.array([0.0, 0.0, 1.0]); v = np.cross(n, z); c = float(n @ z)
    if np.linalg.norm(v) < 1e-9:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 / (1 + c))
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = -R @ p0
    return M


def hand_vertex_ids(verts0: np.ndarray, joints0: np.ndarray, names: list[str]) -> np.ndarray:
    d = np.linalg.norm(verts0[:, None, :] - joints0[None], axis=-1)
    near = np.argmin(d, axis=1)
    hand = np.array([n.startswith("LeftHand") or n.startswith("RightHand") for n in names])
    return np.nonzero(hand[near])[0]


def speeds(obj: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    p = obj[:, :3, 3]; R = obj[:, :3, :3]
    v = np.r_[0.0, np.linalg.norm(np.diff(p, axis=0), axis=1) * fps]
    rel = np.einsum("tji,tjk->tik", R[:-1], R[1:])
    ang = np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1))
    return v, np.r_[0.0, np.degrees(ang) * fps]


# ------------------------------------------------------------------------------------------- render
def look_at(eye, target, up=(0, 0, 1)) -> np.ndarray:
    eye, target, up = map(lambda a: np.asarray(a, float), (eye, target, up))
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r); u = np.cross(r, f)
    M = np.eye(4); M[:3, 0] = r; M[:3, 1] = u; M[:3, 2] = -f; M[:3, 3] = eye
    return M


def unlit(pyrender, rgb):
    return pyrender.MetallicRoughnessMaterial(baseColorFactor=[0, 0, 0, 1], emissiveFactor=list(rgb), metallicFactor=0.0)


def grid_mesh(pyrender, lo, hi, step=0.5):
    xs = np.arange(np.floor(lo[0] / step) * step, hi[0] + step, step)
    ys = np.arange(np.floor(lo[1] / step) * step, hi[1] + step, step)
    seg = [[[x, ys[0], 0], [x, ys[-1], 0]] for x in xs] + [[[xs[0], y, 0], [xs[-1], y, 0]] for y in ys]
    return pyrender.Mesh([pyrender.Primitive(positions=np.asarray(seg, np.float32).reshape(-1, 3), mode=1,
                                             material=unlit(pyrender, GRID_RGB))])


def points_mesh(pyrender, pts, rgb):
    return pyrender.Mesh([pyrender.Primitive(positions=np.asarray(pts, np.float32), mode=0, material=unlit(pyrender, rgb))])


def font(size):
    from PIL import ImageFont
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render_episode(e: dict, soma_faces: np.ndarray, out: Path, args) -> Path:
    import imageio.v2 as imageio
    import pyrender
    import trimesh
    from PIL import Image, ImageDraw

    H, W3, WT, WV = args.height, int(args.height * 4 / 3), args.height, int(args.height * 4 / 3)
    HEAD, FOOT = 32, 32                                   # 480 + 64 = 544, divisible by 16
    M = levelling(e["plane"], e["joints"], e["verts"])
    tf = lambda P: P @ M[:3, :3].T + M[:3, 3]
    verts = tf(e["verts"]); joints = tf(e["joints"])
    obj = np.einsum("ij,tjk->tik", M, e["obj_w"])
    tm = trimesh.load(str(e["mesh"]), force="mesh")
    pelvis = joints[:, SOMA_JOINTS_ORDER.index("Hips")]
    v_lin, v_ang = speeds(obj, e["fps"])
    hand_ids = hand_vertex_ids(e["verts"][0], e["joints"][0], e["names"])
    pq_ = trimesh.proximity.ProximityQuery(tm) if not args.no_penetration else None
    obj_r = float(np.linalg.norm(tm.extents)) / 2

    allp = np.r_[pelvis, obj[:, :3, 3]]
    lo, hi = allp.min(0) - 1.0, allp.max(0) + 1.0
    center = (lo + hi) / 2; center[2] = 0.8
    ext = float(max(hi[0] - lo[0], hi[1] - lo[1]))
    eye = center + np.array([-1.0, -1.25, 0.95]) / np.linalg.norm([-1.0, -1.25, 0.95]) * max(3.5, 1.15 * ext)

    scene = pyrender.Scene(bg_color=[c / 255 for c in BG] + [1], ambient_light=[0.35] * 3)
    scene.add(grid_mesh(pyrender, lo, hi))
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=look_at(center + [2, -3, 5], center))
    obj_node = scene.add(pyrender.Mesh.from_trimesh(tm, material=pyrender.MetallicRoughnessMaterial(
        baseColorFactor=OBJ_RGBA, metallicFactor=0.1, roughnessFactor=0.7)), pose=obj[0])
    cam3 = scene.add(pyrender.PerspectiveCamera(yfov=np.radians(40), aspectRatio=W3 / H), pose=look_at(eye, center))
    half = ext / 2
    camT = pyrender.OrthographicCamera(xmag=half, ymag=half)
    # one renderer for both views: the scene's GPU buffers live in a single GL context
    rnd = pyrender.OffscreenRenderer(W3, H, point_size=3)
    body_mat = pyrender.MetallicRoughnessMaterial(baseColorFactor=BODY_RGBA, metallicFactor=0.0, roughnessFactor=0.9)

    reader = imageio.get_reader(str(e["video"]), "ffmpeg") if e["video"] else None
    vfps = reader.get_meta_data().get("fps", e["fps"]) if reader else e["fps"]
    vframe, vidx = None, -1

    f_big, f_small = font(17), font(15)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=e["fps"] / args.stride, codec="libx264", quality=7, macro_block_size=8)
    body_node = trail_nodes = None
    frames = range(0, e["T"], args.stride)
    try:
        for i in frames:
            # ---- video frame aligned by timestamp
            want = int(round(e["ts"][i] * vfps))
            if reader is not None:
                try:
                    while vidx < want:
                        vframe = reader.get_next_data(); vidx += 1
                except (IndexError, StopIteration):
                    pass
            left = Image.new("RGB", (WV, H), BG)
            if vframe is not None:
                im = Image.fromarray(vframe); s = min(WV / im.width, H / im.height)
                im = im.resize((int(im.width * s), int(im.height * s)))
                left.paste(im, ((WV - im.width) // 2, (H - im.height) // 2))
            # ---- scene update
            if body_node is not None:
                scene.remove_node(body_node)
            for n in trail_nodes or []:
                scene.remove_node(n)
            body_node = scene.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(verts[i], soma_faces, process=False),
                                                             material=body_mat, smooth=True))
            scene.set_pose(obj_node, obj[i])
            a = max(0, i - args.trail)
            trail_nodes = [scene.add(points_mesh(pyrender, pelvis[a:i + 1] * [1, 1, 0] + [0, 0, 0.005], PELVIS_TRAIL_RGB)),
                           scene.add(points_mesh(pyrender, obj[a:i + 1, :3, 3], OBJ_TRAIL_RGB))]
            scene.main_camera_node = cam3
            rnd.viewport_width = W3
            img3, _ = rnd.render(scene)
            topn = scene.add(camT, pose=look_at(np.r_[center[:2], 20.0], np.r_[center[:2], 0.0], up=(0, 1, 0)))
            scene.main_camera_node = topn
            rnd.viewport_width = WT
            imgT, _ = rnd.render(scene)
            scene.remove_node(topn)
            # ---- penetration of hand vertices into the object (approximate; mesh need not be watertight)
            pen = None
            if pq_ is not None:
                hv = verts[i][hand_ids]
                Ri, ti = obj[i, :3, :3], obj[i, :3, 3]
                near = np.linalg.norm(hv - ti, axis=1) < obj_r + 0.05
                if near.any():
                    sd = pq_.signed_distance((hv[near] - ti) @ Ri)      # world -> object local; > 0 inside
                    inside = sd > 0.002
                    pen = (float(sd[inside].max()) if inside.any() else 0.0, int(inside.sum()))
            # ---- compose
            canvas = Image.new("RGB", (WV + W3 + WT, HEAD + H + FOOT), BG)
            canvas.paste(left, (0, HEAD)); canvas.paste(Image.fromarray(img3), (WV, HEAD))
            canvas.paste(Image.fromarray(imgT), (WV + W3, HEAD))
            dr = ImageDraw.Draw(canvas)
            dr.text((8, 6), f"ep{e['ep']:02d}  {e['obj']}   frame {i}/{e['T'] - 1}   t={i / e['fps']:.2f}s   [{e['tier']}]",
                    fill=TEXT, font=f_big)
            dr.text((WV + 8, HEAD + 6), "3D: SOMA-X body + object, ground plane z=0", fill=TEXT_DIM, font=f_small)
            dr.text((WV + 8, HEAD + 26), f"object |v| {v_lin[i]:.2f} m/s   |w| {v_ang[i]:.0f} deg/s", fill=TEXT, font=f_small)
            y = HEAD + 46
            if pen is not None and pen[1] > 0:
                dr.text((WV + 8, y), f"hand INSIDE object: {pen[0] * 100:.1f} cm deep ({pen[1]} hand verts)", fill=WARN, font=f_small); y += 20
            if not e["vis"][i]:
                dr.text((WV + 8, y), "object not visible in source (pose filled)", fill=WARN, font=f_small)
            dr.text((WV + W3 + 8, HEAD + 6), "top-down (ortho), grid 0.5 m", fill=TEXT_DIM, font=f_small)
            dr.text((8, HEAD + H - 22), "exo camera" if reader else "no video found", fill=TEXT_DIM, font=f_small)
            line = textwrap.shorten(e["task"], width=int((WV + W3 + WT) / 8.2), placeholder=" ...")
            dr.text((8, HEAD + H + 7), line, fill=TEXT_DIM, font=f_small)
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close(); rnd.delete()
        if reader is not None:
            reader.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="tier folder with meta/, data/, mesh/, videos/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--height", type=int, default=480, help="panel height in pixels")
    ap.add_argument("--stride", type=int, default=1, help="render every N-th frame (video plays at fps/N, real time)")
    ap.add_argument("--trail", type=int, default=300, help="trail length in frames")
    ap.add_argument("--no-penetration", action="store_true", help="skip the hand-in-object check (faster)")
    ap.add_argument("--soma-data-root", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from robotic_grounding.retarget.read_soma import SOMA
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except ImportError:
        device = None

    info = json.loads((args.root / "meta" / "info.json").read_text())
    metas = read_jsonl(args.root / "meta" / "episodes_metadata.jsonl") or [{"episode_index": i} for i in range(int(info["total_episodes"]))]
    if args.episodes is not None:
        metas = [m for m in metas if int(m["episode_index"]) in set(args.episodes)]
    soma = SOMA(data_root=args.soma_data_root, identity_model_type="mhr", device=device)
    if soma.faces is None:
        raise SystemExit("this SOMA-X install exposes no mesh faces; cannot render the body")
    with tempfile.TemporaryDirectory() as tmp:
        for m in sorted(metas, key=lambda d: int(d["episode_index"])):
            ep = int(m["episode_index"])
            out = args.out / f"ep_{ep:02d}_{m.get('object', 'object')}.mp4"
            if out.exists() and not args.force:
                print(f"ep {ep:02d} skip (exists) {out}"); continue
            e = load_episode(args.root, info, m, soma, Path(tmp))
            print(f"ep {ep:02d} {e['obj']}: {e['T']} frames, video={'yes' if e['video'] else 'NO'}, "
                  f"ground plane={'yes' if e['plane'] is not None else 'NO (body fallback)'}", flush=True)
            render_episode(e, np.asarray(soma.faces), out, args)
            print(f"   -> {out}", flush=True)


if __name__ == "__main__":
    main()
