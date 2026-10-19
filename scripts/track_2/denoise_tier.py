#!/usr/bin/env python3
"""
Denoise a Track 2 tier (LeRobot layout), writing a new tier folder in the same layout, so
convert_tier_to_retarget.py / run_retarget_all.py / run_train_all.py / the packer (--frame-references) use it unchanged.

Stage 1 (always): temporal smoothing to remove frame-to-frame jitter.
  --obj-filter ssa: the object pose uses multichannel singular spectrum analysis instead (Hankel low-rank, rank
  chosen by the Gavish-Donoho threshold unless --ssa-rank is given); the body keeps Savitzky-Golay.
Stage 2 (--hold-static): frames where no body vertex touches the object -> the object cannot move, so its pose is
  held constant over each such run (see hold_static). Needs the retarget environment (robotic_grounding + py-soma-x).
Planned later: projection of resting object poses onto the support, hand-relative grasp smoothing.

Smoothed (Savitzky-Golay, per episode):
  observation.human.translation        position
  observation.human.pose               every joint rotation (axis-angle -> quaternion, sign-continuous, filtered,
                                       renormalised -> axis-angle)
  observation.object.pose              position + quaternion (wxyz), same treatment; invisible frames are first
                                       filled from the visible ones (linear / slerp) so they do not drag the filter,
                                       and the visibility column is kept as is
Everything else in the parquet is copied unchanged (schema and schema metadata included). meta/ is copied with
trajectory_variant renamed; every other top-level folder (mesh, videos, ground_plane, support_surface, ...) is
symlinked, not copied.

Only the tier itself is read: no other tier, no reference data. The report gives physical quantities only
(mean acceleration before/after, size of the correction).

    python denoise_tier.py --root <...>/track_2/tier_2_synthetic_noise --out <...>/track_2/tier_2_denoise_v1 \
        [--win-human 0.33] [--win-object 0.33] [--order 2]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R, Slerp

SOMA_JOINTS_ORDER = None   # filled from convert_tier_to_retarget when --hold-static is used

COL_POSE = "observation.human.pose"
COL_TRANSL = "observation.human.translation"
COL_OBJ = "observation.object.pose"
COL_VIS = "observation.object.visible"
COL_IDENTITY = "observation.human.identity_coeffs"
COL_SCALE = "observation.human.scale_params"
F_CV_TO_SOMA = np.diag([1.0, -1.0, -1.0])


def to_np(table: pa.Table, name: str) -> np.ndarray:
    a = table.column(name).combine_chunks()
    flat = a.flatten().to_numpy(zero_copy_only=False)
    return flat.reshape(len(a), -1).astype(np.float64)


def from_np(x: np.ndarray, like: pa.ChunkedArray) -> pa.Array:
    t = like.type
    vals = pa.array(x.reshape(-1).astype(t.value_type.to_pandas_dtype()), type=t.value_type)
    return pa.FixedSizeListArray.from_arrays(vals, t.list_size)


def win_frames(sec: float, fps: float, T: int, order: int) -> int:
    w = max(order + 2, int(round(sec * fps)))
    w += (w + 1) % 2                                   # odd
    w = min(w, T if T % 2 else T - 1)
    return w if w > order else 0


def sg(x: np.ndarray, w: int, order: int) -> np.ndarray:
    return savgol_filter(x, w, order, axis=0, mode="interp") if w else x


def continuous(q: np.ndarray) -> np.ndarray:
    """(T,4) quaternions with no sign flips between consecutive frames."""
    q = q.copy()
    for t in range(1, len(q)):
        if q[t] @ q[t - 1] < 0:
            q[t] = -q[t]
    return q


def smooth_quat(q: np.ndarray, w: int, order: int) -> np.ndarray:
    q = sg(continuous(q / np.linalg.norm(q, axis=1, keepdims=True)), w, order)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def gd_omega(beta: float) -> float:
    """Gavish-Donoho optimal hard-threshold coefficient for unknown noise level (threshold = omega * median sv)."""
    return 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43


def mssa(x: np.ndarray, L: int, rank: int = 0, max_rank: int = 20) -> tuple[np.ndarray, int]:
    """Multichannel singular spectrum analysis: low-rank denoising of (T, d) series.

    Each channel is embedded in an L x K Hankel matrix (K = T - L + 1), the channels are stacked to (d L) x K, the
    SVD keeps `rank` components (0 = automatic, Gavish-Donoho threshold on the singular values), and every channel is
    rebuilt by anti-diagonal averaging. Smooth motion is low rank; per-frame noise spreads over all components.
    """
    T, d = x.shape
    L = int(min(max(L, 2), T // 2))
    if L < 2:
        return x, 0
    mu = x.mean(0)
    xc = x - mu
    K = T - L + 1
    H = np.concatenate([np.lib.stride_tricks.sliding_window_view(xc[:, c], L).T for c in range(d)], axis=0)  # (dL, K)
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    if rank <= 0:
        m, n = sorted(H.shape)
        tau = gd_omega(m / n) * np.median(S)
        rank = int(max(1, min(max_rank, (S > tau).sum())))
    Hr = (U[:, :rank] * S[:rank]) @ Vt[:rank]
    out = np.zeros_like(xc)
    cnt = np.zeros(T)
    for i in range(L):
        cnt[i:i + K] += 1
    for c in range(d):
        blk = Hr[c * L:(c + 1) * L]
        acc = np.zeros(T)
        for i in range(L):
            acc[i:i + K] += blk[i]
        out[:, c] = acc / cnt
    return out + mu, rank


def ssa_quat(q: np.ndarray, L: int, rank: int) -> tuple[np.ndarray, int]:
    q, r = mssa(continuous(q / np.linalg.norm(q, axis=1, keepdims=True)), L, rank)
    return q / np.linalg.norm(q, axis=1, keepdims=True), r


def fill_invisible(pos: np.ndarray, q_wxyz: np.ndarray, vis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = np.flatnonzero(vis)
    if idx.size == 0 or idx.size == len(vis):
        return pos, q_wxyz
    t = np.arange(len(vis))
    pos = np.stack([np.interp(t, idx, pos[idx, k]) for k in range(3)], 1)
    if idx.size == 1:
        return pos, np.repeat(q_wxyz[idx], len(vis), 0)
    rot = Slerp(idx.astype(float), R.from_quat(q_wxyz[idx], scalar_first=True))(np.clip(t, idx[0], idx[-1]).astype(float))
    return pos, rot.as_quat(scalar_first=True)


def mean_acc(p: np.ndarray, fps: float) -> float:
    return float(np.linalg.norm(np.diff(p, 2, axis=0), axis=1).mean() * fps ** 2) if len(p) > 2 else 0.0


def ang_acc(q_wxyz: np.ndarray, fps: float) -> float:
    rv = R.from_quat(q_wxyz, scalar_first=True)
    w = (rv[:-1].inv() * rv[1:]).as_rotvec() * fps            # body angular velocity, rad/s
    return float(np.degrees(np.linalg.norm(np.diff(w, axis=0), axis=1).mean() * fps)) if len(w) > 1 else 0.0


# ------------------------------------------------------------------ stage 2: hold the object still without contact
class HoldContext:
    """SOMA body model + object surface samples for one episode (needs the retarget environment)."""

    def __init__(self, soma, mesh_path: Path, tmp: Path, n_surface: int = 3000):
        import trimesh
        self.soma, self.tmp = soma, tmp
        m = trimesh.load(str(mesh_path), force="mesh")
        self.surface = np.asarray(m.sample(n_surface), dtype=np.float64)       # object frame, metres

    def body_vertices(self, pose: np.ndarray, transl: np.ndarray, identity: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """(T, V, 3) body vertices in the SOMA world (OpenCV world with y, z flipped)."""
        T = len(pose)
        npz = self.tmp / "hold_params.npz"
        np.savez(npz, poses=pose.reshape(T, -1, 3).astype(np.float32), transl=transl.astype(np.float32),
                 identity_coeffs=identity.astype(np.float32), scale_params=scale.astype(np.float32),
                 joint_names=np.array(SOMA_JOINTS_ORDER), identity_model_type=np.array("mhr"),
                 unit=np.array("meters"), keep_root=np.array(True), absolute_pose=np.array(False))
        m = self.soma.load_motion(params_path=str(npz), normalize=False)
        return np.asarray(m["vertices"], dtype=np.float64)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of every run of True."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def mean_quat(q: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(q.T @ q)
    m = v[:, -1]
    return m if m @ q[0] >= 0 else -m


def hold_static(p: np.ndarray, q: np.ndarray, verts: np.ndarray, surface: np.ndarray, fps: float, args):
    """Frames with no body-object contact: the object cannot move, so hold its pose constant over each such run.

    Contact = some body vertex within --contact-dist of the object surface. Contact runs are dilated by
    --contact-pad (approach / release) and contact gaps shorter than that are closed. A no-contact run is held
    only if it lasts >= --min-static and the object already stays within --hold-max-move / --hold-max-rot of its
    median pose there (larger motion means a missed contact, not noise: that run is left alone and reported).
    Held poses are blended into the neighbouring contact frames over --contact-pad.
    """
    from scipy.ndimage import binary_closing, binary_dilation
    from scipy.spatial import cKDTree
    T = len(p)
    Rm = R.from_quat(q, scalar_first=True).as_matrix()
    vsub = verts[:, ::2]                                                     # every 2nd vertex (~1 cm spacing)
    dist = np.empty(T)
    for i in range(T):
        P = (surface @ Rm[i].T + p[i]) @ F_CV_TO_SOMA.T                     # object surface in the SOMA world
        d, _ = cKDTree(P).query(vsub[i], distance_upper_bound=0.5)
        dist[i] = d.min()
    contact = dist < args.contact_dist
    pad = max(1, int(round(args.contact_pad * fps)))
    if contact.any():
        contact = binary_closing(contact, structure=np.ones(2 * pad + 1), border_value=0) | contact
        contact = binary_dilation(contact, iterations=pad)
    min_len = int(round(args.min_static * fps))
    p2, q2 = p.copy(), continuous(q.copy())
    held, skipped, max_move, max_rot, max_skip = 0, [], 0.0, 0.0, 0.0
    for a, b in runs(~contact):
        if b - a < min_len:
            continue
        pm, qm = np.median(p2[a:b], axis=0), mean_quat(q2[a:b])
        mv = float(np.linalg.norm(p2[a:b] - pm, axis=1).max())
        rt = float(np.degrees((R.from_quat(q2[a:b], scalar_first=True) * R.from_quat(qm, scalar_first=True).inv()).magnitude().max()))
        if mv > args.hold_max_move or rt > args.hold_max_rot:
            skipped.append([int(a), int(b), round(mv * 100, 1), round(rt, 1)]); max_skip = max(max_skip, mv)
            continue
        pa, qa, pb, qb = p2[a].copy(), q2[a].copy(), p2[b - 1].copy(), q2[b - 1].copy()
        p2[a:b], q2[a:b] = pm, qm
        # blend the correction into the contact frames on both sides
        for ref_p, ref_q, frames in ((pa, qa, range(a - 1, max(-1, a - 1 - pad), -1)), (pb, qb, range(b, min(T, b + pad)))):
            dp = pm - ref_p
            dq = R.from_quat(qm, scalar_first=True) * R.from_quat(ref_q, scalar_first=True).inv()
            for j, f in enumerate(frames, start=1):
                w = 1.0 - j / (pad + 1)
                p2[f] = p2[f] + w * dp
                q2[f] = (R.from_rotvec(w * dq.as_rotvec()) * R.from_quat(q2[f], scalar_first=True)).as_quat(scalar_first=True)
        held += b - a; max_move = max(max_move, mv); max_rot = max(max_rot, rt)
    rep = {"contact_frac": round(float(contact.mean()), 3), "held_frac": round(float(held) / T, 3),
           "held_max_cm": round(max_move * 100, 2), "held_max_deg": round(max_rot, 2),
           "skipped_runs": skipped, "skipped_max_cm": round(max_skip * 100, 1)}
    return p2, continuous(q2), rep


def smooth_episode(src: Path, dst: Path, fps: float, args, hold: "HoldContext | None" = None) -> dict:
    t = pq.read_table(src)
    T = t.num_rows
    wh, wo = win_frames(args.win_human, fps, T, args.order), win_frames(args.win_object, fps, T, args.order)
    rep = {"frames": T, "win_human": wh, "win_object": wo}

    # human root translation
    tr = to_np(t, COL_TRANSL)
    tr_s = sg(tr, wh, args.order)
    rep["root_acc"] = (round(mean_acc(tr, fps), 2), round(mean_acc(tr_s, fps), 2))
    rep["root_shift_cm"] = round(float(np.linalg.norm(tr_s - tr, axis=1).mean() * 100), 2)

    # human joint rotations (axis-angle, 3 per joint)
    pose = to_np(t, COL_POSE)
    nj = pose.shape[1] // 3
    rv = pose.reshape(T, nj, 3)
    out = np.empty_like(rv)
    jdeg = []
    for j in range(nj):
        q = R.from_rotvec(rv[:, j]).as_quat(scalar_first=True)
        qs = smooth_quat(q, wh, args.order)
        out[:, j] = R.from_quat(qs, scalar_first=True).as_rotvec()
        jdeg.append(np.degrees((R.from_quat(qs, scalar_first=True) * R.from_quat(q, scalar_first=True).inv()).magnitude()).mean())
    rep["joint_shift_deg"] = round(float(np.mean(jdeg)), 2)

    # object pose (pos + quat wxyz), invisible frames filled first
    obj = to_np(t, COL_OBJ)
    vis = t.column(COL_VIS).to_numpy().astype(bool).reshape(-1) if COL_VIS in t.column_names else np.ones(T, bool)
    p, q = fill_invisible(obj[:, :3], obj[:, 3:7], vis)
    if args.obj_filter == "ssa":
        L = int(round(args.ssa_window * fps))
        p_s, rp = mssa(p, L, args.ssa_rank)
        q_s, rq = ssa_quat(q, L, args.ssa_rank)
        rep["ssa_rank"] = (rp, rq)
    else:
        p_s, q_s = sg(p, wo, args.order), smooth_quat(q, wo, args.order)
    rep["obj_acc"] = (round(mean_acc(p, fps), 2), round(mean_acc(p_s, fps), 2))
    rep["obj_rot_shift_deg"] = round(float(np.degrees((R.from_quat(q_s, scalar_first=True)
                                                       * R.from_quat(q, scalar_first=True).inv()).magnitude()).mean()), 2)
    rep["obj_angacc_deg"] = (round(ang_acc(q, fps), 1), round(ang_acc(q_s, fps), 1))
    rep["obj_shift_cm"] = round(float(np.linalg.norm(p_s - p, axis=1)[vis].mean() * 100), 2) if vis.any() else 0.0
    if hold is not None:
        verts = hold.body_vertices(out.reshape(T, -1), tr_s, to_np(t, COL_IDENTITY)[:1], to_np(t, COL_SCALE)[:1])
        p_s, q_s, hrep = hold_static(p_s, q_s, verts, hold.surface, fps, args)
        rep.update(hrep)
    obj_new = obj.copy(); obj_new[:, :3] = p_s; obj_new[:, 3:7] = q_s

    for name, x in ((COL_TRANSL, tr_s), (COL_POSE, out.reshape(T, -1)), (COL_OBJ, obj_new)):
        i = t.schema.get_field_index(name)
        t = t.set_column(i, t.schema.field(i), from_np(x, t.column(name)))
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, dst)                                        # schema metadata travels with the table
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--win-human", type=float, default=0.33, help="Savitzky-Golay window, seconds")
    ap.add_argument("--win-object", type=float, default=0.33, help="Savitzky-Golay window, seconds")
    ap.add_argument("--obj-filter", choices=["sg", "ssa"], default="sg",
                    help="object pose filter: Savitzky-Golay, or singular spectrum analysis (Hankel low rank)")
    ap.add_argument("--ssa-window", type=float, default=1.0, help="SSA embedding window, seconds")
    ap.add_argument("--ssa-rank", type=int, default=0, help="SSA components kept (0 = automatic, Gavish-Donoho)")
    ap.add_argument("--order", type=int, default=2)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--hold-static", action="store_true", help="stage 2: hold the object still where there is no contact")
    ap.add_argument("--contact-dist", type=float, default=0.04, help="body-object distance counted as contact, m")
    ap.add_argument("--contact-pad", type=float, default=0.2, help="contact dilation and blend length, s")
    ap.add_argument("--min-static", type=float, default=0.3, help="shortest no-contact run that is held, s")
    ap.add_argument("--hold-max-move", type=float, default=0.05, help="do not hold a run moving more than this, m")
    ap.add_argument("--hold-max-rot", type=float, default=10.0, help="do not hold a run rotating more than this, deg")
    ap.add_argument("--soma-data-root", default=None)
    args = ap.parse_args()
    if args.out.resolve() == args.root.resolve():
        raise SystemExit("--out must differ from --root")

    info = json.loads((args.root / "meta" / "info.json").read_text())
    fps = float(info.get("fps", 30))
    chunks = int(info.get("chunks_size", 1000))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # everything except data/ and meta/ is linked
    for p in args.root.iterdir():
        if p.name in ("data", "meta"):
            continue
        link = out / p.name
        if not link.exists() and not link.is_symlink():
            os.symlink(p.resolve(), link)
    (out / "meta").mkdir(exist_ok=True)
    for p in (args.root / "meta").iterdir():
        if p.is_file() and p.name != "info.json":
            shutil.copy2(p, out / "meta" / p.name)
    info2 = dict(info)
    info2["trajectory_variant"] = f"{info.get('trajectory_variant', args.root.name)}_denoise"
    info2["smoothing_note"] = (f"denoise_tier.py: Savitzky-Golay order {args.order}, human window {args.win_human}s, "
                               f"object window {args.win_object}s; rotations filtered as sign-continuous quaternions"
                               + (f"; object pose by SSA (window {args.ssa_window}s, rank {args.ssa_rank or 'auto'})"
                                  if args.obj_filter == "ssa" else "")
                               + (f"; object held still where body-object distance > {args.contact_dist} m"
                                  if args.hold_static else ""))
    (out / "meta" / "info.json").write_text(json.dumps(info2))

    metas = [json.loads(l) for l in (args.root / "meta" / "episodes_metadata.jsonl").read_text().splitlines() if l.strip()]
    eps = sorted(int(m["episode_index"]) for m in metas)
    if args.episodes is not None:
        eps = [e for e in eps if e in set(args.episodes)]
    objs = {int(m["episode_index"]): m.get("object", "") for m in metas}
    soma = None
    if args.hold_static:
        import tempfile
        import torch
        from robotic_grounding.retarget.read_soma import SOMA
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        global SOMA_JOINTS_ORDER
        from convert_tier_to_retarget import SOMA_JOINTS_ORDER as _J
        SOMA_JOINTS_ORDER = _J
        soma = SOMA(data_root=args.soma_data_root, identity_model_type="mhr",
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        tmpdir = Path(tempfile.mkdtemp())
    report = {}
    print(f"fps {fps}  windows: human {args.win_human}s, object {args.win_object}s, order {args.order}")
    print(" ep object               root acc m/s2     obj acc m/s2      obj ang-acc deg/s2   shift: root cm  obj cm  joint deg")
    for ep in eps:
        rel = info["data_path"].format(episode_chunk=ep // chunks, episode_index=ep)
        hold = None
        if soma is not None:
            mesh = args.root / info.get("mesh_path", "mesh/{object_name}/{object_name}.glb").format(object_name=objs[ep])
            hold = HoldContext(soma, mesh, tmpdir)
        r = smooth_episode(args.root / rel, out / rel, fps, args, hold)
        report[ep] = r
        print(f"{ep:3d} {objs.get(ep, ''):<18s} {r['root_acc'][0]:6.2f} -> {r['root_acc'][1]:5.2f}   "
              f"{r['obj_acc'][0]:6.2f} -> {r['obj_acc'][1]:5.2f}   {r['obj_angacc_deg'][0]:7.1f} -> {r['obj_angacc_deg'][1]:6.1f}"
              f"      {r['root_shift_cm']:5.2f}  {r['obj_shift_cm']:6.2f}  {r['joint_shift_deg']:6.2f}"
              f"   obj rot shift {r['obj_rot_shift_deg']:.1f} deg" + (f"  ssa rank pos/rot {r['ssa_rank'][0]}/{r['ssa_rank'][1]}"
                                                                   if "ssa_rank" in r else ""), flush=True)
        if "contact_frac" in r:
            print(f"      contact {r['contact_frac']:.2f}  held {r['held_frac']:.2f} (max {r['held_max_cm']:.1f} cm, "
                  f"{r['held_max_deg']:.1f} deg)  skipped {len(r['skipped_runs'])} moving no-contact run(s)"
                  + (f", max {r['skipped_max_cm']:.1f} cm" if r['skipped_runs'] else ""), flush=True)
    (out / "denoise_report.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                                        "episodes": report}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()