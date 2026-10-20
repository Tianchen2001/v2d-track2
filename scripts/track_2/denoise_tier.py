#!/usr/bin/env python3
"""
Denoise a Track 2 tier (LeRobot layout), writing a new tier folder in the same layout, so
convert_tier_to_retarget.py / run_retarget_all.py / run_train_all.py / the packer (--frame-references) use it unchanged.

Stage 1 (always): temporal smoothing to remove frame-to-frame jitter.
  --obj-filter ssa: the object pose uses multichannel singular spectrum analysis instead (Hankel low-rank, rank
  chosen by the Gavish-Donoho threshold unless --ssa-rank is given); the body keeps Savitzky-Golay.
Stage 1b (--vision-prior <tier folder>): object orientation corrected towards a vision reconstruction of the same
  videos (e.g. tier_3_reconstruction_v1) by a slowly varying rotation (robust RTS smoother); see vision_orientation.
Stage 2 (--hold-static): frames where no body vertex touches the object -> the object cannot move, so its pose is
  held constant over each such run (see hold_static). Needs the retarget environment (robotic_grounding + py-soma-x).
--snap-stable (with --hold-static): the closing rest (reaching the last frame) is rotated to the nearest stable pose of
  the mesh about gravity (trimesh stable poses on a simplified hull; tilt only, yaw kept), put on the ground if near it,
  and faded in over --snap-blend seconds before it with a cosine ramp.
Anchor (default on, --no-anchor-first to disable): the first frame of the input is trusted; the opening rest is held
  at that pose, the vision correction is pinned to zero at frame 0, and frame 0 of the output equals the input.
Per-episode grip constraints for thin handles are a separate later step: grasp_refine.py, run on this output.

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
def stable_downs(mesh, max_points: int = 3000, max_faces: int = 400) -> np.ndarray | None:
    """Object-frame 'down' direction of every stable resting pose of the mesh on a flat floor (trimesh), computed on
    a simplified convex hull (stable poses only depend on the hull)."""
    import trimesh
    try:
        V = np.asarray(mesh.vertices)
        if len(V) > max_points:
            V = V[np.random.default_rng(0).choice(len(V), max_points, replace=False)]
        hull = trimesh.convex.convex_hull(V)
        if len(hull.faces) > max_faces:
            try:
                hull = hull.simplify_quadric_decimation(face_count=max_faces).convex_hull
            except Exception:
                pass
        com = mesh.center_mass if mesh.is_watertight else hull.center_mass
        Ts, _ = trimesh.poses.compute_stable_poses(hull, center_mass=com, n_samples=1)
        return np.array([T[:3, :3].T @ np.array([0.0, 0.0, -1.0]) for T in Ts])
    except Exception as e:
        print(f"   stable poses unavailable: {e}")
        return None


def snap_stable(pm: np.ndarray, qm: np.ndarray, ctx: "HoldContext", args) -> tuple[np.ndarray, np.ndarray, float]:
    """Gravity: rotate a resting pose to the nearest stable pose of the mesh (tilt only; the rotation is the smallest
    one, so the yaw about the vertical is kept), then, if the object is near the ground, put its lowest vertex on it.
    Returns the new position, quaternion and the tilt removed (deg; 0 = unchanged)."""
    if ctx.stable_down is None or len(ctx.stable_down) == 0:
        return pm, qm, 0.0
    Rm = R.from_quat(qm, scalar_first=True)
    u = Rm.inv().apply(-ctx.up)                                       # world down, in the object frame
    ang = np.degrees(np.arccos(np.clip(ctx.stable_down @ u, -1, 1)))
    i = int(np.argmin(ang))
    if ang[i] > args.snap_max_deg or ang[i] < 0.05:
        return pm, qm, 0.0
    v = ctx.stable_down[i]
    axis = np.cross(v, u); s_ = np.linalg.norm(axis)
    delta = R.from_rotvec(axis / s_ * np.arctan2(s_, v @ u)) if s_ > 1e-9 else R.identity()   # object frame: v -> u
    Rn = Rm * delta                                                   # now Rn v = world down
    pn = pm.copy()
    if ctx.ground_off is not None:
        h = (Rn.apply(ctx.vertices) + pm) @ ctx.up + ctx.ground_off   # vertex heights above the ground
        if abs(h.min()) <= args.snap_ground_dist:                     # resting on the floor (not on a table)
            pn = pm - (h.min() - 0.002) * ctx.up
    return pn, Rn.as_quat(scalar_first=True), float(ang[i])


class HoldContext:
    """SOMA body model + object surface samples for one episode (needs the retarget environment)."""

    def __init__(self, soma, mesh_path: Path, tmp: Path, n_surface: int = 3000, prior: Path | None = None,
                 plane: Path | None = None, stable: bool = False):
        import trimesh
        self.soma, self.tmp, self.prior = soma, tmp, prior          # prior: same-episode parquet of the vision prior
        self.ep = -1
        up, off = np.array([0.0, -1.0, 0.0]), None                  # OpenCV world: -y is roughly up
        if plane is not None and plane.exists():
            pl = np.asarray(json.loads(plane.read_text())["plane"], float)
            s_ = np.linalg.norm(pl[:3]); n, d = pl[:3] / s_, pl[3] / s_
            if n @ up < 0:
                n, d = -n, -d
            up, off = n, d                                           # height of a point x above ground: up.x + off
        self.up, self.ground_off = up, off
        m = trimesh.load(str(mesh_path), force="mesh")
        self.vertices = np.asarray(m.vertices, dtype=np.float64)
        self.stable_down = None
        if stable:
            self.stable_down = stable_downs(m)
        try:                                                         # fixed seed: same contact mask every run
            pts = m.sample(n_surface, seed=0)
        except TypeError:                                            # older trimesh: no seed argument
            np.random.seed(0)
            pts = m.sample(n_surface)
        self.surface = np.asarray(pts, dtype=np.float64)              # object frame, metres

    def body_vertices(self, pose: np.ndarray, transl: np.ndarray, identity: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return self.body_motion(pose, transl, identity, scale)[1]

    def body_motion(self, pose: np.ndarray, transl: np.ndarray, identity: np.ndarray, scale: np.ndarray):
        """(T, J, 3) joints and (T, V, 3) vertices in the SOMA world (OpenCV world with y, z flipped)."""
        T = len(pose)
        npz = self.tmp / "hold_params.npz"
        np.savez(npz, poses=pose.reshape(T, -1, 3).astype(np.float32), transl=transl.astype(np.float32),
                 identity_coeffs=identity.astype(np.float32), scale_params=scale.astype(np.float32),
                 joint_names=np.array(SOMA_JOINTS_ORDER), identity_model_type=np.array("mhr"),
                 unit=np.array("meters"), keep_root=np.array(True), absolute_pose=np.array(False))
        m = self.soma.load_motion(params_path=str(npz), normalize=False)
        return np.asarray(m["joints"], dtype=np.float64), np.asarray(m["vertices"], dtype=np.float64)


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of every run of True."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def mean_quat(q: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(q.T @ q)
    m = v[:, -1]
    return m if m @ q[0] >= 0 else -m


def hold_static(p: np.ndarray, q: np.ndarray, verts: np.ndarray, surface: np.ndarray, fps: float, args,
                anchor: tuple[np.ndarray, np.ndarray] | None = None, ctx: "HoldContext | None" = None):
    """Frames with no body-object contact: the object cannot move, so hold its pose constant over each such run.

    Contact = some body vertex within --contact-dist of the object surface. Contact runs are dilated by
    --contact-pad (approach / release) and contact gaps shorter than that are closed. A no-contact run is held
    only if it lasts >= --min-static and the object already stays within --hold-max-move / --hold-max-rot of its
    median pose there (larger motion means a missed contact, not noise: that run is left alone and reported).
    Held poses are blended into the neighbouring contact frames over --contact-pad.
    With `anchor` (first-frame pose of the input), a held run starting at frame 0 takes that pose instead of the
    run median.
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
    snapped = []
    for a, b in runs(~contact):
        if b - a < min_len:
            continue
        pm, qm = np.median(p2[a:b], axis=0), mean_quat(q2[a:b])
        if a == 0 and anchor is not None:                    # the opening rest: keep the (trusted) first frame
            pm, qm = anchor[0].copy(), anchor[1] if anchor[1] @ q2[0] >= 0 else -anchor[1]
        elif args.snap_stable and ctx is not None and b == T:  # the closing rest: gravity, nearest stable pose
            pm, qm, sd = snap_stable(pm, qm, ctx, args)
            if qm @ q2[a] < 0:
                qm = -qm
            if sd > 0:
                snapped.append([int(a), int(b), round(sd, 1)])
        mv = float(np.linalg.norm(p2[a:b] - pm, axis=1).max())
        rt = float(np.degrees((R.from_quat(q2[a:b], scalar_first=True) * R.from_quat(qm, scalar_first=True).inv()).magnitude().max()))
        if mv > args.hold_max_move or rt > args.hold_max_rot:
            skipped.append([int(a), int(b), round(mv * 100, 1), round(rt, 1)]); max_skip = max(max_skip, mv)
            continue
        pa, qa, pb, qb = p2[a].copy(), q2[a].copy(), p2[b - 1].copy(), q2[b - 1].copy()
        p2[a:b], q2[a:b] = pm, qm
        # blend the correction into the contact frames on both sides
        snapped_here = bool(snapped) and snapped[-1][0] == a
        nb = max(1, int(round(args.snap_blend * fps))) if snapped_here else pad
        for ref_p, ref_q, frames in ((pa, qa, range(a - 1, max(-1, a - 1 - nb), -1)), (pb, qb, range(b, min(T, b + nb)))):
            dp = pm - ref_p
            dq = R.from_quat(qm, scalar_first=True) * R.from_quat(ref_q, scalar_first=True).inv()
            for j, f in enumerate(frames, start=1):
                # snapped closing rest: long cosine fade; other rests: short linear fade over the contact pad
                w = 0.5 * (1.0 + np.cos(np.pi * j / (nb + 1))) if snapped_here else 1.0 - j / (pad + 1)
                p2[f] = p2[f] + w * dp
                q2[f] = (R.from_rotvec(w * dq.as_rotvec()) * R.from_quat(q2[f], scalar_first=True)).as_quat(scalar_first=True)
        held += b - a; max_move = max(max_move, mv); max_rot = max(max_rot, rt)
    rep = {"contact_frac": round(float(contact.mean()), 3), "held_frac": round(float(held) / T, 3),
           "held_max_cm": round(max_move * 100, 2), "held_max_deg": round(max_rot, 2),
           "skipped_runs": skipped, "skipped_max_cm": round(max_skip * 100, 1), "snapped_runs": snapped}
    return p2, continuous(q2), rep


# --------------------------------------------- stage 1b: object orientation corrected towards a vision reconstruction
def kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rigid R, t minimising |R src + t - dst| over (N, 3) point pairs."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - ms).T @ (dst - md))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rm = Vt.T @ D @ U.T
    return Rm, md - Rm @ ms


def swing_twist(c: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation vectors (T, 3) -> signed twist angle about `up` and remaining swing (tilt) angle, both rad."""
    q = R.from_rotvec(c).as_quat(scalar_first=True)
    proj = q[:, 1:] @ up
    twist = 2 * np.arctan2(proj, q[:, 0])
    twist = (twist + np.pi) % (2 * np.pi) - np.pi
    swing = R.from_rotvec(c) * R.from_rotvec(np.outer(-twist, up))
    return twist, swing.magnitude()


def rts_huber(z: np.ndarray, w: np.ndarray, fps: float, q_deg: float, r_deg: float, iters: int = 3, k: float = 1.5):
    """Robust Rauch-Tung-Striebel smoother of a (T, 3) random-walk signal from weighted measurements z (w = 0: none).

    State x_t = x_{t-1} + process noise (q_deg per sqrt(second)); measurement z_t = x_t + noise (r_deg). Measurements
    are re-weighted with a Huber function of their residual to the smoothed estimate (iters passes). Isotropic noise,
    so the three components share one scalar covariance."""
    T = len(z)
    Q = np.radians(q_deg) ** 2 / fps
    Rm = np.radians(r_deg) ** 2
    s = np.radians(r_deg)
    wt = w.astype(float).copy()
    xs = np.zeros_like(z)
    for _ in range(iters):
        x = np.zeros_like(z); P = np.zeros(T); xp = np.zeros_like(z); Pp = np.zeros(T)
        good = wt > 0
        x_prev = np.median(z[good], axis=0) if good.any() else np.zeros(3)
        P_prev = np.radians(60.0) ** 2
        for t in range(T):
            xp[t], Pp[t] = x_prev, P_prev + (Q if t else 0.0)
            if wt[t] > 0:
                K = Pp[t] / (Pp[t] + Rm / wt[t])
                x[t], P[t] = xp[t] + K * (z[t] - xp[t]), (1 - K) * Pp[t]
            else:
                x[t], P[t] = xp[t], Pp[t]
            x_prev, P_prev = x[t], P[t]
        xs = x.copy()
        for t in range(T - 2, -1, -1):
            G = P[t] / Pp[t + 1]
            xs[t] = x[t] + G * (xs[t + 1] - xp[t + 1])
        e = np.linalg.norm(z - xs, axis=1)
        wt = w * np.where(e <= k * s, 1.0, k * s / np.maximum(e, 1e-9))
    return xs


def vision_orientation(q: np.ndarray, vis: np.ndarray, joints_t: np.ndarray, prior: pa.Table, ctx: "HoldContext",
                       fps: float, args) -> tuple[np.ndarray, dict]:
    """Correct the object orientation by a slowly varying rotation estimated from a vision reconstruction.

    Model: true(t) = C(t) tier(t), recon(t) = A true(t) noise(t). A (recon world -> tier world) comes from the human
    joints of both (Kabsch over all frames), never from the object. Per frame residual r(t) = log(A^-1 recon(t)
    tier(t)^-1); frames where either object is invisible, the recon jumps (r more than --vp-jump deg away from its
    1 s running median, +-2 frames), or |r| > --vp-max-res are ignored. C(t) = exp(c(t)), c smoothed by a robust RTS
    smoother (random walk --vp-q deg/sqrt(s), measurement noise --vp-meas deg). Positions are not touched."""
    from scipy.ndimage import binary_dilation
    T = min(len(q), prior.num_rows)
    J3, _ = ctx.body_motion(to_np(prior, COL_POSE)[:T], to_np(prior, COL_TRANSL)[:T],
                            to_np(prior, COL_IDENTITY)[:1], to_np(prior, COL_SCALE)[:1])
    a2 = joints_t[:T:3].reshape(-1, 3) @ F_CV_TO_SOMA                        # SOMA world -> OpenCV world
    a3 = J3[::3].reshape(-1, 3) @ F_CV_TO_SOMA
    Ra, ta = kabsch(a3, a2)
    body_rms = float(np.sqrt(((a3 @ Ra.T + ta - a2) ** 2).sum(1).mean()))
    o3 = to_np(prior, COL_OBJ)[:T]
    vis3 = prior.column(COL_VIS).to_numpy().astype(bool).reshape(-1)[:T] if COL_VIS in prior.column_names else np.ones(T, bool)
    R3 = R.from_matrix(Ra) * R.from_quat(o3[:, 3:7], scalar_first=True)
    R2 = R.from_quat(q[:T], scalar_first=True)
    r = (R3 * R2.inv()).as_rotvec()
    from scipy.ndimage import median_filter
    loc = median_filter(r, size=(max(3, int(round(fps)) | 1), 1), mode="nearest")      # ~1 s running median
    jump = binary_dilation(np.linalg.norm(r - loc, axis=1) > np.radians(args.vp_jump), iterations=2)
    valid = vis[:T] & vis3 & ~jump & (np.linalg.norm(r, axis=1) < np.radians(args.vp_max_res))
    rep = {"vp_frame_offset_deg": round(float(np.degrees(R.from_matrix(Ra).magnitude())), 1),
           "vp_frame_offset_cm": round(float(np.linalg.norm(ta) * 100), 1), "vp_body_rms_cm": round(body_rms * 100, 1),
           "vp_used": round(float(valid.mean()), 3), "vp_jump_frames": int(jump.sum())}
    if valid.sum() < 10:
        rep["vp_skipped"] = "fewer than 10 usable frames"
        return q, rep
    if rep["vp_body_rms_cm"] > args.vp_max_body_rms:
        rep["vp_skipped"] = (f"bodies do not align (rms {rep['vp_body_rms_cm']} cm > {args.vp_max_body_rms:g} cm): "
                             f"reconstruction not trusted")
        return q, rep
    w = valid.astype(float)
    if args.anchor_first:                                     # the first frame is trusted: no correction there
        r = r.copy(); r[0] = 0.0; w[0] = 100.0
        if args.vp_ramp > 0:                                  # and the reconstruction is trusted less early on
            w[1:] *= np.clip(np.arange(1, T) / (args.vp_ramp * fps), 0.0, 1.0) ** 2
    c = rts_huber(r, w, fps, args.vp_q, args.vp_meas)
    if T < len(q):
        c = np.vstack([c, np.repeat(c[-1:], len(q) - T, 0)])
    # split the correction into yaw (about the world up axis, harmless for stability) and tilt (the rest)
    tail = max(1, len(q) // 10)
    yaw, tilt = swing_twist(c, ctx.up)
    for name, v in (("", np.linalg.norm(c, axis=1)), ("_yaw", np.abs(yaw)), ("_tilt", tilt)):
        d = np.degrees(v)
        rep.update({f"vp_corr{name}_mean_deg": round(float(d.mean()), 1), f"vp_corr{name}_end_deg": round(float(d[-tail:].mean()), 1)})
    rep["vp_corr_max_deg"] = round(float(np.degrees(np.linalg.norm(c, axis=1)).max()), 1)
    if args.vp_tilt_only or ctx.ep in set(args.vp_tilt_only_episodes or []):
        rep["vp_tilt_only"] = True
        c = (R.from_rotvec(c) * R.from_rotvec(np.outer(-yaw, ctx.up))).as_rotvec()     # drop the twist about up
    qn = (R.from_rotvec(c) * R.from_quat(q, scalar_first=True)).as_quat(scalar_first=True)
    return continuous(qn), rep


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
        joints, verts = hold.body_motion(out.reshape(T, -1), tr_s, to_np(t, COL_IDENTITY)[:1], to_np(t, COL_SCALE)[:1])
        if hold.prior is not None:
            q_s, vrep = vision_orientation(q_s, vis, joints, pq.read_table(hold.prior), hold, fps, args)
            rep.update(vrep)
        if args.hold_static:
            p_s, q_s, hrep = hold_static(p_s, q_s, verts, hold.surface, fps, args,
                                         anchor=(p[0], q[0]) if args.anchor_first else None, ctx=hold)
            rep.update(hrep)
    if args.anchor_first:                                     # pull frame 0 back onto the input, fading over 0.5 s
        n = min(T, max(2, int(round(0.5 * fps))))
        wgt = np.clip(1.0 - np.arange(n) / n, 0.0, 1.0)
        dp = p[0] - p_s[0]
        dq = (R.from_quat(q[0], scalar_first=True) * R.from_quat(q_s[0], scalar_first=True).inv()).as_rotvec()
        rep["anchor_first_cm_deg"] = (round(float(np.linalg.norm(dp) * 100), 2), round(float(np.degrees(np.linalg.norm(dq))), 2))
        p_s = p_s.copy(); p_s[:n] += wgt[:, None] * dp
        q_s = q_s.copy()
        q_s[:n] = (R.from_rotvec(wgt[:, None] * dq) * R.from_quat(q_s[:n], scalar_first=True)).as_quat(scalar_first=True)
        q_s = continuous(q_s)
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
    ap.add_argument("--no-anchor-first", dest="anchor_first", action="store_false",
                    help="do not trust the first frame (default: frame 0 pose of the input is kept; vision correction "
                         "starts from zero there; an opening rest is held at the first-frame pose)")
    ap.add_argument("--vp-max-body-rms", type=float, default=15.0,
                    help="vision prior: skip the episode if the aligned bodies still differ by more than this, cm")
    ap.add_argument("--vp-ramp", type=float, default=3.0,
                    help="with the first-frame anchor: reconstruction weight ramps from 0 to 1 over this many seconds")
    ap.add_argument("--vision-prior", type=Path, default=None,
                    help="tier folder of a vision reconstruction of the same episodes (e.g. tier_3_reconstruction_v1): "
                         "the object orientation is corrected towards it (slowly varying correction, positions kept)")
    ap.add_argument("--vp-tilt-only", action="store_true",
                    help="vision prior: apply only the tilt part of the correction, keep the yaw about the up axis")
    ap.add_argument("--snap-stable", action="store_true",
                    help="with --hold-static: the closing rest (the one reaching the last frame) takes the nearest stable "
                         "pose of the mesh about gravity (tilt only, yaw kept) and, if near the ground, sits on it")
    ap.add_argument("--snap-blend", type=float, default=1.0,
                    help="the snapped closing rest fades in over this many seconds before it (cosine)")
    ap.add_argument("--snap-max-deg", type=float, default=30.0, help="snap only rests within this of a stable pose")
    ap.add_argument("--snap-ground-dist", type=float, default=0.05,
                    help="put the lowest vertex on the ground if it is within this many metres of it")
    ap.add_argument("--vp-skip-episodes", type=int, nargs="*", default=None,
                    help="vision prior: do not use it for these episodes (reconstruction unreliable there)")
    ap.add_argument("--vp-tilt-only-episodes", type=int, nargs="*", default=None,
                    help="vision prior: apply only the tilt part for these episodes (all other episodes: tilt and yaw)")
    ap.add_argument("--vp-q", type=float, default=5.0, help="vision prior: correction random walk, deg per sqrt(s)")
    ap.add_argument("--vp-meas", type=float, default=10.0, help="vision prior: reconstruction orientation noise, deg")
    ap.add_argument("--vp-jump", type=float, default=30.0,
                    help="vision prior: ignore frames whose residual is this far from its 1 s running median, deg")
    ap.add_argument("--vp-max-res", type=float, default=60.0, help="vision prior: ignore frames disagreeing more, deg")
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
                                  if args.hold_static else "")
                               + (f"; object orientation corrected towards {args.vision_prior.name} (vision prior)"
                                  if args.vision_prior is not None else ""))
    (out / "meta" / "info.json").write_text(json.dumps(info2))

    metas = [json.loads(l) for l in (args.root / "meta" / "episodes_metadata.jsonl").read_text().splitlines() if l.strip()]
    eps = sorted(int(m["episode_index"]) for m in metas)
    if args.episodes is not None:
        eps = [e for e in eps if e in set(args.episodes)]
    objs = {int(m["episode_index"]): m.get("object", "") for m in metas}
    sids = {int(m["episode_index"]): m.get("sequence_id", "") for m in metas}
    soma = None
    if args.hold_static or args.vision_prior is not None:
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
            prior = None
            if args.vision_prior is not None and ep not in set(args.vp_skip_episodes or []):
                pinfo = json.loads((args.vision_prior / "meta" / "info.json").read_text())
                prior = args.vision_prior / pinfo["data_path"].format(episode_chunk=ep // chunks, episode_index=ep)
                if not prior.exists():
                    print(f"   ep {ep}: no vision prior at {prior}; orientation not corrected"); prior = None
            sid = sids.get(ep)
            gp = args.root / info.get("ground_plane_path", "ground_plane/{sequence_id}.json").format(sequence_id=sid)
            hold = HoldContext(soma, mesh, tmpdir, prior=prior, plane=gp, stable=args.snap_stable)
            hold.ep = ep
        r = smooth_episode(args.root / rel, out / rel, fps, args, hold)
        report[ep] = r
        print(f"{ep:3d} {objs.get(ep, ''):<18s} {r['root_acc'][0]:6.2f} -> {r['root_acc'][1]:5.2f}   "
              f"{r['obj_acc'][0]:6.2f} -> {r['obj_acc'][1]:5.2f}   {r['obj_angacc_deg'][0]:7.1f} -> {r['obj_angacc_deg'][1]:6.1f}"
              f"      {r['root_shift_cm']:5.2f}  {r['obj_shift_cm']:6.2f}  {r['joint_shift_deg']:6.2f}"
              f"   obj rot shift {r['obj_rot_shift_deg']:.1f} deg" + (f"  ssa rank pos/rot {r['ssa_rank'][0]}/{r['ssa_rank'][1]}"
                                                                   if "ssa_rank" in r else ""), flush=True)
        if "vp_used" in r:
            print(f"      vision prior: frame offset {r['vp_frame_offset_deg']:.1f} deg / {r['vp_frame_offset_cm']:.1f} cm "
                  f"(body rms {r['vp_body_rms_cm']:.1f} cm)  used {r['vp_used']:.2f}  jumps {r['vp_jump_frames']}  "
                  + (f"correction mean/end {r['vp_corr_mean_deg']:.1f}/{r['vp_corr_end_deg']:.1f} deg "
                     f"= yaw {r['vp_corr_yaw_mean_deg']:.1f}/{r['vp_corr_yaw_end_deg']:.1f} + tilt "
                     f"{r['vp_corr_tilt_mean_deg']:.1f}/{r['vp_corr_tilt_end_deg']:.1f}"
                     + (" (tilt only applied)" if r.get("vp_tilt_only") else "")
                     if "vp_corr_mean_deg" in r else r.get("vp_skipped", "")), flush=True)
        if "anchor_first_cm_deg" in r:
            print(f"      anchor: frame 0 pulled back {r['anchor_first_cm_deg'][0]:.2f} cm / {r['anchor_first_cm_deg'][1]:.2f} deg", flush=True)
        if r.get("snapped_runs"):
            print("      gravity snap: " + ", ".join(f"frames {a}-{b} tilt {d} deg" for a, b, d in r["snapped_runs"]), flush=True)
        if "contact_frac" in r:
            print(f"      contact {r['contact_frac']:.2f}  held {r['held_frac']:.2f} (max {r['held_max_cm']:.1f} cm, "
                  f"{r['held_max_deg']:.1f} deg)  skipped {len(r['skipped_runs'])} moving no-contact run(s)"
                  + (f", max {r['skipped_max_cm']:.1f} cm" if r['skipped_runs'] else ""), flush=True)
    (out / "denoise_report.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()},
                                                        "episodes": report}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()