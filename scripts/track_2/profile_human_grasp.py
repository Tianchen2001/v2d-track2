#!/usr/bin/env python3
"""
Profile the HUMAN grasp in each Track 2 episode from the SOMA-X hand joints, to decide which
episodes a 3-finger Dex3 (7 DOF, ~0.76 N m per joint, 500 g rated payload) can plausibly do.

Per frame and hand (from the SOMA joints in the retarget input folder, object pose from poses.npy):
  fingertip gap (5 tips) to the object surface, palm gap, number of fingertips within --touch,
  finger closure (mean PIP flexion of index..pinky, deg), thumb-index aperture (m),
  contact = palm within --touch or >= 2 fingertips within --touch.
Per episode:
  contact fraction per hand, two-hand fraction, fraction of contact frames using >= 4 fingers
  (ring/pinky = beyond Dex3), object lift (m) and cumulative object rotation (deg) while held,
  max object angular speed while held, and a coarse grasp class:
    none      : hands almost never touch the object
    push/carry: contact but object never lifted > --lift, or mostly palm contact
    pinch     : <= 3 fingertips in contact most of the time
    power     : >= 4 fingertips in contact most of the time
  plus a dex3_risk score 0..3 (power grasp +1, rotation while held > 45 deg +1, one-handed lift +1).

    python profile_human_grasp.py --toolkit video_to_data \
        --inputs "runs/track2/tier1/retarget_in/sequence_id=*" \
        --manifest runs/track2/tier1/motion/retarget_results.json \
        --out runs/track2/tier1/grasp_profile

Runs in the retarget env (py-soma-x, torch, trimesh, scipy; robotic_grounding importable via --toolkit).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

R_CV_TO_SOMA = np.diag([1.0, -1.0, -1.0])  # OpenCV world -> SOMA world (soma_to_g1.py)
FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"]


class MeshGap:
    """Signed gap (positive outside) from points in the mesh local frame, via dense surface samples."""
    def __init__(self, mesh: trimesh.Trimesh, spacing: float = 0.003, n_min: int = 30_000, n_max: int = 300_000):
        n = int(np.clip(mesh.area / spacing**2, n_min, n_max))
        pts, fid = trimesh.sample.sample_surface(mesh, n)
        self.pts = np.asarray(pts, dtype=np.float64); self.nrm = np.asarray(mesh.face_normals[fid], dtype=np.float64)
        self.tree = cKDTree(self.pts); self.n = n

    def __call__(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64).reshape(-1, 3)
        d, i = self.tree.query(q, k=min(8, self.n), workers=1)
        dots = np.einsum("ijk,ijk->ij", q[:, None, :] - self.pts[i], self.nrm[i])
        return np.where(dots.max(axis=1) > 1e-4, 1.0, -1.0) * d[:, 0]


def load_mesh(path: Path) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh")
    if not isinstance(m, trimesh.Trimesh):
        m = trimesh.util.concatenate(m.dump())
    return m


def angle(a, b, c):
    """Angle at b (deg) between segments b->a and b->c; 180 = straight."""
    u = a - b; v = c - b
    cos = np.einsum("ij,ij->i", u, v) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def profile_episode(folder: Path, soma, args) -> tuple[dict, list[dict]]:
    sid = folder.name.removeprefix("sequence_id=")
    motion = soma.load_motion(folder / "soma_params.npz", normalize=False)
    J = motion["joints"]; names = list(motion["joint_names"]); ix = {n: names.index(n) for n in names}
    poses = np.load(folder / "poses.npy")
    T = min(len(J), len(poses))
    mesh = load_mesh(folder / "reconstructed_mesh" / "output_aligned.glb")
    gap = MeshGap(mesh)
    ep = None
    try:
        ep = json.loads((folder / "episode.json").read_text())
    except Exception:  # noqa: BLE001
        pass

    rows = []
    obj_R = np.zeros((T, 3, 3)); obj_t = np.zeros((T, 3))
    for t in range(T):
        obj_R[t] = R_CV_TO_SOMA @ poses[t, :3, :3]; obj_t[t] = R_CV_TO_SOMA @ poses[t, :3, 3]
    obj_min_y0 = None
    for t in range(T):
        r = {"frame": t}
        for side in ("Left", "Right"):
            tips = np.stack([J[t, ix[f"{side}Hand{f}End"]] for f in FINGERS])
            palm = 0.5 * (J[t, ix[f"{side}Hand"]] + J[t, ix[f"{side}HandMiddle1"]])
            g = gap((np.vstack([tips, palm[None]]) - obj_t[t]) @ obj_R[t])
            tip_gap, palm_gap = g[:5], g[5]
            n_touch = int((tip_gap < args.touch).sum())
            flex = np.mean([angle(J[t, ix[f"{side}Hand{f}1"]][None], J[t, ix[f"{side}Hand{f}2"]][None], J[t, ix[f"{side}Hand{f}3"]][None])[0]
                            for f in FINGERS[1:]])
            s = side.lower()
            r[f"{s}_palm_gap"] = float(palm_gap); r[f"{s}_tip_min_gap"] = float(tip_gap.min())
            r[f"{s}_n_tips_touch"] = n_touch
            r[f"{s}_ring_pinky_touch"] = int((tip_gap[3:] < args.touch).any())
            r[f"{s}_closure_deg"] = float(180.0 - flex)  # 0 = straight fingers, larger = more curled
            r[f"{s}_aperture"] = float(np.linalg.norm(tips[0] - tips[1]))
            r[f"{s}_contact"] = int(palm_gap < args.touch or n_touch >= 2)
        rows.append(r)

    # object motion while held (orientation smoothed over ~0.3 s before differencing: raw poses jitter 1-3 deg/frame)
    held = np.array([r["left_contact"] or r["right_contact"] for r in rows], dtype=bool)
    both = np.array([r["left_contact"] and r["right_contact"] for r in rows], dtype=bool)
    verts = mesh.vertices[np.random.default_rng(0).choice(len(mesh.vertices), size=min(2000, len(mesh.vertices)), replace=False)]
    low = np.array([(verts @ obj_R[t].T + obj_t[t])[:, 1].min() for t in range(T)])  # SOMA world is y-up
    lift = low - low[:T // 10 + 1].min()
    fps = float(ep.get("fps", 30.0)) if ep else 30.0
    k = max(1, int(round(0.3 * fps)))
    rv = R.from_matrix(obj_R).as_rotvec()
    # moving average on rotvec is fine for small windows; re-orthonormalise through from_rotvec
    rv_s = np.stack([np.convolve(np.pad(rv[:, j], (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid") for j in range(3)], 1)
    Rs = R.from_rotvec(rv_s)
    rot_step = np.zeros(T)
    for t in range(1, T):
        rot_step[t] = np.degrees((Rs[t] * Rs[t - 1].inv()).magnitude())
    ang_speed = rot_step * fps
    # net rotation over the longest held segment
    net_rot = 0.0
    if held.any():
        idx = np.flatnonzero(held); splits = np.flatnonzero(np.diff(idx) > 1)
        segs = np.split(idx, splits + 1); seg = max(segs, key=len)
        net_rot = float(np.degrees((Rs[seg[-1]] * Rs[seg[0]].inv()).magnitude()))

    def frac(mask):
        return float(mask.mean()) if len(mask) else 0.0
    summ = {"sequence_id": sid, "frames": T, "object": (ep or {}).get("object", ""), "episode_index": (ep or {}).get("episode_index")}
    for s in ("left", "right"):
        c = np.array([r[f"{s}_contact"] for r in rows], dtype=bool)
        nt = np.array([r[f"{s}_n_tips_touch"] for r in rows]); rp = np.array([r[f"{s}_ring_pinky_touch"] for r in rows], dtype=bool)
        summ[f"{s}_contact_frac"] = frac(c)
        summ[f"{s}_power_frac"] = frac(nt[c] >= 4) if c.any() else 0.0
        summ[f"{s}_ring_pinky_frac"] = frac(rp[c]) if c.any() else 0.0
        summ[f"{s}_median_tips_when_contact"] = float(np.median(nt[c])) if c.any() else 0.0
        summ[f"{s}_palm_only_frac"] = frac((nt[c] < 2)) if c.any() else 0.0
        clo = np.array([r[f"{s}_closure_deg"] for r in rows]); ap_ = np.array([r[f"{s}_aperture"] for r in rows])
        summ[f"{s}_closure_when_contact_deg"] = float(np.median(clo[c])) if c.any() else None
        summ[f"{s}_aperture_when_contact_cm"] = float(np.median(ap_[c]) * 100) if c.any() else None
        # per-frame grasp type for this hand
        wrap = c & (nt >= 3) & (clo >= args.closure)          # fingers on the object and curled: enveloping grasp
        flat = c & (nt >= 3) & (clo < args.flat)              # fingers on the object but straight: palm/flat contact
        pinch = c & (nt >= 2) & (nt <= 3) & ~flat & ~wrap
        summ[f"{s}_wrap_frac"] = frac(wrap[c]) if c.any() else 0.0
        summ[f"{s}_flat_frac"] = frac(flat[c]) if c.any() else 0.0
        summ[f"{s}_pinch_frac"] = frac(pinch[c]) if c.any() else 0.0
    summ["any_contact_frac"] = frac(held); summ["two_hand_frac"] = frac(both)
    summ["max_lift_m"] = float(lift.max())
    summ["rotation_while_held_deg"] = float(rot_step[held].sum())
    summ["net_rotation_longest_hold_deg"] = net_rot
    summ["max_ang_speed_while_held_dps"] = float(ang_speed[held].max()) if held.any() else 0.0
    one_hand_lift = bool(((lift > args.lift) & held & ~both).any())
    wrap = max(summ["left_wrap_frac"], summ["right_wrap_frac"]); flat = max(summ["left_flat_frac"], summ["right_flat_frac"])
    if summ["any_contact_frac"] < 0.1:
        cls = "none"
    elif wrap >= 0.4:
        cls = "wrap"
    elif flat >= 0.4:
        cls = "flat"
    else:
        cls = "pinch"
    summ["grasp_class"] = cls
    summ["dex3_risk"] = (int(cls == "wrap") + int(cls != "none" and net_rot > 45) + int(cls != "none" and one_hand_lift)
                         + int(cls == "wrap" and min(v for v in (summ["left_aperture_when_contact_cm"] or 99, summ["right_aperture_when_contact_cm"] or 99)) < 7))
    return summ, rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toolkit", type=Path, required=True)
    ap.add_argument("--inputs", required=True, help="glob of retarget input folders (sequence_id=*)")
    ap.add_argument("--manifest", type=Path, default=None, help="retarget_results.json for episode/object names if episode.json is missing")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--touch", type=float, default=0.025, help="m, fingertip/palm counted as touching below this gap (label threshold is 0.02)")
    ap.add_argument("--closure", type=float, default=35.0, help="deg, median PIP flexion above which a contact frame counts as an enveloping grasp")
    ap.add_argument("--flat", type=float, default=20.0, help="deg, below which fingers count as straight (flat/palm contact)")
    ap.add_argument("--lift", type=float, default=0.03, help="m, object counted as lifted above this rise of its lowest point")
    args = ap.parse_args()
    sys.path.insert(0, str(args.toolkit.resolve() / "robotic_grounding" / "source" / "robotic_grounding"))
    import torch
    from robotic_grounding.retarget.read_soma import SOMA
    soma = SOMA(identity_model_type="mhr", device=torch.device("cpu"))

    index = {}
    if args.manifest and args.manifest.exists():
        for e in json.loads(args.manifest.read_text()):
            if isinstance(e, dict) and e.get("sequence_id"):
                index[e["sequence_id"]] = e
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for folder in sorted(Path(p) for p in glob.glob(args.inputs)):
        if not (folder / "soma_params.npz").exists():
            continue
        print(f"[{folder.name}]", flush=True)
        summ, rows = profile_episode(folder, soma, args)
        meta = index.get(summ["sequence_id"], {})
        if summ.get("episode_index") is None:
            summ["episode_index"] = meta.get("episode_index")
        if not summ.get("object"):
            summ["object"] = meta.get("object", "")
        with (args.out / f"{summ['sequence_id']}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        summaries.append(summ)
        print(f"  {summ['grasp_class']:<10} risk {summ['dex3_risk']}  contact L {summ['left_contact_frac']:.2f} R {summ['right_contact_frac']:.2f}  "
              f"power L {summ['left_power_frac']:.2f} R {summ['right_power_frac']:.2f}  lift {summ['max_lift_m']*100:.0f} cm  rot held {summ['rotation_while_held_deg']:.0f} deg")
    summaries.sort(key=lambda s: (s["episode_index"] is None, s["episode_index"] or 0))
    keys = ["episode_index", "object", "sequence_id", "frames", "grasp_class", "dex3_risk", "any_contact_frac", "two_hand_frac",
            "left_contact_frac", "right_contact_frac", "left_wrap_frac", "right_wrap_frac", "left_flat_frac", "right_flat_frac",
            "left_pinch_frac", "right_pinch_frac", "left_power_frac", "right_power_frac", "left_ring_pinky_frac", "right_ring_pinky_frac",
            "left_median_tips_when_contact", "right_median_tips_when_contact", "left_palm_only_frac", "right_palm_only_frac",
            "left_closure_when_contact_deg", "right_closure_when_contact_deg", "left_aperture_when_contact_cm", "right_aperture_when_contact_cm",
            "max_lift_m", "rotation_while_held_deg", "net_rotation_longest_hold_deg", "max_ang_speed_while_held_dps"]
    with (args.out / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(summaries)
    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2))
    def fv(v, fmt="{:.0f}"):
        return "-" if v is None else fmt.format(v)
    print(f"\n{'ep':>3} {'object':<18} {'class':<6} {'risk':>4} {'contact':>7} {'2hand':>5} {'wrapL':>5} {'wrapR':>5} {'flatL':>5} {'flatR':>5} {'closL':>5} {'closR':>5} {'apL':>4} {'apR':>4} {'lift':>5} {'netrot':>6} {'rot':>5}")
    print("   wrap/flat = fraction of contact frames with >=3 fingertips on the object and fingers curled / straight; clos = median PIP flexion deg; ap = thumb-index aperture cm; netrot = net object rotation over the longest hold; rot = smoothed cumulative rotation while held")
    for s in summaries:
        print(f"{str(s['episode_index']):>3} {str(s['object'])[:18]:<18} {s['grasp_class']:<6} {s['dex3_risk']:>4} {s['any_contact_frac']:>7.2f} {s['two_hand_frac']:>5.2f} "
              f"{s['left_wrap_frac']:>5.2f} {s['right_wrap_frac']:>5.2f} {s['left_flat_frac']:>5.2f} {s['right_flat_frac']:>5.2f} "
              f"{fv(s['left_closure_when_contact_deg']):>5} {fv(s['right_closure_when_contact_deg']):>5} {fv(s['left_aperture_when_contact_cm']):>4} {fv(s['right_aperture_when_contact_cm']):>4} "
              f"{s['max_lift_m']*100:>5.0f} {s['net_rotation_longest_hold_deg']:>6.0f} {s['rotation_while_held_deg']:>5.0f}")
    print(f"wrote {args.out / 'summary.csv'}")


if __name__ == "__main__":
    main()