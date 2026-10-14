#!/usr/bin/env python3
"""
Contact schedule and required wrench from the OBJECT motion alone.

Idea: apart from resting and free fall, any object motion needs an external force, and in these clips
that force can only come from the hands. So the object trajectory tells when contact must exist and
what wrench the hands must deliver, independent of the (noisy) hand reconstruction.

Per frame (object pose from the retargeted parquet, smoothed with a Savitzky-Golay filter):
  height of the lowest object point above its support (ground z=0, or a support prim under it),
  supported = lowest point within --rest-gap of the support,
  moving    = |v| > --v-thresh or |omega| > --w-thresh,
  free_fall = not supported and |a - g| < --ff-thresh,
  contact_required = (not supported and not free_fall) or (supported and moving),
  F = m (a - g)  [N per kg, m = 1 nominal],  alpha = angular acceleration [rad/s^2].
Segments of contact_required (holes/runs < --min-segment cleaned) are classified:
  push  : supported throughout, F mostly horizontal
  hold  : unsupported, F mostly vertical (carrying)
  hold+torque : unsupported and net rotation > --rot-thresh or large alpha
and for each the median force direction, the object-surface patch whose inward normal faces that
force (where a push would be applied), and the distance of that patch to the recorded palm poses.

    python contact_schedule.py --toolkit video_to_data \
        --motion-root /localscratch/wangti98/track2_tier1_motion --out runs/track2/tier1/contact_schedule [--episodes 4 7 3]
    python contact_schedule.py --toolkit video_to_data --parquet <data.parquet> --out <dir> [--support <usda>]

Outputs per episode: <name>.csv (per frame), <name>.json (segments), <name>.png; plus summary.csv.
Pure CPU; needs numpy, scipy, trimesh, pyarrow, matplotlib. Reuses render_retarget_check.support_boxes.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_retarget_check import load_mesh_any  # noqa: E402

G = np.array([0.0, 0.0, -9.81])


def support_prims(usda: Path | None):
    """[(center_xyz, size_xyz)] for Cube and Cylinder prims (axis Z) in a *_support.usda, best-effort regex."""
    if usda is None or not usda.exists() or usda.stat().st_size == 0:
        return []
    txt = usda.read_text(errors="ignore"); out = []
    for block in re.finditer(r'def (Cube|Cylinder) "[^"]*"\s*\{(.*?)\n\s*\}', txt, re.S):
        kind, body = block.group(1), block.group(2)
        tr = re.search(r"xformOp:translate\s*=\s*\(([^)]*)\)", body)
        c = np.array([float(v) for v in tr.group(1).split(",")]) if tr else np.zeros(3)
        sc = re.search(r"xformOp:scale\s*=\s*\(([^)]*)\)", body)
        scale = np.array([float(v) for v in sc.group(1).split(",")]) if sc else np.ones(3)
        if kind == "Cube":
            sz = re.search(r"double size\s*=\s*([\d.eE+-]+)", body); size = np.full(3, float(sz.group(1)) if sz else 2.0) * scale
        else:
            rd = re.search(r"double radius\s*=\s*([\d.eE+-]+)", body); ht = re.search(r"double height\s*=\s*([\d.eE+-]+)", body)
            r_ = float(rd.group(1)) if rd else 1.0; h_ = float(ht.group(1)) if ht else 2.0
            size = np.array([2 * r_ * scale[0], 2 * r_ * scale[1], h_ * scale[2]])
        out.append((c, size))
    return out


def rest_levels(low: np.ndarray, speed: np.ndarray, wmag: np.ndarray, fps: float, min_s: float = 1.0):
    """Support heights the object itself reveals: lowest-point height during rests of >= min_s seconds."""
    still = (speed < 0.02) & (wmag < np.radians(10))
    levels = [0.0]; run = 0
    for t in range(len(still) + 1):
        if t < len(still) and still[t]:
            run += 1; continue
        if run >= int(min_s * fps):
            lv = float(np.median(low[t - run:t]))
            if all(abs(lv - l_) > 0.02 for l_ in levels):
                levels.append(lv)
        run = 0
    return sorted(levels)


def load_parquet(path: Path) -> dict:
    row = {k: v[0] for k, v in pq.read_table(path).to_pydict().items()}
    obp = np.asarray(row["object_body_position"], dtype=np.float64); obq = np.asarray(row["object_body_wxyz"], dtype=np.float64)
    if obp.ndim == 3:
        obp, obq = obp[:, 0], obq[:, 0]
    sides = list(row.get("hand_sides") or ["left", "right"])
    hca = row.get("hand_contact_active") or []
    labels = {s: np.asarray(hca[i], dtype=np.float64).reshape(-1) for i, s in enumerate(sides) if i < len(hca) and len(hca[i])}
    ee = np.asarray(row["ee_pose_w"], dtype=np.float64) if row.get("ee_pose_w") is not None else None
    ee_names = list(row.get("ee_link_names") or [])
    return {"fps": float(row.get("fps") or 30.0), "p": obp, "q": obq, "labels": labels, "sides": sides,
            "ee": ee, "ee_names": ee_names, "mesh": row["object_mesh_paths"][0] if row.get("object_mesh_paths") else None}


def smooth_rotations(q_wxyz: np.ndarray, window: int):
    """Smooth orientation through a cumulative rotation-vector representation; return R objects, omega (body->world), alpha."""
    rots = R.from_quat(q_wxyz, scalar_first=True)
    T = len(rots)
    cum = np.zeros((T, 3))
    for t in range(1, T):
        cum[t] = cum[t - 1] + (rots[t] * rots[t - 1].inv()).as_rotvec()  # small increments add up fine
    cum_s = savgol_filter(cum, window, 3, axis=0)
    omega = savgol_filter(cum, window, 3, deriv=1, axis=0)
    alpha = savgol_filter(cum, window, 3, deriv=2, axis=0)
    rot_s = [rots[0] * R.from_rotvec(cum_s[t] - cum_s[0]) for t in range(T)]
    return rot_s, omega, alpha


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toolkit", type=Path, required=True)
    ap.add_argument("--parquet", type=Path, default=None); ap.add_argument("--support", type=Path, default=None)
    ap.add_argument("--motion-root", type=Path, default=None); ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window-s", type=float, default=0.4, help="Savitzky-Golay window in seconds")
    ap.add_argument("--rest-gap", type=float, default=0.015, help="m: lowest point within this of the support = supported")
    ap.add_argument("--v-thresh", type=float, default=0.05, help="m/s"); ap.add_argument("--w-thresh", type=float, default=0.3, help="rad/s")
    ap.add_argument("--ff-thresh", type=float, default=2.0, help="m/s^2: |a - g| below this while airborne = free fall")
    ap.add_argument("--rot-thresh", type=float, default=30.0, help="deg: net rotation in a held segment above this = torque needed")
    ap.add_argument("--min-segment", type=int, default=10)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = []
    if args.motion_root is not None:
        root = args.motion_root.resolve(); index = {}
        man = root / "retarget_results.json"
        if man.exists():
            data = json.loads(man.read_text())
            for e in (data if isinstance(data, list) else list(data.values())):
                if isinstance(e, dict) and e.get("sequence_id"):
                    index[e["sequence_id"]] = e
        for p in sorted(root.glob("whole_body/soma/sequence_id=*/robot_name=g1/data.parquet")):
            sid = p.parents[1].name.removeprefix("sequence_id="); meta = index.get(sid, {})
            ep = meta.get("episode_index", meta.get("episode"))
            if args.episodes is not None and ep not in args.episodes:
                continue
            name = f"ep{int(ep):02d}_{meta.get('object', '')}_{sid}" if ep is not None else sid
            sup = root / "whole_body" / "reconstructed_stage" / f"{sid}_support.usda"
            jobs.append((name.replace(" ", "_"), p, sup if sup.exists() else None, ep, meta.get("object", "")))
    else:
        m = re.search(r"sequence_id=([^/]+)", str(args.parquet.resolve()))
        jobs.append((m.group(1) if m else args.parquet.stem, args.parquet, args.support, None, ""))

    summary_rows = []
    for name, parquet, support, ep, obj in jobs:
        d = load_parquet(parquet); T = len(d["p"]); fps = d["fps"]
        win = int(round(args.window_s * fps)) | 1; win = max(win, 7)
        # ---- kinematics
        p_s = savgol_filter(d["p"], win, 3, axis=0)
        v = savgol_filter(d["p"], win, 3, deriv=1, axis=0) * fps
        a = savgol_filter(d["p"], win, 3, deriv=2, axis=0) * fps**2
        rot_s, omega, alpha = smooth_rotations(d["q"], win); omega *= fps; alpha *= fps**2
        F = a - G                              # N per kg
        # ---- support / height
        mesh = load_mesh_any(Path(d["mesh"])) if d["mesh"] and Path(d["mesh"]).exists() else None
        verts = mesh.vertices[np.random.default_rng(0).choice(len(mesh.vertices), size=min(3000, len(mesh.vertices)), replace=False)] if mesh is not None else np.zeros((1, 3))
        prims = support_prims(support)
        low = np.zeros(T)
        for t in range(T):
            low[t] = (verts @ rot_s[t].as_matrix().T + p_s[t])[:, 2].min()
        speed = np.linalg.norm(v, axis=1); wmag = np.linalg.norm(omega, axis=1)
        levels = rest_levels(low, speed, wmag, fps)
        print(f"  support prims from usda: {len(prims)}; rest levels from the trajectory (m): {np.round(levels, 3).tolist()}")
        top = np.zeros(T)
        for t in range(T):
            c = p_s[t]; tops = list(levels)
            for (bc, bs) in prims:  # support prims whose footprint contains the object centre
                if abs(c[0] - bc[0]) < bs[0] / 2 + 0.05 and abs(c[1] - bc[1]) < bs[1] / 2 + 0.05:
                    tops.append(bc[2] + bs[2] / 2)
            below = [t_ for t_ in tops if t_ < low[t] + args.rest_gap]  # the highest support not above the object
            top[t] = max(below) if below else 0.0
        height = low - top
        supported = np.abs(height) < args.rest_gap
        moving = (speed > args.v_thresh) | (wmag > args.w_thresh)
        ff = (~supported) & (np.linalg.norm(a - G, axis=1) < args.ff_thresh)
        required = ((~supported) & (~ff)) | (supported & moving)
        # clean
        from fix_hand_reference import clean_segments
        required_c = clean_segments(required, args.min_segment)
        # ---- segments
        idx = np.flatnonzero(required_c); segs = []
        if len(idx):
            cuts = np.flatnonzero(np.diff(idx) > 1)
            segs = [(int(s_[0]), int(s_[-1]) + 1) for s_ in np.split(idx, cuts + 1)]
        # split each segment where the object changes between supported and airborne (push phase vs hold phase)
        sup_c = clean_segments(supported, args.min_segment)
        split = []
        for (s0, s1) in segs:
            start = s0
            for t in range(s0 + 1, s1):
                if sup_c[t] != sup_c[t - 1] and t - start >= args.min_segment and s1 - t >= args.min_segment:
                    split.append((start, t)); start = t
            split.append((start, s1))
        segs = split
        seg_info = []
        palm_idx = {n: i for i, n in enumerate(d["ee_names"])} if d["ee"] is not None else {}
        for (s0, s1) in segs:
            Fm = F[s0:s1]; Fdir = Fm / (np.linalg.norm(Fm, axis=1, keepdims=True) + 1e-9)
            vert_frac = float(np.mean(np.abs(Fdir[:, 2])))
            sup_frac = float(supported[s0:s1].mean())
            net_rot = float(np.degrees((rot_s[s1 - 1] * rot_s[s0].inv()).magnitude()))
            alpha_mag = float(np.linalg.norm(alpha[s0:s1], axis=1).mean())
            torque = net_rot > args.rot_thresh or alpha_mag > 30.0
            if sup_frac > 0.8:
                kind = "push+torque" if torque else "push"
            else:
                kind = "hold+torque" if torque else "hold"
            # where on the surface would a push along median F be applied: object-frame samples whose inward normal faces F
            patch = None; palm_dist = {}
            if mesh is not None:
                pts, fid = trimesh.sample.sample_surface(mesh, 4000); nrm = mesh.face_normals[fid]
                tm = (s0 + s1) // 2; Rm = rot_s[tm].as_matrix()
                Fw = np.median(Fdir, axis=0); Fw /= np.linalg.norm(Fw) + 1e-9
                score = -(nrm @ (Rm.T @ Fw))  # inward normal aligned with F: force enters the object here
                k = np.argsort(score)[-200:]
                patch_obj = np.asarray(pts)[k].mean(axis=0); patch_w = Rm @ patch_obj + p_s[tm]
                patch = {"object_frame": patch_obj.tolist(), "world_mid_frame": patch_w.tolist(), "alignment": float(score[k].mean())}
                for n_, i in palm_idx.items():
                    palm_dist[n_] = float(np.linalg.norm(d["ee"][tm, i, :3] - patch_w) * 100)
            lab = {s: float(d["labels"][s][s0:s1].mean()) if s in d["labels"] else None for s in d["sides"]}
            seg_info.append({"start": s0, "end": s1, "frames": s1 - s0, "kind": kind, "supported_frac": sup_frac,
                             "F_vertical_frac": vert_frac, "mean_F_N_per_kg": float(np.linalg.norm(Fm, axis=1).mean()),
                             "net_rotation_deg": net_rot, "mean_alpha": alpha_mag, "max_height_m": float(height[s0:s1].max()),
                             "label_frac": lab, "push_patch": patch, "palm_to_patch_cm": palm_dist})
        # ---- outputs
        rows = []
        for t in range(T):
            r = {"frame": t, "time_s": t / fps, "height_m": height[t], "supported": int(supported[t]), "moving": int(moving[t]),
                 "free_fall": int(ff[t]), "contact_required": int(required_c[t]), "speed": speed[t], "omega": wmag[t],
                 "Fx": F[t, 0], "Fy": F[t, 1], "Fz": F[t, 2], "alpha": float(np.linalg.norm(alpha[t]))}
            for s in d["sides"]:
                r[f"label_{s}"] = int(d["labels"][s][t]) if s in d["labels"] else -1
            rows.append(r)
        with (args.out / f"{name}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        lab_any = np.zeros(T, dtype=bool)
        for s in d["sides"]:
            if s in d["labels"]:
                lab_any |= d["labels"][s] > 0.5
        agree = float((required_c == lab_any).mean())
        summ = {"name": name, "episode_index": ep, "object": obj, "frames": T, "fps": fps, "segments": seg_info,
                "required_frac": float(required_c.mean()), "label_frac": float(lab_any.mean()),
                "required_but_unlabeled_frames": int((required_c & ~lab_any).sum()), "labeled_but_not_required_frames": int((lab_any & ~required_c).sum()),
                "agreement": agree, "kinds": sorted({s_["kind"] for s_ in seg_info})}
        (args.out / f"{name}.json").write_text(json.dumps(summ, indent=2))
        summary_rows.append(summ)
        print(f"{name}: {len(segs)} segments {[(s_['start'], s_['end'], s_['kind']) for s_ in seg_info]}; required {required_c.mean():.2f} "
              f"label {lab_any.mean():.2f} agree {agree:.2f}; required-but-unlabeled {summ['required_but_unlabeled_frames']}, labeled-but-not-required {summ['labeled_but_not_required_frames']}")
        if not args.no_plot:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            tt = np.arange(T) / fps
            fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True, dpi=120)
            ax[0].plot(tt, height * 100, color="#2a78d6", label="lowest point above support, cm"); ax[0].axhline(args.rest_gap * 100, color="#8a8984", ls=":")
            ax[0].legend(loc="upper right", fontsize=8); ax[0].set_title(f"{name}", loc="left", fontsize=10)
            ax[1].plot(tt, speed * 100, color="#2a78d6", label="speed, cm/s"); ax[1].plot(tt, np.degrees(wmag), color="#e07a1f", label="angular speed, deg/s")
            ax[1].legend(loc="upper right", fontsize=8)
            ax[2].plot(tt, F[:, 0], label="Fx"); ax[2].plot(tt, F[:, 1], label="Fy"); ax[2].plot(tt, F[:, 2], label="Fz (9.8 = holding still)")
            ax[2].axhline(0, color="#52514e", lw=0.6); ax[2].set_ylabel("N per kg"); ax[2].legend(loc="upper right", fontsize=8, ncol=3)
            ax[3].fill_between(tt, 0, 1, where=required_c, color="#c0392b", alpha=0.35, label="contact required (dynamics)")
            ax[3].fill_between(tt, 1, 2, where=lab_any, color="#2a78d6", alpha=0.35, label="contact label (geometry)")
            ax[3].fill_between(tt, 2, 3, where=supported, color="#8a8984", alpha=0.35, label="supported")
            ax[3].fill_between(tt, 3, 4, where=ff, color="#8e44ad", alpha=0.35, label="free fall")
            for s_ in seg_info:
                ax[3].text((s_["start"] + s_["end"]) / 2 / fps, 0.5, s_["kind"], ha="center", va="center", fontsize=8)
            ax[3].set_ylim(0, 4); ax[3].set_yticks([]); ax[3].legend(loc="upper right", fontsize=8, ncol=4); ax[3].set_xlabel("time, s")
            fig.tight_layout(); fig.savefig(args.out / f"{name}.png"); plt.close(fig)
    with (args.out / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["episode_index", "object", "name", "frames", "kinds", "required_frac", "label_frac", "agreement",
                                           "required_but_unlabeled_frames", "labeled_but_not_required_frames"], extrasaction="ignore")
        w.writeheader()
        for s_ in sorted(summary_rows, key=lambda x: (x["episode_index"] is None, x["episode_index"] or 0)):
            w.writerow(s_ | {"kinds": "+".join(s_["kinds"])})
    print(f"wrote {args.out / 'summary.csv'}")


if __name__ == "__main__":
    main()