#!/usr/bin/env python3
"""
Wrench analysis per contact segment: where must the hand force come from?

Contact segments come from the geometric labels (hand_contact_active, cleaned). For each segment:
  * required wrench from the object motion, per kg: F_req = a - g, tau_req = I alpha (I = 0.01 nominal),
  * subtract what the support can give while the object rests on it (normal force up, Coulomb friction
    mu_s against the sliding direction) -> F_hand,
  * single-contact ("push") feasibility over the object surface: fraction of segment frames in which the
    hand force direction lies inside the friction cone (half-angle atan(mu_h)) of the inward normal at
    that surface point -> a heat map on the mesh; the hottest patch is where a palm push works,
  * two-contact ("pinch") requirement: squeeze force needed along the best antipodal axis so friction
    carries the tangential load, S = max_t |F_perp| / mu_h,
  * torque: direction and magnitude of tau_req, and the lever arm |tau|/|F| it implies,
  * a verdict: push at patch / pinch along axis with squeeze S / plus torque,
and a figure: hand-force direction on the unit sphere (coloured by time), the surface heat map with the
recorded palm positions, and the verdict. All of it is written to <out>/<name>_wrench.json as a prior
for the grasp planner (patch centre + inward normal and pinch axis, in the OBJECT frame).

    python wrench_analysis.py --toolkit video_to_data \
        --motion-root /localscratch/wangti98/track2_tier1_motion --out runs/track2/tier1/wrench --episodes 4 7 3

Needs contact_schedule.py and render_retarget_check.py next to it. Pure CPU.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contact_schedule import G, load_parquet, rest_levels, smooth_rotations, support_prims  # noqa: E402
from fix_hand_reference import clean_segments  # noqa: E402
from render_retarget_check import load_mesh_any  # noqa: E402


def analyse(name, parquet: Path, support, args, out: Path, ep=None, obj=""):
    d = load_parquet(parquet); T = len(d["p"]); fps = d["fps"]
    win = max(int(round(args.window_s * fps)) | 1, 7)
    p_s = savgol_filter(d["p"], win, 3, axis=0)
    v = savgol_filter(d["p"], win, 3, deriv=1, axis=0) * fps
    a = savgol_filter(d["p"], win, 3, deriv=2, axis=0) * fps**2
    rot_s, omega, alpha = smooth_rotations(d["q"], win); omega *= fps; alpha *= fps**2
    mesh = load_mesh_any(Path(d["mesh"]))
    rng = np.random.default_rng(0)
    pts_o, fid = trimesh.sample.sample_surface(mesh, args.samples); pts_o = np.asarray(pts_o); nrm_o = mesh.face_normals[fid]
    centre_o = mesh.centroid
    low = np.array([(mesh.vertices @ rot_s[t].as_matrix().T + p_s[t])[:, 2].min() for t in range(T)])
    speed = np.linalg.norm(v, axis=1); wmag = np.linalg.norm(omega, axis=1)
    levels = rest_levels(low, speed, wmag, fps); prims = support_prims(support)
    top = np.zeros(T)
    for t in range(T):
        tops = list(levels)
        for (bc, bs) in prims:
            if abs(p_s[t, 0] - bc[0]) < bs[0] / 2 + 0.05 and abs(p_s[t, 1] - bc[1]) < bs[1] / 2 + 0.05:
                tops.append(bc[2] + bs[2] / 2)
        below = [x for x in tops if x < low[t] + args.rest_gap]
        top[t] = max(below) if below else 0.0
    supported = np.abs(low - top) < args.rest_gap

    # required and hand wrench, per kg
    F_req = a - G
    tau_req = args.inertia * alpha + np.cross(omega, args.inertia * omega)
    F_hand = F_req.copy()
    for t in range(T):
        if supported[t]:
            N = max(0.0, F_req[t, 2]); F_hand[t, 2] -= N
            vh = v[t].copy(); vh[2] = 0.0
            if np.linalg.norm(vh) > 0.02:
                F_hand[t] += args.mu_support * N * vh / np.linalg.norm(vh)   # the hand must overcome sliding friction
    Fmag = np.linalg.norm(F_hand, axis=1)

    # segments from the geometric labels (either hand)
    lab_any = np.zeros(T, dtype=bool); lab = {}
    for s in d["sides"]:
        if s in d["labels"]:
            lab[s] = clean_segments(d["labels"][s] > 0.5, args.min_segment); lab_any |= lab[s]
    idx = np.flatnonzero(lab_any); segs = []
    if len(idx):
        cuts = np.flatnonzero(np.diff(idx) > 1)
        segs = [(int(x[0]), int(x[-1]) + 1) for x in np.split(idx, cuts + 1)]
    sup_c = clean_segments(supported, args.min_segment); split = []
    for (s0, s1) in segs:  # split where the object leaves / lands on its support
        start = s0
        for t in range(s0 + 1, s1):
            if sup_c[t] != sup_c[t - 1] and t - start >= args.min_segment and s1 - t >= args.min_segment:
                split.append((start, t)); start = t
        split.append((start, s1))
    segs = split
    cos_cone = np.cos(np.arctan(args.mu_hand))
    palm_idx = {n: i for i, n in enumerate(d["ee_names"])} if d["ee"] is not None else {}
    results = []
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for k, (s0, s1) in enumerate(segs):
        fr = np.arange(s0, s1); active = fr[Fmag[fr] > args.f_min]
        hands = [s for s in lab if lab[s][s0:s1].mean() > 0.5]
        sup_frac = float(supported[s0:s1].mean())
        if len(active) < 3:
            results.append({"segment": [s0, s1], "hands": hands, "verdict": "no significant force needed (resting or very slow)", "supported_frac": sup_frac})
            continue
        # hand-force direction in the object frame per active frame
        Fo = np.stack([rot_s[t].as_matrix().T @ (F_hand[t] / Fmag[t]) for t in active])
        # single-contact feasibility per surface point: inward normal (-n) within the cone of F
        cosang = (-nrm_o) @ Fo.T                       # (P, n_active)
        heat = (cosang >= cos_cone).mean(axis=1)       # fraction of frames feasible
        best = int(np.argmax(heat)); hot = heat >= max(0.5, heat.max() - 0.1)
        patch_c = pts_o[hot].mean(axis=0) if hot.any() else pts_o[best]
        patch_n = -nrm_o[hot].mean(axis=0) if hot.any() else -nrm_o[best]; patch_n /= np.linalg.norm(patch_n) + 1e-9
        push_ok = float(heat.max())
        # pinch: antipodal point pairs NEAR THE LABELLED PALM (so handles are found), width <= hand aperture
        Fo_full = np.stack([rot_s[t].as_matrix().T @ F_hand[t] for t in active])
        tm = int(active[len(active) // 2]); Rm = rot_s[tm].as_matrix()
        palm_o = []
        for h in hands:
            i = palm_idx.get(f"{h}_hand_palm_link")
            if i is not None:
                palm_o.append(Rm.T @ (d["ee"][tm, i, :3] - p_s[tm]))
        near_mask = np.zeros(len(pts_o), dtype=bool)
        for po in palm_o:
            near_mask |= np.linalg.norm(pts_o - po, axis=1) < args.pinch_radius
        cand_i = np.flatnonzero(near_mask) if near_mask.any() else np.arange(len(pts_o))
        best_u, best_S, best_pair = None, np.inf, None
        rng_i = rng.choice(cand_i, size=min(300, len(cand_i)), replace=False)
        for i in rng_i:
            n_i = -nrm_o[i]                                  # inward
            dvec = pts_o - pts_o[i]; dist = np.linalg.norm(dvec, axis=1)
            ok = (dist > 0.005) & (dist <= args.max_pinch_width)
            if not ok.any():
                continue
            along = (dvec[ok] @ n_i) / dist[ok]              # the partner must lie along the inward normal
            opp = (nrm_o[ok] @ nrm_o[i]) < -0.7              # with an opposite normal
            good = np.flatnonzero((along > 0.9) & opp)
            if not len(good):
                continue
            j = np.flatnonzero(ok)[good[np.argmin(dist[ok][good])]]
            u = dvec[j] / dist[j]
            para = Fo_full @ u; perp = Fo_full - np.outer(para, u)
            S = float(np.max(np.linalg.norm(perp, axis=1)) / args.mu_hand + np.max(np.abs(para)) / 2)
            if S < best_S:
                best_S, best_u, best_pair = S, u, (pts_o[i], pts_o[j])
        width = float(np.linalg.norm(best_pair[1] - best_pair[0])) if best_pair is not None else None
        pinch_centre = ((best_pair[0] + best_pair[1]) / 2).tolist() if best_pair is not None else None
        # torque
        tau = tau_req[active]; tau_mag = np.linalg.norm(tau, axis=1)
        tau_dir_o = np.stack([rot_s[t].as_matrix().T @ tau_req[t] for t in active]); tau_dir_o = tau_dir_o[tau_mag > 1e-4]
        tau_axis = (tau_dir_o / np.linalg.norm(tau_dir_o, axis=1, keepdims=True)).mean(axis=0) if len(tau_dir_o) else np.zeros(3)
        tau_axis = tau_axis / (np.linalg.norm(tau_axis) + 1e-9)
        net_rot = float(np.degrees((rot_s[s1 - 1] * rot_s[s0].inv()).magnitude()))
        # verdict
        if push_ok >= 0.8:
            verdict = f"push: a single contact on the hot patch carries the force in {push_ok*100:.0f}% of frames"
        elif best_u is not None:
            verdict = f"pinch/wrap: no single patch works ({push_ok*100:.0f}%); squeeze >= {best_S:.1f} N per kg along the best antipodal axis (width {width*100:.0f} cm)"
        else:
            verdict = "no single patch and no antipodal axis found: wrap or two hands"
        if net_rot > 30 or tau_mag.mean() > 0.2:
            verdict += f"; torque needed (net rotation {net_rot:.0f} deg)"
        palm_dist = {}
        patch_w = Rm @ patch_c + p_s[tm]
        for n_, i in palm_idx.items():
            palm_dist[n_] = float(np.linalg.norm(d["ee"][tm, i, :3] - patch_w) * 100)
        near = min((palm_dist.get(f"{h}_hand_palm_link", 1e9) for h in hands), default=1e9)
        if push_ok >= 0.8 and near > args.push_reach * 100:
            verdict = (f"push patch exists ({push_ok*100:.0f}%) but it is {near:.0f} cm from the labelled palm, so not where the human held it; "
                       + (f"pinch/wrap instead: squeeze >= {best_S:.1f} N per kg along the best antipodal axis (width {width*100:.0f} cm)" if best_u is not None
                          else "wrap or two hands"))
            if net_rot > 30 or tau_mag.mean() > 0.2:
                verdict += f"; torque needed (net rotation {net_rot:.0f} deg)"
        res = {"segment": [s0, s1], "hands": hands, "supported_frac": sup_frac, "mean_F_hand_N_per_kg": float(Fmag[active].mean()),
               "push_feasibility": push_ok, "push_patch_object_frame": patch_c.tolist(), "push_inward_normal_object_frame": patch_n.tolist(),
               "pinch_axis_object_frame": best_u.tolist() if best_u is not None else None, "pinch_squeeze_N_per_kg": best_S if best_u is not None else None,
               "pinch_width_cm": width * 100 if width else None, "pinch_centre_object_frame": pinch_centre,
               "torque_axis_object_frame": tau_axis.tolist(), "mean_torque": float(tau_mag.mean()),
               "net_rotation_deg": net_rot, "palm_to_patch_cm": palm_dist, "verdict": verdict}
        results.append(res)
        print(f"  seg {s0}-{s1} hands {hands} sup {sup_frac:.2f} |F| {Fmag[active].mean():.1f} -> {verdict}")
        if args.no_plot:
            continue
        # ---- figure: sphere of force directions (world), surface heat map (object frame), text
        fig = plt.figure(figsize=(15, 5), dpi=110)
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        Fw = np.stack([F_hand[t] / Fmag[t] for t in active])
        u_, v_ = np.mgrid[0:2 * np.pi:30j, 0:np.pi:15j]
        ax1.plot_wireframe(np.cos(u_) * np.sin(v_), np.sin(u_) * np.sin(v_), np.cos(v_), color="#d0d0d0", lw=0.3)
        sc = ax1.scatter(Fw[:, 0], Fw[:, 1], Fw[:, 2], c=(active - s0) / fps, cmap="viridis", s=12)
        ax1.set_title("hand force direction (world), colour = s into segment", fontsize=9); ax1.set_box_aspect((1, 1, 1))
        for axn in ("x", "y", "z"):
            getattr(ax1, f"set_{axn}lim")(-1, 1)
        fig.colorbar(sc, ax=ax1, shrink=0.6)
        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        sc2 = ax2.scatter(pts_o[:, 0], pts_o[:, 1], pts_o[:, 2], c=heat, cmap="hot", s=4, vmin=0, vmax=1)
        for n_, i in palm_idx.items():
            pw = d["ee"][tm, i, :3]; po = Rm.T @ (pw - p_s[tm])
            ax2.scatter([po[0]], [po[1]], [po[2]], c="#2a78d6" if n_.startswith("left") else "#e07a1f", s=60, marker="^", label=n_)
        ax2.scatter([patch_c[0]], [patch_c[1]], [patch_c[2]], c="#00c000", s=80, marker="*", label="hot patch")
        ax2.quiver(patch_c[0], patch_c[1], patch_c[2], *(patch_n * 0.08), color="#00c000")
        if best_u is not None:
            ax2.plot(*(np.stack([best_pair[0], best_pair[1]]).T), color="#8e44ad", lw=3, label="pinch pair")
        lim = mesh.extents.max() / 2 * 1.1
        for axn in ("x", "y", "z"):
            getattr(ax2, f"set_{axn}lim")(centre_o["xyz".index(axn)] - lim, centre_o["xyz".index(axn)] + lim)
        ax2.set_title("single-contact feasibility on the surface (object frame)", fontsize=9); ax2.legend(fontsize=7, loc="upper left")
        fig.colorbar(sc2, ax=ax2, shrink=0.6)
        ax3 = fig.add_subplot(1, 3, 3); ax3.axis("off")
        txt = (f"{name}\nsegment {s0}-{s1} ({(s1-s0)/fps:.1f} s), hands {hands}\nsupported {sup_frac*100:.0f}% of frames\n"
               f"|F_hand| mean {Fmag[active].mean():.1f} N/kg (9.8 = carrying)\npush feasibility {push_ok*100:.0f}%\n"
               + (f"pinch squeeze {best_S:.1f} N/kg, width {width*100:.0f} cm\n" if best_u is not None else "")
               + f"net rotation {net_rot:.0f} deg\npalm to hot patch: " + ", ".join(f"{k.split('_')[0]} {v_:.0f} cm" for k, v_ in palm_dist.items())
               + f"\n\n{verdict}")
        ax3.text(0.0, 1.0, txt, va="top", fontsize=10, family="monospace", wrap=True)
        fig.tight_layout(); fig.savefig(out / f"{name}_seg{k}_{s0}-{s1}.png"); plt.close(fig)
    summ = {"name": name, "episode_index": ep, "object": obj, "frames": T, "fps": fps, "rest_levels_m": levels, "segments": results}
    (out / f"{name}_wrench.json").write_text(json.dumps(summ, indent=2))
    return summ


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toolkit", type=Path, required=True)
    ap.add_argument("--parquet", type=Path, default=None); ap.add_argument("--support", type=Path, default=None)
    ap.add_argument("--motion-root", type=Path, default=None); ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window-s", type=float, default=0.4); ap.add_argument("--rest-gap", type=float, default=0.015)
    ap.add_argument("--min-segment", type=int, default=10)
    ap.add_argument("--mu-hand", type=float, default=1.0, help="friction at the hand (sim uses 1.0)")
    ap.add_argument("--mu-support", type=float, default=0.5, help="friction between object and support")
    ap.add_argument("--inertia", type=float, default=0.01, help="nominal inertia (URDF)")
    ap.add_argument("--f-min", type=float, default=0.5, help="N/kg: frames with less required hand force are ignored")
    ap.add_argument("--samples", type=int, default=4000)
    ap.add_argument("--max-pinch-width", type=float, default=0.10, help="m: antipodal axes wider than the hand aperture are not pinchable")
    ap.add_argument("--pinch-radius", type=float, default=0.08, help="m: pinch pairs are searched within this distance of the labelled palm")
    ap.add_argument("--push-reach", type=float, default=0.15, help="m: a push patch farther than this from the labelled palm is not where the human acted")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
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
            sid = p.parents[1].name.removeprefix("sequence_id="); meta = index.get(sid, {}); ep = meta.get("episode_index")
            if args.episodes is not None and ep not in args.episodes:
                continue
            name = f"ep{int(ep):02d}_{meta.get('object', '')}_{sid}" if ep is not None else sid
            sup = root / "whole_body" / "reconstructed_stage" / f"{sid}_support.usda"
            jobs.append((name, p, sup if sup.exists() else None, ep, meta.get("object", "")))
    else:
        m = re.search(r"sequence_id=([^/]+)", str(args.parquet.resolve()))
        jobs.append((m.group(1) if m else args.parquet.stem, args.parquet, args.support, None, ""))
    for name, p, sup, ep, obj in jobs:
        print(f"[{name}]", flush=True)
        analyse(name, p, sup, args, args.out, ep, obj)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()