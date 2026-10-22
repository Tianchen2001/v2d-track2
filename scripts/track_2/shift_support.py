#!/usr/bin/env python3
"""
Translate the support surface of each sequence so it sits under the reconstructed object (Tier 3).

Shape and orientation of the official support are kept; every Cube / Cylinder prim of the sequence gets the same
offset (dx, dy, dz) added to its xformOp:translate. The offset comes from the reference itself (the retargeted
motion, sim frame, z up), no Tier 1 trajectory is read:

  rest frames : only the rest at the start of the clip and the one at the end (object speed, median-filtered, below
                --still-speed for at least --min-rest s, touching the clip edge within --edge s); holds in the hand
                in between are never used. A rest counts only if the object bottom is above --ground-band (not
                on the floor) and within --search-z of some prim top
  dz          : max (--dz-mode) of the start and end medians of (lowest object point - top of the prim it rests on)
  dx, dy      : smallest shift that puts every rest contact point (object points within --contact-band of its
                bottom) inside that top face, shrunk by --margin; 0 if they already are
  episodes whose |offset| < --min-shift are left unchanged.

Works in place on a COPY of the motion root (cp -a first); the original usda is kept next to it as *.usda.orig.
Without --write it only prints the table. Only the usda changes: no retarget needed, re-render to check.

    python shift_support.py --motion-root <copy of track2_tier3_motion_v4.6> [--toolkit video_to_data] [--write]

Runs in the retarget env (trimesh, scipy, pyarrow).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation as R

PRIM_RE = re.compile(r'def (Cube|Cylinder) "([^"]*)"\s*(?:\((?:[^()]|\([^()]*\))*\))?\s*\{(.*?)\n\s*\}', re.S)
TR_RE = re.compile(r"(xformOp:translate\s*=\s*\()([^)]*)(\))")


def parse_prims(txt: str) -> list[dict]:
    """Cube / Cylinder prims as flash_chord/scene/support.py builds them (own translate, size*scale, identity rotation)."""
    prims = []
    for m in PRIM_RE.finditer(txt):
        kind, name, body = m.groups()
        tr = TR_RE.search(body)
        c = np.array([float(v) for v in tr.group(2).split(",")]) if tr else np.zeros(3)
        sc = re.search(r"xformOp:scale\s*=\s*\(([^)]*)\)", body)
        scale = np.array([float(v) for v in sc.group(1).split(",")]) if sc else np.ones(3)
        if kind == "Cube":
            sz = re.search(r"(?:double|float) size\s*=\s*([-\d.eE+]+)", body)
            half = 0.5 * (float(sz.group(1)) if sz else 2.0) * np.abs(scale)
            prims.append({"kind": "box", "name": name, "c": c, "half": half, "top": c[2] + half[2], "has_tr": tr is not None})
        else:
            rd = re.search(r"(?:double|float) radius\s*=\s*([-\d.eE+]+)", body)
            ht = re.search(r"(?:double|float) height\s*=\s*([-\d.eE+]+)", body)
            r_, hh = float(rd.group(1)) if rd else 1.0, 0.5 * (float(ht.group(1)) if ht else 2.0)
            prims.append({"kind": "cyl", "name": name, "c": c, "r": r_, "top": c[2] + hh, "has_tr": tr is not None})
    return prims


def inside(p: dict, xy: np.ndarray, grow: float) -> np.ndarray:
    d = xy - p["c"][:2]
    if p["kind"] == "box":
        return np.all(np.abs(d) <= p["half"][:2] + grow, axis=-1)
    return np.linalg.norm(d, axis=-1) <= p["r"] + grow


def xy_shift(p: dict, pts: np.ndarray, margin: float) -> np.ndarray:
    """Smallest move of the top face so all pts are inside it (shrunk by margin); centres them if they cannot fit."""
    c = p["c"][:2]
    if p["kind"] == "box":
        h = p["half"][:2] - margin
        lo, hi = pts.max(0) - h, pts.min(0) + h            # allowed range of the new centre per axis
        new = np.where(lo <= hi, np.clip(c, lo, hi), 0.5 * (pts.min(0) + pts.max(0)))
        return new - c
    co = 0.5 * (pts.min(0) + pts.max(0))
    ro = float(np.linalg.norm(pts - co, axis=1).max())
    room = p["r"] - margin - ro
    if room <= 0:
        return co - c
    d = c - co
    n = np.linalg.norm(d)
    return np.zeros(2) if n <= room else (co + d * room / n) - c


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, s = [], None
    for i, v in enumerate(np.r_[mask, False]):
        if v and s is None:
            s = i
        elif not v and s is not None:
            out.append((s, i)); s = None
    return out


def support_for(stage: Path, sid: str) -> Path | None:
    for c in (stage / f"{sid}_g1_support.usda", stage / f"{sid}_support.usda"):
        if c.is_file():
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motion-root", type=Path, required=True)
    ap.add_argument("--toolkit", type=Path, default=None, help="video_to_data checkout, for relative object mesh paths")
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--still-speed", type=float, default=0.03, help="m/s")
    ap.add_argument("--min-rest", type=float, default=0.3, help="s")
    ap.add_argument("--edge", type=float, default=0.5, help="s: the start rest must begin, the end rest must end, this close to the clip edge")
    ap.add_argument("--dz-mode", choices=["max", "mean"], default="max",
                    help="start and end rest disagree: max = no penetration at either (default), mean = split the error")
    ap.add_argument("--ground-band", type=float, default=0.05, help="m: object bottom below this is on the floor")
    ap.add_argument("--search-z", type=float, default=0.15, help="m: max |object bottom - prim top| to count as resting on it")
    ap.add_argument("--search-xy", type=float, default=0.20, help="m: top face grown by this when looking for the prim")
    ap.add_argument("--contact-band", type=float, default=0.01, help="m")
    ap.add_argument("--margin", type=float, default=0.02, help="m kept between contact points and the top edge")
    ap.add_argument("--min-shift", type=float, default=0.01, help="m: smaller offsets are not applied")
    ap.add_argument("--no-xy", action="store_true", help="only shift z")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    root = args.motion_root.resolve()
    stage = root / "whole_body" / "reconstructed_stage"
    index = {}
    man = root / "retarget_results.json"
    if man.exists():
        data = json.loads(man.read_text())
        for e in (data if isinstance(data, list) else list(data.values())):
            if isinstance(e, dict) and e.get("sequence_id"):
                index[e["sequence_id"]] = (e.get("episode_index"), e.get("object", ""))
    results = {}
    print(f"{'ep':>3} {'object':<20} {'prim':<14} {'rest f':>6} {'dz cm':>7} {'dz spread':>9} {'dx cm':>6} {'dy cm':>6}  "
          f"{'dz from (cm)':<22} action")
    for pq_path in sorted(root.glob("whole_body/soma/sequence_id=*/robot_name=g1/data.parquet")):
        sid = pq_path.parents[1].name.removeprefix("sequence_id=")
        ep, obj = index.get(sid, (None, ""))
        if args.episodes is not None and ep not in args.episodes:
            continue
        usda = support_for(stage, sid)
        tag = f"{ep if ep is not None else '?':>3} {obj[:20]:<20}"
        if usda is None:
            print(f"{tag} {'-':<14} no support usda"); continue
        orig = usda.with_suffix(".usda.orig")
        txt = (orig if orig.exists() else usda).read_text(errors="ignore")   # always start from the official one
        prims = parse_prims(txt)
        if not prims:
            print(f"{tag} {'-':<14} no Cube/Cylinder prims"); continue

        row = {k: v[0] for k, v in pq.read_table(pq_path).to_pydict().items()}
        P = np.asarray(row["object_body_position"], float); Q = np.asarray(row["object_body_wxyz"], float)
        if P.ndim == 3:
            P, Q = P[:, 0], Q[:, 0]
        fps = float(row.get("fps") or 30.0)
        mp = Path(row["object_mesh_paths"][0])
        if not mp.is_absolute() and args.toolkit is not None:
            mp = args.toolkit.resolve() / "robotic_grounding" / mp
        mesh = trimesh.load(str(mp), force="mesh")
        S = np.asarray(trimesh.sample.sample_surface(mesh, 2000, seed=0)[0])

        Ps = median_filter(P, size=(5, 1), mode="nearest")
        speed = np.r_[0, np.linalg.norm(np.diff(Ps, axis=0), axis=1)] * fps
        T = len(P)
        edge = int(round(args.edge * fps))
        cand = [r for r in runs(speed < args.still_speed) if r[1] - r[0] >= args.min_rest * fps]
        # only the rest at the start (before the first pick) and at the end (after the last place): in between the
        # object can be held still in the hand, which would look like a rest
        ends = {"start": next((r for r in cand if r[0] <= edge), None), "end": next((r for r in cand[::-1] if r[1] >= T - edge), None)}
        if ends["start"] is not None and ends["start"] == ends["end"]:
            ends["end"] = None                              # never moves: one run
        dzs, foot, which = {}, {}, {}
        for side, run in ends.items():
            if run is None:
                continue
            a, b = run
            for t in range(a, b):
                W = S @ R.from_quat(Q[t], scalar_first=True).as_matrix().T + P[t]
                bot = W[:, 2].min()
                if bot < args.ground_band:
                    continue
                cxy = W[W[:, 2] < bot + args.contact_band, :2]
                ok = [i for i, p in enumerate(prims)
                      if inside(p, cxy.mean(0), args.search_xy) and abs(bot - p["top"]) < args.search_z]
                if not ok:
                    continue
                i = min(ok, key=lambda i: abs(bot - prims[i]["top"]))
                dzs.setdefault(i, []).append(bot - prims[i]["top"]); foot.setdefault(i, []).append(cxy)
                which.setdefault(i, []).append(side)
        if not dzs:
            print(f"{tag} {'-':<14} {0:6d}   no start/end rest on the support (floor or in hand): unchanged")
            results[sid] = {"episode_index": ep, "object": obj, "status": "no start/end rest on support"}
            continue
        pi = max(dzs, key=lambda i: len(dzs[i]))           # the prim it rests on most
        p = prims[pi]
        d, w = np.array(dzs[pi]), np.array(which[pi])
        side_dz = {s: float(np.median(d[w == s])) for s in ("start", "end") if (w == s).any()}
        # max: the higher of the two rests, so the object is never inside the top (the other rest floats a bit)
        dz = float(max(side_dz.values()) if args.dz_mode == "max" else np.mean(list(side_dz.values())))
        spread = float(np.percentile(d, 90) - np.percentile(d, 10))
        src = "+".join(f"{s} {v * 100:.1f}" for s, v in side_dz.items())
        dxy = np.zeros(2) if args.no_xy else xy_shift(p, np.concatenate(foot[pi]), args.margin)
        off = np.array([dxy[0], dxy[1], dz])
        apply = bool(np.abs(off).max() >= args.min_shift)
        if apply and not all(q["has_tr"] for q in prims):
            apply = False; act = "prim without xformOp:translate: not handled"
        else:
            act = "shift" if apply else "unchanged (< min shift)"
        print(f"{tag} {p['name'][:14]:<14} {len(dzs[pi]):6d} {dz * 100:7.1f} {spread * 100:9.1f} "
              f"{dxy[0] * 100:6.1f} {dxy[1] * 100:6.1f}  {src:<22} {act}")
        results[sid] = {"episode_index": ep, "object": obj, "prim": p["name"], "rest_frames": len(dzs[pi]),
                        "dz_start_end_m": {k: round(v, 4) for k, v in side_dz.items()},
                        "offset_m": off.round(4).tolist(), "dz_spread_m": round(spread, 4), "applied": apply and args.write}
        if apply and args.write:
            if not orig.exists():
                shutil.copy2(usda, orig)

            def add(m):
                v = np.array([float(x) for x in m.group(2).split(",")]) + off
                return f"{m.group(1)}{v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}{m.group(3)}"

            new = PRIM_RE.sub(lambda m: TR_RE.sub(add, m.group(0), count=1), txt)
            usda.write_text(new.rstrip("\n") + f'\n# shift_support.py offset (m): {off.round(4).tolist()}\n')
    if args.write:
        (root / "support_shift_results.json").write_text(json.dumps(results, indent=1))
        print(f"\nwrote {root / 'support_shift_results.json'}")


if __name__ == "__main__":
    main()