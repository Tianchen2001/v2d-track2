#!/usr/bin/env python3
"""
How far is an object, at given frames, from a stable resting pose of its mesh?

trimesh.poses.compute_stable_poses(mesh) gives the poses in which the mesh can rest on a flat floor (with how likely
each is when dropped). Each pose fixes which object-frame direction points down. At a frame we take the world down
direction from the ground plane, express it in the object frame, and report the angle to the closest stable "down":
that angle is the tilt a gravity snap would remove (rotation about the vertical is not affected).

    python stable_check.py --root <tier folder> --episodes 14 15 19 --frames 0 -1      # first and last frame
    python stable_check.py --root <tier folder> --episodes 14 --frames -1 --list-poses
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation as R


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[0, -1], help="frame indices (negative = from the end)")
    ap.add_argument("--max-points", type=int, default=3000, help="vertices used for the convex hull")
    ap.add_argument("--max-faces", type=int, default=400, help="simplify the hull to about this many faces")
    ap.add_argument("--list-poses", action="store_true", help="also list every stable pose and its probability")
    args = ap.parse_args()
    info = json.loads((args.root / "meta" / "info.json").read_text())
    metas = {int(json.loads(l)["episode_index"]): json.loads(l)
             for l in (args.root / "meta" / "episodes_metadata.jsonl").read_text().splitlines() if l.strip()}
    for ep in args.episodes:
        m = metas[ep]
        mesh = trimesh.load(str(args.root / info.get("mesh_path", "mesh/{object_name}/{object_name}.glb").format(object_name=m["object"])), force="mesh")
        # stable poses only depend on the convex hull; a hull of a few hundred faces keeps this fast on dense meshes
        V = np.asarray(mesh.vertices)
        if len(V) > args.max_points:
            V = V[np.random.default_rng(0).choice(len(V), args.max_points, replace=False)]
        hull = trimesh.convex.convex_hull(V)
        if len(hull.faces) > args.max_faces:
            try:
                hull = hull.simplify_quadric_decimation(face_count=args.max_faces).convex_hull
            except Exception:
                pass
        hull.density = 1.0
        com = mesh.center_mass if mesh.is_watertight else hull.center_mass
        Ts, probs = trimesh.poses.compute_stable_poses(hull, center_mass=com, n_samples=1)
        downs = np.array([T[:3, :3].T @ np.array([0.0, 0.0, -1.0]) for T in Ts])
        t = pq.read_table(args.root / info["data_path"].format(episode_chunk=0, episode_index=ep))
        a = t.column("observation.object.pose").combine_chunks()
        o = a.flatten().to_numpy(zero_copy_only=False).reshape(len(a), -1).astype(float)
        gp = args.root / info.get("ground_plane_path", "ground_plane/{sequence_id}.json").format(sequence_id=m["sequence_id"])
        up = np.array([0.0, -1.0, 0.0])
        if gp.exists():
            n = np.asarray(json.loads(gp.read_text())["plane"][:3], float); n /= np.linalg.norm(n)
            up = n if n @ up > 0 else -n
        print(f"\nep {ep} {m['object']}: hull {len(hull.faces)} faces, {len(Ts)} stable poses, top probabilities "
              + ", ".join(f"{p:.2f}" for p in probs[:5]))
        if args.list_poses:
            for i, (d, p) in enumerate(zip(downs, probs)):
                print(f"   pose {i}: prob {p:.3f}   object-frame down = [{d[0]:+.2f} {d[1]:+.2f} {d[2]:+.2f}]")
        for f in args.frames:
            fi = f if f >= 0 else len(o) + f
            u = R.from_quat(o[fi, 3:7], scalar_first=True).inv().apply(-up)        # world down, object frame
            ang = np.degrees(np.arccos(np.clip(downs @ u, -1, 1)))
            i = int(np.argmin(ang))
            print(f"   frame {fi:4d}: nearest stable pose {i} (prob {probs[i]:.2f}) is {ang[i]:5.1f} deg away"
                  f"   [2nd nearest: {np.sort(ang)[1] if len(ang) > 1 else float('nan'):5.1f} deg]")


if __name__ == "__main__":
    main()