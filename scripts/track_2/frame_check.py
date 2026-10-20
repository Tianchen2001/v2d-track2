#!/usr/bin/env python3
"""
How well does a reconstruction tier sit in the world frame of a reference tier (same episodes, same frames)?

Per episode, frame by frame, without the body model:
  root   distance between the human root translations (observation.human.translation), cm
  obj    distance between the object positions, cm
  objrot angle between the object orientations, deg
reported at frame 0, as mean / median / 90th percentile over the clip, and at the last frame. A global rigid offset
(Kabsch of the root + object position tracks, recon -> reference) is also reported: if the per-frame errors are large
but the rigid offset explains most of them, the frames differ by a fixed transform; if not, the reconstruction drifts.

Use the Tier 2 input (released ground truth + noise, same world frame) as the reference so that no Tier 1 trajectory
is read:
    python frame_check.py --recon <tier_3_reconstruction_v4> --ref <tier_2_synthetic_noise>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R


def load(root: Path, ep: int):
    info = json.loads((root / "meta" / "info.json").read_text())
    t = pq.read_table(root / info["data_path"].format(episode_chunk=ep // int(info.get("chunks_size", 1000)), episode_index=ep))
    col = lambda c: np.stack(t.column(c).to_numpy(zero_copy_only=False)).astype(float)
    return col("observation.human.translation"), col("observation.object.pose")


def kabsch(a: np.ndarray, b: np.ndarray):
    ca, cb = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1, 1, d]) @ U.T
    return Rm, cb - Rm @ ca


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon", type=Path, required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    args = ap.parse_args()
    metas = [json.loads(l) for l in (args.recon / "meta" / "episodes_metadata.jsonl").read_text().splitlines() if l.strip()]
    objs = {int(m["episode_index"]): m.get("object", "") for m in metas}
    eps = sorted(objs) if args.episodes is None else args.episodes
    print(f"{'ep':>3} {'object':<19}{'frames':>7} | {'root cm: f0  mean  p90  end':>28} | {'obj cm: f0  mean  p90  end':>27} |"
          f" {'obj deg: f0 mean  end':>21} | rigid offset deg / cm, root resid mean cm")
    allr, allo = [], []
    for ep in eps:
        tr_a, ob_a = load(args.recon, ep)
        tr_b, ob_b = load(args.ref, ep)
        T = min(len(tr_a), len(tr_b))
        if len(tr_a) != len(tr_b):
            print(f"    ep {ep}: frame counts differ ({len(tr_a)} vs {len(tr_b)}), using the first {T}")
        tr_a, tr_b, ob_a, ob_b = tr_a[:T], tr_b[:T], ob_a[:T], ob_b[:T]
        dr = np.linalg.norm(tr_a - tr_b, axis=1) * 100
        do = np.linalg.norm(ob_a[:, :3] - ob_b[:, :3], axis=1) * 100
        qa, qb = ob_a[:, 3:7].copy(), ob_b[:, 3:7].copy()
        ok = (np.linalg.norm(qa, axis=1) > 1e-6) & (np.linalg.norm(qb, axis=1) > 1e-6)   # skip empty (invisible) rows
        qa[~ok], qb[~ok] = [1, 0, 0, 0], [1, 0, 0, 0]
        da = np.degrees((R.from_quat(qa, scalar_first=True) * R.from_quat(qb, scalar_first=True).inv()).magnitude())
        Rm, tv = kabsch(np.vstack([tr_a, ob_a[:, :3]]), np.vstack([tr_b, ob_b[:, :3]]))
        res = np.linalg.norm(tr_a @ Rm.T + tv - tr_b, axis=1) * 100
        ang = np.degrees(R.from_matrix(Rm).magnitude())
        allr.append(dr.mean()); allo.append(do.mean())
        print(f"{ep:3d} {objs.get(ep, ''):<19}{T:7d} | {dr[0]:6.1f} {dr.mean():5.1f} {np.percentile(dr, 90):5.1f} {dr[-1]:5.1f}"
              f"    | {do[0]:6.1f} {do.mean():5.1f} {np.percentile(do, 90):5.1f} {do[-1]:5.1f}   |"
              f" {da[0]:5.1f} {da.mean():5.1f} {da[-1]:5.1f}      | {ang:5.1f} / {np.linalg.norm(tv) * 100:5.1f}, {res.mean():5.1f}")
    print(f"\nmean over episodes: root {np.mean(allr):.1f} cm, object {np.mean(allo):.1f} cm (median {np.median(allr):.1f} / {np.median(allo):.1f})")


if __name__ == "__main__":
    main()
