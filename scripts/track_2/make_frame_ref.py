#!/usr/bin/env python3
"""
Build a frame-reference tier folder for the packer when a run tracked a modified object reference.

The packer checks that the reference stored in each rollout (reference_*, sim frame) and the released tier
input differ by one rigid transform. A run trained and evaluated on an edited object reference (e.g. hold_rest.py)
fails that check. This writes a tier folder that is the released tier (symlinks) except for the given
episodes, whose object pose is the rollout reference mapped back into the released frame:

  1. fit the rigid map sim <- tier (Kabsch on positions) and the constant object-frame rotation offset on
     frames where the edit did not change the reference (--fit-from .. end);
  2. map every rollout reference frame back: p_tier = R^T (p_sim - t), q_tier = R^T q_sim off^-1.

Scoring is unaffected: the packer still scores against the clean reference; this folder is only passed as
--frame-references for the frame conversion, as for the denoised Tier 2 inputs.

    python make_frame_ref.py --tier-root <track_2/tier_1_multiview_caption> \
        --out-root <track_2/tier_1_frameref_v15> \
        --episode 25=<mix_tier1_v15/results/episode_000025.parquet> --fit-from 130
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

COL = "observation.object.pose"


def rollout_reference(f: Path):
    t = pq.read_table(f)
    g = lambda k: t.column(f"reference_{k}").to_numpy().astype(float)
    p = np.stack([g("position_x"), g("position_y"), g("position_z")], 1)
    q = np.stack([g("quaternion_x"), g("quaternion_y"), g("quaternion_z"), g("quaternion_w")], 1)
    return p, R.from_quat(q)                                                      # xyzw


def kabsch(a, b):
    """R, t with b ~ R a + t."""
    ca, cb = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1, 1, d]) @ U.T
    return Rm, cb - Rm @ ca


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--episode", action="append", required=True, metavar="EP=rollout.parquet")
    ap.add_argument("--fit-from", type=int, default=130, help="first frame used for the fit (after the edited part)")
    args = ap.parse_args()
    src, dst = args.tier_root.resolve(), args.out_root.resolve()
    if dst.exists():
        raise SystemExit(f"{dst} exists; remove it first")
    jobs = {}
    for s in args.episode:
        e, _, f = s.partition("=")
        jobs[f"episode_{int(e):06d}.parquet"] = Path(f)
    dst.mkdir(parents=True)
    for item in src.iterdir():                                   # everything but data/ is a link to the tier
        if item.name != "data":
            os.symlink(item, dst / item.name)
    written = set()
    for d, _, files in os.walk(src / "data"):
        rel = Path(d).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for fn in files:
            s_f, d_f = Path(d) / fn, dst / rel / fn
            if fn not in jobs:
                os.symlink(s_f, d_f)
                continue
            t = pq.read_table(s_f)
            o = np.asarray(t.column(COL).to_pylist(), float)
            if o.ndim != 2 or o.shape[1] != 7:
                raise SystemExit(f"{fn}: {COL} has shape {o.shape}, expected (T, 7)")
            pt, qt = o[:, :3], R.from_quat(o[:, 3:7], scalar_first=True)       # tier quaternions are wxyz
            ps, qs = rollout_reference(jobs[fn])
            if len(ps) != len(pt):
                raise SystemExit(f"{fn}: rollout reference has {len(ps)} frames, tier has {len(pt)}")
            k = slice(args.fit_from, len(pt))
            Rm, tv = kabsch(pt[k], ps[k])
            Rr = R.from_matrix(Rm)
            off = (Rr * qt[k]).inv() * qs[k]                             # q_sim = Rr q_tier off
            off_mean = off.mean()
            res_p = np.linalg.norm(pt[k] @ Rm.T + tv - ps[k], axis=1).max()
            res_r = ((Rr * qt[k] * off_mean).inv() * qs[k]).magnitude().max()
            p_new = (ps - tv) @ Rm                                       # R^T (p - t)
            q_new = Rr.inv() * qs * off_mean.inv()
            ch_p = 100 * np.linalg.norm(p_new - pt, axis=1).max()
            ch_r = np.degrees((q_new * qt.inv()).magnitude().max())
            o2 = np.hstack([p_new, q_new.as_quat(scalar_first=True)])
            col = pa.array(o2.tolist(), type=t.schema.field(COL).type)
            t2 = t.set_column(t.schema.get_field_index(COL), COL, col)
            pq.write_table(t2, d_f)
            written.add(fn)
            print(f"{fn}: fit on frames {args.fit_from}-{len(pt) - 1}: residual {res_p * 1000:.3f} mm, "
                  f"{np.degrees(res_r):.3f} deg; object pose changed by up to {ch_p:.2f} cm / {ch_r:.2f} deg")
    missing = set(jobs) - written
    if missing:
        raise SystemExit(f"not found under {src / 'data'}: {sorted(missing)}")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
