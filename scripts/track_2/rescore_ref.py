#!/usr/bin/env python3
"""
Re-score evaluator trajectories of several runs against ONE common reference tier (e.g. tier_2_denoise_v7), so runs
trained on different references (raw Tier 2 input, denoise v4, denoise v7) can be compared without the clean data.

Each run is  label=<results dir>@<tier the run was trained on>.  Per episode:
  1. results/episode_NNNNNN.parquet holds achieved_* (the rollout) and reference_* (what the run tracked), sim frame;
  2. the rigid map sim <- tier world is fitted (Kabsch) between reference_* positions and the run tier object
     positions, plus the constant object-frame offset between the two orientation tracks (reported: fit residual);
  3. the common reference tier object track is mapped into that sim frame the same way;
  4. AUC / SP-SR / MP-SR / MPPE are computed as in flash_chord/evaluation/metrics.py (1 world, 1 body).
ADD uses 500 mesh vertices (seed 0) of the dataset mesh, an approximation of the evaluator sampling; the same for all
runs, so comparisons are fair. Sanity column "self": the same metrics against the run's own reference_*.

    python rescore_ref.py --ref <track_2/tier_2_denoise_v7> \
        posA_v7=<.../tier2_denoise_v7_50m_30hz_posA/results>@<track_2/tier_2_denoise_v7> \
        oriB_raw=<.../tier2_50m_30hz_oriB_pub/results>@<track_2/tier_2_synthetic_noise> ... [--csv out.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial.transform import Rotation as R

ADD_T = np.arange(0.01, 0.10, 0.01)
KP = 0.05 * np.concatenate((np.eye(3), -np.eye(3)))


def tier_track(root: Path, ep: int):
    info = json.loads((root / "meta" / "info.json").read_text())
    t = pq.read_table(root / info["data_path"].format(episode_chunk=ep // int(info.get("chunks_size", 1000)), episode_index=ep),
                      columns=["observation.object.pose"])
    a = t.column("observation.object.pose").combine_chunks()
    o = a.flatten().to_numpy(zero_copy_only=False).reshape(len(a), -1).astype(float)
    p, q = o[:, :3].copy(), o[:, 3:7].copy()                              # tier quaternions are wxyz
    ok = (np.linalg.norm(q, axis=1) > 1e-6) & np.isfinite(o[:, :7]).all(1)  # invisible frames are stored as zeros
    if not ok.all():                                                      # fill them like the retarget input does
        from scipy.spatial.transform import Slerp
        idx, t = np.flatnonzero(ok), np.arange(len(o))
        p = np.stack([np.interp(t, idx, p[idx, k]) for k in range(3)], 1)
        rot = Slerp(idx, R.from_quat(q[idx], scalar_first=True))(np.clip(t, idx[0], idx[-1]))
        return p, rot, ok
    return p, R.from_quat(q, scalar_first=True), ok


def results_track(f: Path):
    t = pq.read_table(f)
    g = lambda p: np.stack([t.column(f"{p}_{k}").to_numpy() for k in ("position_x", "position_y", "position_z",
                                                                      "quaternion_x", "quaternion_y", "quaternion_z",
                                                                      "quaternion_w")], 1).astype(float)
    return g("achieved"), g("reference")                                 # xyzw


def resample(p, rot, n):
    """Tier track (len m) -> n samples over the same duration (linear / slerp)."""
    m = len(p)
    if m == n:
        return p, rot
    s = np.linspace(0, m - 1, n)
    i0 = np.floor(s).astype(int); i1 = np.minimum(i0 + 1, m - 1); w = (s - i0)[:, None]
    pp = (1 - w) * p[i0] + w * p[i1]
    q0, q1 = rot[i0].as_quat(), rot[i1].as_quat()
    q1 = np.where((np.sum(q0 * q1, 1) < 0)[:, None], -q1, q1)
    qq = (1 - w) * q0 + w * q1
    return pp, R.from_quat(qq / np.linalg.norm(qq, axis=1, keepdims=True))


def kabsch(a, b):
    ca, cb = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1, 1, d]) @ U.T
    return Rm, cb - Rm @ ca


def metrics(ach, ref, verts):
    pa_, pr = ach[:, :3], ref[:, :3]
    Ra, Rr = R.from_quat(ach[:, 3:]), R.from_quat(ref[:, 3:])
    pos = np.linalg.norm(pa_ - pr, axis=1)
    rot = (Ra * Rr.inv()).magnitude()
    dR = Ra.as_matrix() - Rr.as_matrix()
    def add(v):
        return np.linalg.norm(np.einsum("tij,vj->tvi", dR, v) + (pa_ - pr)[:, None], axis=-1).mean(1)
    a = add(verts)
    auc = float(np.trapezoid([np.mean(a < t) for t in ADD_T], x=np.linspace(0, 1, ADD_T.size)))
    return {"AUC": auc, "SP": float(pos.mean() <= 0.10 and rot.mean() <= 0.5),
            "MP": float(pos.mean() < 0.03 and np.degrees(rot.mean()) < 30.0),
            "MPPE": float(100 * add(KP).mean()), "pos_cm": float(100 * pos.mean()), "rot_deg": float(np.degrees(rot.mean()))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=<results dir>@<tier dir the run was trained on>")
    ap.add_argument("--ref", type=Path, required=True, help="common reference tier")
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()
    info = json.loads((args.ref / "meta" / "info.json").read_text())
    metas = {int(json.loads(l)["episode_index"]): json.loads(l)
             for l in (args.ref / "meta" / "episodes_metadata.jsonl").read_text().splitlines() if l.strip()}
    eps = sorted(metas) if args.episodes is None else args.episodes
    runs = []
    for s in args.runs:
        lab, rest = s.split("=", 1); d, tier = rest.split("@", 1)
        runs.append((lab, Path(d), Path(tier)))
    verts, rows = {}, []
    for ep in eps:
        obj = metas[ep]["object"]
        if obj not in verts:
            m = trimesh.load(str(args.ref / info.get("mesh_path", "mesh/{object_name}/{object_name}.glb").format(object_name=obj)), force="mesh")
            V = np.asarray(m.vertices)
            verts[obj] = V[np.random.default_rng(0).choice(len(V), min(500, len(V)), replace=False)]
        p_new, r_new, _ = tier_track(args.ref, ep)
        for lab, d, tier in runs:
            f = d / f"episode_{ep:06d}.parquet"
            if not f.exists():
                continue
            ach, ref = results_track(f)
            n = len(ref)
            p0, r0, ok0 = tier_track(tier, ep)
            p_run, r_run = resample(p0, r0, n)
            ok = ok0[np.round(np.linspace(0, len(ok0) - 1, n)).astype(int)]   # fit on visible frames only
            Rk, tk = kabsch(p_run[ok], ref[ok, :3])
            fit_mm = 1000 * np.linalg.norm(p_run[ok] @ Rk.T + tk - ref[ok, :3], axis=1).mean()
            Rw = R.from_matrix(Rk)
            rr = R.from_quat(ref[ok, 3:])
            off = ((Rw * r_run[ok]).inv() * rr).mean()                          # constant object-frame offset
            rot_fit = np.degrees(((Rw * r_run[ok] * off).inv() * rr).magnitude().mean())
            pn, rn = resample(p_new, r_new, n)
            ref_new = np.hstack([pn @ Rk.T + tk, (Rw * rn * off).as_quat()])
            m_new = metrics(ach, ref_new, verts[obj]); m_self = metrics(ach, ref, verts[obj])
            rows.append({"ep": ep, "object": obj, "run": lab, "fit_mm": fit_mm, "fit_deg": rot_fit,
                         **{k: v for k, v in m_new.items()}, **{f"self_{k}": v for k, v in m_self.items()}})
    labs = [r[0] for r in runs]
    print(f"common reference: {args.ref.name}    (fit = how well each run reference maps onto its tier; should be ~mm / <1 deg)")
    print(f"{'ep':>3} {'object':<19}" + "".join(f"{l:>22}" for l in labs))
    for ep in eps:
        line = f"{ep:3d} {metas[ep]['object']:<19}"
        for l in labs:
            r = next((r for r in rows if r["ep"] == ep and r["run"] == l), None)
            line += f"{'-':>22}" if r is None else f"   {int(r['SP'])}{int(r['MP'])} {r['MPPE']:6.1f} {r['AUC']:.3f}  "
        print(line)
    print(f"{'':23}" + "".join(f"{'SP MP  MPPE   AUC':>22}" for _ in labs))
    print("\nmeans (over the episodes each run has):  vs common reference  |  vs own reference (self)  |  max fit residual")
    for l in labs:
        rr = [r for r in rows if r["run"] == l]
        if not rr:
            continue
        mean = lambda k: np.mean([r[k] for r in rr])
        print(f"  {l:<14} n={len(rr):2d}  AUC {mean('AUC'):.3f} SP {mean('SP'):.3f} MP {mean('MP'):.3f} MPPE {mean('MPPE'):6.2f}"
              f"  |  AUC {mean('self_AUC'):.3f} SP {mean('self_SP'):.3f} MP {mean('self_MP'):.3f} MPPE {mean('self_MPPE'):6.2f}"
              f"  |  {max(r['fit_mm'] for r in rr):.1f} mm, {max(r['fit_deg'] for r in rr):.2f} deg")
    best = {}
    for r in rows:
        k = (r["SP"], r["MP"], -r["MPPE"], r["AUC"])
        if r["ep"] not in best or k > best[r["ep"]][0]:
            best[r["ep"]] = (k, r)
    if best:
        b = [v[1] for v in best.values()]
        print(f"  {'best-of-all':<14} n={len(b):2d}  AUC {np.mean([r['AUC'] for r in b]):.3f} SP {np.mean([r['SP'] for r in b]):.3f}"
              f" MP {np.mean([r['MP'] for r in b]):.3f} MPPE {np.mean([r['MPPE'] for r in b]):6.2f}   (per episode, ranked SP > MP > MPPE > AUC)")
        print("  picks: " + " ".join(f"{ep}:{best[ep][1]['run']}" for ep in sorted(best)))
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()