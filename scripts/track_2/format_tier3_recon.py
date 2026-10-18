#!/usr/bin/env python3
"""
Turn a per-episode Tier 3 reconstruction (one folder per episode with episode_NNNNNN.parquet, as delivered:
    <src>/ep00_tall_bar_stool/episode_000000.parquet  (+ a rendered mp4)
) into the same LeRobot tier layout as tier_1_multiview_caption / tier_2_synthetic_noise, so the rest of the
pipeline (convert_tier_to_retarget.py, run_retarget_all.py, render_source_recon.py, run_train_all.py) works
unchanged:

    <out>/data/chunk-000/episode_NNNNNN.parquet       reconstruction (copied as is)
    <out>/meta/info.json, episodes_metadata.jsonl     from Tier 1 (layout, tasks, fps, joint order); variant renamed
    <out>/mesh/<object>/<object>.glb                  object meshes from the dataset (needed by retargeting)
    <out>/videos/chunk-000/observation.images.exo_camera/episode_NNNNNN.mp4   exo videos from the dataset
    <out>/ground_plane/<sequence_id>.json             ESTIMATED from the reconstructed body (feet), --ground-plane
    <out>/recon_renders/ep_NN_<object>.mp4            the renders that came with the reconstruction
    <out>/README_reconstruction.md                    the README that came with the reconstruction

Nothing from the Tier 1 TRAJECTORIES is copied. The Tier 1 ground planes are only read, when --compare-tier1-plane
is given, to report how far the reconstruction frame is from the released world frame (diagnostic only).
support_surface/ is NOT copied: those surfaces are in the Tier 1 world frame and would be misplaced here.

    python format_tier3_recon.py --src <...>/track_2/tier_3_reconstruction_raw \
        --tier1 <...>/track_2/tier_1_multiview_caption --out <...>/track_2/tier_3_reconstruction \
        [--ground-plane estimate|none] [--compare-tier1-plane]

--ground-plane estimate needs the retarget environment (robotic_grounding + py-soma-x); "none" works anywhere and
lets the retargeter fall back to the frame-0 body.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

F_CV_TO_SOMA = np.diag([1.0, -1.0, -1.0])


def find_episodes(src: Path) -> dict[int, dict]:
    eps = {}
    for p in sorted(src.rglob("episode_*.parquet")):
        ep = int(re.findall(r"episode_(\d+)", p.name)[0])
        mp4 = sorted(p.parent.glob("*.mp4"))
        eps[ep] = {"parquet": p, "render": mp4[0] if mp4 else None}
    return eps


def estimate_plane(soma, parquet: Path, tmp: Path, stride: int = 3) -> tuple[np.ndarray, dict]:
    """Ground plane [a,b,c,d] in the OpenCV frame of the parquet, from the lowest body points over the clip."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from convert_tier_to_retarget import COL_IDENTITY, COL_POSE, COL_SCALE, COL_TRANSL, NJ, SOMA_JOINTS_ORDER, col_to_np
    t = pq.read_table(parquet)
    pose = col_to_np(t, COL_POSE).astype(np.float32)[::stride]
    T = len(pose)
    npz = tmp / "p.npz"
    np.savez(npz, poses=pose.reshape(T, NJ, 3), transl=col_to_np(t, COL_TRANSL)[::stride].astype(np.float32),
             identity_coeffs=col_to_np(t, COL_IDENTITY)[:1].astype(np.float32),
             scale_params=col_to_np(t, COL_SCALE)[:1].astype(np.float32), joint_names=np.array(SOMA_JOINTS_ORDER),
             identity_model_type=np.array("mhr"), unit=np.array("meters"), keep_root=np.array(True),
             absolute_pose=np.array(False))
    m = soma.load_motion(params_path=str(npz), normalize=False)
    J, V = np.asarray(m["joints"]), np.asarray(m["vertices"])
    feet = 0.5 * (J[:, SOMA_JOINTS_ORDER.index("LeftFoot")] + J[:, SOMA_JOINTS_ORDER.index("RightFoot")])
    up = (J[:, SOMA_JOINTS_ORDER.index("Head")] - feet).mean(0); up /= np.linalg.norm(up)
    # refine the normal: fit a plane to the per-frame lowest points (feet on the floor), keep it close to "up"
    low = V[np.arange(T), np.argmin(V @ up, axis=1)]
    h = low @ up
    keep = h <= np.percentile(h, 30)                           # frames where a foot is planted
    P = low[keep]
    n = up
    if len(P) >= 10 and np.ptp(P, axis=0).max() > 0.3:         # enough spread to fit a plane
        _, _, vt = np.linalg.svd(P - P.mean(0))
        cand = vt[2] if vt[2] @ up > 0 else -vt[2]
        if np.degrees(np.arccos(np.clip(cand @ up, -1, 1))) <= 15:   # else degenerate fit: keep body up
            n = cand
    d = -float(np.percentile(low @ n, 2))
    n_cv = F_CV_TO_SOMA @ n                                    # SOMA world -> OpenCV world (same d)
    info = {"tilt_from_body_up_deg": float(np.degrees(np.arccos(np.clip(n @ up, -1, 1)))), "frames_used": int(keep.sum())}
    return np.r_[n_cv, d], info


def plane_diff(p: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    a, b = p[:3] / np.linalg.norm(p[:3]), q[:3] / np.linalg.norm(q[:3])
    da, db = p[3] / np.linalg.norm(p[:3]), q[3] / np.linalg.norm(q[:3])
    if a @ b < 0:
        b, db = -b, -db
    return float(np.degrees(np.arccos(np.clip(a @ b, -1, 1)))), float(abs(da - db))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--tier1", type=Path, required=True, help="tier_1_multiview_caption (layout, meta, meshes, videos)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ground-plane", choices=["estimate", "none"], default="estimate")
    ap.add_argument("--compare-tier1-plane", action="store_true", help="diagnostic: distance to the released frame")
    ap.add_argument("--soma-data-root", default=None)
    args = ap.parse_args()
    if args.out.resolve() == args.src.resolve():
        sys.exit("--out must differ from --src")

    info = json.loads((args.tier1 / "meta" / "info.json").read_text())
    metas = [json.loads(l) for l in (args.tier1 / "meta" / "episodes_metadata.jsonl").read_text().splitlines() if l.strip()]
    by_ep = {int(m["episode_index"]): m for m in metas}
    eps = find_episodes(args.src)
    missing = sorted(set(by_ep) - set(eps))
    print(f"{len(eps)} reconstructed episodes; missing vs Tier 1: {missing or 'none'}")

    out = args.out
    (out / "meta").mkdir(parents=True, exist_ok=True)
    chunks = int(info.get("chunks_size", 1000))
    vkey = next(k for k, v in info["features"].items() if v.get("dtype") == "video")
    soma = None
    if args.ground_plane == "estimate":
        from robotic_grounding.retarget.read_soma import SOMA
        soma = SOMA(data_root=args.soma_data_root, identity_model_type="mhr", device=None)
    variant = None
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for ep, e in sorted(eps.items()):
            m = by_ep[ep]
            sid, obj = m["sequence_id"], m["object"]
            t1_pq = args.tier1 / info["data_path"].format(episode_chunk=ep // chunks, episode_index=ep)
            n1, n3 = pq.ParquetFile(t1_pq).metadata.num_rows, pq.ParquetFile(e["parquet"]).metadata.num_rows
            meta3 = {k.decode(): v.decode() for k, v in (pq.read_schema(e["parquet"]).metadata or {}).items()}
            variant = variant or meta3.get("trajectory_variant")
            if meta3.get("source_sequence_id", sid) != sid:
                sys.exit(f"ep {ep}: sequence id {meta3.get('source_sequence_id')} != Tier 1 {sid}")
            # data
            dst = out / info["data_path"].format(episode_chunk=ep // chunks, episode_index=ep)
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(e["parquet"], dst)
            # mesh + exo video from the dataset
            mesh = info.get("mesh_path", "mesh/{object_name}/{object_name}.glb").format(object_name=obj)
            if not (out / mesh).exists():
                (out / mesh).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(args.tier1 / mesh, out / mesh)
            vid = info["video_path"].format(episode_chunk=ep // chunks, video_key=vkey, episode_index=ep)
            if (args.tier1 / vid).exists() and not (out / vid).exists():
                (out / vid).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(args.tier1 / vid, out / vid)
            if e["render"] is not None:
                (out / "recon_renders").mkdir(exist_ok=True)
                shutil.copy2(e["render"], out / "recon_renders" / f"ep_{ep:02d}_{obj}.mp4")
            row = {"ep": ep, "object": obj, "rows": n3, "tier1_rows": n1}
            # ground plane
            if soma is not None:
                plane, pinfo = estimate_plane(soma, e["parquet"], Path(tmp))
                gp = out / info.get("ground_plane_path", "ground_plane/{sequence_id}.json").format(sequence_id=sid)
                gp.parent.mkdir(parents=True, exist_ok=True)
                gp.write_text(json.dumps({"plane": [round(float(x), 6) for x in plane],
                                          "source": "estimated from reconstructed body (lowest points)", **pinfo}))
                row["plane"] = np.round(plane, 3).tolist()
                if args.compare_tier1_plane:
                    g1 = args.tier1 / info.get("ground_plane_path", "ground_plane/{sequence_id}.json").format(sequence_id=sid)
                    if g1.exists():
                        ang, off = plane_diff(plane, np.asarray(json.loads(g1.read_text())["plane"], float))
                        row["vs_tier1_plane"] = f"{ang:.1f} deg, {off * 100:.1f} cm"
            rows.append(row)
            flag = "" if n3 == n1 else f"   <-- ROWS DIFFER (Tier 1 has {n1})"
            print(f"ep {ep:02d} {obj:<20s} rows={n3}{flag}  {row.get('plane', '')}  {row.get('vs_tier1_plane', '')}", flush=True)

    info = dict(info)
    info["trajectory_variant"] = variant or "tier_3_reconstruction"
    info["total_episodes"] = len(eps)
    info["reconstruction_note"] = "Per-episode reconstruction reformatted by format_tier3_recon.py; ground planes estimated from the reconstructed body; no support surfaces."
    info.pop("support_surface_path", None)
    (out / "meta" / "info.json").write_text(json.dumps(info))
    (out / "meta" / "episodes_metadata.jsonl").write_text("".join(json.dumps(by_ep[ep]) + "\n" for ep in sorted(eps)))
    readme = next(iter(sorted(args.src.glob("README*"))), None)
    if readme:
        shutil.copy2(readme, out / "README_reconstruction.md")
    (out / "format_report.json").write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}  (report: {out / 'format_report.json'})")


if __name__ == "__main__":
    main()
