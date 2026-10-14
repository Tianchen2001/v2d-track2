#!/usr/bin/env python3
"""
Run soma_to_g1.py over every episode folder produced by convert_tier_to_retarget.py.

    python run_retarget_all.py --toolkit /path/to/video_to_data \
        --inputs runs/track2/tier1/retarget_in --motion-root runs/track2/tier1/motion \
        --support-dir data/.../track_2/tier_1_multiview_caption/support_surface \
        [--episodes 0 3] [--force]

Writes <motion-root>/whole_body/soma/sequence_id=<sid>/robot_name=g1/data.parquet per episode,
copies the dataset support surfaces to <motion-root>/whole_body/reconstructed_stage/ (where
FlashCHORD looks for them), keeps a log per episode under <motion-root>/logs/, and skips
episodes whose parquet already exists.
Run inside the environment that has pinocchio, pink, py-soma-x and the robotic_grounding package.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--toolkit", type=Path, required=True, help="video_to_data checkout")
    ap.add_argument("--inputs", type=Path, required=True, help="folder with sequence_id=* dirs and manifest.json")
    ap.add_argument("--motion-root", type=Path, required=True)
    ap.add_argument("--support-dir", type=Path, default=None,
                    help="dataset support_surface/ dir; *_support.usda files are copied to "
                         "<motion-root>/whole_body/reconstructed_stage/")
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--soma-subdir", default="soma")
    ap.add_argument("--robot-name", default="g1")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra args passed to soma_to_g1.py")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    args.toolkit = args.toolkit.resolve()
    args.inputs = args.inputs.resolve()
    args.motion_root = args.motion_root.resolve()

    script = args.toolkit / "robotic_grounding" / "scripts" / "retarget" / "soma_to_g1.py"
    if not script.exists():
        raise SystemExit(f"missing {script}")
    manifest = json.loads((args.inputs / "manifest.json").read_text())
    if args.episodes is not None:
        manifest = [m for m in manifest if m["episode_index"] in set(args.episodes)]
    logs = args.motion_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    if args.support_dir is not None:
        stage = args.motion_root / "whole_body" / "reconstructed_stage"
        stage.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(args.support_dir.resolve().glob("*_support.usda")):
            dst = stage / f.name
            if not dst.exists() or args.force:
                shutil.copyfile(f, dst)
            n += 1
        print(f"support surfaces: {n} files in {stage}")

    results = []
    for m in manifest:
        sid = m["sequence_id"]
        folder = args.inputs / f"sequence_id={sid}"
        out_pq = (args.motion_root / "whole_body" / args.soma_subdir / f"sequence_id={sid}"
                  / f"robot_name={args.robot_name}" / "data.parquet")
        if out_pq.exists() and not args.force:
            print(f"ep {m['episode_index']:02d} skip (exists) {out_pq}")
            results.append({**m, "status": "skipped", "parquet": str(out_pq)})
            continue
        cmd = [sys.executable, str(script), str(folder), "--save",
               "--motion-root", str(args.motion_root), "--soma-subdir", args.soma_subdir,
               "--robot-name", args.robot_name, "--fps", str(m.get("fps", 30)), *args.extra]
        log = logs / f"episode_{m['episode_index']:06d}.log"
        print(f"ep {m['episode_index']:02d} {m['object']:<22s} -> {log.name}", flush=True)
        t0 = time.time()
        with log.open("w") as fh:
            rc = subprocess.run(cmd, cwd=args.toolkit / "robotic_grounding", stdout=fh, stderr=subprocess.STDOUT).returncode
        dt = time.time() - t0
        ok = rc == 0 and out_pq.exists()
        print(f"   {'ok' if ok else 'FAIL'} rc={rc} {dt:.0f}s", flush=True)
        results.append({**m, "status": "ok" if ok else "failed", "rc": rc, "seconds": dt, "parquet": str(out_pq)})
    (args.motion_root / "retarget_results.json").write_text(json.dumps(results, indent=2))
    n_ok = sum(r["status"] in ("ok", "skipped") for r in results)
    print(f"{n_ok}/{len(results)} episodes have a retargeted parquet")


if __name__ == "__main__":
    main()