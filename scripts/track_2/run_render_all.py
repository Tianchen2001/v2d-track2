#!/usr/bin/env python3
"""
Run render_retarget_check.py over every retargeted episode and aggregate the summaries.

    python run_render_all.py \
        --toolkit video_to_data \
        --motion-root runs/track2/tier1/motion \
        --out runs/track2/tier1/plausibility \
        --workers 6 [--stride 2] [--no-video] [--episodes 4 25 1]

Finds <motion-root>/whole_body/soma/sequence_id=*/robot_name=g1/data.parquet, maps sid -> episode/object
through <motion-root>/retarget_results.json (or --manifest), uses
<motion-root>/whole_body/reconstructed_stage/<sid>_support.usda when present, runs the checks in parallel
skips episodes whose .json already exists (--force to redo), writes videos to <out>/videos/, then writes\n<out>/summary.csv and prints a table.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECK = HERE / "render_retarget_check.py"


def load_index(manifest: Path | None) -> dict[str, dict]:
    """sid -> {"episode": int|None, "object": str} from retarget_results.json / manifest.json (tolerant keys)."""
    if manifest is None or not manifest.exists():
        return {}
    data = json.loads(manifest.read_text())
    entries = data if isinstance(data, list) else data.get("episodes") or data.get("results") or list(data.values())
    out = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        sid = e.get("sequence_id") or e.get("sid") or e.get("sequence")
        if not sid:
            src = e.get("input") or e.get("folder") or e.get("input_dir") or ""
            if "sequence_id=" in str(src):
                sid = str(src).split("sequence_id=")[1].split("/")[0]
        if not sid:
            continue
        ep = e.get("episode_index", e.get("episode", e.get("ep")))
        out[str(sid)] = {"episode": int(ep) if ep is not None else None,
                         "object": e.get("object") or e.get("object_name") or e.get("object_id") or ""}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toolkit", type=Path, required=True)
    ap.add_argument("--motion-root", type=Path, default=None, help="retarget output dir containing whole_body/soma/ and retarget_results.json")
    ap.add_argument("--parquet-glob", default=None, help="instead of --motion-root: glob of parquet files, e.g. 'rollouts/tier1_350m/ep_*.parquet' (names from the file stem)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None, help="retarget_results.json or manifest.json (default: <motion-root>/retarget_results.json)")
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="only these episode indices")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="everything after --extra is passed to render_retarget_check.py (e.g. --extra --cam -2 2 1.4 --cam-top 3)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--force", action="store_true", help="rerun episodes that already have a .json")
    args = ap.parse_args()

    import glob as globmod
    if args.parquet_glob:
        parquets = [Path(p) for p in sorted(globmod.glob(args.parquet_glob))]
        if not parquets:
            raise SystemExit(f"no parquets match {args.parquet_glob}")
        index = {}
    else:
        if args.motion_root is None:
            raise SystemExit("pass --motion-root or --parquet-glob")
        manifest = args.manifest or (args.motion_root / "retarget_results.json")
        if not manifest.exists():
            alt = args.motion_root / "manifest.json"
            manifest = alt if alt.exists() else manifest
        index = load_index(manifest)
        if not index:
            print(f"warning: no sid->episode index found at {manifest}; outputs will be named by sid only")
        parquets = sorted(args.motion_root.glob("whole_body/soma/sequence_id=*/robot_name=g1/data.parquet"))
        if not parquets:
            raise SystemExit(f"no parquets under {args.motion_root}/whole_body/soma")
    jobs = []
    for pq in parquets:
        if args.parquet_glob:
            sid = pq.stem
            m = re.search(r"ep_?(\d+)", sid)
            meta = {"episode": int(m.group(1)) if m else None, "object": ""}
            name = sid
        else:
            sid = pq.parents[1].name.removeprefix("sequence_id=")
            meta = index.get(sid, {"episode": None, "object": ""})
            name = f"ep{meta['episode']:02d}_{meta['object']}_{sid}" if meta["episode"] is not None else sid
        if args.episodes is not None and meta["episode"] not in args.episodes:
            continue
        name = name.replace(" ", "_").replace("/", "_")
        if not args.force and (args.out / f"{name}.json").exists():
            print(f"skip {name} (done)")
            continue
        stage = (args.motion_root / "whole_body" / "reconstructed_stage") if args.motion_root else Path("/nonexistent")
        support = next((c for c in (stage / f"{sid}_g1_support.usda", stage / f"{sid}_support.usda") if c.exists()), stage / f"{sid}_support.usda")
        cmd = [sys.executable, str(CHECK), "--toolkit", str(args.toolkit), "--parquet", str(pq), "--out", str(args.out),
               "--sid", name, "--stride", str(args.stride), *args.extra]
        if support.exists() and support.stat().st_size > 0:
            cmd += ["--support", str(support)]
        if args.no_video:
            cmd.append("--no-video")
        jobs.append((name, cmd))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "logs").mkdir(exist_ok=True)
    print(f"{len(jobs)} episodes to run with {args.workers} workers")

    def run(job):
        name, cmd = job
        log = args.out / "logs" / f"{name}.log"
        with log.open("w") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
        return name, rc, log

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(run, j) for j in jobs]):
            name, rc, log = fut.result()
            print(f"{'ok  ' if rc == 0 else 'FAIL'} {name}" + ("" if rc == 0 else f"  (see {log})"))
            if rc != 0:
                failed.append(name)

    # ---- aggregate every summary json in --out
    rows = []
    for js in sorted(args.out.glob("*.json")):
        s = json.loads(js.read_text())
        if "sequence_id" not in s:
            continue
        sid_full = s["sequence_id"]
        ep = int(sid_full[2:4]) if sid_full.startswith("ep") and sid_full[2:4].isdigit() else None
        obj = sid_full.split("_")[1] if ep is not None and sid_full.count("_") >= 2 else ""
        r = {"ep": ep, "object": obj, "name": sid_full, "frames": s["frames"], "watertight": s["object_watertight"]}
        for side in ("left", "right"):
            r[f"{side}_label_frames"] = s.get(f"{side}_label_active_frames")
            r[f"{side}_pen_max_mm"] = s.get(f"{side}_pen_max_mm")
            r[f"{side}_pen_frames"] = s.get(f"{side}_pen_frames_gt5mm")
            r[f"{side}_gap_med_labeled_mm"] = s.get(f"{side}_gap_median_when_labeled_mm")
            r[f"{side}_labeled_far_frames"] = s.get(f"{side}_labeled_but_gap_gt2cm_frames")
        r["body_pen_max_mm"] = s.get("body_pen_max_mm")
        r["obj_below_ground_frames"] = s.get("object_below_ground_frames")
        r["obj_floating_frames"] = s.get("object_above_3cm_with_no_contact_label_frames")
        r["obj_through_support_frames"] = s.get("object_through_support_frames")
        r["pos_err_mean_cm"] = s.get("object_pos_err_mean_cm"); r["pos_err_final_cm"] = s.get("object_pos_err_final_cm")
        r["rot_err_mean_deg"] = s.get("object_rot_err_mean_deg"); r["first_frame_err_gt10cm"] = s.get("first_frame_err_gt10cm")
        rows.append(r)
    rows.sort(key=lambda r: (r["ep"] is None, r["ep"] if r["ep"] is not None else 0, r["name"]))
    if rows:
        with (args.out / "summary.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

        def f(v, scale=1.0, fmt="{:.0f}"):
            return "-" if v is None else fmt.format(v * scale)
        print(f"\n{'ep':>3} {'object':<16} {'frames':>6} | {'L lab':>5} {'L pen':>6} {'L gap@lab':>9} {'L far':>5} | "
              f"{'R lab':>5} {'R pen':>6} {'R gap@lab':>9} {'R far':>5} | {'body':>5} {'ground':>6} {'float':>5} {'supp':>5}")
        print("   (pen = max penetration mm; gap@lab = median gap mm while label on; far = labeled frames with gap > 2 cm)")
        for r in rows:
            print(f"{f(r['ep']):>3} {r['object'][:16]:<16} {r['frames']:>6} | "
                  f"{f(r['left_label_frames']):>5} {f(r['left_pen_max_mm']):>6} {f(r['left_gap_med_labeled_mm']):>9} {f(r['left_labeled_far_frames']):>5} | "
                  f"{f(r['right_label_frames']):>5} {f(r['right_pen_max_mm']):>6} {f(r['right_gap_med_labeled_mm']):>9} {f(r['right_labeled_far_frames']):>5} | "
                  f"{f(r['body_pen_max_mm']):>5} {f(r['obj_below_ground_frames']):>6} {f(r['obj_floating_frames']):>5} {f(r['obj_through_support_frames']):>5}")
        if any(r["pos_err_mean_cm"] is not None for r in rows):
            print(f"\n{'ep':>3} {'name':<40} {'pos err mean':>12} {'final':>6} {'rot mean':>8} {'lost@':>6}")
            for r in rows:
                if r["pos_err_mean_cm"] is None:
                    continue
                print(f"{f(r['ep']):>3} {r['name'][:40]:<40} {r['pos_err_mean_cm']:>12.1f} {r['pos_err_final_cm']:>6.1f} {r['rot_err_mean_deg']:>8.0f} {f(r['first_frame_err_gt10cm']):>6}")
        print(f"\nwrote {args.out / 'summary.csv'}")
    if failed:
        print(f"failed: {failed}")


if __name__ == "__main__":
    main()