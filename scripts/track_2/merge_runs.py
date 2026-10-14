#!/usr/bin/env python3
"""
Merge several run_train_all.py output tags into one checkpoints.json (and, optionally, one
results/ directory) for the submission kit. Later tags override earlier ones per episode.

    python merge_runs.py --out /localscratch/wangti98/flash_chord_outputs/merged_tier1 \
        /localscratch/wangti98/flash_chord_outputs/tier1_50m \
        /localscratch/wangti98/flash_chord_outputs/tier1_350m

Each tag dir is scanned for ep_NNN/DONE + the newest policy_*.safetensors (so two GPUs writing
the same tag, or a tag whose checkpoints.json is stale, still merge correctly). Writes
    <out>/checkpoints.json     episode -> checkpoint (absolute paths)
    <out>/results/             copies of episode_NNNNNN.parquet/.json from the winning tag, if present
    <out>/merge_report.json    which tag each episode came from
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def latest_policy(d: Path) -> Path | None:
    cands = sorted(d.glob("policy_*.safetensors"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    return cands[-1] if cands else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", type=Path, nargs="+", help="run_train_all output dirs, lowest priority first")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=30, help="expected episode count")
    args = ap.parse_args()

    chosen: dict[int, dict] = {}
    for tag in args.tags:
        tag = tag.resolve()
        for ep_dir in sorted(tag.glob("ep_*")):
            if not ep_dir.is_dir():
                continue
            ep = int(ep_dir.name.split("_")[1])
            if not (ep_dir / "DONE").exists():
                continue
            ckpt = latest_policy(ep_dir)
            if ckpt is None:
                continue
            traj = tag / "results" / f"episode_{ep:06d}.parquet"
            chosen[ep] = {"tag": tag.name, "checkpoint": str(ckpt),
                          "trajectory": str(traj) if traj.exists() else None,
                          "metrics": str(traj.with_suffix(".json")) if traj.with_suffix(".json").exists() else None}

    out = args.out.resolve()
    (out / "results").mkdir(parents=True, exist_ok=True)
    for ep, c in chosen.items():
        for key in ("trajectory", "metrics"):
            if c[key]:
                shutil.copy2(c[key], out / "results" / Path(c[key]).name)
    (out / "checkpoints.json").write_text(json.dumps({str(ep): chosen[ep]["checkpoint"] for ep in sorted(chosen)}, indent=2))
    (out / "merge_report.json").write_text(json.dumps({str(ep): chosen[ep] for ep in sorted(chosen)}, indent=2))

    missing = [ep for ep in range(args.episodes) if ep not in chosen]
    by_tag: dict[str, list[int]] = {}
    for ep, c in sorted(chosen.items()):
        by_tag.setdefault(c["tag"], []).append(ep)
    for tag, eps in by_tag.items():
        print(f"{tag:<20s} {len(eps):2d} episodes: {eps}")
    print(f"missing: {missing}" if missing else f"all {args.episodes} episodes present")
    print(f"wrote {out / 'checkpoints.json'} and {out / 'results'}")


if __name__ == "__main__":
    main()
