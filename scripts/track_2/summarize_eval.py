#!/usr/bin/env python3
"""
Summarize evaluator metrics from one results directory (episode_NNNNNN.json files written by
evaluate_policy.py), whether it came from run_train_all.py (outputs/<tag>/results) or from the
kit runner (examples/run_policy_evaluation.py --results ...).

    python summarize_eval.py /localscratch/wangti98/eval_merged_tier1
    python summarize_eval.py /localscratch/wangti98/flash_chord_outputs/tier1_50m/results \
        --manifest /localscratch/wangti98/track2_tier1_motion/retarget_results.json --csv out.csv

Prints a per-episode table and the means; --csv also writes the table.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path

KEYS = ["add_auc", "chord_sr", "maniptrans_sr", "mppe_cm"]
HEAD = ["AUC", "SR", "MP-SR", "MPPE"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--manifest", type=Path, default=None, help="retarget_results.json or manifest.json for object names")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    names = {}
    if args.manifest:
        for m in json.loads(args.manifest.read_text()):
            names[int(m["episode_index"])] = m.get("object", "")

    rows = []
    for f in sorted(glob.glob(str(args.results / "episode_*.json"))):
        ep = int(Path(f).stem.split("_")[-1])
        d = json.loads(Path(f).read_text())
        m = d.get("metrics", d)
        worlds = d.get("world_count") or m.get("world_count")
        rows.append({"ep": ep, "object": names.get(ep, ""), "worlds": worlds,
                     **{k: m.get(k) for k in KEYS}})
    if not rows:
        raise SystemExit(f"no episode_*.json in {args.results}")

    print(f"{'ep':>3} {'object':<22} {'AUC':>6} {'SR':>5} {'MP-SR':>6} {'MPPE':>7}")
    for r in rows:
        print(f"{r['ep']:>3} {r['object']:<22} {r['add_auc']:6.3f} {r['chord_sr']:5.2f} {r['maniptrans_sr']:6.2f} {r['mppe_cm']:7.1f}")
    n = len(rows)
    means = {k: sum(r[k] for r in rows) / n for k in KEYS}
    mppe = sorted(r["mppe_cm"] for r in rows)
    median = mppe[n // 2] if n % 2 else (mppe[n // 2 - 1] + mppe[n // 2]) / 2
    print("-" * 54)
    print(f"{'mean':>3} {f'({n} episodes)':<22} {means['add_auc']:6.3f} {means['chord_sr']:5.2f} {means['maniptrans_sr']:6.2f} {means['mppe_cm']:7.1f}")
    print(f"MPPE median {median:.1f} cm; episodes with MPPE <= 10 cm: {sum(v <= 10 for v in mppe)}/{n}; "
          f"<= 3.5 cm: {sum(v <= 3.5 for v in mppe)}/{n}; worlds per episode: {rows[0]['worlds']}")
    print("baseline (350M): AUC 0.51  SP-SR 0.70  MP-SR 0.20  MPPE 17.7")

    if args.csv:
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["ep", "object", "worlds", *KEYS])
            w.writeheader(); w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
