#!/usr/bin/env python3
"""
Side-by-side comparison of several evaluation result directories (episode_*.json written by
evaluate_policy.py or by the kit's run_policy_evaluation.py), per episode and on average.

    python compare_evals.py \
        --manifest /localscratch/wangti98/track2_tier1_motion/retarget_results.json \
        50M=/localscratch/wangti98/flash_chord_outputs/tier1_50m/results \
        350M=/localscratch/wangti98/flash_chord_outputs/merged_tier1/results \
        350M-1w=/localscratch/wangti98/eval_merged_tier1 \
        [--metric mppe_cm] [--csv out.csv] [--low-sr 0.1]

Prints one table per metric (AUC, SR, MP-SR, MPPE) with one column per run, the means, the
baseline row from the leaderboard, and the episodes whose SR is <= --low-sr in the last run.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path

KEYS = {"add_auc": "AUC", "chord_sr": "SR", "maniptrans_sr": "MP-SR", "mppe_cm": "MPPE"}
BASELINE = {"add_auc": 0.51, "chord_sr": 0.70, "maniptrans_sr": 0.20, "mppe_cm": 17.7}  # Tier 1 CHORD baseline, leaderboard Oct 6
HIGHER = {"add_auc": True, "chord_sr": True, "maniptrans_sr": True, "mppe_cm": False}


def load_dir(d: Path) -> dict[int, dict]:
    out = {}
    for f in sorted(glob.glob(str(d / "episode_*.json"))):
        ep = int(Path(f).stem.split("_")[-1])
        j = json.loads(Path(f).read_text()); m = j.get("metrics", j)
        out[ep] = {k: m.get(k) for k in KEYS} | {"worlds": j.get("world_count") or m.get("world_count")}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=dir")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--low-sr", type=float, default=0.1)
    args = ap.parse_args()

    names = {}
    if args.manifest and args.manifest.exists():
        data = json.loads(args.manifest.read_text())
        for e in (data if isinstance(data, list) else list(data.values())):
            if isinstance(e, dict) and e.get("episode_index") is not None:
                names[int(e["episode_index"])] = e.get("object", "")
    runs = []
    for spec in args.runs:
        label, _, d = spec.partition("=")
        res = load_dir(Path(d))
        if not res:
            raise SystemExit(f"no episode_*.json in {d}")
        runs.append((label, res))
    episodes = sorted(set().union(*(r.keys() for _, r in runs)))

    def fmt(v, k):
        return "   -  " if v is None else (f"{v:6.3f}" if k != "mppe_cm" else f"{v:6.1f}")
    rows = []
    for k, title in KEYS.items():
        print(f"\n{title}  ({'higher' if HIGHER[k] else 'lower'} is better)")
        print(f"{'ep':>3} {'object':<20} " + " ".join(f"{lab:>8}" for lab, _ in runs))
        for ep in episodes:
            vals = [r.get(ep, {}).get(k) for _, r in runs]
            print(f"{ep:>3} {names.get(ep, '')[:20]:<20} " + " ".join(f"{fmt(v, k):>8}" for v in vals))
        means = []
        for _, r in runs:
            v = [r[e][k] for e in episodes if e in r and r[e][k] is not None]
            means.append(sum(v) / len(v) if v else None)
        print(f"{'':>3} {'mean':<20} " + " ".join(f"{fmt(m, k):>8}" for m in means))
        print(f"{'':>3} {'baseline (CHORD)':<20} " + " ".join(f"{fmt(BASELINE[k], k):>8}" for _ in runs))
        worlds = [next((r[e]['worlds'] for e in episodes if e in r), None) for _, r in runs]
        print(f"{'':>3} {'worlds':<20} " + " ".join(f"{str(w):>8}" for w in worlds))
    for ep in episodes:
        row = {"ep": ep, "object": names.get(ep, "")}
        for lab, r in runs:
            for k in KEYS:
                row[f"{lab}_{KEYS[k]}"] = r.get(ep, {}).get(k)
        rows.append(row)
    last_label, last = runs[-1]
    low = [ep for ep in episodes if ep in last and last[ep]["chord_sr"] is not None and last[ep]["chord_sr"] <= args.low_sr]
    print(f"\nepisodes with SR <= {args.low_sr} in {last_label}: {low}  ({', '.join(names.get(e, str(e)) for e in low)})")
    if args.csv:
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
