#!/usr/bin/env python3
"""
How much does the object move by itself while it should be resting, in evaluator rollouts?

The rest window is the start of the clip until the REFERENCE object first moves (> --move-cm or > --move-deg
from its frame-0 pose), i.e. before the hand picks it up. In that window the rollout object should not move;
any tilt or drift there is the object settling / chattering on the support, not tracking.

    python rest_check.py label=<dir with run_*/episode_000025.parquet or episode_*.parquet> ... --episode 25

Per label: number of rollouts, rest window length, per rollout max tilt / drift inside the window,
fall rate (max tilt > --fall-deg), and the final MPPE / SP / MP from the metrics JSON next to each parquet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R


def track(t, prefix):
    cols = [f"{prefix}_{k}" for k in ("position_x", "position_y", "position_z",
                                      "quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w")]
    a = np.stack([t.column(c).to_numpy() for c in cols], 1).astype(float)
    return a[:, :3], R.from_quat(a[:, 3:])                                  # xyzw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", metavar="label=dir")
    ap.add_argument("--episode", type=int, default=25)
    ap.add_argument("--move-cm", type=float, default=1.0)
    ap.add_argument("--move-deg", type=float, default=5.0)
    ap.add_argument("--fall-deg", type=float, default=30.0)
    ap.add_argument("--window", type=int, default=0, help="fixed window length in frames (0 = until the reference moves)")
    ap.add_argument("--show-ref", action="store_true", help="print reference displacement / rotation every 10 frames")
    args = ap.parse_args()
    name = f"episode_{args.episode:06d}"
    for spec in args.runs:
        label, _, d = spec.partition("=")
        files = sorted(Path(d).glob(f"**/{name}.parquet"))
        if not files:
            print(f"{label}: no {name}.parquet under {d}")
            continue
        rows = []
        for f in files:
            t = pq.read_table(f)
            pa, ra = track(t, "achieved")
            pr, rr = track(t, "reference")
            ref_move = np.maximum(100 * np.linalg.norm(pr - pr[0], axis=1) / args.move_cm,
                                  np.degrees((rr * rr[0].inv()).magnitude()) / args.move_deg)
            moving = np.nonzero(ref_move > 1.0)[0]
            w = int(moving[0]) if len(moving) else len(pr)
            if args.window > 0:
                w = args.window
            w = max(min(w, len(pr)), 2)
            if args.show_ref and f == files[0]:
                dp = 100 * np.linalg.norm(pr - pr[0], axis=1); dr = np.degrees((rr * rr[0].inv()).magnitude())
                print(f"   reference ({f.parent.name}): frame / move cm / rot deg")
                for i in range(0, min(len(pr), 300), 10):
                    print(f"     {i:4d} {dp[i]:6.2f} {dr[i]:6.2f}")
            tilt = np.degrees((ra[:w] * ra[0].inv()).magnitude())
            drift = 100 * np.linalg.norm(pa[:w] - pa[0], axis=1)
            m = {}
            jf = f.with_suffix(".json")
            if jf.exists():
                m = json.load(open(jf)); m = m.get("metrics", m)
            rows.append((f.parent.name if f.parent.name.startswith("run_") else f.parent.parent.name,
                         w, tilt.max(), drift.max(), int(np.argmax(tilt > args.fall_deg)) if (tilt > args.fall_deg).any() else -1,
                         m.get("mppe_cm", np.nan), m.get("spider_sr_uncentered", np.nan), m.get("maniptrans_sr", np.nan)))
        falls = sum(r[2] > args.fall_deg for r in rows)
        print(f"== {label}: {len(rows)} rollouts, rest window {rows[0][1]} frames, "
              f"fell during rest {falls}/{len(rows)}, median max tilt {np.median([r[2] for r in rows]):.1f} deg, "
              f"median MPPE {np.nanmedian([r[5] for r in rows]):.1f} cm, SP {np.nansum([r[6] for r in rows]):.0f}/{len(rows)}")
        for r in rows:
            print(f"   {r[0]:<10} max tilt {r[2]:6.1f} deg  drift {r[3]:5.1f} cm  fell@{r[4]:>4}  "
                  f"MPPE {r[5]:6.1f}  SP {r[6]:.0f}  MP {r[7]:.0f}")


if __name__ == "__main__":
    main()