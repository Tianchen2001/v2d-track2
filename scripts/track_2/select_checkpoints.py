#!/usr/bin/env python3
"""
Pick one checkpoint per episode from several training runs by an evaluation metric, and assemble a
submission-ready results directory.

Each candidate is  label=<train tag dir>:<eval results dir>
  train tag dir   : run_train_all.py output (ep_NNN/policy_*.safetensors)
  eval results dir: episode_NNNNNN.json/.parquet from the kit evaluator (1 world) for that run
An episode is a candidate for a run only if that run's eval json AND parquet exist (a failed or
non-finite 1-world rollout drops out automatically).

    python select_checkpoints.py --out /localscratch/wangti98/flash_chord_outputs/mix_tier1 \
        50M=/localscratch/wangti98/flash_chord_outputs/tier1_50m:/localscratch/wangti98/eval_50m_1w \
        350M=/localscratch/wangti98/flash_chord_outputs/tier1_350m:/localscratch/wangti98/eval_350m_1w \
        hand=/localscratch/wangti98/flash_chord_outputs/tier1_350m_hand:/localscratch/wangti98/eval_hand_1w \
        [--manifest .../retarget_results.json] [--episodes 30] [--force ep=label ...]

Writes
  <out>/checkpoints.json   episode -> checkpoint (absolute), for run_policy_evaluation.py / record_rollout.py
  <out>/results/           the chosen episode_NNNNNN.parquet/.json, ready for tools/pack_submission.py
  <out>/selection.json     per episode: chosen label, all candidates' metrics
Ranking (--order): SP-SR, then MP-SR, then MPPE, then AUC (1-world SP-SR / MP-SR are 0/1 per episode).
and prints the per-episode choice and the mean metrics of the mixture vs each pure run.
Tier 1 only: choosing by a score against the clean reference is not allowed for Tiers 2/3.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

KEYS = ["add_auc", "spider_sr_uncentered", "maniptrans_sr", "mppe_cm"]
HIGHER = {"add_auc": True, "spider_sr_uncentered": True, "maniptrans_sr": True, "mppe_cm": False}
SHORT = {"add_auc": "AUC", "spider_sr_uncentered": "SP-SR", "maniptrans_sr": "MP-SR", "mppe_cm": "MPPE"}


def latest_policy(d: Path) -> Path | None:
    c = sorted(d.glob("policy_*.safetensors"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    return c[-1] if c else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="label=<train tag dir>:<eval results dir>")
    ap.add_argument("--order", nargs="+", default=["spider_sr_uncentered", "maniptrans_sr", "mppe_cm", "add_auc"], choices=KEYS,
                    help="lexicographic ranking: first metric decides, later ones break ties (default SP-SR > MP-SR > MPPE > AUC)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--force", nargs="*", default=[], help="ep=label overrides")
    args = ap.parse_args()

    names = {}
    if args.manifest and args.manifest.exists():
        data = json.loads(args.manifest.read_text())
        for e in (data if isinstance(data, list) else list(data.values())):
            if isinstance(e, dict) and e.get("episode_index") is not None:
                names[int(e["episode_index"])] = e.get("object", "")
    runs = []
    for spec in args.runs:
        label, _, rest = spec.partition("=")
        tag, _, ev = rest.partition(":")
        runs.append((label, Path(tag).resolve(), Path(ev).resolve()))
    forced = {int(k): v for k, v in (f.split("=") for f in args.force)}

    cands: dict[int, dict[str, dict]] = {}
    for label, tag, ev in runs:
        for ep in range(args.episodes):
            js = ev / f"episode_{ep:06d}.json"; pq = js.with_suffix(".parquet")
            ck = latest_policy(tag / f"ep_{ep:03d}")
            if not (js.exists() and pq.exists() and ck is not None):
                continue
            j = json.loads(js.read_text()); m = j.get("metrics", j)
            if any(m.get(k) is None for k in KEYS):
                continue
            cands.setdefault(ep, {})[label] = {"metrics": {k: float(m[k]) for k in KEYS}, "checkpoint": str(ck),
                                                "json": str(js), "parquet": str(pq)}

    def key(c):
        return tuple((1 if HIGHER[k] else -1) * c["metrics"][k] for k in args.order)

    out = args.out.resolve(); (out / "results").mkdir(parents=True, exist_ok=True)
    chosen, selection, missing = {}, {}, []
    for ep in range(args.episodes):
        if ep not in cands:
            missing.append(ep); continue
        if ep in forced and forced[ep] in cands[ep]:
            lab = forced[ep]
        else:
            lab = max(cands[ep], key=lambda l: key(cands[ep][l]))
        c = cands[ep][lab]; chosen[ep] = lab
        for f in ("json", "parquet"):
            shutil.copy2(c[f], out / "results" / Path(c[f]).name)
        selection[str(ep)] = {"object": names.get(ep, ""), "chosen": lab, "checkpoint": c["checkpoint"],
                              "candidates": {l: v["metrics"] for l, v in cands[ep].items()}}
    (out / "checkpoints.json").write_text(json.dumps({str(ep): selection[str(ep)]["checkpoint"] for ep in sorted(chosen)}, indent=2))
    (out / "selection.json").write_text(json.dumps(selection, indent=2))

    labels = [l for l, _, _ in runs]
    print(f"ranking: {' > '.join(SHORT[k] for k in args.order)}")
    print(f"{'ep':>3} {'object':<20} {'chosen':<8} " + " ".join(f"{l + ' SP/MP/MPPE':>20}" for l in labels))
    for ep in sorted(chosen):
        row = []
        for l in labels:
            if l in cands[ep]:
                m = cands[ep][l]["metrics"]
                row.append(f"{m['spider_sr_uncentered']:>6.0f}{m['maniptrans_sr']:>4.0f}{m['mppe_cm']:>8.1f}  ")
            else:
                row.append(f"{'-':>20}")
        print(f"{ep:>3} {names.get(ep, '')[:20]:<20} {chosen[ep]:<8} " + " ".join(f"{x:>20}" for x in row))
    print("\nmeans over the episodes each run covers (mixture over all selected):")
    for l in labels + ["MIX"]:
        eps = [e for e in chosen if (l == "MIX" or l in cands[e])]
        vals = {k: [cands[e][chosen[e] if l == "MIX" else l]["metrics"][k] for e in eps] for k in KEYS}
        print(f"  {l:<6} n={len(eps):>2}  " + "  ".join(f"{SHORT[k]} {sum(v)/len(v):.3f}" for k, v in vals.items() if v))
    print("  baseline       AUC 0.510  SP-SR 0.700  MP-SR 0.200  MPPE 17.700")
    if missing:
        print(f"\nWARNING: no valid candidate for episodes {missing}; the submission needs all {args.episodes}")
    print(f"\nwrote {out / 'checkpoints.json'}, {out / 'results'}, {out / 'selection.json'}")


if __name__ == "__main__":
    main()