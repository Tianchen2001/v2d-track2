#!/usr/bin/env python3
"""
Train one FlashCHORD policy per Track 2 episode, evaluate it, and collect everything the
submission kit needs. Run INSIDE the FlashCHORD container from /workspace (flash_chord root).

    python /localscratch/wangti98/v2d_scripts/run_train_all.py \
        --results /localscratch/wangti98/track2_tier1_motion/retarget_results.json \
        --tag tier1_50m --steps 50000000 [--episodes 0 3] [--eval-worlds 1024] [--skip-eval]

Per episode it writes
    outputs/<tag>/ep_NNN/policy_*.safetensors   (training checkpoints; the last one is used)
    outputs/<tag>/ep_NNN/train.log
    outputs/<tag>/results/episode_NNNNNN.parquet + .json   (evaluator trajectory + metrics)
and at the end
    outputs/<tag>/checkpoints.json      {episode_index: checkpoint path}   -> kit run_policy_evaluation.py
    outputs/<tag>/summary.json          per-episode status, steps, seconds, metrics

Shorter runs (--steps < 349999104) scale the nine curriculum thresholds proportionally so the
virtual object controller (VOC) is faded out by the end of training; evaluation runs with VOC off.
Episodes whose support USDA holds no Cube/Cylinder prims get scene.support=false.
Already-finished episodes are skipped unless --force.
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

FULL_STEPS = 349999104
FULL_THRESHOLDS = [35000320, 70000640, 104998912, 139999232, 174999552, 209999872, 288749568, 306249728, 349999104]
STEP_QUANTUM = 2048


def q(n: float) -> int:
    return max(STEP_QUANTUM, int(round(n / STEP_QUANTUM)) * STEP_QUANTUM)


def support_has_shapes(usda: Path) -> bool:
    if not usda.is_file():
        return False
    text = usda.read_text(errors="ignore")
    return bool(re.search(r"def\s+(Cube|Cylinder)\b", text))


def latest_policy(d: Path) -> Path | None:
    cands = sorted(d.glob("policy_*.safetensors"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    return cands[-1] if cands else None


def run(cmd: list[str], log: Path) -> int:
    with log.open("a") as fh:
        fh.write("$ " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        fh.flush()
        return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="retarget_results.json from run_retarget_all.py")
    ap.add_argument("--tag", required=True, help="run name; outputs go to outputs/<tag>/")
    ap.add_argument("--steps", type=int, default=FULL_STEPS, help="total environment steps per episode")
    ap.add_argument("--experiment", default="g1_recon_body_flash_sac")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--eval-worlds", type=int, default=4096)
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra Hydra overrides for training")
    args = ap.parse_args()

    results = json.loads(args.results.read_text())
    results = [r for r in results if r.get("status") in ("ok", "skipped")]
    if args.episodes is not None:
        results = [r for r in results if r["episode_index"] in set(args.episodes)]
    results.sort(key=lambda r: r["episode_index"])

    steps = q(args.steps)
    ratio = steps / FULL_STEPS
    thresholds = [q(t * ratio) for t in FULL_THRESHOLDS]
    thresholds[-1] = steps
    for i in range(1, len(thresholds)):            # keep strictly increasing
        thresholds[i] = max(thresholds[i], thresholds[i - 1] + STEP_QUANTUM)
    thr_override = "curriculum.thresholds=[" + ",".join(str(t) for t in thresholds) + "]"

    out_root = Path("outputs") / args.tag
    res_dir = out_root / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(results)} episodes, {steps:,} steps each, thresholds {thresholds[0]:,} ... {thresholds[-1]:,}")

    summary, checkpoints = [], {}
    for r in results:
        ep = int(r["episode_index"])
        pq = Path(r["parquet"])
        if not pq.is_file():
            print(f"ep {ep:02d} MISSING parquet {pq}")
            summary.append({"episode_index": ep, "status": "missing_parquet"})
            continue
        stage = pq.parents[1].parent.parent / "reconstructed_stage" / f"{r['sequence_id']}_support.usda"
        support = support_has_shapes(stage)
        ep_dir = out_root / f"ep_{ep:03d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        log = ep_dir / "train.log"
        done_marker = ep_dir / "DONE"
        entry = {"episode_index": ep, "sequence_id": r["sequence_id"], "object": r["object"], "support": support}

        # ---- train ----
        if done_marker.exists() and not args.force:
            print(f"ep {ep:02d} train: skip (DONE)")
        else:
            cmd = [sys.executable, "scripts/train_flash_sac.py", f"experiment={args.experiment}",
                   f"task.parquet='{pq}'", f"reset.seed={args.seed}", "logging.mode=disabled",
                   f"training.total_environment_steps={steps}", thr_override,
                   f"scene.support={'true' if support else 'false'}",
                   f"output_dir={ep_dir}", *args.extra]
            print(f"ep {ep:02d} train: {r['object']} support={support} ...", flush=True)
            t0 = time.time()
            rc = run(cmd, log)
            dt = time.time() - t0
            entry.update({"train_rc": rc, "train_seconds": round(dt)})
            if rc == 0 and latest_policy(ep_dir) is not None:
                done_marker.write_text(json.dumps({"steps": steps, "seconds": dt}))
                print(f"   ok {dt/3600:.2f} h", flush=True)
            else:
                print(f"   FAIL rc={rc} ({dt:.0f}s), see {log}", flush=True)
                entry["status"] = "train_failed"
                summary.append(entry)
                continue

        ckpt = latest_policy(ep_dir)
        entry["checkpoint"] = str(ckpt)
        checkpoints[str(ep)] = str(ckpt)

        # ---- evaluate (official evaluator, exports the trajectory parquet the packer reads) ----
        traj = res_dir / f"episode_{ep:06d}.parquet"
        metrics = res_dir / f"episode_{ep:06d}.json"
        if args.skip_eval:
            entry["status"] = "trained"
        elif traj.exists() and not args.force:
            entry["status"] = "evaluated"
            print(f"ep {ep:02d} eval: skip (exists)")
        else:
            cmd = [sys.executable, "scripts/evaluate_policy.py", f"evaluation.checkpoint={ckpt}",
                   f"evaluation.world_count={args.eval_worlds}", "evaluation.reset_mode=explicit",
                   f"evaluation.metrics_output={metrics}", f"evaluation.object_trajectories_output={traj}"]
            print(f"ep {ep:02d} eval ...", flush=True)
            rc = run(cmd, log)
            entry["eval_rc"] = rc
            if rc == 0 and traj.exists():
                m = json.loads(metrics.read_text()).get("metrics", {})
                entry["metrics"] = {k: m.get(k) for k in ("add_auc", "chord_sr", "maniptrans_sr", "mppe_cm")}
                entry["status"] = "evaluated"
                print(f"   AUC={m.get('add_auc',0):.3f} MPPE={m.get('mppe_cm',0):.1f}cm SR={m.get('chord_sr',0):.2f}", flush=True)
            else:
                entry["status"] = "eval_failed"
                print(f"   eval FAIL rc={rc}", flush=True)
        summary.append(entry)
        (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
        (out_root / "checkpoints.json").write_text(json.dumps(checkpoints, indent=2))

    n_ev = sum(e.get("status") == "evaluated" for e in summary)
    mppe = [e["metrics"]["mppe_cm"] for e in summary if e.get("metrics") and e["metrics"].get("mppe_cm") is not None]
    print(f"\n{n_ev}/{len(summary)} evaluated; mean MPPE over evaluated = {sum(mppe)/len(mppe):.2f} cm" if mppe else f"\n{n_ev}/{len(summary)} evaluated")
    print(f"checkpoints: {out_root / 'checkpoints.json'}\nresults dir: {res_dir}")


if __name__ == "__main__":
    main()
