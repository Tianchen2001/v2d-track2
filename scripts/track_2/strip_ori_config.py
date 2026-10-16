#!/usr/bin/env python3
"""
Make a checkpoint trained with the make_ori_runtime.py copy loadable by the public FlashCHORD runtime (aee95f25).

The training-only reward keys are removed from the resolved config stored in the checkpoint metadata:
  objective.object_orientation_weight / object_orientation_std (wherever an `objective` block appears, e.g. also
  under env), and an `object_orientation` entry in any curriculum `objective_weights` block.
Tensors (actor, critic, optimizer state) are copied unchanged; only the config JSON changes. The original file is
kept; the stripped one is written next to it (or to --out), so the evaluator gets a checkpoint whose config uses
only keys the public runtime knows.

    python strip_ori_config.py --dir /localscratch/wangti98/flash_chord_outputs/tier1_50m_ori
        -> tier1_50m_ori_pub/ep_NNN/policy_<step>.safetensors (latest policy of every ep_NNN, same layout, so
           select_checkpoints.py / run_policy_evaluation.py use tier1_50m_ori_pub like any other run)
    python strip_ori_config.py path/to/policy_X.safetensors   -> path/to/pub/policy_X.safetensors
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.numpy import save_file

KEY = "user.resolved_config_json"
DROP_FIELDS = ("object_orientation_weight", "object_orientation_std")


def strip(node, path=""):
    removed = []
    if isinstance(node, dict):
        for k in list(node):
            if k in DROP_FIELDS:
                node.pop(k); removed.append(f"{path}.{k}".lstrip("."))
            elif k == "objective_weights" and isinstance(node[k], dict) and "object_orientation" in node[k]:
                node[k].pop("object_orientation"); removed.append(f"{path}.{k}.object_orientation".lstrip("."))
        for k, v in node.items():
            removed += strip(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            removed += strip(v, f"{path}[{i}]")
    return removed


def process(src: Path, dst: Path) -> None:
    with safe_open(str(src), framework="numpy") as f:
        meta = dict(f.metadata() or {})
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    if KEY not in meta:
        raise SystemExit(f"{src}: no {KEY} in the metadata")
    cfg = json.loads(meta[KEY])
    removed = strip(cfg)
    meta[KEY] = json.dumps(cfg, separators=(",", ":"), sort_keys=False)
    save_file(tensors, str(dst), metadata=meta)
    print(f"{src.name} -> {dst}: removed {len(removed)} key(s)" + (f": {', '.join(sorted(set(r.split('[')[0] for r in removed)))}" if removed else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", type=Path, nargs="*")
    ap.add_argument("--dir", type=Path, default=None, help="run_train_all output dir: strip the latest policy of every ep_NNN")
    args = ap.parse_args()
    jobs = [(f, f.parent / "pub" / f.name) for f in args.checkpoints]
    if args.dir is not None:
        out = args.dir.with_name(args.dir.name + "_pub")
        for d in sorted(args.dir.glob("ep_*")):
            c = sorted(d.glob("policy_*.safetensors"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
            if c:
                jobs.append((c[-1], out / d.name / c[-1].name))
    if not jobs:
        raise SystemExit("no checkpoints")
    for src, dst in jobs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        process(src, dst)


if __name__ == "__main__":
    main()
