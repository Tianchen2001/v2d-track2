#!/usr/bin/env python3
"""
Copy a FlashCHORD checkpoint with changed simulation / scene settings in its stored resolved config.

The evaluator rebuilds the whole environment from the config stored in the checkpoint
(user.resolved_config_json), so evaluating the copy runs the same policy under the changed physics.
Tensors are copied unchanged; the original checkpoint is not touched.

    python physics_variant.py <policy_X.safetensors> --out <dir>/policy_X.safetensors \
        --set sim.cone=elliptic --set sim.impratio=10 --set scene.decompose_objects=false

--set BLOCK.FIELD=VALUE sets FIELD in every dict stored under a key named BLOCK (the resolved config repeats
some blocks, e.g. sim at the top level and under env). FIELD must already exist there unless --allow-new.
VALUE is parsed as JSON (true / false / numbers), otherwise kept as a string.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.numpy import save_file

KEY = "user.resolved_config_json"


def parse_value(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def apply(node, block: str, field: str, value, allow_new: bool, path: str = "") -> list[str]:
    hits = []
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{path}.{k}".lstrip(".")
            if k == block and isinstance(v, dict) and (field in v or allow_new):
                old = v.get(field, "<new>")
                v[field] = value
                hits.append(f"{p}.{field}: {old} -> {value}")
            hits += apply(v, block, field, value, allow_new, p)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            hits += apply(v, block, field, value, allow_new, f"{path}[{i}]")
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--set", action="append", default=[], metavar="BLOCK.FIELD=VALUE")
    ap.add_argument("--allow-new", action="store_true")
    ap.add_argument("--show", nargs="*", default=[], metavar="BLOCK", help="print these blocks of the stored config")
    args = ap.parse_args()

    with safe_open(str(args.checkpoint), framework="numpy") as f:
        meta = dict(f.metadata() or {})
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    if KEY not in meta:
        raise SystemExit(f"{args.checkpoint}: no {KEY} in the metadata")
    cfg = json.loads(meta[KEY])
    for b in args.show:
        print(f"{b}: {json.dumps(cfg.get(b), indent=1)}")
    for s in args.set:
        lhs, _, rhs = s.partition("=")
        block, _, field = lhs.partition(".")
        if not field or not rhs:
            raise SystemExit(f"bad --set {s!r}")
        hits = apply(cfg, block, field, parse_value(rhs), args.allow_new)
        if not hits:
            raise SystemExit(f"--set {s}: no '{block}' block with field '{field}' (use --allow-new to add it)")
        for h in hits:
            print("  " + h)
    meta[KEY] = json.dumps(cfg, separators=(",", ":"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.out), metadata=meta)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
