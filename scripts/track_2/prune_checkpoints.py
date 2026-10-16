#!/usr/bin/env python3
"""
Free disk space in FlashCHORD training outputs: in every <root>/<tag>/ep_NNN/ keep only the LAST checkpoint of each
kind (policy_<step>.safetensors, learner/critic/... _<step>.safetensors) and delete the older intermediate ones.

Never deleted:
  - the newest file of each kind in each ep_NNN (what run_train_all / select_checkpoints / packing use)
  - any file named in a checkpoints.json or selection.json found under --keep-from (default /localscratch/wangti98)
  - ep_NNN folders without a DONE marker (training still running or failed) unless --include-unfinished

Dry run by default: prints what would go and how much space it frees. Add --delete to actually remove.

    python prune_checkpoints.py /localscratch/wangti98/flash_chord_outputs              # dry run, all tags
    python prune_checkpoints.py /localscratch/wangti98/flash_chord_outputs --tags tier1_50m tier1_350m --delete
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def referenced(root: Path) -> set[str]:
    keep = set()
    for name in ("checkpoints.json", "selection.json"):
        for f in root.rglob(name):
            try:
                txt = f.read_text()
                json.loads(txt)
            except Exception:
                continue
            for m in re.findall(r'"([^"]+\.safetensors)"', txt):
                p = Path(m)
                keep.add(str(p.resolve()) if p.is_absolute() else str((f.parent / p).resolve()))
                keep.add(p.name if not p.is_absolute() else str(p))
    return keep


def gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="outputs folder holding <tag>/ep_NNN")
    ap.add_argument("--tags", nargs="*", default=None, help="only these run tags (default: all)")
    ap.add_argument("--keep-from", type=Path, default=Path("/localscratch/wangti98"))
    ap.add_argument("--include-unfinished", action="store_true")
    ap.add_argument("--delete", action="store_true")
    args = ap.parse_args()

    keep = referenced(args.keep_from)
    print(f"{len(keep)} checkpoint paths referenced by checkpoints.json / selection.json are protected")
    total, per_tag = 0, defaultdict(int)
    tags = [args.root / t for t in args.tags] if args.tags else sorted(p for p in args.root.iterdir() if p.is_dir())
    for tag in tags:
        for ep in sorted(tag.glob("ep_*")):
            if not ep.is_dir():
                continue
            if not (ep / "DONE").exists() and not args.include_unfinished:
                continue
            groups = defaultdict(list)
            for f in ep.glob("*.safetensors"):
                m = re.match(r"(.*?)_?(\d+)$", f.stem)
                if m:
                    groups[m.group(1)].append((int(m.group(2)), f))
            for kind, files in groups.items():
                files.sort()
                for _, f in files[:-1]:            # all but the newest of this kind
                    r = str(f.resolve())
                    if r in keep or str(f) in keep:
                        continue
                    size = f.stat().st_size
                    total += size; per_tag[tag.name] += size
                    if args.delete:
                        f.unlink()
    for t, s in sorted(per_tag.items(), key=lambda x: -x[1]):
        print(f"  {t:40s} {gb(s)}")
    print(("deleted " if args.delete else "would free ") + gb(total) + ("" if args.delete else "   (dry run; add --delete)"))


if __name__ == "__main__":
    main()
