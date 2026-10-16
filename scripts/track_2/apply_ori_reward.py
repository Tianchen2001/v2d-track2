#!/usr/bin/env python3
"""
Add an object-orientation term to the FlashCHORD ReconBody objective, in place, in a flash_chord source tree.

    python apply_ori_reward.py <path to .../flash_chord/src/flash_chord/objectives/recon_body.py> [--check]

New term (13th, "object_orientation"):
    exp(-angle(q_object, q_object_target)^2 / object_orientation_std^2)
q_object_target is command.body_target_quat_w, the same per-world object target the object_position term uses.

New config keys (objective.*), both TRAINING-ONLY:
    object_orientation_weight  default 0.0  -> with the default the objective is exactly the recipe objective
    object_orientation_std     default 0.5  (rad)
Curriculum stages of the recipe do not list the new term; set_weights() then keeps object_orientation_weight.

A checkpoint trained with a nonzero weight stores the two keys in its config: run strip_ori_config.py on it before
evaluating with the public runtime. The patch refuses to run if the file does not look like the expected recipe file
(nothing is written then), and is idempotent (an already patched file is reported and left alone).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MARK = "object_orientation_weight"

EDITS = [
    ("term names",
     r'(\n    "force_closure",\n\))',
     '\n    "force_closure",\n    "object_orientation",\n)', 0),
    ("term count",
     r"_RECON_BODY_OBJECTIVE_TERM_COUNT = wp\.constant\(12\)",
     "_RECON_BODY_OBJECTIVE_TERM_COUNT = wp.constant(13)", 0),
    ("config fields",
     r"(\n    object_position_std: float = [^\n]+\n)",
     r"\1    object_orientation_weight: float = 0.0  # training-only, 0 = recipe objective\n"
     r"    object_orientation_std: float = 0.5  # rad, training-only\n", 0),
    ("weights property",
     r"(\n            self\.force_closure_weight,\n        \))",
     "\n            self.force_closure_weight,\n            self.object_orientation_weight,\n        )", 0),
    ("kernel argument: target quaternion",
     r"(\n    object_target_pos_w: wp\.array\(dtype=wp\.vec3\),\n)",
     r"\1    object_target_quat_w: wp.array(dtype=wp.quat),\n", 0),
    ("kernel argument: std",
     r"(def compose_recon_body_objective\(.*?\n    palm_orientation_std: float,\n)",
     r"\1    object_orientation_std: float,\n", re.S),
    ("kernel: new term",
     r"(\n    terms\[term_base \+ 11\] = force_closure\[world\]\n)",
     r"\1"
     "    object_orientation = wp.normalize(wp.transform_get_rotation(body_q[body_base + object_body_id]))\n"
     "    object_orientation_error = quat_geodesic_angle(object_orientation, wp.normalize(object_target_quat_w[world]))\n"
     "    terms[term_base + 12] = wp.exp(\n"
     "        -(object_orientation_error * object_orientation_error) / (object_orientation_std * object_orientation_std)\n"
     "    )\n", 0),
    ("launch input: target quaternion",
     r"(\n                self\.command\.body_target_pos_w,\n)",
     r"\1                self.command.body_target_quat_w,\n", 0),
    ("launch input: std",
     r"(\n                self\.config\.palm_orientation_std,\n)",
     r"\1                self.config.object_orientation_std,\n", 0),
    ("set_weights fallback",
     r"    def set_weights\(self, weights: Mapping\[str, float\]\) -> None:\n",
     "    def set_weights(self, weights: Mapping[str, float]) -> None:\n"
     "        if \"object_orientation\" not in weights:  # recipe curricula predate the term: keep the config weight\n"
     "            weights = {**weights, \"object_orientation\": self.config.object_orientation_weight}\n", 0),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path)
    ap.add_argument("--check", action="store_true", help="only report whether every edit applies; write nothing")
    args = ap.parse_args()
    s = args.file.read_text()
    if MARK in s:
        print(f"{args.file}: already patched, nothing to do")
        return
    for name, pattern, repl, flags in EDITS:
        new, n = re.subn(pattern, repl, s, count=1, flags=flags)
        if n != 1:
            sys.exit(f"FAILED at '{name}': pattern not found. {args.file} differs from the recipe file this patch "
                     "was written for; nothing was written.")
        s = new
        print(f"  ok  {name}")
    for needle in ('"object_orientation",', "wp.constant(13)", "object_target_quat_w[world]",
                   "self.command.body_target_quat_w,", "self.config.object_orientation_std,"):
        if needle not in s:
            sys.exit(f"FAILED: '{needle}' missing after patching; nothing was written")
    if args.check:
        print("check passed (nothing written)")
        return
    args.file.write_text(s)
    print(f"patched {args.file}")


if __name__ == "__main__":
    main()
