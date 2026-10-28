#!/usr/bin/env python3
"""
Hold the object still in a retargeted reference while it rests at the start of the clip.

Captured references jitter while the object should be at rest (ep 25 bottle: ~1 cm / 2-4 deg over frames 0-110).
Training resets at a random frame place the object at that frame's pose with zero velocity, so a tilted or
lifted pose of an upright bottle tips it over. This writes a copy of the reference where frames 0 .. end-1 hold
the object at the pose of --hold-frame (default 0), then blends back to the original over --blend frames.
Only object_body_position / object_body_wxyz (read by FlashCHORD) and object_root_position (same hold) change;
the robot is untouched. Evaluation still scores against
the official reference; this copy is for training only.

end (first moving frame) = --end, or automatically the first frame where the object is more than --move-cm or
--move-deg away from its frame-0 pose, minus --margin.

    python hold_rest.py --motion-root /localscratch/wangti98/track2_tier1_motion_palm2 \
        --out-root /localscratch/wangti98/track2_tier1_motion_palm2hold --episodes 25

Output: same layout as the input motion root for the given episodes (parquet + support usda) and a
retarget_results.json listing them. Object mesh / urdf paths in the parquet are unchanged.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R, Slerp


def process(args, p_in: Path, p_out: Path) -> dict:
    tbl = pq.read_table(p_in)
    row = {k: v[0] for k, v in tbl.to_pydict().items()}
    op = np.asarray(row["object_body_position"], float)
    oq = np.asarray(row["object_body_wxyz"], float)
    multi = op.ndim == 3
    p, q = (op[:, 0], oq[:, 0]) if multi else (op, oq)
    rot = R.from_quat(q, scalar_first=True)
    T = len(p)
    dpos = 100 * np.linalg.norm(p - p[0], axis=1)
    drot = np.degrees((rot * rot[0].inv()).magnitude())
    if args.end is not None:
        end = args.end
    else:
        moving = np.nonzero((dpos > args.move_cm) | (drot > args.move_deg))[0]
        end = (int(moving[0]) if len(moving) else T) - args.margin
    end = int(np.clip(end, 1, T - 1))
    h = int(np.clip(args.hold_frame, 0, end - 1))
    p_new = p.copy()
    q_new = q.copy()
    p_new[:end] = p[h]
    q_new[:end] = q[h]
    b1 = min(end + args.blend, T - 1)
    if b1 > end:
        w = (np.arange(end, b1) - end + 1) / (b1 - end + 1)
        p_new[end:b1] = (1 - w)[:, None] * p[h] + w[:, None] * p[end:b1]
        for i, (k, wk) in enumerate(zip(range(end, b1), w)):
            s = Slerp([0.0, 1.0], R.from_quat(np.stack([q[h], q[k]]), scalar_first=True))
            q_new[k] = s([wk]).as_quat(scalar_first=True)[0]
    removed_rot = float(np.degrees((rot[:end] * rot[h].inv()).magnitude()).max())
    removed_pos = float(100 * np.linalg.norm(p[:end] - p[h], axis=1).max())
    if multi:
        op[:, 0], oq[:, 0] = p_new, q_new
    else:
        op, oq = p_new, q_new
    if "object_root_position" in row and row["object_root_position"] is not None:
        rp = np.asarray(row["object_root_position"], float)
        r2 = rp[:, 0] if rp.ndim == 3 else rp
        r_new = r2.copy()
        r_new[:end] = r2[h]
        if b1 > end:
            r_new[end:b1] = (1 - w)[:, None] * r2[h] + w[:, None] * r2[end:b1]
        if rp.ndim == 3:
            rp[:, 0] = r_new
        else:
            rp = r_new
        row["object_root_position"] = rp.tolist()
    row["object_body_position"] = op.tolist()
    row["object_body_wxyz"] = oq.tolist()
    p_out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict({k: [v] for k, v in row.items()}, schema=tbl.schema), p_out)
    other = [c for c in tbl.column_names if c.startswith("object") and c not in ("object_body_position", "object_body_wxyz", "object_root_position")
             and any(k in c for k in ("position", "wxyz", "quat", "velocity", "pose"))]
    return {"frames": T, "end": end, "hold_frame": h, "blend": b1 - end,
            "max_pos_removed_cm": round(removed_pos, 2),
            "max_rot_removed_deg": round(removed_rot, 2), "other_object_columns": other}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--motion-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--end", type=int, default=None, help="first frame that is NOT held (default: auto)")
    ap.add_argument("--move-cm", type=float, default=3.0)
    ap.add_argument("--move-deg", type=float, default=10.0)
    ap.add_argument("--margin", type=int, default=5, help="frames before the detected lift-off")
    ap.add_argument("--hold-frame", type=int, default=0)
    ap.add_argument("--blend", type=int, default=10)
    args = ap.parse_args()
    src, dst = args.motion_root.resolve(), args.out_root.resolve()
    if src == dst:
        raise SystemExit("--out-root must differ from --motion-root")
    data = json.loads((src / "retarget_results.json").read_text())
    entries = data if isinstance(data, list) else list(data.values())
    keep, report = [], {}
    for e in entries:
        if not isinstance(e, dict) or e.get("episode_index") not in args.episodes:
            continue
        p_in = Path(e["parquet"])
        try:
            rel = p_in.relative_to(src)
        except ValueError:
            rel = Path(*p_in.parts[-5:])
        p_out = dst / rel
        st = process(args, p_in, p_out)
        print(f"ep {e['episode_index']:02d} {e.get('object', '')}: held frames 0-{st['end'] - 1} at frame {st['hold_frame']} "
              f"(removed up to {st['max_pos_removed_cm']} cm / {st['max_rot_removed_deg']} deg), blend {st['blend']} frames"
              + (f"; NOTE other object columns not changed: {st['other_object_columns']}" if st["other_object_columns"] else ""))
        sid = e["sequence_id"]
        stage_src, stage_dst = src / "whole_body/reconstructed_stage", dst / "whole_body/reconstructed_stage"
        stage_dst.mkdir(parents=True, exist_ok=True)
        for f in stage_src.glob(f"{sid}*_support.usda"):
            shutil.copy2(f, stage_dst / f.name)
        e2 = dict(e)
        e2["parquet"] = str(p_out)
        keep.append(e2)
        report[sid] = {"episode_index": e["episode_index"], **st}
    if not keep:
        raise SystemExit("no episode matched")
    (dst / "retarget_results.json").write_text(json.dumps(keep, indent=1))
    (dst / "hold_rest_results.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {dst / 'retarget_results.json'} ({len(keep)} episodes)")


if __name__ == "__main__":
    main()