#!/usr/bin/env python3
"""
Convert a V2D Track 2 tier (LeRobot v2.1 layout) into the input folders that
robotic_grounding/scripts/retarget/soma_to_g1.py expects.

For every episode this writes

    <out>/sequence_id=<sequence_id>/
        soma_params.npz                  SOMA-X params (poses, transl, identity, scale, ...)
        poses.npy                        (T, 4, 4) object pose in the source OpenCV world frame
        ground_plane.json                copied from <root>/ground_plane/<sequence_id>.json if present
        reconstructed_mesh/output_aligned.glb   copied object mesh
        episode.json                     provenance (episode index, object, task, frame counts)

Then:  python scripts/retarget/soma_to_g1.py <out>/sequence_id=<sid> --save --motion-root <dir>

Usage:
    python convert_tier_to_retarget.py --root data/.../track_2/tier_1_multiview_caption --out runs/track2/tier1/retarget_in
    python convert_tier_to_retarget.py --root ... --out ... --episodes 0 3 7      # subset
    python convert_tier_to_retarget.py --root ... --out ... --dry-run             # just print what would happen

Deps: numpy, pyarrow, scipy.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# Must match robotic_grounding.retarget.params.SOMA_JOINTS_ORDER (77 joints).
SOMA_JOINTS_ORDER = [
    "Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "HeadEnd", "Jaw",
    "LeftEye", "RightEye",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
    "LeftHandThumb1", "LeftHandThumb2", "LeftHandThumb3", "LeftHandThumbEnd",
    "LeftHandIndex1", "LeftHandIndex2", "LeftHandIndex3", "LeftHandIndex4", "LeftHandIndexEnd",
    "LeftHandMiddle1", "LeftHandMiddle2", "LeftHandMiddle3", "LeftHandMiddle4", "LeftHandMiddleEnd",
    "LeftHandRing1", "LeftHandRing2", "LeftHandRing3", "LeftHandRing4", "LeftHandRingEnd",
    "LeftHandPinky1", "LeftHandPinky2", "LeftHandPinky3", "LeftHandPinky4", "LeftHandPinkyEnd",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightHandThumb1", "RightHandThumb2", "RightHandThumb3", "RightHandThumbEnd",
    "RightHandIndex1", "RightHandIndex2", "RightHandIndex3", "RightHandIndex4", "RightHandIndexEnd",
    "RightHandMiddle1", "RightHandMiddle2", "RightHandMiddle3", "RightHandMiddle4", "RightHandMiddleEnd",
    "RightHandRing1", "RightHandRing2", "RightHandRing3", "RightHandRing4", "RightHandRingEnd",
    "RightHandPinky1", "RightHandPinky2", "RightHandPinky3", "RightHandPinky4", "RightHandPinkyEnd",
    "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "LeftToeEnd",
    "RightLeg", "RightShin", "RightFoot", "RightToeBase", "RightToeEnd",
]
NJ = len(SOMA_JOINTS_ORDER)  # 77

COL_POSE = "observation.human.pose"
COL_TRANSL = "observation.human.translation"
COL_IDENTITY = "observation.human.identity_coeffs"
COL_SCALE = "observation.human.scale_params"
COL_BONEFLEX = "observation.human.bone_length_flexibles"
COL_OBJ_POSE = "observation.object.pose"
COL_OBJ_VIS = "observation.object.visible"


def fail(msg: str) -> None:
    raise SystemExit(f"convert_tier_to_retarget: {msg}")


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def col_to_np(table, name: str) -> np.ndarray:
    """fixed_size_list<float32, N> column -> (T, N) float32; scalar column -> (T,)."""
    if name not in table.column_names:
        fail(f"column {name!r} not in parquet; have {table.column_names}")
    arr = table.column(name).combine_chunks()
    import pyarrow as pa
    if pa.types.is_fixed_size_list(arr.type) or pa.types.is_list(arr.type):
        flat = arr.flatten().to_numpy(zero_copy_only=False)
        width = arr.type.list_size if pa.types.is_fixed_size_list(arr.type) else len(arr[0])
        return flat.reshape(len(arr), width)
    return arr.to_numpy(zero_copy_only=False)


def check_joint_names(info: dict) -> None:
    names = info["features"][COL_POSE].get("names")
    if not names:
        print("  warning: info.json has no joint names for human.pose; assuming SOMA order")
        return
    joints = [n.rsplit(".", 1)[0] for n in names[0::3]]
    if joints != SOMA_JOINTS_ORDER:
        bad = [(i, a, b) for i, (a, b) in enumerate(zip(joints, SOMA_JOINTS_ORDER)) if a != b]
        fail(f"human.pose joint order differs from SOMA_JOINTS_ORDER at {bad[:5]} ...")


def fill_invisible(pos: np.ndarray, quat_wxyz: np.ndarray, visible: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Interpolate object pose over invisible frames (linear position, slerp rotation).
    Leading/trailing invisible runs are held at the nearest visible pose."""
    T = len(visible)
    vis_idx = np.flatnonzero(visible)
    report = {"frames": int(T), "visible": int(vis_idx.size), "invisible": int(T - vis_idx.size)}
    if vis_idx.size == 0:
        fail("object is never visible in this episode")
    if vis_idx.size == T:
        return pos, quat_wxyz, report
    # position: linear interpolation, hold at ends
    t_all = np.arange(T)
    pos_f = np.stack([np.interp(t_all, vis_idx, pos[vis_idx, k]) for k in range(3)], axis=1)
    # rotation: slerp between visible keyframes, hold at ends
    key_rots = R.from_quat(quat_wxyz[vis_idx], scalar_first=True)
    if vis_idx.size == 1:
        quat_f = np.repeat(quat_wxyz[vis_idx], T, axis=0)
    else:
        slerp = Slerp(vis_idx.astype(float), key_rots)
        t_clip = np.clip(t_all, vis_idx[0], vis_idx[-1]).astype(float)
        quat_f = slerp(t_clip).as_quat(scalar_first=True)
    # longest invisible gap, for the report
    gaps = np.diff(vis_idx) - 1
    report["longest_gap"] = int(gaps.max()) if gaps.size else 0
    report["leading_invisible"] = int(vis_idx[0])
    report["trailing_invisible"] = int(T - 1 - vis_idx[-1])
    return pos_f.astype(np.float32), quat_f.astype(np.float32), report


def pose7_to_mat(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    T = len(pos)
    mats = np.tile(np.eye(4, dtype=np.float64), (T, 1, 1))
    mats[:, :3, :3] = R.from_quat(quat_wxyz, scalar_first=True).as_matrix()
    mats[:, :3, 3] = pos
    return mats


def convert_episode(root: Path, info: dict, ep_meta: dict, out: Path, dry_run: bool) -> dict:
    ep = int(ep_meta["episode_index"])
    chunks = int(info.get("chunks_size", 1000))
    pq_path = root / info["data_path"].format(episode_chunk=ep // chunks, episode_index=ep)
    if not pq_path.exists():
        fail(f"missing {pq_path}")
    table = pq.read_table(pq_path)

    # --- identify sequence and object ---
    meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
    sequence_id = ep_meta.get("sequence_id") or meta.get("sequence_id")
    if not sequence_id:
        fail(f"episode {ep}: no sequence_id in episodes_metadata.jsonl or parquet metadata")
    object_name = ep_meta.get("object") or ep_meta.get("object_name")
    if not object_name and "objects" in meta:
        objs = json.loads(meta["objects"])
        object_name = objs[0] if isinstance(objs, list) else objs
        if isinstance(object_name, dict):
            object_name = object_name.get("name")
    if not object_name:
        fail(f"episode {ep}: cannot determine object name")

    # --- human ---
    pose = col_to_np(table, COL_POSE).astype(np.float32)          # (T, 231)
    transl = col_to_np(table, COL_TRANSL).astype(np.float32)      # (T, 3)
    identity = col_to_np(table, COL_IDENTITY).astype(np.float32)  # (T, 45)
    scale = col_to_np(table, COL_SCALE).astype(np.float32)        # (T, 68)
    T = pose.shape[0]
    if pose.shape[1] != NJ * 3:
        fail(f"episode {ep}: human.pose width {pose.shape[1]} != {NJ*3}")
    poses = pose.reshape(T, NJ, 3)
    if not np.all(np.isfinite(poses)) or not np.all(np.isfinite(transl)):
        fail(f"episode {ep}: non-finite human params")
    id_spread = float(np.abs(identity - identity[0]).max())
    sc_spread = float(np.abs(scale - scale[0]).max())
    if id_spread > 1e-4 or sc_spread > 1e-4:
        print(f"  warning: identity/scale vary over the episode (max dev {id_spread:.2e}/{sc_spread:.2e}); using frame 0")
    boneflex = col_to_np(table, COL_BONEFLEX).astype(np.float32) if COL_BONEFLEX in table.column_names else None

    # --- object ---
    obj = col_to_np(table, COL_OBJ_POSE).astype(np.float64)       # (T, 7) x y z qw qx qy qz
    vis = col_to_np(table, COL_OBJ_VIS).astype(bool).reshape(-1)
    if obj.shape[0] != T or vis.shape[0] != T:
        fail(f"episode {ep}: object rows {obj.shape[0]}/{vis.shape[0]} != human rows {T}")
    pos, quat_wxyz, vis_report = fill_invisible(obj[:, :3], obj[:, 3:7], vis)
    qn = np.linalg.norm(quat_wxyz, axis=1)
    if np.any(np.abs(qn - 1) > 1e-3):
        fail(f"episode {ep}: non-unit quaternion after fill (min|q|={qn.min():.4f})")
    poses_npy = pose7_to_mat(pos, quat_wxyz)                      # (T, 4, 4), OpenCV world

    # --- source files ---
    mesh_src = root / info.get("mesh_path", "mesh/{object_name}/{object_name}.glb").format(object_name=object_name)
    if not mesh_src.exists():
        fail(f"episode {ep}: mesh {mesh_src} not found")
    gp_src = root / info.get("ground_plane_path", "ground_plane/{sequence_id}.json").format(sequence_id=sequence_id)

    task_idx = int(col_to_np(table, "task_index")[0]) if "task_index" in table.column_names else None
    task = next((t["task"] for t in info.get("tasks", []) if t["task_index"] == task_idx), None)

    dst = out / f"sequence_id={sequence_id}"
    summary = {
        "episode_index": ep, "sequence_id": sequence_id, "object": object_name,
        "task_index": task_idx, "task": task, "frames": T, "fps": info.get("fps", 30),
        "object_visibility": vis_report, "ground_plane": gp_src.exists(), "out": str(dst),
        "source_parquet": str(pq_path), "source_mesh": str(mesh_src),
    }
    if dry_run:
        return summary

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "reconstructed_mesh").mkdir(exist_ok=True)
    np.savez(
        dst / "soma_params.npz",
        poses=poses,                                   # (T, 77, 3) local rotvec, SOMA order
        transl=transl,                                 # (T, 3) metres
        identity_coeffs=identity[:1],                  # (1, 45)  read_soma takes [0]
        scale_params=scale[:1],                        # (1, 68)
        **({"bone_length_flexibles": boneflex} if boneflex is not None else {}),
        joint_names=np.array(SOMA_JOINTS_ORDER),
        identity_model_type=np.array("mhr"),
        unit=np.array("meters"),
        keep_root=np.array(True),
        absolute_pose=np.array(False),
        fps=np.array(float(info.get("fps", 30))),
        source=np.array(f"v2d track_2/{info.get('trajectory_variant','?')} episode {ep}"),
    )
    np.save(dst / "poses.npy", poses_npy.astype(np.float32))
    shutil.copy2(mesh_src, dst / "reconstructed_mesh" / "output_aligned.glb")
    if gp_src.exists():
        shutil.copy2(gp_src, dst / "ground_plane.json")
    (dst / "episode.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="tier folder containing meta/, data/, mesh/")
    ap.add_argument("--out", type=Path, required=True, help="output folder for sequence_id=... dirs")
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="subset of episode indices")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    info = json.loads((args.root / "meta" / "info.json").read_text())
    check_joint_names(info)
    metas = read_jsonl(args.root / "meta" / "episodes_metadata.jsonl")
    if not metas:
        metas = [{"episode_index": i} for i in range(int(info["total_episodes"]))]
    if args.episodes is not None:
        want = set(args.episodes)
        metas = [m for m in metas if int(m["episode_index"]) in want]
    print(f"{len(metas)} episodes from {args.root} -> {args.out}")

    rows = []
    for m in sorted(metas, key=lambda d: int(d["episode_index"])):
        s = convert_episode(args.root, info, m, args.out, args.dry_run)
        v = s["object_visibility"]
        print(f"  ep {s['episode_index']:02d}  {s['object']:<22s} T={s['frames']:4d}  invisible={v['invisible']:4d}"
              f"  gap={v.get('longest_gap',0):3d}  gp={'y' if s['ground_plane'] else 'n'}  {s['sequence_id']}")
        rows.append(s)
    if not args.dry_run:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "manifest.json").write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out / 'manifest.json'}")


if __name__ == "__main__":
    main()
