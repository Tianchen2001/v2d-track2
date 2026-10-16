#!/usr/bin/env python3
"""
Draw the official object-tracking metrics onto a policy rollout video, frame by frame.

Input: the MP4 from render_policy_videos.sh and the evaluator trajectory parquet of the same rollout
(episode_NNNNNN.parquet: achieved_* and reference_* object poses, one row per 50 Hz step).
Video frame k is matched to trajectory step k.

Top-left panel, per frame:
  pos err   |p - p_ref|                                cm
  rot err   geodesic angle between q and q_ref           deg
  kp err    mean distance of 6 keypoints at +/-5 cm (MPPE per frame)  cm
  SP-SR     running frame-mean pos <= 10 cm and rot <= 0.5 rad (28.6 deg)   OK / FAIL
  MP-SR     running frame-mean pos <  3 cm and rot <  30 deg               OK / FAIL
  final     the whole-sequence verdict (what the leaderboard sees)
Bottom strip: pos / rot error over the whole episode with the SP thresholds and a cursor.

    python overlay_metrics.py --video ep_07_overview.mp4 ep_07_side.mp4 ep_07_top.mp4 --view-names overview side top \
        --parquet episode_000007.parquet --out ep_07.mp4 --label "ep 7 iron (hand)"

Needs numpy, pyarrow, imageio-ffmpeg, Pillow (all present in the FlashCHORD container).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

SP_POS_M, SP_ROT_RAD = 0.10, 0.5
MP_POS_M, MP_ROT_DEG = 0.03, 30.0
KP = 0.05 * np.concatenate((np.eye(3), -np.eye(3)))  # MPPE keypoints, body frame

GREEN, RED, AMBER = (64, 200, 110), (235, 80, 70), (240, 180, 60)
WHITE, GREY = (240, 240, 240), (170, 170, 170)
POS_C, ROT_C = (90, 170, 255), (255, 150, 60)


def load_trajectory(path: Path):
    f = pq.ParquetFile(path)
    meta = {k.decode(): v.decode() for k, v in (f.schema_arrow.metadata or {}).items()}
    t = f.read()
    cols = set(t.column_names)
    comp = ("position_x", "position_y", "position_z", "quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w")

    def stack(prefix):
        return np.stack([t.column(f"{prefix}_{c}").to_numpy().astype(np.float64) for c in comp], 1)

    if not {f"achieved_{c}" for c in comp} <= cols or not {f"reference_{c}" for c in comp} <= cols:
        raise SystemExit(f"{path}: needs achieved_* and reference_* pose columns, has {sorted(cols)}")
    a, r = stack("achieved"), stack("reference")
    order = meta.get("flash_chord.quaternion_order") or meta.get("quaternion_layout") or "xyzw"
    if order == "wxyz":  # same convention as v2dlb.submission_parquet: columns hold w,x,y,z in that order
        a[:, 3:] = a[:, [4, 5, 6, 3]]
        r[:, 3:] = r[:, [4, 5, 6, 3]]
    if "world_index" in cols:  # keep world 0 only
        keep = t.column("world_index").to_numpy() == 0
        a, r = a[keep], r[keep]
    return a, r


def rotmat(q):
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1).reshape(-1, 3, 3)


def per_frame_errors(a, r):
    pos = np.linalg.norm(a[:, :3] - r[:, :3], axis=1)
    qa = a[:, 3:] / np.linalg.norm(a[:, 3:], axis=1, keepdims=True)
    qr = r[:, 3:] / np.linalg.norm(r[:, 3:], axis=1, keepdims=True)
    d = np.clip(np.abs(np.sum(qa * qr, axis=1)), 0.0, 1.0)
    rot = 2.0 * np.arctan2(np.sqrt(np.maximum(0.0, 1.0 - d * d)), d)
    off = np.einsum("tij,kj->tki", rotmat(qa) - rotmat(qr), KP) + (a[:, None, :3] - r[:, None, :3])
    kp = np.linalg.norm(off, axis=2).mean(1)
    return pos, rot, kp


def font(size):
    for name in ("DejaVuSansMono.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "LiberationMono-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def open_videos(paths):
    import imageio_ffmpeg
    readers, sizes, fps = [], [], None
    for pth in paths:
        rd = imageio_ffmpeg.read_frames(str(pth))
        info = next(rd)
        readers.append(rd); sizes.append(tuple(info["size"])); fps = fps or info.get("fps")
    return readers, sizes, fps or 50.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, nargs="+", required=True, help="one or more views of the same rollout (placed side by side)")
    ap.add_argument("--view-names", nargs="*", default=None, help="label per view")
    ap.add_argument("--parquet", type=Path, default=None, help="evaluator trajectory; without it the views are only stitched")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--fps", type=float, default=None, help="output fps (default: input fps)")
    ap.add_argument("--step-hz", type=float, default=50.0, help="control rate of the rollout, for the time label")
    args = ap.parse_args()

    import imageio_ffmpeg

    metrics = args.parquet is not None
    if metrics:
        a, r = load_trajectory(args.parquet)
        pos, rot, kp = per_frame_errors(a, r)
        n = len(pos)
        cnt = np.arange(1, n + 1)
        run_pos, run_rot = np.cumsum(pos) / cnt, np.cumsum(rot) / cnt
        fin_pos, fin_rot, fin_kp = pos.mean(), rot.mean(), kp.mean()
        fin_sp = fin_pos <= SP_POS_M and fin_rot <= SP_ROT_RAD
        fin_mp = fin_pos < MP_POS_M and np.degrees(fin_rot) < MP_ROT_DEG
        print(f"{args.parquet.name}: {n} steps  pos {100 * fin_pos:.1f} cm  rot {np.degrees(fin_rot):.1f} deg  "
              f"MPPE {100 * fin_kp:.1f} cm  SP {'OK' if fin_sp else 'FAIL'}  MP {'OK' if fin_mp else 'FAIL'}")

    readers, sizes, in_fps = open_videos(args.video)
    h = min(sz[1] for sz in sizes)
    widths = [int(round(sz[0] * h / sz[1])) // 2 * 2 for sz in sizes]
    w = sum(widths)
    names = args.view_names if args.view_names and len(args.view_names) == len(args.video) else [p.stem for p in args.video]
    fps = args.fps or in_fps
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(str(args.out), (w, h), fps=fps, codec="libx264", pix_fmt_in="rgb24",
                                         pix_fmt_out="yuv420p", macro_block_size=1, quality=6,
                                         ffmpeg_log_level="warning")
    writer.send(None)

    s = max(0.6, h / 720)
    f_big, f_txt = font(int(20 * s)), font(int(16 * s))
    line = int(22 * s)
    sx0, sx1 = int(12 * s), w - int(12 * s)
    sy1 = h - int(10 * s)
    sy0 = sy1 - int(90 * s)
    if metrics:
        pos_max = max(0.25, float(np.percentile(pos, 99)) * 1.1)
        rot_max = max(1.0, float(np.percentile(rot, 99)) * 1.1)
        xs = sx0 + (sx1 - sx0) * np.arange(n) / max(1, n - 1)
        yp = sy1 - (sy1 - sy0) * np.clip(pos / pos_max, 0, 1)
        yr = sy1 - (sy1 - sy0) * np.clip(rot / rot_max, 0, 1)
        thr_p = sy1 - (sy1 - sy0) * min(1, SP_POS_M / pos_max)
        thr_r = sy1 - (sy1 - sy0) * min(1, SP_ROT_RAD / rot_max)

    k = 0
    while True:
        frames = []
        try:
            for rd, sz in zip(readers, sizes):
                frames.append(Image.frombytes("RGB", sz, next(rd)))
        except StopIteration:
            break
        img = Image.new("RGB", (w, h))
        x = 0
        for fr, ww in zip(frames, widths):
            img.paste(fr.resize((ww, h)) if fr.size != (ww, h) else fr, (x, 0)); x += ww
        d = ImageDraw.Draw(img, "RGBA")
        x = 0
        for nm, ww in zip(names, widths):  # view names, top right of each view; thin separators
            tw = d.textlength(nm, font=f_txt)
            d.rectangle([x + ww - tw - int(18 * s), int(8 * s), x + ww - int(6 * s), int(8 * s) + line], fill=(0, 0, 0, 120))
            d.text((x + ww - tw - int(12 * s), int(10 * s)), nm, font=f_txt, fill=WHITE)
            if x:
                d.line([(x, 0), (x, h)], fill=(255, 255, 255, 160), width=2)
            x += ww
        px, py = int(12 * s), int(12 * s)
        if not metrics:
            d.rectangle([px - 6, py - 6, px + int(430 * s), py + line + 6], fill=(0, 0, 0, 150))
            d.text((px, py), f"{args.label}   frame {k + 1}", font=f_big, fill=WHITE)
            writer.send(np.asarray(img, dtype=np.uint8).tobytes()); k += 1
            continue
        i = min(k, n - 1)
        sp_now = run_pos[i] <= SP_POS_M and run_rot[i] <= SP_ROT_RAD
        mp_now = run_pos[i] < MP_POS_M and np.degrees(run_rot[i]) < MP_ROT_DEG
        bad_frame = pos[i] > SP_POS_M or rot[i] > SP_ROT_RAD

        d.rectangle([px - 6, py - 6, px + int(430 * s), py + 8 * line + 6], fill=(0, 0, 0, 150))
        d.text((px, py), f"{args.label}   step {i + 1}/{n}  t={i / args.step_hz:.2f}s", font=f_big, fill=WHITE)
        rows = [
            (f"pos err {100 * pos[i]:6.1f} cm   mean {100 * run_pos[i]:5.1f}", RED if pos[i] > SP_POS_M else WHITE),
            (f"rot err {np.degrees(rot[i]):6.1f} deg  mean {np.degrees(run_rot[i]):5.1f}",
             RED if rot[i] > SP_ROT_RAD else WHITE),
            (f"kp err  {100 * kp[i]:6.1f} cm   MPPE {100 * kp[:i + 1].mean():5.1f}", WHITE),
            (f"SP-SR (mean <=10cm, <=28.6deg): {'OK' if sp_now else 'FAIL'}", GREEN if sp_now else RED),
            (f"MP-SR (mean <3cm, <30deg):      {'OK' if mp_now else 'FAIL'}", GREEN if mp_now else RED),
            (f"final: SP {'OK' if fin_sp else 'FAIL'}  MP {'OK' if fin_mp else 'FAIL'}  "
             f"pos {100 * fin_pos:.1f}cm rot {np.degrees(fin_rot):.0f}deg", GREEN if fin_sp else AMBER),
        ]
        for j, (txt, col) in enumerate(rows):
            d.text((px, py + (j + 1) * line + int(4 * s)), txt, font=f_txt, fill=col)
        if bad_frame:
            d.rectangle([0, 0, w - 1, h - 1], outline=(235, 80, 70, 200), width=max(2, int(4 * s)))

        d.rectangle([sx0 - 4, sy0 - int(18 * s), sx1 + 4, sy1 + 4], fill=(0, 0, 0, 140))
        d.text((sx0, sy0 - int(17 * s)), f"pos (0-{100 * pos_max:.0f} cm)", font=f_txt, fill=POS_C)
        d.text((sx0 + int(200 * s), sy0 - int(17 * s)), f"rot (0-{np.degrees(rot_max):.0f} deg)", font=f_txt,
               fill=ROT_C)
        d.line([(sx0, thr_p), (sx1, thr_p)], fill=POS_C + (110,), width=1)
        d.line([(sx0, thr_r), (sx1, thr_r)], fill=ROT_C + (110,), width=1)
        d.line(list(zip(xs.tolist(), yp.tolist())), fill=POS_C, width=max(1, int(2 * s)))
        d.line(list(zip(xs.tolist(), yr.tolist())), fill=ROT_C, width=max(1, int(2 * s)))
        d.line([(xs[i], sy0), (xs[i], sy1)], fill=WHITE, width=max(1, int(2 * s)))

        writer.send(np.asarray(img, dtype=np.uint8).tobytes())
        k += 1
    writer.close()
    if metrics and k != n:
        print(f"note: video has {k} frames, trajectory {n} steps (frame k shown against step min(k, {n - 1}))")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()