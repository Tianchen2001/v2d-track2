#!/usr/bin/env python3
"""
Inspect NVIDIA V2D Challenge Track 1 (LeRobot v2.1 layout).

Usage:
    python inspect_track1.py /path/to/track_1
    python inspect_track1.py /path/to/track_1 --thumbs out_thumbs   # also dump first/mid/last frames
    python inspect_track1.py /path/to/track_1 --no-video            # skip ffprobe/cv2 checks

Deps: pandas, pyarrow. Optional: opencv-python (thumbnails), ffprobe on PATH (video probing).
"""
import argparse, json, shutil, subprocess, sys
from collections import Counter
from pathlib import Path

import pandas as pd


def read_jsonl(p: Path):
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def ffprobe(path: Path):
    if shutil.which("ffprobe") is None:
        return None
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-count_frames",
           "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames,codec_name,pix_fmt,duration",
           "-of", "json", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        s = json.loads(out)["streams"][0]
        num, den = s["r_frame_rate"].split("/")
        return dict(width=s.get("width"), height=s.get("height"), fps=float(num) / float(den),
                    n_frames=int(s.get("nb_read_frames", 0)), codec=s.get("codec_name"),
                    pix_fmt=s.get("pix_fmt"), duration=float(s.get("duration", 0)))
    except Exception as e:  # noqa
        print(f"  ffprobe failed on {path.name}: {e}", file=sys.stderr)
        return None


def dump_thumbs(video: Path, out_dir: Path, tag: str):
    try:
        import cv2
    except ImportError:
        print("  opencv not installed; skipping thumbnails", file=sys.stderr)
        return
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for name, idx in [("first", 0), ("mid", n // 2), ("last", max(n - 1, 0))]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            cv2.imwrite(str(out_dir / f"{tag}_{name}_f{idx}.jpg"), small)
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="path to track_1/")
    ap.add_argument("--thumbs", type=Path, default=None, help="dir to write first/mid/last frame thumbnails")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    root: Path = args.root
    meta = root / "meta"

    # ---------- meta ----------
    info = json.loads((meta / "info.json").read_text())
    print("=" * 80)
    print("info.json")
    print("=" * 80)
    for k in ["codebase_version", "robot_type", "total_episodes", "total_frames", "total_tasks",
              "total_videos", "total_chunks", "chunks_size", "fps", "splits", "data_path", "video_path"]:
        if k in info:
            print(f"  {k:18s}: {info[k]}")
    print("  features:")
    for name, f in info.get("features", {}).items():
        extra = ""
        if f.get("dtype") == "video":
            vi = f.get("info", {})
            extra = f"  {vi.get('video.codec')} {vi.get('video.pix_fmt')} {vi.get('video.fps')}fps"
        print(f"    {name:40s} dtype={f.get('dtype'):8s} shape={f.get('shape')}{extra}")

    episodes = pd.DataFrame(read_jsonl(meta / "episodes.jsonl"))
    ep_meta = pd.DataFrame(read_jsonl(meta / "episodes_metadata.jsonl"))
    tasks = pd.DataFrame(read_jsonl(meta / "tasks.jsonl"))
    stats = read_jsonl(meta / "episodes_stats.jsonl")

    print("\n" + "=" * 80)
    print(f"tasks.jsonl  ({len(tasks)} action descriptions)")
    print("=" * 80)
    for _, r in tasks.iterrows():
        print(f"  [{r['task_index']:2d}] {r['task']}")

    # episodes.jsonl: {"episode_index","tasks":[...],"length"}
    df = episodes.merge(ep_meta, on="episode_index", how="outer").sort_values("episode_index")
    if "tasks" in df.columns:
        df["task"] = df["tasks"].apply(lambda t: t[0] if isinstance(t, list) and t else t)
    df["seconds"] = df["length"] / info.get("fps", 30)

    # ---------- parquet ----------
    data_path_tpl = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    video_path_tpl = info.get("video_path",
                              "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    chunks_size = info.get("chunks_size", 1000)

    pq_rows, pq_cols = [], None
    for ep in df["episode_index"]:
        p = root / data_path_tpl.format(episode_chunk=ep // chunks_size, episode_index=ep)
        if not p.exists():
            pq_rows.append(None); continue
        t = pd.read_parquet(p)
        if pq_cols is None:
            pq_cols = list(t.columns)
            print("\n" + "=" * 80)
            print(f"parquet schema (from {p.relative_to(root)})")
            print("=" * 80)
            print(t.dtypes.to_string())
            print("\n  head:\n" + t.head(3).to_string())
            print("  tail:\n" + t.tail(2).to_string())
        # sanity: contiguous frame_index, single task_index, timestamps == frame/fps
        fi = t["frame_index"].to_numpy()
        ok_contig = (fi == range(len(fi))).all()
        ok_ts = ((t["timestamp"] - fi / info.get("fps", 30)).abs() < 1e-3).all()
        pq_rows.append(dict(pq_frames=len(t), pq_task_index=int(t["task_index"].iloc[0]),
                            pq_contig=ok_contig, pq_ts_ok=ok_ts,
                            pq_done_last=bool(t["next.done"].iloc[-1]) if "next.done" in t else None))
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame([r or {} for r in pq_rows])], axis=1)

    # ---------- videos ----------
    if not args.no_video:
        if args.thumbs:
            args.thumbs.mkdir(parents=True, exist_ok=True)
        vrows = []
        for _, r in df.iterrows():
            ep = int(r["episode_index"])
            vk = r.get("video_key", "observation.images.exo_camera")
            p = root / video_path_tpl.format(episode_chunk=ep // chunks_size, episode_index=ep, video_key=vk)
            row = dict(video_exists=p.exists(), video_mb=round(p.stat().st_size / 1e6, 1) if p.exists() else None)
            if p.exists():
                pr = ffprobe(p)
                if pr:
                    row.update({f"vid_{k}": v for k, v in pr.items()})
                if args.thumbs:
                    dump_thumbs(p, args.thumbs, f"ep{ep:02d}_{r.get('object','')}")
            vrows.append(row)
        df = pd.concat([df, pd.DataFrame(vrows)], axis=1)
        if "vid_n_frames" in df:
            df["frames_match"] = df["vid_n_frames"] == df["pq_frames"]

    # ---------- per-episode table ----------
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 60)
    show = [c for c in ["episode_index", "object", "camera", "length", "seconds", "pq_frames",
                        "vid_n_frames", "frames_match", "vid_width", "vid_height", "vid_fps",
                        "video_mb", "task_index", "task", "sequence_id"] if c in df]
    print("\n" + "=" * 80)
    print("per-episode")
    print("=" * 80)
    print(df[show].to_string(index=False))

    # ---------- summaries ----------
    print("\n" + "=" * 80)
    print("summary")
    print("=" * 80)
    print(f"  episodes: {len(df)}   frames: {int(df['length'].sum())}   "
          f"min/mean/max len: {df['length'].min()}/{df['length'].mean():.0f}/{df['length'].max()}   "
          f"total {df['seconds'].sum()/60:.1f} min")
    print("\n  episodes per object:")
    print(df.groupby("object")["length"].agg(["count", "sum", "mean"]).to_string())
    print("\n  episodes per camera:")
    print(df["camera"].value_counts().to_string())
    print("\n  object x camera:")
    print(pd.crosstab(df["object"], df["camera"]).to_string())
    print("\n  object prompts:")
    for o, g in df.groupby("object"):
        for pr in sorted(set(g["object_prompt"])):
            print(f"    {o:18s} -> {pr!r}")

    # capture dates (from sequence_id prefix)
    if "sequence_id" in df:
        dates = df["sequence_id"].str.slice(0, 10)
        print("\n  capture dates:", dict(Counter(dates)))

    # sanity flags
    print("\n  sanity:")
    if "pq_contig" in df:
        print(f"    parquet frame_index contiguous : {df['pq_contig'].all()}")
        print(f"    timestamps == frame/fps        : {df['pq_ts_ok'].all()}")
        print(f"    parquet frames == episodes.jsonl length : {(df['pq_frames'] == df['length']).all()}")
        print(f"    task_index matches parquet     : {(df['pq_task_index'] == df['task_index']).all() if 'task_index' in df else 'n/a'}")
    if "frames_match" in df:
        bad = df.loc[~df["frames_match"].fillna(False), ["episode_index", "pq_frames", "vid_n_frames"]]
        print(f"    video frames == parquet frames : {bad.empty}")
        if not bad.empty:
            print(bad.to_string(index=False))

    # episodes_stats: show what keys exist and the RGB mean for one episode
    if stats:
        print("\n  episodes_stats.jsonl keys:", list(stats[0].keys()))
        s0 = stats[0].get("stats", {})
        for k, v in s0.items():
            if isinstance(v, dict) and "mean" in v:
                m = v["mean"]
                flat = m if not isinstance(m, list) else str(m)[:80]
                print(f"    ep0 {k}: mean={flat}")

    out = root / "track1_episodes.csv"
    df.to_csv(out, index=False)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
