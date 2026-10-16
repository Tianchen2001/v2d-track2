#!/usr/bin/env python3
"""
One policy rollout, several cameras: wraps FlashCHORD scripts/view_policy.py so every control step is simulated once
and then drawn from each camera in turn (run INSIDE the container, from the flash_chord checkout).

Separate runs per camera do not show the same motion: GPU contact solving is not bitwise reproducible, so two runs
of the same checkpoint drift apart. Here all views come from the same run, and the object pose of that run is saved,
so the metric overlay can be computed from exactly what is shown.

Called by render_policy_videos.sh; all Hydra overrides are passed through to view_policy.py. Environment:
  MV_VIEWS    JSON list [{"name": ..., "position": [x, y, z], "target": [x, y, z], "path": "<view>.mp4"}, ...]
  MV_TRAJ     output parquet: achieved object pose of this rollout per step, plus the reference pose copied from
  MV_REF      the evaluator trajectory of the same checkpoint (reference_* columns), same schema as the evaluator
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def run_multiview(cls, config, viewer_config, device=None):
    import newton.viewer
    import warp as wp

    from flash_chord.visualization.viewer import _open_video_writer, configure_camera, configure_render_style

    if device is not None:
        wp.set_device(device)
    viewer = newton.viewer.ViewerGL(width=viewer_config.width, height=viewer_config.height, headless=True)
    app = cls(viewer, SimpleNamespace(test=False, config=config))
    configure_render_style(viewer, viewer_config, getattr(app, "scene", None))
    views = json.loads(os.environ["MV_VIEWS"])
    n = int(viewer_config.num_frames)
    try:
        obj_id = int(app.env.command.layout.body_ids[0])
    except AttributeError as error:
        raise SystemExit(f"cannot find the object body in this FlashCHORD version: {error}")
    writers = [None] * len(views)
    buffers = [None] * len(views)
    achieved = np.zeros((n, 7))
    try:
        for k in range(n):
            app.step()
            achieved[k] = np.asarray(app.state.body_q.numpy()[obj_id], dtype=np.float64)  # p xyz, q xyzw
            for i, v in enumerate(views):
                configure_camera(viewer, SimpleNamespace(camera_position=tuple(v["position"]),
                                                         camera_target=tuple(v["target"]), backend="mp4"))
                app.render()
                buffers[i] = viewer.get_frame(target_image=buffers[i])
                frame = buffers[i].numpy()
                if writers[i] is None:
                    h, w, _ = frame.shape
                    writers[i] = _open_video_writer(Path(v["path"]), w, h, float(viewer_config.video_fps))
                writers[i].send(np.ascontiguousarray(frame))
            if k % 100 == 0:
                print(f"  step {k}/{n}", flush=True)
    finally:
        for w in writers:
            if w is not None:
                w.close()
        viewer.close()
    for v in views:
        print(f"Video: {v['path']} ({n} frames)")
    write_trajectory(achieved, os.environ.get("MV_REF"), os.environ.get("MV_TRAJ"))


def write_trajectory(achieved: np.ndarray, ref_path: str | None, out_path: str | None) -> None:
    if not out_path:
        return
    import pyarrow as pa
    import pyarrow.parquet as pq

    comp = ("position_x", "position_y", "position_z", "quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w")
    cols = {f"achieved_{c}": achieved[:, j].astype(np.float32) for j, c in enumerate(comp)}
    n = len(achieved)
    if ref_path and Path(ref_path).exists():
        t = pq.read_table(ref_path)
        meta = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
        if "world_index" in t.column_names:
            t = t.filter(np.asarray(t.column("world_index").to_numpy() == 0))
        ref = np.stack([t.column(f"reference_{c}").to_numpy() for c in comp], 1).astype(np.float64)
        if meta.get("flash_chord.quaternion_order") == "wxyz":  # evaluator columns hold w,x,y,z: store xyzw here
            ref[:, 3:] = ref[:, [4, 5, 6, 3]]
        m = min(n, len(ref))
        if m < n:
            print(f"  note: reference has {len(ref)} steps, rollout {n}; trajectory cut to {m}")
        cols = {k: v[:m] for k, v in cols.items()}
        for j, c in enumerate(comp):
            cols[f"reference_{c}"] = ref[:m, j].astype(np.float32)
    table = pa.table(cols).replace_schema_metadata({"flash_chord.quaternion_order": "xyzw", "source": "multiview_policy.py"})
    pq.write_table(table, out_path)
    print(f"Trajectory: {out_path}")


def main() -> None:
    import runpy

    from flash_chord.visualization.viewer import ViewerApp

    script = Path.cwd() / "scripts" / "view_policy.py"
    if not script.is_file():
        raise SystemExit(f"run from the flash_chord checkout (no {script})")
    # view_policy's viewer apps inherit launch_configured from ViewerApp: patch it there, then run the script as
    # __main__ so Hydra resolves its config path exactly as in a normal `python scripts/view_policy.py ...` call.
    ViewerApp.launch_configured = classmethod(run_multiview)
    sys.argv[0] = str(script)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()