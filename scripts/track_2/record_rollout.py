#!/usr/bin/env python3
"""
Roll out one trained FlashCHORD policy (world 0, deterministic, same setup as evaluate_policy.py)
and record the full robot state + achieved object pose + reference object pose per control step,
as a single-row parquet that render_retarget_check.py can render offline (outside Docker).

Run INSIDE the FlashCHORD container (flash_chord, newton, jax, warp importable), from /workspace:

    python /localscratch/wangti98/v2d_scripts/record_rollout.py \
        --checkpoint outputs/tier1_350m/ep_004/policy_000350000000.safetensors \
        --out /localscratch/wangti98/rollouts/tier1_350m/ep_004.parquet \
        [--world-count 1] [--start-frame 0] [--source-root /localscratch/wangti98/track2_tier1_motion]

Then, outside the container:

    PYOPENGL_PLATFORM=egl python render_retarget_check.py --toolkit video_to_data \
        --parquet /localscratch/wangti98/rollouts/tier1_350m/ep_004.parquet --out <dir>

Parquet columns (one row): fps, robot_joint_names, robot_root_position (T,3), robot_root_wxyz (T,4),
robot_joint_positions (T,J), object_body_position (T,1,3), object_body_wxyz (T,1,4), object_mesh_paths,
reference_object_position (T,3), reference_object_wxyz (T,4), hand_sides [], hand_contact_active [],
plus checkpoint / source_parquet / world_count / step_count / completion info.

Notes
- Joint names come from the sim layout (scene.layout.scalar_joints.names), which are the URDF joint
  names, so the renderer maps them to its Pinocchio model by name.
- Object pose is world 0 of the cohort; with --world-count 1 it is exactly the submission rollout
  (the kit evaluator also uses 1 world).
- Control fps is the sim control rate (env.sim.fps), not the 30 Hz of the source motion; the reference
  object pose stored here is already resampled to the same steps.
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import warp as wp
from omegaconf import OmegaConf

from flash_chord.configuration import instantiate_typed
from flash_chord.embodiments.base import Embodiment
from flash_chord.envs.rl import RLEnv, RLEnvConfig
from flash_chord.evaluation.harness import (
    _evaluation_curriculum_stage,
    _ppo_policy_for_inference,
    _resolve_parquet,
    apply_evaluation_composition,
    checkpoint_training_algorithm,
)
from flash_chord.evaluation.rollout import reference_object_pose
from flash_chord.scene.collision import CollisionPolicy
from flash_chord.scene.setup import setup_scene
from flash_chord.training.checkpoint import read_checkpoint_metadata
from flash_chord.training.checkpoint_metadata import checkpoint_config
from flash_chord.training.environment import WarpRLEnv
from flash_chord.training.flash_sac.config import TrainingConfig as FlashSACTrainingConfig
from flash_chord.training.flash_sac.evaluation import load_policy_for_inference as load_flash_sac_policy
from flash_chord.training.flash_sac.runner import resolve_jax_device
from flash_chord.training.ppo.config import TrainingConfig as PPOTrainingConfig


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", type=Path, required=True, help="output .parquet")
    ap.add_argument("--world-count", type=int, default=1)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--source-root", default=None, help="passed through like evaluation.source_root")
    ap.add_argument("--world", type=int, default=0, help="which world of the cohort to record")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    metadata = read_checkpoint_metadata(checkpoint_path)
    config = checkpoint_config(metadata)
    algorithm = checkpoint_training_algorithm(config)
    apply_evaluation_composition(config, world_count=args.world_count, reset_mode="explicit")
    config["task"]["parquet"] = _resolve_parquet(config, Path(args.source_root) if args.source_root else None)

    cfg = OmegaConf.create(config)
    env_config = instantiate_typed(cfg.env, RLEnvConfig)
    embodiment = instantiate_typed(cfg.embodiment, Embodiment)
    collision = instantiate_typed(cfg.collision, CollisionPolicy)
    if algorithm == "flash_sac":
        training = instantiate_typed(cfg.training, FlashSACTrainingConfig)
    else:
        training = replace(instantiate_typed(cfg.training, PPOTrainingConfig), world_count=args.world_count)
    device = resolve_jax_device(training.device)

    setup = setup_scene(
        parquet=cfg.task.parquet,
        control_fps=env_config.sim.fps,
        motion_speed=cfg.task.motion_speed,
        embodiment=embodiment,
        collision=collision,
        world_count=cfg.scene.world_count,
        include_support=cfg.scene.support,
        decompose_objects=cfg.scene.decompose_objects,
        object_scale_min=cfg.scene.get("object_scale_min", 1.0),
        object_scale_max=cfg.scene.get("object_scale_max", 1.0),
        object_scale_seed=cfg.scene.get("object_scale_seed", 0),
        contact_friction=cfg.scene.get("contact_friction", 1.0),
        object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
        source_frame_playback=bool(cfg.task.get("source_frame_playback", False)),
        motion_start_frame=int(cfg.task.get("motion_start_frame", 0)),
        motion_end_frame=int(cfg.task.get("motion_end_frame", -1)),
    )
    step_count = setup.reference.num_frames - args.start_frame
    env = RLEnv(setup.scene, setup.reference, config=env_config)
    env.reset()
    env.capture_step()
    jax_env = WarpRLEnv(env, publish_diagnostics=True, publish_step_metrics=True)
    if training.curriculum is not None:
        if algorithm == "flash_sac":
            stage_index = int(metadata["curriculum_stage"])
        else:
            stage_index = training.curriculum.stage_index(max(int(metadata["iteration"]) - 1, 0))
        jax_env.set_curriculum(_evaluation_curriculum_stage(training.curriculum.stages[stage_index], "explicit"))
    if algorithm == "ppo":
        actor_state, policy_action, key = _ppo_policy_for_inference(checkpoint_path, training, jax_env)
    env.reset(frame_id=args.start_frame)
    observation = jax.device_put(jnp.asarray(env.observation.numpy().reshape(env.world_count, env.observation_dim)), device)
    if algorithm == "flash_sac":
        actor_state, policy_action, key, _ = load_flash_sac_policy(
            str(checkpoint_path), training, observation, env.action.action_dim, deterministic=True, device=device)

    # ---- layout: robot q slice of world w, object body of world w
    layout = setup.scene.layout
    model = env.model
    W = env.world_count
    q_per_world = int(model.joint_coord_count) // W
    nq = int(layout.num_joint_q)
    joint_names = list(layout.scalar_joints.names)
    q_ids = np.asarray(layout.scalar_joints.q_ids, dtype=np.int64)
    w = args.world
    body_ids_w = env.command.body_ids_w.numpy()
    bodies = env.command.layout.num_bodies
    obj_body = int(body_ids_w[w * bodies + 0])
    print(f"world_count={W} q/world={q_per_world} robot q={nq} joints={len(joint_names)} object body id={obj_body} "
          f"steps={step_count} control fps={env_config.sim.fps}", flush=True)

    root_p = np.zeros((step_count, 3)); root_q = np.zeros((step_count, 4)); qj = np.zeros((step_count, len(joint_names)))
    obj_p = np.zeros((step_count, 3)); obj_q = np.zeros((step_count, 4))
    done_step = -1
    t0 = time.time()

    def grab(step: int) -> None:
        jq = env.state_0.joint_q.numpy()[w * q_per_world: w * q_per_world + nq]
        root_p[step] = jq[:3]; root_q[step] = xyzw_to_wxyz(jq[3:7]); qj[step] = jq[q_ids]
        tf = env.state_0.body_q.numpy()[obj_body]  # (7,) p xyz, q xyzw
        obj_p[step] = tf[:3]; obj_q[step] = xyzw_to_wxyz(tf[3:7])

    for step in range(step_count):
        action, key = policy_action(actor_state, observation, key)
        observation = jax_env.step(action).observation
        wp.synchronize()
        grab(step)
        if done_step < 0 and bool(env.done.numpy()[w]):
            done_step = step
        if step % 100 == 0 or step + 1 == step_count:
            print(f"step {step + 1}/{step_count} ({(time.time() - t0) / (step + 1):.3f}s/step)", flush=True)

    ref = reference_object_pose(setup.reference, args.start_frame, step_count)[:, 0]  # (T,7) xyz + xyzw
    ref_p, ref_q = ref[:, :3], xyzw_to_wxyz(ref[:, 3:7])

    # object mesh path from the source parquet
    src = pq.read_table(cfg.task.parquet, columns=["object_mesh_paths"]).to_pydict()["object_mesh_paths"][0]
    row = {
        "fps": [float(env_config.sim.fps)],
        "sequence_id": [str(setup.parquet_path.parent.parent.name)],
        "robot_name": ["g1"],
        "robot_joint_names": [joint_names],
        "robot_root_position": [root_p.tolist()],
        "robot_root_wxyz": [root_q.tolist()],
        "robot_joint_positions": [qj.tolist()],
        "object_body_position": [obj_p[:, None, :].tolist()],
        "object_body_wxyz": [obj_q[:, None, :].tolist()],
        "object_mesh_paths": [list(src)],
        "reference_object_position": [ref_p.tolist()],
        "reference_object_wxyz": [ref_q.tolist()],
        "hand_sides": [[]],
        "hand_contact_active": [[]],
        "checkpoint": [str(checkpoint_path)],
        "source_parquet": [str(cfg.task.parquet)],
        "world_count": [int(W)],
        "world": [int(w)],
        "start_frame": [int(args.start_frame)],
        "step_count": [int(step_count)],
        "done_step": [int(done_step)],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pydict(row), args.out)
    err = np.linalg.norm(obj_p - ref_p, axis=1)
    print(f"wrote {args.out}; object position error mean {err.mean()*100:.1f} cm, final {err[-1]*100:.1f} cm, "
          f"done_step={done_step}", flush=True)


if __name__ == "__main__":
    main()
