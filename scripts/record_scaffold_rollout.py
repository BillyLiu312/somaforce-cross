"""Record a short Sonic-based scaffold rollout.

The rollout applies the current nominal scaffold action ``a_nom`` directly to
the Sonic manager env. It records numeric traces by default and can optionally
save RGB frames/video when Isaac cameras are enabled.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_sonic_scaffold_env import (  # noqa: E402
    attach_default_door_articulation,
    write_default_box_usd,
    write_default_door_urdf,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sonic-config-dir",
        type=Path,
        default=Path("../GR00T-WholeBodyControl/gear_sonic/config"),
    )
    parser.add_argument("--task", choices=("push_pull_door", "push_pull_box"), default="push_pull_door")
    parser.add_argument("--interaction-mode", choices=("push", "pull"), default="push")
    parser.add_argument("--motion-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/somaforce_scaffold_record"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--motion-frames", type=int, default=101)
    parser.add_argument("--motion-duration-s", type=float, default=2.0)
    parser.add_argument("--motion-fps", type=int, default=50)
    parser.add_argument("--object-usd-path")
    parser.add_argument("--generate-default-box-usd", action="store_true")
    parser.add_argument("--default-box-usd-path", type=Path, default=Path("/tmp/somaforce_default_box.usd"))
    parser.add_argument("--default-box-size-m", type=float, default=0.25)
    parser.add_argument("--generate-default-door-urdf", action="store_true")
    parser.add_argument("--default-door-urdf-path", type=Path, default=Path("/tmp/somaforce_default_hinged_door.urdf"))
    parser.add_argument("--object-position", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--object-mass", type=float, default=2.0)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--render-every", type=int, default=2)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--camera-eye", type=float, nargs=3, default=(2.4, -2.2, 1.6))
    parser.add_argument("--camera-target", type=float, nargs=3, default=(0.55, -0.15, 0.85))
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.render_every < 1:
        raise ValueError("--render-every must be positive")

    sonic_config_dir = args.sonic_config_dir.resolve()
    sonic_root = sonic_config_dir.parents[1]
    os.chdir(sonic_root)
    print(f"sonic_root={sonic_root}", flush=True)

    # Isaac AppLauncher parses sys.argv. Keep script-specific flags out of Kit.
    sys.argv = [sys.argv[0]]

    from somaforce_cross.scaffold import (  # noqa: PLC0415
        ScaffoldTask,
        make_g1_box_sonic_binding,
        make_g1_door_sonic_binding,
        make_sonic_manager_overrides,
        write_single_task_sonic_motion_file,
        write_single_task_sonic_object_motion_file,
    )

    task = ScaffoldTask(args.task)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    if args.record_video:
        frames_dir.mkdir(parents=True, exist_ok=True)

    motion_file = args.motion_file or output_dir / f"{task.value}_{args.interaction_mode}_motion.pkl"
    write_single_task_sonic_motion_file(
        motion_file,
        task=task,
        mode=args.interaction_mode,
        num_frames=args.motion_frames,
        duration_s=args.motion_duration_s,
        fps=args.motion_fps,
    )

    object_usd_path = args.object_usd_path
    if args.generate_default_box_usd:
        if task != ScaffoldTask.PUSH_PULL_BOX:
            raise ValueError("--generate-default-box-usd is only valid for push_pull_box")
        object_usd_path = str(args.default_box_usd_path)
    if args.generate_default_door_urdf:
        if task != ScaffoldTask.PUSH_PULL_DOOR:
            raise ValueError("--generate-default-door-urdf is only valid for push_pull_door")
        object_usd_path = str(args.default_door_urdf_path)

    object_position = tuple(args.object_position) if args.object_position is not None else (0.72, 0.0, 0.75)
    if task == ScaffoldTask.PUSH_PULL_BOX:
        binding = make_g1_box_sonic_binding(
            motion_file,
            args.interaction_mode,
            box_usd_path=object_usd_path,
            object_position=object_position,
            object_mass=args.object_mass,
        )
    else:
        binding = make_g1_door_sonic_binding(
            motion_file,
            args.interaction_mode,
            door_asset_path=object_usd_path,
        )

    if binding.object_motion_file is not None:
        write_single_task_sonic_object_motion_file(
            binding.object_motion_file,
            task=task,
            mode=args.interaction_mode,
            num_frames=args.motion_frames,
            fps=args.motion_fps,
        )

    print(f"output_dir={output_dir}", flush=True)
    print(f"motion_file={motion_file}", flush=True)
    print(f"object_motion_file={binding.object_motion_file}", flush=True)
    print(f"record_video={args.record_video}", flush=True)

    from isaaclab.app import AppLauncher  # noqa: PLC0415

    launcher = AppLauncher(headless=args.headless, enable_cameras=args.record_video)
    app = launcher.app
    print("app_started", flush=True)

    env = None
    frames: list[np.ndarray] = []
    try:
        if args.generate_default_box_usd:
            write_default_box_usd(args.default_box_usd_path, args.default_box_size_m, args.object_mass)
            print(f"default_box_usd_written={args.default_box_usd_path}", flush=True)
        if args.generate_default_door_urdf:
            write_default_door_urdf(args.default_door_urdf_path)
            print(f"default_door_urdf_written={args.default_door_urdf_path}", flush=True)

        from hydra import compose, initialize_config_dir  # noqa: PLC0415

        from gear_sonic.envs.manager_env.mdp.observations import residual_joint_pos_action  # noqa: PLC0415
        from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper  # noqa: PLC0415
        from gear_sonic.trl.utils.common import custom_instantiate  # noqa: PLC0415
        from isaaclab.envs import ManagerBasedRLEnv  # noqa: PLC0415
        from somaforce_cross.scaffold import SonicScaffoldAdapter  # noqa: PLC0415

        overrides = make_sonic_manager_overrides(
            binding,
            num_envs=args.num_envs,
            experiment_dir=output_dir / "sonic_exp",
        )
        with initialize_config_dir(version_base="1.1", config_dir=str(sonic_config_dir)):
            cfg = compose(config_name="base", overrides=overrides)
        env_cfg = custom_instantiate(cfg.manager_env, _resolve=True)
        env_cfg.seed = 0
        env_cfg.sim.device = args.device
        env_cfg.config["headless"] = args.headless
        if args.generate_default_door_urdf:
            attach_default_door_articulation(env_cfg, args.default_door_urdf_path)
            print("door_articulation_attached=True", flush=True)

        env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
        wrapper = ManagerEnvWrapper(env, env_cfg.config)
        wrapper.reset(flatten_dict_obs=False)
        record_camera = _make_record_camera(args, env) if args.record_video else None

        traces: dict[str, list[np.ndarray]] = {
            "a_nom": [],
            "motion_joint_pos": [],
            "robot_joint_pos": [],
            "robot_root_pos_w": [],
            "robot_root_quat_w": [],
            "door_joint_pos": [],
            "object_root_pos_w": [],
        }
        step_rows: list[dict[str, Any]] = []

        for step in range(args.steps):
            motion_cmd = env.command_manager.get_term("motion")
            a_nom = residual_joint_pos_action(env, command_name="motion")
            scaffold = SonicScaffoldAdapter(env, task=task).get_output()
            if not torch.allclose(a_nom, scaffold.a_nom):
                raise RuntimeError("Sonic residual action and scaffold adapter a_nom diverged")

            _append(traces["a_nom"], a_nom)
            _append(traces["motion_joint_pos"], motion_cmd.joint_pos)
            _record_scene_state(env, traces)

            _, reward, terminated, truncated, _ = env.step(a_nom)
            done = bool(torch.any(terminated | truncated).detach().cpu().item())
            reward_mean = float(torch.mean(reward).detach().cpu().item())
            step_rows.append({"step": step, "reward_mean": reward_mean, "done": done})

            if record_camera is not None and step % args.render_every == 0:
                frame = _capture_camera_frame(record_camera, env)
                if frame is not None:
                    import imageio.v2 as imageio  # noqa: PLC0415

                    frames.append(frame)
                    imageio.imwrite(frames_dir / f"frame_{len(frames) - 1:04d}.png", frame)

        trace_path = output_dir / "rollout_traces.npz"
        np.savez_compressed(
            trace_path,
            **{name: _stack(values) for name, values in traces.items() if values},
        )
        summary = _summary(args, binding, output_dir, motion_file, trace_path, step_rows, frames)
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        video_path = None
        if args.record_video and frames:
            import imageio.v2 as imageio  # noqa: PLC0415

            video_path = output_dir / "rollout.mp4"
            try:
                imageio.mimsave(video_path, frames, fps=max(1, args.motion_fps // args.render_every))
            except Exception as exc:  # noqa: BLE001
                print(f"video_write_failed={exc}", flush=True)
                video_path = None

        print(f"record_ok=1", flush=True)
        print(f"summary_file={summary_path}", flush=True)
        print(f"trace_file={trace_path}", flush=True)
        print(f"frames_dir={frames_dir if frames else None}", flush=True)
        print(f"video_file={video_path}", flush=True)
        print(f"num_steps={args.steps}", flush=True)
        print(f"num_frames={len(frames)}", flush=True)
        print(f"a_nom_shape={summary['a_nom_shape']}", flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()
        print("app_closed", flush=True)


def _record_scene_state(env: object, traces: dict[str, list[np.ndarray]]) -> None:
    robot = env.scene.articulations.get("robot")
    if robot is not None:
        _append(traces["robot_joint_pos"], robot.data.joint_pos)
        _append(traces["robot_root_pos_w"], robot.data.root_pos_w)
        _append(traces["robot_root_quat_w"], robot.data.root_quat_w)
    door = env.scene.articulations.get("door")
    if door is not None:
        _append(traces["door_joint_pos"], door.data.joint_pos)
    obj = env.scene.rigid_objects.get("object")
    if obj is not None:
        _append(traces["object_root_pos_w"], obj.data.root_pos_w)


def _make_record_camera(args: argparse.Namespace, env: object) -> object:
    import isaaclab.sim as sim_utils  # noqa: PLC0415
    import isaacsim.core.utils.prims as prim_utils  # noqa: PLC0415
    from isaaclab.sensors.camera import Camera, CameraCfg  # noqa: PLC0415

    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.78, 0.78, 0.78))
    light_cfg.func("/World/SomaForceRecordLight", light_cfg)

    prim_utils.create_prim("/World/SomaForceRecordCameraOrigin", "Xform")
    camera_cfg = CameraCfg(
        prim_path="/World/SomaForceRecordCameraOrigin/Camera",
        update_period=0,
        height=args.height,
        width=args.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 100.0),
        ),
    )
    camera = Camera(cfg=camera_cfg)
    if not camera.is_initialized:
        camera._initialize_impl()
        camera._is_initialized = True
    camera.set_world_poses_from_view(
        torch.tensor([args.camera_eye], device=env.device, dtype=torch.float32),
        torch.tensor([args.camera_target], device=env.device, dtype=torch.float32),
    )
    env.sim.render()
    camera.update(dt=env.step_dt)
    return camera


def _capture_camera_frame(camera: object, env: object) -> np.ndarray | None:
    env.sim.render()
    camera.update(dt=env.step_dt)
    rgb = camera.data.output.get("rgb")
    if rgb is None:
        return None
    frame = rgb[0].detach().cpu().numpy()
    if frame.shape[-1] > 3:
        frame = frame[..., :3]
    return np.asarray(frame, dtype=np.uint8)


def _append(values: list[np.ndarray], tensor: torch.Tensor) -> None:
    values.append(tensor.detach().cpu().numpy().copy())


def _stack(values: list[np.ndarray]) -> np.ndarray:
    return np.stack(values, axis=0)


def _summary(
    args: argparse.Namespace,
    binding: object,
    output_dir: Path,
    motion_file: Path,
    trace_path: Path,
    step_rows: list[dict[str, Any]],
    frames: list[np.ndarray],
) -> dict[str, Any]:
    traces = np.load(trace_path)
    summary: dict[str, Any] = {
        "task": args.task,
        "interaction_mode": args.interaction_mode,
        "steps": args.steps,
        "num_envs": args.num_envs,
        "motion_file": str(motion_file),
        "output_dir": str(output_dir),
        "trace_file": str(trace_path),
        "record_video": args.record_video,
        "num_frames": len(frames),
        "requires_articulation_scene": bool(binding.requires_articulation_scene),
        "object_motion_file": None if binding.object_motion_file is None else str(binding.object_motion_file),
        "a_nom_shape": list(traces["a_nom"].shape),
        "a_nom_first_head": traces["a_nom"][0, 0, :5].tolist(),
        "a_nom_last_head": traces["a_nom"][-1, 0, :5].tolist(),
        "reward_mean_first": step_rows[0]["reward_mean"],
        "reward_mean_last": step_rows[-1]["reward_mean"],
        "done_count": sum(1 for row in step_rows if row["done"]),
    }
    if "door_joint_pos" in traces.files:
        summary["door_joint_pos_shape"] = list(traces["door_joint_pos"].shape)
        summary["door_joint_pos_first"] = traces["door_joint_pos"][0, 0].tolist()
        summary["door_joint_pos_last"] = traces["door_joint_pos"][-1, 0].tolist()
    if "object_root_pos_w" in traces.files:
        summary["object_root_pos_w_shape"] = list(traces["object_root_pos_w"].shape)
        summary["object_root_pos_w_first"] = traces["object_root_pos_w"][0, 0].tolist()
        summary["object_root_pos_w_last"] = traces["object_root_pos_w"][-1, 0].tolist()
    return summary


if __name__ == "__main__":
    main()
