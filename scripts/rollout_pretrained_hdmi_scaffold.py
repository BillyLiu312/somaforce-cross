#!/usr/bin/env python3
"""Run one standalone HDMI scaffold rollout and optionally write its metrics."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

SCAFFOLD_TASKS = (
    "push_door_hand",
    "push_box",
    "move_suitcase",
    "move_largebox",
)
TASK_DEFAULTS = {
    "push_door_hand": {
        "steps": 540,
        "door_friction": 0.3,
        "door_damping": 0.55,
        "object_mass": 8.0,
        "object_friction": 0.5,
        "require_progress": 0.05,
        "require_contact_fraction": 0.0,
    },
    "push_box": {
        "steps": 792,
        "door_friction": 0.3,
        "door_damping": 0.55,
        "object_mass": 8.0,
        "object_friction": 0.5,
        "require_progress": 1.0,
        "require_contact_fraction": 0.9,
    },
    "move_suitcase": {
        "steps": 472,
        "door_friction": 0.3,
        "door_damping": 0.55,
        "object_mass": 1.5,
        "object_friction": 0.55,
        "require_progress": 0.5,
        "require_contact_fraction": 0.9,
    },
    "move_largebox": {
        "steps": 199,
        "door_friction": 0.3,
        "door_damping": 0.55,
        "object_mass": 1.0,
        "object_friction": 0.55,
        "require_progress": 0.5,
        "require_contact_fraction": 0.5,
    },
}


def condition_name(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value):
        raise argparse.ArgumentTypeError(
            "condition must use lowercase letters, digits, underscores, or hyphens"
        )
    return value

parser = argparse.ArgumentParser(
    description=(
        "Standalone HDMI phase=train privileged teacher simulation baseline. "
        "This is not a deployable policy."
    )
)
parser.add_argument(
    "--task",
    choices=SCAFFOLD_TASKS,
    required=True,
)
parser.add_argument("--artifact", type=Path)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int)
parser.add_argument("--log-interval", type=int, default=25)
parser.add_argument("--delay", type=int, default=4)
parser.add_argument("--alpha", type=float, default=0.9)
parser.add_argument("--door-friction", type=float)
parser.add_argument("--door-damping", type=float)
parser.add_argument("--condition", type=condition_name, default="nominal")
parser.add_argument("--object-mass", "--box-mass", dest="object_mass", type=float)
parser.add_argument(
    "--object-friction", "--box-friction", dest="object_friction", type=float
)
parser.add_argument(
    "--object-com-offset",
    "--box-com-offset",
    dest="object_com_offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
)
parser.add_argument("--initial-object-xy", type=float, nargs=2, default=(0.0, 0.0))
parser.add_argument("--initial-object-yaw", type=float, default=0.0)
parser.add_argument("--contact-target-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0))
parser.add_argument(
    "--show-reference-object",
    "--show-reference-box",
    dest="show_reference_object",
    action="store_true",
    help="Render the translucent reference object marker in GUI mode.",
)
parser.add_argument(
    "--realtime",
    action="store_true",
    help="Pace GUI rollout at the 50 Hz policy control rate.",
)
parser.add_argument(
    "--hold-seconds",
    type=float,
    default=0.0,
    help="Keep rendering the final GUI frame for this duration.",
)
parser.add_argument(
    "--playback-rate",
    type=float,
    default=1.0,
    help="Wall-clock playback multiplier used with --realtime.",
)
parser.add_argument("--metrics-json", type=Path)
parser.add_argument("--no-metrics", action="store_true")
parser.add_argument("--video", type=Path, help="Stream one RGB frame per control step to MP4.")
parser.add_argument("--video-width", type=int, default=1280)
parser.add_argument("--video-height", type=int, default=720)
parser.add_argument("--video-fps", type=int, default=50)
parser.add_argument(
    "--require-progress",
    type=float,
    default=None,
    help="Required task-consistent progress in radians for door or meters for a rigid object.",
)
parser.add_argument(
    "--require-contact-fraction",
    type=float,
    default=None,
    help="Required fraction of reference-contact steps with both rigid-object wrists above 1 N.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs != 1:
    parser.error("the unified rollout entry supports exactly one environment")
if args.steps is not None and args.steps <= 0:
    parser.error("--steps must be positive")
if args.log_interval <= 0:
    parser.error("--log-interval must be positive")
if args.metrics_json is not None and args.no_metrics:
    parser.error("--metrics-json cannot be combined with --no-metrics")
if args.video is not None:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main() -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext

    if args.video is not None:
        import imageio.v2 as imageio
        import numpy as np
        import omni.replicator.core as rep

    from somaforce_cross.scaffold.pretrained_hdmi_isaac import (
        PretrainedHDMIIsaacRuntime,
    )

    torch.manual_seed(0)
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        render_interval=4,
        device=args.device,
    )
    sim = SimulationContext(sim_cfg)
    if args.task == "push_box":
        sim.set_camera_view(eye=(3.0, 6.0, 2.0), target=(0.0, 3.0, 0.6))
    elif args.task in ("move_suitcase", "move_largebox"):
        sim.set_camera_view(eye=(2.0, -3.0, 2.0), target=(0.0, -1.0, 0.5))
    else:
        sim.set_camera_view(eye=(3.0, 2.0, 1.0), target=(0.5, -2.5, 0.8))
    artifact = (
        args.artifact
        if args.artifact is not None
        else REPO_ROOT / f"artifacts/scaffolds/hdmi_{args.task}/v1"
    ).expanduser().resolve()
    defaults = TASK_DEFAULTS[args.task]
    object_mass = (
        args.object_mass if args.object_mass is not None else defaults["object_mass"]
    )
    object_friction = (
        args.object_friction
        if args.object_friction is not None
        else defaults["object_friction"]
    )
    door_friction = (
        args.door_friction
        if args.door_friction is not None
        else defaults["door_friction"]
    )
    door_damping = (
        args.door_damping
        if args.door_damping is not None
        else defaults["door_damping"]
    )
    steps = args.steps if args.steps is not None else defaults["steps"]
    nominal = args.condition == "nominal"
    require_progress = (
        args.require_progress
        if args.require_progress is not None
        else defaults["require_progress"] if nominal else 0.0
    )
    require_contact_fraction = (
        args.require_contact_fraction
        if args.require_contact_fraction is not None
        else defaults["require_contact_fraction"] if nominal else 0.0
    )
    runtime = PretrainedHDMIIsaacRuntime(
        sim,
        artifact,
        num_envs=args.num_envs,
        delay=args.delay,
        alpha=args.alpha,
        door_friction=door_friction,
        door_damping=door_damping,
        object_mass=object_mass,
        object_friction=object_friction,
        object_com_offset=tuple(args.object_com_offset),
        initial_object_xy=tuple(args.initial_object_xy),
        initial_object_yaw=args.initial_object_yaw,
        contact_target_offset=tuple(args.contact_target_offset),
        condition_name=args.condition,
        show_reference_object=args.show_reference_object,
    )

    class VideoRecorder:
        def __init__(self, output: Path) -> None:
            if args.video_width <= 0 or args.video_height <= 0 or args.video_fps <= 0:
                raise ValueError("video width, height, and FPS must be positive")
            if args.video_width % 2 or args.video_height % 2:
                raise ValueError("H.264 video width and height must be even")
            self.output = output.expanduser().resolve()
            self.output.parent.mkdir(parents=True, exist_ok=True)
            self.render_product = rep.create.render_product(
                "/OmniverseKit_Persp", (args.video_width, args.video_height)
            )
            self.annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
            self.annotator.attach([self.render_product])
            self.writer = imageio.get_writer(
                self.output,
                fps=args.video_fps,
                codec="libx264",
                quality=8,
                macro_block_size=2,
                ffmpeg_params=["-movflags", "+faststart"],
            )
            self.frames = 0
            for _ in range(4):
                sim.render()

        def capture(self) -> None:
            rgb = self.annotator.get_data()
            if not hasattr(rgb, "shape") or np.prod(rgb.shape) == 0:
                raise RuntimeError("RGB annotator returned an empty frame")
            frame = np.frombuffer(rgb, dtype=np.uint8).reshape(*rgb.shape)[..., :3]
            self.writer.append_data(frame)
            self.frames += 1

        def close(self) -> None:
            # The rollout entry exits with os._exit because Isaac Sim 5.1 can hang
            # while tearing down local USD scenes. Flush the encoder here and
            # let process exit release the Replicator objects as well.
            self.writer.close()

    recorder = VideoRecorder(args.video) if args.video is not None else None
    if recorder is not None:
        runtime.frame_callback = recorder.capture
    print(
        "role=privileged_simulation_baseline deployable=false "
        f"task={args.task} condition={args.condition} device={args.device} "
        f"action_dim=23 steps={steps}"
    )
    print("action_joints=" + ",".join(runtime.scaffold.action_joint_names))
    try:
        metrics = runtime.rollout(
            steps,
            log_interval=args.log_interval,
            realtime=args.realtime,
            playback_rate=args.playback_rate,
            hold_seconds=args.hold_seconds,
        )
    finally:
        if recorder is not None:
            recorder.close()
    if recorder is not None:
        if recorder.frames == 0:
            raise RuntimeError("video recording completed without frames")
        print(
            "VIDEO_OUTPUT="
            + json.dumps(
                {
                    "path": str(recorder.output),
                    "frames": recorder.frames,
                    "fps": args.video_fps,
                    "width": args.video_width,
                    "height": args.video_height,
                },
                sort_keys=True,
            )
        )
    payload = dataclasses.asdict(metrics)
    print("ROLL_OUT_METRICS=" + json.dumps(payload, sort_keys=True))
    metrics_path = None
    if not args.no_metrics:
        metrics_path = args.metrics_json or artifact / (
            f"rollout_metrics_{args.condition}.json"
        )
        metrics_path = metrics_path.expanduser().resolve()
        if metrics_path.name == "SHA256SUMS":
            raise ValueError("metrics output cannot overwrite SHA256SUMS")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = metrics_path.with_name(
            f".{metrics_path.name}.{os.getpid()}.tmp"
        )
        temporary_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary_path, metrics_path)
        print(f"METRICS_OUTPUT={metrics_path}")
    if args.task in ("move_suitcase", "move_largebox"):
        numerically_valid = metrics.nonfinite_count == 0 and metrics.zero_hook_exact
        if nominal:
            passed = (
                numerically_valid
                and metrics.steps == steps
                and metrics.stable
                and metrics.lift_off_completed
                and metrics.horizontal_displacement >= require_progress
                and metrics.set_down_completed
                and metrics.both_hand_contact_fraction
                >= require_contact_fraction
            )
        else:
            passed = numerically_valid
    elif args.task == "push_box":
        passed = (
            metrics.nonfinite_count == 0
            and metrics.stable
            and metrics.zero_hook_exact
            and metrics.max_directional_progress >= require_progress
            and metrics.contact_active_fraction >= require_contact_fraction
        )
    else:
        passed = (
            metrics.nonfinite_count == 0
            and not metrics.terminated
            and metrics.zero_hook_exact
            and metrics.max_task_progress >= require_progress
        )
    return 0 if passed else 2


if __name__ == "__main__":
    # Isaac Sim 5.1 can block indefinitely while releasing a headless local-USD
    # scene. Flush completed metrics and let the OS release the process-owned
    # simulator resources so batch validation has a reliable exit status.
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
