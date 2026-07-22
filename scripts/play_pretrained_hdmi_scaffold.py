#!/usr/bin/env python3
"""Play the frozen HDMI privileged-teacher scaffold in standalone Isaac Lab."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser(
    description=(
        "Standalone HDMI phase=train privileged teacher simulation baseline. "
        "This is not a deployable policy."
    )
)
parser.add_argument(
    "--task",
    choices=("push_door_hand", "push_box", "move_suitcase"),
    default="push_door_hand",
)
parser.add_argument("--artifact", type=Path)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int)
parser.add_argument("--log-interval", type=int, default=25)
parser.add_argument("--delay", type=int, default=4)
parser.add_argument("--alpha", type=float, default=0.9)
parser.add_argument("--door-friction", type=float, default=0.3)
parser.add_argument("--door-damping", type=float, default=0.55)
parser.add_argument(
    "--case",
    choices=(
        "nominal",
        "light",
        "heavy",
        "stress",
        "high_mass",
        "high_friction",
        "custom",
    ),
    default="nominal",
)
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
parser.add_argument(
    "--require-progress",
    type=float,
    default=0.05,
    help="Required task-consistent progress in radians for door or meters for a rigid object.",
)
parser.add_argument(
    "--require-contact-fraction",
    type=float,
    default=0.0,
    help="Required fraction of reference-contact steps with both rigid-object wrists above 1 N.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


def main() -> int:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationContext

    from somaforce_cross.scaffold.pretrained_hdmi_isaac import (
        PretrainedHDMIIsaacRuntime,
    )

    if args.num_envs != 1:
        raise ValueError("this validated play entry currently supports --num-envs=1")
    torch.manual_seed(0)
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        render_interval=4,
        device=args.device,
    )
    sim = SimulationContext(sim_cfg)
    if args.task == "push_box":
        sim.set_camera_view(eye=(3.0, 6.0, 2.0), target=(0.0, 3.0, 0.6))
    elif args.task == "move_suitcase":
        sim.set_camera_view(eye=(2.0, -3.0, 2.0), target=(0.0, -1.0, 0.5))
    else:
        sim.set_camera_view(eye=(3.0, 2.0, 1.0), target=(0.5, -2.5, 0.8))
    artifact = (
        args.artifact
        if args.artifact is not None
        else REPO_ROOT / f"artifacts/scaffolds/hdmi_{args.task}/v1"
    )
    if args.task == "move_suitcase":
        case_defaults = {
            "nominal": (1.5, 0.55),
            "light": (0.5, 0.55),
            "heavy": (3.0, 0.55),
            "stress": (5.5, 0.55),
            "custom": (1.5, 0.55),
        }
    else:
        case_defaults = {
            "nominal": (8.0, 0.5),
            "high_mass": (10.0, 0.5),
            "high_friction": (8.0, 1.2),
            "custom": (8.0, 0.5),
        }
    if args.case not in case_defaults:
        raise ValueError(f"case {args.case!r} is not valid for task {args.task!r}")
    default_mass, default_friction = case_defaults[args.case]
    object_mass = args.object_mass if args.object_mass is not None else default_mass
    object_friction = (
        args.object_friction
        if args.object_friction is not None
        else default_friction
    )
    default_steps = {
        "push_door_hand": 540,
        "push_box": 792,
        "move_suitcase": 472,
    }
    steps = args.steps if args.steps is not None else default_steps[args.task]
    runtime = PretrainedHDMIIsaacRuntime(
        sim,
        artifact,
        num_envs=args.num_envs,
        delay=args.delay,
        alpha=args.alpha,
        door_friction=args.door_friction,
        door_damping=args.door_damping,
        object_mass=object_mass,
        object_friction=object_friction,
        object_com_offset=tuple(args.object_com_offset),
        initial_object_xy=tuple(args.initial_object_xy),
        initial_object_yaw=args.initial_object_yaw,
        contact_target_offset=tuple(args.contact_target_offset),
        case_name=args.case,
        show_reference_object=args.show_reference_object,
    )
    print(
        "role=privileged_simulation_baseline deployable=false "
        f"task={args.task} case={args.case} device={args.device} "
        f"action_dim=23 steps={steps}"
    )
    print("action_joints=" + ",".join(runtime.scaffold.action_joint_names))
    metrics = runtime.rollout(
        steps,
        log_interval=args.log_interval,
        realtime=args.realtime,
        playback_rate=args.playback_rate,
        hold_seconds=args.hold_seconds,
    )
    payload = dataclasses.asdict(metrics)
    print("ROLL_OUT_METRICS=" + json.dumps(payload, sort_keys=True))
    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        if args.task == "move_suitcase":
            aggregate = {
                "task": "move_suitcase",
                "role": "privileged_simulation_baseline_not_deployable",
                "cases": {},
            }
            if args.metrics_json.is_file():
                existing = json.loads(args.metrics_json.read_text(encoding="utf-8"))
                if existing.get("task") == "move_suitcase" and isinstance(
                    existing.get("cases"), dict
                ):
                    aggregate = existing
            aggregate["cases"][args.case] = payload
            args.metrics_json.write_text(
                json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
            )
        else:
            args.metrics_json.write_text(
                json.dumps(payload, indent=2) + "\n", encoding="utf-8"
            )
    if args.task == "move_suitcase":
        numerically_valid = metrics.nonfinite_count == 0 and metrics.zero_hook_exact
        if args.case == "nominal":
            passed = (
                numerically_valid
                and metrics.steps == steps
                and metrics.stable
                and metrics.lift_off_completed
                and metrics.horizontal_displacement >= args.require_progress
                and metrics.set_down_completed
                and metrics.both_hand_contact_fraction
                >= args.require_contact_fraction
            )
        else:
            passed = numerically_valid
    elif args.task == "push_box":
        passed = (
            metrics.nonfinite_count == 0
            and metrics.stable
            and metrics.zero_hook_exact
            and metrics.max_directional_progress >= args.require_progress
            and metrics.contact_active_fraction >= args.require_contact_fraction
        )
    else:
        passed = (
            metrics.nonfinite_count == 0
            and not metrics.terminated
            and metrics.zero_hook_exact
            and metrics.max_task_progress >= args.require_progress
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
