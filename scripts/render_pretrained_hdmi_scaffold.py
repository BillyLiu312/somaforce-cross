#!/usr/bin/env python3
"""Render one standalone HDMI scaffold task and condition to one MP4."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio


REPO_ROOT = Path(__file__).resolve().parents[1]
SCAFFOLD_TASKS = (
    "push_door_hand",
    "push_box",
    "move_suitcase",
    "move_largebox",
)


def condition_name(value: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", value):
        raise argparse.ArgumentTypeError(
            "condition must use lowercase letters, digits, underscores, or hyphens"
        )
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--task", choices=SCAFFOLD_TASKS, required=True)
    parser.add_argument("--condition", type=condition_name, default="nominal")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--delay", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--door-friction", type=float)
    parser.add_argument("--door-damping", type=float)
    parser.add_argument("--object-mass", type=float)
    parser.add_argument("--object-friction", type=float)
    parser.add_argument("--object-com-offset", type=float, nargs=3)
    parser.add_argument("--initial-object-xy", type=float, nargs=2)
    parser.add_argument("--initial-object-yaw", type=float)
    parser.add_argument("--contact-target-offset", type=float, nargs=3)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def verify_video(path: Path, width: int, height: int) -> dict[str, object]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required to verify rendered videos")
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metadata = json.loads(result.stdout)["streams"][0]
    if metadata["codec_name"] != "h264":
        raise RuntimeError(f"unexpected video codec: {metadata['codec_name']}")
    if (metadata["width"], metadata["height"]) != (width, height):
        raise RuntimeError(f"unexpected video resolution: {metadata}")
    reader = imageio.get_reader(path)
    try:
        first_frame = reader.get_data(0)
    finally:
        reader.close()
    pixel_std = float(first_frame.std())
    if pixel_std < 1.0:
        raise RuntimeError(f"rendered video appears blank: pixel_std={pixel_std}")
    return {
        **metadata,
        "bytes": path.stat().st_size,
        "first_frame_pixel_std": pixel_std,
    }


def append_optional(command: list[str], flag: str, value: object) -> None:
    if value is None:
        return
    command.append(flag)
    if isinstance(value, (tuple, list)):
        command.extend(str(item) for item in value)
    else:
        command.append(str(value))


def main() -> int:
    args = parse_args()
    output = (
        args.output
        or REPO_ROOT
        / "videos"
        / args.task
        / f"{args.condition}.mp4"
    ).expanduser().resolve()
    command = [
        str(args.python),
        str(REPO_ROOT / "scripts/rollout_pretrained_hdmi_scaffold.py"),
        "--task",
        args.task,
        "--condition",
        args.condition,
        "--headless",
        "--device",
        args.device,
        "--delay",
        str(args.delay),
        "--alpha",
        str(args.alpha),
        "--video",
        str(output),
        "--video-width",
        str(args.width),
        "--video-height",
        str(args.height),
        "--video-fps",
        str(args.fps),
        "--require-progress",
        "0",
        "--require-contact-fraction",
        "0",
        "--no-metrics",
    ]
    append_optional(command, "--artifact", args.artifact)
    append_optional(command, "--steps", args.steps)
    append_optional(command, "--door-friction", args.door_friction)
    append_optional(command, "--door-damping", args.door_damping)
    append_optional(command, "--object-mass", args.object_mass)
    append_optional(command, "--object-friction", args.object_friction)
    append_optional(command, "--object-com-offset", args.object_com_offset)
    append_optional(command, "--initial-object-xy", args.initial_object_xy)
    append_optional(command, "--initial-object-yaw", args.initial_object_yaw)
    append_optional(command, "--contact-target-offset", args.contact_target_offset)

    print("RENDER_COMMAND=" + json.dumps(command))
    if args.dry_run:
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode != 0:
        return result.returncode
    verification = verify_video(output, args.width, args.height)
    print(
        "RENDER_OUTPUT="
        + json.dumps(
            {"task": args.task, "condition": args.condition, "video": str(output), **verification},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
