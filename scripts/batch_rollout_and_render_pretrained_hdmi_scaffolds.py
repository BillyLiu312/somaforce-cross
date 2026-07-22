#!/usr/bin/env python3
"""Run a validated task/condition matrix through the single-case rollout and render CLIs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = REPO_ROOT / "configs/pretrained_hdmi_rollout_cases.json"
TASK_CATEGORIES = {
    "push_door_hand": "push_pull_door",
    "push_box": "push_box",
    "move_suitcase": "heavy_payload",
    "move_largebox": "heavy_payload",
}
ARGUMENT_SPECS = {
    "steps": ("--steps", 1),
    "delay": ("--delay", 1),
    "alpha": ("--alpha", 1),
    "door_friction": ("--door-friction", 1),
    "door_damping": ("--door-damping", 1),
    "object_mass": ("--object-mass", 1),
    "object_friction": ("--object-friction", 1),
    "object_com_offset": ("--object-com-offset", 3),
    "initial_object_xy": ("--initial-object-xy", 2),
    "initial_object_yaw": ("--initial-object-yaw", 1),
    "contact_target_offset": ("--contact-target-offset", 3),
}
SLUG_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]*")


@dataclass(frozen=True)
class CaseSpec:
    task: str
    category: str
    condition: str
    arguments: dict[str, int | float | tuple[int | float, ...]]


@dataclass(frozen=True)
class CaseCommands:
    rollout: list[str]
    render: list[str]
    metrics: Path
    video: Path


def _require_slug(value: Any, field: str) -> str:
    if not isinstance(value, str) or SLUG_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase slug: {value!r}")
    return value


def _require_number(value: Any, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    return value


def _validate_arguments(raw: Any, location: str) -> dict[str, int | float | tuple[int | float, ...]]:
    if not isinstance(raw, dict):
        raise ValueError(f"{location}.args must be an object")
    unknown = set(raw) - set(ARGUMENT_SPECS)
    if unknown:
        raise ValueError(f"{location}.args has unsupported keys: {sorted(unknown)}")
    arguments: dict[str, int | float | tuple[int | float, ...]] = {}
    for name, value in raw.items():
        _, arity = ARGUMENT_SPECS[name]
        if arity == 1:
            arguments[name] = _require_number(value, f"{location}.args.{name}")
            continue
        if not isinstance(value, list) or len(value) != arity:
            raise ValueError(f"{location}.args.{name} must contain {arity} numbers")
        arguments[name] = tuple(
            _require_number(item, f"{location}.args.{name}") for item in value
        )
    if "steps" in arguments and arguments["steps"] <= 0:
        raise ValueError(f"{location}.args.steps must be positive")
    return arguments


def load_cases(path: Path) -> list[CaseSpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("matrix schema_version must be 1")
    if set(data) != {"schema_version", "tasks"} or not isinstance(data["tasks"], list):
        raise ValueError("matrix must contain only schema_version and a tasks list")

    cases: list[CaseSpec] = []
    seen_tasks: set[str] = set()
    seen_cases: set[tuple[str, str]] = set()
    for task_index, task_entry in enumerate(data["tasks"]):
        location = f"tasks[{task_index}]"
        if not isinstance(task_entry, dict) or set(task_entry) != {"task", "category", "cases"}:
            raise ValueError(f"{location} must contain task, category, and cases")
        task = _require_slug(task_entry["task"], f"{location}.task")
        category = _require_slug(task_entry["category"], f"{location}.category")
        if task not in TASK_CATEGORIES:
            raise ValueError(f"unsupported task: {task}")
        if category != TASK_CATEGORIES[task]:
            raise ValueError(
                f"category for {task} must be {TASK_CATEGORIES[task]}, got {category}"
            )
        if task in seen_tasks:
            raise ValueError(f"task is duplicated in matrix: {task}")
        seen_tasks.add(task)
        if not isinstance(task_entry["cases"], list) or not task_entry["cases"]:
            raise ValueError(f"{location}.cases must be a non-empty list")
        for case_index, case_entry in enumerate(task_entry["cases"]):
            case_location = f"{location}.cases[{case_index}]"
            if not isinstance(case_entry, dict) or set(case_entry) != {"condition", "args"}:
                raise ValueError(f"{case_location} must contain condition and args")
            condition = _require_slug(case_entry["condition"], f"{case_location}.condition")
            identity = (task, condition)
            if identity in seen_cases:
                raise ValueError(f"condition is duplicated for {task}: {condition}")
            seen_cases.add(identity)
            cases.append(
                CaseSpec(
                    task=task,
                    category=category,
                    condition=condition,
                    arguments=_validate_arguments(case_entry["args"], case_location),
                )
            )
    return cases


def _case_cli_arguments(case: CaseSpec) -> list[str]:
    command: list[str] = []
    for name, (flag, _) in ARGUMENT_SPECS.items():
        if name not in case.arguments:
            continue
        command.append(flag)
        value = case.arguments[name]
        if isinstance(value, tuple):
            command.extend(str(item) for item in value)
        else:
            command.append(str(value))
    return command


def build_case_commands(
    case: CaseSpec,
    *,
    python: Path,
    device: str,
    videos_root: Path,
    width: int,
    height: int,
    fps: int,
) -> CaseCommands:
    artifact = REPO_ROOT / f"artifacts/scaffolds/hdmi_{case.task}/v1"
    metrics = artifact / f"rollout_metrics_{case.condition}.json"
    video = videos_root / case.task / f"{case.condition}.mp4"
    case_arguments = _case_cli_arguments(case)
    rollout = [
        str(python),
        str(REPO_ROOT / "scripts/rollout_pretrained_hdmi_scaffold.py"),
        "--task",
        case.task,
        "--condition",
        case.condition,
        "--headless",
        "--device",
        device,
        "--metrics-json",
        str(metrics),
        *case_arguments,
    ]
    render = [
        str(python),
        str(REPO_ROOT / "scripts/render_pretrained_hdmi_scaffold.py"),
        "--python",
        str(python),
        "--task",
        case.task,
        "--condition",
        case.condition,
        "--device",
        device,
        "--output",
        str(video),
        "--width",
        str(width),
        "--height",
        str(height),
        "--fps",
        str(fps),
        *case_arguments,
    ]
    return CaseCommands(rollout=rollout, render=render, metrics=metrics, video=video)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--videos-root", type=Path, default=REPO_ROOT / "videos")
    parser.add_argument("--task", action="append", choices=tuple(TASK_CATEGORIES))
    parser.add_argument("--condition", action="append")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--rollout-only", action="store_true")
    mode.add_argument("--render-only", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        raise ValueError("video width, height, and FPS must be positive")
    if args.width % 2 or args.height % 2:
        raise ValueError("H.264 video width and height must be even")

    matrix = args.matrix.expanduser().resolve()
    python = args.python.expanduser().resolve()
    videos_root = args.videos_root.expanduser().resolve()
    cases = load_cases(matrix)
    if args.task:
        selected_tasks = set(args.task)
        cases = [case for case in cases if case.task in selected_tasks]
    if args.condition:
        selected_conditions = set(args.condition)
        cases = [case for case in cases if case.condition in selected_conditions]
    if not cases:
        raise ValueError("task/condition filters selected no cases")

    stages = ["render"] if args.render_only else ["rollout"]
    if not args.rollout_only and not args.render_only:
        stages.append("render")
    summary = {
        "planned": len(cases) * len(stages),
        "completed": 0,
        "dry_run": 0,
        "skipped": 0,
        "failed": [],
    }

    for case in cases:
        commands = build_case_commands(
            case,
            python=python,
            device=args.device,
            videos_root=videos_root,
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
        case_failed = False
        for stage in stages:
            output = commands.metrics if stage == "rollout" else commands.video
            command = commands.rollout if stage == "rollout" else commands.render
            job = {"stage": stage, "task": case.task, "condition": case.condition, "output": str(output)}
            if case_failed:
                summary["skipped"] += 1
                print("BATCH_SKIP=" + json.dumps({**job, "reason": "rollout_failed"}, sort_keys=True))
                continue
            if args.resume and output.is_file():
                summary["skipped"] += 1
                print("BATCH_SKIP=" + json.dumps({**job, "reason": "output_exists"}, sort_keys=True))
                continue
            print("BATCH_JOB=" + json.dumps({**job, "command": command}, sort_keys=True))
            if args.dry_run:
                summary["dry_run"] += 1
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(command, cwd=REPO_ROOT)
            if result.returncode == 0:
                summary["completed"] += 1
                continue
            case_failed = stage == "rollout"
            summary["failed"].append({**job, "returncode": result.returncode})
            if args.fail_fast:
                print("BATCH_SUMMARY=" + json.dumps(summary, sort_keys=True))
                return 1

    print("BATCH_SUMMARY=" + json.dumps(summary, sort_keys=True))
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
