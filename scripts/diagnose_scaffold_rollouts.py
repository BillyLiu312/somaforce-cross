"""Print scaffold-only rollout diagnostics for the first G1 task variants."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from somaforce_cross.scaffold import (
    ScaffoldTask,
    summarize_first_scaffold_rollouts,
    summarize_scaffold_rollout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("all", "push_pull_door", "push_pull_box"), default="all")
    parser.add_argument("--interaction-mode", choices=("all", "push", "pull"), default="all")
    parser.add_argument("--num-frames", type=int, default=80)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--action-clip", type=float, default=None)
    parser.add_argument("--no-strict", action="store_true", help="Print diagnostics without failing direction checks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = _summaries(args)
    for summary in summaries:
        print(summary.as_line())

    failed = [summary.label for summary in summaries if not summary.direction_ok]
    if failed and not args.no_strict:
        raise SystemExit(f"scaffold_direction_check_failed={failed}")
    print(f"scaffold_rollout_diagnostics_ok={int(not failed)}")


def _summaries(args: argparse.Namespace):
    if args.task == "all" and args.interaction_mode == "all":
        return summarize_first_scaffold_rollouts(args.num_frames, args.duration_s, args.action_clip)

    tasks = (
        (ScaffoldTask.PUSH_PULL_DOOR, ScaffoldTask.PUSH_PULL_BOX)
        if args.task == "all"
        else (ScaffoldTask(args.task),)
    )
    modes = ("push", "pull") if args.interaction_mode == "all" else (args.interaction_mode,)
    return [
        summarize_scaffold_rollout(task, mode, args.num_frames, args.duration_s, args.action_clip)
        for task in tasks
        for mode in modes
    ]


if __name__ == "__main__":
    main()
