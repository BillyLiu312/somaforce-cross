"""Convert HDMI or retargeted OMOMO NPZ data into the canonical library."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from somaforce_cross.scaffold import (  # noqa: E402
    ReferenceLibrary,
    ScaffoldTask,
    load_hdmi_reference,
    load_retargeted_omomo_reference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one canonical HDMI/OMOMO scaffold reference episode."
    )
    parser.add_argument("--source", choices=("hdmi", "omomo-retargeted"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--source-clip-id", required=True)
    parser.add_argument("--source-revision", default="unknown")
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument(
        "--task",
        choices=tuple(task.value for task in ScaffoldTask),
        help="Required for HDMI; OMOMO-retargeted input is always heavy_payload.",
    )
    parser.add_argument("--left-hand-body-index", type=int)
    parser.add_argument("--right-hand-body-index", type=int)
    parser.add_argument("--object-body-index", type=int, default=-1)
    parser.add_argument("--contact-hand-index", type=int, choices=(0, 1), default=1)
    parser.add_argument("--retarget-version")
    parser.add_argument(
        "--retarget-quality",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable OMOMO retarget quality metric.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source == "hdmi":
        if args.task is None:
            raise ValueError("HDMI conversion requires --task")
        if args.left_hand_body_index is None or args.right_hand_body_index is None:
            raise ValueError(
                "HDMI conversion requires --left-hand-body-index and --right-hand-body-index "
                "because body ordering belongs to the source asset revision"
            )
        episode = load_hdmi_reference(
            args.input,
            episode_id=args.episode_id,
            source_clip_id=args.source_clip_id,
            task=ScaffoldTask(args.task),
            hand_body_indices=(args.left_hand_body_index, args.right_hand_body_index),
            object_body_index=args.object_body_index,
            contact_hand_index=args.contact_hand_index,
            fps=args.fps,
            source_revision=args.source_revision,
        )
    else:
        if args.task is not None and args.task != ScaffoldTask.HEAVY_PAYLOAD.value:
            raise ValueError("OMOMO-retargeted input only supports task=heavy_payload")
        if args.retarget_version is None:
            raise ValueError("OMOMO conversion requires --retarget-version")
        episode = load_retargeted_omomo_reference(
            args.input,
            episode_id=args.episode_id,
            source_clip_id=args.source_clip_id,
            retarget_version=args.retarget_version,
            fps=args.fps,
            source_revision=args.source_revision,
            retarget_quality=_quality_metrics(args.retarget_quality),
        )

    library = ReferenceLibrary((episode,))
    library.save(args.output)
    print(f"reference_file={args.output}")
    print(f"episode_id={episode.metadata.episode_id}")
    print(f"source={episode.metadata.source.value}")
    print(f"task={episode.metadata.task.value}")
    print(f"frames={episode.num_frames}")
    print(f"joint_pos={tuple(episode.joint_pos.shape)}")
    print(f"body_pos_w={tuple(episode.body_pos_w.shape)}")
    print(f"hand_pose_w={tuple(episode.hand_pose_w.shape)}")
    print(f"object_root_pose_w={tuple(episode.object_root_pose_w.shape)}")
    print(f"valid_frames={int(episode.reference_valid.sum())}/{episode.num_frames}")


def _quality_metrics(items: list[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise ValueError(
                f"Invalid --retarget-quality value: {item!r}; expected KEY=VALUE"
            )
        metrics[key] = float(value)
    return metrics


if __name__ == "__main__":
    main()
