"""Verify procedural scaffold trajectories produce nominal G1 actions."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from somaforce_cross.scaffold import (
    make_g1_push_pull_box_scaffold,
    make_g1_push_pull_door_scaffold,
)


def main() -> None:
    variants = [
        ("door", "push", make_g1_push_pull_door_scaffold),
        ("door", "pull", make_g1_push_pull_door_scaffold),
        ("box", "push", make_g1_push_pull_box_scaffold),
        ("box", "pull", make_g1_push_pull_box_scaffold),
    ]
    for task_name, mode, factory in variants:
        scaffold = factory(interaction_mode=mode, num_frames=16, action_clip=None)
        out = scaffold.at_frame(15)
        hand_shape = None if out.nominal_hand_ref is None else tuple(out.nominal_hand_ref.shape)
        body_shape = None if out.nominal_body_ref is None else tuple(out.nominal_body_ref.shape)
        cmd_shape = None if out.cmd_6d is None else tuple(out.cmd_6d.shape)
        print(
            f"{task_name}_{mode}: "
            f"a_nom={tuple(out.a_nom.shape)} "
            f"hand_ref={hand_shape} "
            f"body_ref={body_shape} "
            f"cmd_6d={cmd_shape}"
        )


if __name__ == "__main__":
    main()
