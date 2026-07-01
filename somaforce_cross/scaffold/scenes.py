"""First G1 scaffold scene factories."""

from __future__ import annotations

from dataclasses import dataclass

from somaforce_cross.scaffold.contracts import MotionTrajectory, ScaffoldTask
from somaforce_cross.scaffold.trajectory import MotionTrajectoryScaffold


@dataclass(frozen=True)
class ScaffoldSceneSpec:
    """Minimal scene identity for Sonic-based nominal scaffold rollouts."""

    task: ScaffoldTask
    robot: str
    object_type: str
    interaction_modes: tuple[str, ...]
    mismatch_sources: tuple[str, ...]


G1_PUSH_PULL_DOOR_SCENE = ScaffoldSceneSpec(
    task=ScaffoldTask.PUSH_PULL_DOOR,
    robot="unitree_g1",
    object_type="articulated_door",
    interaction_modes=("push", "pull"),
    mismatch_sources=("hinge_axis_error", "handle_pose_error", "friction_damping_variation"),
)

G1_PUSH_PULL_BOX_SCENE = ScaffoldSceneSpec(
    task=ScaffoldTask.PUSH_PULL_BOX,
    robot="unitree_g1",
    object_type="rigid_box",
    interaction_modes=("push", "pull"),
    mismatch_sources=("box_pose_error", "friction_variation", "initial_stance_reach_stress"),
)


def make_g1_push_pull_door_scaffold(
    trajectory: MotionTrajectory,
    action_clip: float | None = 20.0,
) -> MotionTrajectoryScaffold:
    """Create the Unitree G1 door nominal-action scaffold."""

    return MotionTrajectoryScaffold(
        trajectory=trajectory,
        task=ScaffoldTask.PUSH_PULL_DOOR,
        action_clip=action_clip,
    )


def make_g1_push_pull_box_scaffold(
    trajectory: MotionTrajectory,
    action_clip: float | None = 20.0,
) -> MotionTrajectoryScaffold:
    """Create the Unitree G1 box nominal-action scaffold."""

    return MotionTrajectoryScaffold(
        trajectory=trajectory,
        task=ScaffoldTask.PUSH_PULL_BOX,
        action_clip=action_clip,
    )
