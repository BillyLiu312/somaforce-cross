"""First G1 scaffold scene factories."""

from __future__ import annotations

from dataclasses import dataclass

from somaforce_cross.scaffold.contracts import MotionTrajectory, ScaffoldTask
from somaforce_cross.scaffold.hdmi_scaffold import HDMIReferenceScaffold
from somaforce_cross.scaffold.reference_schema import CanonicalReferenceEpisode
from somaforce_cross.scaffold.task_trajectories import (
    InteractionMode,
    make_g1_push_pull_box_trajectory,
    make_g1_push_pull_door_trajectory,
)
from somaforce_cross.scaffold.trajectory import MotionTrajectoryScaffold


@dataclass(frozen=True)
class ScaffoldSceneSpec:
    """Minimal scene identity for nominal scaffold rollouts."""

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

G1_HEAVY_PAYLOAD_SCENE = ScaffoldSceneSpec(
    task=ScaffoldTask.HEAVY_PAYLOAD,
    robot="unitree_g1",
    object_type="rigid_heavy_payload",
    interaction_modes=("lift_carry_place",),
    mismatch_sources=(
        "payload_mass_error",
        "payload_com_error",
        "bilateral_load_share_error",
        "grasp_transform_error",
    ),
)


def make_g1_push_pull_door_scaffold(
    trajectory: MotionTrajectory | None = None,
    interaction_mode: InteractionMode = "push",
    num_frames: int = 80,
    duration_s: float = 2.0,
    action_clip: float | None = 20.0,
) -> MotionTrajectoryScaffold:
    """Create the Unitree G1 door nominal-action scaffold."""

    if trajectory is None:
        trajectory = make_g1_push_pull_door_trajectory(
            mode=interaction_mode,
            num_frames=num_frames,
            duration_s=duration_s,
        )
    return MotionTrajectoryScaffold(
        trajectory=trajectory,
        task=ScaffoldTask.PUSH_PULL_DOOR,
        action_clip=action_clip,
    )


def make_g1_push_pull_box_scaffold(
    trajectory: MotionTrajectory | None = None,
    interaction_mode: InteractionMode = "push",
    num_frames: int = 80,
    duration_s: float = 2.0,
    action_clip: float | None = 20.0,
) -> MotionTrajectoryScaffold:
    """Create the Unitree G1 box nominal-action scaffold."""

    if trajectory is None:
        trajectory = make_g1_push_pull_box_trajectory(
            mode=interaction_mode,
            num_frames=num_frames,
            duration_s=duration_s,
        )
    return MotionTrajectoryScaffold(
        trajectory=trajectory,
        task=ScaffoldTask.PUSH_PULL_BOX,
        action_clip=action_clip,
    )


def make_g1_hdmi_reference_scaffold(
    reference: CanonicalReferenceEpisode,
    action_clip: float | None = 20.0,
) -> HDMIReferenceScaffold:
    """Create the selected HDMI-style scaffold for a canonical reference."""

    return HDMIReferenceScaffold(reference=reference, action_clip=action_clip)


def make_g1_heavy_payload_scaffold(
    reference: CanonicalReferenceEpisode,
    action_clip: float | None = 20.0,
) -> HDMIReferenceScaffold:
    """Create an OMOMO-derived heavy-payload scaffold reference policy."""

    if reference.metadata.task != ScaffoldTask.HEAVY_PAYLOAD:
        raise ValueError("heavy-payload scaffold requires task=heavy_payload")
    return make_g1_hdmi_reference_scaffold(reference, action_clip=action_clip)
