"""HDMI + OMOMO scaffold interfaces for SomaForce-Cross."""

from somaforce_cross.scaffold.contracts import (
    G1_BODY_JOINT_NAMES,
    G1_FULL_JOINT_NAMES,
    MotionTrajectory,
    ScaffoldOutput,
    ScaffoldTask,
)
from somaforce_cross.scaffold.diagnostics import (
    ScaffoldRolloutSummary,
    summarize_first_scaffold_rollouts,
    summarize_scaffold_rollout,
)
from somaforce_cross.scaffold.scenes import (
    G1_HEAVY_PAYLOAD_SCENE,
    G1_PUSH_PULL_BOX_SCENE,
    G1_PUSH_PULL_DOOR_SCENE,
    ScaffoldSceneSpec,
    make_g1_push_pull_box_scaffold,
    make_g1_push_pull_door_scaffold,
    make_g1_hdmi_reference_scaffold,
    make_g1_heavy_payload_scaffold,
)
from somaforce_cross.scaffold.hdmi_adapter import (
    hdmi_mapping_to_reference,
    load_hdmi_reference,
)
from somaforce_cross.scaffold.hdmi_scaffold import HDMIReferenceScaffold
from somaforce_cross.scaffold.omomo_adapter import (
    load_retargeted_omomo_reference,
    omomo_retarget_mapping_to_reference,
)
from somaforce_cross.scaffold.reference_library import ReferenceLibrary
from somaforce_cross.scaffold.reference_schema import (
    CanonicalReferenceEpisode,
    ReferenceMetadata,
    ReferenceSource,
    finite_difference,
    reconstruct_contact_targets,
)
from somaforce_cross.scaffold.sonic_adapter import SonicScaffoldAdapter
from somaforce_cross.scaffold.sonic_env import (
    SonicManagerEnvBinding,
    SonicObjectBinding,
    make_g1_box_sonic_binding,
    make_g1_door_sonic_binding,
    make_sonic_manager_overrides,
    make_sonic_verify_command,
)
from somaforce_cross.scaffold.sonic_motion import (
    make_sonic_motion_library,
    make_sonic_object_motion_library,
    motion_trajectory_to_sonic_entry,
    write_single_task_sonic_object_motion_file,
    write_single_task_sonic_motion_file,
    write_sonic_motion_file,
)
from somaforce_cross.scaffold.task_trajectories import (
    InteractionMode,
    make_g1_push_pull_box_trajectory,
    make_g1_push_pull_door_trajectory,
    make_g1_task_trajectory,
)
from somaforce_cross.scaffold.trajectory import MotionTrajectoryScaffold

__all__ = [
    "G1_BODY_JOINT_NAMES",
    "G1_FULL_JOINT_NAMES",
    "G1_HEAVY_PAYLOAD_SCENE",
    "G1_PUSH_PULL_BOX_SCENE",
    "G1_PUSH_PULL_DOOR_SCENE",
    "InteractionMode",
    "CanonicalReferenceEpisode",
    "HDMIReferenceScaffold",
    "MotionTrajectory",
    "MotionTrajectoryScaffold",
    "ScaffoldOutput",
    "ScaffoldRolloutSummary",
    "ScaffoldSceneSpec",
    "ScaffoldTask",
    "ReferenceLibrary",
    "ReferenceMetadata",
    "ReferenceSource",
    "SonicManagerEnvBinding",
    "SonicObjectBinding",
    "SonicScaffoldAdapter",
    "make_g1_box_sonic_binding",
    "make_g1_door_sonic_binding",
    "make_sonic_motion_library",
    "make_sonic_object_motion_library",
    "make_sonic_manager_overrides",
    "make_sonic_verify_command",
    "make_g1_push_pull_box_scaffold",
    "make_g1_push_pull_box_trajectory",
    "make_g1_push_pull_door_scaffold",
    "make_g1_push_pull_door_trajectory",
    "make_g1_hdmi_reference_scaffold",
    "make_g1_heavy_payload_scaffold",
    "make_g1_task_trajectory",
    "motion_trajectory_to_sonic_entry",
    "finite_difference",
    "hdmi_mapping_to_reference",
    "load_hdmi_reference",
    "load_retargeted_omomo_reference",
    "omomo_retarget_mapping_to_reference",
    "reconstruct_contact_targets",
    "summarize_first_scaffold_rollouts",
    "summarize_scaffold_rollout",
    "write_single_task_sonic_object_motion_file",
    "write_single_task_sonic_motion_file",
    "write_sonic_motion_file",
]
