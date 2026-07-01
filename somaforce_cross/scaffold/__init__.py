"""Sonic-based scaffold interfaces for SomaForce-Cross."""

from somaforce_cross.scaffold.contracts import (
    G1_BODY_JOINT_NAMES,
    G1_FULL_JOINT_NAMES,
    MotionTrajectory,
    ScaffoldOutput,
    ScaffoldTask,
)
from somaforce_cross.scaffold.scenes import (
    G1_PUSH_PULL_BOX_SCENE,
    G1_PUSH_PULL_DOOR_SCENE,
    ScaffoldSceneSpec,
    make_g1_push_pull_box_scaffold,
    make_g1_push_pull_door_scaffold,
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
    motion_trajectory_to_sonic_entry,
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
    "G1_PUSH_PULL_BOX_SCENE",
    "G1_PUSH_PULL_DOOR_SCENE",
    "InteractionMode",
    "MotionTrajectory",
    "MotionTrajectoryScaffold",
    "ScaffoldOutput",
    "ScaffoldSceneSpec",
    "ScaffoldTask",
    "SonicManagerEnvBinding",
    "SonicObjectBinding",
    "SonicScaffoldAdapter",
    "make_g1_box_sonic_binding",
    "make_g1_door_sonic_binding",
    "make_sonic_motion_library",
    "make_sonic_manager_overrides",
    "make_sonic_verify_command",
    "make_g1_push_pull_box_scaffold",
    "make_g1_push_pull_box_trajectory",
    "make_g1_push_pull_door_scaffold",
    "make_g1_push_pull_door_trajectory",
    "make_g1_task_trajectory",
    "motion_trajectory_to_sonic_entry",
    "write_single_task_sonic_motion_file",
    "write_sonic_motion_file",
]
