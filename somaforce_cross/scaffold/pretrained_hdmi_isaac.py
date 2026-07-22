"""Independent Isaac Lab runtime for the frozen HDMI teacher scaffold.

Isaac Sim must be launched before importing this module. No HDMI or
active_adaptation module is imported here.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from somaforce_cross.scaffold.pretrained_hdmi import (
    HDMI_ACTION_JOINT_NAMES,
    HDMI_DEFAULT_JOINT_POS,
    HDMI_PHYSICS_MATERIAL_COMBINE_MODE,
    HDMITaskSpec,
    HDMIJointPositionActionRuntime,
    HDMIObservationBatch,
    HDMIObservationHistory,
    PretrainedHDMIScaffold,
    get_hdmi_task_spec,
    reference_action,
)


TRACKING_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
FUTURE_STEPS: tuple[int, ...] = (1, 2, 8, 16, 32)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    result = q.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_apply(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_xyz = q[..., 1:]
    t = 2.0 * torch.linalg.cross(q_xyz, vector, dim=-1)
    return vector + q[..., :1] * t + torch.linalg.cross(q_xyz, t, dim=-1)


def quat_apply_inverse(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return quat_apply(quat_conjugate(q), vector)


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    half = yaw * 0.5
    zeros = torch.zeros_like(half)
    return torch.stack((torch.cos(half), zeros, zeros, torch.sin(half)), dim=-1)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    two = 2.0
    return torch.stack(
        (
            1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w),
            two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w),
            two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)


class HDMIMotionReference:
    def __init__(self, artifact_dir: Path, device: torch.device) -> None:
        with np.load(artifact_dir / "reference/motion.npz", allow_pickle=False) as archive:
            self.data = {
                key: torch.as_tensor(archive[key], device=device)
                for key in archive.files
            }
        meta = json.loads((artifact_dir / "reference/meta.json").read_text(encoding="utf-8"))
        self.body_names = tuple(meta["body_names"])
        self.joint_names = tuple(meta["joint_names"])
        self.fps = float(meta["fps"])
        self.length = int(self.data["joint_pos"].shape[0])

    def frames(self, step: int, offsets: tuple[int, ...] = FUTURE_STEPS) -> torch.Tensor:
        return torch.tensor(
            [min(step + offset, self.length - 1) for offset in offsets],
            device=self.data["joint_pos"].device,
            dtype=torch.long,
        )


def make_scene_cfg(
    artifact_dir: Path,
    num_envs: int,
    task_spec: HDMITaskSpec | None = None,
    *,
    box_mass: float = 8.0,
) -> InteractiveSceneCfg:
    task_spec = task_spec or get_hdmi_task_spec("push_door_hand")
    natural_frequency = 10.0 * 2.0 * torch.pi
    damping_ratio = 2.0
    armature_5020 = 0.003609725
    armature_7520_14 = 0.010177520
    armature_7520_22 = 0.025101925
    armature_4010 = 0.00425

    def stiffness(armature: float) -> float:
        return float(armature * natural_frequency**2)

    def damping(armature: float) -> float:
        return float(2.0 * damping_ratio * armature * natural_frequency)

    robot_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(artifact_dir / "assets/g1.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.76),
            joint_pos=dict(HDMI_DEFAULT_JOINT_POS),
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint"],
                effort_limit_sim={".*_hip_yaw_joint": 88.0, ".*_hip_roll_joint": 139.0, ".*_hip_pitch_joint": 88.0, ".*_knee_joint": 139.0},
                velocity_limit_sim={".*_hip_yaw_joint": 32.0, ".*_hip_roll_joint": 20.0, ".*_hip_pitch_joint": 32.0, ".*_knee_joint": 20.0},
                stiffness={".*_hip_pitch_joint": stiffness(armature_7520_14), ".*_hip_roll_joint": stiffness(armature_7520_22), ".*_hip_yaw_joint": stiffness(armature_7520_14), ".*_knee_joint": stiffness(armature_7520_22)},
                damping={".*_hip_pitch_joint": damping(armature_7520_14), ".*_hip_roll_joint": damping(armature_7520_22), ".*_hip_yaw_joint": damping(armature_7520_14), ".*_knee_joint": damping(armature_7520_22)},
                armature={".*_hip_pitch_joint": armature_7520_14, ".*_hip_roll_joint": armature_7520_22, ".*_hip_yaw_joint": armature_7520_14, ".*_knee_joint": armature_7520_22},
            ),
            "feet": ImplicitActuatorCfg(joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"], effort_limit_sim=50.0, velocity_limit_sim=37.0, stiffness=2 * stiffness(armature_5020), damping=2 * damping(armature_5020), armature=2 * armature_5020),
            "waist": ImplicitActuatorCfg(joint_names_expr=["waist_roll_joint", "waist_pitch_joint"], effort_limit_sim=50.0, velocity_limit_sim=37.0, stiffness=2 * stiffness(armature_5020), damping=2 * damping(armature_5020), armature=2 * armature_5020),
            "waist_yaw": ImplicitActuatorCfg(joint_names_expr=["waist_yaw_joint"], effort_limit_sim=88.0, velocity_limit_sim=32.0, stiffness=stiffness(armature_7520_14), damping=damping(armature_7520_14), armature=armature_7520_14),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint", ".*_wrist_roll_joint", ".*_wrist_pitch_joint", ".*_wrist_yaw_joint"],
                effort_limit_sim={".*_shoulder_pitch_joint": 25.0, ".*_shoulder_roll_joint": 25.0, ".*_shoulder_yaw_joint": 25.0, ".*_elbow_joint": 25.0, ".*_wrist_roll_joint": 25.0, ".*_wrist_pitch_joint": 5.0, ".*_wrist_yaw_joint": 5.0},
                velocity_limit_sim={".*_shoulder_pitch_joint": 37.0, ".*_shoulder_roll_joint": 37.0, ".*_shoulder_yaw_joint": 37.0, ".*_elbow_joint": 37.0, ".*_wrist_roll_joint": 37.0, ".*_wrist_pitch_joint": 22.0, ".*_wrist_yaw_joint": 22.0},
                stiffness={".*_shoulder_pitch_joint": stiffness(armature_5020), ".*_shoulder_roll_joint": stiffness(armature_5020), ".*_shoulder_yaw_joint": stiffness(armature_5020), ".*_elbow_joint": stiffness(armature_5020), ".*_wrist_roll_joint": stiffness(armature_5020), ".*_wrist_pitch_joint": stiffness(armature_4010), ".*_wrist_yaw_joint": stiffness(armature_4010)},
                damping={".*_shoulder_pitch_joint": damping(armature_5020), ".*_shoulder_roll_joint": damping(armature_5020), ".*_shoulder_yaw_joint": damping(armature_5020), ".*_elbow_joint": damping(armature_5020), ".*_wrist_roll_joint": damping(armature_5020), ".*_wrist_pitch_joint": damping(armature_4010), ".*_wrist_yaw_joint": damping(armature_4010)},
                armature={".*_shoulder_pitch_joint": armature_5020, ".*_shoulder_roll_joint": armature_5020, ".*_shoulder_yaw_joint": armature_5020, ".*_elbow_joint": armature_5020, ".*_wrist_roll_joint": armature_5020, ".*_wrist_pitch_joint": armature_4010, ".*_wrist_yaw_joint": armature_4010},
            ),
        },
    )
    door_cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Door",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(artifact_dir / "assets/door.usd"),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                enabled_self_collisions=False,
            ),
        ),
        actuators={
            "door": IdealPDActuatorCfg(
                joint_names_expr=["door_joint"],
                stiffness=0.0,
                damping=0.0,
                friction=0.0,
                effort_limit=100.0,
                velocity_limit=20.0,
            )
        },
    )

    ground_cfg = AssetBaseCfg(
        prim_path="/World/ground",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
        spawn=sim_utils.CuboidCfg(
            size=(100.0, 100.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode=HDMI_PHYSICS_MATERIAL_COMBINE_MODE,
                restitution_combine_mode=HDMI_PHYSICS_MATERIAL_COMBINE_MODE,
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.20, 0.22),
            ),
        ),
    )
    light_cfg = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    contacts_cfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    if task_spec.object_kind == "articulation":
        @configclass
        class HDMIDoorSceneCfg(InteractiveSceneCfg):
            ground: AssetBaseCfg = ground_cfg
            light: AssetBaseCfg = light_cfg
            robot: ArticulationCfg = robot_cfg
            door: ArticulationCfg = door_cfg
            contacts: ContactSensorCfg = contacts_cfg

        return HDMIDoorSceneCfg(num_envs=num_envs, env_spacing=5.0)

    box_cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(artifact_dir / "assets/box.usd"),
            activate_contact_sensors=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=box_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
        ),
    )

    @configclass
    class HDMIBoxSceneCfg(InteractiveSceneCfg):
        ground: AssetBaseCfg = ground_cfg
        light: AssetBaseCfg = light_cfg
        robot: ArticulationCfg = robot_cfg
        box: RigidObjectCfg = box_cfg
        contacts: ContactSensorCfg = contacts_cfg
        left_box_contacts = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link",
            history_length=0,
            track_air_time=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Box/box"],
        )
        right_box_contacts = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link",
            history_length=0,
            track_air_time=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Box/box"],
        )

    return HDMIBoxSceneCfg(num_envs=num_envs, env_spacing=5.0)

@dataclass
class RolloutMetrics:
    steps: int
    initial_door_joint: float
    final_door_joint: float
    task_progress: float
    target_progress: float
    max_task_progress: float
    min_root_height: float
    max_contact_force: float
    max_wrist_contact_force: float
    mean_support_contact_count: float
    max_abs_action: float
    mean_abs_action: float
    nonfinite_count: int
    terminated: bool
    termination_reason: str
    zero_hook_exact: bool


@dataclass
class PushBoxRolloutMetrics:
    case: str
    steps: int
    initial_box_position: list[float]
    final_box_position: list[float]
    reference_initial_box_position: list[float]
    reference_final_box_position: list[float]
    actual_xy_displacement: list[float]
    reference_xy_displacement: list[float]
    directional_progress: float
    max_directional_progress: float
    path_progress: float
    reference_path_length: float
    final_tracking_error: float
    final_yaw_error: float
    max_left_wrist_contact_force: float
    max_right_wrist_contact_force: float
    mean_left_wrist_contact_force: float
    mean_right_wrist_contact_force: float
    contact_active_fraction: float
    min_root_height: float
    mean_support_contact_count: float
    max_contact_force: float
    max_abs_action: float
    mean_abs_action: float
    nonfinite_count: int
    terminated: bool
    termination_reason: str
    stable: bool
    zero_hook_exact: bool
    settings: dict[str, object]


class PretrainedHDMIIsaacRuntime:
    """Standalone manifest-selected G1 + object HDMI rollout environment."""

    physics_dt = 0.005
    control_dt = 0.02
    decimation = 4

    def __init__(
        self,
        sim: SimulationContext,
        artifact_dir: str | Path,
        *,
        num_envs: int = 1,
        delay: int = 4,
        alpha: float = 0.9,
        door_friction: float = 0.3,
        door_damping: float = 0.55,
        box_mass: float = 8.0,
        box_friction: float = 0.5,
        box_com_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        initial_object_xy: tuple[float, float] = (0.0, 0.0),
        initial_object_yaw: float = 0.0,
        contact_target_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
        case_name: str = "nominal",
        show_reference_box: bool = False,
    ) -> None:
        self.sim = sim
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        self.device = torch.device(sim.device)
        self.num_envs = num_envs
        self.scaffold = PretrainedHDMIScaffold.from_artifact(
            self.artifact_dir, device=self.device
        )
        self.task_spec = self.scaffold.task_spec
        self.scene = InteractiveScene(
            make_scene_cfg(
                self.artifact_dir, num_envs, self.task_spec, box_mass=box_mass
            )
        )
        self.sim.reset()
        self.scene.update(self.physics_dt)
        self.robot = self.scene["robot"]
        self.object = self.scene[self.task_spec.object_asset_name]
        self.door = self.object if self.task_spec.object_kind == "articulation" else None
        self.box = self.object if self.task_spec.object_kind == "rigid_object" else None
        self.contacts = self.scene["contacts"]
        self.filtered_wrist_contacts = (
            [self.scene["left_box_contacts"], self.scene["right_box_contacts"]]
            if self.task_spec.task == "push_box"
            else []
        )
        self.reference = HDMIMotionReference(self.artifact_dir, self.device)
        self.history = HDMIObservationHistory(num_envs, device=self.device)
        self.action_runtime = HDMIJointPositionActionRuntime(
            self.robot.data.default_joint_pos,
            self.robot.joint_names,
            num_envs=num_envs,
            decimation=self.decimation,
            delay=delay,
            alpha=alpha,
            action_joint_names=self.task_spec.action_joint_names,
            action_scale=self.task_spec.action_scale,
            device=self.device,
        )
        self.door_friction = float(door_friction)
        self.door_damping = float(door_damping)
        self.box_mass = float(box_mass)
        self.box_friction = float(box_friction)
        self.box_com_offset = tuple(float(value) for value in box_com_offset)
        self.initial_object_xy = tuple(float(value) for value in initial_object_xy)
        self.initial_object_yaw = float(initial_object_yaw)
        self.contact_target_offset = tuple(float(value) for value in contact_target_offset)
        self.case_name = case_name
        self.reference_step = 0
        self._resolve_indices()
        self.reference_object_marker: VisualizationMarkers | None = None
        if show_reference_box and self.task_spec.task == "push_box" and self.sim.has_gui():
            self.reference_object_marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/HDMIReferenceBox",
                    markers={
                        "reference_box": sim_utils.CuboidCfg(
                            size=(1.0, 0.8, 0.8),
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(0.05, 0.85, 0.25), opacity=0.3
                            ),
                        )
                    },
                )
            )
        if self.task_spec.object_kind == "rigid_object":
            self._configure_rigid_object()
        self.reset()

    def _configure_rigid_object(self) -> None:
        indices = torch.arange(self.num_envs, device="cpu")
        masses = self.object.root_physx_view.get_masses().clone()
        inertias = self.object.root_physx_view.get_inertias().clone()
        scale = self.box_mass / masses
        masses.fill_(self.box_mass)
        inertias *= scale.unsqueeze(-1) if inertias.ndim == 3 else scale
        self.object.root_physx_view.set_masses(masses, indices)
        self.object.root_physx_view.set_inertias(inertias, indices)
        materials = self.object.root_physx_view.get_material_properties().clone()
        materials[..., 0] = self.box_friction
        materials[..., 1] = self.box_friction
        materials[..., 2] = 0.0
        self.object.root_physx_view.set_material_properties(materials, indices)
        coms = self.object.root_physx_view.get_coms().clone()
        coms[..., :3] += torch.tensor(self.box_com_offset, device=coms.device)
        self.object.root_physx_view.set_coms(coms, indices)

    def _resolve_indices(self) -> None:
        shared_joint_names = tuple(
            name for name in self.robot.joint_names if name in self.reference.joint_names
        )
        self.ref_shared_joint_ids = torch.tensor(
            [self.reference.joint_names.index(name) for name in shared_joint_names],
            device=self.device,
        )
        self.robot_shared_joint_ids = torch.tensor(
            [self.robot.joint_names.index(name) for name in shared_joint_names],
            device=self.device,
        )
        self.tracking_robot_body_ids = torch.tensor(
            [self.robot.body_names.index(name) for name in TRACKING_BODY_NAMES],
            device=self.device,
        )
        self.tracking_ref_body_ids = torch.tensor(
            [self.reference.body_names.index(name) for name in TRACKING_BODY_NAMES],
            device=self.device,
        )
        self.ref_action_joint_ids = torch.tensor(
            [self.reference.joint_names.index(name) for name in HDMI_ACTION_JOINT_NAMES],
            device=self.device,
        )
        self.root_ref_body_id = self.reference.body_names.index("pelvis")
        self.object_ref_body_id = self.reference.body_names.index(
            self.task_spec.object_asset_name
        )
        self.object_body_id = self.object.body_names.index(
            self.task_spec.object_body_name
        )
        self.contact_eef_body_ids = torch.tensor(
            [self.robot.body_names.index(name) for name in self.task_spec.contact_eef_names],
            device=self.device,
        )
        if self.task_spec.object_kind == "articulation":
            self.door_ref_body_id = self.object_ref_body_id
            self.door_ref_joint_id = self.reference.joint_names.index("door_joint")
            self.door_panel_body_id = self.object_body_id
            self.right_wrist_body_id = int(self.contact_eef_body_ids[0])
        self.ankle_body_ids = torch.tensor(
            [self.robot.body_names.index("left_ankle_roll_link"), self.robot.body_names.index("right_ankle_roll_link")],
            device=self.device,
        )
        self.height_body_ids = torch.tensor(
            [self.robot.body_names.index(name) for name in ("left_ankle_roll_link", "right_ankle_roll_link", "pelvis", "torso_link")],
            device=self.device,
        )
        self.contact_wrist_ids = torch.tensor(
            [self.contacts.body_names.index(name) for name in self.task_spec.contact_eef_names],
            device=self.device,
        )
        self.contact_wrist_id = int(self.contact_wrist_ids[-1])
        self.contact_support_ids = torch.tensor(
            [
                self.contacts.body_names.index("left_ankle_roll_link"),
                self.contacts.body_names.index("right_ankle_roll_link"),
            ],
            device=self.device,
        )

    def _reference_robot_joint_state(self, key: str, frame: int = 0) -> torch.Tensor:
        result = torch.zeros(
            self.num_envs,
            len(self.robot.joint_names),
            device=self.device,
            dtype=torch.float32,
        )
        source = self.reference.data[key][[frame]].float().expand(self.num_envs, -1)
        result[:, self.robot_shared_joint_ids] = source[:, self.ref_shared_joint_ids]
        return result

    def reset(self) -> None:
        env_origin = self.scene.env_origins
        robot_root = torch.cat(
            (
                self.reference.data["body_pos_w"][[0], self.root_ref_body_id].float().expand(self.num_envs, -1) + env_origin,
                self.reference.data["body_quat_w"][[0], self.root_ref_body_id].float().expand(self.num_envs, -1),
            ),
            dim=-1,
        )
        robot_velocity = torch.cat(
            (
                self.reference.data["body_lin_vel_w"][[0], self.root_ref_body_id].float().expand(self.num_envs, -1),
                self.reference.data["body_ang_vel_w"][[0], self.root_ref_body_id].float().expand(self.num_envs, -1),
            ),
            dim=-1,
        )
        self.robot.write_root_link_pose_to_sim(robot_root)
        self.robot.write_root_com_velocity_to_sim(robot_velocity)
        joint_pos = self._reference_robot_joint_state("joint_pos")
        joint_vel = self._reference_robot_joint_state("joint_vel")
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

        object_position = (
            self.reference.data["body_pos_w"][[0], self.object_ref_body_id]
            .float()
            .expand(self.num_envs, -1)
            + env_origin
        )
        object_position = object_position.clone()
        object_position[:, 0] += self.initial_object_xy[0]
        object_position[:, 1] += self.initial_object_xy[1]
        object_quat = (
            self.reference.data["body_quat_w"][[0], self.object_ref_body_id]
            .float()
            .expand(self.num_envs, -1)
        )
        half_yaw = torch.full(
            (self.num_envs,), self.initial_object_yaw * 0.5, device=self.device
        )
        yaw_delta = torch.stack(
            (
                torch.cos(half_yaw),
                torch.zeros_like(half_yaw),
                torch.zeros_like(half_yaw),
                torch.sin(half_yaw),
            ),
            dim=-1,
        )
        object_root = torch.cat(
            (
                object_position,
                quat_mul(object_quat, yaw_delta),
            ),
            dim=-1,
        )
        self.object.write_root_link_pose_to_sim(object_root)
        self.object.write_root_com_velocity_to_sim(
            torch.zeros(self.num_envs, 6, device=self.device)
        )
        if self.task_spec.object_kind == "articulation":
            door_joint = self.reference.data["joint_pos"][[0], self.door_ref_joint_id].float().expand(self.num_envs, 1).clone()
            door_joint_vel = self.reference.data["joint_vel"][[0], self.door_ref_joint_id].float().expand(self.num_envs, 1).clone()
            self.object.write_joint_state_to_sim(door_joint, door_joint_vel)
        self.scene.reset()
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(self.physics_dt)
        self.reference_step = 0
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.action_runtime.reset(env_ids)
        self.history.reset(
            env_ids,
            root_ang_vel=self.robot.data.root_ang_vel_b,
            projected_gravity=self.robot.data.projected_gravity_b,
            joint_pos=self.robot.data.joint_pos,
            motion_length=self.reference.length,
        )
        self._update_reference_visualization()

    def _update_reference_visualization(self) -> None:
        if self.reference_object_marker is None:
            return
        frame = min(self.reference_step, self.reference.length - 1)
        position = (
            self.reference.data["body_pos_w"][[frame], self.object_ref_body_id].float()
            + self.scene.env_origins
        )
        # The reference data stores the box origin at its bottom face.
        position = position.clone()
        position[:, 2] += 0.4
        orientation = self.reference.data["body_quat_w"][
            [frame], self.object_ref_body_id
        ].float()
        self.reference_object_marker.visualize(position, orientation)

    def _future_reference(self) -> dict[str, torch.Tensor]:
        frames = self.reference.frames(self.reference_step)
        result: dict[str, torch.Tensor] = {}
        for key, value in self.reference.data.items():
            selected = value.index_select(0, frames).float().unsqueeze(0)
            result[key] = selected.expand(self.num_envs, *selected.shape[1:])
        return result

    def _geometry(self, future: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ref_root_pos = future["body_pos_w"][:, 0, self.root_ref_body_id]
        ref_root_quat = future["body_quat_w"][:, 0, self.root_ref_body_id]
        ref_root_pos_ground = ref_root_pos.clone()
        ref_root_pos_ground[:, 2] = 0.0
        ref_root_yaw = yaw_quat(ref_root_quat)
        ref_body_pos = future["body_pos_w"].index_select(2, self.tracking_ref_body_ids)
        ref_body_quat = future["body_quat_w"].index_select(2, self.tracking_ref_body_ids)
        ref_body_lin = future["body_lin_vel_w"].index_select(2, self.tracking_ref_body_ids)
        ref_body_ang = future["body_ang_vel_w"].index_select(2, self.tracking_ref_body_ids)
        ref_body_local = quat_apply_inverse(
            ref_root_yaw[:, None, None], ref_body_pos - ref_root_pos_ground[:, None, None]
        )

        robot_root_pos = self.robot.data.root_link_pos_w
        robot_root_quat = self.robot.data.root_link_quat_w
        robot_root_pos_ground = robot_root_pos.clone()
        robot_root_pos_ground[:, 2] = 0.0
        robot_root_yaw = yaw_quat(robot_root_quat)
        robot_body_pos = self.robot.data.body_link_pos_w.index_select(1, self.tracking_robot_body_ids)
        robot_body_quat = self.robot.data.body_link_quat_w.index_select(1, self.tracking_robot_body_ids)
        robot_body_lin = self.robot.data.body_com_lin_vel_w.index_select(1, self.tracking_robot_body_ids)
        robot_body_ang = self.robot.data.body_com_ang_vel_w.index_select(1, self.tracking_robot_body_ids)
        robot_body_local = quat_apply_inverse(
            robot_root_yaw[:, None], robot_body_pos - robot_root_pos_ground[:, None]
        )

        ref_body_quat_local = quat_mul(
            quat_conjugate(ref_root_yaw)[:, None, None], ref_body_quat
        )
        robot_body_quat_local = quat_mul(
            quat_conjugate(robot_root_yaw)[:, None], robot_body_quat
        )
        body_quat_diff = quat_mul(
            quat_conjugate(robot_body_quat_local)[:, None], ref_body_quat_local
        )
        ref_lin_local = quat_apply_inverse(ref_root_yaw[:, None, None], ref_body_lin)
        ref_ang_local = quat_apply_inverse(ref_root_yaw[:, None, None], ref_body_ang)
        robot_lin_local = quat_apply_inverse(robot_root_yaw[:, None], robot_body_lin)
        robot_ang_local = quat_apply_inverse(robot_root_yaw[:, None], robot_body_ang)

        object_root_pos = self.object.data.root_link_pos_w
        object_root_quat = self.object.data.root_link_quat_w
        object_body_pos = self.object.data.body_link_pos_w[:, self.object_body_id]
        object_body_quat = self.object.data.body_link_quat_w[:, self.object_body_id]
        target_offsets = torch.tensor(
            self.task_spec.contact_target_offsets,
            device=self.device,
            dtype=object_body_pos.dtype,
        )
        target_offsets += torch.tensor(
            self.contact_target_offset,
            device=self.device,
            dtype=object_body_pos.dtype,
        )
        target_offsets = target_offsets.unsqueeze(0).expand(self.num_envs, -1, -1)
        contact_target = object_body_pos[:, None] + quat_apply(
            object_body_quat[:, None], target_offsets
        )
        wrist_pos = self.robot.data.body_link_pos_w.index_select(
            1, self.contact_eef_body_ids
        )
        wrist_quat = self.robot.data.body_link_quat_w.index_select(
            1, self.contact_eef_body_ids
        )
        eef_offsets = torch.tensor(
            self.task_spec.contact_eef_offsets,
            device=self.device,
            dtype=wrist_pos.dtype,
        ).unsqueeze(0).expand(self.num_envs, -1, -1)
        eef_pos = wrist_pos + quat_apply(wrist_quat, eef_offsets)
        ref_object_pos = (
            future["body_pos_w"][:, :, self.object_ref_body_id]
            + self.scene.env_origins[:, None]
        )
        ref_object_quat = future["body_quat_w"][:, :, self.object_ref_body_id]
        # Legacy aliases keep the door observation path numerically unchanged.
        door_root_pos = object_root_pos
        door_root_quat = object_root_quat
        ref_door_pos = ref_object_pos
        ref_door_quat = ref_object_quat
        return locals()

    def build_observation(self) -> HDMIObservationBatch:
        future = self._future_reference()
        g = self._geometry(future)
        ref_joint_future = future["joint_pos"].index_select(2, self.ref_action_joint_ids)
        command = torch.cat(
            (
                g["ref_body_local"].reshape(self.num_envs, -1),
                ref_joint_future.reshape(self.num_envs, -1),
                self.history.phase,
            ),
            dim=-1,
        )
        object_yaw = yaw_quat(g["door_root_quat"])
        relative_yaw = quat_mul(quat_conjugate(g["robot_root_yaw"]), object_yaw)
        relative_yaw_angle = 2.0 * torch.atan2(relative_yaw[:, 3], relative_yaw[:, 0])
        object_obs = torch.cat(
            (
                quat_apply_inverse(g["robot_root_yaw"], g["door_root_pos"] - g["robot_root_pos"])[:, :2],
                torch.cos(relative_yaw_angle).unsqueeze(1),
                torch.sin(relative_yaw_angle).unsqueeze(1),
                quat_apply_inverse(
                    g["robot_root_yaw"][:, None],
                    g["contact_target"] - g["robot_root_pos"][:, None],
                ).reshape(self.num_envs, -1),
            ),
            dim=-1,
        )
        ref_root_pos_b = quat_apply_inverse(
            g["robot_root_quat"][:, None],
            future["body_pos_w"][:, :, self.root_ref_body_id] + self.scene.env_origins[:, None] - g["robot_root_pos"][:, None],
        )
        ref_root_quat_b = quat_mul(
            quat_conjugate(g["robot_root_quat"])[:, None],
            future["body_quat_w"][:, :, self.root_ref_body_id],
        )
        ankle_pos = self.robot.data.body_link_pos_w.index_select(1, self.ankle_body_ids)
        ankle_vel = self.robot.data.body_com_lin_vel_w.index_select(1, self.ankle_body_ids)
        root_ground = g["robot_root_pos_ground"]
        ankle_pos_b = quat_apply_inverse(g["robot_root_yaw"][:, None], ankle_pos - root_ground[:, None])
        ankle_vel_b = quat_apply_inverse(g["robot_root_quat"][:, None], ankle_vel)
        heights = self.robot.data.body_link_pos_w.index_select(1, self.height_body_ids)[..., 2]
        door_quat_diff = quat_mul(
            quat_conjugate(g["door_root_quat"])[:, None], g["ref_door_quat"]
        )
        privileged_parts = [
                self.history.root_ang_vel.reshape(self.num_envs, -1),
                self.history.projected_gravity.reshape(self.num_envs, -1),
                self.history.joint_pos.reshape(self.num_envs, -1),
                ref_root_pos_b.reshape(self.num_envs, -1),
                quat_to_matrix(ref_root_quat_b)[:, :, :2].reshape(self.num_envs, -1),
                (g["ref_body_local"] - g["robot_body_local"][:, None]).reshape(self.num_envs, -1),
                quat_to_matrix(g["body_quat_diff"])[:, :, :, :2].reshape(self.num_envs, -1),
                (g["ref_lin_local"] - g["robot_lin_local"][:, None]).reshape(self.num_envs, -1),
                (g["ref_ang_local"] - g["robot_ang_local"][:, None]).reshape(self.num_envs, -1),
                self.robot.data.root_lin_vel_b,
                ankle_pos_b.reshape(self.num_envs, -1),
                ankle_vel_b.reshape(self.num_envs, -1),
                heights.reshape(self.num_envs, -1),
                self.action_runtime.applied_action,
                self.robot.data.applied_torque,
                quat_apply_inverse(g["robot_root_quat"], g["door_root_pos"] - g["robot_root_pos"]),
                quat_to_matrix(quat_mul(quat_conjugate(g["robot_root_quat"]), g["door_root_quat"])).reshape(self.num_envs, -1),
                quat_apply_inverse(g["door_root_quat"][:, None], g["ref_door_pos"] - g["door_root_pos"][:, None]).reshape(self.num_envs, -1),
                quat_to_matrix(door_quat_diff).reshape(self.num_envs, -1),
                future["object_contact"].reshape(self.num_envs, -1),
                quat_apply_inverse(
                    g["robot_root_quat"][:, None],
                    g["contact_target"] - g["eef_pos"],
                ).reshape(self.num_envs, -1),
        ]
        if self.task_spec.object_kind == "articulation":
            privileged_parts.extend(
                (
                    self.object.data.joint_pos,
                    self.object.data.joint_vel,
                    self.object.data.applied_torque,
                )
            )
        privileged = torch.cat(privileged_parts, dim=-1)
        current_ref = future["joint_pos"][:, 0]
        default_action_pos = self.robot.data.default_joint_pos[:, self.action_runtime.joint_ids]
        ref_action = reference_action(
            current_ref,
            default_action_pos,
            self.reference.joint_names,
            self.task_spec.action_joint_names,
            self.task_spec.action_scale,
        )
        observation = HDMIObservationBatch(
            command=command,
            policy=self.history.policy_observation(),
            object=object_obs,
            privileged=privileged,
            reference_action=ref_action,
        )
        observation.validate(self.task_spec.observation_dims)
        return observation

    def step(self) -> torch.Tensor:
        observation = self.build_observation()
        a_nom = self.scaffold(observation)
        self.action_runtime.set_nominal_action(a_nom)
        self.history.push_action(a_nom)
        for substep in range(self.decimation):
            target = self.action_runtime.substep_target(substep)
            self.robot.set_joint_position_target(target)
            if self.task_spec.object_kind == "articulation":
                door_vel = self.object.data.joint_vel[:, 0]
                friction = (
                    -torch.sign(door_vel)
                    * (door_vel.abs() > 0.01)
                    * self.door_friction
                )
                door_effort = friction - door_vel * self.door_damping
                self.object.set_joint_effort_target(door_effort.unsqueeze(1))
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
        self._update_reference_visualization()
        if self.sim.has_gui():
            self.sim.render()
        self.history.update_state(
            self.robot.data.root_ang_vel_b,
            self.robot.data.projected_gravity_b,
            self.robot.data.joint_pos,
        )
        self.history.advance_reference()
        self.reference_step += 1
        return a_nom

    def rollout(
        self,
        steps: int,
        *,
        log_interval: int = 25,
        realtime: bool = False,
        playback_rate: float = 1.0,
        hold_seconds: float = 0.0,
    ) -> RolloutMetrics | PushBoxRolloutMetrics:
        if self.task_spec.task == "push_box":
            return self._rollout_push_box(
                steps,
                log_interval=log_interval,
                realtime=realtime,
                playback_rate=playback_rate,
                hold_seconds=hold_seconds,
            )
        if playback_rate <= 0.0:
            raise ValueError("playback_rate must be positive")
        initial_door = float(self.door.data.joint_pos[0, 0])
        ref_initial = float(self.reference.data["joint_pos"][0, self.door_ref_joint_id])
        ref_final = float(self.reference.data["joint_pos"][-1, self.door_ref_joint_id])
        direction = 1.0 if ref_final >= ref_initial else -1.0
        target_progress = abs(ref_final - ref_initial)
        min_height = float("inf")
        max_contact = 0.0
        max_wrist_contact = 0.0
        support_contact_sum = 0.0
        max_progress = 0.0
        action_abs_sum = 0.0
        action_count = 0
        max_action = 0.0
        nonfinite = 0
        terminated = False
        reason = "completed_requested_steps"
        actual_steps = 0
        for step in range(steps):
            wall_step_start = time.perf_counter()
            a_nom = self.step()
            actual_steps = step + 1
            root_height = float(self.robot.data.root_link_pos_w[0, 2])
            door_joint = float(self.door.data.joint_pos[0, 0])
            progress = direction * (door_joint - initial_door)
            contact_norms = self.contacts.data.net_forces_w[0].norm(dim=-1)
            contact_force = float(contact_norms.max())
            wrist_contact = float(contact_norms[self.contact_wrist_id])
            support_count = float((contact_norms[self.contact_support_ids] > 1.0).sum())
            min_height = min(min_height, root_height)
            max_contact = max(max_contact, contact_force)
            max_wrist_contact = max(max_wrist_contact, wrist_contact)
            support_contact_sum += support_count
            max_progress = max(max_progress, progress)
            max_action = max(max_action, float(a_nom.abs().max()))
            action_abs_sum += float(a_nom.abs().sum())
            action_count += a_nom.numel()
            tensors = (a_nom, self.robot.data.joint_pos, self.robot.data.root_link_pos_w, self.door.data.joint_pos)
            nonfinite += sum(int((~torch.isfinite(tensor)).sum()) for tensor in tensors)
            if step % log_interval == 0 or step == steps - 1:
                print(
                    f"step={step:04d} phase={float(self.history.phase[0, 0]):.4f} "
                    f"door_joint={door_joint:.6f} progress={progress:.6f} "
                    f"root_z={root_height:.4f} contact_max={contact_force:.3f} "
                    f"wrist_contact={wrist_contact:.3f} support_contacts={support_count:.0f} "
                    f"a_nom_abs_mean={float(a_nom.abs().mean()):.4f} "
                    f"a_nom_abs_max={float(a_nom.abs().max()):.4f}"
                )
            if nonfinite:
                terminated = True
                reason = "nonfinite_state"
                break
            if root_height < 0.45:
                terminated = True
                reason = "root_height_below_0.45m"
                break
            if self.reference_step + max(FUTURE_STEPS) >= self.reference.length:
                reason = "reference_complete"
                break
            if realtime:
                target_wall_dt = self.control_dt / playback_rate
                remaining = target_wall_dt - (time.perf_counter() - wall_step_start)
                if remaining > 0.0:
                    time.sleep(remaining)
        if hold_seconds > 0.0 and self.sim.has_gui():
            hold_until = time.monotonic() + hold_seconds
            while time.monotonic() < hold_until:
                self.sim.render()
                time.sleep(1.0 / 60.0)
        final_door = float(self.door.data.joint_pos[0, 0])
        return RolloutMetrics(
            steps=actual_steps,
            initial_door_joint=initial_door,
            final_door_joint=final_door,
            task_progress=direction * (final_door - initial_door),
            target_progress=target_progress,
            max_task_progress=max_progress,
            min_root_height=min_height,
            max_contact_force=max_contact,
            max_wrist_contact_force=max_wrist_contact,
            mean_support_contact_count=support_contact_sum / max(actual_steps, 1),
            max_abs_action=max_action,
            mean_abs_action=action_abs_sum / max(action_count, 1),
            nonfinite_count=nonfinite,
            terminated=terminated,
            termination_reason=reason,
            zero_hook_exact=torch.equal(
                self.action_runtime.received_action, self.action_runtime.last_a_nom
            ),
        )

    def _rollout_push_box(
        self,
        steps: int,
        *,
        log_interval: int,
        realtime: bool,
        playback_rate: float,
        hold_seconds: float,
    ) -> PushBoxRolloutMetrics:
        if playback_rate <= 0.0:
            raise ValueError("playback_rate must be positive")
        ref_positions = self.reference.data["body_pos_w"][:, self.object_ref_body_id].float()
        ref_initial = ref_positions[0] + self.scene.env_origins[0]
        ref_final = ref_positions[-1] + self.scene.env_origins[0]
        ref_delta_xy = ref_final[:2] - ref_initial[:2]
        ref_distance = float(torch.linalg.vector_norm(ref_delta_xy))
        if ref_distance <= 1e-8:
            raise ValueError("push-box reference has no XY task direction")
        direction = ref_delta_xy / ref_distance
        reference_path_length = float(
            torch.linalg.vector_norm(ref_positions[1:, :2] - ref_positions[:-1, :2], dim=-1).sum()
        )
        initial = self.object.data.root_link_pos_w[0].clone()
        previous_xy = initial[:2].clone()
        actual_path_length = 0.0
        max_progress = 0.0
        min_height = float("inf")
        max_contact = 0.0
        max_wrist = [0.0, 0.0]
        wrist_sum = [0.0, 0.0]
        support_sum = 0.0
        expected_contact_steps = 0
        both_contact_steps = 0
        max_action = 0.0
        action_abs_sum = 0.0
        action_count = 0
        nonfinite = 0
        terminated = False
        reason = "completed_requested_steps"
        actual_steps = 0

        for step in range(steps):
            wall_step_start = time.perf_counter()
            a_nom = self.step()
            actual_steps = step + 1
            root_height = float(self.robot.data.root_link_pos_w[0, 2])
            box_pos = self.object.data.root_link_pos_w[0]
            displacement_xy = box_pos[:2] - initial[:2]
            progress = float(torch.dot(displacement_xy, direction))
            actual_path_length += float(torch.linalg.vector_norm(box_pos[:2] - previous_xy))
            previous_xy = box_pos[:2].clone()

            contact_norms = self.contacts.data.net_forces_w[0].norm(dim=-1)
            contact_force = float(contact_norms.max())
            wrist_forces = []
            for sensor in self.filtered_wrist_contacts:
                force_matrix = sensor.data.force_matrix_w
                wrist_forces.append(
                    float(force_matrix[0].norm(dim=-1).max())
                    if force_matrix is not None
                    else 0.0
                )
            support_count = float((contact_norms[self.contact_support_ids] > 1.0).sum())
            ref_index = min(self.reference_step - 1, self.reference.length - 1)
            reference_contact = bool(self.reference.data["object_contact"][ref_index].any())
            if reference_contact:
                expected_contact_steps += 1
                both_contact_steps += int(all(force > 1.0 for force in wrist_forces))

            min_height = min(min_height, root_height)
            max_contact = max(max_contact, contact_force)
            max_progress = max(max_progress, progress)
            support_sum += support_count
            for index, force in enumerate(wrist_forces):
                max_wrist[index] = max(max_wrist[index], force)
                wrist_sum[index] += force
            max_action = max(max_action, float(a_nom.abs().max()))
            action_abs_sum += float(a_nom.abs().sum())
            action_count += a_nom.numel()
            tensors = (
                a_nom,
                self.robot.data.joint_pos,
                self.robot.data.root_link_pos_w,
                self.object.data.root_link_pos_w,
                self.object.data.root_link_quat_w,
            )
            nonfinite += sum(int((~torch.isfinite(tensor)).sum()) for tensor in tensors)
            if step % log_interval == 0 or step == steps - 1:
                print(
                    f"step={step:04d} phase={float(self.history.phase[0, 0]):.4f} "
                    f"box_xyz={box_pos.tolist()} progress={progress:.6f} "
                    f"path={actual_path_length:.6f} root_z={root_height:.4f} "
                    f"left_contact={wrist_forces[0]:.3f} "
                    f"right_contact={wrist_forces[1]:.3f} "
                    f"support_contacts={support_count:.0f} "
                    f"a_nom_abs_mean={float(a_nom.abs().mean()):.4f} "
                    f"a_nom_abs_max={float(a_nom.abs().max()):.4f}"
                )
            if nonfinite:
                terminated = True
                reason = "nonfinite_state"
                break
            if root_height < 0.45:
                terminated = True
                reason = "root_height_below_0.45m"
                break
            if realtime:
                target_wall_dt = self.control_dt / playback_rate
                remaining = target_wall_dt - (time.perf_counter() - wall_step_start)
                if remaining > 0.0:
                    time.sleep(remaining)

        if hold_seconds > 0.0 and self.sim.has_gui():
            hold_until = time.monotonic() + hold_seconds
            while time.monotonic() < hold_until:
                self.sim.render()
                time.sleep(1.0 / 60.0)
        final = self.object.data.root_link_pos_w[0].clone()
        final_ref_index = min(max(self.reference_step - 1, 0), self.reference.length - 1)
        matched_reference = ref_positions[final_ref_index] + self.scene.env_origins[0]
        final_quat = self.object.data.root_link_quat_w[0]
        reference_quat = self.reference.data["body_quat_w"][
            final_ref_index, self.object_ref_body_id
        ].float()
        yaw_error = torch.atan2(
            torch.sin(
                2.0 * torch.atan2(final_quat[3], final_quat[0])
                - 2.0 * torch.atan2(reference_quat[3], reference_quat[0])
            ),
            torch.cos(
                2.0 * torch.atan2(final_quat[3], final_quat[0])
                - 2.0 * torch.atan2(reference_quat[3], reference_quat[0])
            ),
        ).abs()
        actual_displacement = final[:2] - initial[:2]
        stable = nonfinite == 0 and not terminated and min_height >= 0.45
        return PushBoxRolloutMetrics(
            case=self.case_name,
            steps=actual_steps,
            initial_box_position=initial.tolist(),
            final_box_position=final.tolist(),
            reference_initial_box_position=ref_initial.tolist(),
            reference_final_box_position=ref_final.tolist(),
            actual_xy_displacement=actual_displacement.tolist(),
            reference_xy_displacement=ref_delta_xy.tolist(),
            directional_progress=float(torch.dot(actual_displacement, direction)),
            max_directional_progress=max_progress,
            path_progress=actual_path_length,
            reference_path_length=reference_path_length,
            final_tracking_error=float(torch.linalg.vector_norm(final - matched_reference)),
            final_yaw_error=float(yaw_error),
            max_left_wrist_contact_force=max_wrist[0],
            max_right_wrist_contact_force=max_wrist[1],
            mean_left_wrist_contact_force=wrist_sum[0] / max(actual_steps, 1),
            mean_right_wrist_contact_force=wrist_sum[1] / max(actual_steps, 1),
            contact_active_fraction=both_contact_steps / max(expected_contact_steps, 1),
            min_root_height=min_height,
            mean_support_contact_count=support_sum / max(actual_steps, 1),
            max_contact_force=max_contact,
            max_abs_action=max_action,
            mean_abs_action=action_abs_sum / max(action_count, 1),
            nonfinite_count=nonfinite,
            terminated=terminated,
            termination_reason=reason,
            stable=stable,
            zero_hook_exact=torch.equal(
                self.action_runtime.received_action, self.action_runtime.last_a_nom
            ),
            settings={
                "box_mass": self.box_mass,
                "box_friction": self.box_friction,
                "box_com_offset": self.box_com_offset,
                "initial_object_xy": self.initial_object_xy,
                "initial_object_yaw": self.initial_object_yaw,
                "contact_target_offset": self.contact_target_offset,
                "delay": int(self.action_runtime.delay[0, 0]),
                "alpha": float(self.action_runtime.alpha[0, 0]),
            },
        )

    def close(self) -> None:
        """Release Isaac Lab objects before SimulationApp shutdown."""

        del self.filtered_wrist_contacts
        if self.reference_object_marker is not None:
            del self.reference_object_marker
        del self.contacts
        if self.door is not None:
            del self.door
        if self.box is not None:
            del self.box
        del self.object
        del self.robot
        del self.scene
        self.sim.stop()
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()


class PretrainedHDMIDoorIsaacRuntime(PretrainedHDMIIsaacRuntime):
    """Backward-compatible door runtime name."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        if self.task_spec.task != "push_door_hand":
            raise ValueError("door runtime requires a push_door_hand artifact")


class PretrainedHDMIBoxIsaacRuntime(PretrainedHDMIIsaacRuntime):
    """Explicit push-box runtime name for downstream mismatch development."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        if self.task_spec.task != "push_box":
            raise ValueError("box runtime requires a push_box artifact")
