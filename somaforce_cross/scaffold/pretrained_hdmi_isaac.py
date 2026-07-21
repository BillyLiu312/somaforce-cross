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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

from somaforce_cross.scaffold.pretrained_hdmi import (
    HDMI_ACTION_JOINT_NAMES,
    HDMIJointPositionActionRuntime,
    HDMIObservationBatch,
    HDMIObservationHistory,
    PretrainedHDMIScaffold,
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


def make_scene_cfg(artifact_dir: Path, num_envs: int) -> InteractiveSceneCfg:
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
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.76)),
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

    @configclass
    class HDMIDoorSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(
            prim_path="/World/ground",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
            spawn=sim_utils.CuboidCfg(
                size=(100.0, 100.0, 0.1),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.18, 0.20, 0.22),
                ),
            ),
        )
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        )
        robot: ArticulationCfg = robot_cfg
        door: ArticulationCfg = door_cfg
        contacts = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=3,
            track_air_time=True,
        )

    return HDMIDoorSceneCfg(num_envs=num_envs, env_spacing=5.0)


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


class PretrainedHDMIDoorIsaacRuntime:
    """Minimal standalone G1 + articulated door rollout environment."""

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
    ) -> None:
        self.sim = sim
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        self.device = torch.device(sim.device)
        self.num_envs = num_envs
        self.scene = InteractiveScene(make_scene_cfg(self.artifact_dir, num_envs))
        self.sim.reset()
        self.scene.update(self.physics_dt)
        self.robot = self.scene["robot"]
        self.door = self.scene["door"]
        self.contacts = self.scene["contacts"]
        self.reference = HDMIMotionReference(self.artifact_dir, self.device)
        self.scaffold = PretrainedHDMIScaffold.from_artifact(
            self.artifact_dir, device=self.device
        )
        self.history = HDMIObservationHistory(num_envs, device=self.device)
        self.action_runtime = HDMIJointPositionActionRuntime(
            self.robot.data.default_joint_pos,
            self.robot.joint_names,
            num_envs=num_envs,
            decimation=self.decimation,
            delay=delay,
            alpha=alpha,
            device=self.device,
        )
        self.door_friction = float(door_friction)
        self.door_damping = float(door_damping)
        self.reference_step = 0
        self._resolve_indices()
        self.reset()

    def _resolve_indices(self) -> None:
        self.ref_robot_joint_ids = torch.tensor(
            [self.reference.joint_names.index(name) for name in self.robot.joint_names],
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
        self.door_ref_body_id = self.reference.body_names.index("door")
        self.door_ref_joint_id = self.reference.joint_names.index("door_joint")
        self.door_panel_body_id = self.door.body_names.index("door_panel")
        self.right_wrist_body_id = self.robot.body_names.index("right_wrist_yaw_link")
        self.ankle_body_ids = torch.tensor(
            [self.robot.body_names.index("left_ankle_roll_link"), self.robot.body_names.index("right_ankle_roll_link")],
            device=self.device,
        )
        self.height_body_ids = torch.tensor(
            [self.robot.body_names.index(name) for name in ("left_ankle_roll_link", "right_ankle_roll_link", "pelvis", "torso_link")],
            device=self.device,
        )
        self.contact_wrist_id = self.contacts.body_names.index("right_wrist_yaw_link")
        self.contact_support_ids = torch.tensor(
            [
                self.contacts.body_names.index("left_ankle_roll_link"),
                self.contacts.body_names.index("right_ankle_roll_link"),
            ],
            device=self.device,
        )

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
        joint_pos = self.reference.data["joint_pos"][[0], self.ref_robot_joint_ids].float().expand(self.num_envs, -1).clone()
        joint_vel = self.reference.data["joint_vel"][[0], self.ref_robot_joint_ids].float().expand(self.num_envs, -1).clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)

        door_root = torch.cat(
            (
                self.reference.data["body_pos_w"][[0], self.door_ref_body_id].float().expand(self.num_envs, -1) + env_origin,
                self.reference.data["body_quat_w"][[0], self.door_ref_body_id].float().expand(self.num_envs, -1),
            ),
            dim=-1,
        )
        self.door.write_root_link_pose_to_sim(door_root)
        self.door.write_root_com_velocity_to_sim(torch.zeros(self.num_envs, 6, device=self.device))
        door_joint = self.reference.data["joint_pos"][[0], self.door_ref_joint_id].float().expand(self.num_envs, 1).clone()
        door_joint_vel = self.reference.data["joint_vel"][[0], self.door_ref_joint_id].float().expand(self.num_envs, 1).clone()
        self.door.write_joint_state_to_sim(door_joint, door_joint_vel)
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

        door_root_pos = self.door.data.root_link_pos_w
        door_root_quat = self.door.data.root_link_quat_w
        door_panel_pos = self.door.data.body_link_pos_w[:, self.door_panel_body_id]
        door_panel_quat = self.door.data.body_link_quat_w[:, self.door_panel_body_id]
        target_offset = torch.tensor([0.0, -0.6, 1.0], device=self.device).expand(self.num_envs, -1)
        contact_target = door_panel_pos + quat_apply(door_panel_quat, target_offset)
        wrist_pos = self.robot.data.body_link_pos_w[:, self.right_wrist_body_id]
        wrist_quat = self.robot.data.body_link_quat_w[:, self.right_wrist_body_id]
        eef_offset = torch.tensor([0.05, 0.0, 0.0], device=self.device).expand(self.num_envs, -1)
        eef_pos = wrist_pos + quat_apply(wrist_quat, eef_offset)
        ref_door_pos = future["body_pos_w"][:, :, self.door_ref_body_id] + self.scene.env_origins[:, None]
        ref_door_quat = future["body_quat_w"][:, :, self.door_ref_body_id]
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
                quat_apply_inverse(g["robot_root_yaw"], g["contact_target"] - g["robot_root_pos"]),
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
        privileged = torch.cat(
            (
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
                quat_apply_inverse(g["robot_root_quat"], g["contact_target"] - g["eef_pos"]),
                self.door.data.joint_pos,
                self.door.data.joint_vel,
                self.door.data.applied_torque,
            ),
            dim=-1,
        )
        current_ref = future["joint_pos"][:, 0, :29]
        default_action_pos = self.robot.data.default_joint_pos[:, self.action_runtime.joint_ids]
        ref_action = reference_action(current_ref, default_action_pos)
        observation = HDMIObservationBatch(
            command=command,
            policy=self.history.policy_observation(),
            object=object_obs,
            privileged=privileged,
            reference_action=ref_action,
        )
        observation.validate()
        return observation

    def step(self) -> torch.Tensor:
        observation = self.build_observation()
        a_nom = self.scaffold(observation)
        self.action_runtime.set_nominal_action(a_nom)
        self.history.push_action(a_nom)
        for substep in range(self.decimation):
            target = self.action_runtime.substep_target(substep)
            self.robot.set_joint_position_target(target)
            door_vel = self.door.data.joint_vel[:, 0]
            friction = -torch.sign(door_vel) * (door_vel.abs() > 0.01) * self.door_friction
            door_effort = friction - door_vel * self.door_damping
            self.door.set_joint_effort_target(door_effort.unsqueeze(1))
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
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
    ) -> RolloutMetrics:
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

    def close(self) -> None:
        """Release Isaac Lab objects before SimulationApp shutdown."""

        del self.contacts
        del self.door
        del self.robot
        del self.scene
        self.sim.stop()
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()
