"""Door-only Isaac adapter for the frozen HDMI simulation scaffold."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch

from somaforce_cross.envs.task_adapter import (
    TaskProgressSignals,
    adapter_contact_target_offsets,
    door_custom_effort,
    door_nominal_physics_mismatch,
    quat_mul_wxyz,
    quat_to_rotvec_wxyz,
    rotvec_to_quat_wxyz,
    strict_root_height_failure,
    validate_door_runtime_physics,
)
from somaforce_cross.scaffold.contracts import G1_FULL_JOINT_NAMES
from somaforce_cross.scaffold.pretrained_hdmi import (
    HDMIObservationBatch,
    HDMIObservationHistory,
    HDMITaskSpec,
    reference_action,
)
from somaforce_cross.scaffold.pretrained_hdmi_isaac import (
    FUTURE_STEPS,
    TRACKING_BODY_NAMES,
    HDMIMotionReference,
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
    quat_mul,
    quat_to_matrix,
    yaw_quat,
)


class PushDoorHandTaskAdapter:
    """Map one articulated door scene into the shared Phase 4 schemas."""

    task_name = "push_door_hand"
    object_kind = "articulation"
    root_height_failure = 0.45

    def __init__(
        self,
        *,
        artifact_dir: str | Path,
        scene: object,
        robot: object,
        door: object,
        contacts: object,
        task_spec: HDMITaskSpec,
        scaffold_history: HDMIObservationHistory,
        action_runtime: object,
        source_reset: Callable[[torch.Tensor], None],
        mechanism_friction: float,
        mechanism_damping: float,
    ) -> None:
        if task_spec.task != self.task_name or task_spec.object_kind != "articulation":
            raise ValueError(
                "PushDoorHandTaskAdapter requires the door artifact contract"
            )
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        self.scene = scene
        self.robot = robot
        self.door = door
        self.object = door
        self.contacts = contacts
        self.task_spec = task_spec
        self.history = scaffold_history
        self.action_runtime = action_runtime
        self.source_reset = source_reset
        self.mechanism_friction = float(mechanism_friction)
        self.mechanism_damping = float(mechanism_damping)
        self.device = self.robot.data.joint_pos.device
        self.num_envs = self.robot.data.joint_pos.shape[0]
        self.reference = HDMIMotionReference(self.artifact_dir, self.device)
        self.reference_step = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._resolve_indices()
        initial = float(self.reference.data["joint_pos"][0, self.door_ref_joint_id])
        final = float(self.reference.data["joint_pos"][-1, self.door_ref_joint_id])
        self.progress_direction = 1.0 if final >= initial else -1.0
        self.target_progress = abs(final - initial)
        self.initial_door_joint = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.previous_progress = torch.zeros_like(self.initial_door_joint)
        self.current_progress = torch.zeros_like(self.initial_door_joint)
        self.mechanism_friction_buffer = torch.full_like(
            self.initial_door_joint, self.mechanism_friction
        )
        self.mechanism_damping_buffer = torch.full_like(
            self.initial_door_joint, self.mechanism_damping
        )
        nominal_offsets = torch.tensor(
            self.task_spec.contact_target_offsets,
            device=self.device,
            dtype=torch.float32,
        )
        self._nominal_contact_target_offsets = nominal_offsets
        self.contact_target_offsets = nominal_offsets.expand(
            self.num_envs, -1, -1
        ).clone()
        self.contact_target_translation = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float32
        )
        self.contact_target_rotvec = torch.zeros_like(self.contact_target_translation)
        self._runtime_nominal_door_pose = torch.zeros(
            self.num_envs, 7, device=self.device, dtype=torch.float32
        )
        self._runtime_nominal_robot_pose = torch.zeros_like(
            self._runtime_nominal_door_pose
        )
        self._runtime_nominal_door_pose[:, 3] = 1.0
        self._runtime_nominal_robot_pose[:, 3] = 1.0

    def _resolve_indices(self) -> None:
        shared_joint_names = tuple(
            name
            for name in self.robot.joint_names
            if name in self.reference.joint_names
        )
        self.ref_shared_joint_ids = torch.tensor(
            [self.reference.joint_names.index(name) for name in shared_joint_names],
            device=self.device,
        )
        self.robot_shared_joint_ids = torch.tensor(
            [self.robot.joint_names.index(name) for name in shared_joint_names],
            device=self.device,
        )
        self.full_joint_ids = torch.tensor(
            [self.robot.joint_names.index(name) for name in G1_FULL_JOINT_NAMES],
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
            [
                self.reference.joint_names.index(name)
                for name in self.task_spec.action_joint_names
            ],
            device=self.device,
        )
        self.root_ref_body_id = self.reference.body_names.index("pelvis")
        self.door_ref_body_id = self.reference.body_names.index(
            self.task_spec.object_asset_name
        )
        self.door_ref_joint_id = self.reference.joint_names.index("door_joint")
        self.door_panel_body_id = self.door.body_names.index(
            self.task_spec.object_body_name
        )
        self.contact_eef_body_ids = torch.tensor(
            [
                self.robot.body_names.index(name)
                for name in self.task_spec.contact_eef_names
            ],
            device=self.device,
        )
        self.ankle_body_ids = torch.tensor(
            [
                self.robot.body_names.index("left_ankle_roll_link"),
                self.robot.body_names.index("right_ankle_roll_link"),
            ],
            device=self.device,
        )
        self.height_body_ids = torch.tensor(
            [
                self.robot.body_names.index(name)
                for name in (
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "pelvis",
                    "torso_link",
                )
            ],
            device=self.device,
        )
        self.contact_wrist_ids = torch.tensor(
            [
                self.contacts.body_names.index("left_wrist_yaw_link"),
                self.contacts.body_names.index("right_wrist_yaw_link"),
            ],
            device=self.device,
        )
        self.contact_support_ids = torch.tensor(
            [
                self.contacts.body_names.index("left_ankle_roll_link"),
                self.contacts.body_names.index("right_ankle_roll_link"),
            ],
            device=self.device,
        )

    def validate_reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.ndim != 1 or env_ids.dtype != torch.long:
            raise TypeError("env_ids must be a one-dimensional int64 tensor")
        if env_ids.device != self.device:
            raise ValueError("env_ids must be on the adapter device")

    def reset(self, env_ids: torch.Tensor) -> None:
        self.validate_reset(env_ids)
        self.reference_step[env_ids] = 0
        door_joint = self.door.data.joint_pos[env_ids, 0]
        self.initial_door_joint[env_ids] = door_joint
        self.previous_progress[env_ids] = 0.0
        self.current_progress[env_ids] = 0.0
        self.source_reset(env_ids)

    def _reference_robot_joint_state(
        self, key: str, env_ids: torch.Tensor, frame: int = 0
    ) -> torch.Tensor:
        count = env_ids.numel()
        result = torch.zeros(
            count,
            len(self.robot.joint_names),
            device=self.device,
            dtype=torch.float32,
        )
        source = self.reference.data[key][[frame]].float().expand(count, -1)
        result[:, self.robot_shared_joint_ids] = source[:, self.ref_shared_joint_ids]
        for name, value in self.task_spec.robot_initial_joint_overrides.items():
            if key == "joint_pos":
                result[:, self.robot.joint_names.index(name)] = value
        return result

    def write_scene_reset(self, env_ids: torch.Tensor) -> None:
        """Write the same selected frame-zero state used by the golden runtime."""
        self.validate_reset(env_ids)
        count = env_ids.numel()
        origins = self.scene.env_origins[env_ids]
        root_pose = torch.cat(
            (
                self.reference.data["body_pos_w"][[0], self.root_ref_body_id]
                .float()
                .expand(count, -1)
                + origins,
                self.reference.data["body_quat_w"][[0], self.root_ref_body_id]
                .float()
                .expand(count, -1),
            ),
            dim=-1,
        )
        root_velocity = torch.cat(
            (
                self.reference.data["body_lin_vel_w"][[0], self.root_ref_body_id]
                .float()
                .expand(count, -1),
                self.reference.data["body_ang_vel_w"][[0], self.root_ref_body_id]
                .float()
                .expand(count, -1),
            ),
            dim=-1,
        )
        self.robot.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_com_velocity_to_sim(root_velocity, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(
            self._reference_robot_joint_state("joint_pos", env_ids),
            self._reference_robot_joint_state("joint_vel", env_ids),
            env_ids=env_ids,
        )

        door_pose = torch.cat(
            (
                self.reference.data["body_pos_w"][[0], self.door_ref_body_id]
                .float()
                .expand(count, -1)
                + origins,
                self.reference.data["body_quat_w"][[0], self.door_ref_body_id]
                .float()
                .expand(count, -1),
            ),
            dim=-1,
        )
        self.door.write_root_link_pose_to_sim(door_pose, env_ids=env_ids)
        self.door.write_root_com_velocity_to_sim(
            torch.zeros(count, 6, device=self.device), env_ids=env_ids
        )
        door_joint = (
            self.reference.data["joint_pos"][[0], self.door_ref_joint_id]
            .float()
            .expand(count, 1)
            .clone()
        )
        door_joint_vel = (
            self.reference.data["joint_vel"][[0], self.door_ref_joint_id]
            .float()
            .expand(count, 1)
            .clone()
        )
        self.door.write_joint_state_to_sim(door_joint, door_joint_vel, env_ids=env_ids)

    def validate_runtime_parameters(
        self, env_ids: torch.Tensor, physics_row: torch.Tensor
    ) -> None:
        self.validate_reset(env_ids)
        validate_door_runtime_physics(env_ids, physics_row, device=self.device)

    @staticmethod
    def _from_physx_xyzw(pose: torch.Tensor) -> torch.Tensor:
        return torch.cat((pose[:, :3], pose[:, [6, 3, 4, 5]]), dim=-1).to(
            dtype=torch.float32
        )

    def _root_pose_readback(self, asset: object, env_ids: torch.Tensor) -> torch.Tensor:
        pose = asset.root_physx_view.get_root_transforms().to(self.device)
        return self._from_physx_xyzw(pose.index_select(0, env_ids.to(pose.device)))

    def apply_runtime_parameters(
        self, env_ids: torch.Tensor, physics_row: torch.Tensor
    ) -> None:
        self.validate_runtime_parameters(env_ids, physics_row)
        if env_ids.numel() == 0:
            return
        door_nominal = self._root_pose_readback(self.door, env_ids)
        robot_nominal = self._root_pose_readback(self.robot, env_ids)
        self._runtime_nominal_door_pose[env_ids] = door_nominal
        self._runtime_nominal_robot_pose[env_ids] = robot_nominal

        door_pose = door_nominal.clone()
        door_pose[:, :3] += physics_row[:, 22:25]
        door_pose[:, 3:] = quat_mul_wxyz(
            rotvec_to_quat_wxyz(physics_row[:, 25:28]), door_nominal[:, 3:]
        )
        robot_pose = robot_nominal.clone()
        robot_pose[:, :2] += physics_row[:, 28:30]
        stance_rotvec = torch.zeros_like(physics_row[:, 25:28])
        stance_rotvec[:, 2] = physics_row[:, 30]
        robot_pose[:, 3:] = quat_mul_wxyz(
            rotvec_to_quat_wxyz(stance_rotvec), robot_nominal[:, 3:]
        )
        self.door.write_root_link_pose_to_sim(door_pose, env_ids=env_ids)
        self.robot.write_root_link_pose_to_sim(robot_pose, env_ids=env_ids)
        self.mechanism_friction_buffer[env_ids] = physics_row[:, 3]
        self.mechanism_damping_buffer[env_ids] = physics_row[:, 4]
        self.contact_target_translation[env_ids] = physics_row[:, 31:34]
        self.contact_target_rotvec[env_ids] = physics_row[:, 34:37]
        self.contact_target_offsets[env_ids] = adapter_contact_target_offsets(
            self._nominal_contact_target_offsets, physics_row
        )

    def runtime_parameter_readback(self, env_ids: torch.Tensor) -> torch.Tensor:
        self.validate_reset(env_ids)
        door_pose = self._root_pose_readback(self.door, env_ids)
        robot_pose = self._root_pose_readback(self.robot, env_ids)
        door_nominal = self._runtime_nominal_door_pose[env_ids]
        robot_nominal = self._runtime_nominal_robot_pose[env_ids]
        result = torch.zeros(
            env_ids.numel(), 39, device=self.device, dtype=torch.float32
        )
        result[:, 3] = self.mechanism_friction_buffer[env_ids]
        result[:, 4] = self.mechanism_damping_buffer[env_ids]
        result[:, 22:25] = door_pose[:, :3] - door_nominal[:, :3]
        result[:, 25:28] = quat_to_rotvec_wxyz(
            quat_mul_wxyz(
                door_pose[:, 3:],
                torch.cat((door_nominal[:, 3:4], -door_nominal[:, 4:]), dim=-1),
            )
        )
        result[:, 28:30] = robot_pose[:, :2] - robot_nominal[:, :2]
        robot_delta = quat_mul_wxyz(
            robot_pose[:, 3:],
            torch.cat((robot_nominal[:, 3:4], -robot_nominal[:, 4:]), dim=-1),
        )
        result[:, 30] = quat_to_rotvec_wxyz(robot_delta)[:, 2]
        result[:, 31:34] = self.contact_target_translation[env_ids]
        result[:, 34:37] = self.contact_target_rotvec[env_ids]
        return result

    def advance(self) -> None:
        self.previous_progress.copy_(self.current_progress)
        door_joint = self.door.data.joint_pos[:, 0]
        self.current_progress.copy_(
            self.progress_direction * (door_joint - self.initial_door_joint)
        )
        self.reference_step.add_(1)

    def apply_object_action(self) -> None:
        door_velocity = self.door.data.joint_vel[:, 0]
        effort = door_custom_effort(
            door_velocity,
            self.mechanism_friction_buffer,
            self.mechanism_damping_buffer,
        )
        self.door.set_joint_effort_target(effort.unsqueeze(1))

    def _future_reference(self) -> dict[str, torch.Tensor]:
        offsets = torch.tensor(FUTURE_STEPS, device=self.device, dtype=torch.long)
        frames = (self.reference_step[:, None] + offsets[None]).clamp_max(
            self.reference.length - 1
        )
        result: dict[str, torch.Tensor] = {}
        for key, value in self.reference.data.items():
            selected = value.index_select(0, frames.reshape(-1)).float()
            result[key] = selected.reshape(
                self.num_envs, len(FUTURE_STEPS), *value.shape[1:]
            )
        return result

    def _geometry(self, future: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        ref_root_pos = future["body_pos_w"][:, 0, self.root_ref_body_id]
        ref_root_quat = future["body_quat_w"][:, 0, self.root_ref_body_id]
        ref_root_pos_ground = ref_root_pos.clone()
        ref_root_pos_ground[:, 2] = 0.0
        ref_root_yaw = yaw_quat(ref_root_quat)
        ref_body_pos = future["body_pos_w"].index_select(2, self.tracking_ref_body_ids)
        ref_body_quat = future["body_quat_w"].index_select(
            2, self.tracking_ref_body_ids
        )
        ref_body_lin = future["body_lin_vel_w"].index_select(
            2, self.tracking_ref_body_ids
        )
        ref_body_ang = future["body_ang_vel_w"].index_select(
            2, self.tracking_ref_body_ids
        )
        ref_body_local = quat_apply_inverse(
            ref_root_yaw[:, None, None],
            ref_body_pos - ref_root_pos_ground[:, None, None],
        )

        robot_root_pos = self.robot.data.root_link_pos_w
        robot_root_quat = self.robot.data.root_link_quat_w
        robot_root_pos_ground = robot_root_pos.clone()
        robot_root_pos_ground[:, 2] = 0.0
        robot_root_yaw = yaw_quat(robot_root_quat)
        robot_body_pos = self.robot.data.body_link_pos_w.index_select(
            1, self.tracking_robot_body_ids
        )
        robot_body_quat = self.robot.data.body_link_quat_w.index_select(
            1, self.tracking_robot_body_ids
        )
        robot_body_lin = self.robot.data.body_com_lin_vel_w.index_select(
            1, self.tracking_robot_body_ids
        )
        robot_body_ang = self.robot.data.body_com_ang_vel_w.index_select(
            1, self.tracking_robot_body_ids
        )
        robot_body_local = quat_apply_inverse(
            robot_root_yaw[:, None],
            robot_body_pos - robot_root_pos_ground[:, None],
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
        door_body_pos = self.door.data.body_link_pos_w[:, self.door_panel_body_id]
        door_body_quat = self.door.data.body_link_quat_w[:, self.door_panel_body_id]
        target_offsets = self.contact_target_offsets.to(dtype=door_body_pos.dtype)
        contact_target = door_body_pos[:, None] + quat_apply(
            door_body_quat[:, None], target_offsets
        )
        wrist_pos = self.robot.data.body_link_pos_w.index_select(
            1, self.contact_eef_body_ids
        )
        wrist_quat = self.robot.data.body_link_quat_w.index_select(
            1, self.contact_eef_body_ids
        )
        eef_offsets = (
            torch.tensor(
                self.task_spec.contact_eef_offsets,
                device=self.device,
                dtype=wrist_pos.dtype,
            )
            .unsqueeze(0)
            .expand(self.num_envs, -1, -1)
        )
        eef_pos = wrist_pos + quat_apply(wrist_quat, eef_offsets)
        ref_door_pos = (
            future["body_pos_w"][:, :, self.door_ref_body_id]
            + self.scene.env_origins[:, None]
        )
        ref_door_quat = future["body_quat_w"][:, :, self.door_ref_body_id]
        return locals()

    def build_scaffold_observation(self) -> HDMIObservationBatch:
        """Build the unchanged privileged-teacher input for frozen a_nom only."""
        future = self._future_reference()
        g = self._geometry(future)
        ref_joint_future = future["joint_pos"].index_select(
            2, self.ref_action_joint_ids
        )
        command = torch.cat(
            (
                g["ref_body_local"].reshape(self.num_envs, -1),
                ref_joint_future.reshape(self.num_envs, -1),
                self.history.phase,
            ),
            dim=-1,
        )
        door_yaw = yaw_quat(g["door_root_quat"])
        relative_yaw = quat_mul(quat_conjugate(g["robot_root_yaw"]), door_yaw)
        relative_yaw_angle = 2.0 * torch.atan2(relative_yaw[:, 3], relative_yaw[:, 0])
        object_obs = torch.cat(
            (
                quat_apply_inverse(
                    g["robot_root_yaw"],
                    g["door_root_pos"] - g["robot_root_pos"],
                )[:, :2],
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
            future["body_pos_w"][:, :, self.root_ref_body_id]
            + self.scene.env_origins[:, None]
            - g["robot_root_pos"][:, None],
        )
        ref_root_quat_b = quat_mul(
            quat_conjugate(g["robot_root_quat"])[:, None],
            future["body_quat_w"][:, :, self.root_ref_body_id],
        )
        ankle_pos = self.robot.data.body_link_pos_w.index_select(1, self.ankle_body_ids)
        ankle_vel = self.robot.data.body_com_lin_vel_w.index_select(
            1, self.ankle_body_ids
        )
        ankle_pos_b = quat_apply_inverse(
            g["robot_root_yaw"][:, None],
            ankle_pos - g["robot_root_pos_ground"][:, None],
        )
        ankle_vel_b = quat_apply_inverse(g["robot_root_quat"][:, None], ankle_vel)
        heights = self.robot.data.body_link_pos_w.index_select(1, self.height_body_ids)[
            ..., 2
        ]
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
                (g["ref_body_local"] - g["robot_body_local"][:, None]).reshape(
                    self.num_envs, -1
                ),
                quat_to_matrix(g["body_quat_diff"])[:, :, :, :2].reshape(
                    self.num_envs, -1
                ),
                (g["ref_lin_local"] - g["robot_lin_local"][:, None]).reshape(
                    self.num_envs, -1
                ),
                (g["ref_ang_local"] - g["robot_ang_local"][:, None]).reshape(
                    self.num_envs, -1
                ),
                self.robot.data.root_lin_vel_b,
                ankle_pos_b.reshape(self.num_envs, -1),
                ankle_vel_b.reshape(self.num_envs, -1),
                heights.reshape(self.num_envs, -1),
                self.action_runtime.applied_action,
                self.robot.data.applied_torque,
                quat_apply_inverse(
                    g["robot_root_quat"],
                    g["door_root_pos"] - g["robot_root_pos"],
                ),
                quat_to_matrix(
                    quat_mul(
                        quat_conjugate(g["robot_root_quat"]),
                        g["door_root_quat"],
                    )
                ).reshape(self.num_envs, -1),
                quat_apply_inverse(
                    g["door_root_quat"][:, None],
                    g["ref_door_pos"] - g["door_root_pos"][:, None],
                ).reshape(self.num_envs, -1),
                quat_to_matrix(door_quat_diff).reshape(self.num_envs, -1),
                future["object_contact"].reshape(self.num_envs, -1),
                quat_apply_inverse(
                    g["robot_root_quat"][:, None],
                    g["contact_target"] - g["eef_pos"],
                ).reshape(self.num_envs, -1),
                self.door.data.joint_pos,
                self.door.data.joint_vel,
                self.door.data.applied_torque,
            ),
            dim=-1,
        )
        current_ref = future["joint_pos"][:, 0]
        default_action_pos = self.robot.data.default_joint_pos[
            :, self.action_runtime.joint_ids
        ]
        observation = HDMIObservationBatch(
            command=command,
            policy=self.history.policy_observation(),
            object=object_obs,
            privileged=privileged,
            reference_action=reference_action(
                current_ref,
                default_action_pos,
                self.reference.joint_names,
                self.task_spec.action_joint_names,
                self.task_spec.action_scale,
            ),
        )
        observation.validate(self.task_spec.observation_dims)
        return observation

    def proprio(self) -> torch.Tensor:
        joint_pos = self.robot.data.joint_pos.index_select(1, self.full_joint_ids)
        default = self.robot.data.default_joint_pos.index_select(1, self.full_joint_ids)
        joint_vel = self.robot.data.joint_vel.index_select(1, self.full_joint_ids)
        return torch.cat(
            (
                self.robot.data.root_ang_vel_b,
                self.robot.data.projected_gravity_b,
                joint_pos - default,
                joint_vel,
            ),
            dim=-1,
        )

    def wrist_twist_base_yaw(self) -> torch.Tensor:
        wrist_ids = torch.tensor(
            [
                self.robot.body_names.index("left_wrist_yaw_link"),
                self.robot.body_names.index("right_wrist_yaw_link"),
            ],
            device=self.device,
        )
        yaw = yaw_quat(self.robot.data.root_link_quat_w)
        linear = self.robot.data.body_com_lin_vel_w.index_select(1, wrist_ids)
        angular = self.robot.data.body_com_ang_vel_w.index_select(1, wrist_ids)
        return torch.cat(
            (
                quat_apply_inverse(yaw[:, None], linear),
                quat_apply_inverse(yaw[:, None], angular),
            ),
            dim=-1,
        )

    def expected_contact(self) -> torch.Tensor:
        frame = self.reference_step.clamp_max(self.reference.length - 1)
        source = self.reference.data["object_contact"].index_select(0, frame).bool()
        result = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.bool)
        result[:, 1] = source[:, 0]
        return result

    def contact_truth(self) -> torch.Tensor:
        forces = self.contacts.data.net_forces_w.index_select(1, self.contact_wrist_ids)
        return torch.linalg.vector_norm(forces, dim=-1) > 1.0

    def support_contact_count(self) -> torch.Tensor:
        forces = self.contacts.data.net_forces_w.index_select(
            1, self.contact_support_ids
        )
        return (
            (torch.linalg.vector_norm(forces, dim=-1) > 1.0)
            .sum(dim=-1, keepdim=True)
            .float()
        )

    def build_object_state(self) -> torch.Tensor:
        robot_quat = self.robot.data.root_link_quat_w
        relative_position = quat_apply_inverse(
            robot_quat,
            self.door.data.root_link_pos_w - self.robot.data.root_link_pos_w,
        )
        relative_quat = quat_mul(
            quat_conjugate(robot_quat), self.door.data.root_link_quat_w
        )
        linear = quat_apply_inverse(robot_quat, self.door.data.root_link_lin_vel_w)
        angular = quat_apply_inverse(robot_quat, self.door.data.root_link_ang_vel_w)
        return torch.cat(
            (
                relative_position,
                relative_quat,
                linear,
                angular,
                self.door.data.joint_pos,
                self.door.data.joint_vel,
                self.door.data.applied_torque,
            ),
            dim=-1,
        )

    def nominal_physics_row(self) -> torch.Tensor:
        result = door_nominal_physics_mismatch(
            self.num_envs,
            mechanism_friction=0.0,
            mechanism_damping=0.0,
            device=self.device,
        )
        result[:, 3] = self.mechanism_friction_buffer
        result[:, 4] = self.mechanism_damping_buffer
        return result

    def progress_signals(self) -> TaskProgressSignals:
        expected = self.expected_contact()
        truth = self.contact_truth()
        root_height = self.robot.data.root_link_pos_w[:, 2]
        return TaskProgressSignals(
            progress=(self.current_progress / self.target_progress).unsqueeze(1),
            progress_delta=(
                (self.current_progress - self.previous_progress) / self.target_progress
            ).unsqueeze(1),
            success=(self.current_progress >= self.target_progress)
            .float()
            .unsqueeze(1),
            expected_contact=expected.float(),
            contact_truth=truth.float(),
            stability_margin=(root_height - self.root_height_failure).unsqueeze(1),
            failure=strict_root_height_failure(
                root_height, threshold=self.root_height_failure
            ),
            reference_exhausted=(
                self.reference_step + max(FUTURE_STEPS) >= self.reference.length
            ),
        )
