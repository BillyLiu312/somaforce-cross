"""Shared rigid-object plumbing for the three Phase 4B4 HDMI tasks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from somaforce_cross.envs.task_adapter import (
    TaskProgressSignals,
    rigid_nominal_physics_mismatch,
    rigid_object_state,
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


class RigidObjectTaskAdapter:
    """Common scene, reference, scaffold, contact, and critic plumbing."""

    object_kind = "rigid_object"

    def __init__(
        self,
        *,
        artifact_dir: str | Path,
        scene: object,
        robot: object,
        rigid_object: object,
        contacts: object,
        filtered_wrist_contacts: Sequence[object],
        task_spec: HDMITaskSpec,
        scaffold_history: HDMIObservationHistory,
        action_runtime: object,
        source_reset: Callable[[torch.Tensor], None],
    ) -> None:
        if task_spec.object_kind != self.object_kind:
            raise ValueError("RigidObjectTaskAdapter requires a rigid-object contract")
        if len(filtered_wrist_contacts) != 2:
            raise ValueError("rigid tasks require left/right filtered contact sensors")
        self.task_name = task_spec.task
        self.artifact_dir = Path(artifact_dir).expanduser().resolve()
        self.scene = scene
        self.robot = robot
        self.object = rigid_object
        self.contacts = contacts
        self.filtered_wrist_contacts = tuple(filtered_wrist_contacts)
        self.task_spec = task_spec
        self.history = scaffold_history
        self.action_runtime = action_runtime
        self.source_reset = source_reset
        self.device = self.robot.data.joint_pos.device
        self.num_envs = self.robot.data.joint_pos.shape[0]
        self.reference = HDMIMotionReference(self.artifact_dir, self.device)
        self.reference_step = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._resolve_indices()

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
        if len(set(self.task_spec.action_joint_names)) != len(
            self.task_spec.action_joint_names
        ):
            raise ValueError("artifact action joint names must be unique")
        self.root_ref_body_id = self.reference.body_names.index("pelvis")
        self.object_ref_body_id = self.reference.body_names.index(
            self.task_spec.object_asset_name
        )
        self.object_body_id = self.object.body_names.index(
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
        self._reset_progress(env_ids)
        self.source_reset(env_ids)

    def _reset_progress(self, env_ids: torch.Tensor) -> None:
        raise NotImplementedError

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
        if key == "joint_pos":
            for name, value in self.task_spec.robot_initial_joint_overrides.items():
                result[:, self.robot.joint_names.index(name)] = value
        return result

    def write_scene_reset(self, env_ids: torch.Tensor) -> None:
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

        object_pose = torch.cat(
            (
                self.reference.data["body_pos_w"][[0], self.object_ref_body_id]
                .float()
                .expand(count, -1)
                + origins,
                self.reference.data["body_quat_w"][[0], self.object_ref_body_id]
                .float()
                .expand(count, -1),
            ),
            dim=-1,
        )
        self.object.write_root_link_pose_to_sim(object_pose, env_ids=env_ids)
        self.object.write_root_com_velocity_to_sim(
            torch.zeros(count, 6, device=self.device), env_ids=env_ids
        )

    def advance(self) -> None:
        self._advance_progress()
        self.reference_step.add_(1)

    def _advance_progress(self) -> None:
        raise NotImplementedError

    def apply_object_action(self) -> None:
        return None

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

        object_root_pos = self.object.data.root_link_pos_w
        object_root_quat = self.object.data.root_link_quat_w
        object_body_pos = self.object.data.body_link_pos_w[:, self.object_body_id]
        object_body_quat = self.object.data.body_link_quat_w[:, self.object_body_id]
        target_offsets = (
            torch.tensor(
                self.task_spec.contact_target_offsets,
                device=self.device,
                dtype=object_body_pos.dtype,
            )
            .unsqueeze(0)
            .expand(self.num_envs, -1, -1)
        )
        contact_target = object_body_pos[:, None] + quat_apply(
            object_body_quat[:, None], target_offsets
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
        ref_object_pos = (
            future["body_pos_w"][:, :, self.object_ref_body_id]
            + self.scene.env_origins[:, None]
        )
        ref_object_quat = future["body_quat_w"][:, :, self.object_ref_body_id]
        return locals()

    def build_scaffold_observation(self) -> HDMIObservationBatch:
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
        object_yaw = yaw_quat(g["object_root_quat"])
        relative_yaw = quat_mul(quat_conjugate(g["robot_root_yaw"]), object_yaw)
        relative_yaw_angle = 2.0 * torch.atan2(relative_yaw[:, 3], relative_yaw[:, 0])
        object_obs = torch.cat(
            (
                quat_apply_inverse(
                    g["robot_root_yaw"],
                    g["object_root_pos"] - g["robot_root_pos"],
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
        object_quat_diff = quat_mul(
            quat_conjugate(g["object_root_quat"])[:, None],
            g["ref_object_quat"],
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
                    g["object_root_pos"] - g["robot_root_pos"],
                ),
                quat_to_matrix(
                    quat_mul(
                        quat_conjugate(g["robot_root_quat"]),
                        g["object_root_quat"],
                    )
                ).reshape(self.num_envs, -1),
                quat_apply_inverse(
                    g["object_root_quat"][:, None],
                    g["ref_object_pos"] - g["object_root_pos"][:, None],
                ).reshape(self.num_envs, -1),
                quat_to_matrix(object_quat_diff).reshape(self.num_envs, -1),
                future["object_contact"].reshape(self.num_envs, -1),
                quat_apply_inverse(
                    g["robot_root_quat"][:, None],
                    g["contact_target"] - g["eef_pos"],
                ).reshape(self.num_envs, -1),
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
        yaw = yaw_quat(self.robot.data.root_link_quat_w)
        linear = self.robot.data.body_com_lin_vel_w.index_select(
            1, self.contact_eef_body_ids
        )
        angular = self.robot.data.body_com_ang_vel_w.index_select(
            1, self.contact_eef_body_ids
        )
        return torch.cat(
            (
                quat_apply_inverse(yaw[:, None], linear),
                quat_apply_inverse(yaw[:, None], angular),
            ),
            dim=-1,
        )

    def expected_contact(self) -> torch.Tensor:
        frame = self.reference_step.clamp_max(self.reference.length - 1)
        source = self.reference.data["object_contact"].index_select(0, frame)
        scalar = source.reshape(self.num_envs, -1).bool().any(dim=-1)
        return scalar[:, None].expand(-1, 2)

    def contact_truth(self) -> torch.Tensor:
        results: list[torch.Tensor] = []
        for sensor in self.filtered_wrist_contacts:
            matrix = sensor.data.force_matrix_w
            if matrix is None:
                results.append(
                    torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
                )
            else:
                norm = torch.linalg.vector_norm(matrix, dim=-1).reshape(
                    self.num_envs, -1
                )
                results.append(norm.amax(dim=-1) > 1.0)
        return torch.stack(results, dim=-1)

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
        result = rigid_object_state(
            self.robot.data.root_link_pos_w,
            self.robot.data.root_link_quat_w,
            self.object.data.root_link_pos_w,
            self.object.data.root_link_quat_w,
            self.object.data.root_link_lin_vel_w,
            self.object.data.root_link_ang_vel_w,
        )
        if result.shape != (self.num_envs, 16):
            raise AssertionError("rigid object_state must have shape [B,16]")
        if torch.count_nonzero(result[:, 13:]) != 0:
            raise AssertionError("rigid mechanism slots must be bitwise zero")
        return result

    def nominal_physics_row(self) -> torch.Tensor:
        view = self.object.root_physx_view
        masses = view.get_masses().to(device=self.device, dtype=torch.float32)
        masses = masses.reshape(self.num_envs, -1)
        inertias = view.get_inertias().to(device=self.device, dtype=torch.float32)
        inertias = inertias.reshape(self.num_envs, -1, 9)
        materials = view.get_material_properties().to(
            device=self.device, dtype=torch.float32
        )
        materials = materials.reshape(self.num_envs, -1, 3)
        if masses.shape[1] != 1 or inertias.shape[1] != 1:
            raise ValueError("rigid task must expose exactly one applied mass/inertia")
        static = materials[..., 0]
        dynamic = materials[..., 1]
        if not (
            torch.equal(static, static[:, :1].expand_as(static))
            and torch.equal(dynamic, dynamic[:, :1].expand_as(dynamic))
        ):
            raise ValueError("all rigid collision shapes must share one friction")
        return rigid_nominal_physics_mismatch(
            masses[:, 0],
            inertias[:, 0],
            static[:, 0],
            dynamic[:, 0],
        )

    def progress_signals(self) -> TaskProgressSignals:
        raise NotImplementedError
