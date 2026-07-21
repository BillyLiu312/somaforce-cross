"""Standalone frozen HDMI privileged-teacher scaffold inference contract.

This module intentionally contains no HDMI or active_adaptation imports.  The
exported phase=train policy uses privileged simulator state and is therefore a
simulation baseline, not a deployable policy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import nn


CONTRACT_VERSION = "hdmi_push_door_hand_teacher_v1"
OBSERVATION_DIMS: dict[str, int] = {
    "command": 356,
    "policy": 249,
    "object": 7,
    "privileged": 1714,
    "reference_action": 23,
}

# Isaac articulation order used by the audited checkpoint.
HDMI_ACTION_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
)

HDMI_ACTION_SCALE: tuple[float, ...] = (
    0.55, 0.55, 0.55, 0.35, 0.35, 0.44, 0.55, 0.55, 0.44, 0.35, 0.35,
    0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44,
)

# Canonical HDMI motion order from reference/meta.json.
HDMI_REFERENCE_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

REFERENCE_TO_ACTION_INDICES: tuple[int, ...] = tuple(
    HDMI_REFERENCE_JOINT_NAMES.index(name) for name in HDMI_ACTION_JOINT_NAMES
)


@dataclass(frozen=True)
class HDMIObservationBatch:
    """Flat named teacher inputs before frozen VecNorm normalization."""

    command: torch.Tensor
    policy: torch.Tensor
    object: torch.Tensor
    privileged: torch.Tensor
    reference_action: torch.Tensor

    def validate(self) -> int:
        tensors = {
            "command": self.command,
            "policy": self.policy,
            "object": self.object,
            "privileged": self.privileged,
            "reference_action": self.reference_action,
        }
        batch_size: int | None = None
        for name, tensor in tensors.items():
            if tensor.ndim != 2 or tensor.shape[1] != OBSERVATION_DIMS[name]:
                raise ValueError(
                    f"{name} must have shape [B, {OBSERVATION_DIMS[name]}], "
                    f"got {tuple(tensor.shape)}"
                )
            if not tensor.is_floating_point():
                raise TypeError(f"{name} must be floating point, got {tensor.dtype}")
            if batch_size is None:
                batch_size = tensor.shape[0]
            elif tensor.shape[0] != batch_size:
                raise ValueError("all HDMI observation fields must share batch size")
        return int(batch_size or 0)


class FrozenHDMITeacherPolicy(nn.Module):
    """Dependency-free network equivalent to HDMI encoder_priv + actor mean."""

    def __init__(self) -> None:
        super().__init__()
        self.priv_fc = nn.Linear(1721, 256)
        self.priv_ln = nn.LayerNorm(256)
        self.priv_out = nn.Linear(256, 256)

        self.actor_fc1 = nn.Linear(861, 512)
        self.actor_ln1 = nn.LayerNorm(512)
        self.actor_fc2 = nn.Linear(512, 256)
        self.actor_ln2 = nn.LayerNorm(256)
        self.actor_fc3 = nn.Linear(256, 256)
        self.actor_ln3 = nn.LayerNorm(256)
        self.actor_mean = nn.Linear(256, 23)
        self.activation = nn.Mish()

        for name, dim in (
            ("command", 356),
            ("policy", 249),
            ("object", 7),
            ("privileged", 1714),
        ):
            self.register_buffer(f"{name}_mean", torch.zeros(dim))
            self.register_buffer(f"{name}_scale", torch.ones(dim))

    def _normalize(self, name: str, value: torch.Tensor) -> torch.Tensor:
        mean = getattr(self, f"{name}_mean")
        scale = getattr(self, f"{name}_scale")
        return (value - mean) / scale

    def forward(self, observation: HDMIObservationBatch) -> torch.Tensor:
        observation.validate()
        command = self._normalize("command", observation.command)
        policy = self._normalize("policy", observation.policy)
        object_obs = self._normalize("object", observation.object)
        privileged = self._normalize("privileged", observation.privileged)

        privileged_feature = self.priv_out(
            self.activation(self.priv_ln(self.priv_fc(torch.cat((privileged, object_obs), dim=-1))))
        )
        actor_input = torch.cat((command, policy, privileged_feature), dim=-1)
        feature = self.activation(self.actor_ln1(self.actor_fc1(actor_input)))
        feature = self.activation(self.actor_ln2(self.actor_fc2(feature)))
        feature = self.activation(self.actor_ln3(self.actor_fc3(feature)))
        return observation.reference_action + self.actor_mean(feature)


class PretrainedHDMIScaffold(nn.Module):
    """Frozen privileged simulation baseline exposing read-only ``a_nom``."""

    def __init__(self, policy: FrozenHDMITeacherPolicy, manifest: Mapping[str, object]):
        super().__init__()
        self.policy = policy
        self.manifest = dict(manifest)
        self.policy.requires_grad_(False)
        super().train(False)

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | torch.device = "cpu",
        verify_checksum: bool = True,
    ) -> "PretrainedHDMIScaffold":
        artifact_dir = Path(artifact_dir).expanduser().resolve()
        manifest_path = artifact_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing HDMI scaffold manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contract_version") != CONTRACT_VERSION:
            raise ValueError(
                f"unsupported HDMI contract {manifest.get('contract_version')!r}; "
                f"expected {CONTRACT_VERSION!r}"
            )
        policy_name = str(manifest["files"]["policy"]["path"])
        policy_path = artifact_dir / policy_name
        if verify_checksum:
            actual = sha256_file(policy_path)
            expected = str(manifest["files"]["policy"]["sha256"])
            if actual != expected:
                raise ValueError(f"policy checksum mismatch: expected {expected}, got {actual}")
        payload = torch.load(policy_path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != 1:
            raise ValueError("unsupported portable HDMI policy format")
        policy = FrozenHDMITeacherPolicy()
        policy.load_state_dict(payload["state_dict"], strict=True)
        policy.to(device=device)
        return cls(policy, manifest)

    def train(self, mode: bool = True) -> "PretrainedHDMIScaffold":
        if mode:
            raise RuntimeError("PretrainedHDMIScaffold is frozen and cannot enter train mode")
        super().train(False)
        return self

    @torch.inference_mode()
    def forward(self, observation: HDMIObservationBatch) -> torch.Tensor:
        # Clone prevents a downstream module from mutating the scaffold output in-place.
        return self.policy(observation).detach().clone()

    @torch.inference_mode()
    def nominal_action(self, observation: HDMIObservationBatch) -> torch.Tensor:
        """Read-only ``a_nom`` hook reserved for a future Cross wrapper."""

        return self.forward(observation)

    @property
    def action_joint_names(self) -> tuple[str, ...]:
        return HDMI_ACTION_JOINT_NAMES

    @property
    def is_privileged_simulation_baseline(self) -> bool:
        return True


class HDMIObservationHistory:
    """HDMI reset and history semantics for policy and privileged inputs."""

    policy_joint_steps = (0, 1, 2, 3, 4, 8)
    privileged_joint_steps = tuple(range(9))

    def __init__(self, num_envs: int, *, device: str | torch.device = "cpu") -> None:
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.root_ang_vel = torch.zeros(num_envs, 9, 3, device=self.device)
        self.projected_gravity = torch.zeros(num_envs, 9, 3, device=self.device)
        self.joint_pos = torch.zeros(num_envs, 9, 29, device=self.device)
        # HDMI layout is [B, action, newest-to-oldest], then flattened.
        self.previous_actions = torch.zeros(num_envs, 23, 3, device=self.device)
        self.reference_step = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.motion_length = torch.ones(num_envs, dtype=torch.long, device=self.device)

    def reset(
        self,
        env_ids: torch.Tensor,
        *,
        root_ang_vel: torch.Tensor,
        projected_gravity: torch.Tensor,
        joint_pos: torch.Tensor,
        motion_length: int | torch.Tensor,
    ) -> None:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        count = env_ids.numel()
        self.root_ang_vel[env_ids] = root_ang_vel.to(self.device).reshape(count, 1, 3)
        self.projected_gravity[env_ids] = projected_gravity.to(self.device).reshape(count, 1, 3)
        self.joint_pos[env_ids] = joint_pos.to(self.device).reshape(count, 1, 29)
        self.previous_actions[env_ids] = 0.0
        self.reference_step[env_ids] = 0
        lengths = torch.as_tensor(motion_length, device=self.device, dtype=torch.long)
        self.motion_length[env_ids] = lengths.expand(count)

    def update_state(
        self,
        root_ang_vel: torch.Tensor,
        projected_gravity: torch.Tensor,
        joint_pos: torch.Tensor,
    ) -> None:
        self.root_ang_vel = self.root_ang_vel.roll(1, dims=1)
        self.projected_gravity = self.projected_gravity.roll(1, dims=1)
        self.joint_pos = self.joint_pos.roll(1, dims=1)
        self.root_ang_vel[:, 0] = root_ang_vel
        self.projected_gravity[:, 0] = projected_gravity
        self.joint_pos[:, 0] = joint_pos

    def push_action(self, action: torch.Tensor) -> None:
        self.previous_actions[:, :, 1:] = self.previous_actions[:, :, :-1].clone()
        self.previous_actions[:, :, 0] = action

    def advance_reference(self) -> None:
        self.reference_step.add_(1)

    @property
    def phase(self) -> torch.Tensor:
        return (self.reference_step / self.motion_length).unsqueeze(1)

    def policy_observation(self) -> torch.Tensor:
        return torch.cat(
            (
                self.root_ang_vel[:, [0]].reshape(self.num_envs, -1),
                self.projected_gravity[:, [0]].reshape(self.num_envs, -1),
                self.joint_pos[:, self.policy_joint_steps].reshape(self.num_envs, -1),
                self.previous_actions.reshape(self.num_envs, -1),
            ),
            dim=-1,
        )


class HDMIJointPositionActionRuntime:
    """HDMI-compatible delay, smoothing, mapping, and position target runtime."""

    def __init__(
        self,
        default_joint_pos: torch.Tensor,
        articulation_joint_names: tuple[str, ...] | list[str],
        *,
        num_envs: int,
        decimation: int = 4,
        delay: int = 4,
        alpha: float = 0.9,
        device: str | torch.device | None = None,
    ) -> None:
        if not 2 <= delay <= 6:
            raise ValueError("audited HDMI delay must be within [2, 6] physics substeps")
        if not 0.8 <= alpha <= 1.0:
            raise ValueError("audited HDMI alpha must be within [0.8, 1.0]")
        self.device = torch.device(device or default_joint_pos.device)
        self.default_joint_pos = default_joint_pos.to(self.device)
        if self.default_joint_pos.shape != (num_envs, len(articulation_joint_names)):
            raise ValueError("default_joint_pos shape does not match articulation joints")
        self.joint_ids = torch.tensor(
            [articulation_joint_names.index(name) for name in HDMI_ACTION_JOINT_NAMES],
            device=self.device,
            dtype=torch.long,
        )
        self.scale = torch.tensor(HDMI_ACTION_SCALE, device=self.device)
        self.decimation = decimation
        self.delay = torch.full((num_envs, 1), delay, device=self.device, dtype=torch.long)
        self.alpha = torch.full((num_envs, 1), alpha, device=self.device)
        self.action_buffer = torch.zeros(num_envs, 23, 3, device=self.device)
        self.applied_action = torch.zeros(num_envs, 23, device=self.device)
        self.last_a_nom = torch.zeros_like(self.applied_action)
        self.received_action = torch.zeros_like(self.applied_action)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.action_buffer[env_ids] = 0.0
        self.applied_action[env_ids] = 0.0
        self.last_a_nom[env_ids] = 0.0
        self.received_action[env_ids] = 0.0

    def set_nominal_action(self, a_nom: torch.Tensor) -> None:
        if a_nom.shape != self.applied_action.shape:
            raise ValueError(f"a_nom must have shape {tuple(self.applied_action.shape)}")
        self.last_a_nom.copy_(a_nom)
        self.received_action.copy_(a_nom)
        if not torch.equal(self.received_action, self.last_a_nom):
            raise AssertionError("zero-hook violated: environment action differs from a_nom")

    def substep_target(self, substep: int) -> torch.Tensor:
        if not 0 <= substep < self.decimation:
            raise ValueError("substep is outside the control decimation")
        if substep == 0:
            self.action_buffer[:, :, 1:] = self.action_buffer[:, :, :-1].clone()
            self.action_buffer[:, :, 0] = self.received_action
        delayed_index = (self.delay - substep + self.decimation - 1) // self.decimation
        delayed = self.action_buffer.take_along_dim(delayed_index.unsqueeze(1), dim=-1)
        self.applied_action.lerp_(delayed.squeeze(-1), self.alpha)
        target = self.default_joint_pos.clone()
        target[:, self.joint_ids] += self.applied_action * self.scale
        return target


def reference_to_action(reference_joint_pos: torch.Tensor) -> torch.Tensor:
    """Select canonical 29-D reference joints in audited 23-D action order."""

    if reference_joint_pos.shape[-1] != 29:
        raise ValueError("canonical HDMI reference must have 29 robot joints")
    indices = torch.tensor(REFERENCE_TO_ACTION_INDICES, device=reference_joint_pos.device)
    return reference_joint_pos.index_select(-1, indices)


def reference_action(
    reference_joint_pos: torch.Tensor,
    default_joint_pos_action_order: torch.Tensor,
) -> torch.Tensor:
    selected = reference_to_action(reference_joint_pos)
    scale = torch.tensor(HDMI_ACTION_SCALE, device=selected.device, dtype=selected.dtype)
    return (selected - default_joint_pos_action_order) / scale


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
