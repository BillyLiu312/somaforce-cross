"""Stateless fixed-field normalizers driven only by NumericContract values."""

from __future__ import annotations

from typing import Mapping

import torch

from somaforce_cross.envs.numeric_contract import NumericContract, TASKS
from somaforce_cross.sensing.virtual_ft import VirtualFTSensorParameters


def _finite_float32(value: torch.Tensor, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.device.type == "meta"
        or not torch.isfinite(value).all()
    ):
        raise ValueError(f"{name} must be finite torch.float32")
    return value


def _bounded(value: torch.Tensor, bound: float, name: str) -> torch.Tensor:
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} became nonfinite before clipping")
    result = torch.clamp(value, -bound, bound)
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} became nonfinite after clipping")
    return result


def _same_device_and_batch(
    fields: Mapping[str, torch.Tensor], batch: int, device: torch.device
) -> None:
    for name, value in fields.items():
        if value.shape[0] != batch or value.device != device:
            raise ValueError(f"{name} must share the policy batch and device")


class FixedFieldNormalizer:
    """No online statistics; checkpoint bytes depend only on the contract SHA."""

    def __init__(self, contract: NumericContract) -> None:
        self.contract = contract
        self.contract_hash = contract.sha256
        self.contract_version = str(contract.payload["contract_version"])
        self.actor = contract.payload["normalizer"]["actor"]
        self.critic = contract.payload["normalizer"]["critic"]
        self.final_clip = float(contract.payload["normalizer"]["final_continuous_clip"])

    def state_dict(self) -> dict[str, str]:
        return {
            "contract_hash": self.contract_hash,
            "contract_version": self.contract_version,
        }

    def load_state_dict(self, state: Mapping[str, str]) -> None:
        if dict(state) != self.state_dict():
            raise ValueError("normalizer state must exactly match numeric contract")

    def update(self, *_: object, **__: object) -> None:
        raise RuntimeError("Phase 4B5 fixed-field normalizers do not update")

    def normalize_actor(
        self, fields: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        required = {
            "a_nom_history",
            "previous_a_total",
            "proprio",
            "wrist_tokens",
            "z_cross",
        }
        if set(fields) != required:
            raise ValueError("actor normalizer received unknown or missing fields")
        wrist = _finite_float32(fields["wrist_tokens"], "wrist_tokens")
        proprio = _finite_float32(fields["proprio"], "proprio")
        history = _finite_float32(fields["a_nom_history"], "a_nom_history")
        previous = _finite_float32(fields["previous_a_total"], "previous_a_total")
        cross = _finite_float32(fields["z_cross"], "z_cross")
        batch = wrist.shape[0]
        if (
            wrist.shape != (batch, 2, 16, 14)
            or proprio.shape != (batch, 64)
            or history.shape != (batch, 23, 3)
            or previous.shape != (batch, 23)
            or cross.shape != (batch, 64)
        ):
            raise ValueError("actor field shapes do not match frozen schema")
        _same_device_and_batch(fields, batch, wrist.device)
        output_wrist = wrist.clone()
        output_wrist[..., :6] = _bounded(
            output_wrist[..., :6], float(self.actor["wrench_clip"]), "actor wrench"
        )
        output_wrist[..., 6:9] /= float(self.actor["linear_twist_divisor"])
        output_wrist[..., 9:12] /= float(self.actor["angular_twist_divisor"])
        output_wrist[..., :12] = _bounded(
            output_wrist[..., :12], self.final_clip, "actor continuous wrist"
        )
        output_wrist[..., 12:14] = torch.clamp(output_wrist[..., 12:14], 0.0, 1.0)
        output_proprio = proprio.clone()
        output_proprio[:, :3] /= float(self.actor["root_angular_divisor"])
        output_proprio[:, 6:35] /= float(self.actor["joint_offset_divisor"])
        output_proprio[:, 35:64] /= float(self.actor["joint_velocity_divisor"])
        output_proprio = _bounded(output_proprio, self.final_clip, "actor proprio")
        return {
            "wrist_tokens": output_wrist,
            "proprio": output_proprio,
            "a_nom_history": _bounded(
                history / float(self.actor["action_divisor"]),
                float(self.actor["action_clip"]),
                "a_nom_history",
            ),
            "previous_a_total": _bounded(
                previous / float(self.actor["action_divisor"]),
                float(self.actor["action_clip"]),
                "previous_a_total",
            ),
            "z_cross": _bounded(cross, float(self.actor["z_cross_clip"]), "z_cross"),
        }

    def normalize_policy(self, policy: torch.Tensor) -> torch.Tensor:
        """Normalize the structured 668-D policy prefix through actor-safe fields."""
        policy = _finite_float32(policy, "policy")
        if policy.ndim != 2 or policy.shape[1] != 668:
            raise ValueError("policy must have shape [B,668]")
        batch = policy.shape[0]
        normalized = self.normalize_actor(
            {
                "wrist_tokens": policy[:, :448].reshape(batch, 2, 16, 14),
                "proprio": policy[:, 448:512],
                "a_nom_history": policy[:, 512:581].reshape(batch, 23, 3),
                "previous_a_total": policy[:, 581:604],
                "z_cross": policy[:, 604:668],
            }
        )
        return torch.cat(
            (
                normalized["wrist_tokens"].reshape(batch, 448),
                normalized["proprio"],
                normalized["a_nom_history"].reshape(batch, 69),
                normalized["previous_a_total"],
                normalized["z_cross"],
            ),
            dim=-1,
        )

    def normalize_critic(
        self, fields: Mapping[str, torch.Tensor], *, task: str
    ) -> dict[str, torch.Tensor]:
        if task not in TASKS:
            raise ValueError("unknown critic task")
        required = {
            "contact_state",
            "object_state",
            "physics_mismatch",
            "policy",
            "progress_state",
            "scaffold_mismatch",
            "sensor_mismatch",
            "stability_state",
        }
        if set(fields) != required:
            raise ValueError("critic normalizer received unknown or missing fields")
        output = {
            name: _finite_float32(value, name).clone() for name, value in fields.items()
        }
        batch = output["policy"].shape[0]
        expected = {
            "object_state": 16,
            "physics_mismatch": 39,
            "scaffold_mismatch": 2,
            "sensor_mismatch": 90,
            "progress_state": 4,
            "contact_state": 17,
            "stability_state": 9,
        }
        if output["policy"].shape != (batch, 668) or any(
            output[name].shape != (batch, width) for name, width in expected.items()
        ):
            raise ValueError("critic field shapes do not match frozen schema")
        _same_device_and_batch(output, batch, output["policy"].device)
        output["policy"] = self.normalize_policy(output["policy"])
        c = self.critic
        object_state = output["object_state"]
        object_state[:, :3] /= float(c["object_position_divisor"])
        object_state[:, 7:10] /= float(c["linear_divisor"])
        object_state[:, 10:13] /= float(c["angular_divisor"])
        object_state[:, 13] /= torch.pi
        object_state[:, 14] /= float(c["velocity_divisor"])
        object_state[:, 15] /= float(c["torque_divisor"])
        physics = output["physics_mismatch"]
        sampler = self.contract.payload["sampler"]
        nominal = sampler["nominal_rows"][task]
        c3 = sampler["task_ranges"][task]["C3"]
        if task != "push_door_hand":
            nominal_mass = float(nominal["mass"])
            nominal_inertia = torch.tensor(
                nominal["inertia"], dtype=torch.float32, device=physics.device
            )
            physics[:, 11] = torch.log(
                torch.clamp_min(physics[:, 11], 1.0e-6) / nominal_mass
            ) / float(c["log_ratio_divisor"])
            physics[:, 15:18] = torch.log(
                torch.clamp_min(physics[:, 15:18], 1.0e-6) / nominal_inertia[:3]
            ) / float(c["log_ratio_divisor"])
            offdiag_scale = torch.sqrt(
                torch.stack(
                    (
                        nominal_inertia[0] * nominal_inertia[1],
                        nominal_inertia[0] * nominal_inertia[2],
                        nominal_inertia[1] * nominal_inertia[2],
                    )
                )
            )
            physics[:, 18:21] /= offdiag_scale
            physics[:, 12:15] /= float(c3["com_m"])
            physics[:, 21] = torch.log(
                torch.clamp_min(physics[:, 21], 1.0e-6) / float(nominal["friction"])
            ) / float(c["log_ratio_divisor"])
        if task == "push_door_hand":
            if torch.any(physics[:, 0:3] != 0.0) or torch.any(physics[:, 5:11] != 0.0):
                raise ValueError(
                    "door runtime-deferred hinge-axis and physical-handle fields must be zero"
                )
            physics[:, 3] = torch.log(
                torch.clamp_min(physics[:, 3], 1.0e-6) / float(nominal["friction"])
            ) / float(c["log_ratio_divisor"])
            physics[:, 4] = torch.log(
                torch.clamp_min(physics[:, 4], 1.0e-6) / float(nominal["damping"])
            ) / float(c["log_ratio_divisor"])
            object_position_bound = float(c3["object_m"])
            object_rotation_bound = float(c3["object_yaw_deg"]) * torch.pi / 180.0
            contact_position_bound = float(c3["contact_m"])
            contact_rotation_bound = float(c3["contact_rot_deg"]) * torch.pi / 180.0
        else:
            position_bound = float(c3["contact_m"])
            rotation_bound = float(c3["contact_rot_deg"]) * torch.pi / 180.0
            object_position_bound = float(c3["object_xy_m"])
            object_rotation_bound = float(c3["object_yaw_deg"]) * torch.pi / 180.0
        if task != "push_door_hand":
            physics[:, 5:8] /= position_bound
            physics[:, 8:11] /= rotation_bound
        physics[:, 22:25] /= object_position_bound
        physics[:, 25:28] /= object_rotation_bound
        physics[:, 28:30] /= float(c3["stance_m"])
        physics[:, 30] /= float(c3["stance_yaw_deg"]) * torch.pi / 180.0
        if task == "push_door_hand":
            physics[:, 31:34] /= contact_position_bound
            physics[:, 34:37] /= contact_rotation_bound
        else:
            physics[:, 31:34] /= position_bound
            physics[:, 34:37] /= rotation_bound
        physics[:, 37:39] /= float(sampler["stage_constants"]["C3"]["load_share"])
        scaffold = output["scaffold_mismatch"]
        scaffold[:, 0] = (scaffold[:, 0] - float(c["delay_center"])) / float(
            c["delay_divisor"]
        )
        scaffold[:, 1] = (scaffold[:, 1] - float(c["alpha_center"])) / float(
            c["alpha_divisor"]
        )
        sensor = VirtualFTSensorParameters.from_flat(output["sensor_mismatch"])
        for offset, value in (
            (8, sensor.scale_error),
            (20, sensor.additive_bias),
            (32, sensor.drift_rate),
            (44, sensor.drift_noise_std),
            (56, sensor.white_noise_std),
        ):
            normalized_sensor = value.clone()
            if offset != 8:
                normalized_sensor[..., :3] /= float(c["force_divisor"])
                normalized_sensor[..., 3:6] /= float(c["moment_divisor"])
            output["sensor_mismatch"][:, offset : offset + 12] = (
                normalized_sensor.reshape(batch, 12)
            )
        saturation = torch.cat(
            (sensor.force_saturation, sensor.torque_saturation), dim=-1
        )
        saturation[..., :3] /= float(c["force_divisor"])
        saturation[..., 3:6] /= float(c["moment_divisor"])
        output["sensor_mismatch"][:, 72:84] = saturation.reshape(batch, 12)
        output["sensor_mismatch"][:, 86:88] /= float(c["force_divisor"])
        output["sensor_mismatch"][:, 88:90] /= float(c["moment_divisor"])
        contact = output["contact_state"]
        clean = contact[:, :12].reshape(batch, 2, 6)
        normalized_clean = torch.cat(
            (
                clean[..., :3] / float(c["force_divisor"]),
                clean[..., 3:6] / float(c["moment_divisor"]),
            ),
            dim=-1,
        )
        contact[:, :12] = normalized_clean.reshape(batch, 12)
        contact[:, 12:16] = torch.clamp(contact[:, 12:16], 0.0, 1.0)
        contact[:, 16] /= float(c["support_divisor"])
        stability = output["stability_state"]
        stability[:, 0] /= float(c["root_height_divisor"])
        stability[:, 4:7] /= float(c["linear_divisor"])
        stability[:, 7:9] = torch.clamp(stability[:, 7:9], 0.0, 1.0)
        for name in (
            "object_state",
            "physics_mismatch",
            "scaffold_mismatch",
            "sensor_mismatch",
            "progress_state",
            "contact_state",
            "stability_state",
        ):
            output[name] = _bounded(output[name], self.final_clip, name)
        output["contact_state"][:, 12:16] = torch.clamp(
            output["contact_state"][:, 12:16], 0.0, 1.0
        )
        output["stability_state"][:, 7:9] = torch.clamp(
            output["stability_state"][:, 7:9], 0.0, 1.0
        )
        return output
