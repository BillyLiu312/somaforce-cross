"""Structured pure-PyTorch observation assembly for Phase 4B1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn

from somaforce_cross.force import (
    CrossEncoder,
    ForceSemanticHeads,
    WristTokenEncoder,
    two_wrist_semantic_targets,
)


_FLOAT_DTYPE = torch.float32


def _validate_batched_tensors(
    fields: tuple[tuple[str, object, tuple[int, ...]], ...],
) -> tuple[int, torch.device]:
    for name, value, _ in fields:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.dtype != _FLOAT_DTYPE:
            raise TypeError(f"{name} must have dtype torch.float32")

    first = fields[0][1]
    assert isinstance(first, torch.Tensor)
    batch_size = first.shape[0] if first.ndim else 0
    if batch_size <= 0:
        raise ValueError("batch size must be greater than zero")
    device = first.device

    for name, value, trailing_shape in fields:
        assert isinstance(value, torch.Tensor)
        expected_shape = (batch_size, *trailing_shape)
        if value.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {list(expected_shape)}, got {list(value.shape)}"
            )
        if value.device != device:
            raise ValueError(f"{name} must be on {device}, got {value.device}")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
    return batch_size, device


def _validate_scale(
    value: object,
    name: str,
    *,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != _FLOAT_DTYPE:
        raise TypeError(f"{name} must have dtype torch.float32")
    if value.numel() != 1:
        raise ValueError(f"{name} must be a scalar tensor")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    if not torch.isfinite(value).all() or torch.any(value <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return value


class ForceSemanticOutput(NamedTuple):
    """All approved intermediate and final force-semantic tensors."""

    per_wrist_features: torch.Tensor
    fused_features: torch.Tensor
    dir_logits: torch.Tensor
    p_dir: torch.Tensor
    mag_logits: torch.Tensor
    p_mag: torch.Tensor
    P_cross: torch.Tensor
    z_joint: torch.Tensor
    z_cross: torch.Tensor


class ForceSemanticPipeline(nn.Module):
    """Compose the committed wrist, semantic-head, and cross-encoder APIs."""

    def __init__(self) -> None:
        super().__init__()
        self.wrist_encoder = WristTokenEncoder()
        self.semantic_heads = ForceSemanticHeads()
        self.cross_encoder = CrossEncoder()

    def forward(self, wrist_tokens: torch.Tensor) -> ForceSemanticOutput:
        _validate_batched_tensors((("wrist_tokens", wrist_tokens, (2, 16, 14)),))
        for name, parameter in self.named_parameters():
            if parameter.dtype != wrist_tokens.dtype:
                raise ValueError(
                    "ForceSemanticPipeline parameter "
                    f"{name!r} has dtype {parameter.dtype}, expected {wrist_tokens.dtype}"
                )
            if parameter.device != wrist_tokens.device:
                raise ValueError(
                    "ForceSemanticPipeline parameter "
                    f"{name!r} is on {parameter.device}, expected {wrist_tokens.device}"
                )

        per_wrist, fused = self.wrist_encoder(wrist_tokens)
        semantic = self.semantic_heads(fused)
        cross = self.cross_encoder(semantic.p_dir, semantic.p_mag)
        return ForceSemanticOutput(
            per_wrist_features=per_wrist,
            fused_features=fused,
            dir_logits=semantic.dir_logits,
            p_dir=semantic.p_dir,
            mag_logits=semantic.mag_logits,
            p_mag=semantic.p_mag,
            P_cross=cross.P_cross,
            z_joint=cross.z_joint,
            z_cross=cross.z_cross,
        )

    def assemble_policy_observation(
        self,
        wrist_tokens: torch.Tensor,
        proprio: torch.Tensor,
        a_nom_history: torch.Tensor,
        previous_a_total: torch.Tensor,
    ) -> PolicyObservationAssembly:
        """Derive semantics and assemble their paired policy observation."""
        _validate_batched_tensors(
            (
                ("wrist_tokens", wrist_tokens, (2, 16, 14)),
                ("proprio", proprio, (64,)),
                ("a_nom_history", a_nom_history, (23, 3)),
                ("previous_a_total", previous_a_total, (23,)),
            )
        )
        semantic_output = self(wrist_tokens)
        bundle = PolicyObservationBundle._from_semantic_output(
            wrist_tokens=wrist_tokens,
            proprio=proprio,
            a_nom_history=a_nom_history,
            previous_a_total=previous_a_total,
            semantic_output=semantic_output,
        )
        return PolicyObservationAssembly(
            bundle=bundle,
            semantic_output=semantic_output,
        )


@dataclass(frozen=True, init=False)
class PolicyObservationBundle:
    """Named deployable fields at the environment/model-wrapper boundary."""

    wrist_tokens: torch.Tensor
    proprio: torch.Tensor
    a_nom_history: torch.Tensor
    previous_a_total: torch.Tensor
    z_cross: torch.Tensor

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError(
            "PolicyObservationBundle cannot be constructed directly; "
            "use ForceSemanticPipeline.assemble_policy_observation()"
        )

    @classmethod
    def _from_semantic_output(
        cls,
        *,
        wrist_tokens: torch.Tensor,
        proprio: torch.Tensor,
        a_nom_history: torch.Tensor,
        previous_a_total: torch.Tensor,
        semantic_output: ForceSemanticOutput,
    ) -> PolicyObservationBundle:
        bundle = object.__new__(cls)
        object.__setattr__(bundle, "wrist_tokens", wrist_tokens)
        object.__setattr__(bundle, "proprio", proprio)
        object.__setattr__(bundle, "a_nom_history", a_nom_history)
        object.__setattr__(bundle, "previous_a_total", previous_a_total)
        object.__setattr__(bundle, "z_cross", semantic_output.z_cross)
        bundle.__post_init__()
        return bundle

    def __post_init__(self) -> None:
        _validate_batched_tensors(
            (
                ("wrist_tokens", self.wrist_tokens, (2, 16, 14)),
                ("proprio", self.proprio, (64,)),
                ("a_nom_history", self.a_nom_history, (23, 3)),
                ("previous_a_total", self.previous_a_total, (23,)),
                ("z_cross", self.z_cross, (64,)),
            )
        )

    @property
    def batch_size(self) -> int:
        return self.wrist_tokens.shape[0]

    def semantic_prefix(self) -> torch.Tensor:
        """Return the exact flattened wrist-token prefix ``[B,448]``."""
        return self.wrist_tokens.reshape(self.batch_size, 448)

    def residual_actor_input(self) -> torch.Tensor:
        """Return only the approved deployable actor fields ``[B,220]``."""
        return torch.cat(
            (
                self.z_cross,
                self.proprio,
                self.a_nom_history.reshape(self.batch_size, 69),
                self.previous_a_total,
            ),
            dim=-1,
        )

    def flatten(self) -> torch.Tensor:
        """Return the structured environment policy boundary ``[B,668]``."""
        return torch.cat(
            (
                self.semantic_prefix(),
                self.proprio,
                self.a_nom_history.reshape(self.batch_size, 69),
                self.previous_a_total,
                self.z_cross,
            ),
            dim=-1,
        )


class PolicyObservationAssembly(NamedTuple):
    """Paired policy bundle and semantic output from one pipeline call."""

    bundle: PolicyObservationBundle
    semantic_output: ForceSemanticOutput


@dataclass(frozen=True)
class CriticObservationBundle:
    """Named policy and privileged fields for the critic boundary."""

    policy: torch.Tensor
    object_state: torch.Tensor
    physics_mismatch: torch.Tensor
    scaffold_mismatch: torch.Tensor
    sensor_mismatch: torch.Tensor
    progress_state: torch.Tensor
    contact_state: torch.Tensor
    stability_state: torch.Tensor

    def __post_init__(self) -> None:
        _validate_batched_tensors(
            (
                ("policy", self.policy, (668,)),
                ("object_state", self.object_state, (16,)),
                ("physics_mismatch", self.physics_mismatch, (39,)),
                ("scaffold_mismatch", self.scaffold_mismatch, (2,)),
                ("sensor_mismatch", self.sensor_mismatch, (90,)),
                ("progress_state", self.progress_state, (4,)),
                ("contact_state", self.contact_state, (17,)),
                ("stability_state", self.stability_state, (9,)),
            )
        )

    def flatten(self) -> torch.Tensor:
        """Return the approved policy-first critic observation ``[B,845]``."""
        return torch.cat(
            (
                self.policy,
                self.object_state,
                self.physics_mismatch,
                self.scaffold_mismatch,
                self.sensor_mismatch,
                self.progress_state,
                self.contact_state,
                self.stability_state,
            ),
            dim=-1,
        )


@dataclass(frozen=True)
class SemanticTargetBundle:
    """Named clean-wrench auxiliary-supervision fields."""

    clean_wrench: torch.Tensor
    p_dir_target: torch.Tensor
    p_mag_target: torch.Tensor
    semantic_loss_weight: torch.Tensor

    def __post_init__(self) -> None:
        _validate_batched_tensors(
            (
                ("clean_wrench", self.clean_wrench, (2, 6)),
                ("p_dir_target", self.p_dir_target, (13,)),
                ("p_mag_target", self.p_mag_target, (5,)),
                ("semantic_loss_weight", self.semantic_loss_weight, (1,)),
            )
        )

    def flatten(self) -> torch.Tensor:
        """Return the approved clean semantic target ``[B,31]``."""
        return torch.cat(
            (
                self.clean_wrench.reshape(self.clean_wrench.shape[0], 12),
                self.p_dir_target,
                self.p_mag_target,
                self.semantic_loss_weight,
            ),
            dim=-1,
        )


def build_semantic_target_bundle(
    clean_wrench: torch.Tensor,
    contact_probability: torch.Tensor,
    F_scale: torch.Tensor,
    M_scale: torch.Tensor,
) -> SemanticTargetBundle:
    """Build clean targets by delegating to the committed Phase 3C API."""
    _, device = _validate_batched_tensors(
        (
            ("clean_wrench", clean_wrench, (2, 6)),
            ("contact_probability", contact_probability, (2,)),
        )
    )
    force_scale = _validate_scale(F_scale, "F_scale", device=device)
    moment_scale = _validate_scale(M_scale, "M_scale", device=device)
    targets = two_wrist_semantic_targets(
        clean_wrench,
        contact_probability,
        force_scale,
        moment_scale,
    )
    return SemanticTargetBundle(
        clean_wrench=clean_wrench,
        p_dir_target=targets.global_direction,
        p_mag_target=targets.global_magnitude,
        semantic_loss_weight=targets.g_contact,
    )
