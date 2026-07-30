"""Common task-adapter contract for Phase 4 vectorized environments."""

from __future__ import annotations

from typing import NamedTuple, Protocol

import torch


def rotvec_to_quat_wxyz(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert finite float32 rotation vectors to normalized wxyz quaternions."""
    if not isinstance(rotvec, torch.Tensor) or rotvec.ndim != 2 or rotvec.shape[1] != 3:
        raise ValueError("rotvec must have shape [B,3]")
    if rotvec.dtype != torch.float32 or not torch.isfinite(rotvec).all():
        raise ValueError("rotvec must be finite float32")
    angle = torch.linalg.vector_norm(rotvec, dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    scale = torch.where(
        angle > 1.0e-6,
        torch.sin(half_angle) / angle,
        0.5 - angle.square() / 48.0,
    )
    return torch.cat((torch.cos(half_angle), rotvec * scale), dim=-1)


def quat_mul_wxyz(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Multiply matching wxyz quaternion batches."""
    if left.shape != right.shape or left.shape[-1] != 4:
        raise ValueError("quaternion operands must have matching [...,4] shapes")
    w1, x1, y1, z1 = left.unbind(-1)
    w2, x2, y2, z2 = right.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_apply_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by broadcast-compatible normalized wxyz quaternions."""
    xyz = quaternion[..., 1:]
    cross = 2.0 * torch.linalg.cross(xyz, vector, dim=-1)
    return vector + quaternion[..., :1] * cross + torch.linalg.cross(xyz, cross, dim=-1)


def quat_to_rotvec_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the principal finite rotation vector for normalized wxyz quaternions."""
    if (
        not isinstance(quaternion, torch.Tensor)
        or quaternion.ndim != 2
        or quaternion.shape[1] != 4
    ):
        raise ValueError("quaternion must have shape [B,4]")
    if quaternion.dtype != torch.float32 or not torch.isfinite(quaternion).all():
        raise ValueError("quaternion must be finite float32")
    normalized = quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True
    ).clamp_min(1.0e-12)
    normalized = torch.where(normalized[:, :1] < 0.0, -normalized, normalized)
    vector = normalized[:, 1:]
    length = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(length, normalized[:, :1].clamp_min(1.0e-12))
    scale = torch.where(length > 1.0e-6, angle / length, torch.full_like(length, 2.0))
    return vector * scale


def adapter_contact_target_offsets(
    nominal_offsets: torch.Tensor, physics_row: torch.Tensor
) -> torch.Tensor:
    """Apply the object-local ``31:37`` contact-target transform."""
    if (
        not isinstance(nominal_offsets, torch.Tensor)
        or nominal_offsets.ndim != 2
        or nominal_offsets.shape[0] < 1
        or nominal_offsets.shape[1] != 3
        or nominal_offsets.dtype != torch.float32
        or not torch.isfinite(nominal_offsets).all()
    ):
        raise ValueError("nominal contact offsets must be finite float32 [K,3], K >= 1")
    if (
        not isinstance(physics_row, torch.Tensor)
        or physics_row.ndim != 2
        or physics_row.shape[1] != 39
        or physics_row.dtype != torch.float32
        or not torch.isfinite(physics_row).all()
    ):
        raise ValueError("physics_row must be finite float32 [B,39]")
    rotation = rotvec_to_quat_wxyz(physics_row[:, 34:37])[:, None]
    source = nominal_offsets.to(physics_row.device).expand(physics_row.shape[0], -1, -1)
    return quat_apply_wxyz(rotation, source) + physics_row[:, None, 31:34]


def validate_door_runtime_physics(
    env_ids: torch.Tensor, physics_row: torch.Tensor, *, device: torch.device
) -> None:
    """Validate the runtime-supported door ownership slices without mutation."""
    if (
        not isinstance(physics_row, torch.Tensor)
        or physics_row.shape != (env_ids.numel(), 39)
        or physics_row.dtype != torch.float32
        or physics_row.device != device
        or not torch.isfinite(physics_row).all()
    ):
        raise ValueError("door runtime physics must be finite float32 [B,39]")
    if torch.count_nonzero(physics_row[:, 0:3]) or torch.count_nonzero(
        physics_row[:, 5:11]
    ):
        raise ValueError("door runtime-deferred slices 0:3 and 5:11 must be zero")
    if torch.any(physics_row[:, 3:5] < 0.0):
        raise ValueError("door mechanism friction and damping must be nonnegative")
    if torch.count_nonzero(physics_row[:, 11:22]) or torch.count_nonzero(
        physics_row[:, 37:39]
    ):
        raise ValueError("door runtime owns only rows 3:5 and 22:37")


def door_custom_effort(
    door_velocity: torch.Tensor, friction: torch.Tensor, damping: torch.Tensor
) -> torch.Tensor:
    """Compute the adapter-owned per-environment door effort target."""
    if (
        door_velocity.ndim != 1
        or friction.shape != door_velocity.shape
        or damping.shape != door_velocity.shape
        or any(
            value.dtype != torch.float32 for value in (door_velocity, friction, damping)
        )
        or any(
            not torch.isfinite(value).all()
            for value in (door_velocity, friction, damping)
        )
    ):
        raise ValueError("door effort inputs must be matching finite float32 [B]")
    return (
        -torch.sign(door_velocity) * (door_velocity.abs() > 0.01) * friction
        - door_velocity * damping
    )


def apply_selected_rigid_physx_parameters(
    view: object,
    env_ids: torch.Tensor,
    physics_row: torch.Tensor,
) -> None:
    """Apply selected rigid mass/COM/inertia/material rows through PhysX getters."""
    indices = env_ids.to(device="cpu", dtype=torch.int64)
    masses = view.get_masses().clone()
    inertias = view.get_inertias().clone()
    if masses.ndim != 2 or masses.shape[1] != 1:
        raise ValueError(
            "rigid runtime requires exactly one PhysX body per environment"
        )
    if inertias.ndim != 2 or inertias.shape[1] != 9:
        raise ValueError("rigid runtime requires one flattened inertia per environment")
    local = env_ids.to(masses.device)
    values = physics_row.to(masses.device)
    masses[local, 0] = values[:, 11]
    inertia = torch.zeros(values.shape[0], 3, 3, device=masses.device)
    inertia[:, 0, 0] = values[:, 15]
    inertia[:, 1, 1] = values[:, 16]
    inertia[:, 2, 2] = values[:, 17]
    inertia[:, 0, 1] = inertia[:, 1, 0] = values[:, 18]
    inertia[:, 0, 2] = inertia[:, 2, 0] = values[:, 19]
    inertia[:, 1, 2] = inertia[:, 2, 1] = values[:, 20]
    inertias[local] = inertia.reshape(-1, 9)
    view.set_masses(masses, indices)
    view.set_inertias(inertias, indices)

    # Fresh COM read preserves PhysX's current principal-axis quaternion.
    coms = view.get_coms().clone()
    if coms.ndim != 2 or coms.shape[1] != 7:
        raise ValueError("rigid runtime requires one [xyz,quat] COM per environment")
    coms[local, :3] = values[:, 12:15]
    view.set_coms(coms, indices)

    materials = view.get_material_properties().clone()
    if materials.ndim != 3 or materials.shape[-1] != 3:
        raise ValueError("rigid runtime material getter must return [B,S,3]")
    materials[local, :, 0] = values[:, 21:22]
    materials[local, :, 1] = values[:, 21:22]
    view.set_material_properties(materials, indices)


def rigid_physx_readback(
    view: object, env_ids: torch.Tensor, *, device: torch.device
) -> torch.Tensor:
    """Build selected applied ``11:22`` values from direct PhysX getters."""
    masses_all = view.get_masses().to(device=device, dtype=torch.float32)
    local = env_ids.to(masses_all.device)
    masses = masses_all[local]
    inertias = view.get_inertias().to(device=device, dtype=torch.float32)[local]
    coms = view.get_coms().to(device=device, dtype=torch.float32)[local]
    materials = view.get_material_properties().to(device=device, dtype=torch.float32)[
        local
    ]
    if masses.shape[1:] != (1,) or inertias.shape[1:] != (9,):
        raise ValueError("rigid PhysX readback shape changed")
    result = rigid_nominal_physics_mismatch(
        masses[:, 0], inertias, materials[:, 0, 0], materials[:, 0, 1]
    )
    result[:, 12:15] = coms[:, :3]
    return result


def stable_env_seeds(
    base_seed: int,
    env_ids: torch.Tensor,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Map each selected environment to ``base_seed + env_id``.

    The mapping is intentionally independent of the order of ``env_ids`` so a
    selected reset cannot change an environment's random stream identity.
    """
    if not isinstance(base_seed, int) or isinstance(base_seed, bool) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    if not isinstance(env_ids, torch.Tensor):
        raise TypeError("env_ids must be a torch.Tensor")
    if env_ids.ndim != 1 or env_ids.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("env_ids must be a one-dimensional integer tensor")
    target = env_ids.device if device is None else torch.device(device)
    ids = env_ids.to(device=target, dtype=torch.long)
    if torch.any(ids < 0):
        raise ValueError("env_ids must be nonnegative")
    return ids + base_seed


def door_nominal_physics_mismatch(
    batch_size: int,
    *,
    mechanism_friction: float,
    mechanism_damping: float,
    device: torch.device | str,
) -> torch.Tensor:
    """Build the frozen door 39-D critic row in the audited field order."""
    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    values = torch.zeros(batch_size, 39, device=device, dtype=torch.float32)
    values[:, 3] = mechanism_friction
    values[:, 4] = mechanism_damping
    return values


def rigid_nominal_physics_mismatch(
    mass: torch.Tensor,
    inertia: torch.Tensor,
    static_friction: torch.Tensor,
    dynamic_friction: torch.Tensor,
) -> torch.Tensor:
    """Build the rigid-object 39-D row from applied PhysX readback."""
    tensors = {
        "mass": mass,
        "inertia": inertia,
        "static_friction": static_friction,
        "dynamic_friction": dynamic_friction,
    }
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.dtype != torch.float32:
            raise TypeError(f"{name} must have dtype torch.float32")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
    if mass.ndim != 1 or mass.numel() == 0:
        raise ValueError("mass must have shape [B] with B > 0")
    batch_size = mass.shape[0]
    if inertia.shape == (batch_size, 9):
        matrix = inertia.reshape(batch_size, 3, 3)
    elif inertia.shape == (batch_size, 3, 3):
        matrix = inertia
    else:
        raise ValueError("inertia must have shape [B,9] or [B,3,3]")
    for name, value in (
        ("static_friction", static_friction),
        ("dynamic_friction", dynamic_friction),
    ):
        if value.shape != (batch_size,):
            raise ValueError(f"{name} must have shape [B]")
        if value.device != mass.device:
            raise ValueError(f"{name} must be on the mass device")
    if matrix.device != mass.device:
        raise ValueError("inertia must be on the mass device")
    if torch.any(mass <= 0):
        raise ValueError("applied mass must be positive")
    if not (
        torch.equal(matrix[:, 0, 1], matrix[:, 1, 0])
        and torch.equal(matrix[:, 0, 2], matrix[:, 2, 0])
        and torch.equal(matrix[:, 1, 2], matrix[:, 2, 1])
    ):
        raise ValueError("applied inertia readback must be symmetric")
    if not torch.equal(static_friction, dynamic_friction):
        raise ValueError("applied static and dynamic friction must be equal")

    result = torch.zeros(batch_size, 39, device=mass.device, dtype=torch.float32)
    result[:, 11] = mass
    flattened = matrix.reshape(batch_size, 9)
    result[:, 15:21] = flattened[:, [0, 4, 8, 1, 2, 5]]
    result[:, 21] = static_friction
    return result


def rigid_object_state(
    robot_position: torch.Tensor,
    robot_quaternion_wxyz: torch.Tensor,
    object_position: torch.Tensor,
    object_quaternion_wxyz: torch.Tensor,
    object_linear_velocity_world: torch.Tensor,
    object_angular_velocity_world: torch.Tensor,
) -> torch.Tensor:
    """Map rigid-object root state to the frozen robot-root-frame schema."""
    values = (
        robot_position,
        robot_quaternion_wxyz,
        object_position,
        object_quaternion_wxyz,
        object_linear_velocity_world,
        object_angular_velocity_world,
    )
    batch_size = robot_position.shape[0]
    expected_shapes = ((batch_size, 3), (batch_size, 4)) * 2 + (
        (batch_size, 3),
        (batch_size, 3),
    )
    for value, shape in zip(values, expected_shapes, strict=True):
        if not isinstance(value, torch.Tensor):
            raise TypeError("rigid object-state inputs must be torch.Tensor values")
        if value.shape != shape:
            raise ValueError("rigid object-state input shape mismatch")
        if value.dtype != torch.float32:
            raise TypeError("rigid object-state inputs must use torch.float32")
        if value.device != robot_position.device:
            raise ValueError("rigid object-state inputs must share one device")
        if not torch.isfinite(value).all():
            raise ValueError("rigid object-state inputs must be finite")

    def conjugate(quaternion: torch.Tensor) -> torch.Tensor:
        return torch.cat((quaternion[:, :1], -quaternion[:, 1:]), dim=-1)

    def multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = left.unbind(-1)
        w2, x2, y2, z2 = right.unbind(-1)
        return torch.stack(
            (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ),
            dim=-1,
        )

    inverse = conjugate(robot_quaternion_wxyz)

    def rotate(vector: torch.Tensor) -> torch.Tensor:
        xyz = inverse[:, 1:]
        cross = 2.0 * torch.linalg.cross(xyz, vector, dim=-1)
        return vector + inverse[:, :1] * cross + torch.linalg.cross(xyz, cross, dim=-1)

    mechanism = torch.zeros(
        batch_size, 3, device=robot_position.device, dtype=torch.float32
    )
    return torch.cat(
        (
            rotate(object_position - robot_position),
            multiply(inverse, object_quaternion_wxyz),
            rotate(object_linear_velocity_world),
            rotate(object_angular_velocity_world),
            mechanism,
        ),
        dim=-1,
    )


def strict_root_height_failure(
    root_height: torch.Tensor, *, threshold: float
) -> torch.Tensor:
    """Apply the report-frozen strict fall boundary without inclusivity drift."""
    if not isinstance(root_height, torch.Tensor) or root_height.ndim != 1:
        raise TypeError("root_height must be a one-dimensional torch.Tensor")
    if not root_height.is_floating_point() or not torch.isfinite(root_height).all():
        raise ValueError("root_height must be finite floating point")
    if not isinstance(threshold, float) or not torch.isfinite(torch.tensor(threshold)):
        raise ValueError("threshold must be a finite float")
    return root_height < threshold


class TaskProgressSignals(NamedTuple):
    """Fixed task-agnostic progress, contact, and stability fields."""

    progress: torch.Tensor
    progress_delta: torch.Tensor
    success: torch.Tensor
    expected_contact: torch.Tensor
    contact_truth: torch.Tensor
    stability_margin: torch.Tensor
    failure: torch.Tensor
    reference_exhausted: torch.Tensor


class TaskAdapter(Protocol):
    """Door/payload data boundary without task-specific residual networks."""

    task_name: str
    task_spec: object
    object_kind: str
    reference: object
    reference_step: torch.Tensor

    def validate_reset(self, env_ids: torch.Tensor) -> None:
        """Validate a selected reset without mutating state."""

    def reset(self, env_ids: torch.Tensor) -> None:
        """Reset exactly the selected reference and progress rows."""

    def write_scene_reset(self, env_ids: torch.Tensor) -> None:
        """Write selected robot and object state to the simulator."""

    def validate_runtime_parameters(
        self, env_ids: torch.Tensor, physics_row: torch.Tensor
    ) -> None:
        """Validate selected B5 runtime values without mutating adapter state."""

    def apply_runtime_parameters(
        self, env_ids: torch.Tensor, physics_row: torch.Tensor
    ) -> None:
        """Apply selected B5 runtime values after the nominal reset pose is written."""

    def runtime_parameter_readback(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Return the selected applied physics row from runtime getters/buffers."""

    def advance(self) -> None:
        """Advance reference and progress once after a control step."""

    def apply_object_action(self) -> None:
        """Apply task-owned object actuation for one physics substep."""

    def build_scaffold_observation(self) -> object:
        """Build the artifact-specific frozen scaffold observation."""

    def build_object_state(self) -> torch.Tensor:
        """Return the common critic object state ``[B,16]``."""

    def proprio(self) -> torch.Tensor:
        """Return deployable whole-body proprioception ``[B,64]``."""

    def wrist_twist_base_yaw(self) -> torch.Tensor:
        """Return two-wrist twist in the base-yaw frame ``[B,2,6]``."""

    def nominal_physics_row(self) -> torch.Tensor:
        """Return the applied nominal physics row ``[B,39]``."""

    def expected_contact(self) -> torch.Tensor:
        """Return left/right reference contact intent ``[B,2]``."""

    def contact_truth(self) -> torch.Tensor:
        """Return privileged left/right simulator contact truth ``[B,2]``."""

    def support_contact_count(self) -> torch.Tensor:
        """Return privileged support-contact count ``[B,1]``."""

    def progress_signals(self) -> TaskProgressSignals:
        """Return the fixed common task signals."""


class SmokeDoneOutput(NamedTuple):
    """Separated terminated and timeout tensors for bounded smoke runs."""

    terminated: torch.Tensor
    time_outs: torch.Tensor


def smoke_done_flags(
    nonfinite_state: torch.Tensor,
    task_failure: torch.Tensor,
    episode_length: torch.Tensor,
    reference_exhausted: torch.Tensor,
    *,
    episode_length_steps: int,
) -> SmokeDoneOutput:
    """Apply the approved Phase 4B3 smoke-only done contract."""
    for value, name in (
        (nonfinite_state, "nonfinite_state"),
        (task_failure, "task_failure"),
        (reference_exhausted, "reference_exhausted"),
    ):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
            raise TypeError(f"{name} must be a boolean torch.Tensor")
        if value.ndim != 1:
            raise ValueError(f"{name} must have shape [B]")
    if not isinstance(episode_length, torch.Tensor):
        raise TypeError("episode_length must be a torch.Tensor")
    if episode_length.ndim != 1 or episode_length.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise TypeError("episode_length must be a one-dimensional integer tensor")
    shape = nonfinite_state.shape
    tensors = (task_failure, reference_exhausted, episode_length)
    if any(value.shape != shape for value in tensors):
        raise ValueError("all done inputs must have the same batch shape")
    if any(value.device != nonfinite_state.device for value in tensors):
        raise ValueError("all done inputs must be on the same device")
    if (
        not isinstance(episode_length_steps, int)
        or isinstance(episode_length_steps, bool)
        or episode_length_steps <= 0
    ):
        raise ValueError("episode_length_steps must be a positive integer")
    return SmokeDoneOutput(
        terminated=nonfinite_state | task_failure,
        time_outs=(episode_length >= episode_length_steps) | reference_exhausted,
    )
