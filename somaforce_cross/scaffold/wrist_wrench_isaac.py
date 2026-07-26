"""Isaac articulation source adapter for V1 two-wrist 6-D joint wrenches.

PXR is imported lazily only while resolving immutable joint metadata. The
per-physics-step path is pure vectorized PyTorch and never queries USD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, NamedTuple

import torch

from somaforce_cross.sensing import WristWrenchTransform


WRIST_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
PELVIS_BODY_NAME = "pelvis"


@dataclass(frozen=True)
class ChildJointPose:
    """Cached USD child-joint pose, expressed in its child body frame."""

    joint_path: str
    local_pos: tuple[float, float, float]
    local_quat_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class WristJointMetadata:
    body_name: str
    body_id: int
    joint_path: str
    child_local_pos: tuple[float, float, float]
    child_local_quat_wxyz: tuple[float, float, float, float]
    artifact_identity: str


class IsaacWristWrenchOutput(NamedTuple):
    raw_joint_wrench: torch.Tensor
    clean_total_wrench_base_yaw: torch.Tensor
    valid: torch.Tensor


JointPoseResolver = Callable[[object, str], ChildJointPose]


def _unique_body_id(robot: object, body_name: str) -> int:
    try:
        ids, names = robot.find_bodies(body_name, preserve_order=True)
    except AttributeError as exc:
        raise TypeError("robot must provide find_bodies()") from exc
    if len(ids) != 1 or len(names) != 1 or names[0] != body_name:
        raise ValueError(
            f"body name {body_name!r} must resolve uniquely, got ids={ids}, names={names}"
        )
    return int(ids[0])


def _robot_instance_root(robot: object) -> str | None:
    view = getattr(robot, "root_physx_view", None)
    paths = getattr(view, "prim_paths", None)
    if not paths:
        return None
    path = str(paths[0])
    if path.endswith(f"/{PELVIS_BODY_NAME}"):
        path = path.rsplit("/", 1)[0]
    return path.rstrip("/")


def _pxr_child_joint_pose(robot: object, body_name: str) -> ChildJointPose:
    """Resolve one incoming joint from the live stage during initialization."""

    try:
        import omni.usd
        from pxr import UsdPhysics
    except ImportError as exc:
        raise RuntimeError(
            "Isaac Sim with PXR is required for USD joint resolution"
        ) from exc

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("no live USD stage is available")
    root = _robot_instance_root(robot)
    matches: list[tuple[object, object]] = []
    for prim in stage.Traverse():
        joint = UsdPhysics.Joint(prim)
        if not joint:
            continue
        targets = joint.GetBody1Rel().GetTargets()
        if len(targets) != 1:
            continue
        child_path = str(targets[0])
        if child_path.rsplit("/", 1)[-1] != body_name:
            continue
        if root is not None and not child_path.startswith(f"{root}/"):
            continue
        matches.append((prim, joint))
    if len(matches) != 1:
        paths = [str(prim.GetPath()) for prim, _ in matches]
        raise ValueError(
            f"incoming joint for {body_name!r} must resolve uniquely under {root!r}, got {paths}"
        )

    prim, joint = matches[0]
    local_pos = joint.GetLocalPos1Attr().Get()
    local_quat = joint.GetLocalRot1Attr().Get()
    if local_pos is None or local_quat is None:
        raise ValueError(f"joint {prim.GetPath()} has no complete child local pose")
    imaginary = local_quat.GetImaginary()
    return ChildJointPose(
        joint_path=str(prim.GetPath()),
        local_pos=tuple(float(local_pos[index]) for index in range(3)),
        local_quat_wxyz=(
            float(local_quat.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ),
    )


def _env_ids(
    env_ids: int | Iterable[int] | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(env_ids, int) and not isinstance(env_ids, bool):
        ids = torch.tensor([env_ids], device=device, dtype=torch.long)
    elif isinstance(env_ids, torch.Tensor):
        if env_ids.dtype == torch.bool:
            if env_ids.shape != (batch_size,):
                raise ValueError("boolean env_ids must have shape [batch_size]")
            ids = torch.nonzero(env_ids, as_tuple=False).flatten().to(device)
        else:
            if env_ids.ndim != 1 or env_ids.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise TypeError("env_ids must be a one-dimensional integer tensor")
            ids = env_ids.to(device=device, dtype=torch.long)
    else:
        try:
            ids = torch.as_tensor(list(env_ids), device=device, dtype=torch.long)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "env_ids must be an integer or iterable of integers"
            ) from exc
        if ids.ndim != 1:
            raise ValueError("env_ids must be one-dimensional")
    if ids.numel() and ((ids < 0).any() or (ids >= batch_size).any()):
        raise IndexError("env_ids contains an index outside the source")
    if ids.numel() and torch.unique(ids).numel() != ids.numel():
        raise ValueError("env_ids must not contain duplicates")
    return ids


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
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


def _quat_apply(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_xyz = q[..., 1:]
    t = 2.0 * torch.linalg.cross(q_xyz, vector, dim=-1)
    return vector + q[..., :1] * t + torch.linalg.cross(q_xyz, t, dim=-1)


class IsaacWristWrenchSource:
    """Cache Isaac joint metadata and expose a GPU-only two-wrist wrench source."""

    def __init__(
        self,
        robot: object,
        *,
        artifact_identity: str,
        sensor_offset_J: torch.Tensor,
        warmup_steps: int,
        joint_pose_resolver: JointPoseResolver | None = None,
    ) -> None:
        if not isinstance(artifact_identity, str) or not artifact_identity.strip():
            raise ValueError("artifact_identity must be a non-empty string")
        if (
            not isinstance(warmup_steps, int)
            or isinstance(warmup_steps, bool)
            or warmup_steps <= 0
        ):
            raise ValueError("warmup_steps must be a positive integer")
        self.robot = robot
        self.artifact_identity = artifact_identity
        self.warmup_steps = warmup_steps

        self.wrist_body_ids = tuple(
            _unique_body_id(robot, body_name) for body_name in WRIST_BODY_NAMES
        )
        self.pelvis_body_id = _unique_body_id(robot, PELVIS_BODY_NAME)
        resolver = joint_pose_resolver or _pxr_child_joint_pose
        poses = tuple(resolver(robot, body_name) for body_name in WRIST_BODY_NAMES)

        try:
            body_pos_w = robot.data.body_pos_w
            body_quat_w = robot.data.body_quat_w
        except AttributeError as exc:
            raise TypeError(
                "robot.data must provide body_pos_w and body_quat_w"
            ) from exc
        if not isinstance(body_pos_w, torch.Tensor) or not isinstance(
            body_quat_w, torch.Tensor
        ):
            raise TypeError("robot body pose fields must be torch.Tensor values")
        if (
            body_pos_w.ndim != 3
            or body_pos_w.shape[0] == 0
            or body_pos_w.shape[-1] != 3
            or body_quat_w.shape != (*body_pos_w.shape[:-1], 4)
        ):
            raise ValueError("robot body pose fields have invalid shapes")
        if not body_pos_w.is_floating_point() or body_quat_w.dtype != body_pos_w.dtype:
            raise TypeError("robot body poses must share a floating-point dtype")
        if body_quat_w.device != body_pos_w.device:
            raise ValueError("robot body pose fields must share a device")
        self.batch_size = body_pos_w.shape[0]
        self.device = body_pos_w.device
        self.dtype = body_pos_w.dtype

        if not isinstance(sensor_offset_J, torch.Tensor):
            raise TypeError("sensor_offset_J must be a torch.Tensor")
        if not sensor_offset_J.is_floating_point():
            raise TypeError("sensor_offset_J must have a floating-point dtype")
        if sensor_offset_J.shape != (2, 3):
            raise ValueError("sensor_offset_J must have shape [2, 3]")
        if sensor_offset_J.device != self.device or sensor_offset_J.dtype != self.dtype:
            raise ValueError("sensor_offset_J must match robot pose device and dtype")
        if not torch.isfinite(sensor_offset_J).all():
            raise ValueError("sensor_offset_J must contain only finite meter offsets")
        self.sensor_offset_J = sensor_offset_J.clone()

        local_pos = torch.tensor(
            [pose.local_pos for pose in poses], device=self.device, dtype=self.dtype
        )
        local_quat = torch.tensor(
            [pose.local_quat_wxyz for pose in poses],
            device=self.device,
            dtype=self.dtype,
        )
        if not torch.isfinite(local_pos).all() or not torch.isfinite(local_quat).all():
            raise ValueError("cached child joint poses must be finite")
        quaternion_norm = torch.linalg.vector_norm(local_quat, dim=-1, keepdim=True)
        if torch.any(quaternion_norm == 0):
            raise ValueError("cached child joint quaternions must be non-zero")
        self.child_local_pos = local_pos
        self.child_local_quat_wxyz = local_quat / quaternion_norm
        self.wrist_body_ids_tensor = torch.tensor(
            self.wrist_body_ids, device=self.device, dtype=torch.long
        )
        self.metadata = tuple(
            WristJointMetadata(
                body_name=body_name,
                body_id=body_id,
                joint_path=pose.joint_path,
                child_local_pos=pose.local_pos,
                child_local_quat_wxyz=pose.local_quat_wxyz,
                artifact_identity=artifact_identity,
            )
            for body_name, body_id, pose in zip(
                WRIST_BODY_NAMES, self.wrist_body_ids, poses, strict=True
            )
        )
        self.age = torch.zeros(self.batch_size, device=self.device, dtype=torch.long)
        self.valid = torch.zeros(
            self.batch_size, 2, device=self.device, dtype=torch.bool
        )
        self.transform = WristWrenchTransform()

    def reset(self, env_ids: int | Iterable[int] | torch.Tensor) -> None:
        ids = _env_ids(env_ids, batch_size=self.batch_size, device=self.device)
        self.age[ids] = 0
        self.valid[ids] = False

    def update_after_physics_step(self) -> IsaacWristWrenchOutput:
        """Read one frame after the caller completed physics step and scene update."""

        data = self.robot.data
        raw_all = data.body_incoming_joint_wrench_b
        body_pos_w = data.body_pos_w
        body_quat_w = data.body_quat_w
        expected_bodies = body_pos_w.shape[1]
        if not isinstance(raw_all, torch.Tensor):
            raise TypeError("body_incoming_joint_wrench_b must be a torch.Tensor")
        if raw_all.shape != (self.batch_size, expected_bodies, 6):
            raise ValueError(
                "body_incoming_joint_wrench_b must have shape [B, num_bodies, 6]"
            )
        for value, name, width in (
            (body_pos_w, "body_pos_w", 3),
            (body_quat_w, "body_quat_w", 4),
        ):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.shape != (self.batch_size, expected_bodies, width):
                raise ValueError(
                    f"{name} has changed shape after source initialization"
                )
        for value, name in (
            (raw_all, "body_incoming_joint_wrench_b"),
            (body_pos_w, "body_pos_w"),
            (body_quat_w, "body_quat_w"),
        ):
            if not value.is_floating_point():
                raise TypeError(f"{name} must have a floating-point dtype")
            if value.device != self.device or value.dtype != self.dtype:
                raise ValueError(f"{name} must retain the source device and dtype")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")

        raw_joint_wrench = raw_all.index_select(1, self.wrist_body_ids_tensor)
        wrist_pos_w = body_pos_w.index_select(1, self.wrist_body_ids_tensor)
        wrist_quat_w = body_quat_w.index_select(1, self.wrist_body_ids_tensor)
        child_quat = self.child_local_quat_wxyz[None].expand(self.batch_size, -1, -1)
        child_pos = self.child_local_pos[None].expand(self.batch_size, -1, -1)
        sensor_offset = self.sensor_offset_J[None].expand(self.batch_size, -1, -1)
        joint_quat_w = _quat_mul(wrist_quat_w, child_quat)
        joint_pos_w = wrist_pos_w + _quat_apply(wrist_quat_w, child_pos)
        sensor_pos_w = joint_pos_w + _quat_apply(joint_quat_w, sensor_offset)
        pelvis_quat_w = body_quat_w[:, self.pelvis_body_id]
        clean_total_wrench = self.transform(
            raw_joint_wrench,
            joint_quat_w,
            joint_pos_w,
            sensor_pos_w,
            pelvis_quat_w,
        )

        self.age.add_(1)
        ready = self.age >= self.warmup_steps
        self.valid.copy_(ready[:, None].expand(-1, 2))
        return IsaacWristWrenchOutput(
            raw_joint_wrench=raw_joint_wrench,
            clean_total_wrench_base_yaw=clean_total_wrench,
            valid=self.valid.clone(),
        )
