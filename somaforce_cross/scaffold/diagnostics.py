"""Scaffold-only rollout diagnostics.

These checks stay on the nominal-motion side of the SomaForce-Cross boundary:
they inspect motion trajectory references and ``a_nom`` without force semantics,
contact truth, object state, or residual policy outputs.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from somaforce_cross.scaffold.contracts import ScaffoldTask
from somaforce_cross.scaffold.scenes import (
    make_g1_push_pull_box_scaffold,
    make_g1_push_pull_door_scaffold,
)
from somaforce_cross.scaffold.task_trajectories import InteractionMode


@dataclass(frozen=True)
class ScaffoldRolloutSummary:
    """Compact diagnostic summary for one scaffold-only task rollout."""

    task: ScaffoldTask
    mode: InteractionMode
    num_frames: int
    a_nom_shape: tuple[int, ...]
    joint_delta_norm: float
    max_abs_a_nom: float
    left_hand_x_delta: float
    right_hand_x_delta: float
    body_x_delta: float
    cmd_x: float
    direction_ok: bool

    @property
    def label(self) -> str:
        return f"{self.task.value}_{self.mode}"

    def as_line(self) -> str:
        return (
            f"{self.label}: "
            f"a_nom={self.a_nom_shape} "
            f"joint_delta_norm={self.joint_delta_norm:.4f} "
            f"max_abs_a_nom={self.max_abs_a_nom:.4f} "
            f"left_hand_x_delta={self.left_hand_x_delta:.4f} "
            f"right_hand_x_delta={self.right_hand_x_delta:.4f} "
            f"body_x_delta={self.body_x_delta:.4f} "
            f"cmd_x={self.cmd_x:.1f} "
            f"direction_ok={int(self.direction_ok)}"
        )


def summarize_scaffold_rollout(
    task: ScaffoldTask,
    mode: InteractionMode,
    num_frames: int = 80,
    duration_s: float = 2.0,
    action_clip: float | None = None,
) -> ScaffoldRolloutSummary:
    """Generate and summarize one scaffold-only G1 nominal rollout."""

    if task == ScaffoldTask.PUSH_PULL_DOOR:
        scaffold = make_g1_push_pull_door_scaffold(
            interaction_mode=mode,
            num_frames=num_frames,
            duration_s=duration_s,
            action_clip=action_clip,
        )
    elif task == ScaffoldTask.PUSH_PULL_BOX:
        scaffold = make_g1_push_pull_box_scaffold(
            interaction_mode=mode,
            num_frames=num_frames,
            duration_s=duration_s,
            action_clip=action_clip,
        )
    else:
        raise ValueError(f"Unsupported scaffold task: {task}")

    frames = torch.arange(num_frames, device=scaffold.trajectory.joint_pos.device)
    rollout = scaffold.at_frame(frames)
    start = scaffold.at_frame(0)
    end = scaffold.at_frame(num_frames - 1)

    if rollout.nominal_hand_ref is None:
        raise ValueError("Scaffold diagnostics require nominal hand references")
    if rollout.nominal_body_ref is None:
        raise ValueError("Scaffold diagnostics require nominal body references")
    if end.cmd_6d is None:
        raise ValueError("Scaffold diagnostics require cmd_6d")

    left_hand_x_delta = _float(end.nominal_hand_ref[0, 0] - start.nominal_hand_ref[0, 0])
    right_hand_x_delta = _float(end.nominal_hand_ref[1, 0] - start.nominal_hand_ref[1, 0])
    body_x_delta = _float(end.nominal_body_ref[0, 0] - start.nominal_body_ref[0, 0])
    cmd_x = _float(end.cmd_6d[0])

    return ScaffoldRolloutSummary(
        task=task,
        mode=mode,
        num_frames=num_frames,
        a_nom_shape=tuple(rollout.a_nom.shape),
        joint_delta_norm=_float(torch.linalg.vector_norm(end.a_nom - start.a_nom)),
        max_abs_a_nom=_float(torch.max(torch.abs(rollout.a_nom))),
        left_hand_x_delta=left_hand_x_delta,
        right_hand_x_delta=right_hand_x_delta,
        body_x_delta=body_x_delta,
        cmd_x=cmd_x,
        direction_ok=_direction_ok(task, cmd_x, left_hand_x_delta, right_hand_x_delta, body_x_delta),
    )


def summarize_first_scaffold_rollouts(
    num_frames: int = 80,
    duration_s: float = 2.0,
    action_clip: float | None = None,
) -> list[ScaffoldRolloutSummary]:
    """Summarize the first selected door/box push/pull scaffold variants."""

    return [
        summarize_scaffold_rollout(ScaffoldTask.PUSH_PULL_DOOR, "push", num_frames, duration_s, action_clip),
        summarize_scaffold_rollout(ScaffoldTask.PUSH_PULL_DOOR, "pull", num_frames, duration_s, action_clip),
        summarize_scaffold_rollout(ScaffoldTask.PUSH_PULL_BOX, "push", num_frames, duration_s, action_clip),
        summarize_scaffold_rollout(ScaffoldTask.PUSH_PULL_BOX, "pull", num_frames, duration_s, action_clip),
    ]


def _direction_ok(
    task: ScaffoldTask,
    cmd_x: float,
    left_hand_x_delta: float,
    right_hand_x_delta: float,
    body_x_delta: float,
) -> bool:
    if abs(cmd_x) <= 0.0:
        return False
    sign = 1.0 if cmd_x > 0.0 else -1.0
    if task == ScaffoldTask.PUSH_PULL_DOOR:
        return sign * right_hand_x_delta > 0.0 and sign * body_x_delta > 0.0
    if task == ScaffoldTask.PUSH_PULL_BOX:
        return (
            sign * left_hand_x_delta > 0.0
            and sign * right_hand_x_delta > 0.0
            and sign * body_x_delta > 0.0
        )
    return False


def _float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)
