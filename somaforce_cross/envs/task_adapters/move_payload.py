"""Shared suitcase/largebox progress semantics for Phase 4B4."""

from __future__ import annotations

import torch

from somaforce_cross.envs.task_adapter import (
    TaskProgressSignals,
    strict_root_height_failure,
)
from somaforce_cross.envs.task_adapters.rigid_object import RigidObjectTaskAdapter


class MovePayloadTaskAdapter(RigidObjectTaskAdapter):
    """Apply the report-frozen payload path, lift, set-down, and failure rules."""

    root_height_failure = 0.25
    lift_threshold = 0.10
    set_down_tolerance = 0.25
    supported_tasks = ("move_suitcase", "move_largebox")

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if self.task_spec.task not in self.supported_tasks:
            raise ValueError(
                "MovePayloadTaskAdapter requires suitcase or largebox contract"
            )
        self.reference_positions = self.reference.data["body_pos_w"][
            :, self.object_ref_body_id
        ].float()
        segments = torch.linalg.vector_norm(
            self.reference_positions[1:] - self.reference_positions[:-1], dim=-1
        )
        self.reference_path_cumulative = torch.cat(
            (torch.zeros(1, device=self.device), segments.cumsum(dim=0))
        )
        self.reference_path_length = self.reference_path_cumulative[-1]
        if not bool(self.reference_path_length > 0):
            raise ValueError("payload reference path length must be positive")
        self.reset_object_z = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.max_object_z = torch.zeros_like(self.reset_object_z)
        self.path_progress = torch.zeros_like(self.reset_object_z)
        self.previous_progress = torch.zeros_like(self.reset_object_z)
        self.current_progress = torch.zeros_like(self.reset_object_z)
        self.lift_latched = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

    def _reset_progress(self, env_ids: torch.Tensor) -> None:
        object_z = self.object.data.root_link_pos_w[env_ids, 2]
        self.reset_object_z[env_ids] = object_z
        self.max_object_z[env_ids] = object_z
        self.path_progress[env_ids] = 0.0
        self.previous_progress[env_ids] = 0.0
        self.current_progress[env_ids] = 0.0
        self.lift_latched[env_ids] = False

    def _advance_progress(self) -> None:
        self.previous_progress.copy_(self.current_progress)
        object_local = self.object.data.root_link_pos_w - self.scene.env_origins
        distances = torch.linalg.vector_norm(
            object_local[:, None] - self.reference_positions[None], dim=-1
        )
        nearest = distances.argmin(dim=-1)
        nearest_progress = self.reference_path_cumulative.index_select(0, nearest)
        self.path_progress.copy_(torch.maximum(self.path_progress, nearest_progress))
        self.current_progress.copy_(self.path_progress / self.reference_path_length)
        self.max_object_z.copy_(
            torch.maximum(self.max_object_z, self.object.data.root_link_pos_w[:, 2])
        )
        self.lift_latched |= (
            self.max_object_z - self.reset_object_z >= self.lift_threshold
        )

    def progress_signals(self) -> TaskProgressSignals:
        root_height = self.robot.data.root_link_pos_w[:, 2]
        failure = strict_root_height_failure(
            root_height, threshold=self.root_height_failure
        )
        exhausted = self.reference_step >= self.reference.length
        object_local = self.object.data.root_link_pos_w - self.scene.env_origins
        set_down_error = torch.linalg.vector_norm(
            object_local - self.reference_positions[-1], dim=-1
        )
        success = (
            exhausted
            & self.lift_latched
            & (set_down_error <= self.set_down_tolerance)
            & ~failure
        )
        return TaskProgressSignals(
            progress=self.current_progress.unsqueeze(1),
            progress_delta=(self.current_progress - self.previous_progress).unsqueeze(
                1
            ),
            success=success.float().unsqueeze(1),
            expected_contact=self.expected_contact().float(),
            contact_truth=self.contact_truth().float(),
            stability_margin=(root_height - self.root_height_failure).unsqueeze(1),
            failure=failure,
            reference_exhausted=exhausted,
        )
