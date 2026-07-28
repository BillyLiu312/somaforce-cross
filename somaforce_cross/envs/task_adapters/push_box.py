"""Push-box progress semantics on the shared rigid-object adapter."""

from __future__ import annotations

import torch

from somaforce_cross.envs.task_adapter import (
    TaskProgressSignals,
    strict_root_height_failure,
)
from somaforce_cross.envs.task_adapters.rigid_object import RigidObjectTaskAdapter


class PushBoxTaskAdapter(RigidObjectTaskAdapter):
    """Apply the report-frozen push-box progress and failure formulas."""

    root_height_failure = 0.45

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if self.task_spec.task != "push_box":
            raise ValueError("PushBoxTaskAdapter requires the push_box contract")
        reference_xy = self.reference.data["body_pos_w"][
            :, self.object_ref_body_id, :2
        ].float()
        delta = reference_xy[-1] - reference_xy[0]
        self.reference_distance = torch.linalg.vector_norm(delta)
        if not bool(self.reference_distance > 0):
            raise ValueError("push-box reference has no XY task direction")
        self.reference_direction = delta / self.reference_distance
        self.reset_object_xy = torch.zeros(
            self.num_envs, 2, device=self.device, dtype=torch.float32
        )
        self.previous_progress = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )
        self.current_progress = torch.zeros_like(self.previous_progress)
        self.directional_displacement = torch.zeros_like(self.previous_progress)

    def _reset_progress(self, env_ids: torch.Tensor) -> None:
        self.reset_object_xy[env_ids] = self.object.data.root_link_pos_w[env_ids, :2]
        self.previous_progress[env_ids] = 0.0
        self.current_progress[env_ids] = 0.0
        self.directional_displacement[env_ids] = 0.0

    def _advance_progress(self) -> None:
        self.previous_progress.copy_(self.current_progress)
        displacement = self.object.data.root_link_pos_w[:, :2] - self.reset_object_xy
        self.directional_displacement.copy_(
            (displacement * self.reference_direction).sum(dim=-1)
        )
        self.current_progress.copy_(
            self.directional_displacement / self.reference_distance
        )

    def progress_signals(self) -> TaskProgressSignals:
        root_height = self.robot.data.root_link_pos_w[:, 2]
        return TaskProgressSignals(
            progress=self.current_progress.unsqueeze(1),
            progress_delta=(self.current_progress - self.previous_progress).unsqueeze(
                1
            ),
            success=(self.directional_displacement >= self.reference_distance)
            .float()
            .unsqueeze(1),
            expected_contact=self.expected_contact().float(),
            contact_truth=self.contact_truth().float(),
            stability_margin=(root_height - self.root_height_failure).unsqueeze(1),
            failure=strict_root_height_failure(
                root_height, threshold=self.root_height_failure
            ),
            reference_exhausted=self.reference_step >= self.reference.length,
        )
