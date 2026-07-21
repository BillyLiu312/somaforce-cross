from __future__ import annotations

import numpy as np
import torch

from somaforce_cross.scaffold import (
    G1_FULL_JOINT_NAMES,
    ReferenceLibrary,
    ReferenceSource,
    ScaffoldTask,
    hdmi_mapping_to_reference,
    omomo_retarget_mapping_to_reference,
    reconstruct_contact_targets,
)


def test_hdmi_mapping_builds_door_reference_with_object_joint() -> None:
    frames = 5
    bodies = 4
    joint_pos = torch.zeros(frames, len(G1_FULL_JOINT_NAMES) + 1)
    joint_pos[:, -1] = torch.linspace(0.0, 0.5, frames)
    body_pos = torch.zeros(frames, bodies, 3)
    body_pos[:, 1] = torch.tensor([0.4, 0.2, 1.0])
    body_pos[:, 2, 0] = torch.linspace(0.5, 0.8, frames)
    body_pos[:, 2, 2] = 1.0
    body_pos[:, 3, 0] = 0.7
    body_quat = _identity_quaternion(frames, bodies)
    contact = torch.tensor([[0], [1], [1], [1], [0]], dtype=torch.bool)

    episode = hdmi_mapping_to_reference(
        {
            "joint_pos": joint_pos,
            "body_pos_w": body_pos,
            "body_quat_w": body_quat,
            "object_contact": contact,
        },
        episode_id="door-001",
        source_clip_id="push-door-hand",
        task=ScaffoldTask.PUSH_PULL_DOOR,
        hand_body_indices=(1, 2),
        object_body_index=3,
        fps=50.0,
        source_revision="test",
    )

    assert episode.metadata.source == ReferenceSource.HDMI
    assert episode.metadata.retarget_version == "native"
    assert episode.joint_pos.shape == (frames, 29)
    assert episode.object_joint_pos.shape == (frames, 1)
    assert episode.body_pos_w.shape == (frames, bodies - 1, 3)
    assert torch.equal(episode.contact_intent[:, 1], contact[:, 0])
    assert not torch.any(episode.contact_intent[:, 0])
    assert torch.all(episode.reference_valid)
    assert torch.allclose(
        episode.contact_target_obj[-1, 1], torch.tensor([0.1, 0.0, 1.0])
    )


def test_omomo_retarget_builds_payload_reference_and_contact_targets() -> None:
    data = _retargeted_omomo_mapping()

    episode = omomo_retarget_mapping_to_reference(
        data,
        episode_id="payload-001",
        source_clip_id="omomo-box-clip",
        retarget_version="retarget-v1",
        source_revision="omomo-test",
        retarget_quality={"foot_slip_m": 0.01, "grasp_error_m": 0.02},
    )

    assert episode.metadata.source == ReferenceSource.OMOMO
    assert episode.metadata.task == ScaffoldTask.HEAVY_PAYLOAD
    assert episode.metadata.retarget_quality["foot_slip_m"] == 0.01
    assert episode.object_joint_pos.shape == (4, 0)
    assert episode.contact_target_obj.shape == (4, 2, 3)
    assert torch.allclose(
        episode.contact_target_obj[0, 0], torch.tensor([0.0, 0.2, 0.0])
    )
    assert torch.allclose(
        episode.contact_target_obj[0, 1], torch.tensor([0.0, -0.2, 0.0])
    )
    assert torch.all(episode.reference_valid)


def test_omomo_adapter_rejects_raw_incomplete_data() -> None:
    try:
        omomo_retarget_mapping_to_reference(
            {"joint_pos": torch.zeros(3, 29)},
            episode_id="bad",
            source_clip_id="raw-omomo",
            retarget_version="none",
        )
    except ValueError as exc:
        assert "Raw OMOMO is not accepted" in str(exc)
    else:
        raise AssertionError("raw incomplete OMOMO data was accepted")


def test_reference_library_round_trip(tmp_path) -> None:
    episode = omomo_retarget_mapping_to_reference(
        _retargeted_omomo_mapping(),
        episode_id="payload-library",
        source_clip_id="clip",
        retarget_version="v1",
    )
    path = tmp_path / "references.pt"
    ReferenceLibrary((episode,)).save(path)

    loaded = ReferenceLibrary.load(path)

    assert len(loaded) == 1
    restored = loaded.get("payload-library")
    assert restored.metadata.source == ReferenceSource.OMOMO
    assert torch.equal(restored.joint_pos, episode.joint_pos)
    assert loaded.select(task=ScaffoldTask.HEAVY_PAYLOAD) == [restored]


def test_contact_target_reconstruction_respects_object_rotation() -> None:
    hand_pose = torch.zeros(2, 2, 7)
    hand_pose[..., 3] = 1.0
    hand_pose[..., 0] = 1.0
    object_pose = torch.zeros(2, 7)
    half = 2**-0.5
    object_pose[:, 3] = half
    object_pose[:, 6] = half

    targets = reconstruct_contact_targets(hand_pose, object_pose)

    assert torch.allclose(targets[..., 0], torch.zeros(2, 2), atol=1e-6)
    assert torch.allclose(targets[..., 1], -torch.ones(2, 2), atol=1e-6)


def _retargeted_omomo_mapping() -> dict[str, np.ndarray | torch.Tensor]:
    frames = 4
    joint_pos = torch.zeros(frames, len(G1_FULL_JOINT_NAMES))
    joint_pos[:, 0] = torch.linspace(0.0, 0.2, frames)
    body_pos = torch.zeros(frames, 3, 3)
    body_pos[:, 0, 2] = 0.76
    body_quat = _identity_quaternion(frames, 3)
    object_pose = torch.zeros(frames, 7)
    object_pose[:, 0] = torch.linspace(0.5, 0.8, frames)
    object_pose[:, 2] = 0.9
    object_pose[:, 3] = 1.0
    hand_pose = torch.zeros(frames, 2, 7)
    hand_pose[:, 0, :3] = object_pose[:, :3] + torch.tensor([0.0, 0.2, 0.0])
    hand_pose[:, 1, :3] = object_pose[:, :3] + torch.tensor([0.0, -0.2, 0.0])
    hand_pose[..., 3] = 1.0
    return {
        "joint_pos": joint_pos,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat,
        "hand_pose_w": hand_pose,
        "object_root_pose_w": object_pose,
        "contact_intent": torch.ones(frames, 2, dtype=torch.bool),
        "contact_confidence": torch.full((frames, 2), 0.9),
    }


def _identity_quaternion(frames: int, count: int) -> torch.Tensor:
    quaternion = torch.zeros(frames, count, 4)
    quaternion[..., 0] = 1.0
    return quaternion
