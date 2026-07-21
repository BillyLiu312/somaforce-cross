from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from somaforce_cross.scaffold.pretrained_hdmi import (
    HDMI_ACTION_JOINT_NAMES,
    HDMI_ACTION_SCALE,
    HDMI_REFERENCE_JOINT_NAMES,
    REFERENCE_TO_ACTION_INDICES,
    HDMIJointPositionActionRuntime,
    HDMIObservationBatch,
    HDMIObservationHistory,
    PretrainedHDMIScaffold,
    reference_to_action,
)


ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts/scaffolds/hdmi_push_door_hand/v1"
)


def test_audited_action_order_and_reference_mapping() -> None:
    assert len(HDMI_ACTION_JOINT_NAMES) == 23
    assert len(set(HDMI_ACTION_JOINT_NAMES)) == 23
    assert len(REFERENCE_TO_ACTION_INDICES) == 23
    assert all("wrist" not in name for name in HDMI_ACTION_JOINT_NAMES)
    assert {name for name in HDMI_REFERENCE_JOINT_NAMES if "wrist" in name} == {
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    }
    canonical = torch.arange(29, dtype=torch.float32).unsqueeze(0)
    selected = reference_to_action(canonical)
    assert selected.tolist()[0] == list(REFERENCE_TO_ACTION_INDICES)


def test_reset_history_previous_action_and_phase() -> None:
    history = HDMIObservationHistory(2)
    env_ids = torch.tensor([0, 1])
    root_ang_vel = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    gravity = torch.tensor([[0.0, 0.0, -1.0], [0.1, 0.0, -0.99]])
    joints = torch.arange(58, dtype=torch.float32).reshape(2, 29)
    history.reset(
        env_ids,
        root_ang_vel=root_ang_vel,
        projected_gravity=gravity,
        joint_pos=joints,
        motion_length=torch.tensor([573, 573]),
    )
    assert torch.equal(history.root_ang_vel, root_ang_vel[:, None].expand(-1, 9, -1))
    assert torch.equal(history.projected_gravity, gravity[:, None].expand(-1, 9, -1))
    assert torch.equal(history.joint_pos, joints[:, None].expand(-1, 9, -1))
    assert torch.count_nonzero(history.previous_actions) == 0
    assert torch.equal(history.phase, torch.zeros(2, 1))
    assert history.policy_observation().shape == (2, 249)

    first = torch.arange(23, dtype=torch.float32).expand(2, -1)
    second = first + 100
    history.push_action(first)
    history.push_action(second)
    assert torch.equal(history.previous_actions[:, :, 0], second)
    assert torch.equal(history.previous_actions[:, :, 1], first)
    assert torch.count_nonzero(history.previous_actions[:, :, 2]) == 0
    history.advance_reference()
    assert torch.allclose(history.phase, torch.full((2, 1), 1.0 / 573.0))


def test_zero_hook_and_action_scaling_are_exact() -> None:
    articulation_names = list(HDMI_ACTION_JOINT_NAMES) + [
        name for name in HDMI_REFERENCE_JOINT_NAMES if "wrist" in name
    ]
    default = torch.zeros(1, 29)
    runtime = HDMIJointPositionActionRuntime(
        default,
        articulation_names,
        num_envs=1,
        delay=2,
        alpha=1.0,
    )
    a_nom = torch.linspace(-1.0, 1.0, 23).unsqueeze(0)
    runtime.set_nominal_action(a_nom)
    assert torch.equal(runtime.received_action, a_nom)
    # delay=2 selects the newest action at substep 2.
    runtime.substep_target(0)
    runtime.substep_target(1)
    target = runtime.substep_target(2)
    expected = a_nom * torch.tensor(HDMI_ACTION_SCALE)
    assert torch.equal(target[:, runtime.joint_ids], expected)
    wrist_ids = [articulation_names.index(name) for name in articulation_names if "wrist" in name]
    assert torch.count_nonzero(target[:, wrist_ids]) == 0


@pytest.mark.skipif(not (ARTIFACT / "manifest.json").is_file(), reason="local HDMI artifact not materialized")
def test_portable_policy_is_frozen_eval_and_matches_oracle_fixture() -> None:
    scaffold = PretrainedHDMIScaffold.from_artifact(ARTIFACT)
    assert not scaffold.training
    assert not scaffold.policy.training
    assert all(not parameter.requires_grad for parameter in scaffold.parameters())
    with pytest.raises(RuntimeError, match="frozen"):
        scaffold.train(True)

    with np.load(ARTIFACT / "parity/source_outputs.npz", allow_pickle=False) as data:
        observation = HDMIObservationBatch(
            command=torch.from_numpy(data["command"]),
            policy=torch.from_numpy(data["policy"]),
            object=torch.from_numpy(data["object"]),
            privileged=torch.from_numpy(data["privileged"]),
            reference_action=torch.from_numpy(data["reference_action"]),
        )
        oracle = torch.from_numpy(data["oracle_action"])
    actual = scaffold(observation)
    assert torch.equal(actual, scaffold.nominal_action(observation))
    assert actual.shape == (4, 23)
    assert float((actual - oracle).abs().max()) <= 1e-5
    assert not actual.requires_grad


def test_observation_contract_rejects_wrong_shape() -> None:
    observation = HDMIObservationBatch(
        command=torch.zeros(1, 355),
        policy=torch.zeros(1, 249),
        object=torch.zeros(1, 7),
        privileged=torch.zeros(1, 1714),
        reference_action=torch.zeros(1, 23),
    )
    with pytest.raises(ValueError, match="command"):
        observation.validate()
