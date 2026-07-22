from __future__ import annotations

from pathlib import Path, PureWindowsPath
import json

import numpy as np
import pytest
import torch

from somaforce_cross.scaffold.pretrained_hdmi import (
    HDMI_ACTION_JOINT_NAMES,
    HDMI_ACTION_SCALE,
    HDMI_DEFAULT_JOINT_POS,
    HDMI_PHYSICS_MATERIAL_COMBINE_MODE,
    HDMI_REFERENCE_JOINT_NAMES,
    MOVE_SUITCASE_OBSERVATION_DIMS,
    REFERENCE_TO_ACTION_INDICES,
    HDMIJointPositionActionRuntime,
    HDMIObservationBatch,
    HDMIObservationHistory,
    PretrainedHDMIScaffold,
    PUSH_BOX_OBSERVATION_DIMS,
    get_hdmi_task_spec,
    reference_action,
    reference_to_action,
    sha256_file,
    write_artifact_checksums,
)
from somaforce_cross.scaffold.contracts import ScaffoldTask


ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts/scaffolds/hdmi_push_door_hand/v1"
)
PUSH_BOX_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts/scaffolds/hdmi_push_box/v1"
)
MOVE_SUITCASE_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts/scaffolds/hdmi_move_suitcase/v1"
)
MOVE_LARGEBOX_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "artifacts/scaffolds/hdmi_move_largebox/v1"
)

ARTIFACT_DIRS = (
    ARTIFACT,
    PUSH_BOX_ARTIFACT,
    MOVE_SUITCASE_ARTIFACT,
    MOVE_LARGEBOX_ARTIFACT,
)


def _manifest_strings(value: object, key: str = "$"):
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            yield from _manifest_strings(child_value, f"{key}.{child_key}")
    elif isinstance(value, list):
        for index, child_value in enumerate(value):
            yield from _manifest_strings(child_value, f"{key}[{index}]")
    elif isinstance(value, str):
        yield key, value


@pytest.mark.parametrize("artifact_dir", ARTIFACT_DIRS)
def test_manifest_contains_no_absolute_filesystem_paths(artifact_dir: Path) -> None:
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    absolute_paths = [
        (key, value)
        for key, value in _manifest_strings(manifest)
        if Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
    ]
    assert absolute_paths == []


@pytest.mark.parametrize("artifact_dir", ARTIFACT_DIRS)
def test_artifact_checksums_cover_every_file(artifact_dir: Path) -> None:
    checksum_entries = {}
    for line in (artifact_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, relative_path = line.split("  ", maxsplit=1)
        checksum_entries[relative_path] = digest

    artifact_files = {
        path.relative_to(artifact_dir).as_posix()
        for path in artifact_dir.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS"
        and not path.match("rollout_metrics*.json")
    }
    assert set(checksum_entries) == artifact_files
    for relative_path, expected_digest in checksum_entries.items():
        assert sha256_file(artifact_dir / relative_path) == expected_digest


def test_artifact_checksum_writer_excludes_rollout_metrics(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "manifest.json").write_text("{}\n", encoding="utf-8")
    metrics_path = artifact_dir / "rollout_metrics_heavy.json"
    metrics_path.write_text('{"task": "move_suitcase"}\n', encoding="utf-8")

    write_artifact_checksums(artifact_dir)

    entries = (artifact_dir / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert [entry.split("  ", maxsplit=1)[1] for entry in entries] == [
        "manifest.json",
    ]
    assert sha256_file(metrics_path) not in entries[0]


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


def test_hdmi_spawn_pose_and_material_combine_contract() -> None:
    assert HDMI_DEFAULT_JOINT_POS == {
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
        ".*_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_pitch_joint": 0.2,
    }
    assert HDMI_PHYSICS_MATERIAL_COMBINE_MODE == "multiply"


def test_push_box_task_identity_and_25_joint_reference_mapping() -> None:
    assert ScaffoldTask.PUSH_BOX.value == "push_box"
    assert ScaffoldTask.PUSH_BOX is not ScaffoldTask.HEAVY_PAYLOAD
    task_spec = get_hdmi_task_spec("push_box")
    assert task_spec.observation_dims == PUSH_BOX_OBSERVATION_DIMS
    assert task_spec.network.privileged_encoder_input_dim == 1724
    reference_names = (
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    )
    source = torch.arange(25, dtype=torch.float32).unsqueeze(0)
    selected = reference_to_action(
        source, reference_names, task_spec.action_joint_names
    )
    assert selected.shape == (1, 23)
    expected = [reference_names.index(name) for name in task_spec.action_joint_names]
    assert selected.tolist()[0] == expected
    normalized = reference_action(
        source,
        torch.zeros(1, 23),
        reference_names,
        task_spec.action_joint_names,
        task_spec.action_scale,
    )
    assert normalized.shape == (1, 23)


def test_move_suitcase_task_contract_and_reference_mapping() -> None:
    assert ScaffoldTask.MOVE_SUITCASE.value == "move_suitcase"
    assert ScaffoldTask.MOVE_SUITCASE is not ScaffoldTask.HEAVY_PAYLOAD
    task_spec = get_hdmi_task_spec("move_suitcase")
    assert task_spec.research_category is ScaffoldTask.HEAVY_PAYLOAD
    assert task_spec.observation_dims == MOVE_SUITCASE_OBSERVATION_DIMS
    assert task_spec.network.privileged_encoder_input_dim == 1724
    assert task_spec.object_asset_name == "suitcase"
    assert task_spec.object_body_name == "suitcase"
    assert task_spec.nominal_object_mass == 1.5
    assert task_spec.robot_initial_joint_overrides == {
        "left_wrist_yaw_joint": -0.4,
        "right_wrist_yaw_joint": 0.4,
    }
    assert task_spec.contact_target_offsets == (
        (-0.1, 0.18, 0.25),
        (-0.1, -0.18, 0.25),
    )
    assert task_spec.contact_eef_offsets == (
        (0.05, 0.0, 0.0),
        (0.05, 0.0, 0.0),
    )
    source = torch.arange(29, dtype=torch.float32).unsqueeze(0)
    selected = reference_to_action(
        source, HDMI_REFERENCE_JOINT_NAMES, task_spec.action_joint_names
    )
    assert selected.shape == (1, 23)
    assert selected.tolist()[0] == list(REFERENCE_TO_ACTION_INDICES)


def test_move_largebox_task_contract_and_reference_mapping() -> None:
    assert ScaffoldTask.MOVE_LARGEBOX.value == "move_largebox"
    task_spec = get_hdmi_task_spec("move_largebox")
    assert task_spec.research_category is ScaffoldTask.HEAVY_PAYLOAD
    assert task_spec.observation_dims == MOVE_SUITCASE_OBSERVATION_DIMS
    assert task_spec.network.privileged_encoder_input_dim == 1724
    assert task_spec.object_asset_name == "largebox"
    assert task_spec.object_asset_file == "largebox.urdf"
    assert task_spec.object_body_name == "largebox_link"
    assert task_spec.nominal_object_mass == 1.0
    assert task_spec.training_mass_range == (0.8, 1.2)
    assert task_spec.contact_target_offsets == (
        (-0.027635, 0.244158, 0.099234),
        (0.198793, -0.151816, 0.149164),
    )
    source = torch.arange(29, dtype=torch.float32).unsqueeze(0)
    selected = reference_to_action(
        source, HDMI_REFERENCE_JOINT_NAMES, task_spec.action_joint_names
    )
    assert selected.shape == (1, 23)
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


@pytest.mark.skipif(
    not (PUSH_BOX_ARTIFACT / "manifest.json").is_file(),
    reason="local HDMI push-box artifact not materialized",
)
def test_push_box_policy_contract_is_frozen_and_matches_oracle_fixture() -> None:
    scaffold = PretrainedHDMIScaffold.from_artifact(PUSH_BOX_ARTIFACT)
    assert scaffold.task_spec.task == "push_box"
    assert scaffold.task_spec.observation_dims["object"] == 10
    assert len(scaffold.task_spec.reference_joint_names) == 25
    assert scaffold.task_spec.contact_target_offsets == (
        (0.0, -0.2, 0.8),
        (0.0, 0.2, 0.8),
    )
    assert scaffold.policy.priv_fc.in_features == 1724
    assert scaffold.policy.actor_fc1.in_features == 861
    assert scaffold.policy.actor_mean.out_features == 23
    assert not scaffold.training and not scaffold.policy.training
    assert all(not parameter.requires_grad for parameter in scaffold.parameters())

    with np.load(PUSH_BOX_ARTIFACT / "parity/source_outputs.npz", allow_pickle=False) as data:
        observation = HDMIObservationBatch(
            command=torch.from_numpy(data["command"]),
            policy=torch.from_numpy(data["policy"]),
            object=torch.from_numpy(data["object"]),
            privileged=torch.from_numpy(data["privileged"]),
            reference_action=torch.from_numpy(data["reference_action"]),
        )
        oracle = torch.from_numpy(data["oracle_action"])
    actual = scaffold.nominal_action(observation)
    assert actual.shape == (4, 23)
    assert float((actual - oracle).abs().max()) <= 1e-5
    assert not actual.requires_grad


@pytest.mark.skipif(
    not (MOVE_SUITCASE_ARTIFACT / "manifest.json").is_file(),
    reason="local HDMI move-suitcase artifact not materialized",
)
def test_move_suitcase_policy_contract_provenance_and_oracle_fixture() -> None:
    scaffold = PretrainedHDMIScaffold.from_artifact(MOVE_SUITCASE_ARTIFACT)
    assert scaffold.task_spec.task == "move_suitcase"
    assert scaffold.task_spec.research_category is ScaffoldTask.HEAVY_PAYLOAD
    assert scaffold.policy.priv_fc.in_features == 1724
    assert scaffold.policy.actor_fc1.in_features == 861
    assert scaffold.policy.actor_mean.out_features == 23
    assert not scaffold.training and not scaffold.policy.training
    assert all(not parameter.requires_grad for parameter in scaffold.parameters())

    with np.load(
        MOVE_SUITCASE_ARTIFACT / "parity/source_outputs.npz", allow_pickle=False
    ) as data:
        observation = HDMIObservationBatch(
            command=torch.from_numpy(data["command"]),
            policy=torch.from_numpy(data["policy"]),
            object=torch.from_numpy(data["object"]),
            privileged=torch.from_numpy(data["privileged"]),
            reference_action=torch.from_numpy(data["reference_action"]),
        )
        oracle = torch.from_numpy(data["oracle_action"])
    actual = scaffold.nominal_action(observation)
    assert actual.shape == (4, 23)
    assert float((actual - oracle).abs().max()) <= 1e-5
    assert not actual.requires_grad

    manifest = json.loads(
        (MOVE_SUITCASE_ARTIFACT / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["checkpoint_audit"]["passed"] is True
    assert manifest["checkpoint_audit"]["forbidden_parameter_paths"] == []
    assert manifest["checkpoint_audit"]["run_saved_policy_source_checked"] is True
    assert manifest["task_contract"]["nominal_object_mass"] == 1.5
    assert manifest["task_contract"]["research_category"] == "heavy_payload"
    assert manifest["material_dependencies"][0]["required_for_headless_physics"] is False
    assert manifest["source"]["source_files"]["run_saved_policy"]["sha256"] == (
        "de7b1546227b64e6ba7f1269753f6a445113838712e252f00006501604f2fd18"
    )
    door_g1 = ARTIFACT / "assets/g1.usd"
    assert sha256_file(MOVE_SUITCASE_ARTIFACT / "assets/g1.usd") == sha256_file(
        door_g1
    )


@pytest.mark.skipif(
    not (MOVE_LARGEBOX_ARTIFACT / "manifest.json").is_file(),
    reason="local HDMI move-largebox artifact not materialized",
)
def test_move_largebox_policy_contract_provenance_and_oracle_fixture() -> None:
    scaffold = PretrainedHDMIScaffold.from_artifact(MOVE_LARGEBOX_ARTIFACT)
    assert scaffold.task_spec.task == "move_largebox"
    assert scaffold.task_spec.research_category is ScaffoldTask.HEAVY_PAYLOAD
    assert scaffold.policy.priv_fc.in_features == 1724
    assert scaffold.policy.actor_fc1.in_features == 861
    assert scaffold.policy.actor_mean.out_features == 23
    assert not scaffold.training and not scaffold.policy.training
    assert all(not parameter.requires_grad for parameter in scaffold.parameters())

    with np.load(
        MOVE_LARGEBOX_ARTIFACT / "parity/source_outputs.npz", allow_pickle=False
    ) as data:
        observation = HDMIObservationBatch(
            command=torch.from_numpy(data["command"]),
            policy=torch.from_numpy(data["policy"]),
            object=torch.from_numpy(data["object"]),
            privileged=torch.from_numpy(data["privileged"]),
            reference_action=torch.from_numpy(data["reference_action"]),
        )
        oracle = torch.from_numpy(data["oracle_action"])
    actual = scaffold.nominal_action(observation)
    assert actual.shape == (4, 23)
    assert float((actual - oracle).abs().max()) <= 1e-5

    manifest = json.loads(
        (MOVE_LARGEBOX_ARTIFACT / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["checkpoint_audit"]["passed"] is True
    assert manifest["task_contract"]["object_asset_file"] == "largebox.urdf"
    assert manifest["task_contract"]["training_mass_range"] == [0.8, 1.2]
    assert manifest["license"]["redistribution"] == (
        "prohibited_until_permissions_are_documented"
    )
    assert (MOVE_LARGEBOX_ARTIFACT / "assets/largebox.obj").is_file()


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
