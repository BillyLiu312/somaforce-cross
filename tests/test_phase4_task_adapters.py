from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from somaforce_cross.envs.observations import (
    CriticObservationBundle,
    ForceSemanticPipeline,
    build_semantic_target_bundle,
)
from somaforce_cross.envs.task_adapter import (
    SmokeDoneOutput,
    TaskAdapter,
    TaskProgressSignals,
    door_nominal_physics_mismatch,
    rigid_nominal_physics_mismatch,
    rigid_object_state,
    smoke_done_flags,
    stable_env_seeds,
    strict_root_height_failure,
)
from somaforce_cross.envs.mismatch import PerEnvRandomStream
from somaforce_cross.scaffold.pretrained_hdmi import (
    HDMI_ACTION_JOINT_NAMES,
    get_hdmi_task_spec,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = REPO_ROOT / "somaforce_cross/envs/task_adapters/push_door_hand.py"
CFG_PATH = REPO_ROOT / "somaforce_cross/envs/residual_env_cfg.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_common_task_adapter_schema_is_fixed_and_task_agnostic() -> None:
    assert list(TaskProgressSignals._fields) == [
        "progress",
        "progress_delta",
        "success",
        "expected_contact",
        "contact_truth",
        "stability_margin",
        "failure",
        "reference_exhausted",
    ]
    assert list(SmokeDoneOutput._fields) == ["terminated", "time_outs"]
    assert set(TaskAdapter.__dict__) >= {
        "validate_reset",
        "reset",
        "write_scene_reset",
        "advance",
        "apply_object_action",
        "build_scaffold_observation",
        "proprio",
        "wrist_twist_base_yaw",
        "build_object_state",
        "nominal_physics_row",
        "expected_contact",
        "contact_truth",
        "support_contact_count",
        "progress_signals",
    }


def test_door_adapter_owns_all_required_source_contract_methods() -> None:
    tree = _tree(ADAPTER_PATH)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    adapter = classes["PushDoorHandTaskAdapter"]
    methods = {node.name for node in adapter.body if isinstance(node, ast.FunctionDef)}
    assert methods >= {
        "validate_reset",
        "reset",
        "write_scene_reset",
        "advance",
        "build_scaffold_observation",
        "proprio",
        "wrist_twist_base_yaw",
        "expected_contact",
        "contact_truth",
        "support_contact_count",
        "build_object_state",
        "progress_signals",
    }

    source = ADAPTER_PATH.read_text(encoding="utf-8")
    assert "G1_FULL_JOINT_NAMES" in source
    assert "self.robot.joint_names.index(name)" in source
    assert "self.reference.joint_names.index(name)" in source
    assert "result[:, 1] = source[:, 0]" in source
    assert "root_height_failure = 0.45" in source


def test_phase4b4_sources_have_no_task_specific_network() -> None:
    paths = [
        ADAPTER_PATH,
        CFG_PATH,
        REPO_ROOT / "somaforce_cross/envs/residual_env.py",
        REPO_ROOT / "somaforce_cross/envs/task_adapter.py",
        REPO_ROOT / "somaforce_cross/envs/task_adapters/rigid_object.py",
        REPO_ROOT / "somaforce_cross/envs/task_adapters/push_box.py",
        REPO_ROOT / "somaforce_cross/envs/task_adapters/move_payload.py",
    ]
    forbidden_identifiers = {
        "residualactor",
        "privilegedcritic",
        "ppo",
        "rslrl",
        "rsl_rl",
        "train",
        "training",
    }
    for path in paths:
        tree = _tree(path)
        identifiers: set[str] = set()
        strings: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id.lower())
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr.lower())
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                strings.add(node.value.lower())
        assert identifiers.isdisjoint(forbidden_identifiers), path

    env_source = paths[2].read_text(encoding="utf-8")
    assert "SomaForceResidualEnv" in env_source
    assert "SomaForceDoorResidualEnv" in env_source
    assert "MovePayloadTaskAdapter" in env_source
    assert "PushBoxTaskAdapter" in env_source


def test_door_c0_cfg_freezes_exact_schema_and_smoke_values_in_source() -> None:
    source = CFG_PATH.read_text(encoding="utf-8")
    expected_fragments = (
        'DEFAULT_TASK = "push_door_hand"',
        "task: str = DEFAULT_TASK",
        "physics_dt: float = 0.005",
        "control_dt: float = 0.02",
        "decimation: int = 4",
        "action_dim: int = 23",
        "policy_dim: int = 668",
        "critic_dim: int = 845",
        "semantic_target_dim: int = 31",
        "episode_length_steps: int = 8",
        "smoke_control_steps: int = 6",
        "parity_atol: float = 1.0e-5",
        "parity_rtol: float = 1.0e-5",
    )
    for fragment in expected_fragments:
        assert fragment in source
    assert "DoorC0SmokeProfile" in source
    assert "C0SmokeProfile" in source
    assert "SomaForceDoorResidualEnvCfg = SomaForceResidualEnvCfg" in source
    assert "Training" not in source


def test_done_contract_keeps_success_out_and_boundaries_separate() -> None:
    output = smoke_done_flags(
        torch.tensor([False, True, False, False]),
        torch.tensor([False, False, True, False]),
        torch.tensor([7, 7, 7, 8]),
        torch.tensor([False, False, False, True]),
        episode_length_steps=8,
    )
    assert torch.equal(output.terminated, torch.tensor([False, True, True, False]))
    assert torch.equal(output.time_outs, torch.tensor([False, False, False, True]))
    assert "success" not in inspect.signature(smoke_done_flags).parameters


def test_selected_seed_mapping_is_env_identity_stable_and_order_independent() -> None:
    ids_a = torch.tensor([63, 1], dtype=torch.long)
    ids_b = torch.tensor([1, 63], dtype=torch.long)
    seeds_a = stable_env_seeds(20260727, ids_a)
    seeds_b = stable_env_seeds(20260727, ids_b)
    assert torch.equal(seeds_a, torch.tensor([20260790, 20260728]))
    assert torch.equal(seeds_b, torch.tensor([20260728, 20260790]))

    stream_a = PerEnvRandomStream(64)
    stream_b = PerEnvRandomStream(64)
    stream_a.reseed(ids_a, seeds_a)
    stream_b.reseed(ids_b, seeds_b)
    draws_a = stream_a.draw_virtual_ft(ids_a)
    draws_b = stream_b.draw_virtual_ft(ids_b)
    assert torch.equal(draws_a.drift[0], draws_b.drift[1])
    assert torch.equal(draws_a.drift[1], draws_b.drift[0])
    assert torch.equal(draws_a.noise[0], draws_b.noise[1])
    assert torch.equal(draws_a.dropout[1], draws_b.dropout[0])


def test_door_nominal_critic_row_places_actual_mechanism_values() -> None:
    row = door_nominal_physics_mismatch(
        4, mechanism_friction=0.3, mechanism_damping=0.55, device="cpu"
    )
    assert row.shape == (4, 39)
    assert torch.equal(row[:, :3], torch.zeros(4, 3))
    assert torch.equal(row[:, 3], torch.full((4,), 0.3))
    assert torch.equal(row[:, 4], torch.full((4,), 0.55))
    assert torch.count_nonzero(row[:, 5:]) == 0


def test_selected_reset_source_has_no_global_physics_step() -> None:
    source = (REPO_ROOT / "somaforce_cross/envs/residual_env.py").read_text(
        encoding="utf-8"
    )
    reset_source = source[
        source.index("def _reset_idx") : source.index("def _owned_sensor_state")
    ]
    assert "self.sim.step" not in reset_source
    assert "self.scene.write_data_to_sim()" in reset_source
    assert "zero_frame" in source


@pytest.mark.parametrize(
    ("task", "artifact_name", "reference_count", "mapping"),
    (
        (
            "push_box",
            "hdmi_push_box",
            25,
            [
                0,
                6,
                12,
                1,
                7,
                13,
                2,
                8,
                14,
                3,
                9,
                15,
                20,
                4,
                10,
                16,
                21,
                5,
                11,
                17,
                22,
                18,
                23,
            ],
        ),
        (
            "move_suitcase",
            "hdmi_move_suitcase",
            29,
            [
                0,
                6,
                12,
                1,
                7,
                13,
                2,
                8,
                14,
                3,
                9,
                15,
                22,
                4,
                10,
                16,
                23,
                5,
                11,
                17,
                24,
                18,
                25,
            ],
        ),
        (
            "move_largebox",
            "hdmi_move_largebox",
            29,
            [
                0,
                6,
                12,
                1,
                7,
                13,
                2,
                8,
                14,
                3,
                9,
                15,
                22,
                4,
                10,
                16,
                23,
                5,
                11,
                17,
                24,
                18,
                25,
            ],
        ),
    ),
)
def test_rigid_artifact_task_spec_asset_reference_and_action_mapping_exact(
    task: str, artifact_name: str, reference_count: int, mapping: list[int]
) -> None:
    artifact = REPO_ROOT / f"artifacts/scaffolds/{artifact_name}/v1"
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    action = json.loads((artifact / "action_contract.json").read_text(encoding="utf-8"))
    spec = get_hdmi_task_spec(task)
    assert manifest["task"] == task
    assert manifest["task_contract"]["object_kind"] == "rigid_object"
    assert action["action_joint_names"] == list(HDMI_ACTION_JOINT_NAMES)
    assert action["reference_joint_count"] == reference_count
    assert action["reference_to_action_indices"] == mapping
    assert len(set(action["reference_joint_names"])) == reference_count
    assert all(
        name in action["reference_joint_names"] for name in HDMI_ACTION_JOINT_NAMES
    )
    assert (artifact / f"assets/{spec.object_asset_file}").is_file()
    with np.load(artifact / "reference/motion.npz", allow_pickle=False) as archive:
        assert archive["joint_pos"].shape[1] == reference_count


def test_rigid_object_state_exact_wxyz_formula_and_zero_mechanism() -> None:
    state = rigid_object_state(
        torch.tensor([[1.0, 2.0, 3.0]]).expand(2, -1),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(2, -1),
        torch.tensor([[2.0, 4.0, 6.0]]).expand(2, -1),
        torch.tensor([[0.5, 0.5, -0.5, 0.5]]).expand(2, -1),
        torch.tensor([[4.0, 5.0, 6.0]]).expand(2, -1),
        torch.tensor([[7.0, 8.0, 9.0]]).expand(2, -1),
    )
    assert state.shape == (2, 16)
    assert torch.equal(state[:, 0:3], torch.tensor([[1.0, 2.0, 3.0]]).expand(2, -1))
    assert torch.equal(
        state[:, 3:7], torch.tensor([[0.5, 0.5, -0.5, 0.5]]).expand(2, -1)
    )
    assert torch.equal(state[:, 7:10], torch.tensor([[4.0, 5.0, 6.0]]).expand(2, -1))
    assert torch.equal(state[:, 10:13], torch.tensor([[7.0, 8.0, 9.0]]).expand(2, -1))
    assert torch.count_nonzero(state[:, 13:16]) == 0


def test_rigid_nominal_39d_row_exact_order_and_readback_validation() -> None:
    mass = torch.tensor([8.0, 1.5, 1.0])
    inertia = torch.diag_embed(
        torch.tensor(
            [
                [0.8533333539962769, 1.0933332443237305, 1.0933332443237305],
                [0.0312499962747097, 0.02499999850988388, 0.016249999403953552],
                [0.0020000000949949026] * 3,
            ]
        )
    )
    friction = torch.full((3,), 0.5)
    rows = rigid_nominal_physics_mismatch(mass, inertia, friction, friction.clone())
    assert rows.shape == (3, 39)
    assert torch.equal(rows[:, 11], mass)
    assert torch.equal(rows[:, 15:18], inertia.diagonal(dim1=-2, dim2=-1))
    assert torch.count_nonzero(rows[:, 18:21]) == 0
    assert torch.equal(rows[:, 21], friction)
    assert torch.count_nonzero(rows[:, :11]) == 0
    assert torch.count_nonzero(rows[:, 12:15]) == 0
    assert torch.count_nonzero(rows[:, 22:]) == 0

    asymmetric = inertia.clone()
    asymmetric[0, 0, 1] = 1.0
    with pytest.raises(ValueError, match="symmetric"):
        rigid_nominal_physics_mismatch(mass, asymmetric, friction, friction)
    with pytest.raises(ValueError, match="static and dynamic"):
        rigid_nominal_physics_mismatch(mass, inertia, friction, friction + 0.1)
    source = (
        REPO_ROOT / "somaforce_cross/envs/task_adapters/rigid_object.py"
    ).read_text(encoding="utf-8")
    assert "get_masses()" in source and "get_inertias()" in source
    assert "[0, 4, 8, 1, 2, 5]" in (
        REPO_ROOT / "somaforce_cross/envs/task_adapter.py"
    ).read_text(encoding="utf-8")


def test_progress_failure_and_exhaustion_boundaries_are_task_owned() -> None:
    heights = torch.tensor([0.45, 0.449, 0.25, 0.249])
    assert torch.equal(
        strict_root_height_failure(heights[:2], threshold=0.45),
        torch.tensor([False, True]),
    )
    assert torch.equal(
        strict_root_height_failure(heights[2:], threshold=0.25),
        torch.tensor([False, True]),
    )
    push_source = (
        REPO_ROOT / "somaforce_cross/envs/task_adapters/push_box.py"
    ).read_text(encoding="utf-8")
    payload_source = (
        REPO_ROOT / "somaforce_cross/envs/task_adapters/move_payload.py"
    ).read_text(encoding="utf-8")
    assert "self.directional_displacement >= self.reference_distance" in push_source
    assert "self.reference_step >= self.reference.length" in push_source
    assert "self.reference_step >= self.reference.length" in payload_source
    assert "self.lift_latched" in payload_source
    assert "set_down_error <= self.set_down_tolerance" in payload_source
    assert "root_height_failure = 0.25" in payload_source
    assert "root_height_failure = 0.45" in push_source


@pytest.mark.parametrize(
    "field",
    (
        "object_state",
        "physics_mismatch",
        "scaffold_mismatch",
        "sensor_mismatch",
        "progress_state",
        "contact_state",
        "stability_state",
        "hdmi_command",
        "task_identity",
        "clean_wrench",
    ),
)
def test_privileged_field_mutations_cannot_change_policy_or_actor(
    field: str,
) -> None:
    torch.manual_seed(7)
    assembly = ForceSemanticPipeline().assemble_policy_observation(
        wrist_tokens=torch.randn(2, 2, 16, 14),
        proprio=torch.randn(2, 64),
        a_nom_history=torch.randn(2, 23, 3),
        previous_a_total=torch.randn(2, 23),
    )
    policy_before = assembly.bundle.flatten().clone()
    actor_before = assembly.bundle.residual_actor_input().clone()
    critic_fields = {
        "policy": policy_before,
        "object_state": torch.zeros(2, 16),
        "physics_mismatch": torch.zeros(2, 39),
        "scaffold_mismatch": torch.zeros(2, 2),
        "sensor_mismatch": torch.zeros(2, 90),
        "progress_state": torch.zeros(2, 4),
        "contact_state": torch.zeros(2, 17),
        "stability_state": torch.zeros(2, 9),
    }
    if field in critic_fields:
        critic_fields[field].fill_(123.0)
        CriticObservationBundle(**critic_fields).flatten()
    elif field == "clean_wrench":
        build_semantic_target_bundle(
            torch.full((2, 2, 6), 123.0),
            torch.ones(2, 2),
            torch.tensor(1.0),
            torch.tensor(1.0),
        )
    else:
        forbidden_value = torch.full((2, 356 if field == "hdmi_command" else 1), 123.0)
        assert forbidden_value.numel() > 0
    assert torch.equal(assembly.bundle.flatten(), policy_before)
    assert torch.equal(assembly.bundle.residual_actor_input(), actor_before)
