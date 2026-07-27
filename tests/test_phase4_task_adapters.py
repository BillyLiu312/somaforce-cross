from __future__ import annotations

import ast
import inspect
from pathlib import Path

import torch

from somaforce_cross.envs.task_adapter import (
    SmokeDoneOutput,
    TaskAdapter,
    TaskProgressSignals,
    door_nominal_physics_mismatch,
    smoke_done_flags,
    stable_env_seeds,
)
from somaforce_cross.envs.mismatch import PerEnvRandomStream


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
        "build_scaffold_observation",
        "build_object_state",
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


def test_phase4b3_sources_are_door_only_and_have_no_task_specific_network() -> None:
    paths = [
        ADAPTER_PATH,
        CFG_PATH,
        REPO_ROOT / "somaforce_cross/envs/residual_env.py",
        REPO_ROOT / "somaforce_cross/envs/task_adapter.py",
    ]
    forbidden_task_names = {
        "push_box",
        "move_suitcase",
        "move_largebox",
    }
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
        assert strings.isdisjoint(forbidden_task_names), path


def test_door_c0_cfg_freezes_exact_schema_and_smoke_values_in_source() -> None:
    source = CFG_PATH.read_text(encoding="utf-8")
    expected_fragments = (
        'task: str = "push_door_hand"',
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
