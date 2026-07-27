from __future__ import annotations

import ast
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import smoke_phase4_door_env as runner

from somaforce_cross.envs import NominalActionHistory
from somaforce_cross.residual import (
    ExecutedActionHistory,
    JointMarginLimiter,
    PerJointAuthority,
    ResidualActionComposer,
    VelocityMarginLimiter,
)
from somaforce_cross.scaffold.pretrained_hdmi import (
    HDMI_ACTION_JOINT_NAMES,
    HDMIJointPositionActionRuntime,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / "somaforce_cross/envs/residual_env.py"
GOLDEN_PATH = REPO_ROOT / "somaforce_cross/scaffold/pretrained_hdmi_isaac.py"
ARTIFACT_DIR = REPO_ROOT / "artifacts/scaffolds/hdmi_push_door_hand/v1"


def _composer() -> ResidualActionComposer:
    default = torch.linspace(-0.2, 0.2, 23)
    scale = torch.linspace(0.3, 0.6, 23)
    return ResidualActionComposer(
        JointMarginLimiter(
            default_joint_pos=default,
            joint_lower=torch.full((23,), -3.0),
            joint_upper=torch.full((23,), 3.0),
            action_scale=scale,
            margin=0.0,
        ),
        VelocityMarginLimiter(
            default_joint_pos=default,
            action_scale=scale,
            velocity_limit=torch.full((23,), 100.0),
            dt=0.02,
        ),
    )


def test_c0_nonzero_probe_is_eliminated_bitwise() -> None:
    batch = 64
    nominal = torch.linspace(-1.5, 1.5, batch * 23).reshape(batch, 23)
    output = _composer()(
        nominal,
        torch.ones_like(nominal),
        PerJointAuthority()(0),
        torch.ones(batch, 1),
        torch.ones(batch, 1),
        torch.zeros_like(nominal),
    )
    assert torch.count_nonzero(output.delta_bounded) == 0
    assert torch.count_nonzero(output.delta_safe) == 0
    assert torch.equal(output.a_total, nominal)


def test_audited_sink_delays_then_smooths_normalized_action_and_scales_once() -> None:
    default = torch.zeros(1, 23)
    runtime = HDMIJointPositionActionRuntime(
        default,
        HDMI_ACTION_JOINT_NAMES,
        num_envs=1,
        decimation=4,
        delay=4,
        alpha=0.9,
        action_joint_names=HDMI_ACTION_JOINT_NAMES,
        action_scale=tuple(2.0 for _ in range(23)),
    )
    normalized = torch.ones(1, 23)
    runtime.set_nominal_action(normalized)
    targets = [runtime.substep_target(index) for index in range(4)]
    assert torch.equal(targets[0], torch.zeros_like(targets[0]))
    assert torch.equal(targets[1], torch.zeros_like(targets[1]))
    assert torch.equal(targets[2], torch.zeros_like(targets[2]))
    assert torch.equal(targets[3], torch.zeros_like(targets[3]))

    delayed_targets = [runtime.substep_target(index) for index in range(4)]
    assert torch.allclose(delayed_targets[0], torch.full((1, 23), 1.8))
    assert torch.allclose(runtime.applied_action, torch.full((1, 23), 0.9999))
    assert torch.allclose(delayed_targets[3], torch.full((1, 23), 1.9998))


def test_new_sink_delegates_scaling_to_existing_runtime_only() -> None:
    tree = ast.parse(ENV_PATH.read_text(encoding="utf-8"), filename=str(ENV_PATH))
    sink = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NormalizedDoorActionSink"
    )
    method_names = {
        node.name for node in sink.body if isinstance(node, ast.FunctionDef)
    }
    assert method_names == {
        "__init__",
        "validate_reset",
        "reset",
        "set_action",
        "substep_target",
    }
    source = ast.get_source_segment(ENV_PATH.read_text(encoding="utf-8"), sink)
    assert source is not None
    assert "HDMIJointPositionActionRuntime" in source
    assert "action_scale" in source
    assert "torch.clamp" not in source
    assert "+=" not in source


def test_nominal_scaffold_and_executed_histories_are_separate() -> None:
    nominal = NominalActionHistory(2)
    executed = ExecutedActionHistory(2)
    assert nominal.storage.data_ptr() != executed.storage.data_ptr()
    nominal.push(torch.full((2, 23), 2.0))
    executed.push(torch.full((2, 23), 3.0))
    assert torch.equal(nominal.storage[:, :, 0], torch.full((2, 23), 2.0))
    assert torch.equal(executed.storage[:, :, 0], torch.full((2, 23), 3.0))


def test_golden_runtime_and_door_artifact_remain_immutable() -> None:
    golden_sha = hashlib.sha256(GOLDEN_PATH.read_bytes()).hexdigest()
    assert (
        golden_sha == "e924dadeb945278ee85fad82e31be56f78bce4e62f5bb9f575f5938f6c4868a0"
    )
    for line in (ARTIFACT_DIR / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        actual = hashlib.sha256((ARTIFACT_DIR / relative).read_bytes()).hexdigest()
        assert actual == expected


def test_runner_has_three_modes_nonzero_exit_and_all_row_comparison() -> None:
    path = REPO_ROOT / "scripts/smoke_phase4_door_env.py"
    source = path.read_text(encoding="utf-8")
    assert 'choices=("golden", "residual", "compare")' in source
    assert 'Path("/tmp/somaforce_phase4b3")' in source
    assert "return 0 if passed else 1" in source
    assert "np.argwhere(~equal)[0]" in source
    assert "rows_compared" in source


def _base_trace(path: Path, *, mode: str) -> None:
    shapes = {
        "a_nom": (1, 2, 23),
        "raw_residual": (1, 2, 23),
        "authority": (1, 2, 23),
        "a_total": (1, 2, 23),
        "applied_action": (1, 2, 23),
        "joint_target": (1, 2, 23),
        "scaffold_history": (1, 2, 23, 3),
        "nominal_history": (1, 2, 23, 3),
        "executed_history": (1, 2, 23, 3),
        "reference_step": (1, 2),
        "reference_phase": (1, 2, 1),
        "robot_root_state": (1, 2, 13),
        "robot_joint_pos": (1, 2, 29),
        "robot_joint_vel": (1, 2, 29),
        "door_root_state": (1, 2, 13),
        "door_joint_pos": (1, 2, 1),
        "door_joint_vel": (1, 2, 1),
        "door_applied_effort": (1, 2, 1),
        "env_origin": (1, 2, 3),
        "physics_mismatch": (1, 2, 39),
        "reset_seed": (1, 2),
        "finite": (1, 2),
        "terminated": (1, 2),
        "time_outs": (1, 2),
    }
    arrays = {name: np.zeros(shape, dtype=np.float32) for name, shape in shapes.items()}
    arrays["raw_residual"] = np.ones(shapes["raw_residual"], dtype=np.float32)
    arrays["finite"] = np.ones(shapes["finite"], dtype=bool)
    arrays["a_total"] = arrays["a_nom"].copy()
    arrays["metadata_json"] = np.asarray(
        json.dumps(
            {
                "mode": mode,
                "configuration": {"task": "push_door_hand"},
                "configuration_hash": "test-hash",
                "action_joint_names": [f"action_{i}" for i in range(23)],
                "robot_joint_names": [f"joint_{i}" for i in range(29)],
                "robot_body_names": ["pelvis", "door_panel"],
                "dtype": "torch.float32",
                "device": "cpu",
            },
            sort_keys=True,
        )
    )
    runner._save_trace(path, arrays)


def _compare_args(golden: Path, residual: Path) -> Namespace:
    return Namespace(
        golden_trace=golden,
        residual_trace=residual,
        parity_atol=1.0e-5,
        parity_rtol=1.0e-5,
    )


def _mutated_dtype(array: np.ndarray) -> np.dtype:
    if np.issubdtype(array.dtype, np.bool_):
        return np.dtype(np.uint8)
    if np.issubdtype(array.dtype, np.floating):
        return np.dtype(np.float64 if array.dtype != np.float64 else np.float32)
    return np.dtype(np.float64)


@pytest.mark.parametrize("field", runner.TRACE_FIELDS)
@pytest.mark.parametrize("archive_name", ("golden", "residual"))
def test_compare_rejects_dtype_metadata_mutation_for_each_trace_field(
    tmp_path: Path, field: str, archive_name: str
) -> None:
    golden_path = tmp_path / "golden.npz"
    residual_path = tmp_path / "residual.npz"
    _base_trace(golden_path, mode="golden")
    _base_trace(residual_path, mode="residual")
    target_path = golden_path if archive_name == "golden" else residual_path
    with np.load(target_path, allow_pickle=False) as archive:
        mutated = {name: archive[name] for name in archive.files}
    mutated[field] = mutated[field].astype(_mutated_dtype(mutated[field]))
    np.savez_compressed(target_path, **mutated)

    with pytest.raises(
        ValueError,
        match=rf"{archive_name} metadata dtype mismatch at {field}:",
    ):
        runner._compare(_compare_args(golden_path, residual_path))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("finite", False),
        ("terminated", True),
        ("time_outs", True),
        ("raw_residual", 2.0),
        ("authority", 1.0),
        ("robot_joint_pos", 1.0),
    ),
)
def test_compare_reports_first_divergence_for_each_mutated_trace_field(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
    value: object,
) -> None:
    golden_path = tmp_path / "golden.npz"
    residual_path = tmp_path / "residual.npz"
    _base_trace(golden_path, mode="golden")
    _base_trace(residual_path, mode="residual")
    with np.load(residual_path, allow_pickle=False) as archive:
        mutated = {name: archive[name] for name in archive.files}
    mutated[field] = mutated[field].copy()
    mutated[field].flat[0] = value
    np.savez_compressed(residual_path, **mutated)

    result = runner._compare(_compare_args(golden_path, residual_path))
    assert result == 1
    output = json.loads(capsys.readouterr().out)
    assert output["passed"] is False
    assert output["first_divergence"] is not None
    assert output["first_divergence"]["field"]
    assert "control_step" in output["first_divergence"]
    assert "env_id" in output["first_divergence"]
    assert "component" in output["first_divergence"]


def test_compare_rejects_shape_and_key_contract_mutations(tmp_path: Path) -> None:
    golden_path = tmp_path / "golden.npz"
    residual_path = tmp_path / "residual.npz"
    _base_trace(golden_path, mode="golden")
    _base_trace(residual_path, mode="residual")
    with np.load(residual_path, allow_pickle=False) as archive:
        mutated = {name: archive[name] for name in archive.files if name != "finite"}
    np.savez_compressed(residual_path, **mutated)
    with pytest.raises(ValueError, match="trace key contract mismatch"):
        runner._compare(_compare_args(golden_path, residual_path))

    _base_trace(residual_path, mode="residual")
    with np.load(residual_path, allow_pickle=False) as archive:
        mutated = {name: archive[name] for name in archive.files}
    mutated["robot_joint_pos"] = mutated["robot_joint_pos"][:, :, :-1]
    np.savez_compressed(residual_path, **mutated)
    with pytest.raises(ValueError, match="robot_joint_pos trace shapes differ"):
        runner._compare(_compare_args(golden_path, residual_path))
