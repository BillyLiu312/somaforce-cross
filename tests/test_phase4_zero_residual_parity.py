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
from scripts import smoke_phase4_multitask_env as multitask_runner

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
ENV_CFG_PATH = REPO_ROOT / "somaforce_cross/envs/residual_env_cfg.py"
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


def test_generic_sink_delegates_scaling_to_existing_runtime_only() -> None:
    tree = ast.parse(ENV_PATH.read_text(encoding="utf-8"), filename=str(ENV_PATH))
    sink = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NormalizedActionSink"
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


def test_generic_environment_uses_generic_primary_names_with_door_aliases() -> None:
    env_source = ENV_PATH.read_text(encoding="utf-8")
    env_tree = ast.parse(env_source, filename=str(ENV_PATH))
    classes = {
        node.name: node for node in env_tree.body if isinstance(node, ast.ClassDef)
    }
    assert "FrozenScaffoldState" in classes
    assert "NormalizedActionSink" in classes
    assert "FrozenDoorScaffoldState" not in classes
    assert "NormalizedDoorActionSink" not in classes
    generic_init = next(
        node
        for node in classes["SomaForceResidualEnv"].body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    generic_source = ast.get_source_segment(env_source, generic_init)
    assert generic_source is not None
    assert "FrozenScaffoldState(" in generic_source
    assert "NormalizedActionSink(" in generic_source
    assert "FrozenDoorScaffoldState" not in generic_source
    assert "NormalizedDoorActionSink" not in generic_source
    assert "FrozenDoorScaffoldState = FrozenScaffoldState" in env_source
    assert "NormalizedDoorActionSink = NormalizedActionSink" in env_source

    cfg_source = ENV_CFG_PATH.read_text(encoding="utf-8")
    cfg_tree = ast.parse(cfg_source, filename=str(ENV_CFG_PATH))
    cfg_classes = {
        node.name: node for node in cfg_tree.body if isinstance(node, ast.ClassDef)
    }
    profile_source = ast.get_source_segment(cfg_source, cfg_classes["C0SmokeProfile"])
    generic_cfg_source = ast.get_source_segment(
        cfg_source, cfg_classes["SomaForceResidualEnvCfg"]
    )
    assert profile_source is not None and generic_cfg_source is not None
    assert "task: str = DEFAULT_TASK" in profile_source
    assert "DEFAULT_ARTIFACT" in generic_cfg_source
    assert "DEFAULT_TASK" in generic_cfg_source
    assert "DOOR_ARTIFACT" not in generic_cfg_source
    assert "DOOR_ARTIFACT = DEFAULT_ARTIFACT" in cfg_source


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


def _new_snapshot(batch: int = 2) -> dict[str, np.ndarray]:
    snapshot: dict[str, np.ndarray] = {}
    for field in multitask_runner.TRACE_FIELDS:
        if field in {"terminated", "time_outs"}:
            continue
        shape = (batch, *multitask_runner.FIELD_TRAILING_SHAPES[field])
        snapshot[field] = np.zeros(
            shape, dtype=np.dtype(multitask_runner.FIELD_DTYPES[field])
        )
    snapshot["raw_residual"].fill(1.0)
    snapshot["finite"].fill(True)
    return snapshot


def _new_metadata(mode: str, batch: int = 2) -> dict[str, object]:
    configuration = {"task": "push_door_hand"}
    artifact_identity = hashlib.sha256(
        (ARTIFACT_DIR / "SHA256SUMS").read_bytes()
    ).hexdigest()
    return {
        "action_joint_names": list(HDMI_ACTION_JOINT_NAMES),
        "artifact_sha256s_sha256": artifact_identity,
        "configuration": configuration,
        "configuration_hash": hashlib.sha256(
            json.dumps(
                configuration,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest(),
        "device": "cpu",
        "dtype": "torch.float32",
        "mode": mode,
        "object_kind": "articulation",
        "robot_body_names": ["pelvis", "door_panel"],
        "robot_joint_names": [f"joint_{index}" for index in range(29)],
        "task": "push_door_hand",
        "trace_contract_version": multitask_runner.TRACE_CONTRACT_VERSION,
        "trace_dtypes": dict(multitask_runner.FIELD_DTYPES),
        "trace_fields": list(sorted(multitask_runner.TRACE_FIELDS)),
        "trace_shapes": {
            name: [batch, *multitask_runner.FIELD_TRAILING_SHAPES[name]]
            for name in multitask_runner.TRACE_FIELDS
        },
    }


def _write_new_trace(path: Path, *, mode: str, batch: int = 2) -> None:
    snapshot = _new_snapshot(batch)
    multitask_runner._save_trace(
        path,
        [snapshot],
        [np.zeros(batch, dtype=bool)],
        [np.zeros(batch, dtype=bool)],
        _new_metadata(mode, batch),
    )


def _multitask_compare_args(golden: Path, residual: Path) -> Namespace:
    return Namespace(
        golden_trace=golden,
        residual_trace=residual,
        parity_atol=1.0e-5,
        parity_rtol=1.0e-5,
    )


def _read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def test_phase4b4_new_trace_exact_24_field_15_metadata_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    golden = tmp_path / "golden.npz"
    residual = tmp_path / "residual.npz"
    _write_new_trace(golden, mode="golden")
    _write_new_trace(residual, mode="residual")
    assert multitask_runner._compare(_multitask_compare_args(golden, residual)) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] is True
    assert result["rows_compared"] == 2
    assert result["golden_contract"] == "phase4b4_c0_v1"
    assert result["residual_contract"] == "phase4b4_c0_v1"
    with np.load(golden, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        assert set(archive.files) == set(multitask_runner.TRACE_FIELDS) | {
            "metadata_json"
        }
    assert set(metadata) == multitask_runner.NEW_METADATA_KEYS
    assert metadata["trace_fields"] == sorted(multitask_runner.TRACE_FIELDS)


def _mutate_metadata(metadata: dict[str, object], field: str) -> None:
    if field == "action_joint_names":
        metadata[field] = list(reversed(metadata[field]))  # type: ignore[arg-type]
    elif field == "artifact_sha256s_sha256":
        metadata[field] = "0" * 64
    elif field == "configuration":
        metadata[field] = {"task": "push_door_hand", "mutation": True}
    elif field == "configuration_hash":
        metadata[field] = "0" * 64
    elif field == "device":
        metadata[field] = 7
    elif field == "dtype":
        metadata[field] = "torch.float64"
    elif field == "mode":
        metadata[field] = "invalid"
    elif field == "object_kind":
        metadata[field] = "rigid_object"
    elif field == "robot_body_names":
        metadata[field] = ["pelvis", "pelvis"]
    elif field == "robot_joint_names":
        metadata[field] = ["joint"] * 29
    elif field == "task":
        metadata[field] = "push_box"
    elif field == "trace_contract_version":
        metadata[field] = "unknown"
    elif field == "trace_dtypes":
        value = dict(metadata[field])  # type: ignore[arg-type]
        value["a_nom"] = "float64"
        metadata[field] = value
    elif field == "trace_fields":
        metadata[field] = list(metadata[field])[:-1]  # type: ignore[arg-type]
    elif field == "trace_shapes":
        value = dict(metadata[field])  # type: ignore[arg-type]
        value["a_nom"] = [2, 22]
        metadata[field] = value
    else:
        raise AssertionError(field)


@pytest.mark.parametrize("field", sorted(multitask_runner.NEW_METADATA_KEYS))
@pytest.mark.parametrize("archive_name", ("golden", "residual"))
def test_phase4b4_compare_rejects_each_metadata_mutation_on_both_sides(
    tmp_path: Path, field: str, archive_name: str
) -> None:
    golden = tmp_path / "golden.npz"
    residual = tmp_path / "residual.npz"
    _write_new_trace(golden, mode="golden")
    _write_new_trace(residual, mode="residual")
    target = golden if archive_name == "golden" else residual
    arrays = _read_npz(target)
    metadata = json.loads(str(arrays["metadata_json"].item()))
    _mutate_metadata(metadata, field)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(target, **arrays)
    with pytest.raises(ValueError):
        multitask_runner._compare(_multitask_compare_args(golden, residual))


@pytest.mark.parametrize("contract", ("new", "legacy"))
@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    (
        (
            "robot_joint_names",
            list(range(29)),
            r"robot_joint_names must be a length-29 list\[str\] with unique entries",
        ),
        (
            "robot_body_names",
            [1001, 1002],
            r"robot_body_names must be a nonempty list\[str\] with unique entries",
        ),
    ),
)
@pytest.mark.parametrize("invalid_archives", (("golden", "residual"), ("residual",)))
def test_compare_rejects_integer_robot_name_lists_independently_and_when_equal(
    tmp_path: Path,
    contract: str,
    field: str,
    invalid_value: list[int],
    message: str,
    invalid_archives: tuple[str, ...],
) -> None:
    golden = tmp_path / "golden.npz"
    residual = tmp_path / "residual.npz"
    writer = _write_new_trace if contract == "new" else _write_legacy_trace
    writer(golden, mode="golden")
    writer(residual, mode="residual")
    for archive_name in invalid_archives:
        target = golden if archive_name == "golden" else residual
        arrays = _read_npz(target)
        metadata = json.loads(str(arrays["metadata_json"].item()))
        metadata[field] = invalid_value
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
        np.savez_compressed(target, **arrays)
    expected_label = "golden" if "golden" in invalid_archives else "residual"
    with pytest.raises(ValueError, match=rf"{expected_label} {message}"):
        multitask_runner._compare(_multitask_compare_args(golden, residual))


@pytest.mark.parametrize(
    "field",
    (
        "trace_contract_version",
        "task",
        "object_kind",
        "artifact_sha256s_sha256",
    ),
)
@pytest.mark.parametrize("archive_name", ("golden", "residual"))
def test_phase4b4_compare_rejects_missing_required_identity_metadata(
    tmp_path: Path, field: str, archive_name: str
) -> None:
    golden = tmp_path / "golden.npz"
    residual = tmp_path / "residual.npz"
    _write_new_trace(golden, mode="golden")
    _write_new_trace(residual, mode="residual")
    target = golden if archive_name == "golden" else residual
    arrays = _read_npz(target)
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata.pop(field)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(target, **arrays)
    with pytest.raises(ValueError):
        multitask_runner._compare(_multitask_compare_args(golden, residual))


@pytest.mark.parametrize("field", multitask_runner.TRACE_FIELDS)
@pytest.mark.parametrize("archive_name", ("golden", "residual"))
def test_phase4b4_compare_detects_each_trace_field_mutation_on_both_sides(
    tmp_path: Path, field: str, archive_name: str
) -> None:
    golden = tmp_path / "golden.npz"
    residual = tmp_path / "residual.npz"
    _write_new_trace(golden, mode="golden")
    _write_new_trace(residual, mode="residual")
    target = golden if archive_name == "golden" else residual
    arrays = _read_npz(target)
    mutated = arrays[field].copy()
    if mutated.dtype == np.bool_:
        mutated.flat[0] = not bool(mutated.flat[0])
    else:
        mutated.flat[0] += 1
    arrays[field] = mutated
    np.savez_compressed(target, **arrays)
    assert multitask_runner._compare(_multitask_compare_args(golden, residual)) == 1


@pytest.mark.parametrize("field", multitask_runner.TRACE_FIELDS)
@pytest.mark.parametrize("archive_name", ("golden", "residual"))
def test_phase4b4_compare_rejects_each_trace_dtype_mutation_on_both_sides(
    tmp_path: Path, field: str, archive_name: str
) -> None:
    golden = tmp_path / "golden.npz"
    residual = tmp_path / "residual.npz"
    _write_new_trace(golden, mode="golden")
    _write_new_trace(residual, mode="residual")
    target = golden if archive_name == "golden" else residual
    arrays = _read_npz(target)
    arrays[field] = arrays[field].astype(_mutated_dtype(arrays[field]))
    np.savez_compressed(target, **arrays)
    with pytest.raises(ValueError, match="metadata dtype mismatch"):
        multitask_runner._compare(_multitask_compare_args(golden, residual))


def _write_legacy_trace(path: Path, *, mode: str, batch: int = 2) -> None:
    snapshot = _new_snapshot(batch)
    arrays = {name: np.expand_dims(value, 0) for name, value in snapshot.items()}
    arrays["terminated"] = np.zeros((1, batch), dtype=bool)
    arrays["time_outs"] = np.zeros((1, batch), dtype=bool)
    for old, new in multitask_runner.LEGACY_ALIASES.items():
        arrays[old] = arrays.pop(new)
    configuration = {"task": "push_door_hand"}
    fields = sorted(arrays)
    metadata = {
        "action_joint_names": list(HDMI_ACTION_JOINT_NAMES),
        "configuration": configuration,
        "configuration_hash": hashlib.sha256(
            json.dumps(
                configuration,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest(),
        "device": "cpu",
        "dtype": "torch.float32",
        "mode": mode,
        "robot_body_names": ["pelvis", "door_panel"],
        "robot_joint_names": [f"joint_{index}" for index in range(29)],
        "trace_dtypes": {name: str(arrays[name].dtype) for name in fields},
        "trace_fields": fields,
        "trace_shapes": {name: list(arrays[name].shape[1:]) for name in fields},
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **arrays)


@pytest.mark.parametrize("legacy_side", ("golden", "residual"))
def test_legacy_door_is_accepted_on_either_side_and_byte_immutable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    legacy_side: str,
) -> None:
    golden = tmp_path / "golden.npz"
    residual = tmp_path / "residual.npz"
    if legacy_side == "golden":
        _write_legacy_trace(golden, mode="golden")
        _write_new_trace(residual, mode="residual")
    else:
        _write_new_trace(golden, mode="golden")
        _write_legacy_trace(residual, mode="residual")
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (golden, residual)
    }
    assert multitask_runner._compare(_multitask_compare_args(golden, residual)) == 0
    output = json.loads(capsys.readouterr().out)
    assert "legacy_phase4b3_door_unversioned" in {
        output["golden_contract"],
        output["residual_contract"],
    }
    assert output["archives_byte_immutable"] is True
    assert before == {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (golden, residual)
    }


@pytest.mark.parametrize("legacy_side", ("golden", "residual"))
@pytest.mark.parametrize("alias", tuple(multitask_runner.LEGACY_ALIASES))
@pytest.mark.parametrize("mutation", ("missing_source", "target_present"))
def test_legacy_alias_mutations_are_rejected_on_both_sides(
    tmp_path: Path, legacy_side: str, alias: str, mutation: str
) -> None:
    golden = tmp_path / "golden.npz"
    residual = tmp_path / "residual.npz"
    if legacy_side == "golden":
        _write_legacy_trace(golden, mode="golden")
        _write_new_trace(residual, mode="residual")
        target = golden
    else:
        _write_new_trace(golden, mode="golden")
        _write_legacy_trace(residual, mode="residual")
        target = residual
    arrays = _read_npz(target)
    metadata = json.loads(str(arrays["metadata_json"].item()))
    if mutation == "missing_source":
        arrays.pop(alias)
    else:
        arrays[multitask_runner.LEGACY_ALIASES[alias]] = arrays[alias].copy()
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(target, **arrays)
    with pytest.raises(ValueError):
        multitask_runner._compare(_multitask_compare_args(golden, residual))


def test_unknown_unversioned_and_legacy_metadata_map_mutations_are_rejected(
    tmp_path: Path,
) -> None:
    golden = tmp_path / "golden.npz"
    residual = tmp_path / "residual.npz"
    _write_legacy_trace(golden, mode="golden")
    _write_new_trace(residual, mode="residual")
    arrays = _read_npz(golden)
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["unknown"] = True
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(golden, **arrays)
    with pytest.raises(ValueError, match="unknown unversioned"):
        multitask_runner._compare(_multitask_compare_args(golden, residual))

    _write_legacy_trace(golden, mode="golden")
    arrays = _read_npz(golden)
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata["trace_shapes"].pop("door_joint_pos")
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(golden, **arrays)
    with pytest.raises(ValueError, match="trace_shapes key contract"):
        multitask_runner._compare(_multitask_compare_args(golden, residual))
