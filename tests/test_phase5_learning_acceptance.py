from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import pytest
import torch

from somaforce_cross.learning.acceptance import (
    FixedSemanticProbe,
    canonical_sha256,
    gradient_summary,
    load_learning_acceptance_config,
    semantic_probe_metrics,
    source_allowlist_sha256,
    source_sha256,
    validate_final_evaluation,
    validate_learning_acceptance_config,
    validate_learning_gates,
)
from scripts.train_phase5 import (
    _load_semantic_probe_artifact,
    _persist_final_metrics,
    _save_semantic_probe_artifact,
    _sha256,
    _write_final_evaluation_progress,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/phase5_learning_acceptance_v1.json"


def _config():
    return load_learning_acceptance_config(CONFIG_PATH)


def _target(batch: int = 4) -> torch.Tensor:
    p_dir = torch.full((batch, 13), 1.0 / 13.0)
    p_mag = torch.full((batch, 5), 1.0 / 5.0)
    return torch.cat(
        (torch.zeros(batch, 12), p_dir, p_mag, torch.ones(batch, 1)), dim=1
    )


def _probe_metrics() -> dict[str, float | int]:
    target = _target()
    return semantic_probe_metrics(target[:, 12:25], target[:, 25:30], target)


def _learning_metrics() -> dict[str, object]:
    initial = {
        **_probe_metrics(),
        "semantic_dir_loss": 1.0,
        "semantic_mag_loss": 1.0,
        "semantic_total": 0.1,
    }
    final = {
        **_probe_metrics(),
        "semantic_dir_loss": 0.8,
        "semantic_mag_loss": 0.8,
        "semantic_total": 0.08,
        "target_aggregate_entropy": 0.6,
        "prediction_aggregate_entropy": 0.55,
        "target_max_aggregate_class_probability": 0.3,
        "prediction_max_aggregate_class_probability": 0.3,
    }
    return {
        "training_contact_transitions": 50000,
        "initial_probe": initial,
        "recent_probe": [dict(final), dict(final), dict(final)],
        "final_probe": final,
        "gradients": {
            "auxiliary_gradient_rms": 0.2,
            "ppo_gradient_rms": 1.0,
            "rms_aux_to_ppo": 0.2,
            "ratio_p95": 0.4,
        },
    }


def _final_metrics() -> dict[str, object]:
    return {
        "outcome": {"success": 0.8, "failure": 0.02, "episode_invalid": 0},
        "retention": {
            "median_progress": 2.5,
            "scaffold_only_median_progress": 2.6,
            "trained_to_scaffold_ratio": 0.96,
        },
        "action": {
            "contact_saturation_fraction": 0.04,
            "contact_saturation_window_max": 0.09,
        },
        "residual": {
            "contact_mean_delta_safe_norm": 0.02,
            "contact_norm_fraction_ge_min": 0.3,
        },
    }


def test_learning_acceptance_config_freezes_single_task_budget_and_hash_bindings() -> (
    None
):
    config = _config()
    assert config.payload["contract_version"] == "phase5_learning_acceptance_v4"
    assert config.payload["thresholds"]["contact_bearing_rms_gradient_ratio_max"] == 1.0
    run = config.payload["run"]
    assert run["task"] == "push_door_hand"
    assert run["stage"] == "C1"
    assert run["episode_horizon"] == 508
    assert run["iterations"] * run["num_envs"] * run["num_steps_per_env"] == 5001216
    assert config.payload["probe"]["samples"] == 4096
    assert "contact_bearing_minibatch_ratio_p95_max" not in config.payload["thresholds"]
    assert "nominal_progress_median_min" not in config.payload["thresholds"]
    assert "contact_bearing_saturation_fraction_max" not in config.payload["thresholds"]
    assert config.payload["evaluation"] == {
        "c1_episodes": 256,
        "deterministic_actor_mean": True,
        "nominal_paired_episodes": 128,
    }


def test_learning_acceptance_config_rejects_frozen_binding_mutation() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["bindings"]["phase5_raw_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bind"):
        validate_learning_acceptance_config(payload)


def test_learning_acceptance_config_canonical_identity_is_stable() -> None:
    config = _config()
    assert config.canonical_sha256 == canonical_sha256(config.payload)
    copied = copy.deepcopy(config.payload)
    assert canonical_sha256(copied) == config.canonical_sha256


def test_source_allowlist_hash_is_deterministic_and_file_sensitive(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    first = source_sha256([source])
    assert source_allowlist_sha256(first) == source_allowlist_sha256(dict(first))
    source.write_text("value = 2\n", encoding="utf-8")
    assert source_sha256([source]) != first


def test_fixed_probe_retains_exactly_contact_weighted_samples() -> None:
    probe = FixedSemanticProbe(3)
    policy = torch.zeros(4, 668)
    target = _target()
    target[1, 30] = 0.0
    assert probe.add(policy, target) == 3
    retained_policy, retained_target = probe.tensors()
    assert retained_policy.shape == (3, 668)
    assert retained_target.shape == (3, 31)
    assert torch.all(retained_target[:, 30] > 0.0)


def test_fixed_probe_rejects_incomplete_contact_collection() -> None:
    probe = FixedSemanticProbe(2)
    target = _target(batch=1)
    assert probe.add(torch.zeros(1, 668), target) == 1
    with pytest.raises(ValueError, match="exactly"):
        probe.tensors()


def test_semantic_probe_metrics_report_finite_aggregate_distribution_values() -> None:
    target = _target()
    metrics = semantic_probe_metrics(target[:, 12:25], target[:, 25:30], target)
    assert metrics["samples"] == 4
    assert metrics["semantic_total"] == 0.0
    assert metrics["target_aggregate_entropy"] > 0.05
    assert metrics["prediction_max_aggregate_class_probability"] == pytest.approx(
        0.2, abs=1.0e-7, rel=0.0
    )


def test_gradient_summary_uses_contact_minibatch_rms_and_p95() -> None:
    summary = gradient_summary(
        (
            {"ppo_grad_norm": 2.0, "aux_grad_norm": 0.5},
            {"ppo_grad_norm": 1.0, "aux_grad_norm": 1.0},
            {"ppo_grad_norm": 0.0, "aux_grad_norm": 0.25},
        )
    )
    assert summary["contact_bearing_minibatches"] == 3
    assert summary["ppo_gradient_rms"] > 0.0
    assert summary["auxiliary_gradient_rms"] > 0.0
    assert summary["ratio_p95"] == 1.0


def test_learning_gates_require_semantic_improvement_and_gradient_dominance() -> None:
    assert validate_learning_gates(_learning_metrics(), _config())
    p95_diagnostic = _learning_metrics()
    p95_diagnostic["gradients"]["ratio_p95"] = 1.379668274878303  # type: ignore[index]
    assert validate_learning_gates(p95_diagnostic, _config())
    v4_passing = _learning_metrics()
    v4_passing["gradients"]["rms_aux_to_ppo"] = 0.51  # type: ignore[index]
    assert validate_learning_gates(v4_passing, _config())
    broken = _learning_metrics()
    broken["gradients"]["rms_aux_to_ppo"] = 1.000001  # type: ignore[index]
    with pytest.raises(AssertionError, match="RMS"):
        validate_learning_gates(broken, _config())


def test_final_gates_require_deterministic_outcome_retention_and_residual() -> None:
    real = _final_metrics()
    real["retention"] = {
        "median_progress": 1.6424626111984253,
        "scaffold_only_median_progress": 1.6180477738380432,
        "trained_to_scaffold_ratio": 1.0150890707648696,
    }
    real["action"]["contact_saturation_fraction"] = 0.05834576679308731  # type: ignore[index]
    real["action"]["contact_saturation_window_max"] = 0.05902777846333467  # type: ignore[index]
    assert validate_final_evaluation(real, _config())
    broken_ratio = copy.deepcopy(real)
    broken_ratio["retention"]["trained_to_scaffold_ratio"] = 0.949  # type: ignore[index]
    with pytest.raises(AssertionError, match="retention ratio"):
        validate_final_evaluation(broken_ratio, _config())
    broken_window = copy.deepcopy(real)
    broken_window["action"]["contact_saturation_window_max"] = 0.101  # type: ignore[index]
    with pytest.raises(AssertionError, match="saturation window"):
        validate_final_evaluation(broken_window, _config())
    broken = _final_metrics()
    broken["outcome"]["success"] = 0.7  # type: ignore[index]
    with pytest.raises(AssertionError, match="success"):
        validate_final_evaluation(broken, _config())


def test_semantic_probe_artifact_round_trip_and_learning_lifecycle_boundary(
    tmp_path: Path,
) -> None:
    context = {
        "contracts": {
            "phase4_raw_sha256": "phase4-raw",
            "phase4_canonical_sha256": "phase4-canonical",
            "phase5_raw_sha256": "phase5-raw",
            "phase5_canonical_sha256": "phase5-canonical",
            "learning_acceptance_raw_sha256": "acceptance-raw",
            "learning_acceptance_canonical_sha256": "acceptance-canonical",
        },
        "source_allowlist_sha256": "source-allowlist",
    }
    policy = torch.zeros(4096, 668)
    semantic_target = torch.zeros(4096, 31)
    semantic_target[:, 30] = 1.0
    path = tmp_path / "semantic_probe.pt"
    _save_semantic_probe_artifact(
        path,
        policy=policy,
        semantic_target=semantic_target,
        collection={
            "samples": 4096,
            "control_steps": 128,
            "num_envs": 64,
            "seed": 20261801,
        },
        context=context,
        torch=torch,
    )
    loaded = _load_semantic_probe_artifact(path, context=context, torch=torch)
    assert loaded["policy"].shape == (4096, 668)
    assert loaded["semantic_target"].shape == (4096, 31)

    for mutation in (
        lambda value: value["contracts"].update({"phase4_raw_sha256": "wrong"}),
        lambda value: value["policy"].resize_(4096, 667),
        lambda value: value["semantic_target"].fill_(0.0),
        lambda value: value["policy"].fill_(float("nan")),
        lambda value: value.update({"source_allowlist_sha256": "wrong"}),
    ):
        value = torch.load(path, map_location="cpu", weights_only=False)
        mutation(value)
        torch.save(value, path)
        with pytest.raises(ValueError):
            _load_semantic_probe_artifact(path, context=context, torch=torch)
        _save_semantic_probe_artifact(
            path,
            policy=policy,
            semantic_target=semantic_target,
            collection={
                "samples": 4096,
                "control_steps": 128,
                "num_envs": 64,
                "seed": 20261801,
            },
            context=context,
            torch=torch,
        )

    source = (ROOT / "scripts/train_phase5.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_learning"
    )
    function_source = ast.get_source_segment(source, function)
    assert function_source is not None
    assert "probe_env" not in function_source
    assert "_collect_fixed_probe" not in function_source
    assert (
        sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_build_environment"
            for node in ast.walk(function)
        )
        == 1
    )


def test_final_evaluation_progress_reports_phase_and_episode_counts(
    tmp_path: Path,
) -> None:
    args = type("Args", (), {"case": "final_evaluation", "case_dir": tmp_path})()
    _write_final_evaluation_progress(
        args,
        started=0.0,
        state="nominal_pair_batch_progress",
        phase="nominal_paired",
        episodes_completed=64,
        episodes_total=128,
        batch_index=1,
        batches_completed=0,
        batches_total=2,
    )
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress == {
        "batch_index": 1,
        "batches_completed": 0,
        "batches_total": 2,
        "case": "final_evaluation",
        "contact_bearing_transitions": 0,
        "elapsed_s": pytest.approx(progress["elapsed_s"]),
        "episodes_completed": 64,
        "episodes_total": 128,
        "iteration": 0,
        "phase": "nominal_paired",
        "state": "nominal_pair_batch_progress",
        "transitions": 0,
        "version": "phase5_progress_v1",
    }
    source = (ROOT / "scripts/train_phase5.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_final_evaluation"
    )
    function_source = ast.get_source_segment(source, function)
    assert function_source is not None
    assert function_source.count("_build_environment(") == 1
    assert function_source.count("env.close()") == 1
    assert "trained_env" not in function_source
    assert "scaffold_env" not in function_source
    for state in (
        "c1_evaluation_progress",
        "nominal_pair_batch_constructing",
        "scaffold_only_pair_batch_start",
    ):
        assert state in function_source
    assert function_source.index("_persist_final_metrics") < function_source.index(
        "gates = validate_final_evaluation"
    )
    assert 'resource_state["env_already_closed"] = True' in function_source

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    context = {
        "acceptance": type(
            "Acceptance",
            (),
            {"payload": {"thresholds": {"nominal_progress_median_min": 2.4871}}},
        )(),
        "contracts": {"phase4_canonical_sha256": "phase4"},
        "source_allowlist_sha256": "source-allowlist",
    }
    final_metrics = {
        "outcome": {"success": 0.8, "failure": 0.02, "episode_invalid": 0},
        "retention": {
            "median_progress": 2.5,
            "scaffold_only_median_progress": 2.6,
            "trained_to_scaffold_ratio": 0.96,
        },
        "action": {
            "contact_saturation_fraction": 0.04,
            "contact_saturation_window_max": 0.09,
        },
        "residual": {
            "contact_mean_delta_safe_norm": 0.02,
            "contact_norm_fraction_ge_min": 0.3,
        },
    }
    evidence = _persist_final_metrics(
        args,
        final_metrics=final_metrics,
        context=context,
        checkpoint=checkpoint,
    )
    stored = json.loads((tmp_path / "final_metrics.json").read_text(encoding="utf-8"))
    assert stored == evidence
    assert stored["retention"] == final_metrics["retention"]
    assert (
        stored["acceptance_thresholds"] == context["acceptance"].payload["thresholds"]
    )
    assert stored["checkpoint"]["sha256"] == _sha256(checkpoint)
    assert stored["contracts"] == context["contracts"]
    assert stored["source_allowlist_sha256"] == "source-allowlist"
    assert stored["git"]["diff_sha256"]
