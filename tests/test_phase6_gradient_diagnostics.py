from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from somaforce_cross.learning.actor_critic import ResidualActorCritic
from somaforce_cross.learning.semantic_ppo import SemanticPPO


ROOT = Path(__file__).parents[1]
PATH = ROOT / "scripts/diagnose_phase6_gradients.py"
SHELL = (
    ROOT.parents[0]
    / "training-job-scripts/somaforce_cross/phase6_joint_training"
    / "diagnose_segment47_authority_contact_drift_4gpu.sh"
)
SPEC = importlib.util.spec_from_file_location("phase6_gradient_diagnostics", PATH)
assert SPEC and SPEC.loader
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class FakeStorage:
    """CPU implementation of the exact storage surface used by the diagnostic."""

    def __init__(self, _kind, envs, steps, observations, _action_shape, device):
        self.observations = {
            name: value.clone() for name, value in observations.items()
        }
        self.envs, self.steps, self.device = envs, steps, device
        self.rows = []

    def add_transitions(self, transition):
        self.rows.append(transition)

    def compute_returns(self, *_args, **_kwargs):
        pass

    def mini_batch_generator(self, minibatches, epochs):
        assert len(self.rows) == 32 and minibatches == 8 and epochs == 3
        for _ in range(24):
            batch = self.envs
            policy = torch.zeros(batch, 668)
            policy[:, 12] = 1.0
            policy[:, 25] = 1.0
            target = torch.zeros(batch, 31)
            target[:, 12] = 1.0
            target[:, 25] = 1.0
            yield (
                {
                    "policy": policy,
                    "critic": torch.zeros(batch, 845),
                    "semantic_target": target,
                },
                torch.zeros(batch, 23),
                torch.zeros(batch, 1),
                torch.ones(batch, 1),
                torch.zeros(batch, 1),
                torch.zeros(batch, 1),
                None,
                None,
                (None, None),
                None,
            )

    def clear(self):
        pass


def _observations(batch: int = 2):
    policy = torch.zeros(batch, 668)
    policy[:, 12] = 1.0
    policy[:, 25] = 1.0
    target = torch.zeros(batch, 31)
    target[:, 12] = 1.0
    target[:, 25] = 1.0
    return {
        "policy": policy,
        "critic": torch.zeros(batch, 845),
        "semantic_target": target,
    }


def _outcome_rows():
    diagnostics = {
        "p50_wrench": 1.0,
        "p95_wrench": 1.0,
        "p99_wrench": 1.0,
        "impulse": 1.0,
        "p95_force_rate": 1.0,
        "contact_fraction": 1.0,
        "contact_loss": 0.0,
        "sensor_quality": 1.0,
        "stability_margin": 1.0,
        "arms_residual_norm": 1.0,
        "waist_residual_norm": 1.0,
        "legs_residual_norm": 1.0,
        "saturation": 0.0,
        "dropout": 0.0,
        "semantic_entropy": 1.0,
        "semantic_kl": 0.0,
    }
    reward = {
        "contact": 0.0,
        "force": 0.0,
        "nonfinite": 0.0,
        "progress": 0.0,
        "rate": 0.0,
        "residual": 0.0,
        "stability": 0.0,
        "terminal_failure": 0.0,
        "terminal_success": 0.0,
    }
    stage = [
        {
            "task": "move_suitcase",
            "mode": "residual",
            "stage": "C1",
            "subset": "stage",
            "family": "physical",
            "seed": index,
            "env_id": index % 64,
            "steps": 10,
            "return": 1.0,
            "success": True,
            "failure": False,
            "invalid": False,
            "timeout": False,
            "raw_reward_sums": reward,
            "weighted_reward_sums": reward,
            "diagnostics": diagnostics,
            "acceptance_diagnostics": {
                "residual_norm_sum": 10.0,
                "contact_residual_norm_sum": 5.0,
                "transition_count": 10,
                "residual_mean_norm": 1.0,
                "contact_bearing_residual_fraction": 0.5,
            },
        }
        for index in range(256)
    ]
    nominal = [
        {**row, "subset": "nominal", "seed": 1_000 + index}
        for index, row in enumerate(stage[:128])
    ]
    return [*stage, *nominal]


def test_module_has_no_top_level_isaac_or_rsl_import() -> None:
    imports = [
        node
        for node in ast.parse(PATH.read_text()).body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert "isaac" not in "\n".join(ast.unparse(node) for node in imports).lower()
    assert "rsl" not in "\n".join(ast.unparse(node) for node in imports).lower()


def test_parser_and_shell_argv_are_consistent() -> None:
    shell = SHELL.read_text()
    assert "--mode plan" in shell and "--mode rank-wrapper" in shell
    assert "--headless" not in shell
    assert "for PROBE in gradient residual scaffold_only; do" in shell
    assert '--probe "${PROBE}"' in shell
    assert "--authority-stage" in shell
    assert "for SEGMENT in 47 51 52 59 79; do" in shell
    assert "chmod" not in shell
    assert "REBOUND_ROOT" in shell
    assert "source_rebind_mainrunner.json" in shell
    assert "post_evaluation_mainrunner_rebound.pt" in shell
    wrapper = diagnostic._plain_parser("rank-wrapper")
    parsed = wrapper.parse_args(
        [
            "--mode",
            "rank-wrapper",
            "--output-dir",
            "x",
            "--checkpoint",
            "c",
            "--stage",
            "C2",
            "--authority-stage",
            "C1",
            "--probe",
            "gradient",
            "--timeout-s",
            "1",
        ]
    )
    assert "--headless" in diagnostic._worker_command(parsed)
    assert diagnostic._worker_command(parsed)[-2:] == ["--authority-stage", "C1"]
    worker_source = ast.get_source_segment(
        PATH.read_text(),
        next(
            node
            for node in ast.parse(PATH.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "_worker_main"
        ),
    )
    assert worker_source is not None
    assert worker_source.index(
        'args.device = f"cuda:{local_rank}"'
    ) < worker_source.index("launcher = AppLauncher(args)")
    assert worker_source.index("args.num_envs = 64") < worker_source.index(
        "launcher = AppLauncher(args)"
    )
    assert worker_source.index(
        "torch.cuda.set_device(local_rank)"
    ) < worker_source.index("launcher = AppLauncher(args)")
    assert worker_source.count("_build_environment(") == 1
    assert "_bind_gradient_authority(" in worker_source
    assert 'if environment_stage != "C2":' in PATH.read_text()
    assert (
        'rollout_steps = 472 if task.task == "move_suitcase" else 32' in worker_source
    )
    outcome_source = ast.get_source_segment(
        PATH.read_text(),
        next(
            node
            for node in ast.parse(PATH.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_outcome_mode"
        ),
    )
    assert (
        outcome_source is not None and outcome_source.count("_build_environment(") == 1
    )


def test_outcome_residual_uses_gradient_authority_binding() -> None:
    outcome_source = ast.get_source_segment(
        PATH.read_text(),
        next(
            node
            for node in ast.parse(PATH.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "_run_outcome_mode"
        ),
    )
    assert outcome_source is not None
    assert outcome_source.count("_bind_gradient_authority(") == 1
    assert "_bind_diagnostic_authority(" not in outcome_source


@pytest.mark.parametrize(
    ("segment", "expected_iteration"),
    ((47, 1465), (51, 1587), (52, 1618), (59, 1832), (79, 2442)),
)
def test_admitted_checkpoint_binding_uses_source_rebind_endpoint_iteration(
    segment: int, expected_iteration: int
) -> None:
    binding = diagnostic._diagnostic_checkpoint_binding(
        diagnostic._rebound_checkpoint(segment)
    )
    assert binding["segment"] == segment
    assert binding["endpoint_iteration"] == expected_iteration
    if segment == 51:
        assert binding["endpoint_iteration"] != 1465


def test_worker_probes_share_strict_checkpoint_binding_path() -> None:
    worker_source = ast.get_source_segment(
        PATH.read_text(),
        next(
            node
            for node in ast.parse(PATH.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "_worker_main"
        ),
    )
    assert worker_source is not None
    binding_index = worker_source.index(
        "checkpoint_binding = _diagnostic_checkpoint_binding(args.checkpoint)"
    )
    restore_index = worker_source.index("restored = restore_learning_checkpoint(")
    outcome_index = worker_source.index("if args.probe in OUTCOME_MODES:")
    gradient_index = worker_source.index('args.runtime_mode = "residual"')
    assert binding_index < restore_index < outcome_index < gradient_index
    assert worker_source.count("restore_learning_checkpoint(") == 1
    assert (
        'expected_iteration=checkpoint_binding["endpoint_iteration"]' in worker_source
    )


def test_production_restore_validator_rejects_segment51_iteration_1465(
    tmp_path: Path,
) -> None:
    from scripts.train_phase6 import _current_mainrunner_source_manifest
    from somaforce_cross.learning.joint_runner import (
        atomic_torch_save,
        load_learning_acceptance_config,
        load_phase6_config,
        load_phase6_task_roster,
        restore_learning_checkpoint,
    )

    def assert_semantically_equal(expected: object, actual: object) -> None:
        if isinstance(expected, torch.Tensor):
            assert isinstance(actual, torch.Tensor)
            assert torch.equal(expected, actual)
        elif isinstance(expected, dict):
            assert isinstance(actual, dict)
            assert set(expected) == set(actual)
            for key in expected:
                assert_semantically_equal(expected[key], actual[key])
        elif isinstance(expected, (list, tuple)):
            assert isinstance(actual, type(expected))
            assert len(expected) == len(actual)
            for expected_item, actual_item in zip(expected, actual, strict=True):
                assert_semantically_equal(expected_item, actual_item)
        else:
            assert expected == actual

    checkpoint = diagnostic._rebound_checkpoint(51)
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    binding = diagnostic._diagnostic_checkpoint_binding(checkpoint)
    v1_manifest = _current_mainrunner_source_manifest(
        ROOT / "configs/phase6_joint_training_v1.json",
        ROOT / "configs/phase6_task_roster_v1.json",
        ROOT / "configs/phase6_learning_acceptance_v1.json",
    )
    config = load_phase6_config(ROOT / "configs/phase6_joint_training_v1.json")
    acceptance = load_learning_acceptance_config(
        ROOT / "configs/phase6_learning_acceptance_v1.json"
    )
    roster = load_phase6_task_roster(
        ROOT / "configs/phase6_task_roster_v1.json", phase6_config=config
    )
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    with pytest.raises(
        ValueError, match="learning checkpoint source manifest mismatch"
    ):
        restore_learning_checkpoint(
            checkpoint,
            config=config,
            roster=roster,
            acceptance=acceptance,
            policy=policy,
            optimizer=optimizer,
            source_manifest=v1_manifest,
            expected_iteration=binding["endpoint_iteration"],
            device="cpu",
        )

    original_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    rebound_payload = copy.deepcopy(original_payload)
    rebound_payload["source_manifest"] = dict(v1_manifest)
    for field, original_value in original_payload.items():
        if field != "source_manifest":
            assert_semantically_equal(original_value, rebound_payload[field])
    changed_paths = {
        path
        for path in set(original_payload["source_manifest"]) | set(v1_manifest)
        if original_payload["source_manifest"].get(path) != v1_manifest.get(path)
    }
    assert changed_paths == {
        str(ROOT / "scripts/train_phase6.py"),
        str(ROOT / "somaforce_cross/learning/joint_runner.py"),
    }
    rebound_checkpoint = tmp_path / "segment_0051_v1_manifest_rebound.pt"
    atomic_torch_save(rebound_checkpoint, rebound_payload)

    restored_policy = ResidualActorCritic()
    restored_optimizer = torch.optim.Adam(restored_policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        rebound_checkpoint,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=restored_policy,
        optimizer=restored_optimizer,
        source_manifest=v1_manifest,
        expected_iteration=binding["endpoint_iteration"],
        device="cpu",
    )
    assert restored["iteration"] == 1587

    rejected_policy = ResidualActorCritic()
    rejected_optimizer = torch.optim.Adam(rejected_policy.parameters(), lr=3.0e-4)
    with pytest.raises(ValueError, match="learning resume iteration mismatch"):
        restore_learning_checkpoint(
            rebound_checkpoint,
            config=config,
            roster=roster,
            acceptance=acceptance,
            policy=rejected_policy,
            optimizer=rejected_optimizer,
            source_manifest=v1_manifest,
            expected_iteration=1465,
            device="cpu",
        )
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == checkpoint_sha256


def test_gradient_authority_binding_keeps_c1_native_and_overrides_only_c2() -> None:
    class FakeAuthority(torch.nn.Module):
        def forward(self, stage: int) -> torch.Tensor:
            return torch.tensor([float(stage)])

    class FakeEnvironment:
        def __init__(self) -> None:
            self.authority = FakeAuthority()

    for authority_stage in (None, "C1"):
        c1_environment = FakeEnvironment()
        c1_forward = c1_environment.authority.forward
        c1 = diagnostic._bind_gradient_authority(
            c1_environment,
            environment_stage="C1",
            authority_stage=authority_stage,
        )
        assert c1["authority_stage"] == "C1"
        assert c1["process_local_override"] is False
        assert c1_environment.authority.forward == c1_forward
        assert c1_environment.authority(1).tolist() == [1.0]
    with pytest.raises(ValueError, match="native C1 authority"):
        diagnostic._bind_gradient_authority(
            FakeEnvironment(), environment_stage="C1", authority_stage="C2"
        )

    c2_environment = FakeEnvironment()
    c2 = diagnostic._bind_gradient_authority(
        c2_environment, environment_stage="C2", authority_stage="C1"
    )
    assert c2["authority_stage"] == "C1"
    assert c2["process_local_override"] is True
    assert c2_environment.authority(2).tolist() == [1.0]


def test_real_actor_critic_parameter_groups_use_log_std() -> None:
    policy = ResidualActorCritic()
    groups = diagnostic._parameter_groups(policy)
    assert groups["log_std"] == (policy.log_std,)
    assert set(groups) == set(diagnostic.GROUPS)


def test_fake_storage_uses_production_32_step_and_24_minibatch_path() -> None:
    policy = ResidualActorCritic()
    algorithm = SemanticPPO(policy, storage_class=FakeStorage, device="cpu")
    observations = _observations()
    algorithm.init_storage(
        num_envs=2, num_transitions_per_env=32, observations=observations
    )
    for _ in range(32):
        algorithm.act(observations)
        algorithm.process_env_step(
            observations,
            torch.zeros(2),
            torch.zeros(2, dtype=torch.bool),
            {"time_outs": torch.zeros(2)},
        )
    algorithm.compute_returns(observations)
    records = diagnostic._gradient_probe(
        algorithm=algorithm,
        observations=observations,
        ppo={
            "num_mini_batches": 8,
            "num_learning_epochs": 3,
            "clip_param": 0.2,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.001,
        },
    )
    summary = diagnostic._summarize_gradient_records(records)
    assert len(records) == 24 and len(summary["records"]) == 24
    assert set(summary["losses"]) == set(diagnostic.OBJECTIVES)
    assert all(parameter.grad is None for parameter in policy.parameters())


def test_outcome_aggregation_rejects_zero_placeholder() -> None:
    with pytest.raises(ValueError, match="quota"):
        diagnostic._aggregate_outcomes(
            [], task="move_suitcase", mode="residual", stage_name="C1"
        )


def test_outcome_aggregation_rejects_missing_schema() -> None:
    rows = _outcome_rows()
    del rows[0]["diagnostics"]
    with pytest.raises(ValueError, match="schema"):
        diagnostic._aggregate_outcomes(
            rows, task="move_suitcase", mode="residual", stage_name="C1"
        )


def test_outcome_aggregation_requires_production_quotas() -> None:
    residual = _outcome_rows()
    scaffold = [
        {
            **row,
            "mode": "scaffold_only",
            "acceptance_diagnostics": {
                "residual_norm_sum": 0.0,
                "contact_residual_norm_sum": 0.0,
                "transition_count": row["steps"],
                "residual_mean_norm": 0.0,
                "contact_bearing_residual_fraction": 0.0,
            },
        }
        for row in residual
    ]
    summary = diagnostic._paired_outcome_summary(
        residual, scaffold, task="move_suitcase", stage="C1"
    )
    assert summary["retention"] == 1.0 and summary["success"] == 1.0
    with pytest.raises(ValueError, match="quota"):
        diagnostic._aggregate_outcomes(
            residual[:-1], task="move_suitcase", mode="residual", stage_name="C1"
        )
    assert "schedule_sha256" in PATH.read_text()
    assert diagnostic._authority_case(authority_stage="C1", probe="residual") == (
        "c2_env_c1_authority"
    )
    assert diagnostic._authority_case(authority_stage=None, probe="scaffold_only") == (
        "scaffold_only"
    )
    with pytest.raises(ValueError, match="C1 or C2"):
        diagnostic._authority_case(authority_stage=None, probe="gradient")


def test_runtime_rng_capture_is_state_sensitive() -> None:
    first = diagnostic._hash_object(torch.get_rng_state())
    _ = torch.rand(1)
    second = diagnostic._hash_object(torch.get_rng_state())
    source = PATH.read_text()
    assert first != second
    assert "runtime_rng_start = _hash_object" in source
    assert "runtime_rng_end = _hash_object" in source
    assert source.count("torch.cuda.get_rng_state(local_rank)") == 3


def test_wrapper_timeout_handles_silent_child(tmp_path: Path, monkeypatch) -> None:
    script = tmp_path / "silent.py"
    script.write_text("import time; time.sleep(10)\n")
    monkeypatch.setattr(
        diagnostic, "_worker_command", lambda _args: [sys.executable, str(script)]
    )
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")
    args = diagnostic._plain_parser("rank-wrapper").parse_args(
        [
            "--mode",
            "rank-wrapper",
            "--output-dir",
            str(tmp_path / "attempt"),
            "--checkpoint",
            "x",
            "--stage",
            "C1",
            "--probe",
            "gradient",
            "--timeout-s",
            "1",
        ]
    )
    with pytest.raises(RuntimeError, match="wrapper evidence"):
        diagnostic._run_rank_wrapper(args)
    assert (
        diagnostic._read_json(tmp_path / "attempt/C1/gradient/rank_0/wrapper.json")[
            "timed_out"
        ]
        is True
    )


def test_plan_is_written_before_worker_launch(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(diagnostic, "CHECKPOINT_SHA256", diagnostic._sha256(checkpoint))
    monkeypatch.setattr(
        diagnostic,
        "_diagnostic_checkpoint_binding",
        lambda path: {
            "diagnostic_only": True,
            "rebound_path": str(path),
            "rebound_sha256": diagnostic._sha256(path),
            "segment": 47,
        },
    )
    args = diagnostic._plain_parser("plan").parse_args(
        [
            "--mode",
            "plan",
            "--output-dir",
            str(tmp_path / "attempt"),
            "--checkpoint",
            str(checkpoint),
            "--timeout-s",
            "1",
        ]
    )
    diagnostic._run_plan(args)
    assert (tmp_path / "attempt/plan.json").is_file() and diagnostic._read_json(
        tmp_path / "attempt/progress.json"
    )["status"] == "planned"
    plan = diagnostic._read_json(tmp_path / "attempt/plan.json")
    assert plan["environment_stage"] == "C2"
    assert plan["authority_cases"] == list(diagnostic.AUTHORITY_CASES)
    assert plan["suitcase_event_trace"]["horizon_steps"] == 472
    assert plan["c1_nominal_checkpoint_drift"]["segments"] == [47, 51, 52, 59, 79]
    assert plan["checkpoint"]["diagnostic_only"] is True
    assert "rebound_checkpoints" in plan["c1_nominal_checkpoint_drift"]


def test_plan_refuses_to_overwrite_failed_attempt_and_stage_merge_rejects_bad_wrappers(
    tmp_path: Path,
) -> None:
    args = diagnostic._plain_parser("plan").parse_args(
        [
            "--mode",
            "plan",
            "--output-dir",
            str(tmp_path),
            "--checkpoint",
            "x",
            "--timeout-s",
            "1",
        ]
    )
    with pytest.raises(FileExistsError, match="refuses"):
        diagnostic._run_plan(args)
    stage = tmp_path / "stage"
    with pytest.raises(ValueError, match="missing or duplicated"):
        diagnostic._stage_probe_records(stage)
    for probe in diagnostic.PROBES:
        for rank, task in enumerate(diagnostic.TASKS):
            rank_dir = stage / probe / f"rank_{rank}"
            rank_dir.mkdir(parents=True)
            diagnostic._atomic_json(rank_dir / "command.json", {"probe": probe})
            diagnostic._atomic_json(rank_dir / "progress.json", {"status": "complete"})
            diagnostic._atomic_json(
                rank_dir / "result.json",
                {"status": "ok", "probe": probe, "rank": rank, "task": task},
            )
            diagnostic._atomic_json(
                rank_dir / "wrapper.json",
                {
                    "passed": True,
                    "exit_code": 0,
                    "signal": None,
                    "timed_out": False,
                    "timeout_kind": None,
                    "marker_order_valid": True,
                },
            )
            (rank_dir / "worker.log").write_text("ok\n")
    assert set(diagnostic._stage_probe_records(stage)) == set(diagnostic.PROBES)
    wrapper = stage / "residual/rank_1/wrapper.json"
    payload = diagnostic._read_json(wrapper)
    payload["timed_out"] = True
    diagnostic._atomic_json(wrapper, payload)
    with pytest.raises(ValueError, match="clean completion"):
        diagnostic._stage_probe_records(stage)
    payload["timed_out"] = False
    payload["marker_order_valid"] = False
    diagnostic._atomic_json(wrapper, payload)
    with pytest.raises(ValueError, match="clean completion"):
        diagnostic._stage_probe_records(stage)
    extra = stage / "gradient/duplicate_rank_0"
    extra.mkdir()
    with pytest.raises(ValueError, match="missing or duplicated"):
        diagnostic._stage_probe_records(stage)


def test_support_requires_both_material_classes_and_invariants() -> None:
    stage = {
        "invariants_ok": True,
        "tasks": {
            task: {"reward": 1.0, "success": 1.0, "retention": 1.0, "failure": 0.0}
            for task in diagnostic.TASKS
        },
        "gradient": {
            "combined_norm": {"mean": 1.0},
            "ppo_semantic_cosine": {"mean": 1.0},
            "preclip_fraction": 0.0,
        },
    }
    c2 = {
        **stage,
        "tasks": {
            **stage["tasks"],
            "move_suitcase": {
                "reward": 0.8,
                "success": 0.85,
                "retention": 1.0,
                "failure": 0.0,
            },
        },
        "gradient": {
            "combined_norm": {"mean": 2.0},
            "ppo_semantic_cosine": {"mean": 1.0},
            "preclip_fraction": 0.0,
        },
    }
    assert diagnostic._support_decision(stage, c2)["status"] == "supported"
    source = PATH.read_text()
    assert "suitcase_event_time_series.json" in source
    assert "_window_gradient_diagnostic(" in source
    assert "strict restore of five diagnostic-only rebound endpoints" in source
    assert "diagnostic refuses a historical or unbound checkpoint input" in source
    assert "source_manifest_only_change" in source
    assert "payload_except_source_manifest" in source
