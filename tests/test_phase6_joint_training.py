from __future__ import annotations

import ast
import copy
import inspect
import json
import hashlib
import os
import subprocess
import sys
import time
import traceback
from argparse import Namespace
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing

import scripts.train_phase6 as train_phase6
from scripts.train_phase6 import (
    HARNESS_REBIND_FUNCTIONS,
    PAIREDBATCH_ENV_SNAPSHOT_SHA256,
    PAIREDBATCH_INPUT_SHA256,
    PAIREDBATCH_JOINT_SNAPSHOT_SHA256,
    PAIREDBATCH_TRAIN_SNAPSHOT_SHA256,
    PHASE6_CONFIG,
    PHASE6_ROSTER,
    WORKER_CRITICAL_FUNCTIONS,
    _EvaluationProgress,
    _ast_function_dumps,
    _aggregate_evaluation_progress,
    _aggregate_modeprocess_progress,
    _command_argument,
    _current_mainrunner_source_manifest,
    _evaluation_devicefix_ast_proof,
    _evaluation_devicefix_invariants,
    _episodeorder_ast_proof_for_sources,
    _mainrunner_ast_source_proof,
    _merge_modeprocess_evaluation,
    _modeprocess_ast_proof,
    _module_ast_sha256,
    _pairedbatch_ast_proof_for_sources,
    _pairedbatch_source_manifest_for_sources,
    _validate_rankseed_failed_smoke_manifest,
    _validate_episodeorder_failed_smoke_manifest,
    _validate_modeprocess_failed_smoke_manifest,
    _validate_warningpolicy_failed_smoke_manifest,
    _validate_pairedbatch_frozen_attempt,
    _validate_legacy_central_markers,
    _validate_production_command,
    _parent_close_verified,
    _profile,
    _run_devicefix_recovery_evaluate,
    _run_modeprocess_recovery_evaluate,
    _run_paired_evaluation_rank,
    _run_production,
    _run_stepboundary_recovery_evaluate,
    _semantic_equal,
    _summaryschema_ast_proof,
    _stepboundary_ast_proof,
    _write_mainrunner_root_latest,
    _validate_historical_warningpolicy_rebind_record,
    _validate_summaryschema_failed_finalize_manifest,
    _warningpolicy_ast_proof,
    summarize_phase6_run,
)
from somaforce_cross.learning.actor_critic import ResidualActorCritic
from somaforce_cross.learning.joint_runner import (
    PHASE6_LEARNING_CONFIG,
    JointCurriculum,
    JointTaskScheduler,
    Phase6Config,
    TransitionMetricCollector,
    assert_matching_hash_records,
    assert_transition_balance,
    atomic_torch_save,
    canonical_sha256,
    crossing_target_iteration,
    joint_checkpoint_payload,
    learning_checkpoint_payload,
    load_learning_acceptance_config,
    load_phase5_policy_only,
    load_phase6_config,
    load_phase6_task_roster,
    optimizer_step,
    paired_evaluation_plan,
    paired_evaluation_rank_seed,
    paired_evaluation_schedule,
    plan_learning_segments,
    pooled_macro_metrics,
    restore_learning_checkpoint,
    restore_joint_checkpoint,
    state_dict_sha256,
    transition_accounting,
    validate_learning_acceptance_config,
    validate_learning_evaluation_summary,
    validate_learning_final_acceptance,
    validate_paired_evaluation_plan,
    validate_paired_evaluation_schedule,
    validate_phase6_joint_metrics,
    validate_phase6_config,
    validate_rank_rng_states,
)
from somaforce_cross.learning.semantic_ppo import SemanticPPO


CONFIG_PATH = Path("configs/phase6_joint_training_v1.json")
ROSTER_PATH = Path("configs/phase6_task_roster_v1.json")
PHASE5_CHECKPOINT = Path(
    "outputs/phase5_learning_acceptance/phase5_learning_v3_20260805/"
    "learning_64env_2442iter/checkpoints/"
    "iteration_0002442_transitions_5001216.pt"
)
CONFIG_RAW_SHA256 = "f26f8e118decb13066362b7a8612b1e2c5fbba37050a38b7ab3880f796c2131d"
CONFIG_CANONICAL_SHA256 = (
    "8e733479baada2334636106ee352aeb8ab8bf8dc7ad4ce3b39b844019c74df42"
)
ROSTER_RAW_SHA256 = "8ddcdd05f423138eb855d9c7cb3478ffb373f718484f0a948512f9bce1492fdb"
ROSTER_CANONICAL_SHA256 = (
    "14dfe15db96daa6c8d0bac93965fcdca3d9e815a1ac69a7adaa44d9f0db45bc4"
)
V2_CONFIG_PATH = Path("configs/phase6_joint_training_v2.json")
V2_ROSTER_PATH = Path("configs/phase6_task_roster_v2.json")
V2_ACCEPTANCE_PATH = Path("configs/phase6_learning_acceptance_v2.json")
V2_SCRIPT_ROOT = Path(
    "/inspire/hdd/global_user/liumengfan-253108110079/lmf-workspace/"
    "training-job-scripts/somaforce_cross/phase6_joint_training"
)


def _v2() -> tuple[Phase6Config, object, object]:
    config = load_phase6_config(V2_CONFIG_PATH)
    roster = load_phase6_task_roster(V2_ROSTER_PATH, phase6_config=config)
    acceptance = load_learning_acceptance_config(V2_ACCEPTANCE_PATH)
    return config, roster, acceptance


def _write_gloo_v2_stage(
    evidence_dir: Path,
    *,
    rank: int,
    stage: str,
    error: BaseException | None = None,
) -> None:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    path = evidence_dir / f"rank_{rank}.json"
    payload = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {
            "rank": rank,
            "stages": [],
        }
    )
    payload["stages"].append(stage)
    if error is not None:
        payload["error"] = {
            "message": str(error),
            "type": type(error).__name__,
            "traceback": traceback.format_exc(),
        }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _gloo_v2_worker(
    rank: int,
    world_size: int,
    init_path: str,
    output_dir: str,
    evidence_dir: str,
    mode: str,
) -> None:
    from somaforce_cross.learning.joint_runner import (
        broadcast_independent_policy,
        initialize_independent_policy,
        independent_optimizer,
        state_dict_sha256,
    )

    output = Path(output_dir)
    evidence = Path(evidence_dir)
    try:
        distributed.init_process_group(
            backend="gloo",
            init_method=f"file://{init_path}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=60),
        )
        _write_gloo_v2_stage(evidence, rank=rank, stage="process_group_ready")
        config = load_phase6_config(V2_CONFIG_PATH)
        policy, record = initialize_independent_policy(config)
        if rank == 1:
            next(policy.parameters()).data.add_(1.0)
        policy_hash = broadcast_independent_policy(policy, rank=rank, device="cpu")
        _write_gloo_v2_stage(evidence, rank=rank, stage="broadcast_complete")
        optimizer, record = independent_optimizer(
            policy, learning_rate=3.0e-4, initialization=record
        )
        _write_gloo_v2_stage(evidence, rank=rank, stage="optimizer_initialized")
        row: dict[str, object] = {
            "initialization": record,
            "policy": policy_hash,
            "optimizer": state_dict_sha256(optimizer.state_dict()),
        }
        if mode == "update":
            from somaforce_cross.learning.joint_runner import GradientAverager

            for parameter in policy.parameters():
                parameter.grad = torch.full_like(parameter, float(rank + 1))
            GradientAverager(world_size=world_size, allow_test_backend=True)(policy)
            _write_gloo_v2_stage(evidence, rank=rank, stage="gradient_sync_complete")
            optimizer.step()
            _write_gloo_v2_stage(evidence, rank=rank, stage="optimizer_step_complete")
            row["updated_policy"] = state_dict_sha256(policy.state_dict())
            row["updated_optimizer"] = state_dict_sha256(optimizer.state_dict())
        elif mode != "broadcast":
            raise ValueError(f"unknown Gloo V2 mode: {mode}")
        torch.save(row, output / f"v2_rank_{rank}.pt")
        _write_gloo_v2_stage(evidence, rank=rank, stage="hashes_written")
    except BaseException as exc:
        _write_gloo_v2_stage(evidence, rank=rank, stage="error", error=exc)
        raise
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()


def _gloo_v2_stages(evidence_dir: Path) -> dict[int, object]:
    result: dict[int, object] = {}
    for rank in range(4):
        path = evidence_dir / f"rank_{rank}.json"
        result[rank] = (
            json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        )
    return result


def _run_gloo_v2_workers(
    tmp_path: Path, *, mode: str
) -> tuple[list[dict[str, object]], dict[int, object]]:
    root = tmp_path / f"v2_{mode}"
    output_dir = root / "outputs"
    evidence_dir = root / "stages"
    output_dir.mkdir(parents=True)
    context = multiprocessing.spawn(
        _gloo_v2_worker,
        args=(4, str(root / "filestore"), str(output_dir), str(evidence_dir), mode),
        nprocs=4,
        join=False,
    )
    deadline = time.monotonic() + 120.0
    timed_out = False
    failure: BaseException | None = None
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                timed_out = True
                break
            try:
                if context.join(timeout=min(1.0, remaining)):
                    break
            except BaseException as exc:
                failure = exc
                break
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=10.0)
    stages = _gloo_v2_stages(evidence_dir)
    unreaped = [process.pid for process in context.processes if process.is_alive()]
    if timed_out:
        raise TimeoutError(
            f"V2 Gloo worker timeout; stages={stages}; unreaped={unreaped}"
        )
    if failure is not None:
        raise RuntimeError(
            f"V2 Gloo worker failed; stages={stages}; unreaped={unreaped}"
        ) from failure
    if unreaped or any(process.exitcode != 0 for process in context.processes):
        raise RuntimeError(
            f"V2 Gloo worker exit failure; stages={stages}; unreaped={unreaped}"
        )
    rows = [
        torch.load(output_dir / f"v2_rank_{rank}.pt", weights_only=False)
        for rank in range(4)
    ]
    return rows, stages


def test_phase6_v2_schema_rejects_phase5_fields() -> None:
    payload = json.loads(V2_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["bindings"]["phase5_initial_checkpoint"] = {"path": "x", "sha256": "0" * 64}
    with pytest.raises(ValueError):
        validate_phase6_config(payload)


def test_phase6_v2_v1_contracts_remain_byte_identical_and_load() -> None:
    assert hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest() == CONFIG_RAW_SHA256
    assert hashlib.sha256(ROSTER_PATH.read_bytes()).hexdigest() == ROSTER_RAW_SHA256
    assert load_phase6_config(CONFIG_PATH).canonical_sha256 == CONFIG_CANONICAL_SHA256


def test_phase6_v2_seed_reproduces_initial_policy_sha() -> None:
    from somaforce_cross.learning.joint_runner import initialize_independent_policy

    config, _, acceptance = _v2()
    policy, _ = initialize_independent_policy(config)
    assert (
        state_dict_sha256(policy.state_dict())
        == acceptance.payload["bindings"]["initial_policy_sha256"]
    )


def test_phase6_v2_different_seed_changes_policy_sha() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260807)
        policy = ResidualActorCritic()
        policy.actor.mlp[4].weight.data.zero_()
        policy.actor.mlp[4].bias.data.zero_()
    assert (
        state_dict_sha256(policy.state_dict())
        != "cc3116f729f05b536084a50323a613c6dbdb73149424d314864196e402bad01d"
    )


def test_phase6_v2_initial_actor_mean_is_exactly_zero() -> None:
    from somaforce_cross.learning.joint_runner import initialize_independent_policy

    policy, _ = initialize_independent_policy(_v2()[0])
    observation = {"policy": torch.zeros(2, 668), "critic": torch.zeros(2, 845)}
    assert torch.equal(policy.act_inference(observation), torch.zeros(2, 23))


def test_phase6_v2_non_actor_parameters_are_finite_and_nonzero() -> None:
    from somaforce_cross.learning.joint_runner import initialize_independent_policy

    policy, _ = initialize_independent_policy(_v2()[0])
    values = torch.cat(
        [
            parameter.detach().flatten()
            for name, parameter in policy.named_parameters()
            if not name.startswith("actor.mlp.4")
        ]
    )
    assert torch.isfinite(values).all() and torch.count_nonzero(values) > 0


def test_phase6_v2_optimizer_is_empty_at_initialization() -> None:
    from somaforce_cross.learning.joint_runner import (
        independent_optimizer,
        initialize_independent_policy,
    )

    policy, record = initialize_independent_policy(_v2()[0])
    optimizer, record = independent_optimizer(
        policy, learning_rate=3.0e-4, initialization=record
    )
    assert (
        optimizer.state == {}
        and optimizer_step(optimizer) == 0
        and record["optimizer_initial_step"] == 0
    )


def test_phase6_v2_four_process_gloo_broadcast_and_optimizer_hashes(
    tmp_path: Path,
) -> None:
    rows, stages = _run_gloo_v2_workers(tmp_path, mode="broadcast")
    assert (
        len({row["policy"] for row in rows}) == 1
        and len({row["optimizer"] for row in rows}) == 1
    )
    assert all(
        value
        == {
            "rank": rank,
            "stages": [
                "process_group_ready",
                "broadcast_complete",
                "optimizer_initialized",
                "hashes_written",
            ],
        }
        for rank, value in stages.items()
    )


def test_phase6_v2_divergent_rank_is_rejected_before_rollout() -> None:
    with pytest.raises(RuntimeError):
        assert_matching_hash_records(
            [
                {"policy": "a" * 64, "optimizer": "b" * 64, "step": 0},
                {"policy": "c" * 64, "optimizer": "b" * 64, "step": 0},
            ]
        )


def test_phase6_v2_cpu_synchronized_update_keeps_hashes_equal(tmp_path: Path) -> None:
    rows, stages = _run_gloo_v2_workers(tmp_path, mode="update")
    assert len({row["updated_policy"] for row in rows}) == 1
    assert len({row["updated_optimizer"] for row in rows}) == 1
    assert all(
        value is not None
        and value["stages"][-3:]
        == ["gradient_sync_complete", "optimizer_step_complete", "hashes_written"]
        for value in stages.values()
    )


def test_phase6_v2_rejects_v1_and_diagnostic_resume_before_environment(
    tmp_path: Path,
) -> None:
    config, roster, acceptance = _v2()
    checkpoint = tmp_path / "v1.pt"
    torch.save({"checkpoint_version": "phase6_learning_checkpoint_v1"}, checkpoint)
    with pytest.raises(ValueError):
        restore_learning_checkpoint(
            checkpoint,
            config=config,
            roster=roster,
            acceptance=acceptance,
            policy=ResidualActorCritic(),
            optimizer=torch.optim.Adam(ResidualActorCritic().parameters()),
            source_manifest={"x": "a" * 64},
        )


def test_phase6_v2_checkpoint_round_trips_initialization_provenance(
    tmp_path: Path,
) -> None:
    from somaforce_cross.learning.joint_runner import (
        independent_optimizer,
        initialize_independent_policy,
    )

    config, roster, acceptance = _v2()
    policy, initialization = initialize_independent_policy(config)
    optimizer, initialization = independent_optimizer(
        policy, learning_rate=3.0e-4, initialization=initialization
    )
    optimizer.state[next(policy.parameters())]["step"] = torch.tensor(24.0)
    payload = learning_checkpoint_payload(
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        iteration=1,
        task_transitions={task.task: 2048 for task in roster.tasks},
        curriculum=JointCurriculum(transitions=8192),
        next_crossing=None,
        evaluation_history=[],
        rank_rng={
            str(rank): {
                "cpu": torch.arange(2, dtype=torch.uint8),
                "cuda": [torch.arange(2, dtype=torch.uint8) for _ in range(4)],
            }
            for rank in range(4)
        },
        source_manifest={"x": "a" * 64},
        metrics={},
        pre_evaluation=True,
        initialization=initialization,
    )
    path = tmp_path / "v2.pt"
    atomic_torch_save(path, payload)
    restored = restore_learning_checkpoint(
        path,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=ResidualActorCritic(),
        optimizer=torch.optim.Adam(ResidualActorCritic().parameters()),
        source_manifest={"x": "a" * 64},
        expected_iteration=1,
    )
    assert restored["initialization"] == initialization


def _write_v2_paired_checkpoint(
    tmp_path: Path,
) -> tuple[Path, object, object, object, dict[str, str]]:
    from somaforce_cross.learning.joint_runner import (
        independent_optimizer,
        initialize_independent_policy,
    )

    config, roster, acceptance = _v2()
    policy, initialization = initialize_independent_policy(config)
    optimizer, initialization = independent_optimizer(
        policy, learning_rate=3.0e-4, initialization=initialization
    )
    optimizer.state[next(policy.parameters())]["step"] = torch.tensor(24.0)
    source_manifest = _current_mainrunner_source_manifest()
    payload = learning_checkpoint_payload(
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        iteration=1,
        task_transitions={task.task: 2048 for task in roster.tasks},
        curriculum=JointCurriculum(transitions=8192),
        next_crossing=None,
        evaluation_history=[],
        rank_rng={
            str(rank): {
                "cpu": torch.arange(2, dtype=torch.uint8),
                "cuda": [torch.arange(2, dtype=torch.uint8) for _ in range(4)],
            }
            for rank in range(4)
        },
        source_manifest=source_manifest,
        metrics={},
        pre_evaluation=True,
        initialization=initialization,
    )
    path = tmp_path / "v2_pre_evaluation.pt"
    atomic_torch_save(path, payload)
    return path, config, roster, acceptance, source_manifest


def test_v2_paired_evaluation_strict_restore_enters_schedule_without_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, config, roster, acceptance, source_manifest = (
        _write_v2_paired_checkpoint(tmp_path)
    )

    class _Environment:
        def __init__(self) -> None:
            self.mode = ""
            self.schedule: dict[str, object] | None = None
            self.done = False

        def evaluation_nominal_probe(self, _env_id: int, _episode_index: int) -> bool:
            return True

        def bind_evaluation_schedule(
            self, schedule: dict[str, object], *, mode: str
        ) -> None:
            self.schedule = schedule
            self.mode = mode

        def reset(self) -> tuple[dict[str, torch.Tensor], None]:
            return {"policy": torch.zeros(1, 668)}, None

        def step(
            self, _actions: torch.Tensor
        ) -> tuple[dict[str, torch.Tensor], None, None, None, None]:
            self.done = True
            assert self.schedule is not None
            self.records = tuple(
                {
                    "mode": self.mode,
                    "seed": slot["seed"],
                    "subset": slot["subset"],
                }
                for slot in self.schedule["rows"]["0"]
            )
            return {"policy": torch.zeros(1, 668)}, None, None, None, None

        def drain_evaluation_completions(self) -> tuple[dict[str, object], ...]:
            if not self.done:
                return ()
            self.done = False
            return self.records

    monkeypatch.setattr(
        "somaforce_cross.learning.joint_runner.paired_evaluation_plan",
        lambda _roster, **_kwargs: {
            "tasks": {
                roster.tasks[0].task: {
                    "horizon": 1,
                    "nominal_pairs": (),
                    "stage_pairs": (object(),),
                }
            }
        },
    )
    monkeypatch.setattr(
        "scripts.train_phase6._build_environment",
        lambda *_args, **_kwargs: _Environment(),
    )
    result, environment = _run_paired_evaluation_rank(
        Namespace(
            device="cpu",
            evaluation_seed=20262806,
            paired_mode="residual",
            paired_nominal_quota=1,
            paired_num_envs=1,
            paired_stage_quota=1,
            resume=checkpoint,
            runtime_mode="residual",
            stage="C1",
        ),
        config=config,
        roster=roster,
        rank=0,
        world_size=4,
        acceptance=acceptance,
        source_manifest=source_manifest,
    )
    environment.close() if hasattr(environment, "close") else None
    assert result["status"] == "ok"
    assert result["episodes"] == 2
    assert result["optimizer_steps"] == result["normalizer_updates"] == 0


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda payload: payload["source_manifest"].__setitem__(
                "scripts/train_phase6.py", "0" * 64
            ),
            "protected state",
        ),
        (
            lambda payload: payload["initialization"].__setitem__("source", "external"),
            "initialization",
        ),
        (lambda payload: payload.__setitem__("iteration", 2), "arithmetic"),
        (
            lambda payload: payload.__setitem__("pre_evaluation", False),
            "pre-evaluation",
        ),
    ],
)
def test_v2_paired_evaluation_rejects_protected_checkpoint_mutations(
    tmp_path: Path, mutation: object, match: str
) -> None:
    checkpoint, config, roster, acceptance, source_manifest = (
        _write_v2_paired_checkpoint(tmp_path)
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mutation(payload)
    mutated = tmp_path / f"mutated_{match}.pt"
    atomic_torch_save(mutated, payload)
    with pytest.raises(ValueError, match=match):
        _run_paired_evaluation_rank(
            Namespace(device="cpu", resume=mutated, stage="C1", paired_mode="residual"),
            config=config,
            roster=roster,
            rank=0,
            world_size=4,
            acceptance=acceptance,
            source_manifest=source_manifest,
        )


def test_v2_paired_evaluation_rejects_v1_checkpoint_before_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, config, roster, acceptance, source_manifest = _write_v2_paired_checkpoint(
        tmp_path
    )
    checkpoint = tmp_path / "v1.pt"
    torch.save({"checkpoint_version": "phase6_learning_checkpoint_v1"}, checkpoint)
    monkeypatch.setattr(
        "scripts.train_phase6._build_environment",
        lambda *_args, **_kwargs: pytest.fail("environment must not be built"),
    )
    with pytest.raises(ValueError, match="schema|non-v2 checkpoint"):
        _run_paired_evaluation_rank(
            Namespace(
                device="cpu", resume=checkpoint, stage="C1", paired_mode="residual"
            ),
            config=config,
            roster=roster,
            rank=0,
            world_size=4,
            acceptance=acceptance,
            source_manifest=source_manifest,
        )


def test_v2_paired_checkpoint_validation_is_cpu_static_and_has_no_update_path() -> None:
    source = inspect.getsource(_run_paired_evaluation_rank)
    assert "AppLauncher" not in source
    assert "torch.load" not in source
    assert "optimizer.step(" not in source
    assert ".backward(" not in source
    assert "policy.eval()" in source


def test_worker_evaluation_launches_isaac_before_environment_and_keeps_lifecycles_separate() -> (
    None
):
    source = inspect.getsource(train_phase6._worker_main)
    assert source.count("AppLauncher(args)") == 2
    evaluate_start = source.index('if args.mode == "evaluate":')
    train_start = source.index("scheduler = JointTaskScheduler(", evaluate_start)
    cleanup_start = source.index("except _EvaluationComplete:")
    evaluate_branch = source[evaluate_start:train_start]
    train_branch = source[train_start:cleanup_start]
    assert evaluate_branch.count("AppLauncher(args)") == 1
    assert train_branch.count("AppLauncher(args)") == 1
    assert evaluate_branch.index("AppLauncher(args)") < evaluate_branch.index(
        "_run_paired_evaluation_rank("
    )
    assert train_branch.index("AppLauncher(args)") < train_branch.index(
        "_build_environment("
    )
    assert source.index("torch.cuda.set_device(local_rank)") < source.index(
        "AppLauncher(args)", evaluate_start
    )
    assert source.index("args.num_envs = num_envs") < source.index(
        "AppLauncher(args)", evaluate_start
    )
    assert source.index('args.device = f"cuda:{local_rank}"') < source.index(
        "AppLauncher(args)", evaluate_start
    )
    assert "finally:" in source and "launcher.app.close()" in source


def test_phase6_v2_initial_branch_has_no_phase5_loader_or_external_torch_load() -> None:
    source = inspect.getsource(train_phase6._worker_main)
    branch = source[
        source.index(
            'if config.payload["contract_version"] == "phase6_joint_training_v2"'
        ) : source.index(
            "else:\n            algorithm",
            source.index(
                'if config.payload["contract_version"] == "phase6_joint_training_v2"'
            ),
        )
    ]
    assert "load_phase5_policy_only" not in branch and "torch.load" not in branch


def test_phase6_v2_task_balance_and_actor_privilege_boundary_are_unchanged() -> None:
    scheduler = JointTaskScheduler(task_count=4, world_size=4)
    assert scheduler.expected_task_transitions(
        cycle_index=0, window_index=0, slot_transitions=2048
    ) == {0: 2048, 1: 2048, 2: 2048, 3: 2048}
    assert "task_id" not in inspect.getsource(ResidualActorCritic.actor_forward)


def test_phase6_v2_init_probe_entrypoint_is_native_and_isaac_free() -> None:
    source = inspect.getsource(train_phase6._run_v2_init_probe)
    parser = train_phase6._init_probe_parser().parse_args(
        [
            "--mode",
            "init-probe",
            "--output-dir",
            "outputs/probe",
            "--timeout-s",
            "900",
        ]
    )
    assert parser.mode == "init-probe"
    assert "torch.distributed.init_process_group" in source
    assert 'backend="nccl"' in source
    assert "broadcast_independent_policy" in source
    assert "independent_optimizer" in source
    assert "PROCESS_GROUP_CLOSED" in source
    assert "AppLauncher" not in source


def test_phase6_v2_smoke_shells_use_torchrun_then_summarize() -> None:
    for name, profile in (
        ("smoke_independent_4gpu_2env_1iter.sh", "smoke_4gpu_2env_1iter"),
        ("smoke_independent_4gpu_64env_2iter.sh", "smoke_4gpu_64env_2iter"),
    ):
        source = (V2_SCRIPT_ROOT / name).read_text(encoding="utf-8")
        assert (
            "-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4"
            in source
        )
        assert "--mode rank-wrapper" in source
        assert f"--profile {profile}" in source
        assert "--stage C1" in source
        assert "--mode summarize" in source
        assert "suite_summary.json and rank_*/wrapper.json" in source

    parser = train_phase6._rank_wrapper_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--mode",
                "rank-wrapper",
                "--output-dir",
                "outputs/v2_smoke",
                "--profile",
                "smoke_4gpu_2env_1iter",
                "--timeout-s",
                "900",
            ]
        )
    smoke_args = parser.parse_args(
        [
            "--mode",
            "rank-wrapper",
            "--output-dir",
            "outputs/v2_smoke",
            "--profile",
            "smoke_4gpu_2env_1iter",
            "--timeout-s",
            "900",
            "--stage",
            "C1",
        ]
    )
    smoke_command = train_phase6._worker_command(smoke_args)
    assert smoke_command.count("--stage") == 1
    assert train_phase6._command_argument(smoke_command, "--stage") == "C1"
    assert train_phase6._command_argument(smoke_command, "--mode") == "worker"

    production_args = parser.parse_args(
        [
            "--mode",
            "rank-wrapper",
            "--worker-mode",
            "train-segment",
            "--output-dir",
            "outputs/v2_production",
            "--profile",
            "production_segment",
            "--timeout-s",
            "900",
            "--segment-end",
            "1",
            "--stage",
            "C1",
        ]
    )
    production_command = train_phase6._worker_command(production_args)
    assert production_command.count("--stage") == 1
    assert train_phase6._command_argument(production_command, "--stage") == "C1"


def test_phase6_v2_pilot_shell_starts_fresh_c1_lineage() -> None:
    source = (V2_SCRIPT_ROOT / "pilot_independent_4gpu_64env_31iter.sh").read_text(
        encoding="utf-8"
    )
    assert '[[ ! -e "${RUN_DIR}" ]]' in source
    assert "--mode production" in source
    assert "--profile pilot --max-segments 1" in source
    assert "--resume" not in source
    assert "--source-rebind" not in source
    assert "segment_0000/post_evaluation.pt latest.json progress.json" in source


def test_phase6_v2_main_shell_reuses_admitted_pilot_run_dir() -> None:
    source = (V2_SCRIPT_ROOT / "train_independent_4gpu_64env_2442iter.sh").read_text(
        encoding="utf-8"
    )
    assert (
        'RUN_DIR="${ADMITTED_V2_PILOT_RUN_DIR:?set ADMITTED_V2_PILOT_RUN_DIR}"'
        in source
    )
    assert 'test -f "${RUN_DIR}/latest.json"' in source
    assert 'test -f "${RUN_DIR}/segment_0000/post_evaluation.pt"' in source
    assert '--output-dir "${RUN_DIR}" --profile main --resume "${RUN_DIR}"' in source
    assert "--source-rebind" not in source
    assert "[[ ! -e" not in source


def test_phase6_v2_progress_stall_recovery_bootstrap_defers_root_latest(
    tmp_path: Path,
) -> None:
    source = inspect.getsource(train_phase6._run_v2_progress_stall_recovery)
    assert 'work_root / "latest.json"' not in source
    assert 'work_root / "progress.json"' in source
    assert 'work_root / "v2_recovery_record.json"' in source
    assert "segment_0043/recovery/pre_evaluation_progress_stall_rebound.pt" in source

    shell = (
        V2_SCRIPT_ROOT / "recover_independent_segment43_progress_stall_4gpu.sh"
    ).read_text(encoding="utf-8")
    assert 'test ! -e "${TARGET_ROOT}/latest.json"' in shell
    assert shell.index('test ! -e "${TARGET_ROOT}/latest.json"') < shell.index(
        "for MODE in residual scaffold_only"
    )
    assert 'test -f "${TARGET_ROOT}/latest.json"' in shell
    assert 'root_latest.get("checkpoint")' in shell
    assert "segment_0043/post_evaluation.pt" in shell
    production_source = inspect.getsource(train_phase6._run_production)
    assert "and recovery_record is None" in production_source

    root = tmp_path / "root"
    root.mkdir()
    payload = {
        "checkpoint": str(tmp_path / "segment_0043/post_evaluation.pt"),
        "segment": 43,
    }
    _write_mainrunner_root_latest(root, payload)
    assert json.loads((root / "latest.json").read_text()) == payload
    with pytest.raises(FileExistsError, match="refuses to move backwards or overwrite"):
        _write_mainrunner_root_latest(root, payload)


def test_phase6_v2_production_rejects_invalid_bootstrap_and_progress(
    tmp_path: Path,
) -> None:
    parser = train_phase6._production_parser()
    for values in (
        [
            "--mode",
            "production",
            "--output-dir",
            str(tmp_path),
            "--profile",
            "pilot",
            "--resume",
            str(tmp_path / "checkpoint.pt"),
        ],
        [
            "--mode",
            "production",
            "--output-dir",
            str(tmp_path),
            "--profile",
            "pilot",
            "--source-rebind",
            str(tmp_path / "rebind.json"),
        ],
        ["--mode", "production", "--output-dir", str(tmp_path), "--profile", "main"],
        [
            "--mode",
            "production",
            "--output-dir",
            str(tmp_path),
            "--profile",
            "main",
            "--resume",
            str(tmp_path / "other_run"),
        ],
    ):
        with pytest.raises(ValueError):
            _run_production(parser.parse_args(values))
    progress = tmp_path / "progress.json"
    progress.write_text(
        json.dumps(
            {
                "active_mode": None,
                "active_substage": "segment_complete",
                "completed_rows": 3072,
                "control_steps": 1,
                "current_iteration": 31,
                "error_state": None,
                "eta_s": None,
                "last_update_utc": "2026-08-19T00:00:00Z",
                "segment": 0,
                "stage": "C1",
                "target_iteration": 31,
            }
        ),
        encoding="utf-8",
    )
    train_phase6._update_mainrunner_progress(
        progress,
        target_iteration=2442,
        current_iteration=31,
        segment=0,
        stage="C1",
        active_substage="ready",
        active_mode=None,
        completed_rows=3072,
        control_steps=1,
        eta_s=None,
        error_state=None,
    )
    with pytest.raises(ValueError, match="not monotonic"):
        train_phase6._update_mainrunner_progress(
            progress,
            target_iteration=31,
            current_iteration=31,
            segment=0,
            stage="C1",
            active_substage="ready",
            active_mode=None,
            completed_rows=3072,
            control_steps=1,
            eta_s=None,
            error_state=None,
        )


def test_phase6_v2_production_preserves_v1_rebind_requirement(tmp_path: Path) -> None:
    args = train_phase6._production_parser().parse_args(
        [
            "--mode",
            "production",
            "--output-dir",
            str(tmp_path),
            "--profile",
            "main",
            "--resume",
            str(tmp_path / "legacy.pt"),
            "--phase6-config",
            str(CONFIG_PATH),
            "--roster",
            str(ROSTER_PATH),
            "--acceptance-config",
            "configs/phase6_learning_acceptance_v1.json",
        ]
    )
    with pytest.raises(
        ValueError, match="historical main production requires source rebind"
    ):
        _run_production(args)
    source = inspect.getsource(_run_production)
    assert 'if not v2 and args.profile == "main"' in source
    assert "with_source_rebind" in source


def _gloo_gradient_worker(
    rank: int, world_size: int, init_path: str, output_dir: str
) -> None:
    from somaforce_cross.learning.joint_runner import GradientAverager

    distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    policy = torch.nn.Linear(1, 1)
    for parameter in policy.parameters():
        parameter.grad = torch.full_like(parameter, float(rank + 1))
    GradientAverager(world_size=world_size, allow_test_backend=True)(policy)
    torch.save(
        [parameter.grad.detach().clone() for parameter in policy.parameters()],
        Path(output_dir) / f"rank_{rank}.pt",
    )
    distributed.destroy_process_group()


def _roster() -> tuple[Phase6Config, object]:
    config = load_phase6_config(CONFIG_PATH)
    return config, load_phase6_task_roster(ROSTER_PATH, phase6_config=config)


def _uniform_p_cross_trace() -> list[list[list[float]]]:
    return [[[1.0 / 65.0 for _ in range(5)] for _ in range(13)]]


def _joint_metrics(config: Phase6Config, roster: object) -> dict[str, object]:
    schema = config.payload["metrics"]
    records = {
        task.task: {
            "episode_count": 0,
            "episode_metrics": {name: None for name in schema["episode_metric_names"]},
            "p_cross_trace": _uniform_p_cross_trace(),
            "task_transitions": 64,
            "transition_metrics": {
                name: 1.0 for name in schema["transition_metric_names"]
            },
        }
        for task in roster.tasks
    }
    return {"pooled": pooled_macro_metrics(records)["pooled"], "task_records": records}


def _final_assembly_inputs() -> tuple[dict[str, object], dict[str, object], object]:
    config, roster = _roster()
    acceptance = load_learning_acceptance_config(PHASE6_LEARNING_CONFIG)
    records: dict[str, object] = {}
    evaluation: dict[str, object] = {}
    transitions = (3, 5, 7, 11)
    rewards = (-2.0, 3.0, -5.0, 7.0)
    semantics = (-1.0, 2.0, 4.0, 8.0)
    for index, task in enumerate(roster.tasks):
        records[task.task] = {
            "task_transitions": transitions[index],
            "transition_metrics": {
                "reward": rewards[index],
                "semantic_total": semantics[index],
            },
        }
        family = f"family_{index}"
        evaluation[task.task] = {
            "nominal_retention": 1.0,
            "nominal_invalid": 0,
            "nominal_saturation": 0.0,
            "residual_mean_norm": 0.1,
            "contact_bearing_residual_fraction": 0.5,
            "family_metrics": {
                family: {
                    "residual": {
                        "success": 1.0,
                        "progress": 1.0,
                        "p95_wrench": 0.9,
                        "stability_margin": 1.0,
                    },
                    "scaffold_only": {
                        "success": 0.0,
                        "progress": 0.0,
                        "p95_wrench": 1.0,
                        "stability_margin": 1.0,
                    },
                }
            },
        }
    return (
        {"task_records": records},
        {"curriculum_task_metrics": evaluation},
        acceptance,
    )


def test_mainrunner_final_acceptance_metrics_are_production_assembled() -> None:
    train, evaluation, acceptance = _final_assembly_inputs()
    assembled = train_phase6._mainrunner_final_acceptance_metrics(
        train_metrics=train, evaluation_metrics=evaluation, acceptance=acceptance
    )
    assert set(assembled) == set(evaluation["curriculum_task_metrics"])
    assert assembled["push_door_hand"]["transition_fraction"] == pytest.approx(3 / 26)
    assert assembled["move_largebox"]["transition_fraction"] == pytest.approx(11 / 26)
    assert assembled["push_door_hand"]["normalized_reward_share"] == pytest.approx(
        6 / 133
    )
    assert assembled["move_largebox"]["normalized_reward_share"] == pytest.approx(
        77 / 133
    )
    assert assembled["push_door_hand"]["semantic_total_share"] == 0.0
    assert assembled["move_largebox"]["semantic_total_share"] == pytest.approx(88 / 126)


def test_mainrunner_final_acceptance_rejects_missing_evidence() -> None:
    train, evaluation, acceptance = _final_assembly_inputs()
    del evaluation["curriculum_task_metrics"]["push_box"]["residual_mean_norm"]
    with pytest.raises(KeyError):
        train_phase6._mainrunner_final_acceptance_metrics(
            train_metrics=train, evaluation_metrics=evaluation, acceptance=acceptance
        )
    base = {
        "task": "push_door_hand",
        "mode": "residual",
        "stage": "C3",
        "subset": "stage",
        "family": "physical",
        "env_id": 0,
        "seed": 1,
        "steps": 2,
        "success": True,
        "failure": False,
        "timeout": False,
        "invalid": False,
        "return": 0.0,
        "raw_reward_sums": {name: 0.0 for name in train_phase6._EVALUATION_REWARD_KEYS},
        "weighted_reward_sums": {
            name: 0.0 for name in train_phase6._EVALUATION_REWARD_KEYS
        },
        "diagnostics": {name: 0.0 for name in train_phase6._EVALUATION_DIAGNOSTIC_KEYS},
        "acceptance_diagnostics": {
            "residual_norm_sum": 2.0,
            "contact_residual_norm_sum": 1.0,
            "transition_count": 2,
            "residual_mean_norm": 1.0,
            "contact_bearing_residual_fraction": 0.5,
        },
    }
    cases = []
    missing = copy.deepcopy(base)
    del missing["acceptance_diagnostics"]["residual_norm_sum"]
    cases.append(missing)
    extra = copy.deepcopy(base)
    extra["acceptance_diagnostics"]["extra"] = 0.0
    cases.append(extra)
    for key, value in (
        ("residual_norm_sum", float("nan")),
        ("residual_norm_sum", -1.0),
        ("contact_residual_norm_sum", 3.0),
        ("transition_count", 0),
        ("transition_count", 3),
        ("residual_mean_norm", 2.0),
        ("contact_bearing_residual_fraction", 0.0),
    ):
        case = copy.deepcopy(base)
        case["acceptance_diagnostics"][key] = value
        cases.append(case)
    zero = copy.deepcopy(base)
    zero["acceptance_diagnostics"] = {
        "residual_norm_sum": 0.0,
        "contact_residual_norm_sum": 0.0,
        "transition_count": 2,
        "residual_mean_norm": 1.0,
        "contact_bearing_residual_fraction": 0.0,
    }
    cases.append(zero)
    for case in cases:
        with pytest.raises(ValueError):
            train_phase6._validate_completed_episode_record(
                case, task="push_door_hand", mode="residual", stage="C3"
            )
    scaffold = copy.deepcopy(base)
    scaffold["mode"] = "scaffold_only"
    with pytest.raises(ValueError):
        train_phase6._validate_completed_episode_record(
            scaffold, task="push_door_hand", mode="scaffold_only", stage="C3"
        )
    residual_records = []
    scaffold_records = []
    for seed, subset, family in ((1, "stage", "physical"), (2, "nominal", "nominal")):
        residual_record = copy.deepcopy(base)
        residual_record.update({"seed": seed, "subset": subset, "family": family})
        scaffold_record = copy.deepcopy(residual_record)
        scaffold_record["mode"] = "scaffold_only"
        scaffold_record["acceptance_diagnostics"] = {
            "residual_norm_sum": 0.0,
            "contact_residual_norm_sum": 0.0,
            "transition_count": 2,
            "residual_mean_norm": 0.0,
            "contact_bearing_residual_fraction": 0.0,
        }
        residual_records.append(residual_record)
        scaffold_records.append(scaffold_record)
    train_phase6._mainrunner_task_metrics(
        residual_records=residual_records, scaffold_records=scaffold_records
    )
    for records, mutation in (
        (scaffold_records[:-1], lambda row: row),
        (copy.deepcopy(scaffold_records), lambda row: row.update({"subset": "stage"})),
        (copy.deepcopy(scaffold_records), lambda row: row.update({"family": "sensor"})),
    ):
        mutated = copy.deepcopy(records)
        mutation(mutated[-1])
        with pytest.raises(ValueError):
            train_phase6._mainrunner_task_metrics(
                residual_records=residual_records, scaffold_records=mutated
            )


def test_mainrunner_benefit_assignment_requires_two_tasks_and_distinct_families() -> (
    None
):
    train, evaluation, acceptance = _final_assembly_inputs()
    rows = evaluation["curriculum_task_metrics"]
    physical = copy.deepcopy(rows["push_door_hand"]["family_metrics"]["family_0"])
    sensor = copy.deepcopy(physical)
    sensor["residual"]["success"] = 0.1
    sensor["residual"]["p95_wrench"] = 0.94
    rows["push_door_hand"]["family_metrics"] = {"physical": physical}
    rows["push_box"]["family_metrics"] = {"physical": physical, "sensor": sensor}
    for task in ("move_suitcase", "move_largebox"):
        rows[task]["family_metrics"] = {
            "physical": {
                **physical,
                "residual": {
                    **physical["residual"],
                    "success": 0.0,
                    "progress": 0.0,
                    "p95_wrench": 1.1,
                },
            }
        }
    assembled = train_phase6._mainrunner_final_acceptance_metrics(
        train_metrics=train, evaluation_metrics=evaluation, acceptance=acceptance
    )
    families = {
        row["benefit"].get("mismatch_family")
        for row in assembled.values()
        if row["benefit"]["passes"]
    }
    assert assembled["push_door_hand"]["benefit"]["mismatch_family"] == "physical"
    assert assembled["push_box"]["benefit"]["mismatch_family"] == "sensor"
    assert families == {"physical", "sensor"}
    assert sum(row["benefit"]["passes"] for row in assembled.values()) == 2
    reversed_evaluation = {
        "curriculum_task_metrics": dict(reversed(list(rows.items())))
    }
    assert (
        train_phase6._mainrunner_final_acceptance_metrics(
            train_metrics=train,
            evaluation_metrics=reversed_evaluation,
            acceptance=acceptance,
        )
        == assembled
    )


def test_mainrunner_reward_and_semantic_shares_are_normalized() -> None:
    train, evaluation, acceptance = _final_assembly_inputs()
    assembled = train_phase6._mainrunner_final_acceptance_metrics(
        train_metrics=train, evaluation_metrics=evaluation, acceptance=acceptance
    )
    assert sum(
        row["normalized_reward_share"] for row in assembled.values()
    ) == pytest.approx(1.0)
    assert sum(
        row["semantic_total_share"] for row in assembled.values()
    ) == pytest.approx(1.0)
    for row in train["task_records"].values():
        row["transition_metrics"]["reward"] = 0.0
    with pytest.raises(ValueError, match="reward mass is zero"):
        train_phase6._mainrunner_final_acceptance_metrics(
            train_metrics=train, evaluation_metrics=evaluation, acceptance=acceptance
        )


def test_mainrunner_zero_semantic_total_uses_contract_zero_share() -> None:
    train, evaluation, acceptance = _final_assembly_inputs()
    for row in train["task_records"].values():
        row["transition_metrics"]["semantic_total"] = -1.0
    assembled = train_phase6._mainrunner_final_acceptance_metrics(
        train_metrics=train, evaluation_metrics=evaluation, acceptance=acceptance
    )
    assert {row["semantic_total_share"] for row in assembled.values()} == {0.0}


def test_mainrunner_final_c3_endpoint_calls_validator_with_assembled_metrics(
    tmp_path: Path,
) -> None:
    # The production finalizer imports these boundaries locally.  Keep the train
    # metrics and final assembly real while replacing only checkpoint/Isaac evidence.
    import somaforce_cross.learning.joint_runner as joint_runner

    train, evaluation, acceptance = _final_assembly_inputs()
    _, roster = _roster()
    payload = {
        "pre_evaluation": True,
        "curriculum": {"stage": "C3"},
        "evaluation_history": [],
        "next_crossing": {
            "logical_transitions": 250000,
            "target_iteration": 2442,
            "actual_transitions": 20004864,
        },
        "actual_global_transitions": 20004864,
        "actual_per_task_transitions": 5001216,
        "optimizer_step": 58608,
        "metrics": train,
    }
    calls: list[dict[str, object]] = []
    source_manifest = {"source": hashlib.sha256(b"source").hexdigest()}
    tmp = tmp_path
    segment = tmp / "segment"
    segment.mkdir()
    evaluation_dir = segment / "evaluation"
    evaluation_dir.mkdir()
    evaluation_path = evaluation_dir / "evaluation_summary.json"
    evaluation_path.write_text("{}", encoding="utf-8")
    args = Namespace(
        segment_index=1,
        iteration=2442,
        segment_dir=segment,
        run_dir=tmp,
        pre_checkpoint=tmp / "pre.pt",
        evaluation_summary=evaluation_path,
        logical_crossing=250000,
        phase6_config=CONFIG_PATH,
        roster=ROSTER_PATH,
        acceptance_config=PHASE6_LEARNING_CONFIG,
        source_rebind=tmp / "rebind.json",
    )

    class _Curriculum:
        def load_state_dict(self, value: object) -> None:
            self.value = value

        def state_dict(self) -> object:
            return self.value

        def evaluate_window(self, **_: object) -> object:
            return Namespace(
                crossing=250000, promoted=False, rolled_back=False, stage="C3"
            )

    saved_payload: dict[str, object] | None = None

    def restore(path: Path, *_: object, **__: object) -> dict[str, object]:
        resolved = path.resolve()
        if resolved == args.pre_checkpoint.resolve():
            return copy.deepcopy(payload)
        if resolved == (segment / "post_evaluation.pt").resolve():
            assert saved_payload is not None
            return copy.deepcopy(saved_payload)
        raise AssertionError(f"unexpected checkpoint restore path: {resolved}")

    def save(path: Path, value: object) -> str:
        nonlocal saved_payload
        assert path.resolve() == (segment / "post_evaluation.pt").resolve()
        assert isinstance(value, dict)
        saved_payload = copy.deepcopy(value)
        path.write_bytes(b"checkpoint")
        return train_phase6._sha256(path)

    def validate(metrics: dict[str, object], **_: object) -> dict[str, bool]:
        calls.append(metrics)
        return {"c3": True}

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(joint_runner, "load_phase6_config", lambda _: _roster()[0])
        monkeypatch.setattr(
            joint_runner, "load_phase6_task_roster", lambda *_args, **_kwargs: roster
        )
        monkeypatch.setattr(
            joint_runner, "load_learning_acceptance_config", lambda _: acceptance
        )
        monkeypatch.setattr(joint_runner, "restore_learning_checkpoint", restore)
        monkeypatch.setattr(joint_runner, "atomic_learning_checkpoint", save)
        monkeypatch.setattr(
            joint_runner, "validate_learning_final_acceptance", validate
        )
        monkeypatch.setattr(joint_runner, "JointCurriculum", _Curriculum)
        monkeypatch.setattr(
            train_phase6,
            "_validate_mainrunner_rebind_record",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            train_phase6,
            "_validate_mainrunner_evaluation_summary",
            lambda *_args, **_kwargs: {**evaluation, "evidence_manifests": {}},
        )
        monkeypatch.setattr(
            train_phase6,
            "_current_mainrunner_source_manifest",
            lambda: dict(source_manifest),
        )
        assert train_phase6._run_mainrunner_finalize(args) == 0
    finally:
        monkeypatch.undo()

    failure_run = tmp_path / "validator_failure"
    failure_segment = failure_run / "segment"
    failure_evaluation = failure_segment / "evaluation" / "evaluation_summary.json"
    failure_evaluation.parent.mkdir(parents=True)
    failure_evaluation.write_text("{}", encoding="utf-8")
    failure_args = Namespace(
        segment_index=1,
        iteration=2442,
        segment_dir=failure_segment,
        run_dir=failure_run,
        pre_checkpoint=failure_run / "pre.pt",
        evaluation_summary=failure_evaluation,
        logical_crossing=250000,
        phase6_config=CONFIG_PATH,
        roster=ROSTER_PATH,
        acceptance_config=PHASE6_LEARNING_CONFIG,
        source_rebind=failure_run / "rebind.json",
    )
    failure_payload = copy.deepcopy(payload)
    failure_saved: dict[str, object] | None = None
    validator_calls: list[dict[str, object]] = []

    class _ValidatorSentinel(Exception):
        pass

    sentinel = _ValidatorSentinel("final validator failure")

    def failure_restore(path: Path, *_: object, **__: object) -> dict[str, object]:
        if path.resolve() == failure_args.pre_checkpoint.resolve():
            return copy.deepcopy(failure_payload)
        if path.resolve() == (failure_segment / "post_evaluation.pt").resolve():
            assert failure_saved is not None
            return copy.deepcopy(failure_saved)
        raise AssertionError(f"unexpected failure fixture restore path: {path}")

    def failure_save(path: Path, value: object) -> str:
        nonlocal failure_saved
        assert path.resolve() == (failure_segment / "post_evaluation.pt").resolve()
        assert isinstance(value, dict)
        failure_saved = copy.deepcopy(value)
        path.write_bytes(b"failure-checkpoint")
        return train_phase6._sha256(path)

    def failure_validate(metrics: dict[str, object], **_: object) -> dict[str, bool]:
        validator_calls.append(metrics)
        raise sentinel

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(joint_runner, "load_phase6_config", lambda _: _roster()[0])
        monkeypatch.setattr(
            joint_runner, "load_phase6_task_roster", lambda *_args, **_kwargs: roster
        )
        monkeypatch.setattr(
            joint_runner, "load_learning_acceptance_config", lambda _: acceptance
        )
        monkeypatch.setattr(
            joint_runner, "restore_learning_checkpoint", failure_restore
        )
        monkeypatch.setattr(joint_runner, "atomic_learning_checkpoint", failure_save)
        monkeypatch.setattr(
            joint_runner, "validate_learning_final_acceptance", failure_validate
        )
        monkeypatch.setattr(joint_runner, "JointCurriculum", _Curriculum)
        monkeypatch.setattr(
            train_phase6,
            "_validate_mainrunner_rebind_record",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            train_phase6,
            "_validate_mainrunner_evaluation_summary",
            lambda *_args, **_kwargs: {**evaluation, "evidence_manifests": {}},
        )
        monkeypatch.setattr(
            train_phase6,
            "_current_mainrunner_source_manifest",
            lambda: dict(source_manifest),
        )
        with pytest.raises(_ValidatorSentinel) as raised:
            train_phase6._run_mainrunner_finalize(failure_args)
        assert raised.value is sentinel
    finally:
        monkeypatch.undo()
    assert len(validator_calls) == 1
    assert validator_calls[0]["iteration"] == 2442
    assert validator_calls[0]["stage"] == "C3"
    assert set(validator_calls[0]["tasks"]) == set(
        evaluation["curriculum_task_metrics"]
    )
    assert failure_saved is not None
    assert failure_saved["pre_evaluation"] is False
    assert failure_saved["curriculum"] == {"stage": "C3"}
    assert failure_saved["evaluation_history"] == [
        {**evaluation, "evidence_manifests": {}}
    ]
    for name in (
        "actual_global_transitions",
        "actual_per_task_transitions",
        "metrics",
        "next_crossing",
        "optimizer_step",
    ):
        assert _semantic_equal(failure_saved[name], failure_payload[name])
    assert (failure_segment / "post_evaluation.pt").is_file()
    assert (failure_segment / "latest.json").is_file()
    assert not (failure_segment / "segment_summary.json").exists()
    assert not (failure_run / "latest.json").exists()
    assert all(
        "final_acceptance_complete" not in path.read_text(encoding="utf-8")
        for path in failure_run.rglob("*.json")
    )
    assert saved_payload is not None
    assert saved_payload["pre_evaluation"] is False
    assert saved_payload["curriculum"] == {"stage": "C3"}
    assert saved_payload["evaluation_history"] == [
        {**evaluation, "evidence_manifests": {}}
    ]
    for name in (
        "actual_global_transitions",
        "actual_per_task_transitions",
        "metrics",
        "next_crossing",
        "optimizer_step",
    ):
        assert _semantic_equal(saved_payload[name], payload[name])
    summary = json.loads((segment / "segment_summary.json").read_text())
    assert calls and calls[0]["tasks"] == summary["final_acceptance_metrics"]
    assert (
        summary["final_acceptance_metrics_sha256"]
        == hashlib.sha256(
            json.dumps(
                summary["final_acceptance_metrics"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
    )
    assert summary["final_acceptance_inputs"]["acceptance"] == {
        "raw_sha256": acceptance.raw_sha256,
        "canonical_sha256": acceptance.canonical_sha256,
    }
    assert summary["evaluation_summary"] == {
        "path": str(evaluation_path),
        "sha256": train_phase6._sha256(evaluation_path),
    }
    assert summary["final_acceptance_inputs"]["evaluation_summary"] == {
        "path": str(evaluation_path),
        "sha256": train_phase6._sha256(evaluation_path),
    }
    expected_manifest_sha256 = train_phase6._source_manifest_digest(source_manifest)
    assert summary["source_manifest_sha256"] == expected_manifest_sha256
    assert (
        summary["final_acceptance_inputs"]["source_manifest_sha256"]
        == expected_manifest_sha256
    )
    completed_segment = Namespace(
        end_iteration=2442,
        end_transitions=20004864,
        segment_index=1,
        crossing=Namespace(logical_transitions=250000),
    )
    command = [
        str(train_phase6.ISAAC_PYTHON),
        str(Path(train_phase6.__file__).resolve()),
        "--mode",
        "mainrunner-finalize",
        "--run-dir",
        str(tmp),
        "--segment-dir",
        str(segment),
        "--pre-checkpoint",
        str(args.pre_checkpoint),
        "--evaluation-summary",
        str(evaluation_path),
        "--iteration",
        "2442",
        "--segment-index",
        "1",
        "--logical-crossing",
        "250000",
        "--stage",
        "C3",
        "--source-rebind",
        str(args.source_rebind),
    ]
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(joint_runner, "restore_learning_checkpoint", restore)
        monkeypatch.setattr(joint_runner, "JointCurriculum", _Curriculum)
        monkeypatch.setattr(
            joint_runner, "validate_learning_final_acceptance", validate
        )
        monkeypatch.setattr(
            train_phase6,
            "_validate_mainrunner_stage_evidence",
            lambda *_args, **_kwargs: command,
        )
        assert (
            train_phase6._validate_mainrunner_completed_finalize(
                run_dir=tmp,
                segment_dir=segment,
                segment=completed_segment,
                pre_checkpoint=args.pre_checkpoint,
                pre_restored=payload,
                evaluation={**evaluation, "evidence_manifests": {}},
                source_rebind=args.source_rebind,
                config=_roster()[0],
                roster=roster,
                acceptance=acceptance,
                source_manifest=source_manifest,
            )["segment"]
            == 1
        )
        pristine = copy.deepcopy(summary)
        mutations = (
            lambda value: value["final_acceptance_metrics"]["push_box"].__setitem__(
                "nominal_retention", 0.0
            ),
            lambda value: value.__setitem__(
                "final_acceptance_metrics_sha256", "0" * 64
            ),
            lambda value: value["final_acceptance_inputs"]["checkpoint"].__setitem__(
                "path", "tampered.pt"
            ),
            lambda value: value["final_acceptance_inputs"]["checkpoint"].__setitem__(
                "sha256", "0" * 64
            ),
            lambda value: value["final_acceptance_inputs"][
                "evaluation_summary"
            ].__setitem__("path", "tampered.json"),
            lambda value: value["final_acceptance_inputs"][
                "evaluation_summary"
            ].__setitem__("sha256", "0" * 64),
            lambda value: value["final_acceptance_inputs"].__setitem__(
                "source_manifest_sha256", "0" * 64
            ),
            lambda value: value["final_acceptance_inputs"]["acceptance"].__setitem__(
                "raw_sha256", "0" * 64
            ),
            lambda value: value["final_acceptance_inputs"]["acceptance"].__setitem__(
                "canonical_sha256", "0" * 64
            ),
            lambda value: value.__setitem__("source_manifest_sha256", "0" * 64),
        )
        for mutate in mutations:
            altered = copy.deepcopy(pristine)
            mutate(altered)
            (segment / "segment_summary.json").write_text(
                json.dumps(altered), encoding="utf-8"
            )
            with pytest.raises(ValueError):
                train_phase6._validate_mainrunner_completed_finalize(
                    run_dir=tmp,
                    segment_dir=segment,
                    segment=completed_segment,
                    pre_checkpoint=args.pre_checkpoint,
                    pre_restored=payload,
                    evaluation={**evaluation, "evidence_manifests": {}},
                    source_rebind=args.source_rebind,
                    config=_roster()[0],
                    roster=roster,
                    acceptance=acceptance,
                    source_manifest=source_manifest,
                )
        (segment / "segment_summary.json").write_text(
            json.dumps(pristine), encoding="utf-8"
        )
        assert (
            train_phase6._validate_mainrunner_completed_finalize(
                run_dir=tmp,
                segment_dir=segment,
                segment=completed_segment,
                pre_checkpoint=args.pre_checkpoint,
                pre_restored=payload,
                evaluation={**evaluation, "evidence_manifests": {}},
                source_rebind=args.source_rebind,
                config=_roster()[0],
                roster=roster,
                acceptance=acceptance,
                source_manifest=source_manifest,
            )["segment"]
            == 1
        )
    finally:
        monkeypatch.undo()


def test_phase6_algorithm_schema_raw_canonical_and_bindings() -> None:
    config = load_phase6_config(CONFIG_PATH)

    assert config.raw_sha256 == CONFIG_RAW_SHA256
    assert config.canonical_sha256 == CONFIG_CANONICAL_SHA256
    assert canonical_sha256(config.payload) == CONFIG_CANONICAL_SHA256
    assert config.payload["dimensions"] == {
        "action": 23,
        "actor_input": 220,
        "critic": 845,
        "policy": 668,
        "semantic_target": 31,
    }
    assert config.payload["runtime"] == {
        "backend": "nccl",
        "init_method": "env://",
        "process_group_timeout_s": 180,
        "required_visible_gpus": 4,
        "world_size": 4,
    }
    assert config.payload["metrics"]["impulse_dt_s"] == 0.02
    assert config.payload["metrics"]["p_cross_shape"] == [13, 5]
    assert config.payload["metrics"]["force_rate_excludes_first_control_step"]


def test_phase6_algorithm_rejects_unknown_nan_and_binding_mutations() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    cases = []
    unknown = copy.deepcopy(payload)
    unknown["unknown"] = 1
    cases.append(unknown)
    nonfinite = copy.deepcopy(payload)
    nonfinite["ppo"]["learning_rate"] = float("nan")
    cases.append(nonfinite)
    bad_binding = copy.deepcopy(payload)
    bad_binding["bindings"]["phase5"]["raw_sha256"] = "0" * 64
    cases.append(bad_binding)
    bad_metrics = copy.deepcopy(payload)
    bad_metrics["metrics"]["unknown"] = "value"
    cases.append(bad_metrics)

    for value in cases:
        with pytest.raises((TypeError, ValueError)):
            validate_phase6_config(value)


def test_phase6_roster_schema_raw_canonical_and_artifact_verification() -> None:
    config, roster = _roster()

    assert roster.raw_sha256 == ROSTER_RAW_SHA256
    assert roster.canonical_sha256 == ROSTER_CANONICAL_SHA256
    assert roster.phase6_canonical_sha256 == config.canonical_sha256
    assert [item.task for item in roster.tasks] == [
        "push_door_hand",
        "push_box",
        "move_suitcase",
        "move_largebox",
    ]
    assert all(item.artifact_path.is_dir() for item in roster.tasks)


def test_phase6_roster_rejects_unadmitted_duplicate_and_unknown_adapter(
    tmp_path: Path,
) -> None:
    config = load_phase6_config(CONFIG_PATH)
    payload = json.loads(ROSTER_PATH.read_text(encoding="utf-8"))
    variants: list[dict[str, object]] = []
    unadmitted = copy.deepcopy(payload)
    unadmitted["tasks"][0]["admitted"] = False
    variants.append(unadmitted)
    duplicate = copy.deepcopy(payload)
    duplicate["tasks"][1]["task"] = "push_door_hand"
    variants.append(duplicate)
    unknown = copy.deepcopy(payload)
    unknown["tasks"][0]["adapter_identity"] = "unknown.Adapter"
    variants.append(unknown)

    for index, value in enumerate(variants):
        path = tmp_path / f"roster_{index}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(ValueError):
            load_phase6_task_roster(path, phase6_config=config)


def test_scheduler_current_four_task_window_assigns_one_rank_per_task() -> None:
    scheduler = JointTaskScheduler(task_count=4, world_size=4)

    assert scheduler.supercycle_windows == 1
    assert scheduler.slots_per_task == 1
    assert [
        scheduler.assignment(cycle_index=0, window_index=0, rank=rank).task_index
        for rank in range(4)
    ] == [0, 1, 2, 3]


def test_scheduler_seven_tasks_four_ranks_has_exact_supercycle_balance() -> None:
    scheduler = JointTaskScheduler(task_count=7, world_size=4)

    assert scheduler.supercycle_windows == 7
    assert scheduler.slots_per_task == 4
    assert scheduler.supercycle_slot_counts(cycle_index=3) == {
        index: 4 for index in range(7)
    }


def test_transition_accounting_is_per_transition_and_not_per_episode() -> None:
    config = load_phase6_config(CONFIG_PATH)
    global_transitions = transition_accounting(
        world_size=4, num_envs=2, num_steps_per_env=32, iterations=1
    )

    assert global_transitions == 256
    assert_transition_balance(
        {task: 64 for task in ("door", "box", "suitcase", "largebox")},
        expected_total=global_transitions,
    )
    with pytest.raises(ValueError):
        assert_transition_balance(
            {"door": 65, "box": 64, "suitcase": 64, "largebox": 63},
            expected_total=global_transitions,
        )
    collector = TransitionMetricCollector(config.payload["metrics"])
    p_cross = torch.full((2, 13, 5), 1.0 / 65.0)
    common = {
        "contact_fraction": torch.tensor([0.0, 1.0]),
        "stability_margin": torch.tensor([0.1, 0.2]),
        "residual_arms_norm": torch.tensor([1.0, 2.0]),
        "residual_waist_norm": torch.tensor([3.0, 4.0]),
        "residual_legs_norm": torch.tensor([5.0, 6.0]),
        "saturation_fraction": torch.tensor([0.0, 0.5]),
        "p_cross": p_cross,
    }
    first = torch.zeros(2, 2, 6)
    first[0, 0, 0] = 3.0
    first[0, 1, 1] = 4.0
    first[1, :, 2] = 5.0
    collector.observe(observed_wrench=first, **common)
    collector.observe(observed_wrench=first * 2.0, **common)
    collected = collector.transition_metrics()
    assert collector.transitions == 4
    assert collected["force_impulse_Ns_mean"] == pytest.approx(0.255)
    assert collected["force_rate_p95_N_per_s"] > 0.0
    trace = collector.p_cross_trace()
    assert len(trace) == 2 and len(trace[0]) == 13 and len(trace[0][0]) == 5
    assert sum(value for row in trace[0] for value in row) == pytest.approx(1.0)


def test_joint_runner_keeps_task_metadata_out_of_residual_actor_inputs() -> None:
    actor_source = inspect.getsource(ResidualActorCritic.actor_forward)
    scheduler_source = inspect.getsource(JointTaskScheduler.assignment)

    assert "task_id" not in actor_source
    assert "object_identity" not in actor_source
    assert "task" not in actor_source
    assert "policy" not in scheduler_source


def test_phase5_semantic_ppo_default_has_no_gradient_sync_hook() -> None:
    algorithm = SemanticPPO(ResidualActorCritic(), storage_class=object)

    assert algorithm.gradient_sync_hook is None


def test_cpu_four_process_gloo_gradient_sync_substitute(tmp_path: Path) -> None:
    init_path = tmp_path / "gloo_init"
    multiprocessing.spawn(
        _gloo_gradient_worker,
        args=(4, str(init_path), str(tmp_path)),
        nprocs=4,
        join=True,
    )

    for rank in range(4):
        gradients = torch.load(tmp_path / f"rank_{rank}.pt", weights_only=False)
        assert all(
            torch.equal(value, torch.full_like(value, 2.5)) for value in gradients
        )


def test_gradient_sync_hook_runs_after_backward_before_clip_and_step() -> None:
    source = inspect.getsource(SemanticPPO.update)

    assert (
        source.index("combined_loss.backward()")
        < source.index("self.gradient_sync_hook(self.policy)")
        < source.index("nn.utils.clip_grad_norm_")
        < source.index("self.optimizer.step()")
    )


def test_policy_and_optimizer_hash_records_must_match() -> None:
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    record = {
        "policy": state_dict_sha256(policy.state_dict()),
        "optimizer": state_dict_sha256(optimizer.state_dict()),
        "step": optimizer_step(optimizer),
    }

    assert_matching_hash_records([record, dict(record)])
    changed = dict(record)
    changed["step"] = 1
    with pytest.raises(RuntimeError):
        assert_matching_hash_records([record, changed])


def test_phase6_initialization_loads_policy_with_fresh_optimizer_only() -> None:
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)

    load_phase5_policy_only(policy, PHASE5_CHECKPOINT)

    assert optimizer.state == {}
    assert optimizer_step(optimizer) == 0
    assert policy.state_dict()


def test_joint_checkpoint_restore_rejects_repeated_or_skipped_window(
    tmp_path: Path,
) -> None:
    config, roster = _roster()
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    curriculum = JointCurriculum(transitions=256)
    policy_hash = state_dict_sha256(policy.state_dict())
    optimizer_hash = state_dict_sha256(optimizer.state_dict())
    payload = joint_checkpoint_payload(
        config=config,
        roster=roster,
        policy=policy,
        optimizer=optimizer,
        iteration=1,
        cycle_index=0,
        window_index=0,
        slot_transitions=64,
        task_transitions={task.task: 64 for task in roster.tasks},
        curriculum=curriculum,
        rank_rng={
            str(rank): {
                "cpu": torch.arange(2, dtype=torch.uint8),
                "cuda": [torch.arange(2, dtype=torch.uint8) for _ in range(4)],
            }
            for rank in range(4)
        },
        metrics=_joint_metrics(config, roster),
        hashes={"policy": policy_hash, "optimizer": optimizer_hash, "step": 0},
    )
    path = tmp_path / "joint.pt"
    atomic_torch_save(path, payload)

    restore_joint_checkpoint(
        path,
        config=config,
        roster=roster,
        policy=ResidualActorCritic(),
        optimizer=torch.optim.Adam(ResidualActorCritic().parameters(), lr=3.0e-4),
        expected_cycle_index=1,
        expected_window_index=0,
        device="cpu",
    )
    with pytest.raises(ValueError):
        restore_joint_checkpoint(
            path,
            config=config,
            roster=roster,
            policy=ResidualActorCritic(),
            optimizer=torch.optim.Adam(ResidualActorCritic().parameters(), lr=3.0e-4),
            expected_cycle_index=0,
            expected_window_index=0,
            device="cpu",
        )


def test_pooled_macro_metrics_uses_equal_transition_tasks_and_equal_episodes() -> None:
    records = {
        "door": {
            "episode_count": 1,
            "episode_metrics": {"force_p95": 2.0},
            "p_cross_trace": _uniform_p_cross_trace(),
            "transition_metrics": {"reward": 1.0, "semantic_total": 3.0},
            "task_transitions": 64,
        },
        "box": {
            "episode_count": 3,
            "episode_metrics": {"force_p95": 6.0},
            "p_cross_trace": _uniform_p_cross_trace(),
            "transition_metrics": {"reward": 5.0, "semantic_total": 7.0},
            "task_transitions": 64,
        },
    }

    pooled = pooled_macro_metrics(records)

    assert pooled["pooled"]["transition_mean"] == {
        "reward": 3.0,
        "semantic_total": 5.0,
    }
    assert pooled["pooled"]["episode_mean"]["force_p95"] == 5.0
    assert pooled["pooled"]["task_episode_counts"] == {"door": 1, "box": 3}
    zero_episode = {
        task: {
            "episode_count": 0,
            "episode_metrics": {"force_p95": None},
            "p_cross_trace": _uniform_p_cross_trace(),
            "task_transitions": 64,
            "transition_metrics": {"reward": 1.0},
        }
        for task in ("door", "box")
    }
    zero_pooled = pooled_macro_metrics(zero_episode)
    assert zero_pooled["pooled"]["episode_mean"] == {"force_p95": None}
    missing_trace = copy.deepcopy(records)
    missing_trace["door"].pop("p_cross_trace")
    with pytest.raises(ValueError):
        pooled_macro_metrics(missing_trace)
    malformed_trace = copy.deepcopy(records)
    malformed_trace["box"]["p_cross_trace"] = [[[1.0]]]
    with pytest.raises(ValueError):
        pooled_macro_metrics(malformed_trace)


def test_curriculum_crossings_persistence_and_global_stage_machine() -> None:
    curriculum = JointCurriculum()

    assert JointCurriculum.evaluation_crossings(249999, 750001) == (
        250000,
        500000,
        750000,
    )
    for crossing in (250000, 500000, 750000):
        decision = curriculum.evaluate(crossing=crossing, passed=True, rollback=False)
    assert decision.stage == "C2"
    restored = JointCurriculum()
    restored.load_state_dict(curriculum.state_dict())
    assert restored.state_dict() == curriculum.state_dict()
    assert "broadcast_object_list" in inspect.getsource(
        __import__("somaforce_cross.learning.joint_runner", fromlist=["*"])
    )


def test_bounded_profiles_have_frozen_transition_and_optimizer_arithmetic() -> None:
    assert _profile("smoke_4gpu_2env_1iter") == (2, 1)
    assert _profile("smoke_4gpu_64env_2iter") == (64, 2)
    assert (
        transition_accounting(
            world_size=4, num_envs=2, num_steps_per_env=32, iterations=1
        )
        == 256
    )
    assert (
        transition_accounting(
            world_size=4, num_envs=64, num_steps_per_env=32, iterations=2
        )
        == 16384
    )
    assert 3 * 8 == 24
    assert 3 * 8 * 2 == 48


def test_wrapper_requires_result_order_exit_zero_and_env_closed_last_marker(
    tmp_path: Path,
) -> None:
    markers = ["JSON_WRITTEN", "PHASE6_JOINT_RESULT", "ENV_CLOSED"]

    assert _parent_close_verified(
        result={"status": "ok"},
        exit_code=0,
        timed_out=False,
        markers=markers,
        log_text="JSON_WRITTEN\nPHASE6_JOINT_RESULT={}\nENV_CLOSED\n",
    )
    assert not _parent_close_verified(
        result={"status": "ok"},
        exit_code=0,
        timed_out=False,
        markers=markers + ["APP_CLOSED"],
        log_text="ENV_CLOSED\nAPP_CLOSED\n",
    )
    config = load_phase6_config(V2_CONFIG_PATH)
    roster = load_phase6_task_roster(V2_ROSTER_PATH, phase6_config=config)
    metrics = _joint_metrics(config, roster)
    validate_phase6_joint_metrics(
        metrics,
        config=config,
        roster=roster,
        expected_global_transitions=256,
    )
    checkpoint_path = tmp_path / "checkpoints" / "cycle_0000_window_0000.pt"
    checkpoint_path.parent.mkdir(parents=True)
    torch.save(
        {
            "metrics": metrics,
            "optimizer_sha256": "b" * 64,
            "optimizer_step": 24,
            "policy_sha256": "a" * 64,
            "rank_rng": {str(rank): {} for rank in range(4)},
            "task_transitions": {task.task: 64 for task in roster.tasks},
        },
        checkpoint_path,
    )
    checkpoint = {
        "path": str(checkpoint_path),
        "sha256": __import__(
            "somaforce_cross.learning.joint_runner", fromlist=["sha256_file"]
        ).sha256_file(checkpoint_path),
    }
    profile = "smoke_4gpu_2env_1iter"
    per_task_transitions = {task.task: 64 for task in roster.tasks}
    task_fraction = {task: 0.25 for task in per_task_transitions}
    initialization = {
        "contract_version": "phase6_independent_initialization_v1",
        "source": "random",
        "seed": 20260806,
        "parameter_count": 916881,
        "initial_policy_sha256": (
            "cc3116f729f05b536084a50323a613c6dbdb73149424d314864196e402bad01d"
        ),
        "actor_output_zero_init": True,
        "rank0_broadcast_verified": True,
        "optimizer_initial_sha256": "c" * 64,
        "optimizer_initial_step": 0,
        "external_checkpoint": None,
    }
    for rank in range(4):
        rank_dir = tmp_path / f"rank_{rank}"
        rank_dir.mkdir()
        slot = JointTaskScheduler(task_count=4, world_size=4).assignment(
            cycle_index=0, window_index=0, rank=rank
        )
        command = [
            "python",
            "scripts/train_phase6.py",
            "--mode",
            "worker",
            "--output-dir",
            str(tmp_path),
            "--profile",
            profile,
        ]
        (rank_dir / "command.json").write_text(
            json.dumps(
                {"command": command, "local_rank": rank, "rank": rank, "world_size": 4}
            ),
            encoding="utf-8",
        )
        result = {
            "checkpoint": checkpoint,
            "contracts": {
                "phase6_canonical_sha256": config.canonical_sha256,
                "phase6_raw_sha256": config.raw_sha256,
                "roster_canonical_sha256": roster.canonical_sha256,
                "roster_raw_sha256": roster.raw_sha256,
            },
            "curriculum": JointCurriculum(transitions=256).state_dict(),
            "global_transitions": 256,
            "initialization": initialization,
            "local_transitions": 64,
            "metrics": metrics,
            "optimizer_sha256": "b" * 64,
            "optimizer_step": 24,
            "per_task_transitions": per_task_transitions,
            "policy_sha256": "a" * 64,
            "profile": profile,
            "rank": rank,
            "schedule": {
                "cycle_index": 0,
                "task": roster.tasks[slot.task_index].task,
                "task_index": slot.task_index,
                "window_index": 0,
            },
            "status": "ok",
            "task_fraction": task_fraction,
        }
        (rank_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        (rank_dir / "worker.log").write_text(
            "JSON_WRITTEN\nPHASE6_JOINT_RESULT={}\nENV_CLOSED\n", encoding="utf-8"
        )
        wrapper = {
            "close_verification": "parent_verified_exit0",
            "command": command,
            "elapsed_s": 1.0,
            "exit_code": 0,
            "last_progress_marker": "ENV_CLOSED",
            "local_rank": rank,
            "log": str(rank_dir / "worker.log"),
            "marker_order_valid": True,
            "markers": markers,
            "passed": True,
            "rank": rank,
            "result_error": None,
            "result_json": str(rank_dir / "result.json"),
            "result_status": "ok",
            "signal": None,
            "timed_out": False,
            "timeout_kind": None,
            "vmhwm_kib": 1,
            "warning_counts": {
                "headless_glfw": 0,
                "kvdb_lock": 0,
                "multiple_installable_client_drivers": 0,
            },
            "world_size": 4,
        }
        (rank_dir / "wrapper.json").write_text(json.dumps(wrapper), encoding="utf-8")
    args = Namespace(
        output_dir=tmp_path,
        phase6_config=V2_CONFIG_PATH,
        profile=profile,
        roster=V2_ROSTER_PATH,
    )
    summary = summarize_phase6_run(args)
    assert summary["status"] == "ok"
    assert len(summary["rank_results"]) == len(summary["wrapper_summaries"]) == 4
    assert all(
        result["initialization"] == initialization for result in summary["rank_results"]
    )
    rank_dir = tmp_path / "rank_3"
    valid_wrapper = json.loads((rank_dir / "wrapper.json").read_text(encoding="utf-8"))
    valid_result = json.loads((rank_dir / "result.json").read_text(encoding="utf-8"))
    del valid_result["initialization"]
    (rank_dir / "result.json").write_text(json.dumps(valid_result), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown or missing schema keys"):
        summarize_phase6_run(args)
    valid_result["initialization"] = initialization
    (rank_dir / "result.json").write_text(json.dumps(valid_result), encoding="utf-8")
    for timeout_kind in ("wall_clock", "progress_stall"):
        invalid_wrapper = copy.deepcopy(valid_wrapper)
        invalid_wrapper["timeout_kind"] = timeout_kind
        (rank_dir / "wrapper.json").write_text(
            json.dumps(invalid_wrapper), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="rank wrapper close evidence"):
            summarize_phase6_run(args)
    missing_timeout_kind = copy.deepcopy(valid_wrapper)
    del missing_timeout_kind["timeout_kind"]
    (rank_dir / "wrapper.json").write_text(
        json.dumps(missing_timeout_kind), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown or missing schema keys"):
        summarize_phase6_run(args)
    unknown_wrapper_key = copy.deepcopy(valid_wrapper)
    unknown_wrapper_key["unknown"] = True
    (rank_dir / "wrapper.json").write_text(
        json.dumps(unknown_wrapper_key), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="unknown or missing schema keys"):
        summarize_phase6_run(args)
    (rank_dir / "wrapper.json").write_text(json.dumps(valid_wrapper), encoding="utf-8")
    (rank_dir / "worker.log").write_text(
        "Multiple Installable Client Drivers\n", encoding="utf-8"
    )
    wrapper = json.loads((rank_dir / "wrapper.json").read_text(encoding="utf-8"))
    wrapper["warning_counts"]["multiple_installable_client_drivers"] = 1
    (rank_dir / "wrapper.json").write_text(json.dumps(wrapper), encoding="utf-8")
    with pytest.raises(RuntimeError, match="multiple Vulkan"):
        summarize_phase6_run(args)
    production_source = inspect.getsource(_run_production)
    assert '"rank-wrapper"' in production_source
    assert '"--worker-mode"' in production_source
    production_command = {
        "command": [
            "python",
            "scripts/train_phase6.py",
            "--mode",
            "train-segment",
            "--output-dir",
            str(tmp_path),
            "--profile",
            "production_segment",
            "--acceptance-config",
            str(train_phase6.PHASE6_LEARNING_CONFIG),
            "--phase6-config",
            str(PHASE6_CONFIG),
            "--roster",
            str(PHASE6_ROSTER),
            "--segment-start",
            "0",
            "--segment-end",
            "31",
            "--stage",
            "C1",
            "--headless",
        ],
        "local_rank": 0,
        "rank": 0,
        "world_size": 4,
    }
    assert production_command["command"].count("--stage") == 1
    assert _command_argument(production_command["command"], "--stage") == "C1"
    _validate_production_command(
        production_command,
        rank=0,
        output_dir=tmp_path,
        worker_mode="train-segment",
        iteration=31,
        stage="C1",
    )
    production_command["command"].remove("--headless")
    with pytest.raises(ValueError, match="frozen contracts"):
        _validate_production_command(
            production_command,
            rank=0,
            output_dir=tmp_path,
            worker_mode="train-segment",
            iteration=31,
            stage="C1",
        )
    devicefix_source = inspect.getsource(_run_devicefix_recovery_evaluate)
    assert "evaluation_devicefix_attempt_0001" in devicefix_source
    assert "paired_evaluation_devicefix.log" in devicefix_source
    stepboundary_source = inspect.getsource(_run_stepboundary_recovery_evaluate)
    assert "evaluation_stepboundary_attempt_0002" in stepboundary_source
    assert "paired_evaluation_stepboundary.log" in stepboundary_source
    assert "attempt.exists()" in stepboundary_source
    assert '(recovery / "post_evaluation.pt").exists()' in stepboundary_source
    assert '(recovery / "latest.json").exists()' in stepboundary_source


def test_phase6_learning_acceptance_schema_bindings_and_arithmetic() -> None:
    acceptance = load_learning_acceptance_config(PHASE6_LEARNING_CONFIG)
    assert acceptance.payload["contract_version"] == "phase6_learning_acceptance_v1"
    assert acceptance.payload["runtime"] == {
        "evaluation_seed": 20262806,
        "global_transitions_per_iteration": 8192,
        "initial_stage": "C1",
        "num_envs_per_rank": 64,
        "num_steps_per_env": 32,
        "optimizer_steps_per_iteration": 24,
        "train_seed": 20260806,
        "world_size": 4,
    }
    assert acceptance.payload["profiles"]["pilot"] == {
        "total_iterations": 31,
        "global_transitions": 253952,
        "per_task_transitions": 63488,
        "optimizer_steps": 744,
    }
    assert acceptance.payload["profiles"]["main"] == {
        "total_iterations": 2442,
        "global_transitions": 20004864,
        "per_task_transitions": 5001216,
        "optimizer_steps": 58608,
    }
    assert crossing_target_iteration(250000) == 31
    source = Path("scripts/train_phase6.py").read_text(encoding="utf-8")
    tree = ast.parse(source, filename="scripts/train_phase6.py")
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert any(
        node.module == "somaforce_cross.learning.acceptance"
        and any(alias.name == "source_sha256" for alias in node.names)
        for node in imports
    )
    assert all(
        node.module != "somaforce_cross.learning.joint_runner"
        or all(alias.name != "source_sha256" for alias in node.names)
        for node in imports
    )
    assert 'REPO_ROOT / "somaforce_cross/learning/acceptance.py"' in source


def test_phase6_learning_acceptance_rejects_unknown_nonfinite_and_mutations() -> None:
    payload = json.loads(PHASE6_LEARNING_CONFIG.read_text(encoding="utf-8"))
    cases = []
    unknown = copy.deepcopy(payload)
    unknown["unexpected"] = True
    cases.append(unknown)
    nonfinite = copy.deepcopy(payload)
    nonfinite["final_acceptance"]["stability_degradation_max"] = float("nan")
    cases.append(nonfinite)
    mutated = copy.deepcopy(payload)
    mutated["bindings"]["phase6_raw_sha256"] = "0" * 64
    cases.append(mutated)
    arithmetic = copy.deepcopy(payload)
    arithmetic["profiles"]["main"]["optimizer_steps"] += 1
    cases.append(arithmetic)
    for value in cases:
        with pytest.raises((TypeError, ValueError)):
            validate_learning_acceptance_config(value)


def test_learning_segment_plan_hits_each_crossing_once() -> None:
    pilot = plan_learning_segments(total_iterations=31)
    assert len(pilot) == 1
    assert pilot[0].crossing.logical_transitions == 250000
    assert pilot[0].crossing.actual_transitions == 253952
    main = plan_learning_segments(total_iterations=2442)
    assert len(main) == 80
    assert len({segment.crossing.logical_transitions for segment in main}) == 80
    assert main[-1].crossing.logical_transitions == 20000000
    assert main[-1].crossing.actual_transitions == 20004864
    assert all(
        current.start_iteration == previous.end_iteration
        and current.start_transitions == previous.end_transitions
        for previous, current in zip(main, main[1:])
    )
    for actual, expected in (
        (
            253952,
            {
                "logical_transitions": 250000,
                "target_iteration": 31,
                "actual_transitions": 253952,
            },
        ),
        (
            507904,
            {
                "logical_transitions": 500000,
                "target_iteration": 62,
                "actual_transitions": 507904,
            },
        ),
        (
            753664,
            {
                "logical_transitions": 750000,
                "target_iteration": 92,
                "actual_transitions": 753664,
            },
        ),
        (
            20004864,
            {
                "logical_transitions": 20000000,
                "target_iteration": 2442,
                "actual_transitions": 20004864,
            },
        ),
    ):
        assert train_phase6._current_evaluation_crossing(actual) == expected
    assert (
        train_phase6._current_evaluation_crossing(507904)["logical_transitions"]
        != 750000
    )
    for segment in main:
        assert train_phase6._current_evaluation_crossing(segment.end_transitions) == {
            "logical_transitions": segment.crossing.logical_transitions,
            "target_iteration": segment.crossing.target_iteration,
            "actual_transitions": segment.crossing.actual_transitions,
        }
    for invalid in (0, True, "507904", 507903, 507905):
        with pytest.raises(ValueError, match="evaluation crossing"):
            train_phase6._current_evaluation_crossing(invalid)  # type: ignore[arg-type]
    worker_source = inspect.getsource(train_phase6._worker_main)
    assert "_current_evaluation_crossing(global_transitions)" in worker_source
    assert "((global_transitions // 250000) + 1) * 250000" not in worker_source


def test_mainrunner_source_manifest_and_rank_rng_contract_are_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _current_mainrunner_source_manifest(
        CONFIG_PATH, ROSTER_PATH, PHASE6_LEARNING_CONFIG
    )
    expected_v1_paths = {
        str(Path(path).resolve())
        for path in (
            "configs/phase6_learning_acceptance_v1.json",
            "configs/phase6_joint_training_v1.json",
            "configs/phase6_task_roster_v1.json",
            "somaforce_cross/learning/acceptance.py",
            "somaforce_cross/learning/joint_runner.py",
            "scripts/train_phase6.py",
            "somaforce_cross/envs/residual_env.py",
        )
    }
    assert len(manifest) == 7
    assert set(manifest) == expected_v1_paths
    default_manifest = _current_mainrunner_source_manifest()
    expected_v2_paths = {
        str(Path(path).resolve())
        for path in (
            "configs/phase6_learning_acceptance_v2.json",
            "configs/phase6_joint_training_v2.json",
            "configs/phase6_task_roster_v2.json",
            "somaforce_cross/learning/acceptance.py",
            "somaforce_cross/learning/joint_runner.py",
            "scripts/train_phase6.py",
            "somaforce_cross/envs/residual_env.py",
        )
    }
    assert len(default_manifest) == 7
    assert set(default_manifest) == expected_v2_paths
    with pytest.raises(ValueError, match="complete known lineage"):
        _current_mainrunner_source_manifest(CONFIG_PATH, V2_ROSTER_PATH)
    with pytest.raises(ValueError, match="must be coherent"):
        _current_mainrunner_source_manifest(
            V2_CONFIG_PATH, V2_ROSTER_PATH, PHASE6_LEARNING_CONFIG
        )
    live_sources = {
        "train_phase6": Path("scripts/train_phase6.py").resolve(),
        "joint_runner": Path("somaforce_cross/learning/joint_runner.py").resolve(),
        "residual_env": Path("somaforce_cross/envs/residual_env.py").resolve(),
    }
    for path in live_sources.values():
        assert manifest[str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()
    recovery = Path(
        "outputs/phase6_learning_acceptance/"
        "phase6-learning-pilot-4gpu-64env-31iter-20260806_042940/"
        "segment_0000/recovery"
    )
    snapshot_dir = recovery / "source_snapshot"
    proof = _mainrunner_ast_source_proof(snapshot_dir)
    expected_snapshots = {
        "train_phase6_pre_mainrunner.py": (
            "3db545b1cf8e3436e658151b29b91679e09056bf8bd9e981f58589f36650c2dc"
        ),
        "joint_runner_pre_mainrunner.py": (
            "4cf86dda401494a0793351ce99de775872d52d91647184f0a5a30702e7b16d2c"
        ),
        "residual_env_pre_mainrunner.py": (
            "86aa99b303290a3b746af65936baf93b93fcb00ae99ce6632721df997185293f"
        ),
    }
    assert {
        Path(value["path"]).name: value["sha256"]
        for value in proof["snapshots"].values()
    } == expected_snapshots
    assert {name: value["path"] for name, value in proof["current"].items()} == {
        name: str(path) for name, path in live_sources.items()
    }
    copied_snapshot_dir = tmp_path / "source_snapshot"
    copied_snapshot_dir.mkdir()
    for name in expected_snapshots:
        (copied_snapshot_dir / name).write_bytes((snapshot_dir / name).read_bytes())
    (copied_snapshot_dir / "train_phase6_pre_mainrunner.py").write_bytes(b"broken")
    with pytest.raises(ValueError, match="frozen snapshot differs"):
        _mainrunner_ast_source_proof(copied_snapshot_dir)
    (copied_snapshot_dir / "train_phase6_pre_mainrunner.py").write_bytes(
        (snapshot_dir / "joint_runner_pre_mainrunner.py").read_bytes()
    )
    with pytest.raises(ValueError, match="frozen snapshot differs"):
        _mainrunner_ast_source_proof(copied_snapshot_dir)
    states = {
        str(rank): {
            "cpu": torch.arange(4, dtype=torch.uint8),
            "cuda": [torch.arange(4, dtype=torch.uint8) for _ in range(4)],
        }
        for rank in range(4)
    }
    validate_rank_rng_states(states)
    states["3"] = {"cpu": torch.arange(4, dtype=torch.uint8), "cuda": []}
    with pytest.raises(ValueError, match="device states"):
        validate_rank_rng_states(states)
    source = Path("scripts/train_phase6.py").read_text(encoding="utf-8")
    assert '"--source-rebind"' in inspect.getsource(_run_production)
    assert '"--max-segments"' in source
    assert "restore_rank_rng_state(" in source
    production_source = inspect.getsource(_run_production)
    assert production_source.index("_validate_mainrunner_completed_segments(") < (
        production_source.index("_update_mainrunner_progress(")
    )
    assert "_validate_mainrunner_completed_train(" in source
    assert "_merge_mainrunner_evaluation(" in source
    assert "_validate_mainrunner_stage_evidence(" in source
    worker_source = inspect.getsource(train_phase6._worker_main)
    assert "transition_offset=previous_task_transitions[task.task]" in worker_source
    assert "_canonical_task_metrics(task_metrics, roster=roster)" in worker_source
    wrapper_source = inspect.getsource(train_phase6._run_rank_wrapper)
    assert wrapper_source.count("_aggregate_rank_wrapper_progress(") == 2

    segment = plan_learning_segments(total_iterations=2442)[1]
    resume_checkpoint = tmp_path / "resume.pt"
    source_rebind = tmp_path / "source_rebind_mainrunner.json"

    def outer_command(
        *, worker_mode: str, paired_mode: str | None = None, stage: str = "C1"
    ) -> list[str]:
        start = (
            segment.start_iteration
            if worker_mode == "train-segment"
            else segment.end_iteration
        )
        end = segment.end_iteration if worker_mode == "train-segment" else start + 1
        command = [
            str(train_phase6.ISAAC_PYTHON),
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node=4",
            str(Path("scripts/train_phase6.py").resolve()),
            "--mode",
            "rank-wrapper",
            "--worker-mode",
            worker_mode,
            "--output-dir",
            str(tmp_path / worker_mode / str(paired_mode)),
            "--profile",
            "production_segment",
            "--timeout-s",
            "1",
            "--segment-start",
            str(start),
            "--segment-end",
            str(end),
            "--stage",
            stage,
            "--resume",
            str(resume_checkpoint),
            "--source-rebind",
            str(source_rebind),
            "--acceptance-config",
            str(train_phase6.PHASE6_LEARNING_CONFIG),
            "--phase6-config",
            str(PHASE6_CONFIG),
            "--roster",
            str(PHASE6_ROSTER),
        ]
        if paired_mode is not None:
            command.extend(("--paired-mode", paired_mode))
        return command

    outer_train = outer_command(worker_mode="train-segment", stage="C2")
    train_output = Path(
        train_phase6._mainrunner_command_argument(outer_train, "--output-dir")
    )
    train_phase6._validate_mainrunner_worker_stage_command(
        outer_train,
        worker_mode="train-segment",
        output_dir=train_output,
        segment=segment,
        resume_checkpoint=resume_checkpoint,
        source_rebind=source_rebind,
        stage="C2",
    )
    for paired_mode in ("residual", "scaffold_only"):
        outer_evaluation = outer_command(
            worker_mode="evaluate", paired_mode=paired_mode, stage="C2"
        )
        evaluation_output = Path(
            train_phase6._mainrunner_command_argument(outer_evaluation, "--output-dir")
        )
        train_phase6._validate_mainrunner_worker_stage_command(
            outer_evaluation,
            worker_mode="evaluate",
            output_dir=evaluation_output,
            segment=segment,
            resume_checkpoint=resume_checkpoint,
            source_rebind=source_rebind,
            paired_mode=paired_mode,
            stage="C2",
        )
    for argument, value in (
        ("--headless", None),
        ("--worker-mode", "evaluate"),
        ("--segment-end", "63"),
        ("--resume", str(tmp_path / "wrong_resume.pt")),
        ("--source-rebind", str(tmp_path / "wrong_source_rebind.json")),
    ):
        invalid_outer = [*outer_train]
        if value is None:
            invalid_outer.append(argument)
        else:
            invalid_outer[invalid_outer.index(argument) + 1] = value
        with pytest.raises(ValueError):
            train_phase6._validate_mainrunner_worker_stage_command(
                invalid_outer,
                worker_mode="train-segment",
                output_dir=train_output,
                segment=segment,
                resume_checkpoint=resume_checkpoint,
                source_rebind=source_rebind,
                stage="C2",
            )
    invalid_paired = outer_command(
        worker_mode="evaluate", paired_mode="residual", stage="C3"
    )
    invalid_paired[invalid_paired.index("--paired-mode") + 1] = "scaffold_only"
    with pytest.raises(ValueError):
        train_phase6._validate_mainrunner_worker_stage_command(
            invalid_paired,
            worker_mode="evaluate",
            output_dir=Path(
                train_phase6._mainrunner_command_argument(
                    invalid_paired, "--output-dir"
                )
            ),
            segment=segment,
            resume_checkpoint=resume_checkpoint,
            source_rebind=source_rebind,
            paired_mode="residual",
            stage="C3",
        )
    inner_args = Namespace(
        acceptance_config=train_phase6.PHASE6_LEARNING_CONFIG,
        cycle_index=segment.segment_index,
        evaluation_seed=20262806,
        output_dir=tmp_path / "inner",
        paired_mode="residual",
        paired_nominal_quota=128,
        paired_num_envs=None,
        paired_stage_quota=256,
        phase6_config=PHASE6_CONFIG,
        profile="production_segment",
        progress_stall_timeout_s=1,
        resume=resume_checkpoint,
        roster=PHASE6_ROSTER,
        segment_end=segment.end_iteration,
        segment_start=segment.start_iteration,
        source_rebind=source_rebind,
        stage="C2",
        timeout_s=2,
        window_index=0,
        worker_mode="train-segment",
    )
    inner = train_phase6._worker_command(inner_args)
    assert "--headless" in inner
    train_phase6._validate_production_command(
        {"command": inner, "local_rank": 0, "rank": 0, "world_size": 4},
        rank=0,
        output_dir=tmp_path / "inner",
        worker_mode="train-segment",
        iteration=segment.end_iteration,
        stage="C2",
    )
    inner_evaluation_args = copy.copy(inner_args)
    inner_evaluation_args.worker_mode = "evaluate"
    inner_evaluation_args.output_dir = tmp_path / "inner_evaluation"
    inner_evaluation_args.segment_start = segment.end_iteration
    inner_evaluation_args.segment_end = segment.end_iteration + 1
    inner_evaluation_args.stage = "C3"
    inner_evaluation = train_phase6._worker_command(inner_evaluation_args)
    assert "--headless" in inner_evaluation
    assert train_phase6._command_argument(inner_evaluation, "--stage") == "C3"
    inner_without_headless = [value for value in inner if value != "--headless"]
    with pytest.raises(ValueError, match="production rank command"):
        train_phase6._validate_production_command(
            {
                "command": inner_without_headless,
                "local_rank": 0,
                "rank": 0,
                "world_size": 4,
            },
            rank=0,
            output_dir=tmp_path / "inner",
            worker_mode="train-segment",
            iteration=segment.end_iteration,
            stage="C2",
        )
    aggregate_calls: list[tuple[str, Path, int]] = []
    with monkeypatch.context() as patched:
        patched.setattr(
            train_phase6,
            "_aggregate_evaluation_progress",
            lambda output_dir, world_size: aggregate_calls.append(
                ("evaluation", output_dir, world_size)
            ),
        )
        patched.setattr(
            train_phase6,
            "_aggregate_modeprocess_progress",
            lambda output_dir, world_size: aggregate_calls.append(
                ("modeprocess", output_dir, world_size)
            ),
        )
        train_phase6._aggregate_rank_wrapper_progress(
            Namespace(
                output_dir=tmp_path / "train",
                paired_mode=None,
                worker_mode="train-segment",
            ),
            rank=0,
            world_size=4,
        )
        assert aggregate_calls == []
        train_phase6._aggregate_rank_wrapper_progress(
            Namespace(
                output_dir=tmp_path / "evaluation",
                paired_mode=None,
                worker_mode="evaluate",
            ),
            rank=0,
            world_size=4,
        )
        assert aggregate_calls == [("evaluation", tmp_path / "evaluation", 4)]
        aggregate_calls.clear()
        train_phase6._aggregate_rank_wrapper_progress(
            Namespace(
                output_dir=tmp_path / "evaluation" / "residual",
                paired_mode="residual",
                worker_mode="evaluate",
            ),
            rank=0,
            world_size=4,
        )
        assert aggregate_calls == [
            ("evaluation", tmp_path / "evaluation" / "residual", 4),
            ("modeprocess", tmp_path / "evaluation", 4),
        ]

    direct_attempt = tmp_path / "direct_mainrunner_attempt"
    direct_environment = os.environ.copy()
    direct_environment.pop("PYTHONPATH", None)
    direct_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    direct_entrypoint = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/train_phase6.py").resolve()),
            "--mode",
            "mainrunner-rebind",
            "--input-checkpoint",
            str(tmp_path / "missing_input.pt"),
            "--input-source-record",
            str(tmp_path / "missing_source_record.json"),
            "--attempt-dir",
            str(direct_attempt),
        ],
        capture_output=True,
        cwd=tmp_path,
        env=direct_environment,
        text=True,
    )
    assert direct_entrypoint.returncode != 0
    assert "mainrunner input checkpoint checksum mismatch" in direct_entrypoint.stderr
    assert "ModuleNotFoundError" not in direct_entrypoint.stderr
    assert not direct_attempt.exists()

    config, roster, acceptance = _v2()
    production_manifest = default_manifest
    assert config.payload["contract_version"] == "phase6_joint_training_v2"
    assert (
        json.loads(V2_ROSTER_PATH.read_text(encoding="utf-8"))["contract_version"]
        == "phase6_task_roster_v2"
    )
    assert roster.phase6_canonical_sha256 == config.canonical_sha256
    assert acceptance.payload["contract_version"] == "phase6_learning_acceptance_v2"
    assert production_manifest == _current_mainrunner_source_manifest()
    metric_schema = config.payload["metrics"]

    class MetricCollector:
        def __init__(self, transitions: int) -> None:
            self.transitions = transitions

        def transition_metrics(self) -> dict[str, float]:
            return {
                name: 1.0
                for name in metric_schema["transition_metric_names"]
                if name != "reward" and name not in metric_schema["ppo_metric_names"]
            }

        def raw_samples(self) -> dict[str, object]:
            return {}

        def p_cross_trace(self) -> list[list[list[float]]]:
            return _uniform_p_cross_trace()

    def local_run(transition_count: int) -> dict[str, object]:
        return {
            "iteration_records": [
                {
                    "mean_reward": 1.0,
                    "update": {name: 1.0 for name in metric_schema["ppo_metric_names"]},
                }
            ],
            "transition_count": transition_count,
        }

    segment_delta = 31 * 64 * 32
    fresh = train_phase6._local_metrics(
        task="push_door_hand",
        run=local_run(segment_delta),
        transition_offset=0,
        collector=MetricCollector(segment_delta),
        episode_snapshot=(),
        metric_schema=metric_schema,
    )
    assert fresh["record"]["task_transitions"] == segment_delta
    resumed = train_phase6._local_metrics(
        task="push_door_hand",
        run=local_run(segment_delta * 2),
        transition_offset=segment_delta,
        collector=MetricCollector(segment_delta),
        episode_snapshot=(),
        metric_schema=metric_schema,
    )
    assert resumed["record"]["task_transitions"] == segment_delta
    for incorrect_count in (segment_delta, segment_delta * 3):
        with pytest.raises(ValueError, match="runner transition count"):
            train_phase6._local_metrics(
                task="push_door_hand",
                run=local_run(incorrect_count),
                transition_offset=segment_delta,
                collector=MetricCollector(segment_delta),
                episode_snapshot=(),
                metric_schema=metric_schema,
            )
    for invalid_offset in (True, -1, "63488"):
        with pytest.raises(ValueError, match="transition offset"):
            train_phase6._local_metrics(
                task="push_door_hand",
                run=local_run(segment_delta),
                transition_offset=invalid_offset,
                collector=MetricCollector(segment_delta),
                episode_snapshot=(),
                metric_schema=metric_schema,
            )

    def production_metrics(per_task_transitions: int) -> dict[str, object]:
        records = {
            task.task: {
                "episode_count": 0,
                "episode_metrics": {
                    name: None for name in metric_schema["episode_metric_names"]
                },
                "p_cross_trace": _uniform_p_cross_trace(),
                "task_transitions": per_task_transitions,
                "transition_metrics": {
                    name: 1.0 for name in metric_schema["transition_metric_names"]
                },
            }
            for task in roster.tasks
        }
        metrics = {
            "pooled": pooled_macro_metrics(records)["pooled"],
            "task_records": records,
        }
        validate_phase6_joint_metrics(
            metrics,
            config=config,
            roster=roster,
            expected_global_transitions=per_task_transitions * 4,
        )
        return metrics

    def production_result(
        *,
        end_iteration: int,
        local_transitions: int,
        per_task_transitions: int,
        cycle_index: int,
        rank: int,
    ) -> dict[str, object]:
        global_transitions = end_iteration * 8192
        slot = JointTaskScheduler(task_count=4, world_size=4).assignment(
            cycle_index=cycle_index,
            window_index=0,
            rank=rank,
        )
        initialization = {
            "contract_version": "phase6_independent_initialization_v1",
            "source": "random",
            "seed": 20260806,
            "parameter_count": 916881,
            "initial_policy_sha256": (
                "cc3116f729f05b536084a50323a613c6dbdb73149424d314864196e402bad01d"
            ),
            "actor_output_zero_init": True,
            "rank0_broadcast_verified": True,
            "optimizer_initial_sha256": "d" * 64,
            "optimizer_initial_step": 0,
            "external_checkpoint": None,
        }
        return {
            "checkpoint": {"path": "/tmp/pre_evaluation.pt", "sha256": "a" * 64},
            "contracts": {
                "phase6_canonical_sha256": config.canonical_sha256,
                "phase6_raw_sha256": config.raw_sha256,
                "roster_canonical_sha256": roster.canonical_sha256,
                "roster_raw_sha256": roster.raw_sha256,
            },
            "curriculum": JointCurriculum(transitions=global_transitions).state_dict(),
            "global_transitions": global_transitions,
            "initialization": initialization,
            "local_transitions": local_transitions,
            "metrics": production_metrics(per_task_transitions),
            "optimizer_sha256": "b" * 64,
            "optimizer_step": end_iteration * 24,
            "per_task_transitions": {
                task.task: per_task_transitions for task in roster.tasks
            },
            "policy_sha256": "c" * 64,
            "profile": "production_segment",
            "rank": rank,
            "schedule": {
                "cycle_index": cycle_index,
                "task": roster.tasks[slot.task_index].task,
                "task_index": slot.task_index,
                "window_index": 0,
            },
            "status": "ok",
            "task_fraction": {task.task: 0.25 for task in roster.tasks},
        }

    segment_delta = 31 * 64 * 32
    fresh_results = [
        production_result(
            end_iteration=31,
            local_transitions=segment_delta,
            per_task_transitions=segment_delta,
            cycle_index=0,
            rank=rank,
        )
        for rank in range(4)
    ]
    resumed_results = [
        production_result(
            end_iteration=62,
            local_transitions=segment_delta,
            per_task_transitions=segment_delta * 2,
            cycle_index=1,
            rank=rank,
        )
        for rank in range(4)
    ]
    for result_set, end_iteration, per_task, local, cycle in (
        (fresh_results, 31, segment_delta, segment_delta, 0),
        (resumed_results, 62, segment_delta * 2, segment_delta, 1),
    ):
        for rank, result in enumerate(result_set):
            record = train_phase6._validate_production_train_result(
                result,
                rank=rank,
                config=config,
                roster=roster,
                expected_global=end_iteration * 8192,
                expected_per_task=per_task,
                expected_local_transitions=local,
                expected_cycle_index=cycle,
                expected_window_index=0,
            )
            assert "task" not in record
            assert record["schedule"]["task"] == roster.tasks[(rank + cycle) % 4].task
            assert record["initialization"] == result["initialization"]

    missing_initialization = copy.deepcopy(resumed_results[0])
    del missing_initialization["initialization"]
    with pytest.raises(ValueError, match="unknown or missing schema keys"):
        train_phase6._validate_production_train_result(
            missing_initialization,
            rank=0,
            config=config,
            roster=roster,
            expected_global=507904,
            expected_per_task=126976,
            expected_local_transitions=segment_delta,
            expected_cycle_index=1,
            expected_window_index=0,
        )

    malformed_initialization = copy.deepcopy(resumed_results[0])
    malformed_initialization["initialization"]["seed"] = 0
    with pytest.raises(ValueError, match="independent initialization provenance"):
        train_phase6._validate_production_train_result(
            malformed_initialization,
            rank=0,
            config=config,
            roster=roster,
            expected_global=507904,
            expected_per_task=126976,
            expected_local_transitions=segment_delta,
            expected_cycle_index=1,
            expected_window_index=0,
        )

    for wrong_local in (segment_delta * 2, segment_delta * 3):
        with pytest.raises(ValueError, match="production rank result accounting"):
            train_phase6._validate_production_train_result(
                resumed_results[0],
                rank=0,
                config=config,
                roster=roster,
                expected_global=507904,
                expected_per_task=126976,
                expected_local_transitions=wrong_local,
                expected_cycle_index=1,
                expected_window_index=0,
            )
    wrong_cumulative = copy.deepcopy(resumed_results[0])
    wrong_cumulative["per_task_transitions"] = {
        task.task: segment_delta for task in roster.tasks
    }
    with pytest.raises(ValueError, match="production rank result accounting"):
        train_phase6._validate_production_train_result(
            wrong_cumulative,
            rank=0,
            config=config,
            roster=roster,
            expected_global=507904,
            expected_per_task=126976,
            expected_local_transitions=segment_delta,
            expected_cycle_index=1,
            expected_window_index=0,
        )
    for field, value in (
        ("cycle_index", 0),
        ("window_index", 1),
        ("task_index", 0),
        ("task", "push_door_hand"),
    ):
        mutated = copy.deepcopy(resumed_results[0])
        mutated["schedule"][field] = value
        with pytest.raises(ValueError, match="production rank result task assignment"):
            train_phase6._validate_production_train_result(
                mutated,
                rank=0,
                config=config,
                roster=roster,
                expected_global=507904,
                expected_per_task=126976,
                expected_local_transitions=segment_delta,
                expected_cycle_index=1,
                expected_window_index=0,
            )
    summary_source = inspect.getsource(train_phase6._run_production_summary)
    assert 'task = result["task"]' in summary_source
    assert 'task = record["schedule"]["task"]' in summary_source
    assert 'task = result.get("task")' not in summary_source

    def ordered_metric_record(reward: float) -> dict[str, object]:
        transition_metrics = {
            name: 1.0 for name in metric_schema["transition_metric_names"]
        }
        transition_metrics["reward"] = reward
        return {
            "episode_count": 0,
            "episode_metrics": {
                name: None for name in metric_schema["episode_metric_names"]
            },
            "p_cross_trace": _uniform_p_cross_trace(),
            "task_transitions": segment_delta * 2,
            "transition_metrics": transition_metrics,
        }

    rank_order = (
        "push_box",
        "move_suitcase",
        "move_largebox",
        "push_door_hand",
    )
    rewards = dict(zip(rank_order, (1.0e16, 1.0, -1.0e16, 1.0), strict=True))
    rank_order_records = {
        task: ordered_metric_record(rewards[task]) for task in rank_order
    }
    records_before = copy.deepcopy(rank_order_records)
    rank_pooled = pooled_macro_metrics(rank_order_records)["pooled"]
    assert rank_pooled["transition_mean"]["reward"] == 0.25
    canonical_records = train_phase6._canonical_task_metrics(
        rank_order_records, roster=roster
    )
    expected_order = tuple(task.task for task in roster.tasks)
    assert tuple(canonical_records) == expected_order
    assert rank_order_records == records_before
    for task in expected_order:
        assert canonical_records[task] is rank_order_records[task]
        assert canonical_records[task] == records_before[task]
    canonical_pooled = pooled_macro_metrics(canonical_records)["pooled"]
    assert canonical_pooled["transition_mean"]["reward"] == 0.0
    validate_phase6_joint_metrics(
        {"pooled": canonical_pooled, "task_records": canonical_records},
        config=config,
        roster=roster,
        expected_global_transitions=segment_delta * 2 * len(roster.tasks),
    )
    reversed_records = {
        task: ordered_metric_record(rewards[task]) for task in reversed(rank_order)
    }
    canonical_reversed = train_phase6._canonical_task_metrics(
        reversed_records, roster=roster
    )
    assert tuple(canonical_reversed) == expected_order
    assert pooled_macro_metrics(canonical_reversed)["pooled"] == canonical_pooled
    missing_records = dict(rank_order_records)
    missing_records.pop("push_box")
    with pytest.raises(ValueError, match="keys do not match"):
        train_phase6._canonical_task_metrics(missing_records, roster=roster)
    extra_records = {**rank_order_records, "unknown": ordered_metric_record(1.0)}
    with pytest.raises(ValueError, match="keys do not match"):
        train_phase6._canonical_task_metrics(extra_records, roster=roster)
    duplicate_roster = Namespace(tasks=(roster.tasks[0], roster.tasks[0]))
    with pytest.raises(ValueError, match="duplicate"):
        train_phase6._canonical_task_metrics(
            rank_order_records, roster=duplicate_roster
        )

    worker_source = inspect.getsource(train_phase6._worker_main)
    assert "evaluation_history=resume_evaluation_history" in worker_source
    assert "evaluation_history=[]" not in worker_source

    history_segment = plan_learning_segments(total_iterations=2442)[2]
    history_segment_dir = tmp_path / "history_segment"
    history_train_dir = history_segment_dir / "train"
    history_train_dir.mkdir(parents=True)
    history_resume = tmp_path / "history_resume.pt"
    history_pre = history_train_dir / "pre_evaluation.pt"
    history_source_rebind = tmp_path / "history_source_rebind.json"
    history_resume.write_bytes(b"history resume")
    history_pre.write_bytes(b"history pre")
    history_source_rebind.write_text("{}\n", encoding="utf-8")
    history = [
        {"iteration": 31, "logical_crossing": 250000, "scope": "pilot"},
        {"iteration": 62, "logical_crossing": 500000, "scope": "segment_0001"},
    ]
    history_metrics = {"status": "history-chain"}
    history_rank_results = [
        {
            "checkpoint": {
                "path": str(history_pre),
                "sha256": hashlib.sha256(history_pre.read_bytes()).hexdigest(),
            },
            "metrics": history_metrics,
            "optimizer_sha256": "history-state",
            "optimizer_step": history_segment.end_iteration * 24,
            "policy_sha256": "history-state",
            "curriculum": {"stage": "C2"},
            "schedule": {
                "cycle_index": history_segment.segment_index,
                "task": roster.tasks[rank].task,
            },
        }
        for rank in range(4)
    ]
    history_resume_payload = {
        "actual_global_transitions": history_segment.start_iteration * 8192,
        "actual_per_task_transitions": history_segment.start_iteration * 2048,
        "evaluation_history": history,
        "pre_evaluation": False,
        "curriculum": {"stage": "C2"},
    }
    history_pre_payload = {
        "actual_global_transitions": history_segment.end_iteration * 8192,
        "actual_per_task_transitions": history_segment.end_iteration * 2048,
        "evaluation_history": copy.deepcopy(history),
        "metrics": history_metrics,
        "optimizer_sha256": "history-state",
        "optimizer_step": history_segment.end_iteration * 24,
        "policy_sha256": "history-state",
        "pre_evaluation": True,
        "curriculum": {"stage": "C2"},
    }

    def history_restore(
        path: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        resolved = Path(path).resolve()
        assert kwargs["config"] is config
        assert kwargs["roster"] is roster
        assert kwargs["acceptance"] is acceptance
        assert kwargs["source_manifest"] == production_manifest
        assert kwargs["device"] == "cpu"
        if resolved == history_resume.resolve():
            assert kwargs["expected_iteration"] == history_segment.start_iteration
            return copy.deepcopy(history_resume_payload)
        assert resolved == history_pre.resolve()
        assert kwargs["expected_iteration"] == history_segment.end_iteration
        return copy.deepcopy(history_pre_payload)

    def history_read_json(path: Path) -> dict[str, object]:
        if path.name == "command.json" and path.parent.name.startswith("rank_"):
            command = [
                str(train_phase6.ISAAC_PYTHON),
                str(Path("scripts/train_phase6.py").resolve()),
                "--mode",
                "train-segment",
                "--output-dir",
                str(history_train_dir),
                "--profile",
                "production_segment",
                "--acceptance-config",
                str(train_phase6.PHASE6_LEARNING_CONFIG),
                "--phase6-config",
                str(PHASE6_CONFIG),
                "--roster",
                str(PHASE6_ROSTER),
                "--segment-start",
                str(history_segment.start_iteration),
                "--segment-end",
                str(history_segment.end_iteration),
                "--stage",
                "C2",
                "--resume",
                str(history_resume),
                "--source-rebind",
                str(history_source_rebind),
                "--headless",
            ]
            singleton_options = (
                "--acceptance-config",
                "--phase6-config",
                "--roster",
                "--mode",
                "--output-dir",
                "--profile",
                "--segment-start",
                "--segment-end",
                "--stage",
                "--resume",
                "--source-rebind",
            )
            for option in singleton_options:
                assert command.count(option) == 1
            return {
                "command": command,
                "local_rank": int(path.parent.name.split("_")[-1]),
                "rank": int(path.parent.name.split("_")[-1]),
                "world_size": 4,
            }
        if path == history_train_dir / "segment_summary.json":
            return {
                "checkpoint": {
                    "path": str(history_pre),
                    "sha256": hashlib.sha256(history_pre.read_bytes()).hexdigest(),
                },
                "iteration": history_segment.end_iteration,
                "rank_results": [dict(result) for result in history_rank_results],
                "source_manifest_sha256": train_phase6._source_manifest_digest(
                    production_manifest
                ),
                "source_rebind": {
                    "path": str(history_source_rebind),
                    "sha256": hashlib.sha256(
                        history_source_rebind.read_bytes()
                    ).hexdigest(),
                },
                "status": "ok",
                "task_metrics": history_metrics,
            }
        if path == history_train_dir / "production_evidence_summary.json":
            return {
                "iteration": history_segment.end_iteration,
                "rank_wrappers": [{"vmhwm_kib": 1} for _ in range(4)],
                "status": "ok",
                "topology": "per_rank_wrapper",
                "worker_mode": "train-segment",
            }
        return {}

    def history_command_path(_values: list[str], name: str) -> Path:
        paths = {
            "--output-dir": history_train_dir,
            "--resume": history_resume,
            "--source-rebind": history_source_rebind,
        }
        return paths[name].resolve()

    def history_command_argument(_values: list[str], name: str) -> str:
        if name == "--iteration":
            return str(history_segment.end_iteration)
        if name == "--stage":
            return "C2"
        raise AssertionError(name)

    with monkeypatch.context() as patched:
        import somaforce_cross.learning.joint_runner as joint_runner

        patched.setattr(joint_runner, "restore_learning_checkpoint", history_restore)
        patched.setattr(
            joint_runner, "state_dict_sha256", lambda _state: "history-state"
        )
        patched.setattr(
            joint_runner,
            "optimizer_step",
            lambda _optimizer: history_segment.end_iteration * 24,
        )
        patched.setattr(train_phase6, "_read_json", history_read_json)
        patched.setattr(
            train_phase6,
            "_validate_mainrunner_stage_evidence",
            lambda path, *, stage: (
                [
                    str(train_phase6.ISAAC_PYTHON),
                    str(Path("scripts/train_phase6.py").resolve()),
                    "--mode",
                    "production-summarize",
                    "--output-dir",
                    str(history_train_dir),
                    "--iteration",
                    str(history_segment.end_iteration),
                    "--stage",
                    "C2",
                    "--source-rebind",
                    str(history_source_rebind),
                ]
                if path.name == "summary.log"
                else [
                    str(train_phase6.ISAAC_PYTHON),
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nnodes=1",
                    "--nproc_per_node=4",
                    str(Path("scripts/train_phase6.py").resolve()),
                    "--mode",
                    "rank-wrapper",
                    "--worker-mode",
                    "train-segment",
                    "--output-dir",
                    str(history_train_dir),
                    "--profile",
                    "production_segment",
                    "--acceptance-config",
                    str(train_phase6.PHASE6_LEARNING_CONFIG),
                    "--phase6-config",
                    str(PHASE6_CONFIG),
                    "--roster",
                    str(PHASE6_ROSTER),
                    "--segment-start",
                    str(history_segment.start_iteration),
                    "--segment-end",
                    str(history_segment.end_iteration),
                    "--stage",
                    "C2",
                    "--resume",
                    str(history_resume),
                    "--source-rebind",
                    str(history_source_rebind),
                ]
            ),
        )
        patched.setattr(
            train_phase6,
            "_validate_summary_wrapper",
            lambda *_args, **_kwargs: {"vmhwm_kib": 1},
        )
        patched.setattr(
            train_phase6,
            "_validate_production_train_result",
            lambda _value, *, rank, **_kwargs: history_rank_results[rank],
        )
        train_phase6._validate_mainrunner_completed_train(
            segment_dir=history_segment_dir,
            segment=history_segment,
            resume_checkpoint=history_resume,
            source_rebind=history_source_rebind,
            config=config,
            roster=roster,
            acceptance=acceptance,
            source_manifest=production_manifest,
        )
        for invalid_history in (
            [],
            list(reversed(history)),
            [*history, copy.deepcopy(history[-1])],
        ):
            history_pre_payload["evaluation_history"] = invalid_history
            with pytest.raises(ValueError, match="resume evaluation history"):
                train_phase6._validate_mainrunner_completed_train(
                    segment_dir=history_segment_dir,
                    segment=history_segment,
                    resume_checkpoint=history_resume,
                    source_rebind=history_source_rebind,
                    config=config,
                    roster=roster,
                    acceptance=acceptance,
                    source_manifest=production_manifest,
                )
    bootstrap = tmp_path / "post_evaluation_mainrunner_rebound.pt"
    bootstrap.write_bytes(b"bootstrap")

    def fake_rebind(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "checkpoint": {"output_path": str(bootstrap)},
            "endpoint": {
                "history_length": 1,
                "iteration": 31,
                "optimizer_step": 744,
                "segment": 0,
                "stage": "C1",
                "transitions": 253952,
            },
        }

    def latest(segment: object) -> dict[str, object]:
        index = int(getattr(segment, "segment_index"))
        return {
            "actual_global_transitions": int(getattr(segment, "end_transitions")),
            "checkpoint": str(tmp_path / f"post_{index}.pt"),
            "checkpoint_sha256": f"{index:064x}",
            "curriculum_stage": "C1",
            "iteration": int(getattr(segment, "end_iteration")),
            "logical_crossing": int(getattr(segment, "crossing").logical_transitions),
            "segment": index,
            "source_manifest_sha256": "a" * 64,
        }

    with monkeypatch.context() as patched:
        patched.setattr(train_phase6, "_validate_mainrunner_rebind_record", fake_rebind)
        patched.setattr(
            train_phase6,
            "_validate_mainrunner_completed_segment",
            lambda *, segment, **_kwargs: latest(segment),
        )
        closed = tmp_path / "closed"
        (closed / "segment_0001").mkdir(parents=True)
        (closed / "latest.json").write_text(
            json.dumps(latest(plan_learning_segments(total_iterations=2442)[1])),
            encoding="utf-8",
        )
        train_phase6._validate_mainrunner_completed_segments(
            closed,
            current_iteration=62,
            target_iteration=2442,
            source_rebind=tmp_path / "source_rebind_mainrunner.json",
            config=config,
            roster=roster,
            acceptance=acceptance,
        )

        broken_history = tmp_path / "broken_history"
        (broken_history / "segment_0001").mkdir(parents=True)
        (broken_history / "segment_0002").mkdir(parents=True)
        (broken_history / "latest.json").write_text(
            json.dumps(latest(plan_learning_segments(total_iterations=2442)[2])),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="latest history"):
            train_phase6._validate_mainrunner_completed_segments(
                broken_history,
                current_iteration=92,
                target_iteration=2442,
                source_rebind=tmp_path / "source_rebind_mainrunner.json",
                config=config,
                roster=roster,
                acceptance=acceptance,
            )

    with monkeypatch.context() as patched:
        patched.setattr(train_phase6, "_validate_mainrunner_rebind_record", fake_rebind)
        missing_checkpoint = tmp_path / "missing_checkpoint"
        (missing_checkpoint / "segment_0001").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="train checkpoint"):
            train_phase6._validate_mainrunner_completed_segments(
                missing_checkpoint,
                current_iteration=62,
                target_iteration=2442,
                source_rebind=tmp_path / "source_rebind_mainrunner.json",
                config=config,
                roster=roster,
                acceptance=acceptance,
            )
        illegal_bootstrap = tmp_path / "illegal_bootstrap"
        (illegal_bootstrap / "segment_0000").mkdir(parents=True)
        with pytest.raises(ValueError, match="bootstrap"):
            train_phase6._validate_mainrunner_completed_segments(
                illegal_bootstrap,
                current_iteration=31,
                target_iteration=2442,
                source_rebind=tmp_path / "source_rebind_mainrunner.json",
                config=config,
                roster=roster,
                acceptance=acceptance,
            )

    external_bootstrap = tmp_path / "external_bootstrap.pt"
    external_bootstrap.write_bytes(b"external bootstrap")
    external_record = tmp_path / "external_source_rebind.json"
    external_record.write_text("{}\n", encoding="utf-8")

    def external_rebind(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "checkpoint": {"output_path": str(external_bootstrap)},
            "endpoint": {
                "history_length": 22,
                "iteration": 672,
                "optimizer_step": 16128,
                "segment": 21,
                "stage": "C2",
                "transitions": 5505024,
            },
        }

    def external_latest(segment: object) -> dict[str, object]:
        return {
            "actual_global_transitions": int(getattr(segment, "end_transitions")),
            "checkpoint": str(tmp_path / f"external_post_{segment.segment_index}.pt"),
            "checkpoint_sha256": f"{int(segment.segment_index):064x}",
            "curriculum_stage": "C2",
            "iteration": int(getattr(segment, "end_iteration")),
            "logical_crossing": int(getattr(segment, "crossing").logical_transitions),
            "segment": int(getattr(segment, "segment_index")),
            "source_manifest_sha256": "a" * 64,
        }

    with monkeypatch.context() as patched:
        patched.setattr(
            train_phase6, "_validate_mainrunner_rebind_record", external_rebind
        )
        empty_external = tmp_path / "empty_external"
        train_phase6._validate_mainrunner_completed_segments(
            empty_external,
            current_iteration=672,
            target_iteration=2442,
            source_rebind=external_record,
            config=config,
            roster=roster,
            acceptance=acceptance,
        )
        with pytest.raises(ValueError, match="precedes"):
            train_phase6._validate_mainrunner_completed_segments(
                tmp_path / "before_external",
                current_iteration=641,
                target_iteration=2442,
                source_rebind=external_record,
                config=config,
                roster=roster,
                acceptance=acceptance,
            )
        with pytest.raises(ValueError, match="segment boundary"):
            train_phase6._validate_mainrunner_completed_segments(
                tmp_path / "non_endpoint_before_external",
                current_iteration=671,
                target_iteration=2442,
                source_rebind=external_record,
                config=config,
                roster=roster,
                acceptance=acceptance,
            )
        before_external = tmp_path / "before_external_dir"
        (before_external / "segment_0021").mkdir(parents=True)
        with pytest.raises(ValueError, match="directory set"):
            train_phase6._validate_mainrunner_completed_segments(
                before_external,
                current_iteration=672,
                target_iteration=2442,
                source_rebind=external_record,
                config=config,
                roster=roster,
                acceptance=acceptance,
            )
        completed_external = tmp_path / "completed_external"
        (completed_external / "segment_0022").mkdir(parents=True)
        post = tmp_path / "external_post_22.pt"
        post.write_bytes(b"post")
        latest_22 = external_latest(plan_learning_segments(total_iterations=2442)[22])
        latest_22["checkpoint"] = str(post)
        (completed_external / "latest.json").write_text(
            json.dumps(latest_22), encoding="utf-8"
        )
        patched.setattr(
            train_phase6,
            "_validate_mainrunner_completed_segment",
            lambda *, segment, **_kwargs: latest_22,
        )
        train_phase6._validate_mainrunner_completed_segments(
            completed_external,
            current_iteration=702,
            target_iteration=2442,
            source_rebind=external_record,
            config=config,
            roster=roster,
            acceptance=acceptance,
        )

    stage_log = tmp_path / "finalize.log"
    stage_log.write_text("ok\n", encoding="utf-8")
    stage_log.with_suffix(".command.json").write_text(
        json.dumps({"command": [], "stage": "finalize-checkpoint", "timeout_s": 1}),
        encoding="utf-8",
    )
    stage_status = {
        "elapsed_s": 1.0,
        "exit_code": 0,
        "signal": None,
        "stage": "finalize-checkpoint",
        "timeout": False,
        "vmhwm_kib": 1,
    }
    stage_log.with_suffix(".status.json").write_text(
        json.dumps(stage_status), encoding="utf-8"
    )
    assert (
        train_phase6._validate_mainrunner_stage_evidence(
            stage_log, stage="finalize-checkpoint"
        )
        == []
    )
    stage_status["exit_code"] = 1
    stage_log.with_suffix(".status.json").write_text(
        json.dumps(stage_status), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="stage evidence"):
        train_phase6._validate_mainrunner_stage_evidence(
            stage_log, stage="finalize-checkpoint"
        )

    segment = plan_learning_segments(total_iterations=2442)[1]
    evaluation_dir = tmp_path / "evaluation"
    (evaluation_dir / "residual").mkdir(parents=True)
    with monkeypatch.context() as patched:
        patched.setattr(
            train_phase6,
            "_validate_mainrunner_stage_evidence",
            lambda *_args, **_kwargs: [],
        )
        patched.setattr(
            train_phase6,
            "_validate_mainrunner_worker_stage_command",
            lambda *_args, **_kwargs: None,
        )
        with pytest.raises(FileNotFoundError, match="scaffold_only"):
            train_phase6._validate_mainrunner_completed_evaluation(
                segment_dir=tmp_path,
                segment=segment,
                pre_checkpoint=bootstrap,
                stage="C1",
                source_rebind=tmp_path / "source_rebind_mainrunner.json",
                roster=roster,
                source_manifest=production_manifest,
            )

    (evaluation_dir / "scaffold_only").mkdir()
    (evaluation_dir / "evaluation_summary.json").write_text(
        json.dumps({"status": "tampered"}), encoding="utf-8"
    )

    def evaluation_stage_command(_log_path: Path, *, stage: str) -> list[str]:
        if stage != "evaluation-summary":
            return []
        return [
            str(train_phase6.ISAAC_PYTHON),
            str(Path(train_phase6.__file__).resolve()),
            "--mode",
            "mainrunner-evaluation-merge",
            "--evaluation-dir",
            str(evaluation_dir),
            "--pre-checkpoint",
            str(bootstrap),
            "--iteration",
            str(segment.end_iteration),
            "--stage",
            "C1",
            "--source-rebind",
            str(tmp_path / "source_rebind_mainrunner.json"),
        ]

    with monkeypatch.context() as patched:
        patched.setattr(
            train_phase6,
            "_validate_mainrunner_stage_evidence",
            evaluation_stage_command,
        )
        patched.setattr(
            train_phase6,
            "_validate_mainrunner_worker_stage_command",
            lambda *_args, **_kwargs: None,
        )
        patched.setattr(
            train_phase6,
            "_validate_mainrunner_evaluation_summary",
            lambda summary, **_kwargs: dict(summary),
        )
        patched.setattr(
            train_phase6,
            "_merge_mainrunner_evaluation",
            lambda **_kwargs: {"status": "expected"},
        )
        with pytest.raises(ValueError, match="summary SHA"):
            train_phase6._validate_mainrunner_completed_evaluation(
                segment_dir=tmp_path,
                segment=segment,
                pre_checkpoint=bootstrap,
                stage="C1",
                source_rebind=tmp_path / "source_rebind_mainrunner.json",
                roster=roster,
                source_manifest=production_manifest,
            )


def test_learning_checkpoint_resume_preserves_balance_hash_rng_and_stage(
    tmp_path: Path,
) -> None:
    config, roster = _roster()
    acceptance = load_learning_acceptance_config()
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    for _ in range(24):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(parameter.square().sum() for parameter in policy.parameters())
        loss.backward()
        optimizer.step()
    curriculum = JointCurriculum(stage="C1", transitions=8192)
    source_manifest = {"scripts/train_phase6.py": "a" * 64}
    payload = learning_checkpoint_payload(
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        iteration=1,
        task_transitions={task.task: 2048 for task in roster.tasks},
        curriculum=curriculum,
        next_crossing={
            "logical_transitions": 250000,
            "target_iteration": 31,
            "actual_transitions": 253952,
        },
        evaluation_history=[],
        rank_rng={
            str(rank): {
                "cpu": torch.arange(2, dtype=torch.uint8),
                "cuda": [torch.arange(2, dtype=torch.uint8) for _ in range(4)],
            }
            for rank in range(4)
        },
        source_manifest=source_manifest,
        metrics={"status": "pre_evaluation"},
        pre_evaluation=True,
    )
    path = tmp_path / "pre_evaluation.pt"
    atomic_torch_save(path, payload)
    restored_policy = ResidualActorCritic()
    restored_optimizer = torch.optim.Adam(restored_policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        path,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=restored_policy,
        optimizer=restored_optimizer,
        source_manifest=source_manifest,
        expected_iteration=1,
    )
    assert restored["pre_evaluation"] is True
    assert restored["curriculum"]["stage"] == "C1"
    assert restored["task_transitions"] == {task.task: 2048 for task in roster.tasks}
    assert restored["policy_sha256"] == state_dict_sha256(restored_policy.state_dict())
    assert restored["optimizer_sha256"] == state_dict_sha256(
        restored_optimizer.state_dict()
    )
    with pytest.raises(ValueError):
        restore_learning_checkpoint(
            path,
            config=config,
            roster=roster,
            acceptance=acceptance,
            policy=ResidualActorCritic(),
            optimizer=torch.optim.Adam(ResidualActorCritic().parameters(), lr=3.0e-4),
            source_manifest={"scripts/train_phase6.py": "b" * 64},
            expected_iteration=1,
        )
    devicefix_snapshot = (
        Path("outputs/phase6_learning_acceptance")
        / "phase6-learning-pilot-4gpu-64env-31iter-20260806_042940"
        / "segment_0000/recovery/source_snapshot/train_phase6_pre_devicefix.py"
    )
    stepboundary_snapshot = (
        Path("outputs/phase6_learning_acceptance")
        / "phase6-learning-pilot-4gpu-64env-31iter-20260806_042940"
        / "segment_0000/recovery/source_snapshot/train_phase6_pre_stepboundaryfix.py"
    )
    pairedbatch_snapshot = (
        Path("outputs/phase6_learning_acceptance")
        / "phase6-learning-pilot-4gpu-64env-31iter-20260806_042940"
        / "segment_0000/recovery/source_snapshot/train_phase6_pre_pairedbatchfix.py"
    )
    recovery = pairedbatch_snapshot.parents[1]
    current = recovery / "source_snapshot/train_phase6_pre_mainrunner.py"
    stepboundary_ast = _ast_function_dumps(
        stepboundary_snapshot, WORKER_CRITICAL_FUNCTIONS
    )
    pairedbatch_ast = _ast_function_dumps(
        pairedbatch_snapshot, WORKER_CRITICAL_FUNCTIONS
    )
    current_ast = _ast_function_dumps(current, WORKER_CRITICAL_FUNCTIONS)
    devicefix_proof = _evaluation_devicefix_ast_proof(
        devicefix_snapshot, stepboundary_snapshot
    )
    assert devicefix_proof["non_device_ast_equal"] is True
    assert devicefix_proof["load_follows_device_move"] is True
    assert all(devicefix_proof["other_worker_critical_ast_equality"].values())
    proof = _stepboundary_ast_proof(stepboundary_snapshot, pairedbatch_snapshot)
    assert proof["devicefix_policy_preserved"] is True
    assert proof["inference_mode_context_count"] == 0
    assert proof["no_grad_context_count"] == 1
    assert proof["no_grad_body_is_policy_forward_only"] is True
    assert proof["environment_reset_outside_torch_context"] is True
    assert proof["environment_step_outside_torch_context"] is True
    assert all(proof["other_worker_critical_ast_equality"].values())
    assert (
        stepboundary_ast["_run_paired_evaluation_rank"]
        != pairedbatch_ast["_run_paired_evaluation_rank"]
    )
    assert (
        pairedbatch_ast["_run_paired_evaluation_rank"]
        != current_ast["_run_paired_evaluation_rank"]
    )
    pairedbatch_proof = _pairedbatch_ast_proof_for_sources(
        pairedbatch_snapshot,
        recovery / "source_snapshot/joint_runner_pre_pairedbatchfix.py",
        recovery / "source_snapshot/residual_env_pre_pairedbatchfix.py",
        candidate_train=current,
        candidate_joint=recovery / "source_snapshot/joint_runner_pre_mainrunner.py",
        candidate_env=recovery / "source_snapshot/residual_env_pre_mainrunner.py",
    )
    assert pairedbatch_proof["approved_changed_paths"] == [
        str(current),
        str(recovery / "source_snapshot/joint_runner_pre_mainrunner.py"),
        str(recovery / "source_snapshot/residual_env_pre_mainrunner.py"),
    ]
    mutated = tmp_path / "worker_critical_mutation.py"
    mutated.write_text(
        current.read_text(encoding="utf-8").replace(
            'status = "error"', 'status = "error"\n    _rebind_mutation = True', 1
        ),
        encoding="utf-8",
    )
    assert (
        _ast_function_dumps(mutated, WORKER_CRITICAL_FUNCTIONS)["_worker_main"]
        != current_ast["_worker_main"]
    )
    episodeorder_train = (
        recovery / "source_snapshot/train_phase6_pre_episodeorderfix.py"
    )
    episodeorder_joint = (
        recovery / "source_snapshot/joint_runner_pre_episodeorderfix.py"
    )
    episodeorder_env = recovery / "source_snapshot/residual_env_pre_episodeorderfix.py"
    modeprocess_train = recovery / "source_snapshot/train_phase6_pre_modeprocessfix.py"
    modeprocess_joint = recovery / "source_snapshot/joint_runner_pre_modeprocessfix.py"
    modeprocess_env = recovery / "source_snapshot/residual_env_pre_modeprocessfix.py"
    warningpolicy_train = (
        recovery / "source_snapshot/train_phase6_pre_warningpolicyfix.py"
    )
    warningpolicy_joint = (
        recovery / "source_snapshot/joint_runner_pre_warningpolicyfix.py"
    )
    warningpolicy_env = (
        recovery / "source_snapshot/residual_env_pre_warningpolicyfix.py"
    )
    episodeorder_proof = _episodeorder_ast_proof_for_sources(
        episodeorder_train,
        episodeorder_joint,
        episodeorder_env,
        candidate_train=modeprocess_train,
        candidate_joint=modeprocess_joint,
        candidate_env=modeprocess_env,
    )
    assert episodeorder_proof["joint_runner_byte_equality"] is True
    assert episodeorder_proof["record_before_next_slot_commit"] is True
    assert episodeorder_proof["record_before_reset_metrics"] is True
    assert episodeorder_proof["reset_plan_validation_precedes_record"] is True
    assert episodeorder_proof["applied_readback_validation_follows_record"] is True
    assert all(episodeorder_proof["worker_critical_ast_equality"].values())
    modeprocess_proof = _modeprocess_ast_proof(
        modeprocess_train,
        modeprocess_joint,
        modeprocess_env,
        candidate_train=warningpolicy_train,
        candidate_joint=warningpolicy_joint,
        candidate_env=warningpolicy_env,
    )
    assert modeprocess_proof["evaluator_build_environment_count"] == 1
    assert modeprocess_proof["evaluator_close_call_count"] == 0
    assert modeprocess_proof["evaluator_preserves_progress_parameter"] is True
    warningpolicy_proof = _warningpolicy_ast_proof(
        warningpolicy_train,
        warningpolicy_joint,
        warningpolicy_env,
        candidate_train=recovery
        / "source_snapshot/train_phase6_pre_summaryschemafix.py",
        candidate_joint=recovery
        / "source_snapshot/joint_runner_pre_summaryschemafix.py",
        candidate_env=recovery / "source_snapshot/residual_env_pre_summaryschemafix.py",
    )
    assert warningpolicy_proof["modeprocess_wrapper_recounts_worker_log"] is True
    assert warningpolicy_proof["only_multiple_vulkan_icd_is_fatal"] is True
    assert all(warningpolicy_proof["preserved_function_ast"].values())
    assert all(warningpolicy_proof["worker_critical_ast_equality"].values())
    assert "_run_harness_rebind" in HARNESS_REBIND_FUNCTIONS
    assert _semantic_equal(payload["policy"], restored["policy"])
    assert _semantic_equal(payload["optimizer"], restored["optimizer"])
    assert _semantic_equal(payload["rank_rng"], restored["rank_rng"])
    assert _semantic_equal(payload["task_transitions"], restored["task_transitions"])


def test_evaluation_pair_plan_is_balanced_seed_paired_and_actor_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, roster = _roster()
    del config
    plan = paired_evaluation_plan(roster, stage="C1")
    validate_paired_evaluation_plan(plan, roster=roster)
    assert plan["modes"] == ("residual", "scaffold_only")
    assert set(plan["tasks"]) == {task.task for task in roster.tasks}
    for task in roster.tasks:
        record = plan["tasks"][task.task]
        assert len(record["stage_pairs"]) == 256
        assert len(record["nominal_pairs"]) == 128
        assert all(
            pair.residual_mode == "residual" and pair.scaffold_mode == "scaffold_only"
            for pair in record["stage_pairs"]
        )
        assert len({pair.seed for pair in record["stage_pairs"]}) == 256
    broken = copy.deepcopy(plan)
    broken["tasks"]["push_box"]["stage_pairs"] = broken["tasks"]["push_door_hand"][
        "stage_pairs"
    ]
    with pytest.raises(ValueError):
        validate_paired_evaluation_plan(broken, roster=roster)
    source = inspect.getsource(_run_paired_evaluation_rank)
    assert "ResidualActorCritic(init_noise_std=init_noise_std).to(" in source
    assert (
        source.index("ResidualActorCritic(init_noise_std=init_noise_std).to(")
        < source.index("restored = restore_learning_checkpoint")
        < source.index("policy.act_inference")
    )
    tree = ast.parse(source)
    contexts = [node for node in ast.walk(tree) if isinstance(node, ast.With)]
    assert not any(
        isinstance(item.context_expr, ast.Call)
        and isinstance(item.context_expr.func, ast.Attribute)
        and isinstance(item.context_expr.func.value, ast.Name)
        and item.context_expr.func.value.id == "torch"
        and item.context_expr.func.attr == "inference_mode"
        for node in contexts
        for item in node.items
    )
    no_grad = [
        node
        for node in contexts
        if len(node.items) == 1
        and isinstance(node.items[0].context_expr, ast.Call)
        and isinstance(node.items[0].context_expr.func, ast.Attribute)
        and isinstance(node.items[0].context_expr.func.value, ast.Name)
        and node.items[0].context_expr.func.value.id == "torch"
        and node.items[0].context_expr.func.attr == "no_grad"
    ]
    assert len(no_grad) == 1 and len(no_grad[0].body) == 1
    assert "policy.act_inference(observations)" in ast.unparse(no_grad[0].body[0])

    class _FakePolicy:
        def to(self, *, device: torch.device) -> _FakePolicy:
            del device
            return self

        def load_state_dict(self, state: object, *, strict: bool) -> None:
            del state, strict

        def eval(self) -> _FakePolicy:
            return self

        def act_inference(self, observations: object) -> torch.Tensor:
            del observations
            return torch.zeros(1, 23)

    lifecycle: list[tuple[str, bool]] = []

    class _FakeEnvironment:
        def __init__(self) -> None:
            self.mode = ""
            self.schedule: dict[str, object] | None = None
            self.position = 0
            self.steps = 0
            self.pending: list[dict[str, object]] = []

        def evaluation_nominal_probe(self, _env_id: int, _episode_index: int) -> bool:
            return True

        def bind_evaluation_schedule(
            self, schedule: dict[str, object], *, mode: str
        ) -> None:
            self.mode = mode
            self.schedule = schedule

        def reset(self) -> tuple[dict[str, torch.Tensor], object]:
            lifecycle.append(("reset", torch.is_inference_mode_enabled()))
            return {"policy": torch.zeros(1, 668)}, None

        def step(
            self, actions: torch.Tensor
        ) -> tuple[dict[str, torch.Tensor], object, object, object, object]:
            del actions
            lifecycle.append(("step", torch.is_inference_mode_enabled()))
            self.steps += 1
            if self.steps % 2 == 0:
                assert self.schedule is not None
                slot = self.schedule["rows"]["0"][self.position]
                self.pending.append(
                    {"mode": self.mode, "seed": slot["seed"], "subset": slot["subset"]}
                )
                self.position += 1
            return {"policy": torch.zeros(1, 668)}, None, None, None, None

        def drain_evaluation_completions(self) -> tuple[dict[str, object], ...]:
            records = tuple(self.pending)
            self.pending.clear()
            return records

        def close(self) -> None:
            return None

    config, roster = _roster()
    checkpoint = Path(
        "outputs/phase6_learning_acceptance/"
        "phase6-learning-pilot-4gpu-64env-31iter-20260806_042940/"
        "segment_0000/recovery/pre_evaluation_devicefix_rebound.pt"
    )
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    monkeypatch.setattr(
        "somaforce_cross.learning.joint_runner.paired_evaluation_plan",
        lambda _roster, **_kwargs: {
            "tasks": {
                roster.tasks[0].task: {
                    "horizon": 2,
                    "nominal_pairs": (),
                    "stage_pairs": (object(),),
                }
            }
        },
    )
    monkeypatch.setattr(
        "scripts.train_phase6._build_environment",
        lambda *_args, **_kwargs: _FakeEnvironment(),
    )
    results = []
    for mode in ("residual", "scaffold_only"):
        result, environment = _run_paired_evaluation_rank(
            Namespace(
                device="cpu",
                evaluation_seed=20262806,
                paired_mode=mode,
                paired_nominal_quota=1,
                paired_num_envs=1,
                paired_stage_quota=1,
                resume=checkpoint,
                runtime_mode=mode,
                stage="C1",
            ),
            config=config,
            roster=roster,
            rank=0,
            world_size=4,
            acceptance=load_learning_acceptance_config(
                train_phase6.PHASE6_V1_LEARNING_CONFIG
            ),
            source_manifest=checkpoint_payload["source_manifest"],
        )
        environment.close()
        results.append(result)
    assert all(
        result["optimizer_steps"] == result["normalizer_updates"] == 0
        for result in results
    )
    assert all(result["episodes"] == 2 for result in results)
    assert all(
        result["subset_counts"] == {"stage": 1, "nominal": 1} for result in results
    )
    assert results[0]["schedule_sha256"] == results[1]["schedule_sha256"]
    assert results[0]["completed_seed_sha256"] == results[1]["completed_seed_sha256"]
    assert lifecycle == [
        ("reset", False),
        ("step", False),
        ("step", False),
        ("step", False),
        ("step", False),
        ("reset", False),
        ("step", False),
        ("step", False),
        ("step", False),
        ("step", False),
    ]
    if not torch.cuda.is_available():
        pytest.skip("single-GPU CUDA inference probe requires CUDA")
    device = torch.device("cuda", 0)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    policy = ResidualActorCritic().to(device=device)
    policy.load_state_dict(payload["policy"], strict=True)
    policy.eval()
    before = {
        name: value.detach().clone()
        for name, value in [*policy.named_parameters(), *policy.named_buffers()]
    }
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    with torch.inference_mode():
        action = policy.act_inference({"policy": torch.zeros(1, 668, device=device)})
    assert action.device == device and torch.isfinite(action).all()
    assert all(value.device == device for value in policy.parameters())
    assert optimizer_step(optimizer) == 0
    assert all(
        torch.equal(value, before[name])
        for name, value in [*policy.named_parameters(), *policy.named_buffers()]
    )


def test_pairedbatch_schedule_quota_seed_and_canonical_hash() -> None:
    schedule = paired_evaluation_schedule(
        task="push_box",
        stage="C1",
        evaluation_seed=20262806,
        rank=2,
        num_envs=2,
        stage_quota=2,
        nominal_quota=2,
        nominal_probe=lambda _env_id, _episode_index: True,
    )
    validate_paired_evaluation_schedule(schedule)
    assert schedule["task_base"] == 2_020_262_806
    assert (
        schedule["schedule_sha256"]
        == paired_evaluation_schedule(
            task="push_box",
            stage="C1",
            evaluation_seed=20262806,
            rank=2,
            num_envs=2,
            stage_quota=2,
            nominal_quota=2,
            nominal_probe=lambda _env_id, _episode_index: True,
        )["schedule_sha256"]
    )
    for env_id, row in schedule["rows"].items():
        assert len(row) == 2
        assert [slot["subset"] for slot in row] == ["stage", "nominal"]
        assert row[0]["seed"] == schedule["task_base"] + int(env_id)
        assert row[1]["seed"] == schedule["task_base"] + int(env_id) + 1_000_003
    mutated = copy.deepcopy(schedule)
    mutated["rows"]["0"][1]["seed"] += 1
    with pytest.raises(ValueError, match="hash|seed"):
        validate_paired_evaluation_schedule(mutated)


def test_pairedbatch_nominal_slots_require_sampler_authenticity() -> None:
    with pytest.raises(ValueError, match="nominal sampler"):
        paired_evaluation_schedule(
            task="push_box",
            stage="C1",
            evaluation_seed=1,
            rank=0,
            num_envs=2,
            stage_quota=2,
            nominal_quota=2,
            nominal_probe=lambda _env_id, _episode_index: False,
        )
    schedule = paired_evaluation_schedule(
        task="push_box",
        stage="C1",
        evaluation_seed=1,
        rank=0,
        num_envs=2,
        stage_quota=2,
        nominal_quota=2,
        nominal_probe=lambda _env_id, _episode_index: True,
    )
    broken = copy.deepcopy(schedule)
    broken["rows"]["1"][1]["nominal"] = False
    broken["schedule_sha256"] = canonical_sha256(
        {
            key: broken[key]
            for key in (
                "evaluation_seed",
                "num_envs",
                "nominal_quota",
                "rank",
                "stage",
                "stage_quota",
                "task",
                "task_base",
                "rows",
            )
        }
    )
    with pytest.raises(ValueError, match="sampler-proven"):
        validate_paired_evaluation_schedule(broken)


def test_pairedbatch_rank_seed_matches_phase4_sampler_for_all_ranks() -> None:
    evaluation_seed = 20262806
    for rank in range(4):
        task_base = paired_evaluation_rank_seed(evaluation_seed, rank)
        assert task_base == evaluation_seed + rank * 1_000_000_000
        schedule = paired_evaluation_schedule(
            task="push_box",
            stage="C1",
            evaluation_seed=evaluation_seed,
            rank=rank,
            num_envs=2,
            stage_quota=2,
            nominal_quota=2,
            nominal_probe=lambda _env_id, _episode_index: True,
        )
        assert schedule["task_base"] == task_base
        for env_id, row in schedule["rows"].items():
            for slot in row:
                assert (
                    slot["seed"]
                    == task_base + int(env_id) + 1_000_003 * slot["episode_index"]
                )


def test_pairedbatch_rank_seed_rejects_invalid_inputs() -> None:
    invalid = (
        (True, 0),
        (0, True),
        (-1, 0),
        (0, -1),
        (1.0, 0),
        (0, "1"),
    )
    for evaluation_seed, rank in invalid:
        with pytest.raises(ValueError, match="nonnegative integer"):
            paired_evaluation_rank_seed(evaluation_seed, rank)  # type: ignore[arg-type]


def test_rankseed_failed_smoke_manifest_is_immutable(tmp_path: Path) -> None:
    recovery = Path(
        "outputs/phase6_learning_acceptance/"
        "phase6-learning-pilot-4gpu-64env-31iter-20260806_042940/segment_0000/recovery"
    )
    manifest = recovery / "evaluation_pairedbatch_smoke_0003_frozen_manifest.json"
    frozen = _validate_rankseed_failed_smoke_manifest(manifest)
    assert frozen["status"] == "failed_before_episode"
    assert len(frozen["files"]) == 20
    assert frozen["outer_log"]["sha256"] == (
        "cdaef96782d5273b6f441ff1e3e02e6df42c77a40d9f00127873866d6b5addeb"
    )
    mutated = copy.deepcopy(frozen)
    mutated["files"][0]["sha256"] = "0" * 64
    candidate = tmp_path / "mutated_frozen_manifest.json"
    candidate.write_text(json.dumps(mutated, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence changed"):
        _validate_rankseed_failed_smoke_manifest(candidate)


def test_episodeorder_failed_smoke_manifest_is_immutable(tmp_path: Path) -> None:
    recovery = Path(
        "outputs/phase6_learning_acceptance/"
        "phase6-learning-pilot-4gpu-64env-31iter-20260806_042940/segment_0000/recovery"
    )
    manifest = recovery / "evaluation_rankseed_smoke_0004_frozen_manifest.json"
    frozen = _validate_episodeorder_failed_smoke_manifest(manifest)
    assert frozen["status"] == "failed_after_partial_episodes"
    assert len(frozen["files"]) == 17
    assert frozen["outer_log"]["sha256"] == (
        "e348c4c5d9b0ea9a291ac0d6e8cfc72008a00a28e539bcaa65e651b964a6c3c9"
    )
    mutated = copy.deepcopy(frozen)
    mutated["files"][0]["mtime_ns"] += 1
    candidate = tmp_path / "mutated_episodeorder_manifest.json"
    candidate.write_text(json.dumps(mutated, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence changed"):
        _validate_episodeorder_failed_smoke_manifest(candidate)


def _write_modeprocess_evidence(
    attempt: Path, *, stage_quota: int = 2, nominal_quota: int = 2
) -> tuple[object, dict[tuple[str, int], Path]]:
    _, roster = _roster()
    paths: dict[tuple[str, int], Path] = {}
    expected = stage_quota + nominal_quota
    for mode in ("residual", "scaffold_only"):
        for rank, task in enumerate(roster.tasks):
            rank_dir = attempt / mode / f"rank_{rank}"
            rank_dir.mkdir(parents=True)
            result = {
                "completed_seed_sha256": f"{rank + 1:064x}",
                "control_steps": 10 + rank,
                "deterministic_actor_mean": True,
                "episodes": expected,
                "horizon": 1,
                "mode": mode,
                "mode_counts": {
                    "residual": expected if mode == "residual" else 0,
                    "scaffold_only": expected if mode == "scaffold_only" else 0,
                },
                "normalizer_updates": 0,
                "optimizer_steps": 0,
                "rank": rank,
                "schedule_sha256": f"{rank + 11:064x}",
                "status": "ok",
                "subset_counts": {"stage": stage_quota, "nominal": nominal_quota},
                "task": task.task,
                "unique_seeds": expected,
                "world_size": 4,
            }
            (rank_dir / "result.json").write_text(
                json.dumps(result, sort_keys=True), encoding="utf-8"
            )
            (rank_dir / "worker.log").write_text(
                "JSON_WRITTEN\nPHASE6_JOINT_RESULT="
                + json.dumps(result, sort_keys=True)
                + "\nENV_CLOSED\n",
                encoding="utf-8",
            )
            (rank_dir / "wrapper.json").write_text(
                json.dumps(
                    {
                        "close_verification": "parent_verified_exit0",
                        "command": ["python", "--paired-mode", mode],
                        "elapsed_s": 1.0,
                        "exit_code": 0,
                        "last_progress_marker": "ENV_CLOSED",
                        "local_rank": rank,
                        "log": str(rank_dir / "worker.log"),
                        "marker_order_valid": True,
                        "markers": [
                            "JSON_WRITTEN",
                            "PHASE6_JOINT_RESULT",
                            "ENV_CLOSED",
                        ],
                        "passed": True,
                        "rank": rank,
                        "result_error": None,
                        "result_json": str(rank_dir / "result.json"),
                        "result_status": "ok",
                        "signal": None,
                        "timed_out": False,
                        "timeout_kind": None,
                        "vmhwm_kib": 1,
                        "warning_counts": {
                            "headless_glfw": 0,
                            "kvdb_lock": 0,
                            "multiple_installable_client_drivers": 0,
                        },
                        "world_size": 4,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            paths[(mode, rank)] = rank_dir
    return roster, paths


def _write_modeprocess_warning(
    rank_dir: Path, *, headless_glfw: int, kvdb_lock: int, vulkan_icd: int
) -> None:
    assert kvdb_lock % 2 == 0
    warnings = {
        "headless_glfw": headless_glfw,
        "kvdb_lock": kvdb_lock,
        "multiple_installable_client_drivers": vulkan_icd,
    }
    wrapper_path = rank_dir / "wrapper.json"
    wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    wrapper["warning_counts"] = warnings
    wrapper_path.write_text(json.dumps(wrapper, sort_keys=True), encoding="utf-8")
    result = json.loads((rank_dir / "result.json").read_text(encoding="utf-8"))
    log = "GLFW\n" * headless_glfw
    log += "kvdb lock\n" * (kvdb_lock // 2)
    log += "Multiple Installable Client Drivers\n" * vulkan_icd
    log += "JSON_WRITTEN\nPHASE6_JOINT_RESULT="
    log += json.dumps(result, sort_keys=True) + "\nENV_CLOSED\n"
    (rank_dir / "worker.log").write_text(log, encoding="utf-8")


def test_paired_evaluation_runs_one_mode_per_isaac_process() -> None:
    recovery = Path(
        "outputs/phase6_learning_acceptance/"
        "phase6-learning-pilot-4gpu-64env-31iter-20260806_042940/segment_0000/recovery"
    )
    candidate_train = recovery / "source_snapshot/train_phase6_pre_summaryschemafix.py"
    candidate_joint = recovery / "source_snapshot/joint_runner_pre_summaryschemafix.py"
    candidate_env = recovery / "source_snapshot/residual_env_pre_summaryschemafix.py"
    proof = _modeprocess_ast_proof(
        recovery / "source_snapshot/train_phase6_pre_modeprocessfix.py",
        recovery / "source_snapshot/joint_runner_pre_modeprocessfix.py",
        recovery / "source_snapshot/residual_env_pre_modeprocessfix.py",
        candidate_train=candidate_train,
        candidate_joint=candidate_joint,
        candidate_env=candidate_env,
    )
    assert proof["evaluator_build_environment_count"] == 1
    assert proof["evaluator_close_call_count"] == 0
    assert proof["evaluator_reads_required_paired_mode"] is True
    assert proof["evaluator_rejects_invalid_paired_mode"] is True
    assert proof["evaluator_preserves_progress_parameter"] is True
    assert proof["worker_cleanup_has_barriers"] is True
    source = Path("scripts/train_phase6.py").read_text(encoding="utf-8")
    mutations = (
        source.replace(
            'getattr(args, "paired_mode", None)', 'getattr(args, "wrong", None)', 1
        ),
        source.replace(
            'getattr(args, "paired_mode", None)',
            'getattr(args, "paired_mode", "residual")',
            1,
        ),
        source.replace(
            'if mode not in {"residual", "scaffold_only"}:',
            "if False:",
            1,
        ),
        source.replace(
            'mode = getattr(args, "paired_mode", None)',
            'mode = getattr(args, "paired_mode", None)\n    progress = progress',
            1,
        ),
    )
    for index, mutated_source in enumerate(mutations):
        candidate = Path.cwd() / f".modeprocess_proof_mutation_{index}.py"
        candidate.write_text(mutated_source, encoding="utf-8")
        try:
            with pytest.raises(ValueError):
                _modeprocess_ast_proof(
                    recovery / "source_snapshot/train_phase6_pre_modeprocessfix.py",
                    recovery / "source_snapshot/joint_runner_pre_modeprocessfix.py",
                    recovery / "source_snapshot/residual_env_pre_modeprocessfix.py",
                    candidate_train=candidate,
                    candidate_joint=candidate_joint,
                    candidate_env=candidate_env,
                )
        finally:
            candidate.unlink()


def test_split_mode_evaluation_merge_requires_eight_clean_wrappers(
    tmp_path: Path,
) -> None:
    roster, paths = _write_modeprocess_evidence(tmp_path)
    summary = _merge_modeprocess_evaluation(
        attempt=tmp_path, roster=roster, stage_quota=2, nominal_quota=2
    )
    assert summary["status"] == "ok"
    assert len(summary["rank_results"]) == 4
    wrapper = json.loads(
        (paths[("scaffold_only", 3)] / "wrapper.json").read_text(encoding="utf-8")
    )
    wrapper["timed_out"] = True
    (paths[("scaffold_only", 3)] / "wrapper.json").write_text(
        json.dumps(wrapper, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="timeout evidence"):
        _merge_modeprocess_evaluation(
            attempt=tmp_path, roster=roster, stage_quota=2, nominal_quota=2
        )


def test_split_mode_merge_rejects_schedule_seed_or_quota_mismatch(
    tmp_path: Path,
) -> None:
    roster, paths = _write_modeprocess_evidence(tmp_path / "schedule")
    rank_dir = paths[("scaffold_only", 0)]
    result = json.loads((rank_dir / "result.json").read_text(encoding="utf-8"))
    result["schedule_sha256"] = "f" * 64
    (rank_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (rank_dir / "worker.log").write_text(
        "JSON_WRITTEN\nPHASE6_JOINT_RESULT="
        + json.dumps(result, sort_keys=True)
        + "\nENV_CLOSED\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="schedule or seeds"):
        _merge_modeprocess_evaluation(
            attempt=tmp_path / "schedule", roster=roster, stage_quota=2, nominal_quota=2
        )
    roster, paths = _write_modeprocess_evidence(tmp_path / "quota")
    rank_dir = paths[("residual", 1)]
    result = json.loads((rank_dir / "result.json").read_text(encoding="utf-8"))
    result["subset_counts"] = {"stage": 3, "nominal": 1}
    (rank_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (rank_dir / "worker.log").write_text(
        "JSON_WRITTEN\nPHASE6_JOINT_RESULT="
        + json.dumps(result, sort_keys=True)
        + "\nENV_CLOSED\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="single-mode result"):
        _merge_modeprocess_evaluation(
            attempt=tmp_path / "quota", roster=roster, stage_quota=2, nominal_quota=2
        )


def test_split_mode_progress_aggregates_both_modes_atomically(tmp_path: Path) -> None:
    for mode, status, completed in (
        ("residual", "complete", 16),
        ("scaffold_only", "running", 2),
    ):
        mode_dir = tmp_path / mode
        mode_dir.mkdir()
        (mode_dir / "progress.json").write_text(
            json.dumps(
                {
                    "status": status,
                    "completed_episodes": completed,
                    "total_episodes": 16,
                    "eta_s": 3.0,
                    "rank_progress": [
                        {"optimizer_steps": 0, "normalizer_updates": 0}
                        for _ in range(4)
                    ],
                }
            ),
            encoding="utf-8",
        )
    _aggregate_modeprocess_progress(tmp_path)
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["status"] == "running"
    assert progress["active_mode"] == "scaffold_only"
    assert progress["completed_episodes"] == 18
    assert progress["total_episodes"] == 32
    assert progress["optimizer_steps"] == progress["normalizer_updates"] == 0


def test_modeprocess_warning_policy_accepts_recorded_nonblocking_warnings(
    tmp_path: Path,
) -> None:
    roster, paths = _write_modeprocess_evidence(tmp_path)
    expected = {
        0: (0, 0),
        1: (0, 2),
        2: (0, 2),
        3: (7, 2),
    }
    for mode in ("residual", "scaffold_only"):
        for rank, (headless_glfw, kvdb_lock) in expected.items():
            _write_modeprocess_warning(
                paths[(mode, rank)],
                headless_glfw=headless_glfw,
                kvdb_lock=kvdb_lock,
                vulkan_icd=0,
            )
    summary = _merge_modeprocess_evaluation(
        attempt=tmp_path, roster=roster, stage_quota=2, nominal_quota=2
    )
    assert summary["status"] == "ok"


def test_modeprocess_warning_policy_rejects_vulkan_icd_and_invalid_counts(
    tmp_path: Path,
) -> None:
    roster, paths = _write_modeprocess_evidence(tmp_path / "vulkan")
    _write_modeprocess_warning(
        paths[("residual", 0)], headless_glfw=0, kvdb_lock=0, vulkan_icd=1
    )
    with pytest.raises(RuntimeError, match="multiple Vulkan"):
        _merge_modeprocess_evaluation(
            attempt=tmp_path / "vulkan", roster=roster, stage_quota=2, nominal_quota=2
        )
    roster, paths = _write_modeprocess_evidence(tmp_path / "invalid")
    wrapper_path = paths[("residual", 1)] / "wrapper.json"
    wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    wrapper["warning_counts"]["headless_glfw"] = True
    wrapper_path.write_text(json.dumps(wrapper, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="warning counts are invalid"):
        _merge_modeprocess_evaluation(
            attempt=tmp_path / "invalid", roster=roster, stage_quota=2, nominal_quota=2
        )
    roster, paths = _write_modeprocess_evidence(tmp_path / "mismatch")
    wrapper_path = paths[("scaffold_only", 2)] / "wrapper.json"
    wrapper = json.loads(wrapper_path.read_text(encoding="utf-8"))
    wrapper["warning_counts"]["kvdb_lock"] = 2
    wrapper_path.write_text(json.dumps(wrapper, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="do not match worker log"):
        _merge_modeprocess_evaluation(
            attempt=tmp_path / "mismatch", roster=roster, stage_quota=2, nominal_quota=2
        )


def test_modeprocess_warningpolicy_failed_smoke_manifest_is_immutable(
    tmp_path: Path,
) -> None:
    recovery = Path(
        "outputs/phase6_learning_acceptance/"
        "phase6-learning-pilot-4gpu-64env-31iter-20260806_042940/segment_0000/recovery"
    )
    manifest = recovery / "evaluation_modeprocess_smoke_0006_frozen_manifest.json"
    frozen = _validate_warningpolicy_failed_smoke_manifest(manifest)
    assert frozen["status"] == "business_complete_merge_rejected_warning_policy"
    assert len(frozen["files"]) == 47
    assert frozen["outer_log"]["sha256"] == (
        "0d27848f1dbdac1abc779a0f099c841725fec9d5d28e13f60c2e98b086ad9c48"
    )
    mutated = copy.deepcopy(frozen)
    mutated["files"][0]["mtime_ns"] += 1
    candidate = tmp_path / "mutated_warningpolicy_manifest.json"
    candidate.write_text(json.dumps(mutated, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence changed"):
        _validate_warningpolicy_failed_smoke_manifest(candidate)


def test_warningpolicy_rebind_preserves_checkpoint_and_updates_only_train_source(
    tmp_path: Path,
) -> None:
    recovery = Path(
        "outputs/phase6_learning_acceptance/"
        "phase6-learning-pilot-4gpu-64env-31iter-20260806_042940/segment_0000/recovery"
    )
    proof = _warningpolicy_ast_proof(
        recovery / "source_snapshot/train_phase6_pre_warningpolicyfix.py",
        recovery / "source_snapshot/joint_runner_pre_warningpolicyfix.py",
        recovery / "source_snapshot/residual_env_pre_warningpolicyfix.py",
        candidate_train=recovery
        / "source_snapshot/train_phase6_pre_summaryschemafix.py",
        candidate_joint=recovery
        / "source_snapshot/joint_runner_pre_summaryschemafix.py",
        candidate_env=recovery / "source_snapshot/residual_env_pre_summaryschemafix.py",
    )
    assert proof["changed_existing_functions"] == [
        "_validate_modeprocess_wrapper",
        "main",
    ]
    checkpoint = recovery / "pre_evaluation_modeprocess_rebound.pt"
    old_payload_value = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert isinstance(old_payload_value, dict)
    old_payload = dict(old_payload_value)
    new_payload = dict(old_payload)
    new_manifest = _pairedbatch_source_manifest_for_sources(
        recovery / "source_snapshot/train_phase6_pre_summaryschemafix.py",
        recovery / "source_snapshot/joint_runner_pre_summaryschemafix.py",
        recovery / "source_snapshot/residual_env_pre_summaryschemafix.py",
    )
    new_payload["source_manifest"] = new_manifest
    script_key = str(Path("scripts/train_phase6.py").resolve())
    assert set(old_payload["source_manifest"]) == set(new_manifest)
    assert all(
        old_payload["source_manifest"][key] == new_manifest[key]
        for key in new_manifest
        if key != script_key
    )
    assert old_payload["source_manifest"][script_key] != new_manifest[script_key]
    assert all(_evaluation_devicefix_invariants(old_payload, new_payload).values())
    candidate = tmp_path / "pre_evaluation_warningpolicy_rebound.pt"
    atomic_torch_save(candidate, new_payload)
    config, roster = _roster()
    acceptance = load_learning_acceptance_config()
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        candidate,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=new_manifest,
        expected_iteration=31,
        device="cpu",
    )
    assert _semantic_equal(restored, new_payload)


def test_modern_learning_evaluation_summary_preserves_schedule_sha256(
    tmp_path: Path,
) -> None:
    roster, _ = _write_modeprocess_evidence(tmp_path)
    summary = _merge_modeprocess_evaluation(
        attempt=tmp_path, roster=roster, stage_quota=2, nominal_quota=2
    )
    assert set(summary) == {
        "iteration",
        "rank_results",
        "schedule_sha256",
        "status",
        "task_metrics",
    }
    assert (
        summary["schedule_sha256"]
        == hashlib.sha256(
            json.dumps(
                [row["schedule_sha256"] for row in summary["rank_results"]],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    assert (
        validate_learning_evaluation_summary(
            summary,
            roster=roster,
            expected_iteration=31,
            expected_stage_quota=2,
            expected_nominal_quota=2,
        )["schedule_sha256"]
        == summary["schedule_sha256"]
    )


def test_finalize_accepts_round_tripped_modeprocess_summary_schema(
    tmp_path: Path,
) -> None:
    roster, _ = _write_modeprocess_evidence(tmp_path)
    summary = _merge_modeprocess_evaluation(
        attempt=tmp_path, roster=roster, stage_quota=2, nominal_quota=2
    )
    round_tripped = validate_learning_evaluation_summary(
        json.loads(json.dumps(summary)),
        roster=roster,
        expected_iteration=31,
        expected_stage_quota=2,
        expected_nominal_quota=2,
    )
    assert round_tripped["schedule_sha256"] == summary["schedule_sha256"]
    finalizer_source = inspect.getsource(
        __import__(
            "scripts.train_phase6", fromlist=["_run_summaryschema_finalize"]
        )._run_summaryschema_finalize
    )
    assert "_current_pairedbatch_source_manifest()" in finalizer_source
    assert "_run_finalize" not in finalizer_source


def test_summaryschema_failed_finalize_manifest_is_immutable(tmp_path: Path) -> None:
    recovery = Path(
        "outputs/phase6_learning_acceptance/"
        "phase6-learning-pilot-4gpu-64env-31iter-20260806_042940/segment_0000/recovery"
    )
    manifest = recovery / "evaluation_warningpolicy_attempt_0007_frozen_manifest.json"
    frozen = _validate_summaryschema_failed_finalize_manifest(manifest)
    assert frozen["status"] == (
        "paired_evaluation_complete_finalize_rejected_summary_schema"
    )
    assert len(frozen["files"]) == 50
    mutated = copy.deepcopy(frozen)
    mutated["outer_log"]["mtime_ns"] += 1
    candidate = tmp_path / "mutated_summaryschema_manifest.json"
    candidate.write_text(json.dumps(mutated, sort_keys=True), encoding="utf-8")
    with pytest.raises(
        ValueError, match="summaryschema failed finalize manifest is invalid"
    ):
        _validate_summaryschema_failed_finalize_manifest(candidate)


def test_summaryschema_rebind_preserves_checkpoint_and_updates_train_joint_sources(
    tmp_path: Path,
) -> None:
    recovery = Path(
        "outputs/phase6_learning_acceptance/"
        "phase6-learning-pilot-4gpu-64env-31iter-20260806_042940/segment_0000/recovery"
    )
    proof = _summaryschema_ast_proof(
        recovery / "source_snapshot/train_phase6_pre_summaryschemafix.py",
        recovery / "source_snapshot/joint_runner_pre_summaryschemafix.py",
        recovery / "source_snapshot/residual_env_pre_summaryschemafix.py",
        candidate_train=recovery / "source_snapshot/train_phase6_pre_mainrunner.py",
        candidate_joint=recovery / "source_snapshot/joint_runner_pre_mainrunner.py",
        candidate_env=recovery / "source_snapshot/residual_env_pre_mainrunner.py",
    )
    assert proof["changed_joint_functions"] == ["validate_learning_evaluation_summary"]
    assert proof["changed_train_functions"] == [
        "_modeprocess_ast_proof",
        "_warningpolicy_ast_proof",
        "main",
    ]
    assert proof["joint_runner_only_adds_modern_schedule_sha256"] is True
    assert proof["modeprocess_proof_endpoint_routing_only"] is True
    assert proof["residual_env_byte_equality"] is True
    assert all(proof["worker_critical_ast_equality"].values())
    config, roster = _roster()
    acceptance = load_learning_acceptance_config()
    historical = _validate_historical_warningpolicy_rebind_record(
        record_path=recovery / "source_rebind_warningpolicy.json",
        rebound_checkpoint=(
            recovery / "pre_evaluation_warningpolicy_rebound.pt"
        ).resolve(),
        config=config,
        roster=roster,
        acceptance=acceptance,
    )
    assert historical["status"] == "ok"
    with pytest.raises(ValueError):
        _validate_historical_warningpolicy_rebind_record(
            record_path=recovery / "source_rebind_warningpolicy.json",
            rebound_checkpoint=(
                recovery / "pre_evaluation_modeprocess_rebound.pt"
            ).resolve(),
            config=config,
            roster=roster,
            acceptance=acceptance,
        )
    old_payload_value = torch.load(
        recovery / "pre_evaluation_warningpolicy_rebound.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert isinstance(old_payload_value, dict)
    old_payload = dict(old_payload_value)
    new_payload = dict(old_payload)
    new_manifest = _pairedbatch_source_manifest_for_sources(
        recovery / "source_snapshot/train_phase6_pre_mainrunner.py",
        recovery / "source_snapshot/joint_runner_pre_mainrunner.py",
        recovery / "source_snapshot/residual_env_pre_mainrunner.py",
    )
    new_payload["source_manifest"] = new_manifest
    changed = sorted(
        key
        for key in set(old_payload["source_manifest"]) | set(new_manifest)
        if old_payload["source_manifest"].get(key) != new_manifest.get(key)
    )
    assert changed == sorted(
        [
            str(Path("scripts/train_phase6.py").resolve()),
            str(Path("somaforce_cross/learning/joint_runner.py").resolve()),
        ]
    )
    assert all(_evaluation_devicefix_invariants(old_payload, new_payload).values())
    candidate = tmp_path / "pre_evaluation_summaryschema_rebound.pt"
    atomic_torch_save(candidate, new_payload)
    policy = ResidualActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3.0e-4)
    restored = restore_learning_checkpoint(
        candidate,
        config=config,
        roster=roster,
        acceptance=acceptance,
        policy=policy,
        optimizer=optimizer,
        source_manifest=new_manifest,
        expected_iteration=31,
        device="cpu",
    )
    assert _semantic_equal(restored, new_payload)


def test_evaluation_timeout_nesting_keeps_worker_evidence_inside_stage_deadline() -> (
    None
):
    source = inspect.getsource(_run_modeprocess_recovery_evaluate)
    wrapper_source = inspect.getsource(
        __import__(
            "scripts.train_phase6", fromlist=["_run_rank_wrapper"]
        )._run_rank_wrapper
    )
    assert "180, 600, 900" in source
    assert "300, 12600, 13200" in source
    assert "stall_timeout_s < worker_timeout_s < stage_timeout_s" in source
    assert "progress_stall" in wrapper_source
    assert "wall_clock" in wrapper_source


def test_modeprocess_failed_smoke_manifest_is_immutable(tmp_path: Path) -> None:
    recovery = Path(
        "outputs/phase6_learning_acceptance/"
        "phase6-learning-pilot-4gpu-64env-31iter-20260806_042940/segment_0000/recovery"
    )
    manifest = recovery / "evaluation_episodeorder_smoke_0005_frozen_manifest.json"
    frozen = _validate_modeprocess_failed_smoke_manifest(manifest)
    assert frozen["status"] == "timeout_during_residual_cleanup_before_scaffold"
    assert len(frozen["files"]) == 15
    assert frozen["outer_log"]["sha256"] == (
        "6cf0ee56b8d9a14aec55dc11077ed9faccd6abd721b1b872defb8e0a0a94d339"
    )
    mutated = copy.deepcopy(frozen)
    mutated["outer_log"]["size_bytes"] += 1
    candidate = tmp_path / "mutated_modeprocess_manifest.json"
    candidate.write_text(json.dumps(mutated, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="evidence changed"):
        _validate_modeprocess_failed_smoke_manifest(candidate)


def test_pairedbatch_modes_share_schedule_and_count_real_completed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, roster = _roster()
    checkpoint = Path(
        "outputs/phase6_learning_acceptance/"
        "phase6-learning-pilot-4gpu-64env-31iter-20260806_042940/"
        "segment_0000/recovery/pre_evaluation_stepboundary_rebound.pt"
    )
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)

    class _FakePolicy:
        def to(self, *, device: torch.device) -> _FakePolicy:
            del device
            return self

        def load_state_dict(self, state: object, *, strict: bool) -> None:
            del state, strict

        def eval(self) -> _FakePolicy:
            return self

        def act_inference(self, observations: object) -> torch.Tensor:
            del observations
            return torch.zeros(2, 23)

    bound_hashes: list[str] = []
    build_seeds: list[int] = []

    class _FakeEnvironment:
        def __init__(self) -> None:
            self.schedule: dict[str, object] | None = None
            self.mode = ""
            self.positions = {0: 0, 1: 0}
            self.pending: list[dict[str, object]] = []

        def evaluation_nominal_probe(self, _env_id: int, _episode_index: int) -> bool:
            return True

        def bind_evaluation_schedule(
            self, schedule: dict[str, object], *, mode: str
        ) -> None:
            self.schedule = schedule
            self.mode = mode
            self.positions = {0: 0, 1: 0}
            bound_hashes.append(str(schedule["schedule_sha256"]))

        def reset(self) -> tuple[dict[str, torch.Tensor], None]:
            return {"policy": torch.zeros(2, 668)}, None

        def step(
            self, actions: torch.Tensor
        ) -> tuple[dict[str, torch.Tensor], None, None, None, None]:
            del actions
            assert self.schedule is not None
            for env_id in range(2):
                row = self.schedule["rows"][str(env_id)]
                offset = self.positions[env_id]
                if offset < len(row):
                    slot = row[offset]
                    self.pending.append(
                        {
                            "mode": self.mode,
                            "seed": slot["seed"],
                            "subset": slot["subset"],
                        }
                    )
                    self.positions[env_id] += 1
            return {"policy": torch.zeros(2, 668)}, None, None, None, None

        def drain_evaluation_completions(self) -> tuple[dict[str, object], ...]:
            records = tuple(self.pending)
            self.pending.clear()
            return records

        def close(self) -> None:
            return None

    def build_environment(*_args: object, **kwargs: object) -> _FakeEnvironment:
        build_seeds.append(int(kwargs["seed"]))
        return _FakeEnvironment()

    monkeypatch.setattr("scripts.train_phase6._build_environment", build_environment)
    results = []
    for mode in ("residual", "scaffold_only"):
        result, environment = _run_paired_evaluation_rank(
            Namespace(
                device="cpu",
                evaluation_seed=20262806,
                paired_mode=mode,
                paired_nominal_quota=2,
                paired_num_envs=2,
                paired_stage_quota=2,
                resume=checkpoint,
                runtime_mode=mode,
                stage="C1",
            ),
            config=config,
            roster=roster,
            rank=2,
            world_size=4,
            acceptance=load_learning_acceptance_config(
                train_phase6.PHASE6_V1_LEARNING_CONFIG
            ),
            source_manifest=checkpoint_payload["source_manifest"],
        )
        environment.close()
        results.append(result)
    assert bound_hashes == [result["schedule_sha256"] for result in results]
    assert all(result["episodes"] == 4 for result in results)
    assert results[0]["mode_counts"] == {"residual": 4, "scaffold_only": 0}
    assert results[1]["mode_counts"] == {"residual": 0, "scaffold_only": 4}
    assert all(
        result["subset_counts"] == {"stage": 2, "nominal": 2} for result in results
    )
    assert results[0]["completed_seed_sha256"] == results[1]["completed_seed_sha256"]
    assert all(result["unique_seeds"] == 4 for result in results)
    assert build_seeds == [
        paired_evaluation_rank_seed(20262806, 2),
        paired_evaluation_rank_seed(20262806, 2),
    ]


def test_pairedbatch_async_reset_counts_rows_not_vector_horizons() -> None:
    source = inspect.getsource(_run_paired_evaluation_rank)
    assert "drain_evaluation_completions" in source
    assert "while len(completed_records) < expected_per_mode" in source
    assert "for pair in pairs" not in source
    assert "range(384)" not in source
    assert "environment.close()" not in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "step" and isinstance(node.func.value, ast.Name):
                assert node.func.value.id == "environment"
    reset_source = Path("somaforce_cross/envs/residual_env.py").read_text(
        encoding="utf-8"
    )
    reset_tree = ast.parse(reset_source)
    environment = next(
        node
        for node in reset_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SomaForceResidualEnv"
    )
    reset = next(
        node
        for node in environment.body
        if isinstance(node, ast.FunctionDef) and node.name == "_reset_scaffold_only_idx"
    )
    reset_dump = ast.dump(reset, include_attributes=False)
    assert "staged_evaluation_subsets" in reset_dump
    assert "staged_evaluation_parked" in reset_dump
    record = next(
        node
        for node in ast.walk(reset)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_record_completed_episodes"
    )
    commits = [
        node
        for node in ast.walk(reset)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and target.value.attr in {"_evaluation_subset", "_evaluation_parked"}
            for target in node.targets
        )
    ]
    assert commits and record.lineno < min(node.lineno for node in commits)


def test_pairedbatch_progress_is_atomic_monotonic_and_root_eta(tmp_path: Path) -> None:
    common = {
        "world_size": 4,
        "task": "push_box",
        "device": "cuda:0",
        "total_episodes": 8,
        "schedule_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "source_rebind_sha256": "c" * 64,
    }
    rank_zero = _EvaluationProgress(
        path=tmp_path / "rank_0/progress.json", rank=0, **common
    )
    rank_one = _EvaluationProgress(
        path=tmp_path / "rank_1/progress.json", rank=1, **common
    )
    rank_zero.update(
        status="running",
        completed=2,
        mode_counts={"residual": 2, "scaffold_only": 0, "active": "residual"},
        subset_counts={"stage": 2, "nominal": 0},
        control_steps=10,
        force=True,
    )
    with pytest.raises(ValueError, match="regressed"):
        rank_zero.update(status="running", completed=1, force=True)
    rank_one.update(
        status="running",
        completed=1,
        mode_counts={"residual": 1, "scaffold_only": 0, "active": "residual"},
        subset_counts={"stage": 1, "nominal": 0},
        control_steps=11,
        force=True,
    )
    _aggregate_evaluation_progress(tmp_path, world_size=2)
    progress = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert progress["schema_version"] == "phase6_paired_evaluation_progress_v1"
    assert progress["completed_episodes"] == 3
    assert progress["total_episodes"] == 16
    assert progress["eta_s"] is not None
    assert progress["rank_progress"][0]["optimizer_steps"] == 0
    assert progress["rank_progress"][1]["normalizer_updates"] == 0


def test_pairedbatch_rebind_binds_snapshots_and_cancelled_attempt() -> None:
    recovery = Path(
        "outputs/phase6_learning_acceptance/"
        "phase6-learning-pilot-4gpu-64env-31iter-20260806_042940/segment_0000/recovery"
    )
    manifest = recovery / "evaluation_stepboundary_attempt_0002_frozen_manifest.json"
    frozen = _validate_pairedbatch_frozen_attempt(manifest)
    assert frozen["status"] == "cancelled_before_result"
    assert (recovery / "pre_evaluation_stepboundary_rebound.pt").is_file()
    assert (
        PAIREDBATCH_INPUT_SHA256
        == "f81a54e9cae84ba4a00c96320ac171bb8706d2b6db912ac9b42e1177989c8cb6"
    )
    snapshots = {
        recovery
        / "source_snapshot/train_phase6_pre_pairedbatchfix.py": PAIREDBATCH_TRAIN_SNAPSHOT_SHA256,
        recovery
        / "source_snapshot/joint_runner_pre_pairedbatchfix.py": PAIREDBATCH_JOINT_SNAPSHOT_SHA256,
        recovery
        / "source_snapshot/residual_env_pre_pairedbatchfix.py": PAIREDBATCH_ENV_SNAPSHOT_SHA256,
    }
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest
        for path, digest in snapshots.items()
    )
    proof = _pairedbatch_ast_proof_for_sources(
        *snapshots,
        candidate_train=recovery / "source_snapshot/train_phase6_pre_mainrunner.py",
        candidate_joint=recovery / "source_snapshot/joint_runner_pre_mainrunner.py",
        candidate_env=recovery / "source_snapshot/residual_env_pre_mainrunner.py",
    )
    assert proof["approved_changed_paths"] == [
        str(recovery / "source_snapshot/train_phase6_pre_mainrunner.py"),
        str(recovery / "source_snapshot/joint_runner_pre_mainrunner.py"),
        str(recovery / "source_snapshot/residual_env_pre_mainrunner.py"),
    ]
    assert proof["module_ast_sha256"]["train_phase6_current"] == _module_ast_sha256(
        recovery / "source_snapshot/train_phase6_pre_mainrunner.py"
    )


def test_curriculum_eligibility_promotion_and_adjacent_rollback() -> None:
    curriculum = JointCurriculum()
    tasks = ("push_door_hand", "push_box", "move_suitcase", "move_largebox")
    passing = {
        task: {"success": 0.80, "failure": 0.02, "retention": 0.97} for task in tasks
    }
    for crossing in (5000000, 5250000, 5500000):
        decision = curriculum.evaluate_window(crossing=crossing, task_metrics=passing)
    assert decision.promoted and curriculum.stage == "C2"
    failing = {
        task: {"success": 0.40, "failure": 0.20, "retention": 0.80} for task in tasks
    }
    curriculum.evaluate_window(crossing=10250000, task_metrics=failing)
    decision = curriculum.evaluate_window(crossing=10500000, task_metrics=failing)
    assert decision.rolled_back and curriculum.stage == "C1"


def test_learning_summary_rejects_incomplete_rank_episode_and_close_evidence() -> None:
    _, roster = _roster()
    complete = {
        "iteration": 31,
        "rank_results": [
            {"episodes": 768, "rank": index, "status": "ok", "task": task.task}
            for index, task in enumerate(roster.tasks)
        ],
        "status": "ok",
        "task_metrics": {task.task: {"success": 0.0} for task in roster.tasks},
    }
    assert (
        validate_learning_evaluation_summary(
            complete, roster=roster, expected_iteration=31
        )["status"]
        == "ok"
    )
    incomplete = copy.deepcopy(complete)
    incomplete["rank_results"][2]["episodes"] = 383
    with pytest.raises(ValueError):
        validate_learning_evaluation_summary(
            incomplete, roster=roster, expected_iteration=31
        )
    rank_results = [{"rank": rank, "status": "ok"} for rank in range(4)]
    central_log = "\n".join(
        ["JSON_WRITTEN" for _ in range(4)]
        + [
            "PHASE6_JOINT_RESULT=" + json.dumps(result, sort_keys=True)
            for result in rank_results
        ]
        + ["ENV_CLOSED" for _ in range(4)]
    )
    assert (
        _validate_legacy_central_markers(central_log, rank_results=rank_results)[
            "multiple_installable_client_drivers"
        ]
        == 0
    )
    concatenated_results = "\n".join(
        ["JSON_WRITTEN" for _ in range(4)]
        + [
            "PHASE6_JOINT_RESULT="
            + json.dumps(rank_results[0], sort_keys=True)
            + "PHASE6_JOINT_RESULT="
            + json.dumps(rank_results[1], sort_keys=True),
            "PHASE6_JOINT_RESULT="
            + json.dumps(rank_results[2], sort_keys=True)
            + "PHASE6_JOINT_RESULT="
            + json.dumps(rank_results[3], sort_keys=True),
        ]
        + ["ENV_CLOSED" for _ in range(4)]
    )
    assert (
        _validate_legacy_central_markers(
            concatenated_results, rank_results=rank_results
        )["multiple_installable_client_drivers"]
        == 0
    )
    with pytest.raises(RuntimeError, match="close marker"):
        _validate_legacy_central_markers(
            central_log.replace("ENV_CLOSED\n", "", 1), rank_results=rank_results
        )
    with pytest.raises(RuntimeError, match="close marker"):
        _validate_legacy_central_markers(
            "\n".join(central_log.splitlines()[1:] + ["JSON_WRITTEN"]),
            rank_results=rank_results,
        )
    with pytest.raises(RuntimeError, match="Vulkan ICDs"):
        _validate_legacy_central_markers(
            central_log + "\nMultiple Installable Client Drivers",
            rank_results=rank_results,
        )
    with pytest.raises(RuntimeError, match="close marker"):
        _validate_legacy_central_markers(
            central_log.replace("PHASE6_JOINT_RESULT=", "RESULT_REMOVED=", 1),
            rank_results=rank_results,
        )
    with pytest.raises(RuntimeError, match="result JSON"):
        _validate_legacy_central_markers(
            central_log.replace(json.dumps(rank_results[-1], sort_keys=True), "{", 1),
            rank_results=rank_results,
        )
    incomplete = copy.deepcopy(complete)
    incomplete["rank_results"][1]["status"] = "error"
    with pytest.raises(ValueError):
        validate_learning_evaluation_summary(
            incomplete, roster=roster, expected_iteration=31
        )


def test_learning_acceptance_requires_c3_retention_nondominance_and_two_benefits() -> (
    None
):
    acceptance = load_learning_acceptance_config()
    tasks = {}
    roster_tasks = ("push_door_hand", "push_box", "move_suitcase", "move_largebox")
    for index, task in enumerate(roster_tasks):
        tasks[task] = {
            "transition_fraction": 0.25,
            "nominal_retention": 0.96,
            "nominal_invalid": 0,
            "nominal_saturation": 0.05,
            "residual_mean_norm": 0.02,
            "contact_bearing_residual_fraction": 0.30,
            "normalized_reward_share": 0.25,
            "semantic_total_share": 0.25,
            "benefit": (
                {
                    "passes": True,
                    "mismatch_family": "physical" if index == 0 else "sensor",
                    "success_delta": 0.06,
                    "progress_delta": 0.01,
                    "force_p95_ratio": 1.0,
                    "stability_degradation": 0.02,
                }
                if index < 2
                else {"passes": False}
            ),
        }
    metrics = {
        "stage": "C3",
        "iteration": 2442,
        "global_transitions": 20004864,
        "optimizer_steps": 58608,
        "tasks": tasks,
    }
    assert validate_learning_final_acceptance(metrics, config=acceptance)[
        "two_benefits"
    ]
    failed = copy.deepcopy(metrics)
    failed["tasks"]["push_box"]["nominal_retention"] = 0.89
    with pytest.raises(AssertionError):
        validate_learning_final_acceptance(failed, config=acceptance)
