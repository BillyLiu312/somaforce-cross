from __future__ import annotations

import copy
import inspect
import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing

from scripts.train_phase6 import (
    _parent_close_verified,
    _profile,
    summarize_phase6_run,
)
from somaforce_cross.learning.actor_critic import ResidualActorCritic
from somaforce_cross.learning.joint_runner import (
    JointCurriculum,
    JointTaskScheduler,
    Phase6Config,
    TransitionMetricCollector,
    assert_matching_hash_records,
    assert_transition_balance,
    atomic_torch_save,
    canonical_sha256,
    joint_checkpoint_payload,
    load_phase5_policy_only,
    load_phase6_config,
    load_phase6_task_roster,
    optimizer_step,
    pooled_macro_metrics,
    restore_joint_checkpoint,
    state_dict_sha256,
    transition_accounting,
    validate_phase6_joint_metrics,
    validate_phase6_config,
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
        rank_rng={str(rank): {"cpu": torch.arange(2)} for rank in range(4)},
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
    config, roster = _roster()
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
        phase6_config=CONFIG_PATH,
        profile=profile,
        roster=ROSTER_PATH,
    )
    summary = summarize_phase6_run(args)
    assert summary["status"] == "ok"
    assert len(summary["rank_results"]) == len(summary["wrapper_summaries"]) == 4
    rank_dir = tmp_path / "rank_3"
    (rank_dir / "worker.log").write_text(
        "Multiple Installable Client Drivers\n", encoding="utf-8"
    )
    wrapper = json.loads((rank_dir / "wrapper.json").read_text(encoding="utf-8"))
    wrapper["warning_counts"]["multiple_installable_client_drivers"] = 1
    (rank_dir / "wrapper.json").write_text(json.dumps(wrapper), encoding="utf-8")
    with pytest.raises(RuntimeError, match="multiple Vulkan"):
        summarize_phase6_run(args)
