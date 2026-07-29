from __future__ import annotations

from pathlib import Path

import pytest
import torch

from somaforce_cross.envs.numeric_contract import load_numeric_contract
from somaforce_cross.envs.reward_manager import (
    EpisodeMetricLog,
    EpisodeTermination,
    Phase4B5RewardManager,
    RetentionEvaluator,
)


CONTRACT = load_numeric_contract(Path("configs/phase4b5_numeric_contract.json"))


def _inputs(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "progress_delta": torch.tensor([0.03, -0.03], dtype=torch.float32)[:batch],
        "expected_contact": torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)[
            :batch
        ],
        "contact_truth": torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float32)[
            :batch
        ],
        "wrench": torch.zeros(batch, 2, 6, dtype=torch.float32),
        "stability_margin": torch.tensor([0.15, 0.0], dtype=torch.float32)[:batch],
        "delta_safe": torch.zeros(batch, 23, dtype=torch.float32),
        "authority": torch.ones(batch, 23, dtype=torch.float32),
        "previous_delta_safe": torch.zeros(batch, 23, dtype=torch.float32),
        "success_latched": torch.zeros(batch, dtype=torch.bool),
        "failure_latched": torch.zeros(batch, dtype=torch.bool),
        "nonfinite": torch.zeros(batch, dtype=torch.bool),
        "terminated": torch.zeros(batch, dtype=torch.bool),
        "time_outs": torch.zeros(batch, dtype=torch.bool),
    }


def test_contact_fixture_zero_and_independent_negative_case() -> None:
    manager = Phase4B5RewardManager(CONTRACT, "push_door_hand")
    result = manager.compute(**_inputs())
    q = 1.0 / 508.0
    assert torch.allclose(result.raw_terms["progress"], torch.tensor([0.02, -0.02]))
    assert torch.allclose(result.raw_terms["contact"], torch.tensor([0.375 * q, 0.0]))
    negative = _inputs(batch=1)
    negative["expected_contact"][:] = 1.0
    negative["contact_truth"][:] = 0.0
    assert torch.allclose(
        manager.compute(**negative).raw_terms["contact"], torch.tensor([-q])
    )


def test_terminal_terms_require_terminal_mask_and_precedence() -> None:
    manager = Phase4B5RewardManager(CONTRACT, "push_box")
    values = _inputs(batch=1)
    values["success_latched"][:] = True
    values["failure_latched"][:] = True
    values["nonfinite"][:] = True
    nonterminal = manager.compute(**values)
    assert all(
        torch.equal(nonterminal.raw_terms[key], torch.zeros(1))
        for key in ("terminal_success", "terminal_failure", "nonfinite")
    )
    values["terminated"][:] = True
    output = manager.compute(**values)
    assert output.raw_terms["nonfinite"].item() == 1.0
    assert output.raw_terms["terminal_failure"].item() == 0.0
    assert output.raw_terms["terminal_success"].item() == 0.0
    values["nonfinite"][:] = False
    output = manager.compute(**values)
    assert output.raw_terms["terminal_failure"].item() == 1.0
    assert output.raw_terms["terminal_success"].item() == 0.0
    values["failure_latched"][:] = False
    output = manager.compute(**values)
    assert output.raw_terms["terminal_success"].item() == 1.0


def test_episode_termination_uses_adapter_payload_success_and_latches() -> None:
    termination = EpisodeTermination(CONTRACT, 2)
    common = dict(
        progress=torch.tensor([0.0, 0.0], dtype=torch.float32),
        adapter_task_success=torch.tensor([True, False]),
        failure=torch.tensor([False, True]),
        nonfinite=torch.tensor([False, False]),
        reference_exhausted=torch.tensor([True, True]),
        episode_steps=torch.tensor([472, 25]),
        tare_ready=torch.tensor([True, False]),
    )
    terminated, timeout = termination.update(task="move_suitcase", **common)
    assert torch.equal(termination.success, torch.tensor([True, False]))
    assert torch.equal(terminated, torch.tensor([False, True]))
    assert torch.equal(timeout, torch.tensor([True, False]))
    assert torch.equal(termination.episode_invalid, torch.tensor([False, True]))


def test_c0_bypasses_manager_bitwise() -> None:
    result = Phase4B5RewardManager(CONTRACT, "push_box", c0=True).compute(**_inputs())
    assert torch.equal(result.total, torch.zeros_like(result.total))


@pytest.mark.parametrize(
    "task", ["push_door_hand", "push_box", "move_suitcase", "move_largebox"]
)
def test_all_reward_terms_weights_horizons_dtype_and_finite(task: str) -> None:
    manager = Phase4B5RewardManager(CONTRACT, task)
    values = _inputs(batch=1)
    values["wrench"][:] = torch.tensor([[[100.0, 0.0, 0.0, 10.0, 0.0, 0.0], [0.0] * 6]])
    values["stability_margin"][:] = 0.0
    values["delta_safe"][:] = 1.0
    values["previous_delta_safe"][:] = 0.8
    values["success_latched"][:] = True
    values["time_outs"][:] = True
    result = manager.compute(**values)
    horizon = CONTRACT.payload["episodes"]["horizons"][task]
    expected_progress = torch.tensor(
        [0.02], dtype=result.total.dtype, device=result.total.device
    )
    float32_atol = torch.finfo(result.total.dtype).eps
    torch.testing.assert_close(
        result.raw_terms["progress"], expected_progress, rtol=0.0, atol=float32_atol
    )
    assert result.raw_terms["terminal_success"].item() == 1.0
    assert result.weighted_terms["terminal_success"].item() == 5.0
    assert result.raw_terms["force"].item() <= 0.0
    for name, numerator in (("stability", -1.0), ("residual", -1.0), ("rate", -4.0)):
        expected = torch.tensor(
            [numerator / horizon], dtype=result.total.dtype, device=result.total.device
        )
        torch.testing.assert_close(
            result.raw_terms[name], expected, rtol=0.0, atol=float32_atol
        )
    clipped = _inputs(batch=1)
    clipped["stability_margin"][:] = -0.15
    clipped_result = manager.compute(**clipped)
    clipped_expected = torch.tensor(
        [-4.0 / horizon], dtype=result.total.dtype, device=result.total.device
    )
    torch.testing.assert_close(
        clipped_result.raw_terms["stability"],
        clipped_expected,
        rtol=0.0,
        atol=float32_atol,
    )
    assert result.total.dtype == torch.float32 and torch.isfinite(result.total).all()
    assert torch.all(result.total <= 10.0) and torch.all(result.total >= -20.0)


@pytest.mark.parametrize(
    "task", ["push_door_hand", "push_box", "move_suitcase", "move_largebox"]
)
def test_episode_horizon_exhaustion_tare_and_selected_reset(task: str) -> None:
    termination = EpisodeTermination(CONTRACT, 3)
    horizon = CONTRACT.payload["episodes"]["horizons"][task]
    common = {
        "task": task,
        "progress": torch.zeros(3),
        "adapter_task_success": torch.zeros(3, dtype=torch.bool),
        "failure": torch.zeros(3, dtype=torch.bool),
        "nonfinite": torch.zeros(3, dtype=torch.bool),
        "reference_exhausted": torch.zeros(3, dtype=torch.bool),
        "tare_ready": torch.tensor([True, False, True]),
    }
    for step, expected_timeout in (
        (horizon - 1, False),
        (horizon, True),
        (horizon + 1, True),
    ):
        terminated, timeout = termination.update(
            episode_steps=torch.full((3,), step, dtype=torch.int64), **common
        )
        assert not terminated.any() and bool(timeout[0]) is expected_timeout
    termination.update(episode_steps=torch.tensor([24, 25, 25]), **common)
    assert not termination.episode_invalid[0] and termination.episode_invalid[1]
    termination.success[:] = True
    termination.failure[:] = True
    termination.episode_invalid[:] = True
    termination.reset(torch.tensor([1], dtype=torch.int64))
    assert termination.success.tolist() == [True, False, True]
    assert termination.failure.tolist() == [True, False, True]
    assert termination.episode_invalid.tolist() == [True, False, True]
    with pytest.raises(ValueError):
        termination.reset(torch.tensor([1, 1], dtype=torch.int64))


def _episode_record(task: str, seed: int, value: float) -> dict[str, object]:
    schema = CONTRACT.payload["logging"]
    return {
        "task": task,
        "stage": "C1",
        "family": "nominal",
        "evaluation_id": None,
        "nominal": True,
        "seed": seed,
        "steps": 3,
        "outcome": {key: False for key in schema["outcome_keys"]},
        "raw_reward_sums": {key: value for key in schema["raw_reward_keys"]},
        "weighted_reward_sums": {key: value for key in schema["weighted_reward_keys"]},
        "return": value,
        "diagnostics": {key: value for key in schema["diagnostic_keys"]},
    }


def test_episode_log_schema_equal_episode_pool_and_retention() -> None:
    log = EpisodeMetricLog(CONTRACT)
    log.add_episode(_episode_record("push_box", 1, 1.0))
    log.add_episode(_episode_record("push_door_hand", 2, 3.0))
    pooled = log.pooled("return")
    assert pooled["mean"] == 2.0
    assert pooled["task_episode_counts"] == {
        "push_door_hand": 1,
        "push_box": 1,
        "move_suitcase": 0,
        "move_largebox": 0,
    }
    broken = _episode_record("push_box", 3, 1.0)
    broken.pop("diagnostics")
    with pytest.raises(ValueError):
        log.add_episode(broken)
    metrics = {
        task: {name: spec["baseline"] for name, spec in specification.items()}
        for task, specification in CONTRACT.payload["retention"].items()
    }
    result = RetentionEvaluator(CONTRACT).score(metrics)
    assert result["gate_score"] == 1.0
    metrics["push_box"]["final_error_m"] *= 2.0
    assert RetentionEvaluator(CONTRACT).score(metrics)["task_scores"]["push_box"] == 0.5
