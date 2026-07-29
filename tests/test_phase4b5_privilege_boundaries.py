from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from somaforce_cross.envs.normalizer import FixedFieldNormalizer
from somaforce_cross.envs.numeric_contract import load_numeric_contract


def test_actor_normalizer_has_no_critic_or_clean_wrench_input() -> None:
    source = Path("somaforce_cross/envs/normalizer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "normalize_actor"
    )
    text = ast.get_source_segment(source, method) or ""
    assert (
        "clean_wrench" not in text
        and "physics_mismatch" not in text
        and "object_state" not in text
    )


def test_critic_mutation_cannot_change_actor_values_or_state() -> None:
    normalizer = FixedFieldNormalizer(
        load_numeric_contract("configs/phase4b5_numeric_contract.json")
    )
    actor = {
        "wrist_tokens": torch.zeros(1, 2, 16, 14),
        "proprio": torch.zeros(1, 64),
        "a_nom_history": torch.zeros(1, 23, 3),
        "previous_a_total": torch.zeros(1, 23),
        "z_cross": torch.zeros(1, 64),
    }
    before = normalizer.normalize_actor(actor)
    privileged = torch.full((1, 39), 999.0)
    privileged[:] = -999.0
    after = normalizer.normalize_actor(actor)
    assert normalizer.state_dict() == {
        "contract_hash": normalizer.contract_hash,
        "contract_version": "phase4b5_numeric_v1",
    }
    assert torch.equal(before["z_cross"], after["z_cross"]) and torch.equal(
        before["wrist_tokens"], after["wrist_tokens"]
    )


@pytest.mark.parametrize(
    "task", ["push_door_hand", "push_box", "move_suitcase", "move_largebox"]
)
@pytest.mark.parametrize(
    "forbidden",
    [
        "clean_wrench",
        "physics_mismatch",
        "object_state",
        "progress",
        "contact_truth",
        "task_id",
    ],
)
def test_actor_rejects_all_privileged_fields_for_each_task(
    task: str, forbidden: str
) -> None:
    normalizer = FixedFieldNormalizer(
        load_numeric_contract("configs/phase4b5_numeric_contract.json")
    )
    actor = {
        "wrist_tokens": torch.zeros(1, 2, 16, 14),
        "proprio": torch.zeros(1, 64),
        "a_nom_history": torch.zeros(1, 23, 3),
        "previous_a_total": torch.zeros(1, 23),
        "z_cross": torch.zeros(1, 64),
    }
    baseline = normalizer.normalize_actor(actor)
    contaminated = dict(actor)
    contaminated[forbidden] = torch.full((1, 1), float(len(task)))
    with pytest.raises(ValueError):
        normalizer.normalize_actor(contaminated)
    after = normalizer.normalize_actor(actor)
    assert all(torch.equal(baseline[key], after[key]) for key in baseline)
