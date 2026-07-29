from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from somaforce_cross.envs.numeric_contract import (
    AMENDMENT_BASE_COMMIT,
    AMENDMENT_REASON,
    CONTRACT_AUDIT_COMMIT,
    CONTRACT_VERSION,
    SOURCE_RUNTIME_COMMIT,
    canonical_json_bytes,
    load_numeric_contract,
    validate_numeric_contract,
)


CONTRACT_PATH = Path("configs/phase4b5_numeric_contract.json")
EXPECTED_V2_SHA256 = "214b5328f0705b467f0fe305ec5eec78dc91f3ced163cf1d410ea00d43889ab3"


def _payload() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_canonical_contract_covers_runtime_identity_and_hash() -> None:
    contract = load_numeric_contract(CONTRACT_PATH)
    assert contract.payload["contract_version"] == CONTRACT_VERSION
    assert contract.payload["source_runtime_commit"] == SOURCE_RUNTIME_COMMIT
    assert contract.payload["contract_audit_commit"] == CONTRACT_AUDIT_COMMIT
    assert contract.sha256 == EXPECTED_V2_SHA256
    assert contract.payload["amendment"] == {
        "base_commit": AMENDMENT_BASE_COMMIT,
        "date": "2026-07-29",
        "owner": "project_owner_delegated_planner",
        "reason": AMENDMENT_REASON,
        "status": "approved",
    }
    assert contract.payload["runtime_deferred_fields"] == {
        "push_door_hand": {
            "0:3": "hinge_axis_requires_scene_bank",
            "5:11": "physical_handle_geometry_requires_scene_bank",
        }
    }
    changed = copy.deepcopy(contract.payload)
    changed["sampler"]["seed_offsets"]["physical"] += 1
    assert canonical_json_bytes(changed) != canonical_json_bytes(contract.payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("unknown", 1),
        lambda value: value["sampler"]["task_ranges"]["move_suitcase"].pop("C3"),
        lambda value: value["units"].pop("force"),
        lambda value: value["approval"].__setitem__("status", "pending"),
        lambda value: value["reward"]["constants"].__setitem__(
            "force_scale_N", float("nan")
        ),
    ],
)
def test_contract_rejects_nested_schema_unit_approval_and_nonfinite(
    mutate: object,
) -> None:
    payload = _payload()
    mutate(payload)  # type: ignore[operator]
    with pytest.raises((TypeError, ValueError)):
        validate_numeric_contract(payload)


def test_contract_rejects_invalid_probability_and_c0_upgrade() -> None:
    payload = _payload()
    payload["curriculum"]["stage_families"]["C2"]["physical"] = 0.5
    with pytest.raises(ValueError):
        validate_numeric_contract(payload)
    payload = _payload()
    payload["c0"]["F_scale"] = 100.0
    with pytest.raises(ValueError, match="C0"):
        validate_numeric_contract(payload)


@pytest.mark.parametrize(
    "path,value",
    [
        (("curriculum", "windows", "promotion_windows"), 0),
        (("curriculum", "family_order"), ["physical"]),
        (("sampler", "sensor", "C3", "delay_probability"), [0.5, 0.5]),
        (("sampler", "task_ranges", "push_box", "C3", "mass"), [16.0, 1.0]),
        (("retention", "push_box", "median_progress", "direction"), "sideways"),
        (("held_out", "box_mass_16_friction_1_2", "type"), "unknown"),
        (("sampler", "counter_rng", "algorithm"), "fallback"),
        (("sampler", "counter_rng", "float_divisor"), 1),
        (("sampler", "counter_rng", "mantissa_shift"), 32),
        (("sampler", "counter_rng", "state_shift"), 32),
        (("sampler", "counter_rng", "output_shift"), 0),
        (("sampler", "counter_rng", "state_multiplier"), 4294967296),
    ],
)
def test_contract_rejects_nested_type_support_and_direction_errors(
    path: tuple[str, ...], value: object
) -> None:
    payload = _payload()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises((TypeError, ValueError)):
        validate_numeric_contract(payload)


def test_contract_rejects_counter_rng_unknown_or_missing_fields() -> None:
    payload = _payload()
    payload["sampler"]["counter_rng"]["fallback_seed"] = 1
    with pytest.raises(ValueError):
        validate_numeric_contract(payload)
    payload = _payload()
    payload["sampler"]["counter_rng"].pop("word_mask")
    with pytest.raises(ValueError):
        validate_numeric_contract(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.__setitem__("contract_version", "phase4b5_numeric_v1"),
        lambda value: value["sampler"]["physical_subfamilies"].__setitem__(
            "door", ["mechanism", "axis_handle", "initial_stance", "contact_target"]
        ),
        lambda value: value["sampler"]["task_ranges"]["push_door_hand"][
            "C1"
        ].__setitem__("axis_deg", 1.0),
        lambda value: value["sampler"]["task_ranges"]["push_door_hand"][
            "C1"
        ].__setitem__("handle_m", 0.01),
        lambda value: value["sampler"]["task_ranges"]["push_door_hand"][
            "C1"
        ].__setitem__("handle_rot_deg", 2.0),
        lambda value: value["runtime_deferred_fields"]["push_door_hand"].__setitem__(
            "0:3", 0.0
        ),
    ],
)
def test_v2_rejects_v1_axis_handle_and_deferred_numeric_fallbacks(
    mutate: object,
) -> None:
    payload = _payload()
    mutate(payload)  # type: ignore[operator]
    with pytest.raises((TypeError, ValueError)):
        validate_numeric_contract(payload)


def test_v1_payload_without_v2_metadata_is_rejected() -> None:
    payload = _payload()
    payload["contract_version"] = "phase4b5_numeric_v1"
    payload.pop("amendment")
    payload.pop("runtime_deferred_fields")
    with pytest.raises(ValueError):
        validate_numeric_contract(payload)
