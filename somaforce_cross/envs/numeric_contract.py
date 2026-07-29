"""Frozen Phase 4B5 numeric-contract loading and canonical hashing."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CONTRACT_VERSION = "phase4b5_numeric_v1"
SOURCE_RUNTIME_COMMIT = "caf09bfdaaa50004bead2750b9da495a340adffa"
CONTRACT_AUDIT_COMMIT = "4a3c384f5017b9da96cc5b5abdebda0245ec53bd"
APPROVAL_OWNER = "project_owner_delegated_planner"
APPROVAL_DATE = "2026-07-29"
_ROOT_KEYS = frozenset(
    {
        "acceptance",
        "approval",
        "c0",
        "contract_audit_commit",
        "contract_version",
        "curriculum",
        "episodes",
        "held_out",
        "logging",
        "normalizer",
        "retention",
        "reward",
        "sampler",
        "sensor",
        "source_runtime_commit",
        "units",
    }
)
TASKS = ("push_door_hand", "push_box", "move_suitcase", "move_largebox")
STAGES = ("C1", "C2", "C3")
_RIGID_TASKS = ("push_box", "move_suitcase", "move_largebox")
_SENSOR_STAGE_KEYS = {
    "axis_deg",
    "bias_force",
    "bias_moment",
    "delay_probability",
    "delay_values",
    "drift_force",
    "drift_moment",
    "dropout",
    "filter",
    "random_walk_force",
    "random_walk_moment",
    "scale",
    "saturation_force",
    "saturation_moment",
    "white_force",
    "white_moment",
}
_DOOR_RANGE_KEYS = {
    "axis_deg",
    "damping",
    "friction",
    "handle_m",
    "handle_rot_deg",
    "object_m",
    "object_yaw_deg",
    "stance_m",
    "stance_yaw_deg",
}
_RIGID_RANGE_KEYS = {
    "contact_m",
    "contact_rot_deg",
    "com_m",
    "friction",
    "mass",
    "object_xy_m",
    "object_yaw_deg",
    "stance_m",
    "stance_yaw_deg",
}


def _reject_nonfinite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} must not contain NaN or infinity")
    if value is None:
        raise ValueError(f"{path} must not be null")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key")
            _reject_nonfinite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite(child, f"{path}[{index}]")


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Encode a JSON object using the frozen canonical serialization."""
    if not isinstance(payload, Mapping):
        raise TypeError("numeric contract must be a JSON object")
    _reject_nonfinite(payload)
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("numeric contract is not canonical-JSON serializable") from exc


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _require_exact_keys(value: Any, name: str, keys: set[str]) -> Mapping[str, Any]:
    mapping = _require_mapping(value, name)
    actual = set(mapping)
    if actual != keys:
        missing = sorted(keys - actual)
        unknown = sorted(actual - keys)
        raise ValueError(f"{name} has missing={missing} unknown={unknown} keys")
    return mapping


def _exact(mapping: Mapping[str, Any], path: str, keys: set[str]) -> Mapping[str, Any]:
    return _require_exact_keys(mapping, path, keys)


def _number(value: Any, path: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{path} must be a finite number")
    return float(value)


def _positive(value: Any, path: str) -> float:
    number = _number(value, path)
    if number <= 0.0:
        raise ValueError(f"{path} must be positive")
    return number


def _positive_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _interval(value: Any, path: str, *, nonnegative: bool = False) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{path} must be a two-value interval")
    lo, hi = (_number(item, path) for item in value)
    if lo > hi or (nonnegative and lo < 0.0):
        raise ValueError(f"{path} has an invalid interval")


def _probabilities(value: Any, path: str) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty probability list")
    numbers = [_number(item, path) for item in value]
    if any(item < 0.0 or item > 1.0 for item in numbers) or not math.isclose(
        sum(numbers), 1.0, abs_tol=1e-7
    ):
        raise ValueError(f"{path} must sum to one")


def _validate_schema(payload: Mapping[str, Any]) -> None:
    """Independent exact-key schema; it intentionally never reads the JSON file."""
    _exact(
        payload["acceptance"],
        "acceptance",
        {"float_dtype", "held_out_requires_evaluation_id", "pooled_episode_weighting"},
    )
    if (
        payload["acceptance"]["float_dtype"] != "float32"
        or payload["acceptance"]["pooled_episode_weighting"] != "equal_episode"
        or not isinstance(
            payload["acceptance"]["held_out_requires_evaluation_id"], bool
        )
    ):
        raise ValueError("acceptance schema is invalid")
    _exact(
        payload["c0"],
        "c0",
        {
            "F_scale",
            "M_scale",
            "authority",
            "control_steps",
            "episode_steps",
            "identity_sensor",
            "reward",
            "seed_rule",
        },
    )
    if not isinstance(payload["c0"]["identity_sensor"], bool) or not isinstance(
        payload["c0"]["seed_rule"], str
    ):
        raise ValueError("C0 type schema is invalid")
    curriculum = _exact(
        payload["curriculum"],
        "curriculum",
        {
            "family_order",
            "gates",
            "nominal_atom",
            "stage_families",
            "task_probability",
            "windows",
        },
    )
    _exact(curriculum["gates"], "curriculum.gates", {"C1", "C2", "rollback"})
    for stage in ("C1", "C2"):
        _exact(
            curriculum["gates"][stage],
            f"curriculum.gates.{stage}",
            {"failure_max", "min_transitions", "retention_min", "success_min"},
        )
    _exact(
        curriculum["gates"]["rollback"],
        "curriculum.gates.rollback",
        {"failure_margin", "retention_min", "success_margin"},
    )
    _exact(curriculum["nominal_atom"], "curriculum.nominal_atom", set(STAGES))
    for stage, probability in curriculum["nominal_atom"].items():
        if not 0.0 < _number(probability, f"curriculum.nominal_atom.{stage}") < 1.0:
            raise ValueError("nominal atom must be in (0,1)")
    _exact(curriculum["stage_families"], "curriculum.stage_families", set(STAGES))
    family_keys = {
        "all",
        "physical",
        "physical_scaffold",
        "physical_sensor",
        "scaffold",
        "sensor",
        "sensor_scaffold",
    }
    if (
        not isinstance(curriculum["family_order"], list)
        or set(curriculum["family_order"]) != family_keys
        or len(curriculum["family_order"]) != len(family_keys)
    ):
        raise ValueError("curriculum.family_order must name each family exactly once")
    for stage in STAGES:
        _exact(
            curriculum["stage_families"][stage],
            f"curriculum.stage_families.{stage}",
            family_keys,
        )
        if not math.isclose(
            sum(
                _number(item, stage)
                for item in curriculum["stage_families"][stage].values()
            ),
            1.0,
            abs_tol=1e-7,
        ):
            raise ValueError(f"curriculum.stage_families.{stage} must sum to one")
    _exact(curriculum["task_probability"], "curriculum.task_probability", set(TASKS))
    if not math.isclose(
        sum(
            _number(value, "curriculum.task_probability")
            for value in curriculum["task_probability"].values()
        ),
        1.0,
        abs_tol=1e-7,
    ):
        raise ValueError("task probabilities must sum to one")
    _exact(
        curriculum["windows"],
        "curriculum.windows",
        {
            "evaluation_interval_transitions",
            "nominal_episodes_per_task",
            "promotion_windows",
            "rollback_windows",
            "stage_episodes_per_task",
        },
    )
    for key, value in curriculum["windows"].items():
        _positive_integer(value, f"curriculum.windows.{key}")
    episodes = _exact(payload["episodes"], "episodes", {"horizons", "tare"})
    _exact(episodes["horizons"], "episodes.horizons", set(TASKS))
    for task, horizon in episodes["horizons"].items():
        _positive_integer(horizon, f"episodes.horizons.{task}")
    _exact(
        episodes["tare"],
        "episodes.tare",
        {"max_steps", "samples", "valid_expected_contact"},
    )
    _positive_integer(episodes["tare"]["max_steps"], "episodes.tare.max_steps")
    _positive_integer(episodes["tare"]["samples"], "episodes.tare.samples")
    if not isinstance(episodes["tare"]["valid_expected_contact"], bool):
        raise ValueError("episodes.tare.valid_expected_contact must be bool")
    logging = _exact(
        payload["logging"],
        "logging",
        {"diagnostic_keys", "outcome_keys", "raw_reward_keys", "weighted_reward_keys"},
    )
    for key, value in logging.items():
        if (
            not isinstance(value, list)
            or not value
            or len(value) != len(set(value))
            or any(not isinstance(name, str) or not name for name in value)
        ):
            raise ValueError(f"logging.{key} must contain unique non-empty keys")
    _exact(
        payload["normalizer"],
        "normalizer",
        {"actor", "critic", "final_continuous_clip", "version"},
    )
    _exact(
        payload["normalizer"]["actor"],
        "normalizer.actor",
        {
            "action_divisor",
            "action_clip",
            "angular_twist_divisor",
            "joint_offset_divisor",
            "joint_velocity_divisor",
            "linear_twist_divisor",
            "root_angular_divisor",
            "wrench_clip",
            "z_cross_clip",
        },
    )
    _exact(
        payload["normalizer"]["critic"],
        "normalizer.critic",
        {
            "alpha_center",
            "alpha_divisor",
            "angular_divisor",
            "delay_center",
            "delay_divisor",
            "force_divisor",
            "linear_divisor",
            "log_ratio_divisor",
            "moment_divisor",
            "object_position_divisor",
            "root_height_divisor",
            "support_divisor",
            "torque_divisor",
            "velocity_divisor",
        },
    )
    reward = _exact(
        payload["reward"], "reward", {"aggregate_clip", "constants", "weights"}
    )
    _interval(reward["aggregate_clip"], "reward.aggregate_clip")
    _exact(
        reward["constants"],
        "reward.constants",
        {
            "contact_unexpected_penalty",
            "force_expected_threshold",
            "force_scale_N",
            "force_s_clip",
            "moment_scale_Nm",
            "progress_delta_clip",
            "rate_delta_scale",
            "rate_ratio_clip",
            "residual_epsilon",
            "stability_margin_m",
            "stability_penalty_clip",
        },
    )
    _exact(
        reward["weights"],
        "reward.weights",
        {
            "contact",
            "force",
            "nonfinite",
            "progress",
            "rate",
            "residual",
            "stability",
            "terminal_failure",
            "terminal_success",
        },
    )
    sampler = _exact(
        payload["sampler"],
        "sampler",
        {
            "counter_rng",
            "nominal_rows",
            "nominal_scaffold",
            "physical_family_count",
            "physical_subfamilies",
            "scaffold",
            "seed_offsets",
            "sensor",
            "stage_constants",
            "task_ranges",
        },
    )
    _exact(sampler["nominal_rows"], "sampler.nominal_rows", set(TASKS))
    _exact(
        sampler["nominal_rows"]["push_door_hand"],
        "sampler.nominal_rows.push_door_hand",
        {"damping", "friction"},
    )
    for task in _RIGID_TASKS:
        row = _exact(
            sampler["nominal_rows"][task],
            f"sampler.nominal_rows.{task}",
            {"friction", "inertia", "mass"},
        )
        if not isinstance(row["inertia"], list) or len(row["inertia"]) != 6:
            raise ValueError("nominal inertia must contain six entries")
    _exact(
        sampler["physical_family_count"], "sampler.physical_family_count", set(STAGES)
    )
    for stage in STAGES:
        count = _exact(
            sampler["physical_family_count"][stage],
            f"sampler.physical_family_count.{stage}",
            {"count", "probability"},
        )
        if not isinstance(count["count"], list) or len(count["count"]) != len(
            count["probability"]
        ):
            raise ValueError("physical family count/probability sizes differ")
        _probabilities(
            count["probability"], f"sampler.physical_family_count.{stage}.probability"
        )
    _exact(
        sampler["physical_subfamilies"],
        "sampler.physical_subfamilies",
        {"door", "rigid"},
    )
    if (
        len(sampler["physical_subfamilies"]["door"]) != 4
        or len(sampler["physical_subfamilies"]["rigid"]) != 4
    ):
        raise ValueError("each physical family requires four named subfamilies")
    _exact(sampler["nominal_scaffold"], "sampler.nominal_scaffold", {"alpha", "delay"})
    _positive_integer(
        sampler["nominal_scaffold"]["delay"], "sampler.nominal_scaffold.delay"
    )
    if (
        not 0.0
        <= _number(
            sampler["nominal_scaffold"]["alpha"], "sampler.nominal_scaffold.alpha"
        )
        < 1.0
    ):
        raise ValueError("nominal scaffold alpha must be in [0,1)")
    _exact(sampler["scaffold"], "sampler.scaffold", set(STAGES))
    for stage in STAGES:
        row = _exact(
            sampler["scaffold"][stage],
            f"sampler.scaffold.{stage}",
            {"pairs", "probability"},
        )
        _probabilities(row["probability"], f"sampler.scaffold.{stage}.probability")
        if len(row["pairs"]) != len(row["probability"]) or any(
            pair[1] == 1.0 for pair in row["pairs"]
        ):
            raise ValueError("scaffold pairs are invalid")
    _exact(
        sampler["seed_offsets"],
        "sampler.seed_offsets",
        {"episode_multiplier", "per_step", "physical", "scaffold", "sensor"},
    )
    offsets = [
        _positive_integer(value, f"sampler.seed_offsets.{key}")
        for key, value in sampler["seed_offsets"].items()
    ]
    if len(offsets) != len(set(offsets)):
        raise ValueError("sampler seed offsets must be distinct")
    rng = _exact(
        sampler["counter_rng"],
        "sampler.counter_rng",
        {
            "algorithm",
            "counter_bias",
            "float_divisor",
            "mantissa_shift",
            "output_multiplier",
            "output_shift",
            "slot_bias",
            "slot_multiplier",
            "state_increment",
            "state_multiplier",
            "state_shift",
            "state_shift_bias",
            "word_mask",
        },
    )
    if rng["algorithm"] != "pcg_rxs_m_xs_32":
        raise ValueError("unsupported counter RNG algorithm")
    if rng["word_mask"] != (1 << 32) - 1:
        raise ValueError("counter RNG word mask is invalid")
    word_mask = _positive_integer(rng["word_mask"], "sampler.counter_rng.word_mask")
    for key in (
        "counter_bias",
        "slot_bias",
        "slot_multiplier",
        "state_multiplier",
        "state_increment",
        "output_multiplier",
    ):
        value = _positive_integer(rng[key], f"sampler.counter_rng.{key}")
        if value > word_mask:
            raise ValueError(f"sampler.counter_rng.{key} exceeds word mask")
    for key in ("state_shift", "output_shift", "mantissa_shift"):
        value = _positive_integer(rng[key], f"sampler.counter_rng.{key}")
        if value > 31:
            raise ValueError(f"sampler.counter_rng.{key} must be in [1,31]")
    shift_bias = _positive_integer(
        rng["state_shift_bias"], "sampler.counter_rng.state_shift_bias"
    )
    dynamic_shift = (word_mask >> rng["state_shift"]) + shift_bias
    if dynamic_shift >= 64:
        raise ValueError("counter RNG dynamic shift must be below 64")
    if rng["float_divisor"] != 1 << (32 - rng["mantissa_shift"]):
        raise ValueError("counter RNG mask or float divisor is invalid")
    _exact(sampler["stage_constants"], "sampler.stage_constants", set(STAGES))
    for stage in STAGES:
        _exact(
            sampler["stage_constants"][stage],
            f"sampler.stage_constants.{stage}",
            {"load_share"},
        )
    _exact(sampler["sensor"], "sampler.sensor", set(STAGES))
    for stage in STAGES:
        row = _exact(
            sampler["sensor"][stage], f"sampler.sensor.{stage}", _SENSOR_STAGE_KEYS
        )
        for key in _SENSOR_STAGE_KEYS - {"delay_probability", "delay_values"}:
            _interval(row[key], f"sampler.sensor.{stage}.{key}")
        _probabilities(
            row["delay_probability"], f"sampler.sensor.{stage}.delay_probability"
        )
        if len(row["delay_probability"]) != len(row["delay_values"]):
            raise ValueError("delay support/probability sizes differ")
    _exact(sampler["task_ranges"], "sampler.task_ranges", set(TASKS))
    for task in TASKS:
        _exact(sampler["task_ranges"][task], f"sampler.task_ranges.{task}", set(STAGES))
        keys = _DOOR_RANGE_KEYS if task == "push_door_hand" else _RIGID_RANGE_KEYS
        for stage in STAGES:
            row = _exact(
                sampler["task_ranges"][task][stage],
                f"sampler.task_ranges.{task}.{stage}",
                keys,
            )
            for key, value in row.items():
                if key in {"friction", "damping", "mass"}:
                    _interval(value, f"sampler.task_ranges.{task}.{stage}.{key}")
                else:
                    _number(value, f"sampler.task_ranges.{task}.{stage}.{key}")
        for key in keys:
            values = [sampler["task_ranges"][task][stage][key] for stage in STAGES]
            if key in {"friction", "damping", "mass"}:
                lows = [float(value[0]) for value in values]
                highs = [float(value[1]) for value in values]
                if lows != sorted(lows, reverse=True) or highs != sorted(highs):
                    raise ValueError(f"{task}.{key} stage support is not monotonic")
            elif [float(value) for value in values] != sorted(
                float(value) for value in values
            ):
                raise ValueError(f"{task}.{key} stage support is not monotonic")
    _exact(
        payload["sensor"],
        "sensor",
        {"control_dt_s", "detector", "fixed_scales", "ramp", "saturation"},
    )
    _exact(
        payload["sensor"]["detector"],
        "sensor.detector",
        {"off", "on", "smoothing_alpha", "temperature"},
    )
    _exact(
        payload["sensor"]["fixed_scales"],
        "sensor.fixed_scales",
        {"force_N", "moment_Nm"},
    )
    _exact(payload["sensor"]["ramp"], "sensor.ramp", {"attack", "release"})
    _exact(
        payload["sensor"]["saturation"], "sensor.saturation", {"force_N", "moment_Nm"}
    )
    detector = payload["sensor"]["detector"]
    if (
        not 0.0
        <= _number(detector["off"], "sensor.detector.off")
        < _number(detector["on"], "sensor.detector.on")
        or not 0.0
        <= _number(detector["smoothing_alpha"], "sensor.detector.smoothing_alpha")
        <= 1.0
        or _positive(detector["temperature"], "sensor.detector.temperature") <= 0.0
    ):
        raise ValueError("sensor detector contract is invalid")
    retention = _exact(payload["retention"], "retention", set(TASKS))
    expected_metrics = {
        "push_door_hand": {"median_progress"},
        "push_box": {"final_error_m", "median_progress"},
        "move_suitcase": {"median_progress", "setdown_m"},
        "move_largebox": {"carry_error_m", "median_progress", "setdown_m"},
    }
    for task, metrics in retention.items():
        _exact(metrics, f"retention.{task}", expected_metrics[task])
        for metric, spec in metrics.items():
            _exact(spec, f"retention.{task}.{metric}", {"baseline", "direction"})
            _positive(spec["baseline"], f"retention.{task}.{metric}.baseline")
            if spec["direction"] not in {"higher", "lower"}:
                raise ValueError("retention direction must be higher or lower")
    _exact(
        payload["held_out"],
        "held_out",
        {
            "largebox_mass_2",
            "sensor_stress",
            "suitcase_mass_5_5",
            "box_mass_16_friction_1_2",
            "door_friction_10_damping_10",
        },
    )
    for name, row in payload["held_out"].items():
        _exact(row, f"held_out.{name}", {"task", "type", "values"})
        if row["task"] not in TASKS or row["type"] not in {"physical", "sensor"}:
            raise ValueError(f"held_out.{name} has invalid task/type")
        values = _require_mapping(row["values"], f"held_out.{name}.values")
        for key, value in values.items():
            if key == "delay":
                _positive_integer(value + 1, f"held_out.{name}.values.delay")
            else:
                _positive(value, f"held_out.{name}.values.{key}")


def validate_numeric_contract(payload: Mapping[str, Any]) -> None:
    """Reject fallback, ambiguous, or non-approved contract payloads."""
    _reject_nonfinite(payload)
    _require_exact_keys(payload, "contract", set(_ROOT_KEYS))
    _validate_schema(payload)
    if payload["contract_version"] != CONTRACT_VERSION:
        raise ValueError("unexpected contract_version")
    if payload["source_runtime_commit"] != SOURCE_RUNTIME_COMMIT:
        raise ValueError("unexpected source_runtime_commit")
    if payload["contract_audit_commit"] != CONTRACT_AUDIT_COMMIT:
        raise ValueError("unexpected contract_audit_commit")
    approval = _require_exact_keys(
        payload["approval"], "approval", {"date", "owner", "status"}
    )
    if approval["status"] != "approved":
        raise ValueError("numeric contract approval.status must be approved")
    if approval["owner"] != APPROVAL_OWNER or approval["date"] != APPROVAL_DATE:
        raise ValueError(
            "numeric contract approval identity does not match the frozen approval"
        )
    units = _require_mapping(payload["units"], "units")
    required_units = {
        "angles",
        "control_dt",
        "force",
        "mass",
        "moment",
        "position",
        "reward",
        "time",
    }
    if set(units) != required_units or any(
        not isinstance(unit, str) or not unit for unit in units.values()
    ):
        raise ValueError("numeric contract has missing or invalid units")
    horizons = _require_mapping(
        _require_mapping(payload["episodes"], "episodes")["horizons"],
        "episodes.horizons",
    )
    if horizons != {
        "push_door_hand": 508,
        "push_box": 792,
        "move_suitcase": 472,
        "move_largebox": 199,
    }:
        raise ValueError("episode horizons do not match the frozen contract")
    task_probability = _require_mapping(
        _require_mapping(payload["curriculum"], "curriculum")["task_probability"],
        "curriculum.task_probability",
    )
    if set(task_probability) != set(horizons) or any(
        value != 0.25 for value in task_probability.values()
    ):
        raise ValueError("task probabilities must be four equal 0.25 atoms")
    c0 = _require_mapping(payload["c0"], "c0")
    required_c0 = {
        "F_scale": 1.0,
        "M_scale": 1.0,
        "authority": 0.0,
        "control_steps": 6,
        "episode_steps": 8,
        "identity_sensor": True,
        "reward": 0.0,
        "seed_rule": "base_seed+env_id",
    }
    if c0 != required_c0:
        raise ValueError("C0 values are frozen and must not be upgraded")


@dataclass(frozen=True)
class NumericContract:
    """Validated payload plus the SHA256 recorded by checkpoints and logs."""

    payload: Mapping[str, Any]
    sha256: str

    @property
    def horizons(self) -> Mapping[str, int]:
        return self.payload["episodes"]["horizons"]


def load_numeric_contract(path: str | Path) -> NumericContract:
    source = Path(path)
    try:
        raw = source.read_bytes()
        text = raw.decode("utf-8")
        payload = json.loads(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load numeric contract {source}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("numeric contract root must be an object")
    validate_numeric_contract(payload)
    return NumericContract(payload=payload, sha256=canonical_sha256(payload))
