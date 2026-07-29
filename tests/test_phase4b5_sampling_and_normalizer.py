from __future__ import annotations

import copy
import hashlib
import math
from collections import Counter
from pathlib import Path

import pytest
import torch

from somaforce_cross.envs.mismatch_sampler import (
    Phase4B5MismatchSampler,
    episode_seed,
)
from somaforce_cross.envs.curriculum import Phase4B5Curriculum
from somaforce_cross.envs.normalizer import FixedFieldNormalizer
from somaforce_cross.envs.numeric_contract import (
    NumericContract,
    canonical_sha256,
    load_numeric_contract,
)
from somaforce_cross.sensing.virtual_ft import VirtualFTSensorParameters


CONTRACT = load_numeric_contract(Path("configs/phase4b5_numeric_contract.json"))


def _actor_fields(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "wrist_tokens": torch.full((batch, 2, 16, 14), 8.0, dtype=torch.float32),
        "proprio": torch.full((batch, 64), 8.0, dtype=torch.float32),
        "a_nom_history": torch.full((batch, 23, 3), 20.0, dtype=torch.float32),
        "previous_a_total": torch.full((batch, 23), 20.0, dtype=torch.float32),
        "z_cross": torch.full((batch, 64), 20.0, dtype=torch.float32),
    }


def _nominal_physics(task: str, count: int) -> torch.Tensor:
    row = torch.zeros(count, 39, dtype=torch.float32)
    nominal = CONTRACT.payload["sampler"]["nominal_rows"][task]
    if task == "push_door_hand":
        row[:, 3] = nominal["friction"]
        row[:, 4] = nominal["damping"]
    else:
        row[:, 11] = nominal["mass"]
        row[:, 15:21] = torch.tensor(nominal["inertia"])
        row[:, 21] = nominal["friction"]
    return row


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()


@pytest.mark.parametrize(
    "task", ["push_door_hand", "push_box", "move_suitcase", "move_largebox"]
)
def test_physics_rows_are_full_nominal_plus_selected_subfamilies(task: str) -> None:
    sampler = Phase4B5MismatchSampler(CONTRACT, base_seed=20260729)
    ids = torch.arange(128, dtype=torch.int64)
    sample = sampler.sample(
        task=task, stage="C3", env_ids=ids, episode_indices=torch.zeros_like(ids)
    )
    baseline = _nominal_physics(task, len(ids))
    physical = torch.tensor(
        ["physical" in name or name == "all" for name in sample.family]
    )
    assert torch.equal(sample.physics[~physical], baseline[~physical])
    assert torch.equal(sample.physics[sample.nominal], baseline[sample.nominal])
    if task != "push_door_hand":
        inertial = sample.physical_subfamily_mask[:, 0]
        nominal = CONTRACT.payload["sampler"]["nominal_rows"][task]
        assert torch.allclose(
            sample.physics[inertial, 15:21],
            torch.tensor(nominal["inertia"])
            * (sample.physics[inertial, 11:12] / nominal["mass"]),
        )


def test_seed_order_invariance_held_out_and_suitcase_contact_bounds() -> None:
    sampler = Phase4B5MismatchSampler(CONTRACT, base_seed=20260729)
    ids = torch.tensor([7, 2, 9], dtype=torch.int64)
    episodes = torch.tensor([3, 5, 4], dtype=torch.int64)
    first = sampler.sample(
        task="move_suitcase", stage="C3", env_ids=ids, episode_indices=episodes
    )
    second = sampler.sample(
        task="move_suitcase",
        stage="C3",
        env_ids=ids.flip(0),
        episode_indices=episodes.flip(0),
    )
    assert torch.equal(
        first.physics[torch.argsort(ids)], second.physics[torch.argsort(ids.flip(0))]
    )
    with pytest.raises(PermissionError):
        sampler.sample(
            task="push_box",
            stage="C3",
            env_ids=torch.tensor([0]),
            episode_indices=torch.tensor([0]),
            held_out="box_mass_16_friction_1_2",
        )
    held = sampler.sample(
        task="push_box",
        stage="C3",
        env_ids=torch.tensor([0]),
        episode_indices=torch.tensor([0]),
        held_out="box_mass_16_friction_1_2",
        evaluation_id="frozen-eval",
    )
    assert held.evaluation_id == "frozen-eval"
    expected = torch.tensor(
        [16.0, 1.2],
        dtype=held.physics.dtype,
        device=held.physics.device,
    )
    assert torch.equal(held.physics[0, [11, 21]], expected)
    assert (
        CONTRACT.payload["sampler"]["task_ranges"]["move_suitcase"]["C3"]["contact_m"]
        == 0.03
    )


def test_wrist_major_round_trip_and_fixed_stateless_normalizer() -> None:
    wrench = torch.zeros(1, 2, 6)
    wrench[0, 0, :3] = 11.0
    wrench[0, 0, 3:] = 22.0
    wrench[0, 1, :3] = 33.0
    wrench[0, 1, 3:] = 44.0
    params = VirtualFTSensorParameters(
        torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]),
        wrench,
        wrench + 1,
        wrench + 2,
        wrench + 3,
        wrench + 4,
        torch.zeros(1, 2, dtype=torch.long),
        torch.zeros(1, 2),
        torch.ones(1, 2, 3),
        torch.ones(1, 2, 3),
        torch.zeros(1, 2),
        torch.ones(1, 2),
        torch.ones(1, 2),
    )
    assert torch.equal(
        VirtualFTSensorParameters.from_flat(params.flatten()).white_noise_std,
        wrench + 4,
    )
    normalizer = FixedFieldNormalizer(CONTRACT)
    original = _actor_fields()
    first = normalizer.normalize_actor(original)
    assert torch.equal(first["wrist_tokens"][..., 12:14], torch.ones(2, 2, 16, 2))
    assert torch.all(first["a_nom_history"] == 2.5) and torch.all(
        first["z_cross"] == 10.0
    )
    assert torch.all(original["wrist_tokens"] == 8.0)
    with pytest.raises(RuntimeError):
        normalizer.update(torch.zeros(1))


def test_critic_clean_wrench_uses_wrist_major_force_moment_scales() -> None:
    normalizer = FixedFieldNormalizer(CONTRACT)
    sensor = torch.zeros(1, 90)
    sensor[:, 0:8] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    sensor[:, 72:78] = 200.0
    sensor[:, 78:84] = 20.0
    sensor[:, 86:88] = 100.0
    sensor[:, 88:90] = 10.0
    clean = torch.tensor(
        [
            [
                [100.0, 100.0, 100.0, 10.0, 10.0, 10.0],
                [200.0, 200.0, 200.0, 20.0, 20.0, 20.0],
            ]
        ]
    ).reshape(1, 12)
    fields = {
        "policy": torch.zeros(1, 668),
        "object_state": torch.zeros(1, 16),
        "physics_mismatch": _nominal_physics("push_box", 1),
        "scaffold_mismatch": torch.tensor([[4.0, 0.9]]),
        "sensor_mismatch": sensor,
        "progress_state": torch.zeros(1, 4),
        "contact_state": torch.cat((clean, torch.zeros(1, 5)), dim=1),
        "stability_state": torch.zeros(1, 9),
    }
    contact_snapshot = fields["contact_state"].clone()
    output = normalizer.normalize_critic(fields, task="push_box")
    expected = torch.tensor(
        [[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]]]
    ).reshape(1, 12)
    assert torch.equal(fields["contact_state"], contact_snapshot)
    assert torch.equal(output["contact_state"][:, :12], expected)


@pytest.mark.parametrize(
    "task", ["push_door_hand", "push_box", "move_suitcase", "move_largebox"]
)
@pytest.mark.parametrize("stage", ["C1", "C2", "C3"])
def test_all_task_stage_support_and_inactive_rows(task: str, stage: str) -> None:
    sampler = Phase4B5MismatchSampler(CONTRACT, base_seed=41)
    ids = torch.arange(1024, dtype=torch.int64)
    sample = sampler.sample(
        task=task, stage=stage, env_ids=ids, episode_indices=ids % 7
    )
    baseline = _nominal_physics(task, len(ids))
    inactive = torch.tensor(
        ["physical" not in name and name != "all" for name in sample.family]
    )
    assert sample.physics.dtype == torch.float32
    assert torch.isfinite(sample.physics).all() and torch.isfinite(sample.sensor).all()
    assert torch.equal(sample.physics[inactive], baseline[inactive])
    if task != "push_door_hand":
        selected = sample.physical_subfamily_mask[:, 3]
        assert torch.equal(sample.physics[selected, 37], -sample.physics[selected, 38])
    params = VirtualFTSensorParameters.from_flat(sample.sensor)
    assert params.additive_bias.shape == (len(ids), 2, 6)
    assert torch.equal(params.F_scale, torch.full_like(params.F_scale, 100.0))
    assert torch.equal(params.M_scale, torch.full_like(params.M_scale, 10.0))


def test_seed_validation_stream_separation_readback_and_canonical_order() -> None:
    sampler = Phase4B5MismatchSampler(CONTRACT, base_seed=11)
    ids = torch.tensor([0, 3], dtype=torch.int64)
    episodes = torch.tensor([1, 1], dtype=torch.int64)
    with pytest.raises(ValueError):
        episode_seed(1, ids.to(torch.int32), episodes, episode_multiplier=1_000_003)
    with pytest.raises(ValueError):
        sampler.sample(
            task="push_box",
            stage="C1",
            env_ids=ids,
            episode_indices=torch.tensor([-1, 1]),
        )
    first = sampler.sample(
        task="push_box", stage="C2", env_ids=ids, episode_indices=episodes
    )
    changed_episode = sampler.sample(
        task="push_box", stage="C2", env_ids=ids, episode_indices=episodes + 1
    )
    assert not torch.equal(first.episode_seeds, changed_episode.episode_seeds)
    assert not torch.equal(
        sampler.per_step_uniform(ids, episodes, width=3),
        sampler.per_step_uniform(ids, episodes + 1, width=3),
    )
    golden_sampler = Phase4B5MismatchSampler(CONTRACT, base_seed=9)
    golden = golden_sampler.per_step_uniform(
        torch.tensor([0, 7], dtype=torch.int64),
        torch.tensor([0, 3], dtype=torch.int64),
        width=3,
    )
    expected_golden = torch.tensor(
        [
            [0.8655009865760803, 0.47910356521606445, 0.7159004807472229],
            [0.09284830093383789, 0.11811584234237671, 0.21064722537994385],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(golden, expected_golden)
    readback = first.physics + 1.0
    before = first.physics.clone()
    final = first.with_physx_readback(readback)
    assert torch.equal(first.physics, before) and torch.equal(final.physics, readback)
    with pytest.raises(ValueError):
        first.with_physx_readback(readback.to(torch.float64))
    payload = copy.deepcopy(CONTRACT.payload)
    payload["curriculum"]["stage_families"]["C2"] = dict(
        reversed(tuple(payload["curriculum"]["stage_families"]["C2"].items()))
    )
    reordered = NumericContract(payload=payload, sha256=canonical_sha256(payload))
    reordered_sample = Phase4B5MismatchSampler(reordered, base_seed=11).sample(
        task="push_box", stage="C2", env_ids=ids, episode_indices=episodes
    )
    assert torch.equal(first.physics, reordered_sample.physics)
    assert first.family == reordered_sample.family


def test_v2_keeps_c0_reward_pcg_sensor_and_rigid_samples_bitwise_stable() -> None:
    from somaforce_cross.envs.numeric_contract import canonical_sha256

    assert canonical_sha256(CONTRACT.payload["c0"]) == (
        "eac16c66023b4aa6e7c5205e6efdb5307b7dc30974160086abd619d2c7eff401"
    )
    assert canonical_sha256(CONTRACT.payload["reward"]) == (
        "a0346e056468a316973ec869a6e5c570e3cd8c8cfd155644bbab20885e47978b"
    )
    assert canonical_sha256(CONTRACT.payload["sampler"]["counter_rng"]) == (
        "a5b1631d91896b0cedb8a42c1d614ce655b8fa7fe3a40ca641e340ca875ae96c"
    )
    assert canonical_sha256(CONTRACT.payload["sampler"]["sensor"]) == (
        "d923096e6db1da9b77ff72031d0d5d95776fe1a0f0eea47522e7a12dd6198e0e"
    )
    assert (
        canonical_sha256(
            {
                task: CONTRACT.payload["sampler"]["nominal_rows"][task]
                for task in ("push_box", "move_suitcase", "move_largebox")
            }
        )
        == "09724bd26791448d17652f1a24a3c6cb2134a94b4be022c800b86e17c5b12c11"
    )
    sampler = Phase4B5MismatchSampler(CONTRACT, base_seed=9)
    env_ids = torch.tensor([0, 7], dtype=torch.int64)
    episode_indices = torch.tensor([0, 3], dtype=torch.int64)
    expected_physics = {
        "push_box": "9bc6a301687173f80887a84915d378fd01d7e5ef7c1286758744f98a59f9535b",
        "move_suitcase": "ed2d37d2c54a033aa4f8ff150652c50dfe7946e74cb875da1549a57573650fc8",
        "move_largebox": "36f5fc700a733744accd149a4ef38563c55947cb75d48735749234dc3ba0e78b",
    }
    for task, expected in expected_physics.items():
        sample = sampler.sample(
            task=task,
            stage="C3",
            env_ids=env_ids,
            episode_indices=episode_indices,
        )
        assert _tensor_sha256(sample.physics) == expected
        assert _tensor_sha256(sample.sensor) == (
            "4859f4c731190752c03f1149034d4bbe7d6df33c29aef8dc8b7c09c7bab7e2be"
        )


def test_door_v2_one_million_draws_have_only_runtime_supported_fields() -> None:
    sampler = Phase4B5MismatchSampler(CONTRACT, base_seed=1002)
    stage = "C3"
    total = 1_000_000
    chunk = 50_000
    names = CONTRACT.payload["sampler"]["physical_subfamilies"]["door"]
    assert names == ["mechanism", "initial_stance", "contact_target"]
    count_spec = CONTRACT.payload["sampler"]["physical_family_count"][stage]
    counts: Counter[int] = Counter()
    subfamily = torch.zeros(len(names), dtype=torch.int64)
    active_physical = 0
    for start in range(0, total, chunk):
        env_ids = torch.arange(start, start + chunk, dtype=torch.int64)
        sample = sampler.sample(
            task="push_door_hand",
            stage=stage,
            env_ids=env_ids,
            episode_indices=torch.zeros_like(env_ids),
        )
        assert sample.physical_subfamily_mask.shape == (chunk, len(names))
        assert torch.equal(
            sample.physics[:, 0:3], torch.zeros_like(sample.physics[:, 0:3])
        )
        assert torch.equal(
            sample.physics[:, 5:11], torch.zeros_like(sample.physics[:, 5:11])
        )
        physical = torch.tensor(
            ["physical" in family or family == "all" for family in sample.family]
        )
        selected = sample.physical_subfamily_mask[physical]
        active_physical += int(physical.sum())
        counts.update(selected.sum(dim=1).tolist())
        subfamily += selected.sum(dim=0).cpu()
        contact = sample.physical_subfamily_mask[:, 2]
        ranges = CONTRACT.payload["sampler"]["task_ranges"]["push_door_hand"][stage]
        if contact.any():
            assert torch.all(
                sample.physics[contact, 31:34].abs() <= ranges["contact_m"]
            )
            assert torch.all(
                sample.physics[contact, 34:37].abs()
                <= ranges["contact_rot_deg"] * torch.pi / 180.0
            )
    for count, probability in zip(
        count_spec["count"], count_spec["probability"], strict=True
    ):
        assert abs(
            counts[count] / active_physical - probability
        ) <= _frequency_tolerance(probability, active_physical)
    expected_subfamily_frequency = sum(
        count * probability
        for count, probability in zip(
            count_spec["count"], count_spec["probability"], strict=True
        )
    ) / len(names)
    for value in subfamily.tolist():
        assert value > 0
        assert abs(value / active_physical - expected_subfamily_frequency) <= (
            _frequency_tolerance(expected_subfamily_frequency, active_physical)
        )


def test_door_normalizer_rejects_runtime_deferred_fields_and_uses_contact_support() -> (
    None
):
    normalizer = FixedFieldNormalizer(CONTRACT)
    ranges = CONTRACT.payload["sampler"]["task_ranges"]["push_door_hand"]["C3"]
    physics = _nominal_physics("push_door_hand", 1)
    physics[:, 22] = ranges["object_m"]
    physics[:, 27] = ranges["object_yaw_deg"] * torch.pi / 180.0
    physics[:, 28] = ranges["stance_m"]
    physics[:, 30] = ranges["stance_yaw_deg"] * torch.pi / 180.0
    physics[:, 31] = ranges["contact_m"]
    physics[:, 34] = ranges["contact_rot_deg"] * torch.pi / 180.0
    fields = {
        "policy": torch.zeros(1, 668),
        "object_state": torch.zeros(1, 16),
        "physics_mismatch": physics,
        "scaffold_mismatch": torch.tensor([[4.0, 0.9]]),
        "sensor_mismatch": torch.zeros(1, 90),
        "progress_state": torch.zeros(1, 4),
        "contact_state": torch.zeros(1, 17),
        "stability_state": torch.zeros(1, 9),
    }
    output = normalizer.normalize_critic(fields, task="push_door_hand")
    assert torch.equal(
        output["physics_mismatch"][:, [22, 27, 28, 30, 31, 34]],
        torch.ones(1, 6),
    )
    for deferred_index in (0, 5):
        bad = dict(fields)
        bad["physics_mismatch"] = physics.clone()
        bad["physics_mismatch"][:, deferred_index] = 1.0
        with pytest.raises(ValueError, match="runtime-deferred"):
            normalizer.normalize_critic(bad, task="push_door_hand")

    v1_state = dict(normalizer.state_dict())
    v1_state["contract_version"] = "phase4b5_numeric_v1"
    with pytest.raises(ValueError):
        normalizer.load_state_dict(v1_state)


def test_all_held_out_rows_are_evaluation_only() -> None:
    sampler = Phase4B5MismatchSampler(CONTRACT, base_seed=29)
    ids = torch.tensor([0], dtype=torch.int64)
    for name, specification in CONTRACT.payload["held_out"].items():
        task = specification["task"]
        with pytest.raises(PermissionError):
            sampler.sample(
                task=task, stage="C3", env_ids=ids, episode_indices=ids, held_out=name
            )
        sample = sampler.sample(
            task=task,
            stage="C3",
            env_ids=ids,
            episode_indices=ids,
            held_out=name,
            evaluation_id="evaluation",
        )
        assert sample.family == ("evaluation",)
        assert not sample.nominal.item()
        assert sample.evaluation_id == "evaluation"
        family = "door" if task == "push_door_hand" else "rigid"
        assert sample.physical_subfamily_mask.shape == (
            1,
            len(CONTRACT.payload["sampler"]["physical_subfamilies"][family]),
        )


def _frequency_tolerance(probability: float, count: int) -> float:
    return max(1.0e-3, 8.0 * math.sqrt(probability * (1.0 - probability) / count))


def _float32_pair_key(pair: object) -> tuple[float, ...]:
    return tuple(torch.tensor(pair, dtype=torch.float32).tolist())


def test_one_million_draw_frequency_contract_is_memory_bounded() -> None:
    sampler = Phase4B5MismatchSampler(CONTRACT, base_seed=1001)
    stage = "C2"
    task = "push_box"
    total = 1_000_000
    chunk = 50_000
    nominal_count = 0
    family_counts: Counter[str] = Counter()
    physical_counts: Counter[int] = Counter()
    scaffold_counts: Counter[tuple[float, float]] = Counter()
    delay_counts: Counter[int] = Counter()
    active_physical = active_scaffold = active_sensor = 0
    for start in range(0, total, chunk):
        ids = torch.arange(start, start + chunk, dtype=torch.int64)
        sample = sampler.sample(
            task=task, stage=stage, env_ids=ids, episode_indices=torch.zeros_like(ids)
        )
        nominal_count += int(sample.nominal.sum())
        family_counts.update(name for name in sample.family if name != "nominal")
        physical = torch.tensor(
            ["physical" in name or name == "all" for name in sample.family]
        )
        scaffold = torch.tensor(
            ["scaffold" in name or name == "all" for name in sample.family]
        )
        sensor = torch.tensor(
            ["sensor" in name or name == "all" for name in sample.family]
        )
        active_physical += int(physical.sum())
        active_scaffold += int(scaffold.sum())
        active_sensor += int(sensor.sum())
        physical_counts.update(
            sample.physical_subfamily_mask[physical].sum(-1).tolist()
        )
        scaffold_counts.update(
            _float32_pair_key(row) for row in sample.scaffold[scaffold].tolist()
        )
        delay_counts.update(
            VirtualFTSensorParameters.from_flat(sample.sensor)
            .delay_steps[sensor]
            .flatten()
            .tolist()
        )
    nominal_probability = CONTRACT.payload["curriculum"]["nominal_atom"][stage]
    assert abs(nominal_count / total - nominal_probability) <= _frequency_tolerance(
        nominal_probability, total
    )
    family_probability = CONTRACT.payload["curriculum"]["stage_families"][stage]
    for name, probability in family_probability.items():
        assert abs(
            family_counts[name] / (total - nominal_count) - probability
        ) <= _frequency_tolerance(probability, total - nominal_count)
    count_spec = CONTRACT.payload["sampler"]["physical_family_count"][stage]
    for count, probability in zip(
        count_spec["count"], count_spec["probability"], strict=True
    ):
        assert abs(
            physical_counts[count] / active_physical - probability
        ) <= _frequency_tolerance(probability, active_physical)
    scaffold_spec = CONTRACT.payload["sampler"]["scaffold"][stage]
    for pair, probability in zip(
        scaffold_spec["pairs"], scaffold_spec["probability"], strict=True
    ):
        key = _float32_pair_key(pair)
        assert abs(
            scaffold_counts[key] / active_scaffold - probability
        ) <= _frequency_tolerance(probability, active_scaffold)
    sensor_spec = CONTRACT.payload["sampler"]["sensor"][stage]
    for delay, probability in zip(
        sensor_spec["delay_values"], sensor_spec["delay_probability"], strict=True
    ):
        assert abs(
            delay_counts[delay] / (2 * active_sensor) - probability
        ) <= _frequency_tolerance(probability, 2 * active_sensor)


def test_cuda_sampler_and_normalizer_when_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = torch.device("cuda", torch.cuda.current_device())
    sampler = Phase4B5MismatchSampler(CONTRACT, base_seed=9)
    env_ids = torch.arange(8, device=device, dtype=torch.int64)
    episode_indices = torch.zeros(8, device=device, dtype=torch.int64)
    sample = sampler.sample(
        task="move_largebox",
        stage="C3",
        env_ids=env_ids,
        episode_indices=episode_indices,
    )
    assert sample.physics.device == device and sample.sensor.device == device
    cpu_per_step = sampler.per_step_uniform(
        env_ids.cpu(), episode_indices.cpu(), width=7
    )
    cuda_per_step = sampler.per_step_uniform(env_ids, episode_indices, width=7)
    assert torch.equal(cpu_per_step, cuda_per_step.cpu())
    fields = _actor_fields(8)
    fields = {key: value.to(device) for key, value in fields.items()}
    snapshot = {key: value.clone() for key, value in fields.items()}
    output = FixedFieldNormalizer(CONTRACT).normalize_actor(fields)
    assert all(
        value.device == device
        and value.dtype == torch.float32
        and torch.isfinite(value).all()
        for value in output.values()
    )
    assert all(torch.equal(fields[key], snapshot[key]) for key in fields)


@pytest.mark.parametrize(
    "task", ["push_door_hand", "push_box", "move_suitcase", "move_largebox"]
)
def test_critic_prefix_all_fields_and_input_rejection(task: str) -> None:
    normalizer = FixedFieldNormalizer(CONTRACT)
    policy = torch.full((2, 668), 2.0, dtype=torch.float32)
    sensor = torch.zeros(2, 90, dtype=torch.float32)
    sensor[:, 0:8] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    sensor[:, 20:32] = torch.tensor([100.0, 100.0, 100.0, 10.0, 10.0, 10.0] * 2)
    sensor[:, 32:44] = sensor[:, 20:32]
    sensor[:, 44:56] = sensor[:, 20:32]
    sensor[:, 56:68] = sensor[:, 20:32]
    sensor[:, 72:78] = 100.0
    sensor[:, 78:84] = 10.0
    sensor[:, 86:88] = 100.0
    sensor[:, 88:90] = 10.0
    fields = {
        "policy": policy,
        "object_state": torch.ones(2, 16),
        "physics_mismatch": _nominal_physics(task, 2),
        "scaffold_mismatch": torch.tensor([[4.0, 0.9], [4.0, 0.9]]),
        "sensor_mismatch": sensor,
        "progress_state": torch.ones(2, 4),
        "contact_state": torch.ones(2, 17),
        "stability_state": torch.ones(2, 9),
    }
    snapshot = {key: value.clone() for key, value in fields.items()}
    output = normalizer.normalize_critic(fields, task=task)
    assert torch.equal(output["policy"], normalizer.normalize_policy(policy))
    assert all(torch.equal(fields[key], snapshot[key]) for key in fields)
    assert all(torch.isfinite(value).all() for value in output.values())
    bad = dict(fields)
    bad["policy"] = policy.to(torch.float64)
    with pytest.raises(ValueError):
        normalizer.normalize_critic(bad, task=task)
    bad = dict(fields)
    bad["object_state"] = torch.ones(2, 16, dtype=torch.float32, device="meta")
    with pytest.raises(ValueError):
        normalizer.normalize_critic(bad, task=task)
    bad = dict(fields)
    bad["contact_state"] = fields["contact_state"].clone()
    bad["contact_state"][0, 0] = float("inf")
    with pytest.raises(ValueError):
        normalizer.normalize_critic(bad, task=task)


def _window_metrics(
    *, success: float, failure: float, retention: float
) -> dict[str, dict[str, object]]:
    return {
        task: {
            "success": success,
            "failure": failure,
            "nominal_retention": retention,
            "stage_episodes": 256,
            "nominal_episodes": 128,
        }
        for task in ("push_door_hand", "push_box", "move_suitcase", "move_largebox")
    }


def test_curriculum_cadence_promotion_rollback_and_atomic_restore() -> None:
    curriculum = Phase4B5Curriculum(CONTRACT)
    good_c1 = _window_metrics(success=0.75, failure=0.05, retention=0.95)
    assert (
        curriculum.evaluate_window(transitions=5_000_000, metrics=good_c1).stage == "C1"
    )
    assert (
        curriculum.evaluate_window(transitions=5_250_000, metrics=good_c1).stage == "C1"
    )
    assert curriculum.evaluate_window(transitions=5_500_000, metrics=good_c1).promoted
    assert curriculum.stage == "C2"
    bad_c2 = _window_metrics(success=0.49, failure=0.14, retention=0.89)
    assert not curriculum.evaluate_window(
        transitions=5_750_000, metrics=bad_c2
    ).rolled_back
    assert curriculum.evaluate_window(transitions=6_000_000, metrics=bad_c2).rolled_back
    assert curriculum.stage == "C1"
    before = curriculum.state_dict()
    malformed = dict(before)
    malformed["episode_counters"] = {"push_box": -1}
    with pytest.raises(ValueError):
        curriculum.load_state_dict(malformed)
    assert curriculum.state_dict() == before
    with pytest.raises(ValueError):
        curriculum.evaluate_window(transitions=6_000_000, metrics=good_c1)
    missing = _window_metrics(success=0.75, failure=0.05, retention=0.95)
    missing.pop("push_box")
    with pytest.raises(ValueError):
        curriculum.evaluate_window(transitions=6_250_000, metrics=missing)
