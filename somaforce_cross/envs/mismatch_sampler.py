"""Contract-driven, order-invariant Phase 4B5 episode parameter sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from somaforce_cross.envs.numeric_contract import NumericContract, STAGES, TASKS
from somaforce_cross.sensing.virtual_ft import VirtualFTSensorParameters


def episode_seed(
    base_seed: int,
    env_ids: torch.Tensor,
    episode_indices: torch.Tensor,
    *,
    episode_multiplier: int,
) -> torch.Tensor:
    """Return the frozen episode seed independently for every environment row."""
    if (
        not isinstance(env_ids, torch.Tensor)
        or not isinstance(episode_indices, torch.Tensor)
        or env_ids.dtype != torch.int64
        or episode_indices.dtype != torch.int64
        or env_ids.shape != episode_indices.shape
        or env_ids.ndim != 1
        or env_ids.device != episode_indices.device
        or torch.any(env_ids < 0)
        or torch.any(episode_indices < 0)
    ):
        raise ValueError(
            "env_ids and episode_indices must be nonnegative matching [B] int64 tensors on one device"
        )
    return int(base_seed) + env_ids + int(episode_multiplier) * episode_indices


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _categorical(uniform: torch.Tensor, probabilities: list[float]) -> torch.Tensor:
    thresholds = torch.tensor(
        probabilities, device=uniform.device, dtype=torch.float32
    ).cumsum(0)
    return torch.searchsorted(thresholds, uniform.contiguous()).clamp_max(
        len(probabilities) - 1
    )


@dataclass(frozen=True)
class MismatchSample:
    """Complete requested-applied rows and their selected physical subfamilies."""

    physics: torch.Tensor
    scaffold: torch.Tensor
    sensor: torch.Tensor
    episode_seeds: torch.Tensor
    family: tuple[str, ...]
    nominal: torch.Tensor
    physical_subfamily_mask: torch.Tensor
    evaluation_id: str | None = None

    def with_physx_readback(self, physics_readback: torch.Tensor) -> "MismatchSample":
        """Promote validated PhysX readback to the final critic physics row."""
        if (
            not isinstance(physics_readback, torch.Tensor)
            or physics_readback.shape != self.physics.shape
            or physics_readback.dtype != torch.float32
            or physics_readback.device != self.physics.device
            or not torch.isfinite(physics_readback).all()
        ):
            raise ValueError("PhysX readback must be finite float32 [B,39]")
        return MismatchSample(
            physics=physics_readback.clone(),
            scaffold=self.scaffold,
            sensor=self.sensor,
            episode_seeds=self.episode_seeds,
            family=self.family,
            nominal=self.nominal,
            physical_subfamily_mask=self.physical_subfamily_mask,
            evaluation_id=self.evaluation_id,
        )


class Phase4B5MismatchSampler:
    """Pure PyTorch sampler with no mutable curriculum or episode state."""

    def __init__(self, contract: NumericContract, *, base_seed: int) -> None:
        self.contract = contract
        self.base_seed = int(base_seed)
        self.data = _mapping(contract.payload["sampler"], "sampler")
        self._random = _mapping(self.data["seed_offsets"], "sampler.seed_offsets")
        self._rng = _mapping(self.data["counter_rng"], "sampler.counter_rng")

    def _uniform(self, seeds: torch.Tensor, width: int, stream: int) -> torch.Tensor:
        slots = torch.arange(width, device=seeds.device, dtype=torch.int64)
        counter_slot = slots[None] + stream
        counter = (
            seeds[:, None]
            + counter_slot
            * (counter_slot + int(self._rng["slot_bias"]))
            * int(self._rng["slot_multiplier"])
            + int(self._rng["counter_bias"])
        ) & int(self._rng["word_mask"])
        state = (
            counter * int(self._rng["state_multiplier"])
            + int(self._rng["state_increment"])
        ) & int(self._rng["word_mask"])
        shift = (state >> int(self._rng["state_shift"])) + int(
            self._rng["state_shift_bias"]
        )
        word = (((state >> shift) ^ state) * int(self._rng["output_multiplier"])) & int(
            self._rng["word_mask"]
        )
        word = ((word >> int(self._rng["output_shift"])) ^ word) & int(
            self._rng["word_mask"]
        )
        return (word >> int(self._rng["mantissa_shift"])).to(torch.float32) / float(
            self._rng["float_divisor"]
        )

    def _symmetric(
        self, seeds: torch.Tensor, width: int, bound: float, stream: int
    ) -> torch.Tensor:
        return (self._uniform(seeds, width, stream) * 2.0 - 1.0) * bound

    def _uniform_range(
        self, seeds: torch.Tensor, width: int, bounds: list[float], stream: int
    ) -> torch.Tensor:
        return bounds[0] + self._uniform(seeds, width, stream) * (bounds[1] - bounds[0])

    def _log_uniform(
        self, seeds: torch.Tensor, bounds: list[float], stream: int
    ) -> torch.Tensor:
        lower = torch.tensor(bounds[0], dtype=torch.float32, device=seeds.device)
        ratio = torch.tensor(
            bounds[1] / bounds[0], dtype=torch.float32, device=seeds.device
        )
        return torch.exp(
            torch.log(lower)
            + self._uniform(seeds, 1, stream).squeeze(1) * torch.log(ratio)
        )

    def _nominal_physics(
        self, task: str, count: int, device: torch.device
    ) -> torch.Tensor:
        row = torch.zeros(count, 39, dtype=torch.float32, device=device)
        nominal = _mapping(self.data["nominal_rows"], "sampler.nominal_rows")[task]
        if task == "push_door_hand":
            row[:, 3] = float(nominal["friction"])
            row[:, 4] = float(nominal["damping"])
        else:
            row[:, 11] = float(nominal["mass"])
            row[:, 15:21] = torch.tensor(
                nominal["inertia"], dtype=torch.float32, device=device
            )
            row[:, 21] = float(nominal["friction"])
        return row

    def _nominal_sensor(
        self, count: int, device: torch.device
    ) -> VirtualFTSensorParameters:
        fixed = _mapping(self.contract.payload["sensor"], "sensor")
        saturation = _mapping(fixed["saturation"], "sensor.saturation")
        scales = _mapping(fixed["fixed_scales"], "sensor.fixed_scales")
        zero_wrench = torch.zeros(count, 2, 6, dtype=torch.float32, device=device)
        return VirtualFTSensorParameters(
            axis_misalignment_quat=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
            .expand(count, 2, 4)
            .clone(),
            scale_error=zero_wrench.clone(),
            additive_bias=zero_wrench.clone(),
            drift_rate=zero_wrench.clone(),
            drift_noise_std=zero_wrench.clone(),
            white_noise_std=zero_wrench.clone(),
            delay_steps=torch.zeros(count, 2, dtype=torch.long, device=device),
            filter_alpha=torch.zeros(count, 2, dtype=torch.float32, device=device),
            force_saturation=torch.full(
                (count, 2, 3),
                float(saturation["force_N"]),
                dtype=torch.float32,
                device=device,
            ),
            torque_saturation=torch.full(
                (count, 2, 3),
                float(saturation["moment_Nm"]),
                dtype=torch.float32,
                device=device,
            ),
            dropout_probability=torch.zeros(
                count, 2, dtype=torch.float32, device=device
            ),
            F_scale=torch.full(
                (count, 2), float(scales["force_N"]), dtype=torch.float32, device=device
            ),
            M_scale=torch.full(
                (count, 2),
                float(scales["moment_Nm"]),
                dtype=torch.float32,
                device=device,
            ),
        )

    def _family_masks(
        self, stage: str, seeds: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[str, ...]]:
        curriculum = _mapping(self.contract.payload["curriculum"], "curriculum")
        nominal_probability = float(
            _mapping(curriculum["nominal_atom"], "curriculum.nominal_atom")[stage]
        )
        table = _mapping(curriculum["stage_families"], "curriculum.stage_families")[
            stage
        ]
        names = tuple(curriculum["family_order"])
        draws = self._uniform(seeds, 2, 1)
        nominal = draws[:, 0] < nominal_probability
        family_index = _categorical(draws[:, 1], [float(table[name]) for name in names])
        return nominal, tuple(
            "nominal" if bool(nominal[i]) else names[int(family_index[i])]
            for i in range(seeds.numel())
        )

    def _physical_subfamilies(
        self, task: str, stage: str, seeds: torch.Tensor, active: torch.Tensor
    ) -> torch.Tensor:
        physical = _mapping(
            self.data["physical_family_count"], "sampler.physical_family_count"
        )[stage]
        count_index = _categorical(
            self._uniform(seeds, 1, 11).squeeze(1),
            [float(value) for value in physical["probability"]],
        )
        counts = torch.tensor(physical["count"], device=seeds.device, dtype=torch.long)[
            count_index
        ]
        family_name = "door" if task == "push_door_hand" else "rigid"
        names = _mapping(
            self.data["physical_subfamilies"], "sampler.physical_subfamilies"
        )[family_name]
        scores = self._uniform(seeds, len(names), 13)
        ranks = torch.argsort(scores, dim=1)
        mask = torch.zeros(
            seeds.numel(), len(names), dtype=torch.bool, device=seeds.device
        )
        for number in range(1, len(names) + 1):
            row_mask = active & (counts >= number)
            if row_mask.any():
                mask[row_mask, ranks[row_mask, number - 1]] = True
        return mask

    def _sample_door(
        self, stage: str, seeds: torch.Tensor, row: torch.Tensor, mask: torch.Tensor
    ) -> None:
        ranges = _mapping(
            _mapping(self.data["task_ranges"], "sampler.task_ranges")["push_door_hand"],
            "door ranges",
        )[stage]
        degrees = torch.pi / 180.0
        if mask[:, 0].any():
            row[:, 3] = torch.where(
                mask[:, 0], self._log_uniform(seeds, ranges["friction"], 21), row[:, 3]
            )
            row[:, 4] = torch.where(
                mask[:, 0], self._log_uniform(seeds, ranges["damping"], 22), row[:, 4]
            )
        if mask[:, 1].any():
            row[mask[:, 1], 22:25] = self._symmetric(
                seeds, 3, float(ranges["object_m"]), 38
            )[mask[:, 1]]
            row[mask[:, 1], 27] = self._symmetric(
                seeds, 1, float(ranges["object_yaw_deg"]) * degrees, 41
            )[mask[:, 1], 0]
            row[mask[:, 1], 28:30] = self._symmetric(
                seeds, 2, float(ranges["stance_m"]), 42
            )[mask[:, 1]]
            row[mask[:, 1], 30] = self._symmetric(
                seeds, 1, float(ranges["stance_yaw_deg"]) * degrees, 44
            )[mask[:, 1], 0]
        if mask[:, 2].any():
            row[mask[:, 2], 31:34] = self._symmetric(
                seeds, 3, float(ranges["contact_m"]), 45
            )[mask[:, 2]]
            row[mask[:, 2], 34:37] = self._symmetric(
                seeds, 3, float(ranges["contact_rot_deg"]) * degrees, 48
            )[mask[:, 2]]

    def _sample_rigid(
        self,
        task: str,
        stage: str,
        seeds: torch.Tensor,
        row: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        ranges = _mapping(
            _mapping(self.data["task_ranges"], "sampler.task_ranges")[task],
            "rigid ranges",
        )[stage]
        nominal = _mapping(self.data["nominal_rows"], "sampler.nominal_rows")[task]
        degrees = torch.pi / 180.0
        if mask[:, 0].any():
            mass = self._uniform_range(seeds, 1, ranges["mass"], 51).squeeze(1)
            row[:, 11] = torch.where(mask[:, 0], mass, row[:, 11])
            nominal_inertia = torch.tensor(
                nominal["inertia"], dtype=torch.float32, device=seeds.device
            )
            row[mask[:, 0], 12:15] = self._symmetric(
                seeds, 3, float(ranges["com_m"]), 52
            )[mask[:, 0]]
            row[mask[:, 0], 15:21] = nominal_inertia * (
                row[mask[:, 0], 11:12] / float(nominal["mass"])
            )
        if mask[:, 1].any():
            row[:, 21] = torch.where(
                mask[:, 1],
                self._uniform_range(seeds, 1, ranges["friction"], 55).squeeze(1),
                row[:, 21],
            )
        if mask[:, 2].any():
            row[mask[:, 2], 22:24] = self._symmetric(
                seeds, 2, float(ranges["object_xy_m"]), 56
            )[mask[:, 2]]
            row[mask[:, 2], 27] = self._symmetric(
                seeds, 1, float(ranges["object_yaw_deg"]) * degrees, 58
            )[mask[:, 2], 0]
            row[mask[:, 2], 28:30] = self._symmetric(
                seeds, 2, float(ranges["stance_m"]), 59
            )[mask[:, 2]]
            row[mask[:, 2], 30] = self._symmetric(
                seeds, 1, float(ranges["stance_yaw_deg"]) * degrees, 61
            )[mask[:, 2], 0]
        if mask[:, 3].any():
            row[mask[:, 3], 31:34] = self._symmetric(
                seeds, 3, float(ranges["contact_m"]), 62
            )[mask[:, 3]]
            row[mask[:, 3], 34:37] = self._symmetric(
                seeds, 3, float(ranges["contact_rot_deg"]) * degrees, 65
            )[mask[:, 3]]
            e = self._symmetric(
                seeds, 1, float(self._stage_value(stage, "load_share")), 68
            ).squeeze(1)
            row[mask[:, 3], 37] = e[mask[:, 3]]
            row[mask[:, 3], 38] = -e[mask[:, 3]]

    def _stage_value(self, stage: str, key: str) -> float:
        return float(
            _mapping(self.data["stage_constants"], "sampler.stage_constants")[stage][
                key
            ]
        )

    def _sample_scaffold(self, stage: str, seeds: torch.Tensor) -> torch.Tensor:
        specification = _mapping(self.data["scaffold"], "sampler.scaffold")[stage]
        index = _categorical(
            self._uniform(seeds, 1, 71).squeeze(1),
            [float(value) for value in specification["probability"]],
        )
        return torch.tensor(
            specification["pairs"], dtype=torch.float32, device=seeds.device
        )[index]

    def _axis_quaternion(
        self, seeds: torch.Tensor, max_degrees: float, stream: int
    ) -> torch.Tensor:
        axis = self._symmetric(seeds, 3, 1.0, stream)
        axis = axis / torch.clamp_min(
            torch.linalg.vector_norm(axis, dim=-1, keepdim=True), 1.0e-6
        )
        half_angle = (
            self._uniform(seeds, 1, stream + 3).squeeze(1)
            * max_degrees
            * torch.pi
            / 360.0
        )
        return torch.cat(
            (
                torch.cos(half_angle).unsqueeze(1),
                axis * torch.sin(half_angle).unsqueeze(1),
            ),
            dim=1,
        )

    def _sample_sensor(self, stage: str, seeds: torch.Tensor) -> torch.Tensor:
        spec = _mapping(
            _mapping(self.data["sensor"], "sampler.sensor")[stage], f"sensor.{stage}"
        )
        params = self._nominal_sensor(seeds.numel(), seeds.device)
        scale = torch.zeros_like(params.scale_error)
        bias = torch.zeros_like(params.additive_bias)
        drift = torch.zeros_like(params.drift_rate)
        random_walk = torch.zeros_like(params.drift_noise_std)
        white = torch.zeros_like(params.white_noise_std)
        for target, force_key, moment_key, stream in (
            (scale, "scale", "scale", 81),
            (bias, "bias_force", "bias_moment", 93),
            (drift, "drift_force", "drift_moment", 105),
            (random_walk, "random_walk_force", "random_walk_moment", 117),
            (white, "white_force", "white_moment", 129),
        ):
            target[..., :3] = self._uniform_range(
                seeds, 6, spec[force_key], stream
            ).reshape(-1, 2, 3)
            target[..., 3:6] = self._uniform_range(
                seeds, 6, spec[moment_key], stream + 6
            ).reshape(-1, 2, 3)
        delay_index = _categorical(
            self._uniform(seeds, 2, 141),
            [float(value) for value in spec["delay_probability"]],
        )
        delay_values = torch.tensor(
            spec["delay_values"], dtype=torch.long, device=seeds.device
        )
        axis = torch.stack(
            (
                self._axis_quaternion(seeds, float(spec["axis_deg"][1]), 143),
                self._axis_quaternion(seeds, float(spec["axis_deg"][1]), 147),
            ),
            dim=1,
        )
        saturation_force = self._uniform_range(
            seeds, 6, spec["saturation_force"], 151
        ).reshape(-1, 2, 3)
        saturation_moment = self._uniform_range(
            seeds, 6, spec["saturation_moment"], 157
        ).reshape(-1, 2, 3)
        return VirtualFTSensorParameters(
            axis_misalignment_quat=axis,
            scale_error=scale,
            additive_bias=bias,
            drift_rate=drift,
            drift_noise_std=random_walk,
            white_noise_std=white,
            delay_steps=delay_values[delay_index],
            filter_alpha=self._uniform_range(seeds, 2, spec["filter"], 163),
            force_saturation=saturation_force,
            torque_saturation=saturation_moment,
            dropout_probability=self._uniform_range(seeds, 2, spec["dropout"], 165),
            F_scale=params.F_scale,
            M_scale=params.M_scale,
        ).flatten()

    def per_step_uniform(
        self,
        env_ids: torch.Tensor,
        episode_indices: torch.Tensor,
        *,
        width: int,
    ) -> torch.Tensor:
        """Expose the isolated per-step stream without storing simulator state."""
        if not isinstance(width, int) or width <= 0:
            raise ValueError("width must be a positive integer")
        seeds = episode_seed(
            self.base_seed,
            env_ids,
            episode_indices,
            episode_multiplier=int(self._random["episode_multiplier"]),
        )
        return self._uniform(seeds + int(self._random["per_step"]), width, 181)

    def sample(
        self,
        *,
        task: str,
        stage: str,
        env_ids: torch.Tensor,
        episode_indices: torch.Tensor,
        evaluation_id: str | None = None,
        held_out: str | None = None,
    ) -> MismatchSample:
        if task not in TASKS or stage not in STAGES:
            raise ValueError("task/stage is outside the Phase 4B5 contract")
        seeds = episode_seed(
            self.base_seed,
            env_ids,
            episode_indices,
            episode_multiplier=int(self._random["episode_multiplier"]),
        )
        if torch.any(seeds < 0):
            raise ValueError("derived episode seeds must be nonnegative")
        if held_out is not None:
            return self._held_out(task, seeds, held_out, evaluation_id)
        count = seeds.numel()
        physical_seed = seeds + int(self._random["physical"])
        scaffold_seed = seeds + int(self._random["scaffold"])
        sensor_seed = seeds + int(self._random["sensor"])
        nominal, family = self._family_masks(stage, physical_seed)
        active_physical = torch.tensor(
            ["physical" in name or name == "all" for name in family],
            device=seeds.device,
        )
        subfamily = self._physical_subfamilies(
            task, stage, physical_seed, active_physical
        )
        physics = self._nominal_physics(task, count, seeds.device)
        if task == "push_door_hand":
            self._sample_door(stage, physical_seed, physics, subfamily)
        else:
            self._sample_rigid(task, stage, physical_seed, physics, subfamily)
        nominal_scaffold = _mapping(
            self.data["nominal_scaffold"], "sampler.nominal_scaffold"
        )
        scaffold = (
            torch.tensor(
                [float(nominal_scaffold["delay"]), float(nominal_scaffold["alpha"])],
                device=seeds.device,
                dtype=torch.float32,
            )
            .expand(count, -1)
            .clone()
        )
        sampled_scaffold = self._sample_scaffold(stage, scaffold_seed)
        active_scaffold = torch.tensor(
            ["scaffold" in name or name == "all" for name in family],
            device=seeds.device,
        )
        scaffold[active_scaffold] = sampled_scaffold[active_scaffold]
        sensor = self._nominal_sensor(count, seeds.device).flatten()
        sampled_sensor = self._sample_sensor(stage, sensor_seed)
        active_sensor = torch.tensor(
            ["sensor" in name or name == "all" for name in family], device=seeds.device
        )
        sensor[active_sensor] = sampled_sensor[active_sensor]
        return MismatchSample(
            physics, scaffold, sensor, seeds, family, nominal, subfamily
        )

    def _held_out(
        self, task: str, seeds: torch.Tensor, held_out: str, evaluation_id: str | None
    ) -> MismatchSample:
        if not isinstance(evaluation_id, str) or not evaluation_id:
            raise PermissionError(
                "held-out conditions require a non-empty evaluation ID"
            )
        table = _mapping(self.contract.payload["held_out"], "held_out")
        if held_out not in table:
            raise ValueError("unknown held-out evaluation condition")
        item = _mapping(table[held_out], f"held_out.{held_out}")
        if item["task"] != task:
            raise ValueError("held-out condition does not belong to task")
        count = seeds.numel()
        physics = self._nominal_physics(task, count, seeds.device)
        sensor = self._nominal_sensor(count, seeds.device).flatten()
        nominal_scaffold = _mapping(
            self.data["nominal_scaffold"], "sampler.nominal_scaffold"
        )
        scaffold = (
            torch.tensor(
                [float(nominal_scaffold["delay"]), float(nominal_scaffold["alpha"])],
                dtype=torch.float32,
                device=seeds.device,
            )
            .expand(count, -1)
            .clone()
        )
        values = _mapping(item["values"], "held_out values")
        if item["type"] == "physical":
            if task == "push_door_hand":
                for key, index in (("friction", 3), ("damping", 4)):
                    if key in values:
                        physics[:, index] = float(values[key])
            else:
                if "mass" in values:
                    nominal = _mapping(self.data["nominal_rows"], "nominal rows")[task]
                    mass = float(values["mass"])
                    physics[:, 11] = mass
                    physics[:, 15:21] = torch.tensor(
                        nominal["inertia"], device=seeds.device
                    ) * (mass / float(nominal["mass"]))
                if "friction" in values:
                    physics[:, 21] = float(values["friction"])
        else:
            params = VirtualFTSensorParameters.from_flat(sensor)
            scale = torch.full_like(params.scale_error, float(values["scale"]))
            angle = float(values["axis_deg"])
            axis = torch.stack(
                (
                    self._axis_quaternion(seeds, angle, 171),
                    self._axis_quaternion(seeds, angle, 175),
                ),
                dim=1,
            )
            sensor = VirtualFTSensorParameters(
                axis,
                scale,
                params.additive_bias,
                params.drift_rate,
                params.drift_noise_std,
                params.white_noise_std,
                torch.full_like(params.delay_steps, int(values["delay"])),
                torch.full_like(params.filter_alpha, float(values["filter"])),
                torch.full_like(
                    params.force_saturation, float(values["force_saturation"])
                ),
                torch.full_like(
                    params.torque_saturation, float(values["torque_saturation"])
                ),
                torch.full_like(params.dropout_probability, float(values["dropout"])),
                params.F_scale,
                params.M_scale,
            ).flatten()
        family = tuple("evaluation" for _ in range(count))
        family_size = len(
            _mapping(self.data["physical_subfamilies"], "sampler.physical_subfamilies")[
                "door" if task == "push_door_hand" else "rigid"
            ]
        )
        return MismatchSample(
            physics,
            scaffold,
            sensor,
            seeds,
            family,
            torch.zeros(count, dtype=torch.bool, device=seeds.device),
            torch.zeros(count, family_size, dtype=torch.bool, device=seeds.device),
            evaluation_id,
        )
