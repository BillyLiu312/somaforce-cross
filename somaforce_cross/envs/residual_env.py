"""Manifest-driven Isaac Lab DirectRLEnv for Phase 4B4 C0 parity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from isaaclab.envs import DirectRLEnv

from somaforce_cross.envs.action_history import NominalActionHistory
from somaforce_cross.envs.mismatch import (
    EpisodeParameterStore,
    PerEnvRandomStream,
    validate_env_ids,
)
from somaforce_cross.envs.mismatch_sampler import Phase4B5MismatchSampler
from somaforce_cross.envs.curriculum import Phase4B5Curriculum
from somaforce_cross.envs.normalizer import FixedFieldNormalizer
from somaforce_cross.envs.numeric_contract import (
    CONTRACT_VERSION,
    load_numeric_contract,
)
from somaforce_cross.envs.observations import (
    CriticObservationBundle,
    ForceSemanticPipeline,
    build_semantic_target_bundle,
)
from somaforce_cross.envs.reset import EpisodeResetCoordinator
from somaforce_cross.envs.residual_env_cfg import SomaForceResidualEnvCfg
from somaforce_cross.envs.reward_manager import (
    EpisodeMetricLog,
    EpisodeTermination,
    Phase4B5RewardManager,
    RewardOutput,
)
from somaforce_cross.envs.task_adapter import (
    TaskProgressSignals,
    smoke_done_flags,
    stable_env_seeds,
)
from somaforce_cross.envs.task_adapters.move_payload import (
    MovePayloadTaskAdapter,
)
from somaforce_cross.envs.task_adapters.push_box import PushBoxTaskAdapter
from somaforce_cross.envs.task_adapters.push_door_hand import (
    PushDoorHandTaskAdapter,
)
from somaforce_cross.force import ContactGate, WristHistoryBuffer
from somaforce_cross.residual import (
    ContactGainRamp,
    ExecutedActionHistory,
    JointMarginLimiter,
    PerJointAuthority,
    ResidualActionComposer,
    VelocityMarginLimiter,
)
from somaforce_cross.scaffold.pretrained_hdmi import (
    HDMIJointPositionActionRuntime,
    HDMIObservationHistory,
    PretrainedHDMIScaffold,
)
from somaforce_cross.scaffold.wrist_wrench_isaac import IsaacWristWrenchSource
from somaforce_cross.sensing import (
    ContactDetector,
    VirtualFTSensor,
    WristTareCalibrator,
)
from somaforce_cross.sensing.virtual_ft import VirtualFTSensorParameters


_DEFAULT_NUMERIC_CONTRACT = (
    Path(__file__).resolve().parents[2] / "configs/phase4b5_numeric_contract.json"
)
_STAGE_INDEX = {"C1": 1, "C2": 2, "C3": 3}


class EvaluationScheduleOwner:
    """Own row-local evaluation slots and park rows after exhaustion."""

    def __init__(self) -> None:
        self._queues: dict[int, list[Mapping[str, object]]] = {}
        self.mode: str | None = None
        self.schedule_sha256: str | None = None
        self.active = False

    def bind(self, schedule: Mapping[str, object], *, mode: str) -> None:
        if mode not in {"residual", "scaffold_only"}:
            raise ValueError("evaluation schedule mode is invalid")
        rows = schedule.get("rows")
        if not isinstance(rows, Mapping) or not rows:
            raise ValueError("evaluation schedule rows are required")
        num_envs = schedule.get("num_envs")
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("evaluation schedule environment count is invalid")
        expected_rows = {str(index) for index in range(num_envs)}
        self._queues = {}
        for key, values in rows.items():
            env_id = int(key)
            if not isinstance(values, list) or any(
                not isinstance(value, Mapping) for value in values
            ):
                raise ValueError("evaluation schedule row is invalid")
            self._queues[env_id] = list(values)
        if set(rows) != expected_rows:
            raise ValueError("evaluation schedule rows do not match environment batch")
        self.mode = mode
        schedule_sha256 = schedule.get("schedule_sha256")
        if not isinstance(schedule_sha256, str) or len(schedule_sha256) != 64:
            raise ValueError("evaluation schedule hash is required")
        self.schedule_sha256 = schedule_sha256
        self.active = True

    def take(self, env_id: int) -> Mapping[str, object] | None:
        if not self.active or env_id not in self._queues:
            raise ValueError("evaluation schedule row is not bound")
        if not self._queues[env_id]:
            return None
        return self._queues[env_id].pop(0)

    def pending(self, env_id: int) -> int:
        if env_id not in self._queues:
            raise ValueError("evaluation schedule row is not bound")
        return len(self._queues[env_id])

    def exhausted(self) -> bool:
        return self.active and all(not queue for queue in self._queues.values())

    def remaining(self) -> int:
        if not self.active:
            raise ValueError("evaluation schedule row is not bound")
        return sum(len(queue) for queue in self._queues.values())


class FrozenScaffoldState:
    """Own frozen inference and HDMI-compatible nominal action history."""

    def __init__(self, scaffold: PretrainedHDMIScaffold, robot: object) -> None:
        self.scaffold = scaffold.eval()
        if any(parameter.requires_grad for parameter in self.scaffold.parameters()):
            raise AssertionError(
                "frozen scaffold parameters must not require gradients"
            )
        self.robot = robot
        self.device = robot.data.joint_pos.device
        self.history = HDMIObservationHistory(
            robot.data.joint_pos.shape[0], device=self.device
        )

    def validate_reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.ndim != 1 or env_ids.dtype != torch.long:
            raise TypeError("env_ids must be a one-dimensional int64 tensor")
        if env_ids.device != self.device:
            raise ValueError("env_ids must be on the scaffold device")

    def reset(self, env_ids: torch.Tensor) -> None:
        self.validate_reset(env_ids)
        self.history.reset(
            env_ids,
            root_ang_vel=self.robot.data.root_ang_vel_b[env_ids],
            projected_gravity=self.robot.data.projected_gravity_b[env_ids],
            joint_pos=self.robot.data.joint_pos[env_ids],
            motion_length=1,
        )

    def set_motion_length(self, env_ids: torch.Tensor, length: int) -> None:
        self.history.motion_length[env_ids] = length

    def nominal_action(self, observation: object) -> torch.Tensor:
        return self.scaffold.nominal_action(observation)

    def push_nominal_action(self, action: torch.Tensor) -> None:
        self.history.push_action(action)

    def advance(self) -> None:
        self.history.update_state(
            self.robot.data.root_ang_vel_b,
            self.robot.data.projected_gravity_b,
            self.robot.data.joint_pos,
        )
        self.history.advance_reference()


class NormalizedActionSink:
    """Selected-reset wrapper around the audited normalized HDMI action sink."""

    def __init__(
        self,
        robot: object,
        *,
        action_joint_names: tuple[str, ...],
        action_scale: tuple[float, ...],
        decimation: int,
        delay: int,
        alpha: float,
    ) -> None:
        self.robot = robot
        self.device = robot.data.joint_pos.device
        self.runtime = HDMIJointPositionActionRuntime(
            robot.data.default_joint_pos,
            robot.joint_names,
            num_envs=robot.data.joint_pos.shape[0],
            decimation=decimation,
            delay=delay,
            alpha=alpha,
            action_joint_names=action_joint_names,
            action_scale=action_scale,
            device=self.device,
        )
        self.last_joint_target = torch.zeros_like(self.runtime.applied_action)

    def validate_reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.ndim != 1 or env_ids.dtype != torch.long:
            raise TypeError("env_ids must be a one-dimensional int64 tensor")
        if env_ids.device != self.device:
            raise ValueError("env_ids must be on the action-sink device")

    def reset(self, env_ids: torch.Tensor) -> None:
        self.validate_reset(env_ids)
        self.runtime.reset(env_ids)
        self.last_joint_target[env_ids] = 0.0

    def set_action(self, normalized_action: torch.Tensor) -> None:
        """Receive normalized a_total; physical scaling remains in the runtime."""
        self.runtime.set_nominal_action(normalized_action)

    def substep_target(self, substep: int) -> torch.Tensor:
        target = self.runtime.substep_target(substep)
        self.last_joint_target.copy_(target[:, self.runtime.joint_ids])
        return target


class _ScaffoldRuntimeParameterOwner:
    """Own selected delay/alpha lifecycle without changing the action-sink API."""

    def __init__(self, runtime: object) -> None:
        self.runtime = runtime
        self.device = runtime.device

    def validate(self, env_ids: torch.Tensor, scaffold_row: torch.Tensor) -> None:
        if env_ids.ndim != 1 or env_ids.dtype != torch.long:
            raise TypeError("env_ids must be a one-dimensional int64 tensor")
        if env_ids.device != self.device:
            raise ValueError("env_ids must be on the scaffold runtime device")
        if (
            not isinstance(scaffold_row, torch.Tensor)
            or scaffold_row.shape != (env_ids.numel(), 2)
            or scaffold_row.dtype != torch.float32
            or scaffold_row.device != self.device
            or not torch.isfinite(scaffold_row).all()
        ):
            raise ValueError("scaffold runtime row must be finite float32 [B,2]")
        delay = scaffold_row[:, 0]
        if not torch.equal(delay, delay.round()) or torch.any(
            (delay < 2) | (delay > 6)
        ):
            raise ValueError("scaffold delay must be an integer in [2,6]")
        if torch.any((scaffold_row[:, 1] < 0.8) | (scaffold_row[:, 1] > 1.0)):
            raise ValueError("scaffold alpha must be in [0.8,1.0]")

    def apply(self, env_ids: torch.Tensor, scaffold_row: torch.Tensor) -> None:
        self.validate(env_ids, scaffold_row)
        with torch.no_grad():
            self.runtime.delay[env_ids, 0] = scaffold_row[:, 0].to(torch.long)
            self.runtime.alpha[env_ids, 0] = scaffold_row[:, 1]

    def readback(self, env_ids: torch.Tensor) -> torch.Tensor:
        if env_ids.ndim != 1 or env_ids.dtype != torch.long:
            raise TypeError("env_ids must be a one-dimensional int64 tensor")
        if env_ids.device != self.device:
            raise ValueError("env_ids must be on the scaffold runtime device")
        return torch.stack(
            (
                self.runtime.delay[env_ids, 0].to(dtype=torch.float32),
                self.runtime.alpha[env_ids, 0],
            ),
            dim=-1,
        ).clone()


class SomaForceResidualEnv(DirectRLEnv):
    """Four-task C0 environment; external actions are raw residual probes."""

    cfg: SomaForceResidualEnvCfg

    def __init__(
        self,
        cfg: SomaForceResidualEnvCfg,
        render_mode: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(cfg, render_mode, **kwargs)
        profile = self.cfg.smoke_profile
        self.runtime_mode = profile.runtime_mode
        self.scaffold_stage = profile.scaffold_stage
        contract_path = Path(profile.numeric_contract_path or _DEFAULT_NUMERIC_CONTRACT)
        self.numeric_contract = load_numeric_contract(contract_path)
        if self.numeric_contract.payload["contract_version"] != CONTRACT_VERSION:
            raise ValueError("residual environment requires phase4b5_numeric_v2")
        self.is_c0 = self.runtime_mode == "c0"
        self.is_scaffold_only = self.runtime_mode == "scaffold_only"
        self.is_residual = self.runtime_mode == "residual"
        self.curriculum: Phase4B5Curriculum | None = None
        self.mismatch_sampler: Phase4B5MismatchSampler | None = None
        if not self.is_c0:
            if self.scaffold_stage not in _STAGE_INDEX:
                raise ValueError(
                    "non-C0 runtime requires an explicit C1, C2, or C3 stage"
                )
            if self.is_residual:
                self.curriculum = Phase4B5Curriculum(
                    self.numeric_contract, stage=self.scaffold_stage
                )
            self.mismatch_sampler = Phase4B5MismatchSampler(
                self.numeric_contract, base_seed=self.cfg.seed
            )
        artifact_dir = Path(self.cfg.artifact_dir).expanduser().resolve()
        scaffold = PretrainedHDMIScaffold.from_artifact(
            artifact_dir, device=self.device
        )
        self.task_spec = scaffold.task_spec
        if self.task_spec.object_kind == "rigid_object":
            self._configure_rigid_object_nominal()
        self.scaffold_state = FrozenScaffoldState(scaffold, self.robot)
        self.action_sink = NormalizedActionSink(
            self.robot,
            action_joint_names=self.task_spec.action_joint_names,
            action_scale=self.task_spec.action_scale,
            decimation=profile.decimation,
            delay=profile.delay,
            alpha=profile.alpha,
        )
        self.scaffold_runtime = _ScaffoldRuntimeParameterOwner(self.action_sink.runtime)
        self.wrench_source = IsaacWristWrenchSource(
            self.robot,
            artifact_identity=self.task_spec.artifact_name,
            sensor_offset_J=torch.zeros(2, 3, device=self.device),
            warmup_steps=1,
        )
        common_adapter = {
            "artifact_dir": artifact_dir,
            "scene": self.scene,
            "robot": self.robot,
            "contacts": self.contacts,
            "task_spec": self.task_spec,
            "scaffold_history": self.scaffold_state.history,
            "action_runtime": self.action_sink.runtime,
            "source_reset": self.wrench_source.reset,
        }
        if self.task_spec.task == "push_door_hand":
            self.adapter = PushDoorHandTaskAdapter(
                **common_adapter,
                door=self.object,
                mechanism_friction=profile.door_friction,
                mechanism_damping=profile.door_damping,
            )
        else:
            rigid_adapter = {
                **common_adapter,
                "rigid_object": self.object,
                "filtered_wrist_contacts": self.filtered_wrist_contacts,
            }
            if self.task_spec.task == "push_box":
                self.adapter = PushBoxTaskAdapter(**rigid_adapter)
            else:
                self.adapter = MovePayloadTaskAdapter(**rigid_adapter)
        self._warmup_golden_reset_state()

        self.tare_calibrator = WristTareCalibrator(
            self.num_envs,
            num_samples=profile.tare_num_samples,
            device=self.device,
        )
        axis_quat = torch.tensor(
            profile.virtual_ft_axis_quat,
            device=self.device,
            dtype=torch.float32,
        ).expand(self.num_envs, 2, 4)
        self.virtual_sensor = VirtualFTSensor(
            self.num_envs,
            axis_misalignment_quat=axis_quat,
            scale_error=profile.scale_error,
            additive_bias=profile.bias,
            drift_rate=profile.drift_rate,
            drift_noise_std=profile.drift_noise_std,
            white_noise_std=profile.white_noise_std,
            delay_steps=torch.full(
                (self.num_envs, 2),
                profile.sensor_delay_steps,
                device=self.device,
                dtype=torch.long,
            ),
            filter_alpha=profile.filter_alpha,
            force_saturation=profile.force_saturation,
            torque_saturation=profile.torque_saturation,
            dropout_probability=profile.dropout_probability,
            F_scale=profile.F_scale,
            M_scale=profile.M_scale,
            seed=self.cfg.seed,
            device=self.device,
        )
        self.contact_detector = ContactDetector(
            self.num_envs,
            F_scale=profile.F_scale,
            M_scale=profile.M_scale,
            on_threshold=profile.contact_on_threshold,
            off_threshold=profile.contact_off_threshold,
            probability_temperature=profile.contact_temperature,
            smoothing_alpha=profile.contact_smoothing_alpha,
            device=self.device,
            dtype=torch.float32,
        )
        self.contact_gate = ContactGate().to(self.device)
        self.contact_ramp = ContactGainRamp(
            self.num_envs,
            attack_step=profile.ramp_attack_step,
            release_step=profile.ramp_release_step,
            device=self.device,
        )
        self.wrist_history = WristHistoryBuffer(self.num_envs, device=self.device)
        self.nominal_action_history = NominalActionHistory(
            self.num_envs, device=self.device
        )
        self.executed_action_history = ExecutedActionHistory(
            self.num_envs, device=self.device
        )
        self.parameter_store = EpisodeParameterStore(self.num_envs, device=self.device)
        self.random_stream = PerEnvRandomStream(self.num_envs, device=self.device)
        self.episode_index = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._evaluation_enabled = False
        self._evaluation_schedule_owner = EvaluationScheduleOwner()
        self._evaluation_mode: str | None = None
        self._evaluation_id: str | None = None
        self._evaluation_subset: list[str] = ["parked"] * self.num_envs
        self._evaluation_parked = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._evaluation_completed: list[dict[str, object]] = []
        self.requested_physics = torch.zeros(
            self.num_envs, 39, device=self.device, dtype=torch.float32
        )
        self.requested_scaffold = torch.zeros(
            self.num_envs, 2, device=self.device, dtype=torch.float32
        )
        self.requested_sensor = torch.zeros(
            self.num_envs, 90, device=self.device, dtype=torch.float32
        )
        self.reset_coordinator = EpisodeResetCoordinator(
            parameter_store=self.parameter_store,
            random_stream=self.random_stream,
            scaffold=self.scaffold_state,
            action_sink=self.action_sink,
            reference_adapter=self.adapter,
            tare_calibrator=self.tare_calibrator,
            virtual_sensor=self.virtual_sensor,
            contact_detector=self.contact_detector,
            contact_ramp=self.contact_ramp,
            wrist_history=self.wrist_history,
            nominal_action_history=self.nominal_action_history,
            executed_action_history=self.executed_action_history,
        )
        self.reset_coordinator.reference_step = self.adapter.reference_step
        self.reset_coordinator.episode_length = self.episode_length_buf

        action_ids = self.action_sink.runtime.joint_ids
        default = self.robot.data.default_joint_pos[0, action_ids]
        scale = torch.tensor(
            self.task_spec.action_scale,
            device=self.device,
            dtype=torch.float32,
        )
        joint_limits = self.robot.data.soft_joint_pos_limits[0, action_ids]
        velocity_limits = self.robot.data.joint_vel_limits[0, action_ids]
        self.action_composer = ResidualActionComposer(
            JointMarginLimiter(
                default_joint_pos=default,
                joint_lower=joint_limits[:, 0],
                joint_upper=joint_limits[:, 1],
                action_scale=scale,
                margin=0.0,
            ),
            VelocityMarginLimiter(
                default_joint_pos=default,
                action_scale=scale,
                velocity_limit=velocity_limits,
                dt=profile.control_dt,
            ),
        ).to(self.device)
        self.authority = PerJointAuthority(device=self.device)
        self.semantic_pipeline = ForceSemanticPipeline().to(self.device).eval()
        self.normalizer: FixedFieldNormalizer | None = None
        self.reward_manager: Phase4B5RewardManager | None = None
        self.episode_termination: EpisodeTermination | None = None
        self.episode_metric_log: EpisodeMetricLog | None = None
        if self.is_residual:
            self.normalizer = FixedFieldNormalizer(self.numeric_contract)
            self.reward_manager = Phase4B5RewardManager(
                self.numeric_contract, self.task_spec.task
            )
            self.episode_termination = EpisodeTermination(
                self.numeric_contract, self.num_envs, device=self.device
            )
            self.episode_metric_log = EpisodeMetricLog(self.numeric_contract)

        self._a_nom = torch.zeros(self.num_envs, 23, device=self.device)
        self._nominal_reference_step = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self._raw_residual = torch.zeros_like(self._a_nom)
        self._step_a_nom = torch.zeros_like(self._a_nom)
        self._step_nominal_history = torch.zeros(
            self.num_envs, 23, 3, device=self.device
        )
        self._a_total = torch.zeros_like(self._a_nom)
        self._authority = torch.zeros_like(self._a_nom)
        self._delta_safe = torch.zeros_like(self._a_nom)
        self._previous_delta_safe = torch.zeros_like(self._a_nom)
        self._clean_wrench = torch.zeros(self.num_envs, 2, 6, device=self.device)
        self._normalized_wrench = torch.zeros_like(self._clean_wrench)
        self._observed_wrench = torch.zeros_like(self._clean_wrench)
        self._sensor_quality = torch.zeros(self.num_envs, 2, device=self.device)
        self._contact_probability = torch.zeros_like(self._sensor_quality)
        self._source_valid = torch.zeros(
            self.num_envs, 2, device=self.device, dtype=torch.bool
        )
        self._saturation_mask = torch.zeros_like(self._source_valid)
        self._dropout_mask = torch.zeros_like(self._source_valid)
        self._last_nonfinite = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._last_terminated = torch.zeros_like(self._last_nonfinite)
        self._last_time_outs = torch.zeros_like(self._last_nonfinite)
        self._episode_active = torch.zeros_like(self._last_nonfinite)
        self._episode_reward_steps = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._episode_return = torch.zeros(self.num_envs, device=self.device)
        self._episode_raw_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in self.numeric_contract.payload["logging"]["raw_reward_keys"]
        }
        self._episode_weighted_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in self.numeric_contract.payload["logging"]["weighted_reward_keys"]
        }
        horizon = int(
            self.numeric_contract.payload["episodes"]["horizons"][self.task_spec.task]
        )
        self._episode_wrench_history = torch.zeros(
            self.num_envs, horizon, device=self.device
        )
        self._episode_force_rate_history = torch.zeros_like(
            self._episode_wrench_history
        )
        self._episode_impulse = torch.zeros(self.num_envs, device=self.device)
        self._episode_contact_fraction = torch.zeros(self.num_envs, device=self.device)
        self._episode_contact_loss = torch.zeros(self.num_envs, device=self.device)
        self._episode_sensor_quality = torch.zeros(self.num_envs, device=self.device)
        self._episode_saturation = torch.zeros(self.num_envs, device=self.device)
        self._episode_dropout = torch.zeros(self.num_envs, device=self.device)
        self._episode_arms_residual = torch.zeros(self.num_envs, device=self.device)
        self._episode_waist_residual = torch.zeros(self.num_envs, device=self.device)
        self._episode_legs_residual = torch.zeros(self.num_envs, device=self.device)
        self._episode_semantic_entropy = torch.zeros(self.num_envs, device=self.device)
        self._episode_semantic_kl = torch.zeros(self.num_envs, device=self.device)
        self._episode_stability_margin = torch.zeros(self.num_envs, device=self.device)
        self._previous_wrench_norm = torch.zeros(self.num_envs, device=self.device)
        self._episode_family: list[str] = ["nominal"] * self.num_envs
        self._episode_nominal = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._episode_seed = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._episode_stage: list[str] = ["C1"] * self.num_envs
        self._episode_evaluation_id: list[str | None] = [None] * self.num_envs
        self._substep = 0
        self._last_observations: dict[str, torch.Tensor] = {}
        self._terminal_snapshot: dict[str, torch.Tensor] = {}

    def _setup_scene(self) -> None:
        self.robot = self.scene["robot"]
        task = self.cfg.smoke_profile.task
        self.object = (
            self.scene["door"]
            if task == "push_door_hand"
            else self.scene["rigid_object"]
        )
        self.door = self.object if task == "push_door_hand" else None
        self.contacts = self.scene["contacts"]
        self.filtered_wrist_contacts = (
            ()
            if task == "push_door_hand"
            else (
                self.scene["left_object_contacts"],
                self.scene["right_object_contacts"],
            )
        )

    def _configure_rigid_object_nominal(self) -> None:
        mass = self.task_spec.nominal_object_mass
        if mass is None:
            raise ValueError("rigid task contract must declare nominal object mass")
        view = self.object.root_physx_view
        indices = torch.arange(self.num_envs, device="cpu")
        masses = view.get_masses().clone()
        inertias = view.get_inertias().clone()
        scale = float(mass) / masses
        masses.fill_(float(mass))
        inertias *= scale.unsqueeze(-1) if inertias.ndim == 3 else scale
        view.set_masses(masses, indices)
        view.set_inertias(inertias, indices)
        materials = view.get_material_properties().clone()
        materials[..., 0] = self.cfg.smoke_profile.object_friction
        materials[..., 1] = self.cfg.smoke_profile.object_friction
        materials[..., 2] = 0.0
        view.set_material_properties(materials, indices)
        coms = view.get_coms().clone()
        view.set_coms(coms, indices)

    def _warmup_golden_reset_state(self) -> None:
        """Mirror the golden constructor's one-time pre-rewind physics step."""
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self.adapter.write_scene_reset(env_ids)
        self.scene.reset(env_ids)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(self.cfg.smoke_profile.physics_dt)

    def _sensor_parameter_rows(self, count: int) -> torch.Tensor:
        profile = self.cfg.smoke_profile
        axis = torch.tensor(
            profile.virtual_ft_axis_quat,
            device=self.device,
            dtype=torch.float32,
        ).expand(count, 2, 4)
        parameters = VirtualFTSensorParameters(
            axis_misalignment_quat=axis,
            scale_error=torch.full(
                (count, 2, 6), profile.scale_error, device=self.device
            ),
            additive_bias=torch.full((count, 2, 6), profile.bias, device=self.device),
            drift_rate=torch.full(
                (count, 2, 6), profile.drift_rate, device=self.device
            ),
            drift_noise_std=torch.full(
                (count, 2, 6), profile.drift_noise_std, device=self.device
            ),
            white_noise_std=torch.full(
                (count, 2, 6), profile.white_noise_std, device=self.device
            ),
            delay_steps=torch.full(
                (count, 2),
                profile.sensor_delay_steps,
                device=self.device,
                dtype=torch.long,
            ),
            filter_alpha=torch.full(
                (count, 2), profile.filter_alpha, device=self.device
            ),
            force_saturation=torch.full(
                (count, 2, 3), profile.force_saturation, device=self.device
            ),
            torque_saturation=torch.full(
                (count, 2, 3), profile.torque_saturation, device=self.device
            ),
            dropout_probability=torch.full(
                (count, 2), profile.dropout_probability, device=self.device
            ),
            F_scale=torch.full((count, 2), profile.F_scale, device=self.device),
            M_scale=torch.full((count, 2), profile.M_scale, device=self.device),
        )
        return parameters.flatten()

    def _snapshot_terminal(self, env_ids: torch.Tensor) -> None:
        object_joint_pos, _, _ = self._object_mechanism_state()
        self._terminal_snapshot = {
            "env_ids": env_ids.clone(),
            "terminated": self.reset_terminated[env_ids].clone(),
            "time_outs": self.reset_time_outs[env_ids].clone(),
            "root_state": self.robot.data.root_state_w[env_ids].clone(),
            "joint_pos": self.robot.data.joint_pos[env_ids].clone(),
            "object_joint": object_joint_pos[env_ids].clone(),
            "reference_step": self.adapter.reference_step[env_ids].clone(),
        }
        self.extras["terminal"] = self._terminal_snapshot

    def _object_mechanism_state(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.task_spec.object_kind == "articulation":
            return (
                self.object.data.joint_pos,
                self.object.data.joint_vel,
                self.object.data.applied_torque,
            )
        zeros = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        return zeros, zeros.clone(), zeros.clone()

    def _validate_reset_plan(
        self,
        env_ids: torch.Tensor,
        physics_mismatch: torch.Tensor,
        scaffold_mismatch: torch.Tensor,
        sensor_mismatch: torch.Tensor,
        seeds: torch.Tensor,
    ) -> None:
        self.parameter_store.validate_rows(
            env_ids, physics_mismatch, scaffold_mismatch, sensor_mismatch
        )
        VirtualFTSensorParameters.from_flat(sensor_mismatch)
        self.random_stream.validate_seeds(env_ids, seeds)
        self.scaffold_state.validate_reset(env_ids)
        self.action_sink.validate_reset(env_ids)
        self.adapter.validate_reset(env_ids)

    def _sensor_parameter_readback(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Flatten the public selected VirtualFTSensor parameter buffers."""
        limits = self.virtual_sensor.saturation_limit[env_ids]
        scales = self.virtual_sensor.normalization_scale[env_ids]
        return VirtualFTSensorParameters(
            axis_misalignment_quat=self.virtual_sensor.axis_misalignment_quat[env_ids],
            scale_error=self.virtual_sensor.scale_error[env_ids],
            additive_bias=self.virtual_sensor.additive_bias[env_ids],
            drift_rate=self.virtual_sensor.drift_rate[env_ids],
            drift_noise_std=self.virtual_sensor.drift_noise_std[env_ids],
            white_noise_std=self.virtual_sensor.white_noise_std[env_ids],
            delay_steps=self.virtual_sensor.delay_steps[env_ids],
            filter_alpha=self.virtual_sensor.filter_alpha[env_ids],
            force_saturation=limits[..., :3],
            torque_saturation=limits[..., 3:],
            dropout_probability=self.virtual_sensor.dropout_probability[env_ids],
            F_scale=scales[..., 0],
            M_scale=scales[..., 3],
        ).flatten()

    def set_curriculum_stage(self, stage: str) -> None:
        """Accept a stage selected by the external training coordinator."""
        if stage not in _STAGE_INDEX:
            raise ValueError("stage must be C1, C2, or C3")
        if not self.is_residual or self.curriculum is None:
            raise RuntimeError("only residual training accepts an external stage")
        self.curriculum.stage = stage
        self.scaffold_stage = stage

    def _reset_episode_metrics(self, env_ids: torch.Tensor) -> None:
        if (
            not self.is_residual and not self._evaluation_enabled
        ) or self.episode_termination is None:
            return
        self.episode_termination.reset(env_ids)
        self._episode_active[env_ids] = True
        self._episode_reward_steps[env_ids] = 0
        self._episode_return[env_ids] = 0.0
        self._delta_safe[env_ids] = 0.0
        self._previous_delta_safe[env_ids] = 0.0
        self._previous_wrench_norm[env_ids] = 0.0
        self._episode_wrench_history[env_ids] = 0.0
        self._episode_force_rate_history[env_ids] = 0.0
        for values in (
            self._episode_raw_sums,
            self._episode_weighted_sums,
        ):
            for value in values.values():
                value[env_ids] = 0.0
        for value in (
            self._episode_impulse,
            self._episode_contact_fraction,
            self._episode_contact_loss,
            self._episode_sensor_quality,
            self._episode_saturation,
            self._episode_dropout,
            self._episode_arms_residual,
            self._episode_waist_residual,
            self._episode_legs_residual,
            self._episode_semantic_entropy,
            self._episode_semantic_kl,
            self._episode_stability_margin,
        ):
            value[env_ids] = 0.0
        self._last_nonfinite[env_ids] = False
        self._last_terminated[env_ids] = False
        self._last_time_outs[env_ids] = False

    def bind_evaluation_schedule(
        self, schedule: Mapping[str, object], *, mode: str
    ) -> None:
        """Enable evaluation-only schedule ownership without changing training reset."""
        if self.is_c0:
            raise ValueError("C0 cannot bind an evaluation schedule")
        if self.mismatch_sampler is None:
            raise AssertionError("evaluation schedule requires the mismatch sampler")
        self._evaluation_schedule_owner.bind(schedule, mode=mode)
        self._evaluation_enabled = True
        self._evaluation_mode = mode
        self._evaluation_id = str(schedule.get("schedule_sha256"))
        if self.curriculum is None:
            self.curriculum = Phase4B5Curriculum(
                self.numeric_contract, stage=self.scaffold_stage
            )
        if self.reward_manager is None:
            self.reward_manager = Phase4B5RewardManager(
                self.numeric_contract, self.task_spec.task
            )
        if self.episode_termination is None:
            self.episode_termination = EpisodeTermination(
                self.numeric_contract, self.num_envs, device=self.device
            )
        if self.episode_metric_log is None:
            self.episode_metric_log = EpisodeMetricLog(self.numeric_contract)
        self._evaluation_completed.clear()

    def evaluation_nominal_probe(self, env_id: int, episode_index: int) -> bool:
        """Probe sampler truth without mutating simulator or evaluation state."""
        if self.mismatch_sampler is None:
            raise AssertionError("nominal probe requires the mismatch sampler")
        ids = torch.tensor([env_id], dtype=torch.long, device=self.device)
        indices = torch.tensor([episode_index], dtype=torch.long, device=self.device)
        sample = self.mismatch_sampler.sample(
            task=self.task_spec.task,
            stage=self.scaffold_stage,
            env_ids=ids,
            episode_indices=indices,
        )
        return bool(sample.nominal[0].item())

    def drain_evaluation_completions(self) -> tuple[Mapping[str, object], ...]:
        completed = tuple(self._evaluation_completed)
        self._evaluation_completed.clear()
        return completed

    def _record_completed_episodes(self, env_ids: torch.Tensor) -> None:
        if (
            not self.is_residual
            and not self._evaluation_enabled
            or self.episode_metric_log is None
            or self.episode_termination is None
            or self.curriculum is None
        ):
            return
        completed = env_ids[
            self._episode_active[env_ids]
            & (self._last_terminated[env_ids] | self._last_time_outs[env_ids])
            & ~self._evaluation_parked[env_ids]
        ]
        for env_id in completed.tolist():
            steps = int(self._episode_reward_steps[env_id].item())
            count = max(steps, 1)
            wrench = self._episode_wrench_history[env_id, :count]
            force_rate = self._episode_force_rate_history[env_id, :count]
            failure = bool(
                (
                    self.episode_termination.failure[env_id]
                    | self._last_nonfinite[env_id]
                ).item()
            )
            outcome = {
                "success": bool(
                    (
                        self.episode_termination.success[env_id]
                        & ~self.episode_termination.failure[env_id]
                        & ~self._last_nonfinite[env_id]
                    ).item()
                ),
                "failure": failure,
                "timeout": bool(self._last_time_outs[env_id].item()),
                "episode_invalid": bool(
                    self.episode_termination.episode_invalid[env_id].item()
                ),
            }
            diagnostics = {
                "p50_wrench": torch.quantile(wrench, 0.50),
                "p95_wrench": torch.quantile(wrench, 0.95),
                "p99_wrench": torch.quantile(wrench, 0.99),
                "impulse": self._episode_impulse[env_id],
                "p95_force_rate": torch.quantile(force_rate, 0.95),
                "contact_fraction": self._episode_contact_fraction[env_id] / count,
                "contact_loss": self._episode_contact_loss[env_id] / count,
                "sensor_quality": self._episode_sensor_quality[env_id] / count,
                "saturation": self._episode_saturation[env_id] / count,
                "dropout": self._episode_dropout[env_id] / count,
                "arms_residual_norm": self._episode_arms_residual[env_id] / count,
                "waist_residual_norm": self._episode_waist_residual[env_id] / count,
                "legs_residual_norm": self._episode_legs_residual[env_id] / count,
                "semantic_entropy": self._episode_semantic_entropy[env_id] / count,
                "semantic_kl": self._episode_semantic_kl[env_id] / count,
                "stability_margin": self._episode_stability_margin[env_id] / count,
            }
            self.episode_metric_log.add_episode(
                {
                    "task": self.task_spec.task,
                    "stage": self._episode_stage[env_id],
                    "family": self._episode_family[env_id],
                    "evaluation_id": self._episode_evaluation_id[env_id],
                    "nominal": bool(self._episode_nominal[env_id].item()),
                    "seed": int(self._episode_seed[env_id].item()),
                    "steps": steps,
                    "outcome": outcome,
                    "raw_reward_sums": {
                        name: values[env_id]
                        for name, values in self._episode_raw_sums.items()
                    },
                    "weighted_reward_sums": {
                        name: values[env_id]
                        for name, values in self._episode_weighted_sums.items()
                    },
                    "return": self._episode_return[env_id],
                    "diagnostics": diagnostics,
                }
            )
            if self._evaluation_enabled:
                self._evaluation_completed.append(
                    {
                        "task": self.task_spec.task,
                        "mode": self._evaluation_mode,
                        "stage": self._episode_stage[env_id],
                        "subset": self._evaluation_subset[env_id],
                        "family": self._episode_family[env_id],
                        "seed": int(self._episode_seed[env_id].item()),
                        "env_id": int(env_id),
                        "steps": steps,
                        "success": outcome["success"],
                        "failure": outcome["failure"],
                        "timeout": outcome["timeout"],
                        "invalid": outcome["episode_invalid"],
                        "return": float(self._episode_return[env_id].item()),
                        "raw_reward_sums": {
                            name: float(values[env_id].item())
                            for name, values in self._episode_raw_sums.items()
                        },
                        "weighted_reward_sums": {
                            name: float(values[env_id].item())
                            for name, values in self._episode_weighted_sums.items()
                        },
                        "diagnostics": {
                            name: float(value.item())
                            for name, value in diagnostics.items()
                        },
                    }
                )
            self.curriculum.complete_episode(
                self.task_spec.task, nominal=bool(self._episode_nominal[env_id].item())
            )

    def _set_episode_metadata(
        self,
        env_ids: torch.Tensor,
        *,
        family: tuple[str, ...],
        nominal: torch.Tensor,
        seeds: torch.Tensor,
        stage: str,
        evaluation_id: str | None,
    ) -> None:
        if not self.is_residual and not self._evaluation_enabled:
            return
        self._episode_nominal[env_ids] = nominal
        self._episode_seed[env_ids] = seeds
        for index, env_id in enumerate(env_ids.tolist()):
            self._episode_family[env_id] = family[index]
            self._episode_stage[env_id] = stage
            self._episode_evaluation_id[env_id] = evaluation_id

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        ids = validate_env_ids(env_ids, batch_size=self.num_envs, device=self.device)
        count = ids.numel()
        if not self.is_c0:
            self._reset_scaffold_only_idx(ids)
            return

        profile = self.cfg.smoke_profile
        physics_mismatch = self.adapter.nominal_physics_row().index_select(0, ids)
        scaffold_mismatch = torch.tensor(
            [profile.delay, profile.alpha],
            device=self.device,
            dtype=torch.float32,
        ).expand(count, 2)
        sensor_mismatch = self._sensor_parameter_rows(count)
        seeds = stable_env_seeds(self.cfg.seed, ids, device=self.device)
        self._validate_reset_plan(
            ids,
            physics_mismatch,
            scaffold_mismatch,
            sensor_mismatch,
            seeds,
        )
        self._snapshot_terminal(ids)
        super()._reset_idx(ids)
        self.adapter.write_scene_reset(ids)
        self.scene.write_data_to_sim()

        self.reset_coordinator.reset(
            ids,
            physics_mismatch=physics_mismatch,
            scaffold_mismatch=scaffold_mismatch,
            sensor_mismatch=sensor_mismatch,
            seeds=seeds,
            initial_wrist_frame=torch.zeros(count, 2, 14, device=self.device),
            current_a_nom=torch.zeros(count, 23, device=self.device),
        )
        self.scaffold_state.set_motion_length(ids, self.adapter.reference.length)
        self._initialize_wrist_rows(ids)
        self._refresh_nominal(ids, initialize=True)
        self._last_observations = self._assemble_observations()

    def _reset_scaffold_only_idx(self, ids: torch.Tensor) -> None:
        """Reset selected rows after contract sampling without a physics advance."""
        if self.mismatch_sampler is None:
            raise AssertionError("non-C0 runtime was not initialized")
        count = ids.numel()
        stage = self.curriculum.stage if self.is_residual else self.scaffold_stage
        if stage is None:
            raise AssertionError("non-C0 runtime has no curriculum stage")
        slots: list[Mapping[str, object] | None] = [None] * count
        staged_evaluation_subsets: list[str] = ["parked"] * count
        staged_evaluation_parked: list[bool] = [False] * count
        episode_indices = self.episode_index.index_select(0, ids)
        if self._evaluation_schedule_owner.active:
            for local, env_id in enumerate(ids.tolist()):
                slots[local] = self._evaluation_schedule_owner.take(env_id)
                if slots[local] is not None:
                    if int(slots[local]["env_id"]) != env_id:
                        raise ValueError("evaluation schedule row identity changed")
                    if slots[local]["subset"] not in {"stage", "nominal"}:
                        raise ValueError("evaluation schedule subset is invalid")
                    episode_indices[local] = int(slots[local]["episode_index"])
        sample = self.mismatch_sampler.sample(
            task=self.task_spec.task,
            stage=stage,
            env_ids=ids,
            episode_indices=episode_indices,
        )
        if self._evaluation_schedule_owner.active:
            for local, slot in enumerate(slots):
                if slot is None:
                    staged_evaluation_parked[local] = True
                    continue
                if int(sample.episode_seeds[local].item()) != int(slot["seed"]):
                    raise ValueError("evaluation schedule seed does not match sampler")
                if bool(sample.nominal[local].item()) != bool(slot["nominal"]):
                    raise ValueError(
                        "evaluation schedule nominal atom does not match sampler"
                    )
                if slot["subset"] == "nominal" and not bool(
                    sample.nominal[local].item()
                ):
                    raise ValueError("evaluation nominal slot is not sampler-authentic")
                staged_evaluation_subsets[local] = str(slot["subset"])
        # Validate every sampled/runtime row before scene, PhysX, buffer, seed,
        # or history mutation.  Requested rows remain distinct from critic rows.
        prepared = self.parameter_store.validate_rows(
            ids, sample.physics, sample.scaffold, sample.sensor
        )
        _, _, _, prepared_sensor = prepared
        sensor_parameters = VirtualFTSensorParameters.from_flat(prepared_sensor)
        self.random_stream.validate_seeds(ids, sample.episode_seeds)
        self.scaffold_state.validate_reset(ids)
        self.action_sink.validate_reset(ids)
        self.scaffold_runtime.validate(ids, sample.scaffold)
        self.adapter.validate_reset(ids)
        self.adapter.validate_runtime_parameters(ids, sample.physics)

        if count == 0:
            return

        # DirectRLEnv invokes this reset after done.  Completion must observe
        # the old row metadata before the next slot (or parked state) commits.
        self._record_completed_episodes(ids)
        if self._evaluation_schedule_owner.active:
            for local, subset in enumerate(staged_evaluation_subsets):
                self._evaluation_subset[int(ids[local].item())] = subset
            self._evaluation_parked[ids] = torch.tensor(
                staged_evaluation_parked, device=self.device, dtype=torch.bool
            )
        self._reset_episode_metrics(ids)
        self.requested_physics[ids] = sample.physics
        self.requested_scaffold[ids] = sample.scaffold
        self.requested_sensor[ids] = sample.sensor
        self._snapshot_terminal(ids)
        super()._reset_idx(ids)

        # The nominal pose is installed first.  Selected PhysX/runtime writes
        # then produce the only rows that may enter the critic parameter store.
        self.adapter.write_scene_reset(ids)
        self.scaffold_runtime.apply(ids, sample.scaffold)
        self.adapter.apply_runtime_parameters(ids, sample.physics)
        self.virtual_sensor._apply_validated_parameters(ids, sensor_parameters)
        self.scene.write_data_to_sim()

        applied_physics = self.adapter.runtime_parameter_readback(ids)
        applied_scaffold = self.scaffold_runtime.readback(ids)
        applied_sensor = self._sensor_parameter_readback(ids)
        self.parameter_store.validate_rows(
            ids, applied_physics, applied_scaffold, applied_sensor
        )
        self.reset_coordinator.reset_after_runtime_apply(
            ids,
            applied_physics=applied_physics,
            applied_scaffold=applied_scaffold,
            applied_sensor=applied_sensor,
            seeds=sample.episode_seeds,
            initial_wrist_frame=torch.zeros(count, 2, 14, device=self.device),
            current_a_nom=torch.zeros(count, 23, device=self.device),
        )
        self.scaffold_state.set_motion_length(ids, self.adapter.reference.length)
        self._initialize_wrist_rows(ids)
        self._refresh_nominal(ids, initialize=True)
        self._set_episode_metadata(
            ids,
            family=sample.family,
            nominal=sample.nominal,
            seeds=sample.episode_seeds,
            stage=stage,
            evaluation_id=self._evaluation_id or sample.evaluation_id,
        )
        self._last_observations = self._assemble_observations()
        self.episode_index[ids] += 1

    def _owned_sensor_state(self) -> dict[str, torch.Tensor]:
        return {
            "source_age": self.wrench_source.age,
            "source_valid": self.wrench_source.valid,
            "tare_bias": self.tare_calibrator.bias,
            "tare_sum": self.tare_calibrator.sample_sum,
            "tare_count": self.tare_calibrator.count,
            "tare_ready": self.tare_calibrator.ready,
            "drift": self.virtual_sensor.drift_state,
            "delay": self.virtual_sensor.delay_buffer,
            "filter": self.virtual_sensor.filter_state,
            "detector_state": self.contact_detector.contact_state,
            "detector_probability": self.contact_detector.contact_probability,
            "clean_wrench": self._clean_wrench,
            "normalized_wrench": self._normalized_wrench,
            "observed_wrench": self._observed_wrench,
            "sensor_quality": self._sensor_quality,
            "contact_probability": self._contact_probability,
            "valid": self._source_valid,
            "saturation": self._saturation_mask,
            "dropout": self._dropout_mask,
        }

    def _initialize_wrist_rows(self, env_ids: torch.Tensor) -> None:
        zero_frame = torch.zeros(
            env_ids.numel(), 2, 14, device=self.device, dtype=torch.float32
        )
        self.wrist_history.reset(env_ids, zero_frame)
        for value in (
            self._clean_wrench,
            self._normalized_wrench,
            self._observed_wrench,
            self._sensor_quality,
            self._contact_probability,
        ):
            value[env_ids] = 0
        self._source_valid[env_ids] = False
        self._saturation_mask[env_ids] = False
        self._dropout_mask[env_ids] = False

    def _update_wrist_pipeline(
        self, selected: torch.Tensor | None = None
    ) -> torch.Tensor:
        source = self.wrench_source.update_after_physics_step()
        expected = self.adapter.expected_contact()
        calibration_mask = source.valid & ~expected
        if selected is not None:
            calibration_mask &= selected[:, None]
        tare = self.tare_calibrator.update(
            source.clean_total_wrench_base_yaw, calibration_mask
        )
        if self.runtime_mode == "scaffold_only" or self.is_residual:
            draws = self.random_stream.draw_virtual_ft()
            drift_draw = draws.drift
            noise_draw = draws.noise
            dropout_draw = draws.dropout
        else:
            # The C0 trace is intentionally bitwise isolated from B5 draws.
            drift_draw = torch.zeros_like(source.clean_total_wrench_base_yaw)
            noise_draw = torch.zeros_like(source.clean_total_wrench_base_yaw)
            dropout_draw = torch.ones(self.num_envs, 2, device=self.device)
        virtual = self.virtual_sensor(
            source.clean_total_wrench_base_yaw,
            tare.calibrated_wrench,
            tare.calibration_ready,
            torch.ones(self.num_envs, 2, device=self.device),
            source.valid.float(),
            drift_draw=drift_draw,
            noise_draw=noise_draw,
            dropout_draw=dropout_draw,
        )
        sample_valid = source.valid & tare.calibration_ready
        if selected is not None:
            sample_valid &= selected[:, None]
        detector = self.contact_detector(virtual.observed_wrench_physical, sample_valid)
        self._clean_wrench.copy_(virtual.clean_total_wrench)
        self._normalized_wrench.copy_(virtual.normalized_wrench)
        self._observed_wrench.copy_(virtual.observed_wrench_physical)
        self._sensor_quality.copy_(virtual.sensor_quality)
        self._contact_probability.copy_(detector.contact_probability)
        self._source_valid.copy_(source.valid)
        self._saturation_mask.copy_(virtual.saturation_mask.any(dim=-1))
        self._dropout_mask.copy_(virtual.dropout_mask)
        return torch.cat(
            (
                virtual.normalized_wrench,
                self.adapter.wrist_twist_base_yaw(),
                detector.contact_probability.unsqueeze(-1),
                virtual.sensor_quality.unsqueeze(-1),
            ),
            dim=-1,
        )

    def _refresh_nominal(self, env_ids: torch.Tensor, *, initialize: bool) -> None:
        with torch.inference_mode():
            observation = self.adapter.build_scaffold_observation()
            nominal = self.scaffold_state.nominal_action(observation)
        if initialize:
            self.nominal_action_history.set_current(env_ids, nominal[env_ids])
        else:
            history = self.nominal_action_history.storage
            history[env_ids, :, 1:] = history[env_ids, :, :-1].clone()
            history[env_ids, :, 0] = nominal[env_ids]
        self._a_nom[env_ids] = nominal[env_ids]
        self._nominal_reference_step[env_ids] = self.adapter.reference_step[env_ids]

    def _ensure_current_nominal(self) -> None:
        stale = self._nominal_reference_step != self.adapter.reference_step
        if stale.any():
            self._refresh_nominal(
                stale.nonzero(as_tuple=False).flatten(), initialize=False
            )

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._ensure_current_nominal()
        self._raw_residual = actions.clone()
        self._step_a_nom.copy_(self._a_nom)
        self._step_nominal_history.copy_(self.nominal_action_history.storage)
        stage = _STAGE_INDEX[self.curriculum.stage] if self.is_residual else 0
        authority = self.authority(stage)
        self._authority.copy_(authority.expand(self.num_envs, -1))
        contact = self.contact_gate(
            self._contact_probability, self._sensor_quality
        ).aggregate
        contact_gain = self.contact_ramp(contact)
        action_ids = self.action_sink.runtime.joint_ids
        composed = self.action_composer(
            self._step_a_nom,
            self._raw_residual,
            authority,
            contact_gain,
            torch.ones(self.num_envs, 1, device=self.device),
            self.robot.data.joint_pos.index_select(1, action_ids),
        )
        self._previous_delta_safe.copy_(self._delta_safe)
        self._delta_safe.copy_(composed.delta_safe)
        if not self.is_residual and not torch.equal(composed.a_total, self._step_a_nom):
            raise AssertionError(
                "non-residual violation: a_total is not bitwise equal to a_nom"
            )
        self._a_total.copy_(composed.a_total)
        self.action_sink.set_action(self._a_total)
        self.scaffold_state.push_nominal_action(self._step_a_nom)
        self.executed_action_history.push(self._a_total)
        self._substep = 0

    def _apply_action(self) -> None:
        target = self.action_sink.substep_target(self._substep)
        self.robot.set_joint_position_target(target)
        self.adapter.apply_object_action()
        self._substep += 1

    def _finish_control_step(self) -> None:
        self.scaffold_state.advance()
        self.adapter.advance()
        frame = self._update_wrist_pipeline()
        self.wrist_history.push(frame)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._finish_control_step()
        object_joint_pos, object_joint_vel, _ = self._object_mechanism_state()
        tensors = (
            self.robot.data.root_state_w,
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
            self.object.data.root_state_w,
            object_joint_pos,
            object_joint_vel,
            self._a_total,
        )
        nonfinite = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        for value in tensors:
            nonfinite |= ~torch.isfinite(value).reshape(self.num_envs, -1).all(dim=1)
        signals = self.adapter.progress_signals()
        if not self.is_residual and not self._evaluation_enabled:
            dones = smoke_done_flags(
                nonfinite,
                signals.failure,
                self.episode_length_buf,
                signals.reference_exhausted,
                episode_length_steps=self.cfg.smoke_profile.episode_length_steps,
            )
            self.extras["success"] = signals.success.squeeze(1)
            return dones.terminated, dones.time_outs

        assert self.episode_termination is not None
        terminated, time_outs = self.episode_termination.update(
            task=self.task_spec.task,
            progress=signals.progress.squeeze(1),
            adapter_task_success=signals.success.squeeze(1).to(dtype=torch.bool),
            failure=signals.failure,
            nonfinite=nonfinite,
            reference_exhausted=signals.reference_exhausted,
            episode_steps=self.episode_length_buf,
            tare_ready=self.tare_calibrator.ready.all(dim=1),
        )
        self._last_nonfinite.copy_(nonfinite)
        self._last_terminated.copy_(terminated)
        self._last_time_outs.copy_(time_outs)
        self.extras["success"] = self.episode_termination.success.clone()
        self.extras["failure"] = self.episode_termination.failure.clone()
        self.extras["episode_invalid"] = (
            self.episode_termination.episode_invalid.clone()
        )
        return terminated, time_outs

    def _accumulate_episode_metrics(
        self,
        reward: RewardOutput,
        *,
        signals: TaskProgressSignals,
    ) -> None:
        if not self.is_residual and not self._evaluation_enabled:
            return

        batch = self.num_envs
        rows = torch.arange(batch, device=self.device)
        horizon = self._episode_wrench_history.shape[1]
        index = self._episode_reward_steps.clamp_max(horizon - 1)
        wrench_norm = torch.linalg.vector_norm(
            self._normalized_wrench.reshape(batch, -1), dim=-1
        )
        force_norm = torch.linalg.vector_norm(
            self._observed_wrench[..., :3], dim=-1
        ).mean(dim=-1)
        control_dt = float(self.numeric_contract.payload["sensor"]["control_dt_s"])
        force_rate = torch.abs(force_norm - self._previous_wrench_norm) / control_dt
        self._episode_wrench_history[rows, index] = wrench_norm
        self._episode_force_rate_history[rows, index] = force_rate
        self._previous_wrench_norm.copy_(force_norm)
        self._episode_impulse += force_norm * control_dt
        self._episode_contact_fraction += signals.contact_truth.mean(dim=-1)
        self._episode_contact_loss += (
            signals.expected_contact * (1.0 - signals.contact_truth)
        ).mean(dim=-1)
        self._episode_sensor_quality += self._sensor_quality.mean(dim=-1)
        self._episode_saturation += self._saturation_mask.float().mean(dim=-1)
        self._episode_dropout += self._dropout_mask.float().mean(dim=-1)
        self._episode_arms_residual += torch.linalg.vector_norm(
            self._delta_safe[:, self.authority.ARMS], dim=-1
        )
        self._episode_waist_residual += torch.linalg.vector_norm(
            self._delta_safe[:, self.authority.WAIST], dim=-1
        )
        self._episode_legs_residual += torch.linalg.vector_norm(
            self._delta_safe[:, self.authority.LEGS], dim=-1
        )
        with torch.inference_mode():
            semantic = self.semantic_pipeline(self.wrist_history.storage)
            target = build_semantic_target_bundle(
                self._clean_wrench,
                signals.contact_truth,
                torch.tensor(
                    self.numeric_contract.payload["sensor"]["fixed_scales"]["force_N"],
                    device=self.device,
                ),
                torch.tensor(
                    self.numeric_contract.payload["sensor"]["fixed_scales"][
                        "moment_Nm"
                    ],
                    device=self.device,
                ),
            )
        eps = torch.finfo(torch.float32).eps
        entropy = -0.5 * (
            (semantic.p_dir * semantic.p_dir.clamp_min(eps).log()).sum(dim=-1)
            + (semantic.p_mag * semantic.p_mag.clamp_min(eps).log()).sum(dim=-1)
        )
        kl = 0.5 * (
            (
                target.p_dir_target
                * (
                    target.p_dir_target.clamp_min(eps).log()
                    - semantic.p_dir.clamp_min(eps).log()
                )
            ).sum(dim=-1)
            + (
                target.p_mag_target
                * (
                    target.p_mag_target.clamp_min(eps).log()
                    - semantic.p_mag.clamp_min(eps).log()
                )
            ).sum(dim=-1)
        )
        self._episode_semantic_entropy += entropy
        self._episode_semantic_kl += kl
        self._episode_stability_margin += signals.stability_margin.squeeze(1)
        for name, value in reward.raw_terms.items():
            self._episode_raw_sums[name] += value
        for name, value in reward.weighted_terms.items():
            self._episode_weighted_sums[name] += value
        self._episode_return += reward.total
        self._episode_reward_steps += 1

    def _get_rewards(self) -> torch.Tensor:
        if not self.is_residual and not self._evaluation_enabled:
            return torch.full(
                (self.num_envs,),
                self.cfg.smoke_profile.smoke_reward,
                device=self.device,
                dtype=torch.float32,
            )
        assert self.reward_manager is not None
        assert self.episode_termination is not None
        signals = self.adapter.progress_signals()
        reward = self.reward_manager.compute(
            progress_delta=signals.progress_delta.squeeze(1),
            expected_contact=signals.expected_contact,
            contact_truth=signals.contact_truth,
            wrench=self._observed_wrench,
            stability_margin=signals.stability_margin.squeeze(1),
            delta_safe=self._delta_safe,
            authority=self._authority,
            previous_delta_safe=self._previous_delta_safe,
            success_latched=self.episode_termination.success,
            failure_latched=self.episode_termination.failure,
            nonfinite=self._last_nonfinite,
            terminated=self._last_terminated,
            time_outs=self._last_time_outs,
        )
        self._accumulate_episode_metrics(reward, signals=signals)
        return reward.total

    def _assemble_observations(self) -> dict[str, torch.Tensor]:
        self._ensure_current_nominal()
        with torch.inference_mode():
            policy_assembly = self.semantic_pipeline.assemble_policy_observation(
                self.wrist_history.storage,
                self.adapter.proprio(),
                self.nominal_action_history.storage,
                self.executed_action_history.storage[:, :, 0],
            )
            raw_policy = policy_assembly.bundle.flatten()
            policy = (
                raw_policy
                if not self.is_residual
                else self.normalizer.normalize_policy(raw_policy)
            )
            signals = self.adapter.progress_signals()
            phase = (
                self.adapter.reference_step.float() / self.adapter.reference.length
            ).unsqueeze(1)
            progress_state = torch.cat(
                (
                    phase,
                    signals.progress,
                    signals.progress_delta,
                    signals.success,
                ),
                dim=-1,
            )
            contact_state = torch.cat(
                (
                    self._clean_wrench.reshape(self.num_envs, 12),
                    signals.contact_truth,
                    signals.expected_contact,
                    self.adapter.support_contact_count(),
                ),
                dim=-1,
            )
            support = signals.contact_truth.new_zeros(self.num_envs, 2)
            contact_forces = self.contacts.data.net_forces_w.index_select(
                1, self.adapter.contact_support_ids
            )
            support.copy_(
                (torch.linalg.vector_norm(contact_forces, dim=-1) > 1.0).float()
            )
            stability_state = torch.cat(
                (
                    self.robot.data.root_link_pos_w[:, 2:3],
                    self.robot.data.projected_gravity_b,
                    self.robot.data.root_lin_vel_b,
                    support,
                ),
                dim=-1,
            )
            critic_fields = {
                "policy": raw_policy,
                "object_state": self.adapter.build_object_state(),
                "physics_mismatch": self.parameter_store.physics_mismatch,
                "scaffold_mismatch": self.parameter_store.scaffold_mismatch,
                "sensor_mismatch": self.parameter_store.sensor_mismatch,
                "progress_state": progress_state,
                "contact_state": contact_state,
                "stability_state": stability_state,
            }
            normalized_critic = (
                critic_fields
                if not self.is_residual
                else self.normalizer.normalize_critic(
                    critic_fields, task=self.task_spec.task
                )
            )
            critic = CriticObservationBundle(**normalized_critic).flatten()
            force_scale = (
                self.cfg.smoke_profile.F_scale
                if not self.is_residual
                else self.numeric_contract.payload["sensor"]["fixed_scales"]["force_N"]
            )
            moment_scale = (
                self.cfg.smoke_profile.M_scale
                if not self.is_residual
                else self.numeric_contract.payload["sensor"]["fixed_scales"][
                    "moment_Nm"
                ]
            )
            target = build_semantic_target_bundle(
                self._clean_wrench,
                signals.contact_truth,
                torch.tensor(force_scale, device=self.device),
                torch.tensor(moment_scale, device=self.device),
            ).flatten()
        return {"policy": policy, "critic": critic, "semantic_target": target}

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._last_observations = self._assemble_observations()
        return self._last_observations

    def environment_owned_state(self) -> dict[str, torch.Tensor]:
        """Expose selected-reset evidence without simulator-global buffers."""
        object_joint_pos, object_joint_vel, _ = self._object_mechanism_state()
        state = {
            "scaffold_root_history": self.scaffold_state.history.root_ang_vel,
            "scaffold_gravity_history": self.scaffold_state.history.projected_gravity,
            "scaffold_joint_history": self.scaffold_state.history.joint_pos,
            "scaffold_action_history": self.scaffold_state.history.previous_actions,
            "scaffold_reference": self.scaffold_state.history.reference_step,
            "adapter_reference": self.adapter.reference_step,
            "action_buffer": self.action_sink.runtime.action_buffer,
            "applied_action": self.action_sink.runtime.applied_action,
            "nominal_history": self.nominal_action_history.storage,
            "executed_history": self.executed_action_history.storage,
            "delta_safe": self._delta_safe,
            "previous_delta_safe": self._previous_delta_safe,
            "wrist_history": self.wrist_history.storage,
            "parameter_physics": self.parameter_store.physics_mismatch,
            "parameter_scaffold": self.parameter_store.scaffold_mismatch,
            "parameter_sensor": self.parameter_store.sensor_mismatch,
            "requested_physics": self.requested_physics,
            "requested_scaffold": self.requested_scaffold,
            "requested_sensor": self.requested_sensor,
            "episode_index": self.episode_index,
            "random_seed": self.random_stream.seeds,
            "scaffold_delay": self.action_sink.runtime.delay,
            "scaffold_alpha": self.action_sink.runtime.alpha,
            "robot_root_state": self.robot.data.root_state_w,
            "robot_joint_pos": self.robot.data.joint_pos,
            "robot_joint_vel": self.robot.data.joint_vel,
            "object_root_state": self.object.data.root_state_w,
            "object_joint_pos": object_joint_pos,
            "object_joint_vel": object_joint_vel,
            "joint_target": self.action_sink.last_joint_target,
        }
        for name in (
            "mechanism_friction_buffer",
            "mechanism_damping_buffer",
            "contact_target_offsets",
            "contact_target_translation",
            "contact_target_rotvec",
            "desired_load_share_error",
        ):
            value = getattr(self.adapter, name, None)
            if isinstance(value, torch.Tensor):
                state[f"adapter_{name}"] = value
        if self.is_residual and self.episode_termination is not None:
            state.update(
                {
                    "termination_success": self.episode_termination.success,
                    "termination_failure": self.episode_termination.failure,
                    "episode_invalid": self.episode_termination.episode_invalid,
                    "episode_active": self._episode_active,
                    "episode_reward_steps": self._episode_reward_steps,
                    "episode_return": self._episode_return,
                    "episode_wrench_history": self._episode_wrench_history,
                    "episode_force_rate_history": self._episode_force_rate_history,
                }
            )
            for name, values in (
                ("episode_raw", self._episode_raw_sums),
                ("episode_weighted", self._episode_weighted_sums),
            ):
                state.update({f"{name}_{key}": value for key, value in values.items()})
        state.update(self._owned_sensor_state())
        return state

    def trace_snapshot(self) -> dict[str, torch.Tensor | str]:
        """Return every field required by the independent C0 trace."""
        object_joint_pos, object_joint_vel, object_applied_effort = (
            self._object_mechanism_state()
        )
        finite = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        values = (
            self._step_a_nom,
            self._a_total,
            self.robot.data.root_state_w,
            self.robot.data.joint_pos,
            self.object.data.root_state_w,
            object_joint_pos,
        )
        for value in values:
            finite &= torch.isfinite(value).reshape(self.num_envs, -1).all(dim=1)
        return {
            "a_nom": self._step_a_nom.clone(),
            "raw_residual": self._raw_residual.clone(),
            "authority": self._authority.clone(),
            "a_total": self._a_total.clone(),
            "applied_action": self.action_sink.runtime.applied_action.clone(),
            "joint_target": self.action_sink.last_joint_target.clone(),
            "scaffold_history": self.scaffold_state.history.previous_actions.clone(),
            "nominal_history": self._step_nominal_history.clone(),
            "executed_history": self.executed_action_history.storage.clone(),
            "reference_step": self.adapter.reference_step.clone(),
            "reference_phase": self.scaffold_state.history.phase.clone(),
            "robot_root_state": self.robot.data.root_state_w.clone(),
            "robot_joint_pos": self.robot.data.joint_pos.clone(),
            "robot_joint_vel": self.robot.data.joint_vel.clone(),
            "object_root_state": self.object.data.root_state_w.clone(),
            "object_joint_pos": object_joint_pos.clone(),
            "object_joint_vel": object_joint_vel.clone(),
            "object_applied_effort": object_applied_effort.clone(),
            "env_origin": self.scene.env_origins.clone(),
            "physics_mismatch": self.parameter_store.physics_mismatch.clone(),
            "reset_seed": self.random_stream.seeds.clone(),
            "finite": finite,
            "configuration_hash": getattr(self.cfg, "configuration_hash", ""),
        }


class SomaForceDoorResidualEnv(SomaForceResidualEnv):
    """Accepted Phase 4B3B door name with legacy trace field aliases."""

    def environment_owned_state(self) -> dict[str, torch.Tensor]:
        state = super().environment_owned_state()
        state.update(
            {
                "door_root_state": state["object_root_state"],
                "door_joint_pos": state["object_joint_pos"],
                "door_joint_vel": state["object_joint_vel"],
            }
        )
        return state

    def trace_snapshot(self) -> dict[str, torch.Tensor | str]:
        snapshot = super().trace_snapshot()
        for old, new in (
            ("door_root_state", "object_root_state"),
            ("door_joint_pos", "object_joint_pos"),
            ("door_joint_vel", "object_joint_vel"),
            ("door_applied_effort", "object_applied_effort"),
        ):
            snapshot[old] = snapshot.pop(new)
        return snapshot


# Accepted Phase 4B3B helper names remain explicit aliases of generic helpers.
FrozenDoorScaffoldState = FrozenScaffoldState
NormalizedDoorActionSink = NormalizedActionSink
