"""Manifest-driven Isaac Lab DirectRLEnv for Phase 4B4 C0 parity."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from isaaclab.envs import DirectRLEnv

from somaforce_cross.envs.action_history import NominalActionHistory
from somaforce_cross.envs.mismatch import (
    EpisodeParameterStore,
    PerEnvRandomStream,
    validate_env_ids,
)
from somaforce_cross.envs.observations import (
    CriticObservationBundle,
    ForceSemanticPipeline,
    build_semantic_target_bundle,
)
from somaforce_cross.envs.reset import EpisodeResetCoordinator
from somaforce_cross.envs.residual_env_cfg import SomaForceResidualEnvCfg
from somaforce_cross.envs.task_adapter import smoke_done_flags
from somaforce_cross.envs.task_adapter import stable_env_seeds
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
        self._clean_wrench = torch.zeros(self.num_envs, 2, 6, device=self.device)
        self._normalized_wrench = torch.zeros_like(self._clean_wrench)
        self._observed_wrench = torch.zeros_like(self._clean_wrench)
        self._sensor_quality = torch.zeros(self.num_envs, 2, device=self.device)
        self._contact_probability = torch.zeros_like(self._sensor_quality)
        self._source_valid = torch.zeros(
            self.num_envs, 2, device=self.device, dtype=torch.bool
        )
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

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        ids = validate_env_ids(env_ids, batch_size=self.num_envs, device=self.device)
        count = ids.numel()
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
        zeros = torch.zeros_like(source.clean_total_wrench_base_yaw)
        virtual = self.virtual_sensor(
            source.clean_total_wrench_base_yaw,
            tare.calibrated_wrench,
            tare.calibration_ready,
            torch.ones(self.num_envs, 2, device=self.device),
            source.valid.float(),
            drift_draw=zeros,
            noise_draw=zeros,
            dropout_draw=torch.ones(self.num_envs, 2, device=self.device),
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
        authority = self.authority(0)
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
        if not torch.equal(composed.a_total, self._step_a_nom):
            raise AssertionError("C0 violation: a_total is not bitwise equal to a_nom")
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
        dones = smoke_done_flags(
            nonfinite,
            signals.failure,
            self.episode_length_buf,
            signals.reference_exhausted,
            episode_length_steps=self.cfg.smoke_profile.episode_length_steps,
        )
        self.extras["success"] = signals.success.squeeze(1)
        return dones.terminated, dones.time_outs

    def _get_rewards(self) -> torch.Tensor:
        return torch.full(
            (self.num_envs,),
            self.cfg.smoke_profile.smoke_reward,
            device=self.device,
            dtype=torch.float32,
        )

    def _assemble_observations(self) -> dict[str, torch.Tensor]:
        self._ensure_current_nominal()
        with torch.inference_mode():
            policy_assembly = self.semantic_pipeline.assemble_policy_observation(
                self.wrist_history.storage,
                self.adapter.proprio(),
                self.nominal_action_history.storage,
                self.executed_action_history.storage[:, :, 0],
            )
            policy = policy_assembly.bundle.flatten()
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
            critic = CriticObservationBundle(
                policy=policy,
                object_state=self.adapter.build_object_state(),
                physics_mismatch=self.parameter_store.physics_mismatch,
                scaffold_mismatch=self.parameter_store.scaffold_mismatch,
                sensor_mismatch=self.parameter_store.sensor_mismatch,
                progress_state=progress_state,
                contact_state=contact_state,
                stability_state=stability_state,
            ).flatten()
            target = build_semantic_target_bundle(
                self._clean_wrench,
                signals.contact_truth,
                torch.tensor(self.cfg.smoke_profile.F_scale, device=self.device),
                torch.tensor(self.cfg.smoke_profile.M_scale, device=self.device),
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
            "wrist_history": self.wrist_history.storage,
            "parameter_physics": self.parameter_store.physics_mismatch,
            "parameter_scaffold": self.parameter_store.scaffold_mismatch,
            "parameter_sensor": self.parameter_store.sensor_mismatch,
            "random_seed": self.random_stream.seeds,
            "robot_root_state": self.robot.data.root_state_w,
            "robot_joint_pos": self.robot.data.joint_pos,
            "robot_joint_vel": self.robot.data.joint_vel,
            "object_root_state": self.object.data.root_state_w,
            "object_joint_pos": object_joint_pos,
            "object_joint_vel": object_joint_vel,
            "joint_target": self.action_sink.last_joint_target,
        }
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
