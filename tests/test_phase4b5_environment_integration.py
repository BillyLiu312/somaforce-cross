from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch


SOURCE_PATH = Path("somaforce_cross/envs/residual_env.py")
CONFIG_PATH = Path("somaforce_cross/envs/residual_env_cfg.py")


def _method_source(name: str) -> str:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    environment = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SomaForceResidualEnv"
    )
    method = next(
        node
        for node in environment.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    return ast.get_source_segment(source, method) or ""


def test_environment_loads_contract_and_isolates_c0_from_phase4_outputs() -> None:
    init = _method_source("__init__")
    observations = _method_source("_assemble_observations")
    rewards = _method_source("_get_rewards")
    dones = _method_source("_get_dones")
    config = CONFIG_PATH.read_text(encoding="utf-8")

    assert '("c0", "scaffold_only", "residual")' in config
    assert 'elif profile.runtime_mode == "scaffold_only"' in config
    assert '"residual requires an explicit C1, C2, or C3 stage"' in config
    assert "load_numeric_contract(contract_path)" in init
    assert 'self.is_scaffold_only = self.runtime_mode == "scaffold_only"' in init
    assert 'self.is_residual = self.runtime_mode == "residual"' in init
    assert "FixedFieldNormalizer(self.numeric_contract)" in init
    assert "Phase4B5RewardManager(" in init
    assert "EpisodeTermination(" in init
    assert "EpisodeMetricLog(self.numeric_contract)" in init
    assert "if self.is_residual:" in init
    assert "self._evaluation_enabled = False" in init
    assert "if not self.is_residual" in observations
    assert "self.normalizer.normalize_policy(raw_policy)" in observations
    assert "self.normalizer.normalize_critic(" in observations
    assert "if not self.is_residual and not self._evaluation_enabled:" in rewards
    assert "self.reward_manager.compute(" in rewards
    assert "if not self.is_residual and not self._evaluation_enabled:" in dones
    assert "self.episode_termination.update(" in dones


def test_environment_actor_output_stays_deployable_and_critic_only_fields_are_separate() -> (
    None
):
    observations = _method_source("_assemble_observations")
    policy_prefix = observations[
        observations.index("policy_assembly =") : observations.index("signals =")
    ]

    assert "raw_policy = policy_assembly.bundle.flatten()" in policy_prefix
    assert "self.normalizer.normalize_policy(raw_policy)" in policy_prefix
    for forbidden in (
        "build_object_state",
        "physics_mismatch",
        "scaffold_mismatch",
        "sensor_mismatch",
        "contact_truth",
    ):
        assert forbidden not in policy_prefix
    critic_prefix = observations[
        observations.index("critic_fields =") : observations.index(
            "normalized_critic ="
        )
    ]
    assert '"policy": raw_policy' in critic_prefix
    assert '"policy": policy' not in critic_prefix
    assert "contact_truth" in observations


def test_selected_reset_uses_current_external_stage_and_never_advances_simulation() -> (
    None
):
    reset = _method_source("_reset_scaffold_only_idx")
    setter = _method_source("set_curriculum_stage")
    pre_step = _method_source("_pre_physics_step")

    assert "self.curriculum.stage if self.is_residual else self.scaffold_stage" in reset
    assert "stage=stage" in reset
    assert "self._record_completed_episodes(ids)" in reset
    assert "self._reset_episode_metrics(ids)" in reset
    assert "self._set_episode_metadata(" in reset
    assert "self.sim.step" not in reset
    assert "self.sim.forward" not in reset
    assert "if not self.is_residual or self.curriculum is None:" in setter
    assert "self.curriculum.stage = stage" in setter
    assert "self.scaffold_stage = stage" in setter
    assert 'self.runtime_mode == "scaffold_only"' not in setter
    assert "_STAGE_INDEX[self.curriculum.stage] if self.is_residual else 0" in pre_step
    assert "if not self.is_residual and not torch.equal" in pre_step


def test_episode_recording_uses_every_contract_declared_group_and_selected_reset() -> (
    None
):
    record = _method_source("_record_completed_episodes")
    reset = _method_source("_reset_episode_metrics")
    accumulate = _method_source("_accumulate_episode_metrics")

    for key in (
        "raw_reward_sums",
        "weighted_reward_sums",
        "diagnostics",
        "episode_invalid",
        "semantic_entropy",
        "semantic_kl",
    ):
        assert key in record or key in accumulate
    assert "not self.is_residual and not self._evaluation_enabled" in reset
    assert "self.episode_termination is None" in reset
    assert "self.episode_termination.reset(env_ids)" in reset
    assert "value[env_ids] = 0.0" in reset
    assert "if not self.is_residual and not self._evaluation_enabled:" in accumulate
    assert "self._episode_reward_steps += 1" in accumulate


def test_evaluation_schedule_owner_advances_each_row_after_auto_reset() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "EvaluationScheduleOwner"
    )
    methods = {
        node.name: ast.get_source_segment(source, node) or ""
        for node in owner.body
        if isinstance(node, ast.FunctionDef)
    }
    reset = _method_source("_reset_scaffold_only_idx")
    record = _method_source("_record_completed_episodes")
    assert "def take" in methods["take"]
    assert "pop(0)" in methods["take"]
    assert "def exhausted" in methods["exhausted"]
    assert "def remaining" in methods["remaining"]
    assert "self._evaluation_schedule_owner.take(env_id)" in reset
    assert "self._evaluation_parked[ids]" in reset
    assert "~self._evaluation_parked[env_ids]" in record
    assert reset.index("self._record_completed_episodes(ids)") < reset.index(
        "self._evaluation_subset[int(ids[local].item())] = subset"
    )
    assert reset.index("self._record_completed_episodes(ids)") < reset.index(
        "self._evaluation_parked[ids] = torch.tensor("
    )


def test_evaluation_reset_records_previous_slot_before_committing_next_slot() -> None:
    reset = _method_source("_reset_scaffold_only_idx")

    assert "staged_evaluation_subsets" in reset
    assert "staged_evaluation_parked" in reset
    assert reset.index("self._record_completed_episodes(ids)") < reset.index(
        "for local, subset in enumerate(staged_evaluation_subsets):"
    )
    assert reset.index("self._record_completed_episodes(ids)") < reset.index(
        "self._reset_episode_metrics(ids)"
    )
    assert reset.index(
        "self._evaluation_subset[int(ids[local].item())] = subset"
    ) < reset.index("self._reset_episode_metrics(ids)")


def test_evaluation_exhausted_row_records_final_episode_before_parking() -> None:
    reset = _method_source("_reset_scaffold_only_idx")
    record = _method_source("_record_completed_episodes")

    assert "staged_evaluation_parked[local] = True" in reset
    assert reset.index("self._record_completed_episodes(ids)") < reset.index(
        "staged_evaluation_parked, device=self.device, dtype=torch.bool"
    )
    assert "& ~self._evaluation_parked[env_ids]" in record


def test_evaluation_completion_records_preserve_business_fields() -> None:
    record = _method_source("_record_completed_episodes")
    for field in (
        '"task": self.task_spec.task',
        '"mode": self._evaluation_mode',
        '"stage": self._episode_stage[env_id]',
        '"subset": self._evaluation_subset[env_id]',
        '"family": self._episode_family[env_id]',
        '"invalid": outcome["episode_invalid"]',
        '"raw_reward_sums": {',
        '"weighted_reward_sums": {',
        '"diagnostics": {',
        '"acceptance_diagnostics": {',
        '"transition_count": transition_count',
        '"contact_bearing_residual_fraction": (',
    ):
        assert field in record
    source = SOURCE_PATH.read_text(encoding="utf-8")
    environment = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "SomaForceResidualEnv"
    )
    methods = [
        node
        for node in environment.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_accumulate_episode_metrics", "_reset_episode_metrics"}
    ]
    probe_node = ast.ClassDef(
        name="_Probe", bases=[], keywords=[], body=methods, decorator_list=[]
    )

    def target_stub(clean: torch.Tensor, *_: object) -> object:
        batch = clean.shape[0]
        return SimpleNamespace(
            p_dir_target=torch.full((batch, 13), 1 / 13),
            p_mag_target=torch.full((batch, 5), 1 / 5),
        )

    namespace = {
        "torch": torch,
        "RewardOutput": object,
        "TaskProgressSignals": object,
        "build_semantic_target_bundle": target_stub,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[probe_node], type_ignores=[])),
            str(SOURCE_PATH),
            "exec",
        ),
        namespace,
    )
    probe_type = namespace["_Probe"]

    class _Termination:
        def __init__(self) -> None:
            self.ids: torch.Tensor | None = None

        def reset(self, ids: torch.Tensor) -> None:
            self.ids = ids.clone()

    def make_probe(*, residual: bool, delta: torch.Tensor) -> object:
        value = probe_type()
        value.is_residual, value._evaluation_enabled, value.device, value.num_envs = (
            residual,
            True,
            torch.device("cpu"),
            3,
        )
        value.numeric_contract = SimpleNamespace(
            payload={
                "sensor": {
                    "control_dt_s": 0.02,
                    "fixed_scales": {"force_N": 1.0, "moment_Nm": 1.0},
                }
            }
        )
        value._delta_safe = delta.clone()
        value._previous_delta_safe = torch.ones_like(delta)
        value._normalized_wrench = torch.zeros(3, 2, 6)
        value._observed_wrench = torch.zeros(3, 2, 6)
        value._sensor_quality = torch.ones(3, 2)
        value._saturation_mask = torch.zeros(3, 2, dtype=torch.bool)
        value._dropout_mask = torch.zeros(3, 2, dtype=torch.bool)
        value._episode_reward_steps = torch.zeros(3, dtype=torch.long)
        value._episode_wrench_history = torch.zeros(3, 4)
        value._episode_force_rate_history = torch.zeros(3, 4)
        for name in (
            "_previous_wrench_norm",
            "_episode_impulse",
            "_episode_contact_fraction",
            "_episode_contact_loss",
            "_episode_sensor_quality",
            "_episode_saturation",
            "_episode_dropout",
            "_episode_arms_residual",
            "_episode_waist_residual",
            "_episode_legs_residual",
            "_episode_semantic_entropy",
            "_episode_semantic_kl",
            "_episode_stability_margin",
            "_episode_return",
            "_evaluation_residual_norm_mass",
            "_evaluation_contact_residual_norm_mass",
        ):
            setattr(value, name, torch.zeros(3))
        value._evaluation_transition_count = torch.zeros(3, dtype=torch.long)
        value._episode_raw_sums = {"x": torch.zeros(3)}
        value._episode_weighted_sums = {"x": torch.zeros(3)}
        value.authority = SimpleNamespace(ARMS=[0], WAIST=[1], LEGS=list(range(2, 23)))
        value.semantic_pipeline = lambda _: SimpleNamespace(
            p_dir=torch.full((3, 13), 1 / 13), p_mag=torch.full((3, 5), 1 / 5)
        )
        value.wrist_history = SimpleNamespace(storage=torch.zeros(3, 2, 16, 14))
        value._clean_wrench = torch.zeros(3, 2, 6)
        value.episode_termination = _Termination()
        value._episode_active = torch.zeros(3, dtype=torch.bool)
        value._last_nonfinite = torch.ones(3, dtype=torch.bool)
        value._last_terminated = torch.ones(3, dtype=torch.bool)
        value._last_time_outs = torch.ones(3, dtype=torch.bool)
        return value

    delta = torch.zeros(3, 23)
    delta[0, :2] = torch.tensor([3.0, 4.0])
    delta[1, 0] = 2.0
    delta[2, 0] = 10.0
    probe = make_probe(residual=True, delta=delta)
    signals = SimpleNamespace(
        contact_truth=torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]),
        expected_contact=torch.zeros(3, 2),
        stability_margin=torch.ones(3, 1),
    )
    reward = SimpleNamespace(
        raw_terms={"x": torch.ones(3)},
        weighted_terms={"x": torch.ones(3)},
        total=torch.ones(3),
    )
    probe._accumulate_episode_metrics(reward, signals=signals)
    assert torch.equal(
        probe._evaluation_residual_norm_mass, torch.tensor([5.0, 2.0, 10.0])
    )
    assert torch.equal(
        probe._evaluation_contact_residual_norm_mass, torch.tensor([5.0, 0.0, 10.0])
    )
    assert torch.equal(
        probe._evaluation_transition_count, torch.ones(3, dtype=torch.long)
    )
    assert torch.equal(
        probe._episode_reward_steps, torch.ones(3, dtype=torch.long)
    ) and torch.equal(probe._episode_raw_sums["x"], torch.ones(3))
    before = {
        name: getattr(probe, name).clone()
        for name in (
            "_evaluation_residual_norm_mass",
            "_evaluation_contact_residual_norm_mass",
            "_evaluation_transition_count",
            "_delta_safe",
            "_episode_reward_steps",
        )
    }
    probe._reset_episode_metrics(torch.tensor([0, 2]))
    assert torch.equal(probe.episode_termination.ids, torch.tensor([0, 2]))
    for value in before.values():
        assert torch.equal(
            value[1],
            getattr(probe, next(name for name, old in before.items() if old is value))[
                1
            ],
        )
    scaffold = make_probe(residual=False, delta=torch.zeros(3, 23))
    scaffold._accumulate_episode_metrics(reward, signals=signals)
    assert (
        torch.equal(scaffold._evaluation_residual_norm_mass, torch.zeros(3))
        and torch.equal(scaffold._evaluation_contact_residual_norm_mass, torch.zeros(3))
        and torch.equal(
            scaffold._evaluation_transition_count, torch.ones(3, dtype=torch.long)
        )
    )


def test_evaluation_schedule_rejects_seed_nominal_and_exhaustion_mutations() -> None:
    bind = _method_source("bind_evaluation_schedule")
    probe = _method_source("evaluation_nominal_probe")
    reset = _method_source("_reset_scaffold_only_idx")

    assert "evaluation schedule requires the mismatch sampler" in bind
    assert "if self.is_c0:" in bind
    assert "self._evaluation_schedule_owner.bind(schedule, mode=mode)" in bind
    assert "self._evaluation_enabled = True" in bind
    assert "sample = self.mismatch_sampler.sample(" in probe
    assert "evaluation schedule seed does not match sampler" in reset
    assert "evaluation schedule nominal atom does not match sampler" in reset
    assert "evaluation nominal slot is not sampler-authentic" in reset
