"""Manifest-driven DirectRLEnv configuration for Phase 4B4 C0 parity."""

from __future__ import annotations

from pathlib import Path

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from somaforce_cross.scaffold.pretrained_hdmi import get_hdmi_task_spec
from somaforce_cross.scaffold.pretrained_hdmi_isaac import make_scene_cfg


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASK = "push_door_hand"
DEFAULT_ARTIFACT = REPO_ROOT / "artifacts/scaffolds/hdmi_push_door_hand/v1"
# Accepted Phase 4B3B constant remains an explicit alias of the generic default.
DOOR_ARTIFACT = DEFAULT_ARTIFACT
SUPPORTED_TASKS = (
    "push_door_hand",
    "push_box",
    "move_suitcase",
    "move_largebox",
)


@configclass
class C0SmokeProfile:
    """Owner-approved values used only by bounded C0 parity smokes."""

    task: str = DEFAULT_TASK
    physics_dt: float = 0.005
    control_dt: float = 0.02
    decimation: int = 4
    action_dim: int = 23
    policy_dim: int = 668
    critic_dim: int = 845
    semantic_target_dim: int = 31
    delay: int = 4
    alpha: float = 0.9
    door_friction: float = 0.3
    door_damping: float = 0.55
    object_friction: float = 0.5
    F_scale: float = 1.0
    M_scale: float = 1.0
    tare_num_samples: int = 1
    contact_on_threshold: float = 0.8
    contact_off_threshold: float = 0.4
    contact_temperature: float = 0.2
    contact_smoothing_alpha: float = 0.5
    virtual_ft_axis_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale_error: float = 0.0
    bias: float = 0.0
    drift_rate: float = 0.0
    drift_noise_std: float = 0.0
    white_noise_std: float = 0.0
    sensor_delay_steps: int = 0
    filter_alpha: float = 0.0
    force_saturation: float = 1.0e6
    torque_saturation: float = 1.0e6
    dropout_probability: float = 0.0
    ramp_attack_step: float = 0.1
    ramp_release_step: float = 0.1
    episode_length_steps: int = 8
    smoke_control_steps: int = 6
    smoke_reward: float = 0.0
    parity_atol: float = 1.0e-5
    parity_rtol: float = 1.0e-5
    root_height_failure: float = 0.45
    # C0 is the immutable parity path.  The lifecycle path has no implicit
    # curriculum stage or numeric-contract fallback.
    runtime_mode: str = "c0"
    scaffold_stage: str | None = None
    numeric_contract_path: str | None = None


@configclass
class SomaForceResidualEnvCfg(DirectRLEnvCfg):
    """Exact Isaac Lab 0.47.2 config for the four-task C0 integration gate."""

    seed: int = 20260727
    decimation: int = 4
    episode_length_s: float = 0.16
    action_space: int = 23
    observation_space: int = 668
    state_space: int = 845
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=4, device="cuda:0")
    scene = make_scene_cfg(
        DEFAULT_ARTIFACT,
        1,
        get_hdmi_task_spec(DEFAULT_TASK),
    )
    artifact_dir: str = str(DEFAULT_ARTIFACT)
    configuration_hash: str = ""
    smoke_profile: C0SmokeProfile = C0SmokeProfile()

    def __post_init__(self) -> None:
        profile = self.smoke_profile
        if profile.task not in SUPPORTED_TASKS:
            raise ValueError(f"unsupported Phase 4B4 task: {profile.task!r}")
        if profile.physics_dt * profile.decimation != profile.control_dt:
            raise ValueError("C0 physics/control dt and decimation are inconsistent")
        if (profile.action_dim, profile.policy_dim, profile.critic_dim) != (
            23,
            668,
            845,
        ):
            raise ValueError("C0 action/policy/critic dimensions are frozen")
        if profile.semantic_target_dim != 31:
            raise ValueError("C0 semantic target dimension must be 31")
        if profile.runtime_mode not in ("c0", "scaffold_only"):
            raise ValueError("runtime_mode must be 'c0' or 'scaffold_only'")
        if profile.runtime_mode == "c0":
            if (
                profile.scaffold_stage is not None
                or profile.numeric_contract_path is not None
            ):
                raise ValueError("c0 mode must not declare a B5 runtime contract")
        else:
            if profile.scaffold_stage not in ("C1", "C2", "C3"):
                raise ValueError(
                    "scaffold_only requires an explicit C1, C2, or C3 stage"
                )
            if (
                not isinstance(profile.numeric_contract_path, str)
                or not profile.numeric_contract_path
            ):
                raise ValueError("scaffold_only requires an explicit v2 contract path")
        self.decimation = profile.decimation
        self.episode_length_s = profile.episode_length_steps * profile.control_dt
        self.action_space = profile.action_dim
        self.observation_space = profile.policy_dim
        self.state_space = profile.critic_dim
        self.sim.dt = profile.physics_dt
        self.sim.render_interval = profile.decimation


# Accepted Phase 4B3B API names remain aliases of the single generic config.
DoorC0SmokeProfile = C0SmokeProfile
SomaForceDoorResidualEnvCfg = SomaForceResidualEnvCfg
