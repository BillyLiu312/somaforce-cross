"""Door-only DirectRLEnv configuration for the Phase 4B3 C0 smoke."""

from __future__ import annotations

from pathlib import Path

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from somaforce_cross.scaffold.pretrained_hdmi import get_hdmi_task_spec
from somaforce_cross.scaffold.pretrained_hdmi_isaac import make_scene_cfg


REPO_ROOT = Path(__file__).resolve().parents[2]
DOOR_ARTIFACT = REPO_ROOT / "artifacts/scaffolds/hdmi_push_door_hand/v1"


@configclass
class DoorC0SmokeProfile:
    """Owner-approved values used only by the bounded Phase 4B3 smoke."""

    task: str = "push_door_hand"
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


@configclass
class SomaForceDoorResidualEnvCfg(DirectRLEnvCfg):
    """Exact Isaac Lab 0.47.2 config for the door C0 integration gate."""

    seed: int = 20260727
    decimation: int = 4
    episode_length_s: float = 0.16
    action_space: int = 23
    observation_space: int = 668
    state_space: int = 845
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=4, device="cuda:0")
    scene = make_scene_cfg(
        DOOR_ARTIFACT,
        1,
        get_hdmi_task_spec("push_door_hand"),
    )
    artifact_dir: str = str(DOOR_ARTIFACT)
    configuration_hash: str = ""
    smoke_profile: DoorC0SmokeProfile = DoorC0SmokeProfile()

    def __post_init__(self) -> None:
        profile = self.smoke_profile
        if profile.task != "push_door_hand":
            raise ValueError("SomaForceDoorResidualEnvCfg supports push_door_hand only")
        if profile.physics_dt * profile.decimation != profile.control_dt:
            raise ValueError(
                "door C0 physics/control dt and decimation are inconsistent"
            )
        if (profile.action_dim, profile.policy_dim, profile.critic_dim) != (
            23,
            668,
            845,
        ):
            raise ValueError("door C0 action/policy/critic dimensions are frozen")
        if profile.semantic_target_dim != 31:
            raise ValueError("door C0 semantic target dimension must be 31")
        self.decimation = profile.decimation
        self.episode_length_s = profile.episode_length_steps * profile.control_dt
        self.action_space = profile.action_dim
        self.observation_space = profile.policy_dim
        self.state_space = profile.critic_dim
        self.sim.dt = profile.physics_dt
        self.sim.render_interval = profile.decimation
