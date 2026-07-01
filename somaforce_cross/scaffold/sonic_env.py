"""Sonic manager-env binding helpers for scaffold-only task rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from somaforce_cross.scaffold.contracts import ScaffoldTask
from somaforce_cross.scaffold.task_trajectories import InteractionMode


@dataclass(frozen=True)
class SonicObjectBinding:
    """Object/table settings expressible through Sonic's manager env config."""

    object_usd_path: str | None = None
    object_position: tuple[float, float, float] = (0.75, 0.0, 0.75)
    object_is_dynamic: bool = True
    object_collision_enabled: bool = True
    object_mass: float | None = None
    add_table: bool = False
    table_position: tuple[float, float, float] | None = None
    table_size: tuple[float, float, float] | None = None

    @property
    def add_object(self) -> bool:
        return self.object_usd_path is not None


@dataclass(frozen=True)
class SonicManagerEnvBinding:
    """Complete scaffold binding from task motion file to Sonic overrides."""

    task: ScaffoldTask
    interaction_mode: InteractionMode
    motion_file: Path
    motion_key: str
    object_binding: SonicObjectBinding = SonicObjectBinding()
    terrain_type: str = "plane"
    requires_articulation_scene: bool = False


def make_g1_door_sonic_binding(
    motion_file: Path,
    interaction_mode: InteractionMode,
    door_asset_path: str | None = None,
) -> SonicManagerEnvBinding:
    """Create the current Sonic binding for the G1 door scaffold task.

    Sonic's existing ``add_object`` path is a rigid-object USD hook. A hinged door
    needs an articulation scene, so the asset path is recorded but not routed
    through ``add_object`` here.
    """

    return SonicManagerEnvBinding(
        task=ScaffoldTask.PUSH_PULL_DOOR,
        interaction_mode=interaction_mode,
        motion_file=motion_file,
        motion_key=f"somaforce_g1_door_{interaction_mode}",
        object_binding=SonicObjectBinding(object_usd_path=door_asset_path),
        requires_articulation_scene=True,
    )


def make_g1_box_sonic_binding(
    motion_file: Path,
    interaction_mode: InteractionMode,
    box_usd_path: str | None = None,
    object_position: tuple[float, float, float] = (0.72, 0.0, 0.75),
    object_mass: float | None = 2.0,
) -> SonicManagerEnvBinding:
    """Create a Sonic rigid-object binding for the G1 box scaffold task."""

    return SonicManagerEnvBinding(
        task=ScaffoldTask.PUSH_PULL_BOX,
        interaction_mode=interaction_mode,
        motion_file=motion_file,
        motion_key=f"somaforce_g1_box_{interaction_mode}",
        object_binding=SonicObjectBinding(
            object_usd_path=box_usd_path,
            object_position=object_position,
            object_is_dynamic=True,
            object_collision_enabled=True,
            object_mass=object_mass,
        ),
        requires_articulation_scene=False,
    )


def make_sonic_manager_overrides(
    binding: SonicManagerEnvBinding,
    num_envs: int = 1,
    experiment_dir: Path = Path("/tmp/somaforce_sonic_scaffold"),
) -> list[str]:
    """Build Hydra overrides for Sonic's manager env scaffold rollout."""

    object_cfg = binding.object_binding
    add_rigid_object = object_cfg.add_object and not binding.requires_articulation_scene
    overrides = [
        "+exp=manager/universal_token/all_modes/sonic_release",
        f"num_envs={num_envs}",
        f"manager_env.config.num_envs={num_envs}",
        f"manager_env.config.terrain_type={binding.terrain_type}",
        f"experiment_dir={experiment_dir}",
        f"save_dir={experiment_dir / '.hydra'}",
        f"output_dir={experiment_dir / 'output'}",
        f"manager_env.config.experiment_dir={experiment_dir}",
        f"manager_env.config.save_rendering_dir={experiment_dir / 'renderings'}",
        f"+manager_env.config.add_object={_bool(add_rigid_object)}",
        f"+manager_env.config.add_table={_bool(object_cfg.add_table)}",
        f"manager_env.commands.motion.motion_lib_cfg.motion_file={binding.motion_file}",
        "manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy",
        "manager_env.commands.motion.motion_lib_cfg.multi_thread=false",
        "manager_env.commands.motion.num_future_frames=2",
        "manager_env.commands.motion.smpl_num_future_frames=2",
    ]
    if add_rigid_object:
        overrides.extend(
            [
                f"+manager_env.config.object_usd_path={object_cfg.object_usd_path}",
                f"+manager_env.config.object_position={list(object_cfg.object_position)}",
                f"+manager_env.config.object_is_dynamic={_bool(object_cfg.object_is_dynamic)}",
                (
                    "+manager_env.config.object_collision_enabled="
                    f"{_bool(object_cfg.object_collision_enabled)}"
                ),
            ]
        )
        if object_cfg.object_mass is not None:
            overrides.append(f"manager_env.config.object_mass={object_cfg.object_mass}")
    if object_cfg.add_table:
        if object_cfg.table_position is not None:
            overrides.append(f"+manager_env.config.table_position={list(object_cfg.table_position)}")
        if object_cfg.table_size is not None:
            overrides.append(f"+manager_env.config.table_size={list(object_cfg.table_size)}")
    return overrides


def make_sonic_verify_command(
    task: ScaffoldTask,
    interaction_mode: InteractionMode,
    motion_file: Path,
) -> list[str]:
    """Return the local verification command for a generated task motion."""

    return [
        "PYTHONUNBUFFERED=1",
        "python",
        "scripts/verify_sonic_scaffold_env.py",
        "--motion-source",
        "scaffold-task",
        "--task",
        task.value,
        "--interaction-mode",
        interaction_mode,
        "--motion-file",
        str(motion_file),
    ]


def _bool(value: bool) -> str:
    return "true" if value else "false"
