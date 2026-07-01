"""Verify Sonic scaffold access from an Isaac Lab manager env.

This script is intentionally small and diagnostic. It creates a synthetic static
G1 motion trajectory, composes Sonic's manager-env config with one environment,
and reads:

- ``motion_cmd.joint_pos``;
- Sonic ``residual_joint_pos_action``;
- SomaForce-Cross ``SonicScaffoldAdapter(...).get_output().a_nom``.

For box tasks, pass ``--object-usd-path`` to route a USD through Sonic's
rigid-object hook and verify that the object is registered in the scene. For
door tasks, pass ``--generate-default-door-urdf`` to add a hinged articulation
scene entity without routing the door through Sonic's rigid-object hook.

Run from the repository root inside the ``isaaclab`` conda environment.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import traceback

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def write_synthetic_motion(path: Path, num_frames: int = 20, num_nodes: int = 30) -> None:
    """Write a minimal Sonic-compatible G1 motion file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    root = np.tile(np.array([[0.0, 0.0, 0.76]], dtype=np.float32), (num_frames, 1))
    pose_aa = np.zeros((num_frames, num_nodes, 3), dtype=np.float32)
    joblib.dump(
        {
            "somaforce_static_g1": {
                "root_trans_offset": root,
                "pose_aa": pose_aa,
                "fps": 50.0,
            }
        },
        path,
    )


def write_default_box_usd(path: Path, size_m: float = 0.25, mass_kg: float = 2.0) -> None:
    """Write a minimal rigid-body box USD for Sonic object smoke checks."""

    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    cube = UsdGeom.Cube.Define(stage, "/Box")
    cube.CreateSizeAttr(size_m)
    cube.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    prim = cube.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass_kg)

    stage.SetDefaultPrim(prim)
    stage.GetRootLayer().Save()


def write_default_door_urdf(path: Path) -> None:
    """Write a minimal fixed-frame hinged door URDF for articulation smoke checks."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """<?xml version="1.0"?>
<robot name="somaforce_default_hinged_door">
  <link name="frame">
    <inertial>
      <origin xyz="0 0 1.0" rpy="0 0 0"/>
      <mass value="10.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 1.0" rpy="0 0 0"/>
      <geometry><box size="0.04 0.04 2.0"/></geometry>
      <material name="frame_gray"><color rgba="0.25 0.25 0.25 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 1.0" rpy="0 0 0"/>
      <geometry><box size="0.04 0.04 2.0"/></geometry>
    </collision>
  </link>
  <link name="panel">
    <inertial>
      <origin xyz="0.45 0 1.0" rpy="0 0 0"/>
      <mass value="12.0"/>
      <inertia ixx="4.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="4.0"/>
    </inertial>
    <visual>
      <origin xyz="0.45 0 1.0" rpy="0 0 0"/>
      <geometry><box size="0.9 0.05 2.0"/></geometry>
      <material name="door_blue"><color rgba="0.22 0.38 0.58 1"/></material>
    </visual>
    <collision>
      <origin xyz="0.45 0 1.0" rpy="0 0 0"/>
      <geometry><box size="0.9 0.05 2.0"/></geometry>
    </collision>
  </link>
  <joint name="hinge_joint" type="revolute">
    <parent link="frame"/>
    <child link="panel"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="40.0" velocity="2.0"/>
    <dynamics damping="1.0" friction="0.2"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )


def attach_default_door_articulation(env_cfg: object, door_urdf_path: Path) -> None:
    """Attach a minimal hinged door articulation to a Sonic manager env cfg."""

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    env_cfg.scene.door = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Door",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(door_urdf_path),
            fix_base=True,
            force_usd_conversion=True,
            collision_from_visuals=False,
            make_instanceable=True,
            joint_drive=None,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.85, -0.45, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={"hinge_joint": 0.0},
            joint_vel={"hinge_joint": 0.0},
        ),
        actuators={
            "hinge": ImplicitActuatorCfg(
                joint_names_expr=["hinge_joint"],
                effort_limit_sim=40.0,
                velocity_limit_sim=2.0,
                stiffness=0.0,
                damping=1.0,
            ),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sonic-config-dir",
        type=Path,
        default=Path("../GR00T-WholeBodyControl/gear_sonic/config"),
    )
    parser.add_argument(
        "--motion-file",
        type=Path,
        default=Path("/tmp/somaforce_synth_motion.pkl"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--motion-source", choices=("static", "scaffold-task"), default="static")
    parser.add_argument("--task", choices=("push_pull_door", "push_pull_box"), default="push_pull_door")
    parser.add_argument("--interaction-mode", choices=("push", "pull"), default="push")
    parser.add_argument("--motion-frames", type=int, default=101)
    parser.add_argument("--motion-duration-s", type=float, default=2.0)
    parser.add_argument("--motion-fps", type=int, default=50)
    parser.add_argument("--object-usd-path")
    parser.add_argument("--generate-default-box-usd", action="store_true")
    parser.add_argument("--default-box-usd-path", type=Path, default=Path("/tmp/somaforce_default_box.usd"))
    parser.add_argument("--default-box-size-m", type=float, default=0.25)
    parser.add_argument("--generate-default-door-urdf", action="store_true")
    parser.add_argument("--default-door-urdf-path", type=Path, default=Path("/tmp/somaforce_default_hinged_door.urdf"))
    parser.add_argument("--object-position", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--object-mass", type=float, default=2.0)
    parser.add_argument("--instantiate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sonic_config_dir = args.sonic_config_dir.resolve()
    sonic_root = sonic_config_dir.parents[1]
    os.chdir(sonic_root)
    print(f"sonic_root={sonic_root}", flush=True)

    # Isaac AppLauncher also parses sys.argv. Keep our script-specific flags from
    # being interpreted by Isaac/Kit after argparse has consumed them.
    sys.argv = [sys.argv[0]]
    from somaforce_cross.scaffold import (
        ScaffoldTask,
        make_g1_box_sonic_binding,
        make_g1_door_sonic_binding,
        make_sonic_manager_overrides,
    )

    task = ScaffoldTask(args.task)
    object_usd_path = args.object_usd_path
    if args.generate_default_box_usd:
        if task != ScaffoldTask.PUSH_PULL_BOX:
            raise ValueError("--generate-default-box-usd is only valid for push_pull_box")
        object_usd_path = str(args.default_box_usd_path)
    if args.generate_default_door_urdf:
        if task != ScaffoldTask.PUSH_PULL_DOOR:
            raise ValueError("--generate-default-door-urdf is only valid for push_pull_door")
        object_usd_path = str(args.default_door_urdf_path)
    if object_usd_path is not None and args.motion_source == "static":
        raise ValueError("Object-enabled Sonic checks require --motion-source scaffold-task")
    if args.motion_source == "static":
        write_synthetic_motion(args.motion_file)
        task_name = "static"
    else:
        from somaforce_cross.scaffold import write_single_task_sonic_motion_file

        write_single_task_sonic_motion_file(
            args.motion_file,
            task=task,
            mode=args.interaction_mode,
            num_frames=args.motion_frames,
            duration_s=args.motion_duration_s,
            fps=args.motion_fps,
        )
        task_name = f"{task.value}:{args.interaction_mode}"
    object_position = tuple(args.object_position) if args.object_position is not None else (0.72, 0.0, 0.75)
    if task == ScaffoldTask.PUSH_PULL_BOX:
        binding = make_g1_box_sonic_binding(
            args.motion_file,
            args.interaction_mode,
            box_usd_path=object_usd_path,
            object_position=object_position,
            object_mass=args.object_mass,
        )
    else:
        binding = make_g1_door_sonic_binding(
            args.motion_file,
            args.interaction_mode,
            door_asset_path=object_usd_path,
        )
    if binding.object_motion_file is not None:
        from somaforce_cross.scaffold import write_single_task_sonic_object_motion_file

        write_single_task_sonic_object_motion_file(
            binding.object_motion_file,
            task=task,
            mode=args.interaction_mode,
            num_frames=args.motion_frames,
            fps=args.motion_fps,
        )
    print(f"motion_file={args.motion_file}", flush=True)
    print(f"object_motion_file={binding.object_motion_file}", flush=True)
    print(f"motion_source={args.motion_source}", flush=True)
    print(f"motion_task={task_name}", flush=True)
    print(f"requires_articulation_scene={binding.requires_articulation_scene}", flush=True)
    print(f"object_usd_path={object_usd_path}", flush=True)
    verify_dir = Path("/tmp/somaforce_sonic_verify")

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    app = launcher.app
    print("app_started", flush=True)

    try:
        if args.generate_default_box_usd:
            write_default_box_usd(args.default_box_usd_path, args.default_box_size_m, args.object_mass)
            print(f"default_box_usd_written={args.default_box_usd_path}", flush=True)
        if args.generate_default_door_urdf:
            write_default_door_urdf(args.default_door_urdf_path)
            print(f"default_door_urdf_written={args.default_door_urdf_path}", flush=True)

        print("importing_hydra", flush=True)
        from hydra import compose, initialize_config_dir

        print("importing_sonic_common", flush=True)
        from gear_sonic.trl.utils.common import custom_instantiate

        overrides = make_sonic_manager_overrides(
            binding,
            num_envs=args.num_envs,
            experiment_dir=verify_dir,
        )
        with initialize_config_dir(version_base="1.1", config_dir=str(sonic_config_dir)):
            cfg = compose(config_name="base", overrides=overrides)
        print("compose_ok", flush=True)

        env_cfg = custom_instantiate(cfg.manager_env, _resolve=True)
        env_cfg.seed = 0
        env_cfg.sim.device = args.device
        env_cfg.config["headless"] = True
        if args.generate_default_door_urdf:
            attach_default_door_articulation(env_cfg, args.default_door_urdf_path)
            print("door_articulation_attached=True", flush=True)
        print(f"env_cfg_ok={type(env_cfg).__name__}", flush=True)

        if args.instantiate_only:
            return

        from isaaclab.envs import ManagerBasedRLEnv

        from gear_sonic.envs.manager_env.mdp.observations import residual_joint_pos_action
        from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper
        from somaforce_cross.scaffold import SonicScaffoldAdapter

        env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
        wrapper = ManagerEnvWrapper(env, env_cfg.config)
        wrapper.reset(flatten_dict_obs=False)
        motion_cmd = env.command_manager.get_term("motion")
        a_nom = residual_joint_pos_action(env, command_name="motion")
        scaffold = SonicScaffoldAdapter(
            env,
            task=task,
        ).get_output()

        print(f"env_ok={env.num_envs}", flush=True)
        rigid_object_names = sorted(getattr(env.scene, "rigid_objects", {}).keys())
        articulation_names = sorted(getattr(env.scene, "articulations", {}).keys())
        print(f"rigid_object_names={rigid_object_names}", flush=True)
        print(f"articulation_names={articulation_names}", flush=True)
        print(f"object_enabled={'object' in rigid_object_names}", flush=True)
        print(f"door_enabled={'door' in articulation_names}", flush=True)
        print(f"motion_joint_pos_shape={tuple(motion_cmd.joint_pos.shape)}", flush=True)
        print(f"residual_action_shape={tuple(a_nom.shape)}", flush=True)
        print(f"scaffold_a_nom_shape={tuple(scaffold.a_nom.shape)}", flush=True)
        print(f"a_nom_head={scaffold.a_nom[0, :5].detach().cpu().tolist()}", flush=True)
        env.close()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app.close()
        print("app_closed", flush=True)


if __name__ == "__main__":
    main()
