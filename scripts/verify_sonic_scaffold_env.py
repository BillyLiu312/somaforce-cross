"""Verify Sonic scaffold access from a no-door Isaac Lab manager env.

This script is intentionally small and diagnostic. It creates a synthetic static
G1 motion trajectory, composes Sonic's manager-env config with one environment,
and reads:

- ``motion_cmd.joint_pos``;
- Sonic ``residual_joint_pos_action``;
- SomaForce-Cross ``SonicScaffoldAdapter(...).get_output().a_nom``.

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
    write_synthetic_motion(args.motion_file)
    print(f"synthetic_motion={args.motion_file}", flush=True)
    verify_dir = Path("/tmp/somaforce_sonic_verify")

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    app = launcher.app
    print("app_started", flush=True)

    try:
        print("importing_hydra", flush=True)
        from hydra import compose, initialize_config_dir

        print("importing_sonic_common", flush=True)
        from gear_sonic.trl.utils.common import custom_instantiate

        overrides = [
            "+exp=manager/universal_token/all_modes/sonic_release",
            f"num_envs={args.num_envs}",
            f"manager_env.config.num_envs={args.num_envs}",
            "manager_env.config.terrain_type=plane",
            f"experiment_dir={verify_dir}",
            f"save_dir={verify_dir / '.hydra'}",
            f"output_dir={verify_dir / 'output'}",
            f"manager_env.config.experiment_dir={verify_dir}",
            f"manager_env.config.save_rendering_dir={verify_dir / 'renderings'}",
            "+manager_env.config.add_object=false",
            "+manager_env.config.add_table=false",
            f"manager_env.commands.motion.motion_lib_cfg.motion_file={args.motion_file}",
            "manager_env.commands.motion.motion_lib_cfg.smpl_motion_file=dummy",
            "manager_env.commands.motion.motion_lib_cfg.multi_thread=false",
            "manager_env.commands.motion.num_future_frames=2",
            "manager_env.commands.motion.smpl_num_future_frames=2",
        ]
        with initialize_config_dir(version_base="1.1", config_dir=str(sonic_config_dir)):
            cfg = compose(config_name="base", overrides=overrides)
        print("compose_ok", flush=True)

        env_cfg = custom_instantiate(cfg.manager_env, _resolve=True)
        env_cfg.seed = 0
        env_cfg.sim.device = args.device
        env_cfg.config["headless"] = True
        print(f"env_cfg_ok={type(env_cfg).__name__}", flush=True)

        if args.instantiate_only:
            return

        from isaaclab.envs import ManagerBasedRLEnv

        from gear_sonic.envs.manager_env.mdp.observations import residual_joint_pos_action
        from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper
        from somaforce_cross.scaffold import ScaffoldTask, SonicScaffoldAdapter

        env = ManagerBasedRLEnv(cfg=env_cfg, render_mode=None)
        wrapper = ManagerEnvWrapper(env, env_cfg.config)
        wrapper.reset(flatten_dict_obs=False)
        motion_cmd = env.command_manager.get_term("motion")
        a_nom = residual_joint_pos_action(env, command_name="motion")
        scaffold = SonicScaffoldAdapter(
            env,
            task=ScaffoldTask.PUSH_PULL_DOOR,
        ).get_output()

        print(f"env_ok={env.num_envs}", flush=True)
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
