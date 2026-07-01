"""Verify Isaac/Sonic imports required by the scaffold adapter.

Run from the repository root inside the ``isaaclab`` conda environment.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import sys
import traceback

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sonic-config-dir",
        type=Path,
        default=Path("../GR00T-WholeBodyControl/gear_sonic/config"),
    )
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

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    app = launcher.app
    print("app_started", flush=True)

    try:
        import gear_sonic
        from gear_sonic.envs.manager_env.mdp.commands import TrackingCommand
        from gear_sonic.envs.manager_env.mdp.observations import residual_joint_pos_action

        try:
            version = importlib.metadata.version("gear-sonic")
        except importlib.metadata.PackageNotFoundError:
            version = "editable"

        print(f"gear_sonic_import_ok={gear_sonic.__name__}", flush=True)
        print(f"gear_sonic_version={version}", flush=True)
        print(f"tracking_command_import_ok={TrackingCommand.__name__}", flush=True)
        print(f"residual_joint_pos_action_import_ok={residual_joint_pos_action.__name__}", flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app.close()
        print("app_closed", flush=True)


if __name__ == "__main__":
    main()
