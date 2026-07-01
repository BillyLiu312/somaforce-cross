"""Print Sonic Hydra overrides for a SomaForce scaffold task binding."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from somaforce_cross.scaffold import (
    ScaffoldTask,
    make_g1_box_sonic_binding,
    make_g1_door_sonic_binding,
    make_sonic_manager_overrides,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("push_pull_door", "push_pull_box"), required=True)
    parser.add_argument("--interaction-mode", choices=("push", "pull"), default="push")
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--experiment-dir", type=Path, default=Path("/tmp/somaforce_sonic_scaffold"))
    parser.add_argument("--object-usd-path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = ScaffoldTask(args.task)
    if task == ScaffoldTask.PUSH_PULL_BOX:
        binding = make_g1_box_sonic_binding(
            args.motion_file,
            interaction_mode=args.interaction_mode,
            box_usd_path=args.object_usd_path,
        )
    else:
        binding = make_g1_door_sonic_binding(
            args.motion_file,
            interaction_mode=args.interaction_mode,
            door_asset_path=args.object_usd_path,
        )

    print(f"task={binding.task.value}")
    print(f"interaction_mode={binding.interaction_mode}")
    print(f"motion_key={binding.motion_key}")
    print(f"requires_articulation_scene={binding.requires_articulation_scene}")
    for override in make_sonic_manager_overrides(
        binding,
        num_envs=args.num_envs,
        experiment_dir=args.experiment_dir,
    ):
        print(override)


if __name__ == "__main__":
    main()
