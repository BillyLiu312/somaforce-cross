"""Export first G1 scaffold trajectories as a Sonic motion-lib pkl."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from somaforce_cross.scaffold import write_sonic_motion_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/somaforce_g1_scaffold_motions.pkl"),
    )
    parser.add_argument("--num-frames", type=int, default=101)
    parser.add_argument("--duration-s", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = write_sonic_motion_file(
        args.output,
        num_frames=args.num_frames,
        duration_s=args.duration_s,
        fps=args.fps,
    )
    print(f"motion_file={args.output}")
    for key, entry in library.items():
        print(
            f"{key}: "
            f"root={entry['root_trans_offset'].shape} "
            f"pose_aa={entry['pose_aa'].shape} "
            f"dof={entry['dof'].shape} "
            f"fps={entry['fps']}"
        )


if __name__ == "__main__":
    main()
