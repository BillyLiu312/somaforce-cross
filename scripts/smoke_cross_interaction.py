"""Smoke-test the probabilistic cross interaction module."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from somaforce_cross.models.cross_interaction import ProbabilisticCrossInteraction


def main() -> None:
    module = ProbabilisticCrossInteraction(dir_dim=16, mag_dim=12, n_dir=4, n_mag=4, z_dim=32)
    out = module(torch.randn(2, 16), torch.randn(2, 12))
    print("p_dir:", tuple(out.p_dir.shape))
    print("p_mag:", tuple(out.p_mag.shape))
    print("p_cross:", tuple(out.p_cross.shape))
    print("z_cross:", tuple(out.z_cross.shape))


if __name__ == "__main__":
    main()
