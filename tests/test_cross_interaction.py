import torch

from somaforce_cross.models.cross_interaction import ProbabilisticCrossInteraction


def test_probabilistic_cross_interaction_shapes_and_probability():
    module = ProbabilisticCrossInteraction(dir_dim=8, mag_dim=6, n_dir=4, n_mag=5, z_dim=16)
    out = module(torch.randn(3, 8), torch.randn(3, 6))

    assert out.p_dir.shape == (3, 4)
    assert out.p_mag.shape == (3, 5)
    assert out.p_cross.shape == (3, 4, 5)
    assert out.z_joint.shape == (3, 20)
    assert out.z_cross.shape == (3, 16)
    assert torch.allclose(out.p_cross.sum(dim=(-1, -2)), torch.ones(3), atol=1e-5)
