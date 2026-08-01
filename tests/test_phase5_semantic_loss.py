from __future__ import annotations

import torch

from somaforce_cross.learning.losses import semantic_losses, weighted_soft_target_kl


def _target(
    p_dir: torch.Tensor,
    p_mag: torch.Tensor,
    weight: torch.Tensor,
    *,
    clean: torch.Tensor | None = None,
) -> torch.Tensor:
    if clean is None:
        clean = torch.zeros(p_dir.shape[0], 12)
    return torch.cat((clean, p_dir, p_mag, weight), dim=-1)


def test_soft_target_kl_matches_target_given_prediction_and_weighting() -> None:
    target = torch.tensor([[0.75, 0.25], [0.2, 0.8]], dtype=torch.float32)
    prediction = torch.tensor([[0.5, 0.5], [0.9, 0.1]], dtype=torch.float32)
    weight = torch.tensor([[2.0], [0.5]], dtype=torch.float32)
    per_sample = (target * (target.log() - prediction.log())).sum(dim=-1)
    expected = (weight.squeeze(-1) * per_sample).sum() / weight.sum()
    actual = weighted_soft_target_kl(target, prediction, weight)
    assert torch.allclose(actual, expected, atol=1.0e-7, rtol=0.0)


def test_semantic_weighting_zero_weight_and_clean_target_exclusion() -> None:
    p_dir_target = torch.softmax(torch.randn(3, 13), dim=-1)
    p_mag_target = torch.softmax(torch.randn(3, 5), dim=-1)
    p_dir_pred = torch.softmax(torch.randn(3, 13), dim=-1)
    p_mag_logits = torch.randn(3, 5, requires_grad=True)
    p_mag_pred = torch.softmax(p_mag_logits, dim=-1)
    zero_weight = torch.zeros(3, 1)
    first = semantic_losses(
        p_dir_pred, p_mag_pred, _target(p_dir_target, p_mag_target, zero_weight)
    )
    assert first.dir_loss.item() == 0.0
    assert first.mag_loss.item() == 0.0
    assert first.total.item() == 0.0
    first.total.backward()
    assert p_mag_logits.grad is not None
    assert torch.isfinite(p_mag_logits.grad).all()

    one_weight = torch.ones(3, 1)
    clean_changed = torch.full((3, 12), 999.0)
    baseline = semantic_losses(
        p_dir_pred, p_mag_pred.detach(), _target(p_dir_target, p_mag_target, one_weight)
    )
    changed_clean = semantic_losses(
        p_dir_pred,
        p_mag_pred.detach(),
        _target(p_dir_target, p_mag_target, one_weight, clean=clean_changed),
    )
    assert torch.equal(baseline.total, changed_clean.total)


def test_probability_contract_and_deterministic_loss_decrease() -> None:
    torch.manual_seed(4)
    target = torch.zeros(1, 13)
    target[:, 4] = 1.0
    logits = torch.zeros(1, 13, requires_grad=True)
    optimizer = torch.optim.SGD([logits], lr=0.5)
    weight = torch.ones(1, 1)
    initial = weighted_soft_target_kl(target, torch.softmax(logits, dim=-1), weight)
    for _ in range(60):
        optimizer.zero_grad()
        loss = weighted_soft_target_kl(target, torch.softmax(logits, dim=-1), weight)
        loss.backward()
        optimizer.step()
    final = weighted_soft_target_kl(target, torch.softmax(logits, dim=-1), weight)
    prediction = torch.softmax(logits, dim=-1)
    assert torch.isfinite(prediction).all()
    assert torch.allclose(prediction.sum(dim=-1), torch.ones(1), atol=1.0e-6)
    assert final < initial
