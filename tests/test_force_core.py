from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from somaforce_cross.force import (
    ContactGate,
    CrossEncoder,
    ForceSemanticHeads,
    WristHistoryBuffer,
    WristTokenEncoder,
    magnitude_soft_targets,
)


def test_history_push_is_oldest_to_newest_and_partial_reset() -> None:
    history = WristHistoryBuffer(3)
    initial = torch.full((2, 2, 14), 2.0)
    history.reset(torch.tensor([0, 2]), initial)
    assert torch.all(history.storage[0] == 2)
    assert torch.all(history.storage[2] == 2)
    assert torch.all(history.storage[1] == 0)
    frame = torch.arange(3 * 2 * 14, dtype=torch.float32).reshape(3, 2, 14)
    history.push(frame)
    assert torch.equal(history.storage[:, :, -1], frame)
    assert torch.all(history.storage[0, :, :-1] == 2)
    history.reset([0])
    assert torch.all(history.storage[0] == 0)
    assert torch.equal(history.storage[2, :, -1], frame[2])


@pytest.mark.parametrize("bad", [torch.zeros(2, 16, 14), torch.zeros(2, 2, 15)])
def test_history_rejects_bad_frames(bad: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        WristHistoryBuffer(2).push(bad)


def test_wrist_encoder_shapes_shared_tcn_and_side_distinction() -> None:
    encoder = WristTokenEncoder()
    wrist = torch.randn(4, 1, 16, 14)
    x = wrist.expand(-1, 2, -1, -1).clone()
    per_wrist, fused = encoder(x)
    assert per_wrist.shape == (4, 2, 128)
    assert fused.shape == (4, 272)
    assert len(encoder.tcn) == 3
    assert encoder.side_embedding.shape == (2, 8)
    assert torch.allclose(per_wrist[:, 0], per_wrist[:, 1])
    fused_per_wrist = fused.reshape(4, 2, 136)
    assert not torch.allclose(fused_per_wrist[:, 0], fused_per_wrist[:, 1])
    with pytest.raises(ValueError):
        encoder(torch.zeros(4, 2, 15, 14))


def test_wrist_encoder_is_causal() -> None:
    torch.manual_seed(0)
    encoder = WristTokenEncoder().eval()
    prefix = torch.randn(1, 2, 8, 14)
    first = torch.cat((prefix, torch.zeros(1, 2, 8, 14)), dim=2)
    second = first.clone()
    second[:, :, 8:] = torch.randn(1, 2, 8, 14)
    # The output at time t is not exposed publicly; compare a direct block prefix.
    x = first.reshape(2, 16, 14).transpose(1, 2)
    y = second.reshape(2, 16, 14).transpose(1, 2)
    for block in encoder.tcn:
        x = block(x)
        y = block(y)
    assert torch.allclose(x[:, :, :8], y[:, :, :8])


def test_wrist_encoder_final_feature_uses_oldest_and_current_frames() -> None:
    torch.manual_seed(1)
    encoder = WristTokenEncoder().eval()
    base = torch.zeros(1, 2, 16, 14)
    oldest_changed = base.clone()
    oldest_changed[:, :, 0, :] = 3.0
    current_changed = base.clone()
    current_changed[:, :, 15, :] = 3.0

    per_base, fused_base = encoder(base)
    per_oldest, fused_oldest = encoder(oldest_changed)
    per_current, fused_current = encoder(current_changed)
    assert not torch.allclose(per_base, per_oldest)
    assert not torch.allclose(fused_base, fused_oldest)
    assert not torch.allclose(per_base, per_current)
    assert not torch.allclose(fused_base, fused_current)


def test_semantic_heads_probabilities_are_finite_and_normalized() -> None:
    heads = ForceSemanticHeads()
    output = heads(torch.zeros(5, 272))
    assert output.dir_logits.shape == (5, 13)
    assert output.mag_logits.shape == (5, 5)
    assert torch.allclose(output.p_dir.sum(-1), torch.ones(5))
    assert torch.allclose(output.p_mag.sum(-1), torch.ones(5))
    assert torch.isfinite(torch.cat((output.p_dir, output.p_mag), dim=-1)).all()


def test_cross_encoder_outer_product_normalizes() -> None:
    heads = ForceSemanticHeads()
    semantic = heads(torch.randn(3, 272))
    cross = CrossEncoder()(semantic.p_dir, semantic.p_mag)
    assert cross.P_cross.shape == (3, 13, 5)
    assert cross.z_joint.shape == (3, 65)
    assert cross.z_cross.shape == (3, 64)
    assert torch.allclose(cross.P_cross.sum(dim=(-1, -2)), torch.ones(3), atol=1e-6)


def test_contact_gate_single_double_and_quality_zero() -> None:
    output = ContactGate()(
        torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.5, 0.5]]),
        torch.tensor([[1.0, 1.0], [0.5, 0.25], [0.0, 1.0]]),
    )
    assert torch.allclose(output.per_wrist, torch.tensor([[1.0, 0.0], [0.5, 0.25], [0.0, 0.5]]))
    assert torch.allclose(output.aggregate, torch.tensor([[1.0], [0.625], [0.5]]))
    with pytest.raises(ValueError):
        ContactGate()(torch.tensor([[1.1, 0.0]]), torch.ones(1, 2))


def test_magnitude_targets_zero_and_bins() -> None:
    wrench = torch.zeros(2, 3, 6)
    target = magnitude_soft_targets(wrench, 10.0, 5.0)
    assert target.shape == (2, 3, 5)
    assert torch.isfinite(target).all()
    assert torch.allclose(target.sum(-1), torch.ones(2, 3))
    assert target[..., 0].min() > target[..., 1].min()
    near = magnitude_soft_targets(torch.full((1, 6), 1e-7), 10.0, 5.0)
    assert torch.isfinite(near).all()
    with pytest.raises(ValueError):
        magnitude_soft_targets(torch.zeros(1, 6), 0.0, 1.0)


def test_force_core_float64() -> None:
    dtype = torch.float64
    history = WristHistoryBuffer(2, dtype=dtype)
    frame = torch.randn(2, 2, 14, dtype=dtype)
    history.push(frame)
    assert history.storage.dtype == dtype

    encoder = WristTokenEncoder().to(dtype=dtype)
    per_wrist, fused = encoder(history.storage)
    assert per_wrist.dtype == dtype
    assert fused.dtype == dtype

    heads = ForceSemanticHeads().to(dtype=dtype)
    semantic = heads(fused)
    assert semantic.p_dir.dtype == dtype
    assert semantic.p_mag.dtype == dtype

    cross = CrossEncoder().to(dtype=dtype)(semantic.p_dir, semantic.p_mag)
    assert cross.P_cross.dtype == dtype
    assert cross.z_joint.dtype == dtype
    assert cross.z_cross.dtype == dtype

    gate = ContactGate()(
        torch.full((2, 2), 0.5, dtype=dtype),
        torch.ones(2, 2, dtype=dtype),
    )
    assert gate.per_wrist.dtype == dtype
    assert gate.aggregate.dtype == dtype
    targets = magnitude_soft_targets(torch.randn(2, 3, 6, dtype=dtype), 2.0, 3.0)
    assert targets.dtype == dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_force_core_cuda_smoke() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.float32
    history = WristHistoryBuffer(2, device=device, dtype=dtype)
    history.push(torch.randn(2, 2, 14, device=device, dtype=dtype))
    assert history.storage.device == device

    encoder = WristTokenEncoder().to(device=device)
    per_wrist, fused = encoder(history.storage)
    assert per_wrist.device == device
    assert fused.device == device
    semantic = ForceSemanticHeads().to(device=device)(fused)
    cross = CrossEncoder().to(device=device)(semantic.p_dir, semantic.p_mag)
    assert cross.z_cross.device == device
    gate = ContactGate()(
        torch.full((2, 2), 0.5, device=device),
        torch.ones(2, 2, device=device),
    )
    assert gate.aggregate.device == device
    targets = magnitude_soft_targets(torch.randn(2, 6, device=device), 2.0, 3.0)
    assert targets.device == device


def test_force_package_has_no_forbidden_imports_via_ast() -> None:
    force_dir = Path(__file__).resolve().parents[1] / "somaforce_cross" / "force"
    forbidden = ("isaac", "hdmi", "active_adaptation")
    for path in sorted(force_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported_names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_names.append(node.module or "")
                imported_names.extend(alias.name for alias in node.names)
        assert not any(
            token in name.lower()
            for name in imported_names
            for token in forbidden
        ), f"forbidden import in {path}: {imported_names}"
