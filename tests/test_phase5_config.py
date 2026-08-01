from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from somaforce_cross.learning.config import (
    PHASE4_CANONICAL_SHA256,
    PHASE4_SOURCE_COMMIT,
    PHASE5_CONTRACT_VERSION,
    canonical_sha256,
    load_phase5_config,
    validate_phase5_config,
)


CONFIG_PATH = Path("configs/phase5_ppo_v1.json")
RAW_SHA256 = "ac3ca17ef7b0f5cd800c2c967c7242eec94a2fdaf59eb8ba465e8f3697d9f200"
CANONICAL_SHA256 = "e95eca5f3c1332965967555f81cd33a78f9144280d0aa446e5eefaf5b83ae1f8"


def test_phase5_config_schema_hashes_and_frozen_values() -> None:
    config = load_phase5_config(CONFIG_PATH)
    payload = config.payload

    assert config.raw_sha256 == RAW_SHA256
    assert config.canonical_sha256 == CANONICAL_SHA256
    assert canonical_sha256(payload) == CANONICAL_SHA256
    assert payload["contract_version"] == PHASE5_CONTRACT_VERSION
    assert payload["phase4"] == {
        "canonical_sha256": PHASE4_CANONICAL_SHA256,
        "contract_version": "phase4b5_numeric_v2",
        "source_commit": PHASE4_SOURCE_COMMIT,
    }
    assert payload["observation"]["obs_groups"] == {
        "policy": ["policy"],
        "critic": ["critic"],
    }
    assert payload["network"]["actor"] == [220, 256, 256, 23]
    assert payload["network"]["critic"] == [845, 512, 256, 256, 1]
    assert payload["ppo"] == {
        "clip_param": 0.2,
        "desired_kl": None,
        "entropy_coef": 0.001,
        "gamma": 0.99,
        "init_noise_std": 1.0,
        "lam": 0.95,
        "learning_rate": 0.0003,
        "max_grad_norm": 1.0,
        "noise_std_type": "log",
        "normalize_advantage_per_mini_batch": False,
        "num_learning_epochs": 3,
        "num_mini_batches": 8,
        "num_steps_per_env": 32,
        "ppo_epochs": 3,
        "schedule": "fixed",
        "use_clipped_value_loss": True,
        "value_loss_coef": 1.0,
    }
    assert payload["semantic_loss"] == {
        "clean_wrench_dim": 12,
        "eps": 1.0e-8,
        "lambda_cross": 0.0,
        "lambda_dir": 0.05,
        "lambda_mag": 0.05,
        "p_dir_dim": 13,
        "p_mag_dim": 5,
        "reduction": "sum(weight * per_sample_kl) / clamp(sum(weight), min=1)",
        "semantic_loss_weight_dim": 1,
    }


def test_phase5_config_rejects_phase4_binding_and_ppo_mutations(tmp_path: Path) -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for mutate in (
        lambda value: value["phase4"].__setitem__("canonical_sha256", "0" * 64),
        lambda value: value["ppo"].__setitem__("schedule", "adaptive"),
        lambda value: value["ppo"].__setitem__("desired_kl", 0.01),
        lambda value: value["network"].__setitem__("actor", [220, 128, 23]),
    ):
        broken = copy.deepcopy(payload)
        mutate(broken)
        with pytest.raises(ValueError):
            validate_phase5_config(broken)

    target = tmp_path / "phase5.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    assert load_phase5_config(target).canonical_sha256 == CANONICAL_SHA256
