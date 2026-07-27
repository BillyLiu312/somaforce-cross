from __future__ import annotations

import ast
import inspect
from pathlib import Path

import torch
from torch import nn

import somaforce_cross.envs as envs_api
from somaforce_cross.envs import (
    CriticObservationBundle,
    ForceSemanticOutput,
    ForceSemanticPipeline,
    PolicyObservationAssembly,
    PolicyObservationBundle,
    SemanticTargetBundle,
    build_semantic_target_bundle,
)


def test_envs_package_level_public_imports() -> None:
    expected = {
        "CriticObservationBundle": CriticObservationBundle,
        "ForceSemanticOutput": ForceSemanticOutput,
        "ForceSemanticPipeline": ForceSemanticPipeline,
        "PolicyObservationAssembly": PolicyObservationAssembly,
        "PolicyObservationBundle": PolicyObservationBundle,
        "SemanticTargetBundle": SemanticTargetBundle,
        "build_semantic_target_bundle": build_semantic_target_bundle,
    }
    assert set(expected) == set(envs_api.__all__)
    for name, value in expected.items():
        assert getattr(envs_api, name) is value


def test_envs_ast_has_no_forbidden_runtime_imports() -> None:
    envs_dir = Path(__file__).resolve().parents[1] / "somaforce_cross" / "envs"
    forbidden = ("isaac", "hdmi", "active_adaptation", "rsl_rl")
    paths = sorted(envs_dir.rglob("*.py"))
    assert paths
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
                imports.extend(alias.name for alias in node.names)
        assert not any(
            token in imported.lower() for imported in imports for token in forbidden
        ), f"forbidden import in {path}: {imports}"


def test_policy_api_exposes_only_approved_deployable_inputs() -> None:
    policy_parameters = set(
        inspect.signature(ForceSemanticPipeline.assemble_policy_observation).parameters
    )
    assert policy_parameters == {
        "self",
        "wrist_tokens",
        "proprio",
        "a_nom_history",
        "previous_a_total",
    }
    forbidden = {
        "object_pose",
        "object_state",
        "task_id",
        "object_category",
        "hdmi_command",
        "command",
        "clean_wrench",
        "simulator_contact_truth",
        "foot_wrench",
        "p_dir",
        "p_mag",
        "P_cross",
    }
    assert policy_parameters.isdisjoint(forbidden)
    assert set(inspect.signature(ForceSemanticPipeline.forward).parameters) == {
        "self",
        "wrist_tokens",
    }
    assert "z_cross" not in inspect.signature(PolicyObservationBundle).parameters


def test_semantic_target_signature_has_no_quality_or_cross_target() -> None:
    parameters = set(inspect.signature(build_semantic_target_bundle).parameters)
    assert parameters == {
        "clean_wrench",
        "contact_probability",
        "F_scale",
        "M_scale",
    }
    assert "sensor_quality" not in parameters
    assert "P_cross_target" not in parameters


def test_no_actor_critic_network_or_cross_loss_is_implemented() -> None:
    envs_dir = Path(__file__).resolve().parents[1] / "somaforce_cross" / "envs"
    forbidden_identifiers = {
        "residualactor",
        "privilegedcritic",
        "taskid",
        "p_cross_target",
        "lambda_cross",
        "cross_loss",
    }
    module_subclasses: list[str] = []
    identifiers: set[str] = set()
    for path in sorted(envs_dir.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Name, ast.Attribute)):
                identifiers.add(
                    (node.id if isinstance(node, ast.Name) else node.attr).lower()
                )
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Name) and base.id == "Module":
                        module_subclasses.append(node.name)
                    elif isinstance(base, ast.Attribute) and base.attr == "Module":
                        module_subclasses.append(node.name)
    assert identifiers.isdisjoint(forbidden_identifiers)
    assert module_subclasses == ["ForceSemanticPipeline"]
    assert issubclass(ForceSemanticPipeline, nn.Module)
    assert not issubclass(PolicyObservationBundle, nn.Module)
    assert not issubclass(CriticObservationBundle, nn.Module)


def test_residual_actor_boundary_width_has_one_semantic_latent() -> None:
    assembly = ForceSemanticPipeline().assemble_policy_observation(
        wrist_tokens=torch.full((1, 2, 16, 14), 100.0),
        proprio=torch.full((1, 64), 2.0),
        a_nom_history=torch.full((1, 23, 3), 3.0),
        previous_a_total=torch.full((1, 23), 4.0),
    )
    bundle = assembly.bundle
    actor_input = bundle.residual_actor_input()
    assert actor_input.shape == (1, 220)
    assert bundle.z_cross is assembly.semantic_output.z_cross
    assert torch.equal(actor_input[:, :64], assembly.semantic_output.z_cross)
    assert torch.equal(actor_input[:, 64:128], torch.full((1, 64), 2.0))
    assert torch.equal(actor_input[:, 128:197], torch.full((1, 69), 3.0))
    assert torch.equal(actor_input[:, 197:], torch.full((1, 23), 4.0))
    assert not torch.any(actor_input == 100.0)
