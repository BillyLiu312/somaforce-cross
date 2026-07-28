from __future__ import annotations

import ast
import inspect
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn

import somaforce_cross.envs as envs_api
from somaforce_cross.envs import (
    CriticObservationBundle,
    EpisodeParameterStore,
    EpisodeResetCoordinator,
    ForceSemanticOutput,
    ForceSemanticPipeline,
    NominalActionHistory,
    PerEnvRandomStream,
    PolicyObservationAssembly,
    PolicyObservationBundle,
    SelectedResetHook,
    SemanticTargetBundle,
    VirtualFTDraws,
    VirtualFTSensorParameters,
    build_semantic_target_bundle,
    validate_env_ids,
)


def test_envs_package_level_public_imports() -> None:
    expected = {
        "CriticObservationBundle": CriticObservationBundle,
        "EpisodeParameterStore": EpisodeParameterStore,
        "EpisodeResetCoordinator": EpisodeResetCoordinator,
        "ForceSemanticOutput": ForceSemanticOutput,
        "ForceSemanticPipeline": ForceSemanticPipeline,
        "NominalActionHistory": NominalActionHistory,
        "PerEnvRandomStream": PerEnvRandomStream,
        "PolicyObservationAssembly": PolicyObservationAssembly,
        "PolicyObservationBundle": PolicyObservationBundle,
        "SelectedResetHook": SelectedResetHook,
        "SemanticTargetBundle": SemanticTargetBundle,
        "VirtualFTDraws": VirtualFTDraws,
        "VirtualFTSensorParameters": VirtualFTSensorParameters,
        "build_semantic_target_bundle": build_semantic_target_bundle,
        "validate_env_ids": validate_env_ids,
    }
    assert set(expected) == set(envs_api.__all__)
    for name, value in expected.items():
        assert getattr(envs_api, name) is value


def test_envs_ast_has_no_forbidden_runtime_imports() -> None:
    envs_dir = Path(__file__).resolve().parents[1] / "somaforce_cross" / "envs"
    forbidden = ("isaac", "hdmi", "active_adaptation", "rsl_rl")
    paths = [
        envs_dir / "__init__.py",
        envs_dir / "action_history.py",
        envs_dir / "mismatch.py",
        envs_dir / "observations.py",
        envs_dir / "reset.py",
        envs_dir / "task_adapter.py",
        envs_dir / "task_adapters/__init__.py",
    ]
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


def test_sensing_ast_has_no_environment_layer_dependency() -> None:
    sensing_dir = Path(__file__).resolve().parents[1] / "somaforce_cross" / "sensing"
    paths = sorted(sensing_dir.rglob("*.py"))
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
            imported == "somaforce_cross.envs"
            or imported.startswith("somaforce_cross.envs.")
            for imported in imports
        ), f"sensing layer imports envs in {path}: {imports}"


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


def test_phase4b2_has_no_task_reward_termination_or_distribution_logic() -> None:
    envs_dir = Path(__file__).resolve().parents[1] / "somaforce_cross" / "envs"
    forbidden_identifiers = {
        "directrlenv",
        "reward",
        "rewards",
        "terminated",
        "termination",
        "terminations",
        "timeout",
        "timeouts",
        "residualactor",
        "privilegedcritic",
    }
    task_names = {
        "push_door_hand",
        "push_door-hand",
        "push_box",
        "move_suitcase",
        "move_largebox",
    }
    identifiers: set[str] = set()
    string_values: set[str] = set()
    pure_phase4b2_paths = (
        envs_dir / "action_history.py",
        envs_dir / "mismatch.py",
        envs_dir / "observations.py",
        envs_dir / "reset.py",
    )
    for path in pure_phase4b2_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.add(node.id.lower())
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr.lower())
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                string_values.add(node.value.lower())
    assert identifiers.isdisjoint(forbidden_identifiers)
    assert string_values.isdisjoint(task_names)


def test_isaac_environment_modules_are_not_imported_by_package_init() -> None:
    init_path = Path(envs_api.__file__).resolve()
    source = init_path.read_text(encoding="utf-8")
    assert "residual_env" not in source
    assert "task_adapters" not in source
    command = [
        sys.executable,
        "-c",
        (
            "import sys; import somaforce_cross.envs; "
            "assert not any(name.startswith('isaaclab') or name.startswith('omni') "
            "for name in sys.modules)"
        ),
    ]
    subprocess.run(command, cwd=init_path.parents[2], check=True)


def test_runner_launches_app_before_delayed_isaac_environment_imports() -> None:
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    for runner in (
        scripts / "smoke_phase4_door_env.py",
        scripts / "smoke_phase4_multitask_env.py",
    ):
        source = runner.read_text(encoding="utf-8")
        launch_marker = (
            "app_launcher = AppLauncher(args)"
            if runner.name == "smoke_phase4_door_env.py"
            else "launcher = AppLauncher(args)"
        )
        launch_index = source.index(launch_marker)
        dispatch_index = source.index("_run_residual(args, simulation_app)")
        assert launch_index < dispatch_index
        assert "from somaforce_cross.envs.residual_env import" in source
        assert "def _run_residual" in source


def test_multitask_runner_and_environment_stop_at_c0_without_learning_code() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "scripts/smoke_phase4_multitask_env.py",
        root / "somaforce_cross/envs/residual_env.py",
        root / "somaforce_cross/envs/task_adapters/rigid_object.py",
        root / "somaforce_cross/envs/task_adapters/push_box.py",
        root / "somaforce_cross/envs/task_adapters/move_payload.py",
    )
    forbidden_imports = ("rsl_rl", "active_adaptation")
    forbidden_identifiers = {"residualactor", "privilegedcritic", "ppo"}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports: list[str] = []
        identifiers: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.Name):
                identifiers.add(node.id.lower())
            elif isinstance(node, ast.Attribute):
                identifiers.add(node.attr.lower())
        assert not any(
            token in name.lower() for token in forbidden_imports for name in imports
        )
        assert identifiers.isdisjoint(forbidden_identifiers)


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
