from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEARNING = ROOT / "somaforce_cross" / "learning"
SCRIPTS = ROOT / "scripts"
FORBIDDEN = ("isaac", "omni", "carb", "hdmi", "active_adaptation", "rsl_rl")


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append(node.module or "")
            found.extend(alias.name for alias in node.names)
    return found


def _top_level_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append(node.module or "")
    return found


def test_pure_learning_modules_have_no_isaac_hdmi_or_rsl_imports() -> None:
    paths = sorted(LEARNING.glob("*.py"))
    assert paths
    for path in paths:
        imports = _imports(path)
        assert not any(
            token in imported.lower() for token in FORBIDDEN for imported in imports
        ), f"forbidden import in {path}: {imports}"

    command = [
        sys.executable,
        "-c",
        (
            "import sys; import somaforce_cross.learning; "
            "assert not any(name.startswith(('isaaclab', 'omni', 'carb', 'rsl_rl')) "
            "for name in sys.modules)"
        ),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def test_smoke_delays_isaac_environment_and_rsl_imports_until_worker_factory() -> None:
    smoke = SCRIPTS / "smoke_phase5_ppo.py"
    train = SCRIPTS / "train_phase5.py"
    source = smoke.read_text(encoding="utf-8")
    assert (
        "from somaforce_cross.envs.residual_env import SomaForceResidualEnv" in source
    )
    assert "launcher = AppLauncher(args)" in source
    assert "def make_phase5_runner" not in source
    assert not any(
        token in imported.lower()
        for token in ("rsl_rl", "active_adaptation")
        for imported in _imports(smoke)
    )
    assert not any(
        token in imported.lower()
        for token in ("isaaclab", "omni", "carb", "rsl_rl")
        for imported in _top_level_imports(train)
    )
    assert "phase5_learning_acceptance" in train.read_text(encoding="utf-8")
    train_tree = ast.parse(train.read_text(encoding="utf-8"), filename=str(train))
    local_forbidden_imports: dict[str, list[str]] = {}
    for function in (
        node for node in train_tree.body if isinstance(node, ast.FunctionDef)
    ):
        imports = [
            imported
            for node in ast.walk(function)
            if isinstance(node, ast.ImportFrom)
            for imported in [node.module or ""]
            if any(
                token in imported.lower()
                for token in ("isaaclab", "omni", "carb", "rsl_rl")
            )
        ]
        if imports:
            local_forbidden_imports[function.name] = imports
    assert local_forbidden_imports == {
        "_worker_parser": ["isaaclab.app"],
        "_worker_main": ["isaaclab.app"],
    }
    worker = next(
        node
        for node in train_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_worker_main"
    )
    worker_source = ast.get_source_segment(train.read_text(encoding="utf-8"), worker)
    assert worker_source is not None
    launcher_index = worker_source.index("launcher = AppLauncher(args)")
    for call in (
        "_contract_context(args)",
        "_run_contact_probe",
        "_run_semantic_probe",
        "_run_learning",
        "_run_final_evaluation",
    ):
        assert launcher_index < worker_source.index(call)
    runner_source = (LEARNING / "runner.py").read_text(encoding="utf-8")
    runner_tree = ast.parse(runner_source, filename=str(LEARNING / "runner.py"))
    runner_identifiers = {
        node.id for node in ast.walk(runner_tree) if isinstance(node, ast.Name)
    } | {node.attr for node in ast.walk(runner_tree) if isinstance(node, ast.Attribute)}
    assert "OnPolicyRunner" not in runner_identifiers
    assert 'importlib.import_module("rsl_rl.storage")' in runner_source
