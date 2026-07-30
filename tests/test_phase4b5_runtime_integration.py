from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from somaforce_cross.envs.mismatch_sampler import Phase4B5MismatchSampler
from somaforce_cross.envs.mismatch import EpisodeParameterStore, PerEnvRandomStream
from somaforce_cross.envs.numeric_contract import load_numeric_contract
from somaforce_cross.envs.reset import EpisodeResetCoordinator
from somaforce_cross.envs.task_adapter import (
    adapter_contact_target_offsets,
    apply_selected_rigid_physx_parameters,
    door_custom_effort,
    rigid_physx_readback,
    validate_door_runtime_physics,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = load_numeric_contract(REPO_ROOT / "configs/phase4b5_numeric_contract.json")


class _RigidView:
    def __init__(self, rows: int) -> None:
        self.masses = torch.full((rows, 1), 2.0, dtype=torch.float32)
        self.inertias = torch.zeros(rows, 9, dtype=torch.float32)
        self.inertias[..., 0] = 1.0
        self.inertias[..., 4] = 1.0
        self.inertias[..., 8] = 1.0
        self.coms = torch.zeros(rows, 7, dtype=torch.float32)
        self.coms[..., 3] = 1.0
        self.materials = torch.zeros(rows, 1, 3, dtype=torch.float32)
        self.materials[..., :2] = 0.5
        self.transforms = torch.zeros(rows, 7, dtype=torch.float32)
        self.transforms[:, 6] = 1.0
        self.calls: list[str] = []

    def get_masses(self) -> torch.Tensor:
        self.calls.append("get_masses")
        return self.masses.clone()

    def set_masses(self, value: torch.Tensor, indices: torch.Tensor) -> None:
        self.calls.append("set_masses")
        assert value.shape == self.masses.shape
        assert indices.dtype == torch.int64 and indices.device.type == "cpu"
        self.masses[indices] = value[indices]

    def get_inertias(self) -> torch.Tensor:
        self.calls.append("get_inertias")
        return self.inertias.clone()

    def set_inertias(self, value: torch.Tensor, indices: torch.Tensor) -> None:
        self.calls.append("set_inertias")
        assert value.shape == self.inertias.shape
        assert indices.dtype == torch.int64 and indices.device.type == "cpu"
        self.inertias[indices] = value[indices]

    def get_coms(self) -> torch.Tensor:
        self.calls.append("get_coms")
        return self.coms.clone()

    def set_coms(self, value: torch.Tensor, indices: torch.Tensor) -> None:
        self.calls.append("set_coms")
        assert value.shape == self.coms.shape
        assert indices.dtype == torch.int64 and indices.device.type == "cpu"
        self.coms[indices] = value[indices]

    def get_material_properties(self) -> torch.Tensor:
        self.calls.append("get_material_properties")
        return self.materials.clone()

    def set_material_properties(
        self, value: torch.Tensor, indices: torch.Tensor
    ) -> None:
        self.calls.append("set_material_properties")
        assert value.shape == self.materials.shape
        assert indices.dtype == torch.int64 and indices.device.type == "cpu"
        self.materials[indices] = value[indices]

    def get_transforms(self) -> torch.Tensor:
        return self.transforms.clone()


def test_scaffold_only_profile_has_no_implicit_stage_or_contract_fallback() -> None:
    path = REPO_ROOT / "somaforce_cross/envs/residual_env_cfg.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    profile = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "C0SmokeProfile"
    )
    assignments = {
        node.target.id: node.value
        for node in profile.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    assert isinstance(assignments["runtime_mode"], ast.Constant)
    assert assignments["runtime_mode"].value == "c0"
    assert isinstance(assignments["scaffold_stage"], ast.Constant)
    assert assignments["scaffold_stage"].value is None
    assert isinstance(assignments["numeric_contract_path"], ast.Constant)
    assert assignments["numeric_contract_path"].value is None
    assert "scaffold_only requires an explicit C1, C2, or C3 stage" in source
    assert "scaffold_only requires an explicit v2 contract path" in source


def test_episode_index_sampling_is_per_environment_and_order_independent() -> None:
    sampler = Phase4B5MismatchSampler(CONTRACT, base_seed=20260730)
    ids = torch.tensor([7, 0, 63], dtype=torch.long)
    episode_index = torch.tensor([3, 9, 1], dtype=torch.long)
    first = sampler.sample(
        task="push_box",
        stage="C2",
        env_ids=ids,
        episode_indices=episode_index,
    )
    second = sampler.sample(
        task="push_box",
        stage="C2",
        env_ids=ids.flip(0),
        episode_indices=episode_index.flip(0),
    )
    order_first = torch.argsort(ids)
    order_second = torch.argsort(ids.flip(0))
    assert torch.equal(
        first.episode_seeds[order_first], second.episode_seeds[order_second]
    )
    assert torch.equal(first.physics[order_first], second.physics[order_second])
    changed_episode = sampler.sample(
        task="push_box",
        stage="C2",
        env_ids=ids,
        episode_indices=episode_index + 1,
    )
    assert not torch.equal(first.episode_seeds, changed_episode.episode_seeds)


def test_contact_target_transform_uses_rotated_nominal_offset_then_translation() -> (
    None
):
    nominal = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float32)
    physics = torch.zeros(1, 39, dtype=torch.float32)
    physics[:, 31:34] = torch.tensor([[3.0, -1.0, 2.0]])
    physics[:, 34:37] = torch.tensor([[0.0, 0.0, torch.pi / 2]])
    actual = adapter_contact_target_offsets(nominal, physics)
    expected = torch.tensor([[[3.0, 0.0, 2.0], [1.0, -1.0, 2.0]]])
    assert torch.allclose(actual, expected, atol=1.0e-6, rtol=0.0)

    single_nominal = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    single_physics = torch.zeros(2, 39, dtype=torch.float32)
    single_physics[:, 31:34] = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]])
    single_physics[0, 34:37] = torch.tensor([0.0, 0.0, torch.pi / 2])
    single_actual = adapter_contact_target_offsets(single_nominal, single_physics)
    single_expected = torch.tensor(
        [[[0.0, 2.0, 3.0]], [[-1.0, 1.0, 1.0]]], dtype=torch.float32
    )
    assert torch.allclose(single_actual, single_expected, atol=1.0e-6, rtol=0.0)

    invalid_nominals: tuple[object, ...] = (
        [[0.0, 0.0, 0.0]],
        torch.empty(0, 3, dtype=torch.float32),
        torch.zeros(1, 2, dtype=torch.float32),
        torch.zeros(1, 3, dtype=torch.float64),
        torch.tensor([[float("nan"), 0.0, 0.0]], dtype=torch.float32),
    )
    for invalid in invalid_nominals:
        with pytest.raises(ValueError, match="nominal contact offsets"):
            adapter_contact_target_offsets(invalid, physics)
    with pytest.raises(ValueError, match="physics_row"):
        adapter_contact_target_offsets(nominal, physics.to(torch.float64))


def test_applied_rows_are_atomically_stored_after_runtime_readback() -> None:
    events: list[str] = []

    class _Hook:
        def __init__(self, name: str) -> None:
            self.name = name

        def validate_reset(self, env_ids: torch.Tensor) -> None:
            events.append(f"validate_{self.name}")

        def reset(self, env_ids: torch.Tensor) -> None:
            events.append(f"reset_{self.name}")

    class _State:
        def __init__(self, name: str) -> None:
            self.name = name

        def reset(self, env_ids: torch.Tensor, *args: object) -> None:
            events.append(f"reset_{self.name}")

    class _NominalState(_State):
        def set_current(self, env_ids: torch.Tensor, value: torch.Tensor) -> None:
            events.append("set_nominal")

    coordinator = object.__new__(EpisodeResetCoordinator)
    coordinator.parameter_store = EpisodeParameterStore(2)
    coordinator.random_stream = PerEnvRandomStream(2)
    coordinator.scaffold = _Hook("scaffold")
    coordinator.action_sink = _Hook("sink")
    coordinator.reference_adapter = _Hook("adapter")
    coordinator.tare_calibrator = _State("tare")
    coordinator.virtual_sensor = _State("sensor")
    coordinator.contact_detector = _State("detector")
    coordinator.contact_ramp = _State("ramp")
    coordinator.wrist_history = _State("wrist")
    coordinator.nominal_action_history = _NominalState("nominal")
    coordinator.executed_action_history = _State("executed")
    coordinator.batch_size = 2
    coordinator.device = torch.device("cpu")
    coordinator.reference_step = torch.full((2,), 9, dtype=torch.long)
    coordinator.episode_length = torch.full((2,), 9, dtype=torch.long)

    sample = Phase4B5MismatchSampler(CONTRACT, base_seed=1).sample(
        task="push_box",
        stage="C1",
        env_ids=torch.tensor([0]),
        episode_indices=torch.tensor([0]),
    )
    bad = sample.physics.clone()
    bad[0, 11] = torch.nan
    before = coordinator.parameter_store.physics_mismatch.clone()
    with pytest.raises(ValueError, match="physics_mismatch.*finite"):
        coordinator.reset_after_runtime_apply(
            [0],
            applied_physics=bad,
            applied_scaffold=sample.scaffold,
            applied_sensor=sample.sensor,
            seeds=sample.episode_seeds,
            initial_wrist_frame=torch.zeros(1, 2, 14),
            current_a_nom=torch.zeros(1, 23),
        )
    assert events == []
    assert torch.equal(coordinator.parameter_store.physics_mismatch, before)

    applied = sample.physics.clone()
    applied[:, 11] += 1.0
    coordinator.reset_after_runtime_apply(
        [0],
        applied_physics=applied,
        applied_scaffold=sample.scaffold,
        applied_sensor=sample.sensor,
        seeds=sample.episode_seeds,
        initial_wrist_frame=torch.zeros(1, 2, 14),
        current_a_nom=torch.zeros(1, 23),
    )
    assert torch.equal(coordinator.parameter_store.physics_mismatch[[0]], applied)
    assert torch.equal(
        coordinator.parameter_store.scaffold_mismatch[[0]], sample.scaffold
    )
    assert events[:3] == ["validate_scaffold", "validate_sink", "validate_adapter"]
    assert "reset_sensor" in events


def test_rigid_physx_readback_leaves_unselected_rows_unchanged() -> None:
    view = _RigidView(3)
    unselected = {
        "mass": view.masses[1].clone(),
        "inertia": view.inertias[1].clone(),
        "com": view.coms[1].clone(),
        "material": view.materials[1].clone(),
    }
    selected = torch.tensor([2, 0], dtype=torch.long)
    principal_axis_quaternion = view.coms[selected, 3:].clone()
    physics = torch.zeros(2, 39, dtype=torch.float32)
    physics[:, 11] = torch.tensor([7.0, 5.0])
    physics[:, 12:15] = torch.tensor([[0.1, -0.2, 0.3], [-0.4, 0.5, -0.6]])
    physics[:, 15:21] = torch.tensor(
        [[2.0, 3.0, 4.0, 0.2, -0.1, 0.4], [5.0, 6.0, 7.0, -0.2, 0.3, -0.4]]
    )
    physics[:, 21] = torch.tensor([0.8, 0.6])
    apply_selected_rigid_physx_parameters(view, selected, physics)
    applied = rigid_physx_readback(view, selected, device=torch.device("cpu"))
    assert torch.allclose(applied[:, 11:22], physics[:, 11:22], atol=1.0e-6)
    assert torch.equal(view.coms[selected, 3:], principal_axis_quaternion)
    assert view.calls.index("set_inertias") < view.calls.index("get_coms")
    assert torch.equal(view.masses[1], unselected["mass"])
    assert torch.equal(view.inertias[1], unselected["inertia"])
    assert torch.equal(view.coms[1], unselected["com"])
    assert torch.equal(view.materials[1], unselected["material"])

    adapter_path = REPO_ROOT / "somaforce_cross/envs/task_adapters/rigid_object.py"
    source = adapter_path.read_text(encoding="utf-8")
    adapter_tree = ast.parse(source, filename=str(adapter_path))
    adapter_node = next(
        node
        for node in adapter_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RigidObjectTaskAdapter"
    )
    required_methods = (
        "_from_physx_xyzw",
        "_object_root_pose_readback",
        "_robot_root_pose_readback",
    )
    methods = [
        node
        for node in adapter_node.body
        if isinstance(node, ast.FunctionDef) and node.name in required_methods
    ]
    assert tuple(node.name for node in methods) == required_methods
    probe_node = ast.ClassDef(
        name="_RigidPoseReadbackProbe",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    namespace: dict[str, object] = {"torch": torch}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[probe_node], type_ignores=[])),
            str(adapter_path),
            "exec",
        ),
        namespace,
    )
    probe_type = namespace["_RigidPoseReadbackProbe"]

    class _RigidPoseView:
        def __init__(self, transforms: torch.Tensor) -> None:
            self.transforms = transforms
            self.calls: list[str] = []

        def get_transforms(self) -> torch.Tensor:
            self.calls.append("get_transforms")
            return self.transforms.clone()

    class _ArticulationPoseView:
        def __init__(self, transforms: torch.Tensor) -> None:
            self.transforms = transforms
            self.calls: list[str] = []

        def get_root_transforms(self) -> torch.Tensor:
            self.calls.append("get_root_transforms")
            return self.transforms.clone()

    object_xyzw = torch.tensor(
        [
            [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.4],
            [4.0, 5.0, 6.0, 0.5, 0.6, 0.7, 0.8],
            [7.0, 8.0, 9.0, 0.9, 1.0, 1.1, 1.2],
        ],
        dtype=torch.float32,
    )
    robot_xyzw = torch.tensor(
        [
            [10.0, 20.0, 30.0, 0.2, 0.3, 0.4, 0.5],
            [40.0, 50.0, 60.0, 0.6, 0.7, 0.8, 0.9],
            [70.0, 80.0, 90.0, 1.0, 1.1, 1.2, 1.3],
        ],
        dtype=torch.float32,
    )
    rigid_pose_view = _RigidPoseView(object_xyzw)
    articulation_pose_view = _ArticulationPoseView(robot_xyzw)
    adapter = object.__new__(probe_type)
    adapter.device = torch.device("cpu")
    adapter.object = SimpleNamespace(root_physx_view=rigid_pose_view)
    adapter.robot = SimpleNamespace(root_physx_view=articulation_pose_view)
    selected_pose_ids = torch.tensor([2, 0], dtype=torch.long)
    object_pose = adapter._object_root_pose_readback(selected_pose_ids)
    robot_pose = adapter._robot_root_pose_readback(selected_pose_ids)
    assert torch.equal(
        object_pose,
        torch.tensor(
            [[7.0, 8.0, 9.0, 1.2, 0.9, 1.0, 1.1], [1.0, 2.0, 3.0, 0.4, 0.1, 0.2, 0.3]],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(
        robot_pose,
        torch.tensor(
            [
                [70.0, 80.0, 90.0, 1.3, 1.0, 1.1, 1.2],
                [10.0, 20.0, 30.0, 0.5, 0.2, 0.3, 0.4],
            ],
            dtype=torch.float32,
        ),
    )
    assert object_pose.shape == robot_pose.shape == (2, 7)
    assert object_pose.dtype == robot_pose.dtype == torch.float32
    assert torch.isfinite(object_pose).all() and torch.isfinite(robot_pose).all()
    assert rigid_pose_view.calls == ["get_transforms"]
    assert articulation_pose_view.calls == ["get_root_transforms"]
    assert torch.equal(rigid_pose_view.transforms, object_xyzw)
    assert torch.equal(articulation_pose_view.transforms, robot_xyzw)

    assert "def _root_pose_readback" not in source
    assert "def _object_root_pose_readback" in source
    assert "def _robot_root_pose_readback" in source
    assert "self.object.root_physx_view.get_transforms()" in source
    assert "self.robot.root_physx_view.get_root_transforms()" in source
    test_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(test_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "somaforce_cross.envs.task_adapters.rigid_object" not in imports
    assert "self.desired_load_share_error[env_ids] = physics_row[:, 37:39]" in source
    assert "result[:, 37:39] = self.desired_load_share_error[env_ids]" in source


def test_door_deferred_slices_are_rejected_and_custom_effort_is_per_environment() -> (
    None
):
    valid = torch.zeros(1, 39, dtype=torch.float32)
    valid[:, 3:5] = torch.tensor([[0.2, 0.3]])
    validate_door_runtime_physics(torch.tensor([0]), valid, device=torch.device("cpu"))
    deferred = valid.clone()
    deferred[:, 5] = 1.0
    with pytest.raises(ValueError, match="runtime-deferred"):
        validate_door_runtime_physics(
            torch.tensor([0]), deferred, device=torch.device("cpu")
        )
    actual = door_custom_effort(
        torch.tensor([2.0, -3.0]),
        torch.tensor([0.2, 0.8]),
        torch.tensor([0.3, 0.4]),
    )
    assert torch.equal(actual, torch.tensor([-0.8, 2.0]))


def test_runtime_reset_source_orders_prevalidation_apply_readback_store_without_global_advance(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = REPO_ROOT / "somaforce_cross/envs/residual_env.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    owner_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "_ScaffoldRuntimeParameterOwner"
    )
    namespace: dict[str, object] = {"torch": torch}
    exec(
        compile(ast.Module(body=[owner_node], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    owner_type = namespace["_ScaffoldRuntimeParameterOwner"]
    runtime = SimpleNamespace(
        device=torch.device("cpu"),
        delay=torch.full((3, 1), 4, dtype=torch.long),
        alpha=torch.full((3, 1), 0.9, dtype=torch.float32),
    )
    owner = owner_type(runtime)
    selected = torch.tensor([2, 0], dtype=torch.long)
    expected = torch.tensor([[6.0, 0.97], [2.0, 0.8]], dtype=torch.float32)
    owner.apply(selected, expected)
    assert torch.equal(owner.readback(selected), expected)
    assert torch.equal(runtime.delay[1], torch.tensor([4]))
    assert torch.equal(runtime.alpha[1], torch.tensor([0.9]))

    start = source.index("def _reset_scaffold_only_idx")
    end = source.index("def _owned_sensor_state")
    reset_source = source[start:end]
    assert reset_source.index(
        "self.parameter_store.validate_rows"
    ) < reset_source.index("self.adapter.write_scene_reset")
    assert reset_source.index("self.adapter.write_scene_reset") < reset_source.index(
        "self.adapter.apply_runtime_parameters"
    )
    assert reset_source.index(
        "self.adapter.apply_runtime_parameters"
    ) < reset_source.index("self.adapter.runtime_parameter_readback")
    assert reset_source.index(
        "self.adapter.runtime_parameter_readback"
    ) < reset_source.index("reset_after_runtime_apply")
    assert "self.sim.step" not in reset_source
    assert "self.sim.forward" not in reset_source
    assert "self.random_stream.draw_virtual_ft()" in source
    assert 'self.runtime_mode == "scaffold_only"' in source
    assert "self.scaffold_runtime.validate(ids, sample.scaffold)" in reset_source
    assert "self.scaffold_runtime.apply(ids, sample.scaffold)" in reset_source
    assert "self.scaffold_runtime.readback(ids)" in reset_source

    harness_path = REPO_ROOT / "scripts/smoke_phase4b5_runtime.py"
    module_spec = importlib.util.spec_from_file_location(
        "phase4b5_runtime_harness", harness_path
    )
    assert module_spec is not None and module_spec.loader is not None
    harness = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(harness)
    for status in ("ok", "error"):
        result_path = tmp_path / f"{status}.json"
        result = {"status": status, "case": "synthetic"}
        close_events: list[str] = []

        class _Env:
            def close(self) -> None:
                assert json.loads(result_path.read_text(encoding="utf-8")) == result
                close_events.append("env.close")

        class _App:
            def close(self) -> None:
                close_events.append("app.close")

        harness._persist_worker_result(result_path, result)
        harness._close_worker_resources(
            env=_Env(), launcher=SimpleNamespace(app=_App())
        )
        markers = [
            line
            for line in capsys.readouterr().out.splitlines()
            if line == "JSON_WRITTEN"
            or line == "ENV_CLOSED"
            or line == "APP_CLOSED"
            or line.startswith(harness.RESULT_MARKER)
        ]
        assert markers == [
            "JSON_WRITTEN",
            harness.RESULT_MARKER + json.dumps(result, sort_keys=True),
            "ENV_CLOSED",
            "APP_CLOSED",
        ]
        assert close_events == ["env.close", "app.close"]
