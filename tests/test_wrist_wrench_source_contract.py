from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from somaforce_cross.scaffold.wrist_wrench_isaac import (
    ChildJointPose,
    IsaacWristWrenchSource,
)


@dataclass
class FakeData:
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_incoming_joint_wrench_b: torch.Tensor


class FakeRobot:
    def __init__(
        self,
        body_names: list[str],
        *,
        batch_size: int = 2,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        self.body_names = body_names
        body_count = len(body_names)
        body_pos = torch.zeros(batch_size, body_count, 3, dtype=dtype, device=device)
        body_quat = torch.zeros(batch_size, body_count, 4, dtype=dtype, device=device)
        body_quat[..., 0] = 1.0
        wrench = torch.zeros(batch_size, body_count, 6, dtype=dtype, device=device)
        self.data = FakeData(body_pos, body_quat, wrench)

    def find_bodies(
        self, name: str, *, preserve_order: bool = False
    ) -> tuple[list[int], list[str]]:
        del preserve_order
        ids = [
            index
            for index, candidate in enumerate(self.body_names)
            if candidate == name
        ]
        return ids, [self.body_names[index] for index in ids]


class CountingResolver:
    def __init__(self, poses: dict[str, ChildJointPose]) -> None:
        self.poses = poses
        self.calls: list[str] = []

    def __call__(self, robot: object, body_name: str) -> ChildJointPose:
        del robot
        self.calls.append(body_name)
        return self.poses[body_name]


def _poses() -> dict[str, ChildJointPose]:
    return {
        "left_wrist_yaw_link": ChildJointPose(
            "/Robot/left_parent/left_wrist_yaw_joint",
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
        ),
        "right_wrist_yaw_link": ChildJointPose(
            "/Robot/right_parent/right_wrist_yaw_joint",
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
        ),
    }


def _source(
    robot: FakeRobot,
    *,
    warmup_steps: int = 2,
    sensor_offset: torch.Tensor | None = None,
    resolver: CountingResolver | None = None,
) -> tuple[IsaacWristWrenchSource, CountingResolver]:
    resolver = resolver or CountingResolver(_poses())
    if sensor_offset is None:
        sensor_offset = torch.zeros(2, 3, dtype=robot.data.body_pos_w.dtype)
    source = IsaacWristWrenchSource(
        robot,
        artifact_identity="fake_task/v9",
        sensor_offset_J=sensor_offset,
        warmup_steps=warmup_steps,
        joint_pose_resolver=resolver,
    )
    return source, resolver


def test_name_resolution_is_unique_and_left_right_order_is_fixed() -> None:
    robot = FakeRobot(
        ["right_wrist_yaw_link", "other", "pelvis", "left_wrist_yaw_link"]
    )
    source, _ = _source(robot)
    assert source.wrist_body_ids == (3, 0)
    assert [item.body_name for item in source.metadata] == [
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]
    assert [item.body_id for item in source.metadata] == [3, 0]
    assert all(item.artifact_identity == "fake_task/v9" for item in source.metadata)

    robot.data.body_incoming_joint_wrench_b[:, 3, 0] = 2.0
    robot.data.body_incoming_joint_wrench_b[:, 0, 1] = 3.0
    output = source.update_after_physics_step()
    assert torch.equal(output.raw_joint_wrench[:, 0, 0], torch.full((2,), 2.0))
    assert torch.equal(output.raw_joint_wrench[:, 1, 1], torch.full((2,), 3.0))


@pytest.mark.parametrize(
    "names",
    [
        ["pelvis", "right_wrist_yaw_link"],
        [
            "pelvis",
            "left_wrist_yaw_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ],
        ["left_wrist_yaw_link", "right_wrist_yaw_link"],
    ],
)
def test_name_resolution_failure_conditions(names: list[str]) -> None:
    with pytest.raises(ValueError, match="resolve uniquely"):
        _source(FakeRobot(names))


def test_joint_resolution_failure_is_not_hidden() -> None:
    robot = FakeRobot(["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"])

    def failing_resolver(robot: object, body_name: str) -> ChildJointPose:
        del robot, body_name
        raise ValueError("joint must resolve uniquely")

    with pytest.raises(ValueError, match="joint must resolve uniquely"):
        IsaacWristWrenchSource(
            robot,
            artifact_identity="fake/v1",
            sensor_offset_J=torch.zeros(2, 3, dtype=torch.float64),
            warmup_steps=1,
            joint_pose_resolver=failing_resolver,
        )


def test_nonidentity_child_transform_and_sensor_offset_are_composed() -> None:
    dtype = torch.float64
    robot = FakeRobot(
        ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"], dtype=dtype
    )
    half = torch.tensor(torch.pi / 4, dtype=dtype)
    q_z_90 = (float(torch.cos(half)), 0.0, 0.0, float(torch.sin(half)))
    poses = _poses()
    poses["left_wrist_yaw_link"] = ChildJointPose(
        "/Robot/left_joint", (1.0, 0.0, 0.0), q_z_90
    )
    resolver = CountingResolver(poses)
    offset = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=dtype)
    source, _ = _source(robot, sensor_offset=offset, resolver=resolver)
    robot.data.body_incoming_joint_wrench_b[:, 1, 0] = 2.0
    output = source.update_after_physics_step()

    expected_left = torch.tensor([0.0, -2.0, 0.0, 0.0, 0.0, -2.0], dtype=dtype)
    assert torch.allclose(
        output.clean_total_wrench_base_yaw[:, 0], expected_left.expand(2, 6), atol=1e-7
    )
    assert torch.allclose(
        source.child_local_pos[0], torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    )
    assert source.metadata[0].joint_path == "/Robot/left_joint"


def test_reset_partial_reset_and_warmup_validity() -> None:
    robot = FakeRobot(
        ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"], batch_size=3
    )
    source, _ = _source(robot, warmup_steps=2)
    first = source.update_after_physics_step()
    assert not first.valid.any()
    second = source.update_after_physics_step()
    assert second.valid.all()
    source.reset([1])
    assert torch.equal(source.age, torch.tensor([2, 0, 2]))
    assert torch.equal(
        source.valid, torch.tensor([[True, True], [False, False], [True, True]])
    )
    third = source.update_after_physics_step()
    assert torch.equal(
        third.valid, torch.tensor([[True, True], [False, False], [True, True]])
    )
    fourth = source.update_after_physics_step()
    assert fourth.valid.all()


def test_joint_resolver_is_initialization_only_and_step_ast_has_no_usd_cpu_numpy() -> (
    None
):
    robot = FakeRobot(["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"])
    source, resolver = _source(robot)
    assert resolver.calls == ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    source.update_after_physics_step()
    source.update_after_physics_step()
    assert resolver.calls == ["left_wrist_yaw_link", "right_wrist_yaw_link"]

    module_path = (
        Path(__file__).resolve().parents[1]
        / "somaforce_cross"
        / "scaffold"
        / "wrist_wrench_isaac.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "IsaacWristWrenchSource"
    )
    update = next(
        node
        for node in source_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "update_after_physics_step"
    )
    text = ast.unparse(update).lower()
    assert not any(token in text for token in ("pxr", "usd", "numpy", ".cpu"))


def test_shape_dtype_device_and_metadata_errors() -> None:
    robot = FakeRobot(["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"])
    with pytest.raises(TypeError):
        IsaacWristWrenchSource(
            robot,
            artifact_identity="fake/v1",
            sensor_offset_J=torch.zeros(2, 3, dtype=torch.int64),
            warmup_steps=1,
            joint_pose_resolver=CountingResolver(_poses()),
        )
    with pytest.raises(ValueError):
        _source(robot, warmup_steps=0)
    with pytest.raises(ValueError, match="device and dtype"):
        IsaacWristWrenchSource(
            robot,
            artifact_identity="fake/v1",
            sensor_offset_J=torch.zeros(2, 3, dtype=torch.float64, device="meta"),
            warmup_steps=1,
            joint_pose_resolver=CountingResolver(_poses()),
        )
    with pytest.raises(ValueError):
        IsaacWristWrenchSource(
            robot,
            artifact_identity="",
            sensor_offset_J=torch.zeros(2, 3, dtype=torch.float64),
            warmup_steps=1,
            joint_pose_resolver=CountingResolver(_poses()),
        )

    source, _ = _source(robot)
    robot.data.body_incoming_joint_wrench_b = torch.zeros(2, 3, 5, dtype=torch.float64)
    with pytest.raises(ValueError, match="shape"):
        source.update_after_physics_step()
    robot.data.body_incoming_joint_wrench_b = torch.zeros(2, 3, 6, dtype=torch.float32)
    with pytest.raises(ValueError, match="device and dtype"):
        source.update_after_physics_step()


def test_importing_pure_sensing_does_not_import_isaac_adapter_or_pxr() -> None:
    code = (
        "import sys; import somaforce_cross.sensing; "
        "assert 'somaforce_cross.scaffold.wrist_wrench_isaac' not in sys.modules; "
        "assert 'pxr' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_diagnostic_fixed_duration_and_evidence_contract() -> None:
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "diagnose_wrist_wrench_source.py"
    )
    tree = ast.parse(script_path.read_text(encoding="utf-8"))

    arguments: dict[str, ast.Call] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        flag = node.args[0]
        if isinstance(flag, ast.Constant) and isinstance(flag.value, str):
            arguments[flag.value] = node

    for flag in (
        "--load-steps",
        "--required-consecutive-windows",
        "--output-jsonl",
    ):
        assert flag in arguments
        assert all(keyword.arg != "default" for keyword in arguments[flag].keywords)
        keywords = {keyword.arg: keyword.value for keyword in arguments[flag].keywords}
        assert ast.literal_eval(keywords["required"]) is True

    fixed_load_loops = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and "range(args.load_steps)" in ast.unparse(node.iter)
    ]
    assert len(fixed_load_loops) == 1
    assert not any(
        isinstance(node, ast.Break) for node in ast.walk(fixed_load_loops[0])
    )

    module_text = ast.unparse(tree)
    assert "trace[-args.required_consecutive_windows:]" in module_text

    string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert {
        "convergence_step",
        "convergence_trace",
        "final_clean_wrench_mean",
        "final_raw_incoming_wrench_mean",
        "wrist_body_pose",
        "wrist_joint_state",
        "axis_error",
        "signed_axis_error",
        "absolute_error",
        "relative_error",
        "direction_cosine",
    } <= string_literals
