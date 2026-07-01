from __future__ import annotations

import joblib
import torch

from somaforce_cross.scaffold import (
    G1_PUSH_PULL_BOX_SCENE,
    G1_PUSH_PULL_DOOR_SCENE,
    G1_FULL_JOINT_NAMES,
    MotionTrajectory,
    MotionTrajectoryScaffold,
    ScaffoldTask,
    SonicScaffoldAdapter,
    summarize_first_scaffold_rollouts,
    make_g1_box_sonic_binding,
    make_g1_door_sonic_binding,
    make_sonic_motion_library,
    make_sonic_object_motion_library,
    make_sonic_manager_overrides,
    make_sonic_verify_command,
    make_g1_push_pull_box_trajectory,
    make_g1_push_pull_door_trajectory,
    make_g1_task_trajectory,
    make_g1_push_pull_box_scaffold,
    make_g1_push_pull_door_scaffold,
    motion_trajectory_to_sonic_entry,
    write_single_task_sonic_motion_file,
    write_single_task_sonic_object_motion_file,
)


def test_motion_trajectory_scaffold_outputs_nominal_action() -> None:
    action_dim = len(G1_FULL_JOINT_NAMES)
    joint_pos = torch.arange(3 * action_dim, dtype=torch.float32).reshape(3, action_dim)
    action_offset = torch.ones(action_dim)
    action_scale = torch.full((action_dim,), 2.0)
    trajectory = MotionTrajectory(joint_pos=joint_pos, time_s=torch.tensor([0.0, 0.1, 0.2]))

    scaffold = MotionTrajectoryScaffold(
        trajectory,
        task=ScaffoldTask.PUSH_PULL_DOOR,
        action_offset=action_offset,
        action_scale=action_scale,
        action_clip=None,
    )
    out = scaffold.at_time(0.11)

    assert out.task == ScaffoldTask.PUSH_PULL_DOOR
    assert torch.equal(out.nominal_joint_pos, joint_pos[1])
    assert torch.equal(out.a_nom, (joint_pos[1] - action_offset) / action_scale)
    assert out.confidence is not None
    assert torch.equal(out.confidence, torch.ones(1))


def test_motion_trajectory_rejects_wrong_joint_count() -> None:
    wrong_joint_pos = torch.zeros(4, len(G1_FULL_JOINT_NAMES) - 1)

    try:
        MotionTrajectory(joint_pos=wrong_joint_pos)
    except ValueError as exc:
        assert "G1 29-DOF order" in str(exc)
    else:
        raise AssertionError("MotionTrajectory accepted a non-G1 joint dimension")


def test_motion_trajectory_scaffold_selects_batched_frames_per_env() -> None:
    action_dim = len(G1_FULL_JOINT_NAMES)
    joint_pos = torch.arange(2 * 4 * action_dim, dtype=torch.float32).reshape(2, 4, action_dim)
    hand_ref = torch.arange(2 * 4 * 3 * 3, dtype=torch.float32).reshape(2, 4, 3, 3)
    trajectory = MotionTrajectory(joint_pos=joint_pos, hand_pose_w=hand_ref)
    scaffold = MotionTrajectoryScaffold(
        trajectory,
        task=ScaffoldTask.PUSH_PULL_BOX,
        action_clip=None,
    )

    out = scaffold.at_frame(torch.tensor([1, 3]))

    assert torch.equal(out.a_nom[0], joint_pos[0, 1])
    assert torch.equal(out.a_nom[1], joint_pos[1, 3])
    assert out.nominal_hand_ref is not None
    assert torch.equal(out.nominal_hand_ref[0], hand_ref[0, 1])
    assert torch.equal(out.nominal_hand_ref[1], hand_ref[1, 3])


def test_sonic_adapter_reads_motion_command_and_nominal_action(monkeypatch) -> None:
    class FakeMotionCommand:
        joint_pos = torch.tensor([[0.2, 0.4]])
        vr_3point_body_pos_w_multi_future = torch.ones(1, 5, 3, 3)
        body_pos_w_multi_future = torch.ones(1, 5, 14, 3)
        command_multi_future = torch.ones(1, 5, 10)

    class FakeCommandManager:
        def get_term(self, name: str) -> FakeMotionCommand:
            assert name == "motion"
            return FakeMotionCommand()

    class FakeEnv:
        command_manager = FakeCommandManager()

    def fake_residual_action(env, command_name):
        assert isinstance(env, FakeEnv)
        assert command_name == "motion"
        return torch.tensor([[2.0, -3.0]])

    monkeypatch.setattr(SonicScaffoldAdapter, "_residual_joint_pos_action", staticmethod(fake_residual_action))

    out = SonicScaffoldAdapter(FakeEnv(), task=ScaffoldTask.PUSH_PULL_BOX, action_clip=2.5).get_output()

    assert out.task == ScaffoldTask.PUSH_PULL_BOX
    assert torch.equal(out.a_nom, torch.tensor([[2.0, -2.5]]))
    assert torch.equal(out.nominal_joint_pos, FakeMotionCommand.joint_pos)
    assert out.nominal_hand_ref is FakeMotionCommand.vr_3point_body_pos_w_multi_future


def test_g1_scene_factories_bind_expected_tasks() -> None:
    joint_pos = torch.zeros(2, len(G1_FULL_JOINT_NAMES))
    trajectory = MotionTrajectory(joint_pos=joint_pos)

    door = make_g1_push_pull_door_scaffold(trajectory).at_frame(0)
    box = make_g1_push_pull_box_scaffold(trajectory).at_frame(1)

    assert G1_PUSH_PULL_DOOR_SCENE.robot == "unitree_g1"
    assert G1_PUSH_PULL_DOOR_SCENE.object_type == "articulated_door"
    assert door.task == ScaffoldTask.PUSH_PULL_DOOR
    assert G1_PUSH_PULL_BOX_SCENE.robot == "unitree_g1"
    assert G1_PUSH_PULL_BOX_SCENE.object_type == "rigid_box"
    assert box.task == ScaffoldTask.PUSH_PULL_BOX


def test_default_g1_door_scaffold_generates_push_and_pull_nominal_actions() -> None:
    push = make_g1_push_pull_door_scaffold(interaction_mode="push", num_frames=8, action_clip=None)
    pull = make_g1_push_pull_door_scaffold(interaction_mode="pull", num_frames=8, action_clip=None)

    push_start = push.at_frame(0)
    push_end = push.at_frame(7)
    pull_end = pull.at_frame(7)

    assert push_start.a_nom.shape == (len(G1_FULL_JOINT_NAMES),)
    assert push_end.task == ScaffoldTask.PUSH_PULL_DOOR
    assert push_end.nominal_hand_ref is not None
    assert push_end.nominal_body_ref is not None
    assert push_end.cmd_6d is not None
    assert push_end.cmd_6d[0] > 0
    assert pull_end.cmd_6d is not None
    assert pull_end.cmd_6d[0] < 0
    assert not torch.equal(push_start.a_nom, push_end.a_nom)
    assert not torch.equal(push_end.a_nom, pull_end.a_nom)


def test_default_g1_box_scaffold_generates_two_hand_push_and_pull_actions() -> None:
    push = make_g1_push_pull_box_scaffold(interaction_mode="push", num_frames=8, action_clip=None).at_frame(7)
    pull = make_g1_push_pull_box_scaffold(interaction_mode="pull", num_frames=8, action_clip=None).at_frame(7)

    assert push.a_nom.shape == (len(G1_FULL_JOINT_NAMES),)
    assert push.task == ScaffoldTask.PUSH_PULL_BOX
    assert push.nominal_hand_ref is not None
    assert push.nominal_hand_ref.shape == (2, 3)
    assert push.cmd_6d is not None
    assert pull.cmd_6d is not None
    assert push.cmd_6d[0] > 0
    assert pull.cmd_6d[0] < 0
    assert not torch.equal(push.a_nom, pull.a_nom)


def test_g1_task_trajectory_dispatches_first_scaffold_tasks() -> None:
    door = make_g1_task_trajectory(ScaffoldTask.PUSH_PULL_DOOR, mode="push", num_frames=5)
    box = make_g1_task_trajectory(ScaffoldTask.PUSH_PULL_BOX, mode="pull", num_frames=6)

    assert torch.equal(door.joint_pos, make_g1_push_pull_door_trajectory("push", num_frames=5).joint_pos)
    assert torch.equal(box.joint_pos, make_g1_push_pull_box_trajectory("pull", num_frames=6).joint_pos)
    assert door.time_s is not None
    assert box.time_s is not None
    assert door.joint_pos.shape == (5, len(G1_FULL_JOINT_NAMES))
    assert box.joint_pos.shape == (6, len(G1_FULL_JOINT_NAMES))


def test_scaffold_rollout_diagnostics_cover_first_task_variants() -> None:
    summaries = summarize_first_scaffold_rollouts(num_frames=8, duration_s=0.2)

    assert [summary.label for summary in summaries] == [
        "push_pull_door_push",
        "push_pull_door_pull",
        "push_pull_box_push",
        "push_pull_box_pull",
    ]
    for summary in summaries:
        assert summary.a_nom_shape == (8, len(G1_FULL_JOINT_NAMES))
        assert summary.joint_delta_norm > 0.0
        assert summary.max_abs_a_nom > 0.0
        assert summary.direction_ok
    assert summaries[0].right_hand_x_delta > 0.0
    assert summaries[1].right_hand_x_delta < 0.0
    assert summaries[2].left_hand_x_delta > 0.0
    assert summaries[3].left_hand_x_delta < 0.0


def test_sonic_motion_entry_contains_motion_lib_fields() -> None:
    trajectory = make_g1_push_pull_door_trajectory("push", num_frames=5, duration_s=0.1)

    entry = motion_trajectory_to_sonic_entry(trajectory, fps=50)

    assert entry["root_trans_offset"].shape == (5, 3)
    assert entry["pose_aa"].shape == (5, 30, 3)
    assert entry["dof"].shape == (5, len(G1_FULL_JOINT_NAMES))
    assert entry["root_rot"].shape == (5, 4)
    assert entry["smpl_joints"].shape == (5, 24, 3)
    assert entry["object_root_pos"].shape == (5, 1, 3)
    assert entry["object_root_quat"].shape == (5, 1, 4)
    assert entry["fps"] == 50
    assert torch.allclose(torch.from_numpy(entry["dof"]), trajectory.joint_pos)
    assert torch.allclose(torch.from_numpy(entry["root_trans_offset"][:, 2]), torch.full((5,), 0.76))
    assert torch.allclose(torch.from_numpy(entry["object_root_pos"][0, 0]), torch.tensor([0.72, 0.0, 0.75]))
    assert torch.allclose(torch.from_numpy(entry["object_root_quat"][0, 0]), torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_sonic_motion_library_writes_first_task_variants(tmp_path) -> None:
    path = tmp_path / "motions.pkl"

    library = write_single_task_sonic_motion_file(
        path,
        task=ScaffoldTask.PUSH_PULL_BOX,
        mode="pull",
        num_frames=7,
        duration_s=0.12,
        fps=50,
    )
    loaded = joblib.load(path)
    object_path = tmp_path / "objects.pkl"
    object_library = write_single_task_sonic_object_motion_file(
        object_path,
        task=ScaffoldTask.PUSH_PULL_BOX,
        mode="pull",
        num_frames=7,
        fps=50,
    )
    loaded_objects = joblib.load(object_path)
    all_motions = make_sonic_motion_library(num_frames=7, duration_s=0.12, fps=50)
    all_objects = make_sonic_object_motion_library(("somaforce_g1_box_pull",), num_frames=7, fps=50)

    assert list(library.keys()) == ["somaforce_g1_box_pull"]
    assert list(loaded.keys()) == ["somaforce_g1_box_pull"]
    assert list(object_library.keys()) == ["somaforce_g1_box_pull"]
    assert loaded_objects["somaforce_g1_box_pull"]["root_pos"].shape == (7, 1, 3)
    assert loaded_objects["somaforce_g1_box_pull"]["root_quat"].shape == (7, 1, 4)
    assert loaded["somaforce_g1_box_pull"]["dof"].shape == (7, len(G1_FULL_JOINT_NAMES))
    assert all_objects["somaforce_g1_box_pull"]["root_quat"][0, 0, 0] == 1.0
    assert set(all_motions) == {
        "somaforce_g1_door_push",
        "somaforce_g1_door_pull",
        "somaforce_g1_box_push",
        "somaforce_g1_box_pull",
    }


def test_sonic_manager_overrides_keep_door_out_of_rigid_object_path(tmp_path) -> None:
    motion_file = tmp_path / "door.pkl"
    binding = make_g1_door_sonic_binding(
        motion_file,
        interaction_mode="pull",
        door_asset_path="/assets/door.usd",
    )

    overrides = make_sonic_manager_overrides(binding, experiment_dir=tmp_path / "exp")

    assert binding.requires_articulation_scene
    assert binding.motion_key == "somaforce_g1_door_pull"
    assert "+manager_env.config.add_object=false" in overrides
    assert all("object_usd_path" not in item for item in overrides)
    assert f"manager_env.commands.motion.motion_lib_cfg.motion_file={motion_file}" in overrides


def test_sonic_manager_overrides_route_box_as_rigid_object(tmp_path) -> None:
    motion_file = tmp_path / "box.pkl"
    binding = make_g1_box_sonic_binding(
        motion_file,
        interaction_mode="push",
        box_usd_path="/assets/box.usd",
        object_mass=3.5,
    )

    overrides = make_sonic_manager_overrides(binding, num_envs=2, experiment_dir=tmp_path / "exp")
    command = make_sonic_verify_command(
        ScaffoldTask.PUSH_PULL_BOX,
        "push",
        motion_file,
        object_usd_path="/assets/box.usd",
    )

    assert not binding.requires_articulation_scene
    assert binding.motion_key == "somaforce_g1_box_push"
    assert "num_envs=2" in overrides
    assert "+manager_env.config.add_object=true" in overrides
    assert "+manager_env.config.object_usd_path=/assets/box.usd" in overrides
    assert "manager_env.config.object_mass=3.5" in overrides
    assert f"+manager_env.commands.motion.motion_lib_cfg.object_motion_file={motion_file.with_name('box_object.pkl')}" in overrides
    assert "+manager_env.commands.motion.motion_lib_cfg.max_num_objects=1" in overrides
    assert command[-3:] == [str(motion_file), "--object-usd-path", "/assets/box.usd"]
