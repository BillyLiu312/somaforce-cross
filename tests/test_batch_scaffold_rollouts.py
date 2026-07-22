from __future__ import annotations

from pathlib import Path

from scripts.batch_rollout_and_render_pretrained_hdmi_scaffolds import (
    DEFAULT_MATRIX,
    TASK_CATEGORIES,
    build_case_commands,
    load_cases,
)


def _flag_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_default_matrix_covers_every_task_and_case() -> None:
    cases = load_cases(DEFAULT_MATRIX)
    assert len(cases) == 14
    assert {case.task for case in cases} == set(TASK_CATEGORIES)
    assert {(case.task, case.condition) for case in cases} == {
        ("push_door_hand", "nominal"),
        ("push_door_hand", "high_friction"),
        ("push_door_hand", "high_damping"),
        ("push_box", "nominal"),
        ("push_box", "light_slippery"),
        ("push_box", "heavy_friction"),
        ("move_suitcase", "nominal"),
        ("move_suitcase", "light"),
        ("move_suitcase", "heavy"),
        ("move_suitcase", "stress"),
        ("move_largebox", "nominal"),
        ("move_largebox", "light"),
        ("move_largebox", "heavy"),
        ("move_largebox", "stress"),
    }


def test_case_commands_share_parameters_and_classify_video(tmp_path: Path) -> None:
    case = next(
        case
        for case in load_cases(DEFAULT_MATRIX)
        if (case.task, case.condition) == ("move_suitcase", "heavy")
    )
    commands = build_case_commands(
        case,
        python=Path("/opt/isaac-python"),
        device="cuda:1",
        videos_root=tmp_path / "videos",
        width=640,
        height=480,
        fps=50,
    )

    assert commands.metrics.name == "rollout_metrics_heavy.json"
    assert commands.video == tmp_path / "videos/move_suitcase/heavy.mp4"
    for flag in ("--steps", "--object-mass", "--object-friction"):
        assert _flag_value(commands.rollout, flag) == _flag_value(commands.render, flag)
    assert _flag_value(commands.rollout, "--device") == "cuda:1"
    assert _flag_value(commands.render, "--output") == str(commands.video)
