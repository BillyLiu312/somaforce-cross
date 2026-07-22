#!/usr/bin/env python3
"""Materialize the audited HDMI teacher as a standalone local artifact.

This is an export/oracle-boundary tool. It intentionally reads a trusted HDMI
checkpoint and source assets. The generated play/runtime artifact does not
import HDMI or load the source checkpoint pickle.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from somaforce_cross.scaffold.pretrained_hdmi import (
    HDMINetworkContract,
    HDMITaskSpec,
    FrozenHDMITeacherPolicy,
    HDMIObservationBatch,
    get_hdmi_task_spec,
    sha256_file,
)


EXPECTED_HDMI_REVISION = "32282f6dcf26cae70b814d585ceb12cc38aa1b60"
SOURCE_SPECS = {
    "push_door_hand": {
        "checkpoint": "outputs/doorpushhand/hdmi-doorpushhand-4gpu/rank_0/checkpoint_final.pt",
        "motion_dir": "data/motion/data_for_sim/push_door-hand-0828",
        "g1_asset": "active_adaptation/assets/g1/g1_29dof_rubberhand-feet_sphere-eef_box-body_capsule.usd",
        "object_asset": "active_adaptation/assets/objects/door/door.usd",
        "checksums": {
            "checkpoint": "24682b7b66c38f784ca5939dad3841b1b1e5b801c47d2fea59c5f85fd90c238c",
            "resolved_config": "fb6ddd53135550b244fd35b707cf2b98e069ddf38dfe456af5b930e4189a074c",
        },
    },
    "push_box": {
        "checkpoint": "outputs/push_box/hdmi-push-box-4gpu-20260721_200242/rank_0/checkpoint_final.pt",
        "motion_dir": "data/motion/g1/push_box/push_box-VID_20250423_220958-light-high-adjust_root_height",
        "g1_asset": "active_adaptation/assets/g1/g1_29dof_nohand-feet_sphere-eef_L-body_capsule.usd",
        "object_asset": "active_adaptation/assets/objects/box/box.usd",
        "reference_video": "scripts/recording-07-21_21-43.mp4",
        "checksums": {
            "checkpoint": "83ddea2ed34919be575d678fcac43c3b3d65643a22d3e45ea995f80e9c0a6d41",
            "resolved_config": "5902fe6b0e44a597c9e98504cfbb83110b6baf036d0b827cff122bfdc2a937c3",
            "asset_meta": "314eae4a38925411cc37cdb56a7fb8a57174d51ae130118f23f0e4aa2b92537d",
            "motion": "8921f589f8470d25ab1dd0fa65f9b9057676ab122fdcc8d87f43d9c527ed461e",
            "motion_meta": "dffeeed04f3001710bb8ab83061e8e1d1be256d9e2e35a4efa0c6011ae1d4105",
            "reference_video": "65d39e36ce7dded75e9f3c1513aa4a6641589e763fea28c25289a799e2fd7a71",
        },
    },
}


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdmi-root", type=Path, default=workspace / "HDMI")
    parser.add_argument(
        "--task", choices=tuple(SOURCE_SPECS), default="push_door_hand"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def copy_tensor(target: torch.Tensor, source: torch.Tensor) -> None:
    if target.shape != source.shape:
        raise ValueError(f"tensor shape mismatch: {target.shape} != {source.shape}")
    target.copy_(source)


def discover_network_contract(checkpoint: dict[str, object]) -> HDMINetworkContract:
    source_policy = checkpoint["policy"]
    encoder = source_policy["encoder_priv"]
    actor = source_policy["actor"]
    return HDMINetworkContract(
        privileged_encoder_input_dim=int(encoder["module.1.module.0.0.weight"].shape[1]),
        privileged_hidden_dim=int(encoder["module.1.module.1.weight"].shape[0]),
        actor_input_dim=int(actor["module.0.module.1.module.0.weight"].shape[1]),
        actor_hidden_dims=(
            int(actor["module.0.module.1.module.0.weight"].shape[0]),
            int(actor["module.0.module.1.module.3.weight"].shape[0]),
            int(actor["module.0.module.1.module.6.weight"].shape[0]),
        ),
        action_dim=int(actor["module.0.module.2.module.actor_mean.weight"].shape[0]),
    )


def build_portable_policy(
    checkpoint: dict[str, object], task_spec: HDMITaskSpec
) -> FrozenHDMITeacherPolicy:
    policy_state = checkpoint["policy"]
    source_encoder = policy_state["encoder_priv"]
    source_actor = policy_state["actor"]
    discovered = discover_network_contract(checkpoint)
    if discovered != task_spec.network:
        raise ValueError(
            f"checkpoint network contract {discovered} does not match {task_spec.network}"
        )
    model = FrozenHDMITeacherPolicy(task_spec.observation_dims, discovered)
    mapping = {
        "priv_fc.weight": source_encoder["module.1.module.0.0.weight"],
        "priv_fc.bias": source_encoder["module.1.module.0.0.bias"],
        "priv_ln.weight": source_encoder["module.1.module.0.1.weight"],
        "priv_ln.bias": source_encoder["module.1.module.0.1.bias"],
        "priv_out.weight": source_encoder["module.1.module.1.weight"],
        "priv_out.bias": source_encoder["module.1.module.1.bias"],
        "actor_fc1.weight": source_actor["module.0.module.1.module.0.weight"],
        "actor_fc1.bias": source_actor["module.0.module.1.module.0.bias"],
        "actor_ln1.weight": source_actor["module.0.module.1.module.1.weight"],
        "actor_ln1.bias": source_actor["module.0.module.1.module.1.bias"],
        "actor_fc2.weight": source_actor["module.0.module.1.module.3.weight"],
        "actor_fc2.bias": source_actor["module.0.module.1.module.3.bias"],
        "actor_ln2.weight": source_actor["module.0.module.1.module.4.weight"],
        "actor_ln2.bias": source_actor["module.0.module.1.module.4.bias"],
        "actor_fc3.weight": source_actor["module.0.module.1.module.6.weight"],
        "actor_fc3.bias": source_actor["module.0.module.1.module.6.bias"],
        "actor_ln3.weight": source_actor["module.0.module.1.module.7.weight"],
        "actor_ln3.bias": source_actor["module.0.module.1.module.7.bias"],
        "actor_mean.weight": source_actor["module.0.module.2.module.actor_mean.weight"],
        "actor_mean.bias": source_actor["module.0.module.2.module.actor_mean.bias"],
    }
    state = model.state_dict()
    with torch.no_grad():
        for name, source in mapping.items():
            copy_tensor(state[name], source.detach().cpu())

        stats = checkpoint["vecnorm"]["_extra_state"]
        for name in ("command", "policy", "object", "privileged"):
            source_name = "priv" if name == "privileged" else name
            count = stats[f"{source_name}_count"]
            mean = stats[f"{source_name}_sum"] / count
            scale = (
                stats[f"{source_name}_ssq"] / count - mean.square()
            ).clamp_min(1e-4).sqrt()
            copy_tensor(state[f"{name}_mean"], mean.detach().cpu())
            copy_tensor(state[f"{name}_scale"], scale.detach().cpu())
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    return model


def oracle_action(
    model: FrozenHDMITeacherPolicy, observation: HDMIObservationBatch
) -> torch.Tensor:
    """Evaluate source tensors with the source checkpoint's exact graph order."""

    state = model.state_dict()
    normalize = lambda name, value: (value - state[f"{name}_mean"]) / state[f"{name}_scale"]
    command = normalize("command", observation.command)
    policy = normalize("policy", observation.policy)
    object_obs = normalize("object", observation.object)
    privileged = normalize("privileged", observation.privileged)
    mish = torch.nn.functional.mish
    feature = torch.nn.functional.linear(
        torch.cat((privileged, object_obs), dim=-1),
        state["priv_fc.weight"],
        state["priv_fc.bias"],
    )
    feature = torch.nn.functional.layer_norm(
        feature,
        (256,),
        state["priv_ln.weight"],
        state["priv_ln.bias"],
    )
    feature = torch.nn.functional.linear(
        mish(feature), state["priv_out.weight"], state["priv_out.bias"]
    )
    actor = torch.cat((command, policy, feature), dim=-1)
    for prefix, width in zip(
        ("actor_fc1", "actor_fc2", "actor_fc3"),
        model.network_contract.actor_hidden_dims,
        strict=True,
    ):
        actor = torch.nn.functional.linear(
            actor, state[f"{prefix}.weight"], state[f"{prefix}.bias"]
        )
        ln_prefix = prefix.replace("fc", "ln")
        actor = torch.nn.functional.layer_norm(
            actor,
            (width,),
            state[f"{ln_prefix}.weight"],
            state[f"{ln_prefix}.bias"],
        )
        actor = mish(actor)
    residual = torch.nn.functional.linear(
        actor, state["actor_mean.weight"], state["actor_mean.bias"]
    )
    return observation.reference_action + residual


def hdmi_source_oracle_action(
    hdmi_root: Path,
    checkpoint: dict[str, object],
    observation: HDMIObservationBatch,
    task_spec: HDMITaskSpec,
) -> torch.Tensor:
    """Run the fixed batch through HDMI's own make_mlp and Actor classes."""

    sys.path.insert(0, str(hdmi_root))
    try:
        from active_adaptation.learning.ppo.common import Actor, make_mlp

        source_policy = checkpoint["policy"]
        latent = task_spec.network.privileged_hidden_dim
        encoder = torch.nn.Sequential(make_mlp([latent]), torch.nn.LazyLinear(latent))
        encoder(torch.zeros(1, task_spec.network.privileged_encoder_input_dim))
        encoder.load_state_dict(
            {
                key.removeprefix("module.1.module."): value
                for key, value in source_policy["encoder_priv"].items()
            },
            strict=True,
        )
        actor_backbone = make_mlp(list(task_spec.network.actor_hidden_dims))
        actor_backbone(torch.zeros(1, task_spec.network.actor_input_dim))
        actor_backbone.load_state_dict(
            {
                key.removeprefix("module.0.module.1.module."): value
                for key, value in source_policy["actor"].items()
                if key.startswith("module.0.module.1.module.")
            },
            strict=True,
        )
        actor_head = Actor(task_spec.network.action_dim)
        actor_head(torch.zeros(1, task_spec.network.actor_hidden_dims[-1]))
        actor_head.load_state_dict(
            {
                key.removeprefix("module.0.module.2.module."): value
                for key, value in source_policy["actor"].items()
                if key.startswith("module.0.module.2.module.")
            },
            strict=True,
        )
        stats = checkpoint["vecnorm"]["_extra_state"]

        def normalize(name: str, value: torch.Tensor) -> torch.Tensor:
            source_name = "priv" if name == "privileged" else name
            count = stats[f"{source_name}_count"]
            mean = stats[f"{source_name}_sum"] / count
            scale = (
                stats[f"{source_name}_ssq"] / count - mean.square()
            ).clamp_min(1e-4).sqrt()
            return (value - mean) / scale

        command = normalize("command", observation.command)
        policy = normalize("policy", observation.policy)
        object_obs = normalize("object", observation.object)
        privileged = normalize("privileged", observation.privileged)
        feature = encoder(torch.cat((privileged, object_obs), dim=-1))
        actor_feature = actor_backbone(torch.cat((command, policy, feature), dim=-1))
        loc, _ = actor_head(actor_feature)
        return observation.reference_action + loc
    finally:
        sys.path.remove(str(hdmi_root))


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def main() -> None:
    args = parse_args()
    hdmi_root = args.hdmi_root.expanduser().resolve()
    task_spec = get_hdmi_task_spec(args.task)
    source_spec = SOURCE_SPECS[args.task]
    output = (
        args.output
        if args.output is not None
        else REPO_ROOT / f"artifacts/scaffolds/{task_spec.artifact_name}/v1"
    ).expanduser().resolve()
    checkpoint_path = hdmi_root / str(source_spec["checkpoint"])
    config_path = checkpoint_path.parent / ".hydra/config.yaml"
    asset_meta_path = checkpoint_path.parent / "asset_meta.json"
    motion_dir = hdmi_root / str(source_spec["motion_dir"])
    sources = {
        "checkpoint": checkpoint_path,
        "resolved_config": config_path,
        "asset_meta": asset_meta_path,
        "motion": motion_dir / "motion.npz",
        "motion_meta": motion_dir / "meta.json",
        "g1_asset": hdmi_root / str(source_spec["g1_asset"]),
        "object_asset": hdmi_root / str(source_spec["object_asset"]),
    }
    if "reference_video" in source_spec:
        sources["reference_video"] = hdmi_root / str(source_spec["reference_video"])
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing HDMI source files: {missing}")
    for name, expected in dict(source_spec["checksums"]).items():
        actual = sha256_file(sources[name])
        if actual != expected:
            raise ValueError(
                f"source {name} checksum does not match audited value: {actual}"
            )
    revision = git_output(hdmi_root, "rev-parse", "HEAD")
    if revision != EXPECTED_HDMI_REVISION:
        raise ValueError(f"HDMI revision mismatch: {revision}")

    if output.exists() and any(output.iterdir()) and not args.force:
        raise FileExistsError(f"output is not empty; pass --force to replace: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for child in output.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    (output / "reference").mkdir()
    (output / "assets").mkdir()
    (output / "parity").mkdir()

    # The source is a trusted local training artifact. This is the only pickle
    # load in the migration path; runtime uses weights_only=True below.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_portable_policy(checkpoint, task_spec)
    policy_path = output / "policy_state.pt"
    torch.save({"format_version": 1, "state_dict": model.state_dict()}, policy_path)
    # Assert the generated artifact is accepted by the restricted loader.
    torch.load(policy_path, map_location="cpu", weights_only=True)

    shutil.copy2(sources["motion"], output / "reference/motion.npz")
    shutil.copy2(sources["motion_meta"], output / "reference/meta.json")
    shutil.copy2(sources["g1_asset"], output / "assets/g1.usd")
    object_asset_path = output / f"assets/{task_spec.object_asset_name}.usd"
    shutil.copy2(sources["object_asset"], object_asset_path)

    reference_meta = json.loads(sources["motion_meta"].read_text(encoding="utf-8"))
    reference_joint_names = tuple(reference_meta["joint_names"])
    reference_to_action_indices = tuple(
        reference_joint_names.index(name) for name in task_spec.action_joint_names
    )

    state = model.state_dict()
    np.savez(
        output / "normalization.npz",
        **{
            f"{name}_{suffix}": state[f"{name}_{suffix}"].numpy()
            for name in ("command", "policy", "object", "privileged")
            for suffix in ("mean", "scale")
        },
    )

    generator = torch.Generator().manual_seed(20260721)
    values = {
        name: torch.randn(4, dim, generator=generator, dtype=torch.float32)
        for name, dim in task_spec.observation_dims.items()
    }
    observation = HDMIObservationBatch(
        command=values["command"],
        policy=values["policy"],
        object=values["object"],
        privileged=values["privileged"],
        reference_action=values["reference_action"],
    )
    with torch.inference_mode():
        oracle = hdmi_source_oracle_action(
            hdmi_root, checkpoint, observation, task_spec
        )
        portable = model(observation)
    error = float((oracle - portable).abs().max())
    if error > 1e-5:
        raise AssertionError(f"portable action parity failed: max_abs_error={error}")
    np.savez(
        output / "parity/source_outputs.npz",
        command=values["command"].numpy(),
        policy=values["policy"].numpy(),
        object=values["object"].numpy(),
        privileged=values["privileged"].numpy(),
        reference_action=values["reference_action"].numpy(),
        oracle_action=oracle.numpy(),
    )

    action_contract = {
        "contract_version": task_spec.contract_version,
        "output": {
            "name": "a_nom",
            "dtype": "float32",
            "shape": ["B", task_spec.network.action_dim],
            "coordinates": "normalized_joint_position",
        },
        "action_joint_names": list(task_spec.action_joint_names),
        "action_scale_rad": list(task_spec.action_scale),
        "reference_joint_names": list(reference_joint_names),
        "reference_joint_count": len(reference_joint_names),
        "reference_to_action_indices": list(reference_to_action_indices),
        "articulation_joint_count": 29,
        "reference_to_articulation": "map by joint name; absent articulation joints are initialized to zero",
        "excluded_wrist_joints": [
            name
            for name in reference_joint_names
            if "wrist" in name and name not in task_spec.action_joint_names
        ],
        "joint_target": "default_joint_position + a_applied * action_scale",
        "physics_dt_s": 0.005,
        "control_dt_s": 0.02,
        "decimation": 4,
        "delay_physics_substeps": {"training_range": [2, 6], "runtime_default": 4},
        "alpha": {"training_range": [0.8, 1.0], "runtime_default": 0.9, "operation": "applied.lerp(delayed, alpha) every physics substep"},
        "zero_hook": "environment input is bitwise equal to a_nom when no later module is present",
    }
    write_json(output / "action_contract.json", action_contract)

    observation_contract = {
        "contract_version": task_spec.contract_version,
        "role": "privileged_simulation_baseline_not_deployable",
        "concatenation_order": ["command", "policy", "object", "privileged"],
        "groups": {
            "command": {
                "shape": ["B", 356],
                "fields": [
                    {"name": "ref_body_pos_future_local", "shape": ["B", 5, 16, 3], "frame": "reference root yaw frame"},
                    {"name": "ref_joint_pos_future", "shape": ["B", 5, task_spec.network.action_dim], "joint_order": "action_joint_names"},
                    {"name": "ref_motion_phase", "shape": ["B", 1], "formula": "reference_step / motion_length"},
                ],
            },
            "policy": {
                "shape": ["B", 249],
                "fields": [
                    {"name": "root_ang_vel_history", "steps": [0], "shape": ["B", 3], "frame": "root"},
                    {"name": "projected_gravity_history", "steps": [0], "shape": ["B", 3], "frame": "root"},
                    {"name": "joint_pos_history", "steps": [0, 1, 2, 3, 4, 8], "shape": ["B", 174], "joint_order": "Isaac articulation 29-D order"},
                    {"name": "prev_actions", "steps": 3, "shape": ["B", task_spec.network.action_dim * 3], "layout": "action-major, newest-to-oldest"},
                ],
            },
            "object": {
                "shape": ["B", task_spec.observation_dims["object"]],
                "fields": [
                    "object_xy_in_root_yaw_frame",
                    "cos_sin_relative_heading",
                    *[
                        f"contact_target_{index}_in_root_yaw_frame"
                        for index in range(len(task_spec.contact_target_offsets))
                    ],
                ],
            },
            "privileged": {
                "shape": ["B", task_spec.observation_dims["privileged"]],
                "classification": "simulator_privileged",
                "field_order": [
                    "root_ang_vel_history_0_to_8", "projected_gravity_history_0_to_8",
                    "joint_pos_history_0_to_8", "ref_root_pos_future_b",
                    "ref_root_ori_future_b_first_two_rows", "diff_body_pos_future_local",
                    "diff_body_ori_future_local_first_two_rows", "diff_body_lin_vel_future_local",
                    "diff_body_ang_vel_future_local", "root_linvel_b", "ankle_pos_b",
                    "ankle_linvel_b", "ankle_pelvis_torso_height", "applied_action",
                    "applied_torque", "object_pos_b", "object_ori_b_full_matrix",
                    "diff_object_pos_future", "diff_object_ori_future_full_matrix",
                    "ref_object_contact_future", "diff_contact_pos_b",
                    *(
                        ["object_joint_pos", "object_joint_vel", "object_joint_torque"]
                        if task_spec.object_kind == "articulation"
                        else []
                    ),
                ],
            },
            "reference_action": {"shape": ["B", task_spec.network.action_dim], "formula": "(reference_joint_pos[action_mapping] - default_joint_pos) / action_scale"},
        },
        "normalization": {"groups": ["command", "policy", "object", "privileged"], "formula": "(x - frozen_mean) / frozen_scale", "variance_floor": 0.0001, "reference_action_normalized": False},
        "reset": {"state_histories": "fill all slots with reset state", "previous_actions": "zero", "reference_step": 0, "phase": 0.0},
        "deterministic_action": "teacher distribution mean; no sampling",
    }
    write_json(output / "observation_contract.json", observation_contract)

    third_party = """# Third-Party Artifact Notice

This local artifact was derived from the HDMI checkout and checkpoint identified
in `manifest.json`. HDMI is prior work and must be cited. At export time the HDMI
checkout had no root LICENSE, DATA_LICENSE, or NOTICE file. Redistribution of
the checkpoint, exported weights, reference motion/video, G1 USD, or object USD
is not authorized by this repository. Keep these files local/private until
permissions are documented. Runtime independence does not imply independent
authorship.
"""
    (output / "THIRD_PARTY.md").write_text(third_party, encoding="utf-8")

    materialized = {
        "policy": output / "policy_state.pt",
        "normalization": output / "normalization.npz",
        "reference_motion": output / "reference/motion.npz",
        "reference_meta": output / "reference/meta.json",
        "g1_asset": output / "assets/g1.usd",
        f"{task_spec.object_asset_name}_asset": object_asset_path,
        "parity": output / "parity/source_outputs.npz",
        "observation_contract": output / "observation_contract.json",
        "action_contract": output / "action_contract.json",
        "third_party": output / "THIRD_PARTY.md",
    }
    dirty = bool(git_output(hdmi_root, "status", "--porcelain"))
    manifest = {
        "artifact_version": "1.0.0-local",
        "task": task_spec.task,
        "contract_version": task_spec.contract_version,
        "observation_dims": dict(task_spec.observation_dims),
        "network_contract": task_spec.network.to_dict(),
        "task_contract": {
            "action_joint_names": task_spec.action_joint_names,
            "action_scale": task_spec.action_scale,
            "object_kind": task_spec.object_kind,
            "object_asset_name": task_spec.object_asset_name,
            "object_body_name": task_spec.object_body_name,
            "contact_target_offsets": task_spec.contact_target_offsets,
            "contact_eef_names": task_spec.contact_eef_names,
            "contact_eef_offsets": task_spec.contact_eef_offsets,
            "reference_joint_names": reference_joint_names,
        },
        "role": "frozen_hdmi_phase_train_privileged_teacher_simulation_baseline",
        "deployable": False,
        "exported_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "export_tool": str(Path(__file__).resolve()),
        "source": {
            "project": "HDMI",
            "revision": revision,
            "worktree_dirty": dirty,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "resolved_config_sha256": sha256_file(config_path),
            "asset_meta_sha256": sha256_file(asset_meta_path),
            "motion_sha256": sha256_file(sources["motion"]),
            "motion_meta_sha256": sha256_file(sources["motion_meta"]),
            "g1_asset_sha256": sha256_file(sources["g1_asset"]),
            f"{task_spec.object_asset_name}_asset_sha256": sha256_file(sources["object_asset"]),
            **(
                {"reference_video_sha256": sha256_file(sources["reference_video"])}
                if "reference_video" in sources
                else {}
            ),
        },
        "oracle_parity": {"backend": "HDMI active_adaptation make_mlp + Actor deterministic mean", "batch_size": 4, "seed": 20260721, "max_abs_action_error": error, "tolerance": 1e-5},
        "files": {
            name: {"path": str(path.relative_to(output)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in materialized.items()
        },
        "license": {"status": "undocumented_in_source_checkout", "redistribution": "prohibited_until_permissions_are_documented", "notice": "THIRD_PARTY.md"},
    }
    write_json(output / "manifest.json", manifest)

    checksums = [
        f"{sha256_file(path)}  {path.relative_to(output)}"
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    (output / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "parity_max_abs_error": error, "policy_sha256": manifest["files"]["policy"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
