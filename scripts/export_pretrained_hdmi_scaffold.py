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
    CONTRACT_VERSION,
    HDMI_ACTION_JOINT_NAMES,
    HDMI_ACTION_SCALE,
    HDMI_REFERENCE_JOINT_NAMES,
    OBSERVATION_DIMS,
    REFERENCE_TO_ACTION_INDICES,
    FrozenHDMITeacherPolicy,
    HDMIObservationBatch,
    sha256_file,
)


EXPECTED_CHECKPOINT_SHA256 = "24682b7b66c38f784ca5939dad3841b1b1e5b801c47d2fea59c5f85fd90c238c"
EXPECTED_CONFIG_SHA256 = "fb6ddd53135550b244fd35b707cf2b98e069ddf38dfe456af5b930e4189a074c"
EXPECTED_HDMI_REVISION = "32282f6dcf26cae70b814d585ceb12cc38aa1b60"


def parse_args() -> argparse.Namespace:
    workspace = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdmi-root", type=Path, default=workspace / "HDMI")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "artifacts/scaffolds/hdmi_push_door_hand/v1",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def copy_tensor(target: torch.Tensor, source: torch.Tensor) -> None:
    if target.shape != source.shape:
        raise ValueError(f"tensor shape mismatch: {target.shape} != {source.shape}")
    target.copy_(source)


def build_portable_policy(checkpoint: dict[str, object]) -> FrozenHDMITeacherPolicy:
    policy_state = checkpoint["policy"]
    source_encoder = policy_state["encoder_priv"]
    source_actor = policy_state["actor"]
    model = FrozenHDMITeacherPolicy()
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
    for prefix, width in (("actor_fc1", 512), ("actor_fc2", 256), ("actor_fc3", 256)):
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
) -> torch.Tensor:
    """Run the fixed batch through HDMI's own make_mlp and Actor classes."""

    sys.path.insert(0, str(hdmi_root))
    try:
        from active_adaptation.learning.ppo.common import Actor, make_mlp

        source_policy = checkpoint["policy"]
        encoder = torch.nn.Sequential(make_mlp([256]), torch.nn.LazyLinear(256))
        encoder(torch.zeros(1, 1721))
        encoder.load_state_dict(
            {
                key.removeprefix("module.1.module."): value
                for key, value in source_policy["encoder_priv"].items()
            },
            strict=True,
        )
        actor_backbone = make_mlp([512, 256, 256])
        actor_backbone(torch.zeros(1, 861))
        actor_backbone.load_state_dict(
            {
                key.removeprefix("module.0.module.1.module."): value
                for key, value in source_policy["actor"].items()
                if key.startswith("module.0.module.1.module.")
            },
            strict=True,
        )
        actor_head = Actor(23)
        actor_head(torch.zeros(1, 256))
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
    output = args.output.expanduser().resolve()
    checkpoint_path = hdmi_root / "outputs/doorpushhand/hdmi-doorpushhand-4gpu/rank_0/checkpoint_final.pt"
    config_path = checkpoint_path.parent / ".hydra/config.yaml"
    asset_meta_path = checkpoint_path.parent / "asset_meta.json"
    motion_dir = hdmi_root / "data/motion/data_for_sim/push_door-hand-0828"
    sources = {
        "checkpoint": checkpoint_path,
        "resolved_config": config_path,
        "asset_meta": asset_meta_path,
        "motion": motion_dir / "motion.npz",
        "motion_meta": motion_dir / "meta.json",
        "g1_asset": hdmi_root / "active_adaptation/assets/g1/g1_29dof_rubberhand-feet_sphere-eef_box-body_capsule.usd",
        "door_asset": hdmi_root / "active_adaptation/assets/objects/door/door.usd",
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing HDMI source files: {missing}")
    if sha256_file(checkpoint_path) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("source checkpoint checksum does not match audited checkpoint")
    if sha256_file(config_path) != EXPECTED_CONFIG_SHA256:
        raise ValueError("resolved config checksum does not match audited config")
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
    model = build_portable_policy(checkpoint)
    policy_path = output / "policy_state.pt"
    torch.save({"format_version": 1, "state_dict": model.state_dict()}, policy_path)
    # Assert the generated artifact is accepted by the restricted loader.
    torch.load(policy_path, map_location="cpu", weights_only=True)

    shutil.copy2(sources["motion"], output / "reference/motion.npz")
    shutil.copy2(sources["motion_meta"], output / "reference/meta.json")
    shutil.copy2(sources["g1_asset"], output / "assets/g1.usd")
    shutil.copy2(sources["door_asset"], output / "assets/door.usd")

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
        for name, dim in OBSERVATION_DIMS.items()
    }
    observation = HDMIObservationBatch(
        command=values["command"],
        policy=values["policy"],
        object=values["object"],
        privileged=values["privileged"],
        reference_action=values["reference_action"],
    )
    with torch.inference_mode():
        oracle = hdmi_source_oracle_action(hdmi_root, checkpoint, observation)
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
        "contract_version": CONTRACT_VERSION,
        "output": {"name": "a_nom", "dtype": "float32", "shape": ["B", 23], "coordinates": "normalized_joint_position"},
        "action_joint_names": list(HDMI_ACTION_JOINT_NAMES),
        "action_scale_rad": list(HDMI_ACTION_SCALE),
        "canonical_reference_joint_names": list(HDMI_REFERENCE_JOINT_NAMES),
        "reference_to_action_indices": list(REFERENCE_TO_ACTION_INDICES),
        "excluded_wrist_joints": [name for name in HDMI_REFERENCE_JOINT_NAMES if "wrist" in name],
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
        "contract_version": CONTRACT_VERSION,
        "role": "privileged_simulation_baseline_not_deployable",
        "concatenation_order": ["command", "policy", "object", "privileged"],
        "groups": {
            "command": {
                "shape": ["B", 356],
                "fields": [
                    {"name": "ref_body_pos_future_local", "shape": ["B", 5, 16, 3], "frame": "reference root yaw frame"},
                    {"name": "ref_joint_pos_future", "shape": ["B", 5, 23], "joint_order": "action_joint_names"},
                    {"name": "ref_motion_phase", "shape": ["B", 1], "formula": "reference_step / motion_length"},
                ],
            },
            "policy": {
                "shape": ["B", 249],
                "fields": [
                    {"name": "root_ang_vel_history", "steps": [0], "shape": ["B", 3], "frame": "root"},
                    {"name": "projected_gravity_history", "steps": [0], "shape": ["B", 3], "frame": "root"},
                    {"name": "joint_pos_history", "steps": [0, 1, 2, 3, 4, 8], "shape": ["B", 174], "joint_order": "Isaac articulation 29-D order"},
                    {"name": "prev_actions", "steps": 3, "shape": ["B", 69], "layout": "action-major, newest-to-oldest"},
                ],
            },
            "object": {"shape": ["B", 7], "fields": ["object_xy_in_root_yaw_frame", "cos_sin_relative_heading", "contact_target_in_root_yaw_frame"]},
            "privileged": {"shape": ["B", 1714], "classification": "simulator_privileged", "field_order": ["root_ang_vel_history_0_to_8", "projected_gravity_history_0_to_8", "joint_pos_history_0_to_8", "ref_root_pos_future_b", "ref_root_ori_future_b_first_two_rows", "diff_body_pos_future_local", "diff_body_ori_future_local_first_two_rows", "diff_body_lin_vel_future_local", "diff_body_ang_vel_future_local", "root_linvel_b", "ankle_pos_b", "ankle_linvel_b", "ankle_pelvis_torso_height", "applied_action", "applied_torque", "object_pos_b", "object_ori_b_full_matrix", "diff_object_pos_future", "diff_object_ori_future_full_matrix", "ref_object_contact_future", "diff_contact_pos_b", "object_joint_pos", "object_joint_vel", "object_joint_torque"]},
            "reference_action": {"shape": ["B", 23], "formula": "(reference_joint_pos[action_mapping] - default_joint_pos) / action_scale"},
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
the checkpoint, exported weights, reference motion, G1 USD, or door USD is not
authorized by this repository. Keep these files local/private until permissions
are documented. Runtime independence does not imply independent authorship.
"""
    (output / "THIRD_PARTY.md").write_text(third_party, encoding="utf-8")

    materialized = {
        "policy": output / "policy_state.pt",
        "normalization": output / "normalization.npz",
        "reference_motion": output / "reference/motion.npz",
        "reference_meta": output / "reference/meta.json",
        "g1_asset": output / "assets/g1.usd",
        "door_asset": output / "assets/door.usd",
        "parity": output / "parity/source_outputs.npz",
        "observation_contract": output / "observation_contract.json",
        "action_contract": output / "action_contract.json",
        "third_party": output / "THIRD_PARTY.md",
    }
    dirty = bool(git_output(hdmi_root, "status", "--porcelain"))
    manifest = {
        "artifact_version": "1.0.0-local",
        "contract_version": CONTRACT_VERSION,
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
            "door_asset_sha256": sha256_file(sources["door_asset"]),
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
