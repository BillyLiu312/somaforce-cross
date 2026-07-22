#!/usr/bin/env python3
"""Verify scaffold inference while blocking HDMI runtime imports."""

from __future__ import annotations

import importlib.abc
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class BlockHDMIImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object, target: object = None) -> object:
        if fullname == "active_adaptation" or fullname.startswith("active_adaptation."):
            raise ImportError(f"blocked runtime dependency: {fullname}")
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=("push_door_hand", "push_box", "move_suitcase"),
        default="push_door_hand",
    )
    parser.add_argument("--artifact", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = (
        args.artifact
        if args.artifact is not None
        else REPO_ROOT / f"artifacts/scaffolds/hdmi_{args.task}/v1"
    ).expanduser().resolve()
    sys.path[:] = [entry for entry in sys.path if not entry.rstrip("/").endswith("/HDMI")]
    blocker = BlockHDMIImports()
    sys.meta_path.insert(0, blocker)
    try:
        from somaforce_cross.scaffold.pretrained_hdmi import (
            HDMIObservationBatch,
            PretrainedHDMIScaffold,
        )

        scaffold = PretrainedHDMIScaffold.from_artifact(artifact)
        if scaffold.task_spec.task != args.task:
            raise AssertionError(
                f"artifact task {scaffold.task_spec.task!r} != requested {args.task!r}"
            )
        with np.load(artifact / "parity/source_outputs.npz", allow_pickle=False) as data:
            observation = HDMIObservationBatch(
                command=torch.from_numpy(data["command"][:1]),
                policy=torch.from_numpy(data["policy"][:1]),
                object=torch.from_numpy(data["object"][:1]),
                privileged=torch.from_numpy(data["privileged"][:1]),
                reference_action=torch.from_numpy(data["reference_action"][:1]),
            )
            oracle = torch.from_numpy(data["oracle_action"][:1])
        action = scaffold(observation)
        imported = sorted(
            name for name in sys.modules if name == "active_adaptation" or name.startswith("active_adaptation.")
        )
        if imported:
            raise AssertionError(f"HDMI modules imported at runtime: {imported}")
        error = float((action - oracle).abs().max())
        if error > 1e-5:
            raise AssertionError(f"isolated parity failed: {error}")
        print(
            json.dumps(
                {
                    "artifact": str(artifact),
                    "task": scaffold.task_spec.task,
                    "active_adaptation_modules": imported,
                    "action_shape": list(action.shape),
                    "max_abs_action_error": error,
                    "frozen": all(not parameter.requires_grad for parameter in scaffold.parameters()),
                    "eval_mode": not scaffold.training and not scaffold.policy.training,
                },
                indent=2,
            )
        )
    finally:
        sys.meta_path.remove(blocker)


if __name__ == "__main__":
    main()
