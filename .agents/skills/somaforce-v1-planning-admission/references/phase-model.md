# V1 Phase Responsibility Model

## Contents

1. Purpose
2. Phases 0-3
3. Phase 4
4. Phases 5-8
5. Cross-phase scientific boundaries

## Purpose

This file describes responsibilities from the authoritative implementation plan. It does not say which phase is current, complete, admitted, or next. Discover repository-specific subphases and progress from live source, tests, Git, contracts, and runtime evidence.

## Phases 0-3

- **Phase 0 contract audit:** establish frozen scaffold task/action/reference/contact contracts, normalized action order, reset/history behavior, wrist mapping, and actor privilege exclusions. Audit only; do not mutate artifacts.
- **Phase 1 force core:** implement pure PyTorch two-wrist history, causal encoding, direction/magnitude semantics, cross encoding, and contact gating without Isaac or HDMI runtime imports.
- **Phase 2 action safety:** implement normalized residual composition, per-joint authority, contact ramping, joint/velocity margins, optional caller bounds, and executed-action history. Zero residual must preserve nominal action exactly.
- **Phase 3 virtual wrist F/T:** establish a defensible six-axis Isaac source, frame/sign/unit behavior, calibration and corruption, independent contact detection, semantic targets, and selected reset. Simulation evidence is not hardware calibration.

## Phase 4

Phase 4 owns the vectorized residual environment while preserving the frozen scaffold as the golden rollout reference. Its plan-level responsibilities include:

- canonical policy, critic, and semantic-target observations;
- actor privilege isolation;
- per-environment parameter storage and reset;
- delayed Isaac imports and bounded environment lifecycle;
- task adapters for the active scaffold set;
- zero-residual C0 parity;
- physical, scaffold, and sensor mismatch contracts;
- selected apply/direct readback/unselected isolation;
- fixed normalization, reward, termination, episode logging, and curriculum wiring.

Repositories may split these responsibilities into multiple 4A/4B/B5 subphases. Discover those labels and their Admission state live; do not infer them from this skill.

Useful conceptual layers, independent of local subphase names:

1. tensor and privilege contracts;
2. atomic selected reset and parameter store;
3. DirectRLEnv and task adapters;
4. C0 golden parity;
5. numeric contracts and runtime capabilities;
6. selected per-episode apply/readback lifecycle;
7. normalizer/reward/done/logging/curriculum integration.

Each layer requires its own source, tests, and runtime evidence. A later standalone module does not close an earlier environment-wiring gap.

## Phases 5-8

- **Phase 5 asymmetric PPO:** actor/critic networks and PPO with semantic auxiliary losses and strict privilege isolation.
- **Phase 6 joint training:** balanced active tasks and curriculum with nominal retention and per-task/pooled metrics.
- **Phase 7 new scaffolds:** audit, export, validate, and add each scaffold independently before joint training.
- **Phase 8 held-out zero-shot:** freeze residual weights and evaluate new scaffolds without fine-tuning.

## Cross-Phase Scientific Boundaries

- Frozen teacher artifacts are privileged simulation scaffolds, not the deployable residual actor.
- Nominal and residual actions share the declared normalized action coordinates; physical scaling occurs exactly once.
- Residual actor inputs exclude task/object identity, object state, simulator contact truth, mismatch rows, clean wrench, foot wrench, and teacher-only command fields unless the governing plan changes explicitly.
- Reference/action mapping must follow names and declared contracts rather than assumed equal dimensions.
- New runtime fields require canonical input, selected apply, applied readback, and unselected isolation; otherwise defer or amend.
- Unit tests do not prove real runtime capability.
- C0 parity does not prove learning, robustness, zero-shot transfer, deployment, or hardware performance.
- Phase 4 evidence does not authorize Phase 5 training.
