# SomaForce-Cross V1 Implementation Plan

Date: 2026-07-25
Status: staged implementation plan

This plan implements `docs/v1_force_residual_plan.md`. The implementation is
deliberately incremental. No phase may silently redesign the V1 method or skip
its acceptance gate.

## Phase 0: Contract Audit

Read-only audit of the four existing scaffold artifacts and runtimes.

Deliverables:

- task/action/reference/contact field table;
- 23-D action joint order for every artifact;
- observation history and reset table;
- F/T wrist body mapping;
- frame and unit conventions;
- zero-residual parity entry point;
- HDMI-specific fields excluded from the residual actor.

Acceptance:

- no unresolved action-order or scale ambiguity;
- no undeclared privileged residual input;
- current repository tests remain green;
- no source files modified in this phase.

## Phase 1: Pure PyTorch Force Core

Implement without Isaac or HDMI imports:

```text
somaforce_cross/force/wrist_token.py
somaforce_cross/force/contact_gate.py
somaforce_cross/force/semantic_heads.py
somaforce_cross/force/cross_encoder.py
```

Required behavior:

- validate `[B, 2, 16, 14]`;
- shared left/right temporal encoder and side embedding;
- `p_dir [B,13]` and `p_mag [B,5]`;
- `P_cross [B,13,5]` and `z_cross [B,64]`;
- soft-bin target generation with sigma `0.15`;
- finite behavior for zero and near-zero wrench;
- deterministic reset and history semantics.

Acceptance tests:

- shape/dtype/device tests;
- probability sums and finite checks;
- outer-product normalization;
- left/right side distinction;
- contact gate onset/offset;
- dropout and sensor-quality attenuation;
- no imports from HDMI, Isaac or `active_adaptation`.

## Phase 2: Action Composer and Safety

Implement:

```text
ResidualActionComposer
PerJointAuthority
JointMarginLimiter
ContactGatedResidual
```

Rules:

- normalized 23-D input/output;
- one-time joint-position scaling;
- C0-C3 authority values from the V1 design;
- remaining joint margin and velocity limit checks;
- `a_total` history update;
- no in-place mutation of `a_nom`.

Acceptance:

- zero residual reproduces `a_nom` exactly;
- nonzero residual is bounded;
- action targets remain finite;
- contact/sensor quality correctly scales authority;
- existing scaffold-only action tests are unchanged.

## Phase 3: Virtual Wrist F/T Sensor

Implement:

```text
clean simulated wrench
  -> base-yaw frame transform
  -> calibration
  -> bias/noise/drift
  -> delay/filter
  -> scale error/saturation/dropout
  -> deployable token
```

Implement an independent contact detector with hysteresis and temporal
smoothing. Generate semantic targets from clean normalized wrench and the V1
soft-bin rules.

Acceptance:

- deterministic test seed produces repeatable output;
- every corruption factor has an isolated test;
- delayed and filtered signals have expected response;
- sensor quality decreases on saturation/dropout;
- clean target generation never enters actor observation.

## Phase 4: Vectorized Residual Environment

Keep `PretrainedHDMIIsaacRuntime` as the golden deterministic rollout. Add a
new batched Isaac Lab `DirectRLEnv` for residual training.

Required environment groups:

```text
policy:
  wrist token history, proprioception, a_nom history, previous a_total, z_cross

critic:
  policy observation plus privileged state and mismatch parameters

semantic_target:
  clean-wrench p_dir/p_mag targets
```

Required behavior:

- per-environment reset;
- randomized physical and sensor conditions;
- task-balanced sampling;
- object progress/contact/stability rewards;
- termination and timeout reporting;
- nominal retention metrics.

Acceptance:

- 64-environment headless smoke;
- per-environment reset/history correctness;
- zero-residual parity against the golden runtime;
- no HDMI import at residual runtime;
- all four current tasks pass observation validation.

## Phase 5: Asymmetric PPO

Use RSL-RL PPO with these starting values:

```text
num_steps_per_env = 32
learning_rate = 3e-4
gamma = 0.99
lam = 0.95
ppo_epochs = 3
num_mini_batches = 8
clip_param = 0.2
entropy_coef = 0.001
max_grad_norm = 1.0
lambda_dir = 0.05
lambda_mag = 0.05
```

The actor must never receive critic-only object/mismatch fields. The critic may
use them. `P_cross` is deterministic and has no separate loss in V1.

Acceptance:

- short training remains finite;
- semantic losses decrease without collapsing to one class;
- actor gradient remains dominated by PPO task objective;
- nominal retention is at least 95% of scaffold baseline;
- no persistent action saturation or residual collapse.

## Phase 6: Four-Task Joint Training

Train the existing four artifacts with balanced environments and C0-C3
curricula. Keep a nominal fraction in every batch. Log task-specific and pooled
metrics separately.

Acceptance:

- every task completes the learning smoke;
- no task dominates pooled reward or semantic loss;
- at least two mismatch families show residual benefit;
- force, stability and `P_cross(t)` metrics are emitted.

## Phase 7: Three New Scaffolds

Add one task at a time:

```text
open_foldchair-sit
topple_wood_board_and_cross
roll_ball-hand
```

For each task:

1. Audit checkpoint/config/assets/reference.
2. Export a manifest-driven standalone artifact.
3. Verify fixed-batch source parity.
4. Verify import and artifact isolation.
5. Run scaffold-only nominal and mismatch rollouts.
6. Run zero-residual action parity.
7. Only then add the task to multi-task PPO.

Acceptance:

- the task requires no task-specific residual network;
- task registration is manifest-driven;
- scaffold-only behavior is stable enough to expose force mismatch;
- artifact provenance and permission fields are recorded.

## Phase 8: Held-Out Zero-Shot Evaluation

Freeze universal residual weights and evaluate held-out base scaffolds through
the common adapter. Do not fine-tune residual parameters or change bins,
authority or sensor contract.

Report in-family, compositional and stress zero-shot separately. The base
scaffold may be new, but the residual is not updated.

## Global Stop Conditions

Stop and return to the preceding phase when any of these occur:

- zero-residual parity fails;
- actor reads privileged object keys;
- action scaling is applied more than once;
- semantic target or p_mag distribution collapses;
- residual improves mismatch only by destroying nominal behavior;
- a new task needs a task-specific residual implementation;
- a sensor corruption factor is absent from the deployable simulation path.

