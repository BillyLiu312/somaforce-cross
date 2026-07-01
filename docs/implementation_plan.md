# Implementation Plan

Date: 2026-07-01

This plan follows the SomaForce-Cross architecture figure and the implementation handoff in the research repository.

## Milestone 1: Minimal Diagnostic Loop

Goal:

```text
one articulated object task
  -> Sonic-style nominal motion
  -> virtual F/T observation
  -> probabilistic cross interaction
  -> bounded residual action
  -> logs
```

Deliverables:

- one door or drawer diagnostic task wrapper;
- scaffold output `a_nom`;
- virtual wrist F/T observation `w_ft_obs`;
- cross interaction forward pass;
- residual action shape and bounds;
- logging schema.

## Milestone 2: Privileged Teacher

Goal:

```text
privileged simulation truth
  -> p_dir_T, p_mag_T
  -> P_cross_T
  -> z_cross_T
  -> residual teacher
```

Deliverables:

- privileged label generator;
- teacher policy module;
- BC warm-start script;
- PPO hook placeholder;
- ablation flags.

## Milestone 3: Student Distillation

Goal:

```text
deployable histories
  -> p_hat_dir, p_hat_mag
  -> P_hat_cross
  -> z_hat_cross
  -> residual action
```

Deliverables:

- student contact encoder;
- staged distillation losses;
- teacher-student evaluation script.

## Milestone 4: Safety and Evaluation

Goal:

```text
runtime safety gate
  -> force limits
  -> posture/stability margins
  -> fallback/stop
```

Deliverables:

- safety gate;
- force and progress metrics;
- baseline comparison scripts;
- summary export back to the research repository.
