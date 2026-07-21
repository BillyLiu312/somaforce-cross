# SomaForce-Cross Implementation Plan

Date: 2026-07-01

> Route update (2026-07-21): the first door scaffold now uses a frozen policy
> trained with the official HDMI implementation and exported behind a standalone
> 23-D inference contract. Instructions below that prescribe training an
> HDMI-style door scaffold inside this repository are historical. Follow
> `docs/pretrained_hdmi_scaffold_rules.md` for current door work. Goal-mode
> prompts are supplied separately for each scoped development round. The
> canonical HDMI/OMOMO reference sections remain applicable to reference replay
> and future payload tasks.

This document stages the future engineering work for SomaForce-Cross. It intentionally avoids code-level implementation until the module contracts and task boundary are stable.

## 1. Method Route

The selected route is:

```text
frozen pretrained HDMI door policy
  -> standalone 23-D inference artifact
  -> versioned observation/action adapter
  -> nominal normalized action a_nom
  -> privileged force semantics in simulation
  -> virtual wrist F/T observation model
  -> probabilistic direction/magnitude semantics
  -> outer-product cross interaction
  -> z_cross
  -> bounded residual policy
  -> student distillation from deployable wrist F/T histories
```

## 2. Development Principles

- Start from pipeline contracts before executable code.
- Keep task generation separate from force residual learning.
- Treat the frozen pretrained HDMI artifact as the first door source of nominal task motion.
- Keep the HDMI export/oracle environment outside the released runtime.
- Keep 23-D learned actions distinct from canonical 29-joint references.
- Use canonical HDMI references for replay/diagnostics and jointly retargeted OMOMO references for future heavy payloads.
- Treat privileged simulation force as teacher supervision, not deployment input.
- Treat real wrist F/T sensing as the deployment assumption.
- Make `P_cross(t)` a first-class logged diagnostic.
- Do not add broad abstractions before the first diagnostic task is clear.

## 3. Phase 0: Pipeline Contract Freeze

Goal:

```text
define interfaces before implementation
```

Deliverables:

- scaffold interface;
- virtual F/T sensor interface;
- privileged force semantics interface;
- cross interaction interface;
- teacher/student interface;
- safety gate interface;
- logging schema for diagnosis, ablation, and paper-ready plots.

Exit criteria:

- every module has documented inputs and outputs;
- every logged variable maps to a paper metric or debugging plot;
- the first diagnostic task is selected.

## 4. Phase 1: Minimal Diagnostic Task Design

Goal:

```text
one task that can expose force-constraint mismatch
```

Candidate tasks:

- hinged door with hinge-axis or handle-pose mismatch;
- drawer with slider-direction mismatch;
- cabinet door with larger reach/body pressure.

Recommended first task:

```text
single hinged door diagnostic task
```

Required properties:

- stable contact phase;
- measurable object progress;
- synchronized wrist F/T observation;
- controllable mismatch;
- baseline comparison against scaffold-only and local compliance.

## 5. Phase 2: Standalone Scaffold and Sensor Prototype

Goal:

```text
standalone frozen pretrained scaffold + virtual wrist F/T observation
```

Deliverables:

- versioned deployable policy input contract and artifact manifest;
- explicit canonical-29 to controlled-23 joint mapping;
- parity-verified nominal normalized action `a_nom [B, 23]`;
- zero-residual and one-time action-scaling tests;
- virtual wrist F/T sensor model specification;
- sensor degradation schedule;
- first logging format.

Exit criteria:

- a clean environment without HDMI can run scaffold-only behavior;
- fixed-batch export parity and matched rollout evidence pass;
- the production scaffold has no undeclared privileged input;
- mismatch produces interpretable force events;
- the F/T observation is separated from privileged truth.

## 6. Phase 3: Cross Semantic Teacher Prototype

Goal:

```text
privileged force -> p_dir, p_mag, P_cross, z_cross
```

Selected cross representation:

```text
p_dir = softmax(E_dir(...))
p_mag = softmax(E_mag(...))
P_cross = p_dir outer p_mag
z_joint = flatten(P_cross)
z_cross = CrossEncoder(z_joint)
```

Deliverables:

- direction semantic label definition;
- magnitude semantic label definition;
- `P_cross(t)` logging plan;
- residual action boundary;
- teacher training plan.

Exit criteria:

- `P_cross` semantics are interpretable in at least one task phase;
- actor input is specified to receive only `z_cross`;
- `p_dir` and `p_mag` remain auxiliary/logging variables.

## 7. Phase 4: Student Distillation Plan

Goal:

```text
deployable histories -> p_hat_dir, p_hat_mag, P_hat_cross, z_hat_cross
```

Student inputs:

- wrist F/T history;
- joint torque / velocity history;
- end-effector motion history;
- proprioception history;
- `cmd_6d` history.

Staged distillation:

```text
Stage 1: KL direction/magnitude semantics
Stage 2: KL cross distribution + z_cross + action matching
Stage 3: joint fine-tuning with small semantic auxiliary losses
```

Exit criteria:

- student actor does not use privileged object truth;
- distillation loss terms respect the coupling among `p_dir`, `p_mag`, and `P_cross`;
- teacher-student performance gap is measurable.

## 8. Phase 5: Engineering Code Start

Only after Phases 0-4 are accepted should code directories be added.

Initial code modules should map one-to-one to the contracts:

```text
scaffold
virtual_ft_sensor
force_semantics
cross_interaction
teacher_policy
student_encoder
runtime_safety_gate
logging
metrics
```

This repository should remain small until the first diagnostic loop is executable.
