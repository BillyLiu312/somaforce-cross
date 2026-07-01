# SomaForce-Cross Implementation Plan

Date: 2026-07-01

This document stages the future engineering work for SomaForce-Cross. It intentionally avoids code-level implementation until the module contracts and task boundary are stable.

## 1. Method Route

SomaForce-Cross follows the architecture in the research repository:

```text
../somaforce/docs/figures/06_somaforce_rld_s_architecture_overview_ft_cross.png
```

A copy is included in this repository:

```text
figures/somaforce_cross_pipeline.png
```

The selected route is:

```text
Sonic-style scaffold
  -> nominal task motion a_nom
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
- Treat Sonic-style scaffold as the source of nominal task motion.
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
- logging schema alignment with the research repository.

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

## 5. Phase 2: Scaffold and Sensor Prototype

Goal:

```text
Sonic-style base trajectory + virtual wrist F/T observation
```

Deliverables:

- task base motion trajectory definition;
- nominal action `a_nom`;
- virtual wrist F/T sensor model specification;
- sensor degradation schedule;
- first logging format.

Exit criteria:

- scaffold-only behavior can run in nominal settings;
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
