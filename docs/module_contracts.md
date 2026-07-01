# SomaForce-Cross Module Contracts

Date: 2026-07-01

This file records the expected module contracts before implementation code is added.

## 1. Sonic-Style Scaffold

Purpose:

```text
provide task base motion and nominal action a_nom
```

Inputs:

- `task_id`;
- `cmd_6d`;
- robot state;
- object/task prior;
- safety envelope.

Outputs:

- nominal action `a_nom`;
- nominal hand reference;
- nominal body reference;
- scaffold confidence or validity flag.

Non-goal:

- The scaffold should not solve force adaptation. It provides the base motion that SomaForce-Cross corrects.

## 2. Virtual Wrist F/T Sensor Model

Purpose:

```text
simulate the deployable wrist/end-effector F/T sensor used by the student
```

Inputs:

- clean simulated wrist or joint wrench;
- sensor frame transform;
- compensation settings;
- degradation settings.

Outputs:

- deployable F/T observation `w_ft_obs`;
- force norm and force-rate diagnostics;
- sensor health flag.

Required modeling factors:

- frame transform;
- zero bias;
- hand/tool gravity compensation if needed;
- low-pass filtering;
- delay;
- noise;
- drift;
- saturation.

Non-goal:

- Do not expose object-side contact truth as actor input.

## 3. Privileged Force Semantics

Purpose:

```text
produce teacher-side force semantic targets
```

Inputs:

- privileged contact truth;
- object progress;
- task prior;
- contact phase;
- virtual F/T observation.

Outputs:

- direction semantic distribution `p_dir`;
- magnitude semantic distribution `p_mag`;
- optional direction/magnitude labels;
- diagnostic confidence.

## 4. Cross Interaction

Purpose:

```text
convert direction and magnitude semantics into the force-semantic policy latent
```

Selected representation:

```text
P_cross = p_dir outer p_mag
z_joint = flatten(P_cross)
z_cross = CrossEncoder(z_joint)
```

Outputs:

- `P_cross`;
- `z_joint`;
- `z_cross`.

Actor rule:

```text
actor receives z_cross only
```

Logging rule:

```text
log p_dir, p_mag, P_cross, z_cross
```

## 5. Residual Teacher

Purpose:

```text
learn force-conditioned residuals over the scaffold
```

Inputs:

- proprioception;
- `cmd_6d`;
- `a_nom`;
- previous action;
- `z_cross`;
- optional critic-only privileged state.

Outputs:

- bounded residual `Delta a_force`;
- teacher action `a_teacher = a_nom + clip(Delta a_force)`;
- value estimate for RL training.

## 6. Student Contact Encoder

Purpose:

```text
replace privileged force semantics with deployable histories
```

Inputs:

- wrist F/T history;
- joint torque / velocity history;
- end-effector motion history;
- proprioception history;
- `cmd_6d` history.

Outputs:

- `p_hat_dir`;
- `p_hat_mag`;
- `P_hat_cross`;
- `z_hat_cross`;
- confidence estimate.

## 7. Runtime Safety Gate

Purpose:

```text
bound or stop residual deployment
```

Inputs:

- F/T observation;
- force rate;
- posture/stability margins;
- contact state;
- confidence.

Outputs:

- residual gain scale;
- safe probing flag;
- fallback/stop signal.

## 8. Logging Contract

Minimum logged variables:

- `a_nom`;
- `Delta a_force`;
- `w_ft_obs`;
- privileged force labels if available;
- `p_dir`, `p_mag`, `P_cross`, `z_cross`;
- student predictions;
- contact phase;
- object progress;
- force and constrained-force metrics;
- safety events.
