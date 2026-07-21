# SomaForce-Cross Module Contracts

Date: 2026-07-01

This file records the expected module contracts before implementation code is added.

## 1. Pretrained HDMI Door Scaffold

Purpose:

```text
standalone frozen learned policy -> nominal normalized action a_nom
```

Inputs:

- versioned deployable observation tensors and declared task/reference priors;
- frozen observation-normalization state;
- frozen exported policy artifact;
- explicit mapping from the canonical 29-joint reference to the 23 controlled
  action joints.

Outputs:

- nominal normalized action `a_nom [B, 23]`;
- exact action joint order and action-space metadata;
- scaffold confidence or health flag where available.

Rules:

- Runtime code must not import HDMI/`active_adaptation` or require an HDMI
  checkout.
- The primary scaffold is frozen and runs in evaluation mode.
- The current `phase=train` HDMI teacher is privileged and is not a production
  scaffold. Export and validate a non-privileged `actor_adapt`/finetune path.
- `a_nom` is not a raw 29-DOF joint target. Compose the Cross residual in the
  same 23-D normalized coordinates and apply action scaling once.
- Follow `docs/pretrained_hdmi_scaffold_rules.md` for the complete contract,
  provenance, licensing, and parity gates.

Non-goal:

- The scaffold does not solve force adaptation. It provides the base
  robot-object motion that SomaForce-Cross corrects.
- Wrist control is not silently added to the current 23-D policy.
- The scaffold artifact and HDMI training method are not claimed as a
  SomaForce-Cross contribution.

## 1A. Canonical HDMI + OMOMO Reference Boundary

Purpose:

```text
canonicalize HDMI/OMOMO robot-object references for replay and future tasks
```

Inputs:

- HDMI door `motion.npz` or jointly retargeted OMOMO payload result;
- source provenance, task identity, contact mapping, and retarget metadata.

Outputs:

- canonical 29-joint G1/object/contact reference;
- nominal hand reference;
- nominal body reference;
- nominal object reference;
- contact intent, object-frame target, and confidence;
- source provenance and per-frame validity mask.

Non-goal:

- Canonical reference conversion is not the frozen learned policy scaffold.
- OMOMO must already be jointly retargeted to G1 before ingestion.
- Physical force, payload mass, inertia, CoM, and load share must not be inferred from OMOMO kinematics.

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
