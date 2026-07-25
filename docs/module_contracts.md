# SomaForce-Cross Module Contracts

Updated: 2026-07-25
Status: V1 compatibility summary

Detailed values and acceptance gates live in
`docs/v1_force_residual_plan.md` and
`docs/v1_force_residual_implementation_plan.md`. This document only fixes the
boundaries shared across modules.

## 1. Frozen Task Scaffold

Input and internal observation contracts are artifact-specific. The public
output is:

```text
a_nom: float32 [B,23], normalized joint-position coordinates
```

The current HDMI `phase=train` artifacts are privileged simulation baselines,
not deployable policies. They remain frozen and HDMI-free at SomaForce-Cross
runtime. A deployable end-to-end system still requires a separately validated
sensor-realizable base scaffold.

## 2. Wrist Observation

```text
wrist_tokens: float32 [B,2,16,14]

per wrist:
  wrench_base_yaw       6
  wrist_twist_base      6
  contact_probability  1
  sensor_quality        1
```

The wrist order is left then right. History is oldest-to-newest and the last
index is current. Both wrists are present for every task even when only one is
expected to contact. The actor receives the corrupted deployable signal, never
simulator contact truth or clean target wrench.

## 3. Force Semantic Core

The shared causal wrist encoder processes each wrist independently, adds an
8-D side embedding, then concatenates left and right features in fixed order.
The global semantic outputs are:

```text
p_dir:   float32 [B,13]
p_mag:   float32 [B,5]
P_cross: float32 [B,13,5]
z_cross: float32 [B,64]
```

```text
P_cross = p_dir.unsqueeze(-1) * p_mag.unsqueeze(-2)
```

`P_cross` is deterministic and has no independent V1 loss. Clean simulated
wrench may generate direction and magnitude auxiliary targets but is not an
actor input.

## 4. Contact Gate

The stateless semantic-core gate consumes deployable values:

```text
g_left  = p_left  * quality_left
g_right = p_right * quality_right
g_contact = 1 - (1 - g_left) * (1 - g_right)
```

Contact detection, hysteresis, smoothing, and sensor corruption belong to the
virtual F/T sensing path. Authority attack/release behavior and per-joint
limits belong to residual action composition.

## 5. Residual Actor and Action Composition

Allowed actor inputs are wrist-token history, `z_cross`, deployable whole-body
proprioception, `a_nom` and its short history, optional `Delta a_nom`, and
previous executed `a_total`.

Forbidden actor inputs include object pose/type/velocity, mass, CoM,
hinge/slider state, simulator contact truth, foot wrench, task ID, HDMI
`command[356]`, and other critic-only state.

```text
delta_bounded = authority * tanh(raw_delta)
delta_gated   = contact_gain * sensor_quality * delta_bounded
a_total       = clip(a_nom + delta_gated)
```

`a_nom` and the residual share normalized 23-D coordinates. Joint-position
scaling is applied once after composition. Previous-action history records
executed `a_total`.

## 6. Asymmetric Learning

V1 has no privileged residual teacher actor in the primary route.

```text
actor: noisy deployable wrist F/T and declared proprioceptive/action inputs
critic: actor observations plus simulator progress, object, mismatch,
        contact, and stability state
targets: clean-wrench p_dir and p_mag distributions
```

The auxiliary weights start at `lambda_dir=0.05` and `lambda_mag=0.05`.
A clean-wrench privileged actor is only a later upper-bound/bootstrap
experiment and must not replace the primary V1 result.

## 7. Runtime Safety and Logging

Per-joint authority is further clipped by joint margin, velocity, torque/current,
posture, and stability limits. Minimum logs include `a_nom`, raw/bounded/gated
residual, `a_total`, wrist observations and clean targets, contact/quality,
`p_dir`, `p_mag`, `P_cross`, `z_cross`, force statistics, task progress,
stability, saturation, reset, termination, and curriculum state.
