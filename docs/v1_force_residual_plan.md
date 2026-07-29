# SomaForce-Cross V1 Force Residual Plan

Date: 2026-07-25
Status: authoritative frozen V1 design baseline

## 1. Goal

V1 is an object-agnostic, wrist-force-conditioned whole-body residual policy
layered on top of an object-aware frozen base scaffold.

```text
object-aware base scaffold -> a_nom [B, 23]
two-wrist force history + proprioception + a_nom history
    -> force semantics -> P_cross -> z_cross
    -> bounded contact-gated residual Delta a_force [B, 23]
    -> a_total
```

The base scaffold supplies nominal task/object motion. The residual adapts
after contact by unloading, increasing useful effort, maintaining contact and
mobilizing the whole body. It does not generate the task from scratch.

## 2. Deployment Boundary

The residual actor may consume:

- measured left/right wrist 6D F/T histories;
- wrist twist from proprioception and kinematics;
- contact probability and sensor quality;
- whole-body proprioception;
- `a_nom`, its short history and previous executed action.

The residual actor must not consume object pose, object velocity, object type,
mass, CoM, hinge/slider state, simulator contact truth, foot wrench, task ID,
object-category one-hot or HDMI `command[356]`.

The current HDMI `phase=train` artifacts remain privileged simulation baselines.
A future deployment gate requires a separately validated non-privileged base
scaffold.

## 3. Wrist Token Contract

Control rate is 50 Hz. The first temporal window is 16 frames (0.32 seconds).

```text
token shape = [B, 2 wrists, 16 history steps, 14 features]

wrench_base_yaw:      6  # Fx, Fy, Fz, Mx, My, Mz
wrist_twist_base:     6  # linear and angular velocity
contact_probability:  1
sensor_quality:       1
```

Wrist order is always left then right. History is oldest-to-newest and the
last temporal index is the current frame. At reset, a supplied initial frame is
broadcast across the history; otherwise the history is zero-filled. Both wrist
tokens exist for every task, including tasks such as door pushing where only
one wrist is expected to make primary contact.

The wrench frame is gravity-aligned robot-base-yaw: origin at the wrist sensor
origin, x/y aligned with pelvis yaw, and z aligned with world-up. Only a learned
left/right side embedding is retained; `contact_role` is not a V1 input.

The preprocessing chain is:

```text
raw wrench -> calibration/frame transform -> force/torque normalization
  -> bias/noise/drift/delay/filter/saturation/dropout model
  -> deployable token
```

Contact probability comes from an independent calibrated detector with
hysteresis and temporal smoothing. Sensor quality combines validity,
saturation, dropout and calibration health.

The pure PyTorch semantic core does not implement that detector. It receives
already computed contact probability and sensor quality and applies only the
stateless two-wrist aggregation rule. Detector hysteresis/smoothing belongs to
the virtual F/T sensor stage, while authority attack/release state belongs to
action composition.

## 4. Force Semantics

### 4.1 Direction

`p_dir` is a soft semantic distribution, not a normalized 6D vector.

```text
p_dir in R^13:
+Fx, -Fx, +Fy, -Fy, +Fz, -Fz,
+Mx, -Mx, +My, -My, +Mz, -Mz,
neutral/mixed
```

```text
p_dir = softmax(E_dir(wrist_history))
```

Each wrist is encoded by the same causal TCN. An 8-D side embedding is added to
each per-wrist feature, and the left/right features are concatenated in fixed
order before the global semantic heads. V1 therefore produces one task-agnostic
`p_dir [B,13]` and one `p_mag [B,5]` from the two-wrist history.

The continuous normalized force/torque vectors remain encoder inputs and a
continuous ablation, but are not the V1 semantic latent.

### 4.2 Magnitude

`p_mag` represents force-demand level:

```text
p_mag in R^5:
release/none, low, medium, high, critical/recover
```

```text
r_force  = ||F|| / F_scale
r_torque = ||M|| / M_scale
r_mag    = sqrt(0.5 * (r_force^2 + r_torque^2))
centers  = [0.0, 0.25, 0.5, 0.75, 1.0]
sigma_mag = 0.15
y_k = softmax(-(r_mag - centers[k])^2 / (2 * sigma_mag^2))
```

The initial direction target temperature is `tau_dir=0.15`; it may be calibrated
after target-distribution diagnostics without changing the V1 architecture.
The exact clean-wrench direction-target mapping and calibrated `F_scale` /
`M_scale` values are owned by the virtual F/T sensor stage; the pure PyTorch
core must not invent them. The magnitude soft-target helper may operate on any
leading dimensions and therefore can generate per-wrist targets before later
aggregation is finalized.

### 4.3 Cross

```text
P_cross = p_dir outer p_mag       # [13, 5]
z_joint  = flatten(P_cross)       # [65]
z_cross  = CrossEncoder(z_joint)  # [64] default latent
```

The residual actor receives `z_cross` only. `p_dir`, `p_mag` and `P_cross` are
kept for supervision, logging, visualization and ablations.

## 5. Actor, Gate and Action

Residual actor inputs are:

```text
z_cross
full-body proprioception
a_nom and short a_nom history
Delta a_nom if available
previous executed action a_total
```

```text
raw_delta     = ResidualActor(...)
delta_bounded = authority(t) * tanh(raw_delta)
delta_gated   = contact_gain * sensor_quality * delta_bounded
a_total       = clip(a_nom + delta_gated)
```

`a_nom` and residual share normalized 23-D coordinates. Joint-position scaling
is applied once after summation. `a_total`, not only `a_nom`, enters previous
action history.

Contact gate:

```text
g_left  = p_left * quality_left
g_right = p_right * quality_right
g_contact = 1 - (1 - g_left) * (1 - g_right)
```

Authority ramps up at contact onset and down at contact loss; it is not a hard
binary switch.

## 6. Learning Objective

V1 does not train a privileged teacher actor. It uses an asymmetric actor-critic:

```text
actor:  deployable noisy wrist F/T and proprioceptive inputs
critic: privileged simulator state, progress, contact, mismatch and stability
```

Semantic targets come from clean simulated wrench and calibrated target rules.

```text
L_total = L_PPO
        + 0.05 * KL(p_dir_pred, p_dir_target)
        + 0.05 * KL(p_mag_pred, p_mag_target)
```

`lambda_cross=0` because `P_cross` is deterministic from the two heads. A
clean-wrench privileged actor is a later upper-bound/bootstrap experiment only.

## 7. Network Defaults

```text
WristEncoder:        shared 3-block causal TCN, channels=128, kernel=3,
                     dilations=(1, 2, 4)
side embedding:      8
semantic heads:      hidden=128
CrossEncoder:        65 -> 128 -> 64
proprio encoder:     256 -> 128
a_nom/action encoder:128 -> 64
ResidualActor:        256 -> 256 -> 23
PrivilegedCritic:     512 -> 256 -> 256 -> 1
```

These are engineering defaults and may change only after smoke evidence.

## 8. Curriculum

Residual authority by group (arms/waist/legs):

```text
C0: 0.00 / 0.00 / 0.00
C1: 0.10 / 0.05 / 0.04
C2: 0.18 / 0.10 / 0.08
C3: 0.25 / 0.15 / 0.12
```

Final authority is additionally limited by remaining joint margin,
`velocity_limit * dt`, torque/current safety and posture/stability safety.

Environment mixture:

```text
C1: 70% nominal, 30% mild mismatch
C2: 40% nominal, 60% mild/moderate mismatch
C3: 30% nominal, 70% full training mismatch
```

Door mismatch includes friction, damping, initial object/stance and contact-target
offsets. Episode-time hinge-axis and physical-handle geometry are deferred by
[`phase4b5_runtime_capability_amendment.md`](phase4b5_runtime_capability_amendment.md).
Push-box mismatch includes mass, friction, CoM and initial pose. Suitcase and
largebox mismatch include mass, friction, CoM, contact stability and
lift/carry tracking. Sensor curriculum is low noise and 0-1 step delay at C1,
bias/noise/filtering and 0-2 steps at C2, then drift/scale error/axis
misalignment/0-3 steps/saturation/dropout at C3.

Stress conditions are held-out evaluation conditions until the base and residual
policies pass the normal curriculum.

## 9. Task Scope and Evidence

V1 residual training tasks:

```text
push_door-hand, push_box, move_suitcase, move_largebox,
open_foldchair-sit, topple_wood_board_and_cross, roll_ball-hand
```

Implementation starts with the four existing artifacts. Each new task must
pass audit, export, parity, asset validation, scaffold-only rollout and
zero-residual checks before joining multi-task PPO.

Held-out zero-shot candidates are `carry_and_place_bread_box`,
`carry_box_over_shoulder`, `move_foam`, `move_stool-climb` and `truman`.
Zero-shot may load a new base scaffold and adapter, but may not update residual
weights, semantic bins or authority.

Required evidence includes task progress/success, nominal retention, p95/p99
force, impulse, force-rate, contact fraction/loss, residual norms by group,
action saturation, stability, semantic entropy/KL and `P_cross(t)`.
