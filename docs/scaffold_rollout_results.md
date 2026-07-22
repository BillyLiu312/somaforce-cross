# Scaffold Rollout Results

Date: 2026-07-23

## Scope

This report summarizes the current deterministic single-environment rollouts of
the four frozen HDMI-derived scaffold tasks. Each row compares cases using the
same task-specific indicators. All videos were verified as H.264, 1280x720,
50 FPS. These policies are privileged simulation baselines and are not
deployable policies.

The report treats `nominal` as the normal operating condition. Other cases
introduce physical mismatch without changing the frozen scaffold policy. Video
links are repository-relative and are tracked through Git LFS.

## Result Overview

| Task | Main nominal behavior | Mismatch behavior |
| --- | --- | --- |
| `push_door_hand` | Opens the door beyond the 1.5 rad target | Friction reduces progress; damping triples peak wrist force |
| `push_box` | Tracks the reference displacement with low final error | Light/slippery overshoots and loses contact; heavy/friction under-progresses with higher force |
| `move_suitcase` | Completes lift, carry, and set-down | Heavy misses set-down; stress loses contact and falls early |
| `move_largebox` | Completes lift, carry, and set-down | Increasing mass progressively reduces lift and increases carry/set-down error |

## Push Door Hand

Indicators: door progress and root height are in meters/radians as labeled;
`wrist max` is the maximum measured wrist contact force. The reference task
target is 1.5 rad.

| Case | Friction | Damping | Steps | Door progress (rad) | Progress vs nominal | Wrist max (N) | Wrist max vs nominal | Min root height (m) | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `nominal` | 0.3 | 0.55 | 540 | 2.618 | 100.0% | 19.43 | 100.0% | 0.713 | [JSON](../artifacts/scaffolds/hdmi_push_door_hand/v1/rollout_metrics_nominal.json) / [MP4](../videos/push_door_hand/nominal.mp4) |
| `high_friction` | 10.0 | 0.55 | 540 | 1.103 | 42.1% | 33.88 | 174.4% | 0.685 | [JSON](../artifacts/scaffolds/hdmi_push_door_hand/v1/rollout_metrics_high_friction.json) / [MP4](../videos/push_door_hand/high_friction.mp4) |
| `high_damping` | 0.3 | 10.0 | 540 | 1.825 | 69.7% | 61.31 | 315.6% | 0.727 | [JSON](../artifacts/scaffolds/hdmi_push_door_hand/v1/rollout_metrics_high_damping.json) / [MP4](../videos/push_door_hand/high_damping.mp4) |

### Analysis

- `high_friction` is a progress-limited mismatch. The door reaches only 73.6%
  of the absolute 1.5 rad target while wrist force rises by 74.4%.
- `high_damping` still crosses the target, but progress falls by 30.3% and peak
  wrist force grows to 3.16 times nominal. It is therefore a force-efficiency
  failure rather than a binary task failure.
- The videos show the same nominal motion prior being executed under different
  mechanism resistance. The endpoint door angle separates `high_friction`,
  while `high_damping` is better distinguished by force and progress metrics.

## Push Box

Indicators: `progress` is maximum displacement along the reference direction;
`final error` is physical box-to-reference position error; `two-hand contact`
is the fraction of expected contact steps where both wrists exceed 1 N.

| Case | Mass (kg) | Friction | Steps | Progress (m) | Progress vs nominal | Final error (m) | Two-hand contact | Wrist max (N) | Min root height (m) | Stable | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `nominal` | 8.0 | 0.5 | 792 | 1.467 | 100.0% | 0.062 | 100.0% | 55.10 | 0.718 | yes | [JSON](../artifacts/scaffolds/hdmi_push_box/v1/rollout_metrics_nominal.json) / [MP4](../videos/push_box/nominal.mp4) |
| `light_slippery` | 1.0 | 0.1 | 792 | 2.110 | 143.8% | 1.066 | 69.0% | 46.81 | 0.745 | yes | [JSON](../artifacts/scaffolds/hdmi_push_box/v1/rollout_metrics_light_slippery.json) / [MP4](../videos/push_box/light_slippery.mp4) |
| `heavy_friction` | 16.0 | 0.6 | 792 | 1.006 | 68.5% | 0.440 | 97.1% | 84.74 | 0.688 | yes | [JSON](../artifacts/scaffolds/hdmi_push_box/v1/rollout_metrics_heavy_friction.json) / [MP4](../videos/push_box/heavy_friction.mp4) |

### Analysis

- `light_slippery` is the low-resistance failure mode. The box travels 44%
  farther than nominal, develops 1.066 m final error, and loses two-hand contact
  for 31% of the expected contact interval. The video shows visible overshoot,
  lateral slip, and separation from the nominal hand-box relationship.
- `heavy_friction` is the opposite high-resistance failure mode. It preserves
  contact and remains stable, but progress falls by 31.5%, final error rises to
  0.440 m, and peak wrist force rises by 53.8%. The 0.6 friction coefficient
  avoids the conspicuous box tipping seen in the retired friction-1.2 case.
- Together, the two cases expose complementary adaptation requirements:
  reduce/extract force under low resistance, and mobilize force without losing
  tracking under high resistance.

## Move Suitcase

Indicators are identical across all suitcase cases. `Carry pos/orient error`
are means over the carry window; `set-down error` is final object-position
error. The stress rollout terminates early, so its 279 steps and 47.7% contact
fraction are part of the failure result.

| Case | Mass (kg) | Steps | Max lift (m) | XY displacement (m) | Carry pos error (m) | Carry orient error (rad) | Set-down error (m) | Set-down | Two-hand contact | Wrist max (N) | Min root height (m) | Stable | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| `nominal` | 1.5 | 472 | 0.648 | 1.832 | 0.201 | 0.186 | 0.206 | yes | 99.4% | 72.88 | 0.281 | yes | [JSON](../artifacts/scaffolds/hdmi_move_suitcase/v1/rollout_metrics_nominal.json) / [MP4](../videos/move_suitcase/nominal.mp4) |
| `light` | 0.5 | 472 | 0.715 | 1.834 | 0.147 | 0.207 | 0.188 | yes | 99.4% | 77.97 | 0.281 | yes | [JSON](../artifacts/scaffolds/hdmi_move_suitcase/v1/rollout_metrics_light.json) / [MP4](../videos/move_suitcase/light.mp4) |
| `heavy` | 3.0 | 472 | 0.530 | 1.712 | 0.286 | 0.194 | 0.312 | no | 100.0% | 69.50 | 0.281 | yes | [JSON](../artifacts/scaffolds/hdmi_move_suitcase/v1/rollout_metrics_heavy.json) / [MP4](../videos/move_suitcase/heavy.mp4) |
| `stress` | 5.5 | 279 | 0.135 | 1.110 | 0.617 | 0.996 | 0.852 | no | 47.7% | 121.62 | 0.190 | no | [JSON](../artifacts/scaffolds/hdmi_move_suitcase/v1/rollout_metrics_stress.json) / [MP4](../videos/move_suitcase/stress.mp4) |

### Analysis

- `light` is not a task-failure case: lift increases by 10.3%, carry position
  error falls by 26.6%, and set-down remains successful. It represents an
  out-of-nominal but easier load condition.
- `heavy` is a recoverable degradation. The robot remains upright and maintains
  contact, but lift falls by 18.2%, carry position error rises by 42.6%, and the
  final 0.312 m error fails set-down.
- `stress` is a catastrophic baseline failure. Lift falls to 20.8% of nominal,
  orientation error increases more than fivefold, contact drops below 50%, and
  the rollout stops at step 279 because root height falls below 0.25 m. The
  video clearly shows the carry sequence collapse and object/robot instability.

## Move Large Box

The large-box table uses the same indicator set as the suitcase table. All
cases complete 199 steps and remain stable, so the important differences are
continuous lift/tracking errors rather than termination.

| Case | Mass (kg) | Steps | Max lift (m) | XY displacement (m) | Carry pos error (m) | Carry orient error (rad) | Set-down error (m) | Set-down | Two-hand contact | Wrist max (N) | Min root height (m) | Stable | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| `nominal` | 1.0 | 199 | 0.276 | 1.511 | 0.102 | 0.225 | 0.182 | yes | 100.0% | 92.93 | 0.542 | yes | [JSON](../artifacts/scaffolds/hdmi_move_largebox/v1/rollout_metrics_nominal.json) / [MP4](../videos/move_largebox/nominal.mp4) |
| `light` | 0.8 | 199 | 0.280 | 1.543 | 0.088 | 0.177 | 0.151 | yes | 100.0% | 175.75 | 0.546 | yes | [JSON](../artifacts/scaffolds/hdmi_move_largebox/v1/rollout_metrics_light.json) / [MP4](../videos/move_largebox/light.mp4) |
| `heavy` | 1.2 | 199 | 0.176 | 1.502 | 0.126 | 0.180 | 0.194 | yes | 100.0% | 81.72 | 0.554 | yes | [JSON](../artifacts/scaffolds/hdmi_move_largebox/v1/rollout_metrics_heavy.json) / [MP4](../videos/move_largebox/heavy.mp4) |
| `stress` | 2.0 | 199 | 0.129 | 1.449 | 0.196 | 0.348 | 0.232 | yes | 100.0% | 78.72 | 0.538 | yes | [JSON](../artifacts/scaffolds/hdmi_move_largebox/v1/rollout_metrics_stress.json) / [MP4](../videos/move_largebox/stress.mp4) |

### Analysis

- `light` tracks at least as well as nominal, but its 175.75 N right-wrist peak
  is 1.89 times nominal. This may be a brief impact rather than sustained force;
  percentiles or force impulse are needed before treating it as a force-control
  failure.
- `heavy` reduces maximum lift by 36.2% while leaving horizontal displacement
  almost unchanged. The policy transports the object but does not reproduce the
  intended vertical trajectory.
- `stress` reduces lift by 53.2%, nearly doubles carry position error, and raises
  orientation error by 54.7%. It still passes the current 0.25 m set-down
  tolerance with 0.232 m error, so its degradation is more visible in continuous
  metrics and carry height than in the binary success flag.
- The videos show similar high-level phases across all four cases; mass mismatch
  primarily changes object height, posture, and tracking quality rather than
  causing an immediate fall.

## Cross-Task Interpretation

The cases provide three useful kinds of scaffold mismatch:

1. **Resistance mismatch:** door friction/damping and push-box heavy/friction
   increase force demand and reduce progress.
2. **Low-resistance mismatch:** push-box light/slippery creates overshoot,
   lateral slip, and contact loss rather than insufficient progress.
3. **Payload mismatch:** suitcase and large-box mass changes degrade lift,
   tracking, placement, contact, or stability with task-dependent severity.

This is a suitable baseline set for evaluating a future SomaForce policy: the
nominal scaffold establishes retained task competence, while non-nominal cases
expose distinct force-conditioned failure modes. Improvements should be judged
with both task outcomes and force efficiency; peak contact force alone is not
sufficient because it can be dominated by brief impacts.

## Limitations

- Results are one deterministic rollout per case, not a multi-seed statistical
  evaluation.
- The rollout metrics currently contain peak and mean wrist forces but not
  p95/p99 force, impulse, saturation time, or phase-conditioned force cost.
- Binary set-down thresholds can hide meaningful continuous degradation, as in
  `move_largebox/stress`.
- The `move_suitcase/stress` video is 279 frames (5.58 s) because the baseline
  terminates early; this is expected and not an encoding failure.
