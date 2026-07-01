# AGENTS.md

## Warmup Prompt for SomaForce-Cross

You are entering the **SomaForce-Cross** repository.

This repository is the standalone engineering baseline for the current selected SomaForce-Cross pipeline. It should be treated as a self-contained starting point for future engineering work, not as a place to rediscover or redesign the method from scratch.

SomaForce-Cross studies **force-aware whole-body humanoid manipulation of articulated objects**. The first target domain is constrained interaction with human-scale mechanisms such as doors, cabinet doors, and drawers. The central research idea is:

> Contact force should not be treated only as a safety threshold. During articulated-object manipulation, force is an embodied observation of environmental constraint mismatch.

The method should help a humanoid answer, during contact:

- Which direction appears task-consistent?
- Which direction appears constrained or force-wasting?
- How much force resistance is present?
- Is hand-level compliance enough, or should the torso/body posture be mobilized?
- Can a deployable wrist F/T history recover the same force semantics learned from privileged simulation signals?

## Current Method Identity

Use the method name:

```text
SomaForce-Cross
```

Interpretation:

> Scaffolded humanoid force adaptation via cross semantic force distillation.

The name emphasizes the actual current innovation: privileged force is factorized into direction and magnitude semantic distributions, combined through an outer-product cross representation, and distilled into a deployable student that uses real wrist F/T histories.

Do not revert the method name to `SomaForce-RLD-S` as the main identity. That older label described the training route, but it did not capture the core innovation as clearly.

## Selected Pipeline

The current baseline pipeline is:

```text
Sonic-style task scaffold
  -> nominal action a_nom
  -> virtual wrist F/T observation model
  -> privileged direction/magnitude force semantics
  -> p_dir outer p_mag
  -> P_cross
  -> flatten(P_cross)
  -> CrossEncoder
  -> z_cross
  -> bounded residual actor
  -> a = a_nom + clip(Delta a_force)
  -> student distillation from real wrist F/T histories
```

The included baseline figure is:

```text
figures/somaforce_cross_pipeline.png
```

## Non-Negotiable Pipeline Decisions

The following decisions are already selected and should not be contradicted unless the project owner explicitly changes the method:

- The scaffold is **Sonic-style / Sonic-based**. It provides the task base motion and nominal action `a_nom`.
- SomaForce-Cross learns a **bounded force-conditioned residual**, not the whole task behavior from scratch.
- The deployed hardware assumption is a **real wrist/end-effector F/T sensor**.
- Simulation should include a **virtual wrist F/T sensor model** that mimics deployable sensing.
- The actor must not receive perfect object-side contact truth, true hinge/slider state, or simulator-only privileged labels at deployment.
- Privileged force information is used for teacher supervision, critic inputs, auxiliary labels, and diagnostics.
- Direction and magnitude are represented as **soft semantic distributions**, not as raw uninterpreted force vectors only.
- The selected cross interaction is:

```text
p_dir = softmax(E_dir(...))
p_mag = softmax(E_mag(...))
P_cross = p_dir outer p_mag
z_joint = flatten(P_cross)
z_cross = CrossEncoder(z_joint)
```

- The residual actor receives **only `z_cross`** as the force-semantic latent.
- `p_dir`, `p_mag`, and `P_cross` are retained for auxiliary supervision, student distillation, logging, and visualization.
- Student distillation should respect the coupling among `p_dir`, `p_mag`, and `P_cross`; do not treat them as three unrelated equal-weight latent targets.

## Why Sonic-Style Scaffold

The scaffold should provide different base motion trajectories for different tasks. It is responsible for nominal task tracking, such as:

- approximate hand trajectory for door opening;
- approximate slider motion for drawer pulling;
- approximate body reference for reach maintenance;
- safe nominal posture and task timing.

The force residual is responsible for adapting this base motion under contact:

```text
a = a_nom + clip(Delta a_force)
```

This separation is important. SomaForce-Cross is not trying to prove that it can generate all humanoid task motion. It is trying to prove that, given a reasonable base task motion, force semantics can improve constrained interaction and whole-body adaptation.

The ideal scaffold behavior is:

- low mismatch: scaffold can complete or nearly complete the task;
- medium/high mismatch: scaffold keeps the task meaningful but produces constrained force, inefficiency, or failure;
- SomaForce-Cross residual: reduces constrained force and improves progress, safety, or body adaptation.

## Force Sensing Assumption

The real deployment assumption is a wrist/end-effector F/T sensor.

In simulation, do not model the student input as perfect contact truth. Model a virtual F/T sensor signal:

```text
clean simulated wrist/joint wrench
  -> sensor frame transform
  -> compensation if needed
  -> bias / noise / drift / delay / filtering / saturation
  -> deployable F/T observation w_ft_obs
```

Keep three concepts separate:

```text
privileged contact truth:
  simulation-only teacher labels, critic inputs, metrics

virtual wrist F/T observation:
  simulation version of what the real sensor would provide

real wrist F/T history:
  deployment-side student input
```

This distinction is central to the sim-to-real story.

## Cross Semantic Representation

The cross representation should be understood as a **joint semantic distribution**, not a geometric vector cross product.

If:

```text
p_dir: direction semantic distribution
p_mag: magnitude / mobilization semantic distribution
```

then:

```text
P_cross(i, j) = p_dir(i) * p_mag(j)
```

Examples of joint semantic cells:

```text
pull x hand-level
pull x torso-level
slide x low-force
reorient x high-force
recover x fallback
```

Flattening `P_cross` gives a one-dimensional semantic vector:

```text
z_joint = flatten(P_cross)
```

Then `CrossEncoder` converts it into:

```text
z_cross
```

`z_cross` is the only force-semantic latent that enters the residual actor.

Cross attention is conceptually related but is not the selected first baseline. With only one vector or distribution per branch, standard cross attention can collapse into a trivial interaction. Cross attention may become relevant later only if direction and magnitude are represented as token sets.

## Student Distillation Semantics

The student replaces privileged force semantics with deployable histories:

```text
wrist F/T history
joint torque / velocity history
end-effector motion history
proprioception history
cmd_6d history
  -> p_hat_dir
  -> p_hat_mag
  -> P_hat_cross
  -> z_hat_cross
```

The intended staged supervision is:

```text
Stage 1:
  align p_hat_dir with p_dir
  align p_hat_mag with p_mag

Stage 2:
  align P_hat_cross with P_cross
  align z_hat_cross with z_cross
  match teacher residual/action behavior

Stage 3:
  joint fine-tuning with smaller direction/magnitude auxiliary losses
```

This preserves the factorized semantics without pretending that `p_dir`, `p_mag`, and `P_cross` are independent targets.

## Research Scope

The first research target is articulated-object manipulation:

- hinged doors;
- cabinet or large doors;
- drawers or sliders.

The main mismatch sources are:

- hinge-axis error;
- slider-direction error;
- handle-pose error;
- friction or damping variation;
- initial stance and reach stress;
- force noise, delay, scale mismatch, or saturation.

The minimum evidence chain should show:

- scaffold-only behavior under nominal and mismatch settings;
- force events under mismatch;
- interpretable `p_dir`, `p_mag`, and `P_cross(t)` traces;
- reduced constrained or wasted force from the residual;
- improvement over local compliance or raw force baselines;
- student recovery of teacher force semantics from deployable F/T histories.

## What This Method Is Not

Do not frame SomaForce-Cross as:

- a general-purpose force foundation model;
- a from-scratch humanoid manipulation policy;
- a new whole-body controller formulation;
- a new impedance/admittance theory;
- a method that requires privileged simulator contact truth at deployment;
- a raw-force black-box policy with no semantic factorization;
- a method where `z_dir` and `z_mag` bypass `P_cross` and directly drive the actor as parallel force latents.

The current contribution is narrower and cleaner:

> A Sonic-style task scaffold provides nominal humanoid task motion; privileged force teaches direction/magnitude semantics; their outer-product cross representation becomes `z_cross`; a bounded residual actor uses `z_cross`; a student distills the same semantics from deployable wrist F/T histories.

## Baselines and Ablations to Keep in Mind

Future experiments should be interpretable against these comparison families:

- scaffold only;
- scaffold + local admittance;
- raw force latent residual;
- direction-only residual;
- magnitude-only residual;
- concat direction/magnitude residual;
- outer-product cross residual;
- hand-only residual;
- full hand/body residual;
- privileged teacher;
- deployable student.

The most important diagnostic signal is:

```text
P_cross(t)
```

because it exposes whether the method is using force as a structured semantic signal rather than as an opaque scalar or raw wrench input.
