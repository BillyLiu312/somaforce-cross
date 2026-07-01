# SomaForce-Cross

> SomaForce-Cross is the implementation route for scaffolded humanoid force adaptation via cross semantic force distillation.

This repository is the standalone engineering baseline for developing SomaForce-Cross from the current selected pipeline. It includes the baseline architecture figure, module contracts, and implementation plan needed to start development from scratch.

Baseline pipeline:

![SomaForce-Cross Pipeline](figures/somaforce_cross_pipeline.png)

## Current Purpose

This repository should now contain:

- the baseline SomaForce-Cross pipeline;
- module boundary definitions;
- expected input/output contracts;
- implementation milestones;
- notes for future Isaac Lab and real F/T integration;
- enough context to begin engineering from a self-contained baseline.

It should not yet contain:

- Isaac Lab task code;
- RL training code;
- policy/model implementations;
- sensor simulation code;
- tests for code that has not been approved for implementation;
- checkpoints, logs, videos, or generated experiment artifacts.

## Selected Pipeline

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

Current key decisions:

- The scaffold should be Sonic-style or Sonic-based: it provides task base motion and `a_nom`.
- The deployed sensor assumption is a real wrist/end-effector F/T sensor.
- Simulation should model a virtual wrist F/T sensor, not expose perfect contact truth to the deployed actor.
- Direction and magnitude are represented as soft semantic distributions.
- Cross interaction is implemented conceptually as `P_cross = p_dir outer p_mag`, then flattening and encoding into `z_cross`.
- The actor should receive only `z_cross` as the force-semantic latent.
- `p_dir`, `p_mag`, and `P_cross` are retained for auxiliary supervision, student distillation, logging, and visualization.

## Repository Layout

```text
docs/
  implementation_plan.md
  module_contracts.md
figures/
  somaforce_cross_pipeline.png
configs/
  .gitkeep
```

Future code directories should be added only after the pipeline contracts are stable.

## Next Step

Before adding implementation code, complete:

1. module contracts for scaffold, virtual F/T sensor, force semantics, cross interaction, teacher, student, safety, and logging;
2. minimum diagnostic task definition;
3. first milestone acceptance criteria;
4. decision on the first actual environment backend.
