# SomaForce-Cross

Pipeline staging repository for **SomaForce-Cross**.

> SomaForce-Cross is the implementation route for scaffolded humanoid force adaptation via cross semantic force distillation.

This repository intentionally starts as a **pipeline and engineering handoff repository**, not a code implementation repository. It should first stabilize the development plan, module boundaries, data contracts, and experiment sequence before executable Isaac Lab / RL code is added.

The research repository remains:

```text
../somaforce
```

Primary references:

- `../somaforce/docs/08_discussion.md`
- `../somaforce/docs/09_implementation_handoff.md`
- `../somaforce/docs/figures/06_somaforce_rld_s_architecture_overview_ft_cross.png`
- `../somaforce/method/logging_and_metrics_spec.md`
- `../somaforce/method/experiment_protocol.md`

## Current Purpose

This repository should now contain:

- the selected SomaForce-Cross pipeline;
- module boundary definitions;
- expected input/output contracts;
- implementation milestones;
- notes for future Isaac Lab and real F/T integration;
- links back to the research repository.

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
  .gitkeep
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
