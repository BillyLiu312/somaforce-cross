# SomaForce-Cross

Engineering implementation repository for **SomaForce-Cross**.

> SomaForce-Cross learns a deployable force-conditioned residual over a Sonic-style task scaffold. Privileged force is factorized into direction and magnitude semantic distributions, combined through an outer-product joint distribution, flattened and encoded as `z_cross`, then distilled to a student that uses real wrist F/T histories.

This repository is the engineering body of the SomaForce project. The research brain lives in:

```text
../somaforce
```

Primary research references:

- `../somaforce/docs/08_discussion.md`
- `../somaforce/docs/09_implementation_handoff.md`
- `../somaforce/docs/figures/06_somaforce_rld_s_architecture_overview_ft_cross.png`
- `../somaforce/method/logging_and_metrics_spec.md`
- `../somaforce/method/experiment_protocol.md`

## Scope

This repository owns:

- Isaac Lab task wrappers and environment adapters;
- Sonic-style base motion scaffold;
- virtual wrist F/T sensor model;
- privileged force semantic label generation;
- probabilistic outer-product cross interaction;
- teacher residual policy training;
- student contact encoder distillation;
- runtime safety gate;
- logging, metrics, and evaluation scripts.

The research repository owns paper design, figures, result summaries, benchmark planning, and related work.

## Initial Architecture

```text
Sonic-style scaffold
  -> nominal action a_nom
  -> virtual wrist F/T observation w_ft_obs
  -> privileged direction/magnitude semantics
  -> p_dir outer p_mag
  -> flatten(P_cross)
  -> CrossEncoder
  -> z_cross
  -> bounded residual policy
  -> a = a_nom + clip(Delta a_force)
```

The actor should receive `z_cross` as the only force-semantic latent. `p_dir`, `p_mag`, and `P_cross` are retained for auxiliary supervision, distillation, logging, and visualization.

## Repository Layout

```text
configs/
  env/
  robot/
  task/
  train/
  distill/
somaforce_cross/
  sensors/
  scaffold/
  models/
  teacher/
  student/
  safety/
  logging/
  metrics/
scripts/
tests/
```

## Development Start

First milestones:

1. Implement `VirtualFTSensor`.
2. Implement `SonicScaffold` with a small trajectory library.
3. Implement `ProbabilisticCrossInteraction`.
4. Implement a minimal residual policy forward pass.
5. Add logging schema compatible with the SomaForce research repo.

## Large Files

Do not commit raw logs, videos, checkpoints, or Isaac cache files. Use `runs/`, `checkpoints/`, and `artifacts/` locally; they are gitignored.
