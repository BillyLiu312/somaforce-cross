# SomaForce-Cross

> Scaffolded humanoid force adaptation via cross semantic force distillation.

This repository is the executable engineering baseline for SomaForce-Cross. The selected Phase 0 scaffold uses HDMI-style robot-object co-tracking with HDMI door references and jointly retargeted OMOMO heavy-payload references.

## Selected Pipeline

```text
HDMI door references + retargeted OMOMO payload references
  -> canonical G1/object/contact reference library
  -> HDMI-style robot-object co-tracking scaffold
  -> nominal action a_nom
  -> virtual wrist F/T observation model
  -> privileged task-conditioned force semantics
  -> p_dir outer p_mag
  -> P_cross -> CrossEncoder -> z_cross
  -> bounded residual actor
  -> a = a_nom + clip(Delta a_force)
  -> student distillation from real wrist F/T histories
```

Phase 0 decisions:

- HDMI supplies door task references, object-aware tracking structure, contact targets, and the training pattern.
- OMOMO supplies full-body human-object motion for lift/carry/place after joint G1/object retargeting.
- Both sources enter one strict `CanonicalReferenceEpisode` contract.
- OMOMO does not provide door motion, physical payload labels, or wrist F/T; those are assigned or generated in Isaac Lab.
- Door and payload specialists are trained separately first.
- SONIC remains a legacy tracker baseline and compatibility path, not a default dependency.

## Implemented Scaffold Slice

The current code provides:

- canonical 50 Hz G1/object/contact reference validation;
- HDMI `motion.npz` conversion;
- conversion of jointly retargeted OMOMO NPZ results;
- object-frame contact-target reconstruction;
- reference provenance, quality, confidence, and valid-frame masks;
- deterministic `ReferenceLibrary` persistence;
- `HDMIReferenceScaffold` outputting `a_nom` and nominal task references;
- default door and heavy-payload scaffold configs;
- legacy Sonic adapters and smoke tools for regression/baseline use.

The current code does not yet provide:

- raw SMPL-H OMOMO to G1 nonlinear retarget optimization;
- HDMI PPO/co-tracking training code vendored into this repository;
- force teacher/student policy implementations;
- real F/T integration, checkpoints, or experiment logs.

## Repository Layout

```text
configs/scaffold/
  g1_push_pull_door.yaml
  g1_heavy_payload.yaml
  g1_push_pull_box.yaml          # legacy Sonic baseline
docs/
  hdmi_omomo_scaffold_pipeline.md
somaforce_cross/scaffold/
  reference_schema.py
  hdmi_adapter.py
  omomo_adapter.py
  reference_library.py
  hdmi_scaffold.py
  contracts.py
  scenes.py
  sonic_*.py                     # legacy baseline
scripts/
  build_scaffold_reference.py
tests/
  test_hdmi_omomo_pipeline.py
  test_scaffold_contracts.py
```

## Reference Conversion

HDMI door reference:

```text
python scripts/build_scaffold_reference.py \
  --source hdmi \
  --input /path/to/motion.npz \
  --output /path/to/door_references.pt \
  --episode-id door-push-001 \
  --source-clip-id push_door-hand \
  --task push_pull_door \
  --left-hand-body-index <index> \
  --right-hand-body-index <index> \
  --object-body-index -1
```

OMOMO joint-retarget result:

```text
python scripts/build_scaffold_reference.py \
  --source omomo-retargeted \
  --input /path/to/retargeted_clip.npz \
  --output /path/to/payload_references.pt \
  --episode-id payload-001 \
  --source-clip-id omomo-source-clip \
  --retarget-version g1-retarget-v1 \
  --retarget-quality foot_slip_m=0.01 \
  --retarget-quality grasp_error_m=0.02
```

The OMOMO command intentionally rejects raw/incomplete data. The input must already contain the jointly retargeted G1 body, object trajectory, wrist poses, and contact intent.

## Verification

```text
python -m pytest -q
python -m compileall -q scripts somaforce_cross
```

The Sonic-specific scripts remain available for compatibility regression. See [docs/hdmi_omomo_scaffold_pipeline.md](docs/hdmi_omomo_scaffold_pipeline.md) for contracts, acceptance gates, and the legacy boundary.

## Next Step

Run the converter on the released HDMI `push_door-hand` reference and one simple jointly retargeted OMOMO carry clip. After deterministic replay, connect these canonical episodes to HDMI-style Isaac Lab reference-state initialization and robot-object co-tracking training.
