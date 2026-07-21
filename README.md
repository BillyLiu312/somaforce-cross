# SomaForce-Cross

> Scaffolded humanoid force adaptation via cross semantic force distillation.

This repository is the executable engineering baseline for SomaForce-Cross. The first door scaffold is a frozen policy trained with the official HDMI implementation and migrated behind a standalone inference contract. SomaForce-Cross users should not need to install HDMI at runtime.

## Selected Pipeline

```text
frozen pretrained HDMI door policy
  -> standalone exported artifact
  -> versioned observation/action adapter
  -> nominal normalized action a_nom [B, 23]
  -> virtual wrist F/T observation model
  -> privileged task-conditioned force semantics
  -> p_dir outer p_mag
  -> P_cross -> CrossEncoder -> z_cross
  -> bounded residual actor
  -> a = a_nom + clip(Delta a_force)
  -> student distillation from real wrist F/T histories
```

Door-scaffold decisions:

- The pretrained scaffold is explicitly attributed prior work and remains frozen for the primary baseline.
- HDMI is permitted only in an export/oracle environment; the released runtime must not import `active_adaptation` or depend on an HDMI checkout.
- The audited learned policy action is 23-D. Canonical G1 references remain 29-joint data and require an explicit reference-to-action mapping.
- The current `phase=train` teacher is privileged; a production scaffold must use a separately validated non-privileged `actor_adapt`/finetune export.
- OMOMO supplies full-body human-object motion for lift/carry/place after joint G1/object retargeting.
- Reference data ingested by SomaForce-Cross enters the strict `CanonicalReferenceEpisode` contract; the learned door policy remains a separate artifact.
- OMOMO does not provide door motion, physical payload labels, or wrist F/T; those are assigned or generated in Isaac Lab.
- The door uses the frozen pretrained artifact first; payload scaffolds remain separate later artifacts.
- Legacy tracker compatibility code is not part of this runtime repository.

The complete migration and development rules are in [docs/pretrained_hdmi_scaffold_rules.md](docs/pretrained_hdmi_scaffold_rules.md). Goal-mode prompts are supplied separately for each scoped development round.

## Implemented Scaffold Slice

The current code provides:

- canonical 50 Hz G1/object/contact reference validation;
- HDMI `motion.npz` conversion;
- conversion of jointly retargeted OMOMO NPZ results;
- object-frame contact-target reconstruction;
- reference provenance, quality, confidence, and valid-frame masks;
- deterministic `ReferenceLibrary` persistence;
- standalone frozen HDMI `phase=train` privileged-teacher artifact under
  `artifacts/scaffolds/hdmi_push_door_hand/v1`;
- explicit 29-D canonical reference to audited 23-D action mapping, VecNorm,
  history/reset and JointPosition delay/alpha runtime;
- independent PyTorch inference module, HDMI-source deterministic parity
  fixture, import-isolation check, and an Isaac Lab G1 + articulated-door play
  entry.

The current code does not yet provide:

- raw SMPL-H OMOMO to G1 nonlinear retarget optimization;
- force teacher/student policy implementations;
- real F/T integration, checkpoints, or experiment logs.

The materialized HDMI artifact is deliberately marked
`privileged_simulation_baseline` and `deployable: false`. The checkpoint was
trained with the official HDMI implementation and remains prior work; a
non-privileged `actor_adapt`/finetune export still needs a separate validation
round before any deployment claim.

## Repository Layout

```text
docs/
  hdmi_omomo_scaffold_pipeline.md
  pretrained_hdmi_scaffold_rules.md
  pretrained_hdmi_scaffold_validation.md
somaforce_cross/scaffold/
  pretrained_hdmi.py
  pretrained_hdmi_isaac.py
  reference_schema.py
  hdmi_adapter.py
  omomo_adapter.py
  reference_library.py
  contracts.py
scripts/
  build_scaffold_reference.py
  export_pretrained_hdmi_scaffold.py       # HDMI export/oracle boundary only
  play_pretrained_hdmi_scaffold.py         # standalone Isaac Lab runtime
  verify_pretrained_hdmi_isolation.py
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

# materialize the local/private artifact (requires the trusted HDMI checkout)
python scripts/export_pretrained_hdmi_scaffold.py \
  --hdmi-root /inspire/hdd/global_user/liumengfan-253108110079/lmf-workspace/HDMI

# runtime checks without HDMI on PYTHONPATH
env -u PYTHONPATH python scripts/verify_pretrained_hdmi_isolation.py

# standalone privileged simulation baseline, one G1 + articulated door
python scripts/play_pretrained_hdmi_scaffold.py \
  --headless --num-envs 1 --steps 540 \
  --metrics-json artifacts/scaffolds/hdmi_push_door_hand/v1/rollout_metrics.json
```

The play command reports `a_nom`, the 23-D action order, reference phase, door
joint/progress, contact and support metrics, root-height stability, finite-state
status, and the zero-hook result. Omitting `--headless` enables Isaac Lab GUI
mode where a display is available. Isaac Sim 5.1 may block during teardown after
an interactive GUI session; headless batch mode exits after flushing metrics.

For GUI playback, render occurs once per 50 Hz control step rather than once per
physics substep. `--realtime --playback-rate 1.0` follows reference wall time;
increase the rate or omit `--realtime` for faster visualization.

## Artifact Versioning

The complete `artifacts/` tree is visible to Git. Large runtime files use Git
LFS according to `.gitattributes`, including `.pt`, `.usd`, `.npz`, `.npy`, and
`.onnx`. Training `checkpoints/`, `*.ckpt`, and `checkpoint_*.pt/.pth` remain
ignored. Before committing artifacts, verify redistribution permission in
`THIRD_PARTY.md`; LFS changes storage mechanics, not licensing rights.

## Boundary After This Round

This round stops at the frozen scaffold. It does not implement virtual F/T,
force semantics, `P_cross`, Cross residuals, student distillation, training, or
cluster jobs. The next independent task is validating/exporting a
non-privileged scaffold candidate; only after that gate should the Cross force
pipeline be developed.
