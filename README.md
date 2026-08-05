# SomaForce-Cross

> Scaffolded humanoid force adaptation via cross-semantic force conditioning.

This repository is the executable engineering baseline for SomaForce-Cross. The
first door scaffold is a frozen policy trained with the official HDMI
implementation and migrated behind a standalone inference contract. Runtime
rollout imports only SomaForce-Cross code plus PyTorch/Isaac Lab and reads local
artifacts under `artifacts/`; it neither imports `active_adaptation` nor accesses
an HDMI checkout. The exported weights, reference motion, and USD assets still
retain HDMI provenance and redistribution restrictions.

## Method Overview

![SomaForce-Cross pipeline overview](figures/somaforce_cross_pipeline.png)

The figure is the retained conceptual overview of the complete research
pipeline. Its older `SomaForce-RLD-S` and `Sonic-style` labels should be read as
`SomaForce-Cross` and the nominal interaction scaffold, respectively. The V1
implementation uses standalone frozen HDMI artifacts and a deployable-input
force-semantic residual actor. It does not train a privileged force teacher or
use a teacher-to-student distillation route.

## Selected Pipeline

```text
frozen pretrained HDMI task policy
  -> standalone exported artifact
  -> versioned observation/action adapter
  -> nominal normalized action a_nom [B, 23]
  -> virtual wrist F/T observation model
  -> deployable-input direction/magnitude force semantics
  -> p_dir outer p_mag
  -> P_cross -> CrossEncoder -> z_cross
  -> contact-gated bounded residual actor
  -> a_total in normalized 23-D action coordinates

clean simulated wrench -> auxiliary semantic targets
privileged simulator state -> critic only
```

Door-scaffold decisions:

- The pretrained scaffold is explicitly attributed prior work and remains frozen for the primary baseline.
- HDMI is permitted only in an export/oracle environment; the released runtime must not import `active_adaptation` or depend on an HDMI checkout.
- The audited learned policy action is 23-D. Canonical G1 references remain 29-joint data and require an explicit reference-to-action mapping.
- The current `phase=train` teacher is privileged; a production scaffold must use a separately validated non-privileged `actor_adapt`/finetune export.
- OMOMO supplies full-body human-object motion for lift/carry/place after joint G1/object retargeting.
- Reference data ingested by SomaForce-Cross enters the strict `CanonicalReferenceEpisode` contract; the learned door policy remains a separate artifact.
- OMOMO does not provide door motion, physical payload labels, or wrist F/T; those are assigned or generated in Isaac Lab.
- Door and payload tasks use four independently selectable frozen artifacts; task differences remain in manifests and adapters rather than task-specific residual networks.
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
- a second independently selectable push-box artifact under
  `artifacts/scaffolds/hdmi_push_box/v1`, using the no-hand/eef-L G1 asset and
  a rigid box;
- an independently selectable move-suitcase artifact under
  `artifacts/scaffolds/hdmi_move_suitcase/v1`, using the shared rubber-hand G1
  asset, a rigid suitcase, and a 472-frame OMOMO-derived HDMI reference;
- an independently selectable move-largebox artifact under
  `artifacts/scaffolds/hdmi_move_largebox/v1`, using the shared rubber-hand G1,
  a local URDF/OBJ rigid object, and a 199-frame OMOMO-derived HDMI reference;
- manifest-driven observation and network dimensions for all tasks, including
  the push-box `command[356]`, `policy[249]`, `object[10]`,
  `privileged[1714]`, encoder input `1724`, and action `[23]` contract;
- explicit 29-D canonical reference to audited 23-D action mapping, VecNorm,
  history/reset and JointPosition delay/alpha runtime;
- independent PyTorch inference module, per-task HDMI-source deterministic
  parity fixtures, import-isolation checks, and a shared Isaac Lab rollout entry
  for G1 + articulated door or G1 + rigid object;
- pure-PyTorch two-wrist force semantics, `P_cross`, `CrossEncoder`, contact
  gating, residual authority and action safety;
- a vectorized four-task residual environment with `c0`, `scaffold_only`, and
  `residual` modes, fixed normalization, reward, termination, metrics, and
  C1-C3 curriculum support;
- the Phase 5 asymmetric actor/critic and SemanticPPO path, including bounded
  Isaac smoke and single-task `push_door_hand` C1 learning acceptance.

The current code does not yet provide:

- raw SMPL-H OMOMO to G1 nonlinear retarget optimization;
- four-task joint PPO orchestration or Phase 6 joint-training evidence;
- the three Phase 7 scaffolds or held-out Phase 8 zero-shot evaluation;
- a validated non-privileged production scaffold, real F/T integration, or
  physical hardware evidence.

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
  scaffold_rollout_results.md
configs/
  pretrained_hdmi_rollout_cases.json
  phase4b5_numeric_contract.json
  phase5_ppo_v1.json
  phase5_learning_acceptance_v1.json
figures/
  somaforce_cross_pipeline.png
artifacts/scaffolds/hdmi_push_door_hand/v1/
  manifest.json
  observation_contract.json
  action_contract.json
artifacts/scaffolds/hdmi_push_box/v1/
  manifest.json
  observation_contract.json
  action_contract.json
artifacts/scaffolds/hdmi_move_suitcase/v1/
  manifest.json
  observation_contract.json
  action_contract.json
artifacts/scaffolds/hdmi_move_largebox/v1/
  manifest.json
  observation_contract.json
  action_contract.json
somaforce_cross/scaffold/
  pretrained_hdmi.py
  pretrained_hdmi_isaac.py
  reference_schema.py
  hdmi_adapter.py
  omomo_adapter.py
  reference_library.py
  contracts.py
somaforce_cross/force/        # pure-PyTorch force-semantic core
somaforce_cross/residual/     # bounded residual composition and safety
somaforce_cross/sensing/      # virtual wrist F/T contracts and corruption
somaforce_cross/envs/         # four-task Isaac residual environment
somaforce_cross/learning/     # asymmetric actor/critic and SemanticPPO
scripts/
  build_scaffold_reference.py
  batch_rollout_and_render_pretrained_hdmi_scaffolds.py
  export_pretrained_hdmi_scaffold.py       # HDMI export/oracle boundary only
  rollout_pretrained_hdmi_scaffold.py      # one task/condition -> one metrics file
  render_pretrained_hdmi_scaffold.py       # one task/condition -> one MP4
  verify_pretrained_hdmi_isolation.py
  smoke_phase4b5_environment.py
  smoke_phase5_ppo.py
  train_phase5.py             # governed learning acceptance, not a quick demo
tests/
  test_hdmi_omomo_pipeline.py
  test_pretrained_hdmi_scaffold.py
```

## Git and Git LFS CLI

This repository stores policy weights, simulator assets, reference arrays, and
rollout videos in Git LFS. Install both Git and Git LFS before cloning, then
enable the LFS filters for the current user:

```bash
git --version
git lfs version
git lfs install
```

Clone the active development branch and materialize all LFS objects:

```bash
git clone --branch v1 https://github.com/BillyLiu312/somaforce-cross.git
cd somaforce-cross
git lfs pull
git lfs fsck
git lfs ls-files
```

For a routine update of an existing clean checkout:

```bash
git switch v1
git pull --ff-only origin v1
git lfs pull
git status --short --branch
```

When publishing a scoped change, inspect both ordinary Git state and LFS state.
The normal `git push` runs the Git LFS pre-push hook and uploads referenced LFS
objects before updating the branch:

```bash
git add <paths>
git status --short
git lfs status
git commit -m "Describe the change"
git push origin v1
git lfs push --dry-run origin v1
```

The final dry run should report no pending objects. If an earlier push bypassed
the LFS hook, upload the objects explicitly and then push the Git ref again:

```bash
git lfs push origin v1
git push origin v1
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

Install the pure-PyTorch package, or include the pinned Phase 5 dependencies
when using the PPO components:

```bash
python -m pip install -e .
python -m pip install -e '.[phase5]'

python -m pytest -q
```

Use the Isaac Lab environment for simulator entry points:

```bash
ISAAC_PYTHON=/opt/miniconda3/envs/isaaclab/bin/python

# one rollout with explicit task and evolution parameters
$ISAAC_PYTHON scripts/rollout_pretrained_hdmi_scaffold.py \
  --task push_door_hand --condition high_friction --headless \
  --door-friction 10.0 --door-damping 0.55

# render exactly one task/condition without writing rollout metrics
$ISAAC_PYTHON scripts/render_pretrained_hdmi_scaffold.py \
  --task push_door_hand --condition high_friction \
  --door-friction 10.0 --door-damping 0.55

# sequentially run every configured task/condition and render one MP4 per case
$ISAAC_PYTHON scripts/batch_rollout_and_render_pretrained_hdmi_scaffolds.py

# bounded four-task residual-environment evidence
$ISAAC_PYTHON scripts/smoke_phase4b5_environment.py \
  --mode suite --python "$ISAAC_PYTHON" --output-dir outputs/phase4_environment

# bounded single-task PPO integration evidence, not full learning acceptance
$ISAAC_PYTHON scripts/smoke_phase5_ppo.py \
  --mode suite --python "$ISAAC_PYTHON" --output-dir outputs/phase5_smoke

# materialize the local/private artifact (requires the trusted HDMI checkout)
$ISAAC_PYTHON scripts/export_pretrained_hdmi_scaffold.py \
  --hdmi-root /inspire/hdd/global_user/liumengfan-253108110079/yindianyu_workspace/somaforce/HDMI \
  --task move_suitcase --force

# runtime checks without HDMI on PYTHONPATH
env -u PYTHONPATH $ISAAC_PYTHON scripts/verify_pretrained_hdmi_isolation.py --task move_suitcase

```

Both entries operate on exactly one task and one caller-named `--condition`.
Evolution is defined by explicit arguments such as `--door-friction`,
`--door-damping`, `--object-mass`, `--object-friction`, `--object-com-offset`,
and initial object pose. A rollout writes one independent
`rollout_metrics_<condition>.json`; it never merges multiple conditions into an
aggregate file. The JSON records the actual task-specific settings used.
Rollout metrics are mutable run evidence and are intentionally excluded from
the artifact `SHA256SUMS`; producing a rollout never modifies that checksum file.

The render entry reuses the same evolution arguments, streams frames directly
to H.264, and verifies codec, resolution, frame readability, and nonblank
pixels. It does not write or update rollout metrics.

The current per-task case comparison, qualitative findings, metrics links, and
videos are collected in
[docs/scaffold_rollout_results.md](docs/scaffold_rollout_results.md).

The batch entry reads `configs/pretrained_hdmi_rollout_cases.json` and invokes
the two single-condition entries sequentially. The default matrix contains 14
conditions across all four tasks. Metrics remain one JSON per condition beside
its artifact, while videos are written as
`videos/<task>/<condition>.mp4`. The four task directories are siblings;
`research_category` remains metadata and does not affect filesystem layout.
Use `--dry-run` to inspect the full
command plan, `--resume` to skip existing outputs, repeated `--task` or
`--condition` filters for a subset, and `--rollout-only` or `--render-only` when
only one stage is needed.

The rollout command reports `a_nom`, the 23-D action order, reference phase, door
joint/progress, contact and support metrics, root-height stability, finite-state
status, and the zero-hook result. Omitting `--headless` enables Isaac Lab GUI
mode where a display is available. Isaac Sim 5.1 may block during teardown after
an interactive GUI session; headless batch mode exits after flushing metrics.

The rollout is policy-driven: the environment receives the frozen teacher's
`a_nom` exactly before the declared delay, smoothing, scaling, and joint-position
control stages. Fixed-input action parity is validated to `1e-5`, but a matched
closed-loop rollout against the complete HDMI environment is not yet claimed.

Push-box rollout metrics separate the physical box pose/displacement
from the reference trajectory. GUI mode renders the physical USD normally; the
translucent green reference box is hidden by default and can be enabled with
`--show-reference-box` for trajectory debugging. It is never used as actual
progress. The rollout entry exposes `--box-mass`, `--box-friction`,
`--box-com-offset`, `--initial-object-xy`, `--initial-object-yaw`, and
`--contact-target-offset`. The condition name is metadata and a filename slug;
the numeric arguments are the authority for the physical evolution.

Move-suitcase defaults to a 472-step, 50 Hz deterministic teacher-mean rollout
with a 1.5 kg object. Use explicit object parameters to define each evolution;
each condition receives its own metrics file.

Move-largebox uses the same manifest-driven payload runtime with a local
URDF/OBJ asset closure. It defaults to 199 steps and 1.0 kg; each explicitly
named condition writes physical lift, displacement, tracking, contact,
stability, and zero-hook evidence to a separate file.

For GUI playback, render occurs once per 50 Hz control step rather than once per
physics substep. `--realtime --playback-rate 1.0` follows reference wall time;
increase the rate or omit `--realtime` for faster visualization.

## Artifact Versioning

The complete `artifacts/` tree is visible to Git. Large runtime files use Git
LFS according to `.gitattributes`, including `.pt`, `.pth`, `.onnx`, `.npz`,
`.npy`, `.usd`, `.usda`, `.obj`, and `.mp4`. Training `checkpoints/`, `*.ckpt`,
and `checkpoint_*.pt/.pth` remain ignored. Before committing artifacts, verify
redistribution permission in `THIRD_PARTY.md`; LFS changes storage mechanics,
not licensing rights.

## Current V1 Boundary

Phases 0-5 of the governed V1 implementation plan are implemented and admitted.
The admitted learning evidence is the bounded single-task `push_door_hand` C1
acceptance; it is not four-task learning, robustness, zero-shot, deployment, or
hardware evidence. The next implementation-plan requirement is Phase 6
four-task joint training with balanced task transitions, synchronized shared
weights, per-task metrics, pooled metrics, nominal retention, and curriculum.
