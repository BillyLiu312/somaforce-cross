# Pretrained HDMI Scaffold Integration Rules

Updated: 2026-07-21

## Status and Scope

This document is the source of truth for the first SomaForce-Cross door
scaffold. It supersedes earlier plans to vendor HDMI or to retrain an
HDMI-style door scaffold inside this repository.

The selected route is:

```text
frozen policy trained with the official HDMI implementation
  -> standalone exported scaffold artifact
  -> SomaForce-Cross observation and action adapters
  -> nominal action a_nom
  -> bounded Cross force residual
  -> final action in the same normalized action coordinates
```

The immediate scope is the G1 `push_door-hand` task. The canonical reference
library remains useful for replay, diagnostics, future tasks, and OMOMO payload
work, but canonical reference conversion is not the trained policy scaffold.

## Research Attribution

The pretrained scaffold is prior work, not a SomaForce-Cross contribution.
Repository documentation, paper text, artifacts, and experiment tables must
describe it as a frozen policy trained with the official HDMI implementation
and cite HDMI.

Software independence does not erase provenance. The research contribution is
the force-semantic representation, bounded residual adaptation, virtual wrist
F/T model, and teacher-to-student distillation built on top of the scaffold.

Use wording equivalent to:

> We use a frozen policy trained with the official HDMI implementation as the
> nominal task scaffold. We export it as a standalone inference artifact and
> do not update it during the primary SomaForce-Cross experiments.

Do not claim that the migrated checkpoint, its HDMI training method, or its
released reference motion is an original SomaForce-Cross contribution.

## Standalone Requirement

A clean SomaForce-Cross environment must be able to run the scaffold without:

- cloning HDMI;
- installing the `active_adaptation` package;
- importing an HDMI Python module;
- loading a pickled Python object whose class is defined by HDMI;
- resolving a file by an absolute path into the HDMI checkout.

HDMI may be used in a separate export/oracle environment to produce the frozen
artifact and parity traces. The released runtime boundary must use a flat,
versioned tensor contract and a self-contained PyTorch export, TorchScript, or
ONNX artifact. For batched Isaac Lab training, prefer a dependency-free PyTorch
export; ONNX is an optional interoperability artifact.

Do not vendor the complete HDMI repository. Do not copy HDMI-specific command,
reward, trainer, or configuration code into this repository merely to make the
checkpoint load. Implement only the observation, reference, scene, and action
semantics required by the frozen inference contract.

## Audited Source Snapshot

The first local source candidate was audited from:

```text
HDMI git revision:
  32282f6dcf26cae70b814d585ceb12cc38aa1b60

checkpoint:
  /inspire/hdd/global_user/liumengfan-253108110079/lmf-workspace/HDMI/
  outputs/doorpushhand/hdmi-doorpushhand-4gpu/rank_0/checkpoint_final.pt

resolved training config:
  /inspire/hdd/global_user/liumengfan-253108110079/lmf-workspace/HDMI/
  outputs/doorpushhand/hdmi-doorpushhand-4gpu/rank_0/.hydra/config.yaml

asset metadata:
  /inspire/hdd/global_user/liumengfan-253108110079/lmf-workspace/HDMI/
  outputs/doorpushhand/hdmi-doorpushhand-4gpu/rank_0/asset_meta.json

reference motion:
  /inspire/hdd/global_user/liumengfan-253108110079/lmf-workspace/HDMI/
  data/motion/data_for_sim/push_door-hand-0828/
```

The HDMI worktree had local changes in `scripts/train.py` and
`scripts/train_sequential.py` when audited. The artifact manifest must record
the upstream revision, dirty-worktree state, resolved config checksum, source
checkpoint checksum, export tool revision, and export timestamp.

Audited checksums:

```text
checkpoint SHA256:
  24682b7b66c38f784ca5939dad3841b1b1e5b801c47d2fea59c5f85fd90c238c

resolved config SHA256:
  fb6ddd53135550b244fd35b707cf2b98e069ddf38dfe456af5b930e4189a074c
```

## Current Checkpoint Facts

Do not infer the runtime contract from the G1 model name.

- The canonical reference trajectory contains 29 G1 joints.
- The trained checkpoint actor outputs 23 actions. Its output layer has shape
  `[23, 256]` because the six wrist joints are not included in HDMI's action
  scaling configuration.
- `a_nom` is a normalized joint-position action consumed by the action manager;
  it is not a raw 29-DOF joint-position vector and not a torque command.
- The resolved checkpoint config uses `algo.phase: train`,
  `enable_residual_distillation: true`, and an MLP adaptation module.
- The latest run used 4096 environments with a configured total of 600 million
  training frames.
- In `phase: train`, HDMI's default rollout path uses `encoder_priv + actor`.
  It consumes simulator-privileged observations and is not automatically a
  deployable scaffold.
- The checkpoint also contains `adapt_module` and `actor_adapt` parameters.
  They are only a deployable candidate until their inputs and rollout quality
  have been verified without privileged observations.
- The checkpoint contains observation normalization state. It must be exported
  and tested as part of the policy, not recomputed from new rollouts.

The export process must discover and record the exact 23 action joint names in
their resolved order. Do not assume that canonical 29-joint order is also the
policy action order.

## Teacher and Deployable Boundaries

Maintain two explicit roles if both are exported:

```text
HDMI teacher/oracle artifact:
  may use privileged simulator observations
  used only for parity, diagnosis, or simulation-side comparison

HDMI deployable scaffold artifact:
  uses only sensor-realizable observations and declared task/reference priors
  supplies a_nom to the final SomaForce-Cross deployment path
```

Never silently label the current `phase=train` teacher path as deployable. The
production scaffold must be `actor_adapt`, a finetuned HDMI student, or another
explicitly validated non-privileged export. If the candidate fails rollout
validation, report that a deployable HDMI adaptation/finetuning stage is still
required; do not weaken the input contract to make the test pass.

Object observations must also be classified. Perfect simulator object pose,
joint state, hinge state, or contact truth is privileged unless a corresponding
real perception or sensing path is explicitly part of the deployment system.

## Versioned Inference Contract

The public scaffold input must be a flat, named, versioned tensor contract. It
may be assembled from the following groups after their exact shapes and frame
conventions are audited:

- future robot/object reference command and motion phase;
- root angular velocity;
- projected gravity;
- joint-position history at steps `[0, 1, 2, 3, 4, 8]`;
- previous action history with three steps;
- declared object-relative observations;
- reference contact position;
- adaptation state, if required by the selected deployable policy.

The manifest must specify for every field:

- tensor name, dtype, shape, and concatenation order;
- units and coordinate frame;
- quaternion convention where applicable;
- history order and reset behavior;
- normalization mean/scale and clipping;
- whether the field is deployable, task prior, or privileged;
- noise behavior in training and deterministic parity mode.

The scaffold output contract is initially:

```text
a_nom: float32[B, 23]
```

It must also expose the action joint names, action scaling, default joint
positions, action bounds, delay semantics, smoothing coefficient, control rate,
and physics rate through metadata. Confidence/health output may be added by the
SomaForce-Cross wrapper, but it must not change `a_nom` silently.

## Action Composition Rule

The Cross residual and `a_nom` must share the same 23-dimensional normalized
action coordinates for the first baseline:

```text
delta_bounded = clip(Delta a_force, -residual_limit, residual_limit)
a_total = clip(a_nom + safety_gain * delta_bounded, action_low, action_high)
```

Apply the HDMI-compatible joint-position scaling exactly once after this sum.
Do not add one residual in radians to another value in normalized action units.
Do not expand the force residual to 29 dimensions until wrist control has an
explicit actuator, observation, safety, and evaluation design.

The pretrained scaffold must run in evaluation mode, with frozen parameters and
no gradient updates, for the primary method and scaffold-only baseline. Any
joint finetuning is a separate experiment and must not replace the frozen
baseline.

## Asset and Data Rules

The standalone package needs compatible G1, door, and reference-motion assets,
but technical necessity does not grant redistribution rights.

At the 2026-07-21 audit, the HDMI repository had no root `LICENSE`,
`DATA_LICENSE`, or `NOTICE`. Therefore:

- local/private migration may proceed;
- do not publish or redistribute HDMI code, motion files, USD assets, or derived
  exported artifacts until their applicable permissions are documented;
- record the source and checksum of every third-party file;
- prefer independently licensed G1/door assets when replacement preserves the
  policy's geometry and dynamics;
- treat an asset replacement as a behavior change and rerun rollout parity;
- add third-party notices and citation metadata before public release.

A downloader avoids bundling a file but does not by itself grant permission to
use it. License and provenance gates must be explicit in the artifact manifest.

## Required Artifact Layout

Use a versioned logical layout such as:

```text
artifacts/scaffolds/hdmi_push_door_hand/v1/
  manifest.yaml
  policy.pt2
  policy.onnx                 # optional
  observation_contract.yaml
  normalization.npz
  reference/motion.npz
  reference/meta.json
  assets/g1.usd
  assets/door.usd
  parity/source_outputs.npz
  SHA256SUMS
  THIRD_PARTY.md
```

Large artifacts may remain outside Git and be materialized by an authorized
download/install step. The manifest and schemas should remain versioned in Git.
No code may rely on the example directory existing without a clear setup error.

## Verification Gates

The migration is incomplete until all applicable gates pass:

1. Import isolation: an environment without HDMI can import and instantiate the
   scaffold adapter.
2. Artifact isolation: loading the exported policy imports no HDMI classes and
   resolves no path inside the HDMI checkout.
3. Contract validation: input names, shapes, dtypes, frames, histories, action
   order, normalization, and control rates match the manifest.
4. Fixed-batch parity: on recorded deterministic inputs, the standalone export
   matches the selected HDMI source policy. Target `max_abs_action_error <= 1e-5`
   unless an export backend requires a documented tolerance change.
5. Reset/history parity: episode reset, history buffers, previous actions, and
   adaptation state match on the first steps of an episode.
6. Rollout parity: under matched seed, assets, physics, and deterministic
   settings, door progress and success are statistically consistent with the
   HDMI oracle. Report the comparison rather than relying on video inspection.
7. Scaffold-only gate: nominal settings complete or nearly complete the door
   task, while selected mismatch settings remain meaningful and numerically
   stable.
8. Residual boundary gate: zero Cross residual reproduces the standalone
   scaffold action exactly; bounded nonzero residual is scaled only once.
9. Deployment gate: the selected production scaffold uses no undeclared
   simulator-only privileged input.
10. Provenance gate: every distributed artifact has source, checksum, citation,
    and permission status.

Do not begin full Cross teacher training before gates 1-8 pass. Privileged
force-label collection may be prototyped earlier, but it must not conceal a
broken scaffold migration.

## Relationship to Existing Modules

Keep these concepts separate:

```text
CanonicalReferenceEpisode / ReferenceLibrary:
  29-joint reference data, replay, metadata, diagnostics

PretrainedHDMIScaffold:
  frozen learned policy, versioned deployable inputs, 23-D a_nom

SomaForce-Cross residual:
  learned bounded correction in the same 23-D action coordinates
```

Do not change the canonical 29-joint schema merely because the first learned
policy controls 23 joints. Add explicit reference-to-action joint mappings.

Legacy tracker compatibility code was removed from this runtime repository;
future tracker baselines must live behind an explicit standalone boundary.
