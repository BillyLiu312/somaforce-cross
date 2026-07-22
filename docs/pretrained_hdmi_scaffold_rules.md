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

The implemented scope includes independently selectable G1 `push_door-hand`,
G1 `push_box`, G1 `move_suitcase`, and G1 `move_largebox` frozen artifacts. The
canonical reference library remains useful for replay, diagnostics, future
tasks, and OMOMO payload work, but canonical reference conversion is not the
trained policy scaffold.

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

The push-box source artifact was audited at the same HDMI revision from:

```text
checkpoint:
  outputs/push_box/hdmi-push-box-4gpu-20260721_200242/rank_0/checkpoint_final.pt
resolved config:
  outputs/push_box/hdmi-push-box-4gpu-20260721_200242/rank_0/.hydra/config.yaml
motion:
  data/motion/g1/push_box/push_box-VID_20250423_220958-light-high-adjust_root_height
checkpoint SHA256:
  83ddea2ed34919be575d678fcac43c3b3d65643a22d3e45ea995f80e9c0a6d41
config SHA256:
  5902fe6b0e44a597c9e98504cfbb83110b6baf036d0b827cff122bfdc2a937c3
asset metadata SHA256:
  314eae4a38925411cc37cdb56a7fb8a57174d51ae130118f23f0e4aa2b92537d
motion SHA256:
  8921f589f8470d25ab1dd0fa65f9b9057676ab122fdcc8d87f43d9c527ed461e
motion metadata SHA256:
  dffeeed04f3001710bb8ab83061e8e1d1be256d9e2e35a4efa0c6011ae1d4105
```

Push-box uses a 25-name motion reference mapped by joint name into the
29-joint Isaac articulation and the shared 23-D action order. Its object group
is 10-D because it contains two contact targets. The privileged encoder input
is `1714 + 10 = 1724`; the actor remains `861 -> 23`. These dimensions must
come from the artifact contract rather than task-name conditionals in network
code.

The move-suitcase source was audited at the same HDMI revision from the dirty
checkout at
`/inspire/hdd/global_user/liumengfan-253108110079/yindianyu_workspace/somaforce/HDMI`.
Its W&B run saved `ppo_roa.py`, resolved `cfg.yaml`, and checkpoint tensor
shapes are the policy-structure oracle; the later dirty-tree force-residual
implementation is excluded. Key checksums are:

```text
checkpoint: be31c2a9893898d26bd989c876d11e359bda5e0d36349b4ce9ca85c758237e14
config:     7ead1ad6c79b8ddadf219917dac574f58d59e30e1931d714034ea426b050b1c5
asset_meta: 400425d1137027eb82de4ba1c92bcab7cdfae69defd09a97895d3f2e3b457258
G1 USD:     faf4d267a7a93fd16186e77e4c2802aa8ea977b5bb971b72fa63dee99c33200d
suitcase:   d3c25a338fffa58ddfef084fe2cff06023dead787ab8a3c2aaa1e3a08925cf6b
motion:     d6d08c5792fc7396c89629876d91cfec26404fd803ccb9ffc2439c66ce51f3cf
motion meta:c20c8a5c5094500d8588a9152f77baa65fc397a476c15717d1db78cb1c7093a0
```

Its contract is `command[356]`, `policy[249]`, `object[10]`,
`privileged[1714]`, encoder input `1724`, actor input `861`, and normalized
action `[23]`. The reference has 29 joints, 472 frames, and 50 Hz. The task
category is `HEAVY_PAYLOAD`; nominal mass is 1.5 kg within the 1.2-1.8 kg
training range. Both the 29-D reference and Isaac articulation are mapped by
joint name. Six wrist joints are not policy actions; the initial left/right
wrist-yaw defaults are -0.4/+0.4 radians.

The move-largebox source was audited from the dirty local HDMI checkout at
revision `32282f6dcf26cae70b814d585ceb12cc38aa1b60`. The resolved run config,
checkpoint tensor shapes, asset metadata, and fixed-input HDMI source oracle are
the authorities because no run-saved policy source was available. Key checksums
are:

```text
checkpoint: a4c33009a756c3e29d2d389061eaa96926ab1c868aceaa9f8bc810e4ef98504d
config:     10f222bf5b9170fd316e5071d9e9150cd02b865be2b7f1bbc332a808be1aa678
asset_meta: 400425d1137027eb82de4ba1c92bcab7cdfae69defd09a97895d3f2e3b457258
G1 USD:     faf4d267a7a93fd16186e77e4c2802aa8ea977b5bb971b72fa63dee99c33200d
URDF:       e3281a96e9b6aff6988ae4fc6d219f61751bebdc4a4628ae3045eeb4e9aa99c4
OBJ:        dbcc11281f62e9226f49165252375080d2e490e5a3f0ab6ba917acbd8f7abc1c
motion:     1e79ff64f495200afb4830468a3a050328f835a6b7e0c336a874722f8849d941
motion meta:da27c333286a6718e51198b7321f369b4633e808ca3c7a441a1b478c1e8f1a99
video:      01594f429ecd9a68c93da8183ac43218e412579b4b06034b5ab06a77d8f5bdd7
```

Its contract is `command[356]`, `policy[249]`, `object[10]`,
`privileged[1714]`, encoder input `1724`, actor input `861`, and normalized
action `[23]`. The 199-frame, 50 Hz reference has 29 named joints, two wrist
contact targets, a 1.0 kg nominal mass, and a 0.8-1.2 kg training mass range.
The source video is one 250-frame nominal episode, not a robustness estimate.
The artifact is an HDMI-derived privileged simulation teacher and remains
local/private-only because weight, motion, robot, URDF, and OBJ redistribution
permissions are undocumented.

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

The push-box artifact uses the parallel path
`artifacts/scaffolds/hdmi_push_box/v1` with `assets/box.usd`. Both remain
local/private under the same permission gate.

The move-suitcase artifact uses
`artifacts/scaffolds/hdmi_move_suitcase/v1` with `assets/suitcase.usd` and the
same rubber-hand G1 USD content hash as the door artifact. The suitcase USD
has no sublayers and retains local geometry/collision, but its visual material
references NVIDIA's remote `Cardboard.mdl`. Headless physics must work when
that material is unavailable. The original USD is preserved; no local material
conversion is claimed. Cite HDMI and OMOMO, and keep the artifact local/private
until code, motion, weights, robot/object asset, and OMOMO redistribution rights
are confirmed.

The move-largebox artifact uses
`artifacts/scaffolds/hdmi_move_largebox/v1` with `assets/largebox.urdf`, its
local `assets/largebox.obj` visual/collision mesh, and the same rubber-hand G1
USD hash. Isaac Lab imports the URDF from the standalone artifact; no source
path is resolved at runtime. Cite HDMI and OMOMO, and keep the full artifact
local/private until all redistribution permissions are documented.

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
