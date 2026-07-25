# AGENTS.md

## Warmup Prompt for SomaForce-Cross

You are entering the **SomaForce-Cross** repository.

This repository is the standalone engineering baseline for the current selected SomaForce-Cross pipeline. It should be treated as a self-contained starting point for future engineering work, not as a place to rediscover or redesign the method from scratch.

SomaForce-Cross studies **force-aware whole-body humanoid contact-rich manipulation**. The first target domains are door interaction and heavy-object lift/carry/place. The central research idea is:

> Contact force should not be treated only as a safety threshold. During execution, wrench history is an embodied observation of mechanism constraint mismatch, payload/load distribution, and required whole-body mobilization.

The method should help a humanoid answer, during contact:

- Which direction appears task-consistent?
- Which direction appears constrained or force-wasting?
- How much force resistance is present?
- Is hand-level compliance enough, or should the torso/body posture be mobilized?
- Can deployable wrist F/T history support the same force-semantic adaptation
  across tasks and physical mismatch conditions?

## Current Method Identity

Use the method name:

```text
SomaForce-Cross
```

Interpretation:

> Scaffolded humanoid force adaptation via cross-semantic force conditioning.

For V1, read this as cross-semantic force adaptation rather than a mandatory
teacher-to-student training route. The deployable-input actor directly encodes
noisy wrist F/T history into direction and magnitude semantic distributions,
combines them through an outer-product representation, and learns a bounded
residual with asymmetric PPO. Clean simulated wrench supplies auxiliary
semantic targets; privileged simulator state is critic-only.

Do not revert the method name to `SomaForce-RLD-S` as the main identity. That older label described the training route, but it did not capture the core innovation as clearly.

## Selected Pipeline

The current V1 pipeline is:

```text
frozen policy trained with the official HDMI implementation
  -> standalone exported artifact (no HDMI runtime dependency)
  -> versioned observation/action adapter
  -> nominal normalized action a_nom [B, 23]
  -> deployable virtual wrist F/T histories
  -> predicted direction/magnitude force semantics
  -> p_dir outer p_mag
  -> P_cross
  -> flatten(P_cross)
  -> CrossEncoder
  -> z_cross
  -> contact-gated bounded residual actor
  -> a_total in normalized 23-D action coordinates

clean simulated wrench -> p_dir/p_mag auxiliary targets
privileged simulator state -> critic only
```

The canonical HDMI/OMOMO reference library remains part of the broader task
pipeline for replay, diagnostics, and future payload scaffolds. It is separate
from the first learned door-policy artifact: canonical G1 references have 29
joints, while the audited pretrained door policy currently controls 23 joints.

The current method and implementation sources of truth are:

```text
docs/v1_force_residual_plan.md
docs/v1_force_residual_implementation_plan.md
```

Follow `docs/pretrained_hdmi_scaffold_rules.md` for frozen scaffold provenance,
contracts, and deployment gates. Use `docs/hdmi_omomo_scaffold_pipeline.md` only
for canonical reference construction and future reference-driven tasks.

## Current Pretrained Scaffold Decision

The current four scaffolds are frozen pretrained HDMI policies migrated as
standalone artifacts for `push_door_hand`, `push_box`, `move_suitcase`, and
`move_largebox`. Users of SomaForce-Cross must not need to clone or install
HDMI. The HDMI checkout is allowed only in a separate export/oracle environment
for producing artifacts and deterministic parity traces.

Read and follow:

```text
docs/pretrained_hdmi_scaffold_rules.md
```

That document is the source of truth for checkpoint provenance, teacher versus
deployable policy roles, 23-D action semantics, artifact layout, licensing, and
verification gates. Goal-mode execution prompts are intentionally supplied by
the project owner per development round and are not stored in this repository.

## Non-Negotiable Pipeline Decisions

The following decisions are already selected and should not be contradicted unless the project owner explicitly changes the method:

- The first door scaffold is a **frozen pretrained HDMI policy artifact**, not a new scaffold trained inside SomaForce-Cross.
- SomaForce-Cross runtime code must not import HDMI/`active_adaptation`, require an HDMI checkout, or load a pickle that needs HDMI-defined classes.
- The audited learned action is 23-D and must remain distinct from the canonical 29-joint G1 reference schema. Use an explicit joint mapping.
- `a_nom` and the first Cross residual share normalized 23-D action coordinates. Apply joint-position scaling exactly once after their bounded sum.
- The current `phase=train` HDMI teacher consumes privileged inputs. It is not the production scaffold unless a non-privileged `actor_adapt`/finetune path is separately exported and validated.
- Keep the pretrained scaffold frozen and in evaluation mode for the primary method and scaffold-only baseline.
- Preserve HDMI attribution and artifact provenance. Runtime independence must never be presented as independent authorship of the scaffold.
- Do not redistribute HDMI code, weights, motion, or assets until their permissions are documented; the audited HDMI repository has no root code/data license.
- Every source reference ingested or replayed by SomaForce-Cross must pass through the canonical G1/object/contact schema. HDMI oracle-only parity traces are kept at the export boundary.
- OMOMO is not a door or wrench dataset. Do not infer mass, force, CoM, or load-share labels from human kinematics; generate them in Isaac Lab rollouts.
- Alternative trackers are separate future baselines, not dependencies of the selected door scaffold.
- SomaForce-Cross learns a **bounded force-conditioned residual**, not the whole task behavior from scratch.
- The deployed hardware assumption is a **real wrist/end-effector F/T sensor**.
- Simulation should include a **virtual wrist F/T sensor model** that mimics deployable sensing.
- The actor must not receive perfect object-side contact truth, true hinge/slider state, or simulator-only privileged labels at deployment.
- Clean simulated wrench is used for auxiliary semantic targets and diagnostics.
- Other privileged simulator state is critic-only and must not enter the residual actor.
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
- `p_dir`, `p_mag`, and `P_cross` are retained for auxiliary supervision,
  logging, visualization, and ablations.
- `P_cross` is deterministic from `p_dir` and `p_mag`; V1 has no independent
  `lambda_cross` loss.

## Why a Pretrained HDMI Scaffold

The pretrained HDMI policy already supplies nominal object-aware door behavior.
Migrating the frozen policy avoids rebuilding the prior scaffold method and
keeps engineering effort focused on the SomaForce-Cross force contribution.
The standalone wrapper must nevertheless reproduce HDMI's observation history,
normalization, reference command, object-relative inputs, action order, scaling,
delay, smoothing, and reset semantics.

OMOMO remains the selected source for future heavy-payload references. It must
be jointly retargeted to G1 with the object trajectory and wrist-to-object
transforms preserved. Low-confidence contact frames are masked. A payload
scaffold artifact requires its own documented training, export, and verification
route; it is not implied by the door checkpoint.

The scaffold is responsible for nominal task tracking, such as:

- approximate hand trajectory for door opening;
- approximate lift/carry/place motion for heavy payloads;
- approximate body reference for reach, balance, and contact maintenance;
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

In simulation, do not model the actor input as perfect contact truth. Model a
virtual F/T sensor signal:

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
  simulation-only semantic targets, critic inputs, metrics

virtual wrist F/T observation:
  simulation version of what the real sensor would provide

real wrist F/T history:
  deployment-side residual actor input
```

This distinction is central to the sim-to-real story.

## Cross Semantic Representation

The cross representation should be understood as a **joint semantic distribution**, not a geometric vector cross product.

If:

```text
p_dir: direction semantic distribution
p_mag: magnitude / recovery-level semantic distribution
```

then:

```text
P_cross(i, j) = p_dir(i) * p_mag(j)
```

Examples of joint semantic cells:

```text
+Fx x low
-Fy x medium
+Fz x high
-Mz x critical/recover
neutral/mixed x release/none
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

## V1 Asymmetric Learning Semantics

V1 does not train a privileged residual teacher actor. The residual actor uses
only deployable inputs:

```text
two-wrist noisy F/T history
wrist twist
contact probability and sensor quality
whole-body proprioception
a_nom history and previous executed a_total
  -> p_dir, p_mag
  -> P_cross
  -> z_cross
  -> bounded residual
```

The critic may additionally consume simulator object, mismatch, progress,
contact, and stability state. Clean simulated wrench generates `p_dir` and
`p_mag` targets for the two auxiliary KL losses. A clean-wrench privileged
actor is permitted only as a later upper-bound/bootstrap experiment after the
primary V1 route has been evaluated.

## Research Scope

The V1 residual training targets are:

- `push_door-hand`;
- `push_box`;
- `move_suitcase`;
- `move_largebox`;
- `open_foldchair-sit`;
- `topple_wood_board_and_cross`;
- `roll_ball-hand`.

The first four have current frozen artifacts. The remaining three join training
only after independent audit, export, parity, rollout, and zero-residual gates.
Held-out zero-shot candidates are `carry_and_place_bread_box`,
`carry_box_over_shoulder`, `move_foam`, `move_stool-climb`, and `truman`.
Held-out evaluation freezes residual weights, semantic bins, sensor contract,
and authority limits.

The main mismatch sources are:

- hinge-axis error;
- payload mass, CoM, inertia, and load-share error;
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
- retention of nominal scaffold behavior while using deployable F/T histories.

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

> Frozen, explicitly attributed task scaffolds provide nominal humanoid-object
> motion through standalone artifacts. A deployable-input residual actor maps
> noisy two-wrist F/T histories into direction and magnitude semantic
> distributions, forms their outer-product `P_cross`, and uses `z_cross` to
> produce a contact-gated bounded whole-body residual. Clean simulated wrench
> supplies auxiliary semantic targets and privileged state remains critic-only.

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
- deployable noisy-F/T actor versus clean-F/T upper bound;
- symmetric critic versus privileged asymmetric critic.

The most important diagnostic signal is:

```text
P_cross(t)
```

because it exposes whether the method is using force as a structured semantic signal rather than as an opaque scalar or raw wrench input.

## Training Job Script Workflow

When preparing cluster or distributed training jobs for SomaForce-Cross, HDMI, Isaac Lab, or PyTorch workflows, use the Codex skill:

```text
$training-job-scripts
/inspire/hdd/global_user/liumengfan-253108110079/lmf-workspace/codex/skills/training-job-scripts/SKILL.md
```

Reusable task scripts should live under:

```text
/inspire/hdd/global_user/liumengfan-253108110079/lmf-workspace/training-job-scripts
```

Organize scripts by project and task, for example:

```text
training-job-scripts/hdmi/push_door_hand/smoke_4gpu.sh
training-job-scripts/hdmi/push_door_hand/train_4gpu.sh
```

For platform submissions, prefer returning a one-line command:

```bash
bash /inspire/hdd/global_user/liumengfan-253108110079/lmf-workspace/training-job-scripts/<project>/<task_slug>/<script>.sh
```

Generated scripts should be self-contained, use absolute workspace paths, write logs through `tee` into persistent log directories, and save checkpoints to persistent output directories rather than only `/tmp`. Prefer explicit Python module launch (`/opt/miniconda3/envs/isaaclab/bin/python -m torch.distributed.run`) over `torchrun` on this platform.
