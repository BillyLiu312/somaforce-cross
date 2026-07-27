# Phase 4A Vectorized Residual Environment Contract Audit

Date: 2026-07-27

Status: Phase 4A contracts ratified against committed Phase 3C;
**Phase 4B implementation has not started**

## 1. Scope and decision

This audit is limited to the Phase 4 contracts required for a future batched
residual-training environment. It does not implement a `DirectRLEnv`, residual
actor, critic, PPO, training, or new scaffold migration.

The audit decision is **Ratified With Gates**:

- Phase 1 normalized force semantics and Phase 2 normalized residual action
  composition have source and unit-test evidence.
- Committed Phase 3 source covers 6-D wrist joint-wrench extraction, frame
  transformation, tare, corruption, sensor quality, contact detection,
  clean-wrench semantic targets, and partial reset.
- Phase 3C is committed at `d3ada5801203295364d0718dd7a1ba097d5d791d`.
  `ContactDetector`, `ContactDetectorOutput`, `TwoWristSemanticTargets`,
  `direction_soft_targets`, and `two_wrist_semantic_targets` are package-level
  public APIs. The Phase 1-3 targeted suite passes 92 tests and the full suite
  passes 120 tests in this ratification run.
- No Phase 4 environment, observation builder, task adapter, reward manager,
  termination manager, mismatch sampler, or RSL-RL integration exists.
- The structured observation, reset, runtime, privilege, and staged Phase 4B
  boundaries below are approved for later implementation Goals. This Goal
  authorizes only this report and its local commit; it does not authorize or
  implement Phase 4B.
- Reward weights, reward clipping/scales, observation/critic normalizers,
  mismatch distributions, and curriculum sampling probabilities remain
  unapproved numeric contracts. Phase 4B1 must not invent defaults for them.

## 2. Git and file boundary

Start snapshot:

```text
branch: v1
HEAD: d3ada5801203295364d0718dd7a1ba097d5d791d
local v1: d3ada5801203295364d0718dd7a1ba097d5d791d
origin/v1 tracking ref: d3ada5801203295364d0718dd7a1ba097d5d791d
live origin refs/heads/v1: d3ada5801203295364d0718dd7a1ba097d5d791d
status:
  ## v1...origin/v1
  ?? docs/phase4a_contract_audit.md
```

The audit report was the only untracked file at entry. Phase 3C is part of
HEAD, not a worktree candidate. Commit `d3ada580` contains exactly the two
package export edits, the detector and semantic-target sources, and their test.

Allowed modification list for this Goal:

```text
docs/phase4a_contract_audit.md
```

All source, test, config, artifact, asset, and branch files are read-only. This
Goal permits one local report-only commit after verification. Push, checkout,
reset, history rewriting, and artifact changes remain prohibited.

## 3. Sources inspected

Governing contracts, read completely:

- `AGENTS.md`
- `docs/v1_force_residual_plan.md`
- `docs/v1_force_residual_implementation_plan.md`
- `docs/pretrained_hdmi_scaffold_rules.md`

Compatibility and physical evidence:

- `docs/module_contracts.md`
- `docs/implementation_plan.md`
- `docs/scaffold_rollout_results.md`
- `configs/pretrained_hdmi_rollout_cases.json`
- all four `artifacts/scaffolds/*/v1/{manifest.json,observation_contract.json,action_contract.json}`
- all four parity fixtures and 14 stored rollout metric JSON files

Source paths:

- `somaforce_cross/force/*.py`
- `somaforce_cross/residual/*.py`
- `somaforce_cross/sensing/*.py`
- `somaforce_cross/scaffold/contracts.py`
- `somaforce_cross/scaffold/pretrained_hdmi.py`
- `somaforce_cross/scaffold/pretrained_hdmi_isaac.py`
- `somaforce_cross/scaffold/wrist_wrench_isaac.py`
- `scripts/rollout_pretrained_hdmi_scaffold.py`
- `scripts/batch_rollout_and_render_pretrained_hdmi_scaffolds.py`

Test paths:

- `tests/test_force_core.py`
- `tests/test_residual_action.py`
- `tests/test_virtual_ft.py`
- `tests/test_wrist_wrench_source_contract.py`
- `tests/test_pretrained_hdmi_scaffold.py`
- `tests/test_batch_scaffold_rollouts.py`
- `tests/test_contact_semantics.py`
- the remaining repository tests through the full suite

## 4. Current implementation evidence

| Capability | Actual source fact | Test/runtime evidence | Phase 4 result |
| --- | --- | --- | --- |
| `DirectRLEnv` | No class, config, registration, or environment package exists. | Repository-wide symbol search returned no hit. | Missing. |
| `ResidualActor` | Only the design name exists; no network class exists. | No source/test hit. | Missing. |
| `PrivilegedCritic` | Only the design name exists; no network class exists. | No source/test hit. | Missing. |
| RSL-RL/PPO | No import, dependency, runner config, or wrapper exists; `pyproject.toml` lists only imageio, numpy, and torch dependencies. | Repository-wide import/symbol search returned no hit. | Correctly deferred to Phase 5. |
| Force policy core | `WristHistoryBuffer`, shared causal `WristTokenEncoder`, semantic heads, `CrossEncoder`, and `ContactGate` exist. | Targeted and full tests pass; exact `[B,2,16,14]`, `[B,13]`, `[B,5]`, `[B,13,5]`, and `[B,64]` shapes are tested. | Reusable. |
| Residual action core | Authority, contact ramp, joint/velocity limiters, composer, and executed-action history exist in normalized 23-D coordinates. | Zero residual is bitwise nominal; partial reset, float64, and conditional CUDA paths are tested. | Reusable. |
| Virtual F/T | HEAD contains source, transform, tare, corruption, quality, and state reset. | `test_virtual_ft.py` and `test_wrist_wrench_source_contract.py` pass. | Reusable only after environment integration. |
| Contact detector | HEAD contains `ContactDetector` and `ContactDetectorOutput`; both are exported from `somaforce_cross.sensing`. | Phase 3C tests cover hysteresis, smoothing, invalid-sample hold, partial reset, dtype/device, and conditional CUDA. | Committed reusable Phase 3C API; environment integration remains absent. |
| Semantic targets | HEAD contains per-wrist/global direction and magnitude targets, stable contact aggregation, and no independent cross target; all three symbols are exported from `somaforce_cross.force`. | Tests cover all 13 direction classes, exact magnitude-helper reuse, zero/tiny contact weights, normalization, quality separation, float32/float64, and conditional CUDA. | Committed reusable Phase 3C API; clean target stays outside actor input. |
| Scaffold task metadata | `HDMITaskSpec` and artifact manifests describe action order, object/contact mapping, and artifact-specific dimensions. | Four fixed-batch parity fixtures pass at `<=1e-5`. | Metadata substrate, not a residual task adapter. |
| Golden Isaac runtime | `PretrainedHDMIIsaacRuntime` is a manifest-selected deterministic rollout class with task branches. | Four-task stored rollouts and tests exist. | Must remain golden; it is not the training env. |
| Per-environment reset | Pure buffers/sensor states support `env_ids`; golden Isaac `reset()` resets all environments and has a scalar `reference_step`. | Unit tests cover pure-buffer partial reset only. | Batched environment integration missing. |
| Mismatch resampling | Golden runtime constructor accepts scalar/global physical parameters. `VirtualFTSensor` accepts per-env parameters at construction but has no resampling method. | Stored deterministic mismatch cases are single-environment, fixed-condition rollouts. | Per-env episode resampling missing. |
| Reward/termination | Golden rollout methods compute offline metrics and two failure checks; no per-step reward tensors exist. | Stored JSON proves diagnostic behavior, not RL reward semantics. | Missing. |

### 4.1 Committed Phase 3C baseline

HEAD `d3ada580` exports the Phase 3C APIs through their package namespaces:

```text
somaforce_cross.sensing:
  ContactDetector, ContactDetectorOutput

somaforce_cross.force:
  TwoWristSemanticTargets, direction_soft_targets,
  two_wrist_semantic_targets
```

The Phase 3C acceptance evidence is source-level and tensor-contract evidence:

- exact-zero contact uses zero-wrench fallback targets;
- every positive contact weight, including float32 `1e-20` and float64
  `1e-200`, is normalized by its true positive sum;
- stable `-expm1(sum(log1p(-p)))` aggregation preserves tiny nonzero contact;
- sensor quality is not accepted by the clean target API and is not applied a
  second time;
- package-level API, forbidden-import, float32/float64, and conditional CUDA
  checks are tested;
- 92 Phase 1-3 targeted tests and 120 full-suite tests pass in this run.

The only runtime use of `IsaacWristWrenchSource` is the diagnostic script.
Neither the golden runtime nor another environment yet composes the source,
tare, virtual sensor, committed detector, token history, semantic encoder,
residual composer, and executed action history end to end. This is a Phase 4
integration gap, not an uncommitted Phase 3C blocker.

## 5. Runtime boundary audit

### 5.1 Imports and privilege

- An AST scan of all 16 current Python files under `force/`, `residual/`, and
  `sensing/` found no import containing `isaac`, `hdmi`,
  `active_adaptation`, or `rsl_rl`.
- `pretrained_hdmi.py` is dependency-free at runtime but implements the
  privileged `phase=train` teacher contract (`command`, object, and privileged
  state). It is a simulation scaffold baseline, not a deployable scaffold.
- `pretrained_hdmi_isaac.py` imports Isaac Lab by design and reads object pose,
  object contact, support contact, applied torque, and reference truth. These
  values may build scaffold input or critic/reward diagnostics but must never be
  copied into the residual actor observation.
- `scripts/export_pretrained_hdmi_scaffold.py` imports
  `active_adaptation` inside the separate export/oracle boundary. No runtime
  module does so.

### 5.2 Golden runtime reuse

`PretrainedHDMIIsaacRuntime` must not be subclassed into or converted into the
Phase 4 training environment:

- `reset()` is global, not `reset(env_ids)`;
- `reference_step` is a Python scalar in addition to batched history state;
- `step()` computes and immediately executes `a_nom`;
- it pushes `a_nom`, not executed `a_total`, into HDMI previous-action history;
- reward and done values are post-rollout Python metrics;
- mismatch parameters are constructor-wide scalar values.

The future environment may reuse immutable artifact contracts, scene-building
facts, joint-name mappings, and the frozen `PretrainedHDMIScaffold`. The golden
runtime's control loop and state ownership remain unchanged for parity.

### 5.3 Action-scaling risk

`ResidualActionComposer` returns normalized `a_total [B,23]`.
`HDMIJointPositionActionRuntime.substep_target()` later computes:

```text
joint_target = default_joint_pos + applied_normalized_action * action_scale
```

The Phase 4 action sink must therefore accept normalized `a_total`, perform
delay/smoothing in normalized coordinates, and apply `action_scale` exactly at
the joint-target conversion. Passing a scaled residual into this sink, or
scaling both `a_nom` and `a_total`, is a double-scaling failure.

### 5.4 Task-specific network risk

No task-specific residual network exists today. The four frozen artifact
policies are task-specific scaffolds with one shared 23-D contract; this is
allowed. The explicit golden runtime subclasses only validate artifact task
identity. Phase 4 must use one residual actor and one schema, with differences
contained in adapter data and reward/progress functions. A per-task residual
actor, task ID, or object-category feature in the actor is a stop condition.

## 6. Frozen Phase 4 observation contracts

All environment observation tensors are `torch.float32`, finite, on the
environment device, and batched in the first dimension. Boolean simulator
facts are converted to `0.0/1.0` only inside critic/target groups. The
environment owns named groups first; flattening uses the exact order below.

### 6.1 Environment policy bundle: total width 668

| Order | Field | Shape | Width | Frozen semantics |
| ---: | --- | --- | ---: | --- |
| 1 | `wrist_tokens` | `[B,2,16,14]` | 448 | Wrist order left/right; history oldest-to-newest; feature order per frame is normalized wrench 6, wrist twist in base-yaw frame 6, contact probability 1, sensor quality 1. |
| 2 | `proprio` | `[B,64]` | 64 | Concatenate root angular velocity in base frame `[3]`, projected gravity `[3]`, 29-D joint-position offset from default `[29]`, and 29-D joint velocity `[29]`. Joint order is `G1_FULL_JOINT_NAMES`, resolved by name. |
| 3 | `a_nom_history` | `[B,23,3]` | 69 | Normalized action coordinates; action-major, newest-to-oldest; slot 0 is current `a_nom`, older slots are prior nominal actions. Joint order is `HDMI_ACTION_JOINT_NAMES`. |
| 4 | `previous_a_total` | `[B,23]` | 23 | Previously executed normalized composed action; zero on first step after reset. |
| 5 | `z_cross` | `[B,64]` | 64 | Semantic-encoder output and the only force-semantic latent passed to the residual actor. It is derived from `wrist_tokens`, not an independent simulator observation. |

The exact width calculation is:

```text
wrist-token prefix:       2 * 16 * 14           = 448
ResidualActor boundary:   64 + 64 + (23 * 3) + 23 = 220
environment policy bundle: 448 + 220            = 668
```

The environment may expose the flat runner tensor as:

```text
policy = cat(
  wrist_tokens.reshape(B,448),
  proprio,
  a_nom_history.reshape(B,69),
  previous_a_total,
  z_cross,
)  # [B,668]
```

`[B,668]` is a structured environment/model bundle, not an approved opaque
input to an ordinary MLP. Its first `[B,448]` elements are the wrist-token
prefix consumed by the semantic encoder. The actual `ResidualActor` input is:

```text
z_cross [B,64]
+ proprio [B,64]
+ a_nom_history [B,69]
+ previous_a_total [B,23]
= residual_actor_input [B,220]
```

The required dataflow is therefore:

```text
wrist_tokens [B,2,16,14]
  -> WristTokenEncoder -> ForceSemanticHeads -> P_cross -> CrossEncoder
  -> z_cross [B,64]

z_cross + proprio + a_nom_history + previous_a_total
  -> ResidualActor [B,220] -> raw_delta [B,23]
```

The residual actor must not concatenate raw wrist tokens, `p_dir`, `p_mag`, or
`P_cross` alongside `z_cross`. A model wrapper must parse the named structure;
feeding all 668 flattened values directly to a generic MLP is a contract
violation.

No object state, object/task identity, HDMI `command[356]`, clean wrench,
simulator contact truth, root linear velocity, foot wrench, progress, mismatch,
or stability truth is present in this group. `Delta a_nom` is omitted because
it is optional in the frozen V1 plan and no current scaffold exports it.

### 6.2 Critic group: total width 845

The critic tensor starts with the identical policy tensor, followed by these
fixed named groups. Task-inapplicable mismatch slots are zero deltas from the
task's manifest nominal value; their positions never change.

| Order | Field | Shape | Width | Field order |
| ---: | --- | --- | ---: | --- |
| 1 | `policy` | `[B,668]` | 668 | Exact policy group above. |
| 2 | `object_state` | `[B,16]` | 16 | Root pose in robot base frame `[xyz,qw,qx,qy,qz]` 7; root twist in base frame `[linear,angular]` 6; mechanism `[q,qd,applied_torque]` 3, zero for rigid objects. |
| 3 | `physics_mismatch` | `[B,39]` | 39 | Hinge-axis error 3; mechanism friction 1; mechanism damping 1; handle-pose error `[xyz,rotvec]` 6; object mass 1; CoM offset 3; symmetric inertia components 6; surface friction 1; initial object pose error 6; initial stance error `[x,y,yaw]` 3; contact-target pose error 6; left/right load-share error 2. |
| 4 | `scaffold_mismatch` | `[B,2]` | 2 | Applied delay in physics substeps 1; smoothing alpha 1. |
| 5 | `sensor_mismatch` | `[B,90]` | 90 | Axis quaternion 8; scale error 12; additive bias 12; drift rate 12; drift-noise std 12; white-noise std 12; delay 2; filter alpha 2; force saturation 6; torque saturation 6; dropout probability 2; `F_scale` 2; `M_scale` 2. |
| 6 | `progress_state` | `[B,4]` | 4 | Normalized phase, normalized progress, progress delta, success flag. |
| 7 | `contact_state` | `[B,17]` | 17 | Clean two-wrist wrench 12; simulator contact truth per wrist 2; reference contact intent per wrist 2; support-contact count 1. |
| 8 | `stability_state` | `[B,9]` | 9 | Root height 1; projected gravity 3; root linear velocity 3; left/right foot contact flags 2. |

The critic width is ratified by the field arithmetic, not by an implemented
critic network:

```text
policy                              668
object_state             7 + 6 + 3 = 16
physics_mismatch  3+1+1+6+1+3+6+1+6+3+6+2 = 39
scaffold_mismatch                    2
sensor_mismatch 8+12+12+12+12+12+2+2+6+6+2+2+2 = 90
progress_state                       4
contact_state             12+2+2+1 = 17
stability_state              1+3+3+2 = 9
total                    668+16+39+2+90+4+17+9 = 845
```

The 39-D physics mismatch fields are in physical units before a separately
versioned critic normalizer, except orientation errors, which use radians.
Mass and inertia slots record applied values relative to manifest nominal
metadata. Numeric normalizer statistics are not currently defined and are a
Phase 4B approval gate.

The critic schema intentionally has no task ID. Task adapters map their state
to common progress/contact/object semantics, and no residual network is selected
by task. The critic may use privileged data; its tensor must be stored and fed
through a separate observation key so it cannot be sliced from the actor input
by convention alone.

### 6.3 Semantic-target group: total width 31

| Order | Field | Shape | Width | Frozen semantics |
| ---: | --- | --- | ---: | --- |
| 1 | `clean_wrench` | `[B,2,6]` | 12 | Calibrated clean physical F/T in base-yaw sensor-origin frame; target/diagnostic only. |
| 2 | `p_dir_target` | `[B,13]` | 13 | Global two-wrist direction distribution in the fixed signed-axis plus neutral order. |
| 3 | `p_mag_target` | `[B,5]` | 5 | Global five-bin magnitude distribution with `sigma_mag=0.15`. |
| 4 | `semantic_loss_weight` | `[B,1]` | 1 | Aggregate contact probability; quality is not multiplied here a second time. |

The target width is `12 + 13 + 5 + 1 = 31`. The committed Phase 3C API
provides `global_direction [B,13]`, `global_magnitude [B,5]`, and
`g_contact [B,1]`; the environment must preserve the clean wrench separately
to assemble the full training-only group.

There is no `P_cross_target` and no cross loss. Clean wrench and all target
fields are excluded from `policy`. The committed target implementation is
tensor-contract evidence for these 19 derived widths, but no Phase 4
environment currently assembles `[B,31]`.

## 7. Reset and step semantics

### 7.1 Per-environment reset order

For exactly the provided unique `env_ids`, and without mutating other rows:

1. Save terminal observations/metrics, `terminated`, and `time_outs` before any
   state mutation.
2. Select the already-registered task/reference and sample that episode's
   physical, scaffold, and sensor mismatch parameters.
3. Apply physics properties, then reset robot, object/mechanism, reference
   phase, and task-adapter progress state.
4. Reset frozen scaffold state/history: fill reset robot-state histories,
   zero its previous actions/adaptation state, and set per-env reference step
   and phase to zero. The policy remains frozen and in eval mode.
5. Reset wrench source cache, tare state, virtual-sensor drift/delay/filter
   state, contact-detector hysteresis/smoothing, contact-gain ramp, wrist token
   history, nominal-action history, and executed-action history.
6. Synchronize one physics sample, construct the initial deployable wrist frame,
   and broadcast that frame across all 16 wrist-history slots. If no valid
   calibrated sample exists, zero-fill and set sensor quality to zero.
7. Build the first scaffold observation and current `a_nom`; place it in
   `a_nom_history[...,0]` while older nominal slots and `previous_a_total`
   remain zero.
8. Zero episode length, reward accumulators, success/failure latches, and
   termination-reason codes; then build policy, critic, and semantic-target
   observations.

### 7.2 Control-step order

```text
current state -> scaffold observation -> frozen a_nom
current wrist sample -> deployable token/history -> p_dir,p_mag -> z_cross
policy observation -> raw residual
raw residual + a_nom -> authority/contact/safety composition -> a_total
a_total -> normalized delay/smoothing -> one-time joint-position scaling
physics substeps -> state/history update -> reward/done/next observations
```

Only executed `a_total` is pushed for the next step's
`previous_a_total`. The frozen scaffold's own internal previous-action field
continues to follow its artifact contract and must not be confused with the
residual actor's executed-action field.

### 7.3 Resampling boundary

- Task assignment, physical mismatch, scaffold delay/smoothing, sensor
  corruption parameters, and curriculum stage are sampled per environment at
  episode reset only.
- A task never switches inside an episode.
- Drift state, stochastic drift/noise draws, and dropout events evolve per
  control step; the parameter distributions do not change mid-episode.
- Contact hysteresis/smoothing and authority attack/release are state updates,
  not parameter resampling.
- Deterministic parity mode fixes nominal physical/scaffold parameters,
  disables stochastic sensor corruption, sets residual authority or output to
  zero, and uses the same reference/reset seed as the golden runtime.
- Current source has no partial parameter-resampling API for `VirtualFTSensor`
  and no per-env physical-property sampler. Phase 4B must add these without
  changing the pure corruption formulas or golden runtime.
- Reset-time ownership and reproducibility are approved here; numeric mismatch
  distributions and curriculum sampling probabilities are not. Phase 4B2 may
  implement explicit caller-supplied parameter plumbing and selective-row
  resampling, but must not embed guessed ranges or probabilities.

## 8. Reward, progress, contact, stability, and done contract

Authoritative V1 documents define required reward families but not an
implementation-ready numeric reward contract. This audit freezes the adapter
outputs and term names; reward weights, clipping, scales, and normalizers
require owner approval.

Every task adapter returns per-env float32 tensors:

```text
progress                 [B,1]  normalized task progress potential
progress_delta           [B,1]  current minus previous potential
success                  [B,1]  task-specific success diagnostic
expected_contact         [B,2]  reference contact intent, left/right
contact_truth            [B,2]  simulator-only actual contact
stability_margin         [B,1]  task-specific fall margin
```

The shared reward builder emits separate, logged `[B]` terms:

```text
progress
contact
stability
force_efficiency
nominal_retention
action_safety
```

No task adapter returns a final weighted scalar on its own. This prevents
task-specific residual objectives from hiding behind adapters. Reward weights,
normalization scales, and caps must live in one shared versioned config and be
reported per term and per task.

The V1 design records high-level C1-C3 nominal/mismatch mixture percentages and
qualitative sensor curricula. It does not define a complete executable table
covering task balance, mismatch-family selection, parameter distributions, or
stage transitions. This ratification therefore does not approve curriculum
sampling probabilities or mismatch distributions from those summaries.

The following numeric contracts remain explicitly unapproved:

```text
reward weights
reward clipping and per-term scales
policy-observation and critic normalizers
physical/scaffold/sensor mismatch distributions
task/stage/mismatch curriculum sampling probabilities
```

Phase 4B1 is pure structured integration and must accept explicit contracts or
leave numeric behavior unconfigured; it must not guess any value above.

Contact used for reward/critic may be simulator truth. Actor contact remains
the independent detector probability from the deployable virtual sensor. Force
efficiency uses clean wrench only as a training signal and metric; it never
enters policy observation.

Done semantics are fixed as:

```text
terminated = nonfinite_state OR task_adapter.failure
time_outs  = max_episode_steps OR reference_exhausted
done       = terminated OR time_outs
```

Current evidenced failure thresholds are root height `<0.45 m` for door and
push-box, and `<0.25 m` for suitcase/large-box. These remain adapter data, not
global hard-coded logic. Success is logged separately and does not silently
become `terminated`. `time_outs` must be returned separately so a later PPO
runner can bootstrap truncated episodes correctly.

## 9. Four-task adapter contract

Common interface required from every adapter:

```text
task_spec / artifact_contract
reset(env_ids, sampled_parameters)
build_scaffold_observation() -> HDMIObservationBatch
build_object_state() -> fixed critic object_state
progress_signals() -> progress, progress_delta, success
contact_signals() -> expected_contact, contact_truth
stability_signals() -> stability_margin, failure
reward_diagnostics() -> fixed named diagnostics only
```

All joint/reference mappings are name-based. All adapters emit two wrist token
slots even when only one wrist is the primary contact. Adapter selection changes
data/physics/progress semantics only, never the residual network.

| Task | Actual artifact/runtime differences | Frozen adapter-specific data |
| --- | --- | --- |
| `push_door_hand` | Articulated door; 29 robot reference joints plus door joint; object observation 7-D; one right-wrist contact target; no nominal mass; historical progress is signed door-joint displacement. | Door hinge axis/sign, target angle, friction/damping, handle pose/contact target, one expected primary contact mapped into the fixed right slot, 0.45 m fall threshold. |
| `push_box` | Rigid box; 25 reference joints mapped by name to the shared 23 actions; object observation 10-D; two wrist targets; nominal mass 8 kg in current runtime; historical progress is projection along reference XY displacement. | Reference XY direction/distance, mass/friction/CoM/inertia/initial-pose parameters, two contact targets, path/tracking diagnostics, 0.45 m fall threshold. |
| `move_suitcase` | Rigid object; 29 reference joints; two contacts; wrist-yaw reset overrides -0.4/+0.4 rad; nominal mass 1.5 kg and documented training range 1.2-1.8 kg; 472-frame reference. | Reference object path/orientation, lift/carry/set-down phase masks, mass/friction/CoM/inertia/load-share parameters, two contact targets, 0.1 m lift and 0.25 m set-down diagnostic thresholds, 0.25 m fall threshold. |
| `move_largebox` | Rigid URDF object; 29 reference joints; two asymmetric contact targets; same wrist-yaw reset overrides; nominal mass 1.0 kg and range 0.8-1.2 kg; 199-frame reference. | Same payload interface as suitcase with artifact-specific object/contact/path data; no task-specific network; 0.25 m fall threshold. |

The historical condition matrix proves meaningful deterministic mismatch cases
for friction/damping, mass/friction, and payload mass. It does not prove
per-env randomization distributions, reward correctness, or statistical
robustness: each stored case is one deterministic rollout.

## 10. Ratified Phase 4B internal execution boundary

The formal V1 implementation phases remain Phase 1 through Phase 7 as written
in `docs/v1_force_residual_implementation_plan.md`. The labels below are only
ordered execution subdivisions inside formal Phase 4; they do not add phases,
advance Phase 5, or authorize PPO/training.

### Phase 4B1: pure PyTorch structured observation/sensing integration

- Implement named shape/dtype/device validation and the structured
  `wrist_tokens -> semantic encoder -> z_cross` flow without Isaac, HDMI,
  `active_adaptation`, RSL-RL, or PPO imports.
- Expose the `[B,448]` wrist prefix and `[B,220]` ResidualActor boundary without
  treating `[B,668]` as one opaque MLP input.
- Assemble and test the `[B,668]`, `[B,845]`, and `[B,31]` schemas using
  explicit caller-provided tensors. Preserve clean/privileged separation.
- Do not implement or guess reward values, normalizers, mismatch
  distributions, or curriculum probabilities.

### Phase 4B2: per-environment reset and parameter resampling

- Implement selective `env_ids` reset and state ownership for scaffold,
  sensing, detector, histories, authority, reference phase, and adapter state.
- Add selected-row parameter application/resampling interfaces with explicit
  caller-supplied parameters and deterministic seeding.
- Numeric sampling distributions remain an approval gate; no default ranges or
  probabilities may be inferred from stored deterministic rollout cases.

### Phase 4B3: door-only 64-environment C0 smoke

- Integrate only `push_door_hand` for the first Isaac `DirectRLEnv` smoke.
- Use C0 zero residual, deterministic nominal parameters, and matched seed,
  physics, reference, delay, and smoothing against the unchanged golden
  runtime.
- Validate selective reset, exact structured schemas, `[B,23]` normalized
  action, one-time scaling, `terminated`/`time_outs`, and actor-leak checks.

### Phase 4B4: four-task adapter and parity

- Add `push_box`, `move_suitcase`, and `move_largebox` through the same schema
  and residual network boundary as the door adapter.
- Keep joint/reference mapping name-based and task differences in manifest
  data, geometry, progress, contact, payload, and door semantics.
- Require per-task C0 zero-residual parity; no task ID, object identity, or
  task-specific residual actor/critic is permitted.

### Phase 4B5: mismatch, reward, termination, and curriculum

- Add per-env mismatch sampling, shared named reward terms, vectorized
  termination/timeout logic, curriculum state, and boundary tests.
- This subdivision cannot start its numeric behavior until the owner approves
  reward weights, clipping/scales, policy/critic normalizers, mismatch
  distributions, and curriculum sampling probabilities in a versioned config.

Each subdivision must pass its own tests and preserve preceding gates before
the next begins. Failure does not authorize skipping forward. Phase 4B remains
unimplemented at this ratification commit.

### 10.1 Future implementation whitelist recommendation

A later implementation Goal should authorize only these new or narrowly
extended paths:

```text
somaforce_cross/envs/__init__.py
somaforce_cross/envs/residual_env.py
somaforce_cross/envs/residual_env_cfg.py
somaforce_cross/envs/observations.py
somaforce_cross/envs/reset.py
somaforce_cross/envs/mismatch.py
somaforce_cross/envs/rewards.py
somaforce_cross/envs/terminations.py
somaforce_cross/envs/task_adapter.py
somaforce_cross/envs/task_adapters/*.py
somaforce_cross/envs/action_history.py
tests/test_phase4_observation_contract.py
tests/test_phase4_reset_contract.py
tests/test_phase4_reward_termination.py
tests/test_phase4_task_adapters.py
tests/test_phase4_import_boundaries.py
tests/test_phase4_zero_residual_parity.py
```

Narrow conditional extension for Phase 4B2 only:

```text
somaforce_cross/sensing/virtual_ft.py  # selected-row parameter application only
```

Committed Phase 3C APIs are dependencies, not unfinished Phase 4 candidates.
Changes to `force/semantic_targets.py`, `sensing/contact_detector.py`, or their
package exports require separate justification and focused regression tests.

Explicitly outside the Phase 4B whitelist:

```text
somaforce_cross/scaffold/pretrained_hdmi_isaac.py  # golden runtime
artifacts/scaffolds/**
PPO/RSL-RL/training code
task-specific residual actor/critic networks
open_foldchair-sit, topple_wood_board_and_cross, roll_ball-hand
```

## 11. Blockers and risks

1. **Phase 4 implementation absent:** no vectorized env or learning-facing
   observation/reward/done API exists.
2. **Committed Phase 3 integration absent:** all sensing and semantic APIs are
   in HEAD and publicly exported, but no task runtime composes them end to end.
3. **Privileged scaffold deployment gate open:** all four current artifacts are
   `phase=train`, `deployable=false` privileged teacher baselines. This does not
   authorize privileged residual actor inputs.
4. **Door manifest incomplete:** the door manifest still relies on code
   fallbacks for task, task contract, observation dimensions, and network
   contract. Do not mutate the checksummed V1 artifact in place.
5. **No per-env resampling:** current Isaac mismatch values are constructor-wide;
   current sensor parameters cannot be resampled on selected rows.
6. **Numeric contracts absent:** reward weights/clipping/scales,
   policy-observation and critic normalizers, mismatch distributions, and
   executable curriculum sampling probabilities require owner approval.
7. **Zero-residual closed-loop gate absent for Phase 4:** fixed-input scaffold
   parity and golden zero-hook evidence exist, but no 64-env composed pipeline
   has matched the golden runtime.
8. **Action-history split is easy to violate:** scaffold internal history and
   residual executed `a_total` history have different ownership and must not be
   aliased.
9. **Double scaling risk:** the only physical scaling point must remain after
   normalized residual composition and normalized delay/smoothing.

## 12. Ratified Phase 4B acceptance gates

The gates follow the internal execution order and do not authorize the next
formal phase:

- **4B1:** exact `[B,448]` semantic prefix, `[B,220]` actor boundary,
  `[B,668]` policy bundle, `[B,845]` critic, and `[B,31]` target validation;
  actor-leak and structured-dataflow tests; pure-PyTorch forbidden-import gate.
- **4B2:** selective reset changes only `env_ids`; all histories,
  detector/sensor state, task progress, reference phase, and explicitly
  supplied per-env parameters follow the reset contract under a fixed seed.
- **4B3:** 64-environment door-only headless C0 smoke; zero residual reproduces
  the door golden runtime's normalized action and physical joint target under
  matched seed/physics/reference/delay/smoothing; action scaling occurs once.
- **4B4:** the other three adapters pass the same C0 parity and schema checks;
  all four share one residual network boundary with no task/object identity in
  policy input.
- **4B5:** after numeric approval, mismatch/reward/termination/curriculum
  tensors are vectorized, finite, reproducible, separately logged, and tested
  at each task boundary; `terminated` and `time_outs` remain separate.
- **All subdivisions:** AST/import tests prove no HDMI/`active_adaptation`
  import or HDMI checkout path in the residual runtime, and checksummed
  artifacts plus `PretrainedHDMIIsaacRuntime` remain unchanged.
- Current targeted and full test suites remain green, followed by
  `git diff --check`.

## 13. Commands and results

Git start checks:

```bash
git branch --show-current
git rev-parse HEAD
git status --short --branch
git show-ref refs/heads/v1 refs/remotes/origin/v1
git ls-remote origin refs/heads/v1
```

Search and static checks included `rg --files`, repository-wide symbol/import
searches, structured JSON parsing, and an AST import scan over `force/`,
`residual/`, and `sensing/`.

Targeted tests:

```bash
pytest -q \
  tests/test_force_core.py \
  tests/test_residual_action.py \
  tests/test_virtual_ft.py \
  tests/test_wrist_wrench_source_contract.py \
  tests/test_contact_semantics.py
```

Result: `92 passed in 6.18s`.

Full tests:

```bash
pytest -q
```

Result: `120 passed in 7.52s`.

Additional results:

```text
AST import scan: 16 files, forbidden_import_hits=[]
python -m compileall -q somaforce_cross tests scripts: PASS
Markdown trailing-whitespace check: PASS
untracked report diff whitespace check: PASS
git diff --cached --check: PASS before commit
```

No training, Isaac rollout, download, scaffold migration, Phase 4B
implementation, or push was run. The only authorized mutation is this report
and its report-only local commit.

## 14. Modified files

```text
entry worktree file and only committed path:
  docs/phase4a_contract_audit.md
```

Stop after the report-only local commit. Phase 4B is not implemented and
requires a separately authorized implementation Goal that respects the 4B1-4B5
ordering and unresolved numeric gates above.
