# Phase 6 Independent Four-Task Learning Amendment

Date: 2026-08-19

Status: **admitted research-definition amendment; production implementation,
its Admission, and new training are pending**.

## 1. Superseded Boundary

This amendment supersedes only the Phase 6 residual-policy initialization and
checkpoint-lineage boundary in the frozen V1 plans.

The previous admitted implementation loaded the complete Phase 5
`ResidualActorCritic` state from the single-task `push_door_hand` checkpoint,
created a fresh optimizer, and then continued with four-task distributed PPO.
That experiment is a single-task warm-start followed by multi-task fine-tuning.
It is not independent four-task learning.

The new research definition is:

> Phase 6 must initialize a new residual actor-critic without loading any
> Phase 5 policy weights. Every learnable residual-policy parameter must be
> learned only from batches sampled by the four Phase 6 task ranks.

Phase 5 may remain the implementation source for the admitted architecture,
PPO equations, storage interface, and semantic losses. Reusing code is allowed;
loading Phase 5 policy, optimizer, normalizer, RNG, or checkpoint state is not.

The four frozen HDMI nominal scaffolds remain pretrained inputs. "Independent"
applies to the SomaForce-Cross residual actor-critic, not to the nominal task
scaffolds.

## 2. Scientific Claim

An admitted independent Phase 6 run may claim:

```text
one newly initialized shared residual actor-critic
+ four simultaneous task-owned rollout streams
+ synchronized equal-weight multi-task gradients
= a residual policy learned only from Phase 6 four-task data
```

It may not claim from-scratch whole-body manipulation because each task still
uses a frozen pretrained nominal scaffold. It may not claim task-independent
data collection because each rank owns a named training task.

## 3. Non-Negotiable Independence Rules

For a new Phase 6 run:

1. No Phase 5 checkpoint path or SHA may appear in the active Phase 6 training,
   acceptance, roster, initialization, or checkpoint contracts.
2. The first training process must not open or deserialize a Phase 5 checkpoint.
3. All 916,881 learnable parameters of the current `ResidualActorCritic` must
   be newly initialized: wrist encoder, semantic heads, cross encoder, residual
   actor, privileged critic, and `log_std`.
4. The optimizer must be constructed after distributed policy initialization,
   with empty state and optimizer step zero.
5. Every rank must begin the first rollout with a bitwise-identical policy.
6. A Phase 5 checkpoint, historical Phase 6 v1 checkpoint, rebound endpoint,
   or diagnostic checkpoint must be rejected as a bootstrap or resume input.
7. Only a checkpoint emitted by the same independent Phase 6 contract may be
   used to resume an interrupted independent run.
8. Historical warm-start files and evidence remain immutable and readable as a
   separate baseline lineage.

## 4. Versioning Strategy

Do not edit the admitted v1 JSON files in place. Add a parallel versioned
contract set:

```text
configs/phase6_joint_training_v2.json
configs/phase6_task_roster_v2.json
configs/phase6_learning_acceptance_v2.json
```

Required versions:

```text
phase6_joint_training_v2
phase6_task_roster_v2
phase6_learning_acceptance_v2
phase6_learning_checkpoint_v2
phase6_independent_initialization_v1
```

The v1 loaders and files must remain available for auditing historical v1
evidence. Active production training must explicitly require v2. Version
dispatch must be fail-closed; a v1 checkpoint cannot be silently converted,
rebound, or accepted by a v2 training command.

The roster task content remains the same four tasks, but its canonical binding
must reference the new Phase 6 v2 canonical SHA. This requires a v2 roster even
though task semantics are unchanged.

## 5. Deterministic Initialization Contract

The selected initialization candidate is:

```json
{
  "actor_output_zero_init": true,
  "contract_version": "phase6_independent_initialization_v1",
  "external_checkpoint": null,
  "init_noise_std": 1.0,
  "parameter_count": 916881,
  "policy_constructor": "ResidualActorCritic",
  "rank_owner": 0,
  "seed": 20260806,
  "source": "random",
  "synchronization": "rank0_state_dict_broadcast"
}
```

Initialization order is normative:

1. Initialize the NCCL process group.
2. On every rank, construct the same CPU model structure. Rank 0 performs the
   authoritative initialization inside an isolated CPU RNG scope using seed
   `20260806`.
3. Use the current PyTorch module initialization for all semantic, hidden actor,
   and critic layers.
4. Set only the final residual-actor layer weight and bias to exact zero. This
   gives an exact zero deterministic residual mean at iteration zero while
   preserving stochastic exploration through `init_noise_std=1.0`.
5. Move the model structure to each local CUDA device and broadcast every
   parameter and persistent buffer from rank 0 in a fixed state-dict order.
6. Compute the policy state SHA256 independently on all ranks and reject any
   mismatch before environment construction or rollout collection.
7. Construct Adam independently on each rank only after policy synchronization.
   Require identical optimizer hashes, empty state, and step zero.
8. After initialization evidence is persisted, seed the rank-owned training RNG
   as `20260806 + rank`. Environment sampling keeps its separately declared
   per-rank/per-environment seed ownership.

At current commit `890aa4a15eb756f5733144e0b106ceb3070fb05b`, PyTorch
`2.7.0+cu128`, and the current architecture, the candidate initialization has:

```text
parameter_count=916881
candidate_initial_policy_sha256=cc3116f729f05b536084a50323a613c6dbdb73149424d314864196e402bad01d
historical_phase5_policy_state_sha256=a8f33a7e78da1d9410474172e046238d9c29ba9da1a060abfd02e749a5a513d9
```

The implementation must recompute and freeze the final initial-policy hash
after all allowed source changes. A mismatch is a contract failure, not a reason
to update the expected hash automatically.

## 6. Four-Task Update Semantics

Phase 6 retains the admitted equal-task distributed update:

```text
world_size = 4
num_envs_per_rank = 64
num_steps_per_env = 32
transitions_per_rank_per_iteration = 2048
global_transitions_per_iteration = 8192
task transition fraction = 0.25 exactly
PPO epochs = 3
mini-batches = 8
optimizer steps per iteration = 24
```

Each rank collects a homogeneous batch from one task. After every combined-loss
backward pass, all policy gradients are averaged across the four ranks before
clipping and optimizer step. Because initial parameters and optimizer states
are identical and every subsequent gradient is synchronized, all ranks must
retain identical policy/optimizer hashes after every iteration.

No Phase 5 transitions, replay, semantic-probe samples, normalization state, or
optimizer moments enter this update.

## 7. Checkpoint And Provenance Schema

`phase6_learning_checkpoint_v2` must replace the v1
`phase5_checkpoint_sha256` lineage field with an exact initialization record:

```text
initialization.contract_version
initialization.source
initialization.seed
initialization.parameter_count
initialization.initial_policy_sha256
initialization.actor_output_zero_init
initialization.rank0_broadcast_verified
initialization.optimizer_initial_sha256
initialization.optimizer_initial_step
initialization.external_checkpoint
```

Required values include:

```text
source=random
external_checkpoint=null
rank0_broadcast_verified=true
optimizer_initial_step=0
```

The root plan, first rank results, first pre-evaluation checkpoint, every resume
checkpoint, `latest.json`, segment summary, and final acceptance inputs must
carry or bind the same initialization record. Protected-state checks must reject
its mutation.

Resume rules:

- iteration zero accepts no `--resume` input;
- iteration greater than zero requires a v2 Phase 6 checkpoint;
- v1 checkpoint version, Phase 5 checkpoint version, unknown initialization
  version, missing initialization evidence, or initial-policy hash mismatch is
  rejected before Isaac environment construction;
- source rebind may update only the declared source manifest and may never
  change initialization lineage or learned state;
- no historical v1 rebound endpoint may seed a v2 run.

## 8. Contract Changes

### 8.1 `phase6_joint_training_v2`

Remove these v1 bindings:

```text
bindings.phase5
bindings.phase5_initial_checkpoint
bindings.phase5_learning_acceptance
```

Retain the Phase 4 numeric contract and roster binding. Add the exact
`initialization` object from section 5. Keep PPO, dimensions, scheduler,
curriculum, metrics, runtime, and checkpoint arithmetic unchanged in the first
independent experiment so initialization is the only intended scientific
change.

### 8.2 `phase6_task_roster_v2`

Keep the exact four tasks and equal scheduling semantics. Replace only its
Phase 6 canonical binding and contract version. Task IDs remain excluded from
the residual actor.

### 8.3 `phase6_learning_acceptance_v2`

Replace `phase5_initial_checkpoint_sha256` with bindings to:

```text
phase6_v2 raw/canonical SHA
roster_v2 raw/canonical SHA
initialization contract version
final initial-policy SHA
```

Add hard gates for no external checkpoint, verified rank-0 broadcast, initial
optimizer step zero, and exact initial-policy hash. Preserve the current
learning and final-acceptance criteria for the first run.

## 9. Training Budget And Curriculum

The first independent run keeps the current Phase 6 budget for direct
comparability:

| Profile | Iterations | Global transitions | Per-task transitions | Optimizer steps |
| --- | ---: | ---: | ---: | ---: |
| pilot | 31 | 253,952 | 63,488 | 744 |
| main | 2,442 | 20,004,864 | 5,001,216 | 58,608 |

The pilot and main profile are one v2 lineage: the main run may resume only from
the admitted v2 pilot endpoint, and its total iteration accounting includes the
pilot iterations.

Start at C1. Keep existing C1-C3 authority values, curriculum evaluation
cadence, nominal fractions, promotion rules, and rollback rules unchanged for
the first independent run. The authority diagnostic recorded in
[`phase6_authority_checkpoint_drift_diagnostic_20260818.md`](phase6_authority_checkpoint_drift_diagnostic_20260818.md)
supports a later authority amendment, but changing initialization and authority
in one run would confound causal interpretation.

The historical warm-start baseline received an additional 5,001,216 Phase 5
single-task transitions before its Phase 6 budget. Comparisons must report both
Phase 6 transitions and total residual-policy training transitions; they must
not present the warm-start and independent runs as equal total-compute runs.

## 10. Source Design

### `somaforce_cross/learning/joint_runner.py`

- Add version-aware v1/v2 contract validation.
- Keep `load_phase5_policy_only` available only for explicit historical v1
  audit paths; the v2 training path must not call it.
- Add a pure policy-initialization helper and a distributed synchronization
  helper with exact schema/hash validation.
- Add v2 checkpoint payload and restore validation.
- Preserve strict v1 checkpoint reading for historical evidence without
  allowing v1-to-v2 resume.

### `scripts/train_phase6.py`

- Point active production defaults to v2 contracts.
- Split policy construction from optimizer/algorithm construction.
- Initialize and broadcast the v2 policy after process-group startup and before
  optimizer or environment construction.
- Persist initialization evidence before the first rollout.
- Remove Phase 5 checkpoint path resolution and policy loading from the v2
  initial branch.
- Keep the resume branch strictly v2 once independent production is selected.
- Verify rank policy and optimizer hashes at initialization and after every
  iteration.
- Fail before `AppLauncher` environment construction where the check can be
  performed without Isaac.

The class names `SemanticPPO`, `ResidualActorCritic`, and `Phase5Runner` may be
reused. Their names do not create weight lineage. A broad rename is outside this
amendment unless needed to eliminate a real contract ambiguity.

## 11. Test Design

Pure CPU/static gates must prove:

1. v2 schemas reject every Phase 5 checkpoint/path/SHA field.
2. v1 files remain byte-identical and their historical loaders still work.
3. seed `20260806` produces the exact final admitted initialization hash.
4. a different seed changes the hash.
5. the initial deterministic actor mean is exactly zero for finite valid input.
6. semantic and critic parameters are finite, nonempty, and not all zero.
7. the optimizer is created after policy initialization with empty state and
   step zero.
8. a four-process CPU Gloo substitute produces identical post-broadcast policy
   hashes and identical initial optimizer hashes.
9. deliberately changing one rank before verification is rejected.
10. one synchronized CPU substitute update leaves all rank policy and optimizer
    hashes equal.
11. v2 rejects Phase 5, Phase 6 v1, and diagnostic/rebound checkpoints as
    bootstrap or resume inputs.
12. v2 checkpoints round-trip initialization provenance exactly.
13. source AST proves the v2 initial branch has no call to
    `load_phase5_policy_only` or `torch.load` of an external initialization.
14. exact 25% task transition accounting and actor privilege boundaries remain
    unchanged.

Collection provenance, scoped Ruff/format, Phase 6 selectors, focused
selectors, and full pytest remain Admission gates. Test counts are commit-bound
and must be collected from the final worktree rather than copied from this
document.

## 12. Runtime Gate Sequence

Each runtime step requires a new non-overwriting RUN_ID and persisted command,
plan, per-rank result/wrapper/log, exit/signal/timeout, positive `VmHWM`, close
markers, initialization hashes, and final summary.

1. **Initialization-only 4-rank probe:** no Isaac environment and no optimizer
   step; verify no Phase 5 file access, rank-0 broadcast, exact initial hash,
   zero actor mean, and empty optimizer.
2. **Four-GPU 2-env/1-iteration smoke:** one task per rank; verify 256 global
   transitions, 64 per task, 24 synchronized optimizer steps, and identical
   final rank hashes.
3. **Four-GPU 64-env/2-iteration smoke:** verify 16,384 global transitions,
   4,096 per task, 48 optimizer steps, finite PPO/semantic metrics, and clean
   lifecycle closure.
4. **Four-GPU 64-env/31-iteration pilot:** verify 253,952 global transitions,
   exact balance, first evaluation crossing, nominal retention, and independent
   checkpoint v2 provenance.
5. **Main run:** only after pilot evidence and Admission; resume the v2 pilot
   lineage and run to iteration 2,442.

Stop immediately if any rank starts from a different policy hash, optimizer
step is nonzero, a Phase 5 path is opened, task balance differs from 25%, rank
hashes diverge after an update, evidence is incomplete, or nominal retention
fails the existing gate. Do not proceed to a longer runtime after a failed
earlier gate.

## 13. Historical And Comparative Reporting

The completed warm-start run remains immutable:

```text
outputs/phase6_learning_acceptance/phase6-learning-main-4gpu-64env-2442iter-20260812_185544
```

It must be labeled `phase5_warm_start`. The new run must be labeled
`phase6_independent_random_init`. Results should report:

- Phase 6 and total lineage transitions separately;
- per-task success, reward, nominal retention, force, force-rate, stability,
  residual group norms, semantic metrics, and task shares;
- curriculum stage history;
- checkpoint drift under fixed C1 evaluation;
- initialization identity and absence/presence of external policy weights.

The independent run must not overwrite, resume, source-rebind, or repair the
historical warm-start run.

## 14. Implementation Boundaries

Recommended repository implementation allowlist:

```text
configs/phase6_joint_training_v2.json
configs/phase6_task_roster_v2.json
configs/phase6_learning_acceptance_v2.json
somaforce_cross/learning/joint_runner.py
scripts/train_phase6.py
tests/test_phase6_joint_training.py
docs/phase6_independent_four_task_learning_amendment.md
docs/v1_force_residual_implementation_plan.md
```

Recommended new external job scripts, without modifying historical v1 scripts:

```text
/inspire/hdd/global_user/liumengfan-253108110079/lmf-workspace/training-job-scripts/somaforce_cross/phase6_joint_training/probe_independent_init_4gpu.sh
/inspire/hdd/global_user/liumengfan-253108110079/lmf-workspace/training-job-scripts/somaforce_cross/phase6_joint_training/smoke_independent_4gpu_2env_1iter.sh
/inspire/hdd/global_user/liumengfan-253108110079/lmf-workspace/training-job-scripts/somaforce_cross/phase6_joint_training/smoke_independent_4gpu_64env_2iter.sh
/inspire/hdd/global_user/liumengfan-253108110079/lmf-workspace/training-job-scripts/somaforce_cross/phase6_joint_training/pilot_independent_4gpu_64env_31iter.sh
/inspire/hdd/global_user/liumengfan-253108110079/lmf-workspace/training-job-scripts/somaforce_cross/phase6_joint_training/train_independent_4gpu_64env_2442iter.sh
```

Do not modify Phase 5 contracts/checkpoints, v1 Phase 6 contracts, historical
job scripts, scaffold artifacts, diagnostic RUN_DIRs, or old checkpoints. Do
not enter Phase 7 or Phase 8.

## 15. Authorization Boundary

Admission of this document freezes the owner-defined research requirement and
the proposed implementation boundary. It does not authorize source
implementation, Admission of that implementation, distributed execution, PPO,
training, or checkpoint generation. Those actions require an exact-path
implementation request followed by implementation Admission and explicit
runtime submission.
