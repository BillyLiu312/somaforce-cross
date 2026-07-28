# Phase 4B5A Numeric Contract Audit

Date: 2026-07-28

Status: **read-only owner approval package; Phase 4B5B is not authorized**.

## 1. Scope and Evidence Labels

Audited baseline:

```
branch: v1
HEAD: caf09bfdaaa50004bead2750b9da495a340adffa
origin/v1: caf09bfdaaa50004bead2750b9da495a340adffa
entry worktree: clean
```

This package covers `push_door_hand`, `push_box`, `move_suitcase`, and
`move_largebox`. It implements nothing and does not infer a training
distribution from historical `light`, `heavy`, or `stress` rollouts.

| Level | Meaning | May become a training value without explicit owner approval? |
| --- | --- | --- |
| `FROZEN_CONTRACT` | Committed interface, safety, or C0 parity contract. | No. |
| `SOURCE_NOMINAL` | Frozen nominal-scaffold source/readback fact. | No. |
| `DIAGNOSTIC_CANDIDATE` | Historical diagnostic condition or V1 design candidate. | No. |
| `MISSING` | No current numerical training contract. | No. |

Approval must record the contract version, source commit, canonical payload
SHA256, owner, and date. Missing, `null`, or `pending_owner_approval` values
must cause configuration rejection, never fallback selection.

## 2. Approved Facts and Smoke-Only Numbers

The table records current facts only. None is a Phase 4B5 reward, distribution,
curriculum, normalizer, PPO, or training default.

| Fact | Current value | Evidence | Boundary |
| --- | --- | --- | --- |
| Action contract | `a_nom`, residual, `a_total` are normalized `[B,23]`; one physical joint-position scale after sum/clip. | `FROZEN_CONTRACT`; `AGENTS.md`, scaffold rules, parity tests. | Coordinates/safety only. |
| C0 residual/reward | Authority is bitwise zero; nonzero raw probe yields `a_total == a_nom`; `smoke_reward=0.0`. | `FROZEN_CONTRACT`; env/cfg and C0 tests. | Not a task reward or learned behavior. |
| C0 time | physics `0.005 s`, control `0.02 s`, decimation `4`, smoke `6` steps, profile episode `8` steps. | `FROZEN_CONTRACT`; cfg. | Six/eight steps are not train horizons. |
| C0 sensor | identity axis, zero corruption, delay/filter `0`, no dropout, `F_scale=M_scale=1`, one tare sample; profile detector/ramp values. | `FROZEN_CONTRACT` for smoke. | Not sensor training calibration. |
| C0 scaffold | `[delay, alpha]=[4,0.9]`. | `SOURCE_NOMINAL`; frozen runtime path. | Not a distribution. |
| Rigid C0 rows | all mismatch entries zero except box mass `8 kg`, suitcase `1.5 kg`, largebox `1 kg`, applied PhysX inertia, friction `0.5`. | `SOURCE_NOMINAL`; 4B4A section 8. | Nominal anchor only. |
| Signals | adapters expose progress, delta, success, stability, failure, exhaustion; common done is nonfinite/failure or timeout/exhaustion. | `FROZEN_CONTRACT`; adapters. | Does not assign reward signs/weights/latches. |
| Actor boundary | only wrist tokens, proprioception, nominal history, previous executed action, and derived `z_cross` (220-D). | `FROZEN_CONTRACT`; observations/import tests. | Privileged data remains outside actor. |

The V1 plan's C1--C3 authority/mixture and PPO values are
`DIAGNOSTIC_CANDIDATE`, not owner approval. `docs/scaffold_rollout_results.md`
contains one deterministic diagnostic rollout per condition, never a sampling
estimate.

## 3. Required Owner Decisions: Reward and Logging

No training reward manager or shared term registry exists. Every item below is
`MISSING`. The owner must approve a complete inventory; B5B must reject an
undeclared term or a task-only variation presented as a shared term.

| ID | Owner must freeze | Source / units | Cross-task acceptance gate |
| --- | --- | --- | --- |
| `RWD-TERM-*` for every shared term | ID, task applicability, formula version, named `TaskProgressSignals`/critic inputs, reduction, sign, weight, input scale, transform, clip, normalizer ID/update rule, missing-data behavior. | `MISSING`; progress is dimensionless, height m, wrench N/Nm, impulse N s, action normalized. | Same ID has same units/sign in all four tasks; exact formula/sign/clip/all-task finite/no-undeclared-field tests. |
| `RWD-AGG-001` | aggregate equation/order, task weighting, episode return reduction, terminal contribution, dtype, finite behavior, clip placement. | `MISSING`. | Recompute exactly from term logs; pooled return retains task counts and does not favor reference duration silently. |
| `RWD-LOG-001` | per-step/episode keys, units, reductions/quantiles, denominator, reset handling, task partitions and pooled rule. | `MISSING`. | Schema/recomputed return/no cross-episode leak tests. |
| `RWD-NOMINAL-001` | nominal-retention baseline, horizon, statistic, tolerance and failure action. | `MISSING`; scaffold nominal is not a residual criterion. | Per-task baseline identity and retention report. |

Progress, force, contact, stability, set-down error, orientation error, action
norm, semantic KL, success/failure/exhaustion, and sensor quality are not reward
terms simply because source exposes them. Clean wrench is target/diagnostic data,
not actor input.

## 4. Required Owner Decisions: Normalizers

| ID | Owner must freeze | Evidence / boundary | Acceptance gate |
| --- | --- | --- | --- |
| `NORM-POLICY-001` | exact actor fields/order, transform, center/scale or robust statistic, epsilon, clip, dtype. | Field order is `FROZEN_CONTRACT`; statistics are `MISSING`. Wrench channels are N/Nm after sensor normalization; proprio is mixed SI/radians; action is normalized. | Only actor-legal samples; privileged mutation cannot alter actor values or normalizer state. |
| `NORM-CRITIC-001` | all critic fields, transforms/statistics, shared versus task-specific state, relation to policy-prefix state. | `[668]+[16]+[39]+[2]+[90]+[4]+[17]+[9]` layout frozen; statistics missing. | Critic state cannot flow to actor normalizer. |
| `NORM-UPDATE-001` | estimator, initialization/warmup, per-env/global reduction, cadence, distributed sync, NaN policy, serialization, update/freeze stage. | `MISSING`. | deterministic multi-rank, save/load, freeze, isolation tests. |
| `NORM-EVAL-001` | held-out/evaluation checkpoint statistics and no-update rule. | `MISSING`. | Evaluation preserves normalizer bytes. |

## 5. Required Owner Decisions: Mismatch and Reset Sampling

`EpisodeParameterStore` validates explicit 39/2/90 rows only; it samples no
episode parameter and defines no curriculum. For every group the owner must
freeze distribution family, parameters, support, units/frame, correlation group,
nominal atom probability, seed stream, reset-time application order, rejection
policy, and applied/readback validation.

### 5.1 Physical row (39-D)

| Group | Indices and units | Cross-task semantic rule | Evidence |
| --- | --- | --- | --- |
| Mechanism | `0:3` hinge-axis rad; `3` friction; `4` damping. | Door applies; rigid tasks must explicitly retain zero/unused or define an approved analogue. | `FROZEN_CONTRACT` schema; nominal source values only. |
| Handle/contact frame | `5:11`: xyz m + rotvec rad. | Cannot silently change from door contact-frame meaning to rigid geometry. | `FROZEN_CONTRACT`. |
| Inertial | `11` mass kg; `12:15` CoM m; `15:21` inertia `[Ixx,Iyy,Izz,Ixy,Ixz,Iyz]` kg m2. | Rigid values must be applied PhysX readback; owner must approve physical-realizability and mass/CoM/inertia correlation. | `SOURCE_NOMINAL` anchors and frozen readback rule. |
| Surface | `21` static/dynamic friction coefficient. | Current rigid nominal ties both; split requires approval. | `SOURCE_NOMINAL` `0.5`; no density. |
| Reset pose | `22:28` object xyz m + rotvec rad; `28:31` stance x/y m + yaw rad. | Frame must be named identically per task. | `FROZEN_CONTRACT` schema; distribution `MISSING`. |
| Contact/load share | `31:37` target xyz m + rotvec rad; `37:39` left/right load-share error. | Both wrists always exist; no contact-role actor input. | `FROZEN_CONTRACT`; sampler missing. |

### 5.2 Scaffold row (2-D)

| Field | Unit/current nominal | Evidence | Owner decision |
| --- | --- | --- | --- |
| `delay` | integer physics substeps; C0 nominal `4`. | `SOURCE_NOMINAL`. | support/distribution, correlation with alpha, parity implications, nominal probability. |
| `alpha` | dimensionless smoothing; C0 nominal `0.9`. | `SOURCE_NOMINAL`. | support/distribution, application point, correlation, task sharing. |

### 5.3 Sensor row (90-D)

Validation constraints are not distributions: nonzero quaternion,
`1+scale_error>0`, nonnegative noise, delay `[0,3]`, filter/dropout `[0,1]`,
and positive limits/scales.

| Group | Row range and units | Owner must freeze |
| --- | --- | --- |
| Axis frame | `0:8`, two normalized quaternions. | frame/provenance, parameterization, left/right coupling, nominal atom. |
| Scale/bias | `8:20` fractional scale; `20:32` additive N/Nm. | per-axis distribution/correlation and reset calibration. |
| Drift/noise | `32:44` rate N/Nm per call; `44:56` random-walk std N/Nm per call; `56:68` white std N/Nm. | time model, horizon, axis/wrist correlation, seed ownership. |
| Delay/filter | `68:70` integer control steps; `70:72` alpha. | joint distribution, reset buffers, noise/dropout correlation. |
| Saturation/dropout | `72:78` N; `78:84` Nm; `84:86` probability. | per-axis limits, iid/burst process, quality/validity behavior, failure handling. |
| Semantic/contact scales | `86:88` `F_scale` N; `88:90` `M_scale` Nm. | calibration values, sampling policy, detector/semantic-target relation and version. |

## 6. Required Owner Decisions: Curriculum and Episodes

| ID | Owner must freeze | Evidence | Acceptance gate |
| --- | --- | --- | --- |
| `CUR-TASK-001` | C0--C3 task probabilities, allocation, loss aggregation, minimum nominal fraction, seeds. | `MISSING`; V1 percentages are candidates. | categorical frequency and per-task/pool dominance tests. |
| `CUR-STAGE-001` | entry/exit metric/window/samples/seeds, hysteresis, rollback, max duration, frozen state. | `MISSING`. | synthetic deterministic transition/rollback/no-NaN transition tests. |
| `CUR-MISMATCH-001` | stage family probabilities, nominal atom, distribution reference, correlations, sampler version. | `MISSING`. | sampled row validation and `(base_seed,env_id,episode_index)` order invariance. |
| `CUR-STRESS-001` | held-out inventory/provenance, no-update policy, metrics and decision rule. | `MISSING`; historical stress is candidate only. | training sampler rejects held-out IDs; evaluation preserves training state. |
| `EP-001` | training horizon in steps/seconds and task/stage variance. | C0 `6/8` only; training value `MISSING`. | exact timeout boundaries. |
| `EP-002` | exhaustion/timeout precedence and diagnostics versus termination policy. | Door `step+max(FUTURE_STEPS)>=length`; rigid `step>=length`; lengths 540/792/472/199. | task-specific `-1/0/+1` boundary tests. |
| `EP-003` | success/failure latches, recovery, reward role, reset behavior. | Door/box failure `<0.45 m`; payload `<0.25 m`; payload success needs exhaustion, lift `0.10 m`, set-down `<=0.25 m`, no failure. | latch/reset/precedence/selected-reset tests. |
| `SNS-001` | tare eligibility/samples, `F_scale`, `M_scale`, detector thresholds/temperature/smoothing, ramp attack/release, calibration and stage rules. | C0 values only; training values `MISSING`. | approved response/calibration boundaries; no implicit C0 reuse. |

## 7. Four-Task Privilege Alignment

```
Reward inputs: TaskProgressSignals, explicitly named critic-only privileged
               fields, and deployable-boundary data only.
Actor input:   wrist tokens, proprioception, a_nom history, previous a_total,
               and derived z_cross only.
Forbidden actor data: object/task identity; object pose/velocity/mass/CoM/inertia;
 hinge/mechanism state; contact truth; support/foot contact or wrench; clean wrench;
 semantic targets; mismatch; progress/success/failure/stability/reference truth;
 HDMI command[356].
```

Any critic-only reward input needs a `RWD-TERM-*` declaration and must not leak
through normalizer state, recurrent state, auxiliary target, logging cache, or
concatenation. For every forbidden field and task, mutation must leave the
668-D policy observation and 220-D actor input bitwise unchanged.

## 8. B5B Recommendation (Only After Every Approval)

### Exact proposed whitelist

```
configs/phase4b5_numeric_contract.json
somaforce_cross/envs/numeric_contract.py
somaforce_cross/envs/reward_manager.py
somaforce_cross/envs/mismatch_sampler.py
somaforce_cross/envs/normalizer.py
somaforce_cross/envs/curriculum.py
somaforce_cross/envs/residual_env.py
somaforce_cross/envs/residual_env_cfg.py
tests/test_phase4b5_numeric_contract.py
tests/test_phase4b5_reward_manager.py
tests/test_phase4b5_sampling_and_normalizer.py
tests/test_phase4b5_privilege_boundaries.py
```

Excluded: artifacts, golden runtime, scaffold adapter, action composer, force
core, sensor implementation, PPO/RSL-RL/training scripts, golden archives, and
existing reports. Needing another path requires a replacement owner whitelist.

### Versioned configuration format

Use canonical UTF-8 JSON (`sort_keys=True`, separators `(',', ':')`) and record
its SHA256 in checkpoints/logs. Reject unknown/missing keys, NaN/infinity,
unitless quantities, and non-approved status. Placeholder strings are
deliberately non-numeric and must never become defaults.

```json
{
  "contract_version": "phase4b5_numeric_v1",
  "source_commit": "caf09bfdaaa50004bead2750b9da495a340adffa",
  "approval": {"status": "approved", "owner": "<name>", "date": "<ISO-8601>"},
  "reward_terms": [{"term_id": "<owner-defined>", "inputs": [], "units": "<unit>", "sign": "<+|->", "weight": "<approved number>", "clip": ["<lo>", "<hi>"]}],
  "normalizers": {"policy": {}, "critic": {}, "update_and_freeze": {}},
  "mismatch": {"physical_39": {}, "scaffold_2": {}, "sensor_90": {}},
  "curriculum": {"task_balance": {}, "stages": {}, "held_out_stress": {}},
  "episode_and_latches": {},
  "sensor_training_contract": {},
  "acceptance": {"required_tests": [], "stop_conditions": []}
}
```

### Required B5B tests and stop conditions

1. Canonical JSON hash/version/commit/approval and no-default rejection.
2. Per-term formula/unit/sign/scale/clip/normalizer/aggregate/logging tests.
3. Four-task actor/critic mutation tests including normalizer/reward-cache leaks.
4. 39/2/90 sampler support/correlation/nominal atom/seed/order/reset/readback tests.
5. C0 zero reward/authority/identity sensor/nominal row/6-step/one-scale regression.
6. Normalizer update, distributed reduction, freeze, checkpoint, and held-out no-update tests.
7. C1--C3 balance, transition, rollback, nominal fraction, held-out exclusion tests.
8. Per-task timeout/exhaustion/latch boundaries, including door versus rigid and
   `0.45 m` versus `0.25 m` failures.
9. Existing Phase 1--3/Phase 4/full tests, compileall, AST/import, checksums,
   golden SHA, whitespace, diff check, and exact allowlist verification.

Stop without commit, push, PPO, or training if: any owner item is unapproved; a
historical candidate becomes a default; unit/frame/formula is ambiguous; checksum
or golden SHA changes; C0 changes; actor leak/normalizer leak appears; correlation,
nominal probability, reset seed, task balance, or held-out policy is absent; a
shared term silently differs per task; or any required test fails.

## 9. Verification Performed

```
Phase 1-3 regression: 94 passed in 8.15s
Canonical command:
pytest -q tests/test_force_core.py tests/test_residual_action.py tests/test_virtual_ft.py tests/test_wrist_wrench_source_contract.py tests/test_contact_semantics.py
Current Phase 4 tests: 309 passed in 18.57s
Full pytest: 431 passed in 20.59s
compileall: PASS
AST scan: 39 somaforce_cross Python files; active_adaptation/rsl_rl hits=[]
git diff --check before report: PASS

artifact SHA256SUMS: all entries OK
push_door_hand SHA256SUMS SHA256: 51a1535953d57b2ac63dea7d821edfdd71b95f9ad6e681bb16bd714148a5e6cb
push_box SHA256SUMS SHA256: 5dafd1f2b4b75edec859a8aa9b026158f125d9e2d3e294537b35bf9569729b08
move_suitcase SHA256SUMS SHA256: c0d61caca2ca6aeade7b34278168fda1e9a3f0daaaa36febef536f9a053055d9
move_largebox SHA256SUMS SHA256: bc469c68324f54cbeb8d258b917a3773b82b38091aac3817d465da55971b3baa
golden runtime SHA256: e924dadeb945278ee85fad82e31be56f78bce4e62f5bb9f575f5938f6c4868a0
```

Isaac was not rerun. No report/source/test conflict blocks this audit; absent
training numerical approvals are the intended owner gate.

## 10. Waiting for Project Owner Approval

Phase 4B5B cannot start until the owner approves or rejects, item by item:

1. Every `RWD-TERM-*`, `RWD-AGG-001`, `RWD-LOG-001`, and `RWD-NOMINAL-001`.
2. `NORM-POLICY-001`, `NORM-CRITIC-001`, `NORM-UPDATE-001`, `NORM-EVAL-001`.
3. Every 39-D/2-D/90-D family field, distribution, parameters, correlation,
   nominal probability, units, and reset-time application.
4. `CUR-TASK-001`, `CUR-STAGE-001`, `CUR-MISMATCH-001`, and `CUR-STRESS-001`.
5. `EP-001`, `EP-002`, `EP-003`, and `SNS-001`.
6. The exact B5B whitelist, JSON payload/hash, test matrix, and stop conditions.

Until then: no B5B, PPO, RSL-RL, training, commit, or push.
