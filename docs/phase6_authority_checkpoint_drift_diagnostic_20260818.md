# Phase 6 Authority And Checkpoint-Drift Diagnostic Audit

Date: 2026-08-19

Status: **read-only runtime evidence audit; no training or Phase 6 acceptance
claim**.

## 1. Scope

This document records the completed four-GPU diagnostic run:

```text
RUN_ID=phase6-segment47-authority-contact-drift-4gpu-20260818_155951
RUN_DIR=outputs/phase6_gradient_diagnostics/phase6-segment47-authority-contact-drift-4gpu-20260818_155951
```

The diagnostic was bound to repository commit:

```text
890aa4a15eb756f5733144e0b106ceb3070fb05b
```

Its admitted diagnostic source identities were:

```text
scripts/diagnose_phase6_gradients.py
d39ef4c93e06b3345cb3d4b48a180bea1518d29ec18664480912121b8227173a

tests/test_phase6_gradient_diagnostics.py
8cc4e87208736bb670c7020a903ccaa351a4049299072e82a2c72691900b4e52
```

The run is diagnostic-only. It did not update policy, optimizer, normalizer,
curriculum, checkpoint, or production source, and it does not establish Phase 6
learning acceptance.

## 2. Root Evidence

The persisted root files and SHA256 values are:

| Evidence | SHA256 |
| --- | --- |
| `plan.json` | `dd20641d3efc0253d1554cc061cd9aa4f9f5a081a280d0e45b272be0831d0456` |
| `progress.json` | `fcb458b1f2cd0c87e4408c6fdb4b8de4c426a11cadb4ef60f7830ff1a2576a9a` |
| `strict_restore.json` | `2a041fc16eeae5403ee9349dd5bd5e4013d4772fe6dc5c7ebf1fae399be68948` |
| `summary.json` | `eaa6cecf04bb560543f50d9dc48ec696956cf1d1b8ce678d2c78d160cb86c276` |
| `checkpoint_drift/summary.json` | `4b50b5638368c87368f616bfc0590507f15dafeefff5afb24a5d3cb96b66c785` |
| `job_logs/diagnostic_20260818_165447.log` | `cfcbfc292ad715b80ac43d9a7da905b2494dc4f982b8e04d81e5238b3a42aa64` |

The final `RUN_DIR` exists and the `.incomplete` work root does not. Root
`progress.json` is `complete`, and `strict_restore.json` is `ok`.

## 3. Runtime Closure

The source-derived plan required 24 distributed probes:

```text
authority comparison: 3 conditions x 3 probes = 9 probes
checkpoint drift:      5 checkpoints x 3 probes = 15 probes
total:                                             24 probes
```

Each probe used four ranks. The persisted evidence contains:

| Item | Required | Observed |
| --- | ---: | ---: |
| distributed probes | 24 | 24 |
| rank wrappers | 96 | 96 |
| rank results | 96 | 96 |
| stage summaries | 8 | 8 |
| suitcase event time series | 8 | 8 |

All 96 wrappers recorded:

```text
passed=true
exit_code=0
signal=null
timed_out=false
timeout_kind=null
world_size=4
marker_order_valid=true
markers=[JSON_WRITTEN, PHASE6_DIAGNOSTIC_RESULT, ENV_CLOSED]
result_status=ok
```

Wrapper `VmHWM` ranged from `4,585,892 KiB` to `8,096,736 KiB`. Every
result SHA matched its wrapper, all eight stage summaries reported
`invariants_ok=true`, and policy, optimizer, optimizer step, normalizer,
curriculum, rank RNG payload, source manifest, and checkpoint hashes were
unchanged before and after each read-only probe.

The job log contains 17,029 `errno=28/No space left on device` change-watch
messages. Live filesystem checks showed available storage and inodes; the
messages came from exhausted Isaac/Kit inotify watch capacity, not failed
business-result writes. They did not prevent any wrapper from closing, but a
future runtime wrapper should report this resource condition explicitly.

## 4. Authority Intervention

The authority comparison held all of the following fixed:

- segment-47 rebound checkpoint and checkpoint SHA;
- C2 physical/scaffold/sensor environment;
- task assignment and exact 25% transition balance;
- episode schedule and seeds;
- rank RNG start state;
- policy, optimizer, normalizer, and curriculum payload;
- source hashes.

Only the process-local authority lookup changed from C1 to C2. The main
`move_suitcase` results were:

| Metric | C1 authority | C2 authority | C2 - C1 |
| --- | ---: | ---: | ---: |
| success | 0.828125 | 0.093750 | -0.734375 |
| nominal retention | 1.000000 | 0.070313 | -0.929688 |
| reward | 8.2267 | 4.3938 | -3.8328 |
| arms residual norm | 0.2341 | 0.4212 | +0.1871 |
| waist residual norm | 0.0718 | 0.1441 | +0.0723 |
| legs residual norm | 0.1137 | 0.2274 | +0.1137 |
| p95 force rate | 132.23 | 151.49 | +19.25 |
| stability margin | 0.43169 | 0.43065 | -0.00103 |

The other task success deltas were small:

```text
move_largebox:  -0.0078125
push_box:       +0.00390625
push_door_hand:  0.0
```

Supported conclusion: under this fixed segment-47 C2 intervention, C2
authority directly caused the large suitcase degradation. This does not prove
that authority was the sole cause of the complete training-run drift.

## 5. Fixed-C1 Checkpoint Drift

The five checkpoint comparisons used the same C1 nominal environment,
schedule, seeds, rank assignment, and scaffold-only baseline.

| Segment | Iteration | Stored stage | Suitcase success | Nominal retention | Scaffold success |
| ---: | ---: | --- | ---: | ---: | ---: |
| 47 | 1465 | C2 | 0.972656 | 1.00 | 0.964844 |
| 51 | 1587 | C1 | 0.941406 | 1.00 | 0.964844 |
| 52 | 1618 | C1 | 0.343750 | 0.25 | 0.964844 |
| 59 | 1832 | C1 | 0.000000 | 0.00 | 0.964844 |
| 79 | 2442 | C1 | 0.011719 | 0.00 | 0.964844 |

`push_door_hand`, `push_box`, and `move_largebox` remained approximately
stable across the same checkpoints. The earliest localized suitcase collapse
is therefore segment 51 to segment 52. This supports task-specific learned
weight drift or forgetting. It does not, by itself, prove multi-task gradient
conflict or identify the responsible task.

Historical curriculum evidence records promotion to C2 at segment 47 and
rollback to C1 at segment 49. The rollback changes curriculum stage but does
not restore earlier policy or optimizer weights.

## 6. Contact-Phase Evidence

The complete 472-step suitcase traces showed:

- contact onset near steps 149-151;
- lift/carry onset near steps 193-196;
- no set-down window in any of the eight traces;
- no reference-exhaustion event;
- final set-down error remained nonzero.

Contact/lift/carry combined gradient norms were approximately 75-97, but were
almost entirely critic/value gradients. Actor/surrogate gradient norms were
approximately `1e-7`; semantic gradient norms were approximately 0.01-0.052;
PPO-semantic cosine was near zero. Reset-adjacent minibatch preclip pressure was
1.0 across all compared checkpoints and therefore did not discriminate the
drift.

The evidence does not support gradient explosion, actual clipping, semantic
gradient conflict, or a set-down reward defect as the established cause.
Set-down-phase attribution remains inconclusive because no set-down samples
were collected.

## 7. Evidence-Schema Caveats

The numerical and wrapper evidence is usable, but closure is not schema-perfect:

1. Eight child `progress.json` files remain `status="running"` after their stage
   summaries were successfully written.
2. Wrapper `result_json` paths and suitcase time-series paths retain their old
   `.incomplete` absolute location after the final directory move. The files
   exist at the corresponding final relative paths and their hashes match.

This historical `RUN_DIR` must not be rewritten to repair those fields.

## 8. Boundary Under The Independent-Learning Definition

The diagnosed checkpoints descend from a Phase 5 `push_door_hand` policy
warm-start. They remain valid evidence about that historical warm-start
training lineage. They are not evidence for a Phase 6 residual network trained
independently from four-task data.

No checkpoint or source-rebind artifact from this run may bootstrap the new
independent Phase 6 experiment. The run remains a warm-start baseline and a
source of authority/drift hypotheses only.
