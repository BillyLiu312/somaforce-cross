# Planning And Numeric Contracts

## Contents

1. Evidence hierarchy
2. Selecting the next step
3. Numeric candidate ownership
4. Runtime capability matrix
5. Test and rollout design
6. Planning output

## Evidence Hierarchy

Use authority in this order, resolving conflicts toward the higher and newer source:

1. `AGENTS.md` and the current V1 governing plans;
2. admitted contract JSON and approved amendments;
3. frozen manifests, checksums, and task specifications;
4. active source/API behavior;
5. tests tied to an exact selector and commit;
6. bounded real Isaac/CUDA evidence;
7. historical reports and memory;
8. new candidate proposals.

Never report a proposal as admitted fact. Separate `SOURCE_FACT`, `LIVE_PROVEN`, `CANDIDATE`, `DEFERRED`, and `HELD_OUT` when useful.

## Selecting The Next Step

Build a compact gap table from the implementation plan:

| Requirement | Source | Pure tests | Real runtime | Admission | Earliest gap |
|---|---|---|---|---|---|

Choose the earliest row lacking implementation or Admission. Do not skip forward because later modules already exist in isolation.

For environment lifecycle work, trace the real control sequence through reset, action, physics, done, reward, automatic reset, and observation. Determine ownership and mutation order before proposing files.

For selected reset, require:

```text
sample -> full prevalidation -> scene/reset pose -> runtime apply
-> direct readback -> applied store -> sensor/scaffold/history reset
```

Do not put `sim.step()` or `sim.forward()` in selected reset. Preserve unselected rows bitwise and use deterministic per-environment episode indices.

## Numeric Candidate Ownership

The planning agent chooses complete candidates rather than asking the owner to fill blanks.

For every field record:

- name, width, unit, frame, order, dtype, and valid domain;
- nominal value or source;
- C1/C2/C3 support and distribution;
- held-out stress support, kept outside training;
- exact sampling stream and deterministic key;
- normalizer center/scale/clip, if authorized by the phase;
- reward or termination formula, if authorized;
- runtime owner, apply API, readback API, and tolerance;
- actor/critic visibility and privilege boundary;
- rejection tests and real-runtime evidence required.

Derive values from source manifests, task physics, admitted plans, prior ranges, and live capability. If evidence cannot justify a number, propose a conservative candidate with rationale and mark it `CANDIDATE`; do not guess and call it frozen.

Freeze structured values in a schema-validated JSON contract. Compute canonical JSON SHA separately from the raw file SHA. Never compare one to the other.

Sampling requirements:

- key determinism by `(base_seed, env_id, episode_index)`;
- selected-list-order invariance;
- independent counter/stream domains for family, value, scaffold, and sensor draws;
- exact nominal/C1/C2/C3/held-out separation;
- statistical frequency tests with float32-aware keys and bounded memory;
- CPU/CUDA equality where the contract requires it.

Formula tests must use tensor-aware tolerances or exact tensor equality as appropriate. Avoid Python float `==` for float32 results.

## Runtime Capability Matrix

Before approving a field, classify it:

- `READY_EXISTING_API`: selected setter/getter proven live;
- `NEEDS_IMPLEMENTATION`: owner exists but lifecycle wiring is absent;
- `NOT_PER_ENV_SETTABLE`: no active selected owner/API;
- `DEFERRED_TO_LATER_PHASE`: outside the current scientific boundary.

For each trainable field require canonical input, selected apply, applied readback, and unselected isolation. A source method name is not live proof.

Do not alias immutable geometry to an unrelated adapter buffer. When hinge axis, handle geometry, or another construction-time property lacks an episode-time owner, use an approved scene bank or defer/amend the field.

## Test And Rollout Design

Scale evidence with risk:

1. focused formula or API test;
2. phase-specific pure tests;
3. frozen earlier-phase selectors;
4. exact full collection and full pytest;
5. bounded 2-env selected-row real Isaac case;
6. 64-env selected set such as `{0,7,63}`;
7. task matrix;
8. C0 golden/residual/compare parity.

Keep one process per Isaac case when isolation is required. Persist wrapper/result/log before moving to the next case. Stop on the first new runtime exception and return a continuation prompt rather than repairing multiple unknowns in one pass.

## Planning Output

Lead with the decision: next phase, reason, and why later phases remain closed. Then provide a paste-ready agent prompt with:

- continuity;
- exact baseline and hashes;
- objective and non-goals;
- exact allowlist;
- concrete formulas and values;
- ordering and atomicity;
- test selectors and counts;
- real Isaac matrix;
- failure-stop behavior;
- no commit/push unless explicitly authorized;
- required final evidence report.
