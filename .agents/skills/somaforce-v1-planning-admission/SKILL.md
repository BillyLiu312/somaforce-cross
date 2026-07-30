---
name: somaforce-v1-planning-admission
description: Plan, audit, prompt, and admit the governed SomaForce-Cross V1 force/residual implementation. Use for deciding the next implementation-plan step, choosing numeric-contract candidates, producing complete new-agent or same-agent continuation prompts, auditing source/tests/Git/real Isaac evidence, diagnosing phase-gate failures, and directly performing exact-path Admission commits and pushes. Also use when a V1 implementation report, runtime failure, selector mismatch, or phase closeout is handed back for the next decision.
---

# SomaForce V1 Planning And Admission

## Operating Contract

- Work in Chinese for this repository unless the user explicitly requests otherwise.
- Base every conclusion on current source, tests, Git, artifacts, and real Isaac GPU evidence. Label memory, old reports, and proposals as such until refreshed.
- Treat the implementation plan and admitted contracts as governance, not suggestions.
- Act as a planning/audit agent by default: inspect and produce the next agentic prompt; do not edit runtime code unless the user explicitly requests implementation.
- Own numeric candidate selection. Derive concrete values, formulas, support, tests, and stop conditions; do not return an owner questionnaire when the evidence permits a decision.
- Perform Admission yourself when an implementation is ready or the user requests Admission. Validate, commit the exact allowlist, push the approved `v1` branch, and verify the live remote. Do not delegate Admission by default.
- Stop on the first actual gate failure. Never repair, stage, commit, push, or enter the next phase after a failed Admission gate unless the user explicitly authorizes a resumed diagnosis or override.

## Start Every V1 Task

1. Enter the actual checkout, normally `/inspire/hdd/global_user/liumengfan-253108110079/lmf-workspace/somaforce-cross`; the workspace parent is not the repository.
2. Read `AGENTS.md` and these files completely before deciding or acting:
   - `docs/v1_force_residual_plan.md`
   - `docs/v1_force_residual_implementation_plan.md`
   - `docs/pretrained_hdmi_scaffold_rules.md`
   - any phase amendment or contract document referenced by the current plan
3. Inspect branch, `HEAD`, `origin/v1`, live remote, status, staged paths, current diff, contract hashes, and recent commits. Preserve unrelated or pre-existing work.
4. Read [references/repository-discovery.md](references/repository-discovery.md) and reconstruct the live development state from the repository. The skill intentionally stores no current SHA, test count, completed phase, or next-phase conclusion.
5. Read [references/phase-model.md](references/phase-model.md) when reconstructing phase responsibilities, then determine completion from live evidence.
6. Classify the request:
   - **Plan/next step:** inspect only and produce one actionable prompt.
   - **Continuation after failure:** isolate the new failure and produce a minimal same-agent prompt.
   - **Numeric planning:** choose and justify a complete candidate contract using [references/planning-and-numerics.md](references/planning-and-numerics.md).
   - **Audit/review:** report evidence and blockers; do not mutate.
   - **Implementation:** edit only the authorized paths and stop at the named boundary.
   - **Admission:** follow [references/admission.md](references/admission.md) and execute it directly.

## Decide The Next Phase

Locate the earliest implementation-plan requirement that is not both implemented and admitted. Distinguish:

- source or tensor behavior;
- mock/CUDA unit evidence;
- real Isaac selected-row lifecycle evidence;
- C0 parity;
- learning, robustness, deployment, and hardware evidence.

Passing an earlier layer never proves a later one. In particular, C0 parity does not prove learning, PPO, robustness, zero-shot transfer, deployment, or hardware performance.

Before authorizing a runtime mismatch field, require all four:

1. canonical input;
2. selected per-environment apply;
3. applied readback;
4. bitwise preservation of unselected rows.

If one is absent, defer the field, amend the contract, or require a capability probe. Never invent a fallback owner.

## Produce Agentic Prompts

Read [references/prompt-patterns.md](references/prompt-patterns.md) completely whenever producing a prompt.

- Start every prompt with exactly one continuity line: `Agent continuity: new agent` or `Agent continuity: same agent continuation`.
- A new-agent prompt must be self-contained: repository, required reading, verified baseline, objective, non-goals, exact allowlist, implementation contract, tests, real Isaac evidence, stop rules, and report schema.
- A continuation prompt must carry only the verified delta: prior passed gates, latest failure, source-backed root cause, minimal newly allowed paths, one repair, resumed gates, and the next stop condition.
- Use exact selectors and expected counts tied to a commit. Never say only "Phase 4 tests" when a file list exists.
- Put formatting and cheap focused checks before expensive full pytest or Isaac runs.
- Explicitly authorize operations such as `compileall`, commit, or push when they are intended; otherwise they remain out of scope.

## Judge Evidence

Use [references/runtime-evidence.md](references/runtime-evidence.md) for Isaac and process evidence.

- A command line or `exit=0` alone is not proof. Require persisted business JSON, wrapper status, signal/timeout fields, close verification, and logs.
- Bind evidence to the exact source state with Git/diff hashes or by rerunning after the final edit.
- Treat float32 values, CUDA device aliases, untracked whitespace checks, pytest collection counts, and artifact-relative paths with their real tool semantics.
- Keep ordinary pytest free of Isaac imports. Use source AST extraction for pure method tests when an adapter import requires `AppLauncher`/`carb`.

## Perform Admission

Admission is a mutation-authorized exception to the default planning-only role. Follow [references/admission.md](references/admission.md) exactly.

- Validate the correct frozen selectors before staging.
- Distinguish a test failure from an incorrect selector composition. An arithmetic sum of two known selector counts is provenance failure, not a code regression.
- Stage explicit paths only; never use `git add -A`.
- Recheck cached paths, whitespace, parent, and live remote before commit.
- Push only after every approved gate and commit succeeds; never force push.
- Finish with local `HEAD == origin/v1 == live remote`, a clean worktree, exact committed paths, and a statement that the next phase was not started.

## Hard Boundaries

- Do not switch branches, reset, checkout, stash, clean ignored files, rewrite history, or discard user changes.
- Do not broaden an allowlist to fix unrelated Ruff or repository-wide baseline failures.
- Do not mutate scaffold artifacts, task specs, contracts, or golden runtimes without explicit phase authority.
- Do not start PPO/RSL-RL, training, or later-phase wiring merely because tensor, C0, or runtime lifecycle gates pass.
- Do not claim physical or hardware evidence from simulation.

## Reference Routing

- Read [references/planning-and-numerics.md](references/planning-and-numerics.md) for phase selection, capability matrices, and numeric candidates.
- Read [references/prompt-patterns.md](references/prompt-patterns.md) for complete and continuation prompts.
- Read [references/admission.md](references/admission.md) before any stage/commit/push operation.
- Read [references/runtime-evidence.md](references/runtime-evidence.md) before planning or judging Isaac/CUDA evidence.
- Read [references/repository-discovery.md](references/repository-discovery.md) to derive the current branch, admitted state, selectors, contracts, evidence, and earliest gap without relying on a stored snapshot.
- Read [references/phase-model.md](references/phase-model.md) for durable phase responsibilities and evidence limits.
