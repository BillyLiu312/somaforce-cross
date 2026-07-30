# Agentic Prompt Patterns

## Contents

1. Choosing continuity
2. Complete new-agent prompt
3. Same-agent continuation prompt
4. Failure follow-up rules
5. Prompt quality checklist

## Choosing Continuity

Use `Agent continuity: new agent` when the recipient must reconstruct the phase from source or provide independent implementation/audit.

Use `Agent continuity: same agent continuation` after a stopped gate when the same implementation agent already owns the worktree and only a verified delta is needed.

Never omit the continuity line.

## Complete New-Agent Prompt

Use this structure and replace every placeholder with verified facts:

```text
Agent continuity: new agent

Role: implement Phase <phase> only. Work in Chinese and base all judgments on
current source, tests, Git, artifacts, and real Isaac evidence.

Repository:
<absolute repository path>

Read completely before acting:
- AGENTS.md
- docs/v1_force_residual_plan.md
- docs/v1_force_residual_implementation_plan.md
- docs/pretrained_hdmi_scaffold_rules.md
- <current contract/amendment>

Verified baseline:
- branch / HEAD / origin / live remote
- worktree state
- admitted contract version, raw SHA, canonical SHA
- exact prior selectors and evidence

Objective:
<one bounded implementation result>

Non-goals:
<later phase, PPO/training, docs/artifacts, unrelated cleanup>

Allowed paths:
- <exact path>

Implementation contract:
- <ownership, shape, dtype, formula, order, atomicity>
- <selected/unselected behavior>
- <C0 and privilege invariants>

Validation order, failure stops immediately:
1. focused test
2. scoped Ruff and format
3. phase selectors with exact files/counts
4. full collection/full pytest
5. explicitly authorized compile/import/whitespace gates
6. bounded real Isaac case
7. full runtime/C0 matrix

Do not commit or push. Report modified paths, tests, runtime JSON/log evidence,
contract hashes, Git state, and unresolved gates. Stop at Phase <phase>.
```

The complete prompt must make implementation decisions. Do not leave `TBD`, ask the implementer to choose core numbers, or refer vaguely to "the usual tests".

## Same-Agent Continuation Prompt

Use this structure after a gate stops:

```text
Agent continuity: same agent continuation

Continue Phase <phase>. Do not enter the next phase.

Verified unchanged baseline:
- HEAD / remote / worktree allowlist
- last passed gates
- contract hashes

Latest failure:
- exact command
- exact node/case
- wrapper/result/log or traceback
- what was not run

Source-backed diagnosis:
<one root cause; distinguish implementation defect, test defect, harness defect,
format-only issue, selector provenance error, or missing runtime capability>

This round may modify only:
- <minimal path list>

Required repair:
<one precise behavior change and rejection boundaries>

Validation:
1. failing node/case
2. local test file
3. format before expensive gates
4. frozen selectors and full pytest
5. same real Isaac boundary
6. full suite only if the boundary passes

On any new failure, stop and report evidence. Do not continue repairing.
No commit/push, PPO, training, or next-phase work.
```

Carry only verified delta. Do not repeat the entire original implementation plan unless context was lost or a new agent will receive the prompt.

## Failure Follow-Up Rules

Classify before prompting:

- **Implementation defect:** authorize the owning source plus focused tests.
- **Test defect:** authorize only the test when source behavior is already correct; prove the test expectation from source/runtime.
- **Harness defect:** keep runtime source frozen; fix persistence, import order, timeout, or close evidence only.
- **Format-only:** authorize the exact file and mechanical formatter; compare AST before/after when useful.
- **Selector mismatch:** reconstruct the explicit file list with `--collect-only`; do not change code.
- **New Isaac exception:** stop the task matrix at the first case and authorize only the source owner and its pure test.

Ordinary pytest must not import adapters that transitively require `isaaclab.sim` before `AppLauncher`. Extract the actual pure method nodes with AST instead of mocking `carb` or launching Isaac in pytest.

## Prompt Quality Checklist

- Is continuity explicit?
- Is every baseline fact current or labeled historical?
- Is the next phase the earliest real gap?
- Are numeric values/formulas concrete?
- Is the allowlist exact and minimal?
- Are non-goals and privilege boundaries explicit?
- Are selector file lists and counts both present?
- Are formatting gates early?
- Is the real Isaac case bounded and persistent?
- Does every failure say what must not run next?
- Is commit/push authority unambiguous?
- Can the receiving agent act without asking the owner to fill missing design decisions?
