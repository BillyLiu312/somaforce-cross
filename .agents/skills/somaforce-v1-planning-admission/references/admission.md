# Admission, Commit, And Push

## Contents

1. Admission authority
2. Preflight
3. Evidence gates
4. Selector provenance
5. Staging and commit
6. Push and closeout
7. Failure behavior

## Admission Authority

When the user says implementation is complete, requests Admission, or authorizes direct submission, the primary planning agent performs Admission itself. Do not merely write another Admission prompt unless the user explicitly asks for one.

Admission authorizes validation and, only after success, an exact-path commit and push to the approved branch. It does not authorize source repair or next-phase work.

## Preflight

Read all governing files and verify:

- branch is the approved branch, normally `v1`;
- local `HEAD`, tracking ref, and live `git ls-remote` parent are expected;
- worktree paths exactly equal the implementation allowlist;
- staged set is initially empty unless explicitly handed off staged;
- raw and canonical contract hashes are separately correct;
- artifact/golden hashes remain unchanged when in scope;
- no unrelated or ignored files are treated as evidence.

If live remote moved, stop. Do not pull, merge, rebase, reset, or force.

## Evidence Gates

Audit source contracts before executing gates. Run cheap gates first:

1. focused tests for the changed behavior;
2. scoped Ruff check and `ruff format --check` on the allowlist;
3. frozen phase selectors with explicit file lists;
4. exact full collection and full pytest;
5. explicitly authorized compileall and AST/import checks;
6. tracked and untracked whitespace checks;
7. artifact checksums;
8. bounded real Isaac/CUDA suite when required;
9. final Git and hash precommit gate.

If final source is byte-for-byte unchanged after already accepted full pytest/Isaac evidence, the user may explicitly authorize reuse. Report clearly which gates were rerun and which evidence was reused.

Do not use a passing old log after source changed unless a content hash or a rerun binds it to the final source.

## Selector Provenance

Freeze selectors as explicit path lists. First use `pytest --collect-only -q` and record the count. Do not construct a "candidate selector" by combining phases.

When an observed count is the arithmetic sum of known selectors, diagnose the file composition. A combined count is not a regression merely because it differs from one frozen selector.

An actual failing test node triggers failure-stop. A wrong selector command is a process/provenance failure: stop Admission, correct it only after a user-authorized resume, and rerun the intended selector without editing source.

Full pytest counts are commit-bound. Never hard-code an old number without checking collection at the current `HEAD` plus worktree.

## Staging And Commit

After all approved gates pass:

1. Recheck live remote parent.
2. Stage each allowed path explicitly with `git add -- <paths>`; never use `git add -A` or `git add .`.
3. Sort and compare `git diff --cached --name-only` against the expected list.
4. Run `git diff --cached --check` and inspect cached stat/diff.
5. Confirm there are no unstaged or extra paths.
6. Commit with the approved phase-specific message.
7. Verify new commit parent and `git show --name-only` exact path set.
8. Confirm the worktree is clean.

If commit hooks modify files, the commit fails, or path sets differ, stop before push.

## Push And Closeout

Immediately before push, confirm the live remote still equals the approved parent. Push the approved branch normally; never force.

After push verify:

```text
local HEAD == origin/v1 == git ls-remote origin refs/heads/v1
```

Confirm worktree clean and no residual pytest/Isaac/Kit workers created by the Admission.

Report:

- new commit and parent;
- exact committed paths;
- rerun gates and reused evidence separately;
- test collection and selectors;
- real GPU/Isaac summary and evidence path;
- raw/canonical/artifact hashes;
- local/tracking/live remote equality;
- clean worktree;
- explicit statement that the next phase, PPO, and training were not started.

Do not record the successful Admission SHA, test count, path set, completed phase, or next phase in this skill. The next agent must reconstruct them from the repository. Update the skill only when durable workflow, governance, prompt, or evidence rules change, and never silently add skill files to a phase commit allowlist.

## Failure Behavior

On the first actual gate failure:

- stop subsequent gates;
- do not change source or tests;
- do not stage, commit, or push;
- preserve the exact allowlist;
- report command, node/case, output, what was not run, hashes, and Git state;
- produce a continuation diagnosis only when asked or when continuing planning work.

Never convert an incomplete Admission into a partial commit.
