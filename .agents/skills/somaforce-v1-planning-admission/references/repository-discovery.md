# Repository State Discovery

## Contents

1. No stored progress
2. Establish Git truth
3. Discover governance and contracts
4. Reconstruct implementation coverage
5. Reconstruct selectors and evidence
6. Determine the earliest gap
7. Report the live baseline

## No Stored Progress

Do not look for a current phase, SHA, test count, allowlist, or next-step answer in this skill. Derive all of them from the checkout during the current turn. Historical reports and conversation memory may suggest search terms but never replace live verification.

## Establish Git Truth

Start with read-only Git evidence:

```text
git branch --show-current
git rev-parse HEAD
git rev-parse origin/v1
git ls-remote origin refs/heads/v1
git status --short
git diff --name-status
git diff --cached --name-status
git log --oneline --decorate -n <bounded count>
```

Determine whether the worktree is clean, implementation is uncommitted, Admission is pending, or local/remote refs diverge. Do not assume the latest commit subject proves its tests or runtime evidence.

## Discover Governance And Contracts

Read the governing files completely and follow their references. Then inspect:

- contract JSON files and schema loaders;
- approved amendments and Admission reports in the repository;
- scaffold manifests and `SHA256SUMS`;
- task specifications and active runtime adapters;
- recent commits touching the relevant plan/config/source/tests.

Compute raw file SHA with `sha256sum`. Compute canonical contract SHA through the repository's structured loader. Keep the two identities separate.

Search with `rg` before assuming a module, API, phase label, selector, or report exists.

## Reconstruct Implementation Coverage

Build a live matrix:

| Plan requirement | Owning source | Tests | Real runtime | Commit/Admission evidence | Status |
|---|---|---|---|---|---|

Inspect actual method bodies and call sites, not exported names alone. Trace ownership through reset, apply, readback, observation, reward, done, logging, and curriculum paths as relevant.

Classify each requirement as:

- absent;
- isolated module only;
- wired in the environment;
- mock/CUDA tested;
- real Isaac proven;
- admitted and pushed.

Do not infer environment wiring from the existence of a standalone module.

## Reconstruct Selectors And Evidence

List current tests with `rg --files tests` and inspect their ownership. Recover phase selectors from current plans, test content, accepted reports, and commit history. Express every selector as an explicit sorted file or nodeid list.

Use `pytest --collect-only -q <explicit selector>` to obtain current counts. Compare sorted nodeids when counts disagree. Never copy a stored count from an old prompt without collection at the current source state.

For runtime evidence, locate the active harness and its evidence schema. Verify persisted wrapper/result/log/summary files mentioned by the current implementation report. Bind them to the current source using hashes or rerun requirements. Ephemeral `/tmp` paths are evidence only while they exist and match the handed-off source.

Check local GPU/Isaac versions and APIs when the next decision depends on runtime capability. Source API presence alone is not selected-row live proof.

## Determine The Earliest Gap

Compare the live matrix to the authoritative implementation plan. The next step is the earliest requirement that lacks either implementation or required Admission evidence.

Do not select a phase from its name, commit chronology, or memory. Explain why every earlier requirement is closed and why every later phase remains unauthorized.

When evidence is incomplete, plan a bounded audit or capability probe rather than guessing progress.

## Report The Live Baseline

Before producing a prompt or performing Admission, state the freshly discovered:

- branch, local/tracking/live remote SHAs;
- clean or exact dirty path set;
- contract version and raw/canonical hashes;
- relevant source owners and missing call sites;
- explicit selectors and current collection counts;
- real runtime evidence and remaining evidence gaps;
- earliest unclosed implementation-plan requirement;
- prohibited later phases.

Keep this baseline in the current response or prompt. Do not write it back into the skill.
