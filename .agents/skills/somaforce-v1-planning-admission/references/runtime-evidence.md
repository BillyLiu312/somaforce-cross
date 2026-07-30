# Runtime And Evidence Rules

## Contents

1. Evidence layers
2. Bounded worker contract
3. Selected-row invariants
4. Isaac import and API boundaries
5. Common evidence traps

## Evidence Layers

Keep these claims separate:

- pure PyTorch/source contract;
- CPU/CUDA tensor behavior;
- Isaac API capability;
- selected-row runtime lifecycle;
- C0 golden parity;
- learning/robustness;
- deployment/hardware.

Do not promote a claim across layers without new evidence.

## Bounded Worker Contract

Run each risky Isaac case in a bounded process group with a timeout. Persist:

- complete child command;
- stdout/stderr log;
- atomic business `result.json`;
- wrapper JSON with exit code, signal, timeout, elapsed time, and independent `VmHWM`;
- progress/close markers;
- final suite summary.

Recommended worker order:

```text
produce business result
-> atomically write JSON
-> JSON_WRITTEN
-> B5_RUNTIME_RESULT=<same JSON>
-> env.close()
-> ENV_CLOSED
-> app.close()
-> APP_CLOSED if app.close returns
```

Accept `parent_verified_exit0` only when result status is `ok`, JSON and marker match, `ENV_CLOSED` is the last progress marker, the child exits zero, and there is no signal/timeout. Isaac may terminate during `app.close()` before printing `APP_CLOSED`.

Never print a close marker before the corresponding close call returns. Never relax wrapper conditions to make a run pass.

## Selected-Row Invariants

For 2 environments, select a nontrivial subset such as `{0}` and prove env 1 unchanged. For 64 environments use dispersed rows such as `{0,7,63}` and prove all other rows bitwise unchanged.

Check requested, applied store, and direct readback independently for:

- physics mismatch;
- scaffold delay/alpha;
- sensor parameters;
- scene/object/stance state;
- histories, random streams, counters, and episode indices.

Use direct runtime getters for applied physical evidence. Preserve full getter tensors, clone, mutate selected rows, and call setters with selected indices when the API requires the full tensor contract.

Selected reset must not globally advance physics. Source scan and live behavior should both support this claim.

## Isaac Import And API Boundaries

Launch `AppLauncher` before importing modules that depend on `isaaclab`, `omni`, or `carb`. Absolute script execution may put only `scripts/` on `sys.path`; insert and verify the repository root after AppLauncher when the worker needs the local package.

Keep ordinary pytest importable without Isaac. To test pure methods from an Isaac-dependent adapter:

1. parse its source with `ast`;
2. extract only the actual method nodes;
3. compile them into a minimal temporary class with pure dependencies;
4. run mocks against those method bodies.

Do not mock `carb`, inject fake Isaac modules, or start an app during collection.

Known PhysX tensor distinction:

- rigid object `RigidBodyView`: `get_transforms()`;
- articulation `ArticulationView`: `get_root_transforms()`;
- both return `[N,7]` as `xyz + xyzw` and require conversion to internal `wxyz` when applicable.

Do not use `hasattr`/exception probing to hide a known type distinction, and do not use cached `asset.data` as direct applied readback.

## Common Evidence Traps

- `torch.device("cuda")` and an actual `cuda:0` device can fail strict object equality despite correct execution.
- JSON/Python floats and sampled float32 values should not be dictionary keys compared by exact representation.
- `0.02` is not necessarily the exact Python representation of a float32 tensor value.
- `git diff --no-index --check /dev/null <untracked>` returns 1 for content difference; whitespace diagnostics, not status 1 alone, decide the gate.
- `pytest` output truncation is not a pass or failure. Use a wrapper that records exit, signal, last node, elapsed, and memory.
- `/usr/bin/time` may not exist. Use `/proc/<pid>/status`, `resource`, or the repository wrapper.
- Artifact checksum manifests often use relative paths; execute them from the artifact `v1` directory.
- `compileall` writes ignored bytecode. Run it only when explicitly authorized and never treat `pycache` as source evidence.
- A real RTX GPU run is still simulation evidence, not physical hardware validation.
