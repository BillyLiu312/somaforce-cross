# Pretrained HDMI Scaffold Validation

Date: 2026-07-22

The repository contains four independently selectable, standalone HDMI-derived
scaffold artifacts:

- `push_door_hand`
- `push_box`
- `move_suitcase`
- `move_largebox`

Each is a frozen `phase=train` privileged-teacher simulation baseline. All are
marked `deployable: false` and local/private-only because redistribution
permissions for the source code, weights, motion, and assets remain
undocumented. Runtime rollout and rendering do not import HDMI or resolve an
HDMI checkout.

## Current Validation Boundary

The following validation remains part of the repository:

- manifest-driven observation, action, network, reference, and asset contracts;
- fixed-input HDMI source-oracle parity with a maximum absolute error of `1e-5`;
- frozen/eval policy loading with tensor-only `policy_state.pt`;
- import isolation from HDMI and `active_adaptation`;
- exact 29-D reference to 23-D action mapping and zero-hook action semantics;
- recursive rejection of absolute filesystem paths in manifests;
- exact `SHA256SUMS` coverage of every frozen artifact file, excluding mutable
  rollout metrics.

Matched closed-loop parity against the complete HDMI environment is not yet
claimed. No rollout metrics are currently committed: historical fixed matrices,
aggregate payload results, and their old commands were removed so they cannot
be mistaken for evidence produced by the unified interface.

## One-Condition Rollout

One invocation runs exactly one task and one caller-named condition. The
condition is a filename/metadata slug; explicit numeric arguments are the
authority for its physical evolution.

```bash
/opt/miniconda3/envs/isaaclab/bin/python \
  scripts/rollout_pretrained_hdmi_scaffold.py \
  --task push_door_hand \
  --condition high_friction \
  --door-friction 10.0 \
  --door-damping 0.55 \
  --headless
```

Unless `--metrics-json` overrides the destination, the command writes exactly
one file beside the selected artifact:

```text
artifacts/scaffolds/hdmi_<task>/v1/rollout_metrics_<condition>.json
```

The JSON is self-contained: `task`, `condition`, measured rollout fields, and a
task-specific `settings` object record the actual values used. It never embeds
multiple conditions or merges with an existing aggregate. `--no-metrics`
suppresses file output and is reserved primarily for the render wrapper.
Rollout metrics are mutable run evidence, not frozen artifact payloads. They are
therefore excluded from `SHA256SUMS`, and rollout never rewrites that file.

Task evolution arguments include:

- door: `--door-friction`, `--door-damping`;
- rigid objects: `--object-mass`, `--object-friction`,
  `--object-com-offset`, `--initial-object-xy`, `--initial-object-yaw`, and
  `--contact-target-offset`;
- controller/runtime: `--steps`, `--delay`, and `--alpha`.

Defaults are task-specific nominal values. Non-nominal conditions default to
numerical-validity checks only unless explicit `--require-progress` or
`--require-contact-fraction` thresholds are supplied.

## One-Condition Render

The render wrapper accepts the same task and physical evolution arguments and
produces exactly one MP4 without writing rollout metrics:

```bash
/opt/miniconda3/envs/isaaclab/bin/python \
  scripts/render_pretrained_hdmi_scaffold.py \
  --task move_suitcase \
  --condition heavy \
  --object-mass 3.0 \
  --output videos/move_suitcase/heavy.mp4
```

After rollout, it verifies H.264 codec, requested resolution, first-frame
readability, and nonblank pixels. Rendering streams frames to the encoder rather
than buffering a full episode in memory.

## Lightweight Verification

Smoke outputs should go to `/tmp` so validation does not repopulate artifact
directories:

```bash
/opt/miniconda3/envs/isaaclab/bin/python \
  scripts/rollout_pretrained_hdmi_scaffold.py \
  --task push_door_hand --condition smoke --headless --steps 2 \
  --require-progress 0 \
  --metrics-json /tmp/somaforce_rollout_smoke.json

/opt/miniconda3/envs/isaaclab/bin/python \
  scripts/render_pretrained_hdmi_scaffold.py \
  --task push_door_hand --condition smoke --steps 2 \
  --width 640 --height 480 \
  --output /tmp/somaforce_render_smoke.mp4
```

Repository-level checks are:

```bash
/opt/miniconda3/envs/isaaclab/bin/python -m pytest -q
/opt/miniconda3/envs/isaaclab/bin/python -m compileall -q \
  somaforce_cross scripts tests
for artifact in artifacts/scaffolds/*/v1; do
  (cd "$artifact" && sha256sum -c SHA256SUMS)
done
```

The rollout smoke proves launch, artifact loading, simulation stepping, metrics
serialization, and task-specific settings recording. The render smoke proves
the same launch path plus camera capture, encoding, and video verification; it
is not a full scientific rollout result.

## Batch Matrix

The batch orchestrator expands the versioned case matrix without adding
multi-condition behavior to either single-task entry:

```bash
/opt/miniconda3/envs/isaaclab/bin/python \
  scripts/batch_rollout_and_render_pretrained_hdmi_scaffolds.py
```

The default `configs/pretrained_hdmi_rollout_cases.json` contains all 14 current
task/condition pairs. Each case carries explicit evolution values and is passed
unchanged to both rollout and render. Outputs are organized as:

```text
artifacts/scaffolds/hdmi_<task>/v1/rollout_metrics_<condition>.json
videos/<task>/<condition>.mp4
```

Task directories are siblings. Research categories such as `heavy_payload` are
metadata only and do not introduce parent directories under `videos/`.

Execution is sequential by default because independent Isaac processes can
otherwise overcommit GPU memory. `--resume` skips existing outputs;
`--fail-fast` stops at the first failure. Without `--fail-fast`, the script
continues through the matrix and returns nonzero after reporting all failures.
