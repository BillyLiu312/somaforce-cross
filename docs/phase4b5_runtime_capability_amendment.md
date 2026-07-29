# Phase 4B5 Runtime Capability Amendment

Date: 2026-07-29

Status: approved contract amendment for `phase4b5_numeric_v2`. This is a
runtime-capability correction only. It does not start B5B2 environment
integration, PPO, learning, training, or hardware validation.

## Basis

The RTX 4090 Isaac Sim 5.1 selected-row probes used independent two-environment
AppLauncher processes and C1 canonical values. Rigid `RigidBodyView` mass
`[2,1]`, inertia `[2,9]`, COM `[2,7]`, and material `[2,1,3]` setters/readbacks
were finite and exact for env 0 while env 1 remained bitwise unchanged. The
combined order was mass, inertia, fresh COM read, COM xyz with the current
principal-axis quaternion preserved, then material.

Selected object and stance root-pose setters likewise left env 1 unchanged;
stance preserves the actual PhysX float32 readback rather than replacing it
with the requested row. Scaffold delay/alpha internal selected tensors and the
public 90-D `VirtualFTSensor.apply_parameters` roundtrip were also exact with
unselected rows unchanged. These are Isaac-runtime capability results, not
physical-rollout, learning, PPO, deployment, or hardware evidence.

The same audit found no episode-time selected joint-axis/joint-frame setter or
readback in the active door articulation/tensor API. The physical handle frame
also has no per-environment geometry owner: current contact targets derive from
immutable task-spec offsets. Both are `NOT_PER_ENV_SETTABLE`.

## V2 Training Support

`phase4b5_numeric_v2` records the approved amendment metadata and declares:

```text
push_door_hand 0:3   hinge_axis_requires_scene_bank
push_door_hand 5:11  physical_handle_geometry_requires_scene_bank
```

Those strings are metadata, not numeric training fallbacks. The door physical
subfamilies are exactly `mechanism`, `initial_stance`, and `contact_target`.
They map respectively to rows `3:5`, `22:31`, and `31:37`. Rows `0:3` and
`5:11` are bitwise zero for every nominal/C1/C2/C3 training sample and the
critic normalizer rejects a nonzero value before any clipping.

Door contact-target support is now named explicitly and preserves its approved
numeric values: C1 `0.01 m` / `2 deg`, C2 `0.025 m` / `5 deg`, and C3
`0.05 m` / `10 deg`. Friction/damping, object pose, stance, C0, PCG, reward,
sensor support, and all rigid-task support are unchanged.

## Deferred Boundary

A future scene-bank may provide distinct prebuilt hinge/physical-handle
geometries with variant-specific readback. That work is deferred and is not
part of B5B2. No runtime fallback, alias, or automatic v1-to-v2 migration is
accepted by this amendment.
