# HDMI + OMOMO Scaffold Construction Pipeline

Updated: 2026-07-14

## Selected Phase 0

```text
HDMI motion.npz (door) -------------------------+
                                                 |
retargeted OMOMO NPZ (heavy payload)             |
  joint G1 + object + contact retarget result ---+
                                                 v
                         CanonicalReferenceEpisode
                                                 |
                         ReferenceLibrary (.pt)
                                                 |
                         HDMIReferenceScaffold
                                                 v
                         ScaffoldOutput.a_nom
                         + hand/body/object refs
                         + contact intent/target
                         + phase/confidence
```

The canonical/replay classes do not replace HDMI's PPO trainer. They establish the data and policy boundary used to feed reference-state initialization, robot-object co-tracking, contact rewards, and residual joint-position training in Isaac Lab.

## Source Boundary

HDMI input accepts released robot-object arrays such as:

```text
joint_pos [T, >=29]
joint_vel [T, >=29] optional
body_pos_w [T, B, 3]
body_quat_w [T, B, 4] in wxyz order
body_lin_vel_w/body_ang_vel_w optional
object_contact [T], [T,1], or [T,2]
```

Hand and object body indices are explicit CLI arguments because their ordering belongs to the exact HDMI asset revision.

OMOMO input is not raw SMPL-H. `load_retargeted_omomo_reference` requires output from a joint human-object-to-G1 retargeter:

```text
joint_pos [T,29]
body_pos_w [T,B,3]
body_quat_w [T,B,4]
hand_pose_w [T,2,7]
object_root_pose_w [T,7]
contact_intent [T,2]
contact_confidence [T,2] optional
reference_valid [T] optional
```

The adapter reconstructs object-frame contact targets and validates quaternions, timestamps, masks, finite values, and tensor dimensions. It does not infer payload force, mass, inertia, CoM, or load share.

## Conversion CLI

HDMI example:

```text
python scripts/build_scaffold_reference.py \
  --source hdmi \
  --input /path/to/motion.npz \
  --output /path/to/door_references.pt \
  --episode-id door-push-001 \
  --source-clip-id push_door-hand \
  --task push_pull_door \
  --left-hand-body-index <index> \
  --right-hand-body-index <index> \
  --object-body-index -1
```

Retargeted OMOMO example:

```text
python scripts/build_scaffold_reference.py \
  --source omomo-retargeted \
  --input /path/to/retargeted_clip.npz \
  --output /path/to/payload_references.pt \
  --episode-id payload-001 \
  --source-clip-id omomo-source-clip \
  --retarget-version g1-retarget-v1 \
  --retarget-quality foot_slip_m=0.01 \
  --retarget-quality grasp_error_m=0.02
```

## Code Entry Points

| Component | Entry point |
| --- | --- |
| Canonical schema | `somaforce_cross/scaffold/reference_schema.py` |
| HDMI conversion | `somaforce_cross/scaffold/hdmi_adapter.py` |
| OMOMO retarget-result conversion | `somaforce_cross/scaffold/omomo_adapter.py` |
| Library storage/filtering | `somaforce_cross/scaffold/reference_library.py` |
| Nominal policy boundary | `somaforce_cross/scaffold/hdmi_scaffold.py` |
| Door config | `configs/scaffold/g1_push_pull_door.yaml` |
| Heavy-payload config | `configs/scaffold/g1_heavy_payload.yaml` |

## Acceptance Gates

- HDMI door reference converts and deterministically replays.
- At least one jointly retargeted OMOMO clip passes all schema and contact checks.
- Door easy-distribution scaffold success reaches at least 80% after HDMI-style training.
- Payload scaffold completes lift-off, carry, and set-down on at least three reference variants.
- Hinge/handle and mass/CoM mismatch produce interpretable failures rather than simulator instability.
- All source revisions, clip IDs, retarget versions, and quality metrics are preserved.
- Force-rich supervision is generated only after reference scaffold construction in Isaac Lab.

## Legacy Compatibility

`sonic_adapter.py`, `sonic_motion.py`, `sonic_env.py`, and the Sonic verification scripts remain for regression and tracker-baseline experiments. They are not the default scaffold construction path.
