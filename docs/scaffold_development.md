# Sonic-Based Scaffold Development

This note records the first SomaForce-Cross scaffold engineering slice on the
`dev` branch.

## Current Reuse Boundary

Sonic is reused as the nominal motion scaffold:

```text
Sonic TrackingCommand
  -> nominal joint/body/hand reference
  -> residual_joint_pos_action(...)
  -> a_nom
```

SomaForce-Cross must add force semantics and residual control after this step:

```text
a = a_nom + clip(Delta a_force)
```

The scaffold code must not consume privileged force labels, `p_dir`, `p_mag`,
`P_cross`, or `z_cross`.

## Added Modules

```text
somaforce_cross/scaffold/contracts.py
somaforce_cross/scaffold/trajectory.py
somaforce_cross/scaffold/sonic_adapter.py
somaforce_cross/scaffold/task_trajectories.py
somaforce_cross/scaffold/sonic_motion.py
```

`MotionTrajectoryScaffold` is a pure-Python scaffold used for early tests. It
accepts a G1 29-DOF motion trajectory and returns `ScaffoldOutput.a_nom`.

`SonicScaffoldAdapter` is the Isaac/Sonic bridge. It expects a Sonic manager env
with a `motion` command term and calls Sonic's:

```text
gear_sonic.envs.manager_env.mdp.observations.residual_joint_pos_action
```

to produce `a_nom`.

`task_trajectories.py` provides scaffold-only G1 nominal motion trajectories for
the first scene variants:

- door push;
- door pull;
- box push;
- box pull.

These trajectories provide joint positions, hand references, body references,
timestamps, and `cmd_6d`. They do not model force, contact truth, hinge state, or
box state.

`sonic_motion.py` exports these trajectories into Sonic motion-lib joblib files
with the fields Sonic expects:

```text
root_trans_offset
pose_aa
dof
root_rot
smpl_joints
fps
```

`sonic_env.py` defines the Sonic manager-env binding layer. It generates Hydra
overrides for:

- no-object scaffold task-motion checks;
- rigid-object box rollouts through Sonic's existing `add_object` USD hook;
- door scaffold motion plus an explicit articulation-scene requirement.

This separation is intentional: Sonic's current `add_object` path uses
`RigidObjectCfg`, while the door task needs a hinged articulation scene.

## Scene Config Seeds

```text
configs/scaffold/g1_push_pull_door.yaml
configs/scaffold/g1_push_pull_box.yaml
```

These define the first two scaffold tasks:

- Unitree G1 push/pull door scaffold;
- Unitree G1 push/pull box scaffold.

Both use motion trajectory input and output `a_nom`.

Current scene factories:

```text
make_g1_push_pull_door_scaffold(trajectory=..., interaction_mode="push"|"pull")
make_g1_push_pull_box_scaffold(trajectory=..., interaction_mode="push"|"pull")
```

If `trajectory` is omitted, the factory uses the procedural G1 nominal
trajectory template for the selected interaction mode.

## Environment Status

Verified:

- `isaaclab` conda environment is active;
- Isaac Lab headless app starts and closes;
- `gear_sonic` editable package imports;
- `open3d==0.19.0` is installed and Sonic motion-lib imports pass;
- real Sonic imports pass after Isaac AppLauncher startup:
  `TrackingCommand` and `residual_joint_pos_action`;
- no-door Sonic manager env instantiates with one Unitree G1 environment from a
  synthetic static motion file;
- `motion_cmd.joint_pos`, `residual_joint_pos_action(...)`, and
  `SonicScaffoldAdapter(...).get_output().a_nom` are readable;
- local scaffold tests pass with `python -m pytest -q`.

For a compact gate-by-gate checklist, see `docs/scaffold_verification_matrix.md`.

Sonic import verification:

```text
python scripts/verify_sonic_imports.py
```

Expected output includes:

```text
gear_sonic_import_ok=gear_sonic
tracking_command_import_ok=TrackingCommand
residual_joint_pos_action_import_ok=residual_joint_pos_action
```

Verification command:

```text
PYTHONUNBUFFERED=1 python scripts/verify_sonic_scaffold_env.py
```

Observed output:

```text
env_ok=1
motion_joint_pos_shape=(1, 29)
residual_action_shape=(1, 29)
scaffold_a_nom_shape=(1, 29)
```

Scaffold-only trajectory verification:

```text
python scripts/verify_scaffold_trajectories.py
```

Expected output:

```text
door_push: a_nom=(29,) hand_ref=(2, 3) body_ref=(1, 6) cmd_6d=(6,)
door_pull: a_nom=(29,) hand_ref=(2, 3) body_ref=(1, 6) cmd_6d=(6,)
box_push: a_nom=(29,) hand_ref=(2, 3) body_ref=(1, 6) cmd_6d=(6,)
box_pull: a_nom=(29,) hand_ref=(2, 3) body_ref=(1, 6) cmd_6d=(6,)
```

Sonic motion export verification:

```text
python scripts/export_scaffold_sonic_motion.py \
  --output /tmp/somaforce_g1_scaffold_motions.pkl \
  --num-frames 101 \
  --duration-s 2.0 \
  --fps 50
```

Expected motion keys:

```text
somaforce_g1_door_push
somaforce_g1_door_pull
somaforce_g1_box_push
somaforce_g1_box_pull
```

Task-motion Sonic manager-env checks:

```text
PYTHONUNBUFFERED=1 python scripts/verify_sonic_scaffold_env.py \
  --motion-source scaffold-task \
  --task push_pull_door \
  --interaction-mode push \
  --motion-file /tmp/somaforce_g1_door_push_motion.pkl

PYTHONUNBUFFERED=1 python scripts/verify_sonic_scaffold_env.py \
  --motion-source scaffold-task \
  --task push_pull_box \
  --interaction-mode pull \
  --motion-file /tmp/somaforce_g1_box_pull_motion.pkl
```

Both checks loaded the generated task motion into Sonic's no-door manager env
and produced:

```text
env_ok=1
motion_joint_pos_shape=(1, 29)
residual_action_shape=(1, 29)
scaffold_a_nom_shape=(1, 29)
```

Sonic override inspection:

```text
python scripts/print_sonic_scaffold_overrides.py \
  --task push_pull_box \
  --interaction-mode push \
  --motion-file /tmp/somaforce_g1_box_push_motion.pkl \
  --object-usd-path /path/to/box.usd
```

For box tasks this emits `+manager_env.config.add_object=true` and routes the
USD path through Sonic's rigid-object hook. For door tasks it keeps
`add_object=false` and marks `requires_articulation_scene=true`, because a
hinged door cannot be represented correctly as a rigid object.

Box object smoke verification:

```text
PYTHONUNBUFFERED=1 python scripts/verify_sonic_scaffold_env.py \
  --motion-source scaffold-task \
  --task push_pull_box \
  --interaction-mode push \
  --motion-file /tmp/somaforce_g1_box_push_motion.pkl \
  --generate-default-box-usd
```

Expected object-related output:

```text
object_motion_file=/tmp/somaforce_g1_box_push_motion_object.pkl
rigid_object_names=['object']
object_enabled=True
motion_joint_pos_shape=(1, 29)
residual_action_shape=(1, 29)
scaffold_a_nom_shape=(1, 29)
```

Door articulation smoke verification:

```text
PYTHONUNBUFFERED=1 python scripts/verify_sonic_scaffold_env.py \
  --motion-source scaffold-task \
  --task push_pull_door \
  --interaction-mode push \
  --motion-file /tmp/somaforce_g1_door_push_motion.pkl \
  --generate-default-door-urdf
```

Expected articulation-related output:

```text
requires_articulation_scene=True
door_articulation_attached=True
rigid_object_names=[]
articulation_names=['door', 'robot']
object_enabled=False
door_enabled=True
motion_joint_pos_shape=(1, 29)
residual_action_shape=(1, 29)
scaffold_a_nom_shape=(1, 29)
```

Known environment notes:

- The current Sonic checkout does not include the release motion dataset paths
  referenced by the G1 configs:
  `gear_sonic/data/motion_lib_bones_seed/robot_filtered` and
  `gear_sonic/data/bones_seed_smpl`.
- The verification script therefore writes a synthetic static G1 motion file to
  `/tmp/somaforce_synth_motion.pkl` and overrides Sonic's motion path.
- Sonic robot assets use paths relative to the Sonic checkout root
  (`gear_sonic/data/...`). The verification script infers that root from
  `--sonic-config-dir` and temporarily runs from there.

Numpy decision:

- `gear_sonic` metadata pins `numpy==1.26.4`.
- `isaacsim-kernel 5.1.0.0` requires `numpy==1.26.0`.
- The environment is kept at `numpy==1.26.0` to preserve Isaac Sim compatibility.

Remaining packaging conflicts from `pip check` are accepted for this Isaac/Sonic
rollout path:

- `gear-sonic` declares `numpy==1.26.4`, but Isaac Sim requires `1.26.0`;
- unrelated API packages request newer `starlette`/`uvicorn`, while Isaac-side
  packages currently work with the restored versions.

## Next Engineering Target

Replace the generated smoke assets with task-specific door/box assets and start
collecting scaffold-only rollout diagnostics. The box path supports Sonic's
existing rigid-object USD hook with a generated rigid-body USD and companion
object-motion pkl. The door path supports a generated hinged URDF articulation
hook without routing the door through Sonic's rigid-object path. The scaffold
boundary should remain unchanged: the output is only the nominal scaffold action
`a_nom`.
