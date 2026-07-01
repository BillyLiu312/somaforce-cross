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
```

`MotionTrajectoryScaffold` is a pure-Python scaffold used for early tests. It
accepts a G1 29-DOF motion trajectory and returns `ScaffoldOutput.a_nom`.

`SonicScaffoldAdapter` is the Isaac/Sonic bridge. It expects a Sonic manager env
with a `motion` command term and calls Sonic's:

```text
gear_sonic.envs.manager_env.mdp.observations.residual_joint_pos_action
```

to produce `a_nom`.

## Scene Config Seeds

```text
configs/scaffold/g1_push_pull_door.yaml
configs/scaffold/g1_push_pull_box.yaml
```

These define the first two scaffold tasks:

- Unitree G1 push/pull door scaffold;
- Unitree G1 push/pull box scaffold.

Both use motion trajectory input and output `a_nom`.

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

Replace the synthetic static motion with task motion trajectories for the first
two scaffold scenes, then bind those trajectories into door and box manager-env
variants that still output only the nominal scaffold action `a_nom`.
