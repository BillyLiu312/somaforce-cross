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
- Sonic training dependencies mostly import;
- local scaffold tests pass with `python -m pytest -q`.

Known environment issue:

- `TrackingCommand` import reaches `gear_sonic.utils.motion_lib.torch_humanoid_batch`,
  which imports `open3d`.
- `open3d` is not declared in Sonic's `pyproject.toml`.
- Installing `open3d` from PyPI attempted to download a 447.7MB wheel and was
  cancelled after repeated timeouts.
- The current Sonic checkout does not include the release motion dataset paths
  referenced by the G1 configs:
  `gear_sonic/data/motion_lib_bones_seed/robot_filtered` and
  `gear_sonic/data/bones_seed_smpl`.
- Until those data paths exist, a real no-door Sonic manager env cannot be
  instantiated from `sonic_release.yaml`.

Numpy decision:

- `gear_sonic` metadata pins `numpy==1.26.4`.
- `isaacsim-kernel 5.1.0.0` requires `numpy==1.26.0`.
- The environment is kept at `numpy==1.26.0` to preserve Isaac Sim compatibility.

## Next Verification Target

After `open3d` is installed or Sonic lazy-imports it, run:

```text
from isaaclab.app import AppLauncher
launcher = AppLauncher(headless=True)
app = launcher.app
from gear_sonic.envs.manager_env.mdp.commands import TrackingCommand
from gear_sonic.envs.manager_env.mdp.observations import residual_joint_pos_action
```

Then instantiate a no-door Sonic manager env and verify:

```text
motion_cmd = env.command_manager.get_term("motion")
motion_cmd.joint_pos
residual_joint_pos_action(env, command_name="motion")
SonicScaffoldAdapter(env).get_output().a_nom
```
