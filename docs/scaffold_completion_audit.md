# Scaffold Completion Audit

> Historical audit: this file records the completed Sonic compatibility slice as of 2026-07-02. Sonic is now a legacy baseline. The selected Phase 0 pipeline is documented in `docs/hdmi_omomo_scaffold_pipeline.md`.

Audit date: 2026-07-02

This audit maps the requested SomaForce-Cross scaffold development order to the
current `dev` branch state. The scope is the Sonic-based nominal scaffold only:
motion trajectory input, Sonic/TrackingCommand reuse, and `a_nom` output for the
first Unitree G1 door/box push-pull tasks.

## Requirement Status

| Requirement | Status | Current evidence |
| --- | --- | --- |
| Work happens on `dev` | Passed | `git status --short --branch` reports `## dev...origin/dev`. |
| Sonic dependency is usable | Passed | `PYTHONUNBUFFERED=1 python scripts/verify_sonic_imports.py` imports `gear_sonic`, `TrackingCommand`, and `residual_joint_pos_action` after Isaac AppLauncher startup. |
| No-door Sonic manager env can expose Sonic command/action data | Passed | `PYTHONUNBUFFERED=1 python scripts/verify_sonic_scaffold_env.py` reports `env_ok=1`, `motion_joint_pos_shape=(1, 29)`, `residual_action_shape=(1, 29)`, and `scaffold_a_nom_shape=(1, 29)`. |
| SomaForce-Cross has scaffold-only engineering modules | Passed | `somaforce_cross/scaffold/` contains contracts, trajectory scaffolds, Sonic adapter, Sonic motion export, Sonic env binding, scene specs, and rollout diagnostics. |
| Unitree G1 push/pull door scaffold exists | Passed | `make_g1_push_pull_door_scaffold(...)` and `make_g1_push_pull_door_trajectory(...)` produce G1 29-DOF nominal actions for push and pull. |
| Unitree G1 push/pull box scaffold exists | Passed | `make_g1_push_pull_box_scaffold(...)` and `make_g1_push_pull_box_trajectory(...)` produce G1 29-DOF nominal actions for push and pull. |
| Scaffold input is motion trajectory | Passed | `MotionTrajectory` is the scaffold contract and Sonic motion export writes the selected trajectories into Sonic-compatible motion-lib files. |
| Scaffold output is `a_nom` | Passed | `MotionTrajectoryScaffold.at_frame/at_time(...)` and `SonicScaffoldAdapter.get_output()` return `ScaffoldOutput.a_nom`. |
| Box scene hook is represented in Sonic/Isaac | Passed | Box smoke uses Sonic's rigid-object USD path and reports `rigid_object_names=['object']`, `object_enabled=True`, `scaffold_a_nom_shape=(1, 29)`. |
| Door scene hook is represented as articulation | Passed | Door smoke adds a hinged URDF articulation and reports `articulation_names=['door', 'robot']`, `door_enabled=True`, `object_enabled=False`, `scaffold_a_nom_shape=(1, 29)`. |
| Scaffold remains separate from SomaForce-Cross force semantics | Passed | Scaffold modules do not consume force semantics, privileged contact truth, `p_dir`, `p_mag`, `P_cross`, or `z_cross`. |

## Current Verification Commands

```text
PYTHONUNBUFFERED=1 python scripts/verify_sonic_imports.py
PYTHONUNBUFFERED=1 python scripts/verify_sonic_scaffold_env.py
python scripts/verify_scaffold_trajectories.py
python scripts/diagnose_scaffold_rollouts.py
python scripts/export_scaffold_sonic_motion.py --output /tmp/somaforce_g1_scaffold_motions.pkl
PYTHONUNBUFFERED=1 python scripts/verify_sonic_scaffold_env.py --motion-source scaffold-task --task push_pull_box --interaction-mode push --motion-file /tmp/somaforce_g1_box_push_motion.pkl --generate-default-box-usd
PYTHONUNBUFFERED=1 python scripts/verify_sonic_scaffold_env.py --motion-source scaffold-task --task push_pull_door --interaction-mode push --motion-file /tmp/somaforce_g1_door_push_motion.pkl --generate-default-door-urdf
python -m pytest -q
python -m compileall -q scripts somaforce_cross
```

## Boundary

The completed scaffold stage is not the force-conditioned residual policy. It
does not train force semantics, read deployable wrist F/T histories, or consume
privileged contact labels. It provides the historical Sonic-style nominal task
motion path:

```text
motion trajectory
  -> Sonic TrackingCommand / scaffold trajectory
  -> residual_joint_pos_action(...) or trajectory action conversion
  -> ScaffoldOutput.a_nom
```

The next research-engineering stage is to replace the generated smoke assets
with task-specific assets and collect scaffold-only rollout diagnostics before
adding the virtual wrist F/T model and cross semantic force residual stack.
