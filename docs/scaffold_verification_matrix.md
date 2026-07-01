# Scaffold Verification Matrix

This matrix maps the current SomaForce-Cross scaffold engineering slice to the
requested development gates on the `dev` branch.

For the requirement-by-requirement scaffold audit, see
`docs/scaffold_completion_audit.md`.

## Gates

| Gate | Evidence entry point | Passing signal |
| --- | --- | --- |
| Sonic imports | `python scripts/verify_sonic_imports.py` | `gear_sonic_import_ok=gear_sonic`, `tracking_command_import_ok=TrackingCommand`, `residual_joint_pos_action_import_ok=residual_joint_pos_action` |
| No-door Sonic manager env | `python scripts/verify_sonic_scaffold_env.py` | `env_ok=1`, `motion_joint_pos_shape=(1, 29)`, `residual_action_shape=(1, 29)`, `scaffold_a_nom_shape=(1, 29)` |
| G1 scaffold trajectory contract | `python scripts/verify_scaffold_trajectories.py` | `door_push`, `door_pull`, `box_push`, and `box_pull` each report `a_nom=(29,)` |
| Scaffold-only rollout diagnostics | `python scripts/diagnose_scaffold_rollouts.py` | all four selected variants report `direction_ok=1` and `scaffold_rollout_diagnostics_ok=1` |
| Scaffold rollout recording | `python scripts/record_scaffold_rollout.py --task push_pull_door --interaction-mode push --generate-default-door-urdf --steps 8` | `record_ok=1`, `rollout_traces.npz`, and `summary.json` are written |
| Sonic task motion export | `python scripts/export_scaffold_sonic_motion.py --output /tmp/somaforce_g1_scaffold_motions.pkl` | motion keys include `somaforce_g1_door_push`, `somaforce_g1_door_pull`, `somaforce_g1_box_push`, `somaforce_g1_box_pull` |
| Box object scaffold smoke | `python scripts/verify_sonic_scaffold_env.py --motion-source scaffold-task --task push_pull_box --interaction-mode push --motion-file /tmp/somaforce_g1_box_push_motion.pkl --generate-default-box-usd` | `rigid_object_names=['object']`, `object_enabled=True`, `scaffold_a_nom_shape=(1, 29)` |
| Door articulation scaffold smoke | `python scripts/verify_sonic_scaffold_env.py --motion-source scaffold-task --task push_pull_door --interaction-mode push --motion-file /tmp/somaforce_g1_door_push_motion.pkl --generate-default-door-urdf` | `articulation_names=['door', 'robot']`, `door_enabled=True`, `object_enabled=False`, `scaffold_a_nom_shape=(1, 29)` |
| Unit tests | `python -m pytest -q` | all scaffold tests pass |

## Scope Boundary

The current scaffold stage provides nominal task motion only:

```text
Sonic TrackingCommand
  -> residual_joint_pos_action(...)
  -> ScaffoldOutput.a_nom
```

Door and box scene smoke tests verify that Sonic's manager env can provide
`a_nom` while the scene contains the expected task object representation. They do
not introduce force semantics, privileged contact labels, or residual policy
training into the scaffold.

The generated box USD and door URDF are smoke assets. The next engineering step
is replacing them with task-specific assets and collecting scaffold-only rollout
diagnostics before adding the force-conditioned residual stack.
