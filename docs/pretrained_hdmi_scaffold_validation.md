# Pretrained HDMI Scaffold Validation

Date: 2026-07-21

This round validates the existing `push_door-hand` artifact and adds a second
standalone frozen HDMI `phase=train` privileged-teacher simulation baseline for
`push_box`. Neither artifact is a deployable policy. Runtime imports neither
HDMI nor `active_adaptation`, and it does not resolve paths in the HDMI
checkout.

## Commands

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/lmf-workspace/somaforce-cross
/opt/miniconda3/envs/isaaclab/bin/python scripts/export_pretrained_hdmi_scaffold.py \
  --hdmi-root /inspire/hdd/global_user/liumengfan-253108110079/lmf-workspace/HDMI \
  --task push_box --force
env -u PYTHONPATH /opt/miniconda3/envs/isaaclab/bin/python \
  scripts/verify_pretrained_hdmi_isolation.py --task push_box
/opt/miniconda3/envs/isaaclab/bin/python -m pytest -q
/opt/miniconda3/envs/isaaclab/bin/python -m compileall -q somaforce_cross scripts tests
env PYTHONUNBUFFERED=1 /opt/miniconda3/envs/isaaclab/bin/python \
  scripts/play_pretrained_hdmi_scaffold.py --task push_box --case nominal \
  --headless --num-envs 1 --steps 792 --require-progress 1.0 \
  --metrics-json artifacts/scaffolds/hdmi_push_box/v1/rollout_metrics.json
env PYTHONUNBUFFERED=1 /opt/miniconda3/envs/isaaclab/bin/python \
  scripts/play_pretrained_hdmi_scaffold.py --task push_box --case high_mass \
  --headless --num-envs 1 --steps 792 --require-progress 0 \
  --metrics-json artifacts/scaffolds/hdmi_push_box/v1/rollout_metrics_high_mass.json
env PYTHONUNBUFFERED=1 /opt/miniconda3/envs/isaaclab/bin/python \
  scripts/play_pretrained_hdmi_scaffold.py --task push_box --case high_friction \
  --headless --num-envs 1 --steps 792 --require-progress 0 \
  --metrics-json artifacts/scaffolds/hdmi_push_box/v1/rollout_metrics_high_friction.json
```

## Artifact And Contracts

`artifacts/scaffolds/hdmi_push_door_hand/v1/manifest.json` records contract
version, source revision/checksums, dirty-worktree status, file checksums,
license status, and the export boundary. `policy_state.pt` is a flat tensor
state accepted by `torch.load(..., weights_only=True)`; no HDMI class or pickle
module is needed at runtime. `observation_contract.json` records the exact
`command[356]`, `policy[249]`, `object[7]`, `privileged[1714]`, and
`reference_action[23]` fields, normalization and reset semantics.

The push-box contract is `command[356]`, `policy[249]`, `object[10]`,
`privileged[1714]`, and `reference_action[23]`. Its portable privileged encoder
input is `1724`, actor input is `861`, and actor output is `23`. The motion has
25 named joints; mapping into the 29-joint articulation and 23-D action is by
name.

The audited action order is:

```text
left_hip_pitch, right_hip_pitch, waist_yaw,
left_hip_roll, right_hip_roll, waist_roll,
left_hip_yaw, right_hip_yaw, waist_pitch,
left_knee, right_knee,
left_shoulder_pitch, right_shoulder_pitch,
left_ankle_pitch, right_ankle_pitch,
left_shoulder_roll, right_shoulder_roll,
left_ankle_roll, right_ankle_roll,
left_shoulder_yaw, right_shoulder_yaw,
left_elbow, right_elbow
```

These are normalized joint-position actions. The six wrist joints remain out
of the action space. Canonical reference data stays 29-D and uses the explicit
mapping in `action_contract.json`.

## Results

| Check | Result |
| --- | --- |
| HDMI-source deterministic parity (`make_mlp + Actor` mean) | batch 4, max abs error `0.0` at export; isolated loaded artifact `4.77e-7` (target `1e-5`) |
| Frozen/eval parameters | passed; all `requires_grad=False`, `eval()` |
| Import isolation | passed; `active_adaptation_modules=[]`, output `[1,23]` |
| Reset/history/phase/action reset tests | passed in `tests/test_pretrained_hdmi_scaffold.py` |
| Zero-hook | passed; environment receives action bitwise equal to `a_nom` |
| Pytest | `10 passed` after removal of the legacy scaffold test suite |
| Compileall | passed |
| Isaac rollout | 540 control steps, no NaN, no frame/shape error, no fall termination |
| Door progress | joint `0 -> -2.6180 rad`; task-consistent progress `2.6180 rad`, reference target `1.5 rad` |
| Stability/contact | minimum root height `0.7131 m`; mean support contacts `1.66`; max wrist contact `19.43 N`; max all-body contact `1314.81 N` |
| Action statistics | max abs `4.1308`, mean abs `0.5977` |

Push-box results (all 792 control steps, 50 Hz):

| Case | Actual directional progress | Path / final error | Two-hand contact fraction | Wrist force max L/R | Min root height | Stable |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| nominal, 8 kg, friction 0.5 | `1.4674 m` | `1.5118 / 0.0619 m` | `1.0000` | `52.17 / 55.10 N` | `0.7183 m` | yes |
| high mass, 10 kg | `1.4017 m` | `1.4406 / 0.0586 m` | `1.0000` | `49.90 / 54.52 N` | `0.7135 m` | yes |
| high friction, 1.2 | `0.00735 m` | `0.1773 / 1.4376 m` | `0.9708` | `79.13 / 66.11 N` | `0.6868 m` | yes |

All three runs had `nonfinite_count=0`, exact zero-hook equality, and valid
support/contact/action metrics. The physical box moved in nominal and high-mass
conditions; progress is not derived from the `1.4443 m` reference displacement.
The high-friction case is a deliberate mismatch boundary: it maintains a
stable robot and contact signal but the box remains effectively constrained.

The saved rollout evidence is
`artifacts/scaffolds/hdmi_push_door_hand/v1/rollout_metrics.json`.
Push-box nominal and mismatch evidence is stored alongside its artifact as
`rollout_metrics.json`, `rollout_metrics_high_mass.json`, and
`rollout_metrics_high_friction.json`.

## Limits And Licensing

The rollout is a local privileged simulation baseline. It uses simulator object
pose/joint/contact truth in the declared privileged group and must not be called
deployable. A non-privileged `actor_adapt`/finetune export has not been claimed
or validated. A matched statistical rollout comparison against the full HDMI
environment was not completed in this round; the fixed-input source oracle
parity and standalone Isaac rollout did pass.

The audited HDMI checkout (revision
`32282f6dcf26cae70b814d585ceb12cc38aa1b60`) had no root `LICENSE`,
`DATA_LICENSE`, or `NOTICE` at audit time. The local artifact is therefore
private/local-only and must not be redistributed until permissions are
documented. `THIRD_PARTY.md` and the manifest preserve this restriction.
The push-box source reference video is provenance-only (SHA256
`65d39e36ce7dded75e9f3c1513aa4a6641589e763fea28c25289a799e2fd7a71`)
and is not copied into the runtime artifact. GUI playback was not captured in
this headless environment because no X display was available. GUI mode hides
the translucent green reference box by default; pass `--show-reference-box`
to render it separately from the physical box for trajectory debugging.
