# Pretrained HDMI Scaffold Validation

Date: 2026-07-21

This round implements and validates a standalone frozen HDMI `phase=train`
privileged-teacher simulation baseline for `push_door-hand`. It is explicitly
not a deployable policy. The runtime imports neither HDMI nor
`active_adaptation`, and it does not resolve paths in the HDMI checkout.

## Commands

```bash
cd /inspire/hdd/global_user/liumengfan-253108110079/lmf-workspace/somaforce-cross
/opt/miniconda3/envs/isaaclab/bin/python scripts/export_pretrained_hdmi_scaffold.py \
  --hdmi-root /inspire/hdd/global_user/liumengfan-253108110079/lmf-workspace/HDMI \
  --force
env -u PYTHONPATH /opt/miniconda3/envs/isaaclab/bin/python \
  scripts/verify_pretrained_hdmi_isolation.py
/opt/miniconda3/envs/isaaclab/bin/python -m pytest -q
/opt/miniconda3/envs/isaaclab/bin/python -m compileall -q somaforce_cross scripts tests
env PYTHONUNBUFFERED=1 /opt/miniconda3/envs/isaaclab/bin/python \
  scripts/play_pretrained_hdmi_scaffold.py --headless --num-envs 1 --steps 540 \
  --log-interval 50 --require-progress 0.05 \
  --metrics-json artifacts/scaffolds/hdmi_push_door_hand/v1/rollout_metrics.json
```

## Artifact And Contracts

`artifacts/scaffolds/hdmi_push_door_hand/v1/manifest.json` records contract
version, source revision/checksums, dirty-worktree status, file checksums,
license status, and the export boundary. `policy_state.pt` is a flat tensor
state accepted by `torch.load(..., weights_only=True)`; no HDMI class or pickle
module is needed at runtime. `observation_contract.json` records the exact
`command[356]`, `policy[249]`, `object[7]`, `privileged[1714]`, and
`reference_action[23]` fields, normalization and reset semantics.

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
| Door progress | joint `0 -> -2.6163 rad`; task-consistent progress `2.6163 rad`, reference target `1.5 rad` |
| Stability/contact | minimum root height `0.7090 m`; mean support contacts `1.65`; max wrist contact `18.28 N`; max all-body contact `1070.47 N` |
| Action statistics | max abs `4.2702`, mean abs `0.6562` |

The saved rollout evidence is
`artifacts/scaffolds/hdmi_push_door_hand/v1/rollout_metrics.json`.

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
