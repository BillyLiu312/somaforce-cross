# Phase 4B4A Three-Task Read-Only Isaac Preflight

日期：2026-07-28

状态：**建议项目所有者 Admission 后冻结为 Phase 4B4B 合同**。本报告只审计
`push_box`、`move_suitcase`、`move_largebox`，没有实现 adapter、环境、reward、
mismatch、PPO 或训练，也没有进入 Phase 4B5。

## 1. 结论与证据边界

结论为 **Accept With Gates**：

- 三个 checksummed artifact、资产、reference、23-D normalized action 合同和
  name-based mapping 均可解析，三个 `SHA256SUMS` 全量通过。
- 三任务分别完成 1-env 和 64-env、6 control-step、RTX 4090 headless golden
  bounded smoke。所有 64-env 运行均有 64 个唯一 origin；`a_nom`、applied
  action、joint target、scaffold history、robot/object/reference state 均 finite，
  shape/device/dtype 正确。
- 现有 door-only `DirectRLEnv` 中的 observation/reset/action-history/C0 composition
  可以泛化；scene object、object actuation、reference reset、progress/contact/
  stability/failure 和 task physics row 必须由 adapter 持有。
- 四任务继续共用 policy `[B,668]`、实际 residual actor `[B,220]`、critic
  `[B,845]`、semantic target `[B,31]`，不得加入 task/object identity 或
  task-specific residual network。
- 本轮 Isaac 结果仅证明 unchanged golden HDMI runtime 的短时、有限、数值稳定
  初始化和向量化执行。它不是 residual C0 parity、task completion、learning、
  robustness、部署或硬件证据。

以下仍是 Admission 或更晚 phase 的开放门：

- 本报告建议的 rigid-task adapter、通用环境和通用 trace 尚未实现，所以三任务
  residual C0 closed-loop parity 仍未证明。
- 所有当前 artifact 都是 `phase=train`、`deployable=false` 的 privileged HDMI
  teacher simulation baseline，不能作为 deployable scaffold 证据。
- mismatch 分布、reward 权重/scale/clip、policy/critic normalizer、curriculum
  概率、PPO 和训练值均未批准。本报告不提供或暗示这些数值。

## 2. Git 与只读边界

入口门禁实测：

```text
branch: v1
HEAD: 7f1a96f0a8bd21c8261759e3c47446a5e920ab87
live origin refs/heads/v1: 7f1a96f0a8bd21c8261759e3c47446a5e920ab87
status before report: clean
tracking: ## v1...origin/v1
```

本轮唯一允许且实际新增的仓库文件是：

```text
docs/phase4b4a_multitask_preflight.md
```

没有 checkout、reset、commit 或 push。现有源码、测试、artifact、golden
runtime、资产和既有文档均保持只读。`/tmp/phase4b4a_golden_smoke.py`、
`/tmp/somaforce_phase4b4a/*` 是未提交诊断工具和证据。

## 3. 审计来源

完整读取：

```text
AGENTS.md
docs/v1_force_residual_plan.md
docs/v1_force_residual_implementation_plan.md
docs/pretrained_hdmi_scaffold_rules.md
docs/phase4a_contract_audit.md
docs/phase4b3a_door_preflight.md
docs/scaffold_rollout_results.md
somaforce_cross/envs/residual_env.py
somaforce_cross/envs/residual_env_cfg.py
somaforce_cross/envs/task_adapter.py
somaforce_cross/envs/task_adapters/push_door_hand.py
scripts/smoke_phase4_door_env.py
somaforce_cross/scaffold/pretrained_hdmi.py
somaforce_cross/scaffold/pretrained_hdmi_isaac.py
```

并检查三个 artifact 的 `manifest.json`、`action_contract.json`、
`observation_contract.json`、`reference/meta.json`、`reference/motion.npz`、
`SHA256SUMS`、资产以及当前 Phase 4 tests。合同事实以当前 HEAD、artifact 和
runtime source 为准，历史 rollout JSON 只作为 task diagnostic 证据。

## 4. Artifact、asset、reference 与 action 合同

### 4.1 三任务事实

| Task | Artifact / object | Reference | Contact / reset facts |
| --- | --- | --- | --- |
| `push_box` | `hdmi_push_box/v1`; `rigid_object`; `assets/box.usd`; body `box`; nominal mass 8.0 kg 来自 resolved `HDMITaskSpec`/runtime | 792 frames, 50 Hz, 25 joint names | two targets `(0,-0.2,0.8)`, `(0,0.2,0.8)`; EEF offsets both `(0.1,0,0)` |
| `move_suitcase` | `hdmi_move_suitcase/v1`; `rigid_object`; `assets/suitcase.usd`; body `suitcase`; manifest nominal mass 1.5 kg | 472 frames, 50 Hz, 29 joint names | targets `(-0.1,0.18,0.25)`, `(-0.1,-0.18,0.25)`; EEF offsets both `(0.05,0,0)`; wrist-yaw reset `-0.4/+0.4` rad |
| `move_largebox` | `hdmi_move_largebox/v1`; `rigid_object`; `assets/largebox.urdf` + local `largebox.obj`; body `largebox_link`; nominal mass 1.0 kg | 199 frames, 50 Hz, 29 joint names | asymmetric targets `(-0.027635,0.244158,0.099234)`, `(0.198793,-0.151816,0.149164)`; EEF offsets both `(0.05,0,0)`; wrist-yaw reset `-0.4/+0.4` rad |

Suitcase 的 1.2-1.8 kg historical training range 来自 current task spec 和
governing scaffold rules，但该字段没有写入 suitcase manifest；push-box 的
`object_asset_file`/nominal mass 也由 current task-spec fallback 补齐。这些是
manifest completeness 风险，不得在 4B4B 静默改 artifact。Largebox manifest
明确记录 0.8-1.2 kg range。任何 range 都不是本轮批准的 mismatch 分布。

关键 immutable 文件 SHA256：

| Task | Policy | Reference motion | Object asset |
| --- | --- | --- | --- |
| `push_box` | `60ede9e41c5e03ac745d8e7cdc6c2cc5e88a42f68bf588fccd17f90c5906efb7` | `8921f589f8470d25ab1dd0fa65f9b9057676ab122fdcc8d87f43d9c527ed461e` | box USD `5c299ef74daeaebe1e5be13edc0f8e2352cb9da2f09600b919d89a08a9e2147e` |
| `move_suitcase` | `9f2f389a454ef770f2cc9d2b016b1f36ccc575bbc3680b96048b4649bb72d605` | `d6d08c5792fc7396c89629876d91cfec26404fd803ccb9ffc2439c66ce51f3cf` | suitcase USD `d3c25a338fffa58ddfef084fe2cff06023dead787ab8a3c2aaa1e3a08925cf6b` |
| `move_largebox` | `ef768885e23277d5513e186892f9a902f427fe05963f5165015ecac3c5885da8` | `1e79ff64f495200afb4830468a3a050328f835a6b7e0c336a874722f8849d941` | URDF `e3281a96e9b6aff6988ae4fc6d219f61751bebdc4a4628ae3045eeb4e9aa99c4`; OBJ `dbcc11281f62e9226f49165252375080d2e490e5a3f0ab6ba917acbd8f7abc1c` |

三份 artifact 的每个 `SHA256SUMS` entry 均为 `OK`。Golden runtime source
保持：

```text
somaforce_cross/scaffold/pretrained_hdmi_isaac.py
SHA256 e924dadeb945278ee85fad82e31be56f78bce4e62f5bb9f575f5938f6c4868a0
```

### 4.2 共用 23-D action order

三个 artifact 的 normalized `a_nom` 均按以下顺序：

```text
 0 left_hip_pitch_joint          1 right_hip_pitch_joint
 2 waist_yaw_joint               3 left_hip_roll_joint
 4 right_hip_roll_joint          5 waist_roll_joint
 6 left_hip_yaw_joint            7 right_hip_yaw_joint
 8 waist_pitch_joint             9 left_knee_joint
10 right_knee_joint             11 left_shoulder_pitch_joint
12 right_shoulder_pitch_joint   13 left_ankle_pitch_joint
14 right_ankle_pitch_joint      15 left_shoulder_roll_joint
16 right_shoulder_roll_joint    17 left_ankle_roll_joint
18 right_ankle_roll_joint       19 left_shoulder_yaw_joint
20 right_shoulder_yaw_joint     21 left_elbow_joint
22 right_elbow_joint
```

相同 order 的 action scale 是：

```text
[0.55,0.55,0.55,0.35,0.35,0.44,0.55,0.55,0.44,0.35,0.35,
 0.44,0.44,0.44,0.44,0.44,0.44,0.44,0.44,0.44,0.44,0.44,0.44]
```

所有 reference mapping 必须按 name 解析并验证 23 个名称唯一存在。冻结结果：

```text
push_box 25-name reference -> action indices:
[0,6,12,1,7,13,2,8,14,3,9,15,20,4,10,16,21,5,11,17,22,18,23]

move_suitcase / move_largebox 29-name reference -> action indices:
[0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25]
```

禁止按 reference 索引位置假定 action order，也禁止把 25/29-D reference 直接
当作 23-D action。唯一物理 target 公式为：

```text
joint_target = default_joint_pos + applied_normalized_action * action_scale
```

delay/smoothing 在 normalized 坐标中进行，scaling 只在此处执行一次。

## 5. 推荐 4B4B 架构边界

### 5.1 通用环境所有权

`SomaForceDoorResidualEnv` 应泛化为 manifest/task-spec 驱动的单一
`DirectRLEnv` 实现，同时保留 door class/config alias 以守住已接纳 regression。
环境共用并继续拥有：

- frozen `PretrainedHDMIScaffold.eval()` inference；
- normalized delay/smoothing/action sink 和一次 physical scaling；
- wrench source、tare、VirtualFT、contact detector、contact ramp；
- wrist/nominal/executed histories、parameter store、per-env RNG；
- semantic pipeline、C0 authority、observation bundle、done plumbing；
- selected reset coordinator 和通用 trace serializer/comparator。

以下 current door-specific 项必须泛化：

```text
SomaForceDoorResidualEnv / SomaForceDoorResidualEnvCfg / DoorC0SmokeProfile
DOOR_ARTIFACT and scene["door"] lookup
NormalizedDoorActionSink class name
door_nominal_physics_mismatch()
_apply_action() 内的 door friction/damping effort
door-only nonfinite/object state selection
door_root/joint/effort trace field names
scripts/smoke_phase4_door_env.py 的固定 task/artifact/trace metadata
```

以下必须留在 adapter，不得在 shared residual network 或 task-name branch 中实现：

- scene object lookup、rigid/articulation reset 和 object actuation；
- reference name mapping、object/contact geometry 和 frozen scaffold observation；
- task progress、success diagnostic、contact truth、stability/failure、
  reference-exhausted；
- task nominal physics row、object state、support contacts 和 task diagnostics；
- door hinge/friction/damping/handle facts与 rigid mass/CoM/inertia/friction facts。

`PretrainedHDMIIsaacRuntime` 继续只作 unchanged golden runtime。不得 subclass、
修改或把其 global reset/scalar `reference_step`/control-loop `step()` 作为
DirectRLEnv owner。

### 5.2 四任务 TaskAdapter 统一接口

四任务 adapter 均应提供并验证：

```text
task_name / task_spec / object_kind / reference / reference_step
validate_reset(env_ids)
write_scene_reset(env_ids)
reset(env_ids)
advance()
apply_object_action()
build_scaffold_observation() -> HDMIObservationBatch
proprio() -> [B,64]
wrist_twist_base_yaw() -> [B,2,6]
build_object_state() -> [B,16]
nominal_physics_row() -> [B,39]
expected_contact() -> [B,2]
contact_truth() -> [B,2]
support_contact_count() -> [B,1]
progress_signals() -> TaskProgressSignals
```

Adapter selection只能改变 data/geometry/physics/progress 语义，不能选择另一个
residual actor、critic、semantic encoder 或 tensor schema。

## 6. 四任务 observation 与 privilege 合同

全部环境 tensor 必须是 environment device 上 finite `torch.float32`，除内部
index/boolean state 外。冻结结构保持：

```text
wrist_tokens       [B,2,16,14] = 448
proprio            [B,64]
a_nom_history      [B,23,3] = 69
previous_a_total   [B,23]
z_cross            [B,64]
policy             [B,668]

actual actor input = z_cross + proprio + a_nom_history + previous_a_total
                   = 64 + 64 + 69 + 23 = [B,220]

critic = policy 668 + object_state 16 + physics 39 + scaffold 2
       + sensor 90 + progress 4 + contact 17 + stability 9
       = [B,845]

semantic target = clean_wrench 12 + p_dir 13 + p_mag 5 + weight 1
                = [B,31]
```

`policy [B,668]` 是 named structured bundle，不是批准的 opaque MLP input。
Residual actor 只接收 `[B,220]`，其中 `z_cross` 是唯一 force-semantic latent。

以下不得出现在 policy 或 actor input：

```text
task ID / task name / object kind / object category / object identity
HDMI command[356] 或 privileged teacher observation
object pose、velocity、mass、CoM、inertia、hinge/mechanism state
simulator contact truth、support/foot contact、foot wrench
clean wrench、semantic targets
progress、success、failure、stability、reference truth
physics/scaffold/sensor mismatch rows
```

Artifact-specific privileged HDMI input只允许 frozen base scaffold 生成 `a_nom`；
它与 residual actor observation 必须是两个独立数据路径。Critic/target 必须以独立
named keys 交付，不能依赖从 policy 尾部切片的约定。

## 7. Rigid-object `object_state [B,16]`

对三个 rigid object，冻结映射为：

```text
0:3    p_object - p_robot_root，经 q_robot_root^-1 旋转到完整 robot-root frame
3:7    q_rel = conjugate(q_robot_root) * q_object_root，quaternion order wxyz
7:10   object root world linear velocity，经 q_robot_root^-1 旋转
10:13  object root world angular velocity，经 q_robot_root^-1 旋转
13:16  [mechanism_q, mechanism_qd, mechanism_applied_torque] = [0,0,0]
```

这里的 twist 保持 current door adapter 语义：表达在 robot-root frame 中的 object
root twist，不额外减去 robot twist。位置使用 object root，不使用接触 target、
body COM 或 reference object。Rigid task 的 mechanism 三槽必须 bitwise zero；
door adapter 继续在三槽中放 door joint q/qd/applied torque。

## 8. 三任务 nominal 39-D C0 physics row

### 8.1 固定字段与单位

Admission 后 39-D 顺序冻结为：

| Indices | Field | Nominal rigid semantics |
| --- | --- | --- |
| `0:3` | hinge-axis error xyz, rad | zero |
| `3` | mechanism friction | zero |
| `4` | mechanism damping | zero |
| `5:11` | handle/contact-frame pose error xyz + rotvec, m/rad | zero |
| `11` | applied object mass, kg | task nominal mass |
| `12:15` | applied CoM offset xyz from artifact nominal, m | zero |
| `15:21` | applied inertia `[Ixx,Iyy,Izz,Ixy,Ixz,Iyz]`, kg m2 | read back from PhysX after nominal mass scaling |
| `21` | applied static/dynamic surface friction | 0.5 from golden runtime nominal constructor; both equal |
| `22:28` | initial object pose error xyz + rotvec, m/rad | zero |
| `28:31` | initial stance error x,y,yaw, m/rad | zero |
| `31:37` | contact-target pose error xyz + rotvec, m/rad | zero |
| `37:39` | left/right load-share error | zero |

Inertia 必须从 `root_physx_view.get_inertias()` 的 row-major 3x3 readback 取
`[0,4,8,1,2,5]`，并验证对称项；不能把资产原始 mass 下的 inertia、训练 range
或手写近似值传入 parameter store。CoM row 表示 constructor 的 applied offset，
所以 nominal 是零；不得误填 PhysX local COM pose。

本次 RTX 4090 nominal readback 得到的完整 rows 为：

```text
push_box:
[0,0,0,0,0,0,0,0,0,0,0,8.0,0,0,0,
 0.8533333539962769,1.0933332443237305,1.0933332443237305,0,0,0,
 0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

move_suitcase:
[0,0,0,0,0,0,0,0,0,0,0,1.5,0,0,0,
 0.0312499962747097,0.02499999850988388,0.016249999403953552,0,0,0,
 0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

move_largebox:
[0,0,0,0,0,0,0,0,0,0,0,1.0,0,0,0,
 0.0020000000949949026,0.0020000000949949026,0.0020000000949949026,0,0,0,
 0.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
```

这些 rows 只批准 nominal C0 parity 输入。不得从 stored rollout 的 light/heavy/
stress cases 推导随机分布、normalizer 或 curriculum。

## 9. Progress、contact、stability、failure 与 exhaustion

本节的直接 source evidence 是
`PretrainedHDMIIsaacRuntime._rollout_push_box()`、
`PretrainedHDMIIsaacRuntime._rollout_payload()`、`build_observation()`、
`step()`、`HDMIMotionReference.frames()` 和 current
`PushDoorHandTaskAdapter.progress_signals()`。其中 source 已实现的是 rollout
metric/failure/contact/reference-clamp 语义；把这些转换成 per-env adapter
`progress_delta/success/reference_exhausted` 是本报告提交 Admission 的新合同，
不是声称当前已有的环境行为。

### 9.1 `push_box`

Source-backed geometry：

```text
d_ref = ref_final_object_xy - ref_initial_object_xy
L_ref = norm(d_ref), require L_ref > 0
u_ref = d_ref / L_ref
s = dot(current_object_xy - reset_object_xy, u_ref)
progress = s / L_ref
progress_delta = progress - previous_progress
success = float(s >= L_ref)  # diagnostic only
```

Contact 与 done：

```text
expected_contact: reference object_contact scalar at clamped reference_step,
                  duplicated into fixed [left,right] slots
contact_truth: each filtered wrist-to-object force_matrix_w norm > 1 N
support_contact_count: ankle net force norm > 1 N, summed
stability_margin: root_height - 0.45 m
failure: root_height < 0.45 m
reference_exhausted: reference_step >= 792
```

`success` 不自动 terminated。Tracking/yaw/path error 继续作为 adapter diagnostic，
不在 4B4B 发明 weighted reward。

### 9.2 `move_suitcase` 与 `move_largebox`

两任务共用 payload progress 算法，reference position 均先去掉 per-env origin：

```text
reference_path_cumulative[0] = 0
reference_path_cumulative[t] = sum(norm(ref_pos[i]-ref_pos[i-1]))
nearest = argmin_t norm(current_object_local_pos - ref_pos[t])
s = max(previous_s, reference_path_cumulative[nearest])
progress = s / reference_path_cumulative[-1]
progress_delta = progress - previous_progress
lift_latched |= max_object_z_since_reset - reset_object_z >= 0.10 m
set_down_error = norm(current_object_local_pos - ref_final_pos)
success = reference_exhausted AND lift_latched AND set_down_error <= 0.25 m
          AND NOT failure
```

Contact 和逐任务 done：

```text
expected_contact: reference object_contact scalar duplicated to [left,right]
contact_truth: filtered left/right wrist-to-object force_matrix_w norm > 1 N
support_contact_count: ankle net force norm > 1 N, summed
stability_margin: root_height - 0.25 m
failure: root_height < 0.25 m

move_suitcase reference_exhausted: reference_step >= 472
move_largebox reference_exhausted: reference_step >= 199
```

特别冻结：`move_suitcase` failure boundary 是严格 `< 0.25 m`，不是 door/push-box
的 `< 0.45 m`。0.10 m lift 和 0.25 m set-down 是 source rollout diagnostic；
orientation/carry errors保留为 diagnostics，不作为本轮未批准 reward。

Door 保持其已接纳 adapter 语义，包括 signed door-joint progress、0.45 m failure
和 current `reference_step + max(FUTURE_STEPS) >= length` exhaustion。不同任务的
exhaustion 必须留在 adapter，不能硬编码为一个 door rule。

特别地，current `_rollout_push_box()` / `_rollout_payload()` 不使用 door 的
`+32` early exhaustion，而是让 requested rollout 到各自 reference length，future
frames 由 `HDMIMotionReference.frames()` clamp。因此 rigid adapter 的
`reference_step >= length` 是对当前 golden full-length 行为的明确冻结；若
Admission 不接受这一差异，必须先停下重新定义，不能默默套用 door 条件。

Shared done 仍为：

```text
terminated = nonfinite_state OR adapter.failure
time_outs  = max_episode_steps OR adapter.reference_exhausted
done       = terminated OR time_outs
```

## 10. Selected reset、history 与 action ownership

对唯一、合法、device-local `env_ids`，4B4B 必须在第一次 mutation 前完成所有
input/hook/parameter validation，然后只改变 selected rows：

1. 保存 terminal observation、metrics、`terminated`、`time_outs`。
2. 写入 selected robot/object frame-zero state；不得调用 global `sim.step()`。
3. reset adapter reference/progress/lift state和 wrench source cache。
4. reset frozen scaffold state history，previous scaffold action 清零，phase/step 为零。
5. reset normalized action sink、tare、sensor、detector、ramp 和三个 histories。
6. C0 的 initial wrist frame 固定为全零 `[selected,2,14]`，广播到全部 16 slots；
   source valid=false、sensor quality/contact probability=0，不能混入 reset 前 frame。
7. 计算 selected current `a_nom`；nominal history slot 0 写 current `a_nom`，older
   slots为零；executed history/`previous_a_total` 全零。
8. counters/latches 清零并构造 `[668]/[845]/[31]` observations。

Seed 必须为：

```text
seed(env_id) = base_seed + env_id
```

它不依赖 selected list order。Reset 不得推进 unselected physics/reference/history
row。必须分别维护：

```text
scaffold previous action history: frozen HDMI observation owner，push a_nom
nominal action history: residual policy owner，slot 0 current a_nom
executed action history: residual policy owner，push a_total
```

C0 即使收到非零 raw residual probe，authority 仍是 bitwise zero，且每 joint、
每 env 必须满足 `a_total == a_nom`。随后 normalized delay/smoothing，最后执行
唯一一次 `default + applied * scale`。

## 11. 通用 trace 与 door 24-field 迁移

新的通用 trace 保持 24 fields 数量和前 14 个 action/history/robot fields，
只把四个 door object fields 改为通用 object fields：

```text
a_nom, raw_residual, authority, a_total,
applied_action, joint_target,
scaffold_history, nominal_history, executed_history,
reference_step, reference_phase,
robot_root_state, robot_joint_pos, robot_joint_vel,
object_root_state, object_joint_pos, object_joint_vel, object_applied_effort,
env_origin, physics_mismatch, reset_seed, finite, terminated, time_outs
```

Fixed trailing shapes：

```text
object_root_state [B,13]
object_joint_pos [B,1]
object_joint_vel [B,1]
object_applied_effort [B,1]
```

Rigid task 的后三个 mechanism trace 必须是 bitwise-zero float32，不使用 shape
`[B,0]`，从而保持四任务 schema 固定。

### 11.1 Legacy door archive 是未版本化合同

当前已接纳的 64-env golden/residual door NPZ 实测 metadata key set 精确为：

```text
action_joint_names
configuration
configuration_hash
device
dtype
mode
robot_body_names
robot_joint_names
trace_dtypes
trace_fields
trace_shapes
```

它没有 `trace_contract_version`、top-level `task`、`object_kind` 或 artifact
identity。旧 archive 本身是 **unversioned legacy door archive**；不得声称它已
声明 `phase4b3_door_v1`，也不得为了迁移而重写、重新压缩或补写既有 NPZ。

Legacy detection 必须同时满足以下全部条件：

1. metadata 中 `trace_contract_version` 缺失；
2. metadata key set 与上面的 11-key set 精确相等；
3. `configuration` 是 JSON object，且 `configuration.task` 精确等于
   `push_door_hand`；
4. `trace_fields` 与以下 lexicographically sorted 旧 door 24-field set 精确相等：

```text
a_nom
a_total
applied_action
authority
door_applied_effort
door_joint_pos
door_joint_vel
door_root_state
env_origin
executed_history
finite
joint_target
nominal_history
physics_mismatch
raw_residual
reference_phase
reference_step
reset_seed
robot_joint_pos
robot_joint_vel
robot_root_state
scaffold_history
terminated
time_outs
```

只有全部满足时，reader 才能在内存中标记
`source_contract_kind=legacy_phase4b3_door_unversioned`。任何其他缺 version 的
archive 必须拒绝，不能根据文件名、路径、field 相似度或 caller task hint 猜测。

通过 detection 后还必须验证：NPZ key set 精确等于旧 24 fields 加
`metadata_json`；`trace_shapes` 和 `trace_dtypes` 的 key set 均精确等于旧 24
fields；每个实际 array 的 step 后 shape 和 dtype 与 metadata 一致。任一
extra/missing key、shape 或 dtype 不符都必须拒绝。

### 11.2 Legacy field canonicalization

Legacy canonicalization 只能在内存中执行以下四个一对一 alias：

```text
door_root_state       -> object_root_state
door_joint_pos        -> object_joint_pos
door_joint_vel        -> object_joint_vel
door_applied_effort   -> object_applied_effort
```

Alias 同时作用于 in-memory array map、`trace_fields`、`trace_shapes` 和
`trace_dtypes`；canonical view 中删除四个旧 key并加入四个新 key。禁止保留
old+new duplicate、增加第五个 alias、接受 alias source 缺失，或接受 target
预先存在。Raw archive 始终以 read-only 方式打开且 byte-immutable。

### 11.3 `phase4b4_c0_v1` 新 metadata 精确 schema

新 trace 的 metadata key set 精确为以下 15 keys。全部 required，optional keys
为空；不允许 extra key。

| Key | JSON type / value contract | Required | Golden/residual rule |
| --- | --- | --- | --- |
| `action_joint_names` | `list[str]`，长度 23，精确 artifact action order | yes | exact equal |
| `artifact_sha256s_sha256` | `str`，正则 `^[0-9a-f]{64}$` | yes | exact equal并与 live readback 相等 |
| `configuration` | JSON object；必须含 `task` 且与 top-level `task` 相等；不得含 `mode` | yes | deep exact equal |
| `configuration_hash` | `str`，canonical configuration JSON bytes 的 SHA256 lowercase hex | yes | 各自重算后 exact equal |
| `device` | `str`，例如 `cuda:0` | yes | exact equal |
| `dtype` | `str`，C0 为 `torch.float32` | yes | exact equal |
| `mode` | `str` enum `golden` / `residual` | yes | 左侧必须 golden，右侧必须 residual |
| `object_kind` | `str` enum `articulation` / `rigid_object`，与 task spec 一致 | yes | exact equal |
| `robot_body_names` | nonempty `list[str]`，无重复 | yes | exact equal |
| `robot_joint_names` | `list[str]`，长度 29，无重复 | yes | exact equal |
| `task` | `str` enum `push_door_hand` / `push_box` / `move_suitcase` / `move_largebox` | yes | exact equal |
| `trace_contract_version` | `str`，精确 `phase4b4_c0_v1` | yes | exact equal |
| `trace_dtypes` | `dict[str,str]`，key set 精确等于通用 24 fields | yes | deep exact equal并逐 array 验证 |
| `trace_fields` | `list[str]`，精确等于 lexicographically sorted 通用 24 fields | yes | exact equal |
| `trace_shapes` | `dict[str,list[int]]`，key set 精确等于通用 24 fields；shape 不含 control-step axis、首维为 B | yes | deep exact equal并逐 array 验证 |

`configuration_hash` 的输入精确为：

```python
json.dumps(
    configuration,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
).encode("utf-8")
```

然后计算 SHA256；禁止 hash pretty-printed JSON、metadata 全体或 Python repr。

新 NPZ key set 必须精确等于通用 24 fields 加 `metadata_json`。通用
`trace_fields` 即本节开头列出的 24 个 names，经 lexicographic sort 后写入
metadata。所有 field 都 required，没有 optional trace array。Array dtype 冻结为：

```text
float32:
  a_nom, raw_residual, authority, a_total, applied_action, joint_target,
  scaffold_history, nominal_history, executed_history, reference_phase,
  robot_root_state, robot_joint_pos, robot_joint_vel,
  object_root_state, object_joint_pos, object_joint_vel, object_applied_effort,
  env_origin, physics_mismatch

int64:
  reference_step, reset_seed

bool:
  finite, terminated, time_outs
```

Step 后 shapes 冻结为：action fields `[B,23]`；三个 history fields
`[B,23,3]`；`reference_step/reset_seed/finite/terminated/time_outs` 为 `[B]`；
`reference_phase` `[B,1]`；robot root `[B,13]`；robot joint pos/vel `[B,29]`；
object root `[B,13]`；三个 object mechanism fields `[B,1]`；origin `[B,3]`；
physics row `[B,39]`。

### 11.4 Artifact identity 精确合同

新 metadata field 名固定为：

```text
artifact_sha256s_sha256
```

其输入是该 task `artifacts/scaffolds/<artifact>/v1/SHA256SUMS` 文件的原始
bytes，包含原始换行；算法是 SHA256，编码必须是 64 位 lowercase hex。不得
hash parsed/sorted entries、manifest、artifact directory listing 或文件路径。

本次实测值：

| Task | `artifact_sha256s_sha256` |
| --- | --- |
| `push_door_hand` | `51a1535953d57b2ac63dea7d821edfdd71b95f9ad6e681bb16bd714148a5e6cb` |
| `push_box` | `5dafd1f2b4b75edec859a8aa9b026158f125d9e2d3e294537b35bf9569729b08` |
| `move_suitcase` | `c0d61caca2ca6aeade7b34278168fda1e9a3f0daaaa36febef536f9a053055d9` |
| `move_largebox` | `bc469c68324f54cbeb8d258b917a3773b82b38091aac3817d465da55971b3baa` |

Reader/comparator 必须先对选定 artifact 执行 `sha256sum -c SHA256SUMS` 等价的
全量校验，再从 raw `SHA256SUMS` bytes 重算 identity。Metadata value、golden/
residual value和 live value 三者必须 exact equal。

### 11.5 Golden/residual equality

对两个 new `phase4b4_c0_v1` archives：

- 两侧 raw metadata key set 都必须精确为 15 keys；除 `mode` 外所有 metadata
  value deep exact equal，`mode` 必须分别为 `golden`、`residual`。
- 两侧分别重算 configuration/artifact hashes，再比较二者；不能只比较 metadata
  中声称的字符串。
- 两侧分别验证 NPZ keys、field list、shape、dtype 和 metadata maps，任何一侧
  mutation 都必须失败。
- exact fields 是 finite/done、全部 action/history/reference、origin、39-D
  physics row和reset seed；C0 checks 继续要求 residual `a_total==a_nom`、raw
  probe ones、authority zeros，以及 nominal/executed histories 与 golden scaffold
  history 相等。
- tolerance fields 是 robot root/joint state和四个通用 object state fields，
  只使用 caller 显式 `atol/rtol`。第一处分歧按
  `(step,env_id,field,component)` 报告。

### 11.6 Legacy door 与 new door 比较

Canonicalizer 必须分别处理 golden 和 residual 输入，所以 legacy archive 可在
任一侧出现，但该侧仍必须具有正确 `mode`。Legacy 缺失的 metadata 只能按以下
显式 trusted context 处理：

```text
source_contract_kind = legacy_phase4b3_door_unversioned  # in-memory only
task = push_door_hand                                    # fixed door contract
object_kind = articulation                               # revalidated task spec
artifact_sha256s_sha256 = live raw SHA256SUMS hash       # after full artifact check
```

这四项不得写回 raw metadata。尤其不能伪造 legacy
`trace_contract_version=phase4b4_c0_v1`；new 对侧仍须独立声明并通过完整 15-key
schema。Task/object kind 必须同时与 legacy `configuration.task`、current fixed
door task spec 一致；artifact identity 必须来自运行时重新全量校验的 current
door artifact。Artifact 不可用、checksum 失败或 task spec 不符时拒绝比较。

Comparator 在完成四个 field aliases 后，比较双方 canonical 24 fields及共同
metadata，并把 legacy trusted context 与 new 对侧显式 task/object/hash 做 exact
比较。除上面固定的 door context 外，任何 legacy 缺失或未知 metadata 都不得
静默补齐。

既有 archives 保持 byte-immutable。本次读取前后的文件 SHA256 证据为：

```text
golden:
  /tmp/somaforce_phase4b3/golden_64env_6step_d88684c9bff5.npz
  6f4d84719207cf09304b06fb795968e1143d511c6f89babaaa7aa22b3ce7091d
residual:
  /tmp/somaforce_phase4b3/residual_64env_6step_d88684c9bff5.npz
  d982ba323e5ce7612b1c2bc0a7cbba718e6c576fbc6f0296bf871afdc6ab5c6a
```

## 12. RTX 4090 golden bounded smoke

共同配置：seed `20260728`、physics dt `0.005 s`、control dt `0.02 s`、
decimation 4、delay 4 physics substeps、alpha 0.9、rigid surface friction 0.5、
headless、6 control steps。每次独立 AppLauncher，启动后才 import/instantiate
`PretrainedHDMIIsaacRuntime`。

| Task / envs | Config hash | Origins | GPU memory MiB before/init/after | Torch peak allocated/reserved bytes | Trace |
| --- | --- | ---: | --- | --- | --- |
| `push_box` / 1 | `8e45bac23cb0d9ad46abc1242b30c1320b21a5b46dc5826486c876dd59d64c54` | 1 | 1155 / 3339 / 3343 | 17046528 / 27262976 | `/tmp/somaforce_phase4b4a/push_box_1env_6step_8e45bac23cb0.npz` |
| `push_box` / 64 | `54b24f351e9a64b030535e80489c72ae949ebb6e7789ef660fca538bb70976c3` | 64 | 1155 / 3343 / 3347 | 19715584 / 29360128 | `/tmp/somaforce_phase4b4a/push_box_64env_6step_54b24f351e9a.npz` |
| `move_suitcase` / 1 | `f4804455d9d23ac911fe832dee94975a250079c8b415604e9a3d8392893c9c4e` | 1 | 1155 / 3339 / 3341 | 15976960 / 25165824 | `/tmp/somaforce_phase4b4a/move_suitcase_1env_6step_f4804455d9d2.npz` |
| `move_suitcase` / 64 | `3d60c06000d73a24c8d610f2437f4b832adacb573921a825e4a9a9ab8020ae7d` | 64 | 1155 / 3343 / 3349 | 18673664 / 29360128 | `/tmp/somaforce_phase4b4a/move_suitcase_64env_6step_3d60c06000d7.npz` |
| `move_largebox` / 1 | `2bf601954b209cafe961cd501ee71d65de2c550fb820dc528fb5dc4b53378b58` | 1 | 1155 / 3476 / 3480 | 14682112 / 25165824 | `/tmp/somaforce_phase4b4a/move_largebox_1env_6step_2bf601954b20.npz` |
| `move_largebox` / 64 | `ac73644f9b4cb54cae386bffdee46086f5e142b474bbe24c5314b4851f373bae` | 64 | 1155 / 3482 / 3488 | 17378816 / 29360128 | `/tmp/somaforce_phase4b4a/move_largebox_64env_6step_ac73644f9b4c.npz` |

每次结果：

```text
a_nom:             [B,23] float32 cuda:0 finite
applied_action:    [B,23] float32 cuda:0 finite
joint_target:      [B,23] float32 cuda:0 finite
scaffold_history:  [B,23,3] float32 cuda:0 finite
robot_root_state:  [B,13] float32 cuda:0 finite
robot_joint_pos:   [B,29] float32 cuda:0 finite
robot_joint_vel:   [B,29] float32 cuda:0 finite
object_root_state: [B,13] float32 cuda:0 finite
reference_step:    [B] int64 cuda:0 finite, final 6
reference_phase:   [B,1] float32 cuda:0 finite
zero_hook_exact: true
```

每次在 `finally` 中清理 SimulationContext 并调用 `SimulationApp.close()`，进程均
`exit 0`。最终进程扫描没有 `phase4b4a_golden_smoke` 或本轮 headless Kit 残留；
`nvidia-smi` 中仍有一个本轮之前已存在的其他 608 MiB 进程，不归属本诊断。
`memory.used` 包含驱动和同机驻留，因此只作为观测，不是 isolated GPU allocation。

Again，本节不是 residual parity：没有 DirectRLEnv、adapter、C0 composition 或
golden/residual closed-loop compare。

## 13. Phase 4B4B 精确文件白名单

建议项目所有者 Admission 后只允许以下 13 个路径：

```text
somaforce_cross/envs/residual_env.py
somaforce_cross/envs/residual_env_cfg.py
somaforce_cross/envs/task_adapter.py
somaforce_cross/envs/task_adapters/__init__.py
somaforce_cross/envs/task_adapters/push_door_hand.py
somaforce_cross/envs/task_adapters/rigid_object.py
somaforce_cross/envs/task_adapters/push_box.py
somaforce_cross/envs/task_adapters/move_payload.py
scripts/smoke_phase4_multitask_env.py
tests/test_phase4_reset_contract.py
tests/test_phase4_task_adapters.py
tests/test_phase4_zero_residual_parity.py
tests/test_phase4_import_boundaries.py
```

`rigid_object.py` 只承载三个 rigid task 共用的 scene/reference/scaffold observation/
object-state/contact/reset plumbing；`push_box.py` 持有 push progress/failure，
`move_payload.py` 由 manifest data 同时服务 suitcase/largebox。禁止 task-specific
residual model。

明确不在白名单：

```text
somaforce_cross/scaffold/pretrained_hdmi_isaac.py
somaforce_cross/scaffold/pretrained_hdmi.py
artifacts/scaffolds/**
force/**, residual/**, sensing/**
reward/mismatch/curriculum/normalizer implementation
PPO/RSL-RL/training code或配置
任何其他 task/artifact
```

若实现发现必须修改白名单外路径，立即停止并请求新的 owner authorization；
不得扩大 scope 自行修复。

## 14. 4B4B 测试矩阵与运行命令

### 14.1 必须新增/扩展的 tests

1. 三个 artifact/task-spec/object asset/reference/action name mapping exact test。
2. 四任务同一 TaskAdapter Protocol、同一 `[668]/[220]/[845]/[31]` schema。
3. Actor leak mutation test：每个 forbidden privileged/task/object field 改变时
   policy/actor tensor bitwise 不变。
4. Rigid `object_state [16]` exact formula、wxyz order、mechanism zero test。
5. 三个 nominal 39-D row exact/readback test，拒绝错位和 guessed values。
6. 每任务 selected reset、reordered IDs、atomic failure、unselected rows unchanged、
   `base_seed+env_id`、zero 16-frame wrist history和三历史 ownership test。
7. 非零 residual probe + C0 authority zero，`a_total==a_nom` bitwise，one-time
   scaling test。
8. 每任务 progress/contact/stability/failure/exhaustion boundary tests，包括
   suitcase root `0.25` 上不失败、低于 `0.25` 失败，绝不套用 `0.45`。
9. 精确 legacy door metadata acceptance test，golden/residual 两种 `mode` 都覆盖；
   仅当 11-key set、`configuration.task` 和旧 24 fields 全部精确匹配时接受。
10. 拒绝 unknown unversioned metadata；拒绝任何 legacy extra/missing NPZ、
    metadata、shape/dtype map key；四个 alias source/target mutation 均拒绝。
11. New trace 缺 `trace_contract_version`、`task`、`object_kind` 或
    `artifact_sha256s_sha256` 任一 required field 必须拒绝；task、object kind、
    artifact hash、configuration hash和每个 field alias mutation 必须分别覆盖
    golden/residual 两侧。
12. Legacy migration test 必须在 compare 前后重算 archive file SHA256，证明
    golden/residual NPZ 均 byte-immutable；不得用 rewrite 后等价 arrays 代替。
13. 通用 24-field key/shape/dtype/完整 15-key metadata test；golden/residual 两侧
    每个 trace field 和 metadata field 的 synthetic mutation 必须被拒绝。
14. 每任务 1/64-env、6-step independent golden/residual trace compare；64 rows
    全比较，不能抽样。

### 14.2 实现后精确命令

```bash
pytest -q \
  tests/test_phase4_reset_contract.py \
  tests/test_phase4_task_adapters.py \
  tests/test_phase4_zero_residual_parity.py \
  tests/test_phase4_import_boundaries.py

for task in push_box move_suitcase move_largebox; do
  for n in 1 64; do
    python scripts/smoke_phase4_multitask_env.py \
      --mode golden --task "$task" --num-envs "$n" --steps 6 \
      --seed 20260728 --trace "/tmp/phase4b4b_${task}_${n}_golden.npz" \
      --headless
    python scripts/smoke_phase4_multitask_env.py \
      --mode residual --task "$task" --num-envs "$n" --steps 6 \
      --seed 20260728 --trace "/tmp/phase4b4b_${task}_${n}_residual.npz" \
      --headless
    python scripts/smoke_phase4_multitask_env.py \
      --mode compare \
      --golden-trace "/tmp/phase4b4b_${task}_${n}_golden.npz" \
      --residual-trace "/tmp/phase4b4b_${task}_${n}_residual.npz"
  done
done

python scripts/smoke_phase4_door_env.py \
  --mode golden --num-envs 64 --steps 6 --seed 20260727 \
  --trace /tmp/phase4b4b_door_golden.npz --headless
python scripts/smoke_phase4_door_env.py \
  --mode residual --num-envs 64 --steps 6 --seed 20260727 \
  --trace /tmp/phase4b4b_door_residual.npz \
  --selective-reset-env-ids 0 7 63 --headless
python scripts/smoke_phase4_door_env.py \
  --mode compare \
  --golden-trace /tmp/phase4b4b_door_golden.npz \
  --residual-trace /tmp/phase4b4b_door_residual.npz

pytest -q \
  tests/test_force_core.py \
  tests/test_residual_action.py \
  tests/test_virtual_ft.py \
  tests/test_wrist_wrench_source_contract.py \
  tests/test_contact_semantics.py

pytest -q
python -m compileall -q somaforce_cross tests scripts
git diff --check
```

每个 Isaac 子进程结束后都要检查 exit code、summary、configuration hash、
`SimulationApp.close()` 和残留进程。Compare 必须拒绝 configuration hash 不同的
archives，显式 `--trace` 只固定文件位置，不得绕开 metadata 验证。

## 15. 停止条件

4B4B 遇到任一条件立即停止，不 commit、不 push、不进入 4B5：

- branch/HEAD/live origin/worktree 不满足 owner 指定 preflight；
- 任一 artifact checksum 或 golden runtime SHA 改变；
- action/reference name mapping 缺失、重复、错序或 scaling 多于一次；
- actor/policy 泄漏 task/object identity、HDMI command、object state、contact truth、
  clean wrench、mismatch 或其他 privileged state；
- rigid mechanism 三槽非零或 object-state frame/quaternion/twist mapping 不符；
- nominal 39-D row 与 applied/readback property 不符，或实现加入未批准 range/
  distribution/normalizer；
- selected reset改变 unselected row、调用 global `sim.step()`、seed 不等于
  `base_seed+env_id`、reset wrist frame 非零或 histories alias；
- C0 `a_total != a_nom`、raw probe没有被 zero authority 消除、joint target
  发生 double scaling；
- 任一 trace key/shape/dtype/metadata、finite/done/action/history/reference/physics
  compare 或 mutation test 失败；
- unknown unversioned archive 被接受、legacy detection/alias 不精确、new metadata
  缺 required key，或 legacy archive bytes 在读取/比较后改变；
- suitcase 使用 0.45 m failure boundary，或 0.25 m inclusive boundary 实现错误；
- 1/64-env smoke、targeted/full tests、compileall、AST/import、checksum、whitespace、
  Git allowlist 中任一 gate 失败；
- 出现未关闭的本轮 Isaac/Kit 进程；
- 实现需要 whitelist 外文件或开始 mismatch/reward/curriculum/PPO/training。

## 16. 本轮验证结果

```text
artifact SHA256SUMS:
  push_door_hand: all entries OK（legacy canonicalization revalidation）
  push_box: all entries OK
  move_suitcase: all entries OK
  move_largebox: all entries OK

artifact_sha256s_sha256:
  push_door_hand: 51a1535953d57b2ac63dea7d821edfdd71b95f9ad6e681bb16bd714148a5e6cb
  push_box: 5dafd1f2b4b75edec859a8aa9b026158f125d9e2d3e294537b35bf9569729b08
  move_suitcase: c0d61caca2ca6aeade7b34278168fda1e9a3f0daaaa36febef536f9a053055d9
  move_largebox: bc469c68324f54cbeb8d258b917a3773b82b38091aac3817d465da55971b3baa

golden runtime SHA256: PASS
RTX 4090 golden bounded smoke: 6/6 runs PASS

legacy door 64-env archive metadata:
  exact 11-key unversioned set: PASS (golden and residual)
  configuration.task == push_door_hand: PASS
  exact old door 24-field set: PASS
  NPZ bytes unchanged by read: PASS

Phase 1-3 regression:
  94 passed in 6.11s

current Phase 4 tests:
  126 passed in 8.39s

full pytest:
  248 passed in 10.04s

python -m compileall -q somaforce_cross tests scripts: PASS
recursive AST/import scan: 26 files, forbidden_import_hits=[]
git diff --check: PASS
report trailing-whitespace scan: PASS
git diff --no-index --check /dev/null report: no diagnostics
```

## 17. 已验证事实、未批准数值与风险

已验证事实：

- 当前 Git 起点、实时 remote 和 clean entry state；
- artifact/asset/reference/action contracts 与全量 checksum；
- exact name-based mapping、reference lengths、object kinds 和 nominal runtime
  physical readback；
- current door-specific ownership和通用 tensor/reset/action boundaries；
- 三任务 1/64-env short golden runtime finite/vectorized execution；
- current regression、full suite、compileall 和 import boundary。

未批准数值：

```text
任何 physical/scaffold/sensor mismatch 分布
任何 observation/critic normalizer
任何 reward weight、scale、clip 或 aggregate scalar
任何 task balance、curriculum probability 或 stage transition
任何 PPO/loss/training hyperparameter启用
任何 closed-loop physics parity tolerance，除非 4B4B Goal 显式传入
```

风险/阻塞项：

1. 三任务 residual env/adapter/trace 尚不存在，C0 parity 是开放 gate。
2. Artifacts 是 privileged teacher、`deployable=false`；deployment gate 开放。
3. Push-box/suitcase manifests 有 task-spec fallback 字段，不得原地修 artifact。
4. Golden `PretrainedHDMIIsaacRuntime` 是 global reset/scalar reference owner，不能
   复用为 training env。
5. Rigid filtered wrist-object contact 与 door general wrist contact source 不同，
   shared env 不能隐藏这个 adapter 差异。
6. Inertia row 必须来自 applied PhysX readback，不能仅信 asset authored值。
7. Golden bounded smoke 不证明 full-reference task完成或 residual行为。

## 18. Modified files 与 phase stop

```text
modified files:
  docs/phase4b4a_multitask_preflight.md only

adapter/environment implementation: 未执行
Phase 4B5: 未进入
PPO/RSL-RL/training: 未执行
commit/push: 未执行
```

到此停止。只有项目所有者 Admission 本报告并另行授权精确 4B4B whitelist 后，
才能实现；Admission 之前不得 commit/push 或提前进入 4B4B。
