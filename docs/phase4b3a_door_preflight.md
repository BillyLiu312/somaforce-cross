# Phase 4B3A Door-Only Isaac Preflight

日期：2026-07-27

状态：只读 preflight 报告。该报告冻结正式 Phase 4B3 的 door-only C0
实现合同，但不实现 `DirectRLEnv`、ResidualEnv、PPO、训练或 Phase 4B4。

## 1. Git 与边界

入口和出口检查均为：

```text
branch: v1
HEAD: d370f439e082d405f5846ff0d76efae133c2db1c
tracking: v1...origin/v1 [ahead 3]
status: clean（报告文件创建前）
```

本 Goal 未切换、reset、checkout、commit 或 push。除本报告外，源码、测试、
artifact、资产和 golden runtime 均只读。允许修改文件只有：

```text
docs/phase4b3a_door_preflight.md
```

HEAD 已包含 Phase 4B1/B2 的纯 PyTorch 合同模块
(`somaforce_cross/envs/{observations,reset,mismatch,action_history}.py`)，
但没有 Isaac 环境类、door task adapter 或 Isaac 注册项。这些现状不能被
解释为本次已实现 4B3。

## 2. Isaac Lab API 审计

实际环境：

```text
Isaac Lab: 0.47.2
Isaac Lab source: /opt/IsaacLab/source/isaaclab/isaaclab
Isaac Sim/Kit: 5.1（AppLauncher 启动日志）
GPU: NVIDIA GeForce RTX 4090, 49,386 MiB advertised
DirectRLEnv: /opt/IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py
DirectRLEnvCfg: /opt/IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env_cfg.py
```

本审计先构造 `AppLauncher(['--headless'])` 并取得 `simulation_app`，然后才
导入 `isaaclab.envs.DirectRLEnv`、`DirectRLEnvCfg` 及其含 `omni` 的依赖。
真实签名为：

```python
DirectRLEnv(cfg: DirectRLEnvCfg, render_mode: str | None = None, **kwargs)
DirectRLEnv.__init__(self, cfg, render_mode=None, **kwargs)
DirectRLEnvCfg(..., decimation: int, episode_length_s: float,
               scene: InteractiveSceneCfg, observation_space, action_space,
               state_space=None, ...)
```

`DirectRLEnv` 的 `__abstractmethods__` 在运行时元数据中为空，但以下五个
方法带有 `@abstractmethod`，派生环境必须实现：

```python
_pre_physics_step(self, actions: torch.Tensor)
_apply_action(self)
_get_observations(self) -> VecEnvObs
_get_rewards(self) -> torch.Tensor
_get_dones(self) -> tuple[torch.Tensor, torch.Tensor]
```

`_setup_scene(self)` 和 `_reset_idx(self, env_ids: Sequence[int])` 是可覆盖的
非 abstract hook；后者是选择性 reset 的实际边界。基类 `reset()` 返回
`(obs, extras)`，`step(action)` 要求 action 为 `[num_envs, action_dim]`，并返回：

```text
(observations, reward[num_envs], terminated[num_envs],
 time_outs[num_envs], extras)
```

其中 observations 是以 `policy` 为约定键的字典，可另含 `critic`；布尔
`terminated` 与 `time_outs` 分开返回。基类在 decimation 个 physics
substep 中反复调用 `_apply_action`，之后计算 done/reward、reset selected rows、
再构造 next observation。C0 环境应保留该顺序，而不是调用 golden runtime 的
`step()`。

环境注册不是 DirectRLEnv 本身的必要条件。直接实例化 cfg/class 不需要
`gym.register`；只有采用 `gym.make` 或外部 launcher 时才需要注册。当前仓库
没有 Isaac 注册项，且不应因本 Goal 添加注册副作用。

必须延迟到 AppLauncher 之后的 import 包括：

```text
isaaclab.envs.DirectRLEnv / DirectRLEnvCfg
isaaclab.sim、isaaclab.assets、isaaclab.scene、isaaclab.sensors
omni.*、pxr.*、PretrainedHDMIIsaacRuntime
```

可在未启动 AppLauncher 时导入的纯包包括 `somaforce_cross.force`、
`somaforce_cross.residual`、`somaforce_cross.sensing` 和当前
`somaforce_cross.envs`。现有 AST/import 测试确认这些模块没有 Isaac、HDMI、
`active_adaptation` 或 RSL-RL 依赖。 `IsaacWristWrenchSource` 只在初始化的
USD resolver 中延迟导入 PXR。

## 3. Golden door runtime preflight

唯一使用的 artifact 和 runtime：

```text
artifacts/scaffolds/hdmi_push_door_hand/v1
somaforce_cross.scaffold.pretrained_hdmi_isaac.PretrainedHDMIIsaacRuntime
```

runtime 源文件 SHA256：

```text
e924dadeb945278ee85fad82e31be56f78bce4e62f5bb9f575f5938f6c4868a0
```

artifact `SHA256SUMS` 只读校验全部通过（policy、normalization、reference、
G1/door USD、contracts、parity fixture、manifest、THIRD_PARTY）。manifest
明确 `role=frozen_hdmi_phase_train_privileged_teacher_simulation_baseline`、
`deployable=false`；这不是 residual actor 的部署输入证明。

两次运行都使用 AppLauncher headless、`torch.manual_seed(20260727)`、
`dt=0.005`、render interval 4、`delay=4`、`alpha=0.9`、door friction
`0.3`、door damping `0.55`，各运行 6 个有限 control steps，并在 finally
关闭 SimulationApp。

### 3.1 1-env

```text
初始化：成功；device cuda:0；env_origins [1,3]，unique=1
GPU memory.used: 5459 MiB -> 8231 MiB(init) -> 8235 MiB(after steps)
torch peak allocated/reserved: 16,349,696 / 27,262,976 bytes
每步 a_nom: [1,23], float32, cuda:0, finite=True
applied normalized action: [1,23], finite=True
physical joint target: [1,23], finite=True
history.previous_actions: [1,23,3], finite=True
door joint: [1,1], root position: [1,3]，均 finite
reference_step: 0 -> 6；phase[0]: 0 -> 0.0104712043
zero hook: received_action == last_a_nom，bitwise True
```

首步/末步 `a_nom` 的数值范围分别为 `[-1.3341718, 1.6695007]` 和
`[-1.2058502, 1.8990718]`；这只是 teacher smoke 观测，不是 action bound
批准。

### 3.2 64-env

```text
初始化：成功；env_origins [64,3]，unique=64
首四个 origins: [17.5,-17.5,0], [17.5,-12.5,0], [17.5,-7.5,0], [17.5,-2.5,0]
GPU memory.used: 9318 MiB -> 12032 MiB(init) -> 12040 MiB(after steps)
torch peak allocated/reserved: 19,045,888 / 31,457,280 bytes
每步 a_nom: [64,23], float32, cuda:0, finite=True
applied normalized action: [64,23], finite=True
physical joint target: [64,23], finite=True
history.previous_actions: [64,23,3]，door [64,1]，root [64,3]，均 finite
reference_step: 所有 rows 为 0 -> 6；phase 与 1-env 相同
每步 unique action rows=64；64 个 env 实际独立存在
```

`nvidia-smi` 的 memory.used 受同机 Kit/驱动驻留影响，因此这里只报告运行前后
观测和进程内 torch peak，不把它当作精确的单独物理占用基线。

## 4. 可复用与禁止复用

未来 DirectRLEnv 可以复用：

- artifact contract、manifest 中的 action joint names、23-D normalized
  `a_nom`、normalization 和 checksum/provenance 元数据；
- `PretrainedHDMIScaffold` 的冻结 eval-mode tensor inference；
- `make_scene_cfg` 的 G1/door/ground/actuator/contact-sensor building facts；
- `HDMIMotionReference` 的 reference tensors、fps 和 name-based mappings；
- `HDMIObservationHistory` 的字段布局作为语义参考，但必须由每-env owner
  重新实现；
- `HDMIJointPositionActionRuntime` 中 delay、smoothing、normalized action
  buffer 和一次 joint scaling 的数学事实；
- `IsaacWristWrenchSource` 的 left/right wrist body-name resolution、child
  joint pose cache、`body_incoming_joint_wrench_b` 到 base-yaw 的 GPU path。

明确禁止：

- subclass `PretrainedHDMIIsaacRuntime`；
- 调用其 global `reset()` 或 control-loop `step()` 作为训练环境循环；
- 修改 `somaforce_cross/scaffold/pretrained_hdmi_isaac.py`；
- 把 `command[356]`、object state、reference truth、sim contact truth、foot
  wrench、door state 或其他 privileged 字段复制进 residual actor；
- 修改 checksummed artifact、替换其 USD/policy/reference 或重写 SHA256；
- 引入 HDMI、`active_adaptation`、HDMI checkout path 或 pickle-class runtime
  依赖；
- 为 door 单独复制一个 residual actor/critic 网络。

golden runtime 的全局 `reference_step` 是 Python scalar，`reset()` 是 global，
`step()` 把 `a_nom` 写入 HDMI previous-action history。未来环境必须分开维护：

```text
scaffold previous action: HDMI-compatible scaffold owner/history
previous executed a_total: residual environment ExecutedActionHistory
```

当前纯合同的 `NominalActionHistory` 与 `ExecutedActionHistory` 已有不 alias
检查，但尚未连接 Isaac scene。

## 5. Door C0 实现合同

### 5.1 C0 配置

以下是本次 preflight 的批准/显式 smoke 参数：

```text
task: push_door_hand only
physics_dt: 0.005 s；control_dt: 0.02 s；decimation: 4
delay: 4 physics substeps；smoothing alpha: 0.9
door friction: 0.3；door damping: 0.55
residual authority: C0 = zero for arms/waist/legs
headless: true；GUI/VNC/NoVNC: false
```

未来 runner 必须把以下字段做成显式 CLI/config，而不是隐藏默认值：
`seed`、`num_envs`、`delay`、`alpha`、`door_friction`、`door_damping`、
`episode_length_steps`/`episode_length_s`、`parity_atol`、`parity_rtol`、
`tare_num_samples`、`tare_calibration_mask/gate`、`F_scale`、`M_scale`、
detector 参数、VirtualFT 参数、contact-ramp 参数以及 smoke reward 模式。

### 5.2 每 control step 的严格顺序

```text
current Isaac robot/door state
 -> frozen scaffold observation (privileged fields stay outside actor)
 -> frozen scaffold a_nom [B,23] normalized joint-position action
 -> wrist wrench source -> tare -> VirtualFT -> contact/token history
 -> p_dir [B,13], p_mag [B,5], P_cross [B,13,5], z_cross [B,64]
 -> structured policy [B,668] / actor boundary [B,220]
 -> external residual action [B,23]
 -> C0 authority exactly zero (nonzero external input is eliminated)
 -> a_total exactly equal to a_nom, no clip-induced change
 -> normalized delay/smoothing
 -> default_joint_pos + applied_normalized * action_scale exactly once
 -> 4 physics substeps
 -> state/history update
 -> reward, terminated/time_outs, next observation
```

C0 的非零 residual 输入必须仍然被 zero authority 消除；“调用者传零”不足以
证明该门。`a_total` 只进入 residual 的 `ExecutedActionHistory`；scaffold 自己
继续按 artifact contract 维护 previous nominal action。reset coordinator 的三
个 selected hooks 必须绑定为：

```text
scaffold: frozen standalone scaffold state/history owner
action_sink: 新 DirectRLEnv 的 normalized delay/smoothing/joint-target sink
reference_adapter: door HDMIMotionReference + door progress/reference phase owner
```

三个 hook 不能绑定 `PretrainedHDMIIsaacRuntime` 本体。reset 顺序是先保存
terminal 信息，再采样显式 episode rows，reset scene/scaffold/reference/sensor/
detector/histories/ramp，广播 initial valid wrist frame 到 16 slots，置零
previous executed action 和 counters，最后构造首个 observation。

### 5.3 结构化 observation

冻结的 policy bundle 为：

```text
wrist_tokens       [B,2,16,14] = 448
proprio            [B,64]
a_nom_history      [B,23,3] = 69
previous_a_total   [B,23]
z_cross            [B,64]
policy             [B,668]
residual actor     [B,220] = z_cross + proprio + a_nom_history + previous_a_total
critic             [B,845]
semantic target    [B,31]
```

actor 不得把 raw wrist tokens、`p_dir`、`p_mag` 或 `P_cross` 再拼接到
`z_cross`；critic/object/contact/clean-wrench 字段必须保持独立 key。door 的
policy 中不得出现 `command[356]`、object/task identity、door state、progress、
contact truth、foot wrench、clean wrench 或 mismatch。

### 5.4 Done/reward 边界

合同公式为：

```text
terminated = nonfinite_state OR door_adapter.failure
time_outs  = max_episode_steps OR reference_exhausted
done       = terminated OR time_outs
```

success 只作诊断，不自动变成 terminated。door 当前 source/audit 的跌倒阈值
是 root height `< 0.45 m`，但 reward 权重、clip/scale、normalizer、mismatch
分布、curriculum 概率都未批准。C0 preflight 可使用零 reward 仅作 smoke 输出，
不能把它写成训练合同。

## 6. 数值合同盘点

| 项目 | 分类 | 结论 |
| --- | --- | --- |
| `physics_dt=0.005`、control `0.02`、decimation `4` | artifact/source 批准 | action contract 与 runtime 一致 |
| delay `4`、alpha `0.9`、door friction `0.3`、damping `0.55` | artifact/source 批准 | 本次 1/64 smoke 已实跑 |
| `a_nom`/`a_total` `[B,23]` normalized、一次 scaling | artifact/source 批准 | target = default + applied * scale |
| C0 authority `(0,0,0)` | V1 设计批准 | arms/waist/legs 全为零 |
| `F_scale`/`M_scale` | 待所有者批准 | source 要求 positive；当前没有校准数值 |
| tare sample count、calibration mask/gate | 待所有者批准 | `WristTareCalibrator` 要求显式 `num_samples`，无生产默认 |
| detector on/off threshold、temperature、smoothing | 待所有者批准 | 只批准约束 `0 <= off < on`、temperature>0、alpha∈[0,1] |
| VirtualFT axis/scale/bias/drift/noise/delay/filter/saturation/dropout rows | smoke-only/待批准 | C0 需显式 identity/zero corruption，但每个 row 仍须显式传入 |
| contact ramp attack/release | 待所有者批准 | 必须 positive；C0 authority 为零但不得猜值 |
| episode length | 待所有者批准 | reference length=573 frames、50 Hz，即 11.46 s；smoke finite steps 不是训练长度 |
| zero reward | smoke-only | 只用于 bounded 初始化/reset/action 检查，不是 V1 reward |
| nonfinite termination | source contract 批准 | `nonfinite_state` 必须进入 terminated |
| timeout/reference exhaustion | source contract + 实现待定 | 两者进入 `time_outs`，不得合并 terminated |
| parity `atol/rtol` | 待所有者批准 | artifact fixed-input 只有 max-abs tolerance `1e-5`；closed-loop 尚未批准 |
| joint target compare point | artifact/source 批准 | 在 normalized delay/smoothing 后、physics sink 前比较；scaling 一次 |
| semantic target loss weight | 训练起始值/待批准 | plan 写 `lambda_dir=lambda_mag=0.05`；4B3A 不启用 loss |

### 6.1 必须显式的未来字段

建议 runner/config 至少提供：

```text
--seed
--num-envs
--delay --alpha --door-friction --door-damping
--episode-length-steps (或 --episode-length-s)
--F-scale --M-scale
--tare-num-samples --tare-calibration-steps/策略
--contact-on-threshold --contact-off-threshold
--contact-temperature --contact-smoothing-alpha
--virtual-ft-axis-quat --scale-error --bias --drift-rate
--drift-noise-std --white-noise-std --delay-steps
--filter-alpha --force-saturation --torque-saturation --dropout-probability
--ramp-attack-step --ramp-release-step
--parity-atol --parity-rtol
```

## 7. 独立 parity trace 方法

每个有限 trace 都是独立进程、独立 AppLauncher、相同 seed/door asset/reference/
physics/delay/smoothing。禁止用只固定网络输入的 oracle 代替 closed-loop trace。

### 7.1 Golden trace

每个 control step、每个 env row 记录：

```text
normalized a_nom [B,23]
normalized delayed/applied action [B,23]
physical joint target [B,23]
reference_step/phase/history slots
selected robot root/joint state、door q/qd/applied effort、finite flags
environment origin
```

### 7.2 Residual C0 trace

相同 seed、physics、reference、delay、smoothing 和 initial state；residual raw
input 明确使用非零 probe，但 authority 固定 C0 zero。记录完全相同的字段，并
额外记录 raw residual、authority、`a_total` 和 actor/critic leak check。要求：

```text
a_total == a_nom（每个 row、每个 joint）
```

### 7.3 比较规则与第一处分歧

- bitwise exact：reference_step/phase、a_nom、C0 `a_total`、normalized
  delayed/applied action、action histories、同设备同 kernel 下的 joint target。
- 允许 tolerance：跨 GPU 或 physics 执行得到的 root pose/velocity、door q/qd/
  effort、contact/支持状态和其他物理量。具体 `atol/rtol` 必须来自显式
  `--parity-atol/--parity-rtol`，当前不批准固定数值。
- 比较顺序按 `(control_step, env_id, field, joint/body index)`；第一处不一致
  输出 `/tmp/phase4b3a_trace.jsonl` 的索引、两值、绝对/相对误差和配置 hash。
- 若控制 step 内 divergence，进一步按 decimation 的 physics substep 二分，
  再按 action joint/body name 定位；64-env 不抽样，必须比较全部 rows。
- closed-loop parity 必须同时覆盖 reset、observation/history、action sink、
  physics state 和 reference progress；固定输入 `max_abs_action_error=0` 只能
  作为 artifact oracle 证据。

## 8. Phase 4B3B 建议白名单

基于实际 Isaac API，后续单独授权的 4B3B Goal 最多允许：

```text
somaforce_cross/envs/residual_env.py
somaforce_cross/envs/residual_env_cfg.py
somaforce_cross/envs/task_adapter.py
somaforce_cross/envs/task_adapters/__init__.py
somaforce_cross/envs/task_adapters/push_door_hand.py
scripts/smoke_phase4_door_env.py
tests/test_phase4_task_adapters.py
tests/test_phase4_zero_residual_parity.py
tests/test_phase4_import_boundaries.py
```

现有 `somaforce_cross/envs/__init__.py` 必须保持未启动 AppLauncher 可导入，
且不得从纯 PyTorch package init 自动导入 Isaac 模块；smoke runner 必须先
启动 AppLauncher，再延迟 import `residual_env`。不得扩展到 golden runtime、
artifact、PPO/RSL-RL、四任务 adapter 或 residual network migration。

## 9. 验证、阻塞与停止条件

本轮验证：

```text
pytest -q: 177 passed in 8.46s
python -m compileall -q somaforce_cross tests scripts: PASS
git diff --check: PASS
Markdown trailing-whitespace check: PASS
door artifact SHA256SUMS: all entries OK
1-env/64-env headless golden bounded smoke: PASS, 6 steps each
```

当前 blockers：

1. 没有 `DirectRLEnv`、door cfg、Isaac task adapter、reward/done 环境 API 或
   registration；本报告不实现它们。
2. `PretrainedHDMIIsaacRuntime` 是 global-reset/scalar-reference golden
   runtime，不能作为训练环境基类。
3. door manifest 仍依赖代码 fallback，且 artifact 是 privileged teacher、
   `deployable=false`；不得把它宣传为 deployable scaffold。
4. Isaac 侧尚无每-env physics/sensor mismatch resampling 和完整
   scaffold-sensor-semantic-residual action 闭环。
5. `F_scale/M_scale`、tare、detector、VirtualFT、ramp、episode/reward/
   normalizer、closed-loop parity tolerance 和 semantic loss 仍有批准门。

交付状态：

```text
modified files: docs/phase4b3a_door_preflight.md only
DirectRLEnv: 未实现
Phase 4B4: 未进入
commit/push: 未执行
```

到此停止，等待项目所有者批准 Phase 4B3B 实现 Goal；本报告不授权进入
Phase 4B4、Phase 5/PPO 或训练。
