# twist2_mjlab 系统辨识（sysid，MuJoCo / mujoco_warp）

`twist2_mjlab` 版的系统辨识，从 TWIST2 方案移植而来。与 TWIST2 版的关键差异：

- **引擎**：`mujoco` 3.7 + `mujoco_warp` 3.6（**没有 JAX**）。这里的拟合器用纯 MuJoCo
  前向 + 无梯度进化策略；`mujoco_warp` 批并行是后续加速 29 关节阶段的预期路径。
- **资产**：mjlab 的 G1 entity 已经把 actuator `armature` 和物理推导的 kp/kd 建模好了
  （`mjlab/asset_zoo/robots/unitree_g1/g1_constants.py`），且 sim 与真机使用同一套增益。
  所以 **kp/kd 固定**，`armature` 只做轻微修正；辨识集合是
  `frictionloss / damping / effort_limit / tau_m / d_ctrl / 吊带或软基座刚度与阻尼`。
- **延时**：mjlab 已在 position actuator 上暴露 `delay_min_lag/delay_max_lag`
  （`src/twist2_mjlab/config.py`）。

## 原理

与 TWIST2 版相同：构造仿真参数 θ，用真机记录的 `q_target` 在同一仿真里前向回放，
最小化与真机 `q/dq/IMU/τ_est` 的多步预测误差：

```
θ* = argmin_θ  Σ_j L( f_θ(x0_j, u_j), y_j ) + λ||θ − θ_prior||²
```

损失同样包含关节位置/速度、基座姿态（四元数测地误差）、基座角速度，权重与 TWIST2 版一致。
θ 采用有界 sigmoid 重参数化（`space.py`）。

### 与 TWIST2 的建模差异

- 本版 sysid MJCF 使用 **position actuator**（与 mjlab 训练一致），
  命令 `data.ctrl = 关节目标`，MuJoCo 内部算力矩；`tau = data.actuator_force`。
- `armature / kp / kd` 已在 mjlab 中解析建模，不再重点辨识；重点是摩擦、阻尼、
  effort 实现、伺服滞后、动作延时、支撑柔度。
- `d_ctrl` 通过 `tau_m` 式命令低通 + 延时插值实现（`plant.py`）。

## 代码结构

```
deploy/sysid/
  constants.py          G1 关节名、增益、action scale、KNEES_BENT 姿态、限位
  space.py              有界参数空间（susp / joint / softbase）
  excitation.py         激励信号 + 安全限幅
  build_sysid_xml.py    把 mjlab G1 entity（含 actuator）序列化成 sysid MJCF
  plant.py              纯 MuJoCo 前向（position actuator）+ 无梯度拟合
  fit.py                分阶段拟合 CLI
  record_real_sysid.py  真机激励/采集（unitree_interface）
  apply_params.py       输出拟合 MJCF + 仅供审阅的 DR patch
  g1_sysid.xml          由 build_sysid_xml 生成
```

## 操作流程

在 `twist2_mjlab/` 根目录、uv 环境内运行。

```bash
# 0) 生成独立 sysid MJCF（序列化 actuator + armature，并把 meshdir 写成绝对路径）
uv run python -m deploy.sysid.build_sysid_xml --out deploy/sysid/g1_sysid.xml

# 1) 真机采集（悬吊，逐关节）
uv run python -m deploy.sysid.record_real_sysid --net eth0 --joint 3 \
    --kind multisine --duration 20 --out deploy/sysid/data/susp_knee.npz

# 2) 阶段 0：吊带；阶段 1：关节/执行器/延时（冻结吊带）
uv run python -m deploy.sysid.fit --space susp \
    --runs 'deploy/sysid/data/strap_*.npz' --out deploy/sysid/generated/susp.json
uv run python -m deploy.sysid.fit --space joint \
    --runs 'deploy/sysid/data/susp_*.npz' --susp-params deploy/sysid/generated/susp.json \
    --out deploy/sysid/generated/joint.json

# 3) 阶段 2（v1 路线 2B）：站立软基座 + 可选负载
uv run python -m deploy.sysid.fit --space softbase \
    --runs 'deploy/sysid/data/stand_*.npz' --out deploy/sysid/generated/softbase.json

# 4) 回写拟合 MJCF + 仅供审阅的 DR patch
uv run python -m deploy.sysid.apply_params --identified deploy/sysid/generated/joint.json
```

`fit.py` 的常用参数：`--window`（计分步数，默认 200）、`--warmup`（不计分预热，默认 50）、
`--stride`、`--population`、`--iters-scale`（冒烟时缩小迭代）、`--max-windows`。

## 在 sim2sim 中验证

```bash
TWIST2_MJLAB_G1_XML=deploy/sysid/generated/g1_sysid_fitted.xml \
  TWIST2_MOTION_FILE=/path/to/motion.pkl ./deploy/play_sim_twist2.sh
```

## 真机实时遥操（TWIST2 vs mjlab 对比需要）

`deploy/play_real_twist2.sh` 目前启动的是 motion 文件版策略。要做实时 PICO 遥操，
改为启动 Redis 版策略节点（UDP 协议相同，ghost 字段会被 `hardware_node.py` 忽略）：

```bash
uv run python deploy/policy/twist2_policy_redis.py <model.onnx> --redis-ip localhost &
uv run python deploy/real/hardware_node.py --net eth0
```

`hardware_node.py` 目前只记录 q/dq/IMU；要使用 TWIST2 的
`deploy_real/sysid/compare_metrics.py` 对比工具，需补记
`state.motor.tau_est` / `voltage` / `temperature`。

## 范围与局限（v1）

- 仅离线；在线辨识不在范围内。
- **不做足底接触摩擦辨识**（无动捕/力传感）；站立场景用软基座代理。
- DR 收窄只对未来训练生效：privileged critic 会观测这些 DR 量
  （`src/twist2_mjlab/observations.py:233-275`），对已训练策略无影响。
- `mujoco_warp` 批并行是预期加速路径；当前拟合器是纯 MuJoCo 无梯度实现。
