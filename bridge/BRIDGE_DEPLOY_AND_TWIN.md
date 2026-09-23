# Bridge 部署与数字孪生说明（移植到另一台笔记本）

本文覆盖：bridge 的两种运行模式、环境依赖、移植到新电脑的 XML/机器人适配问题、
使用与按键、以及 B1 数字孪生模式的注意事项。
配套文件：`bridge_twist2_to_orcalab.py`（单文件，无需 OrcaManipulation 仓库）。

---

## 0. TL;DR

- bridge 只依赖 **orca-gym + OrcaLab + redis + onnxruntime**，通过 **Redis** 与 TWIST2 侧通信。
- 两种模式：
  - **策略模式（默认）**：bridge 自己跑 ONNX 策略，吃 teleop 的 mimic，用于双臂采集/遥操作。
  - **B1 孪生模式（`--replay_target`）**：bridge 不跑策略，直接回放**真机下发的关节目标**，两边拿到同一份指令，做 plant 对比。
- 最容易踩的坑：**XML/机器人模型必须满足硬性结构要求**、**多机器人场景必须显式 `--agent_name`**、
  **手部必须是 Dex3-1（7 自由度/手）**、**29 个本体执行器必须是 `<position>`**。
- **观测契约**：本仓库策略是 `1524 = 127(当前) + 11×127(历史)`、无 future 块；bridge 已按此适配（见 1.3）。

---

## 1. 两种模式

### 1.1 策略模式（默认）

```
teleop(GMR) → Redis action_body(35)+action_hand(7+7) → bridge: mimic+OrcaLab proprio → ONNX(29) → OrcaLab position 执行器
```

- 需要 `--policy`。
- 适合：双臂采集、OrcaLab 侧遥操作验证。

### 1.3 观测契约与策略兼容（本仓库）

本仓库训练出的 ONNX（`resources/*.onnx`）的 actor 观测是：

```
obs(1524) = current(127) + history(11 x 127)
current   = mimic(35) + proprio(92)
proprio   = base_ang_vel*0.25(3) + roll/pitch(2) + (joint_pos - default)(29)
            + joint_vel*0.05(29, 踝关节置零) + last_action(29)
```

history 共 11 帧且**包含当前帧**，并且**没有**额外的 future-mimic 块。原版 bridge 面向的是另一套
student_future 规格（`1432 = 127 x (10 + 1) + 35`：10 帧历史 + 35 维未来 mimic），因此本仓库把
bridge 适配为：

- `HISTORY_LEN = 11`
- `TOTAL_OBS_SIZE = 127 x 12 = 1524`（去掉 `+ N_MIMIC_OBS`）
- `build_obs_buf()` 先 append 当前帧再 flatten（历史含当前），并移除 `future_obs`

用 `--policy` 指向 `resources/` 下的策略时不会再触发 obs 维度断言。

### 1.2 B1 孪生模式（`--replay_target`，真机关节指令回放）

```
sim2real → Redis action_low_level_unitree_g1_with_hands(29, 真机电机目标) → bridge → OrcaLab position 执行器（同一份指令）
```

- **不需要 `--policy`**，不跑策略，不做 obs/history。
- 真机侧需开 `--publish_target` 才会发布目标。
- 孪生模式**不写 `state_*`**（避免覆盖真机的 state 遥测）。
- 结果：两边收到同一份关节指令，轨迹差 = plant/接触/仿真差距。
- 建议配 `--fix_feet`（锁腿只比手臂），否则自由基座会倒/漂。

---

## 2. 环境依赖（新电脑必须装）

`setup_env.sh` **不会**装 OrcaGym/OrcaLab，也不会建 bridge 用的 conda 环境。

| 项 | 要求 |
|---|---|
| Python 环境 | 一个含 `orca_gym + orcalab + redis + onnxruntime` 的环境（本机叫 `orcalab_lerobot`，Python 3.12） |
| OrcaGym/OrcaLab | 按 OrcaGym 官方 README 安装；GUI 用 `orcalab` 命令启动 |
| Redis | `redis-server`，teleop/sim2sim/sim2real/bridge 都用；默认 `localhost:6379` |
| 策略文件 | `twist2_1017_20k.onnx`（**仅策略模式需要**） |
| 场景/资产 | OrcaLab 场景 XML + 网格资产（gRPC 模式由 OrcaStudio 下发；本地模式需缓存 XML + 同目录网格） |

安装 bridge 额外依赖：

```bash
conda activate <orcalab 环境>
pip install redis hiredis onnxruntime
```

### 2.1 版本问题

- bridge 不自带版本约束，真正耦合的是 **orca_gym(客户端) ↔ OrcaStudio/OrcaLab(服务端) 的 gRPC 协议**。
- 26.6.3 与 26.7.1 的 proto 差异是**纯增量**（多出的 RPC/optional 字段对老服务端向后兼容），实测 GUI 26.6.3 + bridge 26.7.1 可用。
- **跨代版本**可能缺 RPC → gRPC 报错。最稳：两端同版本；做不到就用 **`--local_xml --no_render` 本地模式绕开 gRPC**。
- bridge 用到的 `OrcaGymLocalEnv` API 在 26.6.3/26.7.1 一致（唯一差异是 anchor body 命名，bridge 不用）。

### 2.2 Redis 网络（跨机时必须）

```bash
# 真机那台（Redis 所在机）
redis-cli config set bind 0.0.0.0
redis-cli config set protected-mode no
ss -tlnp | grep 6379        # 应监听 0.0.0.0:6379
```

- teleop/sim2sim/sim2real **写死 `localhost:6379`**，所以 Redis 只能留在真机那台。
- **只有 bridge 支持远程**：`--redis_host <真机 IP> --redis_port 6379`。
- 跨机注意时钟同步（NTP），否则 `--stale_ms` 判定会漂。

---

## 3. 移植到新电脑的适配问题（重点）

### 3.1 XML 硬性依赖

| 依赖 | 要求 | 不满足后果 |
|---|---|---|
| 29 本体关节 | 名字以 `left_hip_pitch_joint`…`right_wrist_yaw_joint` 结尾，且有对应执行器 | 启动 `KeyError` |
| 根关节 | 必须叫 **`floating_base_joint`** | 启动 `KeyError` |
| 手关节/执行器 | 14 个 Dex3-1（`*_left_hand_thumb_0_joint`…，7/手） | 启动 `KeyError` |
| pelvis body | `--fix_feet` 靠 `*pelvis` 正则注入 weld | 仅 WARNING，下肢硬锁仍生效 |
| 执行器类型 | 29 个本体执行器必须是 **`<position>`** | `<motor>` 会被当力矩，手臂不跟随 |

松耦合（只会 WARNING，不影响运行）：
- `conf.g1_pick_conf` 缺失 → 回退硬编码手部姿势。
- 螺丝刀/手执行器 patch 匹配不到 → `[Hand Grasp] WARNING`。
- 手姿超模型 range → `_validate_hand_poses` WARNING。

### 3.2 多机器人场景

- 若场景里有多个机器人，`_detect_agent_name` 会"平票"，选择不确定。
- **必须显式指定**：`--agent_name unitree_g1_1`（换成你场景里对应机器人的前缀）。
- `--fix_feet` 的 pelvis weld 正则会抓文档里**第一个** `*pelvis`，多机器人场景可能焊错机器人；
  但真正的下肢硬锁是按 agent 前缀的，所以目标机器人不会倒。

### 3.3 手部类型（重要）

bridge 目前**硬编码 Dex3-1（14 关节/7 自由度每手）**：

- 若你的机器人/场景手部是**二指夹爪**（如 `*_gripper_{l,r}_{inner,outer}_joint1..4`）：
  - `HAND_JOINT_NAMES` 找不到 → **启动 `KeyError`**。
  - `--no_hand_grasp` 只关手部补丁，**不解决**缺关节问题。
  - 现状解决办法：**暂不支持夹爪**（后续可加 `--hands {dex3,gripper,none}`）。手臂/本体（29 关节）与策略**不受影响**（mimic_obs 不含手部）。
- 若机器人**没有手**：同样会在手关节解析处失败。

### 3.4 路径与资产

- `--policy`：ONNX 路径。
- `--local_xml`：本地模式 XML；其 `mesh` 相对路径需能解析（XML 与网格同目录，或 gRPC 模式由 OrcaStudio 下发）。
- `--record_file`：录制 JSON 路径（默认 `logs/bridge_record.json`，相对当前工作目录）。
- 资产目录：`~/.orcagym/tmp`（OrcaStudio 缓存）。

### 3.5 patch 说明（2026-09-23 更新）

`patches/twist2.patch` 已重新生成，覆盖 **官方基线 `d5c7108` → 当前 TWIST2** 的**功能性改动（29 个文件）**，
包含此前缺失的 `server_low_level_g1_real.py`、`sim2real.sh`、`robot_control/g1_wrapper.py` 等。
`patches/gmr.patch` 同步重新生成（7 个文件）。

**排除内容**：
- `deploy_real/sysid/`（按要求排除；注意 `server_low_level_g1_sim.py` 里仍保留可选的
  `--sysid_params` 挂钩，默认不启用，无 sysid 目录时不影响运行）
- 论文相关文件（2 个中文翻译 `.md` + `2511.02832v1.txt` + 2 个 PDF；PDF 为二进制，文本 patch 无法表示）
- `MREADME.md`、28 个 scratch `assets/g1/tmp*.xml`

`setup_env.sh` 的 `cp files/twist2/` 步骤仍在，但其中文件与 patch 新增内容完全一致，覆盖无害。
`patches/twist2_publish_target.patch` 是 `--publish_target` 的独立小 diff（用于已 clone 好、只想补这一处的场景）。

---

## 4. 使用

### 4.1 启动顺序

```
Redis → (PICO/adb reverse) → OrcaStudio GUI → teleop → sim2real(或 sim2sim) → bridge
```

- **teleop 必须先于 bridge**，否则读到上次残留的 `action_*`。
- 切场景/重跑前建议清键：
  ```bash
  redis-cli DEL action_body_unitree_g1_with_hands t_action controller_data
  ```

### 4.2 策略模式

```bash
conda activate <orcalab 环境>
python bridge_twist2_to_orcalab.py \
    --policy /path/to/twist2_mjlab/resources/aux_upstream_30k.onnx \
    --fix_feet --device cpu
# 也可换用 resources/ 下的其它策略：
#   pretrained.onnx / pretrained_aux.onnx / amp_upstream_30k.onnx
# 本地无头：加 --local_xml <XML> --no_render
# 跨机：加 --redis_host <真机 IP>
```

### 4.3 B1 孪生模式

```bash
# 真机那台：sim2real 加 --publish_target
python server_low_level_g1_real.py ... --publish_target

# 笔记本：
python bridge_twist2_to_orcalab.py \
    --replay_target --fix_feet --device cpu \
    --redis_host <真机 IP>
# 可选：--replay_gains train（默认 real，用真机 g1.yaml 的 kp/kd 对齐被控对象）
```

### 4.4 无头自检（换 XML/上机前先跑）

```bash
# 策略模式
python bridge_twist2_to_orcalab.py --policy <onnx> \
    --local_xml <XML> --no_render --fix_feet --device cpu --selftest

# 孪生模式
python bridge_twist2_to_orcalab.py --replay_target \
    --local_xml <XML> --no_render --fix_feet --device cpu --selftest
```

看启动日志：
- `Auto-detected robot agent: 'xxx' (matched 29/29 joints)` → 本体关节是否齐全
- `Resolved 14 hand actuator IDs` → 手部是否 Dex3（夹爪会先崩）
- `Overrode 29 body actuator PD gains ...` → 增益覆盖是否成功
- `[Fix Feet] Injected welds ...` / `[Hand Grasp] patched ...` → 补丁是否匹配

### 4.5 按键（真机在跑时特别注意）

| 键 | teleop | bridge |
|---|---|---|
| 右手 A（`RightController.key_one`） | 切四态 | — |
| 右手 B（`RightController.key_two`） | — | 启动 bridge 四态 |
| 左手 A（`LeftController.key_one`） | 退出 teleop | — |
| 左手 B（`LeftController.key_two`） | — | 刷新场景 |
| 右手摇杆按下（`RightController.axis_click`） | — | 录制 toggle |
| **左手摇杆按下（`LeftController.axis_click`）** | **急停（`pkill -f sim2real.sh`）** | 回放 toggle |

> ⚠️ 左手摇杆按下在 teleop 里是急停。**真机在跑时不要按**，否则会杀掉 sim2real。
> 孪生模式不需要录制/回放，但别误按。

### 4.6 Redis 键

| 键 | 生产者 → 消费者 |
|---|---|
| `action_body_unitree_g1_with_hands`(35) | teleop → sim2sim/bridge |
| `action_hand_left/right_unitree_g1_with_hands`(7+7) | teleop → bridge |
| `action_neck_unitree_g1_with_hands`(2) | teleop → sim2sim |
| `t_action`(ms) | teleop → 消费者 stale 判定 |
| `controller_data` | teleop → bridge（按键） |
| `action_low_level_unitree_g1_with_hands`(29) | **sim2real（`--publish_target`）→ bridge（`--replay_target`）** |
| `t_action_low_level`(ms) | sim2real → bridge |
| `state_body_unitree_g1_with_hands`(34) / `t_state` | sim2sim/sim2real/bridge（策略模式）写；注意多写者会互相覆盖 |

### 4.7 同时开 sim2sim + sim2real + bridge 的风险

- **正常**：三者都读 `action_*`，会一起动（无功能冲突）。
- **CPU 争抢**：同一台笔记本跑 OrcaStudio GUI + 两个 MuJoCo + 真机推理 → 掉帧、`t_action` stale 误判 → 真机时序风险。
- **state_* 覆盖**：sim2sim/sim2real/bridge 都写，监控/录制会混；teleop 只写不读，指令不受影响。
- **按键冲突**：见 4.5。
- 建议：**真机运行时不要同机三开**；要看孪生就把 bridge 放另一台笔记本（`--redis_host`）。

### 4.8 常用参数

| 参数 | 说明 |
|---|---|
| `--fix_feet` | 焊接 pelvis + 硬锁下肢（双臂/孪生推荐） |
| `--leg_pd_gain` / `--leg_ema_alpha` | bridge 自己的腿部 PD 调参，与 TWIST2 同名参数无关 |
| `--replay_target` / `--replay_gains` / `--target_key` / `--target_ts_key` | B1 孪生 |
| `--no_pd_override` | 保留 XML 自带 PD 增益（默认覆盖为训练值；孪生默认真机值） |
| `--no_hand_grasp` | 关手部 kp/摩擦补丁（不解决缺手关节） |
| `--stale_ms` | mimic/目标超时判据（默认 200ms） |
| `--agent_name` | 多机器人场景必填 |
| `--selftest` | 无头 5s 自检 |

---

## 5. 已知限制

1. 手部仅支持 **Dex3-1**；夹爪/无手场景会启动失败（待做 `--hands` 抽象）。
2. 29 本体执行器必须 `<position>`；`<motor>` 模型不可用。
3. 多机器人场景不改 `--agent_name` 可能选错机器人；`--fix_feet` weld 可能焊错机器人（硬锁仍正确）。
4. bundle 的 `patches/twist2.patch` 已过期，不能完整复现当前 TWIST2。

---

## 6. 要复制到另一台笔记本 TWIST2 的文件

假设目标路径是 `<另一台>/TWIST2/`。

### 6.1 只有一台笔记本加进现有真机链路（推荐）

复制到 `<另一台>/TWIST2/bridge/`：

```
files/bridge/bridge_twist2_to_orcalab.py
assets/twist2_1017_20k.onnx        →  <另一台>/TWIST2/assets/ckpts/
```

真机那台（本机）加发布器（孪生模式才需要）：
- 方式 A：应用 `patches/twist2_publish_target.patch`（`git apply`）
- 方式 B：直接复制 `deploy_real/server_low_level_g1_real.py`
- 方式 C：手动照第 7 节改 4 处

### 6.2 想在另一台跑完整链路（teleop/sim2sim/sim2real）

推荐：**官方 clone + 并打最新的 `patches/twist2.patch`**（已覆盖当前 TWIST2 全部改动，除 sysid）。
若目标机已有自己的 TWIST2 改动，也可按需直接复制本机的：

```
deploy_real/server_low_level_g1_real.py    # --publish_target / leg 调参
deploy_real/xrobot_teleop_to_robot_w_hand.py
deploy_real/robot_control/g1_wrapper.py
deploy_real/data_utils/
teleop.sh  sim2sim.sh  sim2real.sh
assets/g1/g1_sim2sim_29dof.xml  assets/g1/sim2sim_g1.xml  assets/g1/teleop_g1.xml  assets/g1/compare_dual.xml
assets/ckpts/
```

---

## 7. 真机侧 `--publish_target` 手动改动（4 处）

```python
# 1) 构造函数签名
def __init__(self, ..., leg_pd_gain=1.0, publish_target=False):

# 2) 构造函数体（self.last_pd_target = ... 之后）
self.publish_target = publish_target
if publish_target:
    print("Target publishing enabled: action_low_level_unitree_g1_with_hands + t_action_low_level")

# 3) 主循环（target_dof_pos 计算 + EMA 之后）
if self.publish_target:
    self.redis_client.set("action_low_level_unitree_g1_with_hands",
                          json.dumps(target_dof_pos.tolist()))
    self.redis_client.set("t_action_low_level", int(time.time() * 1000))

# 4) main()：加 --publish_target 参数并透传
parser.add_argument('--publish_target', action='store_true')
controller = RealTimePolicyController(..., publish_target=args.publish_target)
```

---

## 8. 快速排障

| 症状 | 排查 |
|---|---|
| `KeyError: 'xxx_left_hip_pitch_joint'` | XML 缺关节/前缀不对 → 显式 `--agent_name` |
| `KeyError: '..._left_hand_thumb_0_joint'` | 手部不是 Dex3-1（夹爪）→ 暂不支持 |
| 手臂不动 / 乱动 | 29 执行器里有 `<motor>` → 必须 `<position>` |
| OrcaLab 画面不动 | OrcaStudio 是否运行（50051）；是否误加 `--local_xml`/`--no_render` |
| 手部不跟随 | `redis-cli GET action_hand_left_unitree_g1_with_hands` 是否有值 |
| 机器人自己动/陈旧 | `redis-cli GET t_action` 是否在更新；先清 Redis 残留键 |
| gRPC 报错 | 客户端/服务端版本跨代 → 用 `--local_xml --no_render` |
