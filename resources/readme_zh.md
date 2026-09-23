# TWIST2 MJLab — 使用指南

<div align="center">
  <img src="hello.gif" alt="TWIST2 问候 gif" width="360" />
  <img src="real.gif" alt="TWIST2 真实世界动作 gif" width="360" />
</div>

## 概览

`twist2_mjlab` 是一个独立的 MJLab 任务包，用于基于 [TWIST2](https://github.com/amazon-far/TWIST2) 的 Unitree G1 动作跟踪，目的是在受支持的物理引擎（mjwarp）和训练框架上继续开发。注册的任务名称是 `Twist2-Flat-Unitree-G1`，所有任务相关逻辑都保存在本地的 `src/twist2_mjlab/` 下。

该包通过 PKL 动作库加载动作参考，现在支持两条彼此独立的流程。你可以只用其中一条，也可以两条都用，取决于你要使用哪个数据集：

1. TWIST2：使用 MuJoCo 前向运动学补全原始 TWIST2 PKL 文件，
2. SEED：把 SEED 数据集里的 G1 CSV 动作转换成补全后的 TWIST2 PKL，
3. 将任务指向生成后的动作文件或 dataset YAML，
4. 使用 `train_twist2.sh` 或 `train_seed.sh` 训练，
5. 使用 `play_twist2.sh` 或 `play_seed.sh` 可视化。

## TODO
- [x] 解耦式 sim2sim 流水线（sim 节点 + policy 节点通过 UDP 50 Hz 通信，实时 MuJoCo 查看器带参考动作绿色影子叠加）。
- [x] 在 Unitree G1 上的硬件部署（与 sim2sim 共用 policy 节点，硬件状态节点通过内置的 Unitree SDK2 读写机器人，PD 增益与动作缩放沿用 MJLab G1 定义）。

## 包含内容

```
twist2_mjlab/
├── pyproject.toml              # MJLab 任务包 + MJLab 入口点
├── train_twist2.sh             # 训练 `Twist2-Flat-Unitree-G1`
├── train_seed.sh               # 训练补全后的 SEED G1 数据集
├── play_twist2.sh              # 播放最新或指定检查点
├── play_twist2_pretrained.sh   # 直接播放内置的预训练检查点
├── play_seed.sh                # 播放最新或指定的 SEED 检查点
├── play_seed_pretrained.sh     # 直接播放内置的 SEED 预训练检查点
├── sim2sim_pretrained.sh       # 一键启动预训练模型 sim2sim
├── sim2sim_seed.sh             # SEED 版本的一键 sim2sim
├── sim2sim_seed_pretrained.sh  # 使用 SEED 预训练 ONNX 的一键 sim2sim
├── resources/
│   ├── pretrained.pt           # 预训练检查点（30K iterations）
│   ├── pretrained.onnx         # 预训练 ONNX（原版奖励，4096 envs）
│   ├── pretrained_seed.pt      # SEED 预训练检查点（30K iterations）
│   ├── pretrained_seed.onnx    # SEED 预训练 ONNX 模型（用于 sim2sim）
│   ├── pretrained_aux.onnx     # aux（可微目标，tuned 奖励）30K ONNX
│   ├── aux_upstream_30k.onnx   # aux + 原版奖励 30K ONNX
│   ├── amp_upstream_30k.onnx   # AMP + 原版奖励 30K ONNX
│   ├── hello.gif               # README 演示资源
│   ├── example.gif             # README 演示资源
│   └── readme_zh.md            # 中文使用说明
├── bridge/                     # OrcaLab 双臂遥操作 bridge（已适配本仓库 1524 维策略）
│   ├── bridge_twist2_to_orcalab.py
│   └── BRIDGE_DEPLOY_AND_TWIN.md
├── deploy/                     # Sim2sim + 真机部署流水线
│   ├── play_sim_twist2.sh      # Sim2sim 启动脚本（MuJoCo + policy）
│   ├── play_real_twist2.sh     # 真机启动脚本（G1 + policy）
│   ├── install_unitree_sdk.sh  # 一次性安装 Unitree SDK2 绑定
│   ├── export_onnx.py          # 检查点 -> ONNX 导出
│   ├── common/udp_sync.py      # UDP 状态/动作协议
│   ├── sim/sim_node.py         # MuJoCo 物理仿真 + 参考动作半透明绿色叠加
│   ├── policy/twist2_policy.py # ONNX 推理 + 动作库
│   └── real/                   # 真机部署
│       ├── hardware_node.py        # 通过 unitree_interface 运行 50 Hz G1 控制循环
│       ├── g1_robot_constants.py   # 冻结的 PD 增益 / 默认姿态
│       └── unitree_sdk2_wrapper/   # Git 子模块（SDK2 C++ 源码 + pybind11）
└── src/twist2_mjlab/
    ├── __init__.py             # 任务注册
    ├── commands.py             # PKL 动作命令与重采样
    ├── config.py               # 观测、奖励、终止条件、域随机化
    ├── observations.py         # Actor / critic 观测项
    ├── pkl_motion_lib.py       # 补全后的 PKL 读取与插值
    ├── rewards.py              # 跟踪与正则化奖励
    ├── terminations.py         # 失败 / 超时条件
    ├── rl_cfg.py               # 运行器与模型配置
    └── scripts/enrich_pkl.py   # 给 PKL 添加世界系身体数据
```

## 快速开始

### 1) 安装包

请在 `twist2_mjlab/` 目录下运行所有命令：

```bash
cd /path/to/twist2_mjlab
uv sync
```

### 2) 准备动作数据

`PklMotionLib` 期望 PKL 文件符合 BeyondMimic 的格式。如果你是从原始 [TWIST2 motions](https://drive.google.com/file/d/1JbW_InVD0ji5fvsR5kz7nbsXSXZQQXpd/view) 开始，请先运行补全脚本：

```bash
uv run python -m twist2_mjlab.scripts.enrich_pkl \
  --dataset /path/to/twist2_dataset.yaml \
  --output-dir /path/to/enriched/ \
  --workers 8
```

该脚本会读取 dataset YAML，为每个 PKL 运行 MuJoCo 前向运动学，写出包含 `body_pos_w` 和 `body_quat_w` 的补全版 PKL，并在输出目录中生成新的 `dataset.yaml`。

#### Kimodo 文本到动作桥接

如果你已经把同级目录下的 `~/kimodo` 仓库安装到独立的 conda 环境中，并希望通过一条命令完成“文本提示 -> Kimodo 生成 -> TWIST2 播放”，本仓库提供了：

- `src/twist2_mjlab/scripts/kimodo_csv_to_pkl.py` —— 将 Kimodo 导出的 G1 MuJoCo qpos CSV 转成 TWIST2 可用的补全 PKL
- `kimodo_to_twist2.sh` —— 文本提示 -> Kimodo 生成 -> CSV 转 PKL -> TWIST2 播放

这个桥接流程假设 `kimodo` conda 环境已经激活。先在一个终端中保持 Kimodo 文本编码服务运行：

```bash
conda activate kimodo
kimodo_textencoder
```

然后在第二个终端中，从 `twist2_mjlab/` 目录启动整条流程：

```bash
conda activate kimodo
cd /home/yiling/twist2_mjlab
./kimodo_to_twist2.sh "bend down and pick up a box"
```

说明：

- 该桥接脚本会在当前激活的 `kimodo` 环境下通过 `python -m kimodo.scripts.generate` 调用 Kimodo
- 该流程面向 **G1** Kimodo 模型，不能直接使用 `Kimodo-SOMA-RP-v1`
- 包装脚本默认使用的模型是 `Kimodo-G1-RP-v1`
- 默认播放路径使用 `play_seed_pretrained.sh`
- 这是“提示词 -> 先生成完整片段 -> 转换 -> 播放”的自动化流程，不是真正的流式实时生成

#### SEED 数据集支持

仓库里还提供了面向 Hugging Face 数据集 [`bones-studio/seed`](https://huggingface.co/datasets/bones-studio/seed) 的专用 SEED 流程。访问前需要先在 Hugging Face 页面接受数据集条款。这个流程是可选的；你可以只用 TWIST2 流程、只用 SEED 流程，或者两个都用。

SEED 补全脚本需要 G1 CSV 动作文件和 metadata CSV。`seed_enrich.py` 的默认路径是：

- `~/twist2/seed/g1/csv`
- `~/twist2/seed/seed_metadata_v003.csv`

如果你沿用 Hugging Face 的目录结构，可以显式传 `--metadata`，或者把 `metadata/seed_metadata_v003.csv` 复制 / 软链接到默认位置。

把 SEED G1 CSV 转成补全后的 PKL 和 dataset YAML：

```bash
uv run python -m twist2_mjlab.scripts.seed_enrich \
  --csv-dir ~/twist2/seed/g1/csv \
  --metadata ~/twist2/seed/metadata/seed_metadata_v003.csv \
  --output-dir ~/twist2/seed_g1_enriched_pkl \
  --fps 30 \
  --workers 8
```

这会把补全后的 PKL 写到 `~/twist2/seed_g1_enriched_pkl/`，并生成：

- `seed_dataset.yaml` —— 全部动作
- `seed_dataset_filtered.yaml` —— 过滤后的动作

如果你已经把 metadata CSV 放到了默认路径 `~/twist2/seed/seed_metadata_v003.csv`，那就可以省略 `--metadata`。

**注意:** 如果你想先直接体验一下，这个包已经自带了一个训练到 30K iterations 的预训练 checkpoint，直接运行 `play_twist2_pretrained.sh` 即可。

### 3) 训练

对于原始 TWIST2 动作，`TWIST2_MOTION_FILE` 可以指向单个补全后的 `.pkl`，也可以指向包含多个动作的 dataset `.yaml`：

```bash
TWIST2_MOTION_FILE=/path/to/enriched/dataset.yaml bash train_twist2.sh 0
```

对于 SEED 流程，`train_seed.sh` 默认使用 `seed_enrich.py` 的输出：

```bash
bash train_seed.sh 0
```

说明：

- 第一个位置参数是 GPU 编号（默认是 `0`），
- 额外的 CLI 参数会继续传递给 MJLab 的 `train` 命令，
- `train_twist2.sh` 的训练日志会写入 `logs/rsl_rl/g1_twist2_flat/`，
- `train_seed.sh` 的训练日志会写入 `logs/rsl_rl/g1_twist2_seed_flat/`。

如果你修改了 SEED 输出目录，请同步更新 `train_seed.sh` 里的 `MOTION_FILE`，或者创建一个指向 `~/twist2/seed_g1_enriched_pkl/seed_dataset.yaml` 的软链接。

如果你想同时使用 TWIST2 和 SEED，两条流程分别运行即可；它们使用不同的动作文件和日志目录，不会互相影响。

#### 关于 W&B 以及保存内容

这个包默认会把训练记录到 Weights & Biases。

该任务的 W&B 默认值是：

- project：`twist2_mjlab`
- experiment name：`g1_twist2_flat`
- run name：`g1_twist2_flat`

首次运行前，请先完成 W&B 登录：

```bash
wandb login
```

如果你不想使用交互式登录，也可以直接设置 `WANDB_API_KEY`。

默认情况下，W&B 会保存：

- 训练标量，例如 episode 统计、loss、学习率、action 标准差，以及 FPS / 性能指标
- 训练与环境配置（`agent.yaml` 和 `env.yaml`）
- 本次运行所用本地仓库的 git 状态，包括 commit hash、status 和 diff
- 在运行目录下找到的日志视频（`*.mp4`）
- 当启用 `upload_model` 时，模型 checkpoint 和导出的 policy 文件；该选项默认开启

如果你不想使用 W&B，可以在启动训练时将 logger 切换为 TensorBoard：

```bash
TWIST2_MOTION_FILE=/path/to/enriched/dataset.yaml bash train_twist2.sh 0 --agent.logger tensorboard
```

如果你只想在环境层面禁用 W&B，也可以设置 `WANDB_MODE=disabled`。

### 4) 播放 / 可视化

```bash
TWIST2_MOTION_FILE=/path/to/enriched/dataset.yaml bash play_twist2.sh
```

如果你不传入 checkpoint 路径，`play_twist2.sh` 会自动从 `logs/rsl_rl/g1_twist2_flat/` 下最新的 run 目录里选择最新的 `model_*.pt`。

你也可以显式指定 checkpoint：

```bash
TWIST2_MOTION_FILE=/path/to/enriched/dataset.yaml bash play_twist2.sh /path/to/model_12345.pt
```

也可以直接运行预训练脚本：

```bash
TWIST2_MOTION_FILE=/path/to/enriched/dataset.yaml bash play_twist2_pretrained.sh
```

直接运行 `play_twist2_pretrained.sh` 时，会使用训练到 30K 步的预训练模型。

对于 SEED 流程，可以使用对应的脚本，并指向同一个补全后的动作文件或单个动作 PKL：

```bash
TWIST2_MOTION_FILE=/path/to/enriched/seed_dataset.yaml bash play_seed.sh
```

如果想直接试用内置的 SEED 预训练模型：

```bash
TWIST2_MOTION_FILE=/path/to/enriched/seed_motion.pkl bash play_seed_pretrained.sh
```

`play_seed.sh` 在不显式传入检查点时，会自动从 `logs/rsl_rl/g1_twist2_seed_flat/` 下选择最新的检查点。

说明：

- play 脚本默认使用 `--device cpu` 和 `--viewer native`，
- 额外的 CLI 参数会继续传递给 MJLab 的 `play` 命令，
- 如果没有设置 `TWIST2_MOTION_FILE` 且终端是交互式的，脚本会提示输入。

你可以只用 TWIST2 的播放脚本，只用 SEED 的播放脚本，或者两者都用，具体取决于你训练了哪些检查点。

### 5) Sim2sim 部署

sim2sim 流水线采用类硬件的解耦双进程架构运行训练好的策略：**sim 节点**（MuJoCo 物理仿真 + 查看器）和 **policy 节点**（ONNX 推理 + 动作库），通过 UDP 异步通信。两个进程各自维护独立的实时时钟——如果 policy 稍慢，sim 会继续使用上一条指令运行，就像真实执行器一样。查看器中会显示一个半透明的绿色"影子"机器人，表示策略正在跟踪的参考动作。

**最快体验——使用预训练模型：**

```bash
bash sim2sim_pretrained.sh
```

这会使用内置的 ONNX 模型和示例动作片段，无需训练或导出步骤。

对于 SEED，对应脚本是：

```bash
TWIST2_MOTION_FILE=/path/to/enriched/seed_motion.pkl bash sim2sim_seed_pretrained.sh
```

`sim2sim_seed.sh` 和 TWIST2 版本的用法一样，只是它会去 `logs/rsl_rl/g1_twist2_seed_flat/` 查找检查点，并且要求 `TWIST2_MOTION_FILE` 指向单个补全后的 `.pkl` 动作文件。

**使用自己训练的检查点：**

传入 `.pt` 检查点（自动导出为 ONNX）或直接传入 `.onnx` 文件：

```bash
# 从 .pt 检查点启动（自动导出 ONNX）
TWIST2_MOTION_FILE=/path/to/enriched/motion.pkl \
  ./deploy/play_sim_twist2.sh /path/to/model_29999.pt

# 从已导出的 .onnx 启动
TWIST2_MOTION_FILE=/path/to/enriched/motion.pkl \
  ./deploy/play_sim_twist2.sh /path/to/model.onnx

# 不传模型参数：自动从 logs/ 选择最新检查点
TWIST2_MOTION_FILE=/path/to/enriched/motion.pkl \
  ./deploy/play_sim_twist2.sh
```

#### 使用带 aux（可微辅助目标）策略的步骤说明

可微 aux 只是**训练时**的方法；导出的 ONNX 和普通策略完全一致（输入 `(1, 1524)`、输出 `(1, 29)`），因此部署流程不变。分两种情况：

**A. 直接使用内置的 aux 策略（world model，30K）**

```bash
# A1) sim2sim（不需要 Redis / VR）
TWIST2_MOTION_FILE=/path/to/enriched/motion.pkl \
  ./deploy/play_sim_twist2.sh resources/pretrained_aux.onnx

# A2) 实时遥操作仿真（需要 Redis + teleop 发布端，见「实时遥操作仿真（Redis 链路）」）
bash deploy/play_sim_twist2_redis.sh resources/pretrained_aux.onnx
```

**B. 使用自己训练的 aux checkpoint**

```bash
# B1) 训练时开启 aux：环境产出 aux 观测组（TWIST2_ENABLE_AUX=1）+ 辅助损失（aux-coef > 0）
TWIST2_ENABLE_AUX=1 TWIST2_MOTION_FILE=/path/to/enriched/dataset.yaml bash train_twist2.sh 0 \
  --agent.algorithm.aux-mode world_model \
  --agent.algorithm.aux-coef 0.1 \
  --agent.algorithm.aux-coef-warmup-iters 1000 \
  --agent.algorithm.aux-model-lr 3e-4
# checkpoint 位于：logs/rsl_rl/g1_twist2_flat/<RUN>/model_*.pt

# B2) 导出 ONNX
TWIST2_MOTION_FILE=/path/to/enriched/sub1_clothesstand_000.pkl \
  uv run python deploy/export_onnx.py logs/rsl_rl/g1_twist2_flat/<RUN>/model_29999.pt
# 结果：logs/rsl_rl/g1_twist2_flat/<RUN>/<RUN>.onnx

# B3) 用该 ONNX 跑 sim2sim 或遥操作仿真（同 A1 / A2）
TWIST2_MOTION_FILE=/path/to/enriched/motion.pkl \
  ./deploy/play_sim_twist2.sh logs/rsl_rl/g1_twist2_flat/<RUN>/<RUN>.onnx
bash deploy/play_sim_twist2_redis.sh logs/rsl_rl/g1_twist2_flat/<RUN>/<RUN>.onnx
```

> 部署时**不需要**世界模型和 aux 观测：`L_aux`、`aux_state`、`aux_ref_future` 只存在于训练侧，不会进 ONNX。

和训练、播放一样，TWIST2 与 SEED 的 sim2sim 启动器也是彼此独立的：你可以按需使用任意一个，或者都用来对比结果。

**工作原理：**

该流水线将物理仿真与神经网络推理解耦为两个完全独立的实时进程，模拟真实机器人的工作方式——执行器在等待下一条指令时保持上一条指令：

```
sim_node (MuJoCo, 1000 Hz)        policy_node (ONNX, 50 Hz)
  独立实时时钟                        独立实时时钟
  步进物理 (20 × 0.001s)             加载动作库 (PKL)
  打包机器人状态 ──50Hz UDP──>        获取最新状态 (50 Hz)
                                     构建观测：
                                       mimic (35D) 来自动作参考
                                       proprio (92D) 来自机器人状态
                                       history (11 × 127D)
  获取最新动作 <──50Hz UDP──           运行 ONNX 推理 → 29D 动作
  无新动作则保持上一条指令             发送动作 + 参考姿态
  渲染查看器 + 绿色影子
```

- 两个进程各自运行独立的实时时钟，互不阻塞——UDP 采用发射后不管模式。如果任一方暂时偏慢，另一方继续使用最新可用数据运行。
- **sim 节点** (`deploy/sim/sim_node.py`) 以 1000 Hz 运行 MuJoCo G1 物理仿真（时间步长 0.001s，20 倍降采样 → 50 Hz 控制频率）。每个控制周期发送状态并获取 policy 的最新动作。如果没有新动作到达，`data.ctrl` 保持上一条指令——和真实执行器行为一致。
- **policy 节点** (`deploy/policy/twist2_policy.py`) 以 50 Hz 运行。加载动作库以构建 35D mimic 观测（参考关节位置 + 根状态），维护 11 帧的观测历史，并运行导出的 ONNX actor 网络。
- 每段动作播放时会有 **3 秒的渐入**（从默认站立姿态过渡）和 **3 秒的渐出**（过渡回站立姿态），然后循环。
- 绿色影子的朝向会被校正为始终面向 +X 方向。

**环境变量：**

| 变量 | 说明 |
|------|------|
| `TWIST2_MOTION_FILE` | 补全后的 `.pkl` 或 dataset `.yaml` 的路径（必填） |
| `TWIST2_MOTION_INDEX` | 多动作数据集中要播放的动作索引（默认 `0`） |
| `TWIST2_INIT_YAW_DEG` | 机器人初始偏航角，单位度（默认 `0`） |

#### 实时遥操作仿真（Redis 链路）

除了上面基于 pkl 动作库的 sim2sim，本仓库还提供一条**实时遥操作**链路：**遥操作发布端**把 35D mimic 写进 Redis，**policy 节点**（`deploy/policy/twist2_policy_redis.py`）读取该键、运行 ONNX、并通过 UDP 发送动作，**sim 节点**照旧渲染。整个流程由 `deploy/play_sim_twist2_redis.sh` 启动。

Redis key 与格式：`action_body_unitree_g1_with_hands`，值是 35 个浮点数的 JSON 列表：
`[root_vel_x, root_vel_y, root_z, roll, pitch, yaw_ang_vel, 29 × 关节位置]`。

**前置条件：**

- Redis 服务：`redis-server --daemonize yes`（用 `redis-cli ping` 验证返回 `PONG`）。
- 遥操作发布端：原 TWIST2 仓库的 `teleop.sh`（需要 PICO/VR，使用 `gmr` conda 环境），它运行 `deploy_real/xrobot_teleop_to_robot_w_hand.py` 并写入上面的 key。

**步骤：**

```bash
# 1) 把某个 checkpoint 导出为 ONNX（例如用可微 aux 目标训练到 30K 的策略）
cd /path/to/twist2_mjlab
TWIST2_MOTION_FILE=/path/to/enriched/sub1_clothesstand_000.pkl \
  uv run python deploy/export_onnx.py logs/rsl_rl/g1_twist2_flat/<RUN>/model_29999.pt
# 结果：logs/rsl_rl/g1_twist2_flat/<RUN>/<RUN>.onnx

# 2) 启动遥操作发布端（另开终端，需要 PICO/VR）
cd /path/to/TWIST2
bash teleop.sh                     # 可选：--mode tuned / --mode fix_feet
# 验证是否在实时更新（值应持续变化）：
# while true; do redis-cli get action_body_unitree_g1_with_hands | md5sum; sleep 1; done

# 3) 启动 mjlab 侧（policy + sim 查看器）
cd /path/to/twist2_mjlab
bash deploy/play_sim_twist2_redis.sh \
  logs/rsl_rl/g1_twist2_flat/<RUN>/<RUN>.onnx
```

> 仓库已内置可微 aux（world model）策略 30K 的 ONNX：`resources/pretrained_aux.onnx`。若只想试这个策略，可跳过第 1 步，第 3 步直接运行 `bash deploy/play_sim_twist2_redis.sh resources/pretrained_aux.onnx`。

- 绿色半透明“影子”是遥操作参考姿态，实体机器人是策略输出；`Ctrl-C` 会同时结束 policy 与 sim 两个节点。
- policy 节点启动时先用默认站立姿态，读到 Redis 新值后立即切换到遥操作流。
- 如果 Redis 里的值**一直不变**（即没有 teleop 发布端在跑），机器人只会保持最后收到的姿势。这可以单独用来验证策略的站立稳定性，但不构成“遥操作”。

**没有 VR 时的替代方案：**

- 直接用 pkl 动作库的 sim2sim（不需要 Redis / VR）：
  ```bash
  TWIST2_MOTION_FILE=/path/to/enriched/motion.pkl \
    bash deploy/play_sim_twist2.sh /path/to/model_29999.pt
  ```
- 或者写一个「PKL → Redis」的 50 Hz 回放发布器，复用 `deploy/policy/twist2_policy.py` 里的 `build_mimic_from_frame`，把录制的动作当作遥操作流写进同一个 key。这样就可以在没有 VR 的情况下跑完整的 Redis 链路；把上面的第 2 步替换成该回放器即可。

**部署调参（低层 PD/EMA + mimic 平滑）：**

本仓库的 deploy 侧已加入原版 TWIST2 `deploy_real/` 里的腿部 PD 增益、EMA 低通与 mimic 平滑开关。
所有参数**默认全部关闭或中性（1.0 / 0.0）**，不设置时行为与之前逐位一致；参数带范围校验，越界会在启动时直接报错退出。

*改动概览：*

- 新增 `deploy/common/smoothing.py`：共享的 EMA/滑动窗滤波器、35D mimic 平滑流水线，以及参数校验和共享 argparse 接线（`validate_alpha`、`validate_pd_gain`、`add_smoothing_args`、`build_smoother`）。
- 新增 `deploy/common/forward_env_args.sh`：三个启动脚本共用的「环境变量 → 命令行参数」透传，避免各脚本各写一份。
- `deploy/sim/sim_node.py`：新增 `--leg_pd_gain`、`--arm_pd_gain`、`--leg_ema_alpha`。增益只缩放位置执行器的 kp/kd（`gainprm[0]`、`biasprm[1]`、`biasprm[2]`），**不缩放力矩上限**，让 sim 的力矩饱和行为与真机一致；腿 EMA 作用在写 `data.ctrl` 前的 12 个腿关节目标上。
- `deploy/real/hardware_node.py`：新增同样的三个参数。策略环里用缩放后的 kp/kd，并对 12 个腿目标做 EMA；启动插值到默认姿态、以及 SELECT/B 的减阻尼停机仍使用原始 `_KP`/`_KD`，不受调参影响。
- `deploy/policy/twist2_policy.py`、`deploy/policy/twist2_policy_redis.py`：新增 `--leg_smooth_alpha`、`--arm_smooth_alpha`、`--smooth_body`、`--smooth_window_size`，在拼观测/推理前对 35D mimic 做平滑（pkl 策略在动作循环重启时清空平滑状态）。

> **低层参数和 mimic 参数的区别**：低层参数（`PD_GAIN`/`LEG_EMA_ALPHA`）直接改执行器/电机的 PD 刚度和下发目标；mimic 参数改的是**策略的输入观测**（参考动作），不直接改电机。前者用来补接触/跟踪误差，后者用来压参考动作的抖动。二者可叠加使用。

*参数说明：*

| 参数 | 作用位置 | 命令行 | 默认 | 取值范围 | 公式 / 方向 | 推荐值 |
|------|----------|--------|------|----------|-------------|--------|
| `TWIST2_LEG_PD_GAIN` | sim / 真机 | `--leg_pd_gain` | `1.0` | `(0, 2.0]` | 腿 12 关节 kp、kd 同时乘该系数 | 真机 1.5~2.0；sim 2.0 |
| `TWIST2_ARM_PD_GAIN` | sim / 真机 | `--arm_pd_gain` | `1.0` | `(0, 2.0]` | 臂 14 关节（索引 15~28）kp、kd 乘该系数 | 1.0~1.5 |
| `TWIST2_LEG_EMA_ALPHA` | sim / 真机 | `--leg_ema_alpha` | `0.0` | `[0, 1]` | `t = a·prev + (1−a)·new`，**越大越平滑** | 0.5~0.7 |
| `TWIST2_LEG_SMOOTH_ALPHA` | policy | `--leg_smooth_alpha` | `0.0` | `[0, 1]` | `s = a·new + (1−a)·prev`，**越大越不平滑** | 0.8 |
| `TWIST2_ARM_SMOOTH_ALPHA` | policy | `--arm_smooth_alpha` | `0.0` | `[0, 1]` | 同上一行，作用于 mimic 臂段 `[21:35]` | 0.5~0.8 |
| `TWIST2_SMOOTH_BODY` | policy | `--smooth_body` | `0.0` | `[0, 1]` | `s = a·new + (1−a)·prev`，作用于完整 35D mimic | 0.3~0.5 |
| `TWIST2_SMOOTH_WINDOW` | policy | `--smooth_window_size` | `1` | `>= 1` 整数 | 最近 N 帧 mimic 的算术平均；`1` 表示关闭 | 3~5 |

几点具体说明：

- **`LEG_PD_GAIN` 的用途**：原版在 MuJoCo 软接触下腿跟踪会衰减，把腿 kp/kd 整体放大可补偿，官方建议 2.0。真机上它让电机更硬更阻尼，能改善落地/跟踪，但过大会放大接触冲击。
- **`LEG_PD_GAIN` 不会放大力矩上限**：sim 用的是 mjlab 里与真机一致的电机力矩上限（如 5020/7520/4010 的额定值），所以 sim 里调稳后再上真机，饱和行为是可预期的；真机 kp 提高后若力矩打满，表现为跟踪变差而不是获得额外扭矩。
- **两个 alpha 方向相反**：`LEG_EMA_ALPHA` 的 `a` 是「保留旧值的比例」，所以越大越平滑；其余三个 mimic alpha 的 `a` 是「新值的权重」，越大越跟手、平滑越弱。不要混用直觉。
- **`LEG_SMOOTH_ALPHA` 的掩码**：只平滑 35D mimic 的 `[0:6]`（root 的 vx、vy、z、roll、pitch、yaw_vel）和 `[6:18]`（12 个腿关节），臂和其余部分不动，对应原版遥操作里「腿+root 去抖、手臂保低延迟」的做法。
- **`SMOOTH_WINDOW`**：滑动窗均值比 EMA 更“重”，会引入约 `(N−1)/2 × 20ms` 的相位延迟，适合消掉高频抖动，不适合快速动作的开头；一般和 `SMOOTH_BODY` 二选一，避免过平滑。
- **pkl 策略的绿色 ghost**：mimic 平滑作用于策略输入；ghost 仍按原始参考动作渲染。如果发现 ghost 与实际跟踪有偏差，这是预期现象（redis 链路的 ghost 由 mimic 反推，会同步平滑）。

*怎么使用：*

两种等价方式，可任意组合。**推荐用环境变量**，因为启动脚本会自动透传给正确的节点。

方式一：环境变量 + 启动脚本（推荐）

```bash
# sim2sim：同时启用低层腿 PD 增益和腿目标低通
TWIST2_LEG_PD_GAIN=2.0 TWIST2_LEG_EMA_ALPHA=0.6 \
  bash deploy/play_sim_twist2.sh resources/pretrained.onnx

# sim2sim：只压参考动作的腿/root 抖动
TWIST2_LEG_SMOOTH_ALPHA=0.8 bash deploy/play_sim_twist2.sh resources/pretrained.onnx

# Redis 遥操作链路
TWIST2_LEG_PD_GAIN=1.5 TWIST2_LEG_EMA_ALPHA=0.5 \
  bash deploy/play_sim_twist2_redis.sh resources/pretrained_aux.onnx

# 真机
TWIST2_LEG_PD_GAIN=1.5 TWIST2_LEG_EMA_ALPHA=0.5 \
  bash deploy/play_real_twist2.sh /path/to/model.onnx
```

方式二：直接传给节点（不经过启动脚本时）

```bash
# 低层节点
python deploy/sim/sim_node.py --leg_pd_gain 2.0 --leg_ema_alpha 0.6
python deploy/real/hardware_node.py --net eth0 --leg_pd_gain 1.5 --arm_pd_gain 1.2

# policy 节点
python deploy/policy/twist2_policy.py model.onnx --motion-file motion.pkl \
  --leg_smooth_alpha 0.8 --smooth_body 0.3 --smooth_window_size 5
```

*推荐调参顺序：*

1. **先固定模型，只调低层 `LEG_PD_GAIN`**：从 1.0 → 1.5 → 2.0，找腿部跟踪最好且不抖/不弹的档位。sim 里可到 2.0。
2. **再叠加 `LEG_EMA_ALPHA`（0.3 → 0.7）**降腿的高频抖动。注意它是对 PD 目标做低通，太大会让腿反应变迟钝。
3. **mimic 侧最后调**：参考动作本身抖动明显时用 `LEG_SMOOTH_ALPHA`（腿+root）或 `ARM_SMOOTH_ALPHA`（臂）；需要更重的去抖再用 `SMOOTH_WINDOW`，但它有相位延迟。
4. **真机从小值开始**：先 `LEG_PD_GAIN=1.0`（默认）跑通，再逐步加到 1.5 左右；随时准备 SELECT 急停。真机的臂增益默认 1.0 即可。

*注意事项：*

- 取值范围会被校验：`PD_GAIN ∉ (0, 2.0]`、`alpha ∉ [0, 1]`、`SMOOTH_WINDOW < 1` 都会在启动时报错退出，不会带病上机。
- 启动与停机不用缩放后的增益：`hardware_node.py` 只有在进入 50 Hz 策略环后才应用 `leg_pd_gain`/`arm_pd_gain` 和 EMA，START 插值、SELECT/B 停机仍是默认增益。
- 如果上游原版 teleop 已经配了同类平滑，Redis 链路再开 `SMOOTH_*` 会双重平滑；默认全关可规避，需要时二选一。
- 这些只是**部署/推理期**的开关，不影响也不需要重训；训练侧另有域随机化的电机强度（`_TWIST2_MOTOR_STRENGTH_RANGE`）。

### 6) 硬件部署

真机部署与 sim2sim 共用同一个 policy 节点和 UDP 协议，只是把 MuJoCo 仿真换成了一个 50 Hz 的硬件循环：通过内置的 Unitree SDK2 包装层读取 G1 的 IMU 与关节状态，再把 policy 输出的关节目标以 PD 控制下发到电机，PD 增益与 MJLab G1 定义保持一致。

**前置条件：**

- Ubuntu 主机，通过有线网络连接 G1（默认接口 `eth0`），
- Python 3.10（已由 `pyproject.toml` 固定），以便使用预编译的 `.cpython-310` 绑定，
- 首次运行需要 `sudo` 权限以安装 `build-essential`、`cmake`、`python3-dev`、`pybind11-dev`。

**一次性安装 Unitree SDK2 绑定：**

```bash
git submodule update --init deploy/real/unitree_sdk2_wrapper
./deploy/install_unitree_sdk.sh
```

脚本会编译 C++ 包装层，并把 `unitree_interface.so` 安装到 twist2_mjlab 的 uv 环境里。可以用下面的命令验证：

```bash
uv run python -c "import unitree_interface; print('ok')"
```

**在机器人上运行：**

```bash
TWIST2_MOTION_FILE=/path/to/enriched/motion.pkl \
  TWIST2_REAL_NET=eth0 \
  ./deploy/play_real_twist2.sh /path/to/model_29999.pt
```

模型参数约定与 `play_sim_twist2.sh` 完全相同：可以传入 `.pt`（自动导出为 ONNX）、直接传入 `.onnx`，或不传任何参数以自动选择 `logs/rsl_rl/g1_twist2_flat/` 下最新的 checkpoint。

**手柄启动流程**（与 `hardware_node.py` 控制台提示一致）：

1. **START** — 解除阻尼保持，在 2 秒内平滑过渡到默认站立姿态。
2. **A** — 进入 50 Hz policy 控制循环。
3. **B** — 优雅退出（在当前姿态上进入阻尼保持）。
4. **SELECT** — 紧急阻尼停止，发生任何异常情况时优先按这个键。

**与 sim2sim 的差异：**

- 没有 MuJoCo 查看器，也没有绿色影子叠加。硬件节点仍会**接收** policy 发回的参考姿态字段，但直接丢弃，因为没有渲染目标。
- 机器人没有里程计，所以状态包里 `root_pos` 和 `body_lin_vel` 填 0。Policy 的本体感知只使用 `body_ang_vel`、IMU 四元数和关节状态，这与训练时一致。
- 不依赖 ROS。硬件节点只用 UDP + DDS（DDS 部分由内置 SDK 包装层处理）。

**环境变量：**

| 变量 | 说明 |
|------|------|
| `TWIST2_MOTION_FILE` | 动作参考文件（与 sim2sim 相同） |
| `TWIST2_MOTION_INDEX` | dataset YAML 中的动作索引（默认 `0`） |
| `TWIST2_REAL_NET` | 连接 G1 的 DDS 网络接口（默认 `eth0`） |

sim2sim 一节的调参环境变量（`TWIST2_LEG_PD_GAIN`、`TWIST2_ARM_PD_GAIN`、`TWIST2_LEG_EMA_ALPHA`、
`TWIST2_LEG_SMOOTH_ALPHA`、`TWIST2_ARM_SMOOTH_ALPHA`、`TWIST2_SMOOTH_BODY`、`TWIST2_SMOOTH_WINDOW`）
同样会被 `play_real_twist2.sh` 透传。

**安全建议：**

- 始终保持手握手柄，**SELECT** 是最快的应急出口。
- 启动前请把机器人悬吊起来，或者由另一人扶住。按下 **A** 的瞬间控制权就交给 policy 了。
- Ctrl-C 时启动脚本的清理钩子会 `pkill` 掉两个节点，硬件节点退出前会先在当前姿态阻尼保持。

## 可微辅助目标（Differentiable Auxiliary Objective）

本仓库在原有 PPO 训练之上新增了一个**可微的闭环辅助目标**。它**不改变策略网络结构，也不改变推理/部署链路**，只在训练时额外提供一条「动作 → 未来状态 → 未来跟踪误差」的梯度路径。

### 原理

- 原始 PPO 只优化单步代理目标，梯度不经过机器人动力学，credit assignment 只有一步；同时参考运动由时钟驱动（`PklMotionCommand._update_command` 中的 `motion_times += step_dt`），策略只能“盲目”跟随参考。
- 新增的辅助损失 `L_aux` 把策略当前的**均值动作**通过一个可微模型向前滚 `H` 步，惩罚预测状态与未来参考的偏差，再反传回 actor：

  ```
  ∂L_aux/∂θ = Σ_h (∂L_aux/∂ŝ_{t+h}) · (∂ŝ_{t+h}/∂a_t) · (∂a_t/∂θ)
  ```

- 可微模型有两种实现，由 `aux_mode` 选择：
  - `world_model`（默认）：学习式特权动力学模型 `f_θ(s, a, ref) → Δs`（MLP，末层零初始化）。用 rollout buffer 中 on-policy 的相邻转移做监督 MSE 在线训练。它**不接触接触力，也不经过物理求解器**，因此辅助梯度天然平滑，规避了接触带来的梯度爆炸问题。
  - `analytic`：无学习参数的一阶执行器代理（位置执行器映射 + 一阶滞后），用于最低成本地验证“可微闭环梯度是否有用”。
- 梯度只经**动作**回传：起点状态与参考窗口都 `detach`，世界模型参数在对 actor 反传时冻结；`aux_coef` 从 0 线性 warmup，避免早期压过 PPO。
- 关键约束：**不往 actor 增加任何真机不可观测的观测量**。aux 观测（特权状态、未来参考窗口）只写入 rollout buffer，不参与 actor/critic 的输入，因此 ONNX 导出维度、旧 checkpoint 结构都保持不变。
- 闭环语义：rollout 的起点是 on-policy 的真实状态，策略当前动作会改变未来状态、进而改变未来跟踪误差，梯度因此反映了动作的时域后果。

### 实现与文件

| 文件 | 作用 |
|------|------|
| `src/twist2_mjlab/rl/world_model.py` | `PrivilegedDynamicsModel`：一步状态差分预测，末层零初始化 |
| `src/twist2_mjlab/rl/algorithm.py` | `Twist2PPO`：保留 PPO 主体，注入 `aux_coef * L_aux`；含世界模型监督训练、warmup、`save()/load()` 扩展 |
| `src/twist2_mjlab/observations.py` | `aux_privileged_state`（97D 特权状态）与 `aux_reference_future`（`[H+1, 91]` 参考窗口） |
| `src/twist2_mjlab/config.py` | `enable_aux` 时新增两个**不参与 actor/critic** 的观测组 `aux_state` / `aux_ref_future` |
| `src/twist2_mjlab/rl_cfg.py` | `Twist2PpoCfg`（`class_name` 指向 `Twist2PPO`）与全部 `aux_*` 超参 |

默认 `aux_coef=0.0` 且 aux 观测组默认关闭，此时训练与原始 PPO **逐位一致**，也不会产生额外的每步开销与显存占用。

### 使用

```bash
# world_model 路线：学习式世界模型 + H 步闭环 rollout
TWIST2_ENABLE_AUX=1 TWIST2_MOTION_FILE=/path/to/enriched/dataset.yaml bash train_twist2.sh 0 \
  --agent.algorithm.aux-mode world_model \
  --agent.algorithm.aux-coef 0.1 \
  --agent.algorithm.aux-coef-warmup-iters 1000 \
  --agent.algorithm.aux-model-lr 3e-4

# analytic 路线：一阶解析代理，最低成本验证
TWIST2_ENABLE_AUX=1 TWIST2_MOTION_FILE=/path/to/enriched/dataset.yaml bash train_twist2.sh 0 \
  --agent.algorithm.aux-mode analytic \
  --agent.algorithm.aux-coef 0.05
```

- `TWIST2_ENABLE_AUX=1` 让环境产出 aux 观测组；`--agent.algorithm.aux-coef > 0` 才真正启用辅助损失。若只给后者而不开观测组，算法会打印一次警告并自动禁用 aux。
- 监控曲线：`Loss/aux`（辅助损失）、`Loss/world_model`（世界模型监督误差，应下降）、`Loss/aux_coef`（按 warmup 从 0 升到目标值）。
- **要判断净收益，必须跑对照**：用同一配置、同一 seed 跑一条 `TWIST2_ENABLE_AUX=0` 的基线，再和 aux 对比 `Metrics/motion/error_*`、`Train/mean_episode_length` 与 `Episode_Termination/*`。
- aux 目标与奖励权重**完全解耦**：它直接从参考运动取关节 pos/vel、root pos/rpy、key-body，不读取也不修改任何奖励项。`aux_*_weight` 只是辅助损失内部的权重。
- **奖励预设**：默认 `TWIST2_REWARD_PRESET=tuned`（本仓库当前配置）。若要和原版 `ZhaoLong0808/TWIST2_mjlab` 的奖励对齐做对照，加 `TWIST2_REWARD_PRESET=upstream`：跟踪权重回到 `2.0/0.2/1.0/…`、陡度回到 `exp(-0.15·err)`/`exp(-0.01·err)`，并且**不包含**本分支新增的稳定性奖励（`com/capture_in_support_polygon`、`ankle_hip_step`、`*_momentum_change`）。该开关在进程启动时读取，只影响奖励，不影响 aux 机制。
- 部署时不需要世界模型：`L_aux`、`aux_state`、`aux_ref_future` 都只存在于训练侧，导出的 ONNX 仍然只是 actor。

## AMP（对抗式动作先验，可选）

`TWIST2_ENABLE_AMP=1` 启用一个改良版的 AMP，作为**纯训练期**的风格先验：不改变 actor/critic 输入，判别器/专家 buffer/`amp_style` 观测都只存在于训练侧，不进 ONNX。

> **基线说明**：本仓库依赖的 mjlab pinned 版本（`60eca4af...`）**本身没有 AMP**；本仓库在 `5755bf4 "Rebuild AMP"` 之前也没有 AMP（`rl/runner.py` 只是保留 `registry_name` 的薄壳）。所以下面讲的是「当前 AMP 相对基础 mjlab PPO 跟踪器（无 AMP）」新增了什么，而不是相对某个旧 AMP 的改动。

### 原理

标准 AMP = 判别器 + 风格奖励：

1. 训练判别器 `D(s, s')`（输入是相邻两帧 transition），专家样本来自参考动作库，策略样本来自当前策略 rollout。
2. 判别器学会区分「像人（专家）」和「像策略」。
3. 策略奖励加一项 `r_style = f(D(s,s'))`，鼓励策略产生判别器认为像专家的 transition。
4. 判别器与策略交替优化。

本实现的关键选择：**只做 reward shaping，不改策略梯度结构**。判别器有独立 Adam optimizer，策略只通过新增奖励项被间接影响；不引入额外 actor loss，也不改 actor/critic 维度，因此 ONNX 导出与部署完全不变。

### 代码实现（逐个文件）

| 文件 | 性质 | 作用 |
|---|---|---|
| `src/twist2_mjlab/rl/amp.py` | 新增 296 行 | 判别器、专家 buffer、共享归一化、判别器更新、acc 门控 |
| `src/twist2_mjlab/rl/runner.py` | 修改 +11 | 构建 `AmpState`，挂到 env 与 algorithm |
| `src/twist2_mjlab/rl/algorithm.py` | 修改 +23 | 每个 PPO iteration 更新判别器；checkpoint 存取 |
| `src/twist2_mjlab/config.py` | 修改 +32 | `_AMP_ENABLED` 开关、`amp_style` 观测组、`amp_style` 奖励项 |
| `src/twist2_mjlab/observations.py` | 修改 +16 | `amp_style_state()` 风格特征 |
| `src/twist2_mjlab/rewards.py` | 修改 +33 | `amp_reward()` 风格奖励 |

**1) `config.py` — 开关 / 观测组 / 奖励项**（全部 gated，关闭时配置逐位不变）

```python
# 顶部开关（config.py:48-57）
_AMP_ENABLED = os.environ.get("TWIST2_ENABLE_AMP", "").strip().lower() in (
  "1","true","yes","on",
)
_AMP_WEIGHT = float(os.environ.get("TWIST2_AMP_WEIGHT", "0.3"))

# 奖励项（config.py:236-241）
if _AMP_ENABLED:
    rewards["amp_style"] = RewardTermCfg(
      func=twist2_rewards.amp_reward, weight=_AMP_WEIGHT,
      params={"command_name": "motion"},
    )

# 观测组（config.py:583-594）——buffer-only，不进 actor/critic
if _AMP_ENABLED:
    cfg.observations["amp_style"] = ObservationGroupCfg(
      terms={"state": ObservationTermCfg(
          func=twist2_obs.amp_style_state,
          params={"command_name": "motion"})},
      concatenate_terms=True, enable_corruption=False,
    )
```

**2) `observations.py:195-208` — 风格特征**

```python
def amp_style_state(env, command_name="motion"):
    command = get_motion_command(env, command_name)
    root_z = command.robot_body_pos_w[:, 0, 2:3] - env.scene.env_origins[:, 2:3]
    return torch.cat((command.robot_joint_pos, command.robot_joint_vel, root_z), dim=-1)
```

- `STYLE_DIM = 2*29 + 1 = 59`：`joint_pos(29) + joint_vel(29) + root_z(1)`。
- `robot_joint_pos/vel`、`robot_body_pos_w` 是**机器人本体状态**（判别器里的 policy 侧）；专家侧用 motion library 的 `joint_pos/joint_vel/body_pos_w`。

**3) `rewards.py:528-556` — 风格奖励**

```python
def amp_reward(env, command_name="motion"):
    disc = getattr(env, "amp_discriminator", None)
    normalizer = getattr(env, "amp_normalizer", None)
    if disc is None or normalizer is None:
        return torch.zeros(env.num_envs, device=env.device)
    from twist2_mjlab.rl.amp import REWARD_COEF
    s_next = amp_style_state(env, command_name)
    s_prev = getattr(env, "_prev_amp_obs", None)
    if s_prev is None:
        env._prev_amp_obs = s_next.detach()
        return torch.zeros(env.num_envs, device=env.device)
    with torch.no_grad():
        logits = disc(normalizer.normalize(s_prev),
                      normalizer.normalize(s_next)).squeeze(-1)
    env._prev_amp_obs = s_next.detach()
    return torch.clamp(1.0 - REWARD_COEF * (logits - 1.0).pow(2), min=0.0)
```

- `r = clamp(1 − 0.25·(D(s,s′) − 1)², 0)`：D 越接近 +1 奖励越高，clamp 到 0；只消费判别器、不训练（`no_grad`）。
- transition 用 `env._prev_amp_obs`（上一步）与当前 `amp_style_state` 拼成。

**4) `rl/amp.py` — 核心**

常量（`amp.py:33-50`，全部可用 `TWIST2_AMP_*` 覆盖）：
`DISC_LR=3e-5`、`DISC_HIDDEN=(512,256)`、`R1_COEF=5.0`、`DISC_LOGIT_REG=0.05`、`DISC_BATCH=2048`、`EXPERT_BATCH=2048`、`REWARD_COEF=0.25`、`EXPERT_NUM_MOTIONS=200`、`EXPERT_HORIZON_S=4.0`、`AMP_GRAD_CLIP=1.0`、`AMP_ACC_TARGET=0.85`、`DISC_LABEL_SMOOTH=0.1`。

- `RunningMeanStd`（`62-87`）：增量式 running mean/var，**专家与策略共用**，白化后判别器无法靠分布整体偏移区分两者。
- `AMPDiscriminator`（`90-105`）：输入 `2*59=118`，`Linear(118,512)→LeakyReLU(0.2)→Linear(512,256)→LeakyReLU(0.2)→Linear(256,1)`，输出标量 logits（LSGAN，不是概率）。
- `build_expert_transitions`（`108-144`）：每个动作按 `step_dt` 取最多 `horizon_s/step_dt=200` 帧，`state = joint_pos + joint_vel + root_z`，再取相邻对 `s=all[:-1]`、`s'=all[1:]`。
- `_disc_accuracy`（`147-160`）：`acc = 0.5·(P(D_e>0) + P(D_p<0))`。
- `update_discriminator`（`163-211`）：

```python
normalizer.update(expert_s); normalizer.update(policy_s)   # 共享统计
es, esn = normalize(expert); ps, psn = normalize(policy)
es.requires_grad_(True); esn.requires_grad_(True)          # R1 需要输入梯度
target_e = 1.0 - 0.1                                       # 标签平滑
lsgan = 0.5*(logits_e - 0.9)^2.mean() + 0.5*(logits_p + 1)^2.mean()
logit_reg = 0.05 * (logits_e^2.mean() + logits_p^2.mean())
grads = autograd.grad(logits_e.sum(), (es, esn), create_graph=True)
r1 = 0.5 * 5.0 * Σ mean(||∂D/∂s||²)                        # 只惩罚专家输入
loss = lsgan + logit_reg + r1
# Adam(lr=3e-5), clip_grad_norm_(disc, 1.0)
```

- `AmpState`（`214-283`）：持判别器/optimizer/共享 normalizer/专家 buffer；`update(storage)` 从 buffer 取 `amp_style[:-1]`/`[1:]`，用 `storage.dones[:-1] < 0.5` 丢掉跨 reset 的 transition，采样 2048 策略 + 2048 专家；**更新前先算 acc，`>0.85` 就跳过这一步**（仍返回 acc，等 acc 回落自动恢复）。

**5) `rl/runner.py:31-40` — 接线**

```python
if amp_enabled():
    raw_env = self.env.unwrapped
    motion_lib = get_motion_command(raw_env, "motion").motion_lib
    self.amp = AmpState(motion_lib, device, float(raw_env.step_dt))
    raw_env.amp_discriminator = self.amp.disc      # 供 rewards.amp_reward
    raw_env.amp_normalizer   = self.amp.normalizer
    raw_env._prev_amp_obs    = None
    self.alg.amp             = self.amp            # 供 Twist2PPO.update
```

`Twist2OnPolicyRunner` 由 `src/twist2_mjlab/__init__.py:14` 的 `runner_cls=` 注册进 `Twist2-Flat-Unitree-G1`。

**6) `rl/algorithm.py:485-508` — PPO 更新钩子**

```python
amp_stats = {}
if getattr(self, "amp", None) is not None:
    amp_stats = self.amp.update(self.storage)   # 所有 PPO epoch 之后，clear 之前
self.storage.clear()
...
loss_dict.update(amp_stats)  # amp_disc_loss / amp_r1 / amp_disc_acc
```

判别器**每个 PPO iteration 更新一次**（不是每个 minibatch）。checkpoint 侧（`523-552`）额外保存/恢复 `amp_disc_state_dict` 与 `amp_normalizer_state`（mean/var/count）；注意**不保存 Adam 动量**。

### 一次训练迭代的数据流

1. env 每步：`amp_style_state`（59D 机器人状态）写入 `storage.observations["amp_style"]`；同时 `amp_reward` 用 `_prev_amp_obs` 与当前状态算 `amp_style` 奖励（权重 0.3）进入 advantage。
2. PPO 正常更新（value/surrogate/entropy，外加可选 aux/world model）。
3. `Twist2PPO.update` 末尾调 `AmpState.update(storage)`：策略 transition（reset 掩码）+ 专家 buffer → 先 acc 门控，再 LSGAN+logit_reg+R1，Adam(3e-5) 走一步。
4. `storage.clear()`，进入下一轮；判别器与 normalizer 随 checkpoint 存取。

### 相对早期实现修了什么

早期集成里判别器必然坍塌（run `2026-09-10_11-41-16`：`AMP/disc_loss 9.6→0.002`、`AMP/grad_penalty→0.0001`、`Episode_Reward/amp_style≈0`，而 tracking 正常收敛）。原因与修法：

| 问题 | 修法 |
|------|------|
| 专家 30 fps vs 策略 50 Hz（`‖Δq‖` 幅值域差，判别器学个阈值就 100% 分开） | 专家过渡改为从**环境 motion library 按 `env.step_dt` 采样**，与策略同 dt、同关节顺序、同单位 |
| 风格特征太弱（只有 29D 关节角） | 升为 59D：`joint_pos(29) + joint_vel(29) + root_z(1)`，专家/策略同一 extractor |
| 无共享归一化 | 专家与策略共用 `RunningMeanStd` |
| `DISC_LR=1e-3` 过高 | 默认 `3e-5` + logit 正则 + 专家**标签平滑** |
| LSGAN 配 WGAN-GP 目标不自洽 | 改为自洽的 LSGAN + **R1** |
| reset 跨界 transition 被当假样本 | 策略过渡取自 rollout buffer，用 `dones` 掩码丢弃跨界项 |
| 判别器过强导致 style reward 恒 0 | **acc 超过 `TWIST2_AMP_ACC_TARGET` 就跳过判别器更新**（仍每轮评估，acc 回落自动恢复） |
| `amp_style` 权重 0.05 太小 | 默认 `TWIST2_AMP_WEIGHT=0.3` |

### 使用

```bash
TWIST2_ENABLE_AMP=1 TWIST2_REWARD_PRESET=upstream \
TWIST2_MOTION_FILE=/path/to/enriched/dataset.yaml bash train_twist2.sh 0 \
  --agent.max-iterations 30000 --agent.logger tensorboard
```

可调环境变量：`TWIST2_AMP_WEIGHT`（0.3）、`TWIST2_AMP_LR`（3e-5）、`TWIST2_AMP_ACC_TARGET`（0.85）、`TWIST2_AMP_R1`（5.0）、`TWIST2_AMP_LABEL_SMOOTH`（0.1）、`TWIST2_AMP_EXPERT_MOTIONS`（200）、`TWIST2_AMP_EXPERT_HORIZON_S`（4.0）。

监控：`Episode_Reward/amp_style`（应随跟踪变好而上升，健康时 >0.1）与 `Loss/amp_disc_acc`（健康区间约 0.6–0.85，不应到 1.0）。若 `amp_style` 长期贴 0，说明判别器又过强，可下调 `TWIST2_AMP_LR` 或下调 `TWIST2_AMP_ACC_TARGET`。

### 实现注意点

- `amp.py` 顶部导出的 `AMP_WEIGHT` 实际未被引用；真正生效的是 `config.py` 独立读取的 `_AMP_WEIGHT`。两处读同一环境变量，改一处即可（别只改 `amp.py`）。
- 专家 transition 在动作拼接处会跨界：`all_states[:-1]`/`[1:]` 使每个动作末帧与下一个动作首帧组成一条 transition（数据量大时影响很小，但并非严格同动作内相邻）。
- 归一化统计只用 `s` 更新，`s'` 复用同一套统计量白化（同分布，通常无碍）。
- 奖励侧 `env._prev_amp_obs` 不随 episode reset 清空，reset 后第一帧的 `(s_prev, s_next)` 会跨 episode；判别器训练侧则用 `dones` 掩码丢弃跨界项，两边处理不完全一致。
- 专家 `root_z` 是世界系绝对高度，策略 `root_z` 相对 env origin；平地（origin z=0）一致，起伏地形会对不齐。
- checkpoint 不保存判别器 Adam 动量与 `self.amp` 本身（`self.amp` 由 runner 重建，判别器权重与 normalizer 从 dict 恢复）。

## 可用策略（策略库）与如何选择

`resources/` 下内置了几个可直接部署的 ONNX。它们的 actor 观测都是 `[1, 1524]`、部署方式完全相同，**任选其一即可**：

| ONNX | 训练方法 | 奖励配置 | envs | 特点 |
|------|----------|----------|------|------|
| `pretrained.onnx` | 原版 PPO | upstream | 4096 | 官方预训练基线；anchor / body_pos 最好 |
| `pretrained_aux.onnx` | PPO + 可微 aux（world model） | tuned | 2048 | 关节 / 身体跟踪误差最低，但抖动也最大 |
| `aux_upstream_30k.onnx` | PPO + 可微 aux | upstream | 2048 | 同奖励对照里**最平滑**（动作/关节速度抖动最小） |
| `amp_upstream_30k.onnx` | PPO + 改进版 AMP | upstream | 2048 | 判别器健康（disc_acc≈0.87），跟踪/抖动与 aux 基线基本相当 |

以上结论来自同 harness 的确定性 play-eval（256 envs、无 DR）；`amp` 的训练曲线里 `amp_style` 收敛到 ~0.12、`disc_acc` 稳定在 ~0.87（不再坍塌）。

**如何选择并运行**（所有脚本都接受显式 ONNX 路径，或 `.pt` 检查点自动导出）：

```bash
# sim2sim（pkl 动作库）
TWIST2_MOTION_FILE=/path/to/enriched/motion.pkl \
  ./deploy/play_sim_twist2.sh resources/aux_upstream_30k.onnx

# 实时遥操作仿真（Redis）
./deploy/play_sim_twist2_redis.sh resources/aux_upstream_30k.onnx

# 真机
TWIST2_MOTION_FILE=/path/to/enriched/motion.pkl TWIST2_REAL_NET=eth0 \
  ./deploy/play_real_twist2.sh resources/aux_upstream_30k.onnx

# OrcaLab bridge（双臂遥操作仿真）
python bridge/bridge_twist2_to_orcalab.py --policy resources/aux_upstream_30k.onnx --fix_feet
```

选型建议：要**最平滑** → `aux_upstream_30k.onnx`；要**最低跟踪误差**（但动作更激进、抖动更大）→ `pretrained_aux.onnx`；要**贴近官方基线** → `pretrained.onnx`。

## OrcaLab Bridge（双臂遥操作仿真）

`bridge/` 是 TWIST2 → OrcaLab 的 bridge（单文件，不依赖 OrcaManipulation 仓库）：从 Redis 读 teleop 的 35D mimic，在 OrcaLab/MuJoCo 里运行我们导出的 ONNX 策略并驱动 position 执行器。

已适配本仓库策略：`HISTORY_LEN=11`、`TOTAL_OBS_SIZE=127×12=1524`，并移除了原版的 future-mimic 块（原版是 `1432 = 127×11 + 35`）。依赖、启动顺序、按键与 B1 数字孪生模式见 `bridge/BRIDGE_DEPLOY_AND_TWIN.md`。

## 动作文件格式

### 原始 PKL 输入

补全脚本至少需要每个 PKL 包含以下字段：

- `fps`
- `root_pos`
- `root_rot`，顺序为 `[x, y, z, w]`
- `dof_pos`
- `link_body_list`

### 补全后的 PKL 输出

完成补全后，PKL 还会新增：

- `body_pos_w`
- `body_quat_w`

当任务采样动作帧、计算跟踪观测以及构建特权 critic 特征时，本地 motion library 需要这些字段。

### Dataset YAML 示例

```yaml
root_path: /path/to/enriched/pkls
motions:
  - file: walk_forward.pkl
    weight: 1.0
  - file: wave_hands.pkl
    weight: 0.5
```

传给 `TWIST2_MOTION_FILE` 的路径既可以指向这个 YAML，也可以直接指向单个补全后的 PKL。

## 大规模动作数据集：CPU 卸载与 GPU 缓存

默认情况下，`PklMotionLib` 会把所有动作帧拼接成六个张量直接放到 GPU 上。追踪 14 个 body 时每一帧约 960 B，所以 GPU 显存占用大致等于 `total_frames × 960 B`。完整的 SEED 数据集（约 14.2 万条动作，120 fps）已经超出单张 H100 的显存容量。为此提供三种可选模式。

### 模式 1 — 下采样（最简单，最快）

在加载时通过筛选轨迹或对帧进行下采样来加载数据集的子集。

```bash
# 例：加载每 N 个动作或在加载 PKL 时对帧下采样
# 这在数据集/PKL 准备阶段配置，不在运行时配置。
```

- **优点：** 所有数据都装入 GPU 显存；没有流水线停顿；完整训练速度。
- **缺点：** 相比完整数据集损失动作多样性。
- **适用场景：** 可以接受减少一些轨迹或帧分辨率，且对速度有要求。
- **实践建议（SEED）：** 默认情况下，SEED 补全会从 120 fps 下采样到 30 fps，帧数减少约 4 倍，使得完整数据集可以装入单张 H100 而无需任何卸载或缓存开销。

### 模式 2 — 纯 CPU 卸载（简单，无缓存)

把六个动作张量放在 pinned CPU 内存里，每一步将混合后的帧回传到 GPU。

```bash
--env.commands.motion.offload-to-cpu True
```

- **代价：** 每个 env step 都要对帧索引调用 `.cpu()`，触发 CUDA 同步并清空 GPU 流水线。训练循环大约会变慢 **2 倍**，但可以容纳任意大小的数据集（只要 host RAM 够）。
- **适用场景：** 数据集装不下显存，同时不想引入缓存带来的额外复杂度；也适合作为与其他模式对比的基线。

### 模式 3 — GPU 缓存（大规模数据集推荐)

所有数据仍然放在 pinned CPU 内存，但额外维护一块 GPU 常驻缓存，用来存放“活跃工作集”。4096 个 env 各自只播放一条动作，工作集只有约 4 GB（远小于默认 8 GB 缓存）。缓存命中时走的是和纯 GPU 模式相同的 gather 路径；未命中时会按连续帧块从 CPU 加载该动作到缓存。

```bash
--env.commands.motion.gpu-cache True \
--env.commands.motion.cache-capacity-gb 70.0   # 可选，默认 70 GB
```

- **性能：** 在 14.2 万条动作的数据集上，只要缓存尺寸合理，约 98% 的 `get_frame` 调用都是纯 GPU gather，吞吐非常接近“全部放 GPU”时的基线。
- **缓存容量经验公式：** `num_envs × avg_frames_per_motion × 960 B × 2`。在 96 GB H100 上跑完整 SEED 数据集时我们使用 80 GB 缓存。
- **替换策略：** 追加写入，空间不足时整块重置（不是按动作的 LRU 淘汰）。
- **缓存填满时：** 如果下一个动作会超过剩余容量，缓存映射会被 O(1) 清空，活跃动作会在后续访问时重新加载。
- **实践建议（SEED 数据集）：** 缓存越大通常越好；缓存太小反而可能拖慢速度，因为整块缓存会更频繁重置并反复从 CPU 回填。

### 如何选择

| 场景 | 推荐设置 |
|------|----------|
| 数据集可以轻松装入显存 | 两个开关都保持默认关闭 |
| 数据集非常大，可以下采样动作/帧 | 在数据集准备阶段下采样（模式 1） |
| 数据集太大，偏好简单方案，接受变慢 | `--env.commands.motion.offload-to-cpu True`（模式 2） |
| 数据集太大，希望接近基线速度 | `--env.commands.motion.gpu-cache True`（模式 3） |

## 在哪里调整行为

如果你想修改任务，重点查看这些文件：

- `src/twist2_mjlab/config.py` — 观测组、奖励、终止条件、默认值
- `src/twist2_mjlab/observations.py` — 观测构建模块
- `src/twist2_mjlab/rewards.py` — 跟踪与正则化项
- `src/twist2_mjlab/terminations.py` — episode 失败条件
- `src/twist2_mjlab/commands.py` — 动作加载与重采样
- `src/twist2_mjlab/pkl_motion_lib.py` — 动作加载、插值与采样
- `src/twist2_mjlab/rl_cfg.py` — 运行器与模型配置
- `src/twist2_mjlab/rl/algorithm.py` — `Twist2PPO`：PPO + 可微闭环辅助目标（`L_aux`）
- `src/twist2_mjlab/rl/world_model.py` — 可微特权动力学模型 `PrivilegedDynamicsModel`
- `src/twist2_mjlab/rl/amp.py` — AMP 判别器、专家 buffer（按 control dt 采样）、共享归一化与更新逻辑

## 排查问题

- **`TWIST2_MOTION_FILE is required`**：在非交互式 shell 中运行脚本前，请先设置该环境变量。
- **play 时提示 `No runs found`**：至少训练一次，或者显式传入 checkpoint 路径。
- **缺少 `body_pos_w` / `body_quat_w`**：请对原始 PKL 重新运行 `enrich_pkl.py`。
- **日志路径不符合预期**：请在 `twist2_mjlab/` 下运行脚本，这样相对路径 `logs/` 才会和项目结构一致。
- **关于显示、渲染或视频的报错**：训练脚本里设置 `--video False`。

## 一句话总结

`twist2_mjlab` 是 TWIST2 在 Unitree G1 上进行动作跟踪的自包含 MJLab 包：先补全 PKL，再用 `Twist2-Flat-Unitree-G1` 训练，最后播放最新 checkpoint——毕竟机器人最擅长的事情之一，就是认真拒绝无聊。
