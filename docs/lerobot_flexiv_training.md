# 使用 LeRobot Flexiv 力数据训练 FACTR

本文档记录如何用自己的 LeRobot 数据训练 FACTR，避免后续忘记路径、命令和关键配置。

## 数据说明

原始 LeRobot 数据路径：

```bash
/root/autodl-fs/force_vla_data/data_lerobot/flexiv_pump_1bottle_inputForce
```

数据格式：

- episode 文件：`data/chunk-000/episode_*.parquet`
- 元信息：`meta/info.json`
- 总 episode 数：50
- 总 frame 数：17563
- `action`：7 维
- `observation.state`：13 维
- `observation.image`：主相机图像
- `observation.wrist_image`：腕部相机图像

`observation.state` 维度含义：

```text
state[0:7]   本体状态 proprio
state[7:13]  六维力 force
```

重要：正式 FACTR 训练默认只把 `state[7:13]` 六维力送入模型的低维 observation token。`state[0:7]` 本体状态只保留为可选对照实验，不作为默认 FACTR 输入。

## 已新增/修改的代码

### 1. LeRobot 转 FACTR 数据脚本

路径：

```bash
/root/autodl-tmp/FACTR/scripts/convert_lerobot_to_factr.py
```

功能：

- 读取 LeRobot parquet 数据
- 解码 `observation.image` 和 `observation.wrist_image`
- 将图像 resize 到 `256x256`
- 写成 FACTR/robobuf 可读的 `enc_cam_0`、`enc_cam_1`
- 默认将 `observation.state[7:13]` 六维力写入 robobuf state
- 将 7 维 `action` 写入 robobuf action
- 对 state/action 做高斯归一化
- 生成 `buf.pkl` 和 `rollout_config.yaml`

### 2. 数据准备脚本

路径：

```bash
/root/autodl-tmp/FACTR/scripts/prepare_lerobot_flexiv.sh
```

运行：

```bash
cd /root/autodl-tmp/FACTR
conda activate factr
bash scripts/prepare_lerobot_flexiv.sh
```

输出路径：

```bash
/root/autodl-fs/force_vla_data/processed_factr/flexiv_pump_1bottle_inputForce_force
```

输出文件：

```bash
buf.pkl
rollout_config.yaml
```

当前已经生成好的文件：

```bash
/root/autodl-fs/force_vla_data/processed_factr/flexiv_pump_1bottle_inputForce_force/buf.pkl
/root/autodl-fs/force_vla_data/processed_factr/flexiv_pump_1bottle_inputForce_force/rollout_config.yaml
```

### 3. FACTR 任务配置

路径：

```bash
/root/autodl-tmp/FACTR/factr/cfg/task/flexiv_pump_force.yaml
```

关键配置：

```yaml
obs_dim: 6
ac_dim: 7
cam_indexes: [0, 1]
```

含义：

- `obs_dim: 6` 对应 `observation.state[7:13]` 六维力
- `ac_dim: 7` 对应 LeRobot `action`
- `cam_indexes: [0, 1]` 使用两路相机：
  - `cam0` 对应 `observation.image`
  - `cam1` 对应 `observation.wrist_image`

### 4. 训练脚本

路径：

```bash
/root/autodl-tmp/FACTR/scripts/train_flexiv_lerobot.sh
```

正式训练：

```bash
cd /root/autodl-tmp/FACTR
conda activate factr
bash scripts/train_flexiv_lerobot.sh
```

默认配置：

```bash
CUDA_DEVICE_ID=0
TASK_CONFIG=flexiv_pump_force
BUFFER_PATH=/root/autodl-fs/force_vla_data/processed_factr/flexiv_pump_1bottle_inputForce_force/buf.pkl
FEATURE_PATH=/root/autodl-tmp/FACTR/visual_features/vit_base/SOUP_1M_DH.pth
EXP_NAME=flexiv_pump_force
AC_CHUNK=100
IMG_CHUNK=1
BATCH_SIZE=128
NUM_WORKERS=10
MAX_ITERATIONS=20000
```

可以用环境变量覆盖，例如：

```bash
cd /root/autodl-tmp/FACTR
conda activate factr

CUDA_DEVICE_ID=0 \
BATCH_SIZE=64 \
MAX_ITERATIONS=50000 \
EXP_NAME=flexiv_pump_force_v1 \
bash scripts/train_flexiv_lerobot.sh
```

训练 checkpoint 默认保存到：

```bash
/root/autodl-tmp/FACTR/checkpoints/<EXP_NAME>
```

例如默认实验：

```bash
/root/autodl-tmp/FACTR/checkpoints/flexiv_pump_force
```

## 预训练视觉特征

训练脚本默认使用：

```bash
/root/autodl-tmp/FACTR/visual_features/vit_base/SOUP_1M_DH.pth
```

如果缺失，先运行：

```bash
cd /root/autodl-tmp/FACTR
bash scripts/download_features.sh
```

当前已确认该权重文件存在。

## 环境依赖

使用 FACTR 环境：

```bash
conda activate factr
```

为了读取 LeRobot parquet 数据，已在 factr 环境中安装：

```bash
python -m pip install pyarrow
```

如果以后换环境后转换脚本报 `pyarrow` 缺失，重新安装即可。

## 已验证结果

已完成全量数据转换：

```text
episodes: 50
frames: 17563
force-only buf.pkl: 约 445M
```

已验证 FACTR loader 可正常读取：

```text
dataset len: 16685
obs shape: (6,)
actions shape: (10, 7)
mask shape: (10, 7)
cam0 shape: (1, 3, 256, 256)
cam1 shape: (1, 3, 256, 256)
```

已跑通 1-step 训练烟测：

- ViT 权重加载正常
- Hydra 配置解析正常
- train/test buffer 构建正常
- forward/backward 正常
- loss 可正常输出

## 注意事项

1. 正式 `flexiv_pump_force` 配置只把 `observation.state[7:13]` 六维力作为低维输入传给模型。
2. `rollout_config.yaml` 里记录了 state 布局：

```yaml
state_layout:
  force: [0, 6]
  source_in_observation.state: [7, 13]
```

3. 转换脚本默认会归一化 state 和 action。训练和 rollout 时需要使用同一个 `rollout_config.yaml` 中的统计量。
4. 如果只想转换部分 episode 做调试，可以运行：

```bash
cd /root/autodl-tmp/FACTR
conda activate factr

python scripts/convert_lerobot_to_factr.py \
  --output-dir /root/autodl-fs/force_vla_data/processed_factr/debug_flexiv \
  --episode-indices 0 1
```

5. 如果只想跑一个很小的训练烟测，可以运行：

```bash
cd /root/autodl-tmp/FACTR
conda activate factr

MAX_ITERATIONS=1 \
BATCH_SIZE=2 \
NUM_WORKERS=0 \
AC_CHUNK=10 \
EXP_NAME=_smoke_flexiv_factr \
bash scripts/train_flexiv_lerobot.sh
```

烟测会生成 checkpoint，占用较多空间。验证完可以删除：

```bash
rm -rf /root/autodl-tmp/FACTR/checkpoints/_smoke_flexiv_factr
```

## 关于是否用到了力

当前实现用到了力。

论文中写的是 RGB 图像和外部力/力矩输入，力经过 MLP encoder 形成一个 force token。发布代码里没有单独叫 `force` 的字段，而是把低维力数据放进 robobuf 的 `obs.state`，再用 `nn.Linear(obs_dim, token_dim)` 编成一个 observation token。

官方数据处理配置也体现了这一点：

```yaml
obs_topics:
- /franka/right/obs_franka_torque
```

也就是说，代码里的 force 输入路径是：

```text
robobuf step.obs.state
  -> replay_buffer 返回 obs
  -> agent 中 Linear(obs_dim, token_dim)
  -> 作为 obs/force token 拼到视觉 tokens 后面
  -> Transformer policy
```

对于当前 Flexiv LeRobot 数据，正式转换时：

```text
obs.state = observation.state[7:13]
```

因此模型确实使用的是六维力，而不是只使用本体状态。

对应配置：

```yaml
task: flexiv_pump_force
obs_dim: 6
```

对应代码位置：

```text
factr/replay_buffer.py:
  o_t = step.obs.state

factr/agent.py:
  nn.Linear(odim, token_dim)
  tokens = torch.cat((tokens, obs_token), 1)
```
