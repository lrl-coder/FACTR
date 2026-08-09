# 使用 dataset-8Hz 六维力训练 FACTR

本文档按 FACTR 原仓库的数据接口处理 `/root/autodl-tmp/dataset-8Hz`。模型输入为两路 RGB 图像和六维力/力矩，不使用 `observation.state[8]` 本体状态。

## 数据映射

```text
robobuf obs.enc_cam_0 <- observation.images.third_view
robobuf obs.enc_cam_1 <- observation.images.wrist
robobuf obs.state     <- observation.wrench_compensated[6]
robobuf action        <- action[8]
ignore                <- observation.state[8]
```

转换脚本会将图像缩放到 `256x256` 并编码为 JPEG，对六维力和动作分别做高斯归一化，然后输出：

```text
buf.pkl
rollout_config.yaml
```

`rollout_config.yaml` 保存力和动作的均值、标准差，部署时必须使用相同统计量反归一化动作。

FACTR 任务配置：

```text
task: dataset8hz_force
obs_dim: 6
ac_dim: 8
cam_indexes: [0, 1]
```

训练默认使用全部数据：`n_test_ratio=0.0`、`EVAL_FREQ=0`，不切验证集，也不记录 `eval/*`。wandb 默认开启，`AC_CHUNK=16`，对应 8Hz 数据未来约 2 秒动作。

## 单任务训练

以 `insert_plug` 为例，一条命令完成转换和训练：

```bash
cd /root/autodl-tmp/FACTR
conda activate factr

TASK_NAME=insert_plug \
AC_CHUNK=16 \
BATCH_SIZE=32 \
MAX_ITERATIONS=20000 \
bash scripts/run_single_task_dataset8hz_force.sh
```

输出：

```text
/root/autodl-tmp/FACTR/processed_data/insert_plug_force/buf.pkl
/root/autodl-tmp/FACTR/processed_data/insert_plug_force/rollout_config.yaml
/root/autodl-tmp/FACTR/checkpoints/insert_plug_force
```

不要复用旧的 `processed_data/*_state/buf.pkl`，其中低维 observation 不是六维力。

## 四个任务依次训练

单张 RTX 4090 建议串行运行：

```bash
cd /root/autodl-tmp/FACTR
conda activate factr

for TASK_NAME in flip_box insert_plug press_button wipe_board; do
    TASK_NAME="$TASK_NAME" \
    AC_CHUNK=16 \
    BATCH_SIZE=32 \
    MAX_ITERATIONS=20000 \
    bash scripts/run_single_task_dataset8hz_force.sh
done
```

## 手动拆开运行

只转换单个任务：

```bash
cd /root/autodl-tmp/FACTR
conda activate factr

DATASET_DIR=/root/autodl-tmp/dataset-8Hz/insert_plug \
OUTPUT_DIR=/root/autodl-tmp/FACTR/processed_data/insert_plug_force \
bash scripts/prepare_dataset8hz_force.sh
```

只训练已转换数据：

```bash
cd /root/autodl-tmp/FACTR
conda activate factr

BUFFER_PATH=/root/autodl-tmp/FACTR/processed_data/insert_plug_force/buf.pkl \
EXP_NAME=insert_plug_force \
AC_CHUNK=16 \
BATCH_SIZE=32 \
MAX_ITERATIONS=20000 \
bash scripts/train_dataset8hz_force.sh
```

## 常用参数

```text
CUDA_DEVICE_ID=0    使用哪张 GPU
AC_CHUNK=16         预测未来 16 步动作，8Hz 下约 2 秒
IMG_CHUNK=1         每路相机使用当前 1 帧
BATCH_SIZE=32       RTX 4090 推荐起点
NUM_WORKERS=10      DataLoader worker 数
MAX_ITERATIONS=20000
EVAL_FREQ=0         关闭 eval，全部数据用于训练
WANDB_DEBUG=False   开启 wandb
WANDB_PROJECT=factr
WANDB_GROUP=bc
```

checkpoint 默认每 2000 step 保存一次，只保留最近两个编号权重，并持续更新：

```text
/root/autodl-tmp/FACTR/checkpoints/<EXP_NAME>/rollout/latest_ckpt.ckpt
```

## 烟测

只转换一个 episode：

```bash
cd /root/autodl-tmp/FACTR
conda activate factr

python scripts/convert_lerobot21_force_to_factr.py \
  --dataset-dir /root/autodl-tmp/dataset-8Hz/insert_plug \
  --output-dir /root/autodl-tmp/FACTR/processed_data/debug_insert_plug_force \
  --max-episodes-per-dataset 1
```

跑 1 step：

```bash
BUFFER_PATH=/root/autodl-tmp/FACTR/processed_data/debug_insert_plug_force/buf.pkl \
EXP_NAME=_smoke_insert_plug_force \
AC_CHUNK=16 \
BATCH_SIZE=2 \
NUM_WORKERS=0 \
MAX_ITERATIONS=1 \
WANDB_DEBUG=True \
bash scripts/train_dataset8hz_force.sh
```
