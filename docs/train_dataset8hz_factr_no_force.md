# 使用 dataset-8Hz 训练 FACTR（全量训练，不做 eval）

一步训练四个任务（依次执行）：

```bash
for TASK_NAME in flip_box insert_plug press_button wipe_board; do
    echo "========== Training: $TASK_NAME =========="

    TASK_NAME="$TASK_NAME" \
    AC_CHUNK=16 \
    BATCH_SIZE=32 \
    MAX_ITERATIONS=20000 \
    bash scripts/run_single_task_dataset8hz.sh

    echo "========== Finished: $TASK_NAME =========="
done
```

本文档用于 `/root/autodl-tmp/dataset-8Hz` 的 LeRobot 2.1 数据训练 FACTR。低维输入只用 `observation.state[8]`，不使用 `observation.wrench_compensated[6]`。wandb 默认开启，已登录后会直接记录训练日志。

## 数据映射

```text
cam0   <- observation.images.third_view
cam1   <- observation.images.wrist
obs    <- observation.state[8]        # 7关节角 + 夹爪
action <- action[8]                   # 7关节角 + 夹爪命令
ignore <- observation.wrench_compensated[6]
```

FACTR 配置：

```text
task: dataset8hz_state
obs_dim: 8
ac_dim: 8
cam_indexes: [0, 1]
```

训练时 `n_test_ratio=0.0`，`EVAL_FREQ=0`，所以不会切验证集，也不会在 wandb 里出现 `eval/*`。

## 单任务一条龙

以 `insert_plug` 为例：

```bash
cd /root/autodl-tmp/FACTR
conda activate factr

TASK_NAME=insert_plug \
BATCH_SIZE=32 \
MAX_ITERATIONS=20000 \
bash scripts/run_single_task_dataset8hz.sh
```

默认输出：

```text
/root/autodl-tmp/FACTR/processed_data/insert_plug_state/buf.pkl
/root/autodl-tmp/FACTR/checkpoints/insert_plug_state
```

`rollout/` 目录里的文件是 checkpoint 和配置，不是 eval 结果。

## 手动拆开跑

只做数据转换：

```bash
cd /root/autodl-tmp/FACTR
conda activate factr

DATASET_DIR=/root/autodl-tmp/dataset-8Hz/insert_plug \
OUTPUT_DIR=/root/autodl-tmp/FACTR/processed_data/insert_plug_state \
bash scripts/prepare_dataset8hz_state.sh
```

只做训练：

```bash
cd /root/autodl-tmp/FACTR
conda activate factr

BUFFER_PATH=/root/autodl-tmp/FACTR/processed_data/insert_plug_state/buf.pkl \
EXP_NAME=insert_plug_state \
bash scripts/train_dataset8hz_state.sh
```

## 常用参数

```bash
CUDA_DEVICE_ID=0   # GPU 编号
EVAL_FREQ=0        # 0 表示关闭 eval
BATCH_SIZE=64
AC_CHUNK=100       # 一次预测未来 100 步动作
IMG_CHUNK=1
MAX_ITERATIONS=20000
WANDB_DEBUG=False  # 默认写 wandb
```

覆盖示例：

```bash
cd /root/autodl-tmp/FACTR
conda activate factr

CUDA_DEVICE_ID=0 \
BATCH_SIZE=32 \
MAX_ITERATIONS=50000 \
EXP_NAME=insert_plug_state_v1 \
bash scripts/train_dataset8hz_state.sh
```

## 烟测

```bash
cd /root/autodl-tmp/FACTR
conda activate factr

python scripts/convert_lerobot21_state_to_factr.py \
  --dataset-dir /root/autodl-tmp/dataset-8Hz \
  --output-dir /root/autodl-tmp/FACTR/processed_data/debug_dataset8hz_state \
  --max-episodes-per-dataset 1

BUFFER_PATH=/root/autodl-tmp/FACTR/processed_data/debug_dataset8hz_state/buf.pkl \
EXP_NAME=_smoke_dataset8hz_state \
MAX_ITERATIONS=1 \
BATCH_SIZE=2 \
NUM_WORKERS=0 \
AC_CHUNK=10 \
bash scripts/train_dataset8hz_state.sh
```
