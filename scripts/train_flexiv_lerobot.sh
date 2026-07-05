#!/bin/bash
set -euo pipefail

CUDA_DEVICE_ID=${CUDA_DEVICE_ID:-0}
TASK_CONFIG=${TASK_CONFIG:-flexiv_pump_force}
BUFFER_PATH=${BUFFER_PATH:-/root/autodl-fs/force_vla_data/processed_factr/flexiv_pump_1bottle_inputForce_force/buf.pkl}
FEATURE_PATH=${FEATURE_PATH:-$(pwd)/visual_features/vit_base/SOUP_1M_DH.pth}
EXP_NAME=${EXP_NAME:-flexiv_pump_force}
WANDB_DEBUG=${WANDB_DEBUG:-False}
WANDB_ENTITY=${WANDB_ENTITY:-null}
WANDB_PROJECT=${WANDB_PROJECT:-factr}
WANDB_GROUP=${WANDB_GROUP:-bc}

AC_CHUNK=${AC_CHUNK:-100}
IMG_CHUNK=${IMG_CHUNK:-1}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-10}
MAX_ITERATIONS=${MAX_ITERATIONS:-20000}

SPACE_CONFIG=${SPACE_CONFIG:-pixel}
SCHEDULER_CONFIG=${SCHEDULER_CONFIG:-linear}
OPERATOR_CONFIG=${OPERATOR_CONFIG:-blur}
START_SCALE=${START_SCALE:-5}
STOP_SCALE=${STOP_SCALE:-0}

if [ ! -f "$BUFFER_PATH" ]; then
  echo "Buffer not found: $BUFFER_PATH"
  echo "Run: bash scripts/prepare_lerobot_flexiv.sh"
  exit 1
fi

if [ ! -f "$FEATURE_PATH" ]; then
  echo "Feature checkpoint not found: $FEATURE_PATH"
  echo "Run: bash scripts/download_features.sh"
  exit 1
fi

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_ID python factr/train_bc_policy.py \
  exp_name="$EXP_NAME" \
  agent.features.restore_path="$FEATURE_PATH" \
  buffer_path="$BUFFER_PATH" \
  task="$TASK_CONFIG" \
  ac_chunk="$AC_CHUNK" \
  img_chunk="$IMG_CHUNK" \
  batch_size="$BATCH_SIZE" \
  num_workers="$NUM_WORKERS" \
  max_iterations="$MAX_ITERATIONS" \
  curriculum.space="$SPACE_CONFIG" \
  curriculum.operator="$OPERATOR_CONFIG" \
  curriculum.scheduler="$SCHEDULER_CONFIG" \
  curriculum.start_scale="$START_SCALE" \
  curriculum.stop_scale="$STOP_SCALE" \
  wandb.debug="$WANDB_DEBUG" \
  wandb.entity="$WANDB_ENTITY" \
  wandb.project="$WANDB_PROJECT" \
  wandb.group="$WANDB_GROUP" \
  wandb.name="$EXP_NAME"
