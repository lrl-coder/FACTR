#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

DATASET_ROOT=${DATASET_ROOT:-/root/autodl-tmp/dataset-8Hz}
CUDA_DEVICE_ID=${CUDA_DEVICE_ID:-0}
AC_CHUNK=${AC_CHUNK:-16}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-10}
MAX_ITERATIONS=${MAX_ITERATIONS:-20000}
EVAL_FREQ=${EVAL_FREQ:-0}
WANDB_DEBUG=${WANDB_DEBUG:-False}
WANDB_ENTITY=${WANDB_ENTITY:-1559589961-northwestern-university}
WANDB_PROJECT=${WANDB_PROJECT:-factr}
WANDB_GROUP=${WANDB_GROUP:-bc}

TASK_NAMES=(flip_box insert_plug press_button wipe_board)

for task_name in "${TASK_NAMES[@]}"; do
  task_dataset="$DATASET_ROOT/$task_name"
  task_output="$PROJECT_DIR/processed_data/${task_name}_force"
  task_exp="${task_name}_force"

  if [ ! -d "$task_dataset" ]; then
    echo "Dataset not found: $task_dataset"
    exit 1
  fi

  echo "========== Starting: $task_name =========="
  TASK_NAME="$task_name" \
  DATASET_DIR="$task_dataset" \
  OUTPUT_DIR="$task_output" \
  EXP_NAME="$task_exp" \
  CUDA_DEVICE_ID="$CUDA_DEVICE_ID" \
  AC_CHUNK="$AC_CHUNK" \
  BATCH_SIZE="$BATCH_SIZE" \
  NUM_WORKERS="$NUM_WORKERS" \
  MAX_ITERATIONS="$MAX_ITERATIONS" \
  EVAL_FREQ="$EVAL_FREQ" \
  WANDB_DEBUG="$WANDB_DEBUG" \
  WANDB_ENTITY="$WANDB_ENTITY" \
  WANDB_PROJECT="$WANDB_PROJECT" \
  WANDB_GROUP="$WANDB_GROUP" \
  bash scripts/run_single_task_dataset8hz_force.sh
  echo "========== Finished: $task_name =========="
done

echo "All four FACTR tasks completed."
