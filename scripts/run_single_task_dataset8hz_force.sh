#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

TASK_NAME=${TASK_NAME:-insert_plug}
DATASET_DIR=${DATASET_DIR:-/root/autodl-tmp/dataset-8Hz/$TASK_NAME}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/processed_data/${TASK_NAME}_force}
EXP_NAME=${EXP_NAME:-${TASK_NAME}_force}

echo "[1/2] Prepare dataset: $TASK_NAME"
DATASET_DIR="$DATASET_DIR" OUTPUT_DIR="$OUTPUT_DIR" bash scripts/prepare_dataset8hz_force.sh

echo "[2/2] Train FACTR: $EXP_NAME"
BUFFER_PATH="$OUTPUT_DIR/buf.pkl" EXP_NAME="$EXP_NAME" bash scripts/train_dataset8hz_force.sh
