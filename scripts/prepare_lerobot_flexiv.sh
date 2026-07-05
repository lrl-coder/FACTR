#!/bin/bash
set -euo pipefail

DATASET_DIR=${DATASET_DIR:-/root/autodl-fs/force_vla_data/data_lerobot/flexiv_pump_1bottle_inputForce}
OUTPUT_DIR=${OUTPUT_DIR:-/root/autodl-fs/force_vla_data/processed_factr/flexiv_pump_1bottle_inputForce_force}
OBS_MODE=${OBS_MODE:-force}

python scripts/convert_lerobot_to_factr.py \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --image-keys observation.image observation.wrist_image \
  --obs-mode "$OBS_MODE"
