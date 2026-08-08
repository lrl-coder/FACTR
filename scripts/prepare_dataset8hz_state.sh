#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

DATASET_DIR=${DATASET_DIR:-/root/autodl-tmp/dataset-8Hz}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_DIR/processed_data/dataset_8hz_state}

python scripts/convert_lerobot21_state_to_factr.py \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --obs-keys observation.state \
  --image-keys observation.images.third_view observation.images.wrist
