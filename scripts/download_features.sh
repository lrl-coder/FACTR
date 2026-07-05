#!/usr/bin/env bash
set -euo pipefail

FEATURES_URL="https://www.cs.cmu.edu/~data4robotics/release/features.zip"
FEATURES_ZIP="features.zip"

# make sure the folder doesn't already exist
if [ -d "visual_features" ]; then
    echo "Data already downloaded!"
    exit 0
fi

if command -v aria2c >/dev/null 2>&1; then
    aria2c \
        --continue=true \
        --max-connection-per-server=16 \
        --split=16 \
        --min-split-size=1M \
        --file-allocation=none \
        --out="$FEATURES_ZIP" \
        "$FEATURES_URL"
else
    wget --continue --output-document "$FEATURES_ZIP" "$FEATURES_URL"
fi

unzip -t "$FEATURES_ZIP"
unzip "$FEATURES_ZIP"
rm "$FEATURES_ZIP"
