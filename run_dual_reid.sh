#!/bin/bash
# Run script for KABS Enhance Dual Re-ID System

# Script description
echo "==============================================="
echo "KABS Enhance - Dual Re-ID Keyframe Extraction"
echo "==============================================="
echo ""

# Configuration
VIDEO_PATH="${1:-/workspace/data/input_video.mp4}"
OUTPUT_DIR="${2:-/workspace/output}"
PERSON_REID_MODEL="${3:-/workspace/models/osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth}"
VEHICLE_REID_MODEL="${4:-/workspace/models/osnet_veri.pth}"

echo "Configuration:"
echo "  Video: $VIDEO_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Person Re-ID Model: $PERSON_REID_MODEL"
echo "  Vehicle Re-ID Model: $VEHICLE_REID_MODEL"
echo ""

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "✓ Running inside Docker container"

    # Run the dual Re-ID script
    python /workspace/yolo_osnet_4_dual_reid.py \
        --video "$VIDEO_PATH" \
        --output "$OUTPUT_DIR" \
        --person-reid-model "$PERSON_REID_MODEL" \
        --vehicle-reid-model "$VEHICLE_REID_MODEL"
else
    echo "⚠️  Not running inside Docker. Starting Docker container..."

    # Run Docker container
    docker run --gpus all -it --rm \
        -v $(pwd):/workspace \
        -v $(pwd)/models:/workspace/models \
        -v $(pwd)/data:/workspace/data \
        -v $(pwd)/output:/workspace/output \
        --shm-size 8g \
        kabs_enhance:latest \
        bash /workspace/run_dual_reid.sh "$VIDEO_PATH" "$OUTPUT_DIR" "$PERSON_REID_MODEL" "$VEHICLE_REID_MODEL"
fi
