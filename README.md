# KABS Enhance - Dual Re-ID System

Enhanced keyframe extraction algorithm with separate Re-ID models for person and non-person objects.

## Features

- **Dual Re-ID System**: Separate SOTA models for person and vehicle/object re-identification
- **Person Re-ID**: OSNet trained on Market-1501 dataset
- **Vehicle Re-ID**: OSNet trained on VeRi-776 dataset
- **Same Pipeline**: Maintains the 5-stage pipeline from KABS_paper

## Docker Setup

### Build Docker Image

```bash
cd /data1/vailab02_dir/crime_3d/KABS_enhance

# Using docker-compose (recommended)
docker-compose build

# Or using docker directly
docker build -t kabs_enhance:latest .
```

### Run Container

```bash
# Using docker-compose
docker-compose up -d
docker-compose exec kabs_enhance bash

# Or using docker directly
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/output:/workspace/output \
  --shm-size 8g \
  kabs_enhance:latest bash
```

## 🔥 NEW: Multiple Re-ID Models Support

KABS_enhance now supports **multiple SOTA Re-ID models**! You can easily switch between different models to compare performance.

### Available Person Re-ID Models
- `osnet_market1501` - OSNet Market-1501 fine-tuned ⭐ **BEST for person**
- `osnet_x1_0` - OSNet x1.0 pretrained (balanced)
- `osnet_x0_75` - OSNet x0.75 (faster)
- `osnet_x0_5` - OSNet x0.5 (fastest)
- `osnet_ibn_x1_0` - OSNet-IBN (with Instance-Batch Norm)
- `resnet50_fc512` - ResNet50 baseline
- `mlfn` - Multi-Level Factorisation Net

### Available Vehicle Re-ID Models
- `osnet_veri776` - OSNet VeRi-776 fine-tuned ⭐ **BEST for vehicle** (if available)
- `osnet_x1_0` - OSNet x1.0 pretrained
- `resnet50_fc512` - ResNet50 baseline

### How to Select Models

Edit the config in `yolo_osnet_4_dual_reid.py`:

```python
config = {
    # ...
    "person_reid_model_key": "osnet_market1501",  # Change this!
    "vehicle_reid_model_key": "osnet_x1_0",       # Change this!
}
```

See [REID_MODELS_GUIDE.md](REID_MODELS_GUIDE.md) for detailed model comparison and selection guide.

## Usage

### Quick Start - Single Run

```bash
# Method 1: Direct execution
python yolo_osnet_4_dual_reid.py

# Method 2: With Docker
docker-compose up -d
docker-compose exec kabs_enhance bash
python yolo_osnet_4_dual_reid.py
```

### Compare Multiple Re-ID Models

```bash
# Compare all person Re-ID models
python compare_reid_models.py \
  --video test_video.mp4 \
  --output comparison_results \
  --test-person

# Compare all vehicle Re-ID models
python compare_reid_models.py \
  --video test_video.mp4 \
  --output comparison_results \
  --test-vehicle

# Compare ALL models (person + vehicle)
python compare_reid_models.py \
  --video test_video.mp4 \
  --output comparison_results \
  --test-all
```

Results will be saved to `comparison_results/comparison_results.json`

### List Available Models

```python
from reid_model_registry import list_available_models

# List person Re-ID models
list_available_models("person")

# List vehicle Re-ID models
list_available_models("vehicle")
```

### Configuration

Edit `yolo_osnet_4_dual_reid.py`:

```python
config = {
    "video_path": "test_video.mp4",
    "output_folder": "results",
    "use_dual_reid": True,

    # 🔥 Select Re-ID models from registry
    "person_reid_model_key": "osnet_market1501",  # BEST for person
    "vehicle_reid_model_key": "osnet_x1_0",       # General model

    # Or use custom path (overrides registry)
    "person_reid_model_path": None,  # None = use registry
    "vehicle_reid_model_path": None,

    # ... other settings
}
```

### Example: Test Different Models

```python
# Test 1: Best accuracy
config = {
    "person_reid_model_key": "osnet_market1501",
    "vehicle_reid_model_key": "osnet_veri776",
}

# Test 2: Best speed
config = {
    "person_reid_model_key": "osnet_x0_5",  # Fastest
    "vehicle_reid_model_key": "osnet_x1_0",
}

# Test 3: Balanced
config = {
    "person_reid_model_key": "osnet_x1_0",
    "vehicle_reid_model_key": "osnet_x1_0",
}
```

## Architecture

The system uses a 5-stage pipeline:

1. **Primary Selection**: YOLO detection + ByteTrack tracking
2. **Profile Tracking**: Histogram-based filtering (Brightness + Saturation)
3. **Dual Re-ID & Frame Merging**:
   - Person objects (class 0) → Person Re-ID model
   - Other objects → Vehicle Re-ID model
4. **Greedy Coverage Selection**: k-combinations for object diversity
5. **Post-Greedy Profile Tracking**: Final refinement

## Development

The main algorithm is in `yolo_osnet_4_with_filtering_updated (1).py`. The enhanced version with dual Re-ID will be in `yolo_osnet_4_with_filtering_updated_dual.py`.
