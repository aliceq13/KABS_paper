# Re-ID Models Selection Guide

## Overview

KABS_enhance now supports **multiple SOTA Re-ID models** through a model registry system. You can easily switch between different models to compare performance.

## Quick Start

### 1. List Available Models

```python
from reid_model_registry import list_available_models

# List person Re-ID models
list_available_models("person")

# List vehicle Re-ID models
list_available_models("vehicle")
```

### 2. Select Models in Config

Edit `yolo_osnet_4_dual_reid.py` config:

```python
config = {
    # ...
    "use_dual_reid": True,

    # 🔥 Change these to test different models!
    "person_reid_model_key": "osnet_market1501",  # Person model
    "vehicle_reid_model_key": "osnet_x1_0",       # Vehicle model
    # ...
}
```

### 3. Run and Compare

```bash
# Single run with selected models
python yolo_osnet_4_dual_reid.py

# Compare multiple models automatically
python compare_reid_models.py \
    --video test_video.mp4 \
    --output comparison_results \
    --test-all
```

## Available Models

### Person Re-ID Models

| Model Key | Name | Description | Performance | Speed |
|-----------|------|-------------|-------------|-------|
| `osnet_market1501` | OSNet Market-1501 | Fine-tuned on Market-1501 (751 identities) | ⭐⭐⭐⭐⭐ Best | ⭐⭐⭐⭐ Good |
| `osnet_x1_0` | OSNet x1.0 | Pretrained on ImageNet | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Good |
| `osnet_x0_75` | OSNet x0.75 | Lightweight variant | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Fast |
| `osnet_x0_5` | OSNet x0.5 | Most lightweight | ⭐⭐ Fair | ⭐⭐⭐⭐⭐ Fastest |
| `osnet_ibn_x1_0` | OSNet-IBN | With Instance-Batch Normalization | ⭐⭐⭐⭐ Very Good | ⭐⭐⭐ Moderate |
| `resnet50_fc512` | ResNet50 | Baseline model | ⭐⭐⭐ Good | ⭐⭐⭐ Moderate |
| `mlfn` | MLFN | Multi-Level Factorisation Net | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Good |

**Recommended for Person Re-ID:**
1. **Best Accuracy**: `osnet_market1501`
2. **Balanced**: `osnet_x1_0`
3. **Fastest**: `osnet_x0_5`

### Vehicle Re-ID Models

| Model Key | Name | Description | Performance | Speed |
|-----------|------|-------------|-------------|-------|
| `osnet_veri776` | OSNet VeRi-776 | Fine-tuned on VeRi-776 (776 vehicle IDs) | ⭐⭐⭐⭐⭐ Best* | ⭐⭐⭐⭐ Good |
| `osnet_x1_0` | OSNet x1.0 | Pretrained on ImageNet | ⭐⭐⭐ Good | ⭐⭐⭐⭐ Good |
| `resnet50_fc512` | ResNet50 | Baseline model | ⭐⭐⭐ Good | ⭐⭐⭐ Moderate |

*Requires VeRi-776 trained weights (optional)

**Recommended for Vehicle Re-ID:**
1. **Best Accuracy**: `osnet_veri776` (if available)
2. **General**: `osnet_x1_0`

## Usage Examples

### Example 1: Test Market-1501 Person Model

```python
config = {
    "video_path": "test_video.mp4",
    "output_folder": "results_market1501",
    "use_dual_reid": True,
    "person_reid_model_key": "osnet_market1501",  # 🔥 Best for person
    "vehicle_reid_model_key": "osnet_x1_0",
}

main(**config)
```

### Example 2: Test Lightweight Model for Speed

```python
config = {
    "video_path": "test_video.mp4",
    "output_folder": "results_lightweight",
    "use_dual_reid": True,
    "person_reid_model_key": "osnet_x0_5",  # 🔥 Fastest
    "vehicle_reid_model_key": "osnet_x1_0",
}

main(**config)
```

### Example 3: Compare All Person Models

```bash
python compare_reid_models.py \
    --video test_video.mp4 \
    --output person_comparison \
    --test-person
```

This will test:
- `osnet_market1501`
- `osnet_x1_0`
- `osnet_x0_75`
- `resnet50_fc512`

Results saved to `person_comparison/comparison_results.json`

### Example 4: Compare Vehicle Models

```bash
python compare_reid_models.py \
    --video test_video.mp4 \
    --output vehicle_comparison \
    --test-vehicle
```

### Example 5: Compare All Models

```bash
python compare_reid_models.py \
    --video test_video.mp4 \
    --output full_comparison \
    --test-all
```

## Model Comparison Results

After running comparison, check:

```json
{
  "person_osnet_market1501": {
    "model_name": "OSNet Market-1501 (Best)",
    "num_keyframes": 42,
    "runtime_seconds": 125.3,
    "success": true
  },
  "person_osnet_x0_5": {
    "model_name": "OSNet x0.5 (Fastest)",
    "num_keyframes": 45,
    "runtime_seconds": 87.1,
    "success": true
  }
}
```

## Adding Custom Models

### 1. Add to Registry

Edit `reid_model_registry.py`:

```python
PERSON_REID_MODELS = {
    # ... existing models ...

    "my_custom_model": {
        "name": "My Custom Model",
        "description": "Custom Re-ID model",
        "num_classes": 1000,
        "pretrained_name": None,
        "local_path": "path/to/my_model.pth",
        "framework": "torchreid",
    },
}
```

### 2. Use in Config

```python
config = {
    "person_reid_model_key": "my_custom_model",  # Your custom model
}
```

## Model Loading Priority

1. **Registry Key** (recommended)
   ```python
   "person_reid_model_key": "osnet_market1501"
   ```

2. **Local Path Override**
   ```python
   "person_reid_model_key": "osnet_x1_0",  # Registry key
   "person_reid_model_path": "/custom/path/model.pth"  # Overrides registry
   ```

3. **Pretrained Weights** (auto-download)
   - If no local path found, downloads from TorchReID

## Performance Tips

### For Best Accuracy
- Person: `osnet_market1501`
- Vehicle: `osnet_veri776` (if available)

### For Best Speed
- Person: `osnet_x0_5` or `osnet_x0_75`
- Vehicle: `osnet_x1_0`

### For Balanced Performance
- Person: `osnet_x1_0` or `osnet_ibn_x1_0`
- Vehicle: `osnet_x1_0`

### Memory Considerations
- Each model: ~100-200MB
- Dual models: ~200-400MB total
- Lightweight models (`x0_5`, `x0_75`) use less memory

## Troubleshooting

### Model not found error
```
ValueError: Model 'xxx' not found. Available: [...]
```
**Solution**: Check available models with `list_available_models()`

### Weights not found
```
⚠️ No weights found, using random initialization
```
**Solution**:
- For fine-tuned models: Download weights to local path
- For pretrained models: Will auto-download from TorchReID

### Out of memory
**Solution**:
- Use lightweight models: `osnet_x0_5`, `osnet_x0_75`
- Reduce batch processing
- Use CPU instead of GPU

## References

- [OSNet Paper](https://arxiv.org/abs/1905.00953)
- [Market-1501 Dataset](https://paperswithcode.com/dataset/market-1501)
- [VeRi-776 Dataset](https://github.com/JDAI-CV/VeRidataset)
- [TorchReID Library](https://github.com/KaiyangZhou/deep-person-reid)

## Next Steps

1. **Download VeRi-776 model** for better vehicle Re-ID
2. **Test different models** on your videos
3. **Compare results** using `compare_reid_models.py`
4. **Select best model** based on accuracy vs speed tradeoff
