# Keyframe Extraction Evaluation Experiment

This experiment evaluates your keyframe extraction model against baseline methods using the TvSum-based ground truth dataset.

## 📋 Overview

### Evaluation Metrics
- **F1-Score** with multiple temporal tolerances:
  - ±0 frames (exact matching)
  - ±15 frames (~0.5 second @ 30fps)
  - ±30 frames (~1 second @ 30fps)
- **Precision** and **Recall**
- **Compression Ratio** (efficiency measure)

### Methods Compared
1. **Your Model** with different configurations:
   - YOLO + BOT-SORT
   - YOLO + ByteTrack
   - RT-DETR + BOT-SORT
   - RT-DETR + ByteTrack

2. **Baseline Methods**:
   - Uniform sampling every 30 frames
   - Uniform sampling every 60 frames

## 📁 File Structure

```
KABS_paper/
├── evaluation_metrics.py          # Evaluation metrics implementation
├── baseline_methods.py             # Baseline keyframe extraction
├── model_wrapper.py                # Wrapper for your model
├── run_experiments.py              # Main experiment runner
├── yolo_osnet_4_with_filtering_updated (1).py  # Your model (modified)
├── Keyframe-extraction/
│   └── Dataset/
│       ├── Videos/                 # Test videos (20 videos)
│       └── Keyframe/                # Ground truth keyframes
└── experiment_results/             # Output folder (created after running)
```

## 🚀 Running the Experiment

### Prerequisites

1. **Install Dependencies**:
   ```bash
   pip install ultralytics torchreid transformers pandas opencv-python
   ```

2. **Download Model Weights**:
   - YOLO: `yolo11m.pt` (should be in current directory)
   - RT-DETR: `rtdetr-l.pt`
   - TorchReID OSNet: `osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth`

### Execute Experiment

Run the main experiment script:

```bash
python run_experiments.py
```

This will:
1. Process all 20 videos in `Keyframe-extraction/Dataset/Videos/`
2. Extract keyframes using baselines (30fps, 60fps)
3. Extract keyframes using your model (4 configurations)
4. Evaluate all methods against ground truth
5. Generate CSV files and summary statistics

## 📊 Output Structure

After running, you'll get:

```
experiment_results/
└── experiment_YYYYMMDD_HHMMSS/
    ├── baseline_results/
    │   └── [video_name]/
    │       ├── Uniform_30frames/
    │       └── Uniform_60frames/
    ├── model_results/
    │   └── [video_name]/
    │       ├── YOLO_BOTSORT/
    │       ├── YOLO_ByteTrack/
    │       ├── RTDETR_BOTSORT/
    │       └── RTDETR_ByteTrack/
    └── evaluation/
        ├── detailed_results.csv          # Per-video, per-method results
        ├── aggregated_by_method.csv      # Average performance by method
        ├── aggregated_by_video.csv       # Average performance by video
        └── summary.txt                   # Text summary
```

## 📈 Understanding Results

### CSV Columns

**detailed_results.csv**:
- `method`: Method name (e.g., "YOLO_BOTSORT", "Uniform_30frames")
- `video_name`: Video filename
- `num_keyframes`: Number of keyframes extracted
- `compression_ratio`: Ratio of keyframes to total frames
- `precision_tol0`, `recall_tol0`, `f1_score_tol0`: Exact matching metrics
- `precision_tol15`, `recall_tol15`, `f1_score_tol15`: ±15 frame tolerance
- `precision_tol30`, `recall_tol30`, `f1_score_tol30`: ±30 frame tolerance
- `tp_tol*`, `fp_tol*`, `fn_tol*`: True/False Positives/Negatives

**aggregated_by_method.csv**:
- Mean and standard deviation for each metric across all videos
- Allows comparison of average performance

### Interpreting Metrics

- **F1-Score**: Overall quality (0-1, higher is better)
  - F1 @ ±0: Strict evaluation
  - F1 @ ±30: Lenient evaluation (more realistic)

- **Precision**: How many selected keyframes are correct
  - High precision = few false positives

- **Recall**: How many ground truth keyframes were found
  - High recall = few missed keyframes

- **Compression Ratio**: Efficiency (0-1, lower is better)
  - Lower = fewer keyframes selected = more efficient

## 🔧 Customization

### Modify Experiment Configurations

Edit `run_experiments.py`:

```python
# Change baseline intervals
baseline_intervals=[30, 60, 90]  # Add 90-frame sampling

# Change temporal tolerances
tolerances=[0, 15, 30, 45]  # Add 45-frame tolerance

# Modify user model configs
USER_MODEL_CONFIGS = [
    {
        'name': 'YOLO_BOTSORT',
        'model_type': 'yolo',
        'model_path': 'yolo11m.pt',
        'tracker': 'botsort.yaml',
        'frame_skip_interval': 1  # Change to 5 for faster processing
    },
    # Add more configurations...
]
```

### Run Only Baselines or Only Your Model

```python
# In run_full_experiment()
run_baselines=True,      # Set to False to skip baselines
run_user_model=True,     # Set to False to skip your model
```

## 📝 Notes

1. **Processing Time**:
   - Each video takes ~2-5 minutes with your model
   - Baselines are much faster (~10-30 seconds per video)
   - Total time for all: ~2-3 hours

2. **Memory Usage**:
   - Peak GPU memory: ~4-6 GB
   - Ensure sufficient disk space (~10-20 GB for outputs)

3. **Frame Skip**:
   - Default `frame_skip_interval=1` (no skipping)
   - Increase to 5 for 5x speedup but less accurate keyframes

4. **Depth Estimation**:
   - Currently disabled in model_wrapper.py for speed
   - Enable by setting `"use_depth": True` if needed

## 🐛 Troubleshooting

### "Ground truth folder not found"
- Ensure `Keyframe-extraction/Dataset/Keyframe/` exists
- Each video should have a corresponding folder (e.g., `video1.mp4/`)

### "Model weights not found"
- Download YOLO/RT-DETR weights to current directory
- Check model_path in configurations

### "CUDA out of memory"
- Reduce batch size (not applicable here, processing 1 frame at a time)
- Use CPU: `device = "cpu"` in load_models()
- Try smaller models (yolo11s.pt instead of yolo11m.pt)

### "Import errors"
- Install missing packages: `pip install [package_name]`
- For torchreid: `pip install torchreid`

## 📧 Questions?

If you encounter issues:
1. Check error messages in console output
2. Verify all model weights are downloaded
3. Ensure ground truth dataset is properly structured
4. Check that videos are readable (not corrupted)

## 🎯 Expected Results

Good keyframe extraction should achieve:
- F1@±30 > 0.6 (60%+ accuracy with 1-second tolerance)
- F1@±15 > 0.5 (50%+ with 0.5-second tolerance)
- Compression ratio < 0.1 (less than 10% of frames)

Baseline methods typically achieve:
- F1@±30: 0.3-0.5
- F1@±15: 0.2-0.4

Your model should outperform baselines!
