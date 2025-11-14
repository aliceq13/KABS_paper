# Compare Model Frames - Usage Guide

## Overview
This tool compares keyframe selections from different model configurations and provides GT optimization recommendations.

## Prerequisites
1. Run evaluation first: `python evaluate_single_video.py --video your_video.mp4`
2. Make sure Ground Truth exists in: `Keyframe-extraction/Dataset/Keyframe/`

## Basic Usage

### Compare latest experiment results
```bash
python compare_model_frames.py
```

### Compare specific result folder
```bash
python compare_model_frames.py --result-folder experiment_results/single_video_20240115_123456
```

### Skip GT optimization (not recommended)
```bash
python compare_model_frames.py --no-optimize-gt
```

### Custom tolerance and top-N recommendations
```bash
python compare_model_frames.py --tolerance 30 --top-n 20
```

## Output Files

The tool generates files in `<result_folder>/frame_comparison/`:

1. **frame_overlap_detail.csv**
   - Detailed frame-by-frame comparison
   - Shows which models selected each frame
   - Columns: frame number, model1, model2, model3, model4, selection_count, models

2. **model_similarity.csv**
   - Jaccard similarity matrix between all models
   - Range: 0.0 (no overlap) to 1.0 (identical)

3. **gt_optimization.csv** ⭐
   - Frame-by-frame GT optimization recommendations
   - Shows predicted F1-score improvement for each candidate frame
   - Sorted by improvement potential (best first)
   - Columns:
     - `candidate_frame`: Frame number to add to GT
     - `improvement_*`: F1-score improvement for each model
     - `avg_improvement`: Average improvement across all models
     - `current_f1_*`: Current F1-score for each model
     - `new_f1_*`: Predicted F1-score if frame is added

4. **statistics.json**
   - Summary statistics in JSON format

## Troubleshooting

### Problem: gt_optimization.csv not generated

**Possible causes:**

1. **GT optimization was disabled**
   - Solution: GT optimization is now enabled by default. Just run: `python compare_model_frames.py`

2. **No evaluation results found**
   - Error message: "Could not determine total frames from evaluation results"
   - Solution: Run evaluation first: `python evaluate_single_video.py --video your_video.mp4`

3. **Ground Truth not found**
   - Error message: "Error in GT optimization: ..."
   - Solution: Create GT using: `python create_ground_truth.py --video your_video.mp4`

4. **Wrong GT folder path**
   - Solution: Specify correct path: `python compare_model_frames.py --gt-folder custom/path`

### Problem: "No model results found"

- Make sure you're pointing to the correct result folder
- Result folder should contain: `model_results/<video_name>/*/keyframe_summary_unified.json`

### Problem: "Could not find any result folders"

- Run evaluation first: `python evaluate_single_video.py --video your_video.mp4`
- Or specify folder manually: `python compare_model_frames.py --result-folder path/to/results`

## Example Workflow

```bash
# Step 1: Create Ground Truth
python create_ground_truth.py --video my_video.mp4

# Step 2: Run evaluation
python evaluate_single_video.py --video my_video.mp4

# Step 3: Compare models and optimize GT
python compare_model_frames.py

# Step 4: Check results
ls experiment_results/single_video_*/frame_comparison/
```

## Understanding GT Optimization Results

The `gt_optimization.csv` file contains recommendations for improving your Ground Truth:

- **High `avg_improvement`**: Adding this frame would significantly improve all models
- **Positive `improvement_*`**: Adding this frame would improve that specific model
- **Negative values**: Frame already well-covered, adding won't help much

**How to use:**
1. Sort by `avg_improvement` (descending)
2. Look at top 10-20 frames
3. Review those frames in your video
4. Add frames that represent important composition changes
5. Re-run evaluation to see improvement

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--result-folder` | Latest | Path to experiment result folder |
| `--gt-folder` | `Keyframe-extraction/Dataset/Keyframe` | Ground truth folder |
| `--optimize-gt` | `True` | Perform GT optimization (now default) |
| `--no-optimize-gt` | - | Skip GT optimization |
| `--tolerance` | `15` | Temporal tolerance for F1 calculation |
| `--top-n` | `10` | Show top N recommendations |

## Tips

1. **Always run evaluation before comparison**: The tool needs evaluation results
2. **Check terminal output carefully**: Error messages will tell you what's missing
3. **GT optimization is now automatic**: No need to add flags
4. **Use tolerance=15 for standard evaluation**: This is ±0.5 seconds at 30fps
5. **Review top 10-20 recommendations**: Don't try to add all suggested frames

## Recent Changes

**Version 2.0 (Current)**
- ✅ GT optimization now enabled by default
- ✅ Much clearer error messages
- ✅ Better progress indicators
- ✅ Final summary shows which files were generated
- ✅ Explicit warnings when GT optimization is skipped

**What changed:**
- Before: You had to use `--optimize-gt` flag
- Now: GT optimization runs automatically (use `--no-optimize-gt` to skip)
