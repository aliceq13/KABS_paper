"""
Model Frame Comparison and GT Optimization Tool

Compare keyframes selected by different models and analyze GT optimization opportunities.

Usage:
    python compare_model_frames.py
    python compare_model_frames.py --result-folder experiment_results/single_video_20250112_143000
    python compare_model_frames.py --optimize-gt
"""

import os
import json
import argparse
import glob
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import pandas as pd
from evaluation_metrics import calculate_f1_with_tolerance, load_ground_truth_keyframes


def find_latest_result_folder(base_folder: str = "experiment_results") -> str:
    """Find the latest single_video result folder."""
    pattern = os.path.join(base_folder, "single_video_*")
    folders = glob.glob(pattern)

    if not folders:
        raise FileNotFoundError(f"No result folders found in {base_folder}")

    # Sort by modification time, get latest
    latest = max(folders, key=os.path.getmtime)
    return latest


def load_selected_frames(result_folder: str) -> Dict[str, List[int]]:
    """
    Load selected frames from each model's JSON file.

    Returns:
        Dictionary mapping model name to list of selected frame indices
    """
    model_frames = {}
    model_results_folder = os.path.join(result_folder, "model_results")

    if not os.path.exists(model_results_folder):
        print(f"✗ Model results folder not found: {model_results_folder}")
        return model_frames

    # Find video subfolder
    video_folders = [d for d in os.listdir(model_results_folder)
                    if os.path.isdir(os.path.join(model_results_folder, d))]

    if not video_folders:
        print(f"✗ No video folders found in {model_results_folder}")
        return model_frames

    video_folder = os.path.join(model_results_folder, video_folders[0])

    # Load each model's results
    model_names = ["YOLO_BOTSORT", "YOLO_ByteTrack", "RTDETR_BOTSORT", "RTDETR_ByteTrack"]

    for model_name in model_names:
        json_path = os.path.join(video_folder, model_name, "keyframe_summary_unified.json")

        if not os.path.exists(json_path):
            print(f"⚠️ JSON not found for {model_name}: {json_path}")
            continue

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                frame_indices = data.get('selected_frame_indices', [])
                model_frames[model_name] = sorted(frame_indices)
                print(f"✓ Loaded {model_name}: {len(frame_indices)} frames")
        except Exception as e:
            print(f"✗ Error loading {model_name}: {e}")

    return model_frames


def analyze_overlap(model_frames: Dict[str, List[int]]) -> Dict:
    """Analyze frame overlap between models."""
    if not model_frames:
        return {}

    # Convert to sets
    frame_sets = {name: set(frames) for name, frames in model_frames.items()}

    # Get all unique frames
    all_frames = set()
    for frames in frame_sets.values():
        all_frames.update(frames)

    # Count how many models selected each frame
    frame_counts = defaultdict(int)
    frame_models = defaultdict(list)

    for frame in all_frames:
        for model_name, frame_set in frame_sets.items():
            if frame in frame_set:
                frame_counts[frame] += 1
                frame_models[frame].append(model_name)

    # Categorize frames
    all_4 = [f for f in all_frames if frame_counts[f] == 4]
    at_least_3 = [f for f in all_frames if frame_counts[f] >= 3]
    at_least_2 = [f for f in all_frames if frame_counts[f] >= 2]
    unique = [f for f in all_frames if frame_counts[f] == 1]

    return {
        'all_frames': sorted(all_frames),
        'total_unique': len(all_frames),
        'all_4_models': sorted(all_4),
        'at_least_3': sorted(at_least_3),
        'at_least_2': sorted(at_least_2),
        'unique_frames': sorted(unique),
        'frame_counts': dict(frame_counts),
        'frame_models': dict(frame_models)
    }


def calculate_similarity(model_frames: Dict[str, List[int]]) -> pd.DataFrame:
    """Calculate pairwise similarity (Jaccard index) between models."""
    model_names = list(model_frames.keys())
    n = len(model_names)

    # Create similarity matrix
    similarity_matrix = []

    for i, model1 in enumerate(model_names):
        row = []
        set1 = set(model_frames[model1])

        for j, model2 in enumerate(model_names):
            set2 = set(model_frames[model2])

            if len(set1) == 0 and len(set2) == 0:
                similarity = 1.0
            elif len(set1) == 0 or len(set2) == 0:
                similarity = 0.0
            else:
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                similarity = intersection / union if union > 0 else 0.0

            row.append(similarity)

        similarity_matrix.append(row)

    # Create DataFrame
    df = pd.DataFrame(similarity_matrix, index=model_names, columns=model_names)
    return df


def generate_comparison_table(model_frames: Dict[str, List[int]], overlap_analysis: Dict) -> pd.DataFrame:
    """Generate detailed frame-by-frame comparison table."""
    all_frames = overlap_analysis['all_frames']
    frame_counts = overlap_analysis['frame_counts']
    frame_models = overlap_analysis['frame_models']

    rows = []
    for frame in all_frames:
        row = {'frame_idx': frame}

        # Check each model
        for model_name in model_frames.keys():
            row[model_name] = '✓' if frame in model_frames[model_name] else '-'

        # Add count and category
        count = frame_counts[frame]
        row['count'] = count

        if count == 4:
            category = 'All'
        elif count == 3:
            category = '3_models'
        elif count == 2:
            category = '2_models'
        else:
            category = 'Unique'

        row['category'] = category
        row['selected_by'] = ', '.join(frame_models[frame])

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def load_current_gt(result_folder: str, gt_folder: str = "Keyframe-extraction/Dataset/Keyframe") -> Tuple[str, List[int]]:
    """Load current ground truth for the video."""
    # Find video name from result folder
    model_results_folder = os.path.join(result_folder, "model_results")

    if not os.path.exists(model_results_folder):
        raise FileNotFoundError(f"Model results folder not found: {model_results_folder}")

    video_folders = [d for d in os.listdir(model_results_folder)
                    if os.path.isdir(os.path.join(model_results_folder, d))]

    if not video_folders:
        raise FileNotFoundError("No video folders found")

    video_name = video_folders[0] + ".mp4"

    # Load GT
    gt_frames = load_ground_truth_keyframes(video_name, gt_folder)

    return video_name, gt_frames


def find_candidate_frames(model_frames: Dict[str, List[int]], gt_frames: List[int]) -> List[int]:
    """Find frames selected by models but not in GT."""
    gt_set = set(gt_frames)

    candidates = set()
    for frames in model_frames.values():
        for frame in frames:
            if frame not in gt_set:
                candidates.add(frame)

    return sorted(candidates)


def simulate_frame_addition(candidate_frame: int,
                           gt_frames: List[int],
                           model_frames: Dict[str, List[int]],
                           total_frames: int,
                           tolerance: int = 15) -> Dict:
    """
    Simulate adding a frame to GT and recalculate F1-scores.

    Returns:
        Dictionary with original and new F1-scores for each model
    """
    # Create new GT with candidate frame
    new_gt = sorted(gt_frames + [candidate_frame])

    results = {
        'candidate_frame': candidate_frame,
        'models_selected': sum(1 for frames in model_frames.values() if candidate_frame in frames)
    }

    # Calculate F1 for each model
    original_scores = []
    new_scores = []

    for model_name, predicted_frames in model_frames.items():
        # Original F1
        original_metrics = calculate_f1_with_tolerance(predicted_frames, gt_frames, tolerance)
        original_f1 = original_metrics['f1_score']

        # New F1 with candidate frame added
        new_metrics = calculate_f1_with_tolerance(predicted_frames, new_gt, tolerance)
        new_f1 = new_metrics['f1_score']

        gain = new_f1 - original_f1

        results[f'{model_name}_original'] = original_f1
        results[f'{model_name}_new'] = new_f1
        results[f'{model_name}_gain'] = gain

        original_scores.append(original_f1)
        new_scores.append(new_f1)

    # Calculate averages
    results['avg_original_f1'] = sum(original_scores) / len(original_scores)
    results['avg_new_f1'] = sum(new_scores) / len(new_scores)
    results['avg_f1_gain'] = results['avg_new_f1'] - results['avg_original_f1']

    return results


def analyze_gt_optimization(model_frames: Dict[str, List[int]],
                           gt_frames: List[int],
                           total_frames: int,
                           tolerance: int = 15) -> pd.DataFrame:
    """Analyze GT optimization by simulating frame additions."""
    print(f"\n{'='*80}")
    print("GT OPTIMIZATION ANALYSIS")
    print(f"{'='*80}\n")

    # Find candidates
    candidates = find_candidate_frames(model_frames, gt_frames)

    if not candidates:
        print("No candidate frames found (all model selections are in GT)")
        return pd.DataFrame()

    print(f"Found {len(candidates)} candidate frames to analyze...")

    # Simulate each candidate
    simulation_results = []

    for i, candidate in enumerate(candidates, 1):
        if i % 10 == 0:
            print(f"  Processing candidate {i}/{len(candidates)}...")

        result = simulate_frame_addition(candidate, gt_frames, model_frames, total_frames, tolerance)
        simulation_results.append(result)

    # Create DataFrame and sort by gain
    df = pd.DataFrame(simulation_results)
    df = df.sort_values('avg_f1_gain', ascending=False)

    # Add recommendation
    df['recommendation'] = df['avg_f1_gain'].apply(
        lambda x: 'High' if x >= 0.08 else ('Medium' if x >= 0.04 else 'Low')
    )

    return df


def print_summary(model_frames: Dict[str, List[int]],
                 overlap_analysis: Dict,
                 similarity_df: pd.DataFrame,
                 video_name: str,
                 total_frames: int):
    """Print text summary."""
    print(f"\n{'='*80}")
    print("FRAME SELECTION COMPARISON ANALYSIS")
    print(f"{'='*80}")
    print(f"Video: {video_name}")
    print(f"Total Frames: {total_frames}")
    print(f"\nModel Statistics:")

    for model_name, frames in model_frames.items():
        print(f"  {model_name:<20} {len(frames):>3} frames selected")

    print(f"\nOverlap Analysis:")
    print(f"  Total unique frames:   {overlap_analysis['total_unique']}")
    print(f"  All 4 models:          {len(overlap_analysis['all_4_models'])} frames")
    print(f"  At least 3 models:     {len(overlap_analysis['at_least_3'])} frames")
    print(f"  At least 2 models:     {len(overlap_analysis['at_least_2'])} frames")
    print(f"  Unique (1 model only): {len(overlap_analysis['unique_frames'])} frames")

    print(f"\nCommon Frames (All 4 models):")
    if overlap_analysis['all_4_models']:
        print(f"  {overlap_analysis['all_4_models']}")
    else:
        print("  None")

    print(f"\nModel Similarity (Jaccard Index):")
    print(similarity_df.round(3).to_string())
    print(f"{'='*80}\n")


def print_optimization_summary(optimization_df: pd.DataFrame, gt_frames: List[int], top_n: int = 10):
    """Print GT optimization summary."""
    if optimization_df.empty:
        return

    print(f"\n{'='*80}")
    print("GT OPTIMIZATION RECOMMENDATIONS")
    print(f"{'='*80}")

    print(f"\nCurrent GT: {len(gt_frames)} frames")
    print(f"  {gt_frames}")

    print(f"\nTop {min(top_n, len(optimization_df))} frames to add to GT (by F1-score improvement):\n")

    for i, row in optimization_df.head(top_n).iterrows():
        print(f"{i+1}. Frame {int(row['candidate_frame']):<5} [Selected by {int(row['models_selected'])}/4 models]")
        print(f"   Average F1 gain: {row['avg_f1_gain']:+.4f}")
        print(f"   New avg F1: {row['avg_original_f1']:.4f} → {row['avg_new_f1']:.4f}")
        print(f"   Recommendation: {row['recommendation']}")
        print()

    print(f"{'='*80}\n")


def save_results(model_frames: Dict[str, List[int]],
                overlap_analysis: Dict,
                similarity_df: pd.DataFrame,
                comparison_df: pd.DataFrame,
                optimization_df: pd.DataFrame,
                output_folder: str):
    """Save all results to files."""
    comparison_folder = os.path.join(output_folder, "frame_comparison")
    os.makedirs(comparison_folder, exist_ok=True)

    # Save comparison table
    comparison_csv = os.path.join(comparison_folder, "frame_overlap_detail.csv")
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"✓ Saved frame comparison: {comparison_csv}")

    # Save similarity matrix
    similarity_csv = os.path.join(comparison_folder, "model_similarity.csv")
    similarity_df.to_csv(similarity_csv)
    print(f"✓ Saved similarity matrix: {similarity_csv}")

    # Save optimization results
    if not optimization_df.empty:
        optimization_csv = os.path.join(comparison_folder, "gt_optimization.csv")
        optimization_df.to_csv(optimization_csv, index=False)
        print(f"✓ Saved GT optimization: {optimization_csv}")
    else:
        print(f"⚠️ GT optimization CSV not saved (no data generated)")

    # Save statistics JSON
    stats = {
        'model_frame_counts': {name: len(frames) for name, frames in model_frames.items()},
        'overlap_statistics': {
            'total_unique_frames': overlap_analysis['total_unique'],
            'all_4_models': len(overlap_analysis['all_4_models']),
            'at_least_3': len(overlap_analysis['at_least_3']),
            'at_least_2': len(overlap_analysis['at_least_2']),
            'unique': len(overlap_analysis['unique_frames'])
        }
    }

    stats_json = os.path.join(comparison_folder, "statistics.json")
    with open(stats_json, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved statistics: {stats_json}")

    print(f"\n✓ All results saved to: {comparison_folder}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare model frame selections and optimize GT",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--result-folder', type=str,
                       help='Path to experiment result folder (default: latest)')
    parser.add_argument('--gt-folder', type=str,
                       default='Keyframe-extraction/Dataset/Keyframe',
                       help='Ground truth folder')
    parser.add_argument('--optimize-gt', action='store_true', default=True,
                       help='Perform GT optimization analysis (default: True)')
    parser.add_argument('--no-optimize-gt', dest='optimize_gt', action='store_false',
                       help='Skip GT optimization analysis')
    parser.add_argument('--tolerance', type=int, default=15,
                       help='Temporal tolerance for F1 calculation (default: 15)')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Show top N recommendations (default: 10)')

    args = parser.parse_args()

    # Find result folder
    if args.result_folder:
        result_folder = args.result_folder
    else:
        try:
            result_folder = find_latest_result_folder()
            print(f"Using latest result folder: {result_folder}")
        except FileNotFoundError as e:
            print(f"✗ {e}")
            return

    # Load model frames
    print(f"\n{'='*80}")
    print("LOADING MODEL RESULTS")
    print(f"{'='*80}\n")

    model_frames = load_selected_frames(result_folder)

    if not model_frames:
        print("✗ No model results found")
        return

    # Basic analysis
    overlap_analysis = analyze_overlap(model_frames)
    similarity_df = calculate_similarity(model_frames)
    comparison_df = generate_comparison_table(model_frames, overlap_analysis)

    # Print summary
    video_name = os.path.basename(result_folder).replace("single_video_", "")

    # Try to get total frames from evaluation results
    eval_csv = os.path.join(result_folder, "evaluation", "results.csv")
    total_frames = 0
    if os.path.exists(eval_csv):
        df_eval = pd.read_csv(eval_csv)
        if not df_eval.empty and 'total_frames' in df_eval.columns:
            total_frames = int(df_eval['total_frames'].iloc[0])

    print_summary(model_frames, overlap_analysis, similarity_df, video_name, total_frames)

    # GT optimization analysis
    optimization_df = pd.DataFrame()
    if args.optimize_gt:
        print(f"\n{'='*80}")
        print("GT OPTIMIZATION ANALYSIS")
        print(f"{'='*80}\n")

        try:
            video_name_full, gt_frames = load_current_gt(result_folder, args.gt_folder)
            print(f"✓ Loaded GT: {len(gt_frames)} frames")

            if total_frames == 0:
                print("⚠️ Could not determine total frames from evaluation results")
                print("   Make sure results.csv exists in evaluation folder")
                print("⚠️ Skipping GT optimization")
            else:
                print(f"✓ Total frames: {total_frames}")
                print(f"Analyzing {len(model_frames)} model configurations...")

                optimization_df = analyze_gt_optimization(
                    model_frames, gt_frames, total_frames, args.tolerance
                )

                if not optimization_df.empty:
                    print(f"\n✓ Generated {len(optimization_df)} optimization recommendations")
                    print_optimization_summary(optimization_df, gt_frames, args.top_n)
                else:
                    print("⚠️ No optimization recommendations generated")

        except Exception as e:
            print(f"✗ Error in GT optimization: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n{'='*80}")
        print("GT optimization skipped (use --optimize-gt to enable)")
        print(f"{'='*80}\n")

    # Save results
    save_results(model_frames, overlap_analysis, similarity_df,
                comparison_df, optimization_df, result_folder)

    # Final summary
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Results location: {os.path.join(result_folder, 'frame_comparison')}")
    print(f"\nGenerated files:")
    print(f"  • frame_overlap_detail.csv - Detailed frame-by-frame comparison")
    print(f"  • model_similarity.csv - Jaccard similarity between models")
    print(f"  • statistics.json - Summary statistics")
    if not optimization_df.empty:
        print(f"  • gt_optimization.csv - GT optimization recommendations ✓")
    else:
        print(f"  • gt_optimization.csv - NOT GENERATED (see warnings above)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
