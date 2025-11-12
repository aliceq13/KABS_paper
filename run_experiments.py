"""
Main Experiment Runner

This script runs all experiments:
1. Baseline methods (uniform sampling)
2. User's model with different configurations
3. Evaluation and comparison
4. CSV and visualization output
"""

import os
import sys
import json
from typing import List, Dict
import pandas as pd
from datetime import datetime

# Import our modules
from baseline_methods import extract_baseline_keyframes
from model_wrapper import run_multiple_configurations
from evaluation_metrics import (
    load_ground_truth_keyframes,
    evaluate_method,
    save_results_to_csv,
    print_evaluation_summary,
    get_total_frame_count
)


def get_video_list(video_folder: str) -> List[str]:
    """
    Get list of video files from folder.

    Args:
        video_folder: Path to folder containing videos

    Returns:
        List of video file paths
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []

    for filename in os.listdir(video_folder):
        if any(filename.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(video_folder, filename))

    return sorted(video_files)


def run_full_experiment(video_folder: str,
                       ground_truth_folder: str,
                       output_base_folder: str,
                       run_baselines: bool = True,
                       run_user_model: bool = True,
                       baseline_intervals: List[int] = [30, 60],
                       user_model_configs: List[Dict] = None,
                       tolerances: List[int] = [0, 15, 30]):
    """
    Run full experiment pipeline.

    Args:
        video_folder: Folder containing test videos
        ground_truth_folder: Folder containing ground truth keyframes
        output_base_folder: Base folder for all outputs
        run_baselines: Whether to run baseline methods
        run_user_model: Whether to run user's model
        baseline_intervals: List of sampling intervals for baselines
        user_model_configs: List of user model configurations
        tolerances: List of temporal tolerances for evaluation
    """
    # Create output folders
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = os.path.join(output_base_folder, f"experiment_{timestamp}")
    os.makedirs(experiment_folder, exist_ok=True)

    baseline_output = os.path.join(experiment_folder, "baseline_results")
    model_output = os.path.join(experiment_folder, "model_results")
    evaluation_output = os.path.join(experiment_folder, "evaluation")

    os.makedirs(baseline_output, exist_ok=True)
    os.makedirs(model_output, exist_ok=True)
    os.makedirs(evaluation_output, exist_ok=True)

    # Get video list
    video_files = get_video_list(video_folder)
    print(f"\n{'='*100}")
    print(f"EXPERIMENT SETUP")
    print(f"{'='*100}")
    print(f"Found {len(video_files)} videos:")
    for video_file in video_files:
        print(f"  - {os.path.basename(video_file)}")
    print(f"Output folder: {experiment_folder}")
    print(f"{'='*100}\n")

    # Store all results
    all_results = []

    # Process each video
    for video_idx, video_path in enumerate(video_files, 1):
        video_name = os.path.basename(video_path)
        print(f"\n{'='*100}")
        print(f"PROCESSING VIDEO {video_idx}/{len(video_files)}: {video_name}")
        print(f"{'='*100}\n")

        try:
            # Load ground truth
            print("Loading ground truth keyframes...")
            gt_keyframes = load_ground_truth_keyframes(video_name, ground_truth_folder)
            print(f"✓ Ground truth: {len(gt_keyframes)} keyframes")
            print(f"  Keyframes: {gt_keyframes}")

            # Get total frames
            total_frames = get_total_frame_count(video_path)
            print(f"✓ Total frames in video: {total_frames}")

        except Exception as e:
            print(f"✗ Error loading ground truth for {video_name}: {e}")
            print(f"  Skipping this video...")
            continue

        # Run baseline methods
        if run_baselines:
            print(f"\n{'='*80}")
            print("RUNNING BASELINE METHODS")
            print(f"{'='*80}")

            video_baseline_folder = os.path.join(baseline_output, os.path.splitext(video_name)[0])

            try:
                baseline_results = extract_baseline_keyframes(
                    video_path=video_path,
                    output_base_folder=video_baseline_folder,
                    intervals=baseline_intervals,
                    save_images=True
                )

                # Evaluate baselines
                for method_name, keyframe_indices in baseline_results.items():
                    eval_result = evaluate_method(
                        predicted_frames=keyframe_indices,
                        ground_truth_frames=gt_keyframes,
                        total_frames=total_frames,
                        method_name=method_name,
                        video_name=video_name,
                        tolerances=tolerances
                    )
                    all_results.append(eval_result)

            except Exception as e:
                print(f"✗ Error running baselines for {video_name}: {e}")

        # Run user's model with different configurations
        if run_user_model and user_model_configs:
            print(f"\n{'='*80}")
            print("RUNNING USER MODEL")
            print(f"{'='*80}")

            video_model_folder = os.path.join(model_output, os.path.splitext(video_name)[0])

            try:
                model_results = run_multiple_configurations(
                    video_path=video_path,
                    output_base_folder=video_model_folder,
                    configurations=user_model_configs
                )

                # Evaluate user model results
                for config_name, keyframe_indices in model_results.items():
                    if len(keyframe_indices) == 0:
                        print(f"⚠️ Skipping evaluation for {config_name} (no keyframes extracted)")
                        continue

                    eval_result = evaluate_method(
                        predicted_frames=keyframe_indices,
                        ground_truth_frames=gt_keyframes,
                        total_frames=total_frames,
                        method_name=config_name,
                        video_name=video_name,
                        tolerances=tolerances
                    )
                    all_results.append(eval_result)

            except Exception as e:
                print(f"✗ Error running user model for {video_name}: {e}")

    # Save all results
    print(f"\n{'='*100}")
    print("SAVING RESULTS")
    print(f"{'='*100}\n")

    # Save detailed CSV
    detailed_csv_path = os.path.join(evaluation_output, "detailed_results.csv")
    save_results_to_csv(all_results, detailed_csv_path)

    # Create DataFrame for analysis
    df = pd.DataFrame(all_results)

    # Save aggregated results by method
    aggregated_by_method = df.groupby('method').agg({
        'num_keyframes': ['mean', 'std'],
        'compression_ratio': ['mean', 'std'],
        'f1_score_tol0': ['mean', 'std'],
        'f1_score_tol15': ['mean', 'std'],
        'f1_score_tol30': ['mean', 'std'],
        'precision_tol0': ['mean', 'std'],
        'recall_tol0': ['mean', 'std'],
        'precision_tol15': ['mean', 'std'],
        'recall_tol15': ['mean', 'std'],
        'precision_tol30': ['mean', 'std'],
        'recall_tol30': ['mean', 'std'],
    }).round(4)

    aggregated_csv_path = os.path.join(evaluation_output, "aggregated_by_method.csv")
    aggregated_by_method.to_csv(aggregated_csv_path)
    print(f"✓ Aggregated results saved to: {aggregated_csv_path}")

    # Save aggregated results by video
    aggregated_by_video = df.groupby('video_name').agg({
        'num_keyframes': ['mean', 'std'],
        'compression_ratio': ['mean', 'std'],
        'f1_score_tol0': ['mean', 'std'],
        'f1_score_tol15': ['mean', 'std'],
        'f1_score_tol30': ['mean', 'std'],
    }).round(4)

    aggregated_video_csv_path = os.path.join(evaluation_output, "aggregated_by_video.csv")
    aggregated_by_video.to_csv(aggregated_video_csv_path)
    print(f"✓ Video-wise aggregated results saved to: {aggregated_video_csv_path}")

    # Print summary
    print_evaluation_summary(all_results, tolerances)

    # Save summary to text file
    summary_txt_path = os.path.join(evaluation_output, "summary.txt")
    with open(summary_txt_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("EXPERIMENT SUMMARY\n")
        f.write("="*100 + "\n\n")
        f.write(f"Total videos: {len(video_files)}\n")
        f.write(f"Total methods: {df['method'].nunique()}\n")
        f.write(f"Total evaluations: {len(all_results)}\n\n")
        f.write(aggregated_by_method.to_string())

    print(f"✓ Summary saved to: {summary_txt_path}")

    print(f"\n{'='*100}")
    print(f"EXPERIMENT COMPLETE!")
    print(f"{'='*100}")
    print(f"Results saved to: {experiment_folder}")
    print(f"{'='*100}\n")

    return all_results, experiment_folder


if __name__ == "__main__":
    # Configuration
    VIDEO_FOLDER = "Keyframe-extraction/Dataset/Videos"
    GROUND_TRUTH_FOLDER = "Keyframe-extraction/Dataset/Keyframe"
    OUTPUT_BASE_FOLDER = "experiment_results"

    # User model configurations
    # NOTE: User needs to modify yolo_osnet_4_with_filtering_updated (1).py
    # to accept 'tracker' parameter
    USER_MODEL_CONFIGS = [
        {
            'name': 'YOLO_BOTSORT',
            'model_type': 'yolo',
            'model_path': 'yolo11m.pt',
            'tracker': 'botsort.yaml',
            'frame_skip_interval': 1
        },
        {
            'name': 'YOLO_ByteTrack',
            'model_type': 'yolo',
            'model_path': 'yolo11m.pt',
            'tracker': 'bytetrack.yaml',
            'frame_skip_interval': 1
        },
        {
            'name': 'RTDETR_BOTSORT',
            'model_type': 'rt_detr',
            'model_path': 'rtdetr-l.pt',
            'tracker': 'botsort.yaml',
            'frame_skip_interval': 1
        },
        {
            'name': 'RTDETR_ByteTrack',
            'model_type': 'rt_detr',
            'model_path': 'rtdetr-l.pt',
            'tracker': 'bytetrack.yaml',
            'frame_skip_interval': 1
        },
    ]

    # Check if folders exist
    if not os.path.exists(VIDEO_FOLDER):
        print(f"✗ Video folder not found: {VIDEO_FOLDER}")
        sys.exit(1)

    if not os.path.exists(GROUND_TRUTH_FOLDER):
        print(f"✗ Ground truth folder not found: {GROUND_TRUTH_FOLDER}")
        sys.exit(1)

    # Run experiment
    print("\n" + "="*100)
    print("KEYFRAME EXTRACTION EXPERIMENT")
    print("="*100 + "\n")

    results, output_folder = run_full_experiment(
        video_folder=VIDEO_FOLDER,
        ground_truth_folder=GROUND_TRUTH_FOLDER,
        output_base_folder=OUTPUT_BASE_FOLDER,
        run_baselines=True,
        run_user_model=True,
        baseline_intervals=[15, 30, 60],
        user_model_configs=USER_MODEL_CONFIGS,
        tolerances=[0, 15, 30]
    )

    print("\n✓ All experiments completed successfully!")
    print(f"✓ Check results in: {output_folder}")
