"""
Single Video Evaluation Script

Evaluate a single video with all methods and generate results.

Usage:
    python evaluate_single_video.py --video path/to/video.mp4
    python evaluate_single_video.py --video my_video.mp4 --gt-folder custom_gt_folder
"""

import os
import sys
import argparse
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


def evaluate_single_video(video_path: str,
                         ground_truth_folder: str,
                         output_base_folder: str,
                         run_baselines: bool = True,
                         run_user_model: bool = True,
                         baseline_intervals: List[int] = [30, 60],
                         user_model_configs: List[Dict] = None,
                         tolerances: List[int] = [0, 15, 30]):
    """
    Evaluate a single video with all methods.

    Args:
        video_path: Path to video file
        ground_truth_folder: Base folder containing ground truth keyframes
        output_base_folder: Base folder for all outputs
        run_baselines: Whether to run baseline methods
        run_user_model: Whether to run user's model
        baseline_intervals: List of sampling intervals for baselines
        user_model_configs: List of user model configurations
        tolerances: List of temporal tolerances for evaluation
    """
    # Validate video exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    video_name = os.path.basename(video_path)

    # Create output folders
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = os.path.join(output_base_folder, f"single_video_{timestamp}")
    os.makedirs(experiment_folder, exist_ok=True)

    baseline_output = os.path.join(experiment_folder, "baseline_results")
    model_output = os.path.join(experiment_folder, "model_results")
    evaluation_output = os.path.join(experiment_folder, "evaluation")

    os.makedirs(baseline_output, exist_ok=True)
    os.makedirs(model_output, exist_ok=True)
    os.makedirs(evaluation_output, exist_ok=True)

    print(f"\n{'='*100}")
    print(f"SINGLE VIDEO EVALUATION")
    print(f"{'='*100}")
    print(f"Video: {video_name}")
    print(f"Output folder: {experiment_folder}")
    print(f"{'='*100}\n")

    # Store all results
    all_results = []

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
        print(f"\n💡 Did you create ground truth for this video?")
        print(f"   Run: python create_ground_truth.py --video {video_path}")
        sys.exit(1)

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
            import traceback
            traceback.print_exc()

    # Save all results
    print(f"\n{'='*100}")
    print("SAVING RESULTS")
    print(f"{'='*100}\n")

    if not all_results:
        print("✗ No results to save")
        sys.exit(1)

    # Save detailed CSV
    detailed_csv_path = os.path.join(evaluation_output, "results.csv")
    save_results_to_csv(all_results, detailed_csv_path)

    # Create DataFrame for analysis
    df = pd.DataFrame(all_results)

    # Save summary statistics
    summary_stats = df.describe().round(4)
    summary_csv_path = os.path.join(evaluation_output, "summary_statistics.csv")
    summary_stats.to_csv(summary_csv_path)
    print(f"✓ Summary statistics saved to: {summary_csv_path}")

    # Print evaluation summary
    print_evaluation_summary(all_results, tolerances)

    # Save summary to text file
    summary_txt_path = os.path.join(evaluation_output, "summary.txt")
    with open(summary_txt_path, 'w') as f:
        f.write("="*100 + "\n")
        f.write("SINGLE VIDEO EVALUATION SUMMARY\n")
        f.write("="*100 + "\n\n")
        f.write(f"Video: {video_name}\n")
        f.write(f"Total frames: {total_frames}\n")
        f.write(f"Ground truth keyframes: {len(gt_keyframes)}\n")
        f.write(f"Methods evaluated: {len(all_results)}\n\n")
        f.write("="*100 + "\n")
        f.write("RESULTS BY METHOD\n")
        f.write("="*100 + "\n\n")

        for result in all_results:
            f.write(f"\nMethod: {result['method']}\n")
            f.write(f"  Keyframes extracted: {result['num_keyframes']}\n")
            f.write(f"  Compression ratio: {result['compression_ratio']*100:.2f}%\n")
            f.write(f"  F1-Score (tol=0):  {result['f1_score_tol0']:.4f}\n")
            f.write(f"  F1-Score (tol=15): {result['f1_score_tol15']:.4f}\n")
            f.write(f"  F1-Score (tol=30): {result['f1_score_tol30']:.4f}\n")
            f.write(f"  Precision (tol=15): {result['precision_tol15']:.4f}\n")
            f.write(f"  Recall (tol=15): {result['recall_tol15']:.4f}\n")

    print(f"✓ Summary saved to: {summary_txt_path}")

    print(f"\n{'='*100}")
    print(f"EVALUATION COMPLETE!")
    print(f"{'='*100}")
    print(f"Results saved to: {experiment_folder}")
    print(f"{'='*100}\n")

    return all_results, experiment_folder


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Evaluate a single video with all keyframe extraction methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with default GT folder
  python evaluate_single_video.py --video my_video.mp4

  # Evaluate with custom GT folder
  python evaluate_single_video.py --video my_video.mp4 --gt-folder custom_gt

  # Evaluate only baselines (skip model)
  python evaluate_single_video.py --video my_video.mp4 --no-model

  # Evaluate only model (skip baselines)
  python evaluate_single_video.py --video my_video.mp4 --no-baselines

Note: Make sure you created ground truth first using:
  python create_ground_truth.py --video my_video.mp4
        """
    )

    parser.add_argument('--video', '-v', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--gt-folder', '-g', type=str,
                       default='Keyframe-extraction/Dataset/Keyframe',
                       help='Ground truth folder (default: Keyframe-extraction/Dataset/Keyframe)')
    parser.add_argument('--output', '-o', type=str,
                       default='experiment_results',
                       help='Output base folder (default: experiment_results)')
    parser.add_argument('--no-baselines', action='store_true',
                       help='Skip baseline methods')
    parser.add_argument('--no-model', action='store_true',
                       help='Skip user model')
    parser.add_argument('--baseline-intervals', type=int, nargs='+',
                       default=[15, 30, 60],
                       help='Baseline sampling intervals (default: 15 30 60)')
    parser.add_argument('--tolerances', type=int, nargs='+',
                       default=[0, 15, 30],
                       help='Temporal tolerances for evaluation (default: 0 15 30)')

    args = parser.parse_args()

    # User model configurations
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

    # Check if video exists
    if not os.path.exists(args.video):
        print(f"✗ Error: Video file not found: {args.video}")
        sys.exit(1)

    # Run evaluation
    try:
        results, output_folder = evaluate_single_video(
            video_path=args.video,
            ground_truth_folder=args.gt_folder,
            output_base_folder=args.output,
            run_baselines=not args.no_baselines,
            run_user_model=not args.no_model,
            baseline_intervals=args.baseline_intervals,
            user_model_configs=USER_MODEL_CONFIGS,
            tolerances=args.tolerances
        )

        print("\n✓ Evaluation completed successfully!")
        print(f"✓ Check results in: {output_folder}")

    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
