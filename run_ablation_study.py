"""
Ablation Study for Keyframe Extraction

3가지 구성으로 각 컴포넌트의 효과를 평가:
1. Full Model: YOLO + ByteTrack + Profile Tracking (Pre + Post)
2. No Profile: YOLO + ByteTrack (Profile tracking 제거)
3. Profile Only: YOLO + Profile Tracking (Tracking 제거)
"""

import os
import sys
from datetime import datetime
from typing import List, Dict

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


# Ablation Study Configurations
ABLATION_CONFIGS = [
    {
        'name': 'Full_Model',
        'description': 'YOLO+ByteTrack+Profile (Complete)',
        'model_type': 'yolo',
        'model_path': 'yolo11m.pt',
        'tracker': 'bytetrack.yaml',
        'profile_only': False,
        'profile_iterations': 3,
        'apply_post_filter': True,
    },
    {
        'name': 'No_Profile',
        'description': 'YOLO+ByteTrack (No Profile Tracking)',
        'model_type': 'yolo',
        'model_path': 'yolo11m.pt',
        'tracker': 'bytetrack.yaml',
        'profile_only': False,
        'profile_iterations': 0,  # ← Profile tracking 끄기
        'apply_post_filter': False,  # ← Post-filter도 끄기
    },
    {
        'name': 'Profile_Only',
        'description': 'YOLO+Profile (No Tracking)',
        'model_type': 'yolo',
        'model_path': 'yolo11m.pt',
        'profile_only': True,  # ← Profile-only 모드
        # tracker는 사용 안 함
        # profile_iterations는 profile_only 스크립트 내부에서 설정
    }
]


def run_ablation_study(video_path: str,
                       ground_truth_folder: str,
                       output_base_folder: str,
                       run_baselines: bool = True,
                       baseline_intervals: List[int] = [15],  # Ablation study용: 대표 baseline만
                       tolerances: List[int] = [0, 15, 30]):
    """
    Run ablation study experiment.

    Args:
        video_path: Path to video file
        ground_truth_folder: Base folder containing ground truth keyframes
        output_base_folder: Base folder for all outputs
        run_baselines: Whether to run baseline methods
        baseline_intervals: List of sampling intervals for baselines
        tolerances: List of temporal tolerances for evaluation
    """
    # Validate inputs
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Extract video name
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create timestamp for unique folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = os.path.join(
        output_base_folder,
        f"ablation_study_{video_name}_{timestamp}"
    )
    os.makedirs(experiment_folder, exist_ok=True)

    print(f"\n{'='*100}")
    print(f"ABLATION STUDY: {video_name}")
    print(f"{'='*100}")
    print(f"Video: {video_path}")
    print(f"Output: {experiment_folder}")
    print(f"Configurations: {len(ABLATION_CONFIGS)}")
    print(f"{'='*100}\n")

    # Get total frame count
    total_frames = get_total_frame_count(video_path)

    # Load ground truth
    try:
        gt_keyframes = load_ground_truth_keyframes(ground_truth_folder, video_name)
        print(f"✓ Loaded ground truth: {len(gt_keyframes)} keyframes")
    except FileNotFoundError as e:
        print(f"⚠️ Warning: {e}")
        print(f"   Continuing without ground truth evaluation")
        gt_keyframes = None

    # Store all results
    all_results = []

    # ====================================================================
    # BASELINE METHODS
    # ====================================================================
    if run_baselines and gt_keyframes is not None:
        print(f"\n{'='*100}")
        print("BASELINE METHODS")
        print(f"{'='*100}\n")

        for interval in baseline_intervals:
            method_name = f"Uniform-{interval}"
            print(f"Running {method_name}...")

            baseline_keyframes = extract_baseline_keyframes(
                video_path, interval=interval
            )

            # Evaluate with different tolerances
            for tol in tolerances:
                metrics = evaluate_method(baseline_keyframes, gt_keyframes, tol)

                result = {
                    'video': video_name,
                    'method': method_name,
                    'tolerance': tol,
                    'num_keyframes': len(baseline_keyframes),
                    'total_frames': total_frames,
                    **metrics
                }
                all_results.append(result)

            print(f"  ✓ {method_name}: {len(baseline_keyframes)} keyframes\n")

    # ====================================================================
    # ABLATION STUDY CONFIGURATIONS
    # ====================================================================
    print(f"\n{'='*100}")
    print("ABLATION STUDY - MODEL CONFIGURATIONS")
    print(f"{'='*100}\n")

    # Create model results folder
    model_results_folder = os.path.join(experiment_folder, "model_results", video_name)
    os.makedirs(model_results_folder, exist_ok=True)

    # Run all ablation configurations
    model_keyframes = run_multiple_configurations(
        video_path=video_path,
        output_base_folder=model_results_folder,
        configurations=ABLATION_CONFIGS
    )

    # Evaluate each configuration
    if gt_keyframes is not None:
        for config in ABLATION_CONFIGS:
            config_name = config['name']
            keyframes = model_keyframes.get(config_name, [])

            if not keyframes:
                print(f"⚠️ No keyframes from {config_name}, skipping evaluation")
                continue

            print(f"\nEvaluating {config_name}...")
            print(f"  Description: {config['description']}")
            print(f"  Keyframes: {len(keyframes)}")

            # Evaluate with different tolerances
            for tol in tolerances:
                metrics = evaluate_method(keyframes, gt_keyframes, tol)

                result = {
                    'video': video_name,
                    'method': config_name,
                    'tolerance': tol,
                    'num_keyframes': len(keyframes),
                    'total_frames': total_frames,
                    **metrics
                }
                all_results.append(result)

    # ====================================================================
    # SAVE RESULTS
    # ====================================================================
    print(f"\n{'='*100}")
    print("SAVING RESULTS")
    print(f"{'='*100}\n")

    # Save evaluation results to CSV
    evaluation_folder = os.path.join(experiment_folder, "evaluation")
    os.makedirs(evaluation_folder, exist_ok=True)

    results_csv = save_results_to_csv(all_results, evaluation_folder)
    print(f"✓ Results saved to: {results_csv}")

    # Print summary
    if all_results:
        print_evaluation_summary(all_results)

    print(f"\n{'='*100}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*100}")
    print(f"Results location: {experiment_folder}")
    print(f"  • Model outputs: {model_results_folder}")
    print(f"  • Evaluation: {evaluation_folder}")
    print(f"{'='*100}\n")

    return experiment_folder


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Ablation Study for Keyframe Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ablation Study Configurations:
  1. Full_Model    : YOLO + ByteTrack + Profile (Pre + Post)
  2. No_Profile    : YOLO + ByteTrack (No Profile Tracking)
  3. Profile_Only  : YOLO + Profile (No Tracking)

Example:
  python run_ablation_study.py --video my_video.mp4
  python run_ablation_study.py --video my_video.mp4 --no-baselines
        """
    )

    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--gt-folder', type=str,
                       default='Keyframe-extraction/Dataset/Keyframe',
                       help='Ground truth folder (default: Keyframe-extraction/Dataset/Keyframe)')
    parser.add_argument('--output', type=str,
                       default='experiment_results',
                       help='Output base folder (default: experiment_results)')
    parser.add_argument('--no-baselines', action='store_true',
                       help='Skip baseline methods')
    parser.add_argument('--baseline-intervals', type=int, nargs='+',
                       default=[15],
                       help='Baseline sampling intervals (default: 15, for ablation study)')
    parser.add_argument('--tolerances', type=int, nargs='+',
                       default=[0, 15, 30],
                       help='Temporal tolerances for evaluation (default: 0 15 30)')

    args = parser.parse_args()

    # Run ablation study
    run_ablation_study(
        video_path=args.video,
        ground_truth_folder=args.gt_folder,
        output_base_folder=args.output,
        run_baselines=not args.no_baselines,
        baseline_intervals=args.baseline_intervals,
        tolerances=args.tolerances
    )


if __name__ == "__main__":
    main()
