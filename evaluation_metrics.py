"""
Keyframe Selection Evaluation Metrics Module

This module provides evaluation metrics for keyframe selection:
- F1-Score with temporal tolerance
- Precision & Recall
- Compression Ratio
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Tuple
import csv
from collections import defaultdict


def load_ground_truth_keyframes(video_name: str, ground_truth_base_dir: str) -> List[int]:
    """
    Load ground truth keyframe indices from the dataset folder.

    Args:
        video_name: Video filename (e.g., "video1.mp4")
        ground_truth_base_dir: Base directory containing Keyframe folders

    Returns:
        List of ground truth frame indices (sorted)
    """
    # Extract video name without extension
    video_basename = os.path.splitext(video_name)[0]

    # Ground truth keyframe folder path
    gt_folder = os.path.join(ground_truth_base_dir, video_basename + ".mp4")

    if not os.path.exists(gt_folder):
        raise FileNotFoundError(f"Ground truth folder not found: {gt_folder}")

    # Get all keyframe filenames and extract frame indices
    keyframe_files = [f for f in os.listdir(gt_folder) if f.endswith('.jpg')]

    # Extract frame indices from filenames (e.g., "150.jpg" -> 150)
    frame_indices = []
    for filename in keyframe_files:
        if filename == 'result.jpg':
            continue  # Skip summary images
        try:
            frame_idx = int(os.path.splitext(filename)[0])
            frame_indices.append(frame_idx)
        except ValueError:
            print(f"⚠️ Warning: Could not parse frame index from '{filename}'")
            continue

    return sorted(frame_indices)


def calculate_f1_with_tolerance(predicted: List[int],
                                 ground_truth: List[int],
                                 tolerance: int = 0) -> Dict[str, float]:
    """
    Calculate Precision, Recall, and F1-Score with temporal tolerance.

    Args:
        predicted: List of predicted keyframe indices
        ground_truth: List of ground truth keyframe indices
        tolerance: Temporal tolerance in frames (±tolerance)

    Returns:
        Dictionary with precision, recall, f1_score, tp, fp, fn
    """
    if len(predicted) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'tp': 0,
            'fp': len(predicted),
            'fn': len(ground_truth)
        }

    if len(ground_truth) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'tp': 0,
            'fp': len(predicted),
            'fn': 0
        }

    # Convert to sets for faster lookup
    gt_set = set(ground_truth)
    pred_set = set(predicted)

    # True Positives: predicted frames that match with ground truth within tolerance
    tp = 0
    matched_gt = set()
    matched_pred = set()

    for pred_frame in pred_set:
        # Check if any ground truth frame is within tolerance
        for gt_frame in gt_set:
            if gt_frame in matched_gt:
                continue  # Already matched
            if abs(pred_frame - gt_frame) <= tolerance:
                tp += 1
                matched_gt.add(gt_frame)
                matched_pred.add(pred_frame)
                break

    # False Positives: predicted frames that don't match any ground truth
    fp = len(pred_set) - tp

    # False Negatives: ground truth frames that weren't matched
    fn = len(gt_set) - tp

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def calculate_compression_ratio(num_keyframes: int, total_frames: int) -> float:
    """
    Calculate compression ratio.

    Args:
        num_keyframes: Number of selected keyframes
        total_frames: Total number of frames in video

    Returns:
        Compression ratio (0.0 to 1.0)
    """
    if total_frames == 0:
        return 0.0
    return num_keyframes / total_frames


def get_total_frame_count(video_path: str) -> int:
    """
    Get total number of frames in a video.

    Args:
        video_path: Path to video file

    Returns:
        Total frame count
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return total_frames


def evaluate_method(predicted_frames: List[int],
                   ground_truth_frames: List[int],
                   total_frames: int,
                   method_name: str,
                   video_name: str,
                   tolerances: List[int] = [0, 15, 30]) -> Dict:
    """
    Evaluate a keyframe selection method with multiple tolerance levels.

    Args:
        predicted_frames: List of predicted keyframe indices
        ground_truth_frames: List of ground truth keyframe indices
        total_frames: Total number of frames in video
        method_name: Name of the method being evaluated
        video_name: Name of the video
        tolerances: List of temporal tolerances to evaluate

    Returns:
        Dictionary with all evaluation metrics
    """
    results = {
        'method': method_name,
        'video_name': video_name,
        'num_keyframes': len(predicted_frames),
        'num_gt_keyframes': len(ground_truth_frames),
        'total_frames': total_frames,
        'compression_ratio': calculate_compression_ratio(len(predicted_frames), total_frames)
    }

    # Evaluate with different tolerances
    for tolerance in tolerances:
        metrics = calculate_f1_with_tolerance(predicted_frames, ground_truth_frames, tolerance)

        suffix = f'_tol{tolerance}'
        results[f'precision{suffix}'] = metrics['precision']
        results[f'recall{suffix}'] = metrics['recall']
        results[f'f1_score{suffix}'] = metrics['f1_score']
        results[f'tp{suffix}'] = metrics['tp']
        results[f'fp{suffix}'] = metrics['fp']
        results[f'fn{suffix}'] = metrics['fn']

    return results


def save_results_to_csv(results_list: List[Dict], output_path: str):
    """
    Save evaluation results to CSV file.

    Args:
        results_list: List of result dictionaries
        output_path: Output CSV file path
    """
    if not results_list:
        print("⚠️ No results to save")
        return

    # Get all field names from first result
    fieldnames = list(results_list[0].keys())

    # Write to CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_list)

    print(f"✓ Results saved to: {output_path}")


def aggregate_results(results_list: List[Dict], group_by: str = 'method') -> Dict:
    """
    Aggregate results by method or video.

    Args:
        results_list: List of result dictionaries
        group_by: 'method' or 'video_name'

    Returns:
        Dictionary with aggregated statistics
    """
    grouped = defaultdict(list)

    for result in results_list:
        key = result[group_by]
        grouped[key].append(result)

    aggregated = {}
    for key, group in grouped.items():
        agg_stats = {group_by: key}

        # Calculate mean for numeric fields
        numeric_fields = [k for k in group[0].keys()
                         if k not in [group_by, 'video_name', 'method']]

        for field in numeric_fields:
            values = [r[field] for r in group]
            agg_stats[f'{field}_mean'] = np.mean(values)
            agg_stats[f'{field}_std'] = np.std(values)

        agg_stats['num_videos'] = len(group)
        aggregated[key] = agg_stats

    return aggregated


def print_evaluation_summary(results_list: List[Dict], tolerances: List[int] = [0, 15, 30]):
    """
    Print a formatted summary of evaluation results.

    Args:
        results_list: List of result dictionaries
        tolerances: List of tolerance levels used
    """
    print("\n" + "="*100)
    print("EVALUATION SUMMARY")
    print("="*100)

    # Aggregate by method
    method_stats = aggregate_results(results_list, group_by='method')

    for method, stats in method_stats.items():
        print(f"\n📊 Method: {method}")
        print(f"   Videos evaluated: {stats['num_videos']}")
        print(f"   Avg keyframes: {stats['num_keyframes_mean']:.1f} ± {stats['num_keyframes_std']:.1f}")
        print(f"   Avg compression ratio: {stats['compression_ratio_mean']:.4f} ± {stats['compression_ratio_std']:.4f}")
        print()

        for tolerance in tolerances:
            suffix = f'_tol{tolerance}'
            print(f"   Tolerance ±{tolerance} frames:")
            print(f"      Precision: {stats[f'precision{suffix}_mean']:.4f} ± {stats[f'precision{suffix}_std']:.4f}")
            print(f"      Recall:    {stats[f'recall{suffix}_mean']:.4f} ± {stats[f'recall{suffix}_std']:.4f}")
            print(f"      F1-Score:  {stats[f'f1_score{suffix}_mean']:.4f} ± {stats[f'f1_score{suffix}_std']:.4f}")

    print("\n" + "="*100 + "\n")


# Example usage
if __name__ == "__main__":
    # Test the evaluation metrics
    predicted = [100, 200, 300, 400]
    ground_truth = [105, 195, 305, 500]

    print("Test Evaluation:")
    print(f"Predicted: {predicted}")
    print(f"Ground Truth: {ground_truth}")
    print()

    for tolerance in [0, 15, 30]:
        metrics = calculate_f1_with_tolerance(predicted, ground_truth, tolerance)
        print(f"Tolerance ±{tolerance}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
        print()
