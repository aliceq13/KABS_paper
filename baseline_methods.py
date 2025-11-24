"""
Baseline Keyframe Extraction Methods

This module implements simple baseline methods for comparison:
- Uniform sampling (fixed frame interval)
"""

import cv2
import os
from typing import List, Tuple
import numpy as np


def uniform_sampling(video_path: str, interval: int) -> List[int]:
    """
    Extract keyframes using uniform sampling.

    Args:
        video_path: Path to video file
        interval: Frame interval (e.g., 30 means every 30th frame)

    Returns:
        List of selected keyframe indices
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Sample every 'interval' frames
    keyframe_indices = list(range(0, total_frames, interval))

    print(f"Uniform sampling (interval={interval}): Selected {len(keyframe_indices)} frames from {total_frames} total frames")

    return keyframe_indices


def save_baseline_keyframes(video_path: str,
                            keyframe_indices: List[int],
                            output_folder: str,
                            prefix: str = "baseline"):
    """
    Save baseline keyframes as images.

    Args:
        video_path: Path to video file
        keyframe_indices: List of frame indices to save
        output_folder: Output folder path
        prefix: Filename prefix
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    saved_count = 0
    for frame_idx in keyframe_indices:
        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            print(f"⚠️ Warning: Could not read frame {frame_idx}")
            continue

        # Save frame
        output_path = os.path.join(output_folder, f"{prefix}_frame_{frame_idx}.jpg")
        cv2.imwrite(output_path, frame)
        saved_count += 1

    cap.release()
    print(f"✓ Saved {saved_count} baseline keyframes to: {output_folder}")


def extract_baseline_keyframes(video_path: str,
                               output_base_folder: str,
                               intervals: List[int] = [30, 60],
                               save_images: bool = True) -> dict:
    """
    Extract keyframes using multiple baseline methods.

    Args:
        video_path: Path to video file
        output_base_folder: Base output folder
        intervals: List of sampling intervals
        save_images: Whether to save keyframe images

    Returns:
        Dictionary mapping method name to keyframe indices
    """
    results = {}

    video_name = os.path.basename(video_path)
    print(f"\n{'='*80}")
    print(f"Extracting baseline keyframes for: {video_name}")
    print(f"{'='*80}\n")

    for interval in intervals:
        method_name = f"Uniform_{interval}frames"
        print(f"Running: {method_name}")

        # Extract keyframes
        keyframe_indices = uniform_sampling(video_path, interval)
        results[method_name] = keyframe_indices

        # Save images if requested
        if save_images:
            output_folder = os.path.join(output_base_folder, method_name)
            save_baseline_keyframes(video_path, keyframe_indices, output_folder, prefix=method_name.lower())

    return results


# Example usage
if __name__ == "__main__":
    # Test baseline methods
    video_path = "test_video.mp4"  # Replace with actual video path
    output_folder = "baseline_results"

    # Check if video exists
    if not os.path.exists(video_path):
        print(f"⚠️ Test video not found: {video_path}")
        print("Please provide a valid video path for testing.")
    else:
        results = extract_baseline_keyframes(
            video_path=video_path,
            output_base_folder=output_folder,
            intervals=[30, 60],
            save_images=True
        )

        print("\n" + "="*80)
        print("BASELINE EXTRACTION COMPLETE")
        print("="*80)
        for method, indices in results.items():
            print(f"{method}: {len(indices)} keyframes")
