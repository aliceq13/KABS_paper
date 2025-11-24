"""
Baseline Keyframe Extraction

YOLO Detection -> Greedy Selection
(No ByteTrack, No Profile Tracking, No Re-ID)

This is the simplest baseline: each detection gets a unique ID,
then greedy selection picks frames that cover the most objects.
"""

import os
import sys
import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Tuple


def greedy_keyframe_selection(frames_data):
    """Select keyframes using greedy coverage algorithm"""
    print(f"\n Greedy Keyframe Selection...")

    selected_frames = []
    covered_combos = set()

    # Build combinations for each frame
    frame_combos = []
    for frame_data in frames_data:
        track_ids = set(obj['track_id'] for obj in frame_data['objects'])
        # Create combinations of size 1 and 2
        combos = set()
        combos.update(track_ids)  # Individual objects
        # Pairs
        track_list = list(track_ids)
        for i in range(len(track_list)):
            for j in range(i+1, len(track_list)):
                combos.add(tuple(sorted([track_list[i], track_list[j]])))

        frame_combos.append({
            'frame_index': frame_data['frame_index'],
            'combos': combos
        })

    # Greedy selection
    while frame_combos:
        # Find frame with most new combinations
        best_frame = None
        best_new_count = 0
        best_new_combos = set()

        for fc in frame_combos:
            new_combos = fc['combos'] - covered_combos
            if len(new_combos) > best_new_count:
                best_new_count = len(new_combos)
                best_frame = fc
                best_new_combos = new_combos

        if best_new_count == 0:
            break  # No more new combinations

        # Select this frame
        selected_frames.append(best_frame['frame_index'])
        covered_combos.update(best_new_combos)
        frame_combos.remove(best_frame)

        print(f"  Selected frame {best_frame['frame_index']}: +{best_new_count} new combos (total: {len(covered_combos)})")

    print(f"  Total keyframes selected: {len(selected_frames)}")

    return sorted(selected_frames)


def extract_keyframes_baseline(
    video_path: str,
    output_folder: str,
    model_path: str = "yolo11m.pt",
    conf_threshold: float = 0.25,
    save_frames: bool = True
) -> List[int]:
    """
    Extract keyframes using baseline method (YOLO only)

    Pipeline: YOLO Detection -> Greedy Selection
    No ByteTrack, No Profile Tracking, No Re-ID
    """

    os.makedirs(output_folder, exist_ok=True)

    print(f"\n{'='*80}")
    print("Baseline Keyframe Extraction (YOLO Only)")
    print(f"{'='*80}")

    # Load YOLO model
    print(f"Loading YOLO model: {model_path}")
    yolo_model = YOLO(model_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\nVideo info:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")

    # Step 1: Detection (assign simple sequential IDs)
    print(f"\n{'='*80}")
    print("Step 1: YOLO Detection")
    print(f"{'='*80}")

    frames_data = []
    next_track_id = 1
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame, conf=conf_threshold, verbose=False)

        if len(results[0].boxes) > 0:
            objects = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                crop = frame[y1:y2, x1:x2]

                if crop.size > 0:
                    objects.append({
                        'track_id': next_track_id,
                        'crop': crop,
                        'box': [x1, y1, x2, y2],
                        'conf': float(box.conf[0]),
                        'class': int(box.cls[0])
                    })
                    next_track_id += 1

            if objects:
                frames_data.append({
                    'frame_index': frame_idx,
                    'frame': frame.copy(),
                    'objects': objects
                })

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames")

    cap.release()

    print(f"\n  Detected frames with objects: {len(frames_data)}")
    print(f"  Total unique object IDs: {next_track_id - 1}")

    # Step 2: Greedy Selection (No Re-ID, No Profile Tracking)
    print(f"\n{'='*80}")
    print("Step 2: Greedy Keyframe Selection")
    print(f"{'='*80}")

    keyframe_indices = greedy_keyframe_selection(frames_data)

    # Save results
    if save_frames:
        cap = cv2.VideoCapture(video_path)
        for idx in keyframe_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_filename = os.path.join(output_folder, f"keyframe_{idx:06d}.jpg")
                cv2.imwrite(frame_filename, frame)
        cap.release()

    # Save JSON
    summary = {
        "video_path": video_path,
        "total_frames": total_frames,
        "fps": fps,
        "method": "baseline",
        "settings": {
            "conf_threshold": conf_threshold
        },
        "num_keyframes": len(keyframe_indices),
        "keyframe_indices": keyframe_indices,
        "frames": [{"frame_index": idx} for idx in keyframe_indices]
    }

    json_path = os.path.join(output_folder, "keyframe_summary_unified.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print("Extraction Complete")
    print(f"{'='*80}")
    print(f"Total keyframes: {len(keyframe_indices)}")
    print(f"Results saved to: {output_folder}")

    return keyframe_indices


def main(video_path: str = None,
         output_folder: str = None,
         model_path: str = "yolo11m.pt",
         **kwargs):
    """Main function (callable from model_wrapper.py)"""
    if video_path is None:
        raise ValueError("video_path is required")

    if output_folder is None:
        output_folder = "baseline_results"

    keyframes = extract_keyframes_baseline(
        video_path=video_path,
        output_folder=output_folder,
        model_path=model_path
    )

    return keyframes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Baseline Keyframe Extraction (YOLO Only)"
    )
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--output', type=str, default='baseline_results')
    parser.add_argument('--model', type=str, default='yolo11m.pt')

    args = parser.parse_args()

    extract_keyframes_baseline(
        video_path=args.video,
        output_folder=args.output,
        model_path=args.model
    )
