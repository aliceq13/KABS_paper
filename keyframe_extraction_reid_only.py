"""
Re-ID Only Keyframe Extraction

YOLO Detection → Re-ID → Greedy Selection
(ByteTrack tracking 제외, Profile Tracking 제외)
"""

import os
import sys
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Tuple
from scipy.spatial.distance import cosine

# TorchReID
try:
    from torchreid import models, utils
    TORCHREID_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: torchreid library not found. Re-ID will be disabled.")
    TORCHREID_AVAILABLE = False


def load_torchreid_model(model_path=None):
    """Load TorchReID OSNet model"""
    if not TORCHREID_AVAILABLE:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading TorchReID OSNet model...")
    try:
        torchreid_model = models.build_model(
            name='osnet_x1_0',
            num_classes=751,
            loss='softmax',
            pretrained=True
        )

        if model_path and os.path.exists(model_path):
            torchreid_model.load_state_dict(torch.load(model_path, map_location=device))

        torchreid_model = torchreid_model.to(device)
        torchreid_model.eval()

        print(f"✓ TorchReID model loaded (device: {device})")
        return torchreid_model, device

    except Exception as e:
        print(f"✗ Failed to load TorchReID model: {e}")
        return None, None


def extract_reid_feature(crop_bgr, torchreid_model, device):
    """Extract Re-ID feature from object crop"""
    if torchreid_model is None:
        return None

    try:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(crop_rgb, (128, 256))
        crop_tensor = torch.from_numpy(crop_resized).permute(2, 0, 1).float() / 255.0
        crop_tensor = crop_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            reid_feature = torchreid_model(crop_tensor)
            reid_feature = reid_feature.cpu().numpy().flatten()

        return reid_feature

    except Exception as e:
        return None


def reid_merge_duplicates(frames_data, reid_model, device, reid_threshold=0.8):
    """Merge duplicate track IDs using Re-ID"""
    if not TORCHREID_AVAILABLE or reid_model is None:
        print("⚠️ Re-ID unavailable, skipping merge")
        return frames_data

    print(f"\n Re-ID Duplicate Merging...")
    print(f"  Threshold: {reid_threshold}")

    # Collect all track IDs and their Re-ID features
    track_features = {}  # track_id -> list of features

    for frame_data in frames_data:
        for obj in frame_data['objects']:
            track_id = obj['track_id']
            crop = obj['crop']

            reid_feature = extract_reid_feature(crop, reid_model, device)
            if reid_feature is not None:
                if track_id not in track_features:
                    track_features[track_id] = []
                track_features[track_id].append(reid_feature)

    # Calculate average features for each track
    track_avg_features = {}
    for track_id, features in track_features.items():
        if features:
            track_avg_features[track_id] = np.mean(features, axis=0)

    # Find duplicate tracks using Re-ID similarity
    track_ids = list(track_avg_features.keys())
    id_mapping = {}  # old_id -> new_id

    for i, id1 in enumerate(track_ids):
        if id1 in id_mapping:
            continue

        id_mapping[id1] = id1  # Map to itself by default

        for id2 in track_ids[i+1:]:
            if id2 in id_mapping:
                continue

            # Calculate cosine similarity
            similarity = 1 - cosine(track_avg_features[id1], track_avg_features[id2])

            if similarity >= reid_threshold:
                # Merge id2 into id1
                id_mapping[id2] = id1
                print(f"  Merging track {id2} -> {id1} (similarity: {similarity:.3f})")

    # Apply ID mapping
    for frame_data in frames_data:
        for obj in frame_data['objects']:
            old_id = obj['track_id']
            obj['track_id'] = id_mapping.get(old_id, old_id)
            obj['original_track_id'] = old_id

    print(f"  Merged {len(track_ids)} -> {len(set(id_mapping.values()))} unique tracks")

    return frames_data


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


def extract_keyframes_reid_only(
    video_path: str,
    output_folder: str,
    model_path: str = "yolo11m.pt",
    torchreid_model_path: str = None,
    reid_threshold: float = 0.8,
    conf_threshold: float = 0.25,
    save_frames: bool = True
) -> List[int]:
    """
    Extract keyframes using Re-ID only (no ByteTrack, no Profile Tracking)

    Pipeline: YOLO Detection → Re-ID → Greedy Selection
    """

    os.makedirs(output_folder, exist_ok=True)

    print(f"\n{'='*80}")
    print("Re-ID Only Keyframe Extraction")
    print(f"{'='*80}")

    # Load models
    print(f"Loading YOLO model: {model_path}")
    yolo_model = YOLO(model_path)

    reid_model, device = load_torchreid_model(torchreid_model_path)

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

    # Step 2: Re-ID Merge (NO Profile Tracking)
    print(f"\n{'='*80}")
    print("Step 2: Re-ID Duplicate Merging")
    print(f"{'='*80}")

    frames_data = reid_merge_duplicates(
        frames_data,
        reid_model,
        device,
        reid_threshold=reid_threshold
    )

    # Step 3: Greedy Selection
    print(f"\n{'='*80}")
    print("Step 3: Greedy Keyframe Selection")
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
        "method": "reid_only",
        "settings": {
            "reid_threshold": reid_threshold,
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
         torchreid_model_path: str = None,
         **kwargs):
    """Main function (callable from model_wrapper.py)"""
    if video_path is None:
        raise ValueError("video_path is required")

    if output_folder is None:
        output_folder = "reid_only_results"

    keyframes = extract_keyframes_reid_only(
        video_path=video_path,
        output_folder=output_folder,
        model_path=model_path,
        torchreid_model_path=torchreid_model_path
    )

    return keyframes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Re-ID Only Keyframe Extraction"
    )
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--output', type=str, default='reid_only_results')
    parser.add_argument('--model', type=str, default='yolo11m.pt')
    parser.add_argument('--reid-model', type=str, default=None)

    args = parser.parse_args()

    extract_keyframes_reid_only(
        video_path=args.video,
        output_folder=args.output,
        model_path=args.model,
        torchreid_model_path=args.reid_model
    )
