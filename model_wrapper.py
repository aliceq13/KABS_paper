"""
Wrapper for user's keyframe extraction model

This module provides a simplified interface to run the user's
yolo_osnet_4_with_filtering_updated.py with different configurations.
"""

import os
import sys
import json
import cv2
from typing import List, Dict
import importlib.util


def load_user_model_module(model_path: str):
    """
    Dynamically load the user's model module.

    Args:
        model_path: Path to the Python file

    Returns:
        Loaded module
    """
    spec = importlib.util.spec_from_file_location("user_model", model_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_model"] = module
    spec.loader.exec_module(module)
    return module


def extract_keyframes_from_json(json_path: str) -> List[int]:
    """
    Extract keyframe indices from the unified JSON output.

    Args:
        json_path: Path to keyframe_summary_unified.json

    Returns:
        List of keyframe indices
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract frame indices from the JSON
    keyframe_indices = [frame['frame_index'] for frame in data['frames']]

    return sorted(keyframe_indices)


def run_user_model(video_path: str,
                  output_folder: str,
                  model_type: str = "yolo",
                  model_path: str = "yolo11m.pt",
                  tracker: str = "botsort.yaml",
                  frame_skip_interval: int = 1,
                  profile_only: bool = False,
                  profile_iterations: int = 3,
                  apply_post_filter: bool = True) -> List[int]:
    """
    Run the user's keyframe extraction model with specified configuration.

    Args:
        video_path: Path to video file
        output_folder: Output folder for results
        model_type: "yolo" or "rt_detr"
        model_path: Path to YOLO/RT-DETR model weights
        tracker: Tracker configuration ("botsort.yaml" or "bytetrack.yaml")
        frame_skip_interval: Frame skipping interval (1 = no skip)
        profile_only: If True, use profile-only mode (no tracking)
        profile_iterations: Profile tracking iterations (0 = disable)
        apply_post_filter: Apply post-greedy filtering

    Returns:
        List of selected keyframe indices
    """
    # Profile-only mode: Use simplified profile tracking without tracking
    if profile_only:
        profile_only_path = "keyframe_extraction_profile_only.py"

        if not os.path.exists(profile_only_path):
            raise FileNotFoundError(f"Profile-only script not found: {profile_only_path}")

        print(f"\n{'='*80}")
        print(f"Running PROFILE-ONLY mode (No Tracking)")
        print(f"  Model: {model_type.upper()}")
        print(f"  Output: {output_folder}")
        print(f"{'='*80}\n")

        # Import profile-only module
        profile_module = load_user_model_module(profile_only_path)

        # Run profile-only extraction
        try:
            profile_module.main(
                video_path=video_path,
                output_folder=output_folder,
                model_path=model_path,
                hist_threshold=0.3,
                hist_weight_brightness=0.5,
                hist_weight_saturation=0.5
            )
        except Exception as e:
            print(f"✗ Error running profile-only mode: {e}")
            raise

        # Extract keyframe indices from JSON
        json_path = os.path.join(output_folder, "keyframe_summary_unified.json")
        keyframe_indices = extract_keyframes_from_json(json_path)

        print(f"✓ Extracted {len(keyframe_indices)} keyframes (Profile-Only)")

        return keyframe_indices

    # Standard mode: Use full model with tracking
    user_model_path = "yolo_osnet_4_with_filtering_updated (1).py"

    if not os.path.exists(user_model_path):
        raise FileNotFoundError(f"User model not found: {user_model_path}")

    # Import the main function
    user_module = load_user_model_module(user_model_path)

    # Configure the model
    config = {
        "video_path": video_path,
        "output_folder": output_folder,
        "json_save_mode": "unified",  # Only need unified JSON for evaluation
        "create_comparison_videos": False,  # Skip video creation for speed

        # Model configuration
        "model_type": model_type,
        "model_path": model_path,

        # TorchReID
        "torchreid_model_path": "osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth",

        # Depth estimation (disabled for speed)
        "use_depth": False,
        "save_depth_visualization": False,
        "save_foreground_masks": False,

        # Frame skipping
        "frame_skip_interval": frame_skip_interval,

        # Profile tracking (configurable)
        "profile_iterations": profile_iterations,
        "window_size": 15,
        "hist_threshold": 0.3,
        "hist_weight_brightness": 0.5,
        "hist_weight_saturation": 0.5,

        # Post-greedy filtering (configurable)
        "apply_post_filter": apply_post_filter,
        "post_hist_threshold": 0.25,
        "post_window_size": 7,
        "post_profile_iterations": 1,

        # Re-ID
        "reid_threshold": 0.8,
        "min_reid_samples": 2,
        "frame_merge_threshold": 0.7,

        # Greedy selection
        "coverage_k": 2,
        "use_instance_id": True,
        "max_selected": None,
        "min_new_combos": 1,
        "score_thresh": 0.0,

        # Object filtering
        "show_available_classes": False,
        "filter_mode": 3,  # No filtering
        "filter_classes": [],

        # Tracker configuration
        "tracker": tracker,
    }

    print(f"\n{'='*80}")
    print(f"Running user model:")
    print(f"  Model: {model_type.upper()}")
    print(f"  Tracker: {tracker}")
    print(f"  Frame skip: {frame_skip_interval}")
    print(f"  Output: {output_folder}")
    print(f"{'='*80}\n")

    # Run the model
    try:
        user_module.main(**config)
    except Exception as e:
        print(f"✗ Error running user model: {e}")
        raise

    # Extract keyframe indices from JSON
    json_path = os.path.join(output_folder, "keyframe_summary_unified.json")
    keyframe_indices = extract_keyframes_from_json(json_path)

    print(f"✓ Extracted {len(keyframe_indices)} keyframes")

    return keyframe_indices


def run_multiple_configurations(video_path: str,
                                output_base_folder: str,
                                configurations: List[Dict]) -> Dict[str, List[int]]:
    """
    Run the user's model with multiple configurations.

    Args:
        video_path: Path to video file
        output_base_folder: Base output folder
        configurations: List of configuration dictionaries

    Returns:
        Dictionary mapping configuration name to keyframe indices
    """
    results = {}

    for config in configurations:
        config_name = config['name']
        print(f"\n{'='*100}")
        print(f"CONFIGURATION: {config_name}")
        print(f"{'='*100}")

        output_folder = os.path.join(output_base_folder, config_name)

        try:
            keyframe_indices = run_user_model(
                video_path=video_path,
                output_folder=output_folder,
                model_type=config.get('model_type', 'yolo'),
                model_path=config.get('model_path', 'yolo11m.pt'),
                tracker=config.get('tracker', 'botsort.yaml'),
                frame_skip_interval=config.get('frame_skip_interval', 1),
                profile_only=config.get('profile_only', False),
                profile_iterations=config.get('profile_iterations', 3),
                apply_post_filter=config.get('apply_post_filter', True)
            )

            results[config_name] = keyframe_indices

        except Exception as e:
            print(f"✗ Configuration '{config_name}' failed: {e}")
            results[config_name] = []

    return results


# Example usage
if __name__ == "__main__":
    # Test configurations
    test_video = "test_video.mp4"

    if not os.path.exists(test_video):
        print(f"⚠️ Test video not found: {test_video}")
        print("Please provide a valid video path for testing.")
    else:
        configurations = [
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
        ]

        results = run_multiple_configurations(
            video_path=test_video,
            output_base_folder="model_results",
            configurations=configurations
        )

        print("\n" + "="*100)
        print("MODEL EXTRACTION COMPLETE")
        print("="*100)
        for config_name, indices in results.items():
            print(f"{config_name}: {len(indices)} keyframes")
