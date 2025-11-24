"""
Re-ID Model Comparison Script
Compares different Re-ID models on the same video
"""

import os
import sys
import time
import json
from pathlib import Path

# Import the main pipeline
from yolo_osnet_4_dual_reid import main as pipeline_main


# ============================================================================
# Model Configurations to Test
# ============================================================================

PERSON_REID_MODELS_TO_TEST = [
    {"key": "fastreid_sbs_r50", "name": "FastReID SBS-R50 (SOTA)"},
    {"key": "osnet_market1501", "name": "OSNet Market-1501 (Best)"},
    {"key": "osnet_x1_0", "name": "OSNet x1.0 Pretrained"},
    {"key": "osnet_x0_75", "name": "OSNet x0.75 (Faster)"},
    {"key": "resnet50_fc512", "name": "ResNet50 Baseline"},
]

VEHICLE_REID_MODELS_TO_TEST = [
    {"key": "fastreid_veri_sbs_r50", "name": "FastReID VeRi SBS-R50 (SOTA)"},
    {"key": "fastreid_vehicleid_bot_r50", "name": "FastReID VehicleID BoT-R50 (Large Scale)"},
    {"key": "osnet_x1_0", "name": "OSNet x1.0 Pretrained"},
    {"key": "resnet50_fc512", "name": "ResNet50 Baseline"},
    # {"key": "osnet_veri776", "name": "OSNet VeRi-776 (if available)"},  # Uncomment if you have this model
]


def compare_reid_models(
    video_path: str,
    output_base_folder: str = "reid_model_comparison",
    test_person_models: bool = True,
    test_vehicle_models: bool = False,
    base_config: dict = None
):
    """
    Compare different Re-ID models on the same video

    Args:
        video_path: Path to test video
        output_base_folder: Base folder for comparison results
        test_person_models: Test different person Re-ID models
        test_vehicle_models: Test different vehicle Re-ID models
        base_config: Base configuration (uses defaults if None)
    """
    os.makedirs(output_base_folder, exist_ok=True)

    # Default configuration
    if base_config is None:
        base_config = {
            "video_path": video_path,
            "json_save_mode": "unified",
            "create_comparison_videos": False,  # Disable for speed
            "model_type": "yolo",
            "model_path": "yolo11m.pt",
            "use_dual_reid": True,
            "use_depth": False,  # Disable for speed
            "frame_skip_interval": 5,  # Speed up
            "profile_iterations": 1,
            "apply_post_filter": False,  # Disable for speed

            # Re-ID settings
            "reid_threshold": 0.8,
            "min_reid_samples": 2,
            "frame_merge_threshold": 0.7,

            # Greedy settings
            "coverage_k": 2,
            "use_instance_id": True,
            "max_selected": None,
            "min_new_combos": 1,
            "score_thresh": 0.0,

            # Filtering
            "show_available_classes": False,
            "filter_mode": 3,
            "filter_classes": [],

            # Tracker
            "tracker": "botsort.yaml",

            # Profile tracking
            "window_size": 15,
            "hist_threshold": 0.3,
            "hist_weight_brightness": 0.5,
            "hist_weight_saturation": 0.5,
        }

    results = {}

    # Test Person Re-ID Models
    if test_person_models:
        print("\n" + "="*100)
        print("TESTING PERSON RE-ID MODELS")
        print("="*100)

        for model_config in PERSON_REID_MODELS_TO_TEST:
            model_key = model_config["key"]
            model_name = model_config["name"]

            print(f"\n{'='*100}")
            print(f"Testing: {model_name} (key: {model_key})")
            print(f"{'='*100}")

            output_folder = os.path.join(output_base_folder, f"person_{model_key}")
            config = base_config.copy()
            config.update({
                "output_folder": output_folder,
                "person_reid_model_key": model_key,
                "vehicle_reid_model_key": "osnet_x1_0",  # Keep vehicle model constant
            })

            try:
                start_time = time.time()
                pipeline_main(**config)
                end_time = time.time()
                runtime = end_time - start_time

                # Load results
                json_path = os.path.join(output_folder, "keyframe_summary_unified.json")
                with open(json_path, 'r', encoding='utf-8') as f:
                    summary = json.load(f)

                num_keyframes = summary.get("num_keyframes", 0)

                results[f"person_{model_key}"] = {
                    "model_name": model_name,
                    "model_key": model_key,
                    "type": "person",
                    "num_keyframes": num_keyframes,
                    "runtime_seconds": runtime,
                    "output_folder": output_folder,
                    "success": True
                }

                print(f"\n✓ {model_name}: {num_keyframes} keyframes in {runtime:.2f}s")

            except Exception as e:
                print(f"\n✗ Error with {model_name}: {e}")
                results[f"person_{model_key}"] = {
                    "model_name": model_name,
                    "model_key": model_key,
                    "type": "person",
                    "error": str(e),
                    "success": False
                }

    # Test Vehicle Re-ID Models
    if test_vehicle_models:
        print("\n" + "="*100)
        print("TESTING VEHICLE RE-ID MODELS")
        print("="*100)

        for model_config in VEHICLE_REID_MODELS_TO_TEST:
            model_key = model_config["key"]
            model_name = model_config["name"]

            print(f"\n{'='*100}")
            print(f"Testing: {model_name} (key: {model_key})")
            print(f"{'='*100}")

            output_folder = os.path.join(output_base_folder, f"vehicle_{model_key}")
            config = base_config.copy()
            config.update({
                "output_folder": output_folder,
                "person_reid_model_key": "osnet_market1501",  # Keep person model constant
                "vehicle_reid_model_key": model_key,
            })

            try:
                start_time = time.time()
                pipeline_main(**config)
                end_time = time.time()
                runtime = end_time - start_time

                # Load results
                json_path = os.path.join(output_folder, "keyframe_summary_unified.json")
                with open(json_path, 'r', encoding='utf-8') as f:
                    summary = json.load(f)

                num_keyframes = summary.get("num_keyframes", 0)

                results[f"vehicle_{model_key}"] = {
                    "model_name": model_name,
                    "model_key": model_key,
                    "type": "vehicle",
                    "num_keyframes": num_keyframes,
                    "runtime_seconds": runtime,
                    "output_folder": output_folder,
                    "success": True
                }

                print(f"\n✓ {model_name}: {num_keyframes} keyframes in {runtime:.2f}s")

            except Exception as e:
                print(f"\n✗ Error with {model_name}: {e}")
                results[f"vehicle_{model_key}"] = {
                    "model_name": model_name,
                    "model_key": model_key,
                    "type": "vehicle",
                    "error": str(e),
                    "success": False
                }

    # Save comparison results
    comparison_results_path = os.path.join(output_base_folder, "comparison_results.json")
    with open(comparison_results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "="*100)
    print("COMPARISON SUMMARY")
    print("="*100)

    for model_id, result in results.items():
        if result.get("success"):
            print(f"\n✓ {result['model_name']} ({result['type']})")
            print(f"  Keyframes: {result['num_keyframes']}")
            print(f"  Runtime: {result['runtime_seconds']:.2f}s")
            print(f"  Output: {result['output_folder']}")
        else:
            print(f"\n✗ {result['model_name']} ({result['type']})")
            print(f"  Error: {result.get('error', 'Unknown error')}")

    print(f"\n{'='*100}")
    print(f"Comparison results saved to: {comparison_results_path}")
    print(f"{'='*100}\n")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare different Re-ID models")
    parser.add_argument("--video", type=str, required=True, help="Path to test video")
    parser.add_argument("--output", type=str, default="reid_model_comparison", help="Output folder")
    parser.add_argument("--test-person", action="store_true", help="Test person Re-ID models")
    parser.add_argument("--test-vehicle", action="store_true", help="Test vehicle Re-ID models")
    parser.add_argument("--test-all", action="store_true", help="Test both person and vehicle models")

    args = parser.parse_args()

    # If neither flag is set, test person models by default
    test_person = args.test_person or args.test_all or (not args.test_vehicle)
    test_vehicle = args.test_vehicle or args.test_all

    print("\n" + "="*100)
    print("RE-ID MODEL COMPARISON")
    print("="*100)
    print(f"Video: {args.video}")
    print(f"Output: {args.output}")
    print(f"Test Person Models: {test_person}")
    print(f"Test Vehicle Models: {test_vehicle}")
    print("="*100 + "\n")

    results = compare_reid_models(
        video_path=args.video,
        output_base_folder=args.output,
        test_person_models=test_person,
        test_vehicle_models=test_vehicle
    )

    print("\n✓ Comparison complete!")
