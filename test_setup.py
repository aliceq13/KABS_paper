"""
Test Setup Script

This script checks if all requirements are met before running the experiment.
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    if os.path.exists(filepath):
        print(f"✓ {description}: Found")
        return True
    else:
        print(f"✗ {description}: NOT FOUND")
        print(f"  Expected path: {filepath}")
        return False

def check_directory_exists(dirpath, description, check_contents=False):
    """Check if a directory exists and print status."""
    if os.path.exists(dirpath):
        if check_contents:
            contents = os.listdir(dirpath)
            print(f"✓ {description}: Found ({len(contents)} items)")
        else:
            print(f"✓ {description}: Found")
        return True
    else:
        print(f"✗ {description}: NOT FOUND")
        print(f"  Expected path: {dirpath}")
        return False

def check_python_package(package_name):
    """Check if a Python package is installed."""
    try:
        __import__(package_name)
        print(f"✓ {package_name}: Installed")
        return True
    except ImportError:
        print(f"✗ {package_name}: NOT INSTALLED")
        print(f"  Install with: pip install {package_name}")
        return False

def main():
    print("="*80)
    print("KEYFRAME EXTRACTION EXPERIMENT - SETUP CHECK")
    print("="*80 + "\n")

    all_checks_passed = True

    # Check Python packages
    print("Checking Python Packages...")
    print("-"*80)
    packages = ['cv2', 'torch', 'ultralytics', 'torchreid', 'pandas', 'numpy', 'scipy']
    for package in packages:
        if package == 'cv2':
            if not check_python_package('cv2'):
                print("  (cv2 is from opencv-python)")
        else:
            check_passed = check_python_package(package)
            all_checks_passed = all_checks_passed and check_passed
    print()

    # Check code files
    print("Checking Code Files...")
    print("-"*80)
    code_files = [
        ('evaluation_metrics.py', 'Evaluation metrics module'),
        ('baseline_methods.py', 'Baseline methods module'),
        ('model_wrapper.py', 'Model wrapper module'),
        ('run_experiments.py', 'Main experiment runner'),
        ('yolo_osnet_4_with_filtering_updated (1).py', 'Your keyframe extraction model'),
    ]

    for filepath, description in code_files:
        check_passed = check_file_exists(filepath, description)
        all_checks_passed = all_checks_passed and check_passed
    print()

    # Check model weights
    print("Checking Model Weights...")
    print("-"*80)
    model_weights = [
        ('yolo11m.pt', 'YOLO model'),
        ('rtdetr-l.pt', 'RT-DETR model'),
        ('osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth', 'TorchReID OSNet model'),
    ]

    for filepath, description in model_weights:
        check_passed = check_file_exists(filepath, description)
        if not check_passed:
            print(f"  ⚠️  This may be downloaded automatically if not found")
        # Don't fail if model weights are missing (they can be auto-downloaded)
    print()

    # Check dataset
    print("Checking Dataset...")
    print("-"*80)
    video_folder = 'Keyframe-extraction/Dataset/Videos'
    gt_folder = 'Keyframe-extraction/Dataset/Keyframe'

    check_passed = check_directory_exists(video_folder, 'Video folder', check_contents=True)
    all_checks_passed = all_checks_passed and check_passed

    check_passed = check_directory_exists(gt_folder, 'Ground truth folder', check_contents=True)
    all_checks_passed = all_checks_passed and check_passed

    if os.path.exists(video_folder):
        videos = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
        print(f"  Found {len(videos)} video files")
        if len(videos) < 20:
            print(f"  ⚠️  Expected 20 videos, found {len(videos)}")

    if os.path.exists(gt_folder):
        gt_folders = [f for f in os.listdir(gt_folder) if os.path.isdir(os.path.join(gt_folder, f))]
        print(f"  Found {len(gt_folders)} ground truth folders")
        if len(gt_folders) < 20:
            print(f"  ⚠️  Expected 20 ground truth folders, found {len(gt_folders)}")
    print()

    # Summary
    print("="*80)
    if all_checks_passed:
        print("✓ ALL CHECKS PASSED!")
        print("="*80)
        print("\nYou can now run the experiment:")
        print("  python run_experiments.py")
    else:
        print("✗ SOME CHECKS FAILED!")
        print("="*80)
        print("\nPlease fix the issues above before running the experiment.")
        print("Refer to EXPERIMENT_README.md for detailed setup instructions.")
    print()

if __name__ == "__main__":
    main()
