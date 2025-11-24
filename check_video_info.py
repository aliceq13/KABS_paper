"""
Video Information Checker

Display information about videos including frame count, FPS, resolution, and duration.

Usage:
    python check_video_info.py
    python check_video_info.py --folder custom_folder
    python check_video_info.py --video single_video.mp4
"""

import cv2
import os
import argparse
from typing import Dict, List
import pandas as pd


def get_video_info(video_path: str) -> Dict:
    """
    Get detailed information about a video file.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {
            'filename': os.path.basename(video_path),
            'path': video_path,
            'error': 'Cannot open video',
            'total_frames': 0,
            'fps': 0,
            'duration_sec': 0,
            'width': 0,
            'height': 0
        }

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0

    # Get file size
    file_size_bytes = os.path.getsize(video_path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    cap.release()

    return {
        'filename': os.path.basename(video_path),
        'path': video_path,
        'total_frames': total_frames,
        'fps': round(fps, 2),
        'duration_sec': round(duration_sec, 2),
        'duration_min': round(duration_sec / 60, 2),
        'width': width,
        'height': height,
        'resolution': f"{width}x{height}",
        'file_size_mb': round(file_size_mb, 2)
    }


def check_folder_videos(folder_path: str) -> List[Dict]:
    """
    Check all videos in a folder.

    Args:
        folder_path: Path to folder containing videos

    Returns:
        List of video information dictionaries
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
    video_files = []

    if not os.path.exists(folder_path):
        print(f"✗ Folder not found: {folder_path}")
        return []

    for filename in os.listdir(folder_path):
        if any(filename.endswith(ext) for ext in video_extensions):
            video_path = os.path.join(folder_path, filename)
            video_files.append(video_path)

    if not video_files:
        print(f"✗ No video files found in: {folder_path}")
        return []

    video_files.sort()

    print(f"\n{'='*100}")
    print(f"Checking {len(video_files)} videos in: {folder_path}")
    print(f"{'='*100}\n")

    results = []
    for i, video_path in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] Processing: {os.path.basename(video_path)}...", end=' ')
        info = get_video_info(video_path)

        if 'error' in info:
            print(f"✗ {info['error']}")
        else:
            print(f"✓ {info['total_frames']} frames, {info['fps']} fps, {info['duration_sec']}s")

        results.append(info)

    return results


def print_summary_table(results: List[Dict]):
    """Print a formatted summary table."""
    if not results:
        return

    print(f"\n{'='*100}")
    print("VIDEO INFORMATION SUMMARY")
    print(f"{'='*100}\n")

    # Create DataFrame for nice formatting
    df = pd.DataFrame(results)

    if 'error' in df.columns:
        # Remove error column for display
        valid_results = [r for r in results if 'error' not in r]
        if valid_results:
            df = pd.DataFrame(valid_results)
        else:
            print("✗ No valid videos found")
            return

    # Display columns
    display_cols = ['filename', 'total_frames', 'fps', 'duration_sec', 'resolution', 'file_size_mb']
    df_display = df[display_cols].copy()

    # Rename columns for clarity
    df_display.columns = ['Video Name', 'Total Frames', 'FPS', 'Duration (s)', 'Resolution', 'Size (MB)']

    print(df_display.to_string(index=False))

    # Print statistics
    print(f"\n{'='*100}")
    print("STATISTICS")
    print(f"{'='*100}")
    print(f"Total videos: {len(df)}")
    print(f"Total frames: {df['total_frames'].sum():,}")
    print(f"Average frames per video: {df['total_frames'].mean():.0f}")
    print(f"Min frames: {df['total_frames'].min():,} ({df.loc[df['total_frames'].idxmin(), 'filename']})")
    print(f"Max frames: {df['total_frames'].max():,} ({df.loc[df['total_frames'].idxmax(), 'filename']})")
    print(f"Average FPS: {df['fps'].mean():.2f}")
    print(f"Average duration: {df['duration_sec'].mean():.2f}s ({df['duration_sec'].mean()/60:.2f} min)")
    print(f"Total size: {df['file_size_mb'].sum():.2f} MB ({df['file_size_mb'].sum()/1024:.2f} GB)")
    print(f"{'='*100}\n")


def save_to_csv(results: List[Dict], output_path: str):
    """Save results to CSV file."""
    if not results:
        return

    valid_results = [r for r in results if 'error' not in r]
    if not valid_results:
        print("✗ No valid videos to save")
        return

    df = pd.DataFrame(valid_results)
    df.to_csv(output_path, index=False)
    print(f"✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Check video information (frames, FPS, duration, etc.)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check default dataset folder
  python check_video_info.py

  # Check custom folder
  python check_video_info.py --folder path/to/videos

  # Check single video
  python check_video_info.py --video my_video.mp4

  # Save results to CSV
  python check_video_info.py --output video_info.csv
        """
    )

    parser.add_argument('--folder', '-f', type=str,
                       default='Keyframe-extraction/Dataset/Videos',
                       help='Folder containing videos (default: Keyframe-extraction/Dataset/Videos)')
    parser.add_argument('--video', '-v', type=str,
                       help='Check single video file')
    parser.add_argument('--output', '-o', type=str,
                       help='Save results to CSV file')

    args = parser.parse_args()

    results = []

    if args.video:
        # Check single video
        if not os.path.exists(args.video):
            print(f"✗ Video file not found: {args.video}")
            return

        print(f"\n{'='*100}")
        print(f"Checking video: {args.video}")
        print(f"{'='*100}\n")

        info = get_video_info(args.video)

        if 'error' in info:
            print(f"✗ {info['error']}")
        else:
            print(f"Filename:      {info['filename']}")
            print(f"Total Frames:  {info['total_frames']:,}")
            print(f"FPS:           {info['fps']}")
            print(f"Duration:      {info['duration_sec']}s ({info['duration_min']} min)")
            print(f"Resolution:    {info['resolution']}")
            print(f"File Size:     {info['file_size_mb']} MB")

        results = [info]
    else:
        # Check folder
        results = check_folder_videos(args.folder)

        if results:
            print_summary_table(results)

    # Save to CSV if requested
    if args.output and results:
        save_to_csv(results, args.output)


if __name__ == "__main__":
    main()
