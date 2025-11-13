"""
Visualization Tool for Profile Tracking Effectiveness

Creates figure showing histogram similarity for:
1. Similar frames (low Bhattacharyya distance)
2. Different frames (high Bhattacharyya distance)

Layout for each comparison:
┌─────────────┬──────────────┐
│   Image 1   │  Histogram   │
├─────────────┤  (V + Comb)  │
│   Image 2   │              │
└─────────────┴──────────────┘
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import os


def calculate_combined_histogram(frame: np.ndarray,
                                 weight_brightness: float = 0.5,
                                 weight_saturation: float = 0.5,
                                 bins: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate brightness, saturation, and combined histogram.

    Args:
        frame: BGR image
        weight_brightness: Weight for V channel (default: 0.5)
        weight_saturation: Weight for S channel (default: 0.5)
        bins: Number of histogram bins

    Returns:
        Tuple of (v_hist, s_hist, combined_hist)
    """
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Extract V (brightness) and S (saturation) channels
    v_channel = hsv[:, :, 2]  # V channel (brightness)
    s_channel = hsv[:, :, 1]  # S channel (saturation)

    # Calculate histograms
    v_hist = cv2.calcHist([v_channel], [0], None, [bins], [0, 256])
    s_hist = cv2.calcHist([s_channel], [0], None, [bins], [0, 256])

    # Normalize histograms
    v_hist = cv2.normalize(v_hist, v_hist).flatten()
    s_hist = cv2.normalize(s_hist, s_hist).flatten()

    # Weighted combination
    combined_hist = weight_brightness * v_hist + weight_saturation * s_hist
    combined_hist = combined_hist / np.sum(combined_hist)  # Re-normalize

    return v_hist, s_hist, combined_hist


def bhattacharyya_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Calculate Bhattacharyya distance between two histograms.
    Lower value = more similar.

    Args:
        hist1: First histogram
        hist2: Second histogram

    Returns:
        Bhattacharyya distance
    """
    bc = np.sum(np.sqrt(hist1 * hist2))
    return -np.log(bc) if bc > 0 else float('inf')


def visualize_histogram_comparison(img1_path: str,
                                   img2_path: str,
                                   output_path: str,
                                   threshold: float = 0.3,
                                   weight_brightness: float = 0.5,
                                   weight_saturation: float = 0.5):
    """
    Create visualization comparing two frames with their histograms.

    Layout:
    ┌─────────────┬──────────────┐
    │   Image 1   │  Histogram   │
    ├─────────────┤  (V + Comb)  │
    │   Image 2   │              │
    └─────────────┴──────────────┘

    Args:
        img1_path: Path to first image
        img2_path: Path to second image
        output_path: Path to save output figure
        threshold: Bhattacharyya distance threshold (default: 0.3)
        weight_brightness: Weight for V channel
        weight_saturation: Weight for S channel
    """
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Cannot load images: {img1_path}, {img2_path}")

    # Extract frame numbers from filenames (e.g., "frame_0123.jpg" -> "Frame 123")
    import re
    frame1_match = re.search(r'(\d+)', os.path.basename(img1_path))
    frame2_match = re.search(r'(\d+)', os.path.basename(img2_path))
    frame1_label = f"Frame {int(frame1_match.group(1))}" if frame1_match else "Frame 1"
    frame2_label = f"Frame {int(frame2_match.group(1))}" if frame2_match else "Frame 2"

    # Convert BGR to RGB for matplotlib
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Calculate histograms
    v_hist1, s_hist1, combined_hist1 = calculate_combined_histogram(
        img1, weight_brightness, weight_saturation
    )
    v_hist2, s_hist2, combined_hist2 = calculate_combined_histogram(
        img2, weight_brightness, weight_saturation
    )

    # Calculate Bhattacharyya distances
    v_distance = bhattacharyya_distance(v_hist1, v_hist2)
    combined_distance = bhattacharyya_distance(combined_hist1, combined_hist2)

    # Determine if similar or different based on threshold
    is_similar = combined_distance < threshold

    # Create figure with custom layout
    fig = plt.figure(figsize=(12, 8))

    # Define grid: 2 rows (for images) x 2 columns (image column, histogram column)
    # Images on left take 40% width, histogram on right takes 60%
    gs = fig.add_gridspec(2, 2, width_ratios=[0.4, 0.6], height_ratios=[1, 1],
                         hspace=0.05, wspace=0.2)

    # Image 1 (top-left)
    ax_img1 = fig.add_subplot(gs[0, 0])
    ax_img1.imshow(img1_rgb)
    ax_img1.axis('off')
    ax_img1.set_title(frame1_label, fontsize=11, fontweight='normal')

    # Image 2 (bottom-left)
    ax_img2 = fig.add_subplot(gs[1, 0])
    ax_img2.imshow(img2_rgb)
    ax_img2.axis('off')
    ax_img2.set_title(frame2_label, fontsize=11, fontweight='normal')

    # Histogram (spans both rows on right)
    ax_hist = fig.add_subplot(gs[:, 1])

    # Prepare histogram data
    bins = len(v_hist1)
    x = np.arange(bins)
    bar_width = 0.8

    # Plot V (brightness) histograms
    bar1 = ax_hist.bar(x - bar_width/4, v_hist1, width=bar_width/2,
                       alpha=0.5, color='skyblue', edgecolor='blue')
    bar2 = ax_hist.bar(x + bar_width/4, v_hist2, width=bar_width/2,
                       alpha=0.5, color='lightcoral', edgecolor='red')

    # Plot combined histograms as lines
    line1, = ax_hist.plot(x, combined_hist1, linewidth=2.5,
                         color='darkblue', marker='o', markersize=3)
    line2, = ax_hist.plot(x, combined_hist2, linewidth=2.5,
                         color='darkred', marker='s', markersize=3)

    # Styling
    ax_hist.set_xlabel('Histogram Bins', fontsize=10, fontweight='normal')
    ax_hist.set_ylabel('Normalized Frequency', fontsize=10, fontweight='normal')
    ax_hist.set_title('Histogram Comparison',
                     fontsize=11, fontweight='normal')

    # Create clean legend - only histogram type and colors
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='skyblue', edgecolor='blue', alpha=0.5, label='Frame 1 - V (Brightness)'),
        Patch(facecolor='lightcoral', edgecolor='red', alpha=0.5, label='Frame 2 - V (Brightness)'),
        Line2D([0], [0], color='darkblue', linewidth=2.5, marker='o', markersize=4, label='Frame 1 - Combined (V+S)'),
        Line2D([0], [0], color='darkred', linewidth=2.5, marker='s', markersize=4, label='Frame 2 - Combined (V+S)')
    ]
    ax_hist.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
    ax_hist.grid(True, alpha=0.3, linestyle='--')

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved visualization: {output_path}")
    print(f"  - V distance: {v_distance:.4f}")
    print(f"  - Combined distance: {combined_distance:.4f}")
    print(f"  - Status: {'Similar' if is_similar else 'Different'} (threshold={threshold})")


def create_paper_figure(similar_pair: Tuple[str, str],
                        different_pair: Tuple[str, str],
                        output_folder: str,
                        threshold: float = 0.3,
                        weight_brightness: float = 0.5,
                        weight_saturation: float = 0.5):
    """
    Create paper figure with both similar and different frame comparisons.

    Args:
        similar_pair: Tuple of (img1_path, img2_path) for similar frames
        different_pair: Tuple of (img1_path, img2_path) for different frames
        output_folder: Folder to save output figures
        threshold: Bhattacharyya distance threshold
        weight_brightness: Weight for V channel
        weight_saturation: Weight for S channel
    """
    os.makedirs(output_folder, exist_ok=True)

    print(f"\n{'='*80}")
    print("Creating Profile Tracking Visualization for Paper")
    print(f"{'='*80}\n")

    # Create similar frame comparison
    similar_output = os.path.join(output_folder, "profile_tracking_similar.png")
    print("Processing similar frames...")
    visualize_histogram_comparison(
        img1_path=similar_pair[0],
        img2_path=similar_pair[1],
        output_path=similar_output,
        threshold=threshold,
        weight_brightness=weight_brightness,
        weight_saturation=weight_saturation
    )

    # Create different frame comparison
    different_output = os.path.join(output_folder, "profile_tracking_different.png")
    print("\nProcessing different frames...")
    visualize_histogram_comparison(
        img1_path=different_pair[0],
        img2_path=different_pair[1],
        output_path=different_output,
        threshold=threshold,
        weight_brightness=weight_brightness,
        weight_saturation=weight_saturation
    )

    print(f"\n{'='*80}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Output folder: {output_folder}")
    print(f"  • Similar frames: {similar_output}")
    print(f"  • Different frames: {different_output}")
    print(f"{'='*80}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create Profile Tracking visualization for paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single comparison
  python visualize_profile_tracking.py --img1 frame1.jpg --img2 frame2.jpg --output result.png

  # Create paper figure (similar + different)
  python visualize_profile_tracking.py \\
    --similar-pair frame1.jpg frame2.jpg \\
    --different-pair frame3.jpg frame4.jpg \\
    --output-folder figures/ \\
    --threshold 0.3
        """
    )

    # Single comparison mode
    parser.add_argument('--img1', type=str, help='Path to first image')
    parser.add_argument('--img2', type=str, help='Path to second image')
    parser.add_argument('--output', type=str, help='Output file path')

    # Paper figure mode
    parser.add_argument('--similar-pair', type=str, nargs=2,
                       metavar=('IMG1', 'IMG2'),
                       help='Pair of similar frame paths')
    parser.add_argument('--different-pair', type=str, nargs=2,
                       metavar=('IMG1', 'IMG2'),
                       help='Pair of different frame paths')
    parser.add_argument('--output-folder', type=str,
                       help='Output folder for paper figures')

    # Parameters
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Bhattacharyya distance threshold (default: 0.3)')
    parser.add_argument('--weight-brightness', type=float, default=0.5,
                       help='Weight for V channel (default: 0.5)')
    parser.add_argument('--weight-saturation', type=float, default=0.5,
                       help='Weight for S channel (default: 0.5)')

    args = parser.parse_args()

    # Determine mode
    if args.similar_pair and args.different_pair and args.output_folder:
        # Paper figure mode
        create_paper_figure(
            similar_pair=tuple(args.similar_pair),
            different_pair=tuple(args.different_pair),
            output_folder=args.output_folder,
            threshold=args.threshold,
            weight_brightness=args.weight_brightness,
            weight_saturation=args.weight_saturation
        )
    elif args.img1 and args.img2 and args.output:
        # Single comparison mode
        visualize_histogram_comparison(
            img1_path=args.img1,
            img2_path=args.img2,
            output_path=args.output,
            threshold=args.threshold,
            weight_brightness=args.weight_brightness,
            weight_saturation=args.weight_saturation
        )
    else:
        parser.print_help()
        print("\nError: Please provide either:")
        print("  1. --img1, --img2, --output for single comparison")
        print("  2. --similar-pair, --different-pair, --output-folder for paper figure")


if __name__ == "__main__":
    main()
