"""
Ground Truth Keyframe Annotation Tool

A simple OpenCV-based tool for manually selecting keyframes from videos.

Usage:
    python create_ground_truth.py --video path/to/video.mp4
    python create_ground_truth.py --video path/to/video.mp4 --output custom_output_folder

Keyboard Controls:
    SPACE       - Mark current frame as keyframe
    RIGHT (→)   - Next frame
    LEFT (←)    - Previous frame
    UP (↑)      - Skip forward 30 frames
    DOWN (↓)    - Skip backward 30 frames
    D           - Delete last selected keyframe
    R           - Reset all keyframes
    S           - Save and continue
    P           - Play/Pause video
    Q/ESC       - Save and quit
    H           - Show help
"""

import cv2
import os
import json
import argparse
from typing import List, Set
import sys


class GroundTruthAnnotator:
    """Interactive tool for creating ground truth keyframes."""

    def __init__(self, video_path: str, output_base_folder: str = "Keyframe-extraction/Dataset/Keyframe"):
        """
        Initialize the annotator.

        Args:
            video_path: Path to video file
            output_base_folder: Base folder for saving keyframes
        """
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.output_base_folder = output_base_folder

        # Video capture
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        # Video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Current state
        self.current_frame_idx = 0
        self.keyframes: Set[int] = set()
        self.playing = False
        self.current_frame = None

        # Output folder
        video_basename = os.path.splitext(self.video_name)[0]
        self.output_folder = os.path.join(output_base_folder, video_basename + ".mp4")
        os.makedirs(self.output_folder, exist_ok=True)

        # Window name
        self.window_name = "Ground Truth Annotation Tool"

        # Load existing keyframes if any
        self.load_existing_keyframes()

        print(f"\n{'='*80}")
        print(f"Ground Truth Annotation Tool")
        print(f"{'='*80}")
        print(f"Video: {self.video_name}")
        print(f"Total frames: {self.total_frames}")
        print(f"FPS: {self.fps:.2f}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"Output folder: {self.output_folder}")
        if self.keyframes:
            print(f"Loaded {len(self.keyframes)} existing keyframes")
        print(f"{'='*80}\n")
        self.print_help()

    def load_existing_keyframes(self):
        """Load existing keyframes from output folder if they exist."""
        json_path = os.path.join(self.output_folder, "keyframes.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    self.keyframes = set(data.get('keyframes', []))
                    print(f"✓ Loaded {len(self.keyframes)} existing keyframes")
            except Exception as e:
                print(f"⚠️ Could not load existing keyframes: {e}")

    def print_help(self):
        """Print keyboard controls."""
        print("Keyboard Controls:")
        print("  SPACE     - Mark current frame as keyframe")
        print("  → (RIGHT) - Next frame")
        print("  ← (LEFT)  - Previous frame")
        print("  ↑ (UP)    - Skip forward 30 frames")
        print("  ↓ (DOWN)  - Skip backward 30 frames")
        print("  D         - Delete last selected keyframe")
        print("  R         - Reset all keyframes")
        print("  S         - Save and continue")
        print("  P         - Play/Pause video")
        print("  Q/ESC     - Save and quit")
        print("  H         - Show this help")
        print(f"{'='*80}\n")

    def read_frame(self, frame_idx: int):
        """Read a specific frame from video."""
        if frame_idx < 0:
            frame_idx = 0
        elif frame_idx >= self.total_frames:
            frame_idx = self.total_frames - 1

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if ret:
            self.current_frame_idx = frame_idx
            self.current_frame = frame

        return ret, frame

    def draw_overlay(self, frame):
        """Draw information overlay on frame."""
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Semi-transparent background for text
        cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Text information
        is_keyframe = self.current_frame_idx in self.keyframes
        status_text = "★ KEYFRAME ★" if is_keyframe else ""
        status_color = (0, 255, 0) if is_keyframe else (255, 255, 255)

        # Frame info
        time_sec = self.current_frame_idx / self.fps
        text_lines = [
            f"Frame: {self.current_frame_idx}/{self.total_frames} | Time: {time_sec:.2f}s | {status_text}",
            f"Selected Keyframes: {len(self.keyframes)} | Compression: {len(self.keyframes)/self.total_frames*100:.2f}%",
            f"Mode: {'PLAYING' if self.playing else 'PAUSED'} | Press H for help"
        ]

        y_offset = 25
        for i, text in enumerate(text_lines):
            color = status_color if i == 0 and is_keyframe else (255, 255, 255)
            cv2.putText(frame, text, (10, y_offset + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Progress bar
        progress = self.current_frame_idx / self.total_frames
        bar_width = w - 20
        bar_height = 10
        bar_y = h - 30

        cv2.rectangle(frame, (10, bar_y), (10 + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        cv2.rectangle(frame, (10, bar_y), (10 + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)

        # Keyframe markers on progress bar
        for kf in self.keyframes:
            marker_x = 10 + int((kf / self.total_frames) * bar_width)
            cv2.line(frame, (marker_x, bar_y - 5), (marker_x, bar_y + bar_height + 5), (0, 255, 255), 2)

        return frame

    def add_keyframe(self):
        """Add current frame as keyframe."""
        self.keyframes.add(self.current_frame_idx)
        print(f"✓ Added keyframe: {self.current_frame_idx} (Total: {len(self.keyframes)})")

    def remove_last_keyframe(self):
        """Remove the most recently added keyframe."""
        if self.keyframes:
            removed = max(self.keyframes)
            self.keyframes.remove(removed)
            print(f"✗ Removed keyframe: {removed} (Remaining: {len(self.keyframes)})")
        else:
            print("⚠️ No keyframes to remove")

    def reset_keyframes(self):
        """Clear all keyframes."""
        count = len(self.keyframes)
        self.keyframes.clear()
        print(f"⚠️ Reset all keyframes ({count} removed)")

    def save_keyframes(self):
        """Save keyframes to disk."""
        if not self.keyframes:
            print("⚠️ No keyframes to save")
            return

        print(f"\n{'='*60}")
        print(f"Saving {len(self.keyframes)} keyframes...")
        print(f"{'='*60}")

        # Save frame images
        saved_count = 0
        for frame_idx in sorted(self.keyframes):
            ret, frame = self.read_frame(frame_idx)
            if ret:
                output_path = os.path.join(self.output_folder, f"{frame_idx}.jpg")
                cv2.imwrite(output_path, frame)
                saved_count += 1
                print(f"  ✓ Saved: {frame_idx}.jpg")

        # Save JSON metadata
        json_path = os.path.join(self.output_folder, "keyframes.json")
        metadata = {
            'video_name': self.video_name,
            'video_path': self.video_path,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'keyframes': sorted(list(self.keyframes)),
            'num_keyframes': len(self.keyframes),
            'compression_ratio': len(self.keyframes) / self.total_frames
        }

        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ Saved metadata: keyframes.json")
        print(f"{'='*60}")
        print(f"✓ Successfully saved {saved_count} keyframes to: {self.output_folder}")
        print(f"  Compression ratio: {metadata['compression_ratio']*100:.2f}%")
        print(f"{'='*60}\n")

        # Restore current frame
        self.read_frame(self.current_frame_idx)

    def run(self):
        """Main annotation loop."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

        # Read first frame
        ret, frame = self.read_frame(0)
        if not ret:
            print("✗ Cannot read video")
            return

        while True:
            # Read current frame if not playing or need to display
            if not self.playing:
                ret, frame = self.read_frame(self.current_frame_idx)
                if not ret:
                    break
            else:
                # Playing mode - advance frame
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame_idx += 1
                    self.current_frame = frame
                else:
                    self.playing = False
                    self.current_frame_idx = 0

            # Draw overlay
            display_frame = self.draw_overlay(frame.copy())

            # Show frame
            cv2.imshow(self.window_name, display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1 if self.playing else 30) & 0xFF

            if key == ord(' '):  # Space - Add keyframe
                self.add_keyframe()

            elif key == 83 or key == 3:  # Right arrow - Next frame
                self.playing = False
                self.current_frame_idx = min(self.current_frame_idx + 1, self.total_frames - 1)

            elif key == 81 or key == 2:  # Left arrow - Previous frame
                self.playing = False
                self.current_frame_idx = max(self.current_frame_idx - 1, 0)

            elif key == 82 or key == 0:  # Up arrow - Skip forward
                self.playing = False
                self.current_frame_idx = min(self.current_frame_idx + 30, self.total_frames - 1)

            elif key == 84 or key == 1:  # Down arrow - Skip backward
                self.playing = False
                self.current_frame_idx = max(self.current_frame_idx - 30, 0)

            elif key == ord('d') or key == ord('D'):  # Delete last keyframe
                self.remove_last_keyframe()

            elif key == ord('r') or key == ord('R'):  # Reset all keyframes
                self.reset_keyframes()

            elif key == ord('s') or key == ord('S'):  # Save
                self.save_keyframes()

            elif key == ord('p') or key == ord('P'):  # Play/Pause
                self.playing = not self.playing
                print(f"{'▶ Playing' if self.playing else '⏸ Paused'}")

            elif key == ord('h') or key == ord('H'):  # Help
                self.print_help()

            elif key == ord('q') or key == ord('Q') or key == 27:  # Quit (Q or ESC)
                print("\nQuitting...")
                if self.keyframes:
                    self.save_keyframes()
                break

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

        print(f"\n{'='*80}")
        print(f"Annotation session complete!")
        print(f"Total keyframes: {len(self.keyframes)}")
        print(f"Output folder: {self.output_folder}")
        print(f"{'='*80}\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Ground Truth Keyframe Annotation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Keyboard Controls:
  SPACE     - Mark current frame as keyframe
  → (RIGHT) - Next frame
  ← (LEFT)  - Previous frame
  ↑ (UP)    - Skip forward 30 frames
  ↓ (DOWN)  - Skip backward 30 frames
  D         - Delete last selected keyframe
  R         - Reset all keyframes
  S         - Save and continue
  P         - Play/Pause video
  Q/ESC     - Save and quit
  H         - Show help

Examples:
  python create_ground_truth.py --video my_video.mp4
  python create_ground_truth.py --video my_video.mp4 --output custom_folder
        """
    )

    parser.add_argument('--video', '-v', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--output', '-o', type=str,
                       default='Keyframe-extraction/Dataset/Keyframe',
                       help='Base output folder (default: Keyframe-extraction/Dataset/Keyframe)')

    args = parser.parse_args()

    # Check if video exists
    if not os.path.exists(args.video):
        print(f"✗ Error: Video file not found: {args.video}")
        sys.exit(1)

    # Create annotator and run
    try:
        annotator = GroundTruthAnnotator(args.video, args.output)
        annotator.run()
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
