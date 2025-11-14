"""Quick video info from Ground Truth data"""
import os
import glob

gt_folder = "Keyframe-extraction/Dataset/Keyframe"

print("\n" + "="*100)
print("비디오 프레임 정보 (Ground Truth 데이터 기반)")
print("="*100)
print(f"{'비디오 이름':<30} {'최대 프레임 번호':>15} {'GT 키프레임 개수':>20}")
print("="*100)

video_folders = sorted(glob.glob(os.path.join(gt_folder, "*/")))

total_max_frame = 0
total_gt_keyframes = 0

for video_folder in video_folders:
    video_name = os.path.basename(video_folder.rstrip('/'))

    # Find all jpg files with numeric names
    jpg_files = glob.glob(os.path.join(video_folder, "*.jpg"))

    # Extract frame numbers
    frame_numbers = []
    for jpg_file in jpg_files:
        filename = os.path.basename(jpg_file)
        if filename == "result.jpg":
            continue
        try:
            frame_num = int(filename.replace('.jpg', ''))
            frame_numbers.append(frame_num)
        except ValueError:
            continue

    if frame_numbers:
        max_frame = max(frame_numbers)
        gt_count = len(frame_numbers)

        total_max_frame += max_frame
        total_gt_keyframes += gt_count

        print(f"{video_name:<30} {max_frame:>15,} {gt_count:>20}")

print("="*100)
print(f"{'총 20개 비디오':<30} {'평균: '}{total_max_frame//len(video_folders):>9,} {'합계: '}{total_gt_keyframes:>14,}")
print("="*100)

# Additional statistics
print("\n추가 정보:")
print(f"  - 최대 프레임은 대략적인 비디오 길이를 나타냅니다")
print(f"  - 30 FPS 기준: 평균 {(total_max_frame//len(video_folders))/30/60:.1f}분 길이")
print(f"  - GT 키프레임 평균: {total_gt_keyframes/len(video_folders):.1f}개/비디오")
print(f"  - 평균 압축률: {(total_gt_keyframes/len(video_folders))/(total_max_frame//len(video_folders))*100:.2f}%")
print()
