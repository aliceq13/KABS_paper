"""
Profile-Only Keyframe Extraction

Tracking 없이 순수 히스토그램(Brightness+Saturation)만으로 키프레임 선택
- YOLO detection은 사용 (객체 인식)
- ByteTrack/BOT-SORT 없음 (tracking 없음)
- Greedy coverage 없음 (tracking ID가 없으므로)
- 순수 profile tracking (히스토그램 기반 선택)만 사용
"""

import os
import sys
import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from typing import List, Dict, Tuple


def calculate_histogram(frame: np.ndarray,
                        weight_brightness: float = 0.5,
                        weight_saturation: float = 0.5) -> np.ndarray:
    """
    프레임의 히스토그램 계산 (Brightness + Saturation)

    Args:
        frame: BGR 이미지
        weight_brightness: Brightness 가중치
        weight_saturation: Saturation 가중치

    Returns:
        가중 평균된 히스토그램
    """
    # BGR → HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # V channel (Brightness) 히스토그램
    hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    hist_v = cv2.normalize(hist_v, hist_v).flatten()

    # S channel (Saturation) 히스토그램
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    hist_s = cv2.normalize(hist_s, hist_s).flatten()

    # 가중 평균
    combined_hist = weight_brightness * hist_v + weight_saturation * hist_s

    return combined_hist


def compare_histograms(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    두 히스토그램 간의 correlation 계산

    Returns:
        Correlation 값 (1.0 = 완전히 동일, 0.0 = 완전히 다름)
    """
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def extract_keyframes_profile_only(
    video_path: str,
    output_folder: str,
    model_path: str = "yolo11m.pt",
    hist_threshold: float = 0.3,
    hist_weight_brightness: float = 0.5,
    hist_weight_saturation: float = 0.5,
    conf_threshold: float = 0.25,
    save_frames: bool = True
) -> List[int]:
    """
    Profile tracking만으로 키프레임 추출 (Tracking 없이)

    Args:
        video_path: 비디오 경로
        output_folder: 출력 폴더
        model_path: YOLO 모델 경로
        hist_threshold: 히스토그램 차이 임계값 (0.3 = 30% 차이면 새 키프레임)
        hist_weight_brightness: Brightness 가중치
        hist_weight_saturation: Saturation 가중치
        conf_threshold: YOLO detection confidence threshold
        save_frames: 키프레임 이미지 저장 여부

    Returns:
        선택된 키프레임 인덱스 리스트
    """
    # 출력 폴더 생성
    os.makedirs(output_folder, exist_ok=True)

    # YOLO 모델 로드
    print(f"\n{'='*80}")
    print("Profile-Only Keyframe Extraction")
    print(f"{'='*80}")
    print(f"Loading YOLO model: {model_path}")

    model = YOLO(model_path)

    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"\nVideo info:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {total_frames/fps:.2f}s")
    print(f"\nProfile tracking settings:")
    print(f"  Hist threshold: {hist_threshold}")
    print(f"  Brightness weight: {hist_weight_brightness}")
    print(f"  Saturation weight: {hist_weight_saturation}")

    selected_keyframes = []
    keyframe_data = []

    prev_hist = None
    frame_idx = 0

    print(f"\n{'='*80}")
    print("Processing frames...")
    print(f"{'='*80}\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        results = model(frame, conf=conf_threshold, verbose=False)
        num_detections = len(results[0].boxes)

        # 현재 프레임 히스토그램 계산
        current_hist = calculate_histogram(
            frame,
            weight_brightness=hist_weight_brightness,
            weight_saturation=hist_weight_saturation
        )

        # 키프레임 선택 로직
        is_keyframe = False

        if prev_hist is None:
            # 첫 프레임은 무조건 선택
            is_keyframe = True
            reason = "First frame"
        else:
            # 이전 선택된 키프레임과 히스토그램 비교
            correlation = compare_histograms(prev_hist, current_hist)

            # Correlation이 낮으면 (차이가 크면) 새 키프레임
            # correlation < (1 - threshold) 는 차이가 threshold 이상이라는 의미
            if correlation < (1 - hist_threshold):
                is_keyframe = True
                reason = f"Histogram change (corr={correlation:.3f})"

        # 키프레임으로 선택되면
        if is_keyframe:
            selected_keyframes.append(frame_idx)
            prev_hist = current_hist  # 기준 히스토그램 업데이트

            # 프레임 데이터 저장
            keyframe_info = {
                "frame_index": frame_idx,
                "num_detections": num_detections,
                "reason": reason
            }
            keyframe_data.append(keyframe_info)

            # 이미지 저장
            if save_frames:
                frame_filename = os.path.join(
                    output_folder,
                    f"keyframe_{frame_idx:06d}.jpg"
                )
                cv2.imwrite(frame_filename, frame)

            print(f"✓ Frame {frame_idx:6d}: {reason} (detections: {num_detections})")

        frame_idx += 1

        # 진행상황 표시
        if frame_idx % 100 == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"  Progress: {frame_idx}/{total_frames} ({progress:.1f}%)")

    cap.release()

    # 결과 저장
    summary = {
        "video_path": video_path,
        "total_frames": total_frames,
        "fps": fps,
        "method": "profile_only",
        "settings": {
            "hist_threshold": hist_threshold,
            "hist_weight_brightness": hist_weight_brightness,
            "hist_weight_saturation": hist_weight_saturation,
            "conf_threshold": conf_threshold
        },
        "num_keyframes": len(selected_keyframes),
        "keyframe_indices": selected_keyframes,
        "frames": keyframe_data
    }

    json_path = os.path.join(output_folder, "keyframe_summary_unified.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print("Extraction Complete")
    print(f"{'='*80}")
    print(f"Total keyframes: {len(selected_keyframes)}")
    print(f"Compression ratio: {len(selected_keyframes)/total_frames*100:.2f}%")
    print(f"Results saved to: {output_folder}")
    print(f"JSON summary: {json_path}")

    return selected_keyframes


def main(video_path: str = None,
         output_folder: str = None,
         model_path: str = "yolo11m.pt",
         hist_threshold: float = 0.3,
         hist_weight_brightness: float = 0.5,
         hist_weight_saturation: float = 0.5,
         **kwargs):
    """
    메인 함수 (model_wrapper.py에서 호출 가능)

    Args:
        video_path: 비디오 경로
        output_folder: 출력 폴더
        model_path: YOLO 모델 경로
        hist_threshold: 히스토그램 차이 임계값
        hist_weight_brightness: Brightness 가중치
        hist_weight_saturation: Saturation 가중치
        **kwargs: 기타 파라미터 (무시됨)
    """
    if video_path is None:
        raise ValueError("video_path is required")

    if output_folder is None:
        output_folder = "profile_only_results"

    keyframes = extract_keyframes_profile_only(
        video_path=video_path,
        output_folder=output_folder,
        model_path=model_path,
        hist_threshold=hist_threshold,
        hist_weight_brightness=hist_weight_brightness,
        hist_weight_saturation=hist_weight_saturation
    )

    return keyframes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Profile-Only Keyframe Extraction (No Tracking)"
    )
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--output', type=str, default='profile_only_results',
                       help='Output folder')
    parser.add_argument('--model', type=str, default='yolo11m.pt',
                       help='YOLO model path')
    parser.add_argument('--threshold', type=float, default=0.3,
                       help='Histogram difference threshold (default: 0.3)')
    parser.add_argument('--weight-brightness', type=float, default=0.5,
                       help='Brightness weight (default: 0.5)')
    parser.add_argument('--weight-saturation', type=float, default=0.5,
                       help='Saturation weight (default: 0.5)')

    args = parser.parse_args()

    extract_keyframes_profile_only(
        video_path=args.video,
        output_folder=args.output,
        model_path=args.model,
        hist_threshold=args.threshold,
        hist_weight_brightness=args.weight_brightness,
        hist_weight_saturation=args.weight_saturation
    )
