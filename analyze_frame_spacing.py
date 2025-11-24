"""
프레임 간격 분석 도구

모델이 선택한 키프레임들 사이의 간격을 분석하여
연속된 프레임을 많이 선택하는지 확인합니다.
"""

import json
import os
import argparse
from typing import List
import numpy as np


def load_keyframes_from_json(json_path: str) -> List[int]:
    """JSON에서 키프레임 인덱스 로드"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return sorted([frame['frame_index'] for frame in data['frames']])


def analyze_frame_spacing(keyframes: List[int]) -> dict:
    """프레임 간격 분석"""
    if len(keyframes) < 2:
        return {
            'num_frames': len(keyframes),
            'spacings': [],
            'min_spacing': 0,
            'max_spacing': 0,
            'mean_spacing': 0,
            'median_spacing': 0,
            'std_spacing': 0,
            'consecutive_pairs': 0,
            'within_tolerance_15': 0,
        }

    # 연속된 프레임 간의 간격 계산
    spacings = [keyframes[i+1] - keyframes[i] for i in range(len(keyframes)-1)]

    # 통계
    min_spacing = min(spacings)
    max_spacing = max(spacings)
    mean_spacing = np.mean(spacings)
    median_spacing = np.median(spacings)
    std_spacing = np.std(spacings)

    # 문제가 될 수 있는 경우들
    consecutive_pairs = sum(1 for s in spacings if s == 1)  # 연속 프레임 (간격 1)
    within_tolerance_15 = sum(1 for s in spacings if s <= 15)  # tolerance 15 이내

    return {
        'num_frames': len(keyframes),
        'spacings': spacings,
        'min_spacing': min_spacing,
        'max_spacing': max_spacing,
        'mean_spacing': mean_spacing,
        'median_spacing': median_spacing,
        'std_spacing': std_spacing,
        'consecutive_pairs': consecutive_pairs,
        'within_tolerance_15': within_tolerance_15,
    }


def print_analysis(analysis: dict, name: str):
    """분석 결과 출력"""
    print(f"\n{'='*80}")
    print(f"프레임 간격 분석: {name}")
    print(f"{'='*80}")

    print(f"\n기본 정보:")
    print(f"  총 키프레임 수: {analysis['num_frames']}")

    if analysis['num_frames'] < 2:
        print(f"  ⚠️ 키프레임이 2개 미만이므로 간격 분석 불가")
        return

    print(f"\n간격 통계:")
    print(f"  최소 간격: {analysis['min_spacing']} 프레임")
    print(f"  최대 간격: {analysis['max_spacing']} 프레임")
    print(f"  평균 간격: {analysis['mean_spacing']:.1f} 프레임")
    print(f"  중간값: {analysis['median_spacing']:.1f} 프레임")
    print(f"  표준편차: {analysis['std_spacing']:.1f} 프레임")

    print(f"\n잠재적 문제:")
    print(f"  연속 프레임 쌍 (간격=1): {analysis['consecutive_pairs']}")
    print(f"  tolerance 15 이내 쌍: {analysis['within_tolerance_15']}")

    total_pairs = analysis['num_frames'] - 1
    if total_pairs > 0:
        consecutive_percent = (analysis['consecutive_pairs'] / total_pairs) * 100
        within_tol_percent = (analysis['within_tolerance_15'] / total_pairs) * 100

        print(f"\n비율:")
        print(f"  연속 프레임 비율: {consecutive_percent:.1f}%")
        print(f"  tolerance 15 이내 비율: {within_tol_percent:.1f}%")

        # 경고 메시지
        if consecutive_percent > 10:
            print(f"\n⚠️ 경고: 연속 프레임이 {consecutive_percent:.1f}%로 많습니다!")
            print(f"   tolerance 평가 시 일부 프레임이 같은 GT와 매칭 경쟁할 수 있습니다.")
        elif within_tol_percent > 30:
            print(f"\n⚠️ 주의: tolerance 15 이내 프레임이 {within_tol_percent:.1f}%입니다.")
            print(f"   일부 프레임이 같은 GT와 매칭 경쟁할 수 있습니다.")
        else:
            print(f"\n✓ 프레임 간격이 적절합니다. tolerance 평가에 문제 없을 것으로 보입니다.")

    # 간격 분포 히스토그램 (간단한 텍스트 버전)
    print(f"\n간격 분포:")
    bins = [0, 5, 10, 15, 30, 60, 100, float('inf')]
    labels = ['1-5', '6-10', '11-15', '16-30', '31-60', '61-100', '100+']

    for i in range(len(bins)-1):
        count = sum(1 for s in analysis['spacings']
                   if bins[i] < s <= bins[i+1])
        if count > 0:
            bar = '█' * int(count / max(1, total_pairs) * 50)
            print(f"  {labels[i]:>8} 프레임: {bar} {count}")


def analyze_result_folder(result_folder: str):
    """결과 폴더의 모든 모델 구성 분석"""
    model_results_folder = os.path.join(result_folder, "model_results")

    if not os.path.exists(model_results_folder):
        print(f"✗ 모델 결과 폴더를 찾을 수 없습니다: {model_results_folder}")
        return

    # 비디오 이름 추출
    video_name = os.path.basename(result_folder).replace("single_video_", "")

    # 각 모델 구성 찾기
    video_folder = os.path.join(model_results_folder, video_name)
    if not os.path.exists(video_folder):
        print(f"✗ 비디오 폴더를 찾을 수 없습니다: {video_folder}")
        return

    configs = [d for d in os.listdir(video_folder)
               if os.path.isdir(os.path.join(video_folder, d))]

    if not configs:
        print(f"✗ 모델 구성을 찾을 수 없습니다")
        return

    print(f"\n{'='*80}")
    print(f"비디오: {video_name}")
    print(f"찾은 구성: {len(configs)}개")
    print(f"{'='*80}")

    all_analyses = {}

    for config in sorted(configs):
        json_path = os.path.join(video_folder, config, "keyframe_summary_unified.json")

        if not os.path.exists(json_path):
            print(f"\n⚠️ {config}: JSON 파일을 찾을 수 없습니다")
            continue

        keyframes = load_keyframes_from_json(json_path)
        analysis = analyze_frame_spacing(keyframes)
        all_analyses[config] = analysis

        print_analysis(analysis, config)

    # 전체 요약
    if len(all_analyses) > 1:
        print(f"\n{'='*80}")
        print(f"전체 요약")
        print(f"{'='*80}")

        for config, analysis in all_analyses.items():
            if analysis['num_frames'] >= 2:
                consecutive_pct = (analysis['consecutive_pairs'] / (analysis['num_frames']-1)) * 100
                print(f"  {config:30s}: 평균 간격 {analysis['mean_spacing']:6.1f}, "
                      f"연속 프레임 {consecutive_pct:4.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="키프레임 간격 분석 도구"
    )

    parser.add_argument('--result-folder', type=str,
                       help='결과 폴더 경로 (미지정시 최신 폴더)')
    parser.add_argument('--json', type=str,
                       help='직접 JSON 파일 경로 지정')

    args = parser.parse_args()

    if args.json:
        # 직접 JSON 파일 분석
        keyframes = load_keyframes_from_json(args.json)
        analysis = analyze_frame_spacing(keyframes)
        print_analysis(analysis, os.path.basename(args.json))

    elif args.result_folder:
        # 결과 폴더 분석
        analyze_result_folder(args.result_folder)

    else:
        # 최신 결과 폴더 찾기
        experiment_base = "experiment_results"
        if not os.path.exists(experiment_base):
            print(f"✗ {experiment_base} 폴더를 찾을 수 없습니다")
            print(f"사용법: python analyze_frame_spacing.py --result-folder <폴더경로>")
            return

        result_folders = [d for d in os.listdir(experiment_base)
                         if d.startswith("single_video_")]

        if not result_folders:
            print(f"✗ 결과 폴더를 찾을 수 없습니다")
            return

        latest_folder = os.path.join(experiment_base, sorted(result_folders)[-1])
        print(f"최신 결과 폴더 사용: {latest_folder}")
        analyze_result_folder(latest_folder)


if __name__ == "__main__":
    main()
