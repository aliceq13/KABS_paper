# Ablation Study - 논리적 구조 검증

## 🎯 목적

각 구성 요소(Tracking, Profile Tracking)의 효과를 독립적으로 평가하여 시스템의 기여도를 입증

## 📊 3가지 실험 구성

### Case 1: Full Model (전체 시스템)
```
구성 요소:
✓ YOLO Detection
✓ ByteTrack Tracking
✓ Pre-Profile Tracking (greedy 선택 전)
✓ Greedy Coverage Selection
✓ Post-Profile Tracking (greedy 선택 후)
✓ Re-ID

파라미터:
- profile_only: False
- profile_iterations: 3
- apply_post_filter: True

실행 방식:
→ yolo_osnet_4_with_filtering_updated (1).py (기존 모델 그대로)

기대 결과:
- 가장 높은 F1-score (모든 기능 활성화)
- 적절한 compression ratio
```

### Case 2: No Profile (Profile Tracking 제거)
```
구성 요소:
✓ YOLO Detection
✓ ByteTrack Tracking
✓ Greedy Coverage Selection
✓ Re-ID
✗ Pre-Profile Tracking (비활성화)
✗ Post-Profile Tracking (비활성화)

파라미터:
- profile_only: False
- profile_iterations: 0  ← 핵심!
- apply_post_filter: False  ← 핵심!

실행 방식:
→ yolo_osnet_4_with_filtering_updated (1).py (파라미터만 변경)

기대 결과:
- Full Model보다 낮은 F1-score
- 더 많은 keyframe 선택 (필터링이 없으므로)
- Profile Tracking의 효과 입증
```

### Case 3: Profile Only (Tracking 제거)
```
구성 요소:
✓ YOLO Detection (객체 인식만)
✓ Histogram-based Selection (Brightness + Saturation)
✗ ByteTrack Tracking (없음)
✗ Greedy Coverage Selection (tracking ID 필요, 불가능)
✗ Re-ID (tracking 필요, 불가능)

파라미터:
- profile_only: True  ← 핵심!

실행 방식:
→ keyframe_extraction_profile_only.py (새 스크립트)

알고리즘:
1. YOLO로 객체 detection
2. 전체 프레임의 히스토그램 계산 (HSV의 V, S 채널)
3. 이전 선택된 키프레임과 correlation 비교
4. correlation < (1 - threshold) 면 새 키프레임 선택

기대 결과:
- Full Model보다 낮은 F1-score
- Tracking + Greedy의 효과 입증
- 순수 히스토그램 방법의 한계 확인
```

## 🔍 논리적 구조 검증

### ✅ Case 1 vs Case 2: Profile Tracking의 효과
```
차이점: Profile Tracking 유무
동일점: YOLO + ByteTrack + Greedy

예상:
- Case 1 F1-score > Case 2 F1-score
- Case 1 keyframes < Case 2 keyframes (더 정제됨)

결론: Profile Tracking이 중복/유사 프레임 제거에 효과적
```

### ✅ Case 1 vs Case 3: Tracking의 효과
```
차이점: Tracking + Greedy 유무
동일점: YOLO + Histogram

예상:
- Case 1 F1-score > Case 3 F1-score
- Case 1이 더 의미있는 객체 조합 선택

결론: Tracking 기반 객체 조합이 히스토그램만보다 우수
```

### ✅ Case 2 vs Case 3: Greedy Coverage의 효과
```
차이점: Tracking + Greedy vs Histogram만
동일점: Profile tracking 없음

예상:
- Case 2 F1-score > Case 3 F1-score
- Case 2가 더 객체 구성 변화 잘 포착

결론: Greedy coverage가 히스토그램보다 객체 다양성 확보에 효과적
```

## 🧪 검증 체크리스트

### Case 1: Full Model
- [ ] profile_iterations=3 설정 확인
- [ ] apply_post_filter=True 설정 확인
- [ ] ByteTrack 사용 확인
- [ ] Greedy coverage 작동 확인
- [ ] keyframe_summary_unified.json 생성 확인

### Case 2: No Profile
- [ ] profile_iterations=0 설정 확인
- [ ] apply_post_filter=False 설정 확인
- [ ] ByteTrack 사용 확인 (여전히 tracking 사용)
- [ ] Greedy coverage 작동 확인
- [ ] Case 1보다 더 많은 keyframe 선택 예상

### Case 3: Profile Only
- [ ] keyframe_extraction_profile_only.py 실행 확인
- [ ] ByteTrack 미사용 확인
- [ ] Greedy coverage 미사용 확인
- [ ] 히스토그램 기반 선택만 사용 확인
- [ ] YOLO detection은 사용 (객체 수 기록)
- [ ] keyframe_summary_unified.json 생성 확인

## 📝 예상 결과 표

```
Method                  | F1@tol=0 | F1@tol=15 | F1@tol=30 | #Frames | Compression
------------------------|----------|-----------|-----------|---------|-------------
Full Model              | 0.XXX    | 0.XXX     | 0.XXX     | XXX     | XX.X%
  (Best expected)       | (High)   | (High)    | (High)    | (Mid)   | (Low-Mid)

No Profile              | 0.XXX    | 0.XXX     | 0.XXX     | XXX     | XX.X%
  (More frames)         | (Mid)    | (Mid)     | (Mid)     | (High)  | (High)

Profile Only            | 0.XXX    | 0.XXX     | 0.XXX     | XXX     | XX.X%
  (Histogram baseline)  | (Low)    | (Low-Mid) | (Low-Mid) | (?)     | (?)

Baseline: Uniform-15    | 0.XXX    | 0.XXX     | 0.XXX     | XXX     | XX.X%
  (Reference)           | (Low)    | (Low)     | (Low)     | (Fixed) | (Fixed)
```

## 🔧 실행 방법

### 1. Ablation Study 실행
```bash
python run_ablation_study.py --video your_video.mp4
```

### 2. 개별 케이스 테스트
```bash
# Case 1: Full Model
python evaluate_single_video.py --video your_video.mp4

# Case 2: No Profile (수동 설정 필요, model_wrapper.py 수정)

# Case 3: Profile Only
python keyframe_extraction_profile_only.py --video your_video.mp4
```

### 3. 결과 비교
```bash
python compare_model_frames.py --result-folder experiment_results/ablation_study_xxx
```

## ⚠️ 주의사항

### 1. Profile Only의 한계
- Tracking이 없으므로 객체 ID를 알 수 없음
- Greedy coverage를 사용할 수 없음 (객체 조합 계산 불가)
- 순수 히스토그램 기반 선택만 가능
- **이것이 의도된 것**: Profile tracking만의 효과를 측정하기 위함

### 2. 파라미터 일관성
- 모든 케이스에서 동일한 YOLO 모델 사용 (yolo11m.pt)
- 동일한 히스토그램 설정 사용
  - hist_threshold: 0.3
  - hist_weight_brightness: 0.5
  - hist_weight_saturation: 0.5

### 3. Ground Truth 필요
- 평가를 위해 GT가 반드시 있어야 함
- GT는 객체 구성 변화 기준으로 생성되어야 함

## 🎓 논문 작성 가이드

### 실험 설명
```
We conduct an ablation study to evaluate the contribution of each component:
1. Full Model: Complete system with all components
2. w/o Profile: Removes histogram-based filtering (pre/post)
3. Profile Only: Removes tracking, uses only histogram-based selection

This allows us to measure:
- The effectiveness of profile tracking in reducing redundancy
- The advantage of tracking-based object composition over histogram-only methods
- The synergy between tracking and profile filtering
```

### 결과 해석 예시
```
The ablation study demonstrates:
- Profile tracking (Full vs No Profile) improves F1-score by X%
  while reducing keyframes by Y%, confirming its effectiveness in
  removing redundant frames.

- Tracking-based greedy coverage (Full vs Profile Only) outperforms
  histogram-only methods by Z%, showing that object composition
  changes are more meaningful than visual appearance changes alone.
```

## ✅ 논리적 타당성 확인

### 1. 독립 변인 제어
- ✅ Case 1 vs 2: Profile tracking만 차이
- ✅ Case 1 vs 3: Tracking만 차이
- ✅ Case 2 vs 3: 둘 다 비교 가능

### 2. 코드 구현 일관성
- ✅ Case 1, 2: 동일 스크립트, 파라미터만 차이
- ✅ Case 3: 별도 스크립트, 명확히 구분
- ✅ 모든 케이스 동일 JSON 포맷 출력

### 3. 평가 메트릭 공정성
- ✅ 동일한 GT 사용
- ✅ 동일한 tolerance 설정
- ✅ 동일한 평가 함수 사용

**결론: 논리적 구조에 문제 없음 ✓**
