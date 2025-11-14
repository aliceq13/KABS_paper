# 새로운 비디오 평가 가이드

새로 만든 GT 데이터로 평가하는 방법을 단계별로 안내합니다.

## 📋 전체 작업 흐름

```
1. 비디오 준비
   ↓
2. Ground Truth 생성
   ↓
3. 평가 실행
   ↓
4. 결과 확인
```

---

## 🎬 방법 1: 단일 비디오 평가 (추천)

**가장 빠르고 간단한 방법입니다.**

### Step 1: 비디오 준비

```bash
# 본인의 비디오를 준비
# 예: my_new_video.mp4
```

### Step 2: Ground Truth 생성

```bash
python create_ground_truth.py --video my_new_video.mp4
```

**작업 과정:**
1. 비디오가 열립니다
2. `P`로 재생하며 전체 확인
3. 중요한 프레임에서 `SPACE`로 키프레임 선택
4. `Q`로 저장하고 종료

**저장 위치:**
```
Keyframe-extraction/Dataset/Keyframe/
└── my_new_video.mp4/
    ├── 150.jpg
    ├── 570.jpg
    ├── ...
    └── keyframes.json
```

### Step 3: 평가 실행

```bash
python evaluate_single_video.py --video my_new_video.mp4
```

**실행되는 내용:**
- ✓ Baseline 방법 2개 (Uniform_30, Uniform_60)
- ✓ 사용자 모델 4개 (YOLO+BOTSORT, YOLO+ByteTrack, RTDETR+BOTSORT, RTDETR+ByteTrack)
- ✓ 총 6개 방법으로 평가

### Step 4: 결과 확인

결과는 `experiment_results/single_video_YYYYMMDD_HHMMSS/` 폴더에 저장됩니다:

```
experiment_results/single_video_20250112_143000/
├── baseline_results/           # 베이스라인 결과
│   └── my_new_video/
│       ├── Uniform_30/
│       └── Uniform_60/
├── model_results/              # 모델 결과
│   └── my_new_video/
│       ├── YOLO_BOTSORT/
│       ├── YOLO_ByteTrack/
│       ├── RTDETR_BOTSORT/
│       └── RTDETR_ByteTrack/
└── evaluation/                 # 평가 지표
    ├── results.csv             ← 모든 결과 (F1-score 등)
    ├── summary_statistics.csv  ← 통계 요약
    └── summary.txt             ← 텍스트 요약
```

#### results.csv 예시

| method | video_name | num_keyframes | f1_score_tol0 | f1_score_tol15 | f1_score_tol30 | precision_tol15 | recall_tol15 | compression_ratio |
|--------|-----------|---------------|---------------|----------------|----------------|-----------------|--------------|-------------------|
| Uniform_30 | my_new_video.mp4 | 166 | 0.45 | 0.62 | 0.75 | 0.58 | 0.67 | 0.033 |
| Uniform_60 | my_new_video.mp4 | 83 | 0.38 | 0.55 | 0.68 | 0.52 | 0.59 | 0.017 |
| YOLO_BOTSORT | my_new_video.mp4 | 15 | 0.62 | 0.79 | 0.91 | 0.85 | 0.73 | 0.003 |
| YOLO_ByteTrack | my_new_video.mp4 | 18 | 0.58 | 0.75 | 0.88 | 0.80 | 0.70 | 0.004 |
| RTDETR_BOTSORT | my_new_video.mp4 | 14 | 0.65 | 0.82 | 0.93 | 0.88 | 0.76 | 0.003 |
| RTDETR_ByteTrack | my_new_video.mp4 | 16 | 0.60 | 0.77 | 0.89 | 0.82 | 0.72 | 0.003 |

#### 터미널 출력 예시

```
================================================================================
EVALUATION SUMMARY
================================================================================

Method: Uniform_30
  Avg F1-Score (tol=0):  0.4500
  Avg F1-Score (tol=15): 0.6200
  Avg F1-Score (tol=30): 0.7500
  Avg Keyframes: 166.00
  Avg Compression: 3.32%

Method: YOLO_BOTSORT
  Avg F1-Score (tol=0):  0.6200
  Avg F1-Score (tol=15): 0.7900
  Avg F1-Score (tol=30): 0.9100
  Avg Keyframes: 15.00
  Avg Compression: 0.30%

✓ Best F1-Score (tol=15): RTDETR_BOTSORT (0.82)
================================================================================
```

---

## 🎬 방법 2: 기존 데이터셋에 추가

기존 20개 비디오와 함께 평가하고 싶을 때 사용합니다.

### Step 1: 비디오를 데이터셋 폴더에 복사

```bash
cp my_new_video.mp4 Keyframe-extraction/Dataset/Videos/
```

### Step 2: Ground Truth 생성

```bash
python create_ground_truth.py --video Keyframe-extraction/Dataset/Videos/my_new_video.mp4
```

자동으로 `Keyframe-extraction/Dataset/Keyframe/my_new_video.mp4/`에 저장됩니다.

### Step 3: 모든 비디오 평가

```bash
python run_experiments.py
```

**주의:** 기존 20개 비디오 + 새 비디오를 모두 평가하므로 시간이 오래 걸립니다 (수 시간).

### Step 4: 결과 확인

```
experiment_results/experiment_YYYYMMDD_HHMMSS/
└── evaluation/
    ├── detailed_results.csv         ← 21개 비디오 × 6개 방법 = 126 rows
    ├── aggregated_by_method.csv     ← 방법별 평균
    └── aggregated_by_video.csv      ← 비디오별 평균
```

---

## 🎯 고급 옵션

### 특정 베이스라인만 평가

```bash
# 베이스라인 제외, 모델만 평가
python evaluate_single_video.py --video my_video.mp4 --no-baselines

# 모델 제외, 베이스라인만 평가
python evaluate_single_video.py --video my_video.mp4 --no-model
```

### 커스텀 베이스라인 간격

```bash
# 10, 20, 30 프레임 간격으로 평가
python evaluate_single_video.py --video my_video.mp4 --baseline-intervals 10 20 30
```

### 커스텀 Tolerance

```bash
# ±5, ±10, ±20 프레임 tolerance로 평가
python evaluate_single_video.py --video my_video.mp4 --tolerances 5 10 20
```

### 커스텀 GT 폴더

```bash
# GT를 다른 폴더에 저장했을 때
python create_ground_truth.py --video my_video.mp4 --output my_custom_gt_folder
python evaluate_single_video.py --video my_video.mp4 --gt-folder my_custom_gt_folder
```

---

## 📊 평가 지표 이해하기

### F1-Score with Tolerance

| Tolerance | 의미 | 설명 |
|-----------|------|------|
| **tol=0** | 엄격한 평가 | 프레임이 정확히 일치해야 함 |
| **tol=15** | 중간 평가 | ±15 프레임 (30fps 기준 ±0.5초) |
| **tol=30** | 관대한 평가 | ±30 프레임 (30fps 기준 ±1초) |

### 주요 지표

- **F1-Score**: 정밀도와 재현율의 조화평균 (0-1, 높을수록 좋음)
- **Precision**: 선택한 키프레임 중 올바른 비율
- **Recall**: GT 키프레임 중 찾아낸 비율
- **Compression Ratio**: 선택된 키프레임 / 전체 프레임 (낮을수록 압축률 좋음)

### 예시 해석

```
Method: YOLO_BOTSORT
  F1-Score (tol=15): 0.79
  Precision (tol=15): 0.85
  Recall (tol=15): 0.73
  Compression: 0.30%
```

**해석:**
- 선택한 키프레임의 85%가 GT와 ±15프레임 내에 일치 (높은 정확도)
- GT 키프레임의 73%를 찾아냄 (일부 놓침)
- 전체 프레임의 0.3%만 선택 (효율적)
- 종합 F1-Score: 0.79 (우수)

---

## 🔄 전체 워크플로우 예시

### 예제: 새 비디오 3개 평가

```bash
# 1. 첫 번째 비디오
python create_ground_truth.py --video video1.mp4
python evaluate_single_video.py --video video1.mp4

# 2. 두 번째 비디오
python create_ground_truth.py --video video2.mp4
python evaluate_single_video.py --video video2.mp4

# 3. 세 번째 비디오
python create_ground_truth.py --video video3.mp4
python evaluate_single_video.py --video video3.mp4
```

각 비디오마다 독립적인 결과 폴더가 생성됩니다.

### 여러 결과 비교하기

```bash
# 각 결과 폴더에서 results.csv를 확인
cat experiment_results/single_video_*/evaluation/results.csv
```

또는 Excel/Python으로 여러 CSV를 통합하여 분석할 수 있습니다.

---

## 💡 팁 & 트러블슈팅

### GT 생성 팁

1. **적절한 키프레임 개수**
   - 너무 적으면: Recall 낮음, 중요 장면 누락
   - 너무 많으면: 평가 기준이 애매해짐
   - 권장: 전체 프레임의 1-3%

2. **일관성 유지**
   - 여러 비디오의 GT를 만들 때 동일한 기준 적용
   - 예: "장면 전환", "주요 동작", "객체 등장" 등

### 평가 시 주의사항

1. **모델 가중치 확인**
   ```bash
   # yolo11m.pt, rtdetr-l.pt 파일이 있는지 확인
   ls -lh *.pt
   ```

2. **CUDA/GPU 사용 가능 여부**
   - GPU가 있으면 훨씬 빠름
   - CPU로도 실행 가능하지만 느림

3. **디스크 공간**
   - 키프레임 이미지가 많이 저장되므로 충분한 공간 필요
   - 1개 비디오 평가 시 약 100-500MB 사용

### 에러 해결

**에러: "Ground truth not found"**
```bash
# GT를 먼저 생성했는지 확인
python create_ground_truth.py --video my_video.mp4
```

**에러: "Cannot open video"**
```bash
# 비디오 경로가 올바른지 확인
ls -lh my_video.mp4
```

**에러: "Model file not found"**
```bash
# 모델 가중치를 다운로드
# YOLO 모델은 첫 실행 시 자동 다운로드됨
```

---

## 📈 결과 활용

### Excel에서 보기

1. `results.csv`를 Excel로 열기
2. 표로 정리하여 비교

### Python으로 시각화

```python
import pandas as pd
import matplotlib.pyplot as plt

# CSV 로드
df = pd.read_csv('experiment_results/single_video_*/evaluation/results.csv')

# F1-Score 비교
df.plot(x='method', y='f1_score_tol15', kind='bar')
plt.title('F1-Score Comparison (tolerance=15)')
plt.ylabel('F1-Score')
plt.show()
```

### 논문/보고서용

- `summary.txt`를 복사하여 보고서에 첨부
- `results.csv`에서 테이블 생성
- 그래프: F1-Score, Precision, Recall 비교

---

## ✅ 체크리스트

평가 전:
- [ ] 비디오 파일 준비
- [ ] GT 생성 완료
- [ ] 모델 가중치 확인
- [ ] 디스크 공간 확인

평가 후:
- [ ] `results.csv` 확인
- [ ] F1-Score 지표 검토
- [ ] 각 방법의 키프레임 개수 확인
- [ ] 압축률이 적절한지 확인

---

**빠른 시작:**
```bash
python create_ground_truth.py --video my_video.mp4
python evaluate_single_video.py --video my_video.mp4
cat experiment_results/single_video_*/evaluation/summary.txt
```
