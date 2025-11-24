# Ground Truth Annotation Tool 사용 가이드

OpenCV 기반의 간단한 키프레임 선택 도구입니다.

## 🚀 빠른 시작

```bash
# 기본 사용법
python create_ground_truth.py --video path/to/your_video.mp4

# 출력 폴더 지정
python create_ground_truth.py --video my_video.mp4 --output custom_output_folder
```

## ⌨️ 키보드 단축키

| 키 | 기능 |
|---|---|
| **SPACE** | 현재 프레임을 키프레임으로 선택 |
| **→ (오른쪽 화살표)** | 다음 프레임 (1 frame) |
| **← (왼쪽 화살표)** | 이전 프레임 (1 frame) |
| **↑ (위 화살표)** | 30 프레임 앞으로 이동 |
| **↓ (아래 화살표)** | 30 프레임 뒤로 이동 |
| **D** | 마지막 선택한 키프레임 삭제 |
| **R** | 모든 키프레임 초기화 |
| **S** | 저장하고 계속 작업 |
| **P** | 재생/일시정지 |
| **Q or ESC** | 저장하고 종료 |
| **H** | 도움말 표시 |

## 📊 화면 정보

도구를 실행하면 비디오 위에 다음 정보가 표시됩니다:

```
Frame: 150/5000 | Time: 5.00s | ★ KEYFRAME ★
Selected Keyframes: 12 | Compression: 0.24%
Mode: PAUSED | Press H for help
━━━━━━━━━━━━━━━━━━━━━━━━ [진행 바] ━━━━━━━━━━━━━━━━━━━━━━
```

- **현재 프레임 번호 및 시간**
- **선택된 키프레임 개수**
- **압축률** (선택된 키프레임 / 전체 프레임)
- **진행 바**: 현재 위치 및 키프레임 위치 표시 (노란색 마커)

## 💾 저장 형식

키프레임은 다음과 같이 저장됩니다:

```
Keyframe-extraction/Dataset/Keyframe/
└── your_video.mp4/
    ├── 150.jpg          # 프레임 150
    ├── 570.jpg          # 프레임 570
    ├── 810.jpg          # 프레임 810
    ├── 1350.jpg         # 프레임 1350
    └── keyframes.json   # 메타데이터
```

### keyframes.json 형식

```json
{
  "video_name": "your_video.mp4",
  "video_path": "/path/to/your_video.mp4",
  "total_frames": 5000,
  "fps": 30.0,
  "keyframes": [150, 570, 810, 1350],
  "num_keyframes": 4,
  "compression_ratio": 0.0008
}
```

## 🔄 작업 흐름 예시

### 1. 새로운 비디오 작업

```bash
python create_ground_truth.py --video my_video.mp4
```

1. 비디오가 열립니다
2. **P**를 눌러 재생하며 비디오를 확인
3. 중요한 장면에서 **SPACE**를 눌러 키프레임 선택
4. 또는 **화살표 키**로 프레임을 하나씩 이동하며 정밀하게 선택
5. 실수로 선택했다면 **D**를 눌러 삭제
6. 작업 중간에 **S**를 눌러 저장 (중간 저장)
7. 완료되면 **Q**를 눌러 저장하고 종료

### 2. 기존 GT 편집

이미 생성된 GT가 있다면 자동으로 불러옵니다:

```bash
python create_ground_truth.py --video my_video.mp4
# ✓ Loaded 12 existing keyframes  ← 기존 키프레임 로드
```

- 추가 키프레임을 선택하거나
- **D**로 불필요한 키프레임 삭제
- **R**로 전체 초기화하고 처음부터 다시 시작

### 3. 배치 작업

여러 비디오를 순차적으로 작업:

```bash
python create_ground_truth.py --video video1.mp4
# 작업 완료 후 종료
python create_ground_truth.py --video video2.mp4
# 작업 완료 후 종료
...
```

## 💡 팁

### 키프레임 선택 가이드

1. **Scene Change**: 장면이 바뀌는 순간
2. **중요 객체 등장**: 새로운 사람/물체가 등장
3. **주요 동작**: 중요한 동작이나 이벤트
4. **다양성 확보**: 비슷한 프레임 반복 선택 방지

### 효율적인 작업 방법

1. **1단계 - 빠른 탐색**
   - **P**로 재생하며 전체 비디오 파악
   - 중요 장면을 머릿속으로 기억

2. **2단계 - 정밀 선택**
   - **↑/↓**로 30프레임씩 빠르게 이동
   - 중요 장면에서 **→/←**로 정밀 조정
   - **SPACE**로 최적 프레임 선택

3. **3단계 - 검증**
   - 진행 바의 노란색 마커로 키프레임 분포 확인
   - 너무 밀집되지 않았는지 체크
   - 압축률 확인 (보통 1-5% 권장)

### 추천 압축률

- **매우 상세**: 3-5% (100프레임당 3-5개)
- **일반**: 1-3% (100프레임당 1-3개)
- **간략**: 0.5-1% (100프레임당 0.5-1개)

## 🛠️ 문제 해결

### 비디오가 열리지 않을 때

```bash
# OpenCV가 지원하는 코덱인지 확인
python -c "import cv2; print(cv2.getBuildInformation())"
```

### 키보드가 반응하지 않을 때

- 비디오 창이 활성화되어 있는지 확인 (창 클릭)
- 터미널이 아닌 OpenCV 창에서 키 입력

### 저장이 안 될 때

- 출력 폴더에 쓰기 권한이 있는지 확인
- 디스크 공간이 충분한지 확인

## 🔗 평가 실험과 연동

생성한 GT는 바로 평가 실험에 사용할 수 있습니다:

```bash
# 1. GT 생성
python create_ground_truth.py --video Keyframe-extraction/Dataset/Videos/my_video.mp4

# 2. 평가 실험 실행
python run_experiments.py
# 자동으로 새로 만든 GT를 인식하고 평가에 사용
```

## 📝 예제

### 예제 1: 짧은 비디오 (1분)

```bash
python create_ground_truth.py --video short_video.mp4
```

- 30 fps × 60초 = 1,800 프레임
- 권장 키프레임: 18-54개 (1-3%)
- 예상 소요 시간: 5-10분

### 예제 2: 긴 비디오 (10분)

```bash
python create_ground_truth.py --video long_video.mp4
```

- 30 fps × 600초 = 18,000 프레임
- 권장 키프레임: 180-540개 (1-3%)
- 예상 소요 시간: 30-60분
- **팁**: 중간중간 **S**로 저장하며 작업

## ✅ 체크리스트

작업 완료 후 확인사항:

- [ ] 모든 중요 장면이 포함되었는가?
- [ ] 키프레임들이 골고루 분포되어 있는가?
- [ ] 압축률이 적절한가? (1-3% 권장)
- [ ] keyframes.json이 정상적으로 생성되었는가?
- [ ] 이미지 파일들이 모두 저장되었는가?

---

**문의사항이나 버그 발견 시 이슈를 남겨주세요!**
