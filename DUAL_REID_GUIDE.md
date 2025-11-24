# Dual Re-ID System Guide

## 시스템 개요

이 시스템은 **사람(Person)**과 **차량/사물(Vehicle/Object)**을 구분하여 각각에 최적화된 Re-ID 모델을 적용하는 **Dual Re-ID** 방식을 사용합니다.

## Re-ID 모델 구성

### 사람(Person) Re-ID 모델

YOLO가 탐지한 객체 중 **Class ID = 0 (Person)**인 경우 사용됩니다.

| 모델 키 | 모델 이름 | 성능 (Market1501) | 특징 |
|---------|-----------|-------------------|------|
| `fastreid_agw_r101` | FastReID AGW-R101-IBN | Rank@1: 95.5%, mAP: 89.5% | ⭐ SOTA 성능, Attention 메커니즘 |
| `fastreid_mgn_r50` | FastReID MGN-R50-IBN | Rank@1: 95.8%, mAP: 89.8% | ⭐ Multi-Granularity 특징 |
| `fastreid_sbs_r50` | FastReID SBS-R50 | Rank@1: 95.4%, mAP: 88.2% | Strong Baseline |
| `fastreid_sbs_r101` | FastReID SBS-R101-IBN | Rank@1: 96.3%, mAP: 90.3% | 가장 높은 정확도 (무거움) |
| `osnet_market1501` | OSNet Market-1501 | - | TorchReID 기반 경량 모델 |
| `osnet_x1_0` | OSNet x1.0 | - | 범용 경량 모델 |

### 차량/사물(Vehicle/Object) Re-ID 모델

YOLO가 탐지한 객체 중 **Class ID ≠ 0 (차량, 가방, 기타 사물)**인 경우 사용됩니다.

| 모델 키 | 모델 이름 | 성능 | 특징 |
|---------|-----------|------|------|
| `fastreid_veri_sbs_r50` | FastReID VeRi SBS-R50 | VeRi-776: Rank@1 97.0%, mAP 81.9% | ⭐ SOTA for VeRi |
| `fastreid_vehicleid_bot_r50` | FastReID VehicleID BoT-R50 | VehicleID: Rank@1 86.6% | ⭐ 대규모 데이터셋 학습 |
| `fastreid_veri_agw_r50` | FastReID VeRi AGW-R50 | - | Alternative SOTA |
| `osnet_x1_0` | OSNet x1.0 | - | 범용 경량 모델 |
| `resnet50_fc512` | ResNet50 Baseline | - | 기본 베이스라인 |

## 작동 방식

### 1. 객체 탐지 (YOLO)
```
비디오 프레임 → YOLO → 객체 탐지 (bbox + class)
```

### 2. Re-ID 모델 선택
```python
if object_class == 0:  # Person
    selected_model = person_reid_model  # 예: fastreid_agw_r101
else:  # Vehicle/Object
    selected_model = vehicle_reid_model  # 예: fastreid_veri_sbs_r50
```

### 3. 특징 추출 및 매칭
```
객체 crop → Re-ID 모델 → 특징 벡터 (512-dim or 2048-dim)
→ Cosine Similarity 계산 → 동일 객체 판단
```

## Docker 환경 설정

### 1. Docker 이미지 빌드

**최초 빌드 또는 requirements.txt 변경 시:**
```bash
docker build --no-cache -t kabs-enhance .
```

**일반 빌드 (캐시 사용):**
```bash
docker build -t kabs-enhance .
```

### 2. Docker 컨테이너 실행

**GPU 사용 (권장):**
```bash
docker run -it --gpus all --ipc=host \
  -e PYTHONIOENCODING=utf-8 \
  -e LANG=C.UTF-8 \
  -e LC_ALL=C.UTF-8 \
  -v "${PWD}:/workspace" \
  kabs-enhance
```

**CPU만 사용:**
```bash
docker run -it \
  -e PYTHONIOENCODING=utf-8 \
  -e LANG=C.UTF-8 \
  -e LC_ALL=C.UTF-8 \
  -v "${PWD}:/workspace" \
  kabs-enhance
```

**Windows PowerShell에서:**
```powershell
docker run -it --gpus all --ipc=host -e PYTHONIOENCODING=utf-8 -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 -v "${PWD}:/workspace" kabs-enhance
```

**또는 줄 바꿈 사용 (백틱 ` 사용):**
```powershell
docker run -it --gpus all --ipc=host `
  -e PYTHONIOENCODING=utf-8 `
  -e LANG=C.UTF-8 `
  -e LC_ALL=C.UTF-8 `
  -v "${PWD}:/workspace" `
  kabs-enhance
```

## 실행 명령어

### 1. 단일 모델 조합 실행

```bash
python yolo_osnet_4_dual_reid.py \
  --video_path v_2.mp4 \
  --output_folder output/single_test \
  --person_reid_model_key fastreid_agw_r101 \
  --vehicle_reid_model_key fastreid_veri_sbs_r50
```

### 2. 모든 모델 비교 실험

**사람 모델만 비교:**
```bash
python compare_reid_models.py \
  --video v_2.mp4 \
  --output output/results_person \
  --test-person
```

**차량 모델만 비교:**
```bash
python compare_reid_models.py \
  --video v_2.mp4 \
  --output output/results_vehicle \
  --test-vehicle
```

**모든 모델 조합 비교:**
```bash
python compare_reid_models.py \
  --video v_2.mp4 \
  --output output/results_full_2 \
  --test-all
```

## 결과 확인

### 출력 폴더 구조
```
output/results_full/
├── person_fastreid_agw_r101/
│   ├── 1_primary_selection_frames/
│   ├── 2_profile_tracking_frames/
│   ├── 3_after_reid_frames/
│   └── keyframe_summary_unified.json
├── person_fastreid_mgn_r50/
├── vehicle_fastreid_veri_sbs_r50/
└── comparison_results.json  ← 전체 비교 결과
```

### 비교 결과 (comparison_results.json)
```json
{
  "person_models": {
    "fastreid_agw_r101": {
      "num_keyframes": 15,
      "runtime_seconds": 45.2
    },
    "fastreid_mgn_r50": {
      "num_keyframes": 14,
      "runtime_seconds": 42.8
    }
  },
  "vehicle_models": {
    "fastreid_veri_sbs_r50": {
      "num_keyframes": 8,
      "runtime_seconds": 38.5
    }
  }
}
```

## 모델 선택 가이드

### 사람 Re-ID 추천

| 상황 | 추천 모델 | 이유 |
|------|-----------|------|
| 최고 정확도 필요 | `fastreid_sbs_r101` | 96.3% Rank@1 (가장 높음) |
| 균형잡힌 성능 | `fastreid_agw_r101` | 95.5% Rank@1, Attention 메커니즘 |
| 다양한 포즈/각도 | `fastreid_mgn_r50` | Multi-Granularity 특징 |
| 빠른 처리 속도 | `osnet_x1_0` | 경량 모델 |

### 차량 Re-ID 추천

| 상황 | 추천 모델 | 이유 |
|------|-----------|------|
| 도심 CCTV 환경 | `fastreid_veri_sbs_r50` | VeRi-776 데이터셋 학습 (CCTV 특화) |
| 대규모 차량 식별 | `fastreid_vehicleid_bot_r50` | 13,000+ 차량 ID 학습 |
| 빠른 처리 속도 | `osnet_x1_0` | 경량 범용 모델 |

## 문제 해결

### 인코딩 오류 (cp949)
Docker 실행 시 환경 변수 설정 필수:
```bash
-e PYTHONIOENCODING=utf-8 -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8
```

### GPU 메모리 부족
- 경량 모델 사용: `osnet_x1_0`
- 또는 배치 크기 조정

### 모델 가중치 다운로드 실패
- 인터넷 연결 확인
- 수동 다운로드 후 프로젝트 루트에 배치

## 추가 정보

- 모델 레지스트리: `reid_model_registry.py`
- 메인 파이프라인: `yolo_osnet_4_dual_reid.py`
- 비교 스크립트: `compare_reid_models.py`
- 상세 가이드: `REID_MODELS_GUIDE.md`
