import cv2
import torch
import json
import os
import sys
import numpy as np
from ultralytics import YOLO
from PIL import Image
from collections import defaultdict
from scipy.spatial.distance import cosine
from typing import List, Dict, Tuple, Set
import itertools
import copy
import matplotlib.pyplot as plt
from torchvision import transforms

# ============================================================================
# Re-ID Model Registry
# ============================================================================
try:
    from reid_model_registry import (
        load_reid_model,
        list_available_models,
        get_model_info,
        PERSON_REID_MODELS,
        VEHICLE_REID_MODELS
    )
    REID_REGISTRY_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: reid_model_registry not found. Using legacy loading.")
    REID_REGISTRY_AVAILABLE = False

# ============================================================================
# TorchReID
# ============================================================================
try:
    from torchreid import models, utils
    TORCHREID_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: torchreid library not found. Re-ID will be disabled.")
    print("   Install with: pip install torchreid")
    TORCHREID_AVAILABLE = False

# ============================================================================
# DPT Depth Estimation
# ============================================================================
try:
    from transformers import DPTImageProcessor, DPTForDepthEstimation
    DPT_AVAILABLE = True
except ImportError:
    print("⚠️ Warning: transformers library not found. Depth estimation will be disabled.")
    print("   Install with: pip install transformers")
    DPT_AVAILABLE = False

# ============================================================================
# PART 0-A: 필터 설정 처리 함수
# ============================================================================
def process_filter_config(filter_mode, filter_classes, model):
    """
    filter_mode와 filter_classes를 included_classes, excluded_classes로 변환
    
    Args:
        filter_mode (int): 1=블랙리스트, 2=화이트리스트, 3=필터링없음
        filter_classes (list): 클래스 리스트 (문자열 또는 숫자 혼용 가능)
        model: YOLO/RT-DETR 모델 객체
        
    Returns:
        tuple: (included_classes, excluded_classes)
            - filter_mode=1: (None, excluded_classes)
            - filter_mode=2: (included_classes, None)
            - filter_mode=3: (None, None)
    
    Raises:
        ValueError: filter_mode가 1, 2, 3이 아닌 경우
    """
    # 1. filter_mode 검증
    if filter_mode not in [1, 2, 3]:
        raise ValueError(
            f"❌ 오류: filter_mode는 1, 2, 3 중 하나여야 합니다.\n"
            f"   - 1: 블랙리스트 (특정 클래스 제외)\n"
            f"   - 2: 화이트리스트 (특정 클래스만 포함)\n"
            f"   - 3: 필터링 없음 (모든 클래스 포함)\n"
            f"   현재 입력값: {filter_mode}"
        )
    
    # 2. filter_mode=3 (필터링 없음)인 경우
    if filter_mode == 3:
        if filter_classes:
            print(f"⚠️ 경고: filter_mode=3 (필터링 없음)이므로 filter_classes={filter_classes}는 무시됩니다.")
        print("✓ 필터링 모드: 모든 클래스 포함 (필터링 없음)")
        return None, None
    
    # 3. filter_mode=1 또는 2인데 filter_classes가 비어있는 경우
    if not filter_classes:
        mode_name = "블랙리스트" if filter_mode == 1 else "화이트리스트"
        print(f"⚠️ 경고: filter_mode={filter_mode} ({mode_name})인데 filter_classes가 비어있습니다.")
        print("   → 필터링 없음으로 자동 전환합니다.")
        return None, None
    
    # 4. 클래스 이름 매핑 생성 (숫자 → 이름)
    class_names = model.names  # dict: {0: 'person', 1: 'bicycle', ...}
    
    # 5. filter_classes의 숫자를 이름으로 변환
    processed_classes = []
    for item in filter_classes:
        if isinstance(item, int):
            # 숫자인 경우 클래스 이름으로 변환
            if item in class_names:
                class_name = class_names[item]
                processed_classes.append(class_name)
                print(f"   - 숫자 {item} → '{class_name}' 변환")
            else:
                print(f"⚠️ 경고: 클래스 ID {item}는 존재하지 않습니다. 무시합니다.")
                print(f"   사용 가능한 클래스 ID: 0 ~ {len(class_names)-1}")
        elif isinstance(item, str):
            # 문자열인 경우 검증 후 추가
            if item in class_names.values():
                processed_classes.append(item)
            else:
                print(f"⚠️ 경고: 클래스 이름 '{item}'는 존재하지 않습니다. 무시합니다.")
                print(f"   --list-classes 옵션으로 사용 가능한 클래스를 확인하세요.")
        else:
            print(f"⚠️ 경고: '{item}'는 문자열 또는 정수가 아닙니다. 무시합니다.")
    
    # 6. 중복 제거
    processed_classes = list(set(processed_classes))
    
    # 7. 최종 검증
    if not processed_classes:
        mode_name = "블랙리스트" if filter_mode == 1 else "화이트리스트"
        print(f"⚠️ 경고: 유효한 클래스가 없습니다. ({mode_name} 모드)")
        print("   → 필터링 없음으로 자동 전환합니다.")
        return None, None
    
    # 8. filter_mode에 따라 분기
    if filter_mode == 1:
        # 블랙리스트: excluded_classes 설정
        print(f"✓ 필터링 모드: 블랙리스트 (다음 클래스 제외)")
        print(f"   제외 클래스: {processed_classes}")
        return None, processed_classes
    elif filter_mode == 2:
        # 화이트리스트: included_classes 설정
        print(f"✓ 필터링 모드: 화이트리스트 (다음 클래스만 포함)")
        print(f"   포함 클래스: {processed_classes}")
        return processed_classes, None


# ============================================================================
# PART 0-B: DPT 모델 로딩 및 Depth 추정 함수
# ============================================================================
def load_dpt_model(model_name="Intel/dpt-large"):
    """
    DPT 모델 로드
    
    Args:
        model_name: 사용할 DPT 모델
            - "Intel/dpt-large": 고해상도, 느림
            - "Intel/dpt-hybrid-midas": 중간 성능
            - "Intel/dpt-swinv2-tiny-256": 빠름, 저해상도
    
    Returns:
        processor, model, device
    """
    if not DPT_AVAILABLE:
        print("⚠️ DPT model cannot be loaded. Skipping depth estimation.")
        return None, None, None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📊 Loading DPT model: {model_name}")
    
    try:
        processor = DPTImageProcessor.from_pretrained(model_name)
        model = DPTForDepthEstimation.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        print(f"✓ DPT model loaded successfully on {device}")
        return processor, model, device
    except Exception as e:
        print(f"✗ Error loading DPT model: {e}")
        return None, None, None


def estimate_depth(frame, processor, model, device):
    """
    프레임의 depth map 추정
    
    Args:
        frame: BGR 이미지 (OpenCV format)
        processor: DPT processor
        model: DPT model
        device: cuda or cpu
    
    Returns:
        depth_map: normalized depth map (0-255)
        depth_array: raw depth values (float)
    """
    if processor is None or model is None:
        return None, None
    
    try:
        # BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Preprocess
        inputs = processor(images=image_pil, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Post-process
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        )
        
        # Convert to numpy
        depth_array = prediction.squeeze().cpu().numpy()
        
        # Normalize to 0-255 for visualization
        depth_normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())
        depth_map = (depth_normalized * 255).astype(np.uint8)
        
        return depth_map, depth_array
        
    except Exception as e:
        print(f"⚠️ Error in depth estimation: {e}")
        return None, None


def extract_object_depth_info(depth_array, box, percentiles=[25, 50, 75], 
                             use_foreground_only=True, threshold_method='otsu',
                             save_mask=False, mask_save_path=None):
    """
    객체 영역의 depth 통계 추출 (히스토그램 기반 전경/배경 분리)
    
    Args:
        depth_array: raw depth values
        box: [x1, y1, x2, y2]
        percentiles: 추출할 백분위수
        use_foreground_only: True면 전경(객체)만 사용, False면 전체 사용
        threshold_method: 'otsu' (Otsu's method) or 'mean' (평균값 기준)
        save_mask: 마스크를 저장할지 여부
        mask_save_path: 마스크 저장 경로
    
    Returns:
        depth_info: dict with depth statistics (전경 depth 정보 포함)
    """
    if depth_array is None:
        return None
    
    try:
        x1, y1, x2, y2 = map(int, box)
        
        # Boundary check
        h, w = depth_array.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Extract object region
        object_depth = depth_array[y1:y2, x1:x2]
        
        if object_depth.size == 0:
            return None
        
        # Normalize to 0-255 for histogram/thresholding
        object_depth_norm = ((object_depth - object_depth.min()) / 
                            (object_depth.max() - object_depth.min() + 1e-8) * 255).astype(np.uint8)
        
        # Calculate threshold
        if threshold_method == 'otsu':
            # Otsu's method로 최적 임계값 찾기
            threshold, _ = cv2.threshold(object_depth_norm, 0, 255, 
                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold_method == 'mean':
            # 평균값을 임계값으로 사용
            threshold = np.mean(object_depth_norm)
        else:
            threshold = 128  # default
        
        # Create foreground mask (depth가 threshold보다 큰 값 = 더 가까운 객체)
        # Depth는 보통 가까울수록 큰 값이므로, threshold보다 큰 값을 전경으로 간주
        foreground_mask = object_depth_norm >= threshold
        
        # 전경 픽셀 수 확인
        foreground_pixel_count = np.sum(foreground_mask)
        total_pixel_count = object_depth.size
        foreground_ratio = foreground_pixel_count / total_pixel_count if total_pixel_count > 0 else 0
        
        # 마스크 시각화 저장 (옵션)
        if save_mask and mask_save_path:
            mask_vis = np.zeros_like(object_depth_norm)
            mask_vis[foreground_mask] = 255
            cv2.imwrite(mask_save_path, mask_vis)
        
        # 전경만 사용할지 전체를 사용할지 결정
        if use_foreground_only and foreground_pixel_count > 0:
            depth_values = object_depth[foreground_mask]
            depth_type = "foreground_only"
        else:
            depth_values = object_depth.flatten()
            depth_type = "all_pixels"
        
        if depth_values.size == 0:
            return None
        
        # Calculate statistics
        depth_info = {
            "segmentation_method": threshold_method,
            "threshold_value": float(threshold),
            "depth_type": depth_type,
            "foreground_pixel_count": int(foreground_pixel_count),
            "total_pixel_count": int(total_pixel_count),
            "foreground_ratio": float(foreground_ratio),
            "mean": float(np.mean(depth_values)),
            "median": float(np.median(depth_values)),
            "std": float(np.std(depth_values)),
            "min": float(np.min(depth_values)),
            "max": float(np.max(depth_values)),
            "percentiles": {
                f"p{p}": float(np.percentile(depth_values, p))
                for p in percentiles
            }
        }
        
        # 배경 통계도 추가 (비교용)
        if use_foreground_only and foreground_pixel_count > 0:
            background_mask = ~foreground_mask
            background_pixel_count = np.sum(background_mask)
            if background_pixel_count > 0:
                background_values = object_depth[background_mask]
                depth_info["background_stats"] = {
                    "mean": float(np.mean(background_values)),
                    "median": float(np.median(background_values)),
                    "pixel_count": int(background_pixel_count)
                }
        
        return depth_info
        
    except Exception as e:
        print(f"⚠️ Error extracting depth info: {e}")
        return None


def visualize_depth_map(depth_map, output_path):
    """
    Depth map을 컬러맵으로 시각화하여 저장
    
    Args:
        depth_map: normalized depth map (0-255)
        output_path: 저장 경로
    """
    if depth_map is None:
        return
    
    try:
        # Apply colormap (closer = warmer colors)
        depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
        cv2.imwrite(output_path, depth_colored)
    except Exception as e:
        print(f"⚠️ Error saving depth visualization: {e}")


# ============================================================================
# PART 0.5: 객체 필터링 함수 (NEW)
# ============================================================================
def should_include_detection(det, included_classes=None, excluded_classes=None):
    """
    detection이 포함되어야 하는지 판단
    
    Args:
        det: detection 딕셔너리 (class_name 포함)
        included_classes: 포함할 클래스 리스트 (None = 모두 포함)
        excluded_classes: 제외할 클래스 리스트 (None = 제외 없음)
    
    Returns:
        bool: True면 포함, False면 제외
    """
    class_name = det.get('class_name', '')
    
    # 화이트리스트 우선 (지정된 것만 포함)
    if included_classes is not None:
        return class_name in included_classes
    
    # 블랙리스트 (지정된 것 제외)
    if excluded_classes is not None:
        return class_name not in excluded_classes
    
    # 둘 다 None이면 모두 포함
    return True


def print_available_classes(detection_model):
    """
    탐지 모델이 인식 가능한 모든 클래스 출력
    """
    class_names = detection_model.names
    print("\n" + "="*80)
    print("📋 AVAILABLE CLASSES FOR DETECTION")
    print("="*80)
    print(f"Total classes: {len(class_names)}\n")
    
    # 클래스를 4열로 정렬하여 출력
    classes_list = [f"{idx}: {name}" for idx, name in class_names.items()]
    
    for i in range(0, len(classes_list), 4):
        row = classes_list[i:i+4]
        print("  ".join(f"{item:<20}" for item in row))
    
    print("="*80)
    print("💡 TIP: Copy class names from above to use in 'excluded_classes' config")
    print("="*80 + "\n")
    
    return class_names


def list_classes_only(model_path, model_type="yolo"):
    """
    모델의 클래스 목록만 출력하는 독립 함수 (파이프라인 실행 없이)
    
    Usage:
        python yolo_osnet_4_with_filtering.py --list-classes
    """
    print(f"\n🔍 Loading {model_type.upper()} model: {model_path}")
    detection_model = YOLO(model_path)
    
    print_available_classes(detection_model)
    
    print("\n✅ Use these class names in your config's 'excluded_classes' or 'included_classes'")
    print("Example:")
    print('  "excluded_classes": ["person", "car", "bicycle"]')
    print('  "included_classes": ["cell phone", "laptop", "bottle"]\n')


# ============================================================================
# PART 1 & 2: 모델 로딩, 프레임 선택, Re-ID (TorchReID 기반, ORB 제거)
# ============================================================================
def load_dual_reid_models(person_model_key="osnet_market1501", vehicle_model_key="osnet_x1_0",
                           person_model_path=None, vehicle_model_path=None, device="cuda"):
    """
    Load dual Re-ID models: one for person, one for vehicle/objects
    Now supports model selection from registry!

    Args:
        person_model_key: Model key from PERSON_REID_MODELS registry (e.g., "osnet_market1501", "osnet_x1_0")
        vehicle_model_key: Model key from VEHICLE_REID_MODELS registry (e.g., "osnet_veri776", "osnet_x1_0")
        person_model_path: Optional path to override registry (legacy support)
        vehicle_model_path: Optional path to override registry (legacy support)
        device: cuda or cpu

    Returns:
        person_reid_model, vehicle_reid_model
    """
    if not TORCHREID_AVAILABLE:
        print("⚠️ TorchReID not available. Dual Re-ID disabled.")
        return None, None

    person_model, vehicle_model = None, None

    # Use registry if available
    if REID_REGISTRY_AVAILABLE:
        print(f"\n{'='*80}")
        print("Loading Re-ID Models from Registry")
        print(f"{'='*80}")

        # Load Person Re-ID Model
        try:
            person_model, person_info = load_reid_model(
                model_key=person_model_key,
                model_type="person",
                device=device,
                local_weights_path=person_model_path
            )
        except Exception as e:
            print(f"✗ Error loading person Re-ID model: {e}")
            person_model = None

        # Load Vehicle Re-ID Model
        try:
            vehicle_model, vehicle_info = load_reid_model(
                model_key=vehicle_model_key,
                model_type="vehicle",
                device=device,
                local_weights_path=vehicle_model_path
            )
        except Exception as e:
            print(f"✗ Error loading vehicle Re-ID model: {e}")
            vehicle_model = None

        print(f"{'='*80}\n")

    else:
        # Legacy loading (original code)
        print(f"\n📊 Loading Person Re-ID Model (Legacy)")
        try:
            if person_model_path and os.path.exists(person_model_path):
                checkpoint = torch.load(person_model_path, map_location=device)
                person_model = models.build_model(name='osnet_x1_0', num_classes=751)  # Market-1501
                person_model = person_model.to(device)
                person_model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
                person_model.classifier = None  # Extract features only
                person_model.eval()
                print(f"✓ Person Re-ID model loaded from: {person_model_path}")
            else:
                # Fallback to pretrained
                person_model = models.build_model(name='osnet_x1_0', num_classes=0)
                person_model = person_model.to(device)
                utils.load_pretrained_weights(person_model, 'osnet_x1_0')
                person_model.eval()
                print(f"✓ Person Re-ID model loaded (pretrained)")
        except Exception as e:
            print(f"✗ Error loading person Re-ID model: {e}")
            person_model = None

        # Load Vehicle Re-ID Model
        print(f"\n📊 Loading Vehicle Re-ID Model (Legacy)")
        try:
            if vehicle_model_path and os.path.exists(vehicle_model_path):
                checkpoint = torch.load(vehicle_model_path, map_location=device)
                # Try VeRi num_classes (776) or generic
                vehicle_model = models.build_model(name='osnet_x1_0', num_classes=776)  # VeRi-776
                vehicle_model = vehicle_model.to(device)
                vehicle_model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
                vehicle_model.classifier = None  # Extract features only
                vehicle_model.eval()
                print(f"✓ Vehicle Re-ID model loaded from: {vehicle_model_path}")
            else:
                # Fallback to pretrained (same as person for now)
                vehicle_model = models.build_model(name='osnet_x1_0', num_classes=0)
                vehicle_model = vehicle_model.to(device)
                utils.load_pretrained_weights(vehicle_model, 'osnet_x1_0')
                vehicle_model.eval()
                print(f"✓ Vehicle Re-ID model loaded (pretrained)")
        except Exception as e:
            print(f"✗ Error loading vehicle Re-ID model: {e}")
            vehicle_model = None

    return person_model, vehicle_model


def load_models(model_type="yolo", model_path=None,
                person_reid_model_key="osnet_market1501", vehicle_reid_model_key="osnet_x1_0",
                person_reid_model_path=None, vehicle_reid_model_path=None,
                use_depth=True, dpt_model_name="Intel/dpt-large", use_dual_reid=True):
    """
    Load detection model (YOLO or RT-DETR), Dual TorchReID models, and optionally DPT models

    Args:
        model_type: "yolo" or "rt_detr"
        model_path: Path to detection model weights
        person_reid_model_key: Model key from PERSON_REID_MODELS registry
        vehicle_reid_model_key: Model key from VEHICLE_REID_MODELS registry
        person_reid_model_path: Optional path to override registry
        vehicle_reid_model_path: Optional path to override registry
        use_depth: Enable depth estimation
        dpt_model_name: DPT model name
        use_dual_reid: Enable dual Re-ID system
    """
    if model_path is None:
        raise ValueError("model_path must be specified for the detection model.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load detection model (YOLO or RT-DETR via Ultralytics)
    print(f"\n📊 Loading detection model: {model_type.upper()} from {model_path}")
    if model_type.lower() == "yolo":
        detection_model = YOLO(model_path)
    elif model_type.lower() == "rt_detr":
        detection_model = YOLO(model_path)  # Ultralytics YOLO supports RT-DETR
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'yolo' or 'rt_detr'.")

    # Load Dual Re-ID models
    if use_dual_reid:
        person_reid_model, vehicle_reid_model = load_dual_reid_models(
            person_model_key=person_reid_model_key,
            vehicle_model_key=vehicle_reid_model_key,
            person_model_path=person_reid_model_path,
            vehicle_model_path=vehicle_reid_model_path,
            device=device
        )
    else:
        person_reid_model, vehicle_reid_model = None, None

    class_names = detection_model.names

    # Load DPT if requested
    if use_depth:
        dpt_processor, dpt_model, dpt_device = load_dpt_model(dpt_model_name)
    else:
        dpt_processor, dpt_model, dpt_device = None, None, None

    return detection_model, person_reid_model, vehicle_reid_model, device, class_names, dpt_processor, dpt_model, dpt_device

def primary_selection(video_path, detection_model, class_names, frame_skip_interval=1,
                     included_classes=None, excluded_classes=None, tracker="botsort.yaml"):
    """Primary selection: group frames by object set changes (with frame skipping and class filtering)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): raise ValueError("Error opening video file")

    frame_idx, prev_track_set, groups, current_group = 0, set(), [], []
    skip_counter = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Frame skipping logic
        skip_counter += 1
        if skip_counter < frame_skip_interval:
            frame_idx += 1
            continue
        skip_counter = 0

        results = detection_model.track(frame, persist=True, verbose=False, tracker=tracker)
        current_track_set = set(results[0].boxes.id.cpu().numpy()) if results[0].boxes.id is not None else set()
        if current_track_set != prev_track_set:
            if current_group: groups.append(current_group)
            current_group = []
        if results[0].boxes.id is not None:
            # 필터링 적용
            detections = []
            for b, c, l, t in zip(results[0].boxes.xyxy.cpu(), results[0].boxes.conf.cpu(), 
                                 results[0].boxes.cls.cpu(), results[0].boxes.id.cpu()):
                det = {
                    'box': b.tolist(), 
                    'conf': float(c), 
                    'cls': int(l), 
                    'class_name': class_names[int(l)], 
                    'track_id': int(t)
                }
                # 필터링 체크
                if should_include_detection(det, included_classes, excluded_classes):
                    detections.append(det)
            
            if detections:  # 필터링 후 남은 detection이 있을 때만 추가
                current_group.append((frame_idx, frame, detections))
        prev_track_set = current_track_set
        frame_idx += 1
    if current_group: groups.append(current_group)
    cap.release()
    return [group[len(group) // 2] for group in groups if group]

def calculate_combined_histogram(frame, weight_brightness=0.5, weight_saturation=0.5):
    """
    [NEW] 결합 히스토그램 계산: Grayscale brightness + HSV S saturation (50:50 기본)
    
    Args:
        frame: BGR 이미지
        weight_brightness: 밝기 가중치 (기본 0.5)
        weight_saturation: 채도 가중치 (기본 0.5)
    
    Returns:
        combined_hist: 정규화된 결합 히스토그램 (flatten)
    """
    # Brightness (Grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness_hist = cv2.normalize(cv2.calcHist([gray], [0], None, [256], [0, 256]), None).flatten()
    
    # Saturation (HSV S 채널)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]
    saturation_hist = cv2.normalize(cv2.calcHist([s_channel], [0], None, [256], [0, 256]), None).flatten()
    
    # 결합
    combined_hist = weight_brightness * brightness_hist + weight_saturation * saturation_hist
    return combined_hist

def bhattacharyya_distance(hist1, hist2):
    bc = np.sum(np.sqrt(hist1 * hist2))
    return -np.log(bc) if bc > 0 else float('inf')

def profile_tracking(selected_frames, window_size=5, hist_threshold=0.3, 
                     weight_brightness=0.5, weight_saturation=0.5, iterations=1):
    """
    [MODIFIED] Profile Tracking: 결합 히스토그램 사용 + 가중치 파라미터 추가 + 반복 iteration 지원 + Bidirectional Window Size 실제 적용
    """
    if not selected_frames: return []
    print(f"Using combined histogram: Brightness {weight_brightness*100:.0f}% + Saturation {weight_saturation*100:.0f}%")
    print(f"Profile Tracking iterations: {iterations}, window_size: {window_size} (±{window_size//2} frames, bidirectional)")
    
    current_frames = selected_frames
    for iter_num in range(iterations):
        print(f"  Iteration {iter_num + 1}/{iterations}")
        frame_histograms = [(f_idx, f, d, calculate_combined_histogram(f, weight_brightness, weight_saturation), 
                             set(det['track_id'] for det in d)) for f_idx, f, d in current_frames]
        keep_flags = [True] * len(frame_histograms)
        half_window = window_size // 2
        for i in range(len(frame_histograms)):
            if not keep_flags[i]: continue
            _, _, _, hist_i, track_ids_i = frame_histograms[i]
            # Bidirectional Window 적용: i - half_window ~ i + half_window (자신 제외)
            start_j = max(0, i - half_window)
            end_j = min(len(frame_histograms), i + half_window + 1)
            for j in range(start_j, end_j):
                if j == i or not keep_flags[j]: continue  # 자신 & 이미 제거된 스킵
                _, _, _, hist_j, track_ids_j = frame_histograms[j]
                if bhattacharyya_distance(hist_i, hist_j) < hist_threshold:
                    if track_ids_i.issubset(track_ids_j):
                        keep_flags[i] = False
                        break
                    elif track_ids_j.issubset(track_ids_i):
                        keep_flags[j] = False
        current_frames = [current_frames[i] for i, keep in enumerate(keep_flags) if keep]
        print(f"    -> Remaining frames after iteration {iter_num + 1}: {len(current_frames)}")
    
    return current_frames

def extract_torchreid_features_dual(crop_bgr, object_class, person_reid_model, vehicle_reid_model, device):
    """
    Dual Re-ID feature extraction: route to appropriate model based on object class

    Args:
        crop_bgr: BGR format object crop image
        object_class: COCO class ID (0=person, others=vehicle/object)
        person_reid_model: Person Re-ID model
        vehicle_reid_model: Vehicle Re-ID model
        device: cuda or cpu

    Returns:
        features: Feature vector (numpy array, 512-dim or 2048-dim)
    """
    # Select model based on class
    if object_class == 0:  # Person class in COCO
        selected_model = person_reid_model
        model_type = "person"
    else:  # Vehicle or other objects
        selected_model = vehicle_reid_model
        model_type = "vehicle"

    if selected_model is None:
        return None

    try:
        # BGR to RGB
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)

        # Check if model is FastReID
        is_fastreid = False
        if hasattr(selected_model, '__class__') and selected_model.__class__.__name__ == 'FastReIDWrapper':
            is_fastreid = True

        if is_fastreid:
            # FastReID Preprocessing: Resize -> Tensor (0-255)
            # FastReID models typically expect 0-255 inputs and normalize internally
            transform = transforms.Compose([
                transforms.Resize((256, 128)),
                transforms.ToTensor(), # Converts to [0, 1]
            ])
            # Convert back to [0, 255] for FastReID
            tensor = transform(crop_pil).unsqueeze(0).to(device) * 255.0
        else:
            # TorchReID Preprocessing: Resize -> Tensor -> Normalize
            transform = transforms.Compose([
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            tensor = transform(crop_pil).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            features = selected_model(tensor)
            # FastReID might return a tensor on GPU
            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy().flatten()
            else:
                # Fallback if it returns something else (though wrapper ensures tensor)
                features = np.array(features).flatten()

        return features
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None


def extract_torchreid_features(crop_bgr, torchreid_model, device):
    """
    TorchReID 모델로 객체 크롭의 특징 벡터 추출
    (Legacy function for backward compatibility)

    Args:
        crop_bgr: BGR 형식 객체 크롭 이미지
        torchreid_model: TorchReID 모델
        device: cuda or cpu

    Returns:
        features: 특징 벡터 (numpy array)
    """
    if torchreid_model is None:
        return None

    try:
        # BGR to RGB
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)

        # TorchReID transforms (standard for ReID)
        transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Preprocess
        tensor = transform(crop_pil).unsqueeze(0).to(device)

        # Inference (now returns 512-dim features since classifier is None)
        with torch.no_grad():
            features = torchreid_model(tensor)
        
        return features.squeeze().cpu().numpy()
    except Exception as e:
        print(f"⚠️ Error extracting TorchReID features: {e}")
        return None

def unified_torchreid_reid_and_frame_selection_dual(filtered_frames, person_reid_model, vehicle_reid_model, device,
                                                     reid_threshold=0.8,
                                                     frame_merge_threshold=0.9, min_samples=2,
                                                     included_classes=None, excluded_classes=None):
    """
    [DUAL RE-ID] TorchReID 기반 Dual Re-ID 및 프레임 선택 (person/vehicle 분리)
    """
    if person_reid_model is None and vehicle_reid_model is None:
        print("⚠️ No Re-ID models available. Skipping Re-ID and returning filtered frames.")
        return [(f_idx, f, d) for f_idx, f, d in filtered_frames]

    all_frame_data = []
    for frame_idx, frame, detections in filtered_frames:
        frame_objects = []
        for det in detections:
            # 필터링 체크
            if not should_include_detection(det, included_classes, excluded_classes):
                continue

            box = det['box']
            crop_bgr = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

            # Dual Re-ID: use class-specific model
            object_class = det['cls']
            reid_feature = extract_torchreid_features_dual(
                crop_bgr, object_class, person_reid_model, vehicle_reid_model, device
            )
            if reid_feature is None:
                continue  # Skip if feature extraction fails

            frame_objects.append({
                'class_name': det['class_name'],
                'track_id': det['track_id'],
                'original_track_id': det['track_id'],
                'reid_feature': reid_feature,
                'crop_bgr': crop_bgr,
                'conf': det['conf'],
                'box': det['box'],
                'cls': det['cls']
            })

        if frame_objects:  # 필터링 후 남은 객체가 있을 때만 추가
            all_frame_data.append({'frame_idx': frame_idx, 'frame': frame, 'objects': frame_objects})
    
    objects_by_class = defaultdict(list)
    for frame_data in all_frame_data:
        for obj in frame_data['objects']: objects_by_class[obj['class_name']].append(obj)
    
    global_id_mapping = {}
    for class_name, objects in objects_by_class.items():
        if len(objects) < min_samples: continue
        clusters, assigned = [], set()
        for i, obj_i in enumerate(objects):
            if i in assigned: continue
            cluster, assigned_in_cluster = [i], {i}
            for j, obj_j in enumerate(objects):
                if j in assigned or j in assigned_in_cluster: continue
                reid_sim = 1 - cosine(obj_i['reid_feature'], obj_j['reid_feature'])
                if reid_sim > reid_threshold:
                    cluster.append(j); assigned_in_cluster.add(j)
            assigned.update(assigned_in_cluster)
            clusters.append(cluster)
        for cluster_id, cluster in enumerate(clusters, 1):
            for obj_idx in cluster:
                key = (objects[obj_idx]['class_name'], objects[obj_idx]['track_id'])
                if key not in global_id_mapping: global_id_mapping[key] = f"{class_name}_{cluster_id}"
    
    for frame_data in all_frame_data:
        for obj in frame_data['objects']:
            key = (obj['class_name'], obj['original_track_id'])
            obj['track_id'] = global_id_mapping.get(key, f"unclustered_{obj['original_track_id']}")
    
    unique_frames, merged = [], set()
    for i, frame_data_i in enumerate(all_frame_data):
        if i in merged: continue
        objects_i = {(obj['class_name'], obj['track_id']): obj['reid_feature'] for obj in frame_data_i['objects']}
        for j in range(i + 1, len(all_frame_data)):
            if j in merged: continue
            objects_j = {(obj['class_name'], obj['track_id']): obj['reid_feature'] for obj in all_frame_data[j]['objects']}
            common_keys = set(objects_i.keys()) & set(objects_j.keys())
            if not common_keys or len(objects_i) != len(objects_j): continue
            if sum(1 for key in common_keys if 1 - cosine(objects_i[key], objects_j[key]) > frame_merge_threshold) / len(objects_i) > 0.95:
                merged.add(j)
        detections = [{'box': o['box'], 'conf': o['conf'], 'cls': o['cls'], 'class_name': o['class_name'], 'track_id': o['track_id'], 'original_track_id': o['original_track_id']} for o in frame_data_i['objects']]
        unique_frames.append((frame_data_i['frame_idx'], frame_data_i['frame'], detections))
    return unique_frames

# ============================================================================
# PART 3: Greedy Selection
# ============================================================================
def extract_object_key(det: Dict, use_instance_id: bool = False) -> str:
    if use_instance_id and ("track_id" in det):
        return str(det['track_id'])
    else:
        return str(det.get('class_name'))

def image_object_set(detections: List[Dict], use_instance_id: bool = False, score_thresh: float = 0.0,
                    included_classes=None, excluded_classes=None) -> Set[str]:
    return {
        extract_object_key(d, use_instance_id) 
        for d in detections 
        if d.get("conf", 1.0) >= score_thresh
        and should_include_detection(d, included_classes, excluded_classes)
    }

def combos_from_set(objset: Set[str], k: int) -> Set[Tuple[str, ...]]:
    if len(objset) < k:
        return set()
    return {tuple(sorted(c)) for c in itertools.combinations(objset, k)}

def greedy_coverage_selection(
    frames_data: List[Tuple], k: int = 2, use_instance_id: bool = False, 
    max_selected: int = None, min_new_combos: int = 1, score_thresh: float = 0.0,
    included_classes=None, excluded_classes=None
) -> (List[Dict], Dict): 
    n = len(frames_data)
    frame_objsets = [image_object_set(d, use_instance_id, score_thresh, included_classes, excluded_classes) 
                     for _, _, d in frames_data]
    frame_combos = [combos_from_set(o, k) for o in frame_objsets]
    greedy_log = {
        "parameters": {
            "k": k, "use_instance_id": use_instance_id, "max_selected": max_selected,
            "min_new_combos": min_new_combos, "score_thresh": score_thresh
        },
        "initial_state": {},
        "iterations": [],
        "final_result": {}
    }
    total_unique_combos = set.union(*frame_combos) if frame_combos else set()
    greedy_log["initial_state"]["total_unique_combos"] = sorted([list(c) for c in total_unique_combos])
    greedy_log["initial_state"]["total_unique_count"] = len(total_unique_combos)
    
    selected = []
    covered_combos: Set[Tuple[str, ...]] = set()
    remaining = set(range(n))
    iteration_count = 0
    
    print(f"\n{'='*60}\nStarting Greedy Coverage Selection with Logging\n{'='*60}")
    
    while remaining:
        iteration_count += 1
        iteration_log = {
            "iteration": iteration_count,
            "covered_before_count": len(covered_combos),
            "covered_before": sorted([list(c) for c in covered_combos]),
            "candidates": []
        }
        best_idx = -1
        best_new_count = -1
        best_new_set = set()
        candidate_details = []
        for i in remaining:
            new_combos = frame_combos[i] - covered_combos
            new_count = len(new_combos)
            candidate_details.append({
                "candidate_index": i,
                "original_frame_idx": frames_data[i][0],
                "new_count": new_count,
                "new_combos_offered": sorted([list(c) for c in new_combos])
            })
            if new_count > best_new_count:
                best_new_count = new_count
                best_new_set = new_combos
                best_idx = i
        
        iteration_log["candidates"] = sorted(candidate_details, key=lambda x: x['new_count'], reverse=True)
        if best_idx == -1 or best_new_count < min_new_combos:
            iteration_log["selection"] = f"STOP: Best new count ({best_new_count}) is less than min_new_combos ({min_new_combos})."
            greedy_log["iterations"].append(iteration_log)
            break
        
        original_frame_idx = frames_data[best_idx][0]
        iteration_log["selection"] = {
            "selected_index": best_idx,
            "original_frame_idx": original_frame_idx,
            "new_count_added": best_new_count,
            "new_combos_added": sorted([list(c) for c in best_new_set])
        }
        
        frame_idx, frame, detections = frames_data[best_idx]
        selected.append({
            "frame_idx": frame_idx, "frame": frame, "detections": copy.deepcopy(detections),
            "new_combos_added": [list(c) for c in best_new_set],
            "new_count": best_new_count,
        })
        
        print(f"Iteration {iteration_count}: Selected Frame {original_frame_idx} | New Pairs: {best_new_count}")
        
        covered_combos.update(best_new_set)
        remaining.remove(best_idx)
        
        iteration_log["covered_after_count"] = len(covered_combos)
        iteration_log["covered_after"] = sorted([list(c) for c in covered_combos])
        greedy_log["iterations"].append(iteration_log)
        
        if max_selected is not None and len(selected) >= max_selected:
            iteration_log["selection"]["stop_reason"] = f"Reached max_selected limit of {max_selected}."
            break
            
    final_coverage = len(covered_combos)
    total_coverage = len(total_unique_combos)
    coverage_percentage = (final_coverage / total_coverage * 100) if total_coverage > 0 else 0

    greedy_log["final_result"] = {
        "selected_frame_indices": [s['frame_idx'] for s in selected],
        "total_frames_selected": len(selected),
        "final_covered_combos_count": final_coverage,
        "total_unique_combos_count": total_coverage,
        "final_coverage_percentage": f"{coverage_percentage:.2f}%"
    }

    print(f"\nGreedy Selection Complete! Selected {len(selected)} frames.")
    print(f"Covered {final_coverage} out of {total_coverage} unique combinations ({coverage_percentage:.2f}%).\n{'='*60}\n")
    
    return selected, greedy_log

# ============================================================================
# PART 3.5: Post-Greedy Profile Tracking (새 추가)
# ============================================================================
def post_greedy_profile_tracking(final_selected, hist_threshold=0.25, 
                                 weight_brightness=0.5, weight_saturation=0.5, window_size=3, iterations=1):
    """
    [NEW] Greedy 후 추가 Profile Tracking: 유사 프레임 재필터링 (iterations 및 window_size 지원)
    """
    if not final_selected: 
        return [], {"removed_count": 0, "removed_frames": []}
    
    print(f"Post-Greedy Filtering: Combined histogram (Brightness {weight_brightness*100:.0f}% + Saturation {weight_saturation*100:.0f}%), threshold={hist_threshold}, window_size={window_size}, iterations={iterations}")
    
    # final_selected를 Tuple 형식으로 변환 (frame_idx, frame, detections)
    converted_frames = [(item['frame_idx'], item['frame'], item['detections']) for item in final_selected]
    
    # profile_tracking 로직 재사용 (window_size 및 iterations 전달)
    filtered_frames = profile_tracking(converted_frames, window_size=window_size, 
                                       hist_threshold=hist_threshold,
                                       weight_brightness=weight_brightness, 
                                       weight_saturation=weight_saturation,
                                       iterations=iterations)
    
    # 원본 final_selected와 매칭해 필터링된 인덱스 찾기 (frame_idx로만 비교)
    filtered_frame_indices = {frame_idx for frame_idx, _, _ in filtered_frames}
    ultra_filtered = [item for item in final_selected if item['frame_idx'] in filtered_frame_indices]
    
    removed_count = len(final_selected) - len(ultra_filtered)
    removed_frames = [item['frame_idx'] for item in final_selected if item['frame_idx'] not in filtered_frame_indices]
    
    reduction_log = {
        "removed_count": removed_count,
        "removed_frames": removed_frames,
        "reduction_ratio": removed_count / len(final_selected) if final_selected else 0
    }
    
    print(f"-> Removed {removed_count} frames (ratio: {reduction_log['reduction_ratio']:.2f}). Remaining: {len(ultra_filtered)}")
    return ultra_filtered, reduction_log

# ============================================================================
# PART 4: 시각화 및 기본 저장 함수
# ============================================================================

def draw_detections(frame, detections, use_original_id=False):
    """프레임에 탐지 결과를 그리는 범용 함수"""
    frame_copy = frame.copy()
    
    track_ids_in_frame = []
    if detections:
        key_to_use = 'original_track_id' if use_original_id and 'original_track_id' in detections[0] else 'track_id'
        track_ids_in_frame = [det[key_to_use] for det in detections if key_to_use in det]

    unique_ids = sorted(list(set(track_ids_in_frame)))
    
    if not unique_ids:
        colors = [plt.cm.get_cmap('hsv', 20)(i) for i in range(20)]
        color_map = {i: (int(c[0]*255), int(c[1]*255), int(c[2]*255)) for i,c in enumerate(colors)}
    else:
        colors = [plt.cm.get_cmap('hsv', len(unique_ids))(i) for i in range(len(unique_ids))]
        color_map = {uid: (int(c[0]*255), int(c[1]*255), int(c[2]*255)) for uid, c in zip(unique_ids, colors)}

    for i, det in enumerate(detections):
        box = det['box']
        id_key = 'original_track_id' if use_original_id and 'original_track_id' in det else 'track_id'
        
        label = str(det.get(id_key, f"Obj-{i}"))
        color_key = det.get(id_key, i % 20)
        
        color = color_map.get(color_key, (0, 0, 255))
        
        cv2.rectangle(frame_copy, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(frame_copy, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
    return frame_copy

def save_frames_as_images(frames_data, output_folder, prefix="", use_original_id=False):
    """키프레임 목록을 개별 이미지 파일로 저장하는 함수"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    print(f"\nSaving frames to '{output_folder}'...")
    for frame_idx, frame, detections in frames_data:
        drawn_frame = draw_detections(frame, detections, use_original_id)
        output_path = os.path.join(output_folder, f"{prefix}frame_{frame_idx}.jpg")
        cv2.imwrite(output_path, drawn_frame)
    print(f"✓ Saved {len(frames_data)} frames.")

def create_video_from_source(video_path, detection_model, class_names, output_path, use_tracking=True,
                            included_classes=None, excluded_classes=None, tracker="botsort.yaml"):
    """원본 비디오를 처리하여 추적 적용/미적용 비디오를 생성하는 함수 (필터링 추가)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"\nCreating video: '{os.path.basename(output_path)}' (Tracking: {use_tracking})...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if use_tracking:
            results = detection_model.track(frame, persist=True, verbose=False, tracker=tracker)
            if results[0].boxes is not None and results[0].boxes.id is not None:
                detections = []
                for i, (b, t) in enumerate(zip(results[0].boxes.xyxy.cpu(), results[0].boxes.id.cpu())):
                    det = {
                        'box': b.tolist(), 
                        'track_id': int(t),
                        'class_name': class_names[int(results[0].boxes.cls[i])]
                    }
                    if should_include_detection(det, included_classes, excluded_classes):
                        detections.append(det)
            else:
                detections = []
        else:
            results = detection_model.predict(frame, verbose=False)
            if results[0].boxes is not None:
                detections = []
                for i, b in enumerate(results[0].boxes.xyxy.cpu()):
                    det = {
                        'box': b.tolist(),
                        'class_name': class_names[int(results[0].boxes.cls[i])]
                    }
                    if should_include_detection(det, included_classes, excluded_classes):
                        detections.append(det)
            else:
                detections = []
        
        drawn_frame = draw_detections(frame, detections)
        out.write(drawn_frame)
        
    cap.release()
    out.release()
    print(f"✓ Video saved successfully to: {output_path}")

# ============================================================================
# PART 4-1: Depth 정보를 포함한 JSON 저장 함수들 (간결 구조)
# ============================================================================

def save_json_unified(final_selected, greedy_log, output_folder, 
                      dpt_processor=None, dpt_model=None, dpt_device=None,
                      use_foreground_only=True, threshold_method='otsu',
                      histogram_method="brightness_saturation_combined",
                      hist_weights={"brightness": 0.5, "saturation": 0.5},
                      post_filter=None):
    """
    [MODIFIED] unified 모드: 하나의 JSON 파일로 모든 핵심 정보 저장 (PLAN 기반 간결 구조)
    """
    output_path = os.path.join(output_folder, "keyframe_summary_unified.json")
    
    print(f"\n{'='*60}\nSaving unified JSON (core info only)...\n{'='*60}")
    
    # Depth 추정 및 데이터 추출
    frames_data = []
    depth_enabled = dpt_model is not None
    for idx, frame_data in enumerate(final_selected):
        frame_idx = frame_data['frame_idx']
        frame = frame_data['frame']
        
        # Depth 추정 (필요 시)
        depth_map, depth_array = None, None
        if depth_enabled:
            depth_map, depth_array = estimate_depth(frame, dpt_processor, dpt_model, dpt_device)
        
        # 프레임 데이터 구성 (간결화)
        frame_info = {
            "frame_index": frame_idx,
            "selection_order": idx + 1,
            "new_combinations_count": frame_data.get('new_count', 0),
            "detections": []
        }
        
        # Detection 간결화 + foreground depth (mean, std만)
        for det in frame_data['detections']:
            det_info = {
                "class_name": det.get('class_name', 'unknown'),
                "track_id": det.get('track_id', 'unknown'),
                "bounding_box": {
                    "x1": det['box'][0],
                    "y1": det['box'][1],
                    "x2": det['box'][2],
                    "y2": det['box'][3]
                }
            }
            
            # Foreground depth 추가 (depth_enabled 시)
            if depth_enabled and depth_array is not None:
                object_depth_info = extract_object_depth_info(
                    depth_array, det['box'],
                    use_foreground_only=use_foreground_only,
                    threshold_method=threshold_method,
                    save_mask=False  # unified 모드라 마스크 저장 안 함
                )
                if object_depth_info and object_depth_info['depth_type'] == 'foreground_only':
                    det_info["foreground_depth"] = {
                        "mean": object_depth_info['mean'],
                        "std": object_depth_info['std']
                    }
            
            frame_info['detections'].append(det_info)
        
        frames_data.append(frame_info)
    
    # 전체 요약 JSON 구성
    coverage_percentage = greedy_log['final_result'].get('final_coverage_percentage', '0%')
    unified_json = {
        "metadata": {
            "total_frames_selected": len(final_selected),
            "coverage_percentage": coverage_percentage,
            "depth_enabled": depth_enabled,
            "threshold_method": threshold_method if depth_enabled else None,
            "use_foreground_only": use_foreground_only if depth_enabled else None,
            "histogram_method": histogram_method,
            "hist_weights": hist_weights,
            "post_filter": post_filter or {"applied": False},
            "parameters": greedy_log.get('parameters', {})
        },
        "frames": frames_data,
        "explanations": {
            "frame_index": "원본 비디오의 프레임 인덱스 (0부터 시작, 선택된 키프레임의 위치).",
            "selection_order": "Greedy selection에서 선택된 순서 (1부터 시작, 마지막 선택된 프레임일수록 높음).",
            "new_combinations_count": "이 프레임이 추가로 커버한 객체 쌍(combination)의 개수 (k=2 기준).",
            "class_name": "YOLO가 감지한 객체 클래스 (e.g., 'person', 'cell phone').",
            "track_id": "Re-ID 후의 고유 트랙 ID (클래스명_숫자 형식, 동일 객체 추적용).",
            "bounding_box.x1": "박스 왼쪽 상단 x 좌표 (픽셀 단위).",
            "bounding_box.y1": "박스 왼쪽 상단 y 좌표 (픽셀 단위).",
            "bounding_box.x2": "박스 오른쪽 하단 x 좌표 (픽셀 단위).",
            "bounding_box.y2": "박스 오른쪽 하단 y 좌표 (픽셀 단위).",
            "foreground_depth.mean": "객체 전경 영역의 depth 평균값 (DPT 모델 기준, 가까울수록 값이 큼).",
            "foreground_depth.std": "객체 전경 영역의 depth 표준편차 (depth 분포의 퍼짐 정도).",
            "histogram_method": "프레임 필터링에 사용된 히스토그램 방법 (brightness + saturation 결합).",
            "post_filter": "Greedy 후 추가 유사 프레임 필터링 단계."
        }
    }
    
    # 저장
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(unified_json, f, indent=2, ensure_ascii=False)
        print(f"✓ Successfully saved unified JSON to: {output_path}")
    except Exception as e:
        print(f"✗ Error saving unified JSON: {e}")


def save_json_per_frame(final_selected, greedy_log, output_folder, 
                        dpt_processor=None, dpt_model=None, dpt_device=None,
                        save_depth_visualization=True, save_foreground_masks=True,
                        use_foreground_only=True, threshold_method='otsu',
                        histogram_method="brightness_saturation_combined",
                        hist_weights={"brightness": 0.5, "saturation": 0.5},
                        post_filter=None):
    """
    [MODIFIED] per_frame 모드: 각 프레임별로 개별 JSON 저장 + 간결 구조 (PLAN 기반)
    """
    # 개별 프레임 폴더 생성
    per_frame_folder = os.path.join(output_folder, "per_frame_json")
    if not os.path.exists(per_frame_folder):
        os.makedirs(per_frame_folder)
    
    # Depth visualization 폴더 (옵션)
    if save_depth_visualization and dpt_model is not None:
        depth_vis_folder = os.path.join(output_folder, "depth_visualizations")
        if not os.path.exists(depth_vis_folder):
            os.makedirs(depth_vis_folder)
    
    # Foreground mask 폴더 (옵션)
    if save_foreground_masks and dpt_model is not None:
        mask_folder = os.path.join(output_folder, "foreground_masks")
        if not os.path.exists(mask_folder):
            os.makedirs(mask_folder)
    
    print(f"\n{'='*60}\nSaving per-frame JSON files (core info only)...\n{'='*60}")
    
    # Depth 추정 및 데이터 추출 (per_frame용)
    depth_enabled = dpt_model is not None
    explanations = {
        "frame_index": "원본 비디오의 프레임 인덱스 (0부터 시작, 선택된 키프레임의 위치).",
        "selection_order": "Greedy selection에서 선택된 순서 (1부터 시작, 마지막 선택된 프레임일수록 높음).",
        "new_combinations_count": "이 프레임이 추가로 커버한 객체 쌍(combination)의 개수 (k=2 기준).",
        "class_name": "YOLO가 감지한 객체 클래스 (e.g., 'person', 'cell phone').",
        "track_id": "Re-ID 후의 고유 트랙 ID (클래스명_숫자 형식, 동일 객체 추적용).",
        "bounding_box.x1": "박스 왼쪽 상단 x 좌표 (픽셀 단위).",
        "bounding_box.y1": "박스 왼쪽 상단 y 좌표 (픽셀 단위).",
        "bounding_box.x2": "박스 오른쪽 하단 x 좌표 (픽셀 단위).",
        "bounding_box.y2": "박스 오른쪽 하단 y 좌표 (픽셀 단위).",
        "foreground_depth.mean": "객체 전경 영역의 depth 평균값 (DPT 모델 기준, 가까울수록 값이 큼).",
        "foreground_depth.std": "객체 전경 영역의 depth 표준편차 (depth 분포의 퍼짐 정도).",
        "histogram_method": "프레임 필터링에 사용된 히스토그램 방법 (brightness + saturation 결합).",
        "post_filter": "Greedy 후 추가 유사 프레임 필터링 단계."
    }
    
    # 각 프레임별 JSON 저장
    for idx, frame_data in enumerate(final_selected):
        frame_idx = frame_data['frame_idx']
        frame = frame_data['frame']
        
        # Depth 추정
        depth_map, depth_array = None, None
        if depth_enabled:
            print(f"  📊 Estimating depth for frame {frame_idx}...")
            depth_map, depth_array = estimate_depth(frame, dpt_processor, dpt_model, dpt_device)
            
            # Depth visualization 저장
            if save_depth_visualization and depth_map is not None:
                depth_vis_path = os.path.join(depth_vis_folder, f"depth_frame_{frame_idx}.jpg")
                visualize_depth_map(depth_map, depth_vis_path)
        
        # 프레임 메타데이터 구성 (간결화)
        frame_metadata = {
            "frame_index": frame_idx,
            "selection_order": idx + 1,
            "new_combinations_count": frame_data.get('new_count', 0),
            "depth_available": depth_array is not None,
            "detections": []
        }
        
        # Detection 정보 추가 + Depth 정보 (간결)
        for det_idx, det in enumerate(frame_data['detections']):
            # 객체별 depth 정보 추출 (전경/배경 분리)
            object_depth_info = None
            if depth_array is not None:
                # 마스크 저장 경로 설정 (옵션)
                mask_save_path = None
                if save_foreground_masks:
                    mask_save_path = os.path.join(
                        mask_folder, 
                        f"mask_frame_{frame_idx}_obj_{det_idx}_{det.get('track_id', 'unknown')}.jpg"
                    )
                
                object_depth_info = extract_object_depth_info(
                    depth_array, det['box'],
                    use_foreground_only=use_foreground_only,
                    threshold_method=threshold_method,
                    save_mask=save_foreground_masks,
                    mask_save_path=mask_save_path
                )
            
            det_info = {
                "class_name": det.get('class_name', 'unknown'),
                "track_id": det.get('track_id', 'unknown'),
                "bounding_box": {
                    "x1": det['box'][0],
                    "y1": det['box'][1],
                    "x2": det['box'][2],
                    "y2": det['box'][3]
                }
            }
            
            # Foreground depth 추가 (간결)
            if object_depth_info and object_depth_info['depth_type'] == 'foreground_only':
                det_info["foreground_depth"] = {
                    "mean": object_depth_info['mean'],
                    "std": object_depth_info['std']
                }
            
            frame_metadata['detections'].append(det_info)
        
        # 개별 프레임 JSON 저장 (explanations 포함)
        frame_json_path = os.path.join(per_frame_folder, f"frame_{frame_idx}_metadata.json")
        frame_with_explain = {
            "metadata": {
                "frame_info": frame_metadata,
                "histogram_method": histogram_method,
                "hist_weights": hist_weights,
                "post_filter": post_filter or {"applied": False}
            },
            "explanations": explanations
        }
        try:
            with open(frame_json_path, 'w', encoding='utf-8') as f:
                json.dump(frame_with_explain, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Saved: frame_{frame_idx}_metadata.json")
        except Exception as e:
            print(f"  ✗ Error saving frame {frame_idx}: {e}")
    
    # Summary JSON 생성 및 저장 (간결)
    coverage_percentage = greedy_log['final_result'].get('final_coverage_percentage', '0%')
    summary = {
        "total_frames_selected": len(final_selected),
        "selected_frame_indices": [f['frame_idx'] for f in final_selected],
        "depth_enabled": depth_enabled,
        "threshold_method": threshold_method if depth_enabled else None,
        "use_foreground_only": use_foreground_only if depth_enabled else None,
        "histogram_method": histogram_method,
        "hist_weights": hist_weights,
        "post_filter": post_filter or {"applied": False},
        "coverage_statistics": {
            "final_covered_combos_count": greedy_log['final_result'].get('final_covered_combos_count', 0),
            "total_unique_combos_count": greedy_log['final_result'].get('total_unique_combos_count', 0),
            "coverage_percentage": coverage_percentage
        },
        "frame_details": [
            {
                "frame_index": f['frame_idx'],
                "selection_order": idx + 1,
                "new_combos_count": f.get('new_count', 0),
                "num_detections": len(f['detections'])
            }
            for idx, f in enumerate(final_selected)
        ],
        "explanations": explanations
    }
    
    summary_path = os.path.join(per_frame_folder, "summary.json")
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n  ✓ Successfully saved summary.json")
        print(f"  ✓ Total {len(final_selected)} per-frame JSON files saved in: {per_frame_folder}")
        if save_depth_visualization and dpt_model is not None:
            print(f"  ✓ Depth visualizations saved in: {depth_vis_folder}")
        if save_foreground_masks and dpt_model is not None:
            print(f"  ✓ Foreground masks saved in: {mask_folder}")
    except Exception as e:
        print(f"  ✗ Error saving summary: {e}")


def save_json_results(final_selected, greedy_log, output_folder, save_mode="both",
                      dpt_processor=None, dpt_model=None, dpt_device=None,
                      save_depth_visualization=True, save_foreground_masks=True,
                      use_foreground_only=True, threshold_method='otsu',
                      histogram_method="brightness_saturation_combined",
                      hist_weights={"brightness": 0.5, "saturation": 0.5},
                      post_filter=None):
    """
    [MODIFIED] JSON 저장 통합 함수 - 기존 모드에서 간결 구조로 변경 (PLAN 기반)
    """
    print(f"\n{'='*60}\nSaving JSON results (mode: {save_mode})\n{'='*60}")
    
    if save_mode in ["unified", "both"]:
        save_json_unified(
            final_selected, greedy_log, output_folder,
            dpt_processor, dpt_model, dpt_device,
            use_foreground_only, threshold_method,
            histogram_method, hist_weights, post_filter
        )
    
    if save_mode in ["per_frame", "both"]:
        save_json_per_frame(
            final_selected, greedy_log, output_folder,
            dpt_processor, dpt_model, dpt_device,
            save_depth_visualization, save_foreground_masks,
            use_foreground_only, threshold_method,
            histogram_method, hist_weights, post_filter
        )
    
    print(f"{'='*60}\n")


# ============================================================================
# PART 5: Main Pipeline (Post-Filter 추가 + 객체 필터링 추가)
# ============================================================================

def main(video_path, output_folder, json_save_mode="both", model_type="yolo", model_path=None,
         use_dual_reid=True,
         person_reid_model_key="osnet_market1501", vehicle_reid_model_key="osnet_x1_0",
         person_reid_model_path=None, vehicle_reid_model_path=None,
         torchreid_model_path=None,  # legacy single model
         create_comparison_videos=True,  # 비교 비디오 생성 여부
         use_depth=True, dpt_model_name="Intel/dpt-large",
         save_depth_visualization=True, save_foreground_masks=True,
         use_foreground_only=True, threshold_method='otsu',
         hist_weight_brightness=0.5, hist_weight_saturation=0.5,
         apply_post_filter=True, post_hist_threshold=0.25, post_window_size=3, post_profile_iterations=1,
         frame_skip_interval=1,
         profile_iterations=1,
         show_available_classes=True,  # NEW: 클래스 목록 표시 여부
         filter_mode=3,  # NEW: 필터링 모드 (1=블랙리스트, 2=화이트리스트, 3=필터링없음)
         filter_classes=None,  # NEW: 필터링할 클래스 리스트 (문자열 또는 숫자)
         tracker="botsort.yaml",  # NEW: Tracker configuration
         **kwargs):
    """
    [MODIFIED] 메인 파이프라인 - Dual Re-ID with Model Registry + Depth 추정 + 히스토그램 결합 + Post-Filter 추가

    Args:
        ... (기존) + 새 파라미터:
        use_dual_reid: True 시 Dual Re-ID (person/vehicle 분리), False 시 단일 모델
        person_reid_model_key: Person Re-ID 모델 키 (e.g., "osnet_market1501", "osnet_x1_0", "resnet50_fc512")
        vehicle_reid_model_key: Vehicle Re-ID 모델 키 (e.g., "osnet_veri776", "osnet_x1_0")
        person_reid_model_path: Optional path to override registry
        vehicle_reid_model_path: Optional path to override registry
        show_available_classes: True 시 탐지 가능한 클래스 목록 출력
        filter_mode: 필터링 모드 (1=블랙리스트, 2=화이트리스트, 3=필터링없음)
        filter_classes: 필터링할 클래스 리스트 (문자열 또는 숫자 혼용 가능)
    """
    
    # 단계별 결과를 저장할 하위 폴더 경로 재구성
    video_out_folder = os.path.join(output_folder, "0_source_videos")
    primary_select_folder = os.path.join(output_folder, "1_primary_selection_frames")
    profile_track_folder = os.path.join(output_folder, "2_profile_tracking_frames")
    after_reid_folder = os.path.join(output_folder, "3_after_reid_frames")
    final_folder = os.path.join(output_folder, "4_final_greedy_frames")

    for folder in [output_folder, primary_select_folder, 
                   profile_track_folder, after_reid_folder, final_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    if create_comparison_videos:
        if not os.path.exists(video_out_folder):
            os.makedirs(video_out_folder)
    
    # Load models (Dual TorchReID + DPT)
    print("\n" + "="*80 + "\nSTEP 0: LOADING MODELS\n" + "="*80)
    print(f"Using Dual Re-ID: {use_dual_reid}")
    if use_dual_reid:
        print(f"Person Re-ID Model Key: {person_reid_model_key}")
        print(f"Vehicle Re-ID Model Key: {vehicle_reid_model_key}")

    detection_model, person_reid_model, vehicle_reid_model, device, class_names, dpt_processor, dpt_model, dpt_device = load_models(
        model_type=model_type,
        model_path=model_path,
        person_reid_model_key=person_reid_model_key,
        vehicle_reid_model_key=vehicle_reid_model_key,
        person_reid_model_path=person_reid_model_path,
        vehicle_reid_model_path=vehicle_reid_model_path,
        use_depth=use_depth,
        dpt_model_name=dpt_model_name,
        use_dual_reid=use_dual_reid
    )

    # For backward compatibility with legacy code that uses single model
    if not use_dual_reid and torchreid_model_path:
        # Load single model as fallback
        print("⚠️ Using legacy single Re-ID model")
        torchreid_model = person_reid_model  # Use person model as single model
    else:
        torchreid_model = None
    
    # 🆕 필터 설정 처리 (filter_mode와 filter_classes를 included_classes, excluded_classes로 변환)
    print("\n" + "="*80 + "\nPROCESSING FILTER CONFIGURATION\n" + "="*80)
    
    # filter_classes가 None이면 빈 리스트로 처리
    if filter_classes is None:
        filter_classes = []
    
    # 필터 설정 처리 함수 호출
    included_classes, excluded_classes = process_filter_config(
        filter_mode=filter_mode,
        filter_classes=filter_classes,
        model=detection_model
    )
    
    # 🆕 클래스 목록 출력 (선택사항)
    if show_available_classes:
        print("\n" + "="*80 + "\nAVAILABLE CLASSES\n" + "="*80)
        print_available_classes(detection_model)
        print("="*80 + "\n")
    
    # STEP 0: 추적(Tracking) 비교 영상 생성 (옵션)
    if create_comparison_videos:
        print("\n" + "="*80 + "\nCREATING COMPARISON VIDEOS\n" + "="*80)
        no_tracking_video_path = os.path.join(video_out_folder, "video_no_tracking.mp4")
        create_video_from_source(video_path, detection_model, class_names, no_tracking_video_path,
                                use_tracking=False, included_classes=included_classes, excluded_classes=excluded_classes, tracker=tracker)

        tracking_video_path = os.path.join(video_out_folder, "video_with_tracking.mp4")
        create_video_from_source(video_path, detection_model, class_names, tracking_video_path,
                                use_tracking=True, included_classes=included_classes, excluded_classes=excluded_classes, tracker=tracker)

    # 메인 파이프라인
    print("\n" + "="*80 + "\nSTARTING MAIN KEYFRAME EXTRACTION PIPELINE\n" + "="*80)
    print(f"Frame skipping interval: {frame_skip_interval} (higher = faster processing, fewer frames)")
    print(f"Pre-Greedy Profile Tracking iterations: {profile_iterations}")
    print(f"Post-Greedy Profile Tracking iterations: {post_profile_iterations}")
    
    # Step 1: Primary Selection (with frame skipping + filtering)
    print("\nSTEP 1: Primary Selection...")
    selected_frames = primary_selection(video_path, detection_model, class_names,
                                       frame_skip_interval=frame_skip_interval,
                                       included_classes=included_classes,
                                       excluded_classes=excluded_classes,
                                       tracker=tracker)
    print(f"-> Found {len(selected_frames)} initial keyframes.")
    save_frames_as_images(selected_frames, primary_select_folder, prefix="primary_")
    
    # Step 2: Profile Tracking (결합 히스토그램 + 반복 + Window Size 적용)
    print("\nSTEP 2: Profile Tracking (Filtering)...")
    filtered_frames = profile_tracking(selected_frames, kwargs['window_size'], kwargs['hist_threshold'],
                                       hist_weight_brightness, hist_weight_saturation, profile_iterations)
    print(f"-> Filtered down to {len(filtered_frames)} frames after {profile_iterations} iterations.")
    save_frames_as_images(filtered_frames, profile_track_folder, prefix="profile_tracked_")
    
    # Step 3: Re-ID and Frame Merging (Dual TorchReID 기반 + filtering)
    print("\nSTEP 3: Dual TorchReID Re-ID and Frame Merging...")
    if use_dual_reid:
        unique_frames = unified_torchreid_reid_and_frame_selection_dual(
            filtered_frames, person_reid_model, vehicle_reid_model, device,
            reid_threshold=kwargs['reid_threshold'],
            frame_merge_threshold=kwargs['frame_merge_threshold'],
            min_samples=kwargs['min_reid_samples'],
            included_classes=included_classes,
            excluded_classes=excluded_classes
        )
        print(f"-> After Dual Re-ID and merging, {len(unique_frames)} frames remain.")
    else:
        # Legacy single model mode
        unique_frames = unified_torchreid_reid_and_frame_selection(
            filtered_frames, torchreid_model, device,
            reid_threshold=kwargs['reid_threshold'],
            frame_merge_threshold=kwargs['frame_merge_threshold'],
            min_samples=kwargs['min_reid_samples'],
            included_classes=included_classes,
            excluded_classes=excluded_classes
        )
        print(f"-> After Re-ID and merging, {len(unique_frames)} frames remain.")
    save_frames_as_images(unique_frames, after_reid_folder, prefix="after_reid_")

    # Step 4: Greedy coverage selection (with filtering)
    print("\nSTEP 4: Greedy Coverage Selection...")
    final_selected, greedy_log = greedy_coverage_selection(
        unique_frames,
        k=kwargs['coverage_k'],
        use_instance_id=kwargs['use_instance_id'],
        max_selected=kwargs['max_selected'],
        min_new_combos=kwargs['min_new_combos'],
        score_thresh=kwargs['score_thresh'],
        included_classes=included_classes,
        excluded_classes=excluded_classes
    )
    
    # Step 4.5: Post-Greedy Profile Tracking (새 추가, iterations 지원)
    post_filter_log = None
    if apply_post_filter:
        print("\nSTEP 4.5: Post-Greedy Profile Tracking...")
        final_selected, post_filter_log = post_greedy_profile_tracking(
            final_selected, post_hist_threshold,
            hist_weight_brightness, hist_weight_saturation, post_window_size, post_profile_iterations
        )
    
    # Step 5: Save final results and logs
    print("\n" + "="*80 + "\nSTEP 5: SAVING FINAL RESULTS AND LOGS\n" + "="*80)
    
    # 최종 이미지 저장 (bounding box 있는 버전과 없는 버전 모두 저장)
    for item in final_selected:
        # Bounding box 없는 원본 프레임 저장
        original_img_path = os.path.join(final_folder, f"final_key_frame_{item['frame_idx']}_original.jpg")
        cv2.imwrite(original_img_path, item['frame'])

        # Bounding box 있는 프레임 저장
        drawn_frame = draw_detections(item['frame'], item['detections'])
        drawn_img_path = os.path.join(final_folder, f"final_key_frame_{item['frame_idx']}_bbox.jpg")
        cv2.imwrite(drawn_img_path, drawn_frame)

        print(f"✓ Saved final keyframe (original): {os.path.basename(original_img_path)}")
        print(f"✓ Saved final keyframe (with bbox): {os.path.basename(drawn_img_path)}")
    
    # JSON 저장 - 간결 구조 + 히스토그램/포스트 필터 정보 포함
    save_json_results(
        final_selected, greedy_log, output_folder, 
        save_mode=json_save_mode,
        dpt_processor=dpt_processor,
        dpt_model=dpt_model,
        dpt_device=dpt_device,
        save_depth_visualization=save_depth_visualization,
        save_foreground_masks=save_foreground_masks,
        use_foreground_only=use_foreground_only,
        threshold_method=threshold_method,
        histogram_method="brightness_saturation_combined",
        hist_weights={"brightness": hist_weight_brightness, "saturation": hist_weight_saturation},
        post_filter=post_filter_log
    )
    
    print("\n" + "="*80 + "\nPIPELINE COMPLETE!\n" + "="*80)


# ============================================================================
# PART 6: Example Usage (config 업데이트 - 객체 필터링 추가)
# ============================================================================

if __name__ == "__main__":
    # 클래스 목록만 보기 (파이프라인 실행 안 함)
    if len(sys.argv) > 1 and sys.argv[1] == "--list-classes":
        list_classes_only(
            model_path="rtdetr-l.pt",  # config의 model_path와 동일하게 설정
            model_type="rt_detr"
        )
        sys.exit(0)
    
    config = {
        # ========================================
        # 기본 설정 (Basic Configuration)
        # ========================================
        "video_path": "KakaoTalk_20251003_113043691.mp4",  # 분석할 비디오 파일의 전체 경로 (e.g., MP4 파일)
        "output_folder": "results_output",  # 결과 파일(이미지, JSON, 비디오)을 저장할 폴더 경로
        "json_save_mode": "both",  # JSON 저장 모드: "unified" (하나의 요약 파일), "per_frame" (프레임별 파일), "both" (둘 다)
        "create_comparison_videos": False,  # True 시 0_source_videos 폴더에 트래킹 비교 비디오 생성 (False 시 생성 안 함, 시간 절약)
        
        # ========================================
        # 모델 설정 (Model Configuration)
        # ========================================
        "model_type": "rt_detr",  # 감지 모델 타입: "yolo" (YOLOv11) 또는 "rt_detr" (RT-DETR)
        #"model_path": "yolo11m.pt",  # YOLO/RT-DETR 모델 파일의 전체 경로 (.pt 파일)
        "model_path": "rtdetr-l.pt",   # yolo 사용시 2번째 주석 해제하고 3번째 주석(rt-detr 사용시 그반대)
        
        # ========================================
        # TorchReID 모델 설정 (TorchReID Model Configuration)
        # ========================================
        # Dual Re-ID 모드 (사람/차량 분리 Re-ID)
        "use_dual_reid": True,  # True 시 사람/차량 Re-ID 분리 사용, False 시 단일 모델 사용

        # 🆕 Model Registry Keys (추천 방식)
        "person_reid_model_key": "osnet_market1501",  # Person Re-ID 모델 키
        "vehicle_reid_model_key": "osnet_x1_0",  # Vehicle Re-ID 모델 키

        # Available Person Re-ID Models:
        # - "osnet_market1501" : OSNet Market-1501 fine-tuned (BEST for person)
        # - "osnet_x1_0"       : OSNet pretrained (lightweight)
        # - "osnet_x0_75"      : OSNet x0.75 (faster)
        # - "osnet_x0_5"       : OSNet x0.5 (fastest)
        # - "osnet_ibn_x1_0"   : OSNet-IBN (with Instance-Batch Norm)
        # - "resnet50_fc512"   : ResNet50 baseline
        # - "mlfn"             : Multi-Level Factorisation Net (lightweight)

        # Available Vehicle Re-ID Models:
        # - "osnet_veri776"    : OSNet VeRi-776 fine-tuned (BEST for vehicle, if available)
        # - "osnet_x1_0"       : OSNet pretrained (general)
        # - "resnet50_fc512"   : ResNet50 baseline

        # Legacy: Path-based loading (optional, overrides registry)
        "person_reid_model_path": None,  # None = use registry key above
        "vehicle_reid_model_path": None,  # None = use registry key above

        # 단일 Re-ID 모드 (legacy, use_dual_reid=False일 때만 사용)
        "torchreid_model_path": "osnet_x1_0_market_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth",
        
        # ========================================
        # Depth 추정 설정 (Depth Estimation Configuration)
        # ========================================
        "use_depth": True,  # True 시 DPT 모델로 Depth 추정 활성화 (False 시 비활성화, 속도 향상)
        "dpt_model_name": "Intel/dpt-hybrid-midas",  # DPT 모델 이름: "Intel/dpt-large" (고품질/느림), "Intel/dpt-hybrid-midas" (중간), "Intel/dpt-swinv2-tiny-256" (빠름/저품질)
        "save_depth_visualization": True,  # True 시 Depth 맵을 컬러 이미지로 저장 (depth_visualizations 폴더)
        "save_foreground_masks": True,  # True 시 객체별 전경 마스크 이미지 저장 (foreground_masks 폴더)
        "use_foreground_only": True,  # True 시 Depth 통계에서 전경(객체) 영역만 사용 (False 시 전체 영역)
        "threshold_method": "otsu",  # 전경/배경 분리 방법: "otsu" (Otsu 자동 임계값), "mean" (평균값 기준)
        
        # ========================================
        # Frame Skipping 설정 (Frame Skipping Configuration)
        # ========================================
        "frame_skip_interval": 5,  # 비디오 프레임 스킵 간격 (1 = 모든 프레임 처리, 5 = 5프레임마다 하나 처리, 높을수록 빠름)
        
        # ========================================
        # Profile Tracking 반복 설정 (Profile Tracking Iterations Configuration)
        # ========================================
        "profile_iterations": 3,  # Pre-Greedy Profile Tracking 반복 횟수 (1 = 기본 한 번, 2+ = 반복 필터링으로 더 엄격한 제거)
        
        # ========================================
        # Post-Greedy Profile Tracking 설정 (Post-Greedy Profile Tracking Configuration)
        # ========================================
        "apply_post_filter": True,  # True 시 Greedy 후 추가 유사 프레임 필터링 적용 (False 시 생략)
        "post_hist_threshold": 0.25,  # Post-Filter 히스토그램 유사도 임계값 (낮을수록 엄격)
        "post_window_size": 7,  # Post-Filter 필터링 시 고려할 주변 프레임 수 (e.g., 3 = ±1 프레임)
        "post_profile_iterations": 1,  # Post-Greedy Profile Tracking 반복 횟수 (1 = 기본 한 번, 2+ = 반복 필터링으로 더 엄격한 제거)
        
        # ========================================
        # 히스토그램 설정 (Histogram Configuration for Profile Tracking)
        # ========================================
        "hist_weight_brightness": 0.5,  # 밝기(Grayscale) 히스토그램 가중치 (brightness + saturation = 1.0이 되도록 설정)
        "hist_weight_saturation": 0.5,  # 채도(HSV S 채널) 히스토그램 가중치 (brightness + saturation = 1.0이 되도록 설정)
        
        # ========================================
        # Profile Tracking (Pre-Greedy) 설정 (Pre-Greedy Profile Tracking Configuration)
        # ========================================
        "window_size": 15,  # Pre-Greedy 필터링 시 고려할 주변 프레임 수 (e.g., 5 = ±2 프레임)
        "hist_threshold": 0.3,  # Pre-Greedy 히스토그램 유사도 임계값 (낮을수록 유사 프레임 제거 엄격)
        
        # ========================================
        # TorchReID 설정 (TorchReID Configuration for Re-ID)
        # ========================================
        "reid_threshold": 0.8,  # TorchReID 유사도 임계값 (높을수록 엄격)
        
        # ========================================
        # Re-ID 및 프레임 병합 설정 (Re-ID and Frame Merging Configuration)
        # ========================================
        "min_reid_samples": 2,  # Re-ID 클러스터링 최소 객체 샘플 수 (클래스당)
        "frame_merge_threshold": 0.7,  # 프레임 병합 시 TorchReID 유사도 임계값 (높을수록 엄격)
        
        # ========================================
        # Greedy Selection 설정 (Greedy Coverage Selection Configuration)
        # ========================================
        "coverage_k": 2,  # Greedy에서 커버할 객체 조합 크기 (e.g., 2 = 객체 쌍)
        "use_instance_id": True,  # True 시 track_id(인스턴스 ID) 사용, False 시 class_name만 사용
        "max_selected": None,  # 최대 선택할 키프레임 수 (None = 무제한, int = 제한)
        "min_new_combos": 1,  # 선택할 프레임이 추가로 커버할 최소 신규 조합 수 (0 이하 시 모든 프레임 선택)
        "score_thresh": 0.0,  # 객체 감지 신뢰도 임계값 (0.0 ~ 1.0, 낮을수록 더 많은 객체 포함)
        
        # ========================================
        # 🆕 객체 필터링 설정 (Object Filtering Configuration)
        # ========================================
        "show_available_classes": True,  # True 시 탐지 가능한 클래스 목록 출력 (처음 실행 시 True 권장)
        
        # 필터링 모드 설정 (아래 3가지 중 하나 선택)
        # 1 = 블랙리스트 (특정 클래스 제외, 나머지 모두 포함)
        # 2 = 화이트리스트 (특정 클래스만 포함, 나머지 모두 제외)
        # 3 = 필터링 없음 (모든 클래스 포함)
        "filter_mode": 3,
        
        # 필터링할 클래스 리스트 (filter_mode가 1 또는 2일 때만 사용)
        # - 문자열 또는 숫자(클래스 ID)로 지정 가능
        # - 예시: ["person", "car", 2, 7]  # "person", "car", 클래스ID 2(car), 클래스ID 7(truck)
        "filter_classes": [],
        
        # ========================================
        # 사용 예시:
        # ========================================
        # 예시 1: 블랙리스트 - person과 car 제외
        # "filter_mode": 1,
        # "filter_classes": ["person", "car"],
        
        # 예시 2: 블랙리스트 - 클래스 ID로 지정 (person=0, car=2)
        # "filter_mode": 1,
        # "filter_classes": [0, 2],
        
        # 예시 3: 블랙리스트 - 혼합 사용
        # "filter_mode": 1,
        # "filter_classes": ["person", 2, "truck"],  # person, car(2), truck
        
        # 예시 4: 화이트리스트 - cell phone과 bottle만 포함
        # "filter_mode": 2,
        # "filter_classes": ["cell phone", "bottle"],
        
        # 예시 5: 화이트리스트 - 클래스 ID로 지정
        # "filter_mode": 2,
        # "filter_classes": [67, 39],  # cell phone(67), bottle(39)
        
        # 예시 6: 필터링 없음 (현재 설정)
        # "filter_mode": 3,
        # "filter_classes": [],  # filter_mode=3일 때는 무시됨
        
        # ⚠️ 주의사항:
        # - filter_mode=3일 때 filter_classes에 값이 있으면 경고 메시지 출력 후 무시됩니다
        # - filter_mode=1 또는 2일 때 filter_classes가 비어있으면 자동으로 mode=3으로 전환됩니다
        # - 클래스 ID는 --list-classes 옵션으로 확인할 수 있습니다
        
        # ========================================
    }
    
    main(**config)
