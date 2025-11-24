"""
평가 메트릭 핵심 로직 테스트 (의존성 없이)
"""


def calculate_f1_with_tolerance(predicted, ground_truth, tolerance=15):
    """
    핵심 로직 복사 (테스트용)
    """
    if len(predicted) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'tp': 0,
            'fp': len(predicted),
            'fn': len(ground_truth)
        }

    if len(ground_truth) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'tp': 0,
            'fp': len(predicted),
            'fn': 0
        }

    # Convert to sorted lists for consistent ordering
    # Remove duplicates and sort to ensure reproducibility
    gt_list = sorted(set(ground_truth))
    pred_list = sorted(set(predicted))

    # True Positives: predicted frames that match with ground truth within tolerance
    tp = 0
    matched_gt = set()
    matched_pred = set()

    # Process predicted frames in sorted order for consistent results
    for pred_frame in pred_list:
        # Find the closest ground truth frame within tolerance
        best_gt = None
        best_distance = float('inf')

        for gt_frame in gt_list:
            if gt_frame in matched_gt:
                continue  # Already matched

            distance = abs(pred_frame - gt_frame)
            if distance <= tolerance and distance < best_distance:
                best_gt = gt_frame
                best_distance = distance

        # Match with the closest GT frame if found
        if best_gt is not None:
            tp += 1
            matched_gt.add(best_gt)
            matched_pred.add(pred_frame)

    # False Positives: predicted frames that don't match any ground truth
    fp = len(pred_list) - tp

    # False Negatives: ground truth frames that weren't matched
    fn = len(gt_list) - tp

    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def run_test(name, gt, pred, tolerance, expected_tp, expected_fp, expected_fn):
    """테스트 실행"""
    result = calculate_f1_with_tolerance(pred, gt, tolerance)

    print(f"\n{name}")
    print(f"  GT: {gt}")
    print(f"  Pred: {pred}")
    print(f"  Tolerance: {tolerance}")
    print(f"  Results: TP={result['tp']}, FP={result['fp']}, FN={result['fn']}")
    print(f"  Precision={result['precision']:.3f}, Recall={result['recall']:.3f}, F1={result['f1_score']:.3f}")

    success = (result['tp'] == expected_tp and
               result['fp'] == expected_fp and
               result['fn'] == expected_fn)

    if success:
        print(f"  ✓ PASS")
    else:
        print(f"  ✗ FAIL (Expected TP={expected_tp}, FP={expected_fp}, FN={expected_fn})")

    return success


if __name__ == "__main__":
    print("="*80)
    print("평가 메트릭 논리 검증 테스트")
    print("="*80)

    all_pass = True

    # Test 1: 정확히 일치
    all_pass &= run_test(
        "Test 1: 정확히 일치",
        gt=[100, 200, 300],
        pred=[100, 200, 300],
        tolerance=0,
        expected_tp=3, expected_fp=0, expected_fn=0
    )

    # Test 2: Tolerance 내
    all_pass &= run_test(
        "Test 2: Tolerance 내 (±5 프레임)",
        gt=[100, 200, 300],
        pred=[105, 195, 305],
        tolerance=15,
        expected_tp=3, expected_fp=0, expected_fn=0
    )

    # Test 3: 하나의 GT에 여러 후보
    all_pass &= run_test(
        "Test 3: 하나의 GT에 여러 후보",
        gt=[100],
        pred=[95, 105],
        tolerance=15,
        expected_tp=1, expected_fp=1, expected_fn=0
    )

    # Test 4: 연속 프레임들
    all_pass &= run_test(
        "Test 4: 연속 프레임들 (사용자가 우려한 케이스)",
        gt=[110],
        pred=[100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        tolerance=15,
        expected_tp=1, expected_fp=10, expected_fn=0
    )

    # Test 5: 일부 GT 누락
    all_pass &= run_test(
        "Test 5: 일부 GT 누락",
        gt=[100, 200, 300],
        pred=[105],
        tolerance=15,
        expected_tp=1, expected_fp=0, expected_fn=2
    )

    # Test 6: Tolerance 밖
    all_pass &= run_test(
        "Test 6: Tolerance 밖",
        gt=[100, 200, 300],
        pred=[150, 250, 350],
        tolerance=15,
        expected_tp=0, expected_fp=3, expected_fn=3
    )

    # Test 7: 예측 없음
    all_pass &= run_test(
        "Test 7: 예측 없음",
        gt=[100, 200, 300],
        pred=[],
        tolerance=15,
        expected_tp=0, expected_fp=0, expected_fn=3
    )

    # Test 8: 중복 예측
    all_pass &= run_test(
        "Test 8: 중복 예측 (자동 제거)",
        gt=[100, 200],
        pred=[100, 100, 100, 200, 200],
        tolerance=0,
        expected_tp=2, expected_fp=0, expected_fn=0
    )

    # Test 9: 가장 가까운 매칭
    all_pass &= run_test(
        "Test 9: 가장 가까운 GT와 매칭",
        gt=[100, 110],
        pred=[105],
        tolerance=15,
        expected_tp=1, expected_fp=0, expected_fn=1
    )

    # Test 10: 재현성
    print("\nTest 10: 재현성 (3회 실행)")
    gt = [100, 200, 300]
    pred = [105, 195, 305]
    result1 = calculate_f1_with_tolerance(pred, gt, tolerance=15)
    result2 = calculate_f1_with_tolerance(pred, gt, tolerance=15)
    result3 = calculate_f1_with_tolerance(pred, gt, tolerance=15)

    print(f"  GT: {gt}")
    print(f"  Pred: {pred}")
    print(f"  Run 1: TP={result1['tp']}, F1={result1['f1_score']:.3f}")
    print(f"  Run 2: TP={result2['tp']}, F1={result2['f1_score']:.3f}")
    print(f"  Run 3: TP={result3['tp']}, F1={result3['f1_score']:.3f}")

    if result1 == result2 == result3:
        print("  ✓ PASS (재현성 확인)")
    else:
        print("  ✗ FAIL (재현성 문제)")
        all_pass = False

    print("\n" + "="*80)
    if all_pass:
        print("✓ 모든 테스트 통과!")
        print("="*80)
        print("\n수정 사항 요약:")
        print("  ✓ Set → Sorted List: 재현성 보장")
        print("  ✓ First-match → Closest-match: 더 정확한 매칭")
        print("  ✓ One-to-one 매칭 유지")
        print("  ✓ 중복 자동 제거")
        print("  ✓ Edge case 올바르게 처리")
        print("\n발견하고 수정한 문제:")
        print("  1. Set 순회 순서 불확실성 → 정렬된 리스트 사용")
        print("  2. Greedy first-match → 가장 가까운 GT와 매칭")
        print("  3. 재현성 보장")
    else:
        print("✗ 일부 테스트 실패")
        print("="*80)
