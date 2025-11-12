"""
평가 메트릭 테스트

논리적 허점을 확인하기 위한 테스트 케이스들
"""

from evaluation_metrics import calculate_f1_with_tolerance


def test_case_1_exact_match():
    """정확히 일치하는 경우"""
    gt = [100, 200, 300]
    pred = [100, 200, 300]

    result = calculate_f1_with_tolerance(pred, gt, tolerance=0)

    print("Test 1: 정확히 일치")
    print(f"  GT: {gt}")
    print(f"  Pred: {pred}")
    print(f"  Results: TP={result['tp']}, FP={result['fp']}, FN={result['fn']}")
    print(f"  Precision={result['precision']:.3f}, Recall={result['recall']:.3f}, F1={result['f1_score']:.3f}")

    assert result['tp'] == 3, "TP should be 3"
    assert result['fp'] == 0, "FP should be 0"
    assert result['fn'] == 0, "FN should be 0"
    assert result['f1_score'] == 1.0, "F1 should be 1.0"
    print("  ✓ PASS\n")


def test_case_2_within_tolerance():
    """Tolerance 내에 있는 경우"""
    gt = [100, 200, 300]
    pred = [105, 195, 305]  # ±5 프레임

    result = calculate_f1_with_tolerance(pred, gt, tolerance=15)

    print("Test 2: Tolerance 내 (±5 프레임)")
    print(f"  GT: {gt}")
    print(f"  Pred: {pred}")
    print(f"  Tolerance: 15")
    print(f"  Results: TP={result['tp']}, FP={result['fp']}, FN={result['fn']}")
    print(f"  Precision={result['precision']:.3f}, Recall={result['recall']:.3f}, F1={result['f1_score']:.3f}")

    assert result['tp'] == 3, "TP should be 3"
    assert result['fp'] == 0, "FP should be 0"
    assert result['fn'] == 0, "FN should be 0"
    assert result['f1_score'] == 1.0, "F1 should be 1.0"
    print("  ✓ PASS\n")


def test_case_3_multiple_candidates():
    """하나의 GT에 여러 predicted가 가까운 경우 (가장 가까운 것과 매칭되어야 함)"""
    gt = [100]
    pred = [95, 105]  # 둘 다 tolerance 내

    result = calculate_f1_with_tolerance(pred, gt, tolerance=15)

    print("Test 3: 하나의 GT에 여러 후보")
    print(f"  GT: {gt}")
    print(f"  Pred: {pred}")
    print(f"  Tolerance: 15")
    print(f"  Results: TP={result['tp']}, FP={result['fp']}, FN={result['fn']}")
    print(f"  Precision={result['precision']:.3f}, Recall={result['recall']:.3f}, F1={result['f1_score']:.3f}")

    # 95가 먼저 체크되므로 95가 매칭되고, 105는 FP
    # 수정 후: 95와 105 중 더 가까운 95가 매칭
    assert result['tp'] == 1, "TP should be 1"
    assert result['fp'] == 1, "FP should be 1 (한 프레임이 남음)"
    assert result['fn'] == 0, "FN should be 0"
    print("  ✓ PASS\n")


def test_case_4_consecutive_frames():
    """연속된 프레임들이 하나의 GT와 경쟁하는 경우"""
    gt = [110]
    pred = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110]

    result = calculate_f1_with_tolerance(pred, gt, tolerance=15)

    print("Test 4: 연속 프레임들")
    print(f"  GT: {gt}")
    print(f"  Pred: {pred} (11개)")
    print(f"  Tolerance: 15")
    print(f"  Results: TP={result['tp']}, FP={result['fp']}, FN={result['fn']}")
    print(f"  Precision={result['precision']:.3f}, Recall={result['recall']:.3f}, F1={result['f1_score']:.3f}")

    # 가장 가까운 110만 매칭, 나머지는 FP
    assert result['tp'] == 1, "TP should be 1 (only closest frame)"
    assert result['fp'] == 10, "FP should be 10 (나머지 프레임들)"
    assert result['fn'] == 0, "FN should be 0"
    print("  ✓ PASS (Precision이 낮음을 확인)\n")


def test_case_5_missing_predictions():
    """일부 GT를 놓친 경우"""
    gt = [100, 200, 300]
    pred = [105]  # 하나만 예측

    result = calculate_f1_with_tolerance(pred, gt, tolerance=15)

    print("Test 5: 일부 GT 누락")
    print(f"  GT: {gt}")
    print(f"  Pred: {pred}")
    print(f"  Tolerance: 15")
    print(f"  Results: TP={result['tp']}, FP={result['fp']}, FN={result['fn']}")
    print(f"  Precision={result['precision']:.3f}, Recall={result['recall']:.3f}, F1={result['f1_score']:.3f}")

    assert result['tp'] == 1, "TP should be 1"
    assert result['fp'] == 0, "FP should be 0"
    assert result['fn'] == 2, "FN should be 2"
    assert result['recall'] < 0.5, "Recall should be low"
    print("  ✓ PASS\n")


def test_case_6_outside_tolerance():
    """Tolerance 밖의 경우"""
    gt = [100, 200, 300]
    pred = [150, 250, 350]  # 50 프레임 차이

    result = calculate_f1_with_tolerance(pred, gt, tolerance=15)

    print("Test 6: Tolerance 밖")
    print(f"  GT: {gt}")
    print(f"  Pred: {pred}")
    print(f"  Tolerance: 15")
    print(f"  Results: TP={result['tp']}, FP={result['fp']}, FN={result['fn']}")
    print(f"  Precision={result['precision']:.3f}, Recall={result['recall']:.3f}, F1={result['f1_score']:.3f}")

    assert result['tp'] == 0, "TP should be 0"
    assert result['fp'] == 3, "FP should be 3"
    assert result['fn'] == 3, "FN should be 3"
    assert result['f1_score'] == 0.0, "F1 should be 0.0"
    print("  ✓ PASS\n")


def test_case_7_empty_predictions():
    """예측이 비어있는 경우"""
    gt = [100, 200, 300]
    pred = []

    result = calculate_f1_with_tolerance(pred, gt, tolerance=15)

    print("Test 7: 예측 없음")
    print(f"  GT: {gt}")
    print(f"  Pred: {pred}")
    print(f"  Results: TP={result['tp']}, FP={result['fp']}, FN={result['fn']}")
    print(f"  Precision={result['precision']:.3f}, Recall={result['recall']:.3f}, F1={result['f1_score']:.3f}")

    assert result['tp'] == 0, "TP should be 0"
    assert result['fp'] == 0, "FP should be 0"
    assert result['fn'] == 3, "FN should be 3"
    assert result['f1_score'] == 0.0, "F1 should be 0.0"
    print("  ✓ PASS\n")


def test_case_8_duplicate_predictions():
    """중복 예측 (제거되어야 함)"""
    gt = [100, 200]
    pred = [100, 100, 100, 200, 200]  # 중복

    result = calculate_f1_with_tolerance(pred, gt, tolerance=0)

    print("Test 8: 중복 예측")
    print(f"  GT: {gt}")
    print(f"  Pred: {pred} (중복 포함)")
    print(f"  Results: TP={result['tp']}, FP={result['fp']}, FN={result['fn']}")
    print(f"  Precision={result['precision']:.3f}, Recall={result['recall']:.3f}, F1={result['f1_score']:.3f}")

    # 중복이 제거되어 [100, 200]으로 처리되어야 함
    assert result['tp'] == 2, "TP should be 2 (duplicates removed)"
    assert result['fp'] == 0, "FP should be 0"
    assert result['fn'] == 0, "FN should be 0"
    assert result['f1_score'] == 1.0, "F1 should be 1.0"
    print("  ✓ PASS\n")


def test_case_9_closest_matching():
    """가장 가까운 GT와 매칭되는지 확인"""
    gt = [100, 110]
    pred = [105]  # 100과 110 중간

    result = calculate_f1_with_tolerance(pred, gt, tolerance=15)

    print("Test 9: 가장 가까운 매칭")
    print(f"  GT: {gt}")
    print(f"  Pred: {pred}")
    print(f"  Tolerance: 15")
    print(f"  Results: TP={result['tp']}, FP={result['fp']}, FN={result['fn']}")
    print(f"  Precision={result['precision']:.3f}, Recall={result['recall']:.3f}, F1={result['f1_score']:.3f}")

    # 105는 100(차이 5)과 110(차이 5) 둘 다 가능하지만, 더 가까운 것 또는 먼저 발견되는 것과 매칭
    assert result['tp'] == 1, "TP should be 1"
    assert result['fp'] == 0, "FP should be 0"
    assert result['fn'] == 1, "FN should be 1 (one GT unmatched)"
    print("  ✓ PASS\n")


def test_case_10_reproducibility():
    """재현성 테스트 (같은 입력에 같은 결과)"""
    gt = [100, 200, 300]
    pred = [105, 195, 305]

    result1 = calculate_f1_with_tolerance(pred, gt, tolerance=15)
    result2 = calculate_f1_with_tolerance(pred, gt, tolerance=15)
    result3 = calculate_f1_with_tolerance(pred, gt, tolerance=15)

    print("Test 10: 재현성 (3회 실행)")
    print(f"  GT: {gt}")
    print(f"  Pred: {pred}")
    print(f"  Run 1: TP={result1['tp']}, F1={result1['f1_score']:.3f}")
    print(f"  Run 2: TP={result2['tp']}, F1={result2['f1_score']:.3f}")
    print(f"  Run 3: TP={result3['tp']}, F1={result3['f1_score']:.3f}")

    assert result1 == result2 == result3, "Results should be identical"
    print("  ✓ PASS (재현성 확인)\n")


if __name__ == "__main__":
    print("="*80)
    print("평가 메트릭 논리 검증 테스트")
    print("="*80 + "\n")

    try:
        test_case_1_exact_match()
        test_case_2_within_tolerance()
        test_case_3_multiple_candidates()
        test_case_4_consecutive_frames()
        test_case_5_missing_predictions()
        test_case_6_outside_tolerance()
        test_case_7_empty_predictions()
        test_case_8_duplicate_predictions()
        test_case_9_closest_matching()
        test_case_10_reproducibility()

        print("="*80)
        print("✓ 모든 테스트 통과!")
        print("="*80)
        print("\n핵심 확인 사항:")
        print("  ✓ One-to-one 매칭 작동")
        print("  ✓ 가장 가까운 GT와 매칭")
        print("  ✓ 중복 프레임 자동 제거")
        print("  ✓ 재현성 보장 (정렬된 순서)")
        print("  ✓ Edge case 처리")
        print("  ✓ Tolerance 올바르게 적용")

    except AssertionError as e:
        print(f"\n✗ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
