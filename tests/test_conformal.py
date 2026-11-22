"""conformal 计算与合并逻辑的单元测试。"""

import math

import numpy as np
import pytest

from app.conformal import (
    CalibrationValidationError,
    merge_alarm_intervals,
    predict_sets,
    validate_and_build_thresholds,
)
from app.schemas import CalibrationWindow, DetectionWindow


def _w(ts, label, **scores):
    return CalibrationWindow(timestamp=ts, label=label, scores=dict(scores))


def test_quantile_uses_finite_sample_correction():
    # n=5, alpha=0.2 -> k=ceil(6*0.2)=2 -> rank=0.4 -> 低分位（高分似故障）
    windows = [
        _w(float(i), "f1", f1=0.1 * (i + 1), f2=0.0) for i in range(5)
    ]
    windows.append(_w(10.0, "normal", f1=0.0, f2=0.0))
    windows.append(_w(11.0, "f2", f1=0.0, f2=0.9))
    info, normal_n, fault_n = validate_and_build_thresholds(
        windows, ["f1", "f2"], alpha=0.2, min_samples_per_fault=1
    )
    threshold, n_samples, rank = info["f1"]
    assert n_samples == 5
    assert rank == pytest.approx(0.4)
    # numpy 线性插值：rank=0.4 位于第 2、3 个次序统计量之间
    assert threshold == pytest.approx(float(np.quantile([0.1, 0.2, 0.3, 0.4, 0.5], 0.4)))
    assert normal_n == 1
    assert fault_n == 6


def test_quantile_matches_manual_numpy_calculation():
    rng = np.random.default_rng(42)
    values = rng.normal(loc=0.8, scale=0.05, size=20)
    windows = [
        _w(float(i), "f1", f1=float(v), f2=0.0)
        for i, v in enumerate(values)
    ]
    windows += [_w(100.0 + i, "f2", f1=0.0, f2=0.7) for i in range(3)]
    info, _, _ = validate_and_build_thresholds(
        windows, ["f1", "f2"], alpha=0.1, min_samples_per_fault=1
    )
    n = 20
    expected_rank = min(1.0, math.ceil((n + 1) * 0.1) / n)
    expected = float(np.quantile(values, expected_rank, method="linear"))
    assert info["f1"][0] == pytest.approx(expected)
    assert info["f1"][2] == pytest.approx(expected_rank)


def test_missing_fault_class_rejected():
    windows = [_w(1.0, "normal", f1=0.1, f2=0.1), _w(2.0, "f1", f1=0.9, f2=0.1)]
    with pytest.raises(CalibrationValidationError, match="覆盖不足"):
        validate_and_build_thresholds(
            windows, ["f1", "f2"], alpha=0.1, min_samples_per_fault=2
        )


def test_nan_and_inf_scores_rejected():
    for bad in (float("nan"), float("inf"), float("-inf")):
        windows = [
            _w(1.0, "normal", f1=0.1, f2=0.1),
            _w(2.0, "f1", f1=bad, f2=0.2),
            _w(3.0, "f1", f1=0.8, f2=0.2),
        ]
        with pytest.raises(CalibrationValidationError, match="有限数值"):
            validate_and_build_thresholds(
                windows, ["f1", "f2"], alpha=0.1, min_samples_per_fault=1
            )


def test_missing_score_key_rejected():
    windows = [
        CalibrationWindow(timestamp=1.0, label="f1", scores={"f1": 0.9}),
        CalibrationWindow(timestamp=2.0, label="f2", scores={"f1": 0.1, "f2": 0.8}),
    ]
    with pytest.raises(CalibrationValidationError, match="缺少故障类型分数"):
        validate_and_build_thresholds(
            windows, ["f1", "f2"], alpha=0.1, min_samples_per_fault=1
        )


def test_unsorted_timestamps_rejected():
    windows = [
        _w(2.0, "f1", f1=0.9, f2=0.1),
        _w(1.0, "f1", f1=0.8, f2=0.1),
        _w(3.0, "f2", f1=0.1, f2=0.7),
    ]
    with pytest.raises(CalibrationValidationError, match="按时间升序"):
        validate_and_build_thresholds(
            windows, ["f1", "f2"], alpha=0.1, min_samples_per_fault=1
        )


def test_unknown_label_rejected():
    windows = [
        _w(1.0, "exploded", f1=0.9, f2=0.1),
        _w(2.0, "f1", f1=0.8, f2=0.1),
        _w(3.0, "f2", f1=0.1, f2=0.7),
    ]
    with pytest.raises(CalibrationValidationError, match="不在 fault_types"):
        validate_and_build_thresholds(
            windows, ["f1", "f2"], alpha=0.1, min_samples_per_fault=1
        )


def test_predict_sets_and_normal_case():
    dwindows = [
        DetectionWindow(timestamp=1.0, scores={"f1": 0.95, "f2": 0.2}),
        DetectionWindow(timestamp=2.0, scores={"f1": 0.1, "f2": 0.85}),
        DetectionWindow(timestamp=3.0, scores={"f1": 0.05, "f2": 0.05}),
        DetectionWindow(timestamp=4.0, scores={"f1": 0.9, "f2": 0.9}),
    ]
    sets = predict_sets(dwindows, ["f1", "f2"], {"f1": 0.5, "f2": 0.5})
    assert sets == [["f1"], ["f2"], [], ["f1", "f2"]]


def test_predict_invalid_score_raises():
    dwindows = [DetectionWindow(timestamp=1.0, scores={"f1": float("nan"), "f2": 0.2})]
    with pytest.raises(CalibrationValidationError, match="有限数值"):
        predict_sets(dwindows, ["f1", "f2"], {"f1": 0.5, "f2": 0.5})


def test_merge_strictly_adjacent_alarms():
    alarms = [False, True, True, False, True, False]
    assert merge_alarm_intervals(alarms, 0) == [(1, 2), (4, 4)]


def test_merge_with_gap_tolerance():
    alarms = [True, False, False, True, False, True]
    # gap=1：索引 0 和 3 之间有 2 个非告警，不合并；3 和 5 间隔 1 个，合并
    assert merge_alarm_intervals(alarms, 1) == [(0, 0), (3, 5)]
    # gap=2：全部合并
    assert merge_alarm_intervals(alarms, 2) == [(0, 5)]
    assert merge_alarm_intervals([], 0) == []


def test_conformal_coverage_on_synthetic_data():
    """合成数据上的经验覆盖率 sanity check：故障窗口召回率应接近 1-alpha。

    有限样本校正的形式化保证针对新样本；在校准集自身上，线性插值分位
    得到的经验覆盖率与 1-alpha 的偏差不超过 2/n。
    """
    rng = np.random.default_rng(7)
    fault_scores = rng.normal(0.8, 0.1, size=200)
    windows = [
        _w(float(i), "f1", f1=float(v), f2=0.0)
        for i, v in enumerate(fault_scores)
    ]
    windows += [
        _w(1000.0 + i, "f2", f1=0.0, f2=float(v))
        for i, v in enumerate(rng.normal(0.7, 0.1, size=200))
    ]
    info, _, _ = validate_and_build_thresholds(
        windows, ["f1", "f2"], alpha=0.1, min_samples_per_fault=1
    )
    threshold = info["f1"][0]
    covered = float(np.mean(fault_scores >= threshold))
    n = fault_scores.size
    assert abs(covered - 0.9) <= 2.0 / n + 1e-12
    assert covered >= 0.89
