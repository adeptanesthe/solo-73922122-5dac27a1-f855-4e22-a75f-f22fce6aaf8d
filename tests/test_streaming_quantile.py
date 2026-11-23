"""流式分位数实现：与 NumPy 整批计算逐位一致，且内存不随窗口数增长。"""

import gc
import math
import tracemalloc

import numpy as np
import pytest

from app.conformal import validate_and_build_thresholds
from app.schemas import CalibrationWindow


def _windows(labels, values):
    return [
        CalibrationWindow(
            timestamp=float(i), label=label, scores={"f1": float(v), "f2": 0.0}
        )
        for i, (label, v) in enumerate(zip(labels, values))
    ]


def _numpy_threshold(values, alpha):
    values = np.asarray(values, dtype=np.float64)
    m = values.size
    rank = min(1.0, math.ceil((m + 1) * alpha) / m)
    return float(np.quantile(values, rank, method="linear")), rank


@pytest.mark.parametrize("alpha", [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 0.9, 0.99, 0.999])
@pytest.mark.parametrize(
    "generator",
    [
        lambda m, rng: rng.normal(size=m),
        lambda m, rng: rng.integers(-4, 5, size=m).astype(np.float64),
        lambda m, rng: np.round(rng.normal(size=m), 2),
        lambda m, rng: np.where(rng.random(m) < 0.5, 0.0, -0.0),
        lambda m, rng: rng.normal(size=m) * 1e200,
        lambda m, rng: np.full(m, 0.813),
    ],
)
@pytest.mark.parametrize("m", [1, 2, 3, 5, 7, 17, 61, 299])
def test_threshold_bit_exact_vs_numpy(alpha, generator, m):
    rng = np.random.default_rng(m * 100003 + int(alpha * 1000))
    values = generator(m, rng)
    expected, expected_rank = _numpy_threshold(values, alpha)
    windows = _windows(["f1"] * m, values)
    info, _, _ = validate_and_build_thresholds(
        windows, ["f1"], alpha=alpha, min_samples_per_fault=1
    )
    threshold, n_samples, rank = info["f1"]
    assert n_samples == m
    assert rank == expected_rank
    # 逐位一致（-0.0 == 0.0 在数值上等价，单靠 == 即可覆盖本用例）
    assert threshold == expected


def test_duplicate_scores_are_counted_normally():
    values = [0.5] * 4 + [0.1, 0.9, 0.7, 0.3]
    windows = _windows(["f1"] * len(values), values)
    info, _, _ = validate_and_build_thresholds(
        windows, ["f1"], alpha=0.2, min_samples_per_fault=1
    )
    expected, rank = _numpy_threshold(values, 0.2)
    assert info["f1"] == (expected, len(values), rank)


def _batch(per_class: int, classes: int = 4):
    fault_types = [f"f{i}" for i in range(classes)]
    windows = []
    ts = 0.0
    for cls, ft in enumerate(fault_types):
        for i in range(per_class):
            scores = {
                other: (0.8 if j == cls else 0.1) + 0.0001 * i
                for j, other in enumerate(fault_types)
            }
            windows.append(
                CalibrationWindow(timestamp=ts, label=ft, scores=scores)
            )
            ts += 1.0
    return windows, fault_types


def test_extra_memory_does_not_grow_with_window_count():
    """校验 + 分位数计算的额外峰值内存应与窗口数无关（已解析请求不计）。"""
    peaks = []
    for per_class in (200, 800, 3200, 12800):
        windows, fault_types = _batch(per_class)
        gc.collect()
        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]
        validate_and_build_thresholds(windows, fault_types, alpha=0.1, min_samples_per_fault=2)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak - baseline)
    # 窗口数从 800 涨到 51200（64 倍），额外内存波动应远小于总量增长
    assert max(peaks) - min(peaks) < 50_000, peaks
    assert max(peaks) < 1_000_000, peaks


def test_large_batch_thresholds_match_numpy():
    """大批次整批提交：各故障类型阈值与 NumPy 整批计算一致。"""
    rng = np.random.default_rng(11)
    fault_types = [f"f{i}" for i in range(5)]
    n = 4000
    windows = []
    labels = [fault_types[i % len(fault_types)] for i in range(n)]
    scores_by_class = {}
    for i, label in enumerate(labels):
        scores = {}
        for ft in fault_types:
            scores[ft] = float(rng.normal(loc=0.7 if ft == label else 0.2, scale=0.1))
            if ft == label:
                scores_by_class.setdefault(ft, []).append(scores[ft])
        windows.append(CalibrationWindow(timestamp=float(i), label=label, scores=scores))
    info, normal_n, fault_n = validate_and_build_thresholds(
        windows, fault_types, alpha=0.1, min_samples_per_fault=1
    )
    assert normal_n == 0 and fault_n == n
    for ft in fault_types:
        expected, rank = _numpy_threshold(scores_by_class[ft], 0.1)
        assert info[ft][0] == expected
        assert info[ft][1] == len(scores_by_class[ft])
        assert info[ft][2] == rank
