"""Conformal 阈值计算与预测集合 / 复测区间生成（NumPy 实现）。"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .schemas import NORMAL_LABEL


class CalibrationValidationError(ValueError):
    """校准批次校验失败（缺类、分数无效、数据顺序问题等）。"""


def _finite_float(value: object) -> float:
    """把分数转换为有限 float；非法值（NaN/Inf/不可解析）抛出异常。"""
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise CalibrationValidationError(f"分数无法解析为数值: {value!r}") from exc
    if not math.isfinite(f):
        raise CalibrationValidationError(f"分数必须为有限数值，收到 {value!r}")
    return f


def validate_and_build_thresholds(
    windows: Sequence,
    fault_types: Sequence[str],
    alpha: float,
    min_samples_per_fault: int,
) -> Tuple[Dict[str, Tuple[float, int, float]], int, int]:
    """校验校准批次并计算每个故障类型的 conformal 阈值。

    返回 ``(threshold_info, normal_count, fault_count)``，其中
    ``threshold_info[ft] = (阈值, 该类型故障样本数, 实际分位点等级)``。

    校验内容：
      1. fault_types 去重后非空，且不含保留标签 ``normal``；
      2. 时间戳严格升序；
      3. 每个窗口都包含全部故障类型的分数，且分数均为有限数值；
      4. 标签只能是 ``normal`` 或 fault_types 之一；
      5. 每个故障类型至少有 ``min_samples_per_fault`` 个故障窗口（覆盖性检查）。

    阈值：对某故障类型 k，取所有 k 类故障窗口在该类型上的分数 s。
    分数越高越像故障，因此阈值取低分位。等价做法是对非一致性分数
    ``-s`` 使用标准有限样本校正分位点
    ``q_{ceil((n+1)*(1-alpha)) / n}``，映射回原分数即
    ``rank = ceil((n+1)*alpha) / n``，对 s 取该 rank 的线性插值分位数。
    该校正保证故障窗口被预测集合覆盖的边际比例 >= 1-alpha。
    """
    if not 0.0 < alpha < 1.0:
        raise CalibrationValidationError("alpha 必须位于 (0, 1) 开区间内")

    fault_list = list(fault_types)
    if len(set(fault_list)) != len(fault_list):
        raise CalibrationValidationError("fault_types 存在重复项")
    if NORMAL_LABEL in fault_list:
        raise CalibrationValidationError(f"{NORMAL_LABEL!r} 是保留标签，不能作为故障类型")

    fault_set = set(fault_list)
    n = len(windows)
    normal_count = 0
    # fault type -> 该类型故障窗口上的对应分数
    fault_scores: Dict[str, List[float]] = {ft: [] for ft in fault_list}

    prev_ts: float | None = None
    for i, w in enumerate(windows):
        ts = _finite_float(w.timestamp)
        if prev_ts is not None and ts <= prev_ts:
            raise CalibrationValidationError(
                f"第 {i} 个窗口时间戳 {ts} 未严格大于前一个窗口 {prev_ts}，窗口必须按时间升序排列"
            )
        prev_ts = ts

        label = w.label
        if label != NORMAL_LABEL and label not in fault_set:
            raise CalibrationValidationError(
                f"第 {i} 个窗口标签 {label!r} 既不是 {NORMAL_LABEL!r} 也不在 fault_types 中"
            )
        if label == NORMAL_LABEL:
            normal_count += 1

        score_map = w.scores
        missing = fault_set - set(score_map)
        if missing:
            raise CalibrationValidationError(
                f"第 {i} 个窗口缺少故障类型分数: {sorted(missing)}"
            )
        for ft in fault_list:
            value = _finite_float(score_map[ft])
            if label == ft:
                fault_scores[ft].append(value)

    missing_classes = [
        ft for ft in fault_list if len(fault_scores[ft]) < min_samples_per_fault
    ]
    if missing_classes:
        detail = ", ".join(
            f"{ft}(仅 {len(fault_scores[ft])} 个样本，要求 >= {min_samples_per_fault})"
            for ft in missing_classes
        )
        raise CalibrationValidationError(f"故障类型样本覆盖不足: {detail}")

    threshold_info: Dict[str, Tuple[float, int, float]] = {}
    for ft in fault_list:
        values = np.asarray(fault_scores[ft], dtype=np.float64)
        m = values.size
        # 分数越高越像故障：对非一致性分数 -s 取 (1-alpha) 分位等价于
        # 对 s 取 rank = ceil((n+1)*alpha)/n 的低分位（截顶到 [1/n, 1]）
        rank = min(1.0, math.ceil((m + 1) * alpha) / m)
        threshold = float(np.quantile(values, rank, method="linear"))
        threshold_info[ft] = (threshold, m, rank)

    return threshold_info, normal_count, n - normal_count


def predict_sets(
    windows: Sequence,
    fault_types: Sequence[str],
    thresholds: Dict[str, float],
) -> List[List[str]]:
    """根据窗口分数与阈值生成预测集合（分数 >= 阈值即纳入该故障类型）。

    预测集合为空表示判为正常。分数缺失或非有限值会抛出异常。
    """
    results: List[List[str]] = []
    for i, w in enumerate(windows):
        score_map = w.scores
        prediction: List[str] = []
        for ft in fault_types:
            if ft not in score_map:
                raise CalibrationValidationError(
                    f"第 {i} 个检测窗口缺少故障类型 {ft!r} 的分数"
                )
            value = _finite_float(score_map[ft])
            if value >= thresholds[ft]:
                prediction.append(ft)
        results.append(prediction)
    return results


def merge_alarm_intervals(
    alarms: Sequence[bool], merge_gap: int
) -> List[Tuple[int, int]]:
    """把告警窗口合并为连续复测区间。

    返回 ``[起始索引, 结束索引]`` 闭区间列表。两个告警块之间若至多有
    ``merge_gap`` 个非告警窗口，则合并为同一区间（间隔窗口也纳入复测）。
    ``merge_gap=0`` 时只合并严格相邻的告警窗口。
    """
    if merge_gap < 0:
        raise CalibrationValidationError("merge_gap 不能为负数")
    intervals: List[Tuple[int, int]] = []
    for idx, is_alarm in enumerate(alarms):
        if not is_alarm:
            continue
        if intervals and idx - intervals[-1][1] - 1 <= merge_gap:
            start, _ = intervals[-1]
            intervals[-1] = (start, idx)
        else:
            intervals.append((idx, idx))
    return intervals
