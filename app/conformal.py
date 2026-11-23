"""Conformal 阈值计算与预测集合 / 复测区间生成。

校准阈值采用与 NumPy ``np.quantile(..., method="linear")`` 等价的线性
插值分位数。为避免在校准批次之外再保存一份故障分数，本模块对整个批次
做多轮流式扫描，用固定大小的直方图逐步缩小取值区间，直到定位到所需的
次序统计量。额外工作内存只与故障类型数（以及固定的直方图桶数）有关，
不随校准窗口数增长。
"""

from __future__ import annotations

import math
import struct
from array import array
from typing import Dict, List, Optional, Sequence, Tuple

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


# ---------------------------------------------------------------------- #
# 浮点数 <-> 全序 uint64 键：键序与浮点数大小顺序一致，便于按值域分桶。
# 只处理有限 float64；-0.0 与 +0.0 的键不同，但数值比较语义相同。
# ---------------------------------------------------------------------- #
_KEY_SIGN = 1 << 63
_KEY_MASK = (1 << 64) - 1
_HIST_BITS = 12
_HIST_BUCKETS = 1 << _HIST_BITS
_HIST_MASK = _HIST_BUCKETS - 1


def _float_to_key(value: float) -> int:
    bits = struct.unpack("<Q", struct.pack("<d", value))[0]
    if bits & _KEY_SIGN:
        return (~bits) & _KEY_MASK
    return bits | _KEY_SIGN


def _key_to_float(key: int) -> float:
    if key & _KEY_SIGN:
        bits = key & (_KEY_MASK >> 1)
    else:
        bits = (~key) & _KEY_MASK
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _lerp(a: float, b: float, gamma: float) -> float:
    """与 NumPy ``_lerp`` 相同的浮点运算顺序，保证逐位一致。"""
    diff = b - a
    if gamma >= 0.5:
        return b - diff * (1.0 - gamma)
    return a + diff * gamma


class _ClassSelector:
    """某个故障类型的待选次序统计量集合。

    ``targets`` 中每一项为 ``[lo, hi, pos]``：在键区间 ``[lo, hi]`` 内
    仍未定位的第 ``pos`` 个样本（相对该区间内全部样本的 0 基排名）。
    """

    __slots__ = ("min_key", "max_key", "targets")

    def __init__(self) -> None:
        self.min_key: Optional[int] = None
        self.max_key: Optional[int] = None
        self.targets: List[List[int]] = []


def _select_order_statistics(
    windows: Sequence, selectors: Dict[str, _ClassSelector]
) -> None:
    """多轮流式分桶，把每个 selector 的目标排名定位到具体浮点键。

    每一轮只扫描一遍已解析的窗口、只读取窗口上的分数（不复制、不缓存），
    并用大小固定的直方图缩小候选区间。最坏经过 ceil(64 / _HIST_BITS) 轮
    即可把区间缩到单一键。
    """
    active = {}
    for label, selector in selectors.items():
        if any(target[0] != target[1] for target in selector.targets):
            active[label] = selector

    while active:
        plans: Dict[str, List[Tuple[int, int, int, array, List[List[int]]]]] = {}
        for label, selector in active.items():
            by_range: Dict[Tuple[int, int], List[List[int]]] = {}
            for target in selector.targets:
                if target[0] == target[1]:
                    continue
                by_range.setdefault((target[0], target[1]), []).append(target)
            plan: List[Tuple[int, int, int, array, List[List[int]]]] = []
            for (lo, hi), targets in by_range.items():
                width = (hi - lo + _HIST_MASK + 1) // _HIST_BUCKETS
                hist = array("Q", [0]) * _HIST_BUCKETS
                plan.append((lo, hi, width, hist, targets))
            plans[label] = plan

        # 单遍扫描：每个窗口只贡献它自己故障标签对应分数的一次计数
        for window in windows:
            plan = plans.get(window.label)
            if not plan:
                continue
            label = window.label
            key = _float_to_key(float(window.scores[label]))
            for lo, hi, width, hist, _targets in plan:
                if lo <= key <= hi:
                    hist[(key - lo) // width] += 1

        for plan in plans.values():
            for lo, hi, width, hist, targets in plan:
                for target in targets:
                    position = target[2]
                    cumulative = 0
                    for bucket in range(_HIST_BUCKETS):
                        count = hist[bucket]
                        if count and cumulative + count > position:
                            target[0] = lo + bucket * width
                            target[1] = min(hi, lo + (bucket + 1) * width - 1)
                            target[2] = position - cumulative
                            break
                        cumulative += count
                    else:  # pragma: no cover - 只可能在内部状态被破坏时触发
                        raise RuntimeError(
                            "流式分位数定位失败：目标排名不在候选区间内"
                        )

        active = {}
        for label, selector in selectors.items():
            if any(target[0] != target[1] for target in selector.targets):
                active[label] = selector


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

    实现上不保留任何按故障类型聚合的分数副本：第一遍流式校验并统计
    各类型的样本数、最小值/最大值键，第二段用固定 4096 桶的直方图
    多轮扫描定位分位点两侧的次序统计量，再按 NumPy ``linear`` 方法的
    运算顺序做线性插值。重复分数按出现次数正常计数。
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
    # 每个故障类型只维护常数大小的统计量：样本数与值域键边界
    selectors: Dict[str, _ClassSelector] = {
        ft: _ClassSelector() for ft in fault_list
    }
    counts: Dict[str, int] = {ft: 0 for ft in fault_list}

    prev_ts: Optional[float] = None
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
                key = _float_to_key(value)
                counts[ft] += 1
                selector = selectors[ft]
                if selector.min_key is None or key < selector.min_key:
                    selector.min_key = key
                if selector.max_key is None or key > selector.max_key:
                    selector.max_key = key

    missing_classes = [
        ft for ft in fault_list if counts[ft] < min_samples_per_fault
    ]
    if missing_classes:
        detail = ", ".join(
            f"{ft}(仅 {counts[ft]} 个样本，要求 >= {min_samples_per_fault})"
            for ft in missing_classes
        )
        raise CalibrationValidationError(f"故障类型样本覆盖不足: {detail}")

    threshold_info: Dict[str, Tuple[float, int, float]] = {}
    ranks: Dict[str, float] = {}
    for ft in fault_list:
        m = counts[ft]
        if m == 0:
            raise CalibrationValidationError(
                f"故障类型 {ft!r} 没有故障样本，无法计算阈值"
            )
        # 分数越高越像故障：对非一致性分数 -s 取 (1-alpha) 分位等价于
        # 对 s 取 rank = ceil((n+1)*alpha)/n 的低分位（截顶到 [1/n, 1]）
        rank = min(1.0, math.ceil((m + 1) * alpha) / m)
        ranks[ft] = rank
        virtual = (m - 1) * rank
        # NumPy 对 virtual >= m-1 的情形直接取最大值（previous/next 均为 -1）
        if virtual < m - 1:
            selector = selectors[ft]
            lower_index = math.floor(virtual)
            # 两侧次序统计量；相同区间共享同一张直方图
            for position in (lower_index, lower_index + 1):
                selector.targets.append(
                    [selector.min_key, selector.max_key, position]  # type: ignore[list-item]
                )

    _select_order_statistics(windows, selectors)

    for ft in fault_list:
        m = counts[ft]
        rank = ranks[ft]
        selector = selectors[ft]
        if (m - 1) * rank >= m - 1:
            # 与 NumPy 一致：虚拟索引达到上界时直接取最大值本身
            threshold = _key_to_float(selector.max_key)  # type: ignore[arg-type]
        else:
            virtual = (m - 1) * rank
            lower_index = math.floor(virtual)
            gamma = virtual - lower_index
            lower_key, upper_key = (
                target[0] for target in selector.targets
            )
            threshold = _lerp(
                _key_to_float(lower_key), _key_to_float(upper_key), gamma
            )
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
