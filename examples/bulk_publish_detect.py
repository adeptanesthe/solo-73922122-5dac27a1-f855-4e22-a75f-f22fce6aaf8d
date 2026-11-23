"""大批次校准发布 + 大批次检测的调用示例（需先启动服务）。

运行：python3 examples/bulk_publish_detect.py
可用环境变量：
    BASE_URL       默认 http://localhost:8000
    VERSION        默认 bulk-v1（同名版本重复发布会得到 409，便于观察冲突语义）
    WINDOWS        校准窗口数，默认 20000
    DETECT_WINDOWS 检测窗口数，默认 5000
依赖：httpx、numpy（已在 requirements.txt 中）

脚本在本地用 NumPy 整批算法算出参考阈值，与接口返回结果逐项比对，
随后提交一次大批次检测请求，最后验证失败批次不会影响检测。
"""

import math
import os
import time

import httpx
import numpy as np

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
VERSION = os.environ.get("VERSION", "bulk-v1")
N_WINDOWS = int(os.environ.get("WINDOWS", "20000"))
N_DETECT = int(os.environ.get("DETECT_WINDOWS", "5000"))
ALPHA = 0.1
FAULT_TYPES = ["short", "leak", "thermal"]
MIN_SAMPLES = 20


def build_calibration_batch(seed: int = 20261123) -> tuple[list[dict], dict]:
    """按整批方式构造校准窗口，并记录每个故障类型的参考分数。

    约 20% 窗口标签为 normal（分数整体偏低），其余按故障类型轮转，
    保证每类故障样本充足。
    """
    rng = np.random.default_rng(seed)
    labels = []
    for i in range(N_WINDOWS):
        labels.append("normal" if i % 5 == 0 else FAULT_TYPES[i % len(FAULT_TYPES)])
    windows = []
    reference: dict[str, list[float]] = {ft: [] for ft in FAULT_TYPES}
    for i, label in enumerate(labels):
        scores = {}
        for ft in FAULT_TYPES:
            if label == "normal":
                value = float(rng.normal(loc=0.15, scale=0.05))
            else:
                value = float(rng.normal(loc=0.75 if ft == label else 0.20, scale=0.06))
            scores[ft] = value
            if ft == label:
                reference[ft].append(value)
        windows.append(
            {"timestamp": float(i), "label": label, "scores": scores}
        )
    return windows, reference


def numpy_reference(reference: dict[str, list[float]]) -> dict[str, tuple[float, int, float]]:
    out = {}
    for ft, values in reference.items():
        arr = np.asarray(values, dtype=np.float64)
        m = arr.size
        rank = min(1.0, math.ceil((m + 1) * ALPHA) / m)
        out[ft] = (float(np.quantile(arr, rank, method="linear")), m, rank)
    return out


def build_detection_batch(thresholds: dict[str, float], seed: int = 7) -> list[dict]:
    """约 1/3 窗口取某类高分触发告警，其余为正常分数。"""
    rng = np.random.default_rng(seed)
    windows = []
    for i in range(N_DETECT):
        scores = {}
        trigger = FAULT_TYPES[i % len(FAULT_TYPES)] if i % 3 == 0 else None
        for ft in FAULT_TYPES:
            if ft == trigger:
                scores[ft] = float(thresholds[ft] + rng.uniform(0.01, 0.2))
            else:
                scores[ft] = float(rng.uniform(0.0, 0.3))
        windows.append({"timestamp": 1_000_000.0 + i, "scores": scores})
    return windows


def main() -> None:
    windows, reference = build_calibration_batch()
    print(
        f"校准批次: {len(windows)} 个窗口 × {len(FAULT_TYPES)} 类分数"
        f"（约 {len(windows) * len(FAULT_TYPES)} 个分数）"
    )
    payload = {
        "version": VERSION,
        "fault_types": FAULT_TYPES,
        "alpha": ALPHA,
        "min_samples_per_fault": MIN_SAMPLES,
        "windows": windows,
    }

    expected = numpy_reference(reference)

    with httpx.Client(base_url=BASE_URL, timeout=120.0) as client:
        print("== /health ==")
        print(client.get("/health").json())

        print(f"\n== 发布 {VERSION}（整批提交）==")
        started = time.perf_counter()
        resp = client.post("/calibration/publish", json=payload)
        elapsed = time.perf_counter() - started
        if resp.status_code != 200:
            print(resp.status_code, resp.text)
            raise SystemExit(1)
        body = resp.json()
        print(f"状态: {body['status']}，活动版本: {body['active_version']}，"
              f"耗时: {elapsed * 1000:.0f} ms")
        print(
            f"窗口总数: {body['calibration']['total_windows']}，"
            f"正常窗口: {body['calibration']['normal_windows']}"
        )

        print("\n== 阈值一致性（接口 vs NumPy 整批计算）==")
        ok = True
        for ft in FAULT_TYPES:
            got = body["calibration"]["thresholds"][ft]
            exp_threshold, exp_samples, exp_rank = expected[ft]
            match = got["threshold"] == exp_threshold
            ok = ok and match
            print(
                f"  {ft}: threshold={got['threshold']!r} (numpy {exp_threshold!r}) "
                f"match={match} rank={got['quantile_rank']!r} samples={got['fault_samples']}"
            )
        if not ok:
            raise SystemExit("阈值与 NumPy 参考不一致")

        print(f"\n== 检测 {N_DETECT} 个窗口（钉住 {VERSION}，merge_gap=1）==")
        detect_payload = {
            "version": VERSION,
            "merge_gap": 1,
            "windows": build_detection_batch(
                {ft: body["calibration"]["thresholds"][ft]["threshold"] for ft in FAULT_TYPES}
            ),
        }
        started = time.perf_counter()
        resp = client.post("/detection", json=detect_payload)
        elapsed = time.perf_counter() - started
        if resp.status_code != 200:
            print(resp.status_code, resp.text)
            raise SystemExit(1)
        detected = resp.json()
        alarms = sum(1 for w in detected["windows"] if w["is_alarm"])
        print(
            f"检测完成: used_version={detected['used_version']} fallback={detected['fallback']}，"
            f"告警窗口 {alarms}/{len(detected['windows'])}，"
            f"复测区间 {len(detected['retest_intervals'])} 个，耗时 {elapsed * 1000:.0f} ms"
        )
        if detected["retest_intervals"]:
            first = detected["retest_intervals"][0]
            print(f"首个复测区间: {first['start_timestamp']}..{first['end_timestamp']}，"
                  f"{first['window_count']} 个窗口，类型 {first['fault_types']}")

        print("\n== 计算期间检测可用性（并发发布时的上一版检测）==")
        # 提交一个半样本批次；即使它被拒绝，检测也应保持可用
        bad = dict(payload)
        bad["version"] = f"{VERSION}-bad"
        bad["windows"] = [w for w in windows if w["label"] != FAULT_TYPES[-1]]
        resp_bad = client.post("/calibration/publish", json=bad)
        print(f"缺类批次: {resp_bad.status_code} {resp_bad.json().get('retained_version')}")
        probe = client.post(
            "/detection",
            json={
                "version": VERSION,
                "windows": [{"timestamp": 1.0, "scores": {ft: 0.99 for ft in FAULT_TYPES}}],
            },
        )
        print(
            f"失败后检测: {probe.status_code}，used_version="
            f"{probe.json().get('used_version')}"
        )


if __name__ == "__main__":
    main()
