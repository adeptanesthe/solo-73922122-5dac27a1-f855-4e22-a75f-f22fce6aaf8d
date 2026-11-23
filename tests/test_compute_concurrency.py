"""发布计算期间检测可用：阈值计算不持锁，新版在成功发布前不可见。"""

import threading

from tests.conftest import make_window


def _large_payload(version="cal-big", per_class=1500):
    windows = []
    ts = 0.0
    for i in range(per_class // 10):
        windows.append(make_window(ts, "normal", {"short": 0.05, "leak": 0.05}))
        ts += 1.0
    for label, key, other in (("short", "short", "leak"), ("leak", "leak", "short")):
        for i in range(per_class):
            windows.append(
                make_window(
                    ts,
                    label,
                    {key: 0.7 + (i % 100) * 0.001, other: 0.1},
                )
            )
            ts += 1.0
    return {
        "version": version,
        "fault_types": ["short", "leak"],
        "alpha": 0.1,
        "min_samples_per_fault": 10,
        "windows": windows,
    }


def test_detection_during_threshold_computation(client, calibration_payload, monkeypatch):
    """阈值计算进行中（尚未落盘/切换）检测仍按上一版完成，且看不到新版本。"""
    client.post("/calibration/publish", json=calibration_payload("cal-v1"))

    import app.main as main_module

    real_validate = main_module.validate_and_build_thresholds
    computing = threading.Event()
    may_finish = threading.Event()

    def slow_validate(*args, **kwargs):
        computing.set()
        # 在“计算阶段”停住，等待检测线程完成
        assert may_finish.wait(10.0), "检测线程未在计算期间完成"
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(main_module, "validate_and_build_thresholds", slow_validate)

    outcomes = []
    errors = []

    def publisher():
        resp = client.post("/calibration/publish", json=_large_payload())
        outcomes.append(resp.status_code)

    def detector():
        assert computing.wait(10.0), "发布未进入计算阶段"
        # 计算期间：新版本尚不存在，检测应 404 并保留上一版
        r = client.post(
            "/detection",
            json={
                "version": "cal-big",
                "windows": [{"timestamp": 1.0, "scores": {"short": 0.9, "leak": 0.1}}],
            },
        )
        if r.status_code != 404:
            errors.append(("ghost-during-compute", r.status_code, r.text))
        elif r.json()["detail"]["retained_version"] != "cal-v1":
            errors.append(("retained", r.json()["detail"]))
        # 已发布版本检测正常，使用 cal-v1 阈值
        for i in range(5):
            r = client.post(
                "/detection",
                json={
                    "version": "cal-v1",
                    "windows": [
                        {"timestamp": 100.0 + i, "scores": {"short": 0.99, "leak": 0.01}}
                    ],
                },
            )
            if r.status_code != 200:
                errors.append(("detect", r.status_code, r.text))
                continue
            body = r.json()
            if body["used_version"] != "cal-v1" or not body["windows"][0]["is_alarm"]:
                errors.append(("result", body["used_version"]))
        may_finish.set()

    t_pub = threading.Thread(target=publisher)
    t_det = threading.Thread(target=detector)
    t_pub.start()
    t_det.start()
    t_det.join(15.0)
    t_pub.join(30.0)
    monkeypatch.undo()

    assert errors == [], errors
    assert outcomes == [200]
    assert client.get("/health").json()["active_version"] == "cal-big"
    # 发布成功后新版本立即可用
    r = client.post(
        "/detection",
        json={
            "version": "cal-big",
            "windows": [{"timestamp": 1.0, "scores": {"short": 0.99, "leak": 0.99}}],
        },
    )
    assert r.status_code == 200
    assert r.json()["used_version"] == "cal-big"
    assert r.json()["windows"][0]["is_alarm"] is True
