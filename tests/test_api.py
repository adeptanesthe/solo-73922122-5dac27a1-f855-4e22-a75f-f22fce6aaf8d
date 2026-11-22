"""API 层测试：发布、检测、版本钉住、回退、缺类与异常分数。"""

import json

from tests.conftest import make_window


def _publish(client, payload, expected=200):
    resp = client.post("/calibration/publish", json=payload)
    assert resp.status_code == expected, resp.text
    return resp.json()


def test_publish_and_health(client, calibration_payload):
    data = _publish(client, calibration_payload())
    assert data["status"] == "published"
    assert data["active_version"] == "cal-v1"
    thresholds = data["calibration"]["thresholds"]
    assert set(thresholds) == {"short", "leak"}
    assert thresholds["short"]["fault_samples"] == 5

    health = client.get("/health").json()
    assert health["active_version"] == "cal-v1"


def test_detection_returns_prediction_sets_and_merged_intervals(client, calibration_payload):
    _publish(client, calibration_payload())
    short_th = client.get("/calibration/versions/cal-v1").json()["thresholds"]["short"][
        "threshold"
    ]
    leak_th = client.get("/calibration/versions/cal-v1").json()["thresholds"]["leak"][
        "threshold"
    ]

    windows = [
        {"timestamp": 100.0, "scores": {"short": short_th + 0.2, "leak": 0.05}},
        {"timestamp": 101.0, "scores": {"short": short_th + 0.1, "leak": 0.05}},
        {"timestamp": 102.0, "scores": {"short": 0.01, "leak": 0.01}},
        {"timestamp": 103.0, "scores": {"short": 0.01, "leak": leak_th + 0.1}},
        {"timestamp": 104.0, "scores": {"short": 0.01, "leak": 0.01}},
    ]
    resp = client.post(
        "/detection", json={"version": "cal-v1", "windows": windows, "merge_gap": 0}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["used_version"] == "cal-v1"
    assert body["fallback"] is False
    flags = [w["is_alarm"] for w in body["windows"]]
    assert flags == [True, True, False, True, False]
    assert body["windows"][0]["prediction_set"] == ["short"]
    assert body["windows"][3]["prediction_set"] == ["leak"]
    # 严格相邻合并：前两个一组，第四个单独一组
    intervals = body["retest_intervals"]
    assert [(i["start_timestamp"], i["end_timestamp"]) for i in intervals] == [
        (100.0, 101.0),
        (103.0, 103.0),
    ]
    assert intervals[0]["fault_types"] == ["short"]
    assert intervals[0]["indices"] == [0, 1]

    # merge_gap=1 时，第二块与第一块间隔 1 个正常窗口，应合并
    resp2 = client.post(
        "/detection", json={"version": "cal-v1", "windows": windows, "merge_gap": 1}
    )
    intervals2 = resp2.json()["retest_intervals"]
    assert len(intervals2) == 1
    assert intervals2[0]["indices"] == [0, 1, 2, 3]
    assert intervals2[0]["fault_types"] == ["leak", "short"]


def test_detection_uses_requested_version_not_active(client, calibration_payload):
    """发布 v2 后，请求钉住 v1 时全程使用 v1 的阈值。"""
    p1 = calibration_payload(version="cal-v1")
    _publish(client, p1)
    v1 = client.get("/calibration/versions/cal-v1").json()

    p2 = calibration_payload(version="cal-v2")
    # v2 把 short 故障分数整体抬高，阈值应显著高于 v1
    for w in p2["windows"]:
        if w["label"] == "short":
            w["scores"]["short"] += 0.5
    _publish(client, p2)
    v2 = client.get("/calibration/versions/cal-v2").json()
    assert client.get("/health").json()["active_version"] == "cal-v2"
    assert v2["thresholds"]["short"]["threshold"] > v1["thresholds"]["short"]["threshold"]

    probe = [
        {
            "timestamp": 1.0,
            "scores": {
                "short": v1["thresholds"]["short"]["threshold"] + 0.05,
                "leak": 0.01,
            },
        }
    ]
    # 钉住 v1：分数超过 v1 阈值 -> 告警
    r1 = client.post("/detection", json={"version": "cal-v1", "windows": probe}).json()
    assert r1["used_version"] == "cal-v1"
    assert r1["windows"][0]["is_alarm"] is True
    # 同样的分数在 v2 下低于阈值 -> 正常
    r2 = client.post("/detection", json={"version": "cal-v2", "windows": probe}).json()
    assert r2["used_version"] == "cal-v2"
    assert r2["windows"][0]["is_alarm"] is False


def test_unknown_version_404_and_retains_previous(client, calibration_payload):
    _publish(client, calibration_payload())
    resp = client.post(
        "/detection",
        json={"version": "cal-ghost", "windows": [
            {"timestamp": 1.0, "scores": {"short": 0.1, "leak": 0.1}}
        ]},
    )
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["retained_version"] == "cal-v1"
    assert "不存在" in detail["reason"]


def test_unknown_version_fallback_to_active(client, calibration_payload):
    _publish(client, calibration_payload(version="cal-v1"))
    resp = client.post(
        "/detection",
        json={
            "version": "cal-ghost",
            "allow_fallback": True,
            "windows": [
                {"timestamp": 1.0, "scores": {"short": 0.01, "leak": 0.01}}
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["fallback"] is True
    assert body["used_version"] == "cal-v1"
    assert "cal-ghost" in body["warning"]


def test_missing_fault_class_keeps_previous_version(client, calibration_payload):
    _publish(client, calibration_payload(version="cal-v1"))
    bad = calibration_payload(version="cal-bad")
    # 删掉所有 leak 故障窗口
    bad["windows"] = [w for w in bad["windows"] if w["label"] != "leak"]
    resp = client.post("/calibration/publish", json=bad)
    assert resp.status_code == 422
    body = resp.json()
    assert body["status"] == "rejected"
    assert "覆盖不足" in body["detail"]
    assert body["retained_version"] == "cal-v1"
    # 活动版本未变，版本列表里也没有坏版本
    assert client.get("/health").json()["active_version"] == "cal-v1"
    versions = client.get("/calibration/versions").json()
    assert [v["version"] for v in versions] == ["cal-v1"]


def test_invalid_scores_rejected_and_retain(client, calibration_payload):
    _publish(client, calibration_payload(version="cal-v1"))

    # 校准批次含 NaN / Infinity：用原始 JSON 文本发送（标准 JSON 不允许，
    # 服务端解析后必须在有限性校验环节拒绝）
    for token in ("NaN", "Infinity", "-Infinity"):
        bad = calibration_payload(version=f"cal-bad-{token}")
        bad["windows"][0]["scores"]["short"] = "@@"  # 占位，下面替换
        raw = json.dumps(bad).replace('"@@"', token)
        resp = client.post(
            "/calibration/publish",
            content=raw,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422, (token, resp.text)
        assert "有限数值" in resp.json()["detail"]
        assert resp.json()["retained_version"] == "cal-v1"

    # 检测请求含 inf / nan
    for token in ("Infinity", "NaN"):
        raw = (
            '{"version": "cal-v1", "windows": [{"timestamp": 1.0,'
            f' "scores": {{"short": {token}, "leak": 0.1}}}}]}}'
        )
        resp = client.post(
            "/detection",
            content=raw,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400, (token, resp.text)
        assert "有限数值" in resp.json()["detail"]
    assert client.get("/health").json()["active_version"] == "cal-v1"


def test_duplicate_version_rejected(client, calibration_payload):
    _publish(client, calibration_payload(version="cal-v1"))
    resp = client.post("/calibration/publish", json=calibration_payload(version="cal-v1"))
    assert resp.status_code == 409
    assert resp.json()["retained_version"] == "cal-v1"


def test_unsorted_detection_windows_rejected(client, calibration_payload):
    _publish(client, calibration_payload())
    resp = client.post(
        "/detection",
        json={
            "version": "cal-v1",
            "windows": [
                {"timestamp": 2.0, "scores": {"short": 0.1, "leak": 0.1}},
                {"timestamp": 1.0, "scores": {"short": 0.1, "leak": 0.1}},
            ],
        },
    )
    assert resp.status_code == 400
    assert "升序" in resp.json()["detail"]


def test_snapshot_persisted_and_reloaded(client, calibration_payload, tmp_path, monkeypatch):
    _publish(client, calibration_payload(version="cal-v1"))
    from app.main import store

    assert store._snapshot_path
    import os

    assert os.path.exists(store._snapshot_path)

    # 模拟进程重启：新建 store 实例从同一快照恢复
    from app.store import CalibrationStore

    reloaded = CalibrationStore(snapshot_path=store._snapshot_path)
    assert reloaded.active_version == "cal-v1"
    result = reloaded.get("cal-v1")
    assert result.fault_types == ["short", "leak"]


def test_empty_prediction_set_means_normal(client, calibration_payload):
    _publish(client, calibration_payload())
    resp = client.post(
        "/detection",
        json={
            "version": "cal-v1",
            "windows": [
                {"timestamp": 1.0, "scores": {"short": 0.0, "leak": 0.0}},
                {"timestamp": 2.0, "scores": {"short": 0.0, "leak": 0.0}},
            ],
        },
    )
    body = resp.json()
    assert all(not w["is_alarm"] for w in body["windows"])
    assert all(w["prediction_set"] == [] for w in body["windows"])
    assert body["retest_intervals"] == []


def test_malformed_json_body_rejected(client):
    resp = client.post(
        "/calibration/publish",
        content="{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


def test_detection_missing_score_key_rejected(client, calibration_payload):
    _publish(client, calibration_payload())
    resp = client.post(
        "/detection",
        json={
            "version": "cal-v1",
            "windows": [{"timestamp": 1.0, "scores": {"short": 0.5}}],
        },
    )
    assert resp.status_code == 400
    assert "缺少故障类型" in resp.json()["detail"]


def test_fallback_warning_is_explicit(client, calibration_payload):
    _publish(client, calibration_payload(version="cal-v1"))
    resp = client.post(
        "/detection",
        json={
            "version": "cal-missing",
            "allow_fallback": True,
            "windows": [
                {"timestamp": 1.0, "scores": {"short": 0.99, "leak": 0.99}}
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["fallback"] is True
    assert body["used_version"] == "cal-v1"
    assert body["warning"] and "cal-missing" in body["warning"]
    assert body["windows"][0]["is_alarm"] is True
