"""发布失败语义：落盘失败回滚、临时文件清理、计算期间检测仍可用。"""

import glob
import json
import os
import threading

import pytest
from fastapi.testclient import TestClient

from app import store as store_module
from app.store import CalibrationStore, SnapshotWriteError
from tests.conftest import make_window


def _payload(version="cal-v2", bias=0.0):
    windows = [
        make_window(1.0, "normal", {"short": 0.05, "leak": 0.05}),
        make_window(2.0, "normal", {"short": 0.06, "leak": 0.05}),
    ]
    for i in range(5):
        windows.append(
            make_window(10.0 + i, "short", {"short": 0.80 + bias + 0.02 * i, "leak": 0.10})
        )
    for i in range(5):
        windows.append(
            make_window(20.0 + i, "leak", {"short": 0.10, "leak": 0.70 + bias + 0.03 * i})
        )
    return {
        "version": version,
        "fault_types": ["short", "leak"],
        "alpha": 0.2,
        "min_samples_per_fault": 3,
        "windows": windows,
    }


def _temp_files(path):
    directory = os.path.dirname(os.path.abspath(path))
    return glob.glob(os.path.join(directory, ".cal-*.tmp"))


def test_snapshot_write_failure_rolls_back(client, calibration_payload, monkeypatch, tmp_path):
    """落盘失败：返回 500、保留上一版、不留临时文件、新版本不可查询/检测。"""
    _publish_ok = client.post("/calibration/publish", json=calibration_payload("cal-v1"))
    assert _publish_ok.status_code == 200

    from app.main import store

    real_replace = store_module.os.replace

    def boom(src, dst):
        raise OSError("模拟磁盘写入失败")

    monkeypatch.setattr(store_module.os, "replace", boom)
    resp = client.post("/calibration/publish", json=_payload("cal-v2", 0.05))
    monkeypatch.undo()

    assert resp.status_code == 500, resp.text
    body = resp.json()
    assert body["status"] == "rejected"
    assert body["retained_version"] == "cal-v1"
    assert "落盘失败" in body["reason"]

    # 临时文件已清理
    assert _temp_files(store._snapshot_path) == []

    # 活动版本仍为上一版；新版本不可见
    assert client.get("/health").json()["active_version"] == "cal-v1"
    versions = [v["version"] for v in client.get("/calibration/versions").json()]
    assert versions == ["cal-v1"]
    assert client.get("/calibration/versions/cal-v2").status_code == 404

    det = client.post(
        "/detection",
        json={
            "version": "cal-v2",
            "windows": [{"timestamp": 1.0, "scores": {"short": 0.99, "leak": 0.99}}],
        },
    )
    assert det.status_code == 404
    assert det.json()["detail"]["retained_version"] == "cal-v1"

    # 磁盘快照仍是上一版，可恢复
    with open(store._snapshot_path, encoding="utf-8") as fh:
        snapshot = json.load(fh)
    assert snapshot["active_version"] == "cal-v1"
    assert set(snapshot["versions"]) == {"cal-v1"}

    reloaded = CalibrationStore(snapshot_path=store._snapshot_path)
    assert reloaded.active_version == "cal-v1"
    with pytest.raises(Exception):
        reloaded.get("cal-v2")


def test_stage_failure_during_write_cleans_temp_file(client, calibration_payload, monkeypatch):
    """写临时文件阶段失败（fsync 抛错）：不替换正式快照，也不留临时文件。"""
    client.post("/calibration/publish", json=calibration_payload("cal-v1"))
    from app.main import store

    real_fsync = os.fsync

    def boom(fd):
        raise OSError("模拟 fsync 失败")

    monkeypatch.setattr(store_module.os, "fsync", boom)
    resp = client.post("/calibration/publish", json=_payload("cal-v2"))
    monkeypatch.setattr(store_module.os, "fsync", real_fsync)

    assert resp.status_code == 500, resp.text
    assert resp.json()["retained_version"] == "cal-v1"
    assert _temp_files(store._snapshot_path) == []
    assert client.get("/health").json()["active_version"] == "cal-v1"
    with open(store._snapshot_path, encoding="utf-8") as fh:
        assert set(json.load(fh)["versions"]) == {"cal-v1"}


def test_detection_available_during_publish(client, calibration_payload, monkeypatch):
    """发布（含落盘）进行期间，检测线程始终能按已发布版本完成检测。"""
    client.post("/calibration/publish", json=calibration_payload("cal-v1"))

    from app.main import store

    slow_enter = threading.Event()
    release = threading.Event()
    real_stage = store._stage_snapshot

    def slow_stage(versions, active):
        slow_enter.set()
        assert release.wait(5.0), "测试未及时放行发布"
        return real_stage(versions, active)

    monkeypatch.setattr(store, "_stage_snapshot", slow_stage)

    results = []
    errors = []

    def publish():
        resp = client.post("/calibration/publish", json=_payload("cal-v2", 0.05))
        results.append((resp.status_code, resp.json()))

    def detect():
        assert slow_enter.wait(5.0)
        for i in range(20):
            resp = client.post(
                "/detection",
                json={
                    "version": "cal-v1",
                    "windows": [
                        {"timestamp": 100.0 + i, "scores": {"short": 0.9, "leak": 0.1}}
                    ],
                },
            )
            if resp.status_code != 200:
                errors.append((resp.status_code, resp.text))
            else:
                body = resp.json()
                if body["used_version"] != "cal-v1":
                    errors.append(("version", body["used_version"]))

    t_pub = threading.Thread(target=publish)
    t_det = threading.Thread(target=detect)
    t_pub.start()
    t_det.start()
    t_det.join()
    release.set()
    t_pub.join()

    monkeypatch.undo()
    assert errors == [], errors
    assert results and results[0][0] == 200, results
    assert client.get("/health").json()["active_version"] == "cal-v2"


def test_temporary_batch_data_not_retained_after_rejected_publish(
    client, calibration_payload, monkeypatch
):
    """校验失败（缺类）时，批次分数不会进入版本库或快照。"""
    client.post("/calibration/publish", json=calibration_payload("cal-v1"))
    bad = _payload("cal-bad")
    bad["windows"] = [w for w in bad["windows"] if w["label"] != "leak"]
    resp = client.post("/calibration/publish", json=bad)
    assert resp.status_code == 422

    from app.main import store

    assert store.active_version == "cal-v1"
    assert "cal-bad" not in store._versions
    with open(store._snapshot_path, encoding="utf-8") as fh:
        assert set(json.load(fh)["versions"]) == {"cal-v1"}
    assert _temp_files(store._snapshot_path) == []
