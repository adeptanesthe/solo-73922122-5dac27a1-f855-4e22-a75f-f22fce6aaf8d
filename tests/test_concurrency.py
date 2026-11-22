"""并发测试：并发换版发布与并发检测，验证原子性与版本钉住。"""

import threading

from tests.conftest import make_window


def _build_version(version: str, short_bias: float):
    windows = []
    for i, ts in enumerate([1.0, 2.0, 3.0]):
        windows.append(
            make_window(ts, "normal", {"short": 0.05, "leak": 0.05})
        )
    for i, ts in enumerate([10.0, 11.0, 12.0, 13.0]):
        windows.append(
            make_window(
                ts,
                "short",
                {"short": 0.7 + short_bias + i * 0.02, "leak": 0.1},
            )
        )
    for i, ts in enumerate([20.0, 21.0, 22.0, 23.0]):
        windows.append(
            make_window(
                ts,
                "leak",
                {"short": 0.1, "leak": 0.6 + short_bias + i * 0.02},
            )
        )
    return {
        "version": version,
        "fault_types": ["short", "leak"],
        "alpha": 0.2,
        "min_samples_per_fault": 2,
        "windows": windows,
    }


# 预先发布、并发期间必定存在的版本；检测线程只钉这些版本
_PINNED_VERSIONS = ["cal-v0", "cal-v1"]
# 并发期间新发布的版本
_NEW_VERSIONS = [f"cal-v{i}" for i in range(2, 7)]


def test_concurrent_publish_and_detection(client):
    # 先发布两个基线版本，检测线程全程钉住它们
    for v, bias in (("cal-v0", 0.0), ("cal-v1", 0.01)):
        r = client.post("/calibration/publish", json=_build_version(v, bias))
        assert r.status_code == 200

    errors = []
    published = []
    not_found_seen = threading.Event()
    barrier = threading.Barrier(7)  # 1 个发布线程 + 1 个探针 + 5 个检测线程

    def publisher():
        barrier.wait()
        for i, version in enumerate(_NEW_VERSIONS, start=2):
            resp = client.post(
                "/calibration/publish", json=_build_version(version, 0.01 * i)
            )
            if resp.status_code != 200:
                errors.append(("publish", resp.status_code, resp.text))
            else:
                published.append(version)

    def detector(worker_id: int):
        barrier.wait()
        for i in range(30):
            # 只钉预先存在的版本，used_version 必须严格等于声明版本
            version = _PINNED_VERSIONS[i % len(_PINNED_VERSIONS)]
            resp = client.post(
                "/detection",
                json={
                    "version": version,
                    "windows": [
                        {"timestamp": 100.0 + i, "scores": {"short": 0.9, "leak": 0.1}},
                        {"timestamp": 101.0 + i, "scores": {"short": 0.1, "leak": 0.8}},
                    ],
                },
            )
            if resp.status_code != 200:
                errors.append(("detect", resp.status_code, resp.text))
                continue
            body = resp.json()
            if body["used_version"] != version:
                errors.append(("version-mismatch", version, body["used_version"]))
            if len(body["windows"]) != 2 or body["fallback"]:
                errors.append(("response-inconsistent", body))

    def not_yet_published_probe():
        """并发期间请求尚未发布的版本：必须稳定 404 且保留版本不变。"""
        barrier.wait()
        for version in _NEW_VERSIONS:
            resp = client.post(
                "/detection",
                json={
                    "version": version,
                    "windows": [
                        {"timestamp": 1.0, "scores": {"short": 0.9, "leak": 0.1}}
                    ],
                },
            )
            if resp.status_code == 404:
                not_found_seen.set()
                retained = resp.json()["detail"]["retained_version"]
                if retained not in _PINNED_VERSIONS + _NEW_VERSIONS:
                    errors.append(("bad-retained", retained))
            elif resp.status_code == 200:
                # 发布与探针并发，版本可能恰好已发布，此时必须版本钉住
                if resp.json()["used_version"] != version:
                    errors.append(("probe-mismatch", version))
            else:
                errors.append(("probe", resp.status_code, resp.text))

    threads = [threading.Thread(target=publisher)]
    threads += [threading.Thread(target=detector, args=(w,)) for w in range(5)]
    threads.append(threading.Thread(target=not_yet_published_probe))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert sorted(published) == _NEW_VERSIONS
    assert not_found_seen.is_set(), "应至少观察到一次未发布版本的 404"
    # 最终活动版本为最后发布的 cal-v6
    health = client.get("/health").json()
    assert health["active_version"] == "cal-v6"
    # 所有版本均可查询，阈值随偏置单调
    versions = client.get("/calibration/versions").json()
    assert [v["version"] for v in versions] == [f"cal-v{i}" for i in range(7)]
    thresholds = [
        client.get(f"/calibration/versions/cal-v{i}").json()["thresholds"]["short"][
            "threshold"
        ]
        for i in range(7)
    ]
    assert all(thresholds[i] < thresholds[i + 1] for i in range(6))


def test_concurrent_failed_preserves_active(client):
    r = client.post("/calibration/publish", json=_build_version("cal-v0", 0.0))
    assert r.status_code == 200

    errors = []
    stop = threading.Event()

    def bad_publisher():
        # 持续提交缺类批次，全部应被拒绝
        idx = 0
        while not stop.is_set():
            payload = _build_version(f"cal-bad-{idx}", 0.0)
            payload["windows"] = [
                w for w in payload["windows"] if w["label"] != "leak"
            ]
            resp = client.post("/calibration/publish", json=payload)
            if resp.status_code != 422:
                errors.append((resp.status_code, resp.text))
            idx += 1

    def good_publisher():
        for i in range(1, 4):
            resp = client.post(
                "/calibration/publish", json=_build_version(f"cal-g{i}", 0.02 * i)
            )
            if resp.status_code != 200:
                errors.append(("good", resp.status_code, resp.text))

    def reader():
        while not stop.is_set():
            resp = client.get("/health")
            active = resp.json()["active_version"]
            if active is None or "bad" in active:
                errors.append(("bad-active", active))

    t_bad = threading.Thread(target=bad_publisher)
    t_read = threading.Thread(target=reader)
    t_bad.start()
    t_read.start()
    good_publisher()
    stop.set()
    t_bad.join()
    t_read.join()

    assert errors == []
    assert client.get("/health").json()["active_version"] == "cal-g3"
