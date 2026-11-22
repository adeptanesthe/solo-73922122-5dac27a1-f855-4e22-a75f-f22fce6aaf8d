"""针对运行中服务的并发换版 + 并发检测压测。

用法：
    python3 examples/load_switch.py                 # 默认 http://localhost:8000
    BASE_URL=http://localhost:8011 RUN_ID=r2 python3 examples/load_switch.py
前置：服务已启动（docker compose up 或 uvicorn app.main:app）
"""

import concurrent.futures as cf
import os

import httpx

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
# RUN_ID 用于版本号命名空间，避免与历史运行的版本号冲突
RUN_ID = os.environ.get("RUN_ID", "run1")
N_VERSIONS = 8
N_DETECT_WORKERS = 12
N_REQUESTS_PER_WORKER = 100


def make_batch(version: str, bias: float) -> dict:
    windows = [
        {"timestamp": 1.0, "label": "normal", "scores": {"short": 0.05, "leak": 0.05}},
        {"timestamp": 2.0, "label": "normal", "scores": {"short": 0.06, "leak": 0.05}},
    ]
    for i in range(4):
        windows.append(
            {
                "timestamp": 10.0 + i,
                "label": "short",
                "scores": {"short": 0.70 + bias + 0.02 * i, "leak": 0.10},
            }
        )
    for i in range(4):
        windows.append(
            {
                "timestamp": 20.0 + i,
                "label": "leak",
                "scores": {"short": 0.10, "leak": 0.60 + bias + 0.02 * i},
            }
        )
    return {
        "version": version,
        "fault_types": ["short", "leak"],
        "alpha": 0.2,
        "min_samples_per_fault": 2,
        "windows": windows,
    }


def main() -> None:
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        base = make_batch(f"load-{RUN_ID}-v0", 0.0)
        r = client.post("/calibration/publish", json=base)
        r.raise_for_status()

        errors: list[str] = []

        def publish(i: int) -> str:
            resp = client.post(
                "/calibration/publish", json=make_batch(f"load-{RUN_ID}-v{i}", 0.01 * i)
            )
            if resp.status_code != 200:
                errors.append(f"publish load-v{i}: {resp.status_code} {resp.text}")
            return f"load-v{i}"

        def detect(worker: int) -> None:
            for i in range(N_REQUESTS_PER_WORKER):
                # 在已发布/将发布的版本号中随机钉一个；未发布的版本应得 404
                version = f"load-{RUN_ID}-v{i % (N_VERSIONS + 1)}"
                resp = client.post(
                    "/detection",
                    json={
                        "version": version,
                        "windows": [
                            {"timestamp": 100.0 + i, "scores": {"short": 0.9, "leak": 0.1}},
                            {"timestamp": 200.0 + i, "scores": {"short": 0.1, "leak": 0.8}},
                        ],
                    },
                )
                if resp.status_code == 404:
                    detail = resp.json()["detail"]
                    if detail["reason"] is None or "不存在" not in detail["reason"]:
                        errors.append(f"unexpected 404 body: {detail}")
                    continue
                if resp.status_code != 200:
                    errors.append(f"detect {version}: {resp.status_code} {resp.text}")
                    continue
                body = resp.json()
                if body["used_version"] != version:
                    errors.append(
                        f"version pin violated: asked {version}, got {body['used_version']}"
                    )
                if len(body["windows"]) != 2:
                    errors.append(f"bad window count: {body}")

        with cf.ThreadPoolExecutor(max_workers=N_DETECT_WORKERS + 2) as pool:
            detectors = [pool.submit(detect, w) for w in range(N_DETECT_WORKERS)]
            publishers = [pool.submit(publish, i) for i in range(1, N_VERSIONS + 1)]
            for f in cf.as_completed(detectors + publishers):
                f.result()

        versions = client.get("/calibration/versions").json()
        published = [v["version"] for v in versions]
        active = client.get("/health").json()["active_version"]

        print(f"发布版本数: {len(published)}，活动版本: {active}")
        print(f"检测请求总数: {N_DETECT_WORKERS * N_REQUESTS_PER_WORKER}")
        if errors:
            print("发现错误:")
            for e in errors[:10]:
                print(" -", e)
            raise SystemExit(1)
        print("全部通过：无版本钉住违规、无 5xx、发布原子有序。")


if __name__ == "__main__":
    main()
