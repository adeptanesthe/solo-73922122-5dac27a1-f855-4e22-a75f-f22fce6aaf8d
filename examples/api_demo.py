"""端到端调用示例（需先启动服务：docker compose up 或 uvicorn app.main:app）。

运行：python examples/api_demo.py
可用环境变量：BASE_URL（默认 http://localhost:8000）、VERSION（默认 cal-v1）
依赖：httpx（已在 requirements.txt 中）
"""

import os

import httpx

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000")
VERSION = os.environ.get("VERSION", "cal-v1")


def main() -> None:
    with httpx.Client(base_url=BASE_URL, timeout=10.0) as client:
        # 1. 健康检查
        print("== /health ==")
        print(client.get("/health").json())

        # 2. 发布校准批次 cal-v1
        windows = []
        for i in range(3):
            windows.append(
                {
                    "timestamp": float(i + 1),
                    "label": "normal",
                    "scores": {"short": 0.05 + 0.01 * i, "leak": 0.04 + 0.01 * i},
                }
            )
        for i in range(5):
            windows.append(
                {
                    "timestamp": 10.0 + i,
                    "label": "short",
                    "scores": {"short": 0.80 + 0.02 * i, "leak": 0.10 + 0.01 * i},
                }
            )
        for i in range(5):
            windows.append(
                {
                    "timestamp": 20.0 + i,
                    "label": "leak",
                    "scores": {"short": 0.10 + 0.01 * i, "leak": 0.70 + 0.03 * i},
                }
            )
        print(f"\n== 发布 {VERSION} ==")
        r = client.post(
            "/calibration/publish",
            json={
                "version": VERSION,
                "fault_types": ["short", "leak"],
                "alpha": 0.1,
                "min_samples_per_fault": 3,
                "windows": windows,
            },
        )
        r.raise_for_status()
        print(r.json())

        # 3. 检测：声明使用 cal-v1
        print(f"\n== 检测（钉住 {VERSION}，merge_gap=1）==")
        r = client.post(
            "/detection",
            json={
                "version": VERSION,
                "merge_gap": 1,
                "windows": [
                    {"timestamp": 100.0, "scores": {"short": 0.95, "leak": 0.05}},
                    {"timestamp": 101.0, "scores": {"short": 0.90, "leak": 0.05}},
                    {"timestamp": 102.0, "scores": {"short": 0.02, "leak": 0.02}},
                    {"timestamp": 103.0, "scores": {"short": 0.03, "leak": 0.88}},
                    {"timestamp": 104.0, "scores": {"short": 0.02, "leak": 0.02}},
                ],
            },
        )
        r.raise_for_status()
        print(r.json())

        # 4. 故意提交缺类批次，观察拒绝且保留上一版
        print(f"\n== 缺类批次（应 422，保留 {VERSION}）==")
        bad = {
            "version": f"{VERSION}-bad",
            "fault_types": ["short", "leak"],
            "alpha": 0.1,
            "min_samples_per_fault": 3,
            "windows": [w for w in windows if w["label"] != "leak"],
        }
        print(client.post("/calibration/publish", json=bad).status_code)
        print(client.post("/calibration/publish", json=bad).json())

        # 5. 声明不存在的版本（应 404 并说明保留版本）
        print("\n== 不存在版本（应 404）==")
        r = client.post(
            "/detection",
            json={
                "version": "cal-ghost",
                "windows": [
                    {"timestamp": 1.0, "scores": {"short": 0.1, "leak": 0.1}}
                ],
            },
        )
        print(r.status_code, r.json())

        print("\n== 版本列表 ==")
        print(client.get("/calibration/versions").json())


if __name__ == "__main__":
    main()
