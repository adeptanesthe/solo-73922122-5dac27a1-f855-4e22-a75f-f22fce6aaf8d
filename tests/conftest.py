"""pytest 公共夹具：每个测试使用独立的临时快照文件。"""

import os
import tempfile

import pytest

# 必须在导入 app 之前指定快照路径
_TMP_DIR = tempfile.mkdtemp(prefix="cal-test-")
os.environ["CALIBRATION_STATE_FILE"] = os.path.join(_TMP_DIR, "state.json")


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from app.main import app, store

    # 每个测试用例前清空内存版本库，保证相互隔离
    store._versions.clear()
    store._active_version = None
    if os.path.exists(store._snapshot_path):
        os.unlink(store._snapshot_path)

    with TestClient(app) as c:
        yield c


def make_window(ts: float, label: str, scores: dict):
    return {"timestamp": ts, "label": label, "scores": scores}


@pytest.fixture()
def calibration_payload():
    """构造 v1 校准批次：两类故障 short / leak，各 5 个故障样本。"""

    def payload(version="cal-v1", alpha=0.2):
        windows = []
        # 正常窗口分数整体偏低
        for i, ts in enumerate([1.0, 2.0, 3.0]):
            windows.append(
                make_window(
                    ts,
                    "normal",
                    {"short": 0.05 + i * 0.01, "leak": 0.04 + i * 0.01},
                )
            )
        # short 故障窗口：short 分高，leak 分低
        for i, ts in enumerate([10.0, 11.0, 12.0, 13.0, 14.0]):
            windows.append(
                make_window(
                    ts, "short", {"short": 0.80 + i * 0.02, "leak": 0.10 + i * 0.01}
                )
            )
        # leak 故障窗口：leak 分高，short 分低
        for i, ts in enumerate([20.0, 21.0, 22.0, 23.0, 24.0]):
            windows.append(
                make_window(
                    ts, "leak", {"short": 0.10 + i * 0.01, "leak": 0.70 + i * 0.03}
                )
            )
        return {
            "version": version,
            "fault_types": ["short", "leak"],
            "alpha": alpha,
            "min_samples_per_fault": 3,
            "windows": windows,
        }

    return payload
