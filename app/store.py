"""校准版本存储：线程安全、原子发布、落盘快照。"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from typing import Dict, List, Optional

from .schemas import CalibrationResult

_SNAPSHOT_ENV = "CALIBRATION_STATE_FILE"
_DEFAULT_SNAPSHOT = "data/calibration_state.json"


class VersionNotFoundError(KeyError):
    """请求声明的校准版本不存在。"""


class VersionAlreadyExistsError(ValueError):
    """校准版本号重复。"""


class CalibrationStore:
    """内存版本库 + 原子 JSON 快照。

    发布流程保证原子性：先在锁外完成全部校验与阈值计算，拿到完整的
    新版本对象后才进入临界区；临界区内只做指针切换和一次
    ``os.replace`` 原子落盘。任何一步失败，活动版本与磁盘快照都保持
    在上一版状态。
    """

    def __init__(self, snapshot_path: Optional[str] = None) -> None:
        self._lock = threading.RLock()
        self._versions: Dict[str, CalibrationResult] = {}
        self._active_version: Optional[str] = None
        self._snapshot_path = snapshot_path or os.environ.get(
            _SNAPSHOT_ENV, _DEFAULT_SNAPSHOT
        )
        self._load_snapshot()

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #
    def _load_snapshot(self) -> None:
        path = self._snapshot_path
        if not path or not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            versions = {
                v: CalibrationResult.model_validate(data)
                for v, data in payload.get("versions", {}).items()
            }
            active = payload.get("active_version")
            if active is not None and active not in versions:
                raise ValueError(f"快照活动版本 {active!r} 不存在于版本表中")
            self._versions = versions
            self._active_version = active
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            # 快照损坏不阻断启动，但丢弃该快照，后续发布会覆盖它
            print(f"[store] 警告：无法加载校准快照 {path}: {exc}")

    def _persist_locked(self) -> None:
        path = self._snapshot_path
        if not path:
            return
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        payload = {
            "active_version": self._active_version,
            "versions": {
                version: result.model_dump() for version, result in self._versions.items()
            },
        }
        # 同目录临时文件 + os.replace，保证落盘原子性
        fd, tmp_path = tempfile.mkstemp(prefix=".cal-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    # ------------------------------------------------------------------ #
    # 发布 / 查询
    # ------------------------------------------------------------------ #
    def publish(self, result: CalibrationResult, *, activate: bool = True) -> None:
        """原子发布一个已构建完成的校准结果。

        调用方必须先完成全部校验；本方法只负责临界区内的版本切换。
        版本号冲突时拒绝发布，现有版本与活动指针不受影响。
        """
        with self._lock:
            if result.version in self._versions:
                raise VersionAlreadyExistsError(
                    f"校准版本 {result.version!r} 已存在，拒绝重复发布"
                )
            self._versions[result.version] = result
            if activate or self._active_version is None:
                self._active_version = result.version
            self._persist_locked()

    def get(self, version: str) -> CalibrationResult:
        with self._lock:
            try:
                return self._versions[version]
            except KeyError as exc:
                raise VersionNotFoundError(
                    f"校准版本 {version!r} 不存在"
                ) from exc

    def get_active(self) -> Optional[CalibrationResult]:
        with self._lock:
            if self._active_version is None:
                return None
            return self._versions[self._active_version]

    @property
    def active_version(self) -> Optional[str]:
        with self._lock:
            return self._active_version

    def list_versions(self) -> List[CalibrationResult]:
        with self._lock:
            return [
                self._versions[v]
                for v in sorted(self._versions, key=lambda x: self._versions[x].created_at)
            ]
