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


class SnapshotWriteError(RuntimeError):
    """校准快照落盘失败；本次发布已放弃，内存版本库未改动。"""


class CalibrationStore:
    """内存版本库 + 原子 JSON 快照。

    发布流程保证原子性，并尽量缩短对读请求的阻塞：

    1. 调用方在锁外完成全部校验与阈值计算；
    2. 发布者之间用 ``_publish_lock`` 串行化；持锁快照版本表后**不持有
       读锁**写临时快照并 ``fsync``，此间检测/查询线程仍可读取已发布版本；
    3. 落盘只在临时文件上完成，``os.replace`` 原子替换正式快照后，再在
       读锁内一次性登记新版本、切换活动指针；
    4. 任一步失败都会清理临时文件，内存与磁盘都保持在最近一次成功发布
       的状态，新版本对查询与检测完全不可见。
    """

    def __init__(self, snapshot_path: Optional[str] = None) -> None:
        self._lock = threading.RLock()
        self._publish_lock = threading.Lock()
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

    @staticmethod
    def _snapshot_payload(
        versions: Dict[str, CalibrationResult], active: Optional[str]
    ) -> dict:
        return {
            "active_version": active,
            "versions": {
                version: result.model_dump()
                for version, result in versions.items()
            },
        }

    def _stage_snapshot(
        self, versions: Dict[str, CalibrationResult], active: Optional[str]
    ) -> Optional[str]:
        """把版本表写入临时快照文件（不替换正式快照），返回临时文件路径。

        未启用持久化时返回 ``None``。失败时清理临时文件并抛出
        ``SnapshotWriteError``。
        """
        path = self._snapshot_path
        if not path:
            return None
        payload = self._snapshot_payload(versions, active)
        directory = os.path.dirname(os.path.abspath(path))
        try:
            os.makedirs(directory, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                prefix=".cal-", suffix=".tmp", dir=directory
            )
        except OSError as exc:
            raise SnapshotWriteError(f"无法准备校准快照临时文件: {exc}") from exc
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
        except BaseException as exc:
            self._discard_staged(tmp_path)
            raise SnapshotWriteError(f"写入校准快照失败: {exc}") from exc
        return tmp_path

    def _discard_staged(self, tmp_path: Optional[str]) -> None:
        """删除本批次留下的临时快照文件（不存在或已替换则忽略）。"""
        if not tmp_path:
            return
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        except OSError:
            # 清理失败不应掩盖发布结果；残留文件形如 .cal-*.tmp，不会被读取
            pass

    def _commit_staged(self, tmp_path: Optional[str]) -> None:
        path = self._snapshot_path
        if not tmp_path or not path:
            return
        try:
            os.replace(tmp_path, path)
        except OSError as exc:
            self._discard_staged(tmp_path)
            raise SnapshotWriteError(f"替换校准快照失败: {exc}") from exc

    # ------------------------------------------------------------------ #
    # 发布 / 查询
    # ------------------------------------------------------------------ #
    def publish(self, result: CalibrationResult, *, activate: bool = True) -> None:
        """原子发布一个已构建完成的校准结果。

        调用方必须先完成全部校验与阈值计算。本方法先落盘、后登记：只有
        新快照原子替换成功之后，新版本才会出现在版本表并成为活动版本。
        版本号冲突或落盘失败时，现有版本、活动指针与磁盘快照都不受影响
        （临时快照文件会被清理）。
        """
        with self._publish_lock:  # 发布者串行；读请求不需要此锁
            with self._lock:
                if result.version in self._versions:
                    raise VersionAlreadyExistsError(
                        f"校准版本 {result.version!r} 已存在，拒绝重复发布"
                    )
                previous_active = self._active_version
                # 只复制版本表引用（浅拷贝），不复制每个结果对象
                candidate_versions = dict(self._versions)
            candidate_active = (
                result.version
                if activate or previous_active is None
                else previous_active
            )
            candidate_versions[result.version] = result

            # 落盘期间不持有读锁：并发检测仍使用上一版活动版本
            staged = self._stage_snapshot(candidate_versions, candidate_active)
            self._commit_staged(staged)

            # 快照已原子替换成功，才在锁内一次性生效；版本号冲突在进入
            # 落盘前已由本锁 + 发布锁排除，此处不会出现半发布状态
            with self._lock:
                self._versions[result.version] = result
                self._active_version = candidate_active

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
