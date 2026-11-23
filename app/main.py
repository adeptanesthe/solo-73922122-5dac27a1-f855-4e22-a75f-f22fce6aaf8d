"""FastAPI 入口：校准发布、检测、版本查询。"""

from __future__ import annotations

import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .conformal import (
    CalibrationValidationError,
    merge_alarm_intervals,
    predict_sets,
    validate_and_build_thresholds,
)
from .schemas import (
    CalibrationError,
    CalibrationRequest,
    CalibrationResponse,
    CalibrationResult,
    DetectionRequest,
    DetectionResponse,
    RetestInterval,
    ThresholdInfo,
    VersionSummary,
    WindowPrediction,
)
from .store import (
    CalibrationStore,
    SnapshotWriteError,
    VersionAlreadyExistsError,
    VersionNotFoundError,
)

app = FastAPI(
    title="电芯故障 conformal 检测服务",
    version="1.0.0",
    description=(
        "按故障类型生成 conformal 预测阈值，检测时返回每个窗口的预测集合，"
        "并将相邻告警窗口合并为复测区间。"
    ),
)

store = CalibrationStore()


# ---------------------------------------------------------------------- #
# 异常处理：校准失败时明确原因，并回传保留的上一版版本号
# ---------------------------------------------------------------------- #
@app.exception_handler(CalibrationValidationError)
async def calibration_validation_handler(
    request: Request, exc: CalibrationValidationError
) -> JSONResponse:
    retained = store.active_version
    body = CalibrationError(
        reason="校准批次校验失败，未发布新版本，继续保留上一版校准结果",
        detail=str(exc),
        retained_version=retained,
    )
    return JSONResponse(status_code=422, content=body.model_dump())


@app.exception_handler(VersionAlreadyExistsError)
async def version_exists_handler(
    request: Request, exc: VersionAlreadyExistsError
) -> JSONResponse:
    body = CalibrationError(
        reason="版本号冲突，未发布新版本，继续保留上一版校准结果",
        detail=str(exc),
        retained_version=store.active_version,
    )
    return JSONResponse(status_code=409, content=body.model_dump())


@app.exception_handler(SnapshotWriteError)
async def snapshot_write_handler(
    request: Request, exc: SnapshotWriteError
) -> JSONResponse:
    body = CalibrationError(
        reason="校准快照落盘失败，本批次临时数据已清理，未发布新版本，"
        "继续保留最近一次成功发布的校准结果",
        detail=str(exc),
        retained_version=store.active_version,
    )
    return JSONResponse(status_code=500, content=body.model_dump())


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "active_version": store.active_version}


@app.post("/calibration/publish", response_model=CalibrationResponse)
def publish_calibration(payload: CalibrationRequest) -> CalibrationResponse:
    """校验校准批次并原子发布。

    校验（样本覆盖各故障类型、分数有效、时间有序）全部通过后才切换版本；
    任一环节失败均返回错误且活动版本保持不变。
    """
    threshold_info, normal_count, _ = validate_and_build_thresholds(
        windows=payload.windows,
        fault_types=payload.fault_types,
        alpha=payload.alpha,
        min_samples_per_fault=payload.min_samples_per_fault,
    )

    result = CalibrationResult(
        version=payload.version,
        fault_types=list(payload.fault_types),
        alpha=payload.alpha,
        thresholds={
            ft: ThresholdInfo(
                threshold=threshold,
                fault_samples=sample_count,
                quantile_rank=rank,
            )
            for ft, (threshold, sample_count, rank) in threshold_info.items()
        },
        total_windows=len(payload.windows),
        normal_windows=normal_count,
        created_at=time.time(),
    )

    store.publish(result)  # 临界区内原子切换 + 原子落盘
    return CalibrationResponse(
        calibration=result, active_version=store.active_version
    )


@app.post("/detection", response_model=DetectionResponse)
def detect(payload: DetectionRequest) -> DetectionResponse:
    """按请求声明的校准版本执行检测。

    - 声明版本存在：全程使用该版本，即使它不是当前活动版本；
    - 声明版本不存在且 allow_fallback=false（默认）：返回 404，
      不产出检测结果，响应中说明原因并给出保留的活动版本；
    - 声明版本不存在且 allow_fallback=true：回退到活动版本并在
      warning 中显式说明。
    """
    warning: Optional[str] = None
    fallback = False
    try:
        calibration = store.get(payload.version)
    except VersionNotFoundError:
        active = store.get_active()
        if not payload.allow_fallback or active is None:
            raise HTTPException(
                status_code=404,
                detail=CalibrationError(
                    reason=(
                        f"声明的校准版本 {payload.version!r} 不存在，未执行检测，"
                        "继续保留上一版校准结果；如需自动回退请设置 allow_fallback=true"
                    ),
                    retained_version=active.version if active else None,
                ).model_dump(),
            )
        calibration = active
        fallback = True
        warning = (
            f"请求声明版本 {payload.version!r} 不存在，"
            f"已回退使用当前活动版本 {active.version!r}"
        )

    # 检测窗口同样要求按时间升序，合并逻辑以提交顺序为准
    prev_ts: Optional[float] = None
    for i, w in enumerate(payload.windows):
        ts = w.timestamp
        if prev_ts is not None and ts <= prev_ts:
            raise HTTPException(
                status_code=400,
                detail=f"第 {i} 个检测窗口时间戳未严格升序，窗口必须按时间排列",
            )
        prev_ts = ts

    try:
        prediction_sets = predict_sets(
            windows=payload.windows,
            fault_types=calibration.fault_types,
            thresholds={ft: t.threshold for ft, t in calibration.thresholds.items()},
        )
    except CalibrationValidationError as exc:
        raise HTTPException(status_code=400, detail=f"检测分数无效：{exc}") from exc

    window_predictions: List[WindowPrediction] = []
    alarms: List[bool] = []
    for i, (w, pred) in enumerate(zip(payload.windows, prediction_sets)):
        is_alarm = bool(pred)
        alarms.append(is_alarm)
        window_predictions.append(
            WindowPrediction(
                index=i,
                timestamp=w.timestamp,
                prediction_set=pred,
                is_alarm=is_alarm,
                scores=w.scores,
            )
        )

    intervals: List[RetestInterval] = []
    for start, end in merge_alarm_intervals(alarms, payload.merge_gap):
        fault_union = sorted(
            {
                ft
                for j in range(start, end + 1)
                for ft in prediction_sets[j]
            }
        )
        intervals.append(
            RetestInterval(
                start_timestamp=payload.windows[start].timestamp,
                end_timestamp=payload.windows[end].timestamp,
                window_count=end - start + 1,
                fault_types=fault_union,
                indices=list(range(start, end + 1)),
            )
        )

    return DetectionResponse(
        requested_version=payload.version,
        used_version=calibration.version,
        fallback=fallback,
        warning=warning,
        windows=window_predictions,
        retest_intervals=intervals,
    )


@app.get("/calibration/versions", response_model=List[VersionSummary])
def list_versions() -> List[VersionSummary]:
    active = store.active_version
    return [
        VersionSummary(
            version=r.version,
            created_at=r.created_at,
            fault_types=r.fault_types,
            alpha=r.alpha,
            total_windows=r.total_windows,
            active=(r.version == active),
        )
        for r in store.list_versions()
    ]


@app.get("/calibration/versions/{version}", response_model=CalibrationResult)
def get_version(version: str) -> CalibrationResult:
    try:
        return store.get(version)
    except VersionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
