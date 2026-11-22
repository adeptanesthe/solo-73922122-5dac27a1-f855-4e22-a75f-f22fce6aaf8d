"""请求 / 响应数据模型。"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# 正常窗口使用的标签字面值
NORMAL_LABEL = "normal"


class CalibrationWindow(BaseModel):
    """一个历史检测窗口：时间戳、标签以及各故障类型的模型分数。"""

    timestamp: float = Field(..., description="窗口时间戳（Unix 秒，按升序提交）")
    label: str = Field(..., min_length=1, description="窗口真实标签：normal 或某故障类型")
    scores: Dict[str, float] = Field(..., description="故障类型 -> 模型分数")


class CalibrationRequest(BaseModel):
    """校准批次发布请求。"""

    version: str = Field(..., min_length=1, description="校准版本号，例如 cal-2026-11-22-001")
    fault_types: List[str] = Field(..., min_length=1, description="需要覆盖的全部故障类型")
    alpha: float = Field(0.1, gt=0.0, lt=1.0, description="目标误报率，阈值取 (1-alpha) 分位数")
    min_samples_per_fault: int = Field(
        1, ge=1, description="每个故障类型至少需要的故障窗口数，少于该数则校准失败"
    )
    windows: List[CalibrationWindow] = Field(..., min_length=1, description="按时间升序排列的历史窗口")


class ThresholdInfo(BaseModel):
    threshold: float = Field(..., description="conformal 阈值，分数 >= 阈值时该故障类型进入预测集合")
    fault_samples: int = Field(..., description="该校准类型使用的故障窗口数")
    quantile_rank: float = Field(
        ...,
        description=(
            "实际使用的分位点等级 ceil((n+1)*alpha)/n；分数越高越像故障，"
            "对故障类分数取该低分位，保证故障覆盖率 >= 1-alpha"
        ),
    )


class CalibrationResult(BaseModel):
    version: str
    fault_types: List[str]
    alpha: float
    thresholds: Dict[str, ThresholdInfo]
    total_windows: int
    normal_windows: int
    created_at: float


class CalibrationResponse(BaseModel):
    status: str = "published"
    calibration: CalibrationResult
    active_version: Optional[str] = None


class CalibrationError(BaseModel):
    status: str = "rejected"
    reason: str
    detail: Optional[str] = None
    retained_version: Optional[str] = None


class DetectionWindow(BaseModel):
    """待检测窗口：时间戳 + 各故障类型分数。"""

    timestamp: float
    scores: Dict[str, float]


class DetectionRequest(BaseModel):
    version: str = Field(..., min_length=1, description="请求声明使用的校准版本")
    windows: List[DetectionWindow] = Field(..., min_length=1)
    merge_gap: int = Field(
        0,
        ge=0,
        description="合并告警窗口时允许间隔的非告警窗口数；0 表示仅合并严格相邻窗口",
    )
    allow_fallback: bool = Field(
        False, description="声明版本不存在时，是否允许回退到当前活动（上一版）校准"
    )


class WindowPrediction(BaseModel):
    index: int
    timestamp: float
    prediction_set: List[str] = Field(..., description="预测集合；为空表示判为正常")
    is_alarm: bool
    scores: Dict[str, float]


class RetestInterval(BaseModel):
    start_timestamp: float
    end_timestamp: float
    window_count: int
    fault_types: List[str]
    indices: List[int]


class DetectionResponse(BaseModel):
    requested_version: str
    used_version: str
    fallback: bool = False
    warning: Optional[str] = None
    windows: List[WindowPrediction]
    retest_intervals: List[RetestInterval]


class VersionSummary(BaseModel):
    version: str
    created_at: float
    fault_types: List[str]
    alpha: float
    total_windows: int
    active: bool
