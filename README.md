# 电芯故障 conformal 检测服务

电池产线复测决策 API：基于历史标注窗口和模型分数，按故障类型生成
conformal 预测阈值；检测请求声明校准版本，返回每个窗口的预测集合，
并把相邻告警窗口合并为复测区间。

## 功能

- **Conformal 校准（NumPy 实现）**：每个故障类型单独定阈值。故障类分数
  越高越像故障，对该类型故障窗口的分数取有限样本校正低分位
  `rank = ceil((n+1)·α)/n`（等价于对非一致性分数 `-s` 取标准
  `(1-α)` 分位），保证故障窗口被预测集合覆盖的边际比例 ≥ `1-α`。
- **发布前校验，失败保留上一版**：时间戳严格升序、标签合法、每个窗口
  包含全部故障类型分数且均为有限数值（拒绝 NaN/Inf/缺键）、每个故障
  类型样本数达到 `min_samples_per_fault`。任一不满足返回 422/409，
  活动版本与磁盘快照均不变，响应中说明原因和 `retained_version`。
- **原子发布**：校验与阈值计算在锁外完成，临界区内只做版本指针切换和
  `os.replace` 原子落盘（临时文件 + fsync）。
- **版本钉住**：检测全程使用请求声明的 `version`（即使已有更新的活动
  版本）。版本不存在时默认 404 并说明保留版本；`allow_fallback=true`
  时回退到活动版本并在 `warning` 中显式声明。
- **预测集合与复测区间**：窗口分数 ≥ 对应阈值即把该故障类型纳入预测
  集合，空集合判为正常；`merge_gap` 控制相邻（或间隔 ≤ N 个非告警
  窗口的）告警块合并，输出复测区间（起止时间、窗口索引、涉及故障类型）。
- **持久化**：校准快照写入 Docker volume（`/data/calibration_state.json`），
  服务重启自动恢复。

## 目录结构

```
app/
  conformal.py   # 校验、conformal 分位数、预测集合、区间合并（NumPy）
  store.py       # 线程安全版本库 + 原子 JSON 快照
  schemas.py     # Pydantic 模型
  main.py        # FastAPI 路由与异常处理
tests/           # pytest：单元/API/并发共 28 个用例
examples/        # curl 与 Python 调用示例
Dockerfile
docker-compose.yml
```

## Docker Compose 启动

```bash
docker compose up --build
# 服务监听 http://localhost:8000，交互式文档 http://localhost:8000/docs
```

本地无 Docker 时：

```bash
python3 -m pip install -r requirements.txt
CALIBRATION_STATE_FILE=./data/state.json \
  uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API 概览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查与当前活动版本 |
| POST | `/calibration/publish` | 校验并原子发布校准批次 |
| POST | `/detection` | 按声明版本检测，返回预测集合与复测区间 |
| GET | `/calibration/versions` | 全部校准版本列表 |
| GET | `/calibration/versions/{version}` | 单版本阈值详情 |

### 发布校准批次

```bash
curl -s -X POST http://localhost:8000/calibration/publish \
  -H 'Content-Type: application/json' \
  -d '{
    "version": "cal-v1",
    "fault_types": ["short", "leak"],
    "alpha": 0.1,
    "min_samples_per_fault": 3,
    "windows": [
      {"timestamp": 1,  "label": "normal", "scores": {"short": 0.05, "leak": 0.04}},
      {"timestamp": 10, "label": "short",  "scores": {"short": 0.80, "leak": 0.10}},
      {"timestamp": 11, "label": "short",  "scores": {"short": 0.82, "leak": 0.11}},
      {"timestamp": 12, "label": "short",  "scores": {"short": 0.84, "leak": 0.12}},
      {"timestamp": 20, "label": "leak",   "scores": {"short": 0.10, "leak": 0.70}},
      {"timestamp": 21, "label": "leak",   "scores": {"short": 0.11, "leak": 0.73}},
      {"timestamp": 22, "label": "leak",   "scores": {"short": 0.12, "leak": 0.76}}
    ]
  }'
```

### 检测（声明版本，合并相邻告警）

```bash
curl -s -X POST http://localhost:8000/detection \
  -H 'Content-Type: application/json' \
  -d '{
    "version": "cal-v1",
    "merge_gap": 0,
    "windows": [
      {"timestamp": 100, "scores": {"short": 0.95, "leak": 0.05}},
      {"timestamp": 101, "scores": {"short": 0.90, "leak": 0.05}},
      {"timestamp": 102, "scores": {"short": 0.02, "leak": 0.02}},
      {"timestamp": 103, "scores": {"short": 0.03, "leak": 0.88}}
    ]
  }'
```

响应（节选）：前两个窗口预测集合含 `short` 并合并为一个复测区间，
第四个窗口含 `leak` 单独成区间。

更多示例见 `examples/curl_examples.sh` 与 `examples/api_demo.py`：

```bash
bash examples/curl_examples.sh
python3 examples/api_demo.py
```

## 失败语义

- **校准失败（422）**：缺故障类型、样本不足、NaN/Inf、缺分数键、时间
  乱序、未知标签等。响应体：
  `{"status":"rejected","reason":...,"detail":...,"retained_version":"cal-v1"}`
- **版本号冲突（409）**：拒绝重复发布，保留上一版。
- **声明版本不存在（404）**：不执行检测，返回原因与保留的活动版本；
  显式传 `allow_fallback=true` 才回退，且响应中 `fallback=true` 并带
  `warning`。

## 测试

```bash
python3 -m pytest tests/ -v
```

覆盖：分位数数值正确性与覆盖率 sanity check、缺类/异常分数/乱序/未知
标签拒绝、原子发布与快照恢复、版本钉住与回退、多线程并发换版 + 并发
检测、并发失败批次不影响活动版本。
