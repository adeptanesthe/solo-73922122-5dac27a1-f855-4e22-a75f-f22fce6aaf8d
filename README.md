# 电芯故障 conformal 检测服务

电池产线复测决策 API：基于历史标注窗口和模型分数，按故障类型生成
conformal 预测阈值；检测请求声明校准版本，返回每个窗口的预测集合，
并把相邻告警窗口合并为复测区间。

## 功能

- **Conformal 校准（流式分位数）**：每个故障类型单独定阈值。故障类分数
  越高越像故障，对该类型故障窗口的分数取有限样本校正低分位
  `rank = ceil((n+1)·α)/n`（等价于对非一致性分数 `-s` 取标准
  `(1-α)` 分位），保证故障窗口被预测集合覆盖的边际比例 ≥ `1-α`。
- **校准内存不随窗口数增长**：发布仍按整批提交，但服务端不再为每个
  故障类型保留一份分数副本。校验阶段单遍流式统计各类型样本数、最小值/
  最大值；分位数阶段用固定 4096 桶的直方图多轮扫描已解析请求，逐位
  收窄到目标次序统计量，再按 NumPy `linear` 的运算顺序做线性插值。
  故障类型数固定时，算法自身的额外工作内存恒定（实测 < 1 MB，见下表），
  与窗口数无关；已解析的请求数据不计入该结论。
- **与 NumPy 整批计算一致**：重复分数按出现次数正常计数，各类样本数、
  `quantile_rank` 与阈值均与 `np.quantile(..., method="linear")` 相同
  （含重复值、±0.0、量级跨越 1e300 的分数；随机对照数万次无差异）。
- **发布前校验，失败保留上一版**：时间戳严格升序、标签合法、每个窗口
  包含全部故障类型分数且均为有限数值（拒绝 NaN/Inf/缺键）、每个故障
  类型样本数达到 `min_samples_per_fault`。任一不满足返回 422/409，
  活动版本与磁盘快照均不变，响应中说明原因和 `retained_version`。
- **原子发布、计算期间检测可用**：校验与阈值计算全部在锁外完成；
  发布者之间串行，写临时快照 + `fsync` 期间不持有读锁，检测与查询始终
  可用（见下方「余项」中的单 worker 说明）；正式快照 `os.replace` 原子
  替换成功后才一次性登记版本并切换活动指针。新版本在「校验 → 计算 →
  落盘」全部成功前对查询与检测不可见。
- **发布失败清理本批临时数据**：落盘失败返回 500，删除临时快照文件，
  内存版本表与磁盘快照都停在最近一次成功发布的版本。
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
  conformal.py   # 流式校验、conformal 分位数、预测集合、区间合并
  store.py       # 线程安全版本库 + 原子 JSON 快照
  schemas.py     # Pydantic 模型
  main.py        # FastAPI 路由与异常处理
tests/           # pytest：单元/API/并发/内存/失败语义共 468 个用例
examples/        # curl、Python 与大批次发布/检测示例
Dockerfile
docker-compose.yml
```

## 内存与一致性说明

| 项目 | 行为 |
| --- | --- |
| 批次内故障分数副本 | 不保留；校验时只累计样本数与值域边界 |
| 分位数计算额外内存 | 每个故障类型 2 个目标排名 + 固定 4096 桶直方图，与窗口数无关 |
| 重复分数 | 按出现次数计数，结果与整批排序后取分位一致 |
| 计算/落盘期间检测 | 读锁不参与，检测按已发布版本正常返回 |
| 新版本可见时机 | 校验、计算、快照原子替换全部成功后 |
| 发布失败 | 清理临时文件，活动版本与快照保持最近一次成功发布 |

峰值内存实测（tracemalloc，仅统计校验+计算阶段新增分配，不含已解析请求）：

| 故障类型数 | 窗口数 | 额外峰值 |
| --- | --- | --- |
| 2 | 400 / 6,400 / 51,200 | 134 KB / 134 KB / 134 KB |
| 4 | 800 / 12,800 / 102,400 | 264 KB / 265 KB / 265 KB |
| 8 | 1,600 / 25,600 / 204,800 | 527 KB / 527 KB / 527 KB |

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

### 大批次发布与检测

`examples/bulk_publish_detect.py` 会构造约 2 万个校准窗口（× 3 类分数）、
整批提交发布，用 NumPy 整批算法逐项核对返回阈值，再提交 5 千个检测窗口：

```bash
# Docker Compose 启动后（服务在 localhost:8000）
docker compose up -d --build
python3 examples/bulk_publish_detect.py

# 调整规模 / 指向其他地址
WINDOWS=50000 DETECT_WINDOWS=10000 BASE_URL=http://localhost:8000 \
  VERSION=bulk-v2 python3 examples/bulk_publish_detect.py
```

脚本输出包含：发布耗时、`total_windows`/`normal_windows`、每个故障类型
的阈值与 NumPy 参考值 `match=True`、检测告警窗口数/复测区间数，以及缺类
批次被拒绝后检测仍使用上一版的结果。也可在容器内运行：

```bash
docker compose exec conformal-api python examples/bulk_publish_detect.py
```

更多示例见 `examples/curl_examples.sh`、`examples/api_demo.py` 与
`examples/load_switch.py`：

```bash
bash examples/curl_examples.sh
python3 examples/api_demo.py
python3 examples/bulk_publish_detect.py
```

## 失败语义

- **校准失败（422）**：缺故障类型、样本不足、NaN/Inf、缺分数键、时间
  乱序、未知标签等。响应体：
  `{"status":"rejected","reason":...,"detail":...,"retained_version":"cal-v1"}`
- **版本号冲突（409）**：拒绝重复发布，保留上一版。
- **快照落盘失败（500）**：不发布新版本，清理本批临时快照文件，内存与
  磁盘继续使用最近一次成功发布的校准结果；响应中带 `retained_version`。
- **声明版本不存在（404）**：不执行检测，返回原因与保留的活动版本；
  显式传 `allow_fallback=true` 才回退，且响应中 `fallback=true` 并带
  `warning`。

## 测试

```bash
python3 -m pytest tests/ -v
```

覆盖：分位数与 NumPy 整批计算一致（含重复分数、±0.0、极端量级）、
额外内存不随窗口数增长、大批次阈值一致性、缺类/异常分数/乱序/未知标签
拒绝、原子发布与快照恢复、落盘失败回滚与临时文件清理、计算/落盘期间
检测可用、版本钉住与回退、多线程并发换版 + 并发检测。

## 余项

- **单 worker 限制**：版本状态保存在进程内，`Dockerfile` 固定
  `--workers 1`。多 worker 时各进程快照会互相覆盖，暂不支持；如要横向
  扩展需要把版本库外置为共享存储。
- **计算期间的请求排队**：阈值计算在请求线程内以纯 Python 流式扫描完成，
  不持有版本库读锁，但同一 worker 内仍会与其他请求分时使用 CPU。
  实测 6 万窗口发布（约 2.1 s）期间，并发检测 p50 约 17 ms、p95 约 31 ms，
  最大一次约 0.75 s（来自大请求体 JSON 解析阶段的 GIL 占用）；大请求体
  的解析耗时无法在应用层消除。
- **内存统计口径**：表中数值为 `tracemalloc` 统计的校验+计算阶段新增
  分配，不含 Pydantic 已解析的请求对象；快照落盘时会 `model_dump` 当前
  全部版本，版本数量很多时该部分内存与版本数成正比，与批次窗口数无关。
- **未在 Docker 内实测**：当前环境无 Docker，`docker compose config` 的
  YAML 已用 PyYAML 校验，服务以同等 `uvicorn app.main:app --workers 1`
  方式本地启动完成端到端验证；镜像构建与 compose 启动需在有 Docker 的
  环境复跑一次。
