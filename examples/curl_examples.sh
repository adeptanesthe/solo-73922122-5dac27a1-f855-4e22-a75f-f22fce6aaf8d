#!/usr/bin/env bash
# API 调用示例（curl）。默认对接 http://localhost:8000
# 可用 VERSION 环境变量覆盖版本号，便于重复运行：VERSION=cal-x bash examples/curl_examples.sh
set -euo pipefail
BASE="${BASE_URL:-http://localhost:8000}"
VERSION="${VERSION:-cal-v1}"

echo "== 1. 健康检查 =="
curl -s "$BASE/health" | python3 -m json.tool

echo "== 2. 发布校准批次 $VERSION =="
curl -s -X POST "$BASE/calibration/publish" \
  -H 'Content-Type: application/json' \
  -d "{
    \"version\": \"$VERSION\",
    \"fault_types\": [\"short\", \"leak\"],
    \"alpha\": 0.1,
    \"min_samples_per_fault\": 3,
    \"windows\": [
      {\"timestamp\": 1,  \"label\": \"normal\", \"scores\": {\"short\": 0.05, \"leak\": 0.04}},
      {\"timestamp\": 2,  \"label\": \"normal\", \"scores\": {\"short\": 0.06, \"leak\": 0.05}},
      {\"timestamp\": 10, \"label\": \"short\",  \"scores\": {\"short\": 0.80, \"leak\": 0.10}},
      {\"timestamp\": 11, \"label\": \"short\",  \"scores\": {\"short\": 0.82, \"leak\": 0.11}},
      {\"timestamp\": 12, \"label\": \"short\",  \"scores\": {\"short\": 0.84, \"leak\": 0.12}},
      {\"timestamp\": 20, \"label\": \"leak\",   \"scores\": {\"short\": 0.10, \"leak\": 0.70}},
      {\"timestamp\": 21, \"label\": \"leak\",   \"scores\": {\"short\": 0.11, \"leak\": 0.73}},
      {\"timestamp\": 22, \"label\": \"leak\",   \"scores\": {\"short\": 0.12, \"leak\": 0.76}}
    ]
  }" | python3 -m json.tool

echo "== 3. 检测（声明版本 $VERSION，相邻告警合并）=="
curl -s -X POST "$BASE/detection" \
  -H 'Content-Type: application/json' \
  -d "{
    \"version\": \"$VERSION\",
    \"merge_gap\": 0,
    \"windows\": [
      {\"timestamp\": 100, \"scores\": {\"short\": 0.95, \"leak\": 0.05}},
      {\"timestamp\": 101, \"scores\": {\"short\": 0.90, \"leak\": 0.05}},
      {\"timestamp\": 102, \"scores\": {\"short\": 0.02, \"leak\": 0.02}},
      {\"timestamp\": 103, \"scores\": {\"short\": 0.03, \"leak\": 0.88}}
    ]
  }" | python3 -m json.tool

echo "== 4. 缺类批次应被拒绝（422），并保留 $VERSION =="
curl -s -o /tmp/resp.json -w "%{http_code}\n" -X POST "$BASE/calibration/publish" \
  -H 'Content-Type: application/json' \
  -d "{
    \"version\": \"${VERSION}-bad\",
    \"fault_types\": [\"short\", \"leak\"],
    \"alpha\": 0.1,
    \"min_samples_per_fault\": 3,
    \"windows\": [
      {\"timestamp\": 1, \"label\": \"normal\", \"scores\": {\"short\": 0.05, \"leak\": 0.04}},
      {\"timestamp\": 10, \"label\": \"short\", \"scores\": {\"short\": 0.80, \"leak\": 0.10}},
      {\"timestamp\": 11, \"label\": \"short\", \"scores\": {\"short\": 0.82, \"leak\": 0.11}},
      {\"timestamp\": 12, \"label\": \"short\", \"scores\": {\"short\": 0.84, \"leak\": 0.12}}
    ]
  }"
python3 -m json.tool /tmp/resp.json

echo "== 5. 声明不存在的版本（404，说明保留版本）=="
curl -s -X POST "$BASE/detection" \
  -H 'Content-Type: application/json' \
  -d '{"version": "cal-ghost", "windows": [
      {"timestamp": 1, "scores": {"short": 0.1, "leak": 0.1}}]}' \
  | python3 -m json.tool

echo "== 6. 版本列表 =="
curl -s "$BASE/calibration/versions" | python3 -m json.tool
