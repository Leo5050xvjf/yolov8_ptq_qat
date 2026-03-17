#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-8}"
NUM_CALIB_BATCH="${NUM_CALIB_BATCH:-4}"
WORKERS="${WORKERS:-0}"
YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$ROOT_DIR/.yolo-config}"

export YOLO_CONFIG_DIR

echo "[ptq] repo root: $ROOT_DIR"
echo "[ptq] python: $PYTHON_BIN"
echo "[ptq] YOLO_CONFIG_DIR: $YOLO_CONFIG_DIR"

mkdir -p "$YOLO_CONFIG_DIR"
cd "$ROOT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ptq] error: python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

echo "[ptq] checking torch availability"
"$PYTHON_BIN" - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    raise SystemExit(f"[ptq] error: torch is required before running this script: {exc}")

print(f"[ptq] torch={torch.__version__}")
print(f"[ptq] cuda_available={torch.cuda.is_available()}")
print(f"[ptq] cuda_device_count={torch.cuda.device_count()}")
PY

echo "[ptq] installing runtime dependencies"
"$PYTHON_BIN" -m pip install ultralytics==8.4.23 onnx absl-py tqdm pandas seaborn albumentations
"$PYTHON_BIN" -m pip install --extra-index-url https://pypi.nvidia.com pytorch-quantization==2.1.3

if [ ! -f "yolov8n.pt" ]; then
  echo "[ptq] downloading yolov8n.pt via ultralytics"
  "$PYTHON_BIN" - <<'PY'
from ultralytics import YOLO
YOLO("yolov8n.pt")
print("[ptq] yolov8n.pt ready")
PY
else
  echo "[ptq] found existing yolov8n.pt"
fi

if [ ! -d "coco128/images/train2017" ]; then
  echo "[ptq] downloading coco128"
  "$PYTHON_BIN" - <<'PY'
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

root = Path(".")
zip_path = root / "coco128.zip"
url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
urlretrieve(url, zip_path)
with ZipFile(zip_path) as zf:
    zf.extractall(root)
zip_path.unlink()
print("[ptq] coco128 ready")
PY
else
  echo "[ptq] found existing coco128 dataset"
fi

echo "[ptq] running PTQ"
"$PYTHON_BIN" yolov8_ptq_int8.py \
  --weights yolov8n.pt \
  --data coco128.yaml \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --calib-batch-size "$CALIB_BATCH_SIZE" \
  --num-calib-batch "$NUM_CALIB_BATCH" \
  --workers "$WORKERS"

echo "[ptq] done"
echo "[ptq] expected artifacts:"
echo "[ptq]   weights/yolov8n-max-$((NUM_CALIB_BATCH * CALIB_BATCH_SIZE)).pth"
echo "[ptq]   yolov8n_ptq_detect.onnx"
