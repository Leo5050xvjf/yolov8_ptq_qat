# PTQ Run Notes

This note documents the PTQ path that was verified in this repository with a pip-installed Ultralytics environment.

## Quick Reproduce

On another PC, the shortest path is:

```bash
bash scripts/reproduce_ptq.sh
```

The script assumes `python` points to the environment you want to use. It will:

- install the validated Python dependencies
- download `yolov8n.pt` if it is missing
- download `coco128` if it is missing
- set a writable `YOLO_CONFIG_DIR`
- run the verified PTQ command

Useful overrides:

```bash
PYTHON_BIN=python3 DEVICE=0 NUM_CALIB_BATCH=4 BATCH_SIZE=8 CALIB_BATCH_SIZE=8 bash scripts/reproduce_ptq.sh
```

## What Changed

- `yolov8_ptq_int8.py`
  - uses the selected device instead of forcing `.cuda()`
  - defaults to the local `coco128.yaml`
  - defaults to `--device 0`
  - uses `--calib-batch-size` for calibration dataloading
  - fixes the calibration loop off-by-one
  - accepts a single weight path even though `argparse` returns a list
  - handles ONNX export errors cleanly and removes an outdated `torch.onnx.export()` argument
- `utils/dataloaders.py`
  - replaces deprecated `np.int` usage for NumPy 2 compatibility
- `coco128.yaml`
  - adds a local dataset definition compatible with this repo's vendored YOLOv5 utilities

## Environment Notes

The validated environment used:

- Python from the `ultralytics` conda env
- GPU device `0`
- `ultralytics==8.4.23`
- `pytorch-quantization==2.1.3`
- `onnx`
- `absl-py`
- `tqdm`
- `pandas`
- `seaborn`
- `albumentations`

Ultralytics settings were directed to a writable temp directory with:

```bash
YOLO_CONFIG_DIR=/tmp/Ultralytics
```

## Verified PTQ Command

```bash
YOLO_CONFIG_DIR=/tmp/Ultralytics python yolov8_ptq_int8.py \
  --weights yolov8n.pt \
  --data coco128.yaml \
  --device 0 \
  --batch-size 8 \
  --calib-batch-size 8 \
  --num-calib-batch 4 \
  --workers 0
```

## Observed Results

From the validated run on local `coco128`:

- quantized validation: `mAP50-95 = 0.436`
- FP validation inside the same script with quantization disabled: `mAP50-95 = 0.447`

Generated artifacts from that run were:

- `weights/yolov8n-max-32.pth`
- `yolov8n_ptq_detect.onnx`

These are ignored in git because they are generated outputs.

## NMS Timeout Note

No `NMS time limit ... exceeded` warning appeared during the validated PTQ run, so no Ultralytics patch was required.

In the installed Ultralytics version used here, the timeout logic lives in:

```text
/home/leo/miniconda3/envs/ultralytics/lib/python3.9/site-packages/ultralytics/utils/nms.py
```

not in `ops.py`.
