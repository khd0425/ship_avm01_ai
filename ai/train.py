#!/usr/bin/env python3
"""Fine-tune a YOLOv8 detector on the AVM dataset and export ONNX for the C++ pipeline.

  python ai/train.py --data datasets/avm6/data.yaml --epochs 30 --name avm6_n

Output: runs/<name>/weights/best.pt  and  models/<name>.onnx  (opset 12, static 640x640,
output [1, 4+6, 8400] = what src/inference/dnn_detector.cpp decodes with arch "yolov8").
Run it on the Jetson side with TensorRT later (trtexec --onnx=... --fp16).

LICENCE NOTE: the ultralytics package (YOLOv8/YOLO11) is AGPL-3.0. That is fine for research on this
PC, but delivering a model/product built with it to the customer needs an Ultralytics enterprise
licence or a permissive alternative (e.g. YOLOX, Apache-2.0; RT-DETR). Decide before the M3 milestone.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", default="yolov8n.pt", help="COCO-pretrained start point")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="avm6_n")
    ap.add_argument("--project", default="runs")
    a = ap.parse_args()

    model = YOLO(a.weights)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
                project=a.project, name=a.name, exist_ok=True, workers=8, patience=15,
                degrees=0.0, fliplr=0.5, mosaic=1.0)          # no rotation aug: ships are upright
    best = Path(model.trainer.best)                      # ultralytics decides the run directory
    print(f"best weights: {best}")
    model = YOLO(str(best))
    metrics = model.val(data=a.data, imgsz=a.imgsz, device=a.device)
    print("\nper-class mAP50-95:")
    for i, n in model.names.items():
        print(f"  {n:14s} {metrics.box.maps[i]:.3f}")
    print(f"overall mAP50 {metrics.box.map50:.3f}  mAP50-95 {metrics.box.map:.3f}")

    onnx = model.export(format="onnx", imgsz=a.imgsz, opset=12, simplify=True, dynamic=False)
    Path("models").mkdir(exist_ok=True)
    dst = Path("models") / f"{a.name}.onnx"
    shutil.copy(onnx, dst)
    print(f"exported {dst}")


if __name__ == "__main__":
    main()
