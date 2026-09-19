#!/usr/bin/env python3
"""
Export a PyTorch maritime detection model to TensorRT FP16/INT8 engine.

This script runs ONCE offline on an x86 workstation with GPU, producing a
.engine file that can be deployed to the Jetson AGX Orin.

Usage:
    python export_model.py --weights model.pt --output maritime_detector.engine \
                           --precision fp16 --input_size 640 --num_classes 15

Requires:
    torch, onnx, onnxruntime, tensorrt (on x86 with CUDA)
"""

import argparse
import os
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Optional imports with helpful error messages
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None
    warnings.warn("PyTorch not available. Install with: pip install torch")

try:
    import onnx
except ImportError:
    onnx = None
    warnings.warn("ONNX not available. Install with: pip install onnx")

try:
    import tensorrt as trt
except ImportError:
    trt = None
    warnings.warn(
        "TensorRT not available. Install: pip install tensorrt"
    )

# ─── Constants ──────────────────────────────────────────────────────────────

TRT_LOGGER = trt.Logger(trt.Logger.WARNING) if trt else None

# ─── Helper: Construct a simple detection model (for demonstration) ─────────


class DummyDetectionModel(nn.Module if nn else object):
    """
    Placeholder detection model for export testing.
    Replace with your actual model (YOLOv8, etc.) in production.
    """

    def __init__(self, num_classes: int = 15):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes + 5)  # classification + bbox regression

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ─── ONNX Export ────────────────────────────────────────────────────────────


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_size: int = 640,
    batch_size: int = 1,
    device: str = "cpu",
):
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model (eval mode).
        output_path: Path for .onnx file.
        input_size: Model input size (square).
        batch_size: Batch dimension.
        device: 'cpu' or 'cuda'.
    """
    if torch is None:
        raise ImportError("PyTorch is required for ONNX export.")

    model.eval().to(device)
    dummy_input = torch.randn(batch_size, 3, input_size, input_size).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[ONNX] Model exported to {output_path}")


# ─── TensorRT Engine Build ─────────────────────────────────────────────────


def build_trt_engine(
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",
    max_batch_size: int = 1,
    workspace_gb: float = 4.0,
    calibration_data: Optional[np.ndarray] = None,
):
    """
    Build a TensorRT engine from an ONNX model.

    Args:
        onnx_path: Path to input ONNX model.
        output_path: Path for output .engine file.
        precision: 'fp32', 'fp16', or 'int8'.
        max_batch_size: Maximum batch size.
        workspace_gb: Max workspace memory in GB.
        calibration_data: Optional INT8 calibration data (NCHW float32).
    """
    if trt is None:
        raise ImportError("TensorRT is required for engine building.")

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"[TRT] Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for err in range(parser.num_errors):
                print(f"  Parse error {err}: {parser.get_error(err)}")
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_gb * 1 << 30)
    )

    # ─── Precision flags ────────────────────────────────────────────────
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[TRT] Enabled FP16 inference")
        else:
            print("[TRT] Warning: FP16 not supported on this platform")
    elif precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("[TRT] Enabled INT8 inference")
            if calibration_data is not None:
                calibrator = Int8Calibrator(calibration_data)
                config.int8_calibrator = calibrator
            else:
                print("[TRT] Warning: No INT8 calibration data provided")
        else:
            print("[TRT] Warning: INT8 not supported on this platform")

    # ─── Build engine ───────────────────────────────────────────────────
    print(f"[TRT] Building engine (precision={precision}, workspace={workspace_gb}GB)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")

    with open(output_path, "wb") as f:
        f.write(serialized)

    print(f"[TRT] Engine saved to {output_path} ({len(serialized)} bytes)")


# ─── INT8 Calibrator (simplified) ──────────────────────────────────────────


class Int8Calibrator(trt.IInt8EntropyCalibrator2 if trt else object):
    """
    Simplified INT8 calibrator using precomputed calibration data.
    """

    def __init__(self, calibration_data: np.ndarray):
        super().__init__()
        self._data = calibration_data.astype(np.float32).ravel()
        self._cursor = 0
        self._device_buffer = None

    def get_batch_size(self) -> int:
        return 1

    def get_batch(self, names):
        if self._cursor >= len(self._data):
            return None
        batch = self._data[self._cursor:self._cursor + 1]
        self._cursor += 1
        return [batch]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        pass


# ─── Main ───────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Export AVM detection model to TensorRT")
    parser.add_argument("--weights", "-w", help="PyTorch weights file (.pt)")
    parser.add_argument("--output", "-o", default="maritime_detector.engine",
                        help="Output TensorRT engine path")
    parser.add_argument("--onnx_output", default="model.onnx",
                        help="Intermediate ONNX file path")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"],
                        default="fp16", help="Inference precision")
    parser.add_argument("--input_size", type=int, default=640,
                        help="Model input size")
    parser.add_argument("--num_classes", type=int, default=15,
                        help="Number of detection classes")
    parser.add_argument("--workspace", type=float, default=4.0,
                        help="Max TensorRT workspace (GB)")
    parser.add_argument("--device", default="cuda:0",
                        help="Device for export")
    args = parser.parse_args()

    # ─── 1. Load or create model ────────────────────────────────────────
    if torch is None:
        raise ImportError("PyTorch is required. pip install torch")

    if args.weights and os.path.exists(args.weights):
        print(f"[Export] Loading weights from {args.weights}")
        ckpt = torch.load(args.weights, map_location=args.device)
        if isinstance(ckpt, dict):
            # Checkpoint dict: load into the placeholder architecture
            model = DummyDetectionModel(args.num_classes)
            model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        else:
            model = ckpt
    else:
        print("[Export] No weights file found. Using dummy model for testing.")
        model = DummyDetectionModel(args.num_classes)

    # ─── 2. Export to ONNX ──────────────────────────────────────────────
    export_to_onnx(model, args.onnx_output, args.input_size, device=args.device)

    # ─── 3. Build TensorRT engine ───────────────────────────────────────
    build_trt_engine(
        onnx_path=args.onnx_output,
        output_path=args.output,
        precision=args.precision,
        workspace_gb=args.workspace,
    )

    print(f"[Export] Done. Engine: {args.output}")


if __name__ == "__main__":
    main()
