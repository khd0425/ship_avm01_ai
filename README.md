# ship_avm01_ai — Shipboard AVM System

**Around View Monitoring (AVM) and AR visualisation for vessels** — one screen for
berthing/unberthing: fisheye-corrected view, near-range top-view, two IR thermal
picture-in-picture windows, stabilised viewpoint presets, AI object detection and
navigation overlay.

Target platform: **NVIDIA Jetson AGX Orin / Orin NX** (JetPack 5/6, CUDA + OpenCV + TensorRT).  
Compute budget: **≤ 33 ms per frame, ≥ 30 fps** (spec PR-1/PR-2).  
Requirements: *AVM-SPEC-2026-001 Rev.0.1*. **Status per requirement, open issues,
design and verification notes: [docs/spec_compliance.md](docs/spec_compliance.md).**

## Current project status

Version 0.3.0 (early prototype). The whole application (CUDA kernels + OpenCV) **builds and runs on
the development PC** (CUDA 12.9, TITAN RTX sm_75, OpenCV 4.13) with real USB cameras and the
host unit tests pass. It has **not been built or run on a Jetson yet**; the Jetson-specific
paths (zero-copy mapped memory, TensorRT, NVENC, GMSL2 cameras) are untested.

| Area (spec) | State |
|-------------|-------|
| Capture: 1 EO + 2 IR threads, zero-copy mapped memory, simulator + OpenCV/GStreamer sources (FR-1.1) | Implemented, needs Jetson build; real GMSL2 cameras not tested |
| Time sync 16 ms, channel drop-out handling (FR-1.2–1.4) | Implemented + tested |
| Calibration tool, LUT tool, verification tool (FR-2.1/2.2/2.4) | `build_lut`, `verify_calibration` tested; `calib_fisheye` needs OpenCV build |
| Fisheye correction / LUT remap (FR-2.3) | Implemented (CUDA), used for the detector input |
| IR PiP + histogram AGC + show/hide/move (FR-3.1–3.3) | AGC & layout tested; GPU compositing needs Jetson build |
| 3 view presets, top-view r = 15 m, smooth transitions (FR-4) | Geometry tested + visually verified; CUDA kernel needs Jetson build |
| Digital stabilisation (FR-5.1) | Geometry tested; no real IMU input yet (simulator `--sim-motion`) |
| AI detection (FR-7) | Works on the PC via OpenCV DNN (YOLOX pretrained; YOLOv8 trained on our 6 classes): runs on a normal-FOV camera image. Trained data so far covers **person + small_vessel only**; bollard / fender / buoy / quay_edge need data. **TensorRT (Jetson) not done** — see [docs/ai_dataset_plan.md](docs/ai_dataset_plan.md) |
| Operator UI (FR-8) | Temporary OpenCV window (status text + keys); awaiting the customer UI design |
| Recording, logging (FR-9) | Implemented (NVENC via GStreamer untested) |
| AR overlay / NMEA (FR-6, optional) | Simple overlay only, no NMEA parser |
| EO/IR alpha blending (FR-3.4, optional) | Legacy code kept, not in the default pipeline |

---

## Hardware Requirements

| Component | Specification (spec ch.3) |
|-----------|---------------------------|
| **Edge device** | NVIDIA Jetson AGX Orin or Orin NX (to be fixed with the customer) |
| **EO camera** | 1 × 12 MP (4056×3040 assumed) fisheye, FOV ≥ 180°, bow centre, facing forward |
| **IR cameras** | 2 × normal-FOV thermal, left/right of the EO camera |
| **Sensors** | IMU (roll/pitch), NMEA0183 serial (dummy data for now) |
| **Output** | HDMI/DisplayPort 1920×1080 |

---

## Build Instructions

### Prerequisites (Jetson, JetPack 5.x/6.x)

| Dependency | Version | Notes |
|------------|---------|-------|
| **CMake** | ≥ 3.18 | JetPack 5 ships 3.16 → `pip3 install cmake` (or the Kitware apt repo) |
| **C++ compiler** | GCC ≥ 9 | Code is C++17 (JetPack 5's GCC 9.4 and CUDA 11.4 are supported) |
| **CUDA Toolkit** | ≥ 11.4 | Bundled with JetPack; device code is C++17 |
| **TensorRT** | ≥ 8.5 | Bundled with JetPack; set `TENSORRT_ROOT` if not in `/usr` |
| **OpenCV** | ≥ 4.5 | JetPack's OpenCV (with GStreamer) is enough. Used for capture, recording, display and the calibration tool — **not** for per-pixel processing (CUDA kernels do that) |
| **Qt5** | 5.x | Only for the optional live viewers (`AVM_BUILD_VIEWERS`) |
| **ffmpeg** | any | Optional (`test/run_tests.*`) |

```bash
sudo apt update
sudo apt install -y ninja-build build-essential libopencv-dev qtbase5-dev   # OpenCV/Qt5 are usually already present on JetPack

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure        # host unit tests (no camera/GPU needed)
```

The default CUDA architecture is `87` (Orin). For a desktop GPU pass e.g.
`-DCMAKE_CUDA_ARCHITECTURES=86`.

### Development PC without CUDA/OpenCV (host library + tests + LUT/verify tools)

```bash
cmake -S . -B build -DAVM_WITH_CUDA=OFF -DAVM_WITH_OPENCV=OFF -DAVM_BUILD_TOOLS=OFF -DAVM_BUILD_VIEWERS=OFF
cmake --build build -j && ctest --test-dir build --output-on-failure
```

### Development PC with an NVIDIA GPU (no Jetson yet)

Everything runs on a desktop GPU: the pipeline detects a discrete GPU and stages frames into
device memory (zero-copy mapped memory is only used on Jetson).

```bash
# OpenCV C++ dev files without sudo (conda-forge), any environment prefix
conda create -y -p ~/envs/avm-cv -c conda-forge --override-channels libopencv opencv cmake ninja
export PATH=~/envs/avm-cv/bin:$PATH
cmake -S . -B build-pc -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DOpenCV_DIR=~/envs/avm-cv/lib/cmake/opencv4 -DCMAKE_PREFIX_PATH=~/envs/avm-cv \
      -DAVM_BUILD_VIEWERS=OFF -DAVM_ENABLE_INFERENCE=OFF \
      -DCMAKE_EXE_LINKER_FLAGS="-L$HOME/envs/avm-cv/lib -Wl,-rpath-link,$HOME/envs/avm-cv/lib"
cmake --build build-pc -j
./build-pc/avm_system config/avm_pc_demo.json          # real USB cameras (see the file for the mapping)
./build-pc/avm_system config/avm_pc_demo.json --no-display --tour demo/run   # saves one PNG per view
```

`config/avm_pc_demo.json` maps the icSpring fisheye USB camera to the EO channel, a normal
colour USB camera to window 0 (the detector runs on it) and leaves window 1 empty to show the
"channel inactive" marker. The `-L`/`-rpath-link` flags are only needed because the conda OpenCV
needs a newer libstdc++ than the system GCC provides.

### CMake options

| Option | Default | Description |
|--------|---------|-------------|
| `AVM_WITH_CUDA` | `ON` | Build `avm_system` (CUDA). `OFF` builds only the host library, tests and tools |
| `AVM_WITH_OPENCV` | `ON` | Camera capture (V4L2/GStreamer), recording, display in `avm_system`. `OFF` = simulated cameras only |
| `AVM_ENABLE_INFERENCE` | `ON` | Link TensorRT (auto-disabled with a warning if not found) |
| `AVM_ENABLE_AR_OVERLAY` | `ON` | AR HUD overlay |
| `AVM_BUILD_TOOLS` | `ON` | OpenCV tools: `calib_fisheye` and test/utility programs |
| `AVM_BUILD_VIEWERS` | `ON` | Qt5 live viewers |
| `AVM_BUILD_TESTS` | `ON` | Unit tests (CTest) |

---

## Views without black areas

With `view.fit_to_visible` (default on) every view is fitted to what the camera really sees. A
single forward fisheye cannot see behind itself, so the top-view keeps its radius but its centre
moves ahead of the camera (11 m for a 20° tilt at 10 m height, 4.5 m at 45°) instead of showing
an empty half; perspective views narrow their field of view when tilted close to the lens limit;
the raw view is cropped to the image circle (`view.raw_fill: false` = whole sensor image).

## AI detection

Six phase-1 classes ([ai/classes.yaml](ai/classes.yaml)): `person, bollard, fender, quay_edge,
small_vessel, buoy`. The detector first runs on a **normal-FOV camera image**
(`inference.source: "ir0"`), later on the rectified fisheye view or virtual pinhole tiles.
`inference.backend`: `opencv_dnn` (ONNX, PC) · `tensorrt` (Jetson, pending) · `stub`.
`inference.arch`: `yolox` (COCO-pretrained, person/boat only) · `yolov8` (trained on our classes).

```bash
# data: COCO subset re-mapped to our classes (+ any YOLO-format export, e.g. Roboflow)
python ai/prepare_dataset.py --out datasets/avm6 --coco-zip annotations_trainval2017.zip
python ai/prepare_dataset.py --out datasets/avm6 --append --yolo <export> --yolo-map "Bollard:bollard"
# train + evaluate + export ONNX (models/<name>.onnx)
python ai/train.py --data datasets/avm6/data.yaml --epochs 30 --name avm6_n
# check on still images
./build-pc/detect_image --model models/avm6_n.onnx --arch yolov8 --out out img1.jpg img2.jpg
```

Data sources, licences, gaps and the fisheye strategy: [docs/ai_dataset_plan.md](docs/ai_dataset_plan.md).

## Running

```bash
./build/avm_system [config/avm_config.json] [--no-display] [--duration <sec>] [--sim-motion]
```

Without a config file the built-in defaults are used. With `source: ""` in the config
(the default) the cameras are **simulated**, so the whole pipeline, UI and recording can
be exercised before the hardware arrives. Set `source` to `/dev/videoN` or a GStreamer
pipeline ending in `appsink` for real cameras (examples in `src/capture/camera_source.cpp`).

| Key | Action |
|-----|--------|
| `1` `2` `3` | View preset: forward perspective / top-view / ROI zoom |
| `i` `o` | Show/hide IR window 0 / 1 |
| `s` | Digital stabilisation on/off |
| `r` | Recording start/stop (`recordings/avm_<timestamp>.mp4`) |
| `4` | Raw fisheye image as captured (no rotation/geometry); cropped to the image circle, `view.raw_fill: false` shows the whole sensor image |
| `q` / `Esc` | Quit |

The status line shows camera link state, fps and compute time (mean / p99 / max).
Logs go to `logs/avm_YYYYMMDD.log` (all) and `logs/avm_error_YYYYMMDD.log` (warn+error).

**Configuration** — `config/avm_config.json` (documented inline with `_comment` keys):
cameras, calibration and mount, sync tolerance, AGC, PiP windows, view presets, detector,
recording, logging. Values marked PLACEHOLDER (camera height and tilt) must come from the
vessel's installation data.

Environment overrides: `AVM_OUTPUT_WIDTH`, `AVM_OUTPUT_HEIGHT`, `AVM_MODEL_PATH`,
`AVM_DETECTION_THRESHOLD`, `AVM_DISABLE_IR=1`, `AVM_DISABLE_AR=1`, `AVM_DISABLE_INFERENCE=1`.

---

## Calibration workflow (FR-2)

```bash
# 1. Intrinsics from checkerboard photos (20–40 images, also near the image border)
./build/calib_fisheye --images "calib/*.jpg" --board 9x6 --square-mm 25 --out eo_calib.json

# 2. Point the config at it:  "calibration": { "eo_intrinsics": { "file": "eo_calib.json" } }
#    and enter the measured mount height / pitch under "eo_mount".

# 3. Verify against known distances (CSV: kind,label,u1,v1,u2,v2,expected_m), PR-4 tolerance ±5 %
./build/verify_calibration --config config/avm_config.json --points marks.csv --report report.md

# 4. Optional: pre-build LUT files (preset 0 forward, 1 top-view, 2 ROI)
./build/build_lut --config config/avm_config.json --preset 1 --out lut/topview.avmlut
```

Range accuracy is dominated by the mount height and tilt, not by pixel accuracy: at
h = 10 m and 20° tilt, a 1° pitch error or a 0.5 m height error already costs ≈ 4–5 %
(see [docs/spec_compliance.md](docs/spec_compliance.md), §3).

---

## Testing

```bash
ctest --test-dir build --output-on-failure
```

Unit tests (`test/unit/`, no framework dependency): JSON/config, time sync, AGC, fisheye
and view geometry (incl. stabilisation invariance and float-vs-double kernel precision),
LUT file integrity, view transitions, PiP layout, recording control, logging,
frame-ring threading, calibration verification, and validation of the shipped
`config/avm_config.json`.

Legacy OpenCV programs (need `AVM_BUILD_TOOLS`, run from the repository root):
`test/run_tests.sh` / `test/run_tests.bat` run `test_projections`, `gen_fisheye_test`,
`test_undistortion` and `test_pipeline`; `preprocess` batch-processes images.
Qt5 viewers (`AVM_BUILD_VIEWERS`): `fisheye_rectify_live` (4 projection modes on
`/dev/video0`) and `dual_camera_live`; camera indices are hard-coded.

---

## Project Structure

```
shipborad-avm/
├── CMakeLists.txt
├── config/avm_config.json         # runtime configuration
├── docs/spec_compliance.md        # requirement matrix, open issues, design, verification status
├── include/avm/avm_config.hpp     # constants, env overrides, AppConfig -> PipelineConfig
├── scripts/export_model.py        # PyTorch -> ONNX -> TensorRT (x86 GPU workstation)
├── cmake/FindTensorRT.cmake
├── src/
│   ├── main.cpp                   # application: config, logging, UI keys, recording
│   ├── core/                      # Pipeline (processing thread), types, CUDA RAII helpers
│   ├── capture/                   # CaptureManager (threads, zero-copy rings), camera sources
│   ├── render/                    # view kernel + MorphingRenderer, IR PiP kernels, AR overlay
│   ├── undistort/                 # LUT remap (detector input)
│   ├── inference/                 # detector (TensorRT integration pending)
│   ├── alignment/                 # legacy EO/IR alpha blending (optional FR-3.4)
│   ├── geometry/                  # ★ fisheye + virtual-camera math (shared by host tests and CUDA), LUT + file format
│   ├── sync/  imgproc/  view/  record/  calib/  metrics/  config/  util/    # pure C++ (avm_host)
│   └── tools/                     # build_lut, verify_calibration, calib_fisheye, viewers, test programs
└── test/
    ├── unit/                      # CTest unit tests
    └── data/                      # calibration samples
```

---

## Known issues / TODO

- **AI detection (FR-7) is not implemented**: `InferenceEngine` allocates buffers and runs
  preprocessing but never calls TensorRT (results are always empty); no model exists.
  Detection boxes are only drawn on the forward preset.
- **CUDA/OpenCV code is untested on hardware.** Type/syntax checks were done with stubs
  only; expect a round of fixes on the first Jetson build.
- **Camera input path**: the OpenCV/GStreamer source converts on the CPU, which may not
  sustain 12 MP at 30 fps. If it does not, move to Argus/NvBuffer → CUDA (zero CPU
  conversion) or process at 4K. Timestamps are capture-thread times, not sensor times.
- **IR sync** evaluates only the newest IR frame per EO frame (single-slot history).
- **No real IMU/NMEA input** (simulated); AR overlay projection is a rough placeholder
  and is not tied to the view presets. FR-6 is optional in the spec.
- **UI**: temporary OpenCV window; PiP dragging API exists but mouse events are not wired.
- **Alpha blending (FR-3.4)** uses placeholder homographies; it needs EO/IR extrinsic
  calibration and ≤ 10 cm baseline (spec §4.3).
- **INT8 calibration** in `export_model.py` is a stub; use FP16/FP32.
- **Extrinsic calibration**: no automatic horizon-based estimation of mount pitch/roll yet
  (see PR-4 sensitivity in the compliance document).

---

## License

No `LICENSE` file is included in this repository yet; the ownership of the source code is
to be fixed in the contract (spec ch.13).
