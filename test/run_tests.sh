#!/usr/bin/env bash
# =============================================================================
# AVM Test Suite Runner
# =============================================================================
# Runs the native C++ test programs built by CMake (they live in the build
# directory next to avm_system). All programs use paths relative to the
# project root (test/data/...), so this script changes into it first.
#
#   1. test_projections   - projection math unit tests
#   2. gen_fisheye_test   - synthetic fisheye test video (test/data/test_video.mp4)
#   3. test_undistortion  - undistortion validation on the first video frame
#   4. test_pipeline      - pipeline harness (picks up test/data/test_video.mp4)
#   5. Markdown report    - test/data/report_<timestamp>.md
#
# Usage:
#   bash test/run_tests.sh [build_dir]      # build_dir defaults to ./build
# =============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${1:-$PROJECT_DIR/build}"
TEST_DATA="test/data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT="$TEST_DATA/report_${TIMESTAMP}.md"

cd "$PROJECT_DIR"

echo "=============================================="
echo "  AVM Test Suite"
echo "  Project:   $PROJECT_DIR"
echo "  Build dir: $BUILD_DIR"
echo "=============================================="

for exe in test_projections gen_fisheye_test test_undistortion test_pipeline; do
    if [ ! -x "$BUILD_DIR/$exe" ]; then
        echo "[ERROR] $BUILD_DIR/$exe not found. Build first:"
        echo "        cmake -S . -B build && cmake --build build -j\$(nproc)"
        exit 1
    fi
done

mkdir -p "$TEST_DATA/frames_input" "$TEST_DATA/output"

declare -a RESULTS=()
FAILED=0

run_step() {
    local name="$1"; shift
    echo ""
    echo "[Step] $name"
    if "$@"; then
        RESULTS+=("PASS  $name")
    else
        RESULTS+=("FAIL  $name")
        FAILED=1
    fi
}

# ─── 1. Projection math ─────────────────────────────────────────────────────
run_step "test_projections" "$BUILD_DIR/test_projections"

# ─── 2. Synthetic fisheye video (1920x1080, 150 frames, 185° FOV) ───────────
run_step "gen_fisheye_test" \
    "$BUILD_DIR/gen_fisheye_test" "$TEST_DATA/test_video.mp4" 1920 1080 150 185

# ─── 3. Undistortion validation on the first frame ──────────────────────────
FRAME="$TEST_DATA/frames_input/frame_0000.png"
if command -v ffmpeg &> /dev/null && [ -f "$TEST_DATA/test_video.mp4" ]; then
    ffmpeg -loglevel error -y -i "$TEST_DATA/test_video.mp4" -frames:v 1 "$FRAME"
else
    echo "[WARN] ffmpeg not found or video missing; test_undistortion will use its built-in defaults."
fi
run_step "test_undistortion" "$BUILD_DIR/test_undistortion" "$FRAME"

# ─── 4. Pipeline harness ────────────────────────────────────────────────────
run_step "test_pipeline" "$BUILD_DIR/test_pipeline"

# ─── 5. Report ──────────────────────────────────────────────────────────────
{
    echo "# AVM Test Report"
    echo "**Date**: $(date)"
    echo "**Project**: $PROJECT_DIR"
    echo ""
    echo "## Results"
    for r in "${RESULTS[@]}"; do echo "- $r"; done
} > "$REPORT"

echo ""
echo "=============================================="
if [ "$FAILED" -eq 0 ]; then
    echo "  All tests passed"
else
    echo "  Some tests FAILED"
fi
echo "  Report: $REPORT"
echo "=============================================="
exit "$FAILED"
