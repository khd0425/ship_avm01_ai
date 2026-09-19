@echo off
REM =============================================================================
REM AVM Test Suite Runner (Windows Batch)
REM =============================================================================
REM Runs the native C++ test programs built by CMake. All programs use paths
REM relative to the project root (test\data\...), so this script changes into it.
REM
REM Usage:
REM   test\run_tests.bat [build_dir]      (build_dir defaults to build\Release)
REM =============================================================================

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.."
set "PROJECT_DIR=%CD%"

set "BUILD_DIR=%~1"
if "%BUILD_DIR%"=="" set "BUILD_DIR=%PROJECT_DIR%\build\Release"
set "TEST_DATA=test\data"
set "FAILED=0"

echo ==============================================
echo   AVM Test Suite
echo   Project:   %PROJECT_DIR%
echo   Build dir: %BUILD_DIR%
echo ==============================================

for %%E in (test_projections gen_fisheye_test test_undistortion test_pipeline) do (
    if not exist "%BUILD_DIR%\%%E.exe" (
        echo [ERROR] %BUILD_DIR%\%%E.exe not found. Build first:
        echo         cmake -S . -B build ^&^& cmake --build build --config Release
        popd
        exit /b 1
    )
)

mkdir "%TEST_DATA%\frames_input" 2>nul
mkdir "%TEST_DATA%\output" 2>nul

echo:
echo [Step 1/4] test_projections
"%BUILD_DIR%\test_projections.exe"
if errorlevel 1 set "FAILED=1"

echo:
echo [Step 2/4] gen_fisheye_test
"%BUILD_DIR%\gen_fisheye_test.exe" "%TEST_DATA%\test_video.mp4" 1920 1080 150 185
if errorlevel 1 set "FAILED=1"

echo:
echo [Step 3/4] test_undistortion
set "FRAME=%TEST_DATA%\frames_input\frame_0000.png"
where ffmpeg >nul 2>nul
if not errorlevel 1 (
    ffmpeg -loglevel error -y -i "%TEST_DATA%\test_video.mp4" -frames:v 1 "%FRAME%"
) else (
    echo [WARN] ffmpeg not found; test_undistortion will use its built-in defaults.
)
"%BUILD_DIR%\test_undistortion.exe" "%FRAME%"
if errorlevel 1 set "FAILED=1"

echo:
echo [Step 4/4] test_pipeline
"%BUILD_DIR%\test_pipeline.exe"
if errorlevel 1 set "FAILED=1"

echo:
echo ==============================================
if "%FAILED%"=="0" (
    echo   All tests passed
) else (
    echo   Some tests FAILED
)
echo ==============================================

popd
endlocal & exit /b %FAILED%
