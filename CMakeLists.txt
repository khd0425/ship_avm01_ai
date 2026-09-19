# =============================================================================
# ship_avm01_ai — Shipboard AVM System, CMake build
# =============================================================================
# Target platform : NVIDIA Jetson AGX Orin / Orin NX (JetPack 5/6, aarch64)
# Development     : x86_64 Linux (any machine can build/run the host tests)
#
# Layout of the build
#   avm_host      static lib, pure C++17: JSON, config, logging, geometry/LUT,
#                 time sync, AGC, view presets, PiP layout, recording control,
#                 calibration verification. No CUDA/OpenCV -> unit-tested anywhere.
#   avm_system    the application: CUDA kernels + OpenCV (camera/recording/display)
#                 + optional TensorRT.        (needs AVM_WITH_CUDA=ON and OpenCV)
#   tools         build_lut, verify_calibration (pure C++), calib_fisheye,
#                 test/utility programs and Qt viewers (OpenCV / Qt5).
#
# Quick start on the Jetson:
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j$(nproc)
#   ctest --test-dir build --output-on-failure
#
# Quick start on a PC without CUDA/OpenCV (host library + unit tests only):
#   cmake -S . -B build -DAVM_WITH_CUDA=OFF -DAVM_WITH_OPENCV=OFF \
#         -DAVM_BUILD_TOOLS=OFF -DAVM_BUILD_VIEWERS=OFF
#   cmake --build build -j && ctest --test-dir build --output-on-failure
# =============================================================================

# 3.18: CMAKE_CUDA_ARCHITECTURES. JetPack 5 (Ubuntu 20.04) ships CMake 3.16 -> upgrade
# (pip install cmake / Kitware apt repo). JetPack 6 ships 3.22.
cmake_minimum_required(VERSION 3.18)
project(ship_avm01_ai
    VERSION 0.3.0
    DESCRIPTION "Shipboard Around View Monitoring System for NVIDIA Jetson Orin"
    LANGUAGES CXX
)

# ─── C++ Standard ───────────────────────────────────────────────────────────
# C++17: JetPack 5 has GCC 9.4 (incomplete C++20) and CUDA 11.4 (nvcc: C++17 max).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# TensorRT finder lives in cmake/ (must be on the module path or it is never found)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ─── Build Options ─────────────────────────────────────────────────────────
option(AVM_WITH_CUDA         "Build the CUDA application (avm_system)"                     ON)
option(AVM_WITH_OPENCV       "Use OpenCV (camera capture, recording, display)"             ON)
option(AVM_ENABLE_INFERENCE  "Link TensorRT for the AI detector (needs CUDA)"              ON)
option(AVM_ENABLE_AR_OVERLAY "Enable AR HUD overlay"                                      ON)
option(AVM_BUILD_TOOLS       "Build OpenCV tools (calib_fisheye, test/utility programs)"   ON)
option(AVM_BUILD_VIEWERS     "Build Qt5 live viewers (needs OpenCV + Qt5)"                 ON)
option(AVM_BUILD_TESTS       "Build unit tests (CTest)"                                    ON)

# ─── Platform ───────────────────────────────────────────────────────────────
find_package(Threads REQUIRED)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    message(STATUS "Platform: ${CMAKE_SYSTEM_NAME} aarch64 (Jetson class)")
else()
    message(STATUS "Platform: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# ─── Dependencies ───────────────────────────────────────────────────────────
if(AVM_WITH_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    # Default CUDA architecture: Jetson AGX Orin / Orin NX = sm_87.
    # Override for a dev GPU, e.g. -DCMAKE_CUDA_ARCHITECTURES=86 (RTX 30xx).
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "87")
    endif()
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA ${CUDAToolkit_VERSION}, architecture ${CMAKE_CUDA_ARCHITECTURES}")

    if(AVM_ENABLE_INFERENCE)
        find_package(TensorRT QUIET)
        if(TensorRT_FOUND)
            message(STATUS "TensorRT ${TensorRT_VERSION}")
        else()
            message(WARNING "TensorRT not found - detector runs without TensorRT (set TENSORRT_ROOT)")
            set(AVM_ENABLE_INFERENCE OFF)
        endif()
    endif()
endif()

set(AVM_NEED_OPENCV OFF)
if(AVM_WITH_OPENCV OR AVM_BUILD_TOOLS OR AVM_BUILD_VIEWERS)
    set(AVM_NEED_OPENCV ON)
endif()
if(AVM_NEED_OPENCV)
    # On Jetson the JetPack OpenCV (with GStreamer) is sufficient: pixel work is done by our CUDA kernels.
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui calib3d photo dnn)
    message(STATUS "OpenCV ${OpenCV_VERSION}")
endif()

if(AVM_BUILD_VIEWERS)
    find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets)
    set(CMAKE_AUTOMOC ON)
    set(CMAKE_AUTORCC ON)
    set(CMAKE_AUTOUIC ON)
    message(STATUS "Qt5 ${Qt5_VERSION}")
endif()

# ─── Warnings ───────────────────────────────────────────────────────────────
if(MSVC)
    set(AVM_WARN_FLAGS /W4 /utf-8)
    add_compile_definitions(NOMINMAX _USE_MATH_DEFINES _CRT_SECURE_NO_WARNINGS WIN32_LEAN_AND_MEAN)
else()
    # nvcc rejects GCC-style -W options for CUDA sources: restrict to CXX.
    set(AVM_WARN_FLAGS $<$<COMPILE_LANGUAGE:CXX>:-Wall$<SEMICOLON>-Wextra$<SEMICOLON>-Wno-unused-parameter>)
endif()

# =============================================================================
# avm_host : pure C++ core (unit-tested, no CUDA / OpenCV)
# =============================================================================
add_library(avm_host STATIC
    src/util/json.cpp
    src/util/logger.cpp
    src/config/app_config.cpp
    src/geometry/lut_builder.cpp
    src/sync/frame_sync.cpp
    src/imgproc/agc.cpp
    src/view/view_controller.cpp
    src/view/pip_layout.cpp
    src/record/recording_controller.cpp
    src/calib/verification.cpp
)
target_include_directories(avm_host PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(avm_host PUBLIC Threads::Threads)
target_compile_options(avm_host PRIVATE ${AVM_WARN_FLAGS})

# Command-line tools that need no OpenCV/CUDA: LUT generator (FR-2.2) and
# calibration accuracy verification (FR-2.4).
add_executable(build_lut src/tools/build_lut.cpp)
target_link_libraries(build_lut PRIVATE avm_host)
add_executable(verify_calibration src/tools/verify_calibration.cpp)
target_link_libraries(verify_calibration PRIVATE avm_host)

# ─── Unit tests ─────────────────────────────────────────────────────────────
if(AVM_BUILD_TESTS)
    enable_testing()
    foreach(t test_json_config test_sync_agc test_geometry test_view_misc test_verification test_frame_ring)
        add_executable(${t} test/unit/${t}.cpp)
        target_link_libraries(${t} PRIVATE avm_host)
        target_compile_definitions(${t} PRIVATE AVM_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
        add_test(NAME ${t} COMMAND ${t})
    endforeach()
    # Smoke tests of the command-line tools
    add_test(NAME build_lut_smoke
        COMMAND build_lut --preset 1 --out-w 320 --out-h 180 --out ${CMAKE_BINARY_DIR}/smoke_topview.avmlut)
    add_test(NAME verify_calibration_usage
        COMMAND verify_calibration)
    set_tests_properties(verify_calibration_usage PROPERTIES WILL_FAIL TRUE)   # exits 2 without --points
endif()

# =============================================================================
# avm_system : the application (CUDA + OpenCV [+ TensorRT])
# =============================================================================
if(AVM_WITH_CUDA)
    set(AVM_APP_CXX_SOURCES
        src/main.cpp
        src/core/pipeline.cpp
        src/capture/capture_manager.cpp
        src/capture/camera_source.cpp
        src/render/morphing_renderer.cpp
    )
    # ar_overlay.cpp defines __global__ kernels and launches them with <<<>>>,
    # so it needs the CUDA compiler despite its .cpp extension.
    set_source_files_properties(src/render/ar_overlay.cpp PROPERTIES LANGUAGE CUDA)

    set(AVM_APP_CUDA_SOURCES
        src/render/ar_overlay.cpp
        src/render/cuda_view_render.cu       # per-frame view kernel (presets / top-view / stabilisation)
        src/render/cuda_pip.cu               # IR PiP + AGC compositing
        src/undistort/cuda_remap.cu          # LUT remap (detector input)
        src/inference/trt_inference.cu       # detector (TensorRT integration pending)
        src/alignment/cuda_blend.cu          # optional EO/IR alpha blending (FR-3.4, not in the default pipeline)
    )

    add_executable(avm_system ${AVM_APP_CXX_SOURCES} ${AVM_APP_CUDA_SOURCES})
    target_link_libraries(avm_system PRIVATE avm_host CUDA::cudart Threads::Threads)
    target_include_directories(avm_system PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(avm_system PRIVATE
        ${AVM_WARN_FLAGS}
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    set_target_properties(avm_system PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    if(AVM_WITH_OPENCV)
        target_sources(avm_system PRIVATE src/record/opencv_video_sink.cpp src/inference/dnn_detector.cpp)
        target_compile_definitions(avm_system PRIVATE AVM_WITH_OPENCV=1)
        target_include_directories(avm_system PRIVATE ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(avm_system PRIVATE ${OpenCV_LIBS})
    else()
        message(WARNING "AVM_WITH_OPENCV=OFF: avm_system runs with simulated cameras and no display/recording")
    endif()

    if(AVM_ENABLE_INFERENCE)
        target_link_libraries(avm_system PRIVATE TensorRT::nvinfer TensorRT::nvinfer_plugin TensorRT::nvonnxparser)
        target_compile_definitions(avm_system PRIVATE AVM_ENABLE_INFERENCE=1)
    endif()
    if(AVM_ENABLE_AR_OVERLAY)
        target_compile_definitions(avm_system PRIVATE AVM_ENABLE_AR_OVERLAY=1)
    endif()

    include(GNUInstallDirs)
    install(TARGETS avm_system build_lut verify_calibration RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(FILES config/avm_config.json DESTINATION etc/avm)
    install(DIRECTORY scripts/ DESTINATION share/avm/scripts)
else()
    message(STATUS "AVM_WITH_CUDA=OFF: avm_system is not built (host library, tests and tools only)")
endif()

# =============================================================================
# OpenCV tools
# =============================================================================
if(AVM_BUILD_TOOLS)
    # Checkerboard fisheye calibration (FR-2.1)
    add_executable(calib_fisheye src/tools/calib_fisheye.cpp)
    target_link_libraries(calib_fisheye PRIVATE avm_host ${OpenCV_LIBS})
    target_include_directories(calib_fisheye PRIVATE ${OpenCV_INCLUDE_DIRS})

    # Detector validation on still images (OpenCV DNN backend)
    add_executable(detect_image src/tools/detect_image.cpp src/inference/dnn_detector.cpp)
    target_compile_definitions(detect_image PRIVATE AVM_WITH_OPENCV=1)
    target_link_libraries(detect_image PRIVATE avm_host ${OpenCV_LIBS})
    target_include_directories(detect_image PRIVATE ${OpenCV_INCLUDE_DIRS})

    foreach(tool test_projections test_undistortion test_pipeline gen_fisheye_test preprocess)
        add_executable(${tool} src/tools/${tool}.cpp)
        target_include_directories(${tool} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(${tool} PRIVATE ${OpenCV_LIBS})
    endforeach()
endif()

# =============================================================================
# Qt5 live viewers
# =============================================================================
if(AVM_BUILD_VIEWERS)
    foreach(viewer fisheye_rectify_live dual_camera_live)
        add_executable(${viewer} src/tools/${viewer}.cpp src/tools/${viewer}.hpp)
        target_include_directories(${viewer} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
        target_link_libraries(${viewer} PRIVATE ${OpenCV_LIBS} Qt5::Core Qt5::Gui Qt5::Widgets)
    endforeach()
endif()
