# ─── FindTensorRT.cmake ─────────────────────────────────────────────────────
#
# Finds NVIDIA TensorRT on the system.
# Exposes the following imported targets:
#   TensorRT::nvinfer
#   TensorRT::nvinfer_plugin
#   TensorRT::nvonnxparser
#   TensorRT::nvparsers (optional)
#
# Also sets:
#   TensorRT_FOUND       - True if all required libraries were found
#   TensorRT_INCLUDE_DIR - Include directory
#   TensorRT_LIBRARIES   - List of all found TensorRT libraries
#
# On Jetson / JetPack, TensorRT is typically installed at:
#   /usr/include/aarch64-linux-gnu
#   /usr/lib/aarch64-linux-gnu
#
# On x86 workstation:
#   /usr/include/x86_64-linux-gnu
#   /usr/lib/x86_64-linux-gnu
# Or from the Deb/RPM package: /opt/TensorRT-{version}
# ============================================================================

# ─── Search hints ──────────────────────────────────────────────────────────
set(_TRT_SEARCH_HINTS
    /usr
    /usr/local
    /opt/TensorRT
    /opt/TensorRT-*
    $ENV{TENSORRT_ROOT}
    $ENV{TENSORRT_DIR}
    "C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT"
    "C:/Program Files (x86)/NVIDIA GPU Computing Toolkit/TensorRT"
)

# ─── Include directory ─────────────────────────────────────────────────────
find_path(TensorRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS ${_TRT_SEARCH_HINTS}
    PATH_SUFFIXES
        include
        include/aarch64-linux-gnu
        include/x86_64-linux-gnu
)

# ─── Libraries ─────────────────────────────────────────────────────────────

foreach(_COMP nvinfer nvinfer_plugin nvonnxparser nvparsers)
    # On Windows, TensorRT libraries have .lib extension (import libraries)
    # and may have different naming conventions
    find_library(TensorRT_${_COMP}_LIBRARY
        NAMES ${_COMP} ${_COMP}_d nvinfer
        HINTS ${_TRT_SEARCH_HINTS}
        PATH_SUFFIXES
            lib
            lib/aarch64-linux-gnu
            lib/x86_64-linux-gnu
            lib64
    )
    if(TensorRT_${_COMP}_LIBRARY)
        list(APPEND TensorRT_LIBRARIES ${TensorRT_${_COMP}_LIBRARY})
    endif()
endforeach()

# ─── Version ────────────────────────────────────────────────────────────────
if(TensorRT_INCLUDE_DIR AND EXISTS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
    file(STRINGS "${TensorRT_INCLUDE_DIR}/NvInferVersion.h" _TRT_VER
         REGEX "#define NV_TENSORRT_(MAJOR|MINOR|PATCH) ")
    string(REGEX REPLACE ".*NV_TENSORRT_MAJOR ([0-9]+).*" "\\1"
           TensorRT_VERSION_MAJOR "${_TRT_VER}")
    string(REGEX REPLACE ".*NV_TENSORRT_MINOR ([0-9]+).*" "\\1"
           TensorRT_VERSION_MINOR "${_TRT_VER}")
    string(REGEX REPLACE ".*NV_TENSORRT_PATCH ([0-9]+).*" "\\1"
           TensorRT_VERSION_PATCH "${_TRT_VER}")
    set(TensorRT_VERSION
        "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
endif()

# ─── Standard find_package handling ────────────────────────────────────────
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TensorRT_INCLUDE_DIR TensorRT_nvinfer_LIBRARY
    VERSION_VAR TensorRT_VERSION
)

# ─── Imported targets ──────────────────────────────────────────────────────
if(TensorRT_FOUND AND NOT TARGET TensorRT::nvinfer)
    add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
    set_target_properties(TensorRT::nvinfer PROPERTIES
        IMPORTED_LOCATION "${TensorRT_nvinfer_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
    )

    if(TensorRT_nvinfer_plugin_LIBRARY)
        add_library(TensorRT::nvinfer_plugin UNKNOWN IMPORTED)
        set_target_properties(TensorRT::nvinfer_plugin PROPERTIES
            IMPORTED_LOCATION "${TensorRT_nvinfer_plugin_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
        )
    endif()

    if(TensorRT_nvonnxparser_LIBRARY)
        add_library(TensorRT::nvonnxparser UNKNOWN IMPORTED)
        set_target_properties(TensorRT::nvonnxparser PROPERTIES
            IMPORTED_LOCATION "${TensorRT_nvonnxparser_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
        )
    endif()

    if(TensorRT_nvparsers_LIBRARY)
        add_library(TensorRT::nvparsers UNKNOWN IMPORTED)
        set_target_properties(TensorRT::nvparsers PROPERTIES
            IMPORTED_LOCATION "${TensorRT_nvparsers_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR
    TensorRT_nvinfer_LIBRARY
    TensorRT_nvinfer_plugin_LIBRARY
    TensorRT_nvonnxparser_LIBRARY
    TensorRT_nvparsers_LIBRARY
)
