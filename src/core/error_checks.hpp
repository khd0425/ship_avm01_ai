#pragma once

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>

#ifndef AVM_CPU_ONLY
#include <cuda_runtime.h>

// ─── CUDA Error Checking ────────────────────────────────────────────────────

#define CUDA_CHECK(ans)                                                 \
    do {                                                                 \
        cudaError_t _err = (ans);                                        \
        if (_err != cudaSuccess) {                                       \
            fprintf(stderr, "[CUDA ERROR] %s:%d :: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(_err));       \
            throw std::runtime_error(cudaGetErrorString(_err));          \
        }                                                                \
    } while (0)

#define CUDA_CHECK_STR(ans, msg)                                         \
    do {                                                                 \
        cudaError_t _err = (ans);                                        \
        if (_err != cudaSuccess) {                                       \
            fprintf(stderr, "[CUDA ERROR] %s:%d :: %s | %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(_err), msg);  \
            throw std::runtime_error(msg);                               \
        }                                                                \
    } while (0)

// ─── CUDA Memory RAII Deleters ─────────────────────────────────────────────

struct CudaFreeDeleter {
    void operator()(void* ptr) const {
        if (ptr) cudaFree(ptr);
    }
};

struct CudaHostFreeDeleter {
    void operator()(void* ptr) const {
        if (ptr) cudaFreeHost(ptr);
    }
};

using unique_cuda_ptr    = std::unique_ptr<void, CudaFreeDeleter>;
using unique_cuda_host   = std::unique_ptr<void, CudaHostFreeDeleter>;

// ─── CUDA Stream RAII ───────────────────────────────────────────────────────

class CudaStream {
public:
    CudaStream() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }
    ~CudaStream() {
        if (stream_) cudaStreamDestroy(stream_);
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    CudaStream(CudaStream&& other) noexcept
        : stream_(other.stream_) {
        other.stream_ = nullptr;
    }
    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) cudaStreamDestroy(stream_);
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    cudaStream_t get() const { return stream_; }

    void sync() const { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

private:
    cudaStream_t stream_{nullptr};
};

// ─── CUDA Event RAII ────────────────────────────────────────────────────────

class CudaEvent {
public:
    CudaEvent() {
        CUDA_CHECK(cudaEventCreate(&event_));
    }
    ~CudaEvent() {
        if (event_) cudaEventDestroy(event_);
    }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    CudaEvent(CudaEvent&& other) noexcept
        : event_(other.event_) {
        other.event_ = nullptr;
    }

    cudaEvent_t get() const { return event_; }

    void record(cudaStream_t stream) {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }

    void sync() const {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }

    float elapsed_since(const CudaEvent& other) const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, other.event_, event_));
        return ms;
    }

private:
    cudaEvent_t event_{nullptr};
};

// ─── Utility: Allocate Pinned (Zero-Copy) Memory ────────────────────────────

// Mapped (zero-copy) pinned buffer. On Jetson (integrated GPU, one shared DRAM)
// `device` is just another view of `host`: the CPU writes camera data and the
// kernels read it with no copy. On a discrete GPU it is PCIe zero-copy (works,
// but slower) - only for development machines.
struct MappedBuffer {
    unique_cuda_host host;
    void* device{nullptr};
    size_t bytes{0};
};

inline MappedBuffer alloc_mapped(size_t bytes) {
    void* h = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h, bytes, cudaHostAllocMapped));
    MappedBuffer b;
    b.host.reset(h);
    CUDA_CHECK(cudaHostGetDevicePointer(&b.device, h, 0));
    b.bytes = bytes;
    return b;
}

template <typename T>
unique_cuda_host alloc_pinned(size_t count) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaHostAlloc(&ptr, count * sizeof(T),
                             cudaHostAllocDefault));
    return unique_cuda_host(ptr);
}

template <typename T>
unique_cuda_ptr alloc_device(size_t count) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return unique_cuda_ptr(ptr);
}

#endif // AVM_CPU_ONLY
