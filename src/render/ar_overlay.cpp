#include "ar_overlay.hpp"
#include "src/core/error_checks.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace avm {

// ─── CUDA Kernel: draw a colored rectangle ─────────────────────────────────
// Simplified overlay rendering. In production this would use a proper
// 2D vector graphics rasterizer. For the scaffold we use a simple
// pixel-writing kernel.
//
__global__ void kernel_draw_rect_rgba(
    uint8_t* image, int pitch, int w, int h,
    int rx, int ry, int rw, int rh,
    uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x + rx;
    int row = blockIdx.y * blockDim.y + threadIdx.y + ry;
    if (col < rx || col >= rx + rw || col >= w) return;
    if (row < ry || row >= ry + rh || row >= h) return;

    // Only draw border (3px)
    bool border = (col - rx < 3) || (rx + rw - col <= 3) ||
                  (row - ry < 3) || (ry + rh - row <= 3);
    if (!border) return;

    uint8_t* px = image + row * pitch + col * 4;
    px[0] = r; px[1] = g; px[2] = b; px[3] = a;
}

// ─── Kernel: draw a line using Bresenham-like approach ─────────────────────

__global__ void kernel_draw_line_rgba(
    uint8_t* image, int pitch, int w, int h,
    int x0, int y0, int x1, int y1,
    uint8_t r, uint8_t g, uint8_t b, uint8_t a)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx > 4096) return;  // safety limit

    // Simple DDA line
    int dx = x1 - x0, dy = y1 - y0;
    int steps = max(abs(dx), abs(dy));
    if (steps == 0) return;

    float x_inc = static_cast<float>(dx) / steps;
    float y_inc = static_cast<float>(dy) / steps;
    float x = x0, y = y0;

    for (int i = 0; i <= steps; ++i, x += x_inc, y += y_inc) {
        int px = static_cast<int>(x + 0.5f);
        int py = static_cast<int>(y + 0.5f);
        if (px >= 0 && px < w && py >= 0 && py < h) {
            uint8_t* pxel = image + py * pitch + px * 4;
            pxel[0] = r; pxel[1] = g; pxel[2] = b; pxel[3] = a;
        }
    }
}

// ─── Implementation ─────────────────────────────────────────────────────────

AROverlay::AROverlay(uint32_t output_width,
                     uint32_t output_height,
                     const CameraCalibration& eo_calib)
    : out_w_(output_width)
    , out_h_(output_height)
    , eo_calib_(eo_calib) {
}

AROverlay::~AROverlay() {
    if (d_ais_vertices_) cudaFree(d_ais_vertices_);
    if (d_ais_indices_)  cudaFree(d_ais_indices_);
}

void AROverlay::updateTargets(const std::vector<AISTarget>& targets) {
    ais_render_data_.clear();

    for (const auto& t : targets) {
        AISRenderData rd{};

        // Project lat/lon to screen coordinates (simplified)
        // In production this would use the camera projection matrix
        double dlat = t.latitude - current_gps_.latitude;
        double dlon = t.longitude - current_gps_.longitude;
        // Rough: 1° ≈ 111320 m
        float dx = static_cast<float>(dlon * 111320.0 * cos(current_gps_.latitude * M_PI / 180.0));
        float dy = static_cast<float>(dlat * 111320.0);

        // Transform to view space (simplified)
        float view_dist = sqrtf(dx * dx + dy * dy);
        if (view_dist < 1.0f) view_dist = 1.0f;

        // Simple projection: objects further away appear smaller
        float scale = 500.0f / view_dist;
        if (scale > 1.0f) scale = 1.0f;

        rd.x_norm = 0.5f + dx / view_dist * scale * 0.5f;
        rd.y_norm = 0.5f - dy / view_dist * scale * 0.5f;
        rd.heading_rad = t.heading * M_PI / 180.0f;
        rd.sog_knots = t.sog;
        rd.mmsi = t.mmsi;
        snprintf(rd.label, sizeof(rd.label), "%s", t.ship_name);

        // Simple CPA calculation (straight-line extrapolation)
        float rel_speed_x = t.sog * sinf(t.heading * M_PI / 180.0f) -
                            current_gps_.speed_knots * sinf(current_heading_ * M_PI / 180.0f);
        float rel_speed_y = t.sog * cosf(t.heading * M_PI / 180.0f) -
                            current_gps_.speed_knots * cosf(current_heading_ * M_PI / 180.0f);
        float rel_speed = sqrtf(rel_speed_x * rel_speed_x + rel_speed_y * rel_speed_y);

        if (rel_speed > 0.1f) {
            float ttc = view_dist / (rel_speed * 0.514444f);  // knots → m/s
            rd.tcpa_min = ttc / 60.0f;
            // CPA = closest distance assuming constant velocity
            float cross = fabsf(dx * rel_speed_y - dy * rel_speed_x);
            rd.cpa_nm = cross / (rel_speed * 1852.0f);  // meters → NM
        } else {
            rd.tcpa_min = 999.0f;
            rd.cpa_nm = view_dist / 1852.0f;
        }

        ais_render_data_.push_back(rd);
    }
}

void AROverlay::updateDetections(const DetectionResult& detections) {
    current_detections_ = detections;
}

void AROverlay::updateNavigation(const GPSData& gps, float heading_deg) {
    current_gps_ = gps;
    current_heading_ = heading_deg;
}

void AROverlay::render(CameraFrame& image, cudaStream_t stream) {
    // ─── Draw heading indicator ─────────────────────────────────────────
    if (show_heading_) {
        int cx = out_w_ / 2;
        int cy = out_h_ - 80;
        int len = 60;

        int hx = cx + static_cast<int>(len * sinf(current_heading_ * M_PI / 180.0f));
        int hy = cy - static_cast<int>(len * cosf(current_heading_ * M_PI / 180.0f));

        dim3 block(256);
        dim3 grid(16);
        kernel_draw_line_rgba<<<grid, block, 0, stream>>>(
            static_cast<uint8_t*>(image.device_data),
            static_cast<int>(image.device_pitch),
            static_cast<int>(out_w_), static_cast<int>(out_h_),
            cx, cy, hx, hy,
            0, 255, 0, 255);  // green
    }

    // ─── Draw AIS targets ───────────────────────────────────────────────
    if (show_ais_) {
        for (const auto& rd : ais_render_data_) {
            int sx = static_cast<int>(rd.x_norm * out_w_);
            int sy = static_cast<int>(rd.y_norm * out_h_);
            if (sx < 0 || sx >= static_cast<int>(out_w_) ||
                sy < 0 || sy >= static_cast<int>(out_h_)) continue;

            // Draw a small triangle pointing in heading direction
            int size = 8;
            int tip_x = sx + static_cast<int>(size * sinf(rd.heading_rad));
            int tip_y = sy - static_cast<int>(size * cosf(rd.heading_rad));

            dim3 block(256);
            dim3 grid(16);
            kernel_draw_line_rgba<<<grid, block, 0, stream>>>(
                static_cast<uint8_t*>(image.device_data),
                static_cast<int>(image.device_pitch),
                static_cast<int>(out_w_), static_cast<int>(out_h_),
                sx, sy, tip_x, tip_y,
                255, 255, 0, 255);  // yellow
        }
    }

    // ─── Draw detection boxes ───────────────────────────────────────────
    if (show_detections_) {
        for (const auto& box : current_detections_.boxes) {
            int rx = static_cast<int>(box.x * out_w_);
            int ry = static_cast<int>(box.y * out_h_);
            int rw = static_cast<int>(box.w * out_w_);
            int rh = static_cast<int>(box.h * out_h_);
            if (rw <= 0 || rh <= 0) continue;  // empty grid = invalid launch config

            dim3 block(16, 16);
            dim3 grid((rw + 15) / 16, (rh + 15) / 16);
            kernel_draw_rect_rgba<<<grid, block, 0, stream>>>(
                static_cast<uint8_t*>(image.device_data),
                static_cast<int>(image.device_pitch),
                static_cast<int>(out_w_), static_cast<int>(out_h_),
                rx, ry, rw, rh,
                255, 0, 0, 255);  // red
        }
    }

    CUDA_CHECK(cudaGetLastError());
}

} // namespace avm
