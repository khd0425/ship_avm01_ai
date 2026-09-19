#pragma once

// =============================================================================
// Fisheye / virtual-camera geometry shared by host code and CUDA kernels.
//
// Everything here is header-only, templated on the scalar type (double for
// host tools/tests, float inside CUDA kernels) and marked AVM_HD so nvcc can
// compile the very same code for the device. It has no dependency on CUDA,
// OpenCV or the STL containers, so it is unit-tested on any machine.
//
// Coordinate frames (all right-handed-looking, OpenCV style):
//   camera frame : x right, y down, z forward (optical axis)
//   level frame  : x right, y down, z forward, horizontal (z is level with the
//                  water); origin = real EO camera. The water plane is the
//                  plane y = +camera_height.
//
// Model
//   The EO camera is a Kannala-Brandt fisheye (OpenCV cv::fisheye):
//       theta   = angle between ray and optical axis
//       theta_d = theta * (1 + k1 t^2 + k2 t^4 + k3 t^6 + k4 t^8)
//       u = fx * theta_d * x/rho + cx ,  v = fy * theta_d * y/rho + cy
//
//   Every displayed view (forward perspective, top-view, zoom ROI) is a
//   *virtual pinhole camera* placed relative to the real camera. A ray of the
//   virtual camera is intersected with the water plane and the hit point is
//   re-projected into the real fisheye. When the virtual camera is co-located
//   with the real one this degenerates to a pure rotation (no plane needed);
//   for the top-view it is exactly inverse perspective mapping (IPM).
//   Because a view is just (pose, fov), presets can be interpolated smoothly.
//
//   Digital stabilisation (FR-5) is done by feeding the vessel roll/pitch
//   into CameraPose: the virtual camera is defined in the level frame, so
//   the output stays level while the real camera moves.
// =============================================================================

#include <algorithm>
#include <math.h>

#if defined(__CUDACC__)
#define AVM_HD __host__ __device__
#else
#define AVM_HD
#endif

namespace avm::geo {

inline constexpr double kPi = 3.14159265358979323846;
inline constexpr double kDegToRad = kPi / 180.0;

// ─── scalar helpers (overloaded so float stays float on the device) ─────────
AVM_HD inline float  m_sqrt(float v)  { return sqrtf(v); }
AVM_HD inline double m_sqrt(double v) { return sqrt(v); }
AVM_HD inline float  m_sin(float v)   { return sinf(v); }
AVM_HD inline double m_sin(double v)  { return sin(v); }
AVM_HD inline float  m_cos(float v)   { return cosf(v); }
AVM_HD inline double m_cos(double v)  { return cos(v); }
AVM_HD inline float  m_tan(float v)   { return tanf(v); }
AVM_HD inline double m_tan(double v)  { return tan(v); }
AVM_HD inline float  m_atan2(float y, float x)   { return atan2f(y, x); }
AVM_HD inline double m_atan2(double y, double x) { return atan2(y, x); }

// ─── small linear algebra ───────────────────────────────────────────────────
template <typename T> struct Vec3 { T x, y, z; };
template <typename T> struct Mat3 { T m[9]; };   // row-major

template <typename T>
AVM_HD inline Mat3<T> mat_identity() {
    Mat3<T> r{};
    r.m[0] = r.m[4] = r.m[8] = T(1);
    return r;
}

template <typename T>
AVM_HD inline Vec3<T> mul(const Mat3<T>& a, const Vec3<T>& v) {
    return {a.m[0] * v.x + a.m[1] * v.y + a.m[2] * v.z,
            a.m[3] * v.x + a.m[4] * v.y + a.m[5] * v.z,
            a.m[6] * v.x + a.m[7] * v.y + a.m[8] * v.z};
}

template <typename T>
AVM_HD inline Mat3<T> transpose(const Mat3<T>& a) {
    Mat3<T> r{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r.m[i * 3 + j] = a.m[j * 3 + i];
    return r;
}

template <typename T>
AVM_HD inline Mat3<T> mul(const Mat3<T>& a, const Mat3<T>& b) {
    Mat3<T> r{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            T s = T(0);
            for (int k = 0; k < 3; ++k) s += a.m[i * 3 + k] * b.m[k * 3 + j];
            r.m[i * 3 + j] = s;
        }
    return r;
}

// Rotation about x by `a` rad: +a tilts the forward (z) axis DOWN (toward +y).
template <typename T>
AVM_HD inline Mat3<T> rot_x(T a) {
    T c = m_cos(a), s = m_sin(a);
    Mat3<T> r{};
    r.m[0] = 1; r.m[1] = 0; r.m[2] = 0;
    r.m[3] = 0; r.m[4] = c; r.m[5] = s;
    r.m[6] = 0; r.m[7] = -s; r.m[8] = c;
    return r;
}

// Rotation about y (down axis) by `a` rad: +a turns the view to the RIGHT.
template <typename T>
AVM_HD inline Mat3<T> rot_y(T a) {
    T c = m_cos(a), s = m_sin(a);
    Mat3<T> r{};
    r.m[0] = c; r.m[1] = 0; r.m[2] = s;
    r.m[3] = 0; r.m[4] = 1; r.m[5] = 0;
    r.m[6] = -s; r.m[7] = 0; r.m[8] = c;
    return r;
}

// Rotation about z (forward axis) by `a` rad: +a rolls the right side DOWN.
template <typename T>
AVM_HD inline Mat3<T> rot_z(T a) {
    T c = m_cos(a), s = m_sin(a);
    Mat3<T> r{};
    r.m[0] = c; r.m[1] = -s; r.m[2] = 0;
    r.m[3] = s; r.m[4] = c;  r.m[5] = 0;
    r.m[6] = 0; r.m[7] = 0;  r.m[8] = 1;
    return r;
}

// ─── fisheye camera (Kannala-Brandt) ────────────────────────────────────────

/// Calibration as stored on disk / in config (always double).
struct FisheyeModel {
    double fx{1000.0}, fy{1000.0};
    double cx{0.0}, cy{0.0};
    double k1{0.0}, k2{0.0}, k3{0.0}, k4{0.0};
    double fov_deg{180.0};   // full field of view; rays beyond fov/2 are invalid
    int width{0}, height{0}; // calibrated image size (informational)
};

template <typename T>
struct FisheyeIntrinsics {
    T fx, fy, cx, cy, k1, k2, k3, k4;
    T max_theta;   // rad
};

template <typename T>
inline FisheyeIntrinsics<T> to_intrinsics(const FisheyeModel& m) {
    FisheyeIntrinsics<T> r{};
    r.fx = T(m.fx); r.fy = T(m.fy); r.cx = T(m.cx); r.cy = T(m.cy);
    r.k1 = T(m.k1); r.k2 = T(m.k2); r.k3 = T(m.k3); r.k4 = T(m.k4);
    r.max_theta = T(m.fov_deg > 0.0 ? m.fov_deg * 0.5 * kDegToRad : kPi);
    return r;
}

template <typename T>
AVM_HD inline T distort_theta(const FisheyeIntrinsics<T>& in, T theta) {
    T t2 = theta * theta;
    return theta * (T(1) + t2 * (in.k1 + t2 * (in.k2 + t2 * (in.k3 + t2 * in.k4))));
}

/// Project a camera-frame ray to fisheye pixel coordinates (pixel centres at
/// integer coordinates, OpenCV convention). False if outside the lens FOV.
template <typename T>
AVM_HD inline bool project_fisheye(const FisheyeIntrinsics<T>& in, const Vec3<T>& ray,
                                   T& u, T& v) {
    T rho = m_sqrt(ray.x * ray.x + ray.y * ray.y);
    T theta = m_atan2(rho, ray.z);
    if (theta > in.max_theta) return false;
    if (rho < T(1e-9)) {
        u = in.cx; v = in.cy;
        return true;
    }
    T scale = distort_theta(in, theta) / rho;
    u = in.fx * ray.x * scale + in.cx;
    v = in.fy * ray.y * scale + in.cy;
    return true;
}

/// Inverse of project_fisheye: pixel -> unit ray in the camera frame.
template <typename T>
AVM_HD inline bool unproject_fisheye(const FisheyeIntrinsics<T>& in, T u, T v, Vec3<T>& ray) {
    T xd = (u - in.cx) / in.fx;
    T yd = (v - in.cy) / in.fy;
    T thd = m_sqrt(xd * xd + yd * yd);
    if (thd < T(1e-9)) { ray = {T(0), T(0), T(1)}; return true; }

    // Solve theta_d = theta * (1 + k1 t^2 + ...) by Newton iteration.
    T theta = thd;
    for (int i = 0; i < 20; ++i) {
        T t2 = theta * theta;
        T f = theta * (T(1) + t2 * (in.k1 + t2 * (in.k2 + t2 * (in.k3 + t2 * in.k4)))) - thd;
        T df = T(1) + t2 * (T(3) * in.k1 + t2 * (T(5) * in.k2 + t2 * (T(7) * in.k3 + t2 * T(9) * in.k4)));
        if (df == T(0)) break;
        T step = f / df;
        theta -= step;
        if ((step < T(0) ? -step : step) < T(1e-10)) break;
    }
    if (theta > in.max_theta * T(1.0001)) return false;
    T s = m_sin(theta);
    ray = {s * xd / thd, s * yd / thd, m_cos(theta)};
    return true;
}

// ─── camera pose over the water ─────────────────────────────────────────────

/// Attitude/height of the real EO camera relative to the level frame.
struct CameraPose {
    double height_m{10.0};            // camera height above the water plane
    Mat3<double> r_lc{mat_identity<double>()};   // camera -> level rotation
};

/// Build the pose from the fixed mount angles and the current vessel attitude.
///   mount_pitch_deg : camera tilt below the horizon (down positive)
///   mount_roll_deg  : camera roll about its own optical axis
///   imu_pitch_deg   : vessel pitch, bow-up positive
///   imu_roll_deg    : vessel roll, starboard-down positive
/// With imu_* = 0 this is the static mount pose (used by calibration/LUT tools).
inline CameraPose make_camera_pose(double height_m, double mount_pitch_deg, double mount_roll_deg,
                                   double imu_pitch_deg = 0.0, double imu_roll_deg = 0.0) {
    CameraPose p;
    p.height_m = height_m;
    Mat3<double> r_mount = mul(rot_x(mount_pitch_deg * kDegToRad), rot_z(mount_roll_deg * kDegToRad));
    Mat3<double> r_vessel = mul(rot_x(-imu_pitch_deg * kDegToRad), rot_z(imu_roll_deg * kDegToRad));
    p.r_lc = mul(r_vessel, r_mount);
    return p;
}

/// Water-plane point (metres, x forward / y right of the camera's foot point)
/// seen at fisheye pixel (u, v). False if the ray does not hit the water.
inline bool pixel_to_ground(const FisheyeModel& model, const CameraPose& pose,
                            double u, double v, double& x_fwd, double& y_right) {
    Vec3<double> ray;
    if (!unproject_fisheye(to_intrinsics<double>(model), u, v, ray)) return false;
    Vec3<double> d = mul(pose.r_lc, ray);
    if (d.y <= 1e-9) return false;
    double t = pose.height_m / d.y;
    x_fwd = t * d.z;
    y_right = t * d.x;
    return true;
}

/// Fisheye pixel of a water-plane point. False if it is outside the lens FOV.
inline bool ground_to_pixel(const FisheyeModel& model, const CameraPose& pose,
                            double x_fwd, double y_right, double& u, double& v) {
    Vec3<double> w{y_right, pose.height_m, x_fwd};
    Vec3<double> c = mul(transpose(pose.r_lc), w);
    return project_fisheye(to_intrinsics<double>(model), c, u, v);
}

// ─── virtual view ───────────────────────────────────────────────────────────

// kViewRaw = the original fisheye image as captured (no rotation, no geometry): scaled to fit
// the output and centred. Used to check the camera and as the "as-is" operator view.
enum ViewKind : int { kViewRectilinear = 0, kViewCylindrical = 1, kViewRaw = 2 };

/// A displayed view = virtual pinhole camera defined in the level frame.
struct ViewParams {
    int kind{kViewRectilinear};
    double yaw_deg{0.0};      // + turns right
    double pitch_deg{0.0};    // + looks down
    double roll_deg{0.0};
    double hfov_deg{100.0};   // horizontal field of view
    // Position of the virtual camera relative to the real one (metres).
    double offset_forward_m{0.0};
    double offset_right_m{0.0};
    double offset_up_m{0.0};
    // kViewRaw only: 1 = crop to the largest output-shaped rectangle inside the fisheye image
    // circle (no black surround); 0 = show the whole sensor image, black outside the circle.
    int raw_fill{0};
};

/// Everything a pixel needs, precomputed on the host and passed to the kernel
/// by value (POD).
template <typename T>
struct ViewMapper {
    FisheyeIntrinsics<T> intr;
    Mat3<T> r_cl;        // level -> real camera
    Mat3<T> r_lv;        // virtual camera -> level
    Vec3<T> pv;          // virtual camera position in the level frame
    T plane_y;           // water plane y (= real camera height)
    T f_px;              // virtual focal length (px) [rect] or px/rad [cyl]
    T cx, cy;            // virtual principal point
    T max_range;         // beyond this the hit point is treated as infinity
    T raw_scale;         // kViewRaw: output px per source px
    T src_cx, src_cy;    // kViewRaw: source image centre (px)
    int kind;
    int colocated;       // virtual camera at the real camera position

    /// Output pixel -> fisheye source pixel. False if not visible.
    AVM_HD bool map(T u, T v, T& su, T& sv) const {
        if (kind == kViewRaw) {                       // identity: original image, fitted and centred
            su = (u - cx) / raw_scale + src_cx;
            sv = (v - cy) / raw_scale + src_cy;
            return true;                              // callers bounds-check the source coordinate
        }
        Vec3<T> dv;
        if (kind == kViewCylindrical) {
            T phi = (u - cx) / f_px;
            dv = {m_sin(phi), (v - cy) / f_px, m_cos(phi)};
        } else {
            dv = {(u - cx) / f_px, (v - cy) / f_px, T(1)};
        }
        Vec3<T> d = mul(r_lv, dv);
        Vec3<T> w = d;
        if (!colocated && d.y > T(1e-6)) {
            T t = (plane_y - pv.y) / d.y;
            T len = m_sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
            if (t > T(0) && t * len < max_range) {
                w = {pv.x + t * d.x, pv.y + t * d.y, pv.z + t * d.z};
            }
        }
        Vec3<T> c = mul(r_cl, w);
        return project_fisheye(intr, c, su, sv);
    }

    /// Output pixel -> water-plane point (x forward / y right of the real
    /// camera's foot point). Used to range-gate detections and for BEV grids.
    AVM_HD bool view_pixel_to_ground(T u, T v, T& x_fwd, T& y_right) const {
        if (kind == kViewRaw) return false;           // no ground geometry in the raw view
        Vec3<T> dv;
        if (kind == kViewCylindrical) {
            T phi = (u - cx) / f_px;
            dv = {m_sin(phi), (v - cy) / f_px, m_cos(phi)};
        } else {
            dv = {(u - cx) / f_px, (v - cy) / f_px, T(1)};
        }
        Vec3<T> d = mul(r_lv, dv);
        if (d.y <= T(1e-6)) return false;
        T t = (plane_y - pv.y) / d.y;
        if (t <= T(0)) return false;
        x_fwd = pv.z + t * d.z;
        y_right = pv.x + t * d.x;
        return true;
    }

    /// Water-plane point -> output pixel of this view (inverse of the above).
    AVM_HD bool ground_to_view_pixel(T x_fwd, T y_right, T& u, T& v) const {
        if (kind == kViewRaw) return false;
        Vec3<T> w{y_right - pv.x, plane_y - pv.y, x_fwd - pv.z};
        Vec3<T> c = mul(transpose(r_lv), w);
        if (c.z <= T(1e-6)) return false;
        if (kind == kViewCylindrical) {
            T rho = m_sqrt(c.x * c.x + c.z * c.z);
            u = cx + f_px * m_atan2(c.x, c.z);
            v = cy + f_px * c.y / rho;
        } else {
            u = cx + f_px * c.x / c.z;
            v = cy + f_px * c.y / c.z;
        }
        return true;
    }
};

/// Build a ViewMapper for an output image of out_w x out_h pixels.
template <typename T>
inline ViewMapper<T> make_view_mapper(const FisheyeModel& model, const CameraPose& pose,
                                      const ViewParams& view, int out_w, int out_h,
                                      double max_range_m = 5000.0) {
    ViewMapper<T> m{};
    m.intr = to_intrinsics<T>(model);

    Mat3<double> r_cl = transpose(pose.r_lc);
    Mat3<double> r_lv = mul(rot_y(view.yaw_deg * kDegToRad),
                            mul(rot_x(view.pitch_deg * kDegToRad), rot_z(view.roll_deg * kDegToRad)));
    for (int i = 0; i < 9; ++i) { m.r_cl.m[i] = T(r_cl.m[i]); m.r_lv.m[i] = T(r_lv.m[i]); }

    m.pv = {T(view.offset_right_m), T(-view.offset_up_m), T(view.offset_forward_m)};
    m.plane_y = T(pose.height_m);
    m.max_range = T(max_range_m);
    m.kind = view.kind;

    const double half_fov = 0.5 * view.hfov_deg * kDegToRad;
    m.f_px = T(view.kind == kViewCylindrical ? (0.5 * out_w) / half_fov
                                             : (0.5 * out_w) / m_tan(half_fov));
    m.cx = T(0.5 * (out_w - 1));
    m.cy = T(0.5 * (out_h - 1));
    if (view.kind == kViewRaw) {
        const double sw = model.width > 0 ? model.width : out_w;
        const double sh = model.height > 0 ? model.height : out_h;
        if (view.raw_fill) {
            // Image circle radius (px) from the lens model, then the inscribed rectangle of the output aspect.
            const FisheyeIntrinsics<double> in = to_intrinsics<double>(model);
            const double r = 0.97 * std::min(in.fx, in.fy) * distort_theta(in, in.max_theta);
            const double a = static_cast<double>(out_w) / out_h;
            const double half_h = r / std::sqrt(1.0 + a * a);
            m.raw_scale = T(0.5 * out_h / half_h);               // output px per source px
            m.src_cx = T(model.cx);
            m.src_cy = T(model.cy);
        } else {
            const double s = std::min(static_cast<double>(out_w) / sw, static_cast<double>(out_h) / sh);
            m.raw_scale = T(s);
            m.src_cx = T(0.5 * (sw - 1));
            m.src_cy = T(0.5 * (sh - 1));
        }
    }
    m.colocated = (view.offset_forward_m == 0.0 && view.offset_right_m == 0.0 &&
                   view.offset_up_m == 0.0) ? 1 : 0;
    return m;
}

} // namespace avm::geo
