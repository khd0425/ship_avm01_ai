#include "app_config.hpp"

#include "src/util/json.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>

namespace avm::config {

using util::Json;

AppConfig default_app_config() {
    AppConfig c;

    // 12.33 MP fisheye (4056 x 3040: reproduces the RAW10 3.70 Gbps / RGBA 49.3 MB
    // figures of spec ch.3.3), image circle 3040 px = 180 deg  =>  f = r / (pi/2)
    c.eo.width = 4056;
    c.eo.height = 3040;
    c.eo_intrinsics.width = 4056;
    c.eo_intrinsics.height = 3040;
    c.eo_intrinsics.cx = 2027.5;
    c.eo_intrinsics.cy = 1519.5;
    c.eo_intrinsics.fx = c.eo_intrinsics.fy = 1520.0 / (geo::kPi / 2.0);
    c.eo_intrinsics.fov_deg = 185.0;

    for (int i = 0; i < 2; ++i) {                       // spec: IR x 2
        CameraCfg ir;
        ir.type = "IR_THERMAL";
        ir.width = 640;
        ir.height = 480;
        ir.pixel_format = "MONO8";
        ir.gmsl_link_id = i + 1;
        c.ir.push_back(ir);
        c.ir_intrinsics.push_back(PinholeCfg{});
    }

    view::PipWindowCfg left;
    left.anchor = view::PipAnchor::MiddleLeft;
    view::PipWindowCfg right;
    right.anchor = view::PipAnchor::MiddleRight;
    c.pip.windows = {left, right};

    // Phase-1 classes (docs/ai_dataset_plan.md). Deferred: mooring_line (thin object, no public
    // data) and lighthouse (far beyond the 0-50 m berthing range).
    c.inference.classes = {"person", "bollard", "fender", "quay_edge", "small_vessel", "buoy"};
    return c;
}

// ─── helpers ────────────────────────────────────────────────────────────────

static void read_camera(const Json& j, CameraCfg& c) {
    if (!j.is_object()) return;
    c.type = j.string_or("type", c.type);
    c.width = static_cast<std::uint32_t>(j.number_or("width", c.width));
    c.height = static_cast<std::uint32_t>(j.number_or("height", c.height));
    c.pixel_format = j.string_or("pixel_format", c.pixel_format);
    c.fps = j.number_or("fps", c.fps);
    c.gmsl_link_id = static_cast<int>(j.number_or("gmsl_link_id", c.gmsl_link_id));
    c.source = j.string_or("source", c.source);
    c.fourcc = j.string_or("fourcc", c.fourcc);
    c.exposure = static_cast<int>(j.number_or("exposure", c.exposure));
}

static void read_fisheye(const Json& j, geo::FisheyeModel& m) {
    if (!j.is_object()) return;
    m.fx = j.number_or("fx", m.fx);
    m.fy = j.number_or("fy", m.fy);
    m.cx = j.number_or("cx", m.cx);
    m.cy = j.number_or("cy", m.cy);
    m.k1 = j.number_or("k1", m.k1);
    m.k2 = j.number_or("k2", m.k2);
    m.k3 = j.number_or("k3", m.k3);
    m.k4 = j.number_or("k4", m.k4);
    m.fov_deg = j.number_or("fov_deg", m.fov_deg);
    m.width = static_cast<int>(j.number_or("image_width", j.number_or("width", m.width)));
    m.height = static_cast<int>(j.number_or("image_height", j.number_or("height", m.height)));
}

static void read_pinhole(const Json& j, PinholeCfg& p) {
    if (!j.is_object()) return;
    p.fx = j.number_or("fx", p.fx);
    p.fy = j.number_or("fy", p.fy);
    p.cx = j.number_or("cx", p.cx);
    p.cy = j.number_or("cy", p.cy);
    p.fov_deg = j.number_or("fov_deg", p.fov_deg);
}

static std::string read_file(const std::string& path, bool& ok) {
    std::ifstream f(path, std::ios::binary);
    ok = static_cast<bool>(f);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

bool parse_app_config(const std::string& text, AppConfig& cfg, std::string* err) {
    Json root;
    std::string perr;
    if (!Json::try_parse(text, root, &perr)) {
        if (err) *err = perr;
        return false;
    }
    if (!root.is_object()) {
        if (err) *err = "config root must be a JSON object";
        return false;
    }
    AppConfig c = cfg;   // start from the caller's defaults

    if (const Json* cam = root.find("camera")) {
        if (const Json* eo = cam->find("eo")) read_camera(*eo, c.eo);
        if (const Json* ir = cam->find("ir"); ir && ir->is_array()) {
            std::vector<CameraCfg> list;
            for (const Json& item : ir->items()) {
                CameraCfg cc;
                cc.type = "IR_THERMAL"; cc.width = 640; cc.height = 480; cc.pixel_format = "MONO8";
                cc.gmsl_link_id = static_cast<int>(list.size()) + 1;
                read_camera(item, cc);
                list.push_back(cc);
            }
            c.ir = list;
            c.ir_intrinsics.resize(c.ir.size());
        }
    }
    if (const Json* o = root.find("output")) {
        c.output.width = static_cast<std::uint32_t>(o->number_or("width", c.output.width));
        c.output.height = static_cast<std::uint32_t>(o->number_or("height", c.output.height));
    }
    if (const Json* cal = root.find("calibration")) {
        if (const Json* e = cal->find("eo_intrinsics")) {
            read_fisheye(*e, c.eo_intrinsics);
            const std::string file = e->string_or("file", "");
            if (!file.empty()) {
                std::string ferr;
                if (!load_fisheye_calibration_file(file, c.eo_intrinsics, &ferr)) {
                    if (err) *err = "eo_intrinsics.file: " + ferr;
                    return false;
                }
            }
        }
        if (const Json* irs = cal->find("ir_intrinsics")) {
            if (irs->is_array()) {
                for (std::size_t i = 0; i < irs->items().size() && i < c.ir_intrinsics.size(); ++i)
                    read_pinhole(irs->items()[i], c.ir_intrinsics[i]);
            } else {
                for (auto& p : c.ir_intrinsics) read_pinhole(*irs, p);   // one entry for all
            }
        }
        if (const Json* m = cal->find("eo_mount")) {
            c.mount.height_m = m->number_or("height_m", c.mount.height_m);
            c.mount.pitch_deg = m->number_or("pitch_deg", c.mount.pitch_deg);
            c.mount.roll_deg = m->number_or("roll_deg", c.mount.roll_deg);
        }
    }
    if (const Json* s = root.find("sync")) {
        c.sync.tolerance_us = static_cast<std::int64_t>(s->number_or("tolerance_ms", c.sync.tolerance_us / 1000.0) * 1000.0);
        c.sync.stale_timeout_us = static_cast<std::int64_t>(s->number_or("stale_timeout_ms", c.sync.stale_timeout_us / 1000.0) * 1000.0);
    }
    if (const Json* a = root.find("agc")) {
        c.agc.mode = a->string_or("mode", c.agc.mode == imgproc::AgcMode::Plateau ? "plateau" : "linear") == "linear"
                         ? imgproc::AgcMode::Linear : imgproc::AgcMode::Plateau;
        c.agc.clip_low_percent = a->number_or("clip_low_percent", c.agc.clip_low_percent);
        c.agc.clip_high_percent = a->number_or("clip_high_percent", c.agc.clip_high_percent);
        c.agc.min_range_fraction = a->number_or("min_range_fraction", c.agc.min_range_fraction);
        c.agc.smoothing = a->number_or("smoothing", c.agc.smoothing);
        c.agc.plateau_fraction = a->number_or("plateau_fraction", c.agc.plateau_fraction);
        c.agc.plateau_blend = a->number_or("plateau_blend", c.agc.plateau_blend);
        c.agc.max_gain = a->number_or("max_gain", c.agc.max_gain);
    }
    if (const Json* p = root.find("pip")) {
        c.pip.enabled = p->bool_or("enabled", c.pip.enabled);
        if (const Json* w = p->find("windows"); w && w->is_array()) {
            c.pip.windows.clear();
            for (const Json& item : w->items()) {
                view::PipWindowCfg pw;
                pw.visible = item.bool_or("visible", pw.visible);
                pw.anchor = view::parse_anchor(item.string_or("anchor", "top_left"));
                pw.width_frac = item.number_or("width_frac", pw.width_frac);
                pw.margin_px = item.number_or("margin_px", pw.margin_px);
                pw.custom_x = item.number_or("x", pw.custom_x);
                pw.custom_y = item.number_or("y", pw.custom_y);
                c.pip.windows.push_back(pw);
            }
        }
    }
    if (const Json* v = root.find("view")) {
        auto& ps = c.view.presets;
        ps.transition_ms = v->number_or("transition_ms", ps.transition_ms);
        ps.default_preset = static_cast<int>(v->number_or("default_preset", ps.default_preset));
        ps.fit_to_visible = v->bool_or("fit_to_visible", ps.fit_to_visible);
        ps.raw_fill = v->bool_or("raw_fill", ps.raw_fill);
        if (const Json* f = v->find("forward")) {
            ps.forward_hfov_deg = f->number_or("hfov_deg", ps.forward_hfov_deg);
            ps.forward_pitch_deg = f->number_or("pitch_deg", ps.forward_pitch_deg);
        }
        if (const Json* t = v->find("topview")) {
            ps.topview_radius_m = t->number_or("radius_m", ps.topview_radius_m);
            ps.topview_vfov_deg = t->number_or("vfov_deg", ps.topview_vfov_deg);
            ps.topview_center_forward_m = t->number_or("center_forward_m", ps.topview_center_forward_m);
        }
        if (const Json* r = v->find("roi")) {
            ps.roi_yaw_deg = r->number_or("yaw_deg", ps.roi_yaw_deg);
            ps.roi_pitch_deg = r->number_or("pitch_deg", ps.roi_pitch_deg);
            ps.roi_hfov_deg = r->number_or("hfov_deg", ps.roi_hfov_deg);
        }
        if (const Json* s = v->find("stabilization")) c.view.stabilization = s->bool_or("enabled", c.view.stabilization);
    }
    if (const Json* i = root.find("inference")) {
        c.inference.enabled = i->bool_or("enabled", c.inference.enabled);
        c.inference.backend = i->string_or("backend", c.inference.backend);
        c.inference.source = i->string_or("source", c.inference.source);
        c.inference.arch = i->string_or("arch", c.inference.arch);
        c.inference.model_path = i->string_or("model_path", c.inference.model_path);
        c.inference.input_size = static_cast<int>(i->number_or("input_size", c.inference.input_size));
        c.inference.confidence_threshold = i->number_or("confidence_threshold", c.inference.confidence_threshold);
        c.inference.nms_threshold = i->number_or("nms_threshold", c.inference.nms_threshold);
        c.inference.max_detections = static_cast<int>(i->number_or("max_detections", c.inference.max_detections));
        c.inference.precision = i->string_or("precision", c.inference.precision);
        c.inference.valid_range_m = i->number_or("valid_range_m", c.inference.valid_range_m);
        if (const Json* cl = i->find("classes"); cl && cl->is_array()) {
            c.inference.classes.clear();
            for (const Json& n : cl->items()) if (n.is_string()) c.inference.classes.push_back(n.as_string());
        }
    }
    if (const Json* b = root.find("blending")) {
        c.blend.enabled = b->bool_or("enabled", c.blend.enabled);
        c.blend.ir_alpha = b->number_or("ir_alpha", c.blend.ir_alpha);
    }
    if (const Json* a = root.find("ar_overlay")) {
        c.ar.enabled = a->bool_or("enabled", c.ar.enabled);
        c.ar.show_ais = a->bool_or("show_ais", c.ar.show_ais);
        c.ar.show_detections = a->bool_or("show_detections", c.ar.show_detections);
        c.ar.show_heading = a->bool_or("show_heading", c.ar.show_heading);
        c.ar.show_collision_zone = a->bool_or("show_collision_zone", c.ar.show_collision_zone);
        c.ar.collision_zone_meters = a->number_or("collision_zone_meters", c.ar.collision_zone_meters);
    }
    if (const Json* r = root.find("recording")) {
        c.recording.directory = r->string_or("directory", c.recording.directory);
        c.recording.prefix = r->string_or("prefix", c.recording.prefix);
        c.recording.extension = r->string_or("extension", c.recording.extension);
        c.recording.min_free_mb = static_cast<std::uint64_t>(r->number_or("min_free_mb", static_cast<double>(c.recording.min_free_mb)));
        c.recording.fps = r->number_or("fps", c.recording.fps);
    }
    if (const Json* l = root.find("logging")) {
        c.logging.directory = l->string_or("directory", c.logging.directory);
        c.logging.level = l->string_or("level", c.logging.level);
        c.logging.console = l->bool_or("console", c.logging.console);
    }

    cfg = std::move(c);
    return true;
}

bool load_app_config_file(const std::string& path, AppConfig& cfg, std::string* err) {
    bool ok = false;
    const std::string text = read_file(path, ok);
    if (!ok) {
        if (err) *err = "cannot open config file: " + path;
        return false;
    }
    return parse_app_config(text, cfg, err);
}

bool load_fisheye_calibration_file(const std::string& path, geo::FisheyeModel& out, std::string* err) {
    bool ok = false;
    const std::string text = read_file(path, ok);
    if (!ok) {
        if (err) *err = "cannot open calibration file: " + path;
        return false;
    }
    Json j;
    std::string perr;
    if (!Json::try_parse(text, j, &perr)) {
        if (err) *err = perr;
        return false;
    }
    if (!j.is_object() || !j.find("fx")) {
        if (err) *err = "not a fisheye calibration file (missing fx)";
        return false;
    }
    read_fisheye(j, out);
    return true;
}

std::string fisheye_calibration_to_json(const geo::FisheyeModel& m, double rms_px, int num_images) {
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
        "{\n"
        "  \"model\": \"kannala_brandt\",\n"
        "  \"image_width\": %d,\n"
        "  \"image_height\": %d,\n"
        "  \"fx\": %.9g,\n  \"fy\": %.9g,\n  \"cx\": %.9g,\n  \"cy\": %.9g,\n"
        "  \"k1\": %.9g,\n  \"k2\": %.9g,\n  \"k3\": %.9g,\n  \"k4\": %.9g,\n"
        "  \"fov_deg\": %.6g,\n"
        "  \"rms_reprojection_px\": %.6g,\n"
        "  \"num_images\": %d\n"
        "}\n",
        m.width, m.height, m.fx, m.fy, m.cx, m.cy, m.k1, m.k2, m.k3, m.k4, m.fov_deg, rms_px, num_images);
    return buf;
}

std::vector<std::string> validate(const AppConfig& c) {
    std::vector<std::string> e;
    if (c.eo.width == 0 || c.eo.height == 0) e.push_back("camera.eo: width/height must be > 0");
    if (c.output.width == 0 || c.output.height == 0) e.push_back("output: width/height must be > 0");
    if (c.ir.size() > 3) e.push_back("camera.ir: at most 3 IR cameras are supported");
    if (c.ir_intrinsics.size() != c.ir.size()) e.push_back("calibration.ir_intrinsics count must match camera.ir");
    if (c.eo_intrinsics.fx <= 0 || c.eo_intrinsics.fy <= 0) e.push_back("calibration.eo_intrinsics: fx/fy must be > 0");
    if (c.eo_intrinsics.fov_deg <= 0 || c.eo_intrinsics.fov_deg > 360) e.push_back("calibration.eo_intrinsics.fov_deg out of range");
    if (c.mount.height_m <= 0) e.push_back("calibration.eo_mount.height_m must be > 0");
    if (c.sync.tolerance_us <= 0) e.push_back("sync.tolerance_ms must be > 0");
    if (c.sync.stale_timeout_us <= c.sync.tolerance_us) e.push_back("sync.stale_timeout_ms must exceed tolerance_ms");
    if (c.view.presets.topview_radius_m <= 0) e.push_back("view.topview.radius_m must be > 0");
    if (c.view.presets.topview_vfov_deg <= 0 || c.view.presets.topview_vfov_deg >= 170) e.push_back("view.topview.vfov_deg must be in (0,170)");
    if (c.view.presets.forward_hfov_deg <= 0 || c.view.presets.forward_hfov_deg >= 170) e.push_back("view.forward.hfov_deg must be in (0,170)");
    if (c.pip.windows.size() < c.ir.size() && c.pip.enabled) e.push_back("pip.windows: need one window per IR camera");
    if (c.inference.enabled && c.inference.classes.empty()) e.push_back("inference.classes must not be empty");
    if (c.inference.valid_range_m <= 0) e.push_back("inference.valid_range_m must be > 0");
    return e;
}

} // namespace avm::config
