#include "lut_builder.hpp"

#include <array>
#include <cstring>
#include <fstream>

namespace avm::geo {

double LutMap::valid_fraction() const {
    if (map_x.empty()) return 0.0;
    std::size_t n = 0;
    for (float x : map_x) if (x >= 0.0f) ++n;
    return static_cast<double>(n) / static_cast<double>(map_x.size());
}

LutMap build_view_lut(const FisheyeModel& model, const CameraPose& pose, const ViewParams& view,
                      int out_w, int out_h, int src_w, int src_h) {
    LutMap lut;
    lut.width = out_w; lut.height = out_h;
    lut.src_width = src_w; lut.src_height = src_h;
    lut.map_x.assign(lut.pixel_count(), -1.0f);
    lut.map_y.assign(lut.pixel_count(), -1.0f);

    // Double precision on the host; the per-frame CUDA path uses float.
    const ViewMapper<double> mapper = make_view_mapper<double>(model, pose, view, out_w, out_h);
    for (int v = 0; v < out_h; ++v) {
        for (int u = 0; u < out_w; ++u) {
            double su, sv;
            if (!mapper.map(u, v, su, sv)) continue;
            if (su < 0.0 || sv < 0.0 || su > src_w - 1 || sv > src_h - 1) continue;
            const std::size_t i = static_cast<std::size_t>(v) * out_w + u;
            lut.map_x[i] = static_cast<float>(su);
            lut.map_y[i] = static_cast<float>(sv);
        }
    }
    return lut;
}

// ─── CRC32 (IEEE 802.3) ─────────────────────────────────────────────────────

static const std::array<std::uint32_t, 256>& crc_table() {
    static const std::array<std::uint32_t, 256> table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            std::uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[i] = c;
        }
        return t;
    }();
    return table;
}

std::uint32_t crc32(const void* data, std::size_t size, std::uint32_t seed) {
    const auto& t = crc_table();
    const auto* p = static_cast<const std::uint8_t*>(data);
    std::uint32_t c = ~seed;
    for (std::size_t i = 0; i < size; ++i) c = t[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return ~c;
}

// ─── file I/O ───────────────────────────────────────────────────────────────

static constexpr char kMagic[8] = {'A', 'V', 'M', 'L', 'U', 'T', '0', '1'};
static constexpr std::uint32_t kVersion = 1;

static void set_err(std::string* err, const std::string& m) { if (err) *err = m; }

bool save_lut(const std::string& path, const LutMap& lut, const std::string& metadata,
              std::string* err) {
    if (lut.map_x.size() != lut.pixel_count() || lut.map_y.size() != lut.pixel_count()) {
        set_err(err, "LUT arrays do not match width*height");
        return false;
    }
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) { set_err(err, "cannot open for writing: " + path); return false; }

    std::uint32_t crc = 0;
    auto put = [&](const void* p, std::size_t n) {
        f.write(static_cast<const char*>(p), static_cast<std::streamsize>(n));
        crc = crc32(p, n, crc);
    };
    const std::uint32_t hdr[6] = {kVersion,
                                  static_cast<std::uint32_t>(lut.width),
                                  static_cast<std::uint32_t>(lut.height),
                                  static_cast<std::uint32_t>(lut.src_width),
                                  static_cast<std::uint32_t>(lut.src_height),
                                  static_cast<std::uint32_t>(metadata.size())};
    put(kMagic, sizeof(kMagic));
    put(hdr, sizeof(hdr));
    put(metadata.data(), metadata.size());
    put(lut.map_x.data(), lut.map_x.size() * sizeof(float));
    put(lut.map_y.data(), lut.map_y.size() * sizeof(float));
    f.write(reinterpret_cast<const char*>(&crc), sizeof(crc));
    if (!f) { set_err(err, "write failed: " + path); return false; }
    return true;
}

bool load_lut(const std::string& path, LutMap& lut, std::string* metadata, std::string* err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { set_err(err, "cannot open: " + path); return false; }

    std::uint32_t crc = 0;
    auto get = [&](void* p, std::size_t n) {
        f.read(static_cast<char*>(p), static_cast<std::streamsize>(n));
        if (!f) return false;
        crc = crc32(p, n, crc);
        return true;
    };

    char magic[8];
    if (!get(magic, sizeof(magic)) || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        set_err(err, "not an AVM LUT file (bad magic)");
        return false;
    }
    std::uint32_t hdr[6];
    if (!get(hdr, sizeof(hdr))) { set_err(err, "truncated header"); return false; }
    if (hdr[0] != kVersion) { set_err(err, "unsupported LUT version"); return false; }

    const std::uint64_t w = hdr[1], h = hdr[2];
    if (w == 0 || h == 0 || w > 16384 || h > 16384 || hdr[5] > (1u << 20)) {
        set_err(err, "implausible LUT dimensions");
        return false;
    }
    std::string meta(hdr[5], '\0');
    if (!meta.empty() && !get(meta.data(), meta.size())) { set_err(err, "truncated metadata"); return false; }

    LutMap out;
    out.width = static_cast<int>(w); out.height = static_cast<int>(h);
    out.src_width = static_cast<int>(hdr[3]); out.src_height = static_cast<int>(hdr[4]);
    out.map_x.resize(out.pixel_count());
    out.map_y.resize(out.pixel_count());
    if (!get(out.map_x.data(), out.map_x.size() * sizeof(float)) ||
        !get(out.map_y.data(), out.map_y.size() * sizeof(float))) {
        set_err(err, "truncated map data");
        return false;
    }
    std::uint32_t stored = 0;
    f.read(reinterpret_cast<char*>(&stored), sizeof(stored));
    if (!f || stored != crc) { set_err(err, "CRC mismatch (corrupt LUT file)"); return false; }

    lut = std::move(out);
    if (metadata) *metadata = std::move(meta);
    return true;
}

} // namespace avm::geo
