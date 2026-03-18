#include "httplib.h"
#include <opencv2/opencv.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <algorithm>
#include <unordered_map>
#include <deque>
#include <thread>
#include <condition_variable>
#include <functional>
#include <cstdlib>
#include <filesystem>

// Project headers
#include "yolo.h"
#include "restoration.h"
#include "ModelTypes.h"
#include "tracker.h"
#include "colorizer.h" 

// ==============================
// COCO labels (YOLOv8n default)
// ==============================
static const std::vector<std::string> COCO_CLASSES = {
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
};

// ==============================
// Helpers
// ==============================
static std::string readFileText(const std::string& path) {
    std::ifstream t(path, std::ios::in | std::ios::binary);
    if (!t.is_open()) return "";
    std::stringstream buffer;
    buffer << t.rdbuf();
    return buffer.str();
}

static std::string normalizeSlashes(std::string p) {
    std::replace(p.begin(), p.end(), '\\', '/');
    return p;
}

static std::string safeLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c){ return (char)std::tolower(c); });
    return s;
}

static void setCorsHeaders(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type");
}

static void setDownloadHeaders(httplib::Response& res, const std::string& filename) {
    res.set_header("Content-Disposition", "attachment; filename=\"" + filename + "\"");
    res.set_header("Cache-Control", "no-store");
}

static std::string getField(const httplib::Request& req,
                            const std::string& key,
                            const std::string& def = "")
{
    auto it = req.form.fields.find(key);
    if (it != req.form.fields.end()) {
        return it->second.content;
    }
    if (req.has_param(key)) {
        return req.get_param_value(key);
    }
    return def;
}

static float getFloatField(const httplib::Request& req, const std::string& key, float def) {
    try {
        auto s = getField(req, key, "");
        if (s.empty()) return def;
        return std::stof(s);
    } catch (...) { return def; }
}

static int getIntField(const httplib::Request& req, const std::string& key, int def) {
    try {
        auto s = getField(req, key, "");
        if (s.empty()) return def;
        return std::stoi(s);
    } catch (...) { return def; }
}

static std::vector<int> parseClassesCSV(const std::string& s) {
    std::vector<int> out;
    std::string cur;
    for (char c : s) {
        if (c == ',' || c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            if (!cur.empty()) {
                try { out.push_back(std::stoi(cur)); } catch (...) {}
                cur.clear();
            }
        } else {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) {
        try { out.push_back(std::stoi(cur)); } catch (...) {}
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

static bool encodeImage(const cv::Mat& img, const std::string& fmtLower,
                        std::string& outBytes, std::string& outMime)
{
    std::vector<uchar> buf;
    std::vector<int> params;

    std::string ext = ".jpg";
    outMime = "image/jpeg";
    if (fmtLower == "png") {
        ext = ".png";
        outMime = "image/png";
        params = { cv::IMWRITE_PNG_COMPRESSION, 3 };
    } else {
        params = { cv::IMWRITE_JPEG_QUALITY, 95 };
    }

    if (!cv::imencode(ext, img, buf, params)) return false;
    outBytes.assign((const char*)buf.data(), (const char*)buf.data() + buf.size());
    return true;
}

static int pickMp4Fourcc() {
    return cv::VideoWriter::fourcc('m','p','4','v');
}

// ==============================
// ffmpeg transcode (H264 + faststart)
// ==============================
static std::string FFMPEG_EXE = R"(H:\libraries\ffmpeg-2026-01-14-git-6c878f8b82-full_build\ffmpeg-2026-01-14-git-6c878f8b82-full_build\bin\ffmpeg.exe)";

static bool transcodeToWebMp4(const std::string& inMp4, const std::string& outMp4) {
    std::ostringstream cmd;

    
    cmd << "cmd /C \"\""
        << FFMPEG_EXE
        << "\" -y -hide_banner -loglevel error"
        << " -i \"" << inMp4 << "\""
        << " -c:v libx264 -pix_fmt yuv420p -movflags +faststart"
        << " \"" << outMp4 << "\"\"";

    std::cout << "[ffmpeg] cmd = " << cmd.str() << std::endl;

    int code = std::system(cmd.str().c_str());
    std::cout << "[ffmpeg] exitCode = " << code << std::endl;
    return code == 0;
}

// ==============================
// Visualizer (boxes + masks + trails)
// ==============================
class Visualizer {
public:
    static cv::Scalar getColor(int id) {
        static std::vector<cv::Scalar> palette;
        if (palette.empty()) {
            cv::RNG rng(12345);
            for (int i = 0; i < 256; ++i) {
                palette.emplace_back(rng.uniform(60,255), rng.uniform(60,255), rng.uniform(60,255));
            }
        }
        if (id < 0) id = 0;
        return palette[id % palette.size()];
    }

    static void drawTrails(cv::Mat& img,
                           const std::unordered_map<int, std::deque<cv::Point>>& trails,
                           bool fixedColor)
    {
        for (const auto& kv : trails) {
            int id = kv.first;
            const auto& pts = kv.second;
            if (pts.size() < 2) continue;

            cv::Scalar c = fixedColor ? getColor(id) : cv::Scalar(0, 255, 255);
            for (size_t i = 1; i < pts.size(); ++i) {
                cv::line(img, pts[i-1], pts[i], c, 2, cv::LINE_AA);
            }
        }
    }

    static void draw(cv::Mat& img,
                     const ModelOutput& result,
                     const std::vector<std::string>& classNames,
                     bool fixedColor)
    {
        cv::Mat overlay = img.clone();

        for (size_t i = 0; i < result.boxes.size(); i++) {
            int classId = (i < result.classIds.size()) ? result.classIds[i] : -1;
            int trackId = (i < result.trackIds.size()) ? result.trackIds[i] : -1;

            cv::Rect box = result.boxes[i] & cv::Rect(0,0,img.cols,img.rows);
            if (box.width <= 0 || box.height <= 0) continue;

            int colorKey = fixedColor ? (trackId != -1 ? trackId : std::max(classId,0)) : std::max(classId,0);
            cv::Scalar color = getColor(colorKey);

            if (i < result.masks.size() && !result.masks[i].empty()) {
                cv::Mat mask = result.masks[i];
                cv::Rect validBox = box & cv::Rect(0,0,img.cols,img.rows);
                if (validBox.width > 0 && validBox.height > 0) {
                    cv::Mat maskROI = mask(cv::Rect(0,0, validBox.width, validBox.height));
                    cv::Mat imgROI = overlay(validBox);
                    imgROI.setTo(color, maskROI);
                }
            }

            cv::rectangle(img, box, color, 2, cv::LINE_AA);

            std::string label = "Obj";
            if (classId >= 0 && classId < (int)classNames.size()) label = classNames[classId];
            if (trackId != -1) label += " #" + std::to_string(trackId);
            if (i < result.confidences.size()) {
                label += " " + std::to_string((int)(result.confidences[i]*100)) + "%";
            }

            int baseLine = 0;
            double fontScale = 0.45;
            cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_DUPLEX, fontScale, 1, &baseLine);
            int y = std::max(box.y, ts.height + 10);
            cv::Rect bg(box.x, y - ts.height - 10, ts.width + 10, ts.height + 10);
            bg &= cv::Rect(0,0,img.cols,img.rows);
            cv::rectangle(img, bg, color, cv::FILLED);
            cv::putText(img, label, cv::Point(bg.x + 5, bg.y + bg.height - 6),
                        cv::FONT_HERSHEY_DUPLEX, fontScale, cv::Scalar(255,255,255), 1, cv::LINE_AA);
        }

        cv::addWeighted(overlay, 0.35, img, 0.65, 0, img);
    }
};

// ==============================
// Model Manager
// ==============================
enum class LoadedKind { None, Yolo, Restore, Colorizer };

struct LoadedModel {
    LoadedKind kind = LoadedKind::None;
    std::string key;
    std::unique_ptr<BaseModel> ptr;
};

class ModelManager {
public:
    BaseModel* getOrLoad(const std::string& key,
                         LoadedKind kind,
                         const std::function<std::unique_ptr<BaseModel>()>& factory)
    {
        std::lock_guard<std::mutex> lk(mu_);

        if (loaded_.ptr && loaded_.key == key) {
            return loaded_.ptr.get();
        }

        loaded_.ptr.reset();
        loaded_.kind = LoadedKind::None;
        loaded_.key.clear();

        loaded_.ptr = factory();
        loaded_.key = key;
        loaded_.kind = kind;
        return loaded_.ptr.get();
    }

    void unload() {
        std::lock_guard<std::mutex> lk(mu_);
        loaded_.ptr.reset();
        loaded_.kind = LoadedKind::None;
        loaded_.key.clear();
        std::cout << "[ModelManager] Unloaded model (cache cleared)" << std::endl;
    }

private:
    std::mutex mu_;
    LoadedModel loaded_;
};

// ==============================
// Model paths
// ==============================
static const std::string WEIGHTS_DIR = "H:/QtProjects/Unified_IP/weights/";

static const std::string PATH_DEHAZE      = WEIGHTS_DIR + "dehazeformer-s-final.onnx";
static const std::string PATH_DERAIN      = WEIGHTS_DIR + "restormer_rain_raw.onnx";
static const std::string PATH_DESNOW      = WEIGHTS_DIR + "restormer_rain_raw.onnx";
static const std::string PATH_UNDERWATER  = WEIGHTS_DIR + "funie_gan_sim.onnx";
static const std::string PATH_SUPERRES    = WEIGHTS_DIR + "realesrgan-x4plus.onnx";

// ��?新增：黑白上色和老照片修��?
static const std::string PATH_COLORIZE    = WEIGHTS_DIR + "deoldify_artistic_512.onnx";
static const std::string PATH_OLD_PHOTO   = WEIGHTS_DIR + "realesrgan-x4plus.onnx";

static const std::string PATH_YOLO_DET    = WEIGHTS_DIR + "yolov8n.onnx";
static const std::string PATH_YOLO_SEG    = WEIGHTS_DIR + "yolov8n-seg.onnx";

// Style Transfer
static const std::string PATH_STYLE_CANDY  = WEIGHTS_DIR + "style_candy.onnx";
static const std::string PATH_STYLE_MOSAIC = WEIGHTS_DIR + "style_mosaic.onnx";
static const std::string PATH_STYLE_RAIN   = WEIGHTS_DIR + "style_rain_princess.onnx";
static const std::string PATH_STYLE_UDNIE  = WEIGHTS_DIR + "style_udnie.onnx";

static std::string resolveStylePath(const std::string& model_id, const std::string& onnx) {
    const std::string id = safeLower(model_id);
    const std::string ox = safeLower(onnx);

    if (id.find("candy") != std::string::npos || ox.find("style_candy") != std::string::npos) return PATH_STYLE_CANDY;
    if (id.find("mosaic") != std::string::npos || ox.find("style_mosaic") != std::string::npos) return PATH_STYLE_MOSAIC;
    if (id.find("rain") != std::string::npos || ox.find("style_rain_princess") != std::string::npos) return PATH_STYLE_RAIN;
    if (id.find("udnie") != std::string::npos || ox.find("style_udnie") != std::string::npos) return PATH_STYLE_UDNIE;

    return PATH_STYLE_CANDY;
}

static std::string resolveRealModelPath(const std::string& mode,
                                        const std::string& model_id,
                                        const std::string& onnx)
{
    // ��?新增映射
    if (mode == "colorization") return PATH_COLORIZE;
    if (mode == "old_photo")    return PATH_OLD_PHOTO;

    if (mode == "superres")   return PATH_SUPERRES;
    if (mode == "dehaze")     return PATH_DEHAZE;
    if (mode == "derain")     return PATH_DERAIN;
    if (mode == "desnow")     return PATH_DESNOW;
    if (mode == "underwater") return PATH_UNDERWATER;

    if (mode == "style")      return resolveStylePath(model_id, onnx);

    if (mode == "detect")     return PATH_YOLO_DET;
    if (mode == "segment")    return PATH_YOLO_SEG;
    if (mode == "track")      return PATH_YOLO_DET;

    return "";
}

// ==============================
// Restoration config
// ==============================
static RestorationConfig makeRestorationConfigForMode(const std::string& mode) {
    RestorationConfig cfg;

    cfg.is_bgr_input = false;
    cfg.input_0_1 = true;
    cfg.output_0_1 = true;
    cfg.use_tanh_range = false;

    if (mode == "superres") {
        cfg.tile_size = 256;
        cfg.tile_overlap = 32;
        cfg.scale = 4;
        return cfg;
    }

    if (mode == "old_photo") {
        cfg.tile_size = 256;   
        cfg.tile_overlap = 32;
        cfg.scale = 4;         
        return cfg;
    }

    if (mode == "dehaze" || mode == "underwater") {
        cfg.use_tanh_range = true;
        cfg.input_0_1 = false;
        cfg.output_0_1 = false;
        cfg.tile_size = 0;
        cfg.tile_overlap = 32;
        cfg.scale = 1;
        return cfg;
    }

    if (mode == "style") {
        cfg.use_tanh_range = false;
        cfg.input_0_1 = false;
        cfg.output_0_1 = false;
        cfg.tile_size = 0;
        cfg.tile_overlap = 32;
        cfg.scale = 1;
        return cfg;
    }

    cfg.tile_size = 0;
    cfg.tile_overlap = 32;
    cfg.scale = 1;
    return cfg;
}

// ==============================
// Queue
// ==============================
template<typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t cap) : cap_(cap) {}

    void push(T v) {
        std::unique_lock<std::mutex> lk(m_);
        cv_full_.wait(lk, [&]{ return q_.size() < cap_ || closed_; });
        if (closed_) return;
        q_.push_back(std::move(v));
        cv_empty_.notify_one();
    }

    bool pop(T& out) {
        std::unique_lock<std::mutex> lk(m_);
        cv_empty_.wait(lk, [&]{ return !q_.empty() || closed_; });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop_front();
        cv_full_.notify_one();
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lk(m_);
        closed_ = true;
        cv_empty_.notify_all();
        cv_full_.notify_all();
    }

private:
    size_t cap_;
    std::deque<T> q_;
    std::mutex m_;
    std::condition_variable cv_empty_;
    std::condition_variable cv_full_;
    bool closed_ = false;
};

// ==============================
// main
// ==============================
int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    ModelManager modelManager;
    httplib::Server svr;

    svr.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        setCorsHeaders(res);
        if (req.method == "OPTIONS") {
            res.status = 200;
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        std::string html = readFileText("index.html");
        if (html.empty()) {
            res.status = 404;
            res.set_content("index.html not found", "text/plain; charset=utf-8");
            return;
        }
        res.set_content(html, "text/html; charset=utf-8");
    });

    svr.Post("/unload_models", [&](const httplib::Request&, httplib::Response& res) {
        modelManager.unload();
        res.set_content("OK", "text/plain; charset=utf-8");
    });

    // =========================
    // POST /process_image
    // =========================
    svr.Post("/process_image", [&](const httplib::Request& req, httplib::Response& res) {
        setCorsHeaders(res);

        auto fit = req.form.files.find("file");
        if (fit == req.form.files.end()) {
            res.status = 400;
            res.set_content("missing multipart field: file", "text/plain; charset=utf-8");
            return;
        }
        const auto& file = fit->second;

        const std::string mode     = safeLower(getField(req, "mode", ""));
        const std::string model_id = getField(req, "model_id", "");
        const std::string onnx     = getField(req, "onnx", "");
        const std::string img_fmt  = safeLower(getField(req, "img_format", "auto"));

        if (mode.empty()) {
            res.status = 400;
            res.set_content("missing field: mode", "text/plain; charset=utf-8");
            return;
        }

        const std::string modelPath = resolveRealModelPath(mode, model_id, onnx);
        if (modelPath.empty()) {
            res.status = 400;
            res.set_content("cannot resolve model path for mode=" + mode, "text/plain; charset=utf-8");
            return;
        }

        std::vector<uchar> buf(file.content.begin(), file.content.end());
        cv::Mat img = cv::imdecode(buf, cv::IMREAD_COLOR);
        if (img.empty()) {
            res.status = 400;
            res.set_content("invalid image content", "text/plain; charset=utf-8");
            return;
        }

        const bool keep_model = (getField(req, "keep_model", "0") == "1");

        try {
            cv::Mat outImg;

            // ---- YOLO branch ----
            if (mode == "detect" || mode == "segment" || mode == "track") {
                YoloOptions opt;
                opt.conf = getFloatField(req, "conf", 0.25f);
                opt.iou  = getFloatField(req, "iou", 0.45f);
                opt.max_det = getIntField(req, "max_det", 100);
                opt.classes = parseClassesCSV(getField(req, "classes", ""));

                const std::string key = "YOLO::" + normalizeSlashes(modelPath);
                BaseModel* bm = modelManager.getOrLoad(
                    key, LoadedKind::Yolo,
                    [&]() -> std::unique_ptr<BaseModel> {
                        std::cout << "[Load] YOLO: " << modelPath << std::endl;
                        return std::make_unique<YoloModel>(modelPath, COCO_CLASSES);
                    }
                    );
                auto* yolo = dynamic_cast<YoloModel*>(bm);
                ModelOutput out = yolo->predict(img, opt);

                if (mode == "track") {
                    std::string trackerType = safeLower(getField(req, "tracker", "bytetrack"));
                    auto tracker = CreateTracker(trackerType);
                    out.trackIds = tracker->update(out.boxes, out.confidences);
                }
                bool fixedColor = (getField(req, "fixed_color", "1") == "1");
                cv::Mat vis = img.clone();
                Visualizer::draw(vis, out, COCO_CLASSES, fixedColor);
                outImg = vis;
            }
            else if (mode == "colorization") {
                const std::string key = "COLOR::" + normalizeSlashes(modelPath);
                BaseModel* bm = modelManager.getOrLoad(
                    key, LoadedKind::Colorizer,
                    [&]() -> std::unique_ptr<BaseModel> {
                        std::cout << "[Load] Colorizer: " << modelPath << std::endl;
                        return std::make_unique<Colorizer>(modelPath);
                    }
                    );
                auto* col = dynamic_cast<Colorizer*>(bm);
                if (!col) throw std::runtime_error("loaded model is not Colorizer");

                ModelOutput out = col->predict(img);
                if (out.imageResult.empty()) throw std::runtime_error("colorization failed");
                outImg = out.imageResult;
            }
            // ---- Restoration branch (Old Photo 走这��? ----
            else {
                const int sr_scale = getIntField(req, "sr_scale", 4);
                const std::string key = "RESTORE::" + mode + "::" + normalizeSlashes(modelPath);
                BaseModel* bm = modelManager.getOrLoad(
                    key, LoadedKind::Restore,
                    [&]() -> std::unique_ptr<BaseModel> {
                        std::cout << "[Load] Restoration: " << modelPath
                                  << " mode=" << mode << std::endl;
                        RestorationConfig cfg = makeRestorationConfigForMode(mode);
                        return std::make_unique<Restoration>(modelPath, cfg);
                    }
                    );
                auto* r = dynamic_cast<Restoration*>(bm);
                if (!r) throw std::runtime_error("loaded model is not Restoration");

                ModelOutput out = r->predict(img);
                if (out.imageResult.empty()) throw std::runtime_error("restoration output empty");

                if (mode == "superres" && sr_scale == 2) {
                    cv::resize(out.imageResult, outImg,
                               cv::Size(img.cols * 2, img.rows * 2),
                               0, 0, cv::INTER_CUBIC);
                } else {
                    outImg = out.imageResult;
                }
            }

            std::string bytes, mime;
            if (!encodeImage(outImg, img_fmt, bytes, mime)) {
                res.status = 500;
                res.set_content("failed to encode output image", "text/plain; charset=utf-8");
                return;
            }

            std::string ext = (mime == "image/png") ? "png" : "jpg";
            std::string fname = "AI_Result_" + mode + "." + ext;
            if (!model_id.empty()) fname = "AI_Result_" + mode + "_" + model_id + "." + ext;

            setDownloadHeaders(res, fname);
            res.set_content(bytes, mime.c_str());
            res.status = 200;

        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(std::string("error: ") + e.what(), "text/plain; charset=utf-8");
        }
        if (!keep_model) modelManager.unload();
    });

    // =========================
    // POST /process_video
    // =========================
    svr.Post("/process_video", [&](const httplib::Request& req, httplib::Response& res) {
        setCorsHeaders(res);

        auto fit = req.form.files.find("file");
        if (fit == req.form.files.end()) {
            res.status = 400;
            res.set_content("missing multipart field: file", "text/plain; charset=utf-8");
            return;
        }
        const auto& file = fit->second;

        const std::string mode     = safeLower(getField(req, "mode", ""));
        const std::string model_id = getField(req, "model_id", "");
        const std::string onnx     = getField(req, "onnx", "");

        if (mode.empty()) {
            res.status = 400;
            res.set_content("missing field: mode", "text/plain; charset=utf-8");
            return;
        }

        const std::string modelPath = resolveRealModelPath(mode, model_id, onnx);
        if (modelPath.empty()) {
            res.status = 400;
            res.set_content("cannot resolve model path for mode=" + mode, "text/plain; charset=utf-8");
            return;
        }

        const bool keep_model = (getField(req, "keep_model", "0") == "1");
        const std::string inF   = "temp_in.mp4";
        const std::string outF  = "temp_out.mp4";
        const std::string outF2 = "temp_out_web.mp4";
        {
            std::ofstream ofs(inF, std::ios::binary);
            if (!ofs.is_open()) {
                res.status = 500;
                res.set_content("failed to open temp file for input video", "text/plain; charset=utf-8");
                return;
            }
            ofs.write(file.content.data(), (std::streamsize)file.content.size());
        }

        cv::VideoCapture cap(inF);
        if (!cap.isOpened()) {
            res.status = 400;
            res.set_content("failed to open uploaded video", "text/plain; charset=utf-8");
            return;
        }

        const double fpsIn = cap.get(cv::CAP_PROP_FPS);
        const double fps = (fpsIn > 0.1 ? fpsIn : 25.0);
        const int fourcc = pickMp4Fourcc();

        struct FrameItem { int idx; cv::Mat frame; };
        struct OutItem   { int idx; cv::Mat frame; };

        BoundedQueue<FrameItem> qDecode(8);
        BoundedQueue<OutItem>   qEncode(8);

        struct TrailState { std::deque<cv::Point> pts; int lastSeen = 0; };
        std::unordered_map<int, TrailState> trails;
        const int TRAIL_LEN = 30;
        const int TRAIL_TTL = 15;
        int frameNo = 0;

        const bool drawTrail  = (getField(req, "draw_trail", "0") == "1");
        const bool fixedColor = (getField(req, "fixed_color", "1") == "1");

        cv::VideoWriter writer;

        try {
            std::unique_ptr<ITracker> tracker;
            YoloOptions yopt;

            BaseModel* bm = nullptr;
            YoloModel* yolo = nullptr;
            Restoration* rest = nullptr;
            Colorizer* col = nullptr; 

            const bool isVision = (mode == "detect" || mode == "segment" || mode == "track");

            if (isVision) {
                yopt.conf = getFloatField(req, "conf", 0.25f);
                yopt.iou  = getFloatField(req, "iou", 0.45f);
                yopt.max_det = getIntField(req, "max_det", 100);
                yopt.classes = parseClassesCSV(getField(req, "classes", ""));

                const std::string key = "YOLO::" + normalizeSlashes(modelPath);
                bm = modelManager.getOrLoad(
                    key, LoadedKind::Yolo,
                    [&]() -> std::unique_ptr<BaseModel> {
                        std::cout << "[Load] YOLO: " << modelPath << std::endl;
                        return std::make_unique<YoloModel>(modelPath, COCO_CLASSES);
                    }
                    );
                yolo = dynamic_cast<YoloModel*>(bm);
                if (mode == "track") {
                    std::string trackerType = safeLower(getField(req, "tracker", "bytetrack"));
                    tracker = CreateTracker(trackerType);
                }
            }
            else if (mode == "colorization") { // ��?新增
                const std::string key = "COLOR::" + normalizeSlashes(modelPath);
                bm = modelManager.getOrLoad(
                    key, LoadedKind::Colorizer,
                    [&]() -> std::unique_ptr<BaseModel> {
                        std::cout << "[Load] Colorizer: " << modelPath << std::endl;
                        return std::make_unique<Colorizer>(modelPath);
                    }
                    );
                col = dynamic_cast<Colorizer*>(bm);
            }
            else { // Restoration (Old Photo, Dehaze, SR, Style...)
                const std::string key = "RESTORE::" + mode + "::" + normalizeSlashes(modelPath);
                bm = modelManager.getOrLoad(
                    key, LoadedKind::Restore,
                    [&]() -> std::unique_ptr<BaseModel> {
                        std::cout << "[Load] Restoration: " << modelPath << std::endl;
                        RestorationConfig cfg = makeRestorationConfigForMode(mode);
                        return std::make_unique<Restoration>(modelPath, cfg);
                    }
                    );
                rest = dynamic_cast<Restoration*>(bm);
            }

            // Decode thread
            std::thread thDec([&](){
                cv::Mat frame;
                int idx = 0;
                while (cap.read(frame)) {
                    if (frame.empty()) break;
                    qDecode.push(FrameItem{idx++, frame.clone()});
                }
                qDecode.close();
            });

            // Infer thread
            std::thread thInf([&](){
                FrameItem it;
                while (qDecode.pop(it)) {
                    frameNo++;
                    cv::Mat outFrame = it.frame;

                    if (yolo) {
                        ModelOutput out = yolo->predict(outFrame, yopt);
                        if (mode == "track" && tracker) {
                            out.trackIds = tracker->update(out.boxes, out.confidences);
                            if (drawTrail) {
                                for (size_t i = 0; i < out.boxes.size() && i < out.trackIds.size(); ++i) {
                                    int id = out.trackIds[i];
                                    if (id < 0) continue;
                                    const auto& b = out.boxes[i];
                                    cv::Point c(b.x + b.width/2, b.y + b.height/2);
                                    auto& st = trails[id];
                                    st.lastSeen = frameNo;
                                    st.pts.push_back(c);
                                    if ((int)st.pts.size() > TRAIL_LEN) st.pts.pop_front();
                                }
                                for (auto iter = trails.begin(); iter != trails.end(); ) {
                                    if (frameNo - iter->second.lastSeen > TRAIL_TTL) iter = trails.erase(iter);
                                    else ++iter;
                                }
                                std::unordered_map<int, std::deque<cv::Point>> trailsForDraw;
                                for (auto& kv : trails) trailsForDraw[kv.first] = kv.second.pts;
                                Visualizer::drawTrails(outFrame, trailsForDraw, fixedColor);
                            }
                        }
                        Visualizer::draw(outFrame, out, COCO_CLASSES, fixedColor);
                    }
                    else if (col) { 
                        ModelOutput out = col->predict(outFrame);
                        if (!out.imageResult.empty()) outFrame = out.imageResult;
                    }
                    else if (rest) {
                        ModelOutput out = rest->predict(outFrame);
                        if (!out.imageResult.empty()) outFrame = out.imageResult;
                    }

                    qEncode.push(OutItem{it.idx, outFrame});
                }
                qEncode.close();
            });

            // Encode thread
            std::thread thEnc([&](){
                OutItem o;
                bool opened = false;
                while (qEncode.pop(o)) {
                    if (!opened) {
                        if (!writer.open(outF, fourcc, fps, cv::Size(o.frame.cols, o.frame.rows), true)) {
                            std::cerr << "[VideoWriter] open failed" << std::endl;
                            break;
                        }
                        opened = true;
                    }
                    writer.write(o.frame);
                }
            });

            thDec.join();
            thInf.join();
            thEnc.join();

            cap.release();
            if (writer.isOpened()) writer.release();

            std::string finalOut = outF;
            if (transcodeToWebMp4(outF, outF2)) {
                finalOut = outF2;
            }

            std::ifstream ifs(finalOut, std::ios::binary);
            if (!ifs.is_open()) throw std::runtime_error("failed to read output video");

            std::stringstream ss;
            ss << ifs.rdbuf();
            const std::string videoData = ss.str();

            std::string fname = "AI_Result_" + mode + ".mp4";
            if (!model_id.empty()) fname = "AI_Result_" + mode + "_" + model_id + ".mp4";
            setDownloadHeaders(res, fname);

            res.set_content(videoData, "video/mp4");
            res.status = 200;

        } catch (const std::exception& e) {
            cap.release();
            if (writer.isOpened()) writer.release();
            res.status = 500;
            res.set_content(std::string("error: ") + e.what(), "text/plain; charset=utf-8");
        }
        if (!keep_model) modelManager.unload();
    });

    std::cout << "Server running at http://localhost:8080" << std::endl;
    svr.listen("0.0.0.0", 8080);
    return 0;
}





