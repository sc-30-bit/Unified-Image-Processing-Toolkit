#ifndef TRACKER_H
#define TRACKER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <memory>
#include <string>
#include <algorithm>
#include <cmath>

static inline float IoU(const cv::Rect& a, const cv::Rect& b) {
    int inter = (a & b).area();
    int uni = a.area() + b.area() - inter;
    if (uni <= 0) return 0.f;
    return (float)inter / (float)uni;
}

// ============= TrackState =============
struct TrackState {
    int id = -1;
    int age = 0;
    int lost = 0;              
    cv::Rect box;
    cv::Point2f vel{0, 0};    
};


class ITracker {
public:
    virtual ~ITracker() = default;
    virtual void reset() = 0;


    virtual std::vector<int> update(const std::vector<cv::Rect>& dets,
                                    const std::vector<float>& scores) = 0;

    virtual std::string name() const = 0;
};

class ByteTrack : public ITracker {
public:
    ByteTrack(int maxLost = 30, float matchIou = 0.30f, float lowRatio = 0.50f)
        : maxLost_(maxLost), matchIou_(matchIou), lowRatio_(lowRatio) {}

    void reset() override {
        tracks_.clear();
        nextId_ = 1;
    }

    std::string name() const override { return "bytetrack"; }

    std::vector<int> update(const std::vector<cv::Rect>& dets,
                            const std::vector<float>& scores) override
    {
        std::vector<int> ids(dets.size(), -1);

        // 1) aging
        for (auto& t : tracks_) { t.age++; t.lost++; }

        // 2) split high / low
        float maxScore = 0.f;
        for (float s : scores) maxScore = std::max(maxScore, s);
        float lowThr = maxScore * lowRatio_;

        std::vector<int> hi, lo;
        for (int i = 0; i < (int)dets.size(); ++i) {
            if (scores[i] >= lowThr) hi.push_back(i);
            else lo.push_back(i);
        }

        // 3) match high then low
        greedyMatch(dets, hi, ids);
        greedyMatch(dets, lo, ids);

        // 4) create new tracks for unmatched dets
        for (int i = 0; i < (int)dets.size(); ++i) {
            if (ids[i] != -1) continue;
            TrackState t;
            t.id = nextId_++;
            t.age = 1;
            t.lost = 0;
            t.box = dets[i];
            tracks_.push_back(t);
            ids[i] = t.id;
        }

        // 5) remove lost too long
        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                     [&](const TrackState& t){ return t.lost > maxLost_; }),
                      tracks_.end());

        return ids;
    }

private:
    int maxLost_;
    float matchIou_;
    float lowRatio_;
    int nextId_ = 1;
    std::vector<TrackState> tracks_;

    void greedyMatch(const std::vector<cv::Rect>& dets,
                     const std::vector<int>& detIdx,
                     std::vector<int>& outIds)
    {
        std::vector<bool> usedTrack(tracks_.size(), false);

        for (int di : detIdx) {
            float best = 0.f;
            int bestTi = -1;
            for (int ti = 0; ti < (int)tracks_.size(); ++ti) {
                if (usedTrack[ti]) continue;
                float iou = IoU(tracks_[ti].box, dets[di]);
                if (iou > best) { best = iou; bestTi = ti; }
            }
            if (bestTi >= 0 && best >= matchIou_) {
                usedTrack[bestTi] = true;
                tracks_[bestTi].box = dets[di];
                tracks_[bestTi].lost = 0;
                outIds[di] = tracks_[bestTi].id;
            }
        }
    }
};

class DeepSORT : public ITracker {
public:
    DeepSORT(int maxLost = 20, float matchIou = 0.25f)
        : maxLost_(maxLost), matchIou_(matchIou) {}

    void reset() override {
        tracks_.clear();
        nextId_ = 1;
    }

    std::string name() const override { return "deepsort"; }

    std::vector<int> update(const std::vector<cv::Rect>& dets,
                            const std::vector<float>& scores) override
    {
        (void)scores;
        std::vector<int> ids(dets.size(), -1);

        for (auto& t : tracks_) { t.age++; t.lost++; }

        std::vector<bool> usedTrack(tracks_.size(), false);
        for (int i = 0; i < (int)dets.size(); ++i) {
            float best = 0.f;
            int bestTi = -1;
            for (int ti = 0; ti < (int)tracks_.size(); ++ti) {
                if (usedTrack[ti]) continue;
                float iou = IoU(tracks_[ti].box, dets[i]);
                if (iou > best) { best = iou; bestTi = ti; }
            }
            if (bestTi >= 0 && best >= matchIou_) {
                usedTrack[bestTi] = true;
                tracks_[bestTi].box = dets[i];
                tracks_[bestTi].lost = 0;
                ids[i] = tracks_[bestTi].id;
            }
        }

        for (int i = 0; i < (int)dets.size(); ++i) {
            if (ids[i] != -1) continue;
            TrackState t;
            t.id = nextId_++;
            t.age = 1;
            t.lost = 0;
            t.box = dets[i];
            tracks_.push_back(t);
            ids[i] = t.id;
        }

        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                     [&](const TrackState& t){ return t.lost > maxLost_; }),
                      tracks_.end());

        return ids;
    }

private:
    int maxLost_;
    float matchIou_;
    int nextId_ = 1;
    std::vector<TrackState> tracks_;
};

class OCSORT : public ITracker {
public:
    OCSORT(int maxLost = 30, float matchIou = 0.25f, float velWeight = 0.15f)
        : maxLost_(maxLost), matchIou_(matchIou), velWeight_(velWeight) {}

    void reset() override {
        tracks_.clear();
        nextId_ = 1;
    }

    std::string name() const override { return "ocsort"; }

    std::vector<int> update(const std::vector<cv::Rect>& dets,
                            const std::vector<float>& scores) override
    {
        (void)scores;
        std::vector<int> ids(dets.size(), -1);

        for (auto& t : tracks_) { t.age++; t.lost++; }

        std::vector<bool> usedTrack(tracks_.size(), false);

        for (int i = 0; i < (int)dets.size(); ++i) {
            cv::Point2f c = center(dets[i]);
            float bestScore = -1e9f;
            int bestTi = -1;

            for (int ti = 0; ti < (int)tracks_.size(); ++ti) {
                if (usedTrack[ti]) continue;

                float iou = IoU(tracks_[ti].box, dets[i]);
                if (iou < matchIou_) continue;

                cv::Point2f pc = center(tracks_[ti].box) + tracks_[ti].vel;
                float dist = (float)cv::norm(pc - c);

                float s = iou - velWeight_ * dist * 0.01f;
                if (s > bestScore) { bestScore = s; bestTi = ti; }
            }

            if (bestTi >= 0) {
                usedTrack[bestTi] = true;

                cv::Point2f oldc = center(tracks_[bestTi].box);
                tracks_[bestTi].vel = c - oldc;

                tracks_[bestTi].box = dets[i];
                tracks_[bestTi].lost = 0;
                ids[i] = tracks_[bestTi].id;
            }
        }

        for (int i = 0; i < (int)dets.size(); ++i) {
            if (ids[i] != -1) continue;
            TrackState t;
            t.id = nextId_++;
            t.age = 1;
            t.lost = 0;
            t.box = dets[i];
            t.vel = {0,0};
            tracks_.push_back(t);
            ids[i] = t.id;
        }

        tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                     [&](const TrackState& t){ return t.lost > maxLost_; }),
                      tracks_.end());

        return ids;
    }

private:
    int maxLost_;
    float matchIou_;
    float velWeight_;
    int nextId_ = 1;
    std::vector<TrackState> tracks_;

    static cv::Point2f center(const cv::Rect& r) {
        return {r.x + r.width * 0.5f, r.y + r.height * 0.5f};
    }
};

static inline std::unique_ptr<ITracker> CreateTracker(const std::string& type) {
    if (type == "deepsort") return std::make_unique<DeepSORT>();
    if (type == "ocsort")   return std::make_unique<OCSORT>();
    return std::make_unique<ByteTrack>(); // default
}

#endif // TRACKER_H
