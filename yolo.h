#ifndef YOLO_H
#define YOLO_H

#include "basemodel.h"
#include <vector>
#include <string>
#include <algorithm>

struct YoloOptions {
    float conf = 0.25f;
    float iou  = 0.45f;
    int max_det = 100;
    std::vector<int> classes; 
};

class YoloModel : public BaseModel
{
public:
    explicit YoloModel(const std::string& modelPath,
                       const std::vector<std::string>& labels);

    ModelOutput predict(const cv::Mat& input) override;

    ModelOutput predict(const cv::Mat& input, const YoloOptions& opt);

private:
    std::vector<std::string> classNames;
    cv::Size inputSize = {640, 640};

    float maskThreshold = 0.50f;

    std::vector<const char*> inputNodeNames;
    std::vector<const char*> outputNodeNames;
    std::vector<std::string> inputNodeNamesAllocated;
    std::vector<std::string> outputNodeNamesAllocated;

    bool isSegmentation = false;

    void preProcess(const cv::Mat& input, cv::Mat& output, float& scale);
    void blobFromImage(cv::Mat& img, std::vector<float>& blob);
    cv::Mat processMask(const cv::Mat& maskProtos, const cv::Mat& maskCoefs,
                        const cv::Rect& box, float scale, cv::Size oriSize);

    static bool classAllowed(int cls, const std::vector<int>& allow) {
        if (allow.empty()) return true;
        for (int a : allow) if (a == cls) return true;
        return false;
    }
};

#endif // YOLO_H
