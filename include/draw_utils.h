#ifndef DRAW_UTILS_H
#define DRAW_UTILS_H

#include <opencv2/opencv.hpp>
#include "ModelTypes.h" 
#include <random>

static std::vector<cv::Scalar> generateColors(int numColors) {
    std::vector<cv::Scalar> colors;
    std::srand(time(0));
    for(int i=0; i<numColors; i++) {
        int b = std::rand() % 256;
        int g = std::rand() % 256;
        int r = std::rand() % 256;
        colors.push_back(cv::Scalar(b, g, r));
    }
    return colors;
}

static std::vector<cv::Scalar> _colors = generateColors(100);

static void visualizeResult(cv::Mat& img, const ModelOutput& result, const std::vector<std::string>& classNames) {
    for (size_t i = 0; i < result.boxes.size(); i++) {
        int trackId = (i < result.trackIds.size()) ? result.trackIds[i] : -1;
        int classId = result.classIds[i];
        float conf = result.confidences[i];
        cv::Rect box = result.boxes[i];

        cv::Scalar color = (trackId != -1) ? _colors[trackId % _colors.size()] : _colors[classId % _colors.size()];

        if (i < result.masks.size() && !result.masks[i].empty()) {
            cv::Mat mask = result.masks[i];
            cv::Rect validBox = box & cv::Rect(0, 0, img.cols, img.rows);
            if (validBox.width > 0 && validBox.height > 0) {
                cv::Mat roi = img(validBox);

                cv::Mat colorMask(validBox.size(), CV_8UC3, color);
                cv::Mat maskFloat;
                mask.convertTo(maskFloat, CV_32F, 1.0/255.0); 
                cv::addWeighted(roi, 0.6, colorMask, 0.4, 0.0, roi);
            }
        }

        cv::rectangle(img, box, color, 2);

        std::string label = classNames[classId];
        if (trackId != -1) label += " ID:" + std::to_string(trackId);
        label += " " + std::to_string((int)(conf * 100)) + "%";

        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(img, cv::Point(box.x, box.y - labelSize.height - 5),
                      cv::Point(box.x + labelSize.width, box.y), color, cv::FILLED);
        cv::putText(img, label, cv::Point(box.x, box.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
}

#endif // DRAW_UTILS_H
