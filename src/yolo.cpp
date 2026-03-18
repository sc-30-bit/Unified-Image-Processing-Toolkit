#include "yolo.h"
#include <iostream>
#include <opencv2/dnn.hpp>

YoloModel::YoloModel(const std::string& modelPath,
                     const std::vector<std::string>& labels)
    : BaseModel(modelPath),
    classNames(labels)
{
    Ort::AllocatorWithDefaultOptions allocator;

    // input nodes
    size_t numInputNodes = session->GetInputCount();
    inputNodeNamesAllocated.reserve(numInputNodes);
    inputNodeNames.clear();

    for (size_t i = 0; i < numInputNodes; i++) {
        auto name = session->GetInputNameAllocated(i, allocator);
        inputNodeNamesAllocated.push_back(name.get());
    }
    for (size_t i = 0; i < numInputNodes; i++) {
        inputNodeNames.push_back(inputNodeNamesAllocated[i].c_str());
    }

    // output nodes
    size_t numOutputNodes = session->GetOutputCount();
    outputNodeNamesAllocated.reserve(numOutputNodes);
    outputNodeNames.clear();

    for (size_t i = 0; i < numOutputNodes; i++) {
        auto name = session->GetOutputNameAllocated(i, allocator);
        outputNodeNamesAllocated.push_back(name.get());
    }
    for (size_t i = 0; i < numOutputNodes; i++) {
        outputNodeNames.push_back(outputNodeNamesAllocated[i].c_str());
    }

    // segmentation?
    if (numOutputNodes >= 2) {
        isSegmentation = true;
        std::cout << "[YoloModel] Detected Segmentation Model." << std::endl;
    } else {
        isSegmentation = false;
        std::cout << "[YoloModel] Detected Detection Model." << std::endl;
    }
}

ModelOutput YoloModel::predict(const cv::Mat& input) {
    YoloOptions opt;
    opt.conf = 0.25f;
    opt.iou  = 0.45f;
    opt.max_det = 100;
    return predict(input, opt);
}

void YoloModel::preProcess(const cv::Mat& iImg, cv::Mat& oImg, float& resizeScale)
{
    cv::Mat rgbImg;
    if (iImg.channels() == 3) cv::cvtColor(iImg, rgbImg, cv::COLOR_BGR2RGB);
    else cv::cvtColor(iImg, rgbImg, cv::COLOR_GRAY2RGB);

    if (rgbImg.cols >= rgbImg.rows) resizeScale = rgbImg.cols / (float)inputSize.width;
    else resizeScale = rgbImg.rows / (float)inputSize.height;

    int newWidth  = int(rgbImg.cols / resizeScale);
    int newHeight = int(rgbImg.rows / resizeScale);

    cv::Mat resizedImg;
    cv::resize(rgbImg, resizedImg, cv::Size(newWidth, newHeight));

    oImg = cv::Mat::zeros(inputSize, CV_8UC3);
    resizedImg.copyTo(oImg(cv::Rect(0, 0, newWidth, newHeight)));
}

void YoloModel::blobFromImage(cv::Mat& iImg, std::vector<float>& blob)
{
    int channels = iImg.channels();
    int rows = iImg.rows;
    int cols = iImg.cols;
    blob.resize(channels * rows * cols);

    // HWC -> NCHW
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < rows; h++) {
            for (int w = 0; w < cols; w++) {
                blob[c * rows * cols + h * cols + w] =
                    (float)iImg.at<cv::Vec3b>(h, w)[c] / 255.0f;
            }
        }
    }
}

ModelOutput YoloModel::predict(const cv::Mat& input, const YoloOptions& opt)
{
    ModelOutput output;

    // 1) preprocess
    cv::Mat preprocessedImg;
    float scale = 1.f;
    preProcess(input, preprocessedImg, scale);

    // 2) input tensor
    std::vector<float> inputBlob;
    blobFromImage(preprocessedImg, inputBlob);

    std::vector<int64_t> inputShape = {1, 3, inputSize.width, inputSize.height};

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, inputBlob.data(), inputBlob.size(), inputShape.data(), inputShape.size()
        );

    // 3) run
    auto outputTensors = session->Run(Ort::RunOptions{nullptr},
                                      inputNodeNames.data(), &inputTensor, inputNodeNames.size(),
                                      outputNodeNames.data(), outputNodeNames.size());

    // 4) parse output0: [1, dim, 8400]
    std::vector<int64_t> outShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    float* rawData = outputTensors[0].GetTensorMutableData<float>();

    int dimensions = (int)outShape[1];
    int rows = (int)outShape[2];

    cv::Mat outMat(dimensions, rows, CV_32F, rawData);
    cv::Mat tOut = outMat.t();
    float* pdata = (float*)tOut.data;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> picked_mask_coefs;

    classIds.reserve(rows);
    confidences.reserve(rows);
    boxes.reserve(rows);

    for (int i = 0; i < rows; ++i) {
        float* classesScores = pdata + 4;
        cv::Point classIdPoint;
        double maxScore;

        cv::Mat scores(1, (int)classNames.size(), CV_32FC1, classesScores);
        cv::minMaxLoc(scores, 0, &maxScore, 0, &classIdPoint);

        int cls = classIdPoint.x;

        if ((float)maxScore >= opt.conf && classAllowed(cls, opt.classes)) {
            float x = pdata[0];
            float y = pdata[1];
            float w = pdata[2];
            float h = pdata[3];

            int left   = int((x - 0.5f * w) * scale);
            int top    = int((y - 0.5f * h) * scale);
            int width  = int(w * scale);
            int height = int(h * scale);

            left = std::max(0, left);
            top  = std::max(0, top);
            width  = std::min(input.cols - left, width);
            height = std::min(input.rows - top, height);

            if (width > 1 && height > 1) {
                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back((float)maxScore);
                classIds.push_back(cls);

                if (isSegmentation) {
                    int maskStartIdx = dimensions - 32;
                    std::vector<float> coefs(32);
                    for (int j = 0; j < 32; j++) coefs[j] = pdata[maskStartIdx + j];
                    picked_mask_coefs.push_back(std::move(coefs));
                }
            }
        }

        pdata += dimensions;
    }

    // 5) NMS
    std::vector<int> nmsIdx;
    cv::dnn::NMSBoxes(boxes, confidences, opt.conf, opt.iou, nmsIdx);

    // 6) mask proto
    cv::Mat maskProtos;
    if (isSegmentation && outputTensors.size() >= 2) {
        float* protoData = outputTensors[1].GetTensorMutableData<float>();
        // [1, 32, 160, 160] -> [32, 25600]
        maskProtos = cv::Mat(32, 160 * 160, CV_32F, protoData);
    }

    // 7) assemble output (limit max_det)
    int cnt = 0;
    for (int idx : nmsIdx) {
        if (cnt >= opt.max_det) break;

        output.boxes.push_back(boxes[idx]);
        output.confidences.push_back(confidences[idx]);
        output.classIds.push_back(classIds[idx]);

        if (isSegmentation && !maskProtos.empty()) {
            cv::Mat coefsMat(1, 32, CV_32F, picked_mask_coefs[idx].data());
            cv::Mat mask = processMask(maskProtos, coefsMat, boxes[idx], scale, input.size());
            output.masks.push_back(mask);
        }

        cnt++;
    }

    return output;
}

// mask
cv::Mat YoloModel::processMask(const cv::Mat& maskProtos, const cv::Mat& maskCoefs,
                               const cv::Rect& box, float scale, cv::Size oriSize)
{
    (void)oriSize;
    if (box.width <= 0 || box.height <= 0) return cv::Mat();

    cv::Mat maskMat = maskCoefs * maskProtos; // 1x25600
    maskMat = maskMat.reshape(1, 160);        // 160x160

    // sigmoid
    cv::exp(-maskMat, maskMat);
    maskMat = 1.0 / (1.0 + maskMat);

    // map roi
    int x = int(box.x / scale * 0.25f);
    int y = int(box.y / scale * 0.25f);
    int w = int(box.width / scale * 0.25f);
    int h = int(box.height / scale * 0.25f);

    cv::Rect roi(x, y, w, h);
    roi.x = std::max(0, roi.x);
    roi.y = std::max(0, roi.y);
    roi.width  = std::min(160 - roi.x, roi.width);
    roi.height = std::min(160 - roi.y, roi.height);
    if (roi.width <= 0 || roi.height <= 0) return cv::Mat();

    cv::Mat cropped = maskMat(roi);

    cv::Mat resized;
    cv::resize(cropped, resized, box.size(), 0, 0, cv::INTER_LINEAR);

    return resized > maskThreshold;
}
