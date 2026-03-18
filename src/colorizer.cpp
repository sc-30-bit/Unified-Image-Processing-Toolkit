#include "colorizer.h"
#include <algorithm>
#include <iostream>

Colorizer::Colorizer(const std::string& modelPath)
    : BaseModel(modelPath)
{
    Ort::AllocatorWithDefaultOptions allocator;
    m_inputNameAlloc = session->GetInputNameAllocated(0, allocator).get();
    m_outputNameAlloc = session->GetOutputNameAllocated(0, allocator).get();

    m_inputNodeNames.push_back(m_inputNameAlloc.c_str());
    m_outputNodeNames.push_back(m_outputNameAlloc.c_str());
}

ModelOutput Colorizer::predict(const cv::Mat& input)
{
    ModelOutput output;
    if (input.empty()) return output;

    try {
        auto inShape = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        int netH = 512;
        int netW = 512;
        if (inShape.size() >= 4) {
            if (inShape[2] > 0) netH = static_cast<int>(inShape[2]);
            if (inShape[3] > 0) netW = static_cast<int>(inShape[3]);
        }

        const int oriW = input.cols;
        const int oriH = input.rows;

        // DeOldify preprocess: grayscale -> RGB -> ImageNet normalization -> NCHW.
        cv::Mat gray, gray3;
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(gray, gray3, cv::COLOR_GRAY2RGB);

        cv::Mat resized, f32;
        cv::resize(gray3, resized, cv::Size(netW, netH));
        resized.convertTo(f32, CV_32F, 1.0 / 255.0);

        static const float kMean[3] = {0.485f, 0.456f, 0.406f};
        static const float kStd[3] = {0.229f, 0.224f, 0.225f};

        std::vector<float> inputBlob(3 * netH * netW);
        for (int h = 0; h < netH; ++h) {
            const cv::Vec3f* row = f32.ptr<cv::Vec3f>(h);
            for (int w = 0; w < netW; ++w) {
                for (int c = 0; c < 3; ++c) {
                    float v = (row[w][c] - kMean[c]) / kStd[c];
                    inputBlob[c * netH * netW + h * netW + w] = v;
                }
            }
        }

        std::vector<int64_t> inputShape = {1, 3, netH, netW};
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, inputBlob.data(), inputBlob.size(), inputShape.data(), inputShape.size());

        auto outputTensors = session->Run(Ort::RunOptions{nullptr},
                                          m_inputNodeNames.data(), &inputTensor, 1,
                                          m_outputNodeNames.data(), 1);

        float* outData = outputTensors[0].GetTensorMutableData<float>();
        auto outShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

        if (outShape.size() != 4) {
            std::cerr << "[Colorizer-DeOldify] Unexpected output rank: " << outShape.size() << std::endl;
            return output;
        }

        bool nchw = false;
        bool nhwc = false;
        int ch = -1, outH = -1, outW = -1;

        if (outShape[1] > 0 && outShape[2] > 0 && outShape[3] > 0) {
            ch = static_cast<int>(outShape[1]);
            outH = static_cast<int>(outShape[2]);
            outW = static_cast<int>(outShape[3]);
            nchw = true;
        }
        if (ch != 3 && outShape[3] > 0 && outShape[1] > 0 && outShape[2] > 0) {
            ch = static_cast<int>(outShape[3]);
            outH = static_cast<int>(outShape[1]);
            outW = static_cast<int>(outShape[2]);
            nhwc = true;
            nchw = false;
        }

        if (ch != 3 || outH <= 0 || outW <= 0) {
            std::cerr << "[Colorizer-DeOldify] Unexpected output shape:";
            for (auto d : outShape) std::cerr << " " << d;
            std::cerr << std::endl;
            return output;
        }

        cv::Mat outRgbF32(outH, outW, CV_32FC3);
        if (nchw) {
            int area = outH * outW;
            for (int y = 0; y < outH; ++y) {
                cv::Vec3f* row = outRgbF32.ptr<cv::Vec3f>(y);
                for (int x = 0; x < outW; ++x) {
                    int idx = y * outW + x;
                    row[x][0] = outData[idx];
                    row[x][1] = outData[idx + area];
                    row[x][2] = outData[idx + area * 2];
                }
            }
        } else {
            cv::Mat tmp(outH, outW, CV_32FC3, outData);
            outRgbF32 = tmp.clone();
        }

        // De-normalize back to [0,1].
        for (int y = 0; y < outH; ++y) {
            cv::Vec3f* row = outRgbF32.ptr<cv::Vec3f>(y);
            for (int x = 0; x < outW; ++x) {
                for (int c = 0; c < 3; ++c) {
                    float v = row[x][c] * kStd[c] + kMean[c];
                    row[x][c] = std::min(1.0f, std::max(0.0f, v));
                }
            }
        }

        cv::Mat colorRgb8;
        outRgbF32.convertTo(colorRgb8, CV_8UC3, 255.0);

        cv::Mat colorRgbOrig;
        cv::resize(colorRgb8, colorRgbOrig, cv::Size(oriW, oriH));

        // Keep luminance from original and replace chroma from model output.
        cv::Mat origRgb;
        cv::cvtColor(input, origRgb, cv::COLOR_BGR2RGB);

        cv::Mat colorYuv, origYuv;
        cv::cvtColor(colorRgbOrig, colorYuv, cv::COLOR_RGB2YUV);
        cv::cvtColor(origRgb, origYuv, cv::COLOR_RGB2YUV);

        std::vector<cv::Mat> cch, och;
        cv::split(colorYuv, cch);
        cv::split(origYuv, och);
        och[1] = cch[1];
        och[2] = cch[2];

        cv::Mat fusedYuv;
        cv::merge(och, fusedYuv);

        cv::Mat finalRgb;
        cv::cvtColor(fusedYuv, finalRgb, cv::COLOR_YUV2RGB);
        cv::cvtColor(finalRgb, output.imageResult, cv::COLOR_RGB2BGR);

    } catch (const std::exception& e) {
        std::cerr << "[Colorizer-DeOldify] Error: " << e.what() << std::endl;
    }

    return output;
}
