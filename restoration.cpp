#include "restoration.h"
#include <iostream>
#include <algorithm> // for min/max

Restoration::Restoration(const std::string& modelPath, RestorationConfig config)
    : BaseModel(modelPath), m_config(config)
{
    m_inputNameStr = m_config.input_name;
    m_outputNameStr = m_config.output_name;

    m_inputNodeNames.push_back(m_inputNameStr.c_str());
    m_outputNodeNames.push_back(m_outputNameStr.c_str());

    std::cout << "[Restoration] Initialized model: " << modelPath
              << " | Tile: " << m_config.tile_size
              << " | Overlap: " << m_config.tile_overlap
              << " | Scale: " << m_config.scale
              << " | Tanh: " << m_config.use_tanh_range << std::endl;
}

Restoration::~Restoration() {

}


std::vector<int> Restoration::generateIndices(int length, int tileSize, int overlap) {
    std::vector<int> indices;
    int stride = tileSize - overlap;

    for (int i = 0; i < length - tileSize; i += stride) {
        indices.push_back(i);
    }

    if (length >= tileSize) {
        indices.push_back(length - tileSize);
    } else {
        indices.push_back(0);
    }
    return indices;
}


ModelOutput Restoration::predict(const cv::Mat& image) {
    ModelOutput output;
    if (image.empty()) return output;

    int h = image.rows;
    int w = image.cols;
    int tile = m_config.tile_size;
    int overlap = m_config.tile_overlap;


    if (tile <= 0 || (h <= tile && w <= tile)) {

        int pad_h = (32 - h % 32) % 32;
        int pad_w = (32 - w % 32) % 32;

        cv::Mat inputs = image;
        if (pad_h > 0 || pad_w > 0) {
            cv::copyMakeBorder(image, inputs, 0, pad_h, 0, pad_w, cv::BORDER_REFLECT_101);
        }


        cv::Mat pred = inferenceSingleBlock(inputs);
        if (pred.empty()) return output;

        if (pad_h > 0 || pad_w > 0) {
            int scale = pred.rows / inputs.rows;
            int targetH = h * scale;
            int targetW = w * scale;


            if (targetW <= pred.cols && targetH <= pred.rows) {
                pred = pred(cv::Rect(0, 0, targetW, targetH)).clone();
            }
        }
        output.imageResult = pred;
        return output;
    }

    
    else {
        std::vector<int> h_idx_list = generateIndices(h, tile, overlap);
        std::vector<int> w_idx_list = generateIndices(w, tile, overlap);

        cv::Mat outputAccumulator; 
        cv::Mat countMap;         
        int scale = 0;

        for (int h_idx : h_idx_list) {
            for (int w_idx : w_idx_list) {

                int h_end = std::min(h, h_idx + tile);
                int w_end = std::min(w, w_idx + tile);
                cv::Rect inRoi(w_idx, h_idx, w_end - w_idx, h_end - h_idx);

                cv::Mat in_patch = image(inRoi).clone();


                cv::Mat out_patch = inferenceSingleBlock(in_patch);
                if (out_patch.empty()) continue;


                if (outputAccumulator.empty()) {
                    scale = out_patch.rows / in_patch.rows;

                    outputAccumulator = cv::Mat::zeros(h * scale, w * scale, CV_32FC3);
                    countMap = cv::Mat::zeros(h * scale, w * scale, CV_32FC3);
                }

                cv::Mat out_patch_float;
                out_patch.convertTo(out_patch_float, CV_32F);

                cv::Rect targetRoi(w_idx * scale, h_idx * scale, out_patch.cols, out_patch.rows);

                cv::add(outputAccumulator(targetRoi), out_patch_float, outputAccumulator(targetRoi));

                cv::add(countMap(targetRoi), cv::Scalar(1.0f, 1.0f, 1.0f), countMap(targetRoi));
            }
        }

        if (!outputAccumulator.empty()) {
            cv::divide(outputAccumulator, countMap, outputAccumulator);
            outputAccumulator.convertTo(output.imageResult, CV_8UC3);
        }
    }

    return output;
}

cv::Mat Restoration::inferenceSingleBlock(const cv::Mat& image) {
    if (image.empty()) return cv::Mat();

    try {
        cv::Mat blob;
        double scaleFactor = 1.0;
        cv::Scalar meanVal(0, 0, 0);
        bool swapRB = true; 

        if (m_config.use_tanh_range) {
            scaleFactor = 1.0 / 127.5;
            meanVal = cv::Scalar(127.5, 127.5, 127.5);
        }
        else if (m_config.input_0_1) {
            scaleFactor = 1.0 / 255.0;
        }
        else {
            scaleFactor = 1.0;
        }

        cv::dnn::blobFromImage(image, blob, scaleFactor, cv::Size(image.cols, image.rows), meanVal, swapRB, false);

        std::vector<int64_t> inputShape = {1, 3, image.rows, image.cols};
        size_t inputTensorSize = blob.total();

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memory_info, (float*)blob.data, inputTensorSize, inputShape.data(), inputShape.size());

        auto outputTensors = session->Run(Ort::RunOptions{nullptr},
                                          m_inputNodeNames.data(), &inputTensor, 1,
                                          m_outputNodeNames.data(), 1);


        float* floatData = outputTensors[0].GetTensorMutableData<float>();
        auto typeInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
        auto outputShape = typeInfo.GetShape();

        int outH = (int)outputShape[2];
        int outW = (int)outputShape[3];
        int area = outH * outW;

        cv::Mat resultImage(outH, outW, CV_8UC3);

        bool rangeIs01 = false;

        if (!m_config.use_tanh_range) { 
            float centerPixel = floatData[area / 2]; 
            if (centerPixel > -2.0f && centerPixel < 2.0f) {
                rangeIs01 = true;
            }
        }

#pragma omp parallel for
        for (int i = 0; i < area; i++) {
            float r = floatData[i];
            float g = floatData[i + area];
            float b = floatData[i + area * 2];

            // 策略 A: Tanh [-1, 1] -> [0, 255]
            if (m_config.use_tanh_range) {
                r = (r * 0.5f + 0.5f) * 255.0f;
                g = (g * 0.5f + 0.5f) * 255.0f;
                b = (b * 0.5f + 0.5f) * 255.0f;
            }
            // 策略 B: [0, 1] -> [0, 255] (包括强制配置或自动检测)
            else if (m_config.output_0_1 || rangeIs01) {
                r *= 255.0f;
                g *= 255.0f;
                b *= 255.0f;
            }
            // 策略 C: [0, 255] -> 保持原样

            // 截断到 0-255
            r = std::min(std::max(r, 0.0f), 255.0f);
            g = std::min(std::max(g, 0.0f), 255.0f);
            b = std::min(std::max(b, 0.0f), 255.0f);

            // 转为 OpenCV BGR 顺序 (r,g,b -> b,g,r)
            resultImage.data[i * 3 + 0] = (uchar)b;
            resultImage.data[i * 3 + 1] = (uchar)g;
            resultImage.data[i * 3 + 2] = (uchar)r;
        }

        return resultImage;

    } catch (const std::exception& e) {
        std::cerr << "[Inference] Error: " << e.what() << std::endl;
        return cv::Mat();
    }
}
