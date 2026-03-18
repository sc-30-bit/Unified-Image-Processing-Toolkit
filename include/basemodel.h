#ifndef BASEMODEL_H
#define BASEMODEL_H

#include <ModelTypes.h>
#include <string>
#include <memory>
#include <iostream>

class BaseModel
{
public:
    explicit BaseModel(const std::string& modelPath)
        : env(ORT_LOGGING_LEVEL_WARNING, "BaseModel")
    {
        std::wstring wModelPath(modelPath.begin(), modelPath.end());
        initSession(wModelPath);
    }

    BaseModel()
        : env(ORT_LOGGING_LEVEL_WARNING, "BaseModel") {}

    virtual ~BaseModel() {
        std::cout << "Releasing model resource..." << std::endl;
        session.reset(); 
    }

    virtual ModelOutput predict(const cv::Mat& input) = 0;

protected:
    Ort::Env env;
    std::shared_ptr<Ort::Session> session;

    void initSession(const std::wstring& modelPath) {
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        try {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            cuda_options.gpu_mem_limit = 6ULL * 1024 * 1024 * 1024;
            options.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "[BaseModel] >>> CUDA (GPU) Execution Provider ENABLED! <<<" << std::endl;
        } catch (...) {
            std::cout << "[BaseModel] WARNING: Failed to enable CUDA. Fallback to CPU." << std::endl;
        }

        session = std::make_shared<Ort::Session>(env, modelPath.c_str(), options);
    }
};

#endif // BASEMODEL_H
