#ifndef COLORIZER_H
#define COLORIZER_H

#include "basemodel.h"
#include "ModelTypes.h"
#include <string>
#include <vector>

class Colorizer : public BaseModel
{
public:
    // DeOldify ONNX-only colorizer.
    explicit Colorizer(const std::string& modelPath);

    // Run black-and-white to color inference.
    ModelOutput predict(const cv::Mat& input) override;

private:
    std::vector<const char*> m_inputNodeNames;
    std::vector<const char*> m_outputNodeNames;
    std::string m_inputNameAlloc;
    std::string m_outputNameAlloc;
};

#endif // COLORIZER_H
