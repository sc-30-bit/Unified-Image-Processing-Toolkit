#ifndef RESTORATION_H
#define RESTORATION_H

#include "basemodel.h"
#include "ModelTypes.h"
#include <vector>
#include <string>

struct RestorationConfig {
    bool input_0_1 = true;
    bool output_0_1 = true;
    bool is_bgr_input = false;

    bool use_tanh_range = false;

    std::string input_name = "input";
    std::string output_name = "output";

    int tile_size = 256;
    int tile_overlap = 32;
    int scale= 1;
};

class Restoration : public BaseModel
{
public:
    explicit Restoration(const std::string& modelPath,
                         RestorationConfig config = RestorationConfig());
    ~Restoration();

    ModelOutput predict(const cv::Mat& image) override;

private:
    cv::Mat inferenceSingleBlock(const cv::Mat& inputBlock);
    std::vector<int> generateIndices(int length, int tileSize, int overlap);

private:
    RestorationConfig m_config;
    std::string m_inputNameStr;
    std::string m_outputNameStr;
    std::vector<const char*> m_inputNodeNames;
    std::vector<const char*> m_outputNodeNames;
};

#endif // RESTORATION_H
