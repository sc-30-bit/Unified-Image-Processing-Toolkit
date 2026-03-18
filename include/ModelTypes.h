#ifndef MODELTYPES_H
#define MODELTYPES_H
#include<opencv2/opencv.hpp>
#include<onnxruntime_cxx_api.h>


enum class ModelType {
    None,
    YOLO_Detect,    
    YOLO_Segment,   
    SuperRes,       
    Dehaze,        
    Underwater      
};


struct ModelOutput {
    cv::Mat imageResult;    

    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Mat> masks;     
    std::vector<int> trackIds;      
};

#endif // MODELTYPES_H
