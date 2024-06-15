/**
 * @file ball_detection.hpp
 * @class BallDetector
 * @brief Detects and localizes billiard balls in an image.
 * @author Srobona Ghosh
 * @date 2024-06-11
 * 
 * This class uses a pre-trained deep learning model to detect and classify
 * billiard balls in input images.
 */

#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

struct BoundingBox {
    int x, y, width, height;
    int label;
};

class BallDetector {
public:
    /**
     * @brief Construct a new Ball Detector object.
     * 
     * @param model_path Path to the pre-trained ONNX model file.
     */
    BallDetector(const std::string& model_path);

    /**
     * @brief Detect billiard balls in the given frame.
     * 
     * @param frame Input image frame.
     * @return std::vector<BoundingBox> Vector of detected bounding boxes.
     */  
    std::vector<BoundingBox> detect(const cv::Mat& frame);

private:
    cv::dnn::Net net;
    std::vector<std::string> class_names;
};
