/**
 * @file ball_detection.cpp
 * @brief Implementation of the BallDetector class for billiard ball detection.
 * @author Srobona Ghosh
 * @date 2024-06-12
 */

#include "ball_detection.hpp"

/**
 * @brief Constructor for BallDetector class.
 *
 * @param model_path Path to the pre-trained ONNX model file.
 */
BallDetector::BallDetector(const std::string& model_path) {
    // Load the ONNX model
    // The model is expected to be in the Open Neural Network Exchange (ONNX) format.
    // The model should be trained on a dataset of billiard balls with different categories.
    net = cv::dnn::readNetFromONNX(model_path);
    
    // Initialize class names
    // The class names correspond to the different categories of billiard balls.
    // The order of class names should match the order of output classes in the model.
    // In this case, the class names are "cue_ball", "eight_ball", "solid_ball", and "striped_ball".
    class_names = {"cue_ball", "eight_ball", "solid_ball", "striped_ball"};
}

/**
 * @brief Detects billiard balls in the given frame and returns bounding boxes for each detected ball.
 *
 * @param frame Input image frame.
 * @return std::vector<BoundingBox> Vector of detected bounding boxes.
 */
std::vector<BoundingBox> BallDetector::detect(const cv::Mat& frame) {
    std::vector<BoundingBox> boxes;  // Vector to store bounding boxes

    // Prepare the input blob for the neural network
    // The input frame is resized to 320x320 pixels and normalized to the range [0, 1].
    cv::Mat blob = cv::dnn::blobFromImage(frame, 1/255.0, cv::Size(320, 320), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    // Forward pass through the network
    cv::Mat output = net.forward();

    // Post-process the output
    // The output is a 1x1xNx7 tensor, where N is the number of detected objects.
    // Each object is represented by a row of 7 values: (center_x, center_y, width, height, object_class, confidence).
    float x_factor = frame.cols / 320.0;  // Factor to scale the center_x coordinate back to the original frame size
    float y_factor = frame.rows / 320.0;  // Factor to scale the center_y coordinate back to the original frame size

    float* data = (float*)output.data;  // Pointer to the output tensor data
    
    for (int i = 0; i < output.rows; i++) {
        // Extract detection information
        int obj_class = data[i * 6 + 1];  // Object class label
        float confidence = data[i * 6 + 5];  // Confidence score

        if (confidence > 0.5) {
            // Calculate the bounding box coordinates
            int center_x = static_cast<int>(data[i * 6] * frame.cols);
            int center_y = static_cast<int>(data[i * 6 + 2] * frame.rows);
            int width = static_cast<int>(data[i * 6 + 3] * frame.cols);
            int height = static_cast<int>(data[i * 6 + 4] * frame.rows);

            int x = center_x - width / 2;  // Bounding box top-left x-coordinate
            int y = center_y - height / 2;  // Bounding box top-left y-coordinate

            // Create and store the bounding box
            BoundingBox box = {x, y, width, height, obj_class};
            boxes.push_back(box);
        }
    }

    return boxes;
}
