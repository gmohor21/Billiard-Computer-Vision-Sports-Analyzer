/**
 * @file segmentation.cpp
 * @brief Implementation of functions for segmenting billiard balls and the playing field.
 * @author Srobona Ghosh
 * @date 2024-06-11
 */

#include "segmentation.hpp"

/**
 * @brief Segment balls and field in the given frame.
 *
 * @param frame Input image frame.
 * @param ball_boxes Vector of detected ball bounding boxes.
 * @param field_rect Bounding rectangle of the billiard table.
 *
 * @return cv::Mat Segmentation mask.
 *
 * The function creates a mask of the same size as the input frame, fills the
 * playing field region with label 5, and fills the ball regions with their
 * corresponding labels.
 */
cv::Mat segmentBallsAndField(const cv::Mat& frame, const std::vector<BoundingBox>& ball_boxes, const cv::Rect& field_rect) {
    // Create a mask of the same size as the input frame
    cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);

    // Segment the playing field
    // Fill the playing field region with label 5
    cv::rectangle(mask, field_rect, cv::Scalar(5), cv::FILLED);

    // Segment the balls
    // Fill the ball regions with their corresponding labels
    for (const auto& box : ball_boxes) {
        cv::Rect roi(box.x, box.y, box.width, box.height);
        cv::rectangle(mask, roi, cv::Scalar(box.label), cv::FILLED);
    }

    return mask;
}
