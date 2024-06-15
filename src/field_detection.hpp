/**
 * @file field_detection.hpp
 * @brief Header file for field detection functions.
 * @author Srobona Ghosh
 * @date 2024-06-12
 */
 
#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

/**
 * @brief Detect the main lines of the billiard table in the given frame.
 * 
 * @param frame Input image frame.
 * @return std::vector<cv::Vec4i> Vector of detected lines.
 */
std::vector<cv::Vec4i> detectFieldLines(const cv::Mat& frame);

/**
 * @brief Compute the bounding rectangle of the billiard table from the detected lines.
 * 
 * @param lines Vector of detected lines.
 * @return cv::Rect Bounding rectangle of the billiard table.
 */
cv::Rect getBoundingRect(const std::vector<cv::Vec4i>& lines);
