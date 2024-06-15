/**
 * @file segmentation.hpp
 * @brief Header file for segmentation functions.
 * @author Srobona Ghosh
 * @date 2024-06-13
 */
 
#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

#include "ball_detection.hpp"

/**
 * @brief Segment the billiard balls and playing field in the given frame.
 * 
 * @param frame Input image frame.
 * @param ball_boxes Vector of detected ball bounding boxes.
 * @param field_rect Bounding rectangle of the billiard table.
 * @return cv::Mat Segmentation mask.
 */

cv::Mat segmentBallsAndField(const cv::Mat& frame, const std::vector<BoundingBox>& ball_boxes, const cv::Rect& field_rect);
