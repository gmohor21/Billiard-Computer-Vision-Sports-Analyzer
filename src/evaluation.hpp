/**
 * @file evaluation.hpp
 * @brief Header file for evaluation functions.
 * @author Srobona Ghosh
 * @date 2024-06-15
 */

#pragma once

#include <string>
#include <opencv2/opencv.hpp>

/**
 * @brief Evaluate the mean Average Precision (mAP) of ball detection.
 * 
 * @param detected_boxes Vector of detected bounding boxes.
 * @param ground_truth_path Path to the ground truth bounding box annotations.
 * @param frame_index Current frame index.
 * @return double Mean Average Precision (mAP) score.
 */
double evaluateMeanAveragePrecision(const std::vector<BoundingBox>& detected_boxes, const std::string& ground_truth_path, int frame_index);

/**
 * @brief Evaluate the mean Intersection over Union (mIoU) of segmentation.
 * 
 * @param segmentation_mask Predicted segmentation mask.
 * @param ground_truth_path Path to the ground truth segmentation mask.
 * @param frame_index Current frame index.
 * @return double Mean Intersection over Union (mIoU) score.
 */
double evaluateMeanIntersectionOverUnion(const cv::Mat& segmentation_mask, const std::string& ground_truth_path, int frame_index);
