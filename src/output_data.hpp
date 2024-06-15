/**
 * @file output_data.hpp
 * @brief Header file for output data generation functions.
 * @author Srobona Ghosh
 * @date 2024-06-14
 */

#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "ball_detection.hpp"

/**
 * @brief Save output data including images, bounding boxes, and segmentation masks.
 *
 * This function saves the processed data from the billiard analysis to the specified output directory.
 * It includes the original frames, detected bounding boxes, segmentation masks, ground truth data,
 * and evaluation metrics.
 *
 * @param output_dir Output directory path.
 * @param first_frame First frame of the video.
 * @param first_frame_ball_boxes Detected ball bounding boxes in the first frame.
 * @param first_frame_segmentation_mask Segmentation mask for the first frame.
 * @param first_frame_ground_truth_boxes Ground truth bounding boxes for the first frame.
 * @param first_frame_ground_truth_mask Ground truth segmentation mask for the first frame.
 * @param last_frame Last frame of the video.
 * @param last_frame_ball_boxes Detected ball bounding boxes in the last frame.
 * @param last_frame_segmentation_mask Segmentation mask for the last frame.
 * @param last_frame_ground_truth_boxes Ground truth bounding boxes for the last frame.
 * @param last_frame_ground_truth_mask Ground truth segmentation mask for the last frame.
 * @param trajectories Ball trajectories throughout the video.
 * @param mAP Mean Average Precision of the ball detection.
 * @param mIoU Mean Intersection over Union of the segmentation.
 */
void saveOutputData(const std::string& output_dir, const cv::Mat& first_frame, const std::vector<BoundingBox>& first_frame_ball_boxes, const cv::Mat& first_frame_segmentation_mask, const std::vector<BoundingBox>& first_frame_ground_truth_boxes, const cv::Mat& first_frame_ground_truth_mask, const cv::Mat& last_frame, const std::vector<BoundingBox>& last_frame_ball_boxes, const cv::Mat& last_frame_segmentation_mask, const std::vector<BoundingBox>& last_frame_ground_truth_boxes, const cv::Mat& last_frame_ground_truth_mask, const std::vector<cv::Mat>& trajectories, double mAP, double mIoU);

/**
 * @brief Draw bounding boxes on an image.
 *
 * @param frame Input image frame.
 * @param bboxes Vector of bounding boxes to draw.
 * @return cv::Mat Image with drawn bounding boxes.
 */
cv::Mat drawBoundingBoxes(const cv::Mat& frame, const std::vector<BoundingBox>& bboxes);
