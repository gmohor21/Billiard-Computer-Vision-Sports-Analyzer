/**
 * @file visualization.hpp
 * @brief Header file for visualization functions.
 * @author Srobona Ghosh
 * @date 2024-06-14
 */

#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

/**
 * @brief Generate a 2D top-view visualization of the current game state.
 * 
 * @param frame Input image frame.
 * @param segmentation_mask Segmentation mask of the balls and playing field.
 * @param field_rect Bounding rectangle of the billiard table.
 * @return cv::Mat 2D top-view visualization.
 */
cv::Mat visualizeTopView(const cv::Mat& frame, const cv::Mat& segmentation_mask, const cv::Rect& field_rect);

/**
 * @brief Track ball trajectories across frames.
 * 
 * @param trajectories Vector of previous ball trajectories.
 * @param segmentation_mask Current segmentation mask.
 * @param frame_count Current frame index.
 * @return std::vector<cv::Mat> Updated ball trajectories.
 */
std::vector<cv::Mat> trackBallTrajectories(const std::vector<cv::Mat>& trajectories, const cv::Mat& segmentation_mask, int frame_count);
