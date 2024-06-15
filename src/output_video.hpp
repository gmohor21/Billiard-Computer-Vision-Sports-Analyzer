/**
 * @file output_video.hpp
 * @brief Header file for output video generation functions.
 * @author Your Name
 * @date YYYY-MM-DD
 */

#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

/**
 * @brief Save output videos with ball trajectories.
 *
 * This function generates output videos that visualize the detected ball trajectories
 * overlaid on the original video frames.
 *
 * @param trajectories Vector of ball trajectory images.
 * @param input_video_path Path to the input video file.
 * @param output_dir Output directory for the generated video.
 */
void saveOutputVideos(const std::vector<cv::Mat>& trajectories, const std::string& output_path);
