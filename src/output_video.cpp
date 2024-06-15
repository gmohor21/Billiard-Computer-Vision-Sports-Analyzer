/**
 * @file output_video.cpp
 * @brief Implementation of output video generation functions.
 * @author Srobona Ghosh
 * @date 2024-06-15
 */

#include "output_video.hpp"
#include <iostream>

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
void saveOutputVideos(const std::vector<cv::Mat>& trajectories,
                      const std::string& input_video_path,
                      const std::string& output_dir) {
    // Open the input video
    cv::VideoCapture cap(input_video_path);  // Open the input video
    if (!cap.isOpened()) {  // Check if video was successfully opened
        std::cerr << "Error: Could not open input video file: " << input_video_path << std::endl;
        return;
    }

    // Get video properties
    int fps = static_cast<int>(cap.get(cv::CAP_PROP_FPS));  // Frames per second
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));  // Width of each frame
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));  // Height of each frame

    // Create video writer
    std::string output_video_path = output_dir + "/output_video.mp4";  // Path to the output video
    cv::VideoWriter writer;  // Create a video writer object
    writer.open(output_video_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height), true);  // Open the output video

    if (!writer.isOpened()) {  // Check if video writer was successfully opened
        std::cerr << "Error: Could not create output video file: " << output_video_path << std::endl;
        return;
    }

    // Write frames with overlaid trajectories
    int frame_index = 0;  // Index of the current frame
    while (true) {
        cv::Mat frame;  // Create a Mat object to store the current frame
        cap >> frame;  // Read the next frame from the input video
        if (frame.empty()) break;  // Check if the frame is empty (end of video)

        // Overlay trajectories on the frame
        for (const auto& trajectory : trajectories) {
            cv::Mat colored_trajectory;  // Create a Mat object to store the colored trajectory
            cv::applyColorMap(trajectory, colored_trajectory, cv::COLORMAP_JET);  // Apply color to the trajectory
            cv::addWeighted(frame, 1.0, colored_trajectory, 0.5, 0.0, frame);  // Overlay the trajectory on the frame
        }

        writer.write(frame);  // Write the frame to the output video
        frame_index++;  // Increment the frame index
    }

    cap.release();  // Release the input video capture
    writer.release();  // Release the output video writer
    std::cout << "Output video saved to: " << output_video_path << std::endl;  // Print the path to the output video
}
