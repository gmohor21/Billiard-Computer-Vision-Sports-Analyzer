/**
 * @file output_data.cpp
 * @brief Implementation of output data generation functions.
 * @author Srobona Ghosh
 * @date 2024-06-14
 */

#include "output_data.hpp"
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

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
void saveOutputData(const std::string& output_dir, const cv::Mat& first_frame, const std::vector<BoundingBox>& first_frame_ball_boxes, const cv::Mat& first_frame_segmentation_mask, const std::vector<BoundingBox>& first_frame_ground_truth_boxes, const cv::Mat& first_frame_ground_truth_mask, const cv::Mat& last_frame, const std::vector<BoundingBox>& last_frame_ball_boxes, const cv::Mat& last_frame_segmentation_mask, const std::vector<BoundingBox>& last_frame_ground_truth_boxes, const cv::Mat& last_frame_ground_truth_mask, const std::vector<cv::Mat>& trajectories, double mAP, double mIoU) {
    // Save output video
    std::string output_video_path = output_dir + "/output_video.mp4";
    saveOutputVideos(trajectories, output_video_path);

    // Save evaluation metrics
    std::string metrics_file = output_dir + "/metrics.txt";
    std::ofstream file(metrics_file);
    if (file.is_open()) {
        file << "Mean Average Precision (mAP): " << mAP << std::endl;
        file << "Mean Intersection over Union (mIoU): " << mIoU << std::endl;
    } else {
        std::cerr << "Error: Failed to create metrics file: " << metrics_file << std::endl;
    }
    file.close();

    // Save output images (bounding boxes and segmentation masks)
    cv::imwrite(output_dir + "/first_frame_bboxes.png", drawBoundingBoxes(first_frame, first_frame_ball_boxes));
    cv::imwrite(output_dir + "/first_frame_segmentation.png", first_frame_segmentation_mask);
    cv::imwrite(output_dir + "/first_frame_ground_truth_bboxes.png", drawBoundingBoxes(first_frame, first_frame_ground_truth_boxes));
    cv::imwrite(output_dir + "/first_frame_ground_truth_segmentation.png", first_frame_ground_truth_mask);

    cv::imwrite(output_dir + "/last_frame_bboxes.png", drawBoundingBoxes(last_frame, last_frame_ball_boxes));
    cv::imwrite(output_dir + "/last_frame_segmentation.png", last_frame_segmentation_mask);
    cv::imwrite(output_dir + "/last_frame_ground_truth_bboxes.png", drawBoundingBoxes(last_frame, last_frame_ground_truth_boxes));
    cv::imwrite(output_dir + "/last_frame_ground_truth_segmentation.png", last_frame_ground_truth_mask);
}

/**
 * @brief Draw bounding boxes on an image.
 *
 * @param frame Input image frame.
 * @param bboxes Vector of bounding boxes to draw.
 * @return cv::Mat Image with drawn bounding boxes.
 */
cv::Mat drawBoundingBoxes(const cv::Mat& frame, const std::vector<BoundingBox>& bboxes) {
    // Create a copy of the input frame
    cv::Mat output = frame.clone();

    // Draw bounding boxes on the output image
    for (const auto& box : bboxes) {
        // Calculate the coordinates of the bounding box rectangle
        cv::Rect rect(box.x, box.y, box.width, box.height);

        // Draw the rectangle on the output image
        cv::rectangle(output, rect, cv::Scalar(0, 255, 0), 2);
    }

    // Return the output image
    return output;
}
