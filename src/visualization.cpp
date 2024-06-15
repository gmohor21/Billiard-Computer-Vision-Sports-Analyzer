/**
 * @file visualization.cpp
 * @brief Implementation of functions for visualizing the billiard game state and ball trajectories.
 * @author Srobona Ghosh
 * @date 2024-06-13
 */

#include "visualization.hpp"

/**
 * @brief Generate a 2D top-view visualization of the billiard table and ball positions.
 *
 * @param frame Input image frame.
 * @param segmentation_mask Segmentation mask of the balls and playing field.
 * @param field_rect Bounding rectangle of the billiard table.
 *
 * @return cv::Mat 2D top-view visualization.
 *
 * The function creates a top-view visualization of the billiard table and ball positions.
 * It applies a perspective transform to the segmentation mask to project it onto a 2D plane.
 */
cv::Mat visualizeTopView(const cv::Mat& frame, const cv::Mat& segmentation_mask, const cv::Rect& field_rect) {
    // Create a matrix to store the top-view visualization
    cv::Mat top_view = cv::Mat::zeros(field_rect.size(), CV_8UC3);

    // Define source and destination points for perspective transform
    // Source points are the corners of the field rectangle
    std::vector<cv::Point2f> src_points;
    src_points.push_back(cv::Point2f(field_rect.x, field_rect.y));
    src_points.push_back(cv::Point2f(field_rect.x + field_rect.width, field_rect.y));
    src_points.push_back(cv::Point2f(field_rect.x + field_rect.width, field_rect.y + field_rect.height));
    src_points.push_back(cv::Point2f(field_rect.x, field_rect.y + field_rect.height));

    // Destination points are the corners of the top-view image
    std::vector<cv::Point2f> dst_points;
    dst_points.push_back(cv::Point2f(0, 0));
    dst_points.push_back(cv::Point2f(top_view.cols, 0));
    dst_points.push_back(cv::Point2f(top_view.cols, top_view.rows));
    dst_points.push_back(cv::Point2f(0, top_view.rows));

    // Compute perspective transform matrix
    cv::Mat perspective_matrix = cv::getPerspectiveTransform(src_points, dst_points);

    // Apply perspective transform to the segmentation mask
    cv::warpPerspective(segmentation_mask, top_view, perspective_matrix, top_view.size());

    return top_view;
}

/**
 * @brief Track ball trajectories across frames.
 *
 * @param trajectories Vector of previous ball trajectories.
 * @param segmentation_mask Current segmentation mask.
 * @param frame_count Current frame index.
 *
 * @return Updated ball trajectories as a vector of cv::Mat.
 *
 * The function tracks the trajectories of billiard balls across frames.
 * If it's the first frame, it initializes the trajectories.
 * For each pixel in the segmentation mask, if the label is between 1 and 6,
 * it updates the corresponding trajectory matrix.
 */
std::vector<cv::Mat> trackBallTrajectories(const std::vector<cv::Mat>& trajectories, const cv::Mat& segmentation_mask, int frame_count) {
    std::vector<cv::Mat> updated_trajectories = trajectories;

    // Initialize trajectories if it's the first frame
    if (frame_count == 0) {
        updated_trajectories.clear();  // Clear previous trajectories
        for (int i = 0; i < 6; i++) {  // Initialize trajectories for 6 balls
            updated_trajectories.push_back(cv::Mat::zeros(segmentation_mask.size(), CV_8UC1));
        }
    }

    // Update trajectories based on the current segmentation mask
    for (int i = 0; i < segmentation_mask.rows; i++) {
        for (int j = 0; j < segmentation_mask.cols; j++) {
            int label = segmentation_mask.at<uchar>(i, j);  // Get the label of the pixel
            if (label > 0 && label < 6) {  // Check if the label is valid
                updated_trajectories[label - 1].at<uchar>(i, j) = 255;  // Update the corresponding trajectory matrix
            }
        }
    }

    return updated_trajectories;
}
