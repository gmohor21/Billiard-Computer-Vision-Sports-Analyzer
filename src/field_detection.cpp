/**
 * @file field_detection.cpp
 * @brief Implementation of functions for detecting the billiard table and its boundaries.
 * @author Srobona Ghosh
 * @date 2024-06-11
 */

#include "field_detection.hpp"

/**
 * @brief Detect field lines in the input frame using Canny edge detection and Hough line transform.
 * 
 * @param frame Input image frame.
 * @return std::vector<cv::Vec4i> Vector of detected lines represented as 4-tuples (x1, y1, x2, y2).
 * 
 * The function uses the Canny edge detection algorithm to detect edges in the input frame.
 * It then uses the Hough line transform to detect lines in the edge map.
 * The detected lines are returned as a vector of 4-tuples (x1, y1, x2, y2), where (x1, y1) and (x2, y2)
 * are the endpoints of each line.
 */
std::vector<cv::Vec4i> detectFieldLines(const cv::Mat& frame) {
    std::vector<cv::Vec4i> lines;  // Vector to store detected lines

    cv::Mat edges;  // Matrix to store edge map
    
    // Detect edges in the input frame
    cv::Canny(frame, edges, 100, 200);  // Use Canny edge detection

    // Detect lines using Hough line transform
    cv::HoughLinesP(edges, lines, 1, CV_PI/180, 50, 50, 10);  // Use Hough line transform with parameters

    return lines;  // Return the detected lines
}

/**
 * @brief Compute the bounding rectangle of the detected lines
 *
 * @param lines Vector of detected lines represented as 4-tuples (x1, y1, x2, y2).
 *
 * @return cv::Rect Bounding rectangle of the detected lines.
 *
 * The function collects all line endpoints and computes the bounding rectangle
 * using OpenCV's boundingRect function.
 */
cv::Rect getBoundingRect(const std::vector<cv::Vec4i>& lines) {
    std::vector<cv::Point> points;  // Vector to store line endpoints

    // Collect all line endpoints
    for (const auto& line : lines) {
        points.push_back(cv::Point(line[0], line[1]));  // Add the first endpoint of each line
        points.push_back(cv::Point(line[2], line[3]));  // Add the second endpoint of each line
    }

    // Compute the bounding rectangle
    return cv::boundingRect(points);  // Return the bounding rectangle
}
