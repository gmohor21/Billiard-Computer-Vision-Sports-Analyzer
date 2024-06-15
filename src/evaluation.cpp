/**
 * @file evaluation.cpp
 * @brief Implementation of evaluation metrics for ball detection and segmentation.
 * @author Srobona Ghosh
 * @date 2024-06-14
 */

#include "evaluation.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/opencv.hpp>

/**
 * @brief Evaluates the Mean Average Precision (mAP) for ball detection.
 *
 * @param detected_boxes Vector of detected bounding boxes.
 * @param ground_truth_path Path to the ground truth bounding box annotations.
 * @param frame_index Current frame index.
 *
 * @return Mean Average Precision (mAP) score.
 */
double evaluateMeanAveragePrecision(const std::vector<BoundingBox>& detected_boxes, const std::string& ground_truth_path, int frame_index) {
    // Load ground truth bounding boxes from file
    std::vector<BoundingBox> ground_truth_boxes;

    // Form the ground truth file path
    std::string bbox_file = ground_truth_path + "/frame_" + std::to_string(frame_index) + "_bbox.txt";

    // Open the ground truth file
    std::ifstream file(bbox_file);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open ground truth file: " << bbox_file << std::endl;
        return 0.0;
    }

    // Read ground truth bounding boxes from the file
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int x, y, width, height, label;
        if (iss >> x >> y >> width >> height >> label) {
            ground_truth_boxes.push_back({x, y, width, height, label});
        }
    }
    file.close();

    // Calculate mean Average Precision
    double mAP = calculateMeanAveragePrecision(detected_boxes, ground_truth_boxes);

    return mAP;
}

/**
 * @brief Evaluates the Mean Intersection over Union (mIoU) for segmentation.
 *
 * @param segmentation_mask Predicted segmentation mask.
 * @param ground_truth_path Path to the ground truth segmentation mask.
 * @param frame_index Current frame index.
 *
 * @return Mean Intersection over Union (mIoU) score.
 */
double evaluateMeanIntersectionOverUnion(const cv::Mat& segmentation_mask, const std::string& ground_truth_path, int frame_index) {
    // Load ground truth segmentation mask from file
    std::string mask_file = ground_truth_path + "/frame_" + std::to_string(frame_index) + ".png";
    cv::Mat ground_truth_mask = cv::imread(mask_file, cv::IMREAD_GRAYSCALE);

    // Check if the ground truth mask was loaded successfully
    if (ground_truth_mask.empty()) {
        std::cerr << "Error: Failed to load ground truth mask: " << mask_file << std::endl;
        return 0.0;
    }

    // Calculate mean Intersection over Union
    double mIoU = calculateMeanIoU(segmentation_mask, ground_truth_mask);

    return mIoU;
}

/**
 * @brief Calculates the Mean Average Precision (mAP) for ball detection.
 *
 * @param detected_boxes Vector of detected bounding boxes.
 * @param ground_truth_boxes Vector of ground truth bounding boxes.
 *
 * @return Mean Average Precision (mAP) score.
 */
double calculateMeanAveragePrecision(const std::vector<BoundingBox>& detected_boxes, const std::vector<BoundingBox>& ground_truth_boxes) {
    // Calculate precision for each detected bounding box
    std::vector<double> precisions;
    double totalPositives = static_cast<double>(ground_truth_boxes.size());

    for (const auto& detectedBox : detected_boxes) {
        // Calculate true positives for the detected bounding box
        double truePositives = 0.0;
        for (const auto& groundTruthBox : ground_truth_boxes) {
            double iou = calculateIoU(detectedBox, groundTruthBox);
            if (iou >= 0.5 && detectedBox.label == groundTruthBox.label) {
                truePositives += 1.0;
            }
        }

        // Calculate precision for the detected bounding box
        double precision = truePositives / (static_cast<double>(detected_boxes.size()) + std::numeric_limits<double>::epsilon());
        precisions.push_back(precision);
    }

    // Calculate mean Average Precision
    double mAP = 0.0;
    std::sort(precisions.begin(), precisions.end(), std::greater<double>());
    for (int i = 0; i < precisions.size(); ++i) {
        mAP += precisions[i] * ((i + 1.0) / totalPositives);
    }
    mAP /= precisions.size();

    return mAP;
}

/**
 * @brief Calculates the Intersection over Union (IoU) between two bounding boxes.
 *
 * @param box1 First bounding box.
 * @param box2 Second bounding box.
 *
 * @return IoU score between the two bounding boxes.
 */
double calculateIoU(const BoundingBox& box1, const BoundingBox& box2) {
    // Calculate the overlap in the x and y directions
    int xOverlap = std::max(0, std::min(box1.x + box1.width, box2.x + box2.width) - std::max(box1.x, box2.x));
    int yOverlap = std::max(0, std::min(box1.y + box1.height, box2.y + box2.height) - std::max(box1.y, box2.y));

    // Calculate the intersection and union of the bounding boxes
    int intersection = xOverlap * yOverlap;
    int union_ = box1.width * box1.height + box2.width * box2.height - intersection;

    // Calculate and return the IoU score
    return static_cast<double>(intersection) / static_cast<double>(union_);
}

/**
 * @brief Calculates the Mean Intersection over Union (mIoU) between two segmentation masks.
 *
 * @param segmentation_mask Predicted segmentation mask.
 * @param ground_truth_mask Ground truth segmentation mask.
 *
 * @return Mean Intersection over Union (mIoU) score.
 */
double calculateMeanIoU(const cv::Mat& segmentation_mask, const cv::Mat& ground_truth_mask) {
    // Initialize IoU and total_pixels vectors for each class (0-5)
    std::vector<double> ious(6, 0.0); 
    std::vector<int> total_pixels(6, 0);

    // Calculate IoU and total_pixels for each pixel in the segmentation masks
    for (int i = 0; i < segmentation_mask.rows; ++i) {
        for (int j = 0; j < segmentation_mask.cols; ++j) {
            int label = static_cast<int>(segmentation_mask.at<uchar>(i, j));
            int gt_label = static_cast<int>(ground_truth_mask.at<uchar>(i, j));

            // Increment IoU if the predicted and ground truth labels match
            if (label == gt_label) {
                ious[label] += 1.0;
            }

            // Increment total_pixels for both predicted and ground truth labels
            total_pixels[label] += 1;
            total_pixels[gt_label] += 1;
        }
    }

    // Calculate and return the mIoU score
    double mIoU = 0.0;
    int num_valid_classes = 0;
    for (int i = 0; i < ious.size(); ++i) {
        // Only consider classes with non-zero total_pixels
        if (total_pixels[i] > 0) {
            double iou = ious[i] / (total_pixels[i] + (total_pixels[i] - ious[i]));
            mIoU += iou;
            num_valid_classes += 1;
        }
    }

    // Calculate mean IoU only if there are valid classes
    if (num_valid_classes > 0) {
        mIoU /= num_valid_classes;
    }

    return mIoU;
}
