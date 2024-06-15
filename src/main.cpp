/**
 * @file main.cpp
 * @brief Main entry point for the Billiard Analysis application.
 * 
 * This file contains the main function that orchestrates the billiard video analysis process.
 * It loads the dataset, processes each video, and generates output data and visualizations.
 * 
 * @author Srobona Ghosh
 * @date 2024-06-10
 */

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <filesystem>
namespace fs = std::filesystem;

#include "ball_detection.hpp"
#include "field_detection.hpp"
#include "segmentation.hpp"
#include "visualization.hpp"
#include "evaluation.hpp"
#include "output_data.hpp"

/**
 * @brief Load dataset paths from the given directory.
 *
 * This function scans the dataset directory and collects paths to video clips,
 * bounding box annotations, and segmentation masks.
 *
 * @param dataset_path Path to the dataset directory.
 * @return std::vector<std::tuple<std::string, std::string, std::string>> Vector of tuples containing video, bounding box, and mask paths.
 */

std::vector<std::tuple<std::string, std::string, std::string>> loadDatasetPaths(const std::string& dataset_path) {
    std::vector<std::tuple<std::string, std::string, std::string>> dataset_paths;

    for (const auto& entry : fs::directory_iterator(dataset_path)) {
        if (entry.is_directory()) {
            std::string clip_path = entry.path().string();
            std::string video_path = clip_path + "/clip.mp4";
            std::string ground_truth_path = clip_path;

            if (fs::exists(video_path) && fs::exists(ground_truth_path + "/bounding_boxes") && fs::exists(ground_truth_path + "/masks")) {
                dataset_paths.emplace_back(video_path, ground_truth_path + "/bounding_boxes", ground_truth_path + "/masks");
            } else {
                std::cerr << "Warning: Skipping directory " << clip_path << " due to missing files." << std::endl;
            }
        }
    }

    return dataset_paths;
}

int main(int argc, char** argv) {
    // Check if dataset path is provided
    if (argc < 2) {
        std::cout << "Usage: BilliardAnalysis <dataset_path>" << std::endl;
        return 1;
    }

    std::string dataset_path = "model_path";
    std::vector<std::tuple<std::string, std::string, std::string>> dataset_paths = loadDatasetPaths(dataset_path);

    // Process each video in the dataset
    for (const auto& [video_path, bbox_path, mask_path] : dataset_paths) {
        // Create output directory for the current video
        std::string output_dir = fs::path(video_path).stem().string();
        fs::create_directory(output_dir);

        // Open the video file
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "Error: Failed to open video file: " << video_path << std::endl;
            continue;
        }

        // Initialize the ball detection model
        BallDetector ball_detector("model_path");

        // Process the video clip
        int frame_count = 0;
        std::vector<cv::Mat> trajectories;
        double total_mAP = 0.0;
        double total_mIoU = 0.0;

        // Load the first frame
        cv::Mat first_frame;
        cap >> first_frame;
        if (first_frame.empty()) {
            std::cerr << "Error: Failed to load first frame from video: " << video_path << std::endl;
            continue;
        }

        // Load ground truth data for the first frame
        std::vector<BoundingBox> first_frame_ground_truth_boxes = loadGroundTruthBoundingBoxes(bbox_path, 0);
        cv::Mat first_frame_ground_truth_mask = cv::imread(mask_path + "/frame_0.png", cv::IMREAD_GRAYSCALE);
        if (first_frame_ground_truth_mask.empty()) {
            std::cerr << "Error: Failed to load ground truth mask for first frame from: " << mask_path << "/frame_0.png" << std::endl;
            continue;
        }

        // Process the first frame
        // Detect balls in the first frame        
        std::vector<BoundingBox> first_frame_ball_boxes = ball_detector.detect(first_frame);
        double first_frame_mAP = evaluateMeanAveragePrecision(first_frame_ball_boxes, first_frame_ground_truth_boxes, 0);
        total_mAP += first_frame_mAP;

        // Detect field lines and get the bounding rectangle
        std::vector<cv::Vec4i> first_frame_field_lines = detectFieldLines(first_frame);
        cv::Rect first_frame_field_rect = getBoundingRect(first_frame_field_lines);


        // Segment balls and field in the first frame
        cv::Mat first_frame_segmentation_mask = segmentBallsAndField(first_frame, first_frame_ball_boxes, first_frame_field_rect);
        double first_frame_mIoU = evaluateMeanIntersectionOverUnion(first_frame_segmentation_mask, first_frame_ground_truth_mask, 0);
        total_mIoU += first_frame_mIoU;

        // Process the remaining frames
        while (true) {
            cv::Mat frame;
            cap >> frame;
            if (frame.empty())
                break;

            // Step 1: Ball detection and localization
            std::vector<BoundingBox> ball_boxes = ball_detector.detect(frame);
            double mAP = evaluateMeanAveragePrecision(ball_boxes, bbox_path, frame_count + 1);
            total_mAP += mAP;

            // Step 2: Playing field detection
            std::vector<cv::Vec4i> field_lines = detectFieldLines(frame);
            cv::Rect field_rect = getBoundingRect(field_lines);

            // Step 3: Ball and playing field segmentation
            cv::Mat segmentation_mask = segmentBallsAndField(frame, ball_boxes, field_rect);
            double mIoU = evaluateMeanIntersectionOverUnion(segmentation_mask, mask_path, frame_count + 1);
            total_mIoU += mIoU;

            // Step 4: 2D top-view visualization and ball trajectory tracking
            cv::Mat top_view = visualizeTopView(frame, segmentation_mask, field_rect);
            trajectories = trackBallTrajectories(trajectories, segmentation_mask, frame_count);

            // Display the result
            cv::imshow("Top View", top_view);
            cv::waitKey(1);

            frame_count++;
        }

        // Calculate average metrics
        double avg_mAP = total_mAP / (frame_count + 1);
        double avg_mIoU = total_mIoU / (frame_count + 1);
        std::cout << "Average mAP: " << avg_mAP << std::endl;
        std::cout << "Average mIoU: " << avg_mIoU << std::endl;

        // Save the output videos with top-view and trajectories
        saveOutputVideos(trajectories, video_path, output_dir);

        // Save output images, bounding boxes, and segmentation masks
        saveOutputData(output_dir, first_frame, first_frame_ball_boxes, first_frame_segmentation_mask, first_frame_ground_truth_boxes, first_frame_ground_truth_mask);
        saveOutputData(output_dir, cap.get(cv::CAP_PROP_POS_FRAMES), ball_boxes, segmentation_mask, loadGroundTruthBoundingBoxes(bbox_path, frame_count + 1), cv::imread(mask_path + "/frame_" + std::to_string(frame_count + 1) + ".png", cv::IMREAD_GRAYSCALE));
    }

    return 0;
}
