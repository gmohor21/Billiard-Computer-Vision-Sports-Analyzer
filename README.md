# Billiard-Computer-Vision-Sports-Analyzer

A computer vision system for analyzing billiard game events, tracking ball positions, and visualizing game state.


## Project Status

⚠️ **Note: This project is currently under active development. The main branch is protected, and all changes are being made on the `dev/billiard-sports-analyzer-src-codes` branch.**

## Description

BilliardVisionAnalyzer is an advanced computer vision application designed to analyze video footage of "Eight Ball" billiard game events. The system can recognize and localize balls inside the playing field, detect field boundaries, segment the playing area into different categories (cue ball, 8-ball, solid balls, striped balls, and playing field), and represent the current state of the game in a 2D top-view visualization map, including ball positions and trajectories.

Key features:
- Ball detection and localization
- Playing field boundary detection
- Ball and field segmentation
- 2D top-view visualization with ball trajectories
- Performance evaluation using mAP (mean Average Precision) and mIoU (mean Intersection over Union)

## Prerequisites

Before you begin, ensure you have met the following requirements:
- `C++` Compiler (supporting C++11 or later)
- `CMake` (version 3.10 or higher)
- `OpenCV` library (version 4.9.0 or higher)

## Installation

To install BilliardVisionAnalyzer, follow these steps:

1. Clone the repository:
   `git clone https://github.com/YourUsername/BilliardVisionAnalyzer.git`
2. Navigate to the project directory:
   `cd BilliardVisionAnalyzer`
3. Create a build directory:
   `mkdir build`
   `cd build`
4. Configure the project with CMake:
   `cmake ..`
5. Build the project:
   `cmake --build .`

## Usage

To use BilliardVisionAnalyzer, follow these steps:

1. Prepare your dataset:
- Organize your billiard game video clips in a directory.
- Ensure you have ground truth annotations (bounding boxes and segmentation masks) for the first and last frames of each video clip.
2. Run the executable:
   `./BilliardAnalysis /path/to/your/dataset`
3. The program will process each video clip and generate:
- Console output with mAP and mIoU metrics.
- Output videos with top-view visualization superimposed on the input video.
- Output images (bounding boxes and segmentation masks) for the first and last frames.

## Contributing

To contribute to BilliardVisionAnalyzer, follow these steps:

1. Fork this repository.
2. Create a new branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively, see the GitHub documentation on [creating a pull request](https://help.github.com/articles/creating-a-pull-request/).

## License

This project uses the following license: [MIT License](<link_to_license>).
