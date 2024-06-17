# This CMakeLists.txt file builds this project.

# Specify the minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(BilliardAnalysis)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Collect all source files
file(GLOB_RECURSE SOURCE_FILES "${PROJECT_SOURCE_DIR}/src/*.cpp")

# Add executable target
add_executable(BilliardAnalysis src/main.cpp ball_detection.cpp field_detection.cpp segmentation.cpp visualization.cpp evaluation.cpp output_data.cpp)

# Link OpenCV libraries
target_link_libraries(BilliardAnalysis ${OpenCV_LIBS})
