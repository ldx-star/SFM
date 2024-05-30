cmake_minimum_required(VERSION 3.16)
project(SFM)
set(CMAKE_CXX_STANDARD 17)

add_library(lib src/utils.cpp src/CommonView.cpp src/Calibration.cpp src/Blog.cpp src/Camera.cpp src/Rebuild.cpp)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})


add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} lib ${OpenCV_LIBS})