//
// Created by ldx on 24-5-8.
//

#ifndef SFM_CAMERA_H
#define SFM_CAMERA_H

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

class Camera {
public:
    Camera() {};

    ~Camera() {};

    Camera(const cv::Mat &K, const cv::Mat &dist);

public:
    Eigen::Matrix3d _K;// 内参矩阵
    Eigen::VectorXd _dist = Eigen::VectorXd::Zero(5);// 畸变系数
    Eigen::Matrix3d _R = Eigen::Matrix3d::Zero();// 旋转矩阵
    Eigen::Vector3d _t = Eigen::Vector3d::Zero();// 平移矩阵
};


#endif //SFM_CAMERA_H
