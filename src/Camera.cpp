//
// Created by ldx on 24-5-8.
//

#include "../include/Camera.h"


Camera::Camera(const cv::Mat &K, const cv::Mat &dist) {
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            _K(i,j) = K.at<float>(i,j);
        }
    }
    for(int i = 0; i < 5 ;i++){
        _dist(i) = dist.at<float>(0,i);

    }
}
