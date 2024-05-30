//
// Created by ldx on 24-5-8.
//

#ifndef SFM_CALIBRATION_H
#define SFM_CALIBRATION_H
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

class Calibration {
public:
    std::vector<cv::Mat> _chessBoard_imgs;
    int _rows;//棋盘格行数
    int _cols;//棋盘格列数
    double _size;//每个格子大小(单位：m)
    bool _debug;

public:
    Calibration(const std::vector<std::string> &img_dirs,int rows,int cols,double size,bool debug);
    void operation(cv::Mat &K,cv::Mat &dist,double &repError);
    ~Calibration(){};
};

#endif //SFM_CALIBRATION_H
