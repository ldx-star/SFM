//
// Created by ldx on 24-5-8.
//

#include "../include/Calibration.h"
#include "../include/utils.h"

Calibration::Calibration(const std::vector<std::string> &img_dirs, int rows, int cols, double size, bool debug) {
    std::vector<cv::Mat> &chessBoard_imgs = _chessBoard_imgs;
    for (int i = 0; i < img_dirs.size(); i++) {
        cv::Mat img = cv::imread(img_dirs[i]);
        if (img.empty()) {
            std::cerr << "相机标定始化时，图片读取失败" << std::endl;
            exit(1);
        }
        chessBoard_imgs.push_back(img);
    }
    _rows = rows;
    _cols = cols;
    _size = size;
    _debug = debug;
}

void Calibration::operation(cv::Mat &K, cv::Mat &dist,double &repError) {
    const int &rows = _rows;
    const int &cols = _cols;
    const double &size = _size;
//    const bool &debug = _debug;
    const bool &debug = _debug;

    const std::vector<cv::Mat> &images = _chessBoard_imgs;
    std::vector<cv::Point2f> corner;//存储角点
    std::vector<std::vector<cv::Point2f>> corners;//存储角点
    //世界坐标
    std::vector<cv::Point3f> points_world;
    std::vector<std::vector<cv::Point3f>> points_worlds;
    cv::Mat img, gray;
    for (int i = 0; i < images.size(); i++) {
        points_world.clear();
        corner.clear();
        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < cols; k++) {
                points_world.push_back({static_cast<float>(j * size), static_cast<float>(k * size), 0});
            }
        }
        img = images[i];
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        //角点检测
        bool success = cv::findChessboardCorners(gray, cv::Size(cols, rows), corner,
                                                 cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
        if (success) {
            // 求亚像素精度
            std::cout << "角点提出成功：" << i+1 << std::endl;
            cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::Type::MAX_ITER, 30, 0.001);
            cv::cornerSubPix(gray, corner, cv::Size(11, 11), cv::Size(-1, -1), criteria);
            corners.push_back(corner);
            points_worlds.push_back(points_world);
            if (debug) {
                // 显示角点
                cv::drawChessboardCorners(img, cv::Size(cols, rows), corner, success);
                cv::namedWindow("Corner", cv::WINDOW_NORMAL);
                cv::resizeWindow("Corner", img.rows / 2, img.cols / 2);
                cv::imshow("Corner", img);
                cv::waitKey(0);
            }
        }
    }
    cv::destroyAllWindows();
    std::vector<cv::Mat> R, T;
    cv::calibrateCamera(points_worlds, corners, cv::Size(gray.rows, gray.cols), K, dist, R, T, cv::CALIB_FIX_K3 | cv::CALIB_ZERO_TANGENT_DIST);
    // 内参矩阵
    std::cout << "cameraMatrix : " << K << std::endl;
    // 透镜畸变系数
    std::cout << "distCoeffs : " << dist << std::endl;
    double Error = 0;
    int count = 0;
    for(int i = 0 ; i < points_worlds.size();i++){
//        preError += reprojectError(points_worlds[i],corners[i],K,dist,R[i],T[i]);
        std::vector<cv::Point2f> p_2d;
        cv::projectPoints(points_worlds[i],R[i],T[i],K,dist,p_2d);
        for(int j = 0; j < p_2d.size(); j++){
            Error += sqrt((p_2d[j].x - corners[i][j].x)*(p_2d[j].x - corners[i][j].x)+(p_2d[j].y - corners[i][j].y)*(p_2d[j].y - corners[i][j].y));
            count++;
        }
    }
    repError = Error / count;
    std::cout << "重投影误差 : " << repError << std::endl;

}
