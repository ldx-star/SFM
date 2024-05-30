//
// Created by ldx on 24-5-6.
//
#include "../include/utils.h"

std::vector<std::string> load_dirs(const std::string &dir, int num,std::string suffix) {
    std::vector<std::string> res;
    for (int i = 1; i <= num; i++) {
        res.push_back(dir + "/" + std::to_string(i) + suffix);
    }
    return res;
}

double reprojectError(const std::vector<cv::Point3f> &points_3d, const std::vector<cv::Point2f> &points_2d, const cv::Mat &K, const cv::Mat &dist,
                      const cv::Mat &R, const cv::Mat &T) {
    if (points_3d.size() != points_2d.size()) {
        std::cerr << "计算重投影误差错误，世界坐标与像素坐标数量不一致" << std::endl;
        exit(1);
    }
    cv::Mat rotation_matrix;
    cv::Rodrigues(R, rotation_matrix);
    double repError = 0;
    for (int i = 0; i < points_3d.size(); i++) {
        cv::Point3f p_3d = points_3d[i];
        cv::Point2f p_2d = points_2d[i];
        double X = p_3d.x;
        double Y = p_3d.y;
        double Z = p_3d.z;

        double x_cam = X * rotation_matrix.at<double>(0, 0) + Y * rotation_matrix.at<double>(0, 1) + Z * rotation_matrix.at<double>(0, 2) +
                       T.at<double>(0, 0);
        double y_cam = X * rotation_matrix.at<double>(1, 0) + Y * rotation_matrix.at<double>(1, 1) + Z * rotation_matrix.at<double>(1, 2) +
                       T.at<double>(1, 0);
        double z_cam = X * rotation_matrix.at<double>(2, 0) + Y * rotation_matrix.at<double>(2, 1) + Z * rotation_matrix.at<double>(2, 2) +
                       T.at<double>(2, 0);
        x_cam /= z_cam;
        y_cam /= z_cam;

        double r2 = x_cam * x_cam + y_cam * y_cam;

        double x = x_cam * (1 + r2 * dist.at<double>(0, 0) + r2 * r2 * dist.at<double>(0, 1));
        double y = x_cam * (1 + r2 * dist.at<double>(0, 0) + r2 * r2 * dist.at<double>(0, 1));


        double u = x * K.at<double>(0, 0) + y * K.at<double>(0, 1) + 1 * K.at<double>(0, 2);
        double v = x * K.at<double>(1, 0) + y * K.at<double>(1, 1) + 1 * K.at<double>(1, 2);

        repError += sqrt((p_2d.x - u) * (p_2d.x - u) + (p_2d.y - v) * (p_2d.y - v));

    }
    return repError;
}

bool is_includeNum(const std::string &line) {
    for (const auto &e: line)
        if (e >= '0' && e <= '9') {
            return true;
        }
    return false;
}

void delete_escape(std::string &line) {
    std::string str;
    for (auto e: line) {
        if (e != '\r' && e != '\n') {
            str += e;
        } else {
            break;
        }
    }
    line = str;
}

std::string extract_num(const std::string &str) {
    std::string data;
    for (auto e: str) {
        if (e >= '0' && e <= '9' || e == '.' || e == '-' || e == 'e') {
            data += e;
        }
    }
    return data;
}
std::string get_time(){
    auto now = std::chrono::system_clock::now();
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);
    std::tm* local_time = std::localtime(&now_time_t);
    char buffer[80];
    std::strftime(buffer,sizeof buffer,"%Y-%m-%d %H:%M:%S", local_time);

    return std::string(buffer);
}