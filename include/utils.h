//
// Created by ldx on 24-5-6.
//

#ifndef SFM_UTILS_H
#define SFM_UTILS_H

#endif //SFM_UTILS_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <chrono>
#include <ctime>
#include <string>
/**
 * 加载图片路径
 * @param num 加载的数量
 * @return
 */
std::vector<std::string> load_dirs(const std::string &dir, int num,std::string suffix);

/**
 * \decription 计算从投影误差(将三维点，通过标定参数映射成二维点)
 * @param points_3d 世界坐标
 * @param points_2d 像素坐标
 * @param K 内参矩阵
 * @param dist 畸变系数
 * @param R 旋转向量
 * @param T 平移矩阵
 * @return 重投影误差
 */
double reprojectError(const std::vector<cv::Point3f> &points_3d, const std::vector<cv::Point2f> &points_2d,const cv::Mat &K,const cv::Mat &dist, const cv::Mat &R,const cv::Mat & T);

/**
 * \decription 判断字符串中是否存在数字
 * @param line
 * @return
 */
bool is_includeNum(const std::string &line);

/**
 * \decription 删除字符串中的转义字符
 * @param line
 */
void delete_escape(std::string &line);

/**
 * \discription 从字符串中提取数字
 * @param str
 * @return
 */
std::string extract_num(const std::string &str);

/**
 * \descrition 获取当前时间
 * @return
 */
std::string get_time();