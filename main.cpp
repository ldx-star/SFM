//
// Created by ldx on 24-5-6.
//
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "./include/utils.h"
#include "./include/CommonView.h"
#include "./include/Calibration.h"
#include "./include/Rebuild.h"
#include "./include/Camera.h"
#include "./include/Blog.h"


int main() {
    Blog blog;
    bool debug = false;
    std::string out_dir = "../out";
    bool is_calib = false;
    /**相机标定**/
    cv::Mat K;//内参矩阵
    cv::Mat dist;//畸变系数
    if (is_calib) {
        std::cout << "===============相机标定===============" << std::endl;
        std::string chessBoard_dir = "../chessBoard_image3";
        int chessBoard_num = 15;
        std::vector<std::string> chessBoard_dirs = load_dirs(chessBoard_dir, chessBoard_num, ".jpg");
        Calibration calibration(chessBoard_dirs, 5,7, 0.010, debug);
        double repError = 0;
        calibration.operation(K, dist, repError);


        // 保存参数
        std::ofstream outfile(out_dir + "/calib_parameter.txt", std::ios::out); //std::ios::out 如果没有就创建该文件
        if (!outfile.is_open()) {
            std::cerr << "文件无法创建或打开" << std::endl;
            exit(1);
        }
        outfile << "K:" << std::endl;
        outfile << K << std::endl;
        outfile << "dist:" << std::endl;
        outfile << dist << std::endl;
        outfile << "Reproject Error:" << std::endl;
        outfile << repError << std::endl;
        outfile.close();
    } else {
        std::cout << "===============加载相机参数===============" << std::endl;
        // 读取参数
        std::ifstream infile("../chessBoard_image1/calib_parameter.txt");
        if (!infile.is_open()) {
            std::cerr << "文件读取失败" << std::endl;
            exit(1);
        }
        std::string title;
        std::string data;
        std::string line;
        std::vector<float>numbers;

        while(!infile.eof()){
            std::getline(infile,line);
            //linux 下会读回车（\r）而windows却不会
            //为了实现跨平台编译，需统一格式
            delete_escape(line);
            if(is_includeNum(line)){
                data = line;
            }else{
                if(!numbers.empty()){
                    if(title == "K:"){
                        cv::Mat mat(3,3,CV_32F,numbers.data());
                        K = mat.clone();
                    }else if(title == "dist:"){
                        cv::Mat mat(1,5,CV_32F,numbers.data());
                        dist = mat.clone();
                    }
                }
                title = line;
                numbers.clear();
                continue;
            }
            std::istringstream iss(data); // 使用字符串流读取输入
            std::string word;
            while (iss >> word) {
                // 尝试将每个字符串转换为数值
                std::istringstream wordIss(word);

                std::string num = extract_num(word);
                float number = std::stod(num);
                numbers.emplace_back(number);
            }
        }
        blog.write("====================加载相机参数==========================");
        blog.write("K:");
        blog.write(K);
        blog.write("dist:");
        blog.write(dist);
        std::cout << "K:" << K << std::endl;
        std::cout << "dist:" << dist << std::endl;
        infile.close();
    }



    /**创建共视图**/
    std::cout << "===============创建共视图===============" << std::endl;
    blog.write("====================创建共视图==========================");

    std::string images_dir = "../data";
    int images_num = 8;
    std::vector<std::string> images_dirs = load_dirs(images_dir, images_num, ".jpg");
    CommonView commonView(images_dirs, debug,K,dist,&blog);
    blog.write("tracks num:");
    blog.write(std::to_string(commonView._tracks.size()));
    std::cout << "===============重建===============" << std::endl;
    blog.write("====================重建==========================");

    // 选取匹配点最多的两组图作为初始的重建结果
    std::vector<Camera>  cameras(int(images_num*1.5));
    // 每个相机的内参是一样的
    for(int i = 1; i <= images_num;i++){
        Camera camera(K,dist);
        cameras[i] = camera;
    }

    Rebuild rebuild(commonView, cameras,images_num,&blog);
    rebuild.save(blog._points_dir);
    return 0;
}