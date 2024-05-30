//
// Created by ldx on 24-5-18.
//

#include "../include/Blog.h"

Blog::Blog() {
    std::string path = "../out/blogs";
    std::string file_name = get_time();
    std::string dir = path + "/" + file_name;
    std::string img_dir = dir + "/images";
    _points_dir = dir ;
    std::filesystem::create_directory(dir);
    std::filesystem::create_directory(img_dir);
    _image_dir = img_dir;

    _file = std::ofstream(dir + "/" + file_name + ".txt", std::ios::out);
    if (!_file.is_open()) {
        std::cerr << "文件无法创建或打开" << std::endl;
        exit(1);
    }
}

Blog::~Blog() {

    _file.close();
}

void Blog::writeImg(const cv::Mat &img, const std::string &path) {
    cv::imwrite(_image_dir + "/" + path, img);
}
