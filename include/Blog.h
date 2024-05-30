//
// Created by ldx on 24-5-18.
//

#ifndef SFM_BLOG_H
#define SFM_BLOG_H
#include <fstream>
#include <string>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include <filesystem>

class Blog {
public:
    Blog();
    ~Blog();

    template<typename T>
    void write(const T &content){
        _file << content << std::endl;
        _file.flush();
    }

    void writeImg(const cv::Mat &img,const std::string &path);

public:
    std::string _points_dir;
private:
    std::ofstream _file;
    std::string _image_dir;
};


#endif //SFM_BLOG_H
