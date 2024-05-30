//
// Created by ldx on 24-5-6.
//

#ifndef SFM_COMMONVIEW_H
#define SFM_COMMONVIEW_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "../include/Blog.h"
struct Track{
    int img_id;
    int keyPoint_id;
};
struct Edge{
    bool flag;
    std::vector<cv::DMatch> matches;// 匹配情况

};
struct Node{
    cv::Mat img;
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptors;
    std::vector<Edge> edges;
    std::vector<int> trick_id;
};



class CommonView {
public:
    CommonView(std::vector<std::string> images_dirs,bool debug,const cv::Mat &K,const cv::Mat &dist,Blog *blog);
    CommonView(){};
    ~CommonView(){};

private:
    void create_graph();
    void create_tracks();

    /**
     * \根据前驱节点将后继节点插入到tracks中
     * @param predecessor 前驱节点
     * @param successor 后继节点
     */
    void track_insert(std::pair<int,int> predecessor, std::pair<int,int> successor);

public:
    std::vector<Node> _graph;
    std::vector<cv::Mat> _images;
    std::vector<std::list<std::pair<int,int>>> _tracks;

private:
    bool _debug;
    Blog* _blog;
    int _image_nums = 0; //图片数量
};

#endif //SFM_COMMONVIEW_H
