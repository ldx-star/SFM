//
// Created by ldx on 24-5-6.
//
#include "../include/CommonView.h"

CommonView::CommonView(std::vector<std::string> images_dirs, bool debug, const cv::Mat &K, const cv::Mat &dist, Blog *blog) {
    _debug = debug;
    _image_nums = images_dirs.size();
    _blog = blog;
    std::vector<cv::Mat> &images = _images;
    images.resize(_image_nums * 1.5);
    //存放图片的小标从1开始
    for (int i = 0, k = 1; i < images_dirs.size(); i++) {
        cv::Mat img = cv::imread(images_dirs[i]);
        if (img.empty()) {
            std::cerr << "共视图初始化时，图片读取失败" << std::endl;
            exit(1);
        }
//        cv::Mat undistImg;
//        cv::undistort(img, undistImg, K,dist);
//        cv::imshow("Distorted Image", img);
//        cv::imshow("Undistorted Image", undistImg);
//
//        cv::waitKey(0);
        images[k++] = img;
    }
    create_graph();
    create_tracks();
}

void CommonView::create_graph() {
    bool debug = _debug;
    int image_nums = _image_nums;
    std::vector<cv::Mat> &images = CommonView::_images;
    std::vector<Node> &graph = _graph;
    graph.resize(image_nums * 1.5); // 每个节点最多images.size()-1

    //建立节点
    for (int i = 1; i <= image_nums; i++) {
        cv::Mat img = images[i];
        graph[i].img = img;
        std::vector<cv::KeyPoint> keyPoints;
        cv::Mat descriptors;
        cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
        sift->detectAndCompute(img, cv::noArray(), keyPoints, descriptors);
        graph[i].descriptors = descriptors;
        graph[i].keyPoints = keyPoints;
    }
    //建立边 edge[i][j] = edge[j][i]
    for (int i = 1; i <= image_nums; i++) {
        graph[i].edges.resize(image_nums * 1.5);
        for (int j = i + 1; j <= image_nums; j++) {
            cv::Mat img1 = graph[i].img;
            cv::Mat img2 = graph[j].img;
            std::vector<cv::KeyPoint> keyPoints1 = graph[i].keyPoints;
            cv::Mat descriptors1 = graph[i].descriptors;
            std::vector<cv::KeyPoint> keyPoints2 = graph[j].keyPoints;
            cv::Mat descriptors2 = graph[j].descriptors;
            std::vector<cv::DMatch> good_matches;
            cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
            std::vector<std::vector<cv::DMatch>> knn_matches;
            const float ratio_thresh = 0.7f;
            matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);
            for (auto &knn_match: knn_matches) {
                if (knn_match[0].distance < ratio_thresh * knn_match[1].distance) {
                    good_matches.push_back(knn_match[0]);
                }
            }
            graph[i].edges[j] = {true, good_matches};
            //显示匹配结果
//            cv::Mat img_matches;
//            cv::drawMatches(img1, keyPoints1, img2, keyPoints2, good_matches, img_matches, cv::Scalar::all(-1),
//                            cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//            cv::namedWindow("knn_matches", cv::WINDOW_NORMAL);
//            cv::resizeWindow("knn_matches", img1.rows / 2, img2.cols / 2);
//            cv::imshow("knn_matches", img_matches);
//            cv::waitKey(0);
        }
    }
}

void CommonView::create_tracks() {
    int images_num = _image_nums;
    for (int i = 1; i <= images_num; i++) {
        for (int j = i + 1; j <= images_num; j++) {
            _graph[i].trick_id.resize(_graph[i].keyPoints.size());
            _graph[j].trick_id.resize(_graph[j].keyPoints.size());
            Edge &edge = _graph[i].edges[j];
            const std::vector<cv::DMatch> &matches = edge.matches;
            for (int k = 0; k < matches.size(); k++) {
                const cv::DMatch &match = matches[k];
                int queryIdx = match.queryIdx; // 第一幅图的特征点索引
                int trainIdx = match.trainIdx; // 第二幅图的特征点索引
                std::pair<int, int> predecessor = {i, queryIdx}; // 前驱节点
                std::pair<int, int> successor = {j, trainIdx}; // 后继节点
                track_insert(predecessor, successor);
            }
        }
    }
}

void CommonView::track_insert(std::pair<int, int> predecessor, std::pair<int, int> successor) {
    int id1 = predecessor.first;
    int id2 = successor.first;
    int kp_id1 = predecessor.second;
    int kp_id2 = successor.second;

    bool flag = false;
    bool exist = false;
    int track = -1;
    for (int i = 0; i < _tracks.size(); i++) {
        if (!exist || !flag) {
            for (std::pair<int, int> e: _tracks[i]) {
                if (e.first == id2 && e.second == kp_id2) {
                    exist = true;
                }
                if (e.first == id1 && e.second == kp_id1) {
                    flag = true;
                    track = i;
                }
            }
        }
    }
    if (exist) {
        return;
    }
    if (!exist && flag) {
        _tracks[track].push_back(successor);
        _graph[id2].trick_id[kp_id2] = track;
        return;
    }
    std::list<std::pair<int, int>> my_list;
    my_list.push_back(predecessor);
    my_list.push_back(successor);
    _tracks.push_back(my_list);
    _graph[id1].trick_id[kp_id1] = _tracks.size() - 1;
    _graph[id2].trick_id[kp_id2] = _tracks.size() - 1;

}
