//
// Created by ldx on 24-5-11.
//

#include "../include/Rebuild.h"
#include "../include/SnavelyReprojectionError.h"

Rebuild::Rebuild(const CommonView &commonView, const std::vector<Camera> &cameras, int n, Blog *blog) {
    _commonView = commonView;
    _images_num = n;
    _cameras = cameras;
    _blog = blog;
    _tracks_state.resize(commonView._tracks.size());
    for (int i = 0; i < _tracks_state.size(); i++) {
        _tracks_state[i] = false;
    }
    _cameras_state.resize(n + 1);
    for (int i = 0; i < _cameras_state.size(); i++) {
        _cameras_state[i] = false;
    }
    init();
}


void Rebuild::init() {
    const int images_num = _images_num;
    CommonView &commonView = _commonView;
    std::vector<Node> &graph = commonView._graph;
    Blog *blog = _blog;
    auto &points_cloud = _points_cloud;
    auto &cameras = _cameras;
    auto &tracks = _commonView._tracks;


    // 寻找交集最多的两幅视图
    int id1, id2;
    std::vector<cv::DMatch> success_matches;
    for (int i = 1; i <= images_num; i++) {
        for (int j = i + 1; j <= images_num; j++) {
            std::vector<cv::DMatch> temp_matches = get_success_matches(i, j);
            if (temp_matches.size() > success_matches.size()) {
                success_matches = temp_matches;
                id1 = i;
                id2 = j;
            }
        }
    }

    blog->write("初始匹配视图: " + std::to_string(id1) + "  " + std::to_string(id2));
    //重建

    init_reconstruct(id1, id2, success_matches);
    this->save(blog->_points_dir);
    _cameras_state[id1] = true;
    _cameras_state[id2] = true;
    bool flag = false;
    do {
        this->save(blog->_points_dir);
        flag = false;
        // 从以及重建的点中找track最多的视图
        int *views = new int[images_num + 1]; // 每个视图中已经重建的点数
        memset(views, 0, sizeof views[0] * (images_num + 1));
        for (int i = 0; i < _points_state.size(); i++) {
            int track = _points_state[i];
            _tracks_state[track] = true;
            for (auto e: tracks[track]) {
                int id = e.first;
                views[id]++;
            }
        }
        int max_val = 0, max_id = 0;
        for (int i = 1; i <= images_num; i++) {
            if (views[i] > max_val && _cameras_state[i] == false) {
                max_id = i;
                max_val = views[i];
            }
        }
        _cameras_state[max_id] = true;
        /** 用ePnP求解相机(max_id)的外参 **/
        // 建立3D-2D匹配点
        std::vector<Eigen::Vector3d> p3ds;
        std::vector<Eigen::Vector2d> p2ds;
        for (int i = 0; i < _points_state.size(); i++) {
            int track = _points_state[i];
            for (auto e: tracks[track]) {
                int id = e.first;
                int kp_id = e.second;
                if (id == max_id) {
                    Eigen::Vector3d p3d = _points_cloud[i];
                    Eigen::Vector2d p2d;
                    p2d(0) = graph[max_id].keyPoints[kp_id].pt.x;
                    p2d(1) = graph[max_id].keyPoints[kp_id].pt.y;
                    p3ds.push_back(p3d);
                    p2ds.push_back(p2d);
                }
            }
        }
        ePnP(p3ds, p2ds, cameras[max_id]);
        /**用max_id与其他可用相机进行重建**/
        for (int i = 1; i <= images_num; i++) {
            id1 = i;
            id2 = max_id;
            if (id1 > id2) {
                int tmp = id1;
                id1 = id2;
                id2 = tmp;
            }

            if (_cameras_state[i] && graph[id1].edges[id2].flag == true) {
                success_matches = get_success_matches(id1, id2);
                if (success_matches.size() > 100) {
                    reconstruct(max_id, i, success_matches);
                    /**捆绑调整**/
                    ceres_bundle_adjustment();
                    flag = true;
                }
            }
        }

    } while (flag);
//    ceres_bundle_adjustment();
}


void Rebuild::calc_fundamental(const std::vector<Eigen::Vector3d> &points1, const std::vector<Eigen::Vector3d> &points2, Eigen::MatrixXd &F) {
    Eigen::MatrixXd W(points1.size(), 9);
    for (int i = 0; i < points1.size(); i++) {
        double u1 = points1[i](0);
        double v1 = points1[i](1);
        double u2 = points2[i](0);
        double v2 = points2[i](1);

        W(i, 0) = u1 * u2;
        W(i, 1) = v1 * u2;
        W(i, 2) = u2;
        W(i, 3) = u1 * v2;
        W(i, 4) = v1 * v2;
        W(i, 5) = v2;
        W(i, 6) = u1;
        W(i, 7) = v1;
        W(i, 8) = 1;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd1(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd vv = svd1.matrixV();
    Eigen::VectorXd f = vv.col(8);
    F << f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8];

    //奇异值约束
    Eigen::JacobiSVD<Eigen::MatrixXd> svd2(F, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::MatrixXd U, V;
    U = svd2.matrixU();
    V = svd2.matrixV();
    Eigen::VectorXd v_S = svd2.singularValues();
    Eigen::Matrix3d S = Eigen::Matrix3d::Zero();

    S(0, 0) = v_S(0);
    S(1, 1) = v_S(1);
    S(2, 2) = 0;
    F = U * S * V.transpose();
}

std::vector<int>
Rebuild::find_inters(const std::vector<Eigen::Vector3d> &p1, const std::vector<Eigen::Vector3d> &p2, const Eigen::Matrix<double, 3, 3> &F,
                     const double inter_thresh) {
    const double squared_thresh = inter_thresh * inter_thresh;
    std::vector<int> inters;
    for (int i = 0; i < p1.size(); i++) {
        std::pair<Eigen::Vector3d, Eigen::Vector3d> pair_point = {p1[i], p2[i]};
        double error = calc_sampson_distance(F, pair_point);
        if (error < squared_thresh) {
            inters.push_back(i);
        }
    }
    return inters;
}

double Rebuild::calc_sampson_distance(const Eigen::Matrix<double, 3, 3> &F, const std::pair<Eigen::Vector3d, Eigen::Vector3d> &pair_point) {
    auto p1 = pair_point.first;
    auto p2 = pair_point.second;
    double p2Fp1 = 0.0;
    p2Fp1 += ((p2(0) * F(0, 0) + p2(1) * F(1, 0) + 1 * F(2, 0)) * p1(0)) + ((p2(0) * F(0, 1) + p2(1) * F(1, 1) + 1 * F(2, 1)) * p1(1)) +
             ((p2(0) * F(0, 2) + p2(1) * F(1, 2) + 1 * F(2, 2)) * 1);
    p2Fp1 *= p2Fp1;

    double sum = 0;
    double Fp1_x_2 = pow(F(0, 0) * p1(0) + F(0, 1) * p1(1) + F(0, 2) * 1, 2);
    double Fp1_y_2 = pow(F(1, 0) * p1(0) + F(1, 1) * p1(1) + F(1, 2) * 1, 2);
    double p2tF_x_2 = pow(F(0, 0) * p2(0) + F(1, 0) * p2(1) + F(2, 0) * 1, 2);
    double p2tF_y_2 = pow(F(0, 1) * p2(0) + F(1, 1) * p2(1) + F(2, 1) * 1, 2);
    sum += Fp1_x_2 + Fp1_y_2 + p2tF_x_2 + p2tF_y_2;
    return p2Fp1 / sum;
}

void Rebuild::calc_cam_pose(const Eigen::Matrix<double, 3, 3> &F, Camera &camera1, Camera &camera2, const std::vector<Eigen::Vector3d> &points1,
                            const std::vector<Eigen::Vector3d> &points2) {
    Eigen::Matrix3d R_1;
    Eigen::Vector3d t_1;
    Eigen::MatrixXd K1 = camera1._K;
    Eigen::MatrixXd K2 = camera2._K;

    // 计算本质矩阵
    Eigen::Matrix3d E = K2.transpose() * F * K1;
    //验证本质矩阵
//    {
//        std::cout << "E:" << std::endl << E << std::endl;
//        for(int i = 0; i < points1.size();i++){
//            Eigen::Vector3d p1 = points1[i];
//            Eigen::Vector3d p2 = points2[i];
//            auto tmp = p2.transpose() * E *p1;
//            std::cout << "tmp:" << std::endl << tmp << std::endl;
//        }
//    }


    Eigen::JacobiSVD<Eigen::Matrix3d> svd_E(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto U = svd_E.matrixU();
    auto V = svd_E.matrixV();

    //定义W和Z
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    W(0, 1) = -1, W(1, 0) = 1, W(2, 2) = 1;
    Eigen::Matrix3d Z = Eigen::Matrix3d::Zero();
    Z(0, 1) = -1, Z(1, 0) = 1;

    Eigen::Matrix3d UWtVt = U * W.transpose() * V.transpose();
    Eigen::Matrix3d UWVt = U * W * V.transpose();

    Eigen::Matrix3d R1 = UWVt.determinant() * UWVt;
    Eigen::Matrix3d R2 = UWtVt.determinant() * UWtVt;
    Eigen::Vector3d t1 = U.col(2);
    Eigen::Vector3d t2 = -U.col(2);

    std::vector<std::pair<Eigen::Matrix3d, Eigen::Vector3d>> poses(4);
    poses[0] = {R1, t1};
    poses[1] = {R1, t2};
    poses[2] = {R2, t1};
    poses[3] = {R2, t2};

    //判断位姿是否合理
    int min_count = INT_MAX;
    int min_idx = 0;
    for (int i = 0; i < 4; i++) {
        Camera tmp_camera2 = camera2;
        tmp_camera2._R = poses[i].first;
        tmp_camera2._t = poses[i].second;

//        std::cout << poses[i].first << std::endl;
//        std::cout << poses[i].second << std::endl;
//
//        std::cout << tmp_camera2._R << std::endl;
//        std::cout << tmp_camera2._t << std::endl;
        // 进行三角化，如果重建的三维点坐标，在两个相机坐标系下的z值都为正，说明是正确的位姿
        Eigen::Vector3d p1;
        Eigen::Vector3d p2;
        int count = 0;
        for (int j = 0; j < points1.size(); j++) {

            int idx = j;
            p1 = points1[idx];
            p2 = points2[idx];

            // 三角化
            Eigen::Vector3d p3d = triangulation(p1, p2, camera1, tmp_camera2);
            auto x1 = camera1._R * p3d + camera1._t;
            auto x2 = tmp_camera2._R * p3d + tmp_camera2._t;
//            std::cout << "x1:" << std::endl << x1 << std::endl;
//            std::cout << "x2:" << std::endl << x2 << std::endl;
            if (x1(2) < 0 || x2(2) < 0) {
                count += 1;
            }
        }
        if (count < min_count) {
            min_idx = i;
            min_count = count;
            camera2 = tmp_camera2;
        }
    }
    camera2._R = poses[min_idx].first;
    camera2._t = poses[min_idx].second;
}

Eigen::Vector3d Rebuild::triangulation(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2, const Camera &camera1, const Camera &camera2) {
    //投影矩阵
    Eigen::Matrix3d R1 = camera1._R, R2 = camera2._R;
    Eigen::Matrix3d K1 = camera1._K, K2 = camera2._K;
    Eigen::Vector3d t1 = camera1._t, t2 = camera2._t;

    Eigen::Matrix<double, 3, 4> M1, M2;
    M1 << R1, t1;
    M2 << R2, t2;
    M1 = K1 * M1;
    M2 = K2 * M2;
//    std::cout << "M1: " << M1 << std::endl;
//    std::cout << "M2: " << M2 << std::endl;
    //构造A矩阵
    Eigen::Matrix4d A;
    A(0, 0) = p1(0) * M1(2, 0) - M1(0, 0);
    A(0, 1) = p1(0) * M1(2, 1) - M1(0, 1);
    A(0, 2) = p1(0) * M1(2, 2) - M1(0, 2);
    A(0, 3) = p1(0) * M1(2, 3) - M1(0, 3);

    A(1, 0) = p1(1) * M1(2, 0) - M1(1, 0);
    A(1, 1) = p1(1) * M1(2, 1) - M1(1, 1);
    A(1, 2) = p1(1) * M1(2, 2) - M1(1, 2);
    A(1, 3) = p1(1) * M1(2, 3) - M1(1, 3);

    A(2, 0) = p2(0) * M2(2, 0) - M2(0, 0);
    A(2, 1) = p2(0) * M2(2, 1) - M2(0, 1);
    A(2, 2) = p2(0) * M2(2, 2) - M2(0, 2);
    A(2, 3) = p2(0) * M2(2, 3) - M2(0, 3);

    A(3, 0) = p2(1) * M2(2, 0) - M2(1, 0);
    A(3, 1) = p2(1) * M2(2, 1) - M2(1, 1);
    A(3, 2) = p2(1) * M2(2, 2) - M2(1, 2);
    A(3, 3) = p2(1) * M2(2, 3) - M2(1, 3);

    Eigen::Vector3d P;
    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto V = svd.matrixV();
    P(0) = V(0, 3) / V(3, 3);
    P(1) = V(1, 3) / V(3, 3);
    P(2) = V(2, 3) / V(3, 3);

//    Eigen::Vector4d tmp_P;
//    tmp_P(0) = P(0);
//    tmp_P(1) = P(1);
//    tmp_P(2) = P(2);
//    tmp_P(3) = 1;
//
//    Eigen::Vector3d rePoint = M1 * tmp_P;
//    rePoint(0) /= rePoint(2);
//    rePoint(1) /= rePoint(2);
//    rePoint(2) /= rePoint(2);
//    std::cout << "rePoint:" << std::endl << rePoint << std::endl;
//    std::cout << "p2d:" << std::endl << p1 << std::endl;

    return P;
}

void Rebuild::points_normalize(const std::vector<cv::KeyPoint> &cv_points, std::vector<Eigen::Vector3d> &eigen_points, Eigen::MatrixXd &T) {
    double sum_x = 0, sum_y = 0;
    for (int i = 0; i < cv_points.size(); i++) {
        sum_x += cv_points[i].pt.x;
        sum_y += cv_points[i].pt.y;
    }
    double ave_x = sum_x / cv_points.size();
    double ave_y = sum_y / cv_points.size();
    double sigma_x = 0;
    double sigma_y = 0;
    for (int i = 0; i < cv_points.size(); i++) {
        sigma_x += (cv_points[i].pt.x - ave_x) * (cv_points[i].pt.x - ave_x);
        sigma_y += (cv_points[i].pt.y - ave_y) * (cv_points[i].pt.y - ave_y);
    }
    sigma_x = sqrt(sigma_x / cv_points.size());
    sigma_y = sqrt(sigma_y / cv_points.size());

    Eigen::Vector3d point;
    for (int i = 0; i < cv_points.size(); i++) {
        point(0) = (cv_points[i].pt.x - ave_x) / sigma_x;
        point(1) = (cv_points[i].pt.y - ave_y) / sigma_y;
        point(2) = 1;
        eigen_points.push_back(point);
    }
    T = Eigen::MatrixXd::Zero(3, 3);
    T(0, 0) = 1 / sigma_x;
    T(0, 2) = -ave_x / sigma_x;
    T(1, 1) = 1 / sigma_y;
    T(1, 2) = -ave_y / sigma_y;
    T(2, 2) = 1;

    //验证
//    {
//        for(int i = 0 ; i < eigen_points.size();i++){
//            Eigen::Vector3d norm_p = eigen_points[i];
//            Eigen::Vector3d p;
//            p(0) = cv_points[i].pt.x;
//            p(1) = cv_points[i].pt.y;
//            p(2) = 1;
//            auto tmp = T * p;
//            std::cout <<"norm:" <<std::endl << norm_p << std::endl;
//            std::cout <<"p:" <<std::endl << p << std::endl;
//            std::cout <<"tmp:" <<std::endl << tmp << std::endl;
//
//        }
//    }

}

void
Rebuild::init_reconstruct(int id1, int id2, const std::vector<cv::DMatch> &success_matches) {
    if (success_matches.size() < 100) {
        return;
    }
    if (id1 > id2) {
        int tmp = id1;
        id1 = id2;
        id2 = tmp;
    }
    std::cout << ("=========重建 " + std::to_string(id1) + " " + std::to_string(id2) + " " + std::to_string(success_matches.size()) + "=============")
              << std::endl;

    _blog->write("=========重建 " + std::to_string(id1) + " " + std::to_string(id2) + " " + std::to_string(success_matches.size()) + "=============");

    auto &commonView = _commonView;
    std::vector<Node> &graph = _commonView._graph;
    std::vector<Camera> &cameras = _cameras;
    std::vector<Eigen::Vector3d> &points_cloud = _points_cloud;
    // 构建匹配点
    std::map<int, int> state1, state2;
    std::vector<int> used_kp1, used_kp2;

    std::vector<cv::KeyPoint> points1;
    std::vector<cv::KeyPoint> points2;
    for (int i = 0; i < success_matches.size(); i++) {
        auto node1 = graph[id1];
        auto node2 = graph[id2];
        int queryIdx = success_matches[i].queryIdx;
        int trainIdx = success_matches[i].trainIdx;
        auto point1 = node1.keyPoints[queryIdx];
        auto point2 = node2.keyPoints[trainIdx];
        points1.push_back(point1);
        points2.push_back(point2);
        state1[i] = queryIdx;
        state2[i] = trainIdx;
    }

    //显示匹配结果
    cv::Mat img_matches;
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < points1.size(); i++) {
        cv::DMatch match{i, i, 0, 0};
        good_matches.push_back(match);
    }
    cv::drawMatches(commonView._images[id1], points1, commonView._images[id2], points2, good_matches, img_matches, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    _blog->writeImg(img_matches, "matches_" + std::to_string(id1) + "_" + std::to_string(id2) + ".bmp");

//        cv::namedWindow("knn_matches", cv::WINDOW_NORMAL);
//        cv::resizeWindow("knn_matches", commonView._images[id1].rows / 2, commonView._images[id1].cols / 2);
//        cv::imshow("knn_matches", img_matches);
//        cv::waitKey(0);


    // 对点归一化
    Eigen::MatrixXd T1, T2;
    std::vector<Eigen::Vector3d> norm_p1;
    std::vector<Eigen::Vector3d> norm_p2;
    points_normalize(points2, norm_p2, T2);
    points_normalize(points1, norm_p1, T1);

    // 计算基础矩阵
    int max_iter = 100;
    double inter_thresh = 0.5;
    int N = INT_MAX, sample_count = 0;
    std::vector<int> best_inter_idx;
    Eigen::MatrixXd F(3, 3);
    while (sample_count >= max_iter || N > sample_count) {
        //随机选8个点
        std::set<int> idx;
        while (idx.size() < 8) {
            idx.insert(rand() % norm_p1.size());
        }

        std::vector<Eigen::Vector3d> tmp_p1, tmp_p2;
        for (auto e: idx) {
            tmp_p1.push_back(norm_p1[e]);
            tmp_p2.push_back(norm_p2[e]);
        }
        calc_fundamental(tmp_p1, tmp_p2, F);


        // 统计内点个数
        std::vector<int> inter_idx = find_inters(norm_p1, norm_p2, F, inter_thresh);
        if (best_inter_idx.size() < inter_idx.size()) {
            best_inter_idx = inter_idx;
        }

        // RANSAC 迭代
        double e = 1.0 - double(inter_idx.size()) / points1.size();
        N = log(1 - 0.9) / log(1 - pow((1 - e), 8));
        sample_count += 1;
    }


    //显示匹配结果
//    cv::Mat img_matches;
//    std::vector<cv::DMatch> good_matches;
//    for (int i = 0; i < best_inter_idx.size(); i++) {
//        cv::DMatch match{best_inter_idx[i], best_inter_idx[i], 0, 0};
//        good_matches.push_back(match);
//    }
//    cv::drawMatches(commonView._images[id1], points1, commonView._images[id2], points2, good_matches, img_matches, cv::Scalar::all(-1),
//                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//    cv::namedWindow("knn_matches", cv::WINDOW_NORMAL);
//    cv::resizeWindow("knn_matches", commonView._images[id1].rows / 2, commonView._images[id1].cols / 2);
//    cv::imshow("knn_matches", img_matches);
//    cv::waitKey(0);


    //利用所有的内点计算基础矩阵
    std::vector<Eigen::Vector3d> norm_inter_p1;
    std::vector<Eigen::Vector3d> norm_inter_p2;
    std::vector<Eigen::Vector3d> inter_p1;
    std::vector<Eigen::Vector3d> inter_p2;
    for (int i = 0; i < best_inter_idx.size(); i++) {
        Eigen::Vector3d p1, p2;
        int idx = best_inter_idx[i];
        p1 = norm_p1[idx];
        p2 = norm_p2[idx];
        norm_inter_p1.push_back(p1);
        norm_inter_p2.push_back(p2);
        p1(0) = points1[idx].pt.x;
        p1(1) = points1[idx].pt.y;
        p1(2) = 1;
        p2(0) = points2[idx].pt.x;
        p2(1) = points2[idx].pt.y;
        p2(2) = 1;
        inter_p1.push_back(p1);
        inter_p2.push_back(p2);
        used_kp1.push_back(state1[idx]);
        used_kp2.push_back(state2[idx]);
    }
    calc_fundamental(norm_inter_p1, norm_inter_p2, F);

    //验证基础矩阵
//    {
//        std::cout << "F:" << std::endl << F << std::endl;
//        for(int i = 0; i < 5;i++){
//            Eigen::Vector3d p1 = norm_p1[i];
//            Eigen::Vector3d p2 = norm_p2[i];
//            auto tmp = p2.transpose() * F * p1;
//            std::cout << tmp << std::endl;
//        }
//    }

    F = T2.transpose() * F * T1;
    //验证基础矩阵
//    {
//        std::cout << "F:" << std::endl << F << std::endl;
//        for (int i = 0; i < 5; i++) {
//            Eigen::Vector3d p1, p2;
//            p1(0) = points1[i].pt.x;
//            p1(1) = points1[i].pt.y;
//            p1(2) = 1;
//            p2(0) = points2[i].pt.x;
//            p2(1) = points2[i].pt.y;
//            p2(2) = 1;
//            auto tmp = p2.transpose() * F * p1;
//            std::cout << tmp << std::endl;
//        }
//    }

    // 计算相机的R,t
    // 以第一个相机为中心，建立坐标系
    Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    R(0, 0) = 1;
    R(1, 1) = 1;
    R(2, 2) = 1;
    cameras[id1]._R = R;
    cameras[id1]._t = t;

//        std::cout << "R:" << std::endl << R << std::endl;
//        std::cout << "t:" << std::endl << t << std::endl;


    calc_cam_pose(F, cameras[id1], cameras[id2], inter_p1, inter_p2);
    _blog->write("相机参数：");
    _blog->write("id1 " + std::to_string(id1) + ":");
    _blog->write("K:");
    _blog->write(cameras[id1]._K);
    _blog->write("dist:");
    _blog->write(cameras[id1]._dist);
    _blog->write("R:");
    _blog->write(cameras[id1]._R);
    _blog->write("t:");
    _blog->write(cameras[id1]._t);

    _blog->write("id2 " + std::to_string(id2) + ":");
    _blog->write("K:");
    _blog->write(cameras[id2]._K);
    _blog->write("dist:");
    _blog->write(cameras[id2]._dist);
    _blog->write("R:");
    _blog->write(cameras[id2]._R);
    _blog->write("t:");
    _blog->write(cameras[id2]._t);
    double reErr1 = 0;
    double reErr2 = 0;
    for (int i = 0; i < inter_p1.size(); i++) {
        Eigen::Vector3d p3d = triangulation(inter_p1[i], inter_p2[i], cameras[id1], cameras[id2]);
//        if (p3d(2) > 100 || p3d(2) < 0) continue;
        points_cloud.push_back(p3d);
        double err1 = reProjErr(p3d, inter_p1[i], cameras[id1]);
        reErr1 += err1;
        double err2 = reProjErr(p3d, inter_p2[i], cameras[id2]);
        reErr2 += err2;
        if (_commonView._graph[id1].trick_id[used_kp1[i]] != _commonView._graph[id2].trick_id[used_kp2[i]]) {
            std::cerr << "track_id  错误" << std::endl;
            exit(1);
        }
        _points_state.push_back(_commonView._graph[id1].trick_id[used_kp1[i]]);
    }
    _blog->write("reProjErr:");
    _blog->write((reErr1 + reErr2) / (inter_p1.size() * 2));

    //取消两视图的关联
    graph[id1].edges[id2].flag = false;
    graph[id2].edges[id1].flag = false;
}

void
Rebuild::reconstruct(int id1, int id2, const std::vector<cv::DMatch> &success_matches) {
    if (id1 > id2) {
        int tmp = id1;
        id1 = id2;
        id2 = tmp;
    }
    std::cout << ("=========重建 " + std::to_string(id1) + " " + std::to_string(id2) + " " + std::to_string(success_matches.size()) + "=============")
              << std::endl;

    _blog->write("=========重建 " + std::to_string(id1) + " " + std::to_string(id2) + " " + std::to_string(success_matches.size()) + "=============");


    auto &commonView = _commonView;
    std::vector<Node> &graph = _commonView._graph;
    std::vector<Camera> &cameras = _cameras;
    std::vector<Eigen::Vector3d> &points_cloud = _points_cloud;
    // 构建匹配点
    std::vector<int> used_kp1, used_kp2;


    std::vector<cv::KeyPoint> points1;
    std::vector<cv::KeyPoint> points2;
    for (int i = 0; i < success_matches.size(); i++) {
        auto node1 = graph[id1];
        auto node2 = graph[id2];
        int queryIdx = success_matches[i].queryIdx;
        int trainIdx = success_matches[i].trainIdx;

        auto point1 = node1.keyPoints[queryIdx];
        auto point2 = node2.keyPoints[trainIdx];
        points1.push_back(point1);
        points2.push_back(point2);
        used_kp1.push_back(queryIdx);
        used_kp2.push_back(trainIdx);
    }
    //显示匹配结果
    cv::Mat img_matches;
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < points1.size(); i++) {
        cv::DMatch match{i, i, 0, 0};
        good_matches.push_back(match);
    }
    cv::drawMatches(commonView._images[id1], points1, commonView._images[id2], points2, good_matches, img_matches, cv::Scalar::all(-1),
                    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    _blog->writeImg(img_matches, "matches_" + std::to_string(id1) + "_" + std::to_string(id2) + ".bmp");

//        cv::namedWindow("knn_matches", cv::WINDOW_NORMAL);
//        cv::resizeWindow("knn_matches", commonView._images[id1].rows / 2, commonView._images[id1].cols / 2);
//        cv::imshow("knn_matches", img_matches);
//        cv::waitKey(0);

    std::vector<Eigen::Vector3d> inter_p1;
    std::vector<Eigen::Vector3d> inter_p2;
    for (int i = 0; i < points1.size(); i++) {
        Eigen::Vector3d p1, p2;
        p1(0) = points1[i].pt.x;
        p1(1) = points1[i].pt.y;
        p1(2) = 1;
        p2(0) = points2[i].pt.x;
        p2(1) = points2[i].pt.y;
        p2(2) = 1;
        inter_p1.push_back(p1);
        inter_p2.push_back(p2);
    }

    _blog->write("相机参数：");
    _blog->write("id1 " + std::to_string(id1) + ":");
    _blog->write("K:");
    _blog->write(cameras[id1]._K);
    _blog->write("dist:");
    _blog->write(cameras[id1]._dist);
    _blog->write("R:");
    _blog->write(cameras[id1]._R);
    _blog->write("t:");
    _blog->write(cameras[id1]._t);

    _blog->write("id2 " + std::to_string(id2) + ":");
    _blog->write("K:");
    _blog->write(cameras[id2]._K);
    _blog->write("dist:");
    _blog->write(cameras[id2]._dist);
    _blog->write("R:");
    _blog->write(cameras[id2]._R);
    _blog->write("t:");
    _blog->write(cameras[id2]._t);

    double reErr1 = 0;
    double reErr2 = 0;
    for (int i = 0; i < inter_p1.size(); i++) {
        Eigen::Vector3d p3d = triangulation(inter_p1[i], inter_p2[i], cameras[id1], cameras[id2]);
//        if (p3d(2) > 100 || p3d(2) < 0) continue;
        points_cloud.push_back(p3d);
        reErr1 += reProjErr(p3d, inter_p1[i], cameras[id1]);
        reErr2 += reProjErr(p3d, inter_p2[i], cameras[id2]);
        if (_commonView._graph[id1].trick_id[used_kp1[i]] != _commonView._graph[id2].trick_id[used_kp2[i]]) {
            std::cerr << "track_id  错误" << std::endl;
            exit(1);
        }
        _points_state.push_back(_commonView._graph[id1].trick_id[used_kp1[i]]);
    }
    _blog->write("reProjErr:");
    _blog->write((reErr1 + reErr2) / (inter_p1.size() * 2));

    //取消两视图的关联
    graph[id1].edges[id2].flag = false;
    graph[id2].edges[id1].flag = false;
}

std::vector<cv::DMatch> Rebuild::get_success_matches(int id1, int id2) {
    if (id1 > id2) {
        int tmp = id1;
        id1 = id2;
        id2 = tmp;
    }
    std::vector<Node> &graph = _commonView._graph;
    const std::vector<std::list<std::pair<int, int>>> &tracks = _commonView._tracks;
    //    Blog *blog = _blog;
    std::vector<cv::DMatch> temp_matches1;
    Node node1 = graph[id1];
    Node node2 = graph[id2];


    std::vector<cv::DMatch> success_matches;
    std::vector<cv::DMatch> matches = graph[id1].edges[id2].matches;
    for (int i = 0; i < matches.size(); i++) {
        int queryIdx = matches[i].queryIdx;//第一幅图的特征点id;
        int trainIdx = matches[i].trainIdx;//第二幅图的特征点id;
        int track_id1 = node1.trick_id[queryIdx];
        int track_id2 = node2.trick_id[trainIdx];
        if (track_id1 != track_id2) {
            continue;
        }
        if (tracks[track_id1].size() > 2 && _tracks_state[track_id1] == false) {
            success_matches.push_back(matches[i]);
        }
    }
    return success_matches;
}

void Rebuild::save(std::string path) {
    std::ofstream file(path + "/points_cloud.txt");
    auto points_cloud = _points_cloud;
    for (int i = 0; i < points_cloud.size(); i++) {
        auto p3d = points_cloud[i];
        auto x = p3d(0);
        auto y = p3d(1);
        auto z = p3d(2);
        if (z < 0 || z > 100) continue;
        file << x << "\t" << y << "\t" << z << "\t" << std::endl;
    }
    file.flush();
    file.close();
}

void Rebuild::ePnP(const std::vector<Eigen::Vector3d> &p3ds, const std::vector<Eigen::Vector2d> &p2ds, Camera &camera) {
    auto blog = _blog;
//    blog->write("ePnP:");

    /**寻找4个控制点**/
    //求取重心
    Eigen::Vector3d center = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> control_points_w(4);
    for (int i = 0; i < p3ds.size(); i++) {
        center(0) += p3ds[i](0);
        center(1) += p3ds[i](1);
        center(2) += p3ds[i](2);
    }
    center(0) /= p3ds.size();
    center(1) /= p3ds.size();
    center(2) /= p3ds.size();
    control_points_w[0] = center;
//    blog->write("center:");
//    blog->write(center);
//    blog->write("");
    //构建矩阵
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(p3ds.size(), 3);
    for (int i = 0; i < p3ds.size(); i++) {
        A(i, 0) = p3ds[i](0) - center(0);
        A(i, 1) = p3ds[i](1) - center(1);
        A(i, 2) = p3ds[i](2) - center(2);
    }

    Eigen::MatrixXd M = A.transpose() * A;

    Eigen::EigenSolver<Eigen::MatrixXd> solver(M);
    Eigen::VectorXd eigenValues = solver.eigenvalues().real();
    Eigen::MatrixXd eigenVectors = solver.eigenvectors().real();


    for (int i = 1; i < 4; i++) {
        control_points_w[i] = control_points_w[0] + sqrt(eigenValues(i - 1)) * eigenVectors.col(i - 1);
    }
    /**求解alpha**/
    Eigen::MatrixXd C_w = Eigen::MatrixXd::Zero(4, 4);
    for (int i = 0; i < 4; i++) {
        double x = control_points_w[i](0);
        double y = control_points_w[i](1);
        double z = control_points_w[i](2);
        C_w(0, i) = x;
        C_w(1, i) = y;
        C_w(2, i) = z;
        C_w(3, i) = 1;
    }
    Eigen::MatrixXd C_w_inv = C_w.inverse();


    double fu = camera._K(0, 0);
    double fv = camera._K(1, 1);
    double uc = camera._K(0, 2);
    double vc = camera._K(1, 2);

    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(int(2 * p3ds.size()), 12);
    for (int i = 0; i < p3ds.size(); i++) {
        Eigen::Vector3d p3d = p3ds[i];
        Eigen::Vector2d p2d = p2ds[i];
        Eigen::Vector4d b;
        b << p3d(0), p3d(1), p3d(2), 1;
        Eigen::Vector4d alpha = C_w_inv * b;


        D(i * 2, 0) = fu * alpha(0);
        D(i * 2, 1) = fu * alpha(1);
        D(i * 2, 2) = fu * alpha(2);
        D(i * 2, 3) = fu * alpha(3);
        D(i * 2, 8) = (uc - p2d(0)) * alpha(0);
        D(i * 2, 9) = (uc - p2d(0)) * alpha(1);
        D(i * 2, 10) = (uc - p2d(0)) * alpha(2);
        D(i * 2, 11) = (uc - p2d(0)) * alpha(3);

        D(i * 2 + 1, 4) = fv * alpha(0);
        D(i * 2 + 1, 5) = fv * alpha(1);
        D(i * 2 + 1, 6) = fv * alpha(2);
        D(i * 2 + 1, 7) = fv * alpha(3);
        D(i * 2 + 1, 8) = (vc - p2d(1)) * alpha(0);
        D(i * 2 + 1, 9) = (vc - p2d(1)) * alpha(1);
        D(i * 2 + 1, 10) = (vc - p2d(1)) * alpha(2);
        D(i * 2 + 1, 11) = (vc - p2d(1)) * alpha(3);

    }


    Eigen::JacobiSVD<Eigen::MatrixXd> svd(D, Eigen::ComputeFullV | Eigen::ComputeFullU);
    auto V_matrix = svd.matrixV();
    auto V_vector = V_matrix.col(V_matrix.cols() - 1);
    std::vector<Eigen::Vector3d> control_points_c(4);

    for (int i = 0; i < 4; i++) {
        double x = V_vector(i);
        double y = V_vector(i + 4);
        double z = V_vector(i + 8);
        Eigen::Vector3d p3d(x, y, z);
        control_points_c[i] = p3d;

    }


    Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(12, 12);
    Eigen::VectorXd C_c = Eigen::VectorXd::Zero(12);

    for (int i = 0; i < 4; i++) {
        C_c(i * 3) = control_points_c[i](0);
        C_c(i * 3 + 1) = control_points_c[i](1);
        C_c(i * 3 + 2) = control_points_c[i](2);
        Q(i * 3, 0) = control_points_w[i](0);
        Q(i * 3, 1) = control_points_w[i](1);
        Q(i * 3, 2) = control_points_w[i](2);
        Q(i * 3, 9) = 1;

        Q(i * 3 + 1, 3) = control_points_w[i](0);
        Q(i * 3 + 1, 4) = control_points_w[i](1);
        Q(i * 3 + 1, 5) = control_points_w[i](2);
        Q(i * 3 + 1, 10) = 1;

        Q(i * 3 + 2, 6) = control_points_w[i](0);
        Q(i * 3 + 2, 7) = control_points_w[i](1);
        Q(i * 3 + 2, 8) = control_points_w[i](2);
        Q(i * 3 + 2, 11) = 1;

    }

    Eigen::MatrixXd Q_inv = Q.inverse();

    Eigen::VectorXd ans = Eigen::VectorXd::Zero(12);

    ans = Q_inv * C_c;


    camera._R(0, 0) = ans(0);
    camera._R(0, 1) = ans(1);
    camera._R(0, 2) = ans(2);
    camera._R(1, 0) = ans(3);
    camera._R(1, 1) = ans(4);
    camera._R(1, 2) = ans(5);
    camera._R(2, 0) = ans(6);
    camera._R(2, 1) = ans(7);
    camera._R(2, 2) = ans(8);

    camera._t(0) = ans(9);
    camera._t(1) = ans(10);
    camera._t(2) = ans(11);
}

double Rebuild::reProjErr(const Eigen::Vector3d &p3d, const Eigen::Vector3d &p2d, const Camera &camera) {
    Eigen::Matrix3d R = camera._R;
    Eigen::Matrix3d K = camera._K;
    Eigen::Vector3d t = camera._t;
    Eigen::Vector3d rePoint = K * (R * p3d + t);
    rePoint(0) /= rePoint(2);
    rePoint(1) /= rePoint(2);
    rePoint(2) /= rePoint(2);

//    std::cout << "p2d:" << std::endl << p2d << std::endl;
//    std::cout << "rePoint:" << std::endl << rePoint << std::endl;
    double err = sqrt(pow(p2d(0) - rePoint(0), 2) + pow(p2d(1) - rePoint(1), 2));
    return err;
}

//void Rebuild::bundle_adjustment() {
//    const std::vector<std::list<std::pair<int, int>>> &tracks = _commonView._tracks;
//    const std::vector<Node> &graph = _commonView._graph;
//
//    std::vector<int> available_cameras, available_points;
//    for (int i = 1; i <= _images_num; i++) {
//        if (_cameras_state[i]) {
//            available_cameras.push_back(i);
//        }
//    }
//    //建立三维点、二维点和相机的对应关系
//    std::vector<std::vector<Eigen::Vector3d>> c_p3ds(_images_num + 1);
//    std::vector<std::vector<Eigen::Vector3d>> c_p2ds(_images_num + 1);
//    for (int i = 0; i < _points_cloud.size(); i++) {
//        int track = _points_state[i];
//        for (auto e: tracks[track]) {
//            int cam_id = e.first;
//            int kp_id = e.second;
//            Eigen::Vector3d p3d = _points_cloud[i];
//            Eigen::Vector3d p2d;
//            p2d(0) = graph[cam_id].keyPoints[kp_id].pt.x;
//            p2d(1) = graph[cam_id].keyPoints[kp_id].pt.y;
//            p2d(2) = 1;
//            c_p3ds[cam_id].push_back(p3d);
//            c_p2ds[cam_id].push_back(p2d);
//        }
//    }
////    // 根据available_cameras的顺序重新排列三维点和二维点
////    std::vector<Eigen::Vector3d> p3ds;
////    std::vector<Eigen::Vector2d> p2ds;
////    for(int i = 0; i < available_cameras.size();i++){
////        int cam_id = available_cameras[i];
////        for(int j = 0; j < c_p3ds[cam_id].size();j++){
////            p3ds.push_back(c_p3ds[cam_id][j]);
////            p2ds.push_back(c_p2ds[cam_id][j]);
////        }
////    }
//
//    // 计算重投影误差
//    int k = 0;
//    std::vector<double> F;
//    std::vector<Eigen::Vector3d> p3ds;
//    for (int i = 0; i < available_cameras.size(); i++) {
//        int cam_id = available_cameras[i];
//        for (int j = 0; j < c_p3ds[cam_id].size(); j++) {
//            available_points.push_back(k++);
//            Eigen::Vector3d p3d = c_p3ds[cam_id][j];
//            p3ds.push_back(p3d);
//            Eigen::Vector3d p2d = c_p2ds[cam_id][j];
//            reProjErr_bj(p3d, p2d, _cameras[cam_id], F);
//        }
//    }
//    double mse = compute_mse(F);
//    double final_mse, init_mse;
//    final_mse = init_mse = mse;
//
//    /**Levenberg-Marquard 算法**/
//    int lm_max_iterations = 100;
//    for (int lm_iter = 0; lm_iter < lm_max_iterations; lm_iter++) {
//        // 1、计算雅可比矩阵
//        Eigen::MatrixXd Jc, Jp;
//        analytic_jacobian(Jc, Jp, p3ds, available_cameras);
//
//        // 2、高斯牛顿法求解
//        Eigen::MatrixXd delta = my_solve_schur(Jc, Jp, F, available_cameras, available_points);
//
//    }
//
//}

void Rebuild::reProjErr_bj(const Eigen::Vector3d &p3d, const Eigen::Vector3d &p2d, const Camera &camera, std::vector<double> &F) {
    Eigen::Matrix3d R = camera._R;
    Eigen::Matrix3d K = camera._K;
    Eigen::Vector3d t = camera._t;
    Eigen::Vector3d rePoint = K * (R * p3d + t);
    rePoint(0) /= rePoint(2);
    rePoint(1) /= rePoint(2);
    rePoint(2) /= rePoint(2);

    F.push_back(p2d(0) - rePoint(0));
    F.push_back(p2d(1) - rePoint(1));
}

double Rebuild::compute_mse(const std::vector<double> &F) {
    double mse = 0.0;
    for (int i = 0; i < F.size(); i++) {
        mse += F[i] * F[i];
    }
    return mse / (F.size() / 2.0);
}

void
Rebuild::analytic_jacobian(Eigen::MatrixXd &Jc, Eigen::MatrixXd &Jp, const std::vector<Eigen::Vector3d> &p3ds, std::vector<int> available_cameras) {
    int Jc_col = 16 * available_cameras.size(), Jp_col = 3 * p3ds.size(), Jc_row = 0, Jp_row = 0;
    Jc_row = Jp_row = p3ds.size() * 2;

    Jc.resize(Jc_row, Jc_col);
    Jp.resize(Jp_row, Jp_col);

    double cam_x_ptr[16], cam_y_ptr[16], point_x_ptr[3], point_y_ptr[3];
    for (int i = 0; i < available_cameras.size(); i++) {
        int cam_id = available_cameras[i];
        Camera cam = _cameras[cam_id];
        for (int j = 0; j < p3ds.size(); j++) {
            Eigen::Vector3d p3d = p3ds[i];
            my_jacobian(cam, p3d, cam_x_ptr, cam_y_ptr, point_x_ptr, point_y_ptr);
            int row_x = j * 2;
            int row_y = j * 2 + 1;
            int cam_col = i * 16;
            int point_col = j * 3;

            for (int k = 0; k < 16; k++) {
                Jc(row_x, cam_col + k) = cam_x_ptr[k];
                Jc(row_y, cam_col + k) = cam_y_ptr[k];
            }
            for (int k = 0; k < 3; k++) {
                Jp(row_x, point_col + k) = point_x_ptr[k];
                Jp(row_y, point_col + k) = point_y_ptr[k];
            }
        }
    }
    int a = 10;
}

void
Rebuild::my_jacobian(const Camera &cam, const Eigen::Vector3d &p3d, double *cam_x_ptr, double *cam_y_ptr, double *point_x_ptr, double *point_y_ptr) {
    Eigen::Vector3d pc = cam._R * p3d + cam._t;

    double xc = pc(0);
    double yc = pc(1);
    double zc = pc(2);
    double fu = cam._K(0, 0);
    double fv = cam._K(1, 1);
    double uc = cam._K(0, 2);
    double vc = cam._K(1, 2);
    double t0 = cam._t(0);
    double t1 = cam._t(1);
    double t2 = cam._t(2);
    double r0 = cam._R(0, 0);
    double r1 = cam._R(0, 1);
    double r2 = cam._R(0, 2);
    double r3 = cam._R(1, 0);
    double r4 = cam._R(1, 1);
    double r5 = cam._R(1, 2);
    double r6 = cam._R(2, 0);
    double r7 = cam._R(2, 1);
    double r8 = cam._R(2, 2);
    double X = p3d(0);
    double Y = p3d(1);
    double Z = p3d(2);

    cam_x_ptr[0] = xc / zc;
    cam_x_ptr[1] = 0;
    cam_x_ptr[2] = 1;
    cam_x_ptr[3] = 0;
    cam_x_ptr[4] = fu / zc;
    cam_x_ptr[5] = 0;
    cam_x_ptr[6] = -fu * xc / zc / zc;
    cam_x_ptr[7] = fu / zc * X;
    cam_x_ptr[8] = fu / zc * Y;
    cam_x_ptr[9] = fu / zc * Z;
    cam_x_ptr[10] = 0;
    cam_x_ptr[11] = 0;
    cam_x_ptr[12] = 0;
    cam_x_ptr[13] = -fu * xc / zc / zc * X;
    cam_x_ptr[14] = -fu * xc / zc / zc * Y;
    cam_x_ptr[15] = -fu * xc / zc / zc * Z;

    cam_y_ptr[0] = 0;
    cam_y_ptr[1] = yc / zc;
    cam_y_ptr[2] = 0;
    cam_y_ptr[3] = 1;
    cam_y_ptr[4] = 0;
    cam_y_ptr[5] = fv / zc;
    cam_y_ptr[6] = -fv * yc / zc / zc;
    cam_y_ptr[7] = 0;
    cam_y_ptr[8] = 0;
    cam_y_ptr[9] = 0;
    cam_y_ptr[10] = fv / zc * X;
    cam_y_ptr[11] = fv / zc * Y;
    cam_y_ptr[12] = fv / zc * Z;
    cam_y_ptr[13] = -fv * yc / zc / zc * X;
    cam_y_ptr[14] = -fv * yc / zc / zc * Y;
    cam_y_ptr[15] = -fv * yc / zc / zc * Z;

    point_x_ptr[0] = r0 * fu / zc - r6 * fu * xc / zc / zc;
    point_x_ptr[1] = r1 * fu / zc - r7 * fu * xc / zc / zc;
    point_x_ptr[2] = r2 * fu / zc - r8 * fu * xc / zc / zc;

    point_y_ptr[0] = r3 * fv / zc - r6 * fv * yc / zc / zc;
    point_y_ptr[1] = r4 * fv / zc - r7 * fv * yc / zc / zc;
    point_y_ptr[2] = r5 * fv / zc - r8 * fv * yc / zc / zc;

}

Eigen::MatrixXd
Rebuild::my_solve_schur(const Eigen::MatrixXd Jc, const Eigen::MatrixXd Jp, const std::vector<double> &F, const std::vector<int> &available_cameras,
                        const std::vector<int> &available_points) {
    /**
    *  雅可比矩阵：
    *      J = [Jc Jp]
    *
    */
    Eigen::MatrixXd J(Jc.rows(), Jc.cols() + Jp.cols());
    Eigen::VectorXd err = Eigen::VectorXd::Zero(F.size());

    for (int i = 0; i < F.size(); i++) {
        err(i) = F[i];
    }
    _blog->write("Jc:");
    _blog->write(Jc);
    _blog->write("Jp:");
    _blog->write(Jp);

    J << Jc, Jp;

    const Eigen::MatrixXd JT = J.transpose();
//    std::cout << JT << std::endl;
    const Eigen::MatrixXd JTJ = JT * J;
//    std::cout << JTJ << std::endl;

    Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(JTJ.rows(), JTJ.rows());
    identity *= 0.1;
    /* B = JTJ + uI */
    const Eigen::MatrixXd B = JTJ + identity;

    const Eigen::MatrixXd inv_B = B.inverse();
//    std::cout << inv_B << std::endl;

    const Eigen::MatrixXd E = -JT * err;
    const Eigen::VectorXd delta = inv_B * E;
//    std::cout << delta << std::endl;
    int j = 0;
    for (int i = 0; i < available_cameras.size(); i++) {
        //[fu,fv,u_c,v_c,t_0,t1,t2,r0,r1,r2,r3,r4,r5,r6,r7,r8]
        int cam_id = available_cameras[i];
        Camera &cam = _cameras[cam_id];
        cam._K(0, 0) += delta(0);
        cam._K(1, 1) += delta(1);
        cam._K(0, 2) += delta(j++);
        cam._K(1, 2) += delta(j++);
        cam._t(0) += delta(j++);
        cam._t(1) += delta(j++);
        cam._t(2) += delta(j++);
        cam._R(0, 0) += delta(j++);
        cam._R(0, 1) += delta(j++);
        cam._R(0, 2) += delta(j++);
        cam._R(1, 0) += delta(j++);
        cam._R(1, 1) += delta(j++);
        cam._R(1, 2) += delta(j++);
        cam._R(2, 0) += delta(j++);
        cam._R(2, 1) += delta(j++);
        cam._R(2, 2) += delta(j++);
    }
    for (int i = 0; i < available_points.size(); i++) {
        int p_id = available_points[i];
        Eigen::Vector3d &p3d = _points_cloud[p_id];
        p3d(0) += delta(j + 0);
        p3d(1) += delta(j + 1);
        p3d(2) += delta(j + 2);


    }
}

void Rebuild::ceres_bundle_adjustment() {
    _observations.clear();
    const std::vector<std::list<std::pair<int, int>>> &tracks = _commonView._tracks;
    const std::vector<Node> &graph = _commonView._graph;

    int num_cam = 0;
    std::vector<int> available_cam;
    for (int i = 0; i < _cameras_state.size(); i++) {
        if (_cameras_state[i]) {
            available_cam.push_back(i);
            num_cam++;
        }
    }

    //建立三维点、二维点和相机的对应关系
    for (int i = 0; i < _points_cloud.size(); i++) {
        int p3d_id = i;
        int track = _points_state[i];
        for (auto e: tracks[track]) {
            int cam_id = e.first;
            if (!_cameras_state[cam_id]) {
                continue;
            }

            int kp_id = e.second;
            double p2d[2];
            p2d[0] = graph[cam_id].keyPoints[kp_id].pt.x;
            p2d[1] = graph[cam_id].keyPoints[kp_id].pt.y;
            observation obs;
            obs.cam_id = cam_id;
            obs.p3d_id = p3d_id;
            obs.p2d[0] = p2d[0];
            obs.p2d[1] = p2d[1];
            _observations.push_back(obs);
        }
    }
    _blog->write("捆绑调整前重投影误差:");
    double err = 0;
    for (int i = 0; i < _observations.size(); i++) {

        int p3d_id = _observations[i].p3d_id;
        int cam_id = _observations[i].cam_id;
        Eigen::Vector3d p3d;
        p3d(0) = _points_cloud[p3d_id](0);
        p3d(1) = _points_cloud[p3d_id](1);
        p3d(2) = _points_cloud[p3d_id](2);
        Camera camera = _cameras[cam_id];
        Eigen::Vector3d p2d;
        p2d(0) = _observations[i].p2d[0];
        p2d(1) = _observations[i].p2d[1];
        p2d(2) = 1;
        err += reProjErr(p3d, p2d, camera);
//        if(i == 0){
//            _blog->write("第一个点信息：");
//            _blog->write("");
//            _blog->write("相机参数： " + std::to_string(cam_id));
//            _blog->write("K：");
//            _blog->write(camera._K);
//            _blog->write("R：");
//            _blog->write(camera._R);
//            _blog->write("t：");
//            _blog->write(camera._t);
//            _blog->write("");
//            _blog->write("p3d:");
//            _blog->write(p3d);
//            _blog->write("p2d:");
//            _blog->write(p2d);
//            _blog->write("err:");
//            _blog->write(err);
//        }
    }
    _blog->write(err / _observations.size());

    double *cam_parameters = new double[16 * num_cam];
    double *point_parameters = new double[3 * _points_cloud.size()];

    for (int i = 0; i < available_cam.size(); i++) {
        int cam_id = available_cam[i];
        Camera cam = _cameras[cam_id];
        cam_parameters[i * 16 + 0] = cam._K(0, 0);
        cam_parameters[i * 16 + 1] = cam._K(1, 1);
        cam_parameters[i * 16 + 2] = cam._K(0, 2);
        cam_parameters[i * 16 + 3] = cam._K(1, 2);
        cam_parameters[i * 16 + 4] = cam._t(0);
        cam_parameters[i * 16 + 5] = cam._t(1);
        cam_parameters[i * 16 + 6] = cam._t(2);
        cam_parameters[i * 16 + 7] = cam._R(0, 0);
        cam_parameters[i * 16 + 8] = cam._R(0, 1);
        cam_parameters[i * 16 + 9] = cam._R(0, 2);
        cam_parameters[i * 16 + 10] = cam._R(1, 0);
        cam_parameters[i * 16 + 11] = cam._R(1, 1);
        cam_parameters[i * 16 + 12] = cam._R(1, 2);
        cam_parameters[i * 16 + 13] = cam._R(2, 0);
        cam_parameters[i * 16 + 14] = cam._R(2, 1);
        cam_parameters[i * 16 + 15] = cam._R(2, 2);
    }
    for (int i = 0; i < _points_cloud.size(); i++) {
        Eigen::Vector3d p3d = _points_cloud[i];
        point_parameters[i * 3 + 0] = p3d(0);
        point_parameters[i * 3 + 1] = p3d(1);
        point_parameters[i * 3 + 2] = p3d(2);
    }

    ceres::Problem problem;
    for (int i = 0; i < _observations.size(); i++) {
        ceres::CostFunction *cost_function;

        cost_function = SnavelyReprojectionError::Create(_observations[i].p2d[0], _observations[i].p2d[1]);

        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
        int cam_id = _observations[i].cam_id;
        int p3d_id = _observations[i].p3d_id;
        Camera cam = _cameras[cam_id];
        int k = 0;
        for (; k < available_cam.size(); k++) {
            if (available_cam[k] == cam_id) {
                break;
            }
        }
        double *point = point_parameters + p3d_id * 3;
        double *camera = cam_parameters + k * 16;

        problem.AddResidualBlock(cost_function, loss_function, camera, point);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";


    for (int i = 0; i < available_cam.size(); i++) {
        int cam_id = available_cam[i];

        Camera &camera = _cameras[cam_id];
        camera._K(0, 0) = cam_parameters[i * 16 + 0];
        camera._K(1, 1) = cam_parameters[i * 16 + 1];
        camera._K(0, 2) = cam_parameters[i * 16 + 2];
        camera._K(1, 2) = cam_parameters[i * 16 + 3];
        camera._t(0) = cam_parameters[i * 16 + 4];
        camera._t(1) = cam_parameters[i * 16 + 5];
        camera._t(2) = cam_parameters[i * 16 + 6];
        camera._R(0, 0) = cam_parameters[i * 16 + 7];
        camera._R(0, 1) = cam_parameters[i * 16 + 8];
        camera._R(0, 2) = cam_parameters[i * 16 + 9];
        camera._R(1, 0) = cam_parameters[i * 16 + 10];
        camera._R(1, 1) = cam_parameters[i * 16 + 11];
        camera._R(1, 2) = cam_parameters[i * 16 + 12];
        camera._R(2, 0) = cam_parameters[i * 16 + 13];
        camera._R(2, 1) = cam_parameters[i * 16 + 14];
        camera._R(2, 2) = cam_parameters[i * 16 + 15];
    }
    for (int i = 0; i < _points_cloud.size(); i++) {
        Eigen::Vector3d &p3d = _points_cloud[i];
        p3d(0) = point_parameters[i * 3 + 0];
        p3d(1) = point_parameters[i * 3 + 1];
        p3d(2) = point_parameters[i * 3 + 2];
    }
    err = 0;
    for (int i = 0; i < _observations.size(); i++) {
        int p3d_id = _observations[i].p3d_id;
        int cam_id = _observations[i].cam_id;

        Eigen::Vector3d p3d;
        p3d(0) = _points_cloud[p3d_id](0);
        p3d(1) = _points_cloud[p3d_id](1);
        p3d(2) = _points_cloud[p3d_id](2);
        Camera camera = _cameras[cam_id];
        Eigen::Vector3d p2d;
        p2d(0) = _observations[i].p2d[0];
        p2d(1) = _observations[i].p2d[1];
        p2d(2) = 1;
        err += reProjErr(p3d, p2d, camera);
    }

    _blog->write("捆绑调整后重投影误差:");
    _blog->write(err / _observations.size());
    this->save(_blog->_points_dir);
}

