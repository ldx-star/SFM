//
// Created by ldx on 24-5-11.
//

#ifndef SFM_REBUILD_H
#define SFM_REBUILD_H

#include "../include/CommonView.h"
#include "../include/Camera.h"
#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Dense>
#include <fstream>
#include <map>
#include <set>
#include "../include/Blog.h"
#include <sys/stat.h>

/***
 * 对两个视图进行重建
 */
class Rebuild {
public:
    /**
     *
     * @param commonView 公式图
     * @param n 图片数量
     */
    Rebuild(const CommonView &commonView, const std::vector<Camera> &cameras, int n, Blog *blog);

    void save(std::string path);

    ~Rebuild() {};
private:

    void init();

    /**
     * \description 计算基础矩阵和本质矩阵
     * @param points1 视图1的点
     * @param points2 视图2的点
     * @param F 基础矩阵
     * @param E 本质矩阵
     */
    void calc_fundamental(const std::vector<Eigen::Vector3d> &points1, const std::vector<Eigen::Vector3d> &points2, Eigen::MatrixXd &F);

    /**
     * \description 统计内点个数
     * @param p1 视图1的点
     * @param p2 视图2的点
     * @param F 基础矩阵
     * @param inter_thresh 内点门线
     * @return
     */
    std::vector<int> find_inters(const std::vector<Eigen::Vector3d> &p1, const std::vector<Eigen::Vector3d> &p2, const Eigen::Matrix<double, 3, 3> &F,
                                 const double inter_thresh);

    /**
     * \decription 计算距离
     * @param F
     * @param pair_point
     * @return
     */
    double calc_sampson_distance(const Eigen::Matrix<double, 3, 3> &F, const std::pair<Eigen::Vector3d, Eigen::Vector3d> &pair_point);

    /**
     * \description 计算相机位姿(以第一个相机为参考坐标系)
     * @param F 基础矩阵
     * @param camera1
     * @param camera2
     */
    void calc_cam_pose(const Eigen::Matrix<double, 3, 3> &F, Camera &camera1, Camera &camera2, const std::vector<Eigen::Vector3d> &p1,
                       const std::vector<Eigen::Vector3d> &p2);

    Eigen::Vector3d triangulation(const Eigen::Vector3d &p1, const Eigen::Vector3d &p2, const Camera &camera1, const Camera &camera2);

    /**
     * 对二维点进行归一化，将所有坐标都减去其均值
     * @param cv_points
     * @param eigen_points
     */
    void points_normalize(const std::vector<cv::KeyPoint> &cv_points, std::vector<Eigen::Vector3d> &eigen_points, Eigen::MatrixXd &T);

    /**
     * \decription 对初始两幅视图进行重建
     * @param id1
     * @param id2
     * @param used_kp1 记录id1视图中特征点的使用情况
     * @param used_kp1 记录id2视图中特征点的使用情况
     */
    void init_reconstruct(int id1, int id2, const std::vector<cv::DMatch> &success_matches);

    /**
     * \decription 对两幅视图进行重建
     * @param id1
     * @param id2
     * @param used_kp1 记录id1视图中特征点的使用情况
     * @param used_kp1 记录id2视图中特征点的使用情况
     */
    void reconstruct(int id1, int id2, const std::vector<cv::DMatch> &success_matches);

    std::vector<cv::DMatch> get_success_matches(int id1, int id2);

    /**
     * \description 通过ePnP算法求相机参数
     * @param p3ds 三维点
     * @param p2ds 二维点
     * @param camera 相机模型
     */
    void ePnP(const std::vector<Eigen::Vector3d> &p3ds, const std::vector<Eigen::Vector2d> &p2ds, Camera &camera);

    /**
     * \descriptino 计算重投影误差
     * @param p3d
     * @param p2d
     * @param camera
     * @return
     */
    double reProjErr(const Eigen::Vector3d &p3d, const Eigen::Vector3d &p2d, const Camera &camera);

    /**
     * \descriptino 捆绑调整的重投影误差
     * @param p3d
     * @param p2d
     * @param camera
     * @param F
     * @return
     */
    void reProjErr_bj(const Eigen::Vector3d &p3d, const Eigen::Vector3d &p2d, const Camera &camera, std::vector<double> &F);

    void bundle_adjustment();

    /**
     * \description 计算均方误差
     * @param F
     * @return 误差总和
     */
    double compute_mse(const std::vector<double> &F);

    /**
     * \description 计算雅可比矩阵
     * @param Jc
     * @param Jp
     * @param c_p3ds
     * @param c_p2ds
     * @param available_cameras
     */
    void analytic_jacobian(Eigen::MatrixXd &Jc, Eigen::MatrixXd &Jp,const std::vector<std::vector<Eigen::Vector3d>> &c_p3ds,const std::vector< std::vector<Eigen::Vector3d>> &c_p2ds,std::vector<int> available_cameras);

    /**
     * \求雅可比矩阵
     * @param cam 相机参数
     * @param p3d 三维点
     * @param cam_x_ptr u 对 [fu,fv,u_c,v_c,t_0,t1,t2,r0,r1,r2,r3,r4,r5,r6,r7,r8]
     * @param cam_y_ptr v 对 [fu,fv,u_c,v_c,t_0,t1,t2,r0,r1,r2,r3,r4,r5,r6,r7,r8]
     * @param point_x_ptr u 对 [X,Y,Z]
     * @param point_y_ptr v 对 [X,Y,Z]
     */
    void my_jacobian(const Camera &cam, const Eigen::Vector3d &p3d, double *cam_x_ptr, double *cam_y_ptr, double *point_x_ptr, double *point_y_ptr);

    /**
     *\descrition 使用高斯牛顿法进行求解
     * @param Jc 相机参数雅可比矩阵
     * @param Jp 三维点雅可比矩阵
     * @param F y‘ - y
     */
    Eigen::MatrixXd my_solve_schur(const Eigen::MatrixXd Jc, const Eigen::MatrixXd Jp, const std::vector<double> &F,const std::vector<int> &available_cameras);
public:
    CommonView _commonView;
    int _images_num;
    std::vector<Camera> _cameras;
    std::vector<Eigen::Vector3d> _points_cloud;
    std::vector<int> _points_state; // 记录每个三维点的track
    std::vector<bool> _tracks_state; // 每个track是否已经重建
    std::vector<bool> _cameras_state; // 每个相机是否已经获得参数
    Blog *_blog;
};


#endif //SFM_REBUILD_H
