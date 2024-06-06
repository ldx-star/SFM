//
// Created by ldx on 24-6-4.
//

#ifndef SFM_SNAVELYREPROJECTIONERROR_H
#define SFM_SNAVELYREPROJECTIONERROR_H
#include <ceres/ceres.h>
#include <iostream>

class SnavelyReprojectionError{
public:
    SnavelyReprojectionError(double observation_x, double observation_y) : observed_x(observation_x),
                                                                           observed_y(observation_y) {}

    template<typename T>
    bool operator()(const T *const camera, const T *const point, T *residuals) const{
        T predictions[2];

        CamProjectionWithDistortion(camera,point,predictions);
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);
//        if(T(observed_x)-202.246 <0.001 && T(observed_y) - 367.319 < 0.001){
//            std::cout << observed_x << "   " <<  observed_y  << std::endl;
//        }
        return true;
    }


    //camera : [fu,fv,u_c,v_c,t_0,t1,t2,r0,r1,r2,r3,r4,r5,r6,r7,r8]
    //point : [X,Y,Z]
    template<typename T>
    static inline bool CamProjectionWithDistortion(const T *camera, const T *point, T *predictions) {
        // Rodrigues' formula
        T p[3];

        p[0] = camera[7] * point[0] + camera[8] * point[1] + camera[9] * point[2];
        p[1] = camera[10] * point[0] + camera[11] * point[1] + camera[12] * point[2];
        p[2] = camera[13] * point[0] + camera[14] * point[1] + camera[15] * point[2];


        p[0] += camera[4];
        p[1] += camera[5];
        p[2] += camera[6];



//        // Compute the center fo distortion
//        T xp = -p[0] / p[2];
//        T yp = -p[1] / p[2];
//
//        // Apply second and fourth order radial distortion
//        const T &l1 = camera[7];
//        const T &l2 = camera[8];
//
//        T r2 = xp * xp + yp * yp;
//        T distortion = T(1.0) + r2 * (l1 + l2 * r2);
//
//        const T &focal = camera[6];
        predictions[0] = (camera[0] * p[0] + camera[2] * p[2]) / p[2];
        predictions[1] = (camera[1] * p[1] + camera[3] * p[2]) / p[2];
//        std::cout << predictions[0] << "   " << predictions[1] << std::endl;


        return true;
    }

    static ceres::CostFunction *Create(const double observed_x, const double observed_y){
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError,2,16,3>(new SnavelyReprojectionError(observed_x,observed_y)));
    }

private:
    double observed_x;
    double observed_y;
};


#endif //SFM_SNAVELYREPROJECTIONERROR_H
