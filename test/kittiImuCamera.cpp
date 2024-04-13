//
// Created by whitby on 10/17/23.
//

#include "BasicObject/Pose.h"

#include <iostream>

using namespace std;
using namespace mono_orb_slam3;

int main() {
    Eigen::Matrix3f R_velo_imu, R_cam_velo;
    Eigen::Vector3f t_velo_imu, t_cam_velo;
    R_velo_imu << 9.999976e-01, 7.553071e-04, -2.035826e-03, -7.854027e-04, 9.998898e-01, -1.482298e-02, 2.024406e-03, 1.482454e-02, 9.998881e-01;
    t_velo_imu << -8.086759e-01, 3.195559e-01, -7.997231e-01;

    R_cam_velo << 7.027555e-03, -9.999753e-01, 2.599616e-05, -2.254837e-03, -4.184312e-05, -9.999975e-01, 9.999728e-01, 7.027479e-03, -2.255075e-03;
    t_cam_velo << -7.137748e-03, -7.482656e-02, -3.336324e-01;

    Pose T_velo_imu(R_velo_imu, t_velo_imu), T_cam_velo(R_cam_velo, t_cam_velo);
    Pose T_cam_imu = T_cam_velo * T_velo_imu;
    T_cam_imu.normalize();

    Pose T_imu_cam = T_cam_imu.inverse();
    T_imu_cam.normalize();
    cout.precision(10);
    cout << T_imu_cam.R << endl;
    cout << T_imu_cam.t << endl;

    return 0;
}