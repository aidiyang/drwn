
#pragma once
#include <Eigen/Dense>

class MyRobot {
  public:
    MyRobot() {}
    virtual ~MyRobot() {}

    virtual bool get_sensors(double * time, double* sensor, double * conf) { return false; }
    virtual bool set_controls(double * u, int nu, int *pgain, int *dgain) { return false; }
    bool is_running() {return darwin_ok;}

    bool darwin_ok;

    void set_frame_rotation(double * rot) {}
    void set_frame_rotation(Eigen::Matrix3d rot) {}
    void set_initpos_rt(Eigen::Vector3d vec_r, Eigen::Vector3d vec_s,
        double* mrkr, double* ps_c, double *pose) {}

};


