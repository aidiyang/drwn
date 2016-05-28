
#pragma once

class MyRobot {
  public:
    MyRobot() {}

    virtual bool get_sensors(double * time, double* sensor) { }

    virtual bool set_controls(double * u, int *pgain, int *dgain) { }

    bool is_running() {return darwin_ok;}

    bool darwin_ok;

};


