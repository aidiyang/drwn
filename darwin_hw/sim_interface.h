#pragma once

#include "mujoco.h"

//#include "Utilities.h"
#include "robot.h"

#include "util_func.h"

#include <string.h>
#include <random>

class SimDarwin : public MyRobot {
  private:

    mjModel *m;
    mjData *d;

    double sensor_time;
    double s_dt;

    double s_time_noise;

    std::random_device rd;
    std::mt19937 gen;
    std::normal_distribution<> t_noise;
    std::normal_distribution<> sen_noise;
    std::normal_distribution<> ctrl_noise;

    std::normal_distribution<> * rd_vec; 

    // model helpers
    int nq;
    int nv;
    int ns;
    //int nu;

    bool sensor_noise;
    bool control_noise;

  public:
    SimDarwin(
        mjModel *m, mjData * d, double dt,
        double s_noise, double *s_cov, double s_time_noise, double c_noise) : gen(rd()),
    t_noise(0, s_time_noise), sen_noise(0, s_noise), ctrl_noise(0, c_noise) {
      // s_noise ; magnitude of sensors noise 
      // s_time_noise ; noise is getting sensor values

      this->m = m;
      this->d = d;

      this->sensor_time = 0.0; 
      this->s_dt = dt; // 2x the ms of the 'robot' timestep
      this->s_time_noise = s_time_noise;

      darwin_ok = true;

      nq = m->nq;
      nv = m->nv;
      ns = m->nsensordata;
      //nu = m->nu;

      sensor_noise = s_noise > 0 ? true : false;
      control_noise = c_noise > 0 ? true : false;

      if (sensor_noise) printf("Adding SENSOR noise to Simulation\n");
      if (control_noise) printf("Adding CONTROL noise to Simulation\n");

      double *snsr_vec = new double[ns];
      util::fill_sensor_vector(m, snsr_vec, s_cov); // fill up array with covariances
      rd_vec = new std::normal_distribution<>[ns]; 
      for (int i=0; i<ns; i++) {
        rd_vec[i] = std::normal_distribution<>( 0, sqrt(snsr_vec[i]) );
      }
      delete[] snsr_vec;
    }

    ~SimDarwin() {
      //mj_deleteData(d);
      //mj_deleteModel(m);
      //delete[] this->i_pose;
      delete[] rd_vec;
    }

    bool get_sensors(double * time, double* sensor, double* conf) {
      static std::mt19937 s_rng(20212223);

      *time = d->time;
      if (d->time < sensor_time ) {
        //mj_step(m, d); // advanced simulation until we can get new sensor data
        // let render program advance the simulation
        return false;
      }
      //printf("real time %f\nsnsr time: %f\n\n", d->time, sensor_time);

      // current time + next sensor time + sensor time noise
      sensor_time = d->time + s_dt;// + t_noise(gen);
      
      // get sensor values at this timestep
      mj_forwardSkip(m, d, 0, 0); // all calculations

      if (sensor) {
        for (int id=0; id<ns; id++) {
          double r = 0.0; 
          if (sensor_noise) r = rd_vec[id](s_rng);
          sensor[id] = d->sensordata[id] + r; //cs_noise perturbation;
        }
      }
      else {
        printf("Initialize sensor buffer\n");
        return false;
      }

      if (conf) {
        for (int i=0; i<16; i++)
          conf[i] = 6;
      }

      return true;
    }


    // mujoco controls to mujoco controls
    bool set_controls(double * u, int nu, int *pgain, int *dgain) {
      static std::mt19937 c_rng(987612345);
      // converts controls to darwin positions
      for(int id = 0; id < nu; id++) {
        double r = 0.0;
        if (control_noise) r = ctrl_noise(c_rng);
        d->ctrl[id] = u[id] + r;
        //printf("%f %f %f\n", u[id], d->ctrl[id], r);
      }

      // TODO setting pgain and dgain not configured yet
      if (pgain) {
        for(int id = 0; id < nu; id++) {
          m->actuator_gainprm[id*3] = pgain[id]; // mjNGAIN = 3
          m->actuator_biasprm[(id*3)+2] = -1.0 * pgain[id]; // mjNBIAS = 3
        }
      }
      if (dgain) {
        printf("Setting d gain for position actuator in Mujoco is not supported\n");
      }

      return true;
    }

};

