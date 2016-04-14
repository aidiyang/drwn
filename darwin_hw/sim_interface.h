#include "mujoco.h"

#include "Utilities.h"

#include <string.h>
#include <random>

class SimDarwin {
  private:
	bool darwin_ok;

	mjModel *m;
	mjData *d;

	double sensor_time;
	double s_dt;

	double s_noise;
	double s_time_noise;

	std::random_device rd;
	std::mt19937 gen;
	std::normal_distribution<> t_noise;


	// model helpers
	int nq;
	int nv;
	int nu;

  public:
	SimDarwin(
		mjModel *m, mjData * d,
		double s_noise, double s_time_noise) : gen(rd()), t_noise(0, s_time_noise) {
	  // s_noise ; magnitude of sensors noise 
	  // s_time_noise ; noise is getting sensor values

	  this->m = m;
	  this->d = d;

	  this->sensor_time = 0.0; 
	  this->s_dt = 0.0055; //ms in seconds
	  this->s_noise = s_noise;
	  this->s_time_noise = s_time_noise;

	  darwin_ok = true;

	  nq = m->nq;
	  nv = m->nv;
	  nu = m->nu;

	}

	~SimDarwin() {
	  //mj_deleteData(d);
	  //mj_deleteModel(m);
      //delete[] this->i_pose;
	}

	bool is_running() {
	  return darwin_ok;
	}

	//void init_pose(double *p) { // supposed to be for phasespace stuff
	//  if (!i_pose) {
	//	i_pose = new double[7];
	//  }

	//  if (p) {
	//	memcpy(i_pose, p, sizeof(double)*7);
	//  }
	//  else {
	//	memset(i_pose, 0, sizeof(double)*7);
	//  }
	//}

	bool get_state(double* time, double* qpos, double* qvel, double *sensor) {
	  
      *time = d->time;
	  if (d->time < sensor_time ) {
		//mj_step(m, d); // advanced simulation until we can get new sensor data
      // let render program advance the simulation
		return false;
	  }

	  // current time + next sensor time + sensor time noise
	  sensor_time = d->time + s_dt + t_noise(gen);

      for(int id = 0; id < nq; id++) {
        qpos[id] = d->qpos[id];
      }
      for(int id = 0; id < nv; id++) {
        qvel[id] = d->qvel[id];
      }

      // TODO check format of mujoco's sensordata
      if (sensor) {
        for (int id=0; id<m->nsensordata; id++) {
          sensor[id] = d->sensordata[id]; // s_noise perturbation;
        }
      }

      return true;
    }

    bool set_controls(double * u, int *pgain, int *dgain) {
      // converts controls to darwin positions
      for(int id = 0; id < nu; id++) {
        d->ctrl[id] = u[id];
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

      // sets controls, should stay the same for each mjstep called in
      // get_state

      return true;
    }

};

