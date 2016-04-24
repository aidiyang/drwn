
#include "viewer_lib.h"
#include "darwin_hw/sim_interface.h"
#include "darwin_hw/drwn_walker.h"

#include "estimator.h"


#include <omp.h>


std::string model_name;

extern mjModel* m; // defined in viewer_lib.h to capture perturbations
extern mjData* d;


bool walking = false;

double now_t() {
  std::chrono::time_point<std::chrono::high_resolution_clock> t
    = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> d=t.time_since_epoch();
  return d.count(); // returns milliseconds
}

int main(int argc, const char** argv) {

  omp_set_num_threads(8);

  if( argc==2 )
    model_name = argv[1];

  if (init_viz(model_name)) { return 1; }

  // init darwin 'robot'
  double s_noise = 0.0;
  double s_time_noise = 0.0;
  SimDarwin *robot = new SimDarwin(m, d, s_noise, s_time_noise);

  int nq = m->nq;
  int nv = m->nv;
  int nu = m->nu;
  int nsensordata = m->nsensordata;

  double time = 0.0;
  double prev_time = 0.0;
  double *qpos = new double[nq];
  double *qvel = new double[nv];
  double *ctrl = new double[nu];
  for (int i=0; i<nu; i++) {
    ctrl[i] = 0.0;
  }
  double *sensors = new double[nsensordata];


  // init darwin to walker pose
  Walking * walker = new Walking();
  walker->Initialize(ctrl);
  robot->set_controls(ctrl, NULL, NULL);
  
  for (int i=0; i<5; i++)
    render(window, NULL); // get state updated model / data, mj_steps

  // init UKF to darwin data
  //UKF * est = new UKF(m, d, 10e-3, 2, 0);
  UKF * est = 0;


  // init estimator from darwin 'robot'
  double dt = m->opt.timestep;
  printf("DT is set %f\n", dt);
  mjData * est_data;

  while( !closeViewer() ) {

    switch (viewer_signal()) { // signal for keyboard presses (triggers walking)
      case 1:
        if (walking) {
          walker->Stop();
          walking = false;
          printf("Starting to walk\n");
        }
        else {
          walker->Start();
          walking = true;
          printf("Stopping walk\n");
        }
        viewer_set_signal(0);
        break;
      case 2:
        if (est)
          delete est;
        est = new UKF(m, d, 10e-3, 2, 0);
        printf("New UKF initialization\n");
        viewer_set_signal(0);
        break;
      default: 
        break;
    }

    // simulate and render
    //printf("time: %f\t", d->time);
    if (robot->get_sensors(&time, sensors)) {
      // we got sensor values
      //printf("got new sensors at %f \n", time);

      walker->Process(time-prev_time, 0, ctrl);
      robot->set_controls(ctrl, NULL, NULL);

      double t0 = now_t();
      if (est)
        est->predict(ctrl, time-prev_time);

      double t1 = now_t();
      //if (est)
      //  est->correct(sensors);

      double t2 = now_t();

      printf("time: %f\t\t estimator update %f ms, predict %f ms, total %f ms\n", time, t1-t0, t2-t1, t2-t0);
      printf("qpos: ");
      for (int i=0; i<nq; i++) {
        printf("%1.4f ", d->qpos[i]);
      }
      if (est) {
        est_data = est->get_state();
        printf("\n est: ");
        for (int i=0; i<nq; i++) {
          printf("%1.4f ", est_data->qpos[i]);
        }
      }
      printf("\nctrl: ");
      for (int i=0; i<nu; i++) {
        printf("%1.4f ", ctrl[i]);
      }
      printf("\n\n");
      prev_time = time;
    }
    else {
      //printf("no  new sensors at %f \n", time);
      // we just advanced the sim, nothing else

    }

    if (est) {
    //render(window, est_data); // get state updated model / data, mj_steps
    render(window, NULL); // get state updated model / data, mj_steps
    }
    else {
    render(window, NULL); // get state updated model / data, mj_steps
    }

    finalize();
  }

  end_viz();

  // end_estimator
  // end darwin 'robot'
  //for(int id = 0; id < nq; id++) {
  //  printf("%f ", qpos[id]);
  //}

  printf("\n");
  delete[] qpos;
  delete[] qvel;
  delete[] ctrl;
  delete[] sensors;

  return 0;
}
