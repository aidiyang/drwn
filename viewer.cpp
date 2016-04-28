
#include "viewer_lib.h"
#include "darwin_hw/sim_interface.h"
#include "darwin_hw/drwn_walker.h"

#include "estimator.h"

#include <cnpy.h>


#include <omp.h>
#include <iostream>
#include <fstream>


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

std::ofstream myfile;
void save_states(std::string filename, mjData * real, mjData * est, std::string mode = "w") {
  if (mode=="w") {
    // create file
    myfile.open(filename, std::ofstream::out);
    myfile<<"time,";
    for (int i=0; i<m->nq; i++) 
      myfile<<"qpos,";
    for (int i=0; i<m->nv; i++) 
      myfile<<"qvel,";
    for (int i=0; i<m->nu; i++) 
      myfile<<"ctrl,";
    for (int i=0; i<m->nsensordata; i++) 
      myfile<<"snsr,";

    for (int i=0; i<m->nq; i++) 
      myfile<<"est_p,";
    for (int i=0; i<m->nv; i++) 
      myfile<<"est_v,";
    for (int i=0; i<m->nu; i++) 
      myfile<<"est_c,";
    for (int i=0; i<m->nsensordata; i++) 
      myfile<<"est_s,";
    myfile<<"\n";

    myfile.close();
  }
  else if (mode=="c") {
    myfile.close();
    return;
  }
  else {
    if (!myfile.is_open()) {
      printf("HAD TO OPEN OUTPUT FILE AGAIN!!!!!!\n");
      myfile.open(filename, std::ofstream::out | std::ofstream::app );
    }

    myfile<<d->time<<",";
    for (int i=0; i<m->nq; i++) 
      myfile<<d->qpos[i]<<",";
    for (int i=0; i<m->nv; i++) 
      myfile<<d->qvel[i]<<",";
    for (int i=0; i<m->nu; i++) 
      myfile<<d->ctrl[i]<<",";
    for (int i=0; i<m->nsensordata; i++) 
      myfile<<d->sensordata[i]<<",";

    for (int i=0; i<m->nq; i++) 
      myfile<<est->qpos[i]<<",";
    for (int i=0; i<m->nv; i++) 
      myfile<<est->qvel[i]<<",";
    for (int i=0; i<m->nu; i++) 
      myfile<<est->ctrl[i]<<",";
    for (int i=0; i<m->nsensordata; i++) 
      myfile<<est->sensordata[i]<<",";
    myfile<<"\n";
  }

}

void print_state(const mjModel* m, const mjData* d) {
  for (int i=0; i<m->nq; i++) {
    printf("%1.4f ", d->qpos[i]);
  }
  for (int i=0; i<m->nv; i++) {
    printf("%1.4f ", d->qvel[i]);
  }
  printf("\n");
}


int main(int argc, const char** argv) {

  omp_set_num_threads(12);

  if( argc==2 )
    model_name = argv[1];

  if (init_viz(model_name)) { return 1; }


  std::string output_file = "out.csv";
  // init darwin 'robot'
  double s_noise = 0.00;
  double s_time_noise = 0.0;

  
  double dt = m->opt.timestep;
  SimDarwin *robot = new SimDarwin(m, d, 2.0*dt, s_noise, s_time_noise);

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

  //for (int i=0; i<5; i++)
  //  render(window, NULL); // get state updated model / data, mj_steps

  // init UKF to darwin data
  //UKF * est = new UKF(m, d, 10e-3, 2, 0);
  UKF * est = 0;


  // init estimator from darwin 'robot'
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
        printf("New UKF initialization\n");
        est = new UKF(m, d, 10e-5, 2, 0);

        est_data = est->get_state();
        save_states(output_file, d, est_data, "w");
        viewer_set_signal(0);
        break;
      default: 
        break;
    }

    // simulate and render
    //printf("time: %f\t", d->time);
    if (robot->get_sensors(&time, sensors)) {

      printf("robot hw time: %f\n", time);
      printf("true state:\n");
      print_state(m, d);

      double t0 = now_t();
      if (est) est->predict(ctrl, time-prev_time);

      //////////////////////////////////
      //
      printf("robot sensor time: %f\n", d->time);
      //printf("true sensors:\n");
      //for (int i=0; i<nsensordata; i++) {
      //  printf("%1.6f ", d->sensordata[i]);
      //}
      //printf("\n");

      //////////////////////////////////
      double t1 = now_t();
      if (est) est->correct(sensors);

      double t2 = now_t();


      printf("\t\t estimator update %f ms, predict %f ms, total %f ms\n",
          t1-t0, t2-t1, t2-t0);
      printf("qpos at t: ");
      for (int i=0; i<nq; i++) {
        printf("%1.6f ", d->qpos[i]);
      }
      if (est) {
        est_data = est->get_state();
        printf("\n est at t: ");
        for (int i=0; i<nq; i++) {
          printf("%1.6f ", est_data->qpos[i]);
        }
      }printf("\nsnsr: ");
      for (int i=0; i<nsensordata; i++) {
        printf("%1.4f ", d->sensordata[i]);
      }
      printf("\nctrl: ");
      for (int i=0; i<nu; i++) {
        printf("%1.4f ", ctrl[i]);
      }
      printf("\n\n");

      if (est) {
        save_states(output_file, d, est_data, "a");
      }

      // we have estimated and logged the data,
      // now get new controls
      walker->Process(time-prev_time, 0, ctrl);
      robot->set_controls(ctrl, NULL, NULL);

      prev_time = time;

    }
    else {
      //printf("no  new sensors at %f \n", time);
      // we just advanced the sim, nothing else
    }

    if (est) {
      //render(window, est_data); // get state updated model / data, mj_steps
      render(window, est->get_sigmas()); // get state updated model / data, mj_steps
    }
    else {
      std::vector<mjData*> a;
      render(window, a);
    }

    finalize();
  }

  end_viz();
  if (est) {
    save_states(output_file, d, est_data, "c"); // close file
  }

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
