
#include "viewer_lib.h"
#include "darwin_hw/sim_interface.h"
#include "darwin_hw/drwn_walker.h"

#include "estimator.h"


#include <omp.h>
#include <iostream>
#include <fstream>


#include <boost/program_options.hpp>
namespace po = boost::program_options;


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
void save_states(std::string filename,
        mjData * real, mjData * est, double t1, double t2,
        std::string mode = "w") {
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

    myfile<<"predict,";
    myfile<<"correct,";
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

    myfile<<t1<<",";
    myfile<<t2<<",";

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

  int num_threads;
  int estimation_counts;
	bool engage; // for real robot?
  std::string model_name;// = new std::string();
  std::string output_file;// = new std::string();
  double s_noise;
  double c_noise;
  double e_noise;
  double alpha;
  double beta;
  double kappa;
  double s_time_noise=0.0;
  bool do_correct;
  bool debug;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help", "Usage guide")
			//("engage,e", po::value<bool>(&engage)->default_value(false), "Engage motors for live run.")
			("model,m", po::value<std::string>(&model_name)->required(), "Model file to load.")
			("output,o", po::value<std::string>(&output_file)->default_value("out.csv"), "Where to save output of logged data to csv.")
			("timesteps,c", po::value<int>(&estimation_counts)->default_value(-1), "Number of times to allow estimator to run before quitting.")
			("do_correct,d", po::value<bool>(&do_correct)->default_value(true), "Do correction step in estimator.")
			("debug,n", po::value<bool>(&debug)->default_value(false), "Do correction step in estimator.")
			//("velocity,v", po::value<std::string>(vel_file), "Binary file of joint velocity data")
			("s_noise,s", po::value<double>(&s_noise)->default_value(0.0), "Gaussian amount of sensor noise to corrupt data with.")
			("c_noise,p", po::value<double>(&c_noise)->default_value(0.0), "Gaussian amount of control noise to corrupt data with.")
			("e_noise,e", po::value<double>(&e_noise)->default_value(0.0), "Gaussian amount of estimator noise to corrupt data with.")
			("alpha,a", po::value<double>(&alpha)->default_value(10e-3), "Gaussian amount of sensor noise to corrupt data with.")
			("beta,b", po::value<double>(&beta)->default_value(2), "Gaussian amount of control noise to corrupt data with.")
			("kappa,k", po::value<double>(&kappa)->default_value(0), "Gaussian amount of estimator noise to corrupt data with.")
			//("dt,t", po::value<double>(&dt)->default_value(0.02), "Timestep in binary file -- checks for file corruption.")
			("threads,t", po::value<int>(&num_threads)->default_value(omp_get_num_threads()), "Number of OpenMP threads to use.")
			//("i_gain,i", po::value<int>(&i_gain)->default_value(0), "I gain of PiD controller, 0-32")
			//("d_gain,d", po::value<int>(&d_gain)->default_value(0), "D gain of PiD controller, 0-32")
			;

		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);
		if (vm.count("help")) {
			std::cout << desc << std::endl;
			return 0;
		}
		po::notify(vm);
	}
	catch(std::exception& e) {
		std::cerr << "Error: " << e.what() << "\n";
		return 0;
	}
	catch(...) {
		std::cerr << "Unknown error!\n";
		return 0;
	}

  printf("Model:\t\t%s\n", model_name.c_str());
  printf("OMP threads:\t\t%d\n", num_threads);
  printf("Timesteps:\t\t%d\n", estimation_counts);
  printf("Sensor Noise:\t\t%f\n", s_noise);
  printf("Control Noise:\t\t%f\n", c_noise);
  printf("UKF alpha:\t\t%f\n", alpha);
  printf("UKF beta:\t\t%f\n", beta);
  printf("UKF kappa:\t\t%f\n", kappa);

  // Start Initializations
  omp_set_num_threads(num_threads);
  omp_set_dynamic(0);

  if (init_viz(model_name)) { return 1; }

  
  double dt = m->opt.timestep;
  SimDarwin *robot = new SimDarwin(m, d, 2.0*dt, s_noise, s_time_noise, c_noise);

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

  UKF * est = 0;

  // init estimator from darwin 'robot'
  printf("DT is set %f\n", dt);
  mjData * est_data;
  bool color = false;

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
        est = new UKF(m, d, alpha, beta, kappa, e_noise, debug, num_threads);

        est_data = est->get_state();
        save_states(output_file, d, est_data, 0, 0, "w");
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

      //mj_forward(m, d); // to get the sensord data at the current place
      //mj_sensor(m, d);


      //////////////////////////////////
      printf("robot sensor time: %f\n", d->time);

      //////////////////////////////////
      double t0 = now_t();
      if (est) est->predict(ctrl, time-prev_time);


      printf("true state:\n");
      print_state(m, d);
      printf("\nprovided snsr data: ");
      for (int i=0; i<nsensordata; i++) {
        printf("%1.6f ", sensors[i]);
      }
      printf("\n");

      //////////////////////////////////
      double t1 = now_t();
      if (est && do_correct) est->correct(sensors);

      double t2 = now_t();


      printf("\n\t\t estimator predict %f ms, correct %f ms, total %f ms\n\n",
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


      if (est) { save_states(output_file, d, est_data, t1-t0, t2-t1, "a"); }

      // we have estimated and logged the data,
      // now get new controls
      walker->Process(time-prev_time, 0, ctrl);
      robot->set_controls(ctrl, NULL, NULL);

      prev_time = time;
      if (estimation_counts > 0) {
        estimation_counts--;
      }
      else if (estimation_counts == 0) {
        shouldCloseViewer();
      }
      printf("RENDERING CORRECTED SIGMA POINTS\n");
      color = true;
    }
    else {
      // allow time to progress
    }

    if (est) {
      // render sigma points
      render(window, est->get_sigmas(), nq+nv, color); // get state updated model / data, mj_steps
    }
    else {
      std::vector<mjData*> a;
      render(window, a, 0, false);
    }

    finalize();
  }

  end_viz();
  if (est) {
    save_states(output_file, d, est_data, 0, 0, "c"); // close file
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
