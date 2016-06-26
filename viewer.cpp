
#include "viewer_lib.h"
#include "darwin_hw/sim_interface.h"
#ifndef __APPLE__
#include "darwin_hw/interface.h"
#endif
#include "darwin_hw/robot.h"
#include "darwin_hw/drwn_walker.h"

#include "estimator.h"



#ifndef __APPLE__
#include <omp.h>
#endif

#include <iostream>
#include <fstream>


#include <boost/program_options.hpp>
namespace po = boost::program_options;


extern mjModel* m; // defined in viewer_lib.h to capture perturbations
extern mjData* d;

double init_qpos[26] = { 0.03, -0.00, -0.06, 0.00, 0.16, 0.00, -0.00, -0.40, 0.72, 0.29, -0.50, -0.84, -0.29, 0.50,
  -0.00, -0.01, 0.64, -0.94, -0.55, -0.01, 0.00, 0.01, -0.64, 0.94, 0.55, 0.01 };
double init_qvel[26] = { 0.00, -0.00, -0.01, 0.00, -0.00, 0.00, 0.00, -0.01, 0.01, 0.01, -0.01, -0.01, -0.01, 0.01,
  0.00, -0.01, 0.00, -0.02, 0.09, -0.03, -0.00, 0.00, -0.00, 0.01, -0.03, 0.04 };

bool walking = false;

double now_t() {
  std::chrono::time_point<std::chrono::high_resolution_clock> t
    = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> d=t.time_since_epoch();
  return d.count(); // returns milliseconds
}

std::ofstream myfile;
void save_states(std::string filename, double time,
        mjData * real, mjData * est, mjData * stddev,
        double t1, double t2,
        std::string mode = "w") {
  if (mode=="w") {
    // create file
    myfile.open(filename, std::ofstream::out);
    myfile<<"time,";

    if (real) {
      for (int i=0; i<m->nq; i++) 
        myfile<<"qpos,";
      for (int i=0; i<m->nv; i++) 
        myfile<<"qvel,";
      for (int i=0; i<m->nu; i++) 
        myfile<<"ctrl,";
      for (int i=0; i<m->nsensordata; i++) 
        myfile<<"snsr,";
    }

    for (int i=0; i<m->nq; i++) 
      myfile<<"est_p,";
    for (int i=0; i<m->nv; i++) 
      myfile<<"est_v,";
    for (int i=0; i<m->nu; i++) 
      myfile<<"est_c,";
    for (int i=0; i<m->nsensordata; i++) 
      myfile<<"est_s,";

    for (int i=0; i<m->nq; i++) 
      myfile<<"stddev_p,";
    for (int i=0; i<m->nv; i++) 
      myfile<<"stddev_v,";
    for (int i=0; i<m->nu; i++) 
      myfile<<"stddev_c,";
    for (int i=0; i<m->nsensordata; i++) 
      myfile<<"stddev_s,";

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

    myfile<<time<<",";
    if (real) {
      for (int i=0; i<m->nq; i++) 
        myfile<<real->qpos[i]<<",";
      for (int i=0; i<m->nv; i++) 
        myfile<<real->qvel[i]<<",";
      for (int i=0; i<m->nu; i++) 
        myfile<<real->ctrl[i]<<",";
      for (int i=0; i<m->nsensordata; i++) 
        myfile<<real->sensordata[i]<<",";
    }

    for (int i=0; i<m->nq; i++) 
      myfile<<est->qpos[i]<<",";
    for (int i=0; i<m->nv; i++) 
      myfile<<est->qvel[i]<<",";
    for (int i=0; i<m->nu; i++) 
      myfile<<est->ctrl[i]<<",";
    for (int i=0; i<m->nsensordata; i++) 
      myfile<<est->sensordata[i]<<",";

    for (int i=0; i<m->nq; i++) 
      myfile<<stddev->qpos[i]<<",";
    for (int i=0; i<m->nv; i++) 
      myfile<<stddev->qvel[i]<<",";
    for (int i=0; i<m->nu; i++) 
      myfile<<stddev->ctrl[i]<<",";
    for (int i=0; i<m->nsensordata; i++) 
      myfile<<stddev->sensordata[i]<<",";

    myfile<<t1<<",";
    myfile<<t2<<",";

    myfile<<"\n";
  }

}

void print_state(const mjModel* m, const mjData* d) {
  for (int i=0; i<m->nq; i++) {
    printf("%1.4f ", d->qpos[i]);
  }
  printf(":: ");
  for (int i=0; i<m->nv; i++) {
    printf("%1.4f ", d->qvel[i]);
  }
  printf(":: ");
  for (int i=0; i<m->nv; i++) {
    printf("%1.4f ", d->qacc[i]);
  }
  printf("\n");
}


int main(int argc, const char** argv) {

  int num_threads;
  int estimation_counts;
	//bool engage; // for real robot?
  std::string model_name;// = new std::string();
  std::string output_file;// = new std::string();
  double s_noise;
  double c_noise;
  double e_noise;
  double alpha;
  double beta;
  double kappa;
  double diag;
  double Ws0;
  double tol;
  double s_time_noise=0.0;
  //bool do_correct;
  bool debug;
  bool real_robot;

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help", "Usage guide")
			//("engage,e", po::value<bool>(&engage)->default_value(false), "Engage motors for live run.")
			("model,m", po::value<std::string>(&model_name)->required(), "Model file to load.")
			("output,o", po::value<std::string>(&output_file)->default_value("out.csv"), "Where to save output of logged data to csv.")
			("timesteps,c", po::value<int>(&estimation_counts)->default_value(-1), "Number of times to allow estimator to run before quitting.")
			//("do_correct,d", po::value<bool>(&do_correct)->default_value(true), "Do correction step in estimator.")
			("debug,n", po::value<bool>(&debug)->default_value(false), "Do correction step in estimator.")
			("real,r", po::value<bool>(&real_robot)->default_value(false), "Use real robot.")
			//("velocity,v", po::value<std::string>(vel_file), "Binary file of joint velocity data")
			("s_noise,s", po::value<double>(&s_noise)->default_value(0.0), "Gaussian amount of sensor noise to corrupt data with.")
			("c_noise,p", po::value<double>(&c_noise)->default_value(0.0), "Gaussian amount of control noise to corrupt data with.")
			("e_noise,e", po::value<double>(&e_noise)->default_value(0.0), "Gaussian amount of estimator noise to corrupt data with.")
			("alpha,a", po::value<double>(&alpha)->default_value(10e-3), "Alpha: UKF param")
			("beta,b", po::value<double>(&beta)->default_value(2), "Beta: UKF param")
			("kappa,k", po::value<double>(&kappa)->default_value(0), "Kappa: UKF param")
			("diagonal,d", po::value<double>(&diag)->default_value(1), "Diagonal amount to add to UKF covariance matrix.")
      ("weight_s,w", po::value<double>(&Ws0)->default_value(-1.0), "Set inital Ws weight.")
      ("tol,i", po::value<double>(&tol)->default_value(-1.0), "Set Constraint Tolerance (default NONE).")
			//("dt,t", po::value<double>(&dt)->default_value(0.02), "Timestep in binary file -- checks for file corruption.")
			("threads,t", po::value<int>(&num_threads)->default_value(omp_get_num_procs()>>1), "Number of OpenMP threads to use.")
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


  if(Ws0 > 1) {
    Ws0 = 1;
    printf("Inputted weight was larger than 1, Ws0 has been clamped to 1\n\n");
  }


  printf("Model:\t\t%s\n", model_name.c_str());
  printf("OMP threads:\t\t%d\n", num_threads);
  printf("Timesteps:\t\t%d\n", estimation_counts);
  printf("Sensor Noise:\t\t%f\n", s_noise);
  printf("Control Noise:\t\t%f\n", c_noise);
  printf("UKF alpha:\t\t%f\n", alpha);
  printf("UKF beta:\t\t%f\n", beta);
  printf("UKF kappa:\t\t%f\n", kappa);
  printf("UKF diag:\t\t%f\n", diag);
  printf("UKF Ws0:\t\t%f\n", Ws0);
  printf("UKF Tol:\t\t%f\n", tol);

  // Start Initializations
  omp_set_num_threads(num_threads);
  omp_set_dynamic(0);

  if (init_viz(model_name)) { return 1; }
  int nq = m->nq;
  int nv = m->nv;
  int nu = m->nu;
  int nsensordata = m->nsensordata;

  
  ////// SIMULATED ROBOT
  double dt = m->opt.timestep;
  MyRobot *robot;
#ifndef __APPLE__
  real_robot = false;
  if (real_robot) {
    ////// REAL ROBOT
    bool zero_gyro = true;
    bool use_rigid = false;
    bool use_markers = false;
    std::string ps_server = "128.208.4.128";
    double *p = NULL; // initial pose
    bool use_accel = false;
    bool use_gyro = false;
    bool use_ati = false;
    for (int i=0; i<m->nsensor; i++) { // use sensors based on mujoco model
      if (m->sensor_type[i] == mjSENS_ACCELEROMETER) use_accel = true;
      if (m->sensor_type[i] == mjSENS_GYRO) use_gyro = true;
      if (m->sensor_type[i] == mjSENS_FORCE) use_ati = true;
      //if (m->sensor_type[i] == mjSENS_TORQUE) use_ati = true;
    }
    int* p_gain = new int[nu]; 
    for (int i=0; i<m->nu; i++) { // use sensors based on mujoco model
      p_gain[i] = (int) m->actuator_gainprm[i*mjNGAIN];
      //printf("%d\n", p_gain[i]);
    }
    printf("\n\n");
    if (use_accel) printf("Using Accelerometer\n");
    if (use_gyro) printf("Using Gyroscope\n");
    if (use_ati) printf("Using Force/Torque sensors\n");

    robot = new DarwinRobot(zero_gyro, use_rigid, use_markers,
        use_accel, use_gyro, use_ati, p_gain, ps_server, p);
    delete[] p_gain;
  }
  else
#endif
    robot = new SimDarwin(m, d, 2*dt, s_noise, s_time_noise, c_noise);

  double time = 0.0;
  double prev_time = 0.0;
  double *qpos = new double[nq];
  double *qvel = new double[nv];
  double *ctrl = new double[nu];
  for (int i=0; i<nu; i++) {
    ctrl[i] = 0.0;
  }
  double *sensors = new double[nsensordata];
  //double *sensors = new double[nsensordata];

  // init darwin to walker pose
  Walking * walker = new Walking();
  if (nu >= 20) {
      walker->Initialize(ctrl);
  }
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
          printf("Stopping walk\n");
        }
        else {
          walker->Start();
          walking = true;
          printf("Starting to walk\n");
        }
        viewer_set_signal(0);
        break;
      case 2:
        if (est)
          delete est;
        printf("New UKF initialization\n");
        if (real_robot) {
          int c=25;
          for (int i=(nq-1); i>(nq-20); i--) d->qpos[i] = init_qpos[c--];
          c=25;
          for (int i=(nv-1); i>(nv-20); i--) d->qvel[i] = init_qvel[c--];
        }

        est = new UKF(m, d, alpha, beta, kappa, diag, Ws0, e_noise, tol, debug, num_threads);

        est_data = est->get_state();
        if (real_robot) save_states(output_file, 0.0, NULL, est_data, est->get_stddev(), 0, 0, "w");
        else save_states(output_file, 0.0, d, est_data, est->get_stddev(), 0, 0, "w");
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
      printf("robot sensor time: %f, est DT: %f\n", time, time-prev_time);

      //////////////////////////////////
      double t0 = now_t();
      //if (est) est->predict(ctrl, time-prev_time);

      //////////////////////////////////
      double t1 = now_t();
      //if (est) est->correct(sensors);
      if (est) est->predict_correct(ctrl, time-prev_time, sensors);


      double t2 = now_t();


      printf("\n\t\t estimator predict %f ms, correct %f ms, total %f ms\n\n",
          t1-t0, t2-t1, t2-t0);

      //printf("qpos at t: ");
      //for (int i=0; i<nq; i++) {
      //  printf("%1.6f ", d->qpos[i]);
      //}
      //if (est) {
      //  est_data = est->get_state();
      //  printf("\n est at t: ");
      //  for (int i=0; i<nq; i++) {
      //    printf("%1.6f ", est_data->qpos[i]);
      //  }
      //}
      //printf("\nraw snsr: ");
      //for (int i=0; i<nsensordata; i++) {
      //  if (real_robot) printf("%1.4f ", sensors[i]);
      //  else printf("%1.4f ", d->sensordata[i]);
      //}
      if (est) {
        printf("\n\nSensor Compare:\nreal: ");
        for (int i=40; i<nsensordata; i++) {
          if (real_robot) printf("%1.4f ", sensors[i]);
          else printf("%1.4f ", d->sensordata[i]);
        }
        printf("\n est: ");
        for (int i=40; i<nsensordata; i++) {
          printf("%1.4f ", est_data->sensordata[i]);
        }
      }
      printf("\n");

      printf("\nctrl: ");
      for (int i=0; i<nu; i++) {
        printf("%1.4f ", ctrl[i]);
      }
      printf("\n\n");


      if (est) {
        if (real_robot) save_states(output_file, time, NULL, est_data, est->get_stddev(), t1-t0, t2-t1, "a");
        else save_states(output_file, time, d, est_data, est->get_stddev(), t1-t0, t2-t1, "a");
      }

      // we have estimated and logged the data,
      // now get new controls
      if (nu >= 20) {
          walker->Process(time-prev_time, 0, ctrl);
      }
      robot->set_controls(ctrl, NULL, NULL);

      prev_time = time;
      if (est && estimation_counts > 0) {
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
      render(window, est->get_sigmas(), color); // get state updated model / data, mj_steps
    }
    else {
      std::vector<mjData*> a;
      render(window, a, false);
    }

    finalize();
  }

  end_viz();
  if (est) {
    if (real_robot) save_states(output_file, time, NULL, est_data, est->get_stddev(), 0, 0, "c");
    else save_states(output_file, time, d, est_data, est->get_stddev(), 0, 0, "c");
    // close file
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
