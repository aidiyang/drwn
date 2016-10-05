
#include "util_func.h"
#include "viewer_lib.h"
#include "darwin_hw/drwn_walker.h"
#include "darwin_hw/sim_interface.h"
#include "darwin_hw/file_interface.h"

#ifndef __APPLE__
#include "darwin_hw/interface.h"
//#include <omp.h>
#endif

#include "darwin_hw/robot.h"

#include "estimator.h"
//#include "ekf_estimator.h"

#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

extern mjModel* m; // defined in viewer_lib.h to capture perturbations
extern mjData* d;

double s_ps_walk[16*3] = {
   -0.0246,  0.0560, 0.3758,
   -0.0711, -0.0540, 0.3161,
    0.0210,  0.0450, 0.3601,
    0.0417,  0.0180, 0.3144,
   -0.0246, -0.0560, 0.3758,
   -0.0711,  0.0540, 0.3161,
    0.0210, -0.0450, 0.3601,
    0.0417, -0.0180, 0.3144,
   -0.0795, -0.0743, 0.0169,
   -0.0795,  0.0743, 0.0169,
    0.0084, -0.0742, 0.0117,
    0.0084,  0.0742, 0.0117,
    0.0061, -0.1081, 0.2651,
    0.0205, -0.1174, 0.2375,
    0.0161,  0.1082, 0.2655,
    0.0337,  0.1176, 0.2399};


double init_qpos[26] = {
  0.0423, 0.0002, -0.0282, 0.0004, 0.2284, -0.0020, 0.0000, -0.4004, 0.7208, 0.2962, -0.5045, -0.8434, -0.2962, 0.5045, -0.0005, -0.0084, 0.6314, -0.9293, -0.5251, -0.0115, 0.0005, 0.0084, -0.6314, 0.9293, 0.5251, 0.0115};
double init_qvel[26] = { 0.00, -0.00, -0.01, 0.00, -0.00, 0.00, 0.00, -0.01, 0.01, 0.01, -0.01, -0.01, -0.01, 0.01,
  0.00, -0.01, 0.00, -0.02, 0.09, -0.03, -0.00, 0.00, -0.00, 0.01, -0.03, 0.04 };

double fall_qpos[26] = {
  0.40, -0.00, -0.31, 0.25, 1.63, -0.19, 0.00, -0.41, 0.74, 0.30, -0.50, -0.84, -0.29, 0.50, 0.00, -0.01, 0.62, -0.94, -0.52, -0.01, 0.01, 0.00, -0.60, 0.95, 0.52, 0.01};
double fall_qvel[26] = {
  -0.00, -0.00, -0.00, -0.02, 0.00, 0.03, 0.00, 0.01, -0.01, 0.00, -0.00, -0.00, -0.00, -0.00, 0.00, 0.00, -0.00, 0.00, 0.00, -0.00, 0.00, 0.00, -0.00, -0.00, -0.01, 0.00};

bool walking = false;

std::ofstream myfile;
void save_states(std::string filename, double time,
    mjData * real, mjData * est, mjData * stddev,
    double t1, double t2,
    std::string mode = "w") {
  int nu = m->nu;
  //int nu = 20;
  if (mode=="w") {
    // create file
    myfile.open(filename, std::ofstream::out);
    myfile<<"time,";

    if (real) {
      for (int i=0; i<m->nq; i++) 
        myfile<<"qpos,";
      for (int i=0; i<m->nv; i++) 
        myfile<<"qvel,";
      for (int i=0; i<nu; i++) 
        myfile<<"ctrl,";
      for (int i=0; i<m->nsensordata; i++) 
        myfile<<"snsr,";
    }

    for (int i=0; i<m->nq; i++) 
      myfile<<"est_p,";
    for (int i=0; i<m->nv; i++) 
      myfile<<"est_v,";
    for (int i=0; i<nu; i++) 
      myfile<<"est_c,";
    for (int i=0; i<m->nsensordata; i++) 
      myfile<<"est_s,";

    for (int i=0; i<m->nq; i++) 
      myfile<<"stddev_p,";
    for (int i=0; i<m->nv; i++) 
      myfile<<"stddev_v,";
    for (int i=0; i<nu; i++) 
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

    std::stringstream my_ss;
    my_ss<<time<<",";
    if (real) {
      for (int i=0; i<m->nq; i++) 
        my_ss<<real->qpos[i]<<",";
      for (int i=0; i<m->nv; i++) 
        my_ss<<real->qvel[i]<<",";
      for (int i=0; i<nu; i++) 
        my_ss<<real->ctrl[i]<<",";
      for (int i=0; i<m->nsensordata; i++) 
        my_ss<<real->sensordata[i]<<",";
    }

    if (est) {
      for (int i=0; i<m->nq; i++) 
        my_ss<<est->qpos[i]<<",";
      for (int i=0; i<m->nv; i++) 
        my_ss<<est->qvel[i]<<",";
      for (int i=0; i<nu; i++) 
        my_ss<<est->ctrl[i]<<",";
      for (int i=0; i<m->nsensordata; i++) 
        my_ss<<est->sensordata[i]<<",";
    }

    if (stddev) {
      for (int i=0; i<m->nq; i++) 
        my_ss<<stddev->qpos[i]<<",";
      for (int i=0; i<m->nv; i++) 
        my_ss<<stddev->qvel[i]<<",";
      for (int i=0; i<nu; i++) 
        my_ss<<stddev->ctrl[i]<<",";
      for (int i=0; i<m->nsensordata; i++) 
        my_ss<<stddev->sensordata[i]<<",";
    }

    my_ss<<t1<<",";
    my_ss<<t2<<",";

    my_ss<<"\n";

    myfile << my_ss.rdbuf();
  }

}


void print_state(const mjModel* m, const mjData* d) {
  printf("qpos:\n");
  for (int i=0; i<m->nq; i++) {
    printf("%1.4f ", d->qpos[i]);
  }
  printf("\nqvel:\n");
  for (int i=0; i<m->nv; i++) {
    printf("%1.4f ", d->qvel[i]);
  }
  printf("\nctrl:\n");
  for (int i=0; i<m->nu; i++) {
    //printf("%1.4f ", d->qacc[i]);
    printf("%1.4f ", d->ctrl[i]);
  }
  printf("\n");
}

void calc_torques(mjModel * m, mjData *d, double * u) {
  printf("\nControls:\n");
  for(int id = 0; id < m->nu; id++) {
    printf("%1.4f ", u[id]);
  }
  printf("\nPositions:\n");
  for(int id = 0; id < m->nu; id++) {
    printf("%1.4f ", d->sensordata[id]);
  }
  printf("\nmujoco torques:\n");
  for(int id = 0; id < m->nu; id++) {
    printf("%1.4f ", d->actuator_force[id]);
  }
  printf("\nmanual torques:\n");
  for(int id = 0; id < m->nu; id++) {
    double torque = 20 * (u[id] - d->sensordata[id]);
    printf("%1.4f ", torque);
  }
  printf("\n");
}

int main(int argc, const char** argv) {

  int num_threads=1;
  int estimation_counts;
  //bool engage; // for real robot?
  std::string model_name;// = new std::string();
  std::string output_file;// = new std::string();
  std::string input_file;
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
  bool useUKF;
  bool render_robot;

  try {
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help", "Usage guide")
      //("engage,e", po::value<bool>(&engage)->default_value(false), "Engage motors for live run.")
      ("model,m", po::value<std::string>(&model_name)->required(), "Model file to load.")
      ("output,o", po::value<std::string>(&output_file)->default_value("out.csv"), "Where to save output of logged data to csv.")
      ("file,f", po::value<std::string>(&input_file)->default_value(""), "Use saved ctrl/sensor data as real robot.")
      ("timesteps,c", po::value<int>(&estimation_counts)->default_value(-1), "Number of times to allow estimator to run before quitting.")
      //("do_correct,d", po::value<bool>(&do_correct)->default_value(true), "Do correction step in estimator.")
      ("debug,n", po::value<bool>(&debug)->default_value(false), "Debugging output.")
      ("real,r", po::value<bool>(&real_robot)->default_value(false), "Use real robot.")
      ("render,R", po::value<bool>(&render_robot)->default_value(true), "Render the visualizer.")
      //("velocity,v", po::value<std::string>(vel_file), "Binary file of joint velocity data")
      ("s_noise,s", po::value<double>(&s_noise)->default_value(0.0), "Gaussian amount of sensor noise to corrupt data with.")
      ("c_noise,p", po::value<double>(&c_noise)->default_value(0.0), "Gaussian amount of control noise to corrupt data with.")
      ("e_noise,e", po::value<double>(&e_noise)->default_value(0.0), "Gaussian amount of estimator noise to corrupt data with.")
      ("alpha,a", po::value<double>(&alpha)->default_value(1e-3), "Alpha: UKF param")
      ("beta,b", po::value<double>(&beta)->default_value(2), "Beta: UKF param")
      ("kappa,k", po::value<double>(&kappa)->default_value(0), "Kappa: UKF param")
      ("diagonal,d", po::value<double>(&diag)->default_value(1), "Diagonal amount to add to UKF covariance matrix.")
      ("weight_s,w", po::value<double>(&Ws0)->default_value(-1.0), "Set inital Ws weight.")
      ("tol,i", po::value<double>(&tol)->default_value(-1.0), "Set Constraint Tolerance (default NONE).")
      ("UKF,u", po::value<bool>(&useUKF)->default_value(true), "Use UKF or EKF")
      //("dt,t", po::value<double>(&dt)->default_value(0.02), "Timestep in binary file -- checks for file corruption.")
#ifndef __APPLE__
      ("threads,t", po::value<int>(&num_threads)->default_value(std::thread::hardware_concurrency()>>1), "Number of OpenMP threads to use.")
#endif
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

  bool from_file = input_file.length() > 0;
  bool from_hardware = (from_file == false) && real_robot;

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
  if (from_hardware) printf("Using real robot!!!\n");
  if (from_file) printf("Using real data from file!!!\n");

  // Start Initializations
//#ifndef __APPLE__
//  omp_set_dynamic(0);
//  omp_set_num_threads(num_threads);
//#endif

  if (render_robot) {
    if (init_viz(model_name)) { return 1; }
  }
  else {
    printf("MuJoCo Pro library version %.2lf\n\n", 0.01 * mj_version());
    if (mjVERSION_HEADER != mj_version())
      mju_error("Headers and library have different versions");

    // activate MuJoCo license
    mj_activate("mjkey.txt");

    if (!model_name.empty()) {
      //m = mj_loadModel(model_name.c_str(), 0, 0);
      char error[1000] = "could not load binary model";
      m = mj_loadXML(model_name.c_str(), 0, error, 1000);
      if (!m) {
        printf("%s\n", error);
        return 1;
      }

      d = mj_makeData(m);
      mj_forward(m, d);
    }
  }
  int nq = m->nq;
  int nv = m->nv;
  int nu = m->nu;
  int nsensordata = m->nsensordata;
  
  bool darwin_model = model_name.find("darwin") != std::string::npos;

  ////// SIMULATED ROBOT
  double dt = m->opt.timestep;
  double kp = 20;
  double min_t = 0.0;
  int A_SIZE=3;
  int G_SIZE=3;
  int CONTACTS_SIZE=12;
  int NMARKERS=16;
  int MARKER_SIZE = NMARKERS*3;
  
double time = 0.0;
  double prev_time = 0.0;
  double *qpos = new double[nq];
  double *qvel = new double[nv];
  double *ctrl = new double[nu];
  double *trqs = new double[nu];
  for (int i=0; i<nu; i++) {
    ctrl[i] = 0.0;
    trqs[i] = 0.0;
  }
  double *sensors = new double[nsensordata];
  double *conf = new double[NMARKERS];
  //double lp_dt = dt;
  printf("DT of mujoco model %f\n", dt);
  MyRobot *robot;
  int mrkr_idx = 0;
#ifndef __APPLE__
  if (real_robot) {
    ////// REAL ROBOT
    bool use_cm730 = true;
    bool zero_gyro = true;
    bool use_rigid = false;
    bool use_markers = true;
    bool use_raw = false;
    std::string ps_server = "128.208.4.49";
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
    //use_markers = false;
    mrkr_idx = 40+
      use_accel*A_SIZE+
      use_gyro*G_SIZE+
      use_ati*CONTACTS_SIZE;


    int* p_gain = new int[nu]; 
    for (int i=0; i<nu; i++) { // use sensors based on mujoco model
      p_gain[i] = (int) m->actuator_gainprm[i*mjNGAIN];
    }
    kp = p_gain[0];
    printf("KP: %f\n", kp);
    double * min_t_ptr = util::get_numeric_field(m, "min_torque", NULL);
    if (min_t_ptr) {
      min_t = min_t_ptr[0];
      printf("Torque Minimium for Deadband: %f\n", min_t);
    }

    printf("\n\n");
    if (use_accel) printf("Using Accelerometer\n");
    if (use_gyro) printf("Using Gyroscope\n");
    if (use_ati) printf("Using Force/Torque sensors\n");
    if (use_rigid || use_markers) printf("Using Phasespace Tracking\n");

    if (from_file) {
      robot = new FileDarwin(d, dt, nu, m->nsensordata, input_file, render_robot);
    }
    else {
      robot = new DarwinRobot(use_cm730, zero_gyro, use_rigid, use_markers, use_raw,
          use_accel, use_gyro, use_ati, p_gain, ps_server, p);

      // MARKER ROTATION AND OFFSET
      Matrix3d rot;
      rot<<0.996336648486483, -0.0855177343170566, 0, 0.0855177343170566, 0.996336648486483, 0, 0, 0, 1;
      if (use_markers || use_rigid) {
        double *ps = new double[MARKER_SIZE];
        double *ps_c = new double[NMARKERS];
        robot->set_frame_rotation(rot);
        int c1 = 8; //7; // chest marker positions
        int c2 = 9; //3;
        double* mrkr = sensors+mrkr_idx;
        kb_changemode(1);
        while (!kbhit()) { // double check the chest marker positioning
          robot->get_sensors(&time, sensors, conf);
          printf("\r");
          printf("M1: %1.4f %1.4f %1.4f : %1.4f\t\t", mrkr[c1*3+0], mrkr[c1*3+1], mrkr[c1*3+2], conf[c1]);
          printf("M2: %1.4f %1.4f %1.4f : %1.4f", mrkr[c2*3+0], mrkr[c2*3+1], mrkr[c2*3+2], conf[c2]);
        }
        int c=getchar();
        kb_changemode(0);
        // after chest markers are in view, average their values
        int m_count = 100;
        buffer_markers(robot, ps, ps_c, sensors, conf, mrkr_idx, NMARKERS, m_count);
        // ps has been averaged of good markers

        Vector3d v1(ps[c1*3+0], ps[c1*3+1], ps[c1*3+2]); // these should be good from above
        Vector3d v2(ps[c2*3+0], ps[c2*3+1], ps[c2*3+2]);
        //Vector3d vec_r = rot*v1 - rot*v2;
        Vector3d vec_r = v1 - v2;
        vec_r[2] = 0;
        Vector3d vec_s(0, -1, 0); // basically only in y axis

        // set these at the same time?
        robot->set_initpos_rt(vec_r, vec_s, ps, ps_c, s_ps_walk); // pass the clean data

        delete[] ps;
        delete[] ps_c;
      }
      else {
        printf("Skipping Rotation Calibration\n");
      }
    }
    delete[] p_gain;

  }
  else
#endif
    robot = new SimDarwin(m, d, dt, s_noise, s_time_noise, c_noise);


  // init darwin to walker pose
  Walking * walker = new Walking();
  if (nu >= 20) {
    walker->Initialize(ctrl);
  }
  if (from_file) {
    if (input_file.find("fallen") < input_file.length()) { // fallen initial condition
      printf("Loading fallen initial position.\n");
      for (int i=0; i<20; i++) d->ctrl[i] = ctrl[i];
      for (int i=0; i<nq; i++) d->qpos[i] = fall_qpos[i];
    }
    else {
      printf("Loading crouched initial position.\n");
      for (int i=0; i<20; i++) d->ctrl[i] = ctrl[i];
      for (int i=0; i<nq; i++) d->qpos[i] = init_qpos[i];
    }
    //for (int i=0; i<nv; i++) d->qvel[i] = 0;

    for (int i=0; i<100; i++)
      mj_step(m, d);
  }
  //if (from_hardware) 
  if (darwin_model)
  {
      std::cout<< model_name << "\n";
    printf("Loading crouched initial position.\n");
    for (int i=0; i<20; i++) d->ctrl[i] = ctrl[i];
    for (int i=0; i<nq; i++) d->qpos[i] = init_qpos[i];
    for (int i=0; i<1000; i++)
      mj_step(m, d);
  }
  //util::darwin_torques(trqs, m, d, d->ctrl, min_t, kp); // set controls from above
  //robot->set_controls(trqs, 20, NULL, NULL);
  robot->set_controls(ctrl, 20, NULL, NULL);

  Estimator * est = 0;

  // init estimator from darwin 'robot'
  printf("DT is set %f\n", dt);
  mjData * est_data = 0;
  bool render_inplace = false;

  double * s_cov = util::get_numeric_field(m, "snsr_covar", NULL);
  double * p_cov = util::get_numeric_field(m, "covar_diag", NULL);


  if (render_robot == false) { // if we are not rendering just make the ukf
    est = new UKF(m, d, s_cov, p_cov,
        alpha, beta, kappa, diag, Ws0, e_noise, tol, debug, num_threads);
    est_data = est->get_state();
    if (real_robot) save_states(output_file, 0.0, NULL, est_data, est->get_stddev(), 0, 0, "w");
    else save_states(output_file, 0.0, d, est_data, est->get_stddev(), 0, 0, "w");
  }

  std::future<void> t_predict;

  //double render_t = util::now_t();
  bool exit = render_robot ? true : !closeViewer();
  double real_dt = 0.007;
  double lp_alpha = 0.09;
  while ( exit ) {

    if (render_robot) {
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
          if (est) delete est;
          if (useUKF) {
            printf("New UKF initialization\n");

            est = new UKF(m, d, s_cov, p_cov,
                alpha, beta, kappa, diag, Ws0, e_noise, tol, debug, num_threads);
          } else {
            printf("New EKF initialization\n");
            //est = new EKF(m, d, e_noise, tol, diag, debug, num_threads);
          }

          est_data = est->get_state();
          if (real_robot) save_states(output_file, 0.0, NULL, est_data, est->get_stddev(), 0, 0, "w");
          else save_states(output_file, 0.0, d, est_data, est->get_stddev(), 0, 0, "w");
          viewer_set_signal(0);
          break;
        default: 
          break;
      }
    }
    //double t_2 = util::now_t();
    double thread_t=0;

    bool get_data_and_estimate = false;
    if (from_file) { // don't process unless we are estimating
      bool sense = robot->get_sensors(&time, sensors, conf);
      get_data_and_estimate = (est!=NULL) && sense;
      if (sense == false // no data
          && time < 0
          && render_robot == false) {// not rendering
        printf("Done with file, readying to exit.\n");
        exit = false; //shouldCloseViewer(); // end of file
      }
    }
    else if (from_hardware && est) {
      thread_t = util::now_t();
      t_predict = std::async(std::launch::async, &Estimator::predict_correct_p1, est, ctrl, real_dt, sensors, conf);
      double t1 = util::now_t();
      get_data_and_estimate = robot->get_sensors(&time, sensors, conf);
      double t2 = util::now_t();
      printf("Sensor time : %f\n", t2-t1);
    }
    else if (from_hardware && !est) {
      thread_t = util::now_t();
      get_data_and_estimate = robot->get_sensors(&time, sensors, conf);
    }
    else {
      // sim robot
      get_data_and_estimate = robot->get_sensors(&time, sensors, conf);
    }

    // simulate and render
    //printf("time: %f\t", d->time);
    //printf("time to get sensors: %f\n", util::now_t() - thread_t);
    if (get_data_and_estimate) {

      printf("robot hw time: %f\n", time);
      printf("true state:\n");
      //print_state(m, d);

      //////////////////////////////////
      printf("robot sensor time: %f, est DT: %f\n", time, time-prev_time);
      if (est) printf("Estimator time: %f\n", est_data->time);
      real_dt = real_dt + lp_alpha * ((time-prev_time)-real_dt);
      printf("low passed dt: %f\n", real_dt);

      //////////////////////////////////
      double t0 = util::now_t();
      //if (est) est->predict(ctrl, time-prev_time);
      if (est) {
        if (from_hardware) {
          t_predict.get(); // waits for this completion
          //est->predict_correct_p1(ctrl, time-prev_time, sensors, conf);
        }
        else {  
          ///est->predict_correct_p1(trqs, time-prev_time, sensors, conf);
          est->predict_correct_p1(ctrl, time-prev_time, sensors, conf);
        }
      }
      //double a = 0.5;
      //lp_dt = a * lp_dt + (1-a)*((time-prev_time));

      //////////////////////////////////
      double t1 = util::now_t();
      //if (est) est->correct(sensors);
      //if (est) est->predict_correct(ctrl, time-prev_time, sensors, conf);
      if (est) {
        //est->predict_correct_p2(trqs, time-prev_time, sensors, conf);
        est->predict_correct_p2(ctrl, time-prev_time, sensors, conf);
      }
      double t2 = util::now_t();

      if (from_hardware) 
        printf("\n\t\t estimator predict %f ms, correct %f ms, total %f ms\n",
            t1-thread_t, t2-t1, t2-thread_t);
      else
        printf("\n\t\t estimator predict %f ms, correct %f ms, total %f ms\n",
            t1-t0, t2-t1, t2-t0);

      if (est) {
        printf("est  state:\n");
        //print_state(m, est_data);
      }

      if (darwin_model) {
          printf("\n\nSensor Compare:\nreal: ");
          int s = mrkr_idx;
          for (int i=s; i<s+MARKER_SIZE; i++) {
              if (real_robot) printf("%1.4f ", sensors[i]);
              else printf("%1.4f ", d->sensordata[i]);
          }
          if (est) {
              printf("\n est: ");
              for (int i=s; i<s+MARKER_SIZE; i++) {
                  printf("%1.4f ", est_data->sensordata[i]);
              }
          }
      }

      //printf("\n");
      //printf("\nctrl: ");
      //for (int i=0; i<nu; i++) {
      //    printf("%1.4f ", ctrl[i]);
      //}
      //printf("\n\n");

      if (est && est_data) {
        if (from_hardware) save_states(output_file, time, NULL, est_data, est->get_stddev(), t1-thread_t, t2-thread_t, "a");
        else save_states(output_file, time, d, est_data, est->get_stddev(), t1-t0, t2-t1, "a");
      }

      // we have estimated and logged the data,
      // now get new controls
      if (nu >= 20) {
        walker->Process(time-prev_time, 0, ctrl);
      }
      robot->set_controls(ctrl, 20, NULL, NULL);
      //calc_torques(m, d, ctrl);
      //util::darwin_torques(trqs, m, d, ctrl, min_t, kp);
      //robot->set_controls(trqs, 20, NULL, NULL);


      prev_time = time;
      if (est && estimation_counts > 0) {
        estimation_counts--;
      }
      else if (estimation_counts == 0) {
        printf("Done with limited runtime.\n");
        if (render_robot) shouldCloseViewer();
        else exit = true;
      }
    }
    else {
      // allow time to progress
    }

    if (render_robot) {
      //printf("Render time %f\n", util::now_t() - render_t);
      //render_t=util::now_t();
      if (est) {
        // render sigma points
        // TODO rendering is slow on macs
        if (from_hardware) { render_inplace = true; }
        else { render_inplace = false; }
        //render_inplace = true;

        render(window, est->get_sigmas(), render_inplace); // get state updated model / data, mj_steps
      }
      else {
        std::vector<mjData*> a;
        render(window, a, false);
      }

      finalize();
      exit = !closeViewer();
    }

    //double t_1 = util::now_t();
    //printf("\n\t\t Loop time: %f ms\n", t_1-t_2);
  }

  if (render_robot) {
    end_viz();
  }
  else {
    mj_deactivate();
    mj_deleteData(d);
    mj_deleteModel(m);
  }

  if (est && est_data) {
    if (real_robot) save_states(output_file, time, NULL, est_data, est->get_stddev(), 0, 0, "c");
    else save_states(output_file, time, d, est_data, est->get_stddev(), 0, 0, "c");
    // close file
  }

  //printf("\n");
  delete[] qpos;
  delete[] qvel;
  delete[] ctrl;
  delete[] sensors;
  delete[] conf;
  delete est;
  delete robot;

  return 0;
}
