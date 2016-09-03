#include "interface.h"
#include "Utilities.h"
#ifdef _WIN32
#include "WindowsDARwIn.h"
#else
#include "LinuxCM730.h"
#endif

#include "drwn_walker.h"
#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>

#include <math.h>


namespace po = boost::program_options;

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


double s_ps_zero[16*3] = 
  {-0.0440,  0.0560, 0.3918, 
   -0.0750, -0.0540, 0.3228,
    0.0040,  0.0450, 0.3873,
    0.0350,  0.0180, 0.3478,
   -0.0440, -0.0560, 0.3918,
   -0.0750,  0.0540, 0.3228,
    0.0040, -0.0450, 0.3873,
    0.0350, -0.0180, 0.3478,
   -0.0490, -0.0723, 0.0162,
   -0.0490,  0.0723, 0.0162,
    0.0390, -0.0723, 0.0117,
    0.0390,  0.0723, 0.0117,
    0.0710, -0.1324, 0.3141,
    0.0960, -0.1517, 0.3063,
    0.0710,  0.1326, 0.3151,
    0.0960,  0.1519, 0.3073};



int main (int argc, char* argv[]) {

  bool debug;
  bool joints;
  bool zero_gyro;
  bool use_rigid;
  bool use_markers;
  bool raw_markers = false;
  bool use_accel; //true;
  bool use_gyro; //true;
  bool use_ati;
  bool log;
  std::string ps_server;
  std::string output_file;
  std::ofstream myfile;

  try {
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help", "Usage guide")
      ("output,o", po::value<std::string>(&output_file)->default_value("raw.csv"), "Where to save logged sensor data to csv.")
      ("debug,d", po::value<bool>(&debug)->default_value(false), "Use CM730 interface.")
      ("cm730,j", po::value<bool>(&joints)->default_value(true), "Use CM730 interface.")
      ("rigid,r", po::value<bool>(&use_rigid)->default_value(false), "Use Phasespace rigid body tracking.")
      ("markers,m", po::value<bool>(&use_markers)->default_value(true), "Use Phasespace Markers.")
      ("accel,a", po::value<bool>(&use_accel)->default_value(true), "Use accelerometer.")
      ("zero_gyro,g", po::value<bool>(&zero_gyro)->default_value(true), "Use gyroscope.")
      ("contact,c", po::value<bool>(&use_ati)->default_value(true), "Use Contact sensors.")
      ("log,l", po::value<bool>(&log)->default_value(false), "Log sensor data.")
      ("ps_server,p", po::value<std::string>(&ps_server)->default_value("128.208.4.49"), "Phasespace Server IP.")
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
  use_gyro = zero_gyro;

  double *p = NULL; // initial pose
  int nu = 20;
  int* p_gain = new int[nu]; 
  for (int i=0; i<nu; i++) { // use sensors based on mujoco model
    p_gain[i] = 20;
  }
  DarwinRobot *d = new DarwinRobot(joints, zero_gyro, use_rigid, use_markers, raw_markers,
      use_accel, use_gyro, use_ati, p_gain, ps_server, p);

  delete[] p_gain;

  if (!d->is_running()) {
    printf("\tCouldn't initialized Darwin, or some subset of its sensors!!\n");
    return 0;
  }

  Walking * walker = new Walking();

  // TODO get array sizes from mujoco?
  double *ctrl = new double[nu];
  double time = 0.0; 
  int A_SIZE=3;
  int G_SIZE=3;
  int CONTACTS_SIZE=12;
  int NMARKERS=16;
  int MARKER_SIZE = NMARKERS*3;
  int nsensordata = 40+
    use_accel*A_SIZE+
    use_gyro*G_SIZE+
    use_ati*CONTACTS_SIZE+
    use_markers*MARKER_SIZE;
  int mrkr_idx = 40+
    use_accel*A_SIZE+
    use_gyro*G_SIZE+
    use_ati*CONTACTS_SIZE;

  double *sensors = new double[nsensordata];
  double *conf = new double[NMARKERS];
  double *ps = new double[MARKER_SIZE];
  double *ps_c = new double[NMARKERS];

  double gyro[2];
  int count = 1000;
  double avg[count];

  ////////////////////////////////// move to initial position
  walker->Initialize(ctrl); // new goal state
  zero_position(d, ctrl, sensors, nu);
  d->set_gyro_offsets();
  Matrix3d rot;

  //rot<<0.999303305496562, -0.03732162407565985, 0, 0.03732162407565985, 0.999303305496562, 0, 0, 0, 1;
  //rot<<0.9965100112758773,-0.08347333362787641, 0, 0.08347333362787641, 0.9965100112758773, 0, 0, 0, 1;
  rot<<0.9998328991168706, -0.01828042241179796, 0, 0.01828042241179796, 0.9998328991168706, 0, 0, 0, 1;

  d->set_frame_rotation(rot);
  int c1 = 7; // marker positions
  int c2 = 3;
  double* mrkr = sensors+mrkr_idx;
  kb_changemode(1);
  while (!kbhit()) { // double check the chest marker positioning
    d->get_sensors(&time, sensors, conf);
    printf("\r");
    printf("M1: %1.4f %1.4f %1.4f : %1.4f\t\t",
        mrkr[c1*3+0], mrkr[c1*3+1], mrkr[c1*3+2], conf[c1]);
    printf("M2: %1.4f %1.4f %1.4f : %1.4f",
        mrkr[c2*3+0], mrkr[c2*3+1], mrkr[c2*3+2], conf[c2]);
  }
  int c=getchar();
  // after chest markers are in view, average their values
  int m_count = 100;
  buffer_markers(d, ps, ps_c, sensors, conf, mrkr_idx, NMARKERS, m_count);
  // ps has been averaged of good markers
  
  Vector3d v1(ps[c1*3+0], ps[c1*3+1], ps[c1*3+2]); // these should be good from above
  Vector3d v2(ps[c2*3+0], ps[c2*3+1], ps[c2*3+2]);
  //Vector3d vec_r = rot*v1 - rot*v2;
  Vector3d vec_r = v1 - v2;
  vec_r[2] = 0;
  Vector3d vec_s(0, -1, 0); // basically only in y axis

  // set these at the same time?
  d->set_initpos_rt(vec_r, vec_s, ps, ps_c, s_ps_walk); // pass the clean data


  printf("Press w to walk.\n");
  printf("Press q to quit.\n");
  printf("Press enter to begin.\n");

  double init_time;
  while (!kbhit()) {
    d->get_sensors(&init_time, sensors, conf);
  }
  getchar();

  // make the log file to start
  if (log) save_states(myfile, output_file, nu, nsensordata, NMARKERS, time, ctrl, sensors, conf, "w");

  double t1=0.0, t2=0.0;
  double prev_time = 0.0;

  bool exit = false;
  bool walk = false;
  int idx=0;
  while (!exit) {
    if (kbhit()) {
      int c = getchar(); // pop off queue
      switch (c) {
        case 'q':
          exit=true;
          break;
        case 'w':
          if (walk) {
            walker->Stop();
            printf("\nStop walking.\n");
            walk = false;
          }
          else {
            walker->Start();
            printf("\nStart walking.\n");
            walk = true;
          }
          break;
      }
    }
    // get this t's sensor data
    t1 = GetCurrentTimeMS();
    d->get_sensors(&time, sensors, conf);
    t2 = GetCurrentTimeMS();
    time = time - init_time;

    // set this t's ctrl
    //
    d->get_cm730_gyro(gyro);
    printf("\ncm730 %f %f\n", gyro[0], gyro[1]);
    //gyro[0] = sensors[40+3+1]*57.2958;
    //gyro[1] = sensors[40+3+2]*57.2958;

    walker->Process(time-prev_time, gyro, ctrl);
    //walker->Process(time-prev_time, NULL, ctrl);
    d->set_controls(ctrl, nu, NULL, NULL);

    if (log) save_states(myfile, output_file, nu, nsensordata, NMARKERS, time, ctrl, sensors, conf, "a");

    printf("\r");
    printf("%1.3f : %1.3f ms\t", time, t2-t1);
    for (int id=0; id<10; id++) {
      printf("%1.2f ", sensors[id]);
    }
    //printf("\t::\t");
    //for (int id=0; id<6; id++) {
    //  printf("%1.6f ", sensors[40+id]);
    //}
    printf("\n");
    int i=40;
    if (debug) {
      if (use_accel) { printf("Accl: %1.4f %1.4f %1.4f\n", sensors[i+0], sensors[i+1], sensors[i+2]); i+=3; }
      if (use_gyro) { printf("Gyro: %1.4f %1.4f %1.4f\n", sensors[i+0], sensors[i+1], sensors[i+2]); i+=3; }
      if (use_ati) {
        printf("Frce: %1.4f %1.4f %1.4f\n", sensors[i+0], sensors[i+1], sensors[i+2]); i+=3;
        printf("Trqe: %1.4f %1.4f %1.4f\n", sensors[i+0], sensors[i+1], sensors[i+2]); i+=3;
        printf("Frce: %1.4f %1.4f %1.4f\n", sensors[i+0], sensors[i+1], sensors[i+2]); i+=3;
        printf("Trqe: %1.4f %1.4f %1.4f\n", sensors[i+0], sensors[i+1], sensors[i+2]); i+=3;
      }
    }
    //if (use_markers) {
    //    printf("Trqe: %1.4f %1.4f %1.4f\n", sensors[i++], sensors[i++], sensors[i++]);
    //}

    //printf("L: %1.4f %1.4f %1.4f : %1.4f\t\t",
    //    sensors[40+M_L*4+0], sensors[40+M_L*4+1], sensors[40+M_L*4+2], sensors[40+M_L*4+3]);
    //printf("R: %1.4f %1.4f %1.4f : %1.4f",
    //    sensors[40+M_R*4+0], sensors[40+M_R*4+1], sensors[40+M_R*4+2], sensors[40+M_R*4+3]);
    avg[idx++] = (t2-t1);
    if (idx>=count) { idx = 0; }

    prev_time = time;
  }
  printf("\n\n");


  if (log) save_states(myfile, output_file, nu, nsensordata, NMARKERS, 0.0, NULL, NULL, NULL, "c");

  double mean=0.0;
  double stddev=0.0;
  for (int i = 0; i < count; i++) {
    mean += avg[i];
  }
  mean = mean / (double) count;
  for (int i = 0; i < count; i++) {
    double diff = avg[i] - mean;
    stddev += diff * diff;
  }
  stddev = sqrt(stddev / count);
  printf("Average Timing: %f; Standard Deviation: %f\n", mean, stddev);

  kb_changemode(0);

  delete[] ctrl;
  delete[] sensors;

  delete walker;
  delete d;


  return 0;
}

