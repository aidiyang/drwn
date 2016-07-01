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

#include <termios.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/time.h>


namespace po = boost::program_options;

std::ofstream myfile;
int nu = 20;

void save_states(std::string filename,
    int nu, int nsensordata, 
    double time, double * ctrl, double * sensors, 
    std::string mode = "w") {
  if (mode=="w") {
    // create file
    myfile.open(filename, std::ofstream::out);
    myfile<<"time,";
    for (int i=0; i<nu; i++) 
      myfile<<"ctrl,";
    for (int i=0; i<nsensordata; i++) 
      myfile<<"snsr,";

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
    for (int i=0; i<nu; i++) 
      myfile<<ctrl[i]<<",";
    for (int i=0; i<nsensordata; i++) 
      myfile<<sensors[i]<<",";

    myfile<<"\n";
  }
}

void zero_position(DarwinRobot *d, double* ctrl, double* sensors) {
  printf("Moving to initial position");
  double time;
  d->get_sensors(&time, sensors);
  int max_t = 50;
  double init[nu];
  for (int i = 0; i < nu; i++) {
    init[i] = ctrl[i]; // pass in initial goal position with ctrl
    ctrl[i] = sensors[i];
  }
  for (int t=0; t<max_t; t++) { // 25 commands over 5 seconds?
    for (int i = 0; i < nu; i++) {
      double diff = init[i] - sensors[i]; // end - start
      ctrl[i] += diff / (double)max_t;
    }
    d->set_controls(ctrl, NULL, NULL);
    printf(".");
    fflush(stdout);
    // wait for next cmd to interpolate
    std::chrono::milliseconds interval(3000/max_t);
    std::this_thread::sleep_for(interval);
  }
  for (int i = 0; i < nu; i++) {
    ctrl[i] = 0.0;
  }
  d->set_controls(ctrl, NULL, NULL);
  printf(" done.\n");
}

void changemode(int dir)
{
  static struct termios oldt, newt;

  if ( dir == 1 )
  {
    tcgetattr( STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~( ICANON | ECHO );
    tcsetattr( STDIN_FILENO, TCSANOW, &newt);
  }
  else
    tcsetattr( STDIN_FILENO, TCSANOW, &oldt);
}

int kbhit (void)
{
  struct timeval tv;
  fd_set rdfs;

  tv.tv_sec = 0;
  tv.tv_usec = 0;

  FD_ZERO(&rdfs);
  FD_SET (STDIN_FILENO, &rdfs);

  select(STDIN_FILENO+1, &rdfs, NULL, NULL, &tv);
  return FD_ISSET(STDIN_FILENO, &rdfs);

}



int main (int argc, char* argv[]) {

  bool joints;
  bool zero_gyro;
  bool use_rigid;
  bool use_markers;
  bool use_accel; //true;
  bool use_gyro; //true;
  bool use_ati;
  bool log;
  std::string ps_server;
  std::string output_file;

  try {
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help", "Usage guide")
      //("engage,e", po::value<bool>(&engage)->default_value(false), "Engage motors for live run.")
      //("model,m", po::value<std::string>(&model_name)->required(), "Model file to load.")
      ("output,o", po::value<std::string>(&output_file)->default_value("raw.csv"), "Where to save logged sensor data to csv.")
      //("timesteps,c", po::value<int>(&estimation_counts)->default_value(-1), "Number of times to allow estimator to run before quitting.")
      //("debug,n", po::value<bool>(&debug)->default_value(false), "Do correction step in estimator.")
      //("real,r", po::value<bool>(&real_robot)->default_value(false), "Use real robot.")
      ("cm730,j", po::value<bool>(&joints)->default_value(true), "Use CM730 interface.")
      ("rigid,r", po::value<bool>(&use_rigid)->default_value(false), "Use Phasespace rigid body tracking.")
      ("markers,m", po::value<bool>(&use_markers)->default_value(true), "Use Phasespace Markers.")
      ("accel,a", po::value<bool>(&use_accel)->default_value(true), "Use accelerometer.")
      ("zero_gyro,g", po::value<bool>(&zero_gyro)->default_value(true), "Use gyroscope.")
      ("contact,c", po::value<bool>(&use_ati)->default_value(true), "Use Contact sensors.")

      ("log,l", po::value<bool>(&log)->default_value(false), "Log sensor data.")
      ("ps_server,p", po::value<std::string>(&ps_server)->default_value("128.208.4.49"), "Where to save logged sensor data to csv.")

      /*
         ("s_noise,s", po::value<double>(&s_noise)->default_value(0.0), "Gaussian amount of sensor noise to corrupt data with.")
         ("c_noise,p", po::value<double>(&c_noise)->default_value(0.0), "Gaussian amount of control noise to corrupt data with.")
         ("e_noise,e", po::value<double>(&e_noise)->default_value(0.0), "Gaussian amount of estimator noise to corrupt data with.")
         ("alpha,a", po::value<double>(&alpha)->default_value(10e-3), "Alpha: UKF param")
         ("beta,b", po::value<double>(&beta)->default_value(2), "Beta: UKF param")
         ("kappa,k", po::value<double>(&kappa)->default_value(0), "Kappa: UKF param")
         ("diagonal,d", po::value<double>(&diag)->default_value(1), "Diagonal amount to add to UKF covariance matrix.")
         ("weight_s,w", po::value<double>(&Ws0)->default_value(-1.0), "Set inital Ws weight.")
         ("tol,i", po::value<double>(&tol)->default_value(-1.0), "Set Constraint Tolerance (default NONE).")
         */
      //("dt,t", po::value<double>(&dt)->default_value(0.02), "Timestep in binary file -- checks for file corruption.")
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

  double *p = NULL; // initial pose
  int* p_gain = new int[nu]; 
  for (int i=0; i<nu; i++) { // use sensors based on mujoco model
    p_gain[i] = 2;
  }
  DarwinRobot *d = new DarwinRobot(joints, zero_gyro, use_rigid, use_markers,
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
  int MARKER_SIZE = 16*4;
  int nsensordata = 40+
    use_accel*A_SIZE+
    use_gyro*G_SIZE+
    use_ati*CONTACTS_SIZE+
    use_markers*MARKER_SIZE;

  double *sensors = new double[nsensordata];

  ////////////////////////////////// move to initial position
  walker->Initialize(ctrl); // new goal state
  zero_position(d, ctrl, sensors);

  // make the log file to start
  if (log) save_states("raw.csv", nu, nsensordata, time, ctrl, sensors, "w");

  int count = 1000;
  double avg[count];

  double t1=0.0, t2=0.0;
  double prev_time = 0.0;

  printf("Press w to walk.\n");
  printf("Press q to quit.\n");
  changemode(1);
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
            printf("Stop walking.\n");
          }
          else {
            walker->Start();
            printf("Start walking.\n");
          }
          walk = !walk;
          break;
      }
    }
    t1 = GetCurrentTimeMS();
    // get this t's sensor data
    d->get_sensors(&time, sensors);
    t2 = GetCurrentTimeMS();
    //for (int m=0; m<MARKER_SIZE; m++) {
    //  ps[m] += sensors[40+m];
    //}
    // set this t's ctrl
    if (walk) {
      walker->Process(time-prev_time, 0, ctrl);
    }
    if (log) save_states("raw.csv", nu, nsensordata, time, ctrl, sensors, "a");

    printf("\r");
    printf("%f ms\t", t2-t1);
    for (int id=0; id<10; id++) {
      printf("%1.2f ", sensors[id]);
    }
    printf("\t::\t");
    for (int id=0; id<6; id++) {
      printf("%1.6f ", sensors[40+id]);
    }

    //printf("L: %1.4f %1.4f %1.4f : %1.4f\t\t",
    //    sensors[40+M_L*4+0], sensors[40+M_L*4+1], sensors[40+M_L*4+2], sensors[40+M_L*4+3]);
    //printf("R: %1.4f %1.4f %1.4f : %1.4f",
    //    sensors[40+M_R*4+0], sensors[40+M_R*4+1], sensors[40+M_R*4+2], sensors[40+M_R*4+3]);
    avg[idx++] = (t2-t1);
    if (idx>=count) { idx = 0; }

    prev_time = time;
  }
  changemode(0);
  printf("\n\n");

  /*
     for (int i = 0; i < count; i++) {

     t1 = GetCurrentTimeMS();
     d->get_sensors(&time, sensors);
     for (int id=0; id<20; id++) {
     ctrl[id] = sensors[id];
     }
     d->set_controls(ctrl, NULL, NULL);
     t2 = GetCurrentTimeMS();

     printf("%f ms\t", t2-t1);
     for (int id=0; id<10; id++) {
     printf("%1.2f ", sensors[id]);
     }
     printf("\t::\t");
     for (int id=0; id<6; id++) {
     printf("%1.6f ", sensors[40+id]);
     }
     printf("\n");

  // Do stuff with data
  if (log) save_states("raw.csv", nu, nsensordata, time, ctrl, sensors, "a");

  avg[i] = (t2-t1);
  }
  */

  if (log) save_states("raw.csv", nu, nsensordata, 0.0, NULL, NULL, "c");

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

  delete[] ctrl;
  delete[] sensors;

  delete d;
  delete walker;


  return 0;
}

