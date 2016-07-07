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
  int max_t = 150;
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

    d->get_sensors(&time, sensors);
    // wait for next cmd to interpolate
    //std::chrono::milliseconds interval(3000/max_t);
    //std::this_thread::sleep_for(interval);
  }
  d->set_controls(init, NULL, NULL); // force final position
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
  use_gyro = zero_gyro;

  double *p = NULL; // initial pose
  int* p_gain = new int[nu]; 
  for (int i=0; i<nu; i++) { // use sensors based on mujoco model
    p_gain[i] = 20;
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
  d->set_gyro_offsets();
  double gyro[2];

  int count = 1000;
  double avg[count];

  printf("Press w to walk.\n");
  printf("Press q to quit.\n");
  printf("Press enter to begin.\n");

  changemode(1);
  double init_time;
  while (!kbhit()) {
    d->get_sensors(&init_time, sensors);
  }
  getchar();

  // make the log file to start
  if (log) save_states(output_file, nu, nsensordata, time, ctrl, sensors, "w");

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
    d->get_sensors(&time, sensors);
    t2 = GetCurrentTimeMS();
    time = time - init_time;

    // set this t's ctrl
    //
    d->get_cm730_gyro(gyro);
    printf("\ncm730 %f %f\n", gyro[0], gyro[1]);
    gyro[0] = sensors[40+3+1]*57.2958;
    gyro[1] = sensors[40+3+2]*57.2958;

    //walker->Process(time-prev_time, gyro, ctrl);
    walker->Process(time-prev_time, NULL, ctrl);
    d->set_controls(ctrl, NULL, NULL);

    if (log) save_states(output_file, nu, nsensordata, time, ctrl, sensors, "a");

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
    if (use_accel) { printf("Accl: %1.4f %1.4f %1.4f\n", sensors[i+0], sensors[i+1], sensors[i+2]); i+=3; }
    if (use_gyro) { printf("Gyro: %1.4f %1.4f %1.4f\n", sensors[i+0], sensors[i+1], sensors[i+2]); i+=3; }
    if (use_ati) {
        printf("Frce: %1.4f %1.4f %1.4f\n", sensors[i+0], sensors[i+1], sensors[i+2]); i+=3;
        printf("Trqe: %1.4f %1.4f %1.4f\n", sensors[i+0], sensors[i+1], sensors[i+2]); i+=3;
        printf("Frce: %1.4f %1.4f %1.4f\n", sensors[i+0], sensors[i+1], sensors[i+2]); i+=3;
        printf("Trqe: %1.4f %1.4f %1.4f\n", sensors[i+0], sensors[i+1], sensors[i+2]); i+=3;
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


  if (log) save_states(output_file, nu, nsensordata, 0.0, NULL, NULL, "c");

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

  changemode(0);

  delete[] ctrl;
  delete[] sensors;

  delete walker;
  delete d;


  return 0;
}

