#include "interface.h"
#include "Utilities.h"
#ifdef _WIN32
#include "WindowsDARwIn.h"
#else
#include "LinuxCM730.h"
#endif

#include <stdio.h>

#include <iostream>
#include <fstream>

#include <math.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <termios.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/time.h>

using namespace Eigen;

std::ofstream myfile;
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

  int nu = 20;
  bool joints = true;
  bool zero_gyro = false;
  bool use_rigid = false;
  bool use_markers = true;
  bool use_accel = false; //true;
  bool use_gyro  = false; //true;
  bool use_ati   = false;
  std::string ps_server = "128.208.4.49";

  double *p = NULL; // initial pose
  int* p_gain = new int[nu]; 
  for (int i=0; i<nu; i++) { // use sensors based on mujoco model
    p_gain[i] = 32;
  }
  DarwinRobot *d = new DarwinRobot(joints, zero_gyro, use_rigid, use_markers,
      use_accel, use_gyro, use_ati, p_gain, ps_server, p);

  if (!d->is_running()) {
    printf("\tCouldn't initialized Darwin, or some subset of its sensors!!\n");
    return 0;
  }

  // TODO get array sizes from mujoco?
  double *qpos = new double[27];
  double *qvel = new double[26];
  double *ctrl = new double[nu];
  Vector3d * initial = new Vector3d[16];
  Vector3d * rotated = new Vector3d[16];
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
  printf("Sensor Size: %d\n", nsensordata);
  double *sensors = new double[nsensordata];

  //save_states("raw.csv", nu, nsensordata, time, ctrl, sensors, "w");

  ////////////////////////////////// move to initial position
  printf("Moving to initial position");
  d->get_sensors(&time, sensors);
  int max_t = 50;
  for (int i = 0; i < nu; i++) {
    ctrl[i] = sensors[i];
  }
  for (int t=0; t<max_t; t++) { // 25 commands over 5 seconds?
    for (int i = 0; i < nu; i++) {
      double diff = 0.0 - sensors[i]; // end - start
      ctrl[i] += diff / (double)max_t;
    }
    d->set_controls(ctrl, NULL, NULL);
    printf(".");
    fflush(stdout);
    // wait for next cmd to interpolate
    std::chrono::milliseconds interval(3000/max_t);
    std::this_thread::sleep_for(interval);
  }
  printf(" done.\n");

  ////////////////////////////////// get first position of data
  int M_L = 9; // the two markers that should be parallel to the plane
  int M_R = 8;
  double *ps = new double[MARKER_SIZE];

  printf("Set robot at initial position.\n");
  printf("\tPress any key to continue.\n");
  changemode(1);
  while (!kbhit()) {
    d->get_sensors(&time, sensors);
    for (int m=0; m<MARKER_SIZE; m++) {
      ps[m] += sensors[40+m];
    }
    printf("\r");
    printf("L: %1.4f %1.4f %1.4f : %1.4f\t\t",
        sensors[40+M_L*4+0], sensors[40+M_L*4+1], sensors[40+M_L*4+2], sensors[40+M_L*4+3]);
    printf("R: %1.4f %1.4f %1.4f : %1.4f",
        sensors[40+M_R*4+0], sensors[40+M_R*4+1], sensors[40+M_R*4+2], sensors[40+M_R*4+3]);
  }
  int c = getchar(); // pop off queue

  int count = 200;
  for (int m=0; m<MARKER_SIZE; m++) { // clear buffer
    ps[m] = 0.0;
  }
  for (int i = 0; i < count; i++) {
    d->get_sensors(&time, sensors);
    for (int m=0; m<MARKER_SIZE; m++) {
      ps[m] += sensors[40+m];
    }
    printf(".");
    fflush(stdout);
  }
  printf("done.\n");
  for (int m=0; m<MARKER_SIZE; m++) { // average the data points 
    ps[m] = ps[m] / (double)count;
  }

  Vector3d l1(ps[M_L*4+0], ps[M_L*4+1], ps[M_L*4+2]);
  Vector3d r1(ps[M_R*4+0], ps[M_R*4+1], ps[M_R*4+2]);

  std::cout<<"2nd Left  Foot: "<< l1 << "\n";
  std::cout<<"2nd Right Foot: "<< r1 << "\n";

  ////////////////////////////////// get second position of data
  printf("Set robot to the second position.\n");
  printf("\tPress any key to continue.\n");
  while (!kbhit()) {
    d->get_sensors(&time, sensors);
    for (int m=0; m<MARKER_SIZE; m++) {
      ps[m] += sensors[40+m];
    }
    printf("\r");
    printf("L: %1.4f %1.4f %1.4f : %1.4f\t\t",
        sensors[40+M_L*4+0], sensors[40+M_L*4+1], sensors[40+M_L*4+2], sensors[40+M_L*4+3]);
    printf("R: %1.4f %1.4f %1.4f : %1.4f",
        sensors[40+M_R*4+0], sensors[40+M_R*4+1], sensors[40+M_R*4+2], sensors[40+M_R*4+3]);
  }
  c = getchar(); // pop off queue
  changemode(0);

  for (int m=0; m<MARKER_SIZE; m++) { // clear buffer
    ps[m] = 0.0;
  }
  for (int i = 0; i < count; i++) {
    d->get_sensors(&time, sensors);
    for (int m=0; m<MARKER_SIZE; m++) {
      ps[m] += sensors[40+m];
    }
    printf(".");
    fflush(stdout);
  }
  for (int m=0; m<MARKER_SIZE; m++) { // average the data points
    ps[m] = ps[m] / (double)count;
  }

  Vector3d l2(ps[M_L*4+0], ps[M_L*4+1], ps[M_L*4+2]);
  Vector3d r2(ps[M_R*4+0], ps[M_R*4+1], ps[M_R*4+2]);

  std::cout<<"2nd Left  Foot: "<< l2 << "\n";
  std::cout<<"2nd Right Foot: "<< r2 << "\n";

  Vector3d va1 = r1 - l1;
  Vector3d va2 = l2 - l1;

  Vector3d vb1 = l1 - r1; // for verification
  Vector3d vb2 = r2 - r1;

  Vector3d cross1 = va1.cross(va2);
  Vector3d cross2 = vb1.cross(vb2);
  cross1.normalize();
  cross2.normalize();

  JacobiSVD<MatrixXd> svd(cross1, Eigen::ComputeFullU);

  MatrixXd U = svd.matrixU();
  MatrixXd rot(3, 3);
  rot.col(0) = U.col(1);
  rot.col(1) = U.col(2);
  rot.col(2) = U.col(0);

  std::cout<<"\nRotation:\n"<< rot << std::endl;

  std::cout<<"First position Rotated:\n"
    << (rot*l1).transpose() << "\n\n"
    << (rot*r1).transpose() << "\n\n";

  std::cout<<"Second position Rotated:\n"
    << (rot*l2).transpose() << "\n\n"
    << (rot*r2).transpose() << "\n\n";

  printf("Press enter to continue...\n");
  getchar();
  // getting initial position
  d->get_sensors(&time, sensors);
  for (int m=0; m<MARKER_SIZE; m++) {
    ps[m] = sensors[40+m];
  }

  for (int m=0; m<MARKER_SIZE; m++) {
    initial[m] << ps[m*4+0], ps[m*4+1], ps[m*4+2];
    initial[m] = rot*initial[m];
  }

  while (1) {
    d->get_sensors(&time, sensors);
    for (int m=0; m<MARKER_SIZE; m++) {
      ps[m] = sensors[40+m];
    }
    for (int m=0; m<MARKER_SIZE; m++) {
      rotated[m] << ps[m*4+0], ps[m*4+1], ps[m*4+2];
      rotated[m] = rot* rotated[m]; // - initial[m]; plus the mujoco offsets?
    }
    printf("Rotated:\nx: ");
    for (int m=0; m<16; m++) { printf("%1.3f  ", rotated[m](0)); }
    printf("\ny: ");
    for (int m=0; m<16; m++) { printf("%1.3f  ", rotated[m](1)); }
    printf("\nz: ");
    for (int m=0; m<16; m++) { printf("%1.3f  ", rotated[m](2)); }
    printf("\nconf: ");
    for (int m=0; m<16; m++) { printf("%1.3f  ", ps[m*4+3]); }
    printf("\n");
  }

  delete[] qpos;
  delete[] qvel;
  delete[] ctrl;
  delete[] sensors;
  delete[] ps;

  //save_states("raw.csv", nu, nsensordata, 0.0, NULL, NULL, "c");

  return 0;
}


