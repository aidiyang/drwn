#include "interface.h"
#include "Utilities.h"
#ifdef _WIN32
#include "WindowsDARwIn.h"
#else
#include "LinuxCM730.h"
#endif


#include <iostream>
#include <fstream>

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

int main (int argc, char* argv[]) {

  int nu = 20;
  bool zero_gyro = true;
  bool use_rigid = false;
  bool use_markers = false;
  bool use_accel = true;
  bool use_gyro = true;
  bool use_ati = true;
  std::string ps_server = "128.208.4.128";

  double *p = NULL; // initial pose
  int* p_gain = new int[nu]; 
  for (int i=0; i<nu; i++) { // use sensors based on mujoco model
    p_gain[i] = 2;
  }
  DarwinRobot *d = new DarwinRobot(zero_gyro, use_rigid, use_markers,
      use_accel, use_gyro, use_ati, p_gain, ps_server, p);

  if (!d->is_running()) {
    printf("\tCouldn't initialized Darwin, or some subset of its sensors!!\n");
    return 0;
  }

  // TODO get array sizes from mujoco?
  double *qpos = new double[27];
  double *qvel = new double[26];
  double *ctrl = new double[nu];
  double time = 0.0; 
  int IMU_SIZE=6;
  int CONTACTS_SIZE=12;
  int nsensordata = 40+IMU_SIZE+CONTACTS_SIZE;
  double *sensors = new double[nsensordata];

  save_states("raw.csv", nu, nsensordata, time, ctrl, sensors, "w");

  int count = 1000;
  double t1=0.0, t2=0.0;
  printf("\n");
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
    save_states("raw.csv", nu, nsensordata, time, ctrl, sensors, "a");

  }


  delete[] qpos;
  delete[] qvel;
  delete[] ctrl;
  delete[] sensors;
  
  save_states("raw.csv", nu, nsensordata, 0.0, NULL, NULL, "c");

  return 0;
}

