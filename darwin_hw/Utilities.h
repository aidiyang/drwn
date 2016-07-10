
#pragma once

#include <time.h>
#include <chrono>

#include <iostream>
#include <fstream>

#include <termios.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/time.h>

#include "robot.h"



double GetCurrentTimeMS() {
  std::chrono::time_point<std::chrono::high_resolution_clock> t
    = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> d=t.time_since_epoch();
  return d.count(); // returns milliseconds
}

// joint positions for converting to and from
double joint2radian(int joint_value) {
  //return (joint_value * 0.088) * 3.14159265 / 180.0;
  return (joint_value-2048.0) * 0.00153398078;
}

int radian2joint(double radian) {
  return (int)(radian * 651.898650256) + 2048;
}

// joint speeds for converting to and from
double j_rpm2rads_ps(int rpm) {
  //return rpm * 0.11 * 2 * 3.14159265 / 60;

  // bitwise
  int neg = !!(rpm & ~0x3ff); // bool for negative
  rpm = rpm & 0x3ff;			 // get speed magnitude
  rpm = (!neg*rpm)-(neg*rpm); //use bool as switch

  /* logical
     if (rpm > 1023) {
  // negative
  rpm = 1024 - rpm;
  }
  */
  return rpm * 0.01151917306;
}

int rad_ps2rpm(double rad_ps) {
  //return rpm * 0.11 * 2 * 3.14159265 / 60;
  return (int)(rad_ps * 86.8117871649);
}

int rad_ps2jvel(double rps) {
  int jvel = (int) (rps / 0.01151917306);
  if (jvel < 0) jvel = 1023 - jvel;
  return jvel;
}

// gyro & accel
double gyro2rads_ps(int gyro) {
  return (gyro-512)*0.017453229251;
}

double accel2ms2(int accel) {
  //return (accel-512) / 128.0; in G's
  return ((accel-512) / 128.0) * 9.81; // in m/s^2
}

// fsr
double fsr2newton(int fsr) {
  return fsr / 1000.0;
}

void save_states(std::ofstream &myfile, std::string filename,
    int nu, int nsensordata, int nc,
    double time, double * ctrl, double * sensors, double* conf,
    std::string mode = "w")
{
  if (mode=="w") {
    // create file
    myfile.open(filename, std::ofstream::out);
    myfile<<"time,";
    for (int i=0; i<nu; i++) 
      myfile<<"ctrl,";
    for (int i=0; i<nsensordata; i++) 
      myfile<<"snsr,";
    if (conf) {
      for (int i=0; i<nc; i++) 
        myfile<<"conf,";
    }

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
    if (conf) {
      for (int i=0; i<nc; i++) 
        myfile<<conf[i]<<",";
    }

    myfile<<"\n";
  }
}

void zero_position(MyRobot *d, double* ctrl, double* sensors, int nu) {
  printf("Moving to initial position");
  double time;
  d->get_sensors(&time, sensors, NULL);
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

    d->get_sensors(&time, sensors, NULL);
    // wait for next cmd to interpolate
    //std::chrono::milliseconds interval(3000/max_t);
    //std::this_thread::sleep_for(interval);
  }
  d->set_controls(init, NULL, NULL); // force final position
  printf(" done.\n");
}

void buffer_markers(MyRobot *d, double* ps, double* ps_c,
    double* sensors, double* conf, int mrkr_idx, int NMARKER, int b_count) {
  // ASSUMES ps is NMARKER*3 in size
  // ASSUMES ps_c is NMARKER in size
  // ASSUMES other buffers are adequate
  for (int m=0; m<NMARKER; m++) { // clear buffer
    ps[m*3+0] = 0.0; ps[m*3+1] = 0.0; ps[m*3+2] = 0.0;
    ps_c[m] = 0;
  }
  double time;
  for (int i = 0; i < b_count; i++) {
    d->get_sensors(&time, sensors, conf);
    for (int m=0; m<NMARKER; m++) {
      if (conf[m] > 3.9) {
        ps[m*3+0] += sensors[mrkr_idx+ m*3+0];
        ps[m*3+1] += sensors[mrkr_idx+ m*3+1];
        ps[m*3+2] += sensors[mrkr_idx+ m*3+2];
        ps_c[m] += 1.0;
      }
    }
    printf(".");
    fflush(stdout);
  }
  for (int m=0; m<NMARKER; m++) { // average the data points
    if (ps_c[m] > 1e-6) {
      ps[m*3+0] = ps[m*3+0] / ps_c[m];
      ps[m*3+1] = ps[m*3+1] / ps_c[m];
      ps[m*3+2] = ps[m*3+2] / ps_c[m];
    }
  }
  for (int m=0; m<NMARKER; m++) {
    //printf("%d %1.3f %1.3f %1.3f %1.3f\n", m, ps[m*3+0], ps[m*3+1], ps[m*3+2], ps_c[m]);
    ps_c[m] = ps_c[m] / (double) b_count; // threshold these above 0.5
  }
}

void kb_changemode(int dir)
{
#ifndef __WIN32__
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
#endif
}

#ifndef __WIN32__
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
#endif


/*
   std::map<int, int> MJ_JOINT_MAP = {
   {19, 6},
   {20, 7},
   {2, 8},
   {4, 9},
   {6, 10},
   {1, 11},
   {3, 12},
   {5, 13},
   {8, 14},
   {10, 15}, 
   {12, 16},
   {14, 17},
   {16, 18},
   {18, 19},
   {7, 20},
   {9, 21}, 
   {11, 22}, 
   {13, 23},
   {15, 24},
   {17, 25} };

   std::map<int, int> MJ_CTRL_MAP = {
   {19, 6},
   {20, 7},
   {2, 8},
   {4, 9},
   {6, 10},
   {1, 11},
   {3, 12},
   {5, 13},
   {8, 14},
   {10, 15}, 
   {12, 16},
   {14, 17},
   {16, 18},
   {18, 19},
   {7, 20},
   {9, 21}, 
   {11, 22}, 
   {13, 23},
   {15, 24},
   {17, 25} };
   */
