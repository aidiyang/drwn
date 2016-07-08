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

using namespace Eigen;

/*
double norm(double* a) {
  return dot(a, a);
}

void normalize(double *a) {
  double n = norm(a);
  a[0] = a[0] / n;
  a[1] = a[1] / n;
  a[2] = a[2] / n;
}

double dot(double *a, double* b) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

void cross(double *a, double* b, double *c) {
  c[0] = a[1]*b[2] - a[2]*b[1];
  c[1] = a[2]*b[0] - a[0]*b[2];
  c[2] = a[0]*b[1] - a[1]*b[0];
}
*/

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


Matrix3d rot_matrix(Vector3d a, Vector3d b) {
  double dot = a.dot(b);
  double cross = (a.cross(b)).norm();

  Matrix3d rot;
  rot << dot, -1*cross, 0, cross, dot, 0, 0, 0, 1;
  return rot;
}


void orig_ps_order(double * ps, double* sensor, double* conf) {
  for (int i=0; i<16; i++) {
    ps[i*3 + 0] = sensor[i*3 + 0];
    ps[i*3 + 1] = sensor[i*3 + 1];
    ps[i*3 + 2] = sensor[i*3 + 2];
    ps[i*3 + 3] = conf[i];
  }
}

int main (int argc, char* argv[]) {

  bool joints = true;
  bool zero_gyro = false;
  bool use_rigid = false;
  bool use_markers = true;
  bool raw_markers = false;
  bool use_accel = false; //true;
  bool use_gyro  = false; //true;
  bool use_ati   = false;
  std::string ps_server = "128.208.4.49";

  double *p = NULL; // initial pose
  int nu = 20;
  int* p_gain = new int[nu]; 
  for (int i=0; i<nu; i++) { // use sensors based on mujoco model
    p_gain[i] = 32;
  }
  DarwinRobot *d = new DarwinRobot(joints, zero_gyro, use_rigid, use_markers, raw_markers,
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
  double time = 0.0; 
  int A_SIZE=3;
  int G_SIZE=3;
  int CONTACTS_SIZE=12;
  int MARKER_SIZE = 16*3;
  int nsensordata = 40+
    use_accel*A_SIZE+
    use_gyro*G_SIZE+
    use_ati*CONTACTS_SIZE+
    use_markers*MARKER_SIZE;
  int mrkr_idx = 40+
    use_accel*A_SIZE+
    use_gyro*G_SIZE+
    use_ati*CONTACTS_SIZE;
  printf("Sensor Size: %d\n", nsensordata);
  double *sensors = new double[nsensordata];

  ////////////////////////////////// move to initial position
  zero_position(d, ctrl, sensors, nu);

  ////////////////////////////////// get first position of data
  int M_L = 9; // the two markers that should be parallel to the plane
  int M_R = 8;
  double *ps = new double[16*4]; // TODO hack to get around the ps marker + conf order
  double *conf = new double[16];

  printf("Set robot at initial position.\n");
  printf("\tPress any key to continue.\n");
  kb_changemode(1);
  while (!kbhit()) {
    d->get_sensors(&time, sensors, conf);

    printf("\r");
    printf("L: %1.4f %1.4f %1.4f : %1.4f\t\t",
        sensors[40+M_L*3+0], sensors[40+M_L*3+1], sensors[40+M_L*3+2], conf[M_L]);
    printf("R: %1.4f %1.4f %1.4f : %1.4f",
        sensors[40+M_R*3+0], sensors[40+M_R*3+1], sensors[40+M_R*3+2], conf[M_R]);
  }
  int c = getchar(); // pop off queue

  int count = 200;
  for (int m=0; m<(16*4); m++) { // clear buffer
    ps[m] = 0.0;
  }
  for (int i = 0; i < count; i++) {
    d->get_sensors(&time, sensors, conf);
    //orig_ps_order(ps, sensors, conf);
    for (int m=0; m<16; m++) {
      ps[m*4+0] += sensors[40+m*3+0];
      ps[m*4+1] += sensors[40+m*3+1];
      ps[m*4+2] += sensors[40+m*3+2];
      ps[m*4+3] += conf[m];
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
    d->get_sensors(&time, sensors, conf);

    printf("\r");
    printf("L: %1.4f %1.4f %1.4f : %1.4f\t\t",
        sensors[40+M_L*3+0], sensors[40+M_L*3+1], sensors[40+M_L*3+2], conf[M_L]);
    printf("R: %1.4f %1.4f %1.4f : %1.4f",
        sensors[40+M_R*3+0], sensors[40+M_R*3+1], sensors[40+M_R*3+2], conf[M_R]);
  }
  c = getchar(); // pop off queue

  for (int m=0; m<MARKER_SIZE; m++) { // clear buffer
    ps[m] = 0.0;
  }
  for (int i = 0; i < count; i++) {
    d->get_sensors(&time, sensors, conf);
    for (int m=0; m<16; m++) {
      ps[m*4+0] += sensors[40+m*3+0];
      ps[m*4+1] += sensors[40+m*3+1];
      ps[m*4+2] += sensors[40+m*3+2];
      ps[m*4+3] += conf[m];
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

  Vector3d m_normal= va1.cross(va2);
  Vector3d cross2 = vb1.cross(vb2);
  Vector3d s_normal(0, 0, 1);
  m_normal.normalize();
  cross2.normalize();
  s_normal.normalize();

  Matrix3d rot = rot_matrix(m_normal, s_normal);

  std::cout<<"\nRotation:\n"<< rot << std::endl;
  std::cout<<"\n2nd Rotation:\n"<< rot_matrix(cross2, s_normal) << std::endl;

  std::cout<<"First position Rotated:\n"
    << (rot*l1).transpose() << "\n\n"
    << (rot*r1).transpose() << "\n\n";

  std::cout<<"Second position Rotated:\n"
    << (rot*l2).transpose() << "\n\n"
    << (rot*r2).transpose() << "\n\n";

  std::ofstream myfile;
  std::string filename = "rot.csv";
  IOFormat CommaInitFmt(FullPrecision, DontAlignCols, ", ", ", ", "", "", "", "");
  myfile.open(filename, std::ofstream::out);
  //myfile<<"rotation\n";
  myfile<<rot.format(CommaInitFmt)<<"\n";
  //myfile<<"first\n";
  //myfile<<l1<<"\n";
  //myfile<<r1<<"\n";
  //myfile<<"second\n";
  //myfile<<l2<<"\n";
  //myfile<<r2<<"\n";
  myfile.close();

  d->set_frame_rotation(rot);

  printf("Press enter to accept chest markers...\n");
  int c1 = 7; // marker positions
  int c2 = 3;

  while (!kbhit()) {
    d->get_sensors(&time, sensors, conf);

    printf("\r");
    printf("M1: %1.4f %1.4f %1.4f : %1.4f\t\t",
        sensors[40+c1*3+0], sensors[40+c1*3+1], sensors[40+c1*3+2], conf[c1]);
    printf("M2: %1.4f %1.4f %1.4f : %1.4f",
        sensors[40+c2*3+0], sensors[40+c2*3+1], sensors[40+c2*3+2], conf[c2]);
  }
  c=getchar();
  kb_changemode(0);

  // getting initial position for offset and rotation

  d->get_sensors(&time, sensors, conf);
  orig_ps_order(ps, sensors, conf);

  for (int m=0; m<MARKER_SIZE; m++) {
    initial[m] << ps[m*4+0], ps[m*4+1], ps[m*4+2];
    initial[m] = rot*initial[m];
  }

  Vector3d v1(ps[c1*4+0], ps[c1*4+1], ps[c1*4+2]);
  Vector3d v2(ps[c2*4+0], ps[c2*4+1], ps[c2*4+2]);
  Vector3d vec_r = rot*v1 - rot*v2;
  vec_r[2] = 0;
  Vector3d vec_s(0, -1, 0); // basically only in y axis

  double* mrkr = sensors+mrkr_idx;
  d->set_initpos_rt(vec_r, vec_s, mrkr, s_ps_zero);

  while (1) {
    d->get_sensors(&time, sensors, conf);
    orig_ps_order(ps, sensors, conf);

    // should be rotated in interface.h
    //for (int m=0; m<16; m++) {
    //  Map<Vector3d> r(ps + m*4);
    //  r = rot*r;
    //  //rotated[m] << ps[m*4+0], ps[m*4+1], ps[m*4+2];
    //  //rotated[m] = rot*rotated[m]; // - initial[m]; plus the mujoco offsets?
    //}
    printf("Rotated:\nx: ");
    for (int m=0; m<16; m++) { printf("%1.3f  ", ps[m*4+0]); }
    printf("\ny: ");
    for (int m=0; m<16; m++) { printf("%1.3f  ", ps[m*4+1]); }
    printf("\nz: ");
    for (int m=0; m<16; m++) { printf("%1.3f  ", ps[m*4+2]); }
    printf("\nconf: ");
    for (int m=0; m<16; m++) { printf("%1.3f  ", ps[m*4+3]); }
    printf("\n");
  }

  delete[] qpos;
  delete[] qvel;
  delete[] ctrl;
  delete[] sensors;
  delete[] ps;


  return 0;
}


