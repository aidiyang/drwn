#include "interface.h"
#include "Utilities.h"
#ifdef _WIN32
#include "WindowsDARwIn.h"
#else
#include "LinuxCM730.h"
#endif


int main (int argc, char* argv[]) {

  bool zero_gyro = true;
  bool use_rigid = true;
  bool use_markers = false;
  std::string ps_server = "128.208.4.128";

  double *p = NULL; // initial pose
  DarwinRobot *d = new DarwinRobot(zero_gyro, use_rigid, use_markers, ps_server, p);

  if (!d->is_running()) {
    printf("\tCouldn't initialized Darwin, or some subset of its sensors!!\n");
    return 0;
  }

  // TODO get array sizes from mujoco?
  double *qpos = new double[27];
  double *qvel = new double[26];
  double *ctrl = new double[20];
  int IMU_SIZE=6;
  int CONTACTS_SIZE=12;
  double *sensors = new double[IMU_SIZE+CONTACTS_SIZE];

  int count = 1000;
  double t1=0.0, t2=0.0;
  printf("\n");
  for (int i = 0; i < count; i++) {
    
    t1 = GetCurrentTimeMS();
    d->get_state(qpos, qvel, sensors);
    for (int id=0; id<20; id++) {
      ctrl[id] = qpos[id+7];
    }
    d->set_controls(ctrl, NULL, NULL);
    t2 = GetCurrentTimeMS();

    printf("%f ms\t", t2-t1);
    for (int id=0; id<10; id++) {
      printf("%1.2f ", qpos[id+7]);
    }
    printf("\t::\t");
    for (int id=0; id<6; id++) {
      printf("%1.6f ", sensors[id]);
    }
    printf("\n");

    // Do stuff with data

  }


  delete[] qpos;
  delete[] qvel;
  delete[] ctrl;
  delete[] sensors;

  return 0;
}

