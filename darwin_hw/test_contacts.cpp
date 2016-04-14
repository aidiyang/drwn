#include <iostream>
#include "s_contacts.h"
#include "Utilities.h"
#ifdef _WIN32
#include "WindowsDARwIn.h"
#else
//#include "LinuxDARwIn.h"
#include "LinuxCM730.h"
#endif


int main (int argc, char* argv[]) {

  ContactSensors *c = new ContactSensors();

  if (!c->is_running()) {
    printf("\tCouldn't initialized contact sensors!!\n");
    return 0;
  }

  double *r = new double[6];
  double *l = new double[6];

  int count = 100000;
  double t1=0.0, t2=0.0;
  printf("\n");
  double total=0.0;
  for (int i = 0; i < count; i++) {
    
    t1 = GetCurrentTimeMS();
    bool ret = c->getData(r, l);
    t2 = GetCurrentTimeMS();

    if (!ret) break;

    total+=t2-t1;
    printf("%f ms\t", t2-t1);
    for (int id=0; id<6; id++) {
      printf("%1.2f ", r[id]);
    }
    printf("\t::\t");
    for (int id=0; id<6; id++) {
      printf("%1.2f ", l[id]);
    }
    printf("\n");

    // Do stuff with data
  }
  printf("Total time for %d datas: %f ms\t", count, total);

  delete c;
  delete[] l;
  delete[] r;

  return 0;
}


