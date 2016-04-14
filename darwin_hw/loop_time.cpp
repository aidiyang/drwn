#include <iostream>
#include "CM730.h"
#include "interface.h"
//#include "s_imu.h"
//#include "s_phasespace.h"
#ifdef _WIN32
#include "WindowsCM730.h"
#else
#include "LinuxCM730.h"
#endif


using namespace Robot;


int main (int argc, char* argv[]) {

   if (argc < 2) {
      std::cout << "Usage: ./policy 0/1/2/3/4/5 for just read, just write, read & write, imu, phasepace, everything" << std::endl;
      return -1;
   }

   // load params from file
   int option = atoi(argv[1]);

#ifdef _WIN32
   std::string BOARD_NAME="\\\\.\\COM3";
   WindowsCM730 my_cm730(BOARD_NAME.c_str());
#else
   std::string BOARD_NAME="/dev/ttyUSB0";
   LinuxCM730 my_cm730(BOARD_NAME.c_str());
#endif

   CM730 cm730(&my_cm730); // our packet
   //CM730 cm730(&linux_cm730, false);
   //if (MotionManager::GetInstance()->Initialize(&cm730) == false) {
   //   printf("Didn't initialized CM730\n");
   //   return -1;
   //}

   cm730.WriteWord(CM730::ID_BROADCAST, MX28::P_MOVING_SPEED_L, 0, 0); // enable stuff; needed

   int current[JointData::NUMBER_OF_JOINTS];
   int param[JointData::NUMBER_OF_JOINTS * MX28_PARAM_BYTES];
   for(int id=JointData::ID_R_SHOULDER_PITCH; id<JointData::NUMBER_OF_JOINTS; id++) {
      // get current positions
      cm730.ReadWord(id, MX28::P_PRESENT_POSITION_L, current+id, 0);
      //printf("Joint %d Position: %d\n", id, current[id]);
   }
   if (option == 1 || option == 2 || option == 5) {
      // write prep 
      int n = 0;
      for(int id=JointData::ID_R_SHOULDER_PITCH; id<JointData::NUMBER_OF_JOINTS; id++) {
         param[n++] = id;
         param[n++] = 0; // d gain
         param[n++] = 0; // i gain
         param[n++] = 2; // p gain
         param[n++] = 0; // reserved
         param[n++] = CM730::GetLowByte(current[id]); // move to middle
         param[n++] = CM730::GetHighByte(current[id]);
      }
   }

   bool zero_gyro = true;
   PhidgetIMU * imu = new PhidgetIMU(-1, zero_gyro);
   if (zero_gyro) {
      my_cm730.Sleep(2500); // gyro needs 2 seconds to zero itself
   }

   bool use_rigid = true;
   bool use_markers = false;
   std::string ps_server = "128.208.4.128";
   Phasespace * ps = new Phasespace(use_rigid, use_markers, ps_server);



   std::cout << "Begin timing tests?" << std::endl;
   getchar();


   double time = my_cm730.GetCurrentTime();
   double loop = 0.0;
   double time_sum = 0.0;
   int count = 100;
   int ret = cm730.BulkRead(); // need to do an initial read to load mem
   if (option == 0) { // read only
      for (int i=0; i<count; i++) {
         time = my_cm730.GetCurrentTime();

         int ret = cm730.BulkRead();
         std::array<double, 3> accel = imu->get_accel();

         loop = my_cm730.GetCurrentTime() - time;
         time_sum += loop;

         my_cm730.Sleep(10); // some delay between readings seems to be help?

         if (ret != CM730::SUCCESS) printf("BAD READ ");
         printf("read %f ms\n", loop);
      }
   }
   else if (option == 1) { // write only
      for (int i=0; i<count; i++) {
         time = my_cm730.GetCurrentTime();
         cm730.SyncWrite(MX28::P_D_GAIN, MX28::PARAM_BYTES, 20, param);

         loop = my_cm730.GetCurrentTime() - time;
         time_sum += loop;
         printf("write %f ms\n", loop);
      }
      printf("Writing %d bytes.\n", JointData::NUMBER_OF_JOINTS * MX28::PARAM_BYTES);
   }
   else if (option == 2) { // read & write
      for (int i=0; i<count; i++) {
         time = my_cm730.GetCurrentTime();
         int ret = cm730.BulkRead();
         cm730.SyncWrite(MX28::P_D_GAIN, MX28::PARAM_BYTES, 20, param);

         loop = my_cm730.GetCurrentTime() - time;
         time_sum += loop;

         my_cm730.Sleep(10); // some delay between readings seems to be help?

         if (ret != CM730::SUCCESS) printf("BAD READ ");
         printf("r/w %f ms\n", loop);
      }
      printf("Writing %d bytes.\n", JointData::NUMBER_OF_JOINTS * MX28::PARAM_BYTES);
   }
   else if (option == 3) { // test IMU
      double a[3];
      double g[3];
      for (int i = 0; i < count; i++) {
         time = my_cm730.GetCurrentTime();
         imu->getAccel(a);
         imu->getGyro(g);
         loop = my_cm730.GetCurrentTime() - time;
         time_sum += loop;

         my_cm730.Sleep(10); // some delay between readings seems to be help?

         printf("imu %f ms\t%f\t%f\t%f\t%f\t%f\t%f\n", loop, a[0], a[1], a[2], g[0], g[1], g[2]);
      }
   }
   else if (option == 4) { // test Phasespace
      double pose[8];
      double markers[32];
      for (int i = 0; i < count; i++) {
         time = my_cm730.GetCurrentTime();
         ps->getData(pose, markers); // synced. can also call just rigid or just markers
         loop = my_cm730.GetCurrentTime() - time;
         time_sum += loop;

         my_cm730.Sleep(10); // some delay between readings seems to be help?

         printf("phasespace %f ms\n", loop);
      }
   }
   else if (option == 5) { // test all
      double a[3];
      double g[3];
      double pose[8];
      double markers[32];
      for (int i = 0; i < count; i++) {
         time = my_cm730.GetCurrentTime();
         int ret = cm730.BulkRead();
         cm730.SyncWrite(MX28::P_D_GAIN, MX28::PARAM_BYTES, 20, param);
         imu->getAccel(a);
         imu->getGyro(g);
         ps->getData(pose, markers); // synced. can also call just rigid or just markers
         loop = my_cm730.GetCurrentTime() - time;
         time_sum += loop;

         my_cm730.Sleep(10); // some delay between readings seems to be help?
         if (ret != CM730::SUCCESS) printf("BAD READ ");
         printf("ALL: %f ms\n", loop);
      }
   }
   else {
      printf("Bad Option\n");
   }
   printf("Average Loop time %f\n", time_sum / (double)count);

   delete imu;
   delete ps;
   return 0;
}


