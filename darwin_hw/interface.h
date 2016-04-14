//#include "DARwIn.h"
#include "CM730.h"
#include "MX28.h"
#include "JointData.h"

// sensors
#include "s_imu.h"
#include "s_contacts.h"
#include "s_phasespace.h"

#include "Utilities.h"
#include <future>
#include <thread>

#ifdef _WIN32
#include "WindowsCM730.h"
#else
#include "LinuxCM730.h"
#endif


using namespace Robot;

class DarwinRobot {
  private:
    CM730 *cm730;
    Phasespace * ps;
    PhidgetIMU * imu;
    ContactSensors *ati;
    int * cmd_vec;
    double * i_pose;

    int pgain;
    int dgain;

    bool darwin_ok;

#ifdef _WIN32
      std::string BOARD_NAME="\\\\.\\COM3";
      WindowsCM730 *my_cm730;
#else
      std::string BOARD_NAME="/dev/ttyUSB0";
      LinuxCM730 *my_cm730;
#endif

  public:
    DarwinRobot(bool zero_gyro, bool use_rigid, bool use_markers,
        std::string ps_server, double* p) {
      //phasespace: bool use_rigid = true;
      //phasespace: bool use_markers = false;
      //phasespace: std::string ps_server = "128.208.4.127";
      //imu: bool zero_gyro = true;
      //auto init1 = std::async(std::launch::async, &DarwinRobot::init_CM730, this, 2, 0);
      auto init2 = std::async(std::launch::async, &DarwinRobot::init_phasespace, this, ps_server, use_rigid, use_markers);
      int data_rate = -1; // we are not using the imu in streaming mode
      auto init3 = std::async(std::launch::async, &DarwinRobot::init_imu, this, data_rate, zero_gyro);
      auto init4 = std::async(std::launch::async, &DarwinRobot::init_contacts, this);

      darwin_ok = true;

      init_CM730(2, 0);
      //if (init1.get() == false) {printf("Failed to connect to CM730\n"); darwin_ok = false;}
      //else {printf("CM730 initialized.\n");}

      if (init2.get() == false) {printf("Failed to connect to phasespace\n"); darwin_ok = false;}
      else {printf("Phasespace initialized\n");}

      if (init3.get() == false) {printf("Failed to connect to phidgets IMU\n"); darwin_ok = false;}
      else {printf("IMU initialized.");}

      if (init4.get() == false) {printf("Failed to set up Contact Sensors\n"); darwin_ok = false;}
      else {printf("Contact Sensors initialized.");}

      init_pose(p);
    }

    ~DarwinRobot() {
      delete this->my_cm730;
      delete this->cm730;
      delete this->imu;
      delete this->ps;
      delete[] this->cmd_vec;
      delete[] this->i_pose;
    }

    bool is_running() {
        return darwin_ok;
    }

    bool init_CM730(int p, int d) {
#ifdef _WIN32
      my_cm730 = new WindowsCM730(BOARD_NAME.c_str());
#else
      my_cm730 = new LinuxCM730(BOARD_NAME.c_str());
#endif

      cm730 = new CM730(my_cm730); // our packet
      //CM730 cm730(&linux_cm730, false);
      if(cm730->Connect() == false) {
        printf("Fail to initializeCM-730\n");
        return false;
      }

      for(int id=JointData::ID_R_SHOULDER_PITCH; id<JointData::NUMBER_OF_JOINTS; id++) {
        int error=0;
        int value=0;
        if(cm730->ReadWord(id, MX28::P_PRESENT_POSITION_L, &value, &error) != CM730::SUCCESS) {
          printf("Failure to initialize motor %d\n", id);
        }
      }

      cm730->WriteWord(CM730::ID_BROADCAST, MX28::P_MOVING_SPEED_L, 0, 0); // enable stuff; needed

      // configure command vector to accept gains or just positions
      cmd_vec = new int[JointData::NUMBER_OF_JOINTS * MX28::PARAM_BYTES];
      int n = 0;
      int current=2048;
      for(int id=JointData::ID_R_SHOULDER_PITCH; id<JointData::NUMBER_OF_JOINTS; id++) {
        // initialize positions to be the same
        cm730->ReadWord(id, MX28::P_PRESENT_POSITION_L, &current, 0);
        cmd_vec[n++] = id;
        cmd_vec[n++] = 0; // d gain
        cmd_vec[n++] = 0; // i gain
        cmd_vec[n++] = 2; // p gain
        cmd_vec[n++] = 0;
        cmd_vec[n++] = CM730::GetLowByte(current); // move to middle
        cmd_vec[n++] = CM730::GetHighByte(current);
      }

      int ret = cm730->BulkRead(); // need to do a blank read to init things

      this->pgain = p;
      this->dgain = d;
      return true;
    }

    bool init_contacts() {
      this->ati = new ContactSensors();
      return this->ati->is_running();
	  return true;
    }

    bool init_phasespace(std::string server, bool use_rigid, bool use_markers) {
      this->ps = new Phasespace(use_rigid, use_markers, server);
      return this->ps->isRunning();
    }

    bool init_imu(int data_rate, bool zero_gyro) {
      this->imu = new PhidgetIMU(data_rate, zero_gyro);
      if (this->imu->is_running() && zero_gyro) {
        std::chrono::milliseconds interval(2500);
        std::this_thread::sleep_for(interval);
        //my_cm730.Sleep(2500); // gyro needs 2 seconds to zero itself
      }
      return this->imu->is_running();
    }

    void init_pose(double *p) {
      if (!i_pose) {
        i_pose = new double[7];
      }

      if (p) {
        memcpy(i_pose, p, sizeof(double)*7);
      }
      else {
        memset(i_pose, 0, sizeof(double)*7);
      }
    }

    bool get_state(double* qpos, double* qvel, double *sensor) {
      // get data from sensors, process into qpos, qvel
      // converts things to mujoco centric

      // try to asynchronously get the data
      auto body_data = std::async(std::launch::async, &CM730::BulkRead, cm730);
      //int ret = cm730->BulkRead();
      //if (ret != CM730::SUCCESS) {printf("BAD READ ");}


      double pose[8];
      double markers[32];
      // TODO need to do quaternion fix
      //if (!(ps->getData(pose, markers))) {
      //  return false;
      //}

      double a[3];
      double g[3];
      imu->getData(a, g); // should be in m/s^2 and rad/sec


      double r[6];
      double l[6];
      bool c_ret = ati->getData(r, l);

      // TODO add raw joint values to sensors?
      if (sensor) {
        printf(" %f %f %f\n", a[0], a[1], a[2]);
        sensor[0]=a[0]; sensor[1]=a[1]; sensor[2]=a[2];
        sensor[3]=g[0]; sensor[4]=g[1]; sensor[5]=g[2];
        if (c_ret) {
          for (int id=0; id<6; id++) {
            sensor[6+id] = r[id];
          }
          for (int id=0; id<6; id++) {
            sensor[12+id] = l[id];
          }
        }
      }

      if (body_data.get() != CM730::SUCCESS) {printf("BAD READ ");}

      // raw values collected, convert to mujoco
      int i = 0;
      // positions
      double *s_vec = new double[40];
      for(int id = 1; id <= 17; id+=2) // Right Joints
        s_vec[i++] = joint2radian(cm730->m_BulkReadData[id].ReadWord(MX28::P_PRESENT_POSITION_L));
      for(int id = 2; id <= 18; id+=2) // Left Joints
        s_vec[i++] = joint2radian(cm730->m_BulkReadData[id].ReadWord(MX28::P_PRESENT_POSITION_L));
      for(int id = 19; id <= 20; id++) // Head Joints
        s_vec[i++] = joint2radian(cm730->m_BulkReadData[id].ReadWord(MX28::P_PRESENT_POSITION_L));

      // velocities
      for(int id = 1; id <= 17; id+=2) // Right Joints
        s_vec[i++] = j_rpm2rads_ps(cm730->m_BulkReadData[id].ReadWord(MX28::P_PRESENT_SPEED_L));
      for(int id = 2; id <= 18; id+=2) // Left Joints
        s_vec[i++] = j_rpm2rads_ps(cm730->m_BulkReadData[id].ReadWord(MX28::P_PRESENT_SPEED_L));
      for(int id = 19; id <= 20; id++) // Head Joints
        s_vec[i++] = j_rpm2rads_ps(cm730->m_BulkReadData[id].ReadWord(MX28::P_PRESENT_SPEED_L));

      // TODO generate root positions and velocities
      for(int id = 0; id < 20; id++) {
        qpos[id+7] = s_vec[id];
        qvel[id+7] = s_vec[id+20];
      }

      my_cm730->Sleep(10); // some delay between readings seems to be help?
      return true;
    }

    bool set_controls(double * u, int *p, int *d) {
      // converts controls to darwin positions
      int joint_num = 0;
      int current[JointData::NUMBER_OF_JOINTS];
      for (int joint=0; joint<JointData::ID_R_HIP_ROLL; joint++) {
        joint_num++;
        current[joint_num]=radian2joint(u[joint]);
        joint_num++;
        current[joint_num]=radian2joint(u[joint+9]);
      }
      current[JointData::ID_HEAD_PAN]=radian2joint(u[JointData::ID_HEAD_PAN]);
      current[JointData::ID_HEAD_TILT]=radian2joint(u[JointData::ID_HEAD_TILT]);

      // TODO setting pgain and dgain not configured yet
      int n = 0;
      for(int id=JointData::ID_R_SHOULDER_PITCH; id<JointData::NUMBER_OF_JOINTS; id++) {
        cmd_vec[n++] = id;
        cmd_vec[n++] = this->dgain; // d gain
        cmd_vec[n++] = 0; // i gain
        cmd_vec[n++] = this->pgain; // p gain
        cmd_vec[n++] = 0; // reserved
        cmd_vec[n++] = CM730::GetLowByte(current[id]); // move to middle
        cmd_vec[n++] = CM730::GetHighByte(current[id]);
      }
      cm730->SyncWrite(MX28::P_D_GAIN, MX28::PARAM_BYTES, 20, cmd_vec);
      return true;
    }

};
