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

#include "robot.h"

#include <Eigen/Dense>
#include <vector>
#include <math.h>

using namespace Robot;
using namespace Eigen;

class DarwinRobot : public MyRobot {
  private:
    CM730 *cm730;
    Phasespace * ps;
    PhidgetIMU * imu;
    ContactSensors *ati;
    int * cmd_vec;
    double * i_pose;

    int *pgain;
    int *dgain;

    int m_FBGyroCenter;
    int m_RLGyroCenter;

    bool use_ps;
    bool use_imu;
    bool use_accel;
    bool use_gyro;
    bool use_ati;
    bool use_cm730;
    bool use_raw;

    Matrix3d table_rotation;
    Matrix3d init_pos;
    Vector3d *offset;

#ifdef _WIN32
    std::string BOARD_NAME="\\\\.\\COM3";
    WindowsCM730 *my_cm730;
#else
    std::string BOARD_NAME="/dev/ttyUSB0";
    LinuxCM730 *my_cm730;
#endif
    int NMARKER;

  public:
    DarwinRobot(bool joints, bool zero_gyro, bool use_rigid, bool use_markers, bool raw_markers,
        bool _use_accel, bool _use_gyro, bool _use_ati, int* p_gain, 
        std::string ps_server, double* p) {
      //phasespace: bool use_rigid = true;
      //phasespace: bool use_markers = false;
      //phasespace: std::string ps_server = "128.208.4.127";
      //imu: bool zero_gyro = true;
      //auto init1 = std::async(std::launch::async, &DarwinRobot::init_CM730, this, 2, 0);

      this->use_ps = use_rigid | use_markers;
      this->use_raw = raw_markers;
      this->use_accel = _use_accel;
      this->use_gyro = _use_gyro;
      this->use_imu = use_accel| use_gyro;
      this->use_ati = _use_ati;
      this->use_cm730 = joints;

      NMARKER = 16;

      double imu_alpha = 0.2;

      table_rotation = Matrix3d::Identity();
      init_pos = Matrix3d::Identity();
      offset = new Vector3d[NMARKER];

      std::future<bool> init2, init3, init4; 
      if (use_ps) {
        printf("Initializing Phasespace\n");
        init2 = std::async(std::launch::async, &DarwinRobot::init_phasespace, this, ps_server, use_rigid, use_markers);
      }
      //if (use_ps) {
      //  printf("Initializing Phasespace\n");
      //  init2 = std::async(std::launch::async, &DarwinRobot::init_phasespace, this, ps_server, use_rigid, use_markers);
      //}
      int data_rate = -1; // we are not using the imu in streaming mode
      if (use_accel || use_gyro) {
        printf("Initializing IMU\n");
        init3 = std::async(std::launch::async, &DarwinRobot::init_imu, this, data_rate, zero_gyro, imu_alpha);
      }
      if (use_ati) {
        printf("Initializing Contact Sensors\n");
        init4 = std::async(std::launch::deferred, &DarwinRobot::init_contacts, this);
      }


      darwin_ok = true;

      if (use_cm730) {
        printf("Initializing Body Sensors\n");
        init_CM730(p_gain, NULL);
      }

      if (use_ps) {
        if (init2.get() == false) {printf("Failed to connect to phasespace\n"); darwin_ok = false;}
        else {printf("Phasespace initialized\n");}
      }

      if (use_accel || use_gyro) {
        if (init3.get() == false) {printf("Failed to connect to phidgets IMU\n"); darwin_ok = false;}
        else {printf("IMU initialized.");}
      }

      if (use_ati) {
        if (init4.get() == false) {printf("Failed to set up Contact Sensors\n"); darwin_ok = false;}
        else {printf("Contact Sensors initialized.");}
      }

      init_pose(p);
    }

    ~DarwinRobot() {
      //delete this->my_cm730;
      delete this->cm730;
      delete this->imu;
      delete this->ps;
      delete[] offset;
      delete[] this->cmd_vec;
      delete[] this->i_pose;
    }


    bool init_CM730(int *p, int *d) {
#ifdef _WIN32
      my_cm730 = new WindowsCM730(BOARD_NAME.c_str());
#else
      my_cm730 = new LinuxCM730(BOARD_NAME.c_str());
#endif

      cm730 = new CM730(my_cm730); // our packet
      //CM730 cm730(&linux_cm730, false);
      if(cm730->Connect() == false) {
        cmd_vec = 0;
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
      //int n = 0;
      //int mid=2048;
      //for(int id=JointData::ID_R_SHOULDER_PITCH; id<JointData::NUMBER_OF_JOINTS; id++) {
      //  // initialize positions to be the same
      //  cm730->ReadWord(id, MX28::P_PRESENT_POSITION_L, &mid, 0);
      //  cmd_vec[n++] = id;
      //  cmd_vec[n++] = 0; // d gain
      //  cmd_vec[n++] = 0; // i gain
      //  cmd_vec[n++] = 2; // p gain
      //  cmd_vec[n++] = 0;
      //  cmd_vec[n++] = CM730::GetLowByte(mid); // move to middle
      //  cmd_vec[n++] = CM730::GetHighByte(mid);
      //}

      cm730->BulkRead(); // need to do a blank read to init things
      set_gyro_offsets();

      int joint_num = 0;
      //int current[JointData::NUMBER_OF_JOINTS];
      this->pgain = new int[JointData::NUMBER_OF_JOINTS];
      this->dgain = new int[JointData::NUMBER_OF_JOINTS];
      for (int joint=0; joint<JointData::ID_R_HIP_ROLL; joint++) {
        joint_num++;
        this->pgain[joint_num]=p[joint];
        this->dgain[joint_num]=0; //d[joint];
        joint_num++;
        this->pgain[joint_num]=p[joint+9];
        this->dgain[joint_num]=0; //d[joint+9];
      }
      this->pgain[JointData::ID_HEAD_PAN]=p[JointData::ID_HEAD_PAN-1];
      this->pgain[JointData::ID_HEAD_TILT]=p[JointData::ID_HEAD_TILT-1];
      this->dgain[JointData::ID_HEAD_PAN]=0; //d[JointData::ID_HEAD_PAN];
      this->dgain[JointData::ID_HEAD_TILT]=0; //d[JointData::ID_HEAD_TILT];

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

    bool init_imu(int data_rate, bool zero_gyro, double imu_alpha) {
      this->imu = new PhidgetIMU(data_rate, zero_gyro, imu_alpha);
      if (this->imu->is_running() && zero_gyro) {
        std::chrono::milliseconds interval(2500);
        std::this_thread::sleep_for(interval);
        //my_cm730.Sleep(2500); // gyro needs 2 seconds to zero itself
      }
      return this->imu->is_running();
    }

    void init_pose(double *p) {
      //if (!i_pose) {
      i_pose = new double[7];
      //}

      if (p) {
        memcpy(i_pose, p, sizeof(double)*7);
      }
      else {
        for (int i=0; i<7; i++) i_pose[i] = 0;
        //memset(i_pose, 0, sizeof(double)*7);
      }
    }

    void set_gyro_offsets() {
      int GYRO_WINDOW_SIZE = 100;
      double MARGIN_OF_SD = 2.0;
      int fb_gyro_array[GYRO_WINDOW_SIZE];// = {512,};
      int rl_gyro_array[GYRO_WINDOW_SIZE];// = {512,};
      int buf_idx = 0;

      while (buf_idx < GYRO_WINDOW_SIZE){
        cm730->BulkRead(); 
        if(cm730->m_BulkReadData[CM730::ID_CM].error == 0) {
          fb_gyro_array[buf_idx] = cm730->m_BulkReadData[CM730::ID_CM].ReadWord(CM730::P_GYRO_Y_L);
          rl_gyro_array[buf_idx] = cm730->m_BulkReadData[CM730::ID_CM].ReadWord(CM730::P_GYRO_X_L);
          //printf("%d: %d %d\n", buf_idx, fb_gyro_array[buf_idx], rl_gyro_array[buf_idx]);
          buf_idx++;
        }
      }
      double fb_sum = 0.0, rl_sum = 0.0;
      double fb_sd = 0.0, rl_sd = 0.0;
      double fb_diff, rl_diff;
      double fb_mean = 0.0, rl_mean = 0.0;
      buf_idx = 0;
      for(int i = 0; i < GYRO_WINDOW_SIZE; i++) {
        fb_sum += fb_gyro_array[i];
        rl_sum += rl_gyro_array[i];
      }
      fb_mean = fb_sum / GYRO_WINDOW_SIZE;
      rl_mean = rl_sum / GYRO_WINDOW_SIZE;

      fb_sum = 0.0; rl_sum = 0.0;
      for(int i = 0; i < GYRO_WINDOW_SIZE; i++) {
        fb_diff = fb_gyro_array[i] - fb_mean;
        rl_diff = rl_gyro_array[i] - rl_mean;
        fb_sum += fb_diff * fb_diff;
        rl_sum += rl_diff * rl_diff;
      }
      fb_sd = sqrt(fb_sum / GYRO_WINDOW_SIZE);
      rl_sd = sqrt(rl_sum / GYRO_WINDOW_SIZE);

      if(fb_sd < MARGIN_OF_SD && rl_sd < MARGIN_OF_SD) {
        m_FBGyroCenter = (int)fb_mean;
        m_RLGyroCenter = (int)rl_mean;
      }
      else {
        m_FBGyroCenter = 512;
        m_RLGyroCenter = 512;
      }
      printf("\nFBGyroCenter:%d , RLGyroCenter:%d \n", m_FBGyroCenter, m_RLGyroCenter);
    }

    void get_cm730_gyro(double* gyro) {
      gyro[0] =
        cm730->m_BulkReadData[CM730::ID_CM].ReadWord(CM730::P_GYRO_X_L) - m_RLGyroCenter;
      gyro[1] =
        cm730->m_BulkReadData[CM730::ID_CM].ReadWord(CM730::P_GYRO_Y_L) - m_FBGyroCenter;
    }

    bool get_sensors(double * time, double* sensor, double* conf) {
      // TODO convert this to just be for sensor vector: 20, 20, 3, 3, 6, 6
      // and phasespace markers
      // get data from sensors, process into qpos, qvel 
      // converts things to mujoco centric
      static double init_time = GetCurrentTimeMS();

      // try to asynchronously get the data
      if (sensor) {
        double r[6];
        double l[6];

        std::future<int> body_data;
        std::future<bool> ati_data;
        if (use_cm730) {
          body_data = std::async(std::launch::async, &CM730::BulkRead, cm730);
        }
        //if (use_ati) {
        //  ati_data = std::async(std::launch::async, &ContactSensors::getData, ati, r, l);
        //}

        double a[3];
        double g[3];
        int idx = 40; // should these be automatic?
        if (use_imu && imu->getData(a, g)) { // should be in m/s^2 and rad/sec
          if (use_accel) { sensor[idx+0]=a[0]; sensor[idx+1]=a[1]; sensor[idx+2]=a[2]; idx += 3; }
          if (use_gyro) { sensor[idx+0]=g[0]; sensor[idx+1]=g[1]; sensor[idx+2]=g[2]; idx += 3; }
        }

        //double t1 = GetCurrentTimeMS();
        if (use_ati) {
         if ( ati->getData(r, l)) {
         //if ( ati_data.get()) {
          sensor[idx+0] = r[0]; // right force x
          sensor[idx+1] = -1.0*r[1]; // right force y
          sensor[idx+2] = r[2]; // right force z
          sensor[idx+3] = r[3]; // right torque x
          sensor[idx+4] = -1.0*r[4]; // right torque y
          sensor[idx+5] = r[5]; // right torque z
          idx += 6;
          sensor[idx+0] = -1.0*l[0]; // left force x
          sensor[idx+1] = l[1]; // left force y
          sensor[idx+2] = l[2]; // left force z
          sensor[idx+3] = -1.0*l[3]; // left torque x
          sensor[idx+4] = l[4]; // left torque y
          sensor[idx+5] = l[5]; // left torque z
          idx += 6;
         }
        }
        //printf("\ncontact time: %f\n", GetCurrentTimeMS() - t1);//, init_time, GetCurrentTimeMS() - init_time);

        //double pose[8];
        double markers[NMARKER*4]; // 16 markers * (x, y, z, confidence)
        double * m_ptr = sensor+idx;
        if (use_ps) {
          if (!(ps->getMarkers(markers))) {
            return false;
          }
          else {
            // TODO get marker confidence and rotations; etc
            // table rotation, initial rotation
            if (use_raw == false) {
              for (int i=0; i<NMARKER; i++) { // TODO fix this hard coded number
                //Vector3d m(markers[i*4+0],markers[i*4+1],markers[i*4+2]);
                Map<Vector3d> m(markers + i*4);
                m = init_pos*table_rotation*m - offset[i]; // apply the rotation
              }
            }
            for (int i=0; i<NMARKER; i++) { // TODO fix this hard coded number
              m_ptr[i*3+0] = markers[i*4 + 0];
              m_ptr[i*3+1] = markers[i*4 + 1];
              m_ptr[i*3+2] = markers[i*4 + 2];
              if (conf) {
                conf[i] = markers[i*4 + 3];
              }
            }
          }
        }

        if (use_cm730) {
          if (body_data.get() != CM730::SUCCESS) {
            printf("BAD JOINT READ INTERFACE\n");
          }
          else {
            // raw values collected, convert to mujoco
            // positions
            for(int id = 1; id <= 20; id++) {
              int i = id-1;
              sensor[i] = joint2radian(cm730->m_BulkReadData[id].ReadWord(MX28::P_PRESENT_POSITION_L));
              sensor[i+20] = j_rpm2rads_ps(cm730->m_BulkReadData[id].ReadWord(MX28::P_PRESENT_SPEED_L));
            }

          }
        }

        // TODO generate root positions and velocities
        //for(int id = 0; id < 20; id++) {
        //  qpos[id+7] = s_vec[id];
        //  qvel[id+7] = s_vec[id+20];
        //}

        //printf("\n%f %f %f\n", GetCurrentTimeMS(), init_time, GetCurrentTimeMS() - init_time);
        *time = (GetCurrentTimeMS() - init_time) / 1000.0;
      }
      else {
        printf("Initialize sensor buffer\n");
        return false;
      }

      //my_cm730->Sleep(10); // some delay between readings seems to be help?
      return true;
    }

    // mujoco controls to darwin centric controls
    bool set_controls(double * u, int nu, int *p, int *d) {
      // converts controls to darwin positions
      int current[JointData::NUMBER_OF_JOINTS];
      for (int joint=0; joint<nu; joint++) {
        current[joint+1]=radian2joint(u[joint]);
      }

      // TODO setting pgain and dgain not configured yet
      if (use_cm730 && darwin_ok) {
        int n = 0;
        for(int id=JointData::ID_R_SHOULDER_PITCH; id<JointData::NUMBER_OF_JOINTS; id++) {
          cmd_vec[n++] = id;
          //cmd_vec[n++] = this->dgain; // d gain
          cmd_vec[n++] = 0; // d gain
          cmd_vec[n++] = 0; // i gain
          cmd_vec[n++] = this->pgain[id]; // p gain
          cmd_vec[n++] = 0; // reserved
          cmd_vec[n++] = CM730::GetLowByte(current[id]); // move to middle
          cmd_vec[n++] = CM730::GetHighByte(current[id]);
        }
        cm730->SyncWrite(MX28::P_D_GAIN, MX28::PARAM_BYTES, 20, cmd_vec);
        return true;
      }
      else {
        return false;
      }
    }

    void set_frame_rotation(double * rot) {
      for (int i=0; i<3; i++) 
        for (int j=0; j<3; j++) 
          table_rotation(i,j) = rot[i*3+j];
      std::cout<<"Table rotation Matrix set to be:\n"<<table_rotation<<std::endl;
    }

    void set_frame_rotation(Matrix3d rot) {
      table_rotation = rot;
      std::cout<<"Table rotation Matrix set to be:\n"<<table_rotation<<std::endl;
    }

    double angle(Vector3d v1, Vector3d v2) {
      double angle_diff = v1.dot(v2) / (v1.norm() * v2.norm());
      if (angle_diff < -1) angle_diff = -1.0;
      if (angle_diff >  1) angle_diff =  1.0;
      angle_diff = acos(angle_diff);

      return angle_diff;
    }

    // given real_chest marker positions, get the rotational offset
    void set_initpos_rt(Vector3d vec_r, Vector3d vec_s, double* mrkr, double* ps_c, double *pose) {

      Vector3d vec_z(1,0,0);
      vec_z.normalize();
      std::cout<<"V1: "<< vec_r.transpose()<<std::endl;
      std::cout<<"V2: "<< vec_s.transpose()<<std::endl;
      vec_r.normalize();
      vec_s.normalize();

      printf("\n");
      printf("Chest Vector from markers, norm: %f\n", vec_r.norm());
      printf("Chest Vector from Simultr, norm: %f\n", vec_s.norm());

      double angle_diff = angle(vec_r, vec_s);
      std::cout<<"Angle difference: "<<angle_diff*180/3.1415<<", radian:"<<angle_diff<<"\n";

      double a1 = angle(vec_r, vec_z);
      double a2 = angle(vec_s, vec_z);

      std::cout<<"Angles (degrees) from independent point:"<<a1*180/3.1415<<", "<<a2*180/3.1415<<"\n";

      if (a2 > a1) {
        init_pos<<cos(angle_diff),sin(angle_diff),0,-1*sin(angle_diff),cos(angle_diff),0,0,0,1;
      }
      else {
        init_pos<<cos(angle_diff),-1*sin(angle_diff),0,sin(angle_diff),cos(angle_diff),0,0,0,1;
      }

      std::cout<<"Rotation Matrix for initial position:\n"<<init_pos<<std::endl;

      Vector3d off(0,0,0);
      int c=0;
      std::cout<<"Calculating Offset:\n";
      for (int i=0; i<NMARKER; i++) {
        Map<Vector3d> p(pose+i*3);
        Map<Vector3d> m(mrkr+i*3);
        if (ps_c[i] > 0.5) { // saw a marker well more than half the time
            //Vector3d n = init_pos*table_rotation*m - p;
            Vector3d n = init_pos*m - p;
            std::cout<<i<<":"<<ps_c[i]<<": "<<(init_pos*m).transpose()<<" - "<<p.transpose()<<" : "<<n.transpose()<<std::endl;
            off = off + n;
            c++;
        }
      }
      off = off / (double)c; // average the offset
      std::cout<<"Offset:\n"<<off<<"\n";
      for (int i=0; i<NMARKER; i++) { // calculate the offset
          offset[i] = off;
      }
    }
};
