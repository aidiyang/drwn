
#include "util_func.h"

namespace util {
    double now_t() {
        std::chrono::time_point<std::chrono::high_resolution_clock> t
            = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> d=t.time_since_epoch();
        return d.count(); // returns milliseconds
    }

    double * get_numeric_field(const mjModel* m, std::string s, int *size) {
        for (int i=0; i<m->nnumeric; i++) {
            std::string f = m->names + m->name_numericadr[i];
            //printf("%d %s %d\n", m->numeric_adr[i], f.c_str(), m->numeric_size[i]);
            if (s.compare(f) == 0) {
                if (size)
                    *size = m->numeric_size[i];
                return m->numeric_data + m->numeric_adr[i];
            }
        }
        return 0;
    }

    // maybe more optimized than mujoco's built in?
    //void my_copy(double* des, double * src, int n) {
    //    for (int i=0; i<n; i++)
    //        des[i] = src[i];
    //}

    void show_snsr_weights(double * snsr_ptr) {
        printf("Sensor Covariance Matrix Diagonal:\n");
        printf("    Touch        : %e\n",  snsr_ptr[0]);
        printf("    Accelerometer: %e\n",  snsr_ptr[1]);
        printf("    Velocimeter  : %e\n",  snsr_ptr[2]);
        printf("    Gyro         : %e\n",  snsr_ptr[3]);
        printf("    Force        : %e\n",  snsr_ptr[4]);
        printf("    Torque       : %e\n",  snsr_ptr[5]);
        printf("    Magnetometer : %e\n",  snsr_ptr[6]);
        printf("    JointPos     : %e\n",  snsr_ptr[7]);
        printf("    JointVel     : %e\n",  snsr_ptr[8]);
        printf("    TendonPos    : %e\n",  snsr_ptr[9]);
        printf("    TendonVel    : %e\n", snsr_ptr[10]);
        printf("    ActuatorPos  : %e\n", snsr_ptr[11]);
        printf("    ActuatorVel  : %e\n", snsr_ptr[12]);
        printf("    ActuatorFrc  : %e\n", snsr_ptr[13]);
        printf("    BallPos      : %e\n", snsr_ptr[14]);
        printf("    BallQuat     : %e\n", snsr_ptr[15]);
        printf("    FramePos     : %e\n", snsr_ptr[16]);
        printf("    FrameQuat    : %e\n", snsr_ptr[17]);
        printf("    FrameXAxis   : %e\n", snsr_ptr[18]);
        printf("    FrameYAxis   : %e\n", snsr_ptr[19]);
        printf("    FrameZAxis   : %e\n", snsr_ptr[20]);
        printf("    FrameLinVel  : %e\n", snsr_ptr[21]);
        printf("    FrameAngVel  : %e\n", snsr_ptr[22]);
        printf("    FrameLinAcc  : %e\n", snsr_ptr[23]);
        printf("    FrameAngAcc  : %e\n", snsr_ptr[24]);
        printf("    SubTreeCom   : %e\n", snsr_ptr[25]);
        printf("    SubTreeLinVel: %e\n", snsr_ptr[26]);
        printf("    SubTreeAngMom: %e\n", snsr_ptr[27]);
        printf("    Default      : %e\n", snsr_ptr[28]);
    }

    // assumes snsr_vec and snsr_ptr have been allocated
    void fill_sensor_vector(const mjModel* m, double* snsr_vec, double* snsr_ptr) {
      int my_sensordata=0;
      for (int i = 0; i < m->nsensor; i++) {      
        int type = m->sensor_type[i];
        // different sensors have different number of fields
        for (int j=my_sensordata; j<(m->sensor_dim[i]+my_sensordata); j++) {
          snsr_vec[j] = snsr_ptr[type];
          //if (snsr_range) snsr_limit[j] = snsr_range[type];
          //else snsr_limit[j] = 1e6; // filler
        }
        my_sensordata += m->sensor_dim[i];
      }
    }

    void darwin_torques(double * torques, const mjModel * m, mjData *d, double * ctrl, double min_t, double kp) {
        // servo torque = kp * (goal - sensor position)
        //printf("\nctrl input:\n");
        //for(int i = 0; i < m->nu; i++) {
        //    printf("%f %f\n", ctrl[i], d->sensordata[i]);
        //}
        //printf("\npositional difference:\n");
        //for(int i = 0; i < m->nu; i++) {
        //    printf("%f ", ctrl[i] - d->sensordata[i]);
        //}
        //printf("\nlimited torques:\n");
        for(int i = 0; i < m->nu; i++) {
            torques[i] = kp * (ctrl[i] - d->sensordata[i]);
            // use built in force limiting
            //if (torques[i] > 0) 
            //  torques[i] = torques[i] > max_t ? max_t : torques[i];
            //else 
            //  torques[i] = abs(torques[i]) > max_t ? -max_t : torques[i];

            // dead band
            torques[i] = abs(torques[i]) < min_t ? 0.0 : torques[i];
            //printf("%f ", torques[i]);
        }
        //printf("\n");
    }


}
