
#pragma once

#include <string.h>
#include <stdio.h>
#include <math.h>
#include "Vector.h"
#include "Matrix.h"
#include "MX28.h"
#include "JointData.h"

#define WALKING_SECTION "Walking Config"
#define INVALID_VALUE   -1024.0

class Walking
{
  public:
    enum
    {
      PHASE0 = 0,
      PHASE1 = 1,
      PHASE2 = 2,
      PHASE3 = 3
    };

  private:
    //static Walking* m_UniqueInstance;

    double m_PeriodTime;
    double m_DSP_Ratio;
    double m_SSP_Ratio;
    double m_X_Swap_PeriodTime;
    double m_X_Move_PeriodTime;
    double m_Y_Swap_PeriodTime;
    double m_Y_Move_PeriodTime;
    double m_Z_Swap_PeriodTime;
    double m_Z_Move_PeriodTime;
    double m_A_Move_PeriodTime;
    double m_SSP_Time;
    double m_SSP_Time_Start_L;
    double m_SSP_Time_End_L;
    double m_SSP_Time_Start_R;
    double m_SSP_Time_End_R;
    double m_Phase_Time1;
    double m_Phase_Time2;
    double m_Phase_Time3;

    double m_X_Offset;
    double m_Y_Offset;
    double m_Z_Offset;
    double m_R_Offset;
    double m_P_Offset;
    double m_A_Offset;

    double m_X_Swap_Phase_Shift;
    double m_X_Swap_Amplitude;
    double m_X_Swap_Amplitude_Shift;
    double m_X_Move_Phase_Shift;
    double m_X_Move_Amplitude;
    double m_X_Move_Amplitude_Shift;
    double m_Y_Swap_Phase_Shift;
    double m_Y_Swap_Amplitude;
    double m_Y_Swap_Amplitude_Shift;
    double m_Y_Move_Phase_Shift;
    double m_Y_Move_Amplitude;
    double m_Y_Move_Amplitude_Shift;
    double m_Z_Swap_Phase_Shift;
    double m_Z_Swap_Amplitude;
    double m_Z_Swap_Amplitude_Shift;
    double m_Z_Move_Phase_Shift;
    double m_Z_Move_Amplitude;
    double m_Z_Move_Amplitude_Shift;
    double m_A_Move_Phase_Shift;
    double m_A_Move_Amplitude;
    double m_A_Move_Amplitude_Shift;

    double m_Pelvis_Offset;
    double m_Pelvis_Swing;
    double m_Hip_Pitch_Offset;
    double m_Arm_Swing_Gain;

    bool m_Ctrl_Running;
    bool m_Real_Running;
    double m_Time;

    int m_Num_Steps;
    int m_Steps_Taken;

    int    m_Phase;
    double m_Body_Swing_Y;
    double m_Body_Swing_Z;


    double wsin(double time, double period, double period_shift, double mag, double mag_shift);
    bool computeIK(double *out, double x, double y, double z, double a, double b, double c);
    void update_param_time();
    void update_param_move();
    void update_param_balance();

    double Kinematics_CAMERA_DISTANCE;
    double Kinematics_EYE_TILT_OFFSET_ANGLE;
    double Kinematics_LEG_SIDE_OFFSET;
    double Kinematics_THIGH_LENGTH;
    double Kinematics_CALF_LENGTH;
    double Kinematics_ANKLE_LENGTH;
    double Kinematics_LEG_LENGTH;

  public:
    // Walking initial pose
    double X_OFFSET;
    double Y_OFFSET;
    double Z_OFFSET;
    double A_OFFSET;
    double P_OFFSET;
    double R_OFFSET;

    // Walking control
    double PERIOD_TIME;
    double DSP_RATIO;
    double STEP_FB_RATIO;
    double X_MOVE_AMPLITUDE;
    double Y_MOVE_AMPLITUDE;
    double Z_MOVE_AMPLITUDE;
    double A_MOVE_AMPLITUDE;
    bool A_MOVE_AIM_ON;

    // Balance control
    bool   BALANCE_ENABLE;
    double BALANCE_KNEE_GAIN;
    double BALANCE_ANKLE_PITCH_GAIN;
    double BALANCE_HIP_ROLL_GAIN;
    double BALANCE_ANKLE_ROLL_GAIN;
    double Y_SWAP_AMPLITUDE;
    double Z_SWAP_AMPLITUDE;
    double ARM_SWING_GAIN;
    double PELVIS_OFFSET;
    double HIP_PITCH_OFFSET;

    int    P_GAIN;
    int    I_GAIN;
    int    D_GAIN;

    int GetCurrentPhase()		{ return m_Phase; }
    double GetBodySwingY()		{ return m_Body_Swing_Y; }
    double GetBodySwingZ()		{ return m_Body_Swing_Z; }

    Walking();
    virtual ~Walking();

    //static Walking* GetInstance() { return m_UniqueInstance; }

    void Initialize(double * ctrl);
    void Start();
    void TakeSteps(int steps);
    void Stop();
    void Process(double dt, double * gyro, double * ctrl);
    bool IsRunning();

    //void LoadINISettings(minIni* ini);
    //void LoadINISettings(minIni* ini, const std::string &section);
    //void SaveINISettings(minIni* ini);
    //void SaveINISettings(minIni* ini, const std::string &section);
};


