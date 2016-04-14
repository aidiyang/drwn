#pragma once

#include <stdio.h>
#include <phidget21.h>
#include <array>

//#include "Eigen/Dense"
//using namespace Eigen;

class PhidgetIMU {

  private:
	bool imu_ok;
	std::array<double, 3> accel;
	std::array<double, 3> gyro;
	std::array<double, 3> gravi;
	//Vector3d accel, gyro, gravi;

	CPhidgetSpatialHandle imu;

	static int CCONV handler(CPhidgetSpatialHandle device, void *userptr,
		CPhidgetSpatial_SpatialEventDataHandle *data, int count) {

	  PhidgetIMU* imu = (PhidgetIMU*) userptr;
	  if (count > 0) { // gets last data point

		imu->gravi[1] = -1.0*data[count-1]->acceleration[0]; // just get g's in case
		imu->accel[1] = -1.0*9.81*data[count-1]->acceleration[0]; // convert g's to m/ss

		imu->gravi[0] = -1.0*data[count-1]->acceleration[1]; // just get g's in case
		imu->accel[0] = -1.0*9.81*data[count-1]->acceleration[1]; // convert g's to m/ss

		imu->gravi[2] = data[count-1]->acceleration[2]; // just get g's in case
		imu->accel[2] = 9.81*data[count-1]->acceleration[2]; // convert g's to m/ss

		for (int i = 0; i < 3; i++) {
		  imu->gyro[i] = 0.0017453*data[count - 1]->angularRate[(i + 1) % 3]; // convert deg/s to rad/s
		}

		/*
		   imu->gravi(1) = -1.0*data[count - 1]->acceleration[0]; // just get g's in case
		   imu->accel(1) = -1.0*9.81*data[count - 1]->acceleration[0]; // convert g's to m/ss

		   imu->gravi(0) = -1.0*data[count - 1]->acceleration[1]; // just get g's in case
		   imu->accel(0) = -1.0*9.81*data[count - 1]->acceleration[1]; // convert g's to m/ss

		   imu->gravi(2) = data[count - 1]->acceleration[2]; // just get g's in case
		   imu->accel(2) = 9.81*data[count - 1]->acceleration[2]; // convert g's to m/ss

		   for (int i = 0; i < 3; i++) {
		   imu->gyro(i) = 0.0017453*data[count-1]->angularRate[ (i+1) % 3]; // convert deg/s to rad/s
		   }
		   */
	  }
	  return 0;
	}

  public:

	PhidgetIMU(int delay, bool zero_gyro) { // delay in ms, or negative to disable handler
	  imu_ok = true;
	  //imu = NULL;
	  CPhidgetSpatial_create(&imu);

	  // attach listener
	  if (delay > 0) {
		CPhidgetSpatial_set_OnSpatialData_Handler(imu, PhidgetIMU::handler, this);
	  }

	  // initialize device
	  if (CPhidget_open((CPhidgetHandle)imu, -1)) {
		// cant open
		CPhidget_delete((CPhidgetHandle)imu);
        printf("Couldn't open Phidget device.\n");
	  	imu_ok = false;
	  }
	  //if (CPhidget_waitForAttachment((CPhidgetHandle)imu, 2000)) {
      int err;
	  if ((err=CPhidget_waitForAttachment((CPhidgetHandle)imu, 2000)) != EPHIDGET_OK) {
        const char* errStr;
        CPhidget_getErrorDescription(err, &errStr);

        printf("Couldn't attach to Phidget device. Error %d: %s\n", err, errStr);
	    CPhidget_close((CPhidgetHandle)imu);
	    CPhidget_delete((CPhidgetHandle)imu);
	  	imu_ok = false;
	  }

	  if (imu_ok && zero_gyro) {
        printf("Phidget: Zero-ing gyro.\n");
		CPhidgetSpatial_zeroGyro(imu); // takes about 2 seconds
	  }

	  if (imu_ok && delay > 0) {
        printf("Phidget: Setting Data Rate at %d ms.\n", delay);
		CPhidgetSpatial_setDataRate(imu, delay);
	  }

	  // create vector structures
	  /*
		 accel = Vector3d::Zero();
		 gyro = Vector3d::Zero();
		 */

	  accel = { 0.0, 0.0, 0.0 };
	  gyro = { 0.0, 0.0, 0.0 };
	  gravi = { 0.0, 0.0, 0.0 };

	}
	bool is_running() { return imu_ok; }

	bool getData(double *a, double *g) {
	  double r[3];
	  if (CPhidgetSpatial_getAcceleration(imu, 0, r+1)) {imu_ok=false;}
	  if (CPhidgetSpatial_getAcceleration(imu, 1, r+0)) {imu_ok=false;}
	  if (CPhidgetSpatial_getAcceleration(imu, 2, r+2)) {imu_ok=false;}

	  a[0] = -1.0*9.81*r[0];
	  a[1] = -1.0*9.81*r[1];
	  a[2] =      9.81*r[2];
	  if (CPhidgetSpatial_getAngularRate(imu, 0, r + 1)) {imu_ok=false;}
	  if (CPhidgetSpatial_getAngularRate(imu, 1, r + 2)) {imu_ok=false;}
	  if (CPhidgetSpatial_getAngularRate(imu, 2, r + 0)) {imu_ok=false;}
	  for (int i = 0; i < 3; i++) {
		g[i] = 0.0017453*r[i];
	  }
	  return imu_ok;
	}

	void getAccel(double *a) {
	  double r[3];
	  CPhidgetSpatial_getAcceleration(imu, 0, r+1);
	  CPhidgetSpatial_getAcceleration(imu, 1, r+0);
	  CPhidgetSpatial_getAcceleration(imu, 2, r+2);

	  a[0] = -1.0*9.81*r[0];
	  a[1] = -1.0*9.81*r[1];
	  a[2] =      9.81*r[2];

	}

	void getGyro(double *g) {
	  double r[3];
	  CPhidgetSpatial_getAngularRate(imu, 0, r + 1);
	  CPhidgetSpatial_getAngularRate(imu, 1, r + 2);
	  CPhidgetSpatial_getAngularRate(imu, 2, r + 0);
	  for (int i = 0; i < 3; i++) {
		g[i] = 0.0017453*r[i];
	  }
	}

	~PhidgetIMU() {
	  // close handle
	  CPhidget_close((CPhidgetHandle)imu);
	  CPhidget_delete((CPhidgetHandle)imu);
	}

	/*
	   Vector3d get_accel() { return Vector3d(accel); }
	   Vector3d get_grav() { return Vector3d(gravi); }
	   Vector3d get_gyro() { return Vector3d(gyro); }
	   */
	std::array<double, 3> const & get_accel() const { return accel; }
	std::array<double, 3> const & get_grav() const { return gravi; }
	std::array<double, 3> const & get_gyro() const { return gyro; }
};
