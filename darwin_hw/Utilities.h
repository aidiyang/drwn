
#pragma once

#include <time.h>

double GetCurrentTimeMS()
{
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

