
#pragma once 

#ifdef _WIN32
#include "WindowsDARwIn.h"
#else
#include "LinuxCM730.h"
#endif

#include "JointData.h"


void Prompt(int id);
void Help();
void Scan(Robot::CM730 *cm730);
void Dump(Robot::CM730 *cm730, int id);
void Reset(Robot::CM730 *cm730, int id);
void Write(Robot::CM730 *cm730, int id, int addr, int value);

