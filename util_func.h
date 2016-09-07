
#pragma once

#include "mujoco.h"
#include <chrono> 
#include <string>
#include <iostream>

namespace util {
  double now_t();
  double * get_numeric_field(const mjModel* m, std::string s, int *size);
}
