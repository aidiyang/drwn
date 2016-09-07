
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


}
