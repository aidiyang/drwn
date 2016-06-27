#include "mujoco.h"
#include <string.h>
//#include <random>
#include <functional>

#include "estimator.h"
#ifndef __APPLE__
#include <omp.h>
#endif
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>

#include <string>
#include <cstring>
#include <sstream>
#include <iostream>

#include <boost/program_options.hpp>

using namespace Eigen;
namespace po = boost::program_options;
char error[1000];
mjModel* m; // defined in viewer_lib.h to capture perturbations
mjData* d;

std::string makeString(double arr[], int size) {
  std::stringstream ss;
  for(int i = 0; i < size; i++) {
    if(i != 0) {
      ss << ",";
    }
    ss << arr[i];
  }
  std::string s = ss.str();
  return s;
}

std::ofstream myfile;
std::string to_string(double x);
void save_states(std::string filename, mjModel *m, mjData * est, std::string mode = "w") {
  if (mode=="w") {
    // create file
    myfile.open(filename, std::ofstream::out);
    myfile<<"time,";
    for (int i=0; i<m->nq; i++) {
      myfile<<"qpos,";
    }

    for (int i=0; i<m->nv; i++) 
      myfile<<"qvel,";
    for (int i=0; i<m->nu; i++) 
      myfile<<"ctrl,";
    for (int i=0; i<m->nsensordata; i++) 
      myfile<<"snsr,";

    myfile<<"\n";
    myfile.close();
  }
  else if (mode=="c") {
    myfile.close();
    return;
  }
  else {
    if (!myfile.is_open()) {
      //printf("HAD TO OPEN OUTPUT FILE AGAIN!!!!!!\n");
      myfile.open(filename, std::ofstream::out | std::ofstream::app );
    }

    myfile<<est->time<<",";

    std::string eststr = makeString(est->qpos, m->nq) + "," + makeString(est->qvel, m->nv) + "," + makeString(est->ctrl, m->nu)
            + "," + makeString(est->sensordata, m->nsensordata) + ",";
    myfile<<eststr;

    myfile<<"\n";

  }
}


int main(int argc, const char** argv) {
	
  mj_activate("mjkey.txt");

  //bool engage; // for real robot?
  std::string model_name;
  std::string output_file;
  std::string dir; 
  int maxtime;
  double s_noise;
  double c_noise;
  double e_noise;
  int sim;

	
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help", "Usage guide")
			//("engage,e", po::value<bool>(&engage)->default_value(false), "Engage motors for live run.")
			("model,m", po::value<std::string>(&model_name)->required(), "Model file to load.")
      ("directory, d", po::value<std::string>(&dir)->default_value("./mcouts/"), "Directory name to put out files in")
			("output,o", po::value<std::string>(&output_file)->default_value("mcout"), "Output file prefix.")
			("timesteps,t", po::value<int>(&maxtime)->default_value(500), "Number of times to allow estimator to run before quitting.")
			("s_noise,s", po::value<double>(&s_noise)->default_value(0.0), "Gaussian amount of sensor noise to corrupt data with.")
			("c_noise,c", po::value<double>(&c_noise)->default_value(0.0), "Gaussian amount of control noise to corrupt data with.")
			("e_noise,e", po::value<double>(&e_noise)->default_value(0.0), "Gaussian amount of estimator noise to corrupt data with.")
			("simulations,i", po::value<int>(&sim)->default_value(100), "Number of simulations to do")
      ;

		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);
		if (vm.count("help")) {
			std::cout << desc << std::endl;
			return 0;
		}
		po::notify(vm);
	}
	catch(std::exception& e) {
		std::cerr << "Error: " << e.what() << "\n";
		return 0;
	}
	catch(...) {
		std::cerr << "Unknown error!\n";
		return 0;
	}

  printf("Model:\t\t\t%s\n", model_name.c_str());
  printf("Directory:\t\t%s\n", dir.c_str());
  printf("Out file prefix:\t%s\n", output_file.c_str());
  printf("Number of simulations:\t%u\n", sim);
  printf("Timesteps:\t\t%d\n", maxtime);
  printf("Sensor Noise:\t\t%f\n", s_noise);
  printf("Control Noise:\t\t%f\n", c_noise);


  mjModel* m = mj_loadXML(model_name.c_str(), 0, 0, 0);
  //mjData* d = mj_makeData(m);

  printf("total state vector size:\t%u\n", m->nq+m->nv+m->nu+m->nsensordata + 1);

  //Create control noise
  static std::random_device rd;
  static std::mt19937 rng(rd());
  static std::normal_distribution<> nd(0, c_noise);

//#pragma omp parallel for
  for(int i=0; i<sim; i++) {
    mjData* d = mj_makeData(m);   //remake data (reset robot) in order to start at time 0.0 again
    std::stringstream sname;
    sname<<dir<<output_file<<"_"<<i<<".csv";
    std::string filename = sname.str();
    save_states(filename, m, d, "w");  //Create/open file
    for(int j = 0; j<maxtime; j++) {
      //Make control noise array
      double cnoise[m->nu];
      for (int j = 0; j<m->nu; j++) {
        //cnoise[j] = nd(rng);
        d->ctrl[j] = nd(rng);
      }
      //d->ctrl = cnoise;
      save_states(filename, m, d, "a");
      mj_step(m, d);
	  }
    save_states(filename, m, d, "a");
    myfile.close();
    mj_deleteData(d);   //free data
  }

}




