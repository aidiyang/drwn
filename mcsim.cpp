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
#include <boost/filesystem.hpp>

using namespace Eigen;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

//char error[1000];
mjModel* m; // defined in viewer_lib.h to capture perturbations
mjData* d;

//Convert inputted array in a string
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

std::ofstream myfile; //Need global file or no?
std::string to_string(double x);

//Save the states of real, estimate, and covar (stddev) to inputed file
void save_states(std::string filename, mjModel *m, mjData *real, mjData *est, mjData *stddev,
      std::string mode = "w") {
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

    for (int i=0; i<m->nq; i++) 
      myfile<<"est_p,";
    for (int i=0; i<m->nv; i++) 
      myfile<<"est_v,";
    for (int i=0; i<m->nu; i++) 
      myfile<<"est_c,";
    for (int i=0; i<m->nsensordata; i++) 
      myfile<<"est_s,";

    for (int i=0; i<m->nq; i++) 
      myfile<<"stddev_p,";
    for (int i=0; i<m->nv; i++) 
      myfile<<"stddev_v,";
    for (int i=0; i<m->nu; i++) 
      myfile<<"stddev_c,";
    for (int i=0; i<m->nsensordata; i++) 
      myfile<<"stddev_s,";

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

    myfile<<real->time<<",";

    std::string realstr = makeString(real->qpos, m->nq) + "," + makeString(real->qvel, m->nv) + "," + makeString(real->ctrl, m->nu)
            + "," + makeString(real->sensordata, m->nsensordata) + ",";
    myfile<<realstr;

    std::string eststr = makeString(est->qpos, m->nq) + "," + makeString(est->qvel, m->nv) + "," + makeString(est->ctrl, m->nu)
            + "," + makeString(est->sensordata, m->nsensordata) + ",";
    myfile<<eststr;

    std::string stdstr = makeString(stddev->qpos, m->nq) + "," + makeString(stddev->qvel, m->nv) + "," + makeString(stddev->ctrl, m->nu)
            + "," + makeString(stddev->sensordata, m->nsensordata) + ",";
    myfile<<stdstr;

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
  bool debug;
  double alpha;
  double beta;
  double kappa;
  double diag;
  double Ws0;
  int num_threads = 1;//omp_get_num_procs();

	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help", "Usage guide")
			//("engage,e", po::value<bool>(&engage)->default_value(false), "Engage motors for live run.")
			("model,m", po::value<std::string>(&model_name)->required(), "Model file to load.")
      ("dir", po::value<std::string>(&dir)->default_value("./mcouts/"), "Directory name to put out files in")
			("output, o", po::value<std::string>(&output_file)->default_value("mcout"), "Output file prefix.")
			("timesteps,t", po::value<int>(&maxtime)->default_value(100), "Number of times to allow estimator to run before quitting.")
			("s_noise,s", po::value<double>(&s_noise)->default_value(0.0), "Gaussian amount of sensor noise to corrupt data with.")
			("c_noise,c", po::value<double>(&c_noise)->default_value(0.0), "Gaussian amount of control noise to corrupt data with.")
			("e_noise,e", po::value<double>(&e_noise)->default_value(0.0), "Gaussian amount of estimator noise to corrupt data with.")
			("simulations,i", po::value<int>(&sim)->default_value(30), "Number of simulations to do")
      ("debug,n", po::value<bool>(&debug)->default_value(false), "Do correction step in estimator.")
      ("alpha,a", po::value<double>(&alpha)->default_value(10e-3), "Alpha: UKF param")
      ("beta,b", po::value<double>(&beta)->default_value(2), "Beta: UKF param")
      ("kappa,k", po::value<double>(&kappa)->default_value(0), "Kappa: UKF param")
      ("diagonal,d", po::value<double>(&diag)->default_value(1e-6), "Diagonal amount to add to UKF covariance matrix.")
      ("weight_s,w", po::value<double>(&Ws0)->default_value(-1.0), "Set inital Ws weight.")
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

  omp_set_num_threads(num_threads);
  omp_set_dynamic(0);

  if(Ws0 > 1) {
    Ws0 = 1;
    printf("Inputted weight was larger than 1, Ws0 has been clamped to 1\n\n");
  }

  printf("Model:\t\t\t%s\n", model_name.c_str());
  printf("OMP threads:\t\t%d\n", num_threads);
  printf("Directory:\t\t%s\n", dir.c_str());
  printf("Out file prefix:\t%s\n", output_file.c_str());
  printf("Number of simulations:\t%u\n", sim);
  printf("Timesteps:\t\t%d\n", maxtime);
  printf("Sensor Noise:\t\t%f\n", s_noise);
  printf("Control Noise:\t\t%f\n", c_noise);
  printf("UKF alpha:\t\t%f\n", alpha);
  printf("UKF beta:\t\t%f\n", beta);
  printf("UKF kappa:\t\t%f\n", kappa);
  printf("UKF diag:\t\t%f\n", diag);
  printf("UKF Ws0:\t\t%f\n", Ws0);

  //Make directory if does not exist yet
  if(!fs::is_directory(dir)) {
    //boost::filesystem::path dir(dir);
    boost::filesystem::create_directory(dir);
    printf("Made directory");
  }  

  mjModel* m = mj_loadXML(model_name.c_str(), 0, 0, 0);
  //Estimator * est = 0;

  printf("total state vector size:\t%u\n", m->nq+m->nv+m->nu+m->nsensordata + 1);

  //Create control noise
  static std::random_device rd;
  static std::mt19937 rng(rd());
  static std::normal_distribution<> nd(0, c_noise);
  static std::normal_distribution<> snd(0, s_noise);
  double estctrl[m->nu];
  double estsensor[m->nsensordata];
  for (int i = 0; i<m->nu; i++) {
    estctrl[i] = 0;
  }

//#pragma omp parallel for
  for(int i=0; i<sim; i++) {
    mjData* d = mj_makeData(m);   //remake data (reset robot) in order to start at time 0.0 again
    mj_forward(m,d); // init
    Estimator * est = new UKF(m, d, alpha, beta, kappa, diag, Ws0, e_noise, -1, debug, 1);   //Make new UKF estimator
    std::stringstream mcname;
    mcname<<"./"<<dir<<"/"<<output_file<<"_"<<i<<".csv";   //MCout file prefix
    std::string filename = mcname.str();
    save_states(filename, m, d, est->get_state(), est->get_stddev(), "w");  //Create/open MC file
    for(int j = 0; j<maxtime; j++) {
      //Make control noise array
      for (int k = 0; k < m->nu; k++) {
        d->ctrl[k] = nd(rng);
      }
      for (int k = 0; k < m->nsensordata; k++) {
        estsensor[k] = d->sensordata[k] + snd(rng);
      }
      save_states(filename, m, d, est->get_state(), est->get_stddev(), "a");
      //mj_step(m,d);
      mj_Euler(m,d);
      mj_forward(m,d);
      mj_sensor(m,d);
      est->predict_correct(estctrl, m->opt.timestep, estsensor);
	  }
    save_states(filename, m, d, est->get_state(), est->get_stddev(), "a");
    myfile.close();     //Close file, if don't use global file, use save_states close
    mj_deleteData(d);   //free data
    delete est;         //Delete UKF to free data

  }

}




