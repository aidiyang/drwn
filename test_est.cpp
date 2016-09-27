
#define __AVX__ 1
#include "util_func.h"
#include "estimator.h"
#include "darwin_hw/drwn_walker.h"

#include <boost/program_options.hpp>

namespace po = boost::program_options;

// defined in viewer_lib.h to capture perturbations
mjModel* m;
mjData* d;

void diff_state(double* diff, const mjModel* m, mjData* d1, mjData* d2, bool print) {
  for (int i=0; i<m->nq; i++) {
    double d = d1->qpos[i] - d2->qpos[i];
    if (print) printf("%1.4f ", d);
    diff[i] += d;
  }
  if (print) printf(":: ");
  for (int i=0; i<m->nv; i++) {
    double d = d1->qvel[i] - d2->qvel[i];
    if (print) printf("%1.4f ", d);
    diff[i+m->nq] += d;
  }
  //printf(":: ");
  //for (int i=0; i<m->nv; i++) {
  //  printf("%1.4f ", d1->qacc[i] - d2->qacc[i]);
  //}
  printf("\n");
}

void print_state(const mjModel* m, const mjData* d) {
  for (int i=0; i<m->nq; i++) {
    printf("%1.4f ", d->qpos[i]);
  }
  printf(":: ");
  for (int i=0; i<m->nv; i++) {
    printf("%1.4f ", d->qvel[i]);
  }
  printf(":: ");
  for (int i=0; i<m->nv; i++) {
    printf("%1.4f ", d->qacc[i]);
  }
  printf("\n");
}

int main(int argc, const char** argv) {

  int num_threads=1;
  int estimation_counts;
  //bool engage; // for real robot?
  std::string model_name;// = new std::string();
  std::string input_file;
  double s_noise;
  double c_noise;
  double e_noise;
  double alpha;
  double beta;
  double kappa;
  double diag;
  double Ws0;
  double tol;
  double s_time_noise=0.0;
  //bool do_correct;
  bool debug;
  bool useUKF;

  try {
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help", "Usage guide")
      //("engage,e", po::value<bool>(&engage)->default_value(false), "Engage motors for live run.")
      ("model,m", po::value<std::string>(&model_name)->required(), "Model file to load.")
      ("timesteps,c", po::value<int>(&estimation_counts)->default_value(300), "Number of times to allow estimator to run before quitting.")
      //("do_correct,d", po::value<bool>(&do_correct)->default_value(true), "Do correction step in estimator.")
      ("debug,n", po::value<bool>(&debug)->default_value(false), "Debugging output.")
      //("velocity,v", po::value<std::string>(vel_file), "Binary file of joint velocity data")
      ("s_noise,s", po::value<double>(&s_noise)->default_value(0.0), "Gaussian amount of sensor noise to corrupt data with.")
      ("c_noise,p", po::value<double>(&c_noise)->default_value(0.0), "Gaussian amount of control noise to corrupt data with.")
      ("e_noise,e", po::value<double>(&e_noise)->default_value(1.0), "Gaussian amount of estimator noise to corrupt data with.")
      ("alpha,a", po::value<double>(&alpha)->default_value(0.001), "Alpha: UKF param")
      ("beta,b", po::value<double>(&beta)->default_value(2), "Beta: UKF param")
      ("kappa,k", po::value<double>(&kappa)->default_value(0), "Kappa: UKF param")
      ("diagonal,d", po::value<double>(&diag)->default_value(1), "Diagonal amount to add to UKF covariance matrix.")
      ("weight_s,w", po::value<double>(&Ws0)->default_value(-1.0), "Set inital Ws weight.")
      ("tol,i", po::value<double>(&tol)->default_value(-1.0), "Set Constraint Tolerance (default NONE).")
      ("UKF,u", po::value<bool>(&useUKF)->default_value(true), "Use UKF or EKF")
      //("dt,t", po::value<double>(&dt)->default_value(0.02), "Timestep in binary file -- checks for file corruption.")
      ("threads,t", po::value<int>(&num_threads)->default_value(std::thread::hardware_concurrency()>>1), "Number of OpenMP threads to use.")
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


  if(Ws0 > 1) {
    Ws0 = 1;
    printf("Inputted weight was larger than 1, Ws0 has been clamped to 1\n\n");
  }

  printf("Model:\t\t%s\n", model_name.c_str());
  printf("OMP threads:\t\t%d\n", num_threads);
  printf("Timesteps:\t\t%d\n", estimation_counts);
  printf("Sensor Noise:\t\t%f\n", s_noise);
  printf("Control Noise:\t\t%f\n", c_noise);
  printf("UKF alpha:\t\t%f\n", alpha);
  printf("UKF beta:\t\t%f\n", beta);
  printf("UKF kappa:\t\t%f\n", kappa);
  printf("UKF diag:\t\t%f\n", diag);
  printf("UKF Ws0:\t\t%f\n", Ws0);
  printf("UKF Tol:\t\t%f\n", tol);

  // Start Initializations

  printf("MuJoCo Pro library version %.2lf\n\n", 0.01 * mj_version());
  if (mjVERSION_HEADER != mj_version())
    mju_error("Headers and library have different versions");

  // activate MuJoCo license
  mj_activate("mjkey.txt");

  if (!model_name.empty()) {
    //m = mj_loadModel(model_name.c_str(), 0, 0);
    char error[1000] = "could not load binary model";
    m = mj_loadXML(model_name.c_str(), 0, error, 1000);
    if (!m) {
      printf("%s\n", error);
      return 1;
    }

    d = mj_makeData(m);
    mj_forward(m, d);
  }
  int nq = m->nq;
  int nv = m->nv;
  int nu = 20;//m->nu;
  int nsensordata = m->nsensordata;

  ////// SIMULATED ROBOT
  double dt = m->opt.timestep;
  printf("DT of mujoco model %f\n", dt);


  double time = 0.0;
  double prev_time = 0.0;
  double *qpos = new double[nq];
  double *qvel = new double[nv];
  double *ctrl = new double[nu];
  double *sensors = new double[nsensordata];
  double *conf = new double[16];

  for (int i=0; i<nu; i++) { ctrl[i] = 0.0; }
  for (int i=0; i<16; i++) { conf[i] = 4.4; }

  // init darwin to walker pose
  Walking * walker = new Walking();
  if (nu >= 20) {
    walker->Initialize(ctrl);
  }

  double * s_cov = util::get_numeric_field(m, "snsr_covar", NULL);
  double * p_cov = util::get_numeric_field(m, "covar_diag", NULL);

  Estimator * est = 0;
  if (useUKF) {
    est = new UKF(m, d, s_cov, p_cov,
        alpha, beta, kappa, diag, Ws0, e_noise, tol, debug, num_threads);
  }
  //else 
  //  est = new EKF(m, d, e_noise, tol, diag, debug, num_threads);
  mjData * est_data = 0;
  est_data = est->get_state();

  printf("DT is set %f\n", dt);

  bool render_inplace = false;
  bool print = false;

  double *avg_diff = new double[nq+nv];
  double p_time = 0.0;
  double c_time = 0.0;
  for (int i=0; i<(nq+nv); i++) { avg_diff[i] = 0.0; }

  for (int i=0; i<estimation_counts; i++) {

    mj_step(m, d); // to get the sensord data at the current place
    time = d->time;

    // simulate and render
    printf("robot hw time: %f\n", time);
    printf("prev time: %f\n", prev_time);

    //////////////////////////////////
    printf("robot sensor time: %f, est DT: %f\n", time, est_data->time);

    //////////////////////////////////
    double t0 = util::now_t();
    est->predict_correct_p1(ctrl, time-prev_time, d->sensordata, conf);

    //////////////////////////////////
    double t1 = util::now_t();
    est->predict_correct_p2(ctrl, time-prev_time, d->sensordata, conf);

    double t2 = util::now_t();

    printf("\n\t\t estimator predict %f ms, correct %f ms, total %f ms\n",
        t1-t0, t2-t1, t2-t0);
    p_time += t1-t0;
    c_time += t2-t1;

    diff_state(avg_diff, m, d, est_data, print);

    //printf("qpos:\n");
    //for (int i=0; i<nq; i++) {
    //  printf("%1.4f ", avg_diff[i]);
    //}
    //printf("\nqvel:\n");
    //for (int i=0; i<m->nv; i++) {
    //  printf("%1.4f ", avg_diff[i+nq]);
    //}
    //printf("\n");

    // we have estimated and logged the data,
    if (nu >= 20) {
      walker->Process(time-prev_time, 0, ctrl);
    }

    prev_time = time;
  }
  printf("True:\n");
  print_state(m, d);
  printf("est:\n");
  print_state(m, est_data);

  mj_deactivate();
  mj_deleteData(d);
  mj_deleteModel(m);

  for (int i=0; i<(nq+nv); i++) {
    avg_diff[i] /= (double)estimation_counts;
  }

  printf("Average Predict: %f\n", p_time/(double)estimation_counts);
  printf("Average Correct: %f\n", c_time/(double)estimation_counts);

  printf("Average differences\nqpos:\n");
  for (int i=0; i<nq; i++) {
    printf("%1.5f ", avg_diff[i]);
  }
  printf("\nqvel:\n");
  for (int i=0; i<nv; i++) {
    printf("%1.5f ", avg_diff[i+nq]);
  }
  printf("\n");

  //printf("\n");
  delete walker;
  delete[] qpos;
  delete[] qvel;
  delete[] ctrl;
  delete[] sensors;
  delete[] conf;
  delete[] avg_diff;
  delete est;

  return 0;
}
