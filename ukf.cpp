#include "estimator.h"
#include "util_func.h"

#ifdef USE_EIGEN_MKL
#define EIGEN_USE_MKL_ALL
#endif

#define EIGEN_DONT_PARALLELIZE // no major effect
//#define EIGEN_DONT_VECTORIZE // not much difference 
//#define EIGEN_DEFAULT_TO_ROW_MAJOR // might mess up data access
//#define EIGEN_NO_DEBUG // saves a little time

#include <vector>
#include <Eigen/Dense>
//#define SYMMETRIC_SQUARE_ROOT
#ifdef SYMMETRIC_SQUARE_ROOT
#include <unsupported/Eigen/MatrixFunctions>
#endif

using namespace Eigen;


int omp_get_thread_num() { return 0; }
int omp_get_num_threads() { return 1; }


UKF::UKF(mjModel *m, mjData * d, double * snsr_weights, double *P_covar,
    double _alpha, double _beta, double _kappa,
    double _diag, double _Ws0, double _noise,
    double _tol,
    bool debug, int threads) : Estimator(m, d) {

  ctrl_state = false;

  L = nq + nv; // size of state dimensions
  if (ctrl_state) L += nu;

  N = 2*L + 1;

  alpha = _alpha;
  beta = _beta;
  lambda = (alpha*alpha)*((double)L + _kappa) - (double)L;
  diag = _diag;
  Ws0 = _Ws0;
  tol = _tol;
  printf("L size: %d\t N: %d\n", L, N);
  printf("Alpha %f\nBeta %f\nLambda %f\nDiag %f\nWs0 %f\n", alpha, beta, lambda, diag, Ws0);
  printf("Constraint Tolerance: %f\n", tol);

  W_s = new double[N];
  W_theta = new double[N];
  W_c = new double[N];
  sigma_states.resize(2*nq+1); // only useful to visualize qpos perturbed

  my_threads = threads;
  sigmas.resize(threads);
  models.resize(threads);

  //omp_set_dynamic(0);
  //omp_set_num_threads(threads);

  stddev = mj_makeData(this->m);

  printf("Setting up %d thread models\n", threads);
  for (int i=0; i<my_threads; i++) {
    models[i] = mj_copyModel(NULL, m);

    models[i]->opt.iterations = 25; // TODO to make things faster 
    models[i]->opt.tolerance = 0; 

    sigmas[i] = mj_makeData(this->m);
    mj_copyData(sigmas[i], this->m, this->d); // data initialization
    printf("Initialized %d model for UKF\n", i);
  }
  //prev_d = mj_makeData(this->m); // data from t-1

  //raw.resize(N);
  m_x = MatrixXd::Zero(L, N);
  m_gamma = MatrixXd::Zero(ns, N);
  x = new VectorXd[N];
  gamma = new VectorXd[N];
  for (int i=0; i<(2*nq+1); i++) {
    if (i==0) {
      sigma_states[i] = this->d;
    }
    else {
      sigma_states[i] = mj_makeData(this->m);
      mj_copyData(sigma_states[i], this->m, this->d);
    }
    mj_forward(m, sigma_states[i]);
  }

  W_even = 1.0/N;
  for (int i=0; i<N; i++) {
    if (i==0) {
      //sigma_states[i] = this->d;
      W_s[i] = lambda / ((double) L + lambda);
      W_c[i] = W_s[i] + (1-(alpha*alpha)+beta);
      
      W_s[i] = 1.0/N;
      W_c[i] = 1.0/N;
      //W_c[i] = W_s[i];
    }
    else {
      //sigma_states[i] = mj_makeData(this->m);
      //mj_copyData(sigma_states[i], this->m, this->d);

      // Traditional weighting
      W_s[i] = 1.0 / (2.0 * ((double)L + lambda));
      W_c[i] = W_s[i];

      // even weighting
      W_s[i] = 1.0/N;
      W_c[i] = 1.0/N;
    }
    x[i] = VectorXd::Zero(L); // point into raw buffer, not mjdata

    //gamma[i] = Map<VectorXd>(sigma_states[i]->sensordata, ns);
    gamma[i] = VectorXd::Zero(ns);
  }
  double suma = 0.0;
  double sumb = 0.0;
  printf("W_s: ");
  for (int i=0; i<N; i++) {
    suma+=W_s[i];
    printf("%f ", W_s[i]); 
  }
  printf("\nW_c: ");
  for (int i=0; i<N; i++) {
    sumb+=W_c[i];
    printf("%f ", W_c[i]); 
  }
  printf("\nSums %f %f\n", suma, sumb);

  P_t = MatrixXd::Identity(L,L)*1e-3;
  P_z = MatrixXd::Zero(ns, ns);
  Pxz = MatrixXd::Zero(L,ns);

  x_t = VectorXd::Zero(L);
  x_minus = VectorXd::Zero(L);

  z_k = VectorXd::Zero(ns);

  mju_copy(&x_t(0), this->d->qpos, nq);
  mju_copy(&x_t(nq), this->d->qvel, nv);
  if (ctrl_state) mju_copy(&x_t(nq+nv), this->d->ctrl, nu);

  this->noise = _noise;

  this->NUMBER_CHECK = debug;

  PtAdd = MatrixXd::Identity(L, L);
  if (P_covar) {
    // HACK TODO fix this; assumes nq is the same as nv
    for (int i=0; i<nq; i++) {
      PtAdd(i,i) = P_covar[0];
      PtAdd(i+nq,i+nq) = P_covar[0];

      P_t(i,i) = P_covar[0];
      P_t(i+nq,i+nq) = P_covar[0];
    }
  }
  std::cout<<"Ptadd:\n"<<PtAdd.diagonal().transpose()<<std::endl;

  q_hat = VectorXd::Zero(L);
  Q = PtAdd;

  s_hat = VectorXd::Zero(ns);

  PzAdd = MatrixXd::Identity(ns, ns);
  snsr_limit = new double[ns];
  double default_snsr[29] = {
    1e-4,1e-4,1e-4,1e-7,1e-4,1e-6,1e-3,
    1e-7,1e-5,1e-4,1e-5,1e-4,1e-5,1e-3,
    1e-6,1e-6,1e-7,1e-4,1e-7,1e-7,1e-7,
    1e-5,1e-2,1e-5,1e-2,1e-1,1e-1,1e-1,1e-0};
  snsr_ptr = new double[29];
  double* snsr_range = util::get_numeric_field(m, "snsr_range", NULL); 

  if (m->nnumericdata >= 29 && snsr_weights) {
    for (int i=0; i<29; i++)
      snsr_ptr[i] = snsr_weights[i];
  } else {
    for (int i=0; i<29; i++)
      snsr_ptr[i] = default_snsr[i];
  }
  util::show_snsr_weights(snsr_ptr);
  
  int my_sensordata=0;
  for (int i = 0; i < m->nsensor; i++) {      
    int type = m->sensor_type[i];
    // different sensors have different number of fields
    for (int j=my_sensordata; j<(m->sensor_dim[i]+my_sensordata); j++) {
      PzAdd(j, j) = snsr_ptr[type];
      if (snsr_range) snsr_limit[j] = snsr_range[type];
      else snsr_limit[j] = 1e6; // filler
    }
    my_sensordata += m->sensor_dim[i];
  }
  mrkr_conf = snsr_ptr[16];
  printf("Filled PzAdd: %d %d\n", my_sensordata, ns);

  rd_vec = new std::normal_distribution<>[ns]; 
  for (int i=0; i<ns; i++) {
    rd_vec[i] = std::normal_distribution<>( 0, sqrt(PzAdd(i,i)) );
  }

  ct_vec = NULL;
  if (diag > 0) {
    ct_vec = new std::normal_distribution<>[nu]; 
    for (int i=0; i<nu; i++) {
      ct_vec[i] = std::normal_distribution<>( 0, sqrt(diag) );
      //ct_vec[i] = std::normal_distribution<>( 0, sqrt(diag)*this->noise );
    }
  }

  m_noise = util::get_numeric_field(m, "mass_var", NULL);
  t_noise = util::get_numeric_field(m, "time_var", NULL);

  if (m_noise) printf("Total Mass Var: %f\n", (*m_noise));
  if (t_noise) printf("Time Var: %f\n", (*t_noise));

  sigma_handles = new std::future<double>[my_threads];
};

UKF::~UKF() {
  delete[] W_s;
  delete[] W_theta;
  delete[] W_c;
  delete[] x;
  delete[] gamma;
  delete[] snsr_ptr;
  delete[] sigma_handles;

  for (int i=0; i<my_threads; i++) {
    mj_deleteData(sigmas[i]);
    mj_deleteModel(models[i]);
  }
  for (int i=1; i<(2*nq+1); i++) {
    // sigma_state[0] = d main model, deleted in estimator.h
    mj_deleteData(sigma_states[i]);
  }
  mj_deleteData(stddev);

  delete[] rd_vec;
  if (ct_vec) delete[] ct_vec;
};

double UKF::add_time_noise() {
  double dt = 0;
  if (t_noise) {
    static std::mt19937 t_rng(50505);
    static std::normal_distribution<double> nd(0, (*t_noise) * this->noise);
    dt = nd(t_rng);
  }
  return dt;
}

void UKF::add_model_noise(mjModel* t_m) {
  if (m_noise) {
    static std::mt19937 m_rng(12345);
    static std::normal_distribution<double> mass_r(0, (*m_noise) * this->noise);
    // TOTAL MASS
    //double newmass = mj_getTotalmass(m) + mass_r(m_rng);
    //mj_setTotalmass(t_m, newmass);
    for (int i=0; i<t_m->nbody; i++) {
      //t_m->body_mass[i] = m->body_mass[i] + mass_r(m_rng)/(double)m->nbody;
      t_m->body_mass[i] = m->body_mass[i] + mass_r(m_rng);
    }
  }

}

void UKF::add_ctrl_noise(double * ctrl) {
  static std::mt19937 c_rng(505050);
  //static std::normal_distribution<double> nd(0, _sigma);
  if (diag > 0) {
    for (int i=0; i<nu; i++) {
      double r = ct_vec[i](c_rng);
      ctrl[i] += r;
    }
  }
}

void UKF::add_snsr_noise(double * snsr) {
  static std::mt19937 s_rng(494949);
  // make a vector ns in length with appropriate sigmas / variances

  for (int i=0; i<ns; i++) {
    if (noise > 0) { // should be bool...?
      double r = rd_vec[i](s_rng);
      snsr[i] += r;
    }
    // clamp sensor values
    if (snsr[i] > snsr_limit[i]) snsr[i] = snsr_limit[i];
    if (snsr[i] < -1.0*snsr_limit[i]) snsr[i] = -1.0*snsr_limit[i];
  }
}

// vector to data
void UKF::set_data(mjData* data, VectorXd *x) {
  mju_copy(data->qpos,   &x[0](0), nq);
  mju_copy(data->qvel,   &x[0](nq), nv);
  if (ctrl_state) mju_copy(data->ctrl,   &x[0](nq+nv), nu); // set controls for this t
}

// data to vector
void UKF::get_data(mjData* data, VectorXd *x) {
  mju_copy(&x[0](0),  data->qpos, nq);
  mju_copy(&x[0](nq), data->qvel, nv);
  if (ctrl_state) mju_copy(&x[0](nq+nv), data->ctrl, nu);
}

void UKF::copy_state(mjData * dst, mjData * src) {
  mju_copy(dst->qpos, src->qpos, nq);
  mju_copy(dst->qvel, src->qvel, nv);
  if (ctrl_state) mju_copy(dst->ctrl, src->ctrl, nu);
}

void UKF::fast_forward(mjModel *t_m, mjData * t_d, int index, int sensorskip) {
  // NOTE: the mj_forwardSkip doesn't seem to be thread safe
#if 1
  if (ctrl_state && index >= (nq+nv))
    mj_forwardSkip(t_m, t_d, 2, sensorskip); // just ctrl calcs
  else if (index >= nq)
    mj_forwardSkip(t_m, t_d, 1, sensorskip); // velocity cals
  else
    mj_forwardSkip(t_m, t_d, 0, sensorskip); // all calculations
#else
  mj_forward(m, t_d); // all calculations
#endif
}

double UKF::constraint_scale(mjData * t_d, double tol) {
  double max = 0.0;
  if (tol < 0.0) return 1.0; // shortcut out to avoid scaling
  for (int i=0; i<nv; i++) {
    if (t_d->qfrc_constraint[i] > max) {
      max = t_d->qfrc_constraint[i];
    }
  }
  if (max > tol) return tol / max;
  else return 1.0;
}

double UKF::constraint_violated(mjData* t_d) {
  VectorXd constraint = Map<VectorXd>(t_d->qfrc_constraint, nv);
  //VectorXd constraint = Map<VectorXd>(t_d->qfrc_inverse, nv);
  return constraint.maxCoeff();
}

double UKF::handle_constraint(mjData* t_d, double tol,
    VectorXd* t_x, VectorXd* x0) {
  double scale = 1.0;
  set_data(t_d, t_x);
  double vio = constraint_violated(t_d);
  std::cout<< "\tscale: 0.0 violation: "<<vio<<"\n";
  std::cout<< "\terror: "<<x0[0].transpose()<<"\n";
  if (vio <= tol) {
    return scale;
  }
  else {
    //scale = 0.5;
    for (int i=2; i<16; i++) { // limit the line-search

      VectorXd new_x_t = t_x[0] + (scale*x0[0]);

      mj_resetData(m, t_d); 
      set_data(t_d, &(new_x_t));

      std::cout<< "origin:" << t_x[0].transpose() << std::endl;
      std::cout<< "scaled:" << new_x_t.transpose() << std::endl;

      //for (int j=0; j<nv; j++) t_d->qacc[j] = 0.0;
      //for (int j=0; j<nv; j++) t_d->qfrc_constraint[j] = 0.0;
      //mju_copy(t_d->qacc, d->qacc, nv); // copy from center point
      //fast_forward(t_d, j);
      mj_forward(m, t_d);

      vio = constraint_violated(t_d);
      std::cout<< "\tscale: "<< scale<<" violation: "<<vio<<"\n";
      if (vio <= tol) { break; }
      //if (vio > tol) { scale = scale - pow(0.5, i); }
      //else { scale = scale + pow(0.5, i); }
      scale = scale / 10.0;
    }
  }
  return scale;
}

double UKF::handle_constraint(mjData* t_d, mjData* d, double tol,
    VectorXd* t_x, VectorXd* x0, VectorXd* P) {
  double scale = 1.0;
  if (constraint_violated(t_d) <= tol) {
    return scale;
  }
  else {
    scale = 0.5;
    for (int i=2; i<8; i++) { // limit the line-search

      t_x[0] = x0[0] + (scale*P[0]);
      set_data(t_d, t_x);
      mju_copy(t_d->qacc, d->qacc, nv); // copy from center point
      //fast_forward(t_d, j);
      mj_forward(m, t_d);
      //mj_inverse(m, t_d);

      if (constraint_violated(t_d) > tol) {
        scale = scale - pow(0.5, i);
      }
      else {
        scale = scale + pow(0.5, i);
      }
    }
  }
  return scale;
}

double UKF::sigma_samples(mjModel *t_m, mjData *t_d, mjData *d, double* ctrl,
    VectorXd *x, MatrixXd *m_sqrt, int s, int e) {

  double t0 = util::now_t();
  for (int j=s; j<e; j++) {
    // step through all the perturbation cols and collect the data
    int i = j+1;
    ///////////////////////////////////////////////////// sigma point
    x[i+0] = x[0]+m_sqrt->col(i-1);// + q_hat;

    t_d->time = d->time;

    set_data(t_d, &(x[i+0]));
    mju_copy(t_d->qacc, d->qacc, nv); // copy from center point

    if (!ctrl_state) mju_copy(t_d->ctrl, ctrl, nu); // set controls for this t
    add_ctrl_noise(t_d->ctrl);

    add_model_noise(t_m);
    fast_forward(t_m, t_d, j, 1); // skip sensor

    t_m->opt.timestep = m->opt.timestep + add_time_noise();
    mj_Euler(t_m, t_d);

    get_data(t_d, &(x[i+0]));

    fast_forward(t_m, t_d, 0, 0); // dont skip sensor

    add_snsr_noise(t_d->sensordata);
    mju_copy(&(gamma[i](0)), t_d->sensordata, ns);
    mju_copy(&(m_gamma.col(i)(0)), d->sensordata, ns);

    if (j < nq) { // only copy position perturbed
      mju_copy(sigma_states[i+0]->qpos, t_d->qpos, nq);
    }

    ///////////////////////////////////////////////////// symmetric point
    x[i+L] = x[0]-m_sqrt->col(i-1);// + q_hat;

    t_d->time = d->time;
    set_data(t_d, &(x[i+L]));
    if (!ctrl_state) mju_copy(t_d->ctrl, ctrl, nu); // set controls for this t
    add_ctrl_noise(t_d->ctrl);
    mju_copy(t_d->qacc, d->qacc, nv); // copy from center point

    add_model_noise(t_m);
    fast_forward(t_m, t_d, j, 1);

    t_m->opt.timestep = m->opt.timestep + add_time_noise();
    mj_Euler(t_m, t_d);

    get_data(t_d, &(x[i+L]));

    fast_forward(t_m, t_d, 0, 0); // dont skip sensor

    add_snsr_noise(t_d->sensordata);
    mju_copy(&(gamma[i+L](0)), t_d->sensordata, ns);
    mju_copy(&(m_gamma.col(i+L)(0)), d->sensordata, ns);

    if (j < nq) { // only copy position perturbed states
      mju_copy(sigma_states[i+nq]->qpos, t_d->qpos, nq);
    }
  }
  return util::now_t() - t0;
}
void UKF::predict_correct_p10(double * ctrl, double dt, double* sensors, double* conf) {
  //fast_forward(m, d, 0, 0); // dont skip sensor
  //mj_Euler(m, d);
  //fast_forward(m, d, 0, 0); // dont skip sensor

  m->opt.timestep = dt; // input to this function
  m->opt.iterations = 100; 
  m->opt.tolerance = 1e-6; 
  mju_copy(d->ctrl, ctrl, nu); // set controls for the center point
  mj_step(m, d);
  for (int i=1; i<(2*nq+1); i++) {
    mju_copy(sigma_states[i]->qpos, d->qpos, nq);
  }
}

void UKF::predict_correct_p1(double * ctrl, double dt, double* sensors, double* conf) {

  //double t2 = util::now_t();
  //int end = (nq > 26) ? 26 : nq;
  IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");

  //set_data(d, &(x_t)); // did this in _p2 step
  mju_copy(d->ctrl, ctrl, nu); // set controls for the center point

  get_data(d, &(x[0])); //x[0] = x_t;

  //double t3 = util::now_t();

#ifdef SYMMETRIC_SQUARE_ROOT
  MatrixXd m_sqrt = ((L+lambda)*(P_t)).sqrt(); // symmetric; blows up
  // compilation time.... 
#else
  LLT<MatrixXd> chol((L+lambda)*(P_t));
  MatrixXd m_sqrt = chol.matrixL(); // chol
#endif

  //double t4 = util::now_t();
  if (conf) { // our sensors have confidence intervals
    // TODO more than just phasespace markers?
    if (ns > (40+6+12)) { // we have phasespace markers
      int ps_start = ns - 16*3;
      //printf("ps %d ns %d\n", ps_start, ns);
      for (int j=0; j<16; j++) { // DIRTY HACKY HACK
        for (int i=0; i<3; i++) {
          int idx = ps_start + j*3 + i;
          // our conf threshold (basically not visible)
          if (conf[j] < 0.0 || conf[j] > 10.0) { PzAdd(idx, idx) = 1e+100; }
          else { PzAdd(idx, idx) = mrkr_conf; }
        }
      }
      //IOFormat CleanFmt(2, 0, ", ", "\n", "[", "]");
      //std::cout << "pzadd output:\n"<< PzAdd.block(ps_start, ps_start, ns-ps_start, ns-ps_start).format(CleanFmt) << std::endl;
    }
  }
  // Simulation options
  m->opt.timestep = dt; // input to this function
  m->opt.iterations = 100; 
  m->opt.tolerance = 1e-6; 

  mj_forwardSkip(m, d, 0, 1); // skip sensors

  //for (int i=1; i<N; i++) { W_theta[i] = sqrt(L+lambda); }

  // set tolerance to be low, run 50, 100 iterations for mujoco solver
  // copy qacc for sigma points with some higher tolerance
  double times[my_threads];
  for (int i=0; i<my_threads; i++) {
    int s = i * L / my_threads;
    int e = (i + 1 ) * L / my_threads;
    if (i == my_threads-1) e = L;

    times[i] = util::now_t();
    sigma_handles[i] = std::async(std::launch::async,
        &UKF::sigma_samples, this,
        models[i], sigmas[i], d, ctrl, x, &m_sqrt, s, e);
    //printf("Thread %d : %d sigmas\n", i, e-s);
  }

  // step for the central point
  mj_Euler(m, d); // step

  get_data(d, &(x[0]));
  //get_data(d, &(x_minus)); // pre-correction

  fast_forward(m, d, 0, 0); // dont skip sensor
  //gamma[0] = Map<VectorXd>(d->sensordata, ns);
  add_snsr_noise(d->sensordata);
  mju_copy(&(gamma[0](0)), d->sensordata, ns);
  mju_copy(&(m_gamma.col(0)(0)), d->sensordata, ns);

  mju_copy(sigma_states[0]->qpos, d->qpos, nq);
  //mju_copy(sigma_states[0]->ctrl, d->ctrl, nu);
  //double t5 = util::now_t();

  ////////////////// Weights scaled according to constraints
  //double S_theta = 0.0;
  //for (int i=1; i<N; i++) { S_theta += W_theta[i]; }
  //double sq = sqrt(L+lambda);
  //double a = (2.0*lambda - 1.0) / (2.0*(L+lambda)*(S_theta - (N*sq)));
  //double b = (1.0)/(2.0*(L+lambda)) - (2.0*lambda - 1.0)/(2.0*sq*(S_theta - (N*sq)));

  // finish up sigma point processing
  double pure_t = 0.0;
  double pure_ts[my_threads];
  for (int i=my_threads-1; i>=0; i--) {
    pure_ts[i] = sigma_handles[i].get();
    if (pure_ts[i] > pure_t)
      pure_t = pure_ts[i];
    times[i] = util::now_t() - times[i];
  }

  //printf("Max proc time: %f\n", pure_t);
  //for (int i=0; i<my_threads; i++) {
  //  printf("Thread %d : %f / %f : d: %f\n", i, pure_ts[i], times[i], times[i]-pure_ts[i]);
  //}

  // aprior mean
  x_t.setZero();
  z_k.setZero(); // = VectorXd::Zero(ns);
  for (int i=1; i<N; i++) {
    x_t += W_s[i]*x[i];
    z_k += W_s[i]*gamma[i];
    //std::cout<<i<<" : "<<gamma[i].segment(40,18).transpose()<<std::endl;
  }
  x_t = W_s[0]*x[0]     + x_t;
  z_k = W_s[0]*gamma[0] + z_k; // offset the sensors with an adaptive bias
  // TODO  can optimize more if all the weights are the same

  // even weighting
  //for (int i=0; i<N; i++) {
  //z_k += W_even*gamma[i];
  //x_t += W_even*x[i];
  //}
  //x_t = W_even*x_t;
  //z_k = W_even*z_k;

  //std::cout << "snsr mean: " << z_k << std::endl;
  //std::cout << "snsr diff: " << m_gamma.rowwise().mean() << std::endl;
  //std::cout << "snsr diff: " << z_k - m_gamma.colwise().mean() << std::endl;

  P_t.setZero();
  P_z.setZero();
  Pxz.setZero();

  for (int i=1; i<N; i++) {
    VectorXd z(gamma[i] - z_k);
    VectorXd x_i(x[i] - x_t);

    //P_t += W_even*(x_i * x_i.transpose()); // needs to include w_even here to avoid over/underflow?
    //P_z += W_even*(z * z.transpose());
    //Pxz += W_even*(x_i * z.transpose());

    P_t += W_c[i] * (x_i * x_i.transpose());
    P_z += W_c[i] * (z * z.transpose());
    Pxz += W_c[i] * (x_i * z.transpose());
  }
  VectorXd x_i(x[0] - x_t);
  VectorXd z(gamma[0] - z_k);
  P_t = W_c[0]*(x_i * x_i.transpose()) + P_t;
  P_z = W_c[0]*(z * z.transpose())     + P_z + PzAdd;
  Pxz = W_c[0]*(x_i * z.transpose())   + Pxz;

  //P_t = W_even*P_t;
  //P_z = P_z + PzAdd;
  //Pxz = W_even*Pxz;

  if (diag < 0) { // if no control noise, force pt to stay sane
    P_t = P_t + PtAdd; 
  }

  // other method of combining estimates
  // TODO different method for X_1 by multiplying with kalman gain each
  //for (int i=0; i<N; i++) {
  //  x[i] = x[i] + (K*(s-gamma[i])); // other way
  //}
  //// sigma point
  //VectorXd x_t_aug = VectorXd::Zero(L);
  //for (int i=0; i<N; i++) { x_t += W_s[i]*x[i]; }
  //for (int i=1; i<N; i++) { x_t_aug += x[i]; }
  //x_t_aug = W_s[0]*x[0] + (W_s[1])*x_t_aug;

  //MatrixXd P_t_aug = MatrixXd::Zero(L,L);
  //for (int i=0; i<N; i++) {
  //  VectorXd x_i(x[i] - x_t);
  //  P_t_aug += W_c[i] * (x_i * x_i.transpose()); // aprior covarian
  //}
  //double t6 = util::now_t();

  //printf("\ncombo init %f, sqrt %f, mjsteps %f, merge %f\n",
  //        t3-t2, t4-t3, t5-t4, t6-t5);
  //double t2 = util::now_t();

  if (NUMBER_CHECK) {
    int end = (nq > 26) ? 26 : nq;
    VectorXd c = Map<VectorXd>(ctrl, nu);
    std::cout << "\nraw ctrl:\n"<< (c).transpose().format(CleanFmt) << "\n";

    std::cout << "\nPredict Pz  :\n"<< P_z.block(0,0,end,end).format(CleanFmt) << "\n";
    std::cout << "\nPredict Pz next:\n"<< P_z.block(end,end,ns,ns).format(CleanFmt) << "\n";
    std::cout << "\nPredict Pz^-1:\n"<< P_z.inverse().block(0,0,end,end).format(CleanFmt) << "\n";
    std::cout << "\nPredict Pxz :\n"<< Pxz.block(0,0,end,end).format(CleanFmt) << "\n\n";

    std::cout << "\npredict p_t :\n"<< P_t.block(0,0,end,end).format(CleanFmt) << "\n";
    std::cout << "\npredict x_t:\n"<< x_t.transpose().format(CleanFmt) << "\n";
    std::cout << "\npredict z_k hat:\n"<< (z_k).transpose().format(CleanFmt) << "\n";
  }
  //printf("\ncombo est %f\n", util::now_t()-t2);
}

void UKF::predict_correct_p2(double * ctrl, double dt, double* sensors, double* conf) {

  MatrixXd K = Pxz * P_z.inverse();

  VectorXd s = Map<VectorXd>(sensors, ns); // map our real z to vector

  z_k = z_k + s_hat;
  VectorXd c_v = (K*(s-z_k));

  set_data(d, &(x_t)); // set corrected state into mujoco data struct
  mj_forward(m, d); // dont skip sensor
  std::cout<<"Predict Energy: "<<d->energy[0]<<", "<<d->energy[1]<<std::endl;
  double total_energy = d->energy[0] + d->energy[1];


  x_t = x_t + (K*(s-z_k));
  P_t = P_t - (K * P_z * K.transpose());

  set_data(d, &(x_t)); // set corrected state into mujoco data struct

  double dk = 0.09; //0.99;
  s_hat = (1-dk)*s_hat + dk*(s-z_k);
  std::cout<<"Sensor Vector     :\n" << s.transpose() << "\n";
  std::cout<<"Sensor Esti Vector:\n" << z_k.transpose() << "\n";
  std::cout<<"Sensor Bias Vector:\n" << s_hat.transpose() << "\n";
  std::cout<<"Summed:\n" << (s-z_k).transpose() << "\n";

  //VectorXd new_x_t = x_t + (K*(s-z_k));
  //set_data(d, &(x_t)); // set corrected state into mujoco data struct
  //std::cout<< "Correction constraint violation norm predict:" << constraint_violated(d) << "\n";
  //set_data(d, &(new_x_t)); // set corrected state into mujoco data struct
  //std::cout<< "Correction constraint violation norm correct:" << constraint_violated(d) << "\n";

  //double scale = handle_constraint(d, tol, &(x_t), &(c_v));
  //x_t = x_t + scale*(K*(s-z_k));

  //set_data(d, &(x_t)); // set corrected state into mujoco data struct
  //std::cout<< "Correction constraint violation scale:" << scale << "\n";
  //std::cout<< "Correction constraint violation norm scaled :" << constraint_violated(d) << "\n";



  mju_copy(d->sensordata, &(z_k(0)), ns); // copy estimated data for viewing 
#if 1
  //fast_forward(m, d, 0, 0); // dont skip sensor

  mj_forward(m, d); // dont skip sensor
  std::cout<<"Correct Energy: "<<d->energy[0]<<", "<<d->energy[1]<<std::endl;
  total_energy = d->energy[0] + d->energy[1];
  
  if (total_energy > 20) {
    x_t = x_t - (K*(s-z_k));
    P_t = P_t + (K * P_z * K.transpose());
    printf("IGNORING UPDATE STEP!!\n");
    printf("IGNORING UPDATE STEP!!\n");
    printf("IGNORING UPDATE STEP!!\n");
    printf("IGNORING UPDATE STEP!!\n");
    printf("IGNORING UPDATE STEP!!\n");
  }
#endif

  if (NUMBER_CHECK) {
    int end = (nq > 26) ? 26 : nq;
    IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");
    std::cout << "\nKalman Gain:\n"<< K.block(0,0,end,end).format(CleanFmt) << "\n";
    std::cout << "\nraw sensors:\n"<< (s).transpose().format(CleanFmt) << "\n";
    std::cout << "\ndelta:\n"<< (s-z_k).transpose().format(CleanFmt) << "\n";

    std::cout << "\nCorrect Pz  :\n"<< P_z.block(0,0,end,end).format(CleanFmt) << "\n";
    if (nq>26) std::cout << "\nCorrect Pz^-1:\n"<< P_z.inverse().block(0,0,end,end).format(CleanFmt) << "\n";
    std::cout << "\nCorrect Pxz :\n"<< Pxz.block(0,0,end,end).format(CleanFmt) << "\n\n";

    std::cout << "\nCorrect p_t :\n"<< P_t.block(0,0,end,end).format(CleanFmt) << "\n";
    if (nq>26) std::cout << "\nCorrect p_t next :\n"<< P_t.block(end,end,L,L).format(CleanFmt) << "\n";
    std::cout << "\nCorrect x_t:\n"<< x_t.transpose().format(CleanFmt) << "\n";
    std::cout << "\nCorrect z_k hat:\n"<< (z_k).transpose().format(CleanFmt) << "\n";
    std::cout << "Predict and Correct COMBO\n";
  }
}

void UKF::predict_correct(double * ctrl, double dt, double* sensors, double* conf) {

  double t2 = util::now_t();
  int end = (nq > 26) ? 26 : nq;

  set_data(d, &(x_t));
  mju_copy(d->ctrl, ctrl, nu); // set controls for the center point

  get_data(d, &(x[0])); //x[0] = x_t;

  double t3 = util::now_t();

#ifdef SYMMETRIC_SQUARE_ROOT
  MatrixXd m_sqrt = ((L+lambda)*(P_t)).sqrt(); // symmetric; blows up
#else
  LLT<MatrixXd> chol((L+lambda)*(P_t));
  MatrixXd m_sqrt = chol.matrixL(); // chol
#endif
  // compilation time.... 

  double t4 = util::now_t();

  if (NUMBER_CHECK) {
    IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");
    std::cout << "p_t start:\n"<< P_t.block(0,0,end,end).format(CleanFmt) << std::endl;
    std::cout << "sqrt output:\n"<< m_sqrt.block(0,0,end,end).format(CleanFmt) << std::endl;
    //std::cout << "sqrt2 output:\n"<< sqrt2.format(CleanFmt) << std::endl;
    //for (int j=0; j<L; j++) {
    //  if (j >= (nq+nv))
    //    std::cout<<"skip p&v:\t"<<j<<" "<<m_sqrt.col(j).transpose().format(CleanFmt) << std::endl;
    //  else if (j >= nq ) 
    //    std::cout<<"skip pos:\t"<<j<<" "<<m_sqrt.col(j).transpose().format(CleanFmt) << std::endl;
    //  else
    //    std::cout<<"no skips:\t"<<j<<" "<<m_sqrt.col(j).transpose().format(CleanFmt) << std::endl;
    //}
  }

  if (conf) { // our sensors have confidence intervals
    // TODO more than just phasespace markers?
    if (ns > (40+6+12)) { // we have phasespace markers
      int ps_start = ns - 16;
      for (int j=0; j<16; j++) { // DIRTY HACKY HACK
        int idx = ps_start + j;
        // our conf threshold (basically not visible)
        if (conf[j] < 3.0) { PzAdd(idx, idx) = 1e+2; }
        else { PzAdd(idx, idx) = mrkr_conf; }
      }
    }
  }
  // Simulation options
  m->opt.timestep = dt;

  bool INV_CHECK = false;

  if (INV_CHECK) {
    printf("%d: ", 0);
    printf("   b-state: ");
    for (int i=0; i<nq; i++) printf("%1.4f ", d->qpos[i]);
    for (int i=0; i<nv; i++) printf("%1.4f ", d->qvel[i]);
    printf("\tb-qacc: ");
    for (int i=0; i<nv; i++) printf("%1.4f ", d->qacc[i]);
  }

  add_ctrl_noise(d->ctrl);
  m->opt.iterations = 50; 
  m->opt.tolerance = 1e-6; 
  mj_forward(m, d); // solve for center point accurately

  if (INV_CHECK) {
    printf("  \tinv: ");
    for (int i=0; i<nv; i++) printf("%1.4f ", d->qfrc_constraint[i]);
    printf("\ta-qacc: ");
    for (int i=0; i<nv; i++) printf("%1.4f ", d->qacc[i]);
    printf("\n");
  }

  double center_tol = constraint_violated(d);

  for (int i=1; i<N; i++) { W_theta[i] = sqrt(L+lambda); }

  // set tolerance to be low, run 50, 100 iterations for mujoco solver
  // copy qacc for sigma points with some higher tolerance
#pragma omp parallel
  {
    //double omp1 = util::now_t();
    int tid = omp_get_thread_num();
    int t = omp_get_num_threads();
    int s = tid * L / t;
    int e = (tid + 1 ) * L / t;
    if (tid == t-1) e = L;

    mjData *t_d = sigmas[tid];
    mjModel*t_m = models[tid];

    //printf("p thread: %d chunk: %d-%d \n", tid, s, e);
    for (int j=s; j<e; j++) {
      // step through all the perturbation cols and collect the data
      int i = j+1;
      double scale;

      ///////////////////////////////////////////////////// sigma point
      x[i+0] = x[0]+m_sqrt.col(i-1) + q_hat;

      t_d->time = d->time;
      set_data(t_d, &(x[i+0]));
      mju_copy(t_d->qacc, d->qacc, nv); // copy from center point

      if (!ctrl_state) mju_copy(t_d->ctrl, ctrl, nu); // set controls for this t
      add_ctrl_noise(t_d->ctrl);

      if (INV_CHECK) {
        printf("%d: ", i);
        printf("   b-state: ");
        for (int k=0; k<nq; k++) printf("%1.4f ", t_d->qpos[k]);
        for (int k=0; k<nv; k++) printf("%1.4f ", t_d->qvel[k]);
        printf("\tb-qacc: ");
        for (int k=0; k<nv; k++) printf("%1.4f ", t_d->qacc[k]);
      }

      fast_forward(t_m, t_d, j, 0);
      //mj_forward(t_m, t_d);

      if (INV_CHECK) {
        printf("\tinv: ");
        for (int k=0; k<nv; k++) printf("%1.4f ", t_d->qfrc_constraint[k]);
        printf("\ta-qacc: ");
        for (int k=0; k<nv; k++) printf("%1.4f ", t_d->qacc[k]);
      }

      if (tol > 0 ) {
        VectorXd first_sqrt = m_sqrt.col(i-1);
        scale = handle_constraint(t_d, d, center_tol+tol, &(x[i+0]), &(x[0]), &(first_sqrt));
        W_theta[i] *= scale;
      }
      else { scale = 1.0; }

      t_m->opt.timestep = m->opt.timestep + add_time_noise();
      mj_Euler(t_m, t_d);

      if (INV_CHECK) {
        printf(" s: %f ", scale);
        printf("\tinv: ");
        for (int k=0; k<nv; k++) printf("%1.4f ", t_d->qfrc_constraint[k]);
      }

      get_data(t_d, &(x[i+0]));

      mj_forward(t_m, t_d); // at new position; can't assume we didn't move
      //mj_sensor(t_m, t_d);

      if (INV_CHECK) {
        printf("\n");
      }


      add_snsr_noise(t_d->sensordata);
      mju_copy(&(gamma[i](0)), t_d->sensordata, ns);

      if (j < nq) { // only copy position perturbed
        //mju_copy(sigma_states[i+0]->qpos, t_d->qpos, nq);
      }

      ///////////////////////////////////////////////////// symmetric point
      x[i+L] = x[0]-m_sqrt.col(i-1) + q_hat;

      t_d->time = d->time;
      set_data(t_d, &(x[i+L]));
      if (!ctrl_state) mju_copy(t_d->ctrl, ctrl, nu); // set controls for this t
      add_ctrl_noise(t_d->ctrl);
      mju_copy(t_d->qacc, d->qacc, nv); // copy from center point

      if (INV_CHECK) {
        printf("%d: ", i+L);
        printf("   b-state: ");
        for (int k=0; k<nq; k++) printf("%1.4f ", t_d->qpos[k]);
        for (int k=0; k<nv; k++) printf("%1.4f ", t_d->qvel[k]);
        printf("\tb-qacc: ");
        for (int k=0; k<nv; k++) printf("%1.4f ", t_d->qacc[k]);
      }

      fast_forward(t_m, t_d, j, 0);
      //mj_forward(t_m, t_d);

      if (INV_CHECK) {
        printf("\tinv: ");
        for (int k=0; k<nv; k++) printf("%1.4f ", t_d->qfrc_constraint[k]);
        printf("\ta-qacc: ");
        for (int k=0; k<nv; k++) printf("%1.4f ", t_d->qacc[k]);
      }

      if (tol > 0 ) {
        VectorXd other_sqrt = -1*m_sqrt.col(i-1);
        scale = handle_constraint(t_d, d, center_tol+tol, &(x[i+L]), &(x[0]), &(other_sqrt));
        W_theta[i] *= scale;
      }
      else { scale = 1.0; }

      t_m->opt.timestep = m->opt.timestep + add_time_noise();
      mj_Euler(t_m, t_d);

      if (INV_CHECK) {
        printf(" s: %f ", scale);
        printf("\tinv: ");
        for (int k=0; k<nv; k++) printf("%1.4f ", t_d->qfrc_constraint[k]);
      }

      get_data(t_d, &(x[i+L]));

      mj_forward(t_m, t_d); // at new position; can't assume we didn't move
      //mj_sensor(t_m, t_d);

      if (INV_CHECK) {
        printf("\n");
      }

      add_snsr_noise(t_d->sensordata);
      mju_copy(&(gamma[i+L](0)), t_d->sensordata, ns);

      if (j < nq) { // only copy position perturbed states
        //mju_copy(sigma_states[i+nq]->qpos, t_d->qpos, nq);
      }

    }
    //double omp2 = util::now_t();
    //printf("p thread: %d chunk: %d-%d Time: %f\n", tid, s, e, omp2-omp1);
  }

  // step for the central point
  //mj_forward(m, d);
  mj_Euler(m, d); // step

  get_data(d, &(x[0]));
  get_data(d, &(x_minus)); // pre-correction

  mj_forward(m, d);
  //mj_sensor(m, d); // sensor values at new positions
  //gamma[0] = Map<VectorXd>(d->sensordata, ns);
  //add_snsr_noise(d->sensordata);
  mju_copy(&(gamma[0](0)), d->sensordata, ns);

  mju_copy(sigma_states[0]->qpos, d->qpos, nq);
  //copy_state(sigma_states[0], d); // for visualizations
  //mju_copy(sigma_states[0]->ctrl, d->ctrl, nu);

  double t5 = util::now_t();

  ////////////////// Weights scaled according to constraints
  //double S_theta = 0.0;
  //for (int i=1; i<N; i++) { S_theta += W_theta[i]; }
  //double sq = sqrt(L+lambda);
  //double a = (2.0*lambda - 1.0) / (2.0*(L+lambda)*(S_theta - (N*sq)));
  //double b = (1.0)/(2.0*(L+lambda)) - (2.0*lambda - 1.0)/(2.0*sq*(S_theta - (N*sq)));

  // traditional weighting  rescaled
  //W_s[0] = b;
  //W_c[0] = b;
  //for (int i=1; i<N; i++) { W_s[i] = a * W_theta[i] + b; W_c[i] = W_s[i]; }

  // even weighting
  //W_s[0] = 1.0/N;
  //W_c[0] = 1.0/N;
  //for (int i=1; i<N; i++) { W_s[i] = 1.0/N; W_c[i] = W_s[i]; }
  //double sum = 0.0;
  //printf("\nScaled Weights:\n");
  //for (int i=0; i<N; i++) { sum += W_s[i]; printf("%1.3f ", W_s[i]); }
  //printf("\nScaled Weights Sum: %f\n", sum);

  // aprior mean
  x_t.setZero();
  //for (int i=0; i<N; i++) { x_t += W_s[i]*x[i]; }
  for (int i=1; i<N; i++) { x_t += x[i]; }
  x_t = W_s[0]*x[0] + (W_s[1])*x_t;
  // TODO  can optimize more if all the weights are the same

  VectorXd z_k = s_hat; //VectorXd::Zero(ns);
  //for (int i=0; i<N; i++) { z_k += W_s[i]*gamma[i]; }
  for (int i=1; i<N; i++) { z_k += gamma[i]; }
  z_k = W_s[0]*gamma[0] + (W_s[1])*z_k;
  // TODO  can optimize more if all the weights are the same

  if (NUMBER_CHECK) {
    IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");
    printf("Correct: After Step time %f\n", sigma_states[0]->time);
    std::cout << "\nz_k hat:\n"<< (z_k).transpose().format(CleanFmt) << std::endl;
  }

  P_t.setZero();
  P_z.setZero();
  Pxz.setZero();
  //for (int i=0; i<N; i++) {
  //  VectorXd z(gamma[i] - z_k);
  //  VectorXd x_i(x[i] - x_t);

  //  P_t += W_c[i] * (x_i * x_i.transpose()); // aprior covarian
  //  P_z += W_c[i] * (z * z.transpose());
  //  Pxz += W_c[i] * (x_i * z.transpose());
  //}
  for (int i=1; i<N; i++) {
    VectorXd z(gamma[i] - z_k);
    VectorXd x_i(x[i] - x_t);

    P_t += (x_i * x_i.transpose());
    P_z += (z * z.transpose());
    Pxz += (x_i * z.transpose());
  }
  VectorXd x_i(x[0] - x_t);
  VectorXd z(gamma[0] - z_k);
  P_t = W_c[0]*(x_i * x_i.transpose()) + W_c[1]*P_t;
  P_z = W_c[0]*(z * z.transpose())     + W_c[1]*P_z + PzAdd;
  Pxz = W_c[0]*(x_i * z.transpose())   + W_c[1]*Pxz;

  if (NUMBER_CHECK) {
    IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");
    std::cout << "Prediction output:\nx_t\n"<< x_t.transpose().format(CleanFmt) << std::endl;
    std::cout << "Prediction output:\nP_t\n"<< P_t.block(0,0,end,end).format(CleanFmt) << std::endl;
  }

  //P_z = P_z + PzAdd;

  MatrixXd K = Pxz * P_z.inverse();

  // other method of combining estimates
  // TODO different method for X_1 by multiplying with kalman gain each
  //for (int i=0; i<N; i++) {
  //  x[i] = x[i] + (K*(s-gamma[i])); // other way
  //}
  //// sigma point
  //VectorXd x_t_aug = VectorXd::Zero(L);
  //for (int i=0; i<N; i++) { x_t += W_s[i]*x[i]; }
  //for (int i=1; i<N; i++) { x_t_aug += x[i]; }
  //x_t_aug = W_s[0]*x[0] + (W_s[1])*x_t_aug;

  //MatrixXd P_t_aug = MatrixXd::Zero(L,L);
  //for (int i=0; i<N; i++) {
  //  VectorXd x_i(x[i] - x_t);
  //  P_t_aug += W_c[i] * (x_i * x_i.transpose()); // aprior covarian
  //}

  VectorXd s = Map<VectorXd>(sensors, ns); // map our real z to vector
  x_t = x_t + (K*(s-z_k));
  P_t = P_t - (K * P_z * K.transpose());

  //P_t = P_t + PtAdd; 

  set_data(d, &(x_t)); // set corrected state into mujoco data struct
  mju_copy(d->sensordata, &(z_k(0)), ns); // copy estimated data for viewing 
  //mj_forward(m, d);

  //std::cout << "\ncorrect x_t:\n"<< x_t.transpose().format(CleanFmt) << std::endl;
  //std::cout << "\nother   x_t:\n"<< x_t_aug.transpose().format(CleanFmt) << std::endl;

  // MAKING A PREVIOUS DATA
  //set_data(prev_d, &(x_t)); // the 'previous' estimate's mjData
  //mju_copy(prev_d->ctrl, d->ctrl, nu); // set controls for the center point
  //mju_copy(prev_d->qacc, d->qacc, nv); // set controls for the center point
  //mj_forward(m, prev_d);

  double t6 = util::now_t();

  printf("\ncombo init %f, sqrt %f, mjsteps %f, merge %f\n",
      t3-t2, t4-t3, t5-t4, t6-t5);

  /* Entropy Testing?
  */
  //IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");

  //std::cout << "\ndelta:\n"<< (s-z_k).transpose().format(CleanFmt) << std::endl;
  //std::cout << "\ncorrection:\n"<< (K*(s-z_k)).transpose().format(CleanFmt) << std::endl;
  //std::cout << "\np_t :\n"<< P_t.diagonal().transpose().format(CleanFmt) << std::endl;
  //std::cout << "\np_z :\n"<< P_z.diagonal().transpose().format(CleanFmt) << std::endl;

  // Sage-Husa Adaptive Estimate
  //double dk = 0.95;
  //MatrixXd F = x_t - q_hat;
  //q_hat = (1-dk)*q_hat + dk*(x_t - x_minus);
  //dk = 0.5;
  //s_hat = (1-dk)*s_hat + dk*(s-z_k);
  //s_hat.setZero();
  //q_hat.setZero();

  // x_t = F*x_minus + q_hat;
  //Q = (1-dk) * Q + dk*(K*V*V.transpose()*K.transpose() + P_t );

  //std::cout << "\nq_hat:\n"<< q_hat.transpose().format(CleanFmt) << std::endl;

  //double a2 = abs(P_t.determinant());
  //double b2 = pow(2*3.14159265*exp(1), (double)L);
  //double entropy = 0.5*log(b2 * a2);
  //printf("\n\ne %f\ta: %1.32f\tb: %f\tEntropy: %f\n\n", a2*b2, a2, b2, entropy);

  if (NUMBER_CHECK) {
    IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");
    std::cout << "Kalman Gain:\n"<< K.block(0,0,end,end).format(CleanFmt) << std::endl;
    std::cout << "\nPz  :\n"<< P_z.block(0,0,end,end).format(CleanFmt) << std::endl;
    std::cout << "\nPz^-1:\n"<< P_z.inverse().block(0,0,end,end).format(CleanFmt) << std::endl;
    std::cout << "\nPxz :\n"<< Pxz.block(0,0,end,end).format(CleanFmt) << std::endl;
    std::cout << "\ncorrect p_t :\n"<< P_t.block(0,0,end,end).format(CleanFmt) << std::endl;
    std::cout << "\ncorrect x_t:\n"<< x_t.transpose().format(CleanFmt) << std::endl;
    std::cout << "\nraw sensors:\n"<< (s).transpose().format(CleanFmt) << std::endl;
    std::cout << "\nz_k hat:\n"<< (z_k).transpose().format(CleanFmt) << std::endl;
    std::cout << "\ndelta:\n"<< (s-z_k).transpose().format(CleanFmt) << std::endl;
    std::cout << "Predict and Correct COMBO\n";
  }
}

void UKF::predict(double * ctrl, double dt) {

  IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");

  double t2 = util::now_t();

  m->opt.timestep = dt; // smoother way of doing this?


  double t3 = util::now_t();

  //P_t = MatrixXd::Identity(L,L) * 1e-3;
  LLT<MatrixXd> chol((L+lambda)*(P_t));
  MatrixXd sqrt = chol.matrixL(); // chol

  double t4 = util::now_t();

  if (NUMBER_CHECK) {
    std::cout<<"\n\nPrevious Prediction x_t:\n"<<x_t.transpose().format(CleanFmt)<<"\n";
    std::cout<<"\n\nPrediction P_T:\n"<<P_t.format(CleanFmt)<<"\n\n";
    printf("Before Step:\n");
    for (int i=1; i<=L; i++) {
      printf("%d ", i);
      std::cout<<(x[i+0]+sqrt.col(i-1)).transpose().format(CleanFmt)<<"\n";
      std::cout<<(x[i+L]+sqrt.col(i-1)).transpose().format(CleanFmt);
      printf("\n");
    }
  }


#pragma omp parallel
  {
    volatile int tid = omp_get_thread_num();
    int t = omp_get_num_threads();
    int chunk = (L + t-1) / t;
    int s = tid * chunk;
    int e = mjMIN(s+chunk, L);

    //printf("p thread: %d chunk: %d-%d \n", tid, s, e);
    for (int j=s; j<e; j++) {
      // step through all the perturbation cols and collect the data
      int i = j+1;

      x[i+0] = x[0]+sqrt.col(i-1);
      x[i+L] = x[0]-sqrt.col(i-1);

      mju_copy(sigmas[tid]->qpos,   &x[i](0), nq);
      mju_copy(sigmas[tid]->qvel,   &x[i](nq), nv);
      //mju_copy(sigmas[tid]->qacc,   d->qacc, nv);
      if (ctrl_state) mju_copy(sigmas[tid]->ctrl,   &x[i](nq+nv), nu); // set controls for this t
      else mju_copy(sigmas[tid]->ctrl,   ctrl, nu); // set controls for this t

      //mj_step(m, sigmas[tid]);
      mj_forward(m, sigmas[tid]);
      mj_Euler(m, sigmas[tid]);

      sigmas[tid]->time = d->time;
      mju_copy(&x[i](0),  sigmas[tid]->qpos, nq);
      mju_copy(&x[i](nq), sigmas[tid]->qvel, nv);
      if (ctrl_state) mju_copy(&x[i](nq+nv), sigmas[tid]->ctrl, nu);

      if (NUMBER_CHECK) mju_copy(sigma_states[i]->qpos, sigmas[tid]->qpos, nq);

      mju_copy(sigmas[tid]->qpos,   &x[i+L](0), nq);
      mju_copy(sigmas[tid]->qvel,   &x[i+L](nq), nv);
      //mju_copy(sigmas[tid]->qacc,   d->qacc, nv);
      if (ctrl_state) mju_copy(sigmas[tid]->ctrl,   &x[i+L](nq+nv), nu); // set controls for this t
      else mju_copy(sigmas[tid]->ctrl,   ctrl, nu); // set controls for this t

      //mj_step(m, sigmas[tid]);
      mj_forward(m, sigmas[tid]);
      mj_Euler(m, sigmas[tid]);

      mju_copy(&x[i+L](0),  sigmas[tid]->qpos, nq);
      mju_copy(&x[i+L](nq), sigmas[tid]->qvel, nv);
      if (ctrl_state) mju_copy(&x[i+L](nq+nv), sigmas[tid]->ctrl, nu);

      if (NUMBER_CHECK) mju_copy(sigma_states[i+L]->qpos, sigmas[tid]->qpos, nq);
    }
  }

  // step for the central point
  mj_step(m, d);
  mju_copy(&x[0](0), d->qpos, nq);
  mju_copy(&x[0](nq), d->qvel, nv);
  if (ctrl_state) mju_copy(&x[0](nq+nv), d->ctrl, nu);

  if (NUMBER_CHECK) mju_copy(sigma_states[0]->qpos, d->qpos, nq);

  if (NUMBER_CHECK) {
    printf("Predict t-0 = %f seconds\n", sigma_states[0]->time);
    printf("After Step:\n");
    for (int i=0; i<N; i++) {
      printf("%d ", i);
      std::cout<<x[i].transpose().format(CleanFmt);
      printf("\t:: ");
      for (int j=0; j<nu; j++)
        printf("%1.5f ", sigma_states[i]->ctrl[j]);
      printf("\t:: ");
      for (int j=0; j<ns; j++)
        printf("%f ", sigma_states[i]->sensordata[j]);

      printf("\n");
    }
  }

  double t5 = util::now_t();

  x_t.setZero();
  for (int i=0; i<N; i++) {
    if (NUMBER_CHECK) {
      std::cout<<W_s[i]<<"  "<<(W_s[i]*x[i]).transpose().format(CleanFmt)<<"\n";
    }
    x_t += W_s[i]*x[i];
  }
  //x_t = (W_s[0] * x[0]) + ((N-1)*W_s[1] * x_t);

  P_t.setZero();
  for (int i=0; i<N; i++) {
    VectorXd x_i(x[i] - x_t);
    P_t += W_c[i] * (x_i * x_i.transpose());
  }

  for (int i=0; i<L; i++) {
    if (P_t.row(i).isZero(1e-9)) {
      P_t.row(i).setZero();
      P_t.col(i).setZero();
      P_t(i,i) = 1.0;
    }
  }

  //mju_copy(d->qpos, &(x_t(0)), nq); // center point
  //mju_copy(d->qvel, &(x_t(nq)), nv);
  mju_copy(d->qpos, &(x_t(0)), nq);
  mju_copy(d->qvel, &(x_t(nq)), nv);

  double t6 = util::now_t();

  printf("\npredict init %f, sqrt %f, mjsteps %f, merge %f\n",
      t3-t2, t4-t3, t5-t4, t6-t5);

  if (NUMBER_CHECK) std::cout << "Prediction output:\nx_t\n"<< x_t.transpose().format(CleanFmt) << std::endl;
  if (NUMBER_CHECK) std::cout << "Prediction output:\nP_t\n"<< P_t.format(CleanFmt) << std::endl;

}

void UKF::correct(double* sensors) {

  IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
  double t2 = util::now_t();

  mju_copy(d->qpos, &(x_t(0)), nq);
  mju_copy(d->qvel, &(x_t(nq)), nv);
  if (ctrl_state) mju_copy(d->ctrl, &(x_t(nq+nv)), nu);

  double t3 = util::now_t();

  //P_t = MatrixXd::Identity(L,L) * 1e-3;
  LLT<MatrixXd> chol((L+lambda)*(P_t));
  MatrixXd sqrt = chol.matrixL(); // chol

  double t4 = util::now_t();

  // TODO set solver iterations and tolerance
  mju_copy(sigma_states[0]->qpos, d->qpos, nq);
  mju_copy(sigma_states[0]->qvel, d->qvel, nv);
  mju_copy(sigma_states[0]->ctrl, d->ctrl, nu);

  mj_forward(m, d);
  //mj_sensor(m, d);
  //mju_copy(&x[0](0), d->qpos, nq);
  //mju_copy(&x[0](nq), d->qvel, nv);

  x[0] = x_t;
  gamma[0] = Map<VectorXd>(d->sensordata, ns);

  if (NUMBER_CHECK) std::cout << "Correction sqrt:\n"<< sqrt.format(CleanFmt) << std::endl;
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int t = omp_get_num_threads();
    int chunk = (L + t-1) / t;
    int s = tid * chunk;
    int e = mjMIN(s+chunk, L);

    //printf("c thread: %d chunk: %d-%d \n", tid, s, e);
    for (int j=s; j<e; j++) {
      int i = j+1;

      x[i+0] = x[0]+sqrt.col(i-1);
      x[i+L] = x[0]-sqrt.col(i-1);

      mju_copy(sigmas[tid]->qpos, &x[i](0), nq);
      mju_copy(sigmas[tid]->qvel, &x[i](nq), nv);
      //mju_copy(sigmas[tid]->qacc, d->qacc, nv);
      if (ctrl_state) mju_copy(sigmas[tid]->ctrl, &x[i](nq+nv), nu); // set controls for this t

      // TODO set solver iterations and tolerance
      mj_forward(m, sigmas[tid]);
      //mj_sensor(m, sigmas[tid]);

      gamma[i] = Map<VectorXd>(sigmas[tid]->sensordata, ns);

      mju_copy(sigma_states[i+0]->qpos, sigmas[tid]->qpos, nq);
      mju_copy(sigma_states[i+0]->qvel, sigmas[tid]->qvel, nv);
      if (ctrl_state) mju_copy(sigma_states[i+0]->ctrl, sigmas[tid]->ctrl, nu);

      //sigmas[tid]->time = d->time;
      mju_copy(sigmas[tid]->qpos, &x[i+L](0), nq);
      mju_copy(sigmas[tid]->qvel, &x[i+L](nq), nv);
      //mju_copy(sigmas[tid]->qacc, d->qacc, nv);
      //if (!ctrl_state) mju_copy(sigmas[tid]->ctrl,   ctrl, nu); // set controls for this t
      if (ctrl_state) mju_copy(sigmas[tid]->ctrl, &x[i+L](nq+nv), nu); // set controls for this t

      mj_forward(m, sigmas[tid]);
      //mj_sensor(m, sigmas[tid]);

      gamma[i+L] = Map<VectorXd>(sigmas[tid]->sensordata, ns);

      mju_copy(sigma_states[i+L]->qpos, sigmas[tid]->qpos, nq);
      mju_copy(sigma_states[i+L]->qvel, sigmas[tid]->qvel, nv);
      //mju_copy(sigma_states[i+L]->ctrl, sigmas[tid]->ctrl, nu);
    }
  }

  double t5 = util::now_t();

  VectorXd z_k = VectorXd::Zero(ns);
  for (int i=0; i<N; i++) {
    z_k += W_s[i]*gamma[i];
  }

  if (NUMBER_CHECK) {
    printf("Correct: After Step time %f\n", sigma_states[0]->time);
    for (int i=0; i<N; i++) {
      printf("%d\t", i);
      for (int j=0; j<nq; j++)
        printf("%1.5f ", sigma_states[i]->qpos[j]);
      for (int j=0; j<nv; j++)
        printf("%1.5f ", sigma_states[i]->qvel[j]);
      printf("\t:: ");
      for (int j=0; j<nu; j++)
        printf("%1.5f ", sigma_states[i]->ctrl[j]);
      printf("\t:: ");
      for (int j=0; j<ns; j++)
        printf("%f ", sigma_states[i]->sensordata[j]);
      printf("\n");
    }
    std::cout << "\nz_k hat:\n"<< (z_k).transpose().format(CleanFmt) << std::endl;
  }

  P_z.setZero();
  //P_z.setIdentity();
  Pxz.setZero();
  for (int i=0; i<N; i++) {
    VectorXd z(gamma[i] - z_k);
    VectorXd x_i(x[i] - x_t);

    P_z += W_c[i] * (z * z.transpose());
    Pxz += W_c[i] * (x_i * z.transpose());
  }

  // TODO make the identity addition a parameter
  P_z = P_z + (MatrixXd::Identity(ns, ns)*diag);

  //P_z.setIdentity();
  //P_t = MatrixXd::Identity(L,L) * 1e-3;
  MatrixXd K = Pxz * P_z.inverse();

  VectorXd s = Map<VectorXd>(sensors, ns); // map our real z to vector
  //std::cout << "\nbefore x_t:\n"<< x_t.transpose().format(CleanFmt) << std::endl;
  x_t = x_t + (K*(s-z_k));
  P_t = P_t - (K * P_z * K.transpose());

  //std::cout << "\nafter x_t:\n"<< x_t.transpose().format(CleanFmt) << std::endl;

  mju_copy(d->qpos, &(x_t(0)), nq); // center point
  mju_copy(d->qvel, &(x_t(nq)), nv);

  double t6 = util::now_t();

  printf("\n\ncorrect copy %f, sqrt %f, mjsteps %f, merge %f\n",
      t3-t2, t4-t3, t5-t4, t6-t5);

  if (NUMBER_CHECK) {
    std::cout << "Kalman Gain:\n"<< K.format(CleanFmt) << std::endl;
    std::cout << "\nPz  :\n"<< P_z.format(CleanFmt) << std::endl;
    std::cout << "\nPz^-1:\n"<< P_z.inverse().format(CleanFmt) << std::endl;
    std::cout << "\nPxz :\n"<< Pxz.format(CleanFmt) << std::endl;
    std::cout << "\ncorrect p_t :\n"<< P_t.format(CleanFmt) << std::endl;
    std::cout << "\ncorrect x_t:\n"<< x_t.transpose().format(CleanFmt) << std::endl;
    std::cout << "\nraw sensors:\n"<< (s).transpose().format(CleanFmt) << std::endl;
    std::cout << "\nz_k hat:\n"<< (z_k).transpose().format(CleanFmt) << std::endl;
    std::cout << "\ndelta:\n"<< (s-z_k).transpose().format(CleanFmt) << std::endl;
    std::cout << "NEW CORRECTION STEPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP\n";
  }

  /* Entropy Testing?
     std::cout << "\ndelta:\n"<< (s-z_k).transpose().format(CleanFmt) << std::endl;
     std::cout << "\ncorrection:\n"<< (K*(s-z_k)).transpose().format(CleanFmt) << std::endl;
     std::cout << "\np_t :\n"<< P_t.diagonal().transpose().format(CleanFmt) << std::endl;
     double a = abs(P_t.determinant());
     double b = pow(2*3.14159265*exp(1), (double)L);
     double entropy = 0.5*log(b * a);
     printf("\n\ne %f\ta: %1.32f\tb: %f\tEntropy: %f\n\n", a*b, a, b, entropy);
     */
}

mjData* UKF::get_stddev() {
  VectorXd var = P_t.diagonal();
  if (NUMBER_CHECK) {
    std::cout<<"P_T diag:\n";
    std::cout<< var.transpose() << std::endl;
    std::cout<<"\nP_z diag:\n";
    std::cout<< P_z.diagonal().transpose() << std::endl;
  }
  mju_copy(stddev->qpos, &(var(0)), nq);
  mju_copy(stddev->qvel, &(var(nq)), nv);
  if (ctrl_state) mju_copy(stddev->ctrl, &(var(nq+nv)), nu);
  else mju_copy(stddev->ctrl, d->ctrl, nu);

  var = P_z.diagonal();
  mju_copy(stddev->sensordata, &(var(0)), ns);

  return stddev;
}

