#include "mujoco.h"
#include <string.h>
#include <iostream>
#include <random>
#include <functional>

#ifndef __APPLE__
#include <omp.h>
#endif
#include <math.h>
#include <cmath>

#ifdef USE_EIGEN_MKL
#define EIGEN_USE_MKL_ALL
#endif

#define EIGEN_DONT_PARALLELIZE
#define EIGEN_DEFAULT_TO_ROW_MAJOR

#include <vector>
#include <Eigen/Dense>
//#define SYMMETRIC_SQUARE_ROOT
#ifdef SYMMETRIC_SQUARE_ROOT
#include <unsupported/Eigen/MatrixFunctions>
#endif

//#include <boost/random.hpp>
//#include <boost/random/normal_distribution.hpp>

using namespace Eigen;

#ifdef __APPLE__
double omp_get_wtime() {
  std::chrono::time_point<std::chrono::high_resolution_clock> t
    = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> d=t.time_since_epoch();
  return d.count() / 1000.0 ; // returns milliseconds
}
int omp_get_thread_num() { return 0; }
int omp_get_num_threads() { return 1; }
#endif

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

class Estimator {
  public:
    Estimator(mjModel *m, mjData * d) {
      this->m = mj_copyModel(NULL, m); // make local copy of model and data
      this->d = mj_makeData(this->m);
      mj_copyData(this->d, m, d); // get current state

      this->nq = this->m->nq;
      this->nv = this->m->nv;
      this->nu = this->m->nu;
      this->ns = this->m->nsensordata;
    }

    virtual ~Estimator() {
      mj_deleteData(d);
      mj_deleteModel(m);
    }

    //virtual void init();
    virtual void predict(double * ctrl, double dt) {};
    virtual void correct(double* sensors) {};
    virtual void predict_correct(double * ctrl, double dt, double* sensors, double* conf) {};
    virtual void predict_correct_p1(double * ctrl, double dt, double* sensors, double* conf) {};
    virtual void predict_correct_p2(double * ctrl, double dt, double* sensors, double* conf) {};
    //virtual void predict_correct_2stage(double * ctrl, double dt, double* sensors, double* conf) {};
    virtual mjData* get_state() {return this->d; };
    virtual mjData* get_stddev() {return this->d; };
    virtual std::vector<mjData*> get_sigmas() {return sigma_states; };

    std::vector<mjData *> sigma_states;
    mjModel* m;
    mjData* d;
    int nq;
    int nv;
    int nu;
    int ns;
};

class UKF : public Estimator {
  public:
    UKF(mjModel *m, mjData * d, double * snsr_weights, double *P_covar,
        double _alpha, double _beta, double _kappa,
        double _diag, double _Ws0, double _noise,
        double _tol,
        bool debug = false, int threads = 1) : Estimator(m, d) {

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
      //sigma_states = new mjData*[N];
      //sigmas = new mjData*[32];
      //sigma_states.resize(N);
      sigma_states.resize(nq+1); // only useful to visualize qpos perturbed

      my_threads = threads;
      sigmas.resize(threads);
      models.resize(threads);

      omp_set_num_threads(threads);

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
      x = new VectorXd[N];
      gamma = new VectorXd[N];
      for (int i=0; i<nq+1; i++) {
        if (i==0) {
          sigma_states[i] = this->d;
        }
        else {
          sigma_states[i] = mj_makeData(this->m);
          mj_copyData(sigma_states[i], this->m, this->d);
        }
        mj_forward(m, sigma_states[i]);
      }

      for (int i=0; i<N; i++) {
        if (i==0) {
          //sigma_states[i] = this->d;
          //W_s[i] = lambda / ((double) L + lambda);
          //W_c[i] = W_s[i] + (1-(alpha*alpha)+beta);
          W_s[i] = 1.0/N;
          W_c[i] = 1.0/N;
          /*
             if (Ws0 < 0) {
             W_s[i] = 1.0 / N;
             }
             else{
             W_s[i] = Ws0;
             }
             */
          //W_c[i] = W_s[i];
        }
        else {
          //sigma_states[i] = mj_makeData(this->m);
          //mj_copyData(sigma_states[i], this->m, this->d);

          // Traditional weighting
          //W_s[i] = 1.0 / (2.0 * ((double)L + lambda));
          //W_c[i] = W_s[i];

          // even weighting
          W_s[i] = 1.0/N;
          W_c[i] = 1.0/N;

          /*
             if (Ws0 < 0) {
             W_s[i] = 1.0 / N;
             }
             else{
             W_s[i] = (1.0 - Ws0) / ((double) 2 * L);
             }
             */
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

          P_t(i,i) = P_covar[i];
          P_t(i+nq,i+nq) = P_covar[i];
        }
      }
      std::cout<<"Ptadd:\n"<<PtAdd.diagonal().transpose()<<std::endl;


      qhat = VectorXd::Zero(L);
      Q = PtAdd;

      shat = VectorXd::Zero(ns);

      PzAdd = MatrixXd::Identity(ns, ns);
      int my_sensordata=0;
      double default_snsr[15] = {1e-4,1e-7,1e-4,1e-6,1e-7,1e-5,1e-4,1e-5,1e-4,1e-5,1e-3,1e-7,1e-4,1e-5,1e-2};
      snsr_ptr = new double[15];
      if (m->nnumericdata >= 15 && snsr_weights) {
        for (int i=0; i<15; i++)
          snsr_ptr[i] = snsr_weights[i];
      } else {
        for (int i=0; i<15; i++)
          snsr_ptr[i] = default_snsr[i];
      }
      printf("Sensor Covariance Matrix Diagonal:\n");
      printf("    Accelerometer: %e\n", snsr_ptr[0]);
      printf("    Gyro         : %e\n", snsr_ptr[1]);
      printf("    Force        : %e\n", snsr_ptr[2]);
      printf("    Torque       : %e\n", snsr_ptr[3]);
      printf("    JointPos     : %e\n", snsr_ptr[4]);
      printf("    JointVel     : %e\n", snsr_ptr[5]);
      printf("    TendonPos    : %e\n", snsr_ptr[6]);
      printf("    TendonVel    : %e\n", snsr_ptr[7]);
      printf("    ActuatorPos  : %e\n", snsr_ptr[8]);
      printf("    ActuatorVel  : %e\n", snsr_ptr[9]);
      printf("    ActuatorFrc  : %e\n", snsr_ptr[10]);
      printf("    SitePos      : %e\n", snsr_ptr[11]);
      printf("    SiteQuat     : %e\n", snsr_ptr[12]);
      printf("    Magnetometer : %e\n", snsr_ptr[13]);
      printf("    Default      : %e\n", snsr_ptr[14]);
      for (int i = 0; i < m->nsensor; i++) {
        int type = m->sensor_type[i];
        double var = 0;
        switch (type) {
          case 1:  var = snsr_ptr[0]; break;   //Accelerometer
          case 2:  var = snsr_ptr[1]; break;   //Gyro
          case 3:  var = snsr_ptr[2]; break;   //Force
          case 4:  var = snsr_ptr[3]; break;   //Torque
          case 5:  var = snsr_ptr[4]; break;   //JointPos
          case 6:  var = snsr_ptr[5]; break;   //JointVel
          case 7:  var = snsr_ptr[6]; break;   //TendonPos
          case 8:  var = snsr_ptr[7]; break;   //TendonVel
          case 9:  var = snsr_ptr[8]; break;   //ActuatorPos
          case 10: var = snsr_ptr[9]; break;   //ActuatorVel
          case 11: var = snsr_ptr[10]; break;  //ActuatorFrc
          case 12: var = snsr_ptr[11]; break;  //SitePos
          case 13: var = snsr_ptr[12]; break;  //SiteQuat
          case 14: var = snsr_ptr[13]; break;  //Magnetometer (WTF?)
          default: var = snsr_ptr[14]; break;
        }
        for (int j=my_sensordata; j<(m->sensor_dim[i]+my_sensordata); j++) {
          // different sensors have different number of fields
          PzAdd(j, j) = var;
        }
        my_sensordata += m->sensor_dim[i];
      }
      mrkr_conf = snsr_ptr[11];
      printf("Filled PzAdd: %d %d\n", my_sensordata, ns);

      rd_vec = new std::normal_distribution<>[ns]; 
      for (int i=0; i<ns; i++) {
        rd_vec[i] = std::normal_distribution<>( 0, sqrt(PzAdd(i,i)) );
      }

      if (diag > 0) {
        ct_vec = new std::normal_distribution<>[nu]; 
        for (int i=0; i<nu; i++) {
          ct_vec[i] = std::normal_distribution<>( 0, sqrt(diag) );
        }
      }

      m_noise = get_numeric_field(m, "mass_var", NULL);
      t_noise = get_numeric_field(m, "time_var", NULL);
      if (m_noise) printf("Total Mass Var: %f\n", *m_noise);
      if (t_noise) printf("Time Var: %f\n", *t_noise);
    };

    ~UKF() {
      delete[] x;
      delete[] W_s;
      delete[] W_theta;
      delete[] W_c;
      //delete[] p_state;
      delete[] gamma;
      delete[] snsr_ptr;
      //delete[] W_s;
      //delete[] W_c;

      for (int i=0; i<my_threads; i++) {
        mj_deleteData(sigmas[i]);
        mj_deleteModel(models[i]);
      }
      for (int i=1; i<nq; i++) {
        mj_deleteData(sigma_states[i]);
      }

      delete[] rd_vec;
      delete[] ct_vec;
    };

    double add_time_noise() {
      double dt = 0;
      if (t_noise) {
        static std::mt19937 t_rng(50505);
        static std::normal_distribution<double> nd(0, *t_noise);
        dt = nd(t_rng);
      }
      return dt;
    }

    double add_model_noise(mjModel* t_m) {
      if (m_noise) {
        static std::mt19937 m_rng(12345);
        static std::normal_distribution<double> mass_r(0, *m_noise);
        // TOTAL MASS
        //double newmass = mj_getTotalmass(m) + mass_r(m_rng);
        //mj_setTotalmass(t_m, newmass);
        for (int i=0; i<t_m->nbody; i++) {
          //t_m->body_mass[i] = m->body_mass[i] + mass_r(m_rng)/(double)m->nbody;
          t_m->body_mass[i] = m->body_mass[i] + mass_r(m_rng);
        }
      }
    }

    void add_ctrl_noise(double * noise) {
      static std::mt19937 c_rng(505050);
      //static std::normal_distribution<double> nd(0, _sigma);
      if (diag > 0) {
        for (int i=0; i<nu; i++) {
          double r = ct_vec[i](c_rng);
          noise[i] += r;
        }
      }
    }

    void add_snsr_noise(double * noise) {
      static std::mt19937 s_rng(494949);
      // make a vector ns in length with appropriate sigmas / variances

      for (int i=0; i<(ns-16*3); i++) {
        double r = rd_vec[i](s_rng);
        //printf("%1.2f ", r);
        noise[i] += r;
      }
    }

    void set_data(mjData* data, VectorXd *x) {
      mju_copy(data->qpos,   &x[0](0), nq);
      mju_copy(data->qvel,   &x[0](nq), nv);
      if (ctrl_state) mju_copy(data->ctrl,   &x[0](nq+nv), nu); // set controls for this t
    }

    void get_state(mjData* data, VectorXd *x) {
      mju_copy(&x[0](0),  data->qpos, nq);
      mju_copy(&x[0](nq), data->qvel, nv);
      if (ctrl_state) mju_copy(&x[0](nq+nv), data->ctrl, nu);
    }

    void copy_state(mjData * dst, mjData * src) {
      mju_copy(dst->qpos, src->qpos, nq);
      mju_copy(dst->qvel, src->qvel, nv);
      if (ctrl_state) mju_copy(dst->ctrl, src->ctrl, nu);
    }

    void fast_forward(mjModel *t_m, mjData * t_d, int index) {
      // NOTE: the mj_forwardSkip doesn't seem to be thread safe
#if 1
      if (ctrl_state && index >= (nq+nv)) mj_forwardSkip(t_m, t_d, 2); // just ctrl calcs
      else if (index >= nq) mj_forwardSkip(t_m, t_d, 1); // velocity cals
      else mj_forwardSkip(t_m, t_d, 0); // all calculations
#else
      mj_forward(m, t_d); // all calculations
#endif
    }

    double constraint_scale(mjData * t_d, double tol) {
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

    double constraint_violated(mjData* t_d) {
      VectorXd constraint = Map<VectorXd>(t_d->qfrc_constraint, nv);
      //VectorXd constraint = Map<VectorXd>(t_d->qfrc_inverse, nv);
      return constraint.norm();
    }

    double handle_constraint(mjData* t_d, mjData* d, double tol,
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

    void predict_correct_p1(double * ctrl, double dt, double* sensors, double* conf = 0) {

      //double t2 = omp_get_wtime()*1000.0;
      int end = (nq > 26) ? 26 : nq;

      //set_data(d, &(x_t)); // did this in _p2 step
      mju_copy(d->ctrl, ctrl, nu); // set controls for the center point

      get_state(d, &(x[0])); //x[0] = x_t;

      //double t3 = omp_get_wtime()*1000.0;

#ifdef SYMMETRIC_SQUARE_ROOT
      MatrixXd m_sqrt = ((L+lambda)*(P_t)).sqrt(); // symmetric; blows up
      // compilation time.... 
#else
      LLT<MatrixXd> chol((L+lambda)*(P_t));
      MatrixXd m_sqrt = chol.matrixL(); // chol
#endif

      //double t4 = omp_get_wtime()*1000.0;
      if (conf) { // our sensors have confidence intervals
        // TODO more than just phasespace markers?
        if (ns > (40+6+12)) { // we have phasespace markers
          int ps_start = ns - 16*3;
          //printf("ps %d ns %d\n", ps_start, ns);
          for (int j=0; j<16; j++) { // DIRTY HACKY HACK
            for (int i=0; i<3; i++) {
              int idx = ps_start + j*3 + i;
              // our conf threshold (basically not visible)
              //if (conf[j] < 0.0) { PzAdd(idx, idx) = 1e+100; }
              if (j != 2) { PzAdd(idx, idx) = 1e+100; }
              else { PzAdd(idx, idx) = mrkr_conf; }
            }
          }
          //IOFormat CleanFmt(2, 0, ", ", "\n", "[", "]");
          //std::cout << "pzadd 1st block:\n"<< PzAdd.block(0, 0, ps_start, ps_start).format(CleanFmt) << std::endl;
          //std::cout << "pzadd output:\n"<< PzAdd.block(ps_start, ps_start, ns-ps_start, ns-ps_start).format(CleanFmt) << std::endl;
        }
      }
      // Simulation options
      m->opt.timestep = dt; // input to this function

      add_ctrl_noise(d->ctrl);
      m->opt.iterations = 100; 
      m->opt.tolerance = 1e-6; 
      //mj_forward(m, d); // solve for center point accurately

      //double center_tol = constraint_violated(d);

      //for (int i=1; i<N; i++) { W_theta[i] = sqrt(L+lambda); }

      // set tolerance to be low, run 50, 100 iterations for mujoco solver
      // copy qacc for sigma points with some higher tolerance
#pragma omp parallel
      {
        //double omp1 = omp_get_wtime()*1000.0;
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
          x[i+0] = x[0]+m_sqrt.col(i-1) + qhat;

          t_d->time = d->time;

          set_data(t_d, &(x[i+0]));
          mju_copy(t_d->qacc, d->qacc, nv); // copy from center point

          if (!ctrl_state) mju_copy(t_d->ctrl, ctrl, nu); // set controls for this t
          add_ctrl_noise(t_d->ctrl);

          add_model_noise(t_m);
          fast_forward(t_m, t_d, j);
          //mj_forward(t_m, t_d);

          //if (tol > 0 ) {
          //  VectorXd first_sqrt = m_sqrt.col(i-1);
          //  scale = handle_constraint(t_d, d, center_tol+tol, &(x[i+0]), &(x[0]), &(first_sqrt));
          //  W_theta[i] *= scale;
          //}
          //else { scale = 1.0; }

          t_m->opt.timestep = m->opt.timestep + add_time_noise();
          mj_Euler(t_m, t_d);

          get_state(t_d, &(x[i+0]));

          mj_forward(t_m, t_d); // at new position; can't assume we didn't move
          mj_sensor(t_m, t_d);


          add_snsr_noise(t_d->sensordata);
          mju_copy(&(gamma[i](0)), t_d->sensordata, ns);

          if (j < nq) { // only copy position perturbed
            mju_copy(sigma_states[i+0]->qpos, t_d->qpos, nq);
          }

          ///////////////////////////////////////////////////// symmetric point
          x[i+L] = x[0]-m_sqrt.col(i-1) + qhat;

          t_d->time = d->time;
          set_data(t_d, &(x[i+L]));
          if (!ctrl_state) mju_copy(t_d->ctrl, ctrl, nu); // set controls for this t
          add_ctrl_noise(t_d->ctrl);
          mju_copy(t_d->qacc, d->qacc, nv); // copy from center point

          add_model_noise(t_m);
          fast_forward(t_m, t_d, j);
          //mj_forward(t_m, t_d);

          //if (tol > 0 ) {
          //  VectorXd other_sqrt = -1*m_sqrt.col(i-1);
          //  scale = handle_constraint(t_d, d, center_tol+tol, &(x[i+L]), &(x[0]), &(other_sqrt));
          //  W_theta[i] *= scale;
          //}
          //else { scale = 1.0; }

          t_m->opt.timestep = m->opt.timestep + add_time_noise();
          mj_Euler(t_m, t_d);

          get_state(t_d, &(x[i+L]));

          mj_forward(t_m, t_d); // at new position; can't assume we didn't move
          mj_sensor(t_m, t_d);

          add_snsr_noise(t_d->sensordata);
          mju_copy(&(gamma[i+L](0)), t_d->sensordata, ns);

          //if (j < nq) { // only copy position perturbed states
          //  //mju_copy(sigma_states[i+nq]->qpos, t_d->qpos, nq);
          //}

        }
        //double omp2 = omp_get_wtime()*1000.0;
        //printf("p thread: %d chunk: %d-%d Time: %f\n", tid, s, e, omp2-omp1);
      }

      // step for the central point
      //mj_forward(m, d);
      mj_Euler(m, d); // step

      get_state(d, &(x[0]));
      get_state(d, &(x_minus)); // pre-correction

      mj_forward(m, d);
      mj_sensor(m, d); // sensor values at new positions
      //gamma[0] = Map<VectorXd>(d->sensordata, ns);
      //add_snsr_noise(d->sensordata);
      mju_copy(&(gamma[0](0)), d->sensordata, ns);

      mju_copy(sigma_states[0]->qpos, d->qpos, nq);
      //copy_state(sigma_states[0], d); // for visualizations
      //mju_copy(sigma_states[0]->ctrl, d->ctrl, nu);

      //double t5 = omp_get_wtime()*1000.0;

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

      z_k = VectorXd::Zero(ns); // shat;
      //for (int i=0; i<N; i++) { z_k += W_s[i]*gamma[i]; }
      for (int i=1; i<N; i++) { z_k += gamma[i]; }
      z_k = W_s[0]*gamma[0] + (W_s[1])*z_k;
      // TODO  can optimize more if all the weights are the same

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
      P_t = W_c[0]*(x_i * x_i.transpose()) + W_c[1]*P_t; // + PtAdd;
      P_z = W_c[0]*(z * z.transpose())     + W_c[1]*P_z + PzAdd;
      Pxz = W_c[0]*(x_i * z.transpose())   + W_c[1]*Pxz;

      //P_z = P_z + PzAdd;

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
      //double t6 = omp_get_wtime()*1000.0;

      //printf("\ncombo init %f, sqrt %f, mjsteps %f, merge %f\n",
      //        t3-t2, t4-t3, t5-t4, t6-t5);
    }

    void predict_correct_p2(double * ctrl, double dt, double* sensors, double* conf = 0) {

      MatrixXd K = Pxz * P_z.inverse();

      VectorXd s = Map<VectorXd>(sensors, ns); // map our real z to vector

      VectorXd prev_xt = x_t;
      x_t = x_t + (K*(s-z_k));
      P_t = P_t - (K * P_z * K.transpose());
      //double mean = std::abs(P_t.mean());
      //std::cout<<"P_T average:\n"<<mean<<std::endl;
      //if (mean > 1) {
      //  printf("\n\nCovariance blowup!!\n\n");
      //  std::cout<<"P_Tadd average:\n"<<PtAdd.mean()<<std::endl;

      //  // don't allow for correction
      //  P_t = PtAdd;
      //  std::cout<<"x_t :\n"<<x_t.transpose()<<std::endl;
      //  std::cout<<"prev:\n"<<prev_xt.transpose()<<std::endl;
      //  x_t = prev_xt; // undo-correction
      //}
      set_data(d, &(x_t)); // set corrected state into mujoco data struct
      mju_copy(d->sensordata, &(z_k(0)), ns); // copy estimated data for viewing 
#if 1
      mj_forward(m, d); // fwd at new estimate
      mj_sensor(m, d);
#endif
    }

    void predict_correct(double * ctrl, double dt, double* sensors, double* conf = 0) {

      double t2 = omp_get_wtime()*1000.0;
      int end = (nq > 26) ? 26 : nq;

      set_data(d, &(x_t));
      mju_copy(d->ctrl, ctrl, nu); // set controls for the center point

      get_state(d, &(x[0])); //x[0] = x_t;

      double t3 = omp_get_wtime()*1000.0;

#ifdef SYMMETRIC_SQUARE_ROOT
      MatrixXd m_sqrt = ((L+lambda)*(P_t)).sqrt(); // symmetric; blows up
#else
      LLT<MatrixXd> chol((L+lambda)*(P_t));
      MatrixXd m_sqrt = chol.matrixL(); // chol
#endif
      // compilation time.... 

      double t4 = omp_get_wtime()*1000.0;

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
      m->opt.iterations = 100; 
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
        //double omp1 = omp_get_wtime()*1000.0;
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
          x[i+0] = x[0]+m_sqrt.col(i-1);// + qhat;

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

          fast_forward(t_m, t_d, j);
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

          get_state(t_d, &(x[i+0]));

          mj_forward(t_m, t_d); // at new position; can't assume we didn't move
          mj_sensor(t_m, t_d);

          if (INV_CHECK) {
            printf("\n");
          }


          add_snsr_noise(t_d->sensordata);
          mju_copy(&(gamma[i](0)), t_d->sensordata, ns);

          if (j < nq) { // only copy position perturbed
            //mju_copy(sigma_states[i+0]->qpos, t_d->qpos, nq);
          }

          ///////////////////////////////////////////////////// symmetric point
          x[i+L] = x[0]-m_sqrt.col(i-1);// + qhat;

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

          fast_forward(t_m, t_d, j);
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

          get_state(t_d, &(x[i+L]));

          mj_forward(t_m, t_d); // at new position; can't assume we didn't move
          mj_sensor(t_m, t_d);

          if (INV_CHECK) {
            printf("\n");
          }

          add_snsr_noise(t_d->sensordata);
          mju_copy(&(gamma[i+L](0)), t_d->sensordata, ns);

          if (j < nq) { // only copy position perturbed states
            //mju_copy(sigma_states[i+nq]->qpos, t_d->qpos, nq);
          }

        }
        //double omp2 = omp_get_wtime()*1000.0;
        //printf("p thread: %d chunk: %d-%d Time: %f\n", tid, s, e, omp2-omp1);
      }

      // step for the central point
      //mj_forward(m, d);
      mj_Euler(m, d); // step

      get_state(d, &(x[0]));
      get_state(d, &(x_minus)); // pre-correction

      mj_forward(m, d);
      mj_sensor(m, d); // sensor values at new positions
      //gamma[0] = Map<VectorXd>(d->sensordata, ns);
      //add_snsr_noise(d->sensordata);
      mju_copy(&(gamma[0](0)), d->sensordata, ns);

      mju_copy(sigma_states[0]->qpos, d->qpos, nq);
      //copy_state(sigma_states[0], d); // for visualizations
      //mju_copy(sigma_states[0]->ctrl, d->ctrl, nu);

      double t5 = omp_get_wtime()*1000.0;

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

      VectorXd z_k = shat; //VectorXd::Zero(ns);
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

      double t6 = omp_get_wtime()*1000.0;

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
      //MatrixXd F = x_t - qhat;
      //qhat = (1-dk)*qhat + dk*(x_t - x_minus);
      //dk = 0.5;
      //shat = (1-dk)*shat + dk*(s-z_k);
      //shat.setZero();
      //qhat.setZero();

      // x_t = F*x_minus + qhat;
      //Q = (1-dk) * Q + dk*(K*V*V.transpose()*K.transpose() + P_t );

      //std::cout << "\nqhat:\n"<< qhat.transpose().format(CleanFmt) << std::endl;

      //double a2 = fabs(P_t.determinant());
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

    void predict_correct_2stage(double * ctrl, double dt, double* sensors, double* conf = 0) {

      int end = (nq > 26) ? 26 : nq;

      set_data(d, &(x_t));
      mju_copy(d->ctrl, ctrl, nu); // set controls for the center point

      get_state(d, &(x[0])); //x[0] = x_t;


      LLT<MatrixXd> chol((L+lambda)*(P_t));
      MatrixXd m_sqrt = chol.matrixL(); // chol
      //MatrixXd sqrt2 = ((L+lambda)*(P_t)).m_sqrt(); // symmetric; blows up
      // compilation time.... 

      if (NUMBER_CHECK) {
        IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");
        std::cout << "p_t start:\n"<< P_t.block(0,0,end,end).format(CleanFmt) << std::endl;
        std::cout << "sqrt output:\n"<< m_sqrt.block(0,0,end,end).format(CleanFmt) << std::endl;
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
      mj_forward(m, d); // solve for center point accurately

      if (INV_CHECK) {
        printf("  \tinv: ");
        for (int i=0; i<nv; i++) printf("%1.4f ", d->qfrc_constraint[i]);
        printf("\ta-qacc: ");
        for (int i=0; i<nv; i++) printf("%1.4f ", d->qacc[i]);
        printf("\n");
      }

      double center_tol = constraint_violated(d);

      //m->opt.iterations = 15; 
      //m->opt.tolerance = 0; 
      for (int i=1; i<N; i++) { W_theta[i] = sqrt(L+lambda); }

      // set tolerance to be low, run 50, 100 iterations for mujoco solver
      // copy qacc for sigma points with some higher tolerance
#pragma omp parallel
      {
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
          x[i+0] = x[0]+m_sqrt.col(i-1) + qhat;

          t_d->time = d->time;
          set_data(t_d, &(x[i+0]));
          //mju_copy(t_d->qacc, d->qacc, nv); // copy from center point
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

          //fast_forward(t_d, j);
          mj_forward(t_m, t_d);

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

          get_state(t_d, &(x[i+0]));

          mj_forward(t_m, t_d); // at new position; can't assume we didn't move
          mj_sensor(t_m, t_d);

          if (INV_CHECK) {
            printf("\n");
          }


          add_snsr_noise(t_d->sensordata);
          mju_copy(&(gamma[i](0)), t_d->sensordata, ns);

          if (j < nq) { // only copy position perturbed
            //mju_copy(sigma_states[i+0]->qpos, t_d->qpos, nq);
          }

          ///////////////////////////////////////////////////// symmetric point
          x[i+L] = x[0]-m_sqrt.col(i-1) + qhat;

          t_d->time = d->time;
          set_data(t_d, &(x[i+L]));
          if (!ctrl_state) mju_copy(t_d->ctrl, ctrl, nu); // set controls for this t
          add_ctrl_noise(t_d->ctrl);
          //mju_copy(t_d->qacc, d->qacc, nv); // copy from center point

          if (INV_CHECK) {
            printf("%d: ", i+L);
            printf("   b-state: ");
            for (int k=0; k<nq; k++) printf("%1.4f ", t_d->qpos[k]);
            for (int k=0; k<nv; k++) printf("%1.4f ", t_d->qvel[k]);
            printf("\tb-qacc: ");
            for (int k=0; k<nv; k++) printf("%1.4f ", t_d->qacc[k]);
          }

          //fast_forward(t_d, j);
          mj_forward(t_m, t_d);

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

          get_state(t_d, &(x[i+L]));

          mj_forward(t_m, t_d); // at new position; can't assume we didn't move
          mj_sensor(t_m, t_d);

          if (INV_CHECK) {
            printf("\n");
          }

          add_snsr_noise(t_d->sensordata);
          mju_copy(&(gamma[i+L](0)), t_d->sensordata, ns);

          if (j < nq) { // only copy position perturbed states
            //mju_copy(sigma_states[i+nq]->qpos, t_d->qpos, nq);
          }
        }
      }
      // step for the central point
      mj_Euler(m, d); // step

      get_state(d, &(x[0]));
      get_state(d, &(x_minus)); // pre-correction

      // even weighting
      W_s[0] = 1.0/N;
      W_c[0] = 1.0/N;
      for (int i=1; i<N; i++) { W_s[i] = 1.0/N; W_c[i] = W_s[i]; }

      // aprior mean
      x_t.setZero();
      for (int i=0; i<N; i++) { x_t += W_s[i]*x[i]; }
      //for (int i=1; i<N; i++) { x_t += x[i]; }
      //x_t = W_s[0]*x[0] + (W_s[1])*x_t;


      P_t.setZero();
      for (int i=0; i<N; i++) {
        VectorXd x_i(x[i] - x_t);
        P_t += W_c[i] * (x_i * x_i.transpose());
      }

      set_data(d, &(x_t)); // set new central point

      mj_forward(m, d);
      mj_sensor(m, d); // sensor values at new positions
      mju_copy(&(gamma[0](0)), d->sensordata, ns);

      mju_copy(sigma_states[0]->qpos, d->qpos, nq);

      LLT<MatrixXd> chol2((L+lambda)*(P_t));
      m_sqrt = chol2.matrixL(); // chol
#pragma omp parallel
      {
        int tid = omp_get_thread_num();
        int t = omp_get_num_threads();
        int s = tid * L / t;
        int e = (tid + 1 ) * L / t;
        if (tid == t-1) e = L;

        mjData *t_d = sigmas[tid];
        mjModel*t_m = models[tid];

        for (int j=s; j<e; j++) {
          // step through all the perturbation cols and collect the data
          int i = j+1;
          double scale;

          ///////////////////////////////////////////////////// sigma point
          x[i+0] = x[0]+m_sqrt.col(i-1) + qhat;

          t_d->time = d->time;
          set_data(t_d, &(x[i+0]));
          mju_copy(t_d->ctrl, ctrl, nu); // set controls for this t
          add_ctrl_noise(t_d->ctrl);

          mj_forward(t_m, t_d);

          mj_sensor(t_m, t_d);

          add_snsr_noise(t_d->sensordata);
          mju_copy(&(gamma[i](0)), t_d->sensordata, ns);

          ///////////////////////////////////////////////////// symmetric point
          x[i+L] = x[0]-m_sqrt.col(i-1) + qhat;

          t_d->time = d->time;
          set_data(t_d, &(x[i+L]));
          mju_copy(t_d->ctrl, ctrl, nu); // set controls for this t
          add_ctrl_noise(t_d->ctrl);

          mj_forward(t_m, t_d);

          t_m->opt.timestep = m->opt.timestep + add_time_noise();
          mj_Euler(t_m, t_d);


          get_state(t_d, &(x[i+L]));

          mj_forward(t_m, t_d); // at new position; can't assume we didn't move
          mj_sensor(t_m, t_d);

          if (INV_CHECK) {
            printf("\n");
          }

          add_snsr_noise(t_d->sensordata);
          mju_copy(&(gamma[i+L](0)), t_d->sensordata, ns);

          if (j < nq) { // only copy position perturbed states
            //mju_copy(sigma_states[i+nq]->qpos, t_d->qpos, nq);
          }

          ///////////////////////////////////////////////////// symmetric point
          t_d->time = d->time;
          set_data(t_d, &(x[i+L]));
          mju_copy(t_d->qacc, d->qacc, nv); // copy from center point

        }
      }

      VectorXd z_k = shat; //VectorXd::Zero(ns);
      for (int i=0; i<N; i++) { z_k += W_s[i]*gamma[i]; }
      //for (int i=1; i<N; i++) { z_k += gamma[i]; }
      //z_k = W_s[0]*gamma[0] + (W_s[1])*z_k;

      if (NUMBER_CHECK) {
        IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");
        printf("Correct: After Step time %f\n", sigma_states[0]->time);
        std::cout << "\nz_k hat:\n"<< (z_k).transpose().format(CleanFmt) << std::endl;
      }

      P_z.setZero();
      Pxz.setZero();
      for (int i=0; i<N; i++) {
        VectorXd z(gamma[i] - z_k);
        VectorXd x_i(x[i] - x_t);

        P_z += W_c[i] * (z * z.transpose());
        Pxz += W_c[i] * (x_i * z.transpose());
      }

      if (NUMBER_CHECK) {
        IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");
        std::cout << "Prediction output:\nx_t\n"<< x_t.transpose().format(CleanFmt) << std::endl;
        std::cout << "Prediction output:\nP_t\n"<< P_t.block(0,0,end,end).format(CleanFmt) << std::endl;
      }

      P_z = P_z + PzAdd;

      MatrixXd K = Pxz * P_z.inverse();
      VectorXd s = Map<VectorXd>(sensors, ns); // map our real z to vector

      x_t = x_t + (K*(s-z_k));
      P_t = P_t - (K * P_z * K.transpose());

      //P_t = P_t + PtAdd; 

      set_data(d, &(x_t)); // set central point corrected
      mju_copy(d->sensordata, &(z_k(0)), ns); // copy estimated data for viewing 

      //std::cout << "\ncorrect x_t:\n"<< x_t.transpose().format(CleanFmt) << std::endl;
      //std::cout << "\nother   x_t:\n"<< x_t_aug.transpose().format(CleanFmt) << std::endl;

      // MAKING A PREVIOUS DATA
      //set_data(prev_d, &(x_t)); // the 'previous' estimate's mjData
      //mju_copy(prev_d->ctrl, d->ctrl, nu); // set controls for the center point
      //mju_copy(prev_d->qacc, d->qacc, nv); // set controls for the center point
      //mj_forward(m, prev_d);

      IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");

      std::cout << "\ndelta:\n"<< (s-z_k).transpose().format(CleanFmt) << std::endl;
      std::cout << "\ncorrection:\n"<< (K*(s-z_k)).transpose().format(CleanFmt) << std::endl;
      std::cout << "\np_t :\n"<< P_t.diagonal().transpose().format(CleanFmt) << std::endl;
      std::cout << "\np_z :\n"<< P_z.diagonal().transpose().format(CleanFmt) << std::endl;

      //double dk = 0.95;
      //MatrixXd F = x_t - qhat;
      //qhat = (1-dk)*qhat + dk*(x_t - x_minus);
      //dk = 0.5;
      //shat = (1-dk)*shat + dk*(s-z_k);
      //shat.setZero();
      //qhat.setZero();
      // x_t = F*x_minus + qhat;
      //Q = (1-dk) * Q + dk*(K*V*V.transpose()*K.transpose() + P_t );

      //std::cout << "\nqhat:\n"<< qhat.transpose().format(CleanFmt) << std::endl;

      //double a2 = fabs(P_t.determinant());
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

    void predict(double * ctrl, double dt) {

      IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");

      double t2 = omp_get_wtime()*1000.0;

      m->opt.timestep = dt; // smoother way of doing this?



      double t3 = omp_get_wtime()*1000.0;

      //P_t = MatrixXd::Identity(L,L) * 1e-3;
      LLT<MatrixXd> chol((L+lambda)*(P_t));
      MatrixXd sqrt = chol.matrixL(); // chol

      double t4 = omp_get_wtime()*1000.0;

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

      double t5 = omp_get_wtime()*1000.0;

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

      double t6 = omp_get_wtime()*1000.0;

      printf("\npredict init %f, sqrt %f, mjsteps %f, merge %f\n",
          t3-t2, t4-t3, t5-t4, t6-t5);

      if (NUMBER_CHECK) std::cout << "Prediction output:\nx_t\n"<< x_t.transpose().format(CleanFmt) << std::endl;
      if (NUMBER_CHECK) std::cout << "Prediction output:\nP_t\n"<< P_t.format(CleanFmt) << std::endl;

    }

    void correct(double* sensors) {

      IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
      double t2 = omp_get_wtime()*1000.0;

      mju_copy(d->qpos, &(x_t(0)), nq);
      mju_copy(d->qvel, &(x_t(nq)), nv);
      if (ctrl_state) mju_copy(d->ctrl, &(x_t(nq+nv)), nu);

      double t3 = omp_get_wtime()*1000.0;

      //P_t = MatrixXd::Identity(L,L) * 1e-3;
      LLT<MatrixXd> chol((L+lambda)*(P_t));
      MatrixXd sqrt = chol.matrixL(); // chol

      double t4 = omp_get_wtime()*1000.0;

      // TODO set solver iterations and tolerance
      mju_copy(sigma_states[0]->qpos, d->qpos, nq);
      mju_copy(sigma_states[0]->qvel, d->qvel, nv);
      mju_copy(sigma_states[0]->ctrl, d->ctrl, nu);

      mj_forward(m, d);
      mj_sensor(m, d);
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
          mj_sensor(m, sigmas[tid]);

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
          mj_sensor(m, sigmas[tid]);

          gamma[i+L] = Map<VectorXd>(sigmas[tid]->sensordata, ns);

          mju_copy(sigma_states[i+L]->qpos, sigmas[tid]->qpos, nq);
          mju_copy(sigma_states[i+L]->qvel, sigmas[tid]->qvel, nv);
          //mju_copy(sigma_states[i+L]->ctrl, sigmas[tid]->ctrl, nu);
        }
      }

      double t5 = omp_get_wtime()*1000.0;

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

      double t6 = omp_get_wtime()*1000.0;

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
         double a = fabs(P_t.determinant());
         double b = pow(2*3.14159265*exp(1), (double)L);
         double entropy = 0.5*log(b * a);
         printf("\n\ne %f\ta: %1.32f\tb: %f\tEntropy: %f\n\n", a*b, a, b, entropy);
         */
    }

    mjData* get_state() {
      return this->d;
    }

    std::vector<mjData*> get_sigmas() {
      return sigma_states;
    }

    mjData* get_stddev() {
      VectorXd var = P_t.diagonal();
      if (NUMBER_CHECK) {
        IOFormat CleanFmt(2, 0, ", ", "\n", "[", "]");
        std::cout << "\ncorrect x_t:\n"<< x_t.transpose().format(CleanFmt) << std::endl;
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


  private:
    int L;
    int N;
    int my_threads;
    double alpha;
    double beta;
    double lambda;
    double diag;
    double Ws0;
    double noise;
    double tol;
    double * W_s;
    double * W_theta;
    double * W_c;
    double * p_state;
    double * snsr_ptr;

    std::normal_distribution<> * rd_vec; 
    std::normal_distribution<> * ct_vec; 
    std::vector<double *> raw; // raw data storage pointed to by x
    //std::vector<VectorXd *> x;
    VectorXd * x;
    VectorXd * gamma;
    VectorXd x_t;
    VectorXd z_k;
    VectorXd x_minus;
    VectorXd qhat;
    VectorXd shat;
    MatrixXd Q;
    //MatrixXd * P;
    MatrixXd P_t;
    MatrixXd P_z;
    MatrixXd Pxz;
    MatrixXd PzAdd;
    MatrixXd PtAdd;
    double mrkr_conf;

    mjData * prev_d;

    std::vector<mjData *> sigma_states;
    std::vector<mjData *> sigmas;
    std::vector<mjModel *> models;
    mjData * stddev;
    //mjData ** sigma_states;
    //mjData ** sigmas;

    bool NUMBER_CHECK;
    bool ctrl_state;


    double *m_noise;
    double *t_noise;
};




