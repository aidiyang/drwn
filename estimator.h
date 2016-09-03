#pragma once

#include "mujoco.h"
#include <string.h>
#include <iostream>
#include <random>
#include <functional>

#ifndef __APPLE__
#include <omp.h>
#endif
#include <math.h>

#ifdef USE_EIGEN_MKL
#define EIGEN_USE_MKL_ALL
#endif

//#define EIGEN_DONT_PARALLELIZE
//#define EIGEN_DEFAULT_TO_ROW_MAJOR

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigen>
//#define SYMMETRIC_SQUARE_ROOT
#ifdef SYMMETRIC_SQUARE_ROOT
#include <unsupported/Eigen/MatrixFunctions>
#endif

//#include <boost/random.hpp>
//#include <boost/random/normal_distribution.hpp>

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
    virtual void predict_correct(double * ctrl, double dt, double* sensors, double* conf = 0) {};
    virtual void predict_correct_p1(double * ctrl, double dt, double* sensors, double* conf = 0) {};
    virtual void predict_correct_p2(double * ctrl, double dt, double* sensors, double* conf = 0) {};
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
        bool debug = false, int threads = 1);
     
    ~UKF();

    double add_time_noise();
    void add_model_noise(mjModel* t_m);
    void add_ctrl_noise(double * noise);
    void add_snsr_noise(double * noise);
    void set_data(mjData* data, Eigen::VectorXd *x);
    void get_data(mjData* data, Eigen::VectorXd *x);
    void copy_state(mjData * dst, mjData * src);
    void fast_forward(mjModel *t_m, mjData * t_d, int index, int sensorskip);

    double constraint_scale(mjData * t_d, double tol);
    double constraint_violated(mjData* t_d);
    double handle_constraint(mjData* t_d, mjData* d, double tol,
        Eigen::VectorXd* t_x, Eigen::VectorXd* x0, Eigen::VectorXd* P);

    void predict(double * ctrl, double dt);
    void correct(double* sensors);
    void predict_correct(double * ctrl, double dt, double* sensors, double* conf = 0); 
    void predict_correct_p1(double * ctrl, double dt, double* sensors, double* conf = 0); 
    void predict_correct_p2(double * ctrl, double dt, double* sensors, double* conf = 0); 

    double * get_numeric_field(const mjModel* m, std::string s, int *size);
    //void predict_correct_2stage(double * ctrl, double dt, double* sensors, double* conf = 0);
    //void predict(double * ctrl, double dt); 
    //void correct(double* sensors); 

    //mjData* get_state() {return this->d; };
    mjData* get_stddev();
    //std::vector<mjData*> get_sigmas() {return sigma_states; };

  private:
    int L; // state size
    int N; // num sigma points
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
    Eigen::VectorXd * x;
    Eigen::VectorXd * gamma;
    Eigen::VectorXd x_t;
    Eigen::VectorXd z_k;
    Eigen::VectorXd x_minus;
    Eigen::VectorXd qhat;
    Eigen::VectorXd shat;
    Eigen::MatrixXd Q;
    //Eigen::MatrixXd * P;
    Eigen::MatrixXd P_t;
    Eigen::MatrixXd P_z;
    Eigen::MatrixXd Pxz;
    Eigen::MatrixXd PzAdd;
    Eigen::MatrixXd PtAdd;
    double mrkr_conf;

    mjData * prev_d;

    std::vector<mjData *> sigmas;
    std::vector<mjModel *> models;
    mjData * stddev;

    bool NUMBER_CHECK;
    bool ctrl_state;


    double *m_noise;
    double *t_noise;
};




