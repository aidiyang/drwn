#pragma once

#include "mujoco.h"
#include "kNN.h"
#include <string.h>
#include <iostream>
#include <random>
#include <functional>
#include <future>
#include <set>
#include <chrono>

#ifndef __APPLE__
//#include <omp.h>
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
    void add_ctrl_noise(double * ctrl);
    void add_snsr_noise(double * snsr);
    void add_snsr_limit(double * snsr);
    void set_data(mjData* data, Eigen::VectorXd *x);
    void get_data(mjData* data, Eigen::VectorXd *x);
    void copy_state(mjData * dst, mjData * src);
    void fast_forward(mjModel *t_m, mjData * t_d, int index, int sensorskip);

    double constraint_scale(mjData * t_d, double tol);
    double constraint_violated(mjData* t_d);
    double handle_constraint(mjData* t_d, mjData* d, double tol,
        Eigen::VectorXd* t_x, Eigen::VectorXd* x0, Eigen::VectorXd* P);
    double handle_constraint(mjData* t_d, double tol,
        Eigen::VectorXd* t_x, Eigen::VectorXd* x0);

    void predict(double * ctrl, double dt);
    void correct(double* sensors);
    void predict_correct(double * ctrl, double dt, double* sensors, double* conf = 0); 
    void predict_correct_p1(double * ctrl, double dt, double* sensors, double* conf = 0); 
    void predict_correct_p2(double * ctrl, double dt, double* sensors, double* conf = 0); 
    void predict_correct_p10(double * ctrl, double dt, double* sensors, double* conf = 0);

    double * get_numeric_field(const mjModel* m, std::string s, int *size);

    mjData* get_stddev();

    double sigma_samples(mjModel *t_m, mjData *t_d, mjData *d, double* ctrl, 
        Eigen::VectorXd *x, Eigen::MatrixXd *m_sqrt, int s, int e);

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
    double W_even;
    double * snsr_ptr;

    std::normal_distribution<> * rd_vec; 
    std::normal_distribution<> * ct_vec; 
    std::vector<double *> raw; // raw data storage pointed to by x
    //std::vector<VectorXd *> x;
    Eigen::VectorXd * x;
    Eigen::VectorXd * gamma;
    Eigen::MatrixXd m_gamma;
    Eigen::MatrixXd m_x;
    Eigen::VectorXd x_t;
    Eigen::VectorXd z_k;
    Eigen::VectorXd x_minus;
    Eigen::VectorXd q_hat;
    Eigen::VectorXd s_hat;
    Eigen::MatrixXd Q;
    //Eigen::MatrixXd * P;
    Eigen::MatrixXd P_t;
    Eigen::MatrixXd P_z;
    Eigen::MatrixXd Pxz;
    Eigen::MatrixXd PzAdd;
    double * snsr_limit;
    Eigen::MatrixXd PtAdd;
    double mrkr_conf;

    std::future<double> *sigma_handles;

    std::vector<mjData *> sigmas;
    std::vector<mjModel *> models;
    mjData * stddev;

    bool NUMBER_CHECK;
    bool ctrl_state;


    double *m_noise;
    double *t_noise;
};

class PF : public Estimator {
    public:
      PF(mjModel *m, mjData * d, double eps, int numPart, int nResamp, double diag, double snoise, double cnoise, 
      double* P_covar, int render, int thread, bool debug = false);

      ~PF();

      void predict_correct(double* ctrl, double dt, double* sensors, double* conf);
      void set_data(mjData* data, Eigen::VectorXd *x);
      void clampWeights(Eigen::VectorXd &weights);
      double forward_particles(mjModel* m, mjData* d, double* ctrl, int s, int e, int num);
      mjData* get_state();
      mjData* get_kNNstate();
      std::vector<mjData*> get_sigmas();
      void resampPart(int index);
      void kNN_resampPart(int index);
      void printState();
      double getDiff(Eigen::VectorXd part, double* est);
    
    private:
      bool debug;
      int numPart;
      int nResamp;
      double snoise;
      double cnoise;
      double eps;     //Spacing of particles
      double num_diag;
      double* snsr_weights;
      double* S_cov;
      int render;
      int threads;

      double minQfrc;
      double maxQfrc;
      
      //std::vector<mjData*> particles;
      double* p_states;
      double* p_sensors;
      double* p_ctrl;
      Eigen::VectorXd weights;
      Eigen::MatrixXd sensDiff;
      Eigen::VectorXd sumWeights;
      Eigen::VectorXd mu;
      Eigen::VectorXd diff;
      Eigen::VectorXd rms;
      Eigen::VectorXd estSens; 
      Eigen::MatrixXd covar;
      Eigen::MatrixXd sensCovar;
      Eigen::MatrixXd crossCovar;
      Eigen::VectorXd P_add;
      Eigen::MatrixXd S_add;
      Eigen::MatrixXd Kgain;
      Eigen::VectorXd sensor;
      
      std::default_random_engine gen;
      std::mt19937 s_rng;
      std::uniform_real_distribution<double> rand;
      std::normal_distribution<double> crand;
      std::normal_distribution<>* rd_vec; 
      std::normal_distribution<>* qposResamp;
      std::normal_distribution<>* qvelResamp;

      Eigen::VectorXd partCount;
      std::set<int> resample;

      std::vector<mjData *> sigma_states;
      std::future<double>* thread_handles;
      mjData** thread_datas;
};

class kNNPF : public Estimator {
    public:
      kNNPF(mjModel *m, mjData * d, double eps, int numPart, int nResamp, double diag, double snoise, double cnoise, 
      double* P_covar, int render, int thread, bool debug = false);

      ~kNNPF();

      void predict_correct(double* ctrl, double dt, double* sensors, double* conf);
      void set_data(mjData* data, Eigen::VectorXd *x);
      void clampWeights(Eigen::VectorXd &weights);
      mjData* get_state();
      mjData* get_kNNstate();
      std::vector<mjData*> get_sigmas();
      void printState();
      double getDiff(Eigen::VectorXd part, double* est);
    
    private:
      bool debug;
      int numPart;
      int nResamp;
      double snoise;
      double cnoise;
      double eps;     //Spacing of particles
      double num_diag;
      double* snsr_weights;

      int render;
      kNN* kNNdata;
      
      double* kNN_states;
      double* kNN_sensors;
      Eigen::VectorXd kNN_weights;
      Eigen::MatrixXd sensDiff;
      Eigen::VectorXd kNN_mu;
      Eigen::VectorXd kNN_diff;
      Eigen::VectorXd kNN_rms;
      Eigen::MatrixXd kNN_covar;
      Eigen::MatrixXd kNN_sensCovar;
      Eigen::MatrixXd kNN_crossCovar;
      Eigen::VectorXd P_add;
      Eigen::MatrixXd S_add;
      Eigen::MatrixXd kNN_Kgain;
      Eigen::VectorXd sensor;
      Eigen::VectorXd kNNest;
      Eigen::VectorXd kNN_estSens;
      
      std::default_random_engine gen;
      std::mt19937 s_rng;
      std::uniform_real_distribution<double> rand;
      std::normal_distribution<double> crand;
      std::normal_distribution<>* rd_vec; 

      std::set<int> kNN_resample;

      std::vector<mjData *> sigma_states;
      std::list<diffIndex> closeList;
      mjData* saveData;
};
