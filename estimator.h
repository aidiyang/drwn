#include "mujoco.h"

#include <string.h>
#include <iostream>
#include <random>

#include <omp.h>

#ifdef USE_EIGEN_MKL
#define EIGEN_USE_MKL_ALL
#endif

#define EIGEN_DONT_PARALLELIZE

#include <Eigen/StdVector>
#include <Eigen/Dense>


using namespace Eigen;

class Estimator {
  public:
    Estimator(mjModel *m, mjData * d) {
      this->m = mj_copyModel(NULL, m); // make local copy of model and data
      this->d = mj_makeData(this->m);

      this->nq = this->m->nq;
      this->nv = this->m->nv;
      this->nu = this->m->nu;
    }

    ~Estimator() {
      mj_deleteData(d);
      mj_deleteModel(m);
    }

    void get_state(mjData * d, double * state) {
      mju_copy(state, d->qpos, nq);
      mju_copy(state+nq, d->qvel, nv);
    }
    void set_state(mjData * d, double * state) {
      mju_copy(d->qpos, state, nq);
      mju_copy(d->qvel, state+nq, nv);
    }
    //virtual void init();
    virtual void predict(double * ctrl, double dt) {};
    virtual void correct(double* sensors) {};

    mjModel* m;
    mjData* d;
    int nq;
    int nv;
    int nu;
};

/*
class EKF : public Estimator {
  public:
    EKF(mjModel *m, mjData * d) : Estimator() {
    };
    ~EKF() : ~Estimator() {};

};
*/

class UKF : public Estimator {
  public:
    UKF(mjModel *m, mjData * d,
        double _alpha, double _beta, double _kappa) : Estimator(m, d) {
      L = nq + nv; // size of state dimensions
      N = 2*L + 1;

      alpha = _alpha;
      beta = _beta;
      lambda = (alpha*alpha)*((double)L + _kappa) - (double)L;

      W_s = new double[N];
      W_c = new double[N];
      sigma_states.resize(N);
      raw.resize(N);
      x = new VectorXd[N];
      gamma = new VectorXd[N];
      for (int i=0; i<N; i++) {
        if (i==0) {
          sigma_states[i] = this->d;
          W_s[i] = lambda / ((double) L + lambda);
          W_c[i] = W_s[i] + (1-(alpha*alpha)+beta);
        }
        else {
          sigma_states[i] = mj_makeData(this->m);
          W_s[i] = 1.0 / (2.0 * ((double)L + lambda));
          W_c[i] = W_s[i];
        }
        //raw[i] = new double[L];
        //x[i] = Map<VectorXd>(sigma_states[i]->qpos, L);
        //x[i] = Map<VectorXd>(raw[i], L); // point into raw buffer, not mjdata
        //x[i].setZero();
        x[i] = VectorXd::Zero(L); // point into raw buffer, not mjdata
        raw[i] = &x[i](0);

        //gamma[i] = Map<VectorXd>(sigma_states[i]->sensordata, m->nsensordata);
        //x[i] = new VectorXd>(L);
      }

      //P = new MatrixXd::Identity(L,L);
      P_t = MatrixXd::Identity(L,L);
      P_z = MatrixXd::Zero(m->nsensordata,m->nsensordata);
      Pxz = MatrixXd::Zero(L,m->nsensordata);

      x_t = VectorXd::Zero(L);
      p_state = &x_t(0);
      //p_state = new double[L];
      //x_t = Map<VectorXd> (p_state, L);
      //x_t.setZero();
    };

    ~UKF() {
      delete[] x;
      delete[] W_s;
      delete[] W_c;
      //delete[] p_state;
      delete[] gamma;
      delete[] W_s;
      delete[] W_c;
      //for (int i=0; i<N; i++) {
      //  delete[] raw[i];
      //}
    };

    void predict(double * ctrl, double dt) {

      m->opt.timestep = dt; // smoother way of doing this?

      double t1 = omp_get_wtime()*1000.0;
      mju_copy(d->ctrl, ctrl, nu); // set controls for this t
      mj_forward(m, d);

      mju_copy(raw[0], d->qpos, nq);
      mju_copy(raw[0]+nq, d->qvel, nv);
      //#pragma omp parallel for 
      for (int i=1; i<N; i++) { // for all sigma points; not main point
        //get_state(this->d, raw[i]);
        mju_copy(sigma_states[i]->ctrl, ctrl, nu); // set controls for this t

        //mju_copy(sigma_states[i]->qpos, d->qpos, nq);
        //mju_copy(sigma_states[i]->qvel, d->qvel, nv);
        //mju_copy(sigma_states[i]->qact, d->qact, nv); // set controls for this t

        mju_copy(raw[i], d->qpos, nq);
        mju_copy(raw[i]+nq, d->qvel, nv);
        //x[i] = Map<VectorXd>(raw[i], L); // point into raw buffer, not mjdata
        //mj_copyData(sigma_states[i], m, d); // slow

      } // should be accessible from x now
      double t2 = omp_get_wtime()*1000.0;

      // get the matrix square root
      LLT<MatrixXd> chol((L+lambda)*(P_t));
      MatrixXd sqrt = chol.matrixL(); // chol

      double t3 = omp_get_wtime()*1000.0;

      // perturb x with covariance values => make sigma point vectors
//#pragma omp parallel for
      for (int i=1; i<=L; i++) {
        x[i+0] += sqrt.col(i-1);
        x[i+L] -= sqrt.col(i-1);

        

        mju_copy(sigma_states[i]->qpos, raw[i], nq);
        mju_copy(sigma_states[i]->qvel, raw[i]+nq, nv);
        mju_copy(sigma_states[i+L]->qpos, raw[i+L], nq);
        mju_copy(sigma_states[i+L]->qvel, raw[i+L]+nq, nv);
      }
      double t4 = omp_get_wtime()*1000.0;

      // copy back to sigma_states and step forward in time
//#pragma omp parallel for
      for (int i=0; i<N; i++) {
        //mj_copyData(sigma_states[i], this->m, this->d);
        //set_state(sigma_states[i], raw[i]);
        mj_step(m, sigma_states[i]);

        mju_copy(raw[i], sigma_states[i]->qpos, nq);
        mju_copy(raw[i]+nq, sigma_states[i]->qvel, nv);

        if (i == 1) {
          printf("raw\tvec:\n");
          for (int j=1; j<L; j++) {
            printf("%1.3f\t%1.3f\n", raw[i][j], x[i](j));
          }
          printf("\n");
        }
      }
      double t5 = omp_get_wtime()*1000.0;

      x_t.setZero();
      std::cout << "\n0pred: "<< x_t.transpose() << std::endl;
      for (int i=0; i<N; i++) {
        x_t += W_s[i] * x[i];
      }
      std::cout << "\n1pred: "<< x_t.transpose() << std::endl;
      P_t.setZero();
      for (int i=0; i<N; i++) {
        VectorXd x_i(x[i] - x_t);
        P_t += W_c[i] * (x_i * x_i.transpose());
      }
      //std::cout << "\nP_t:\n"<< P_t << std::endl;
      double t6 = omp_get_wtime()*1000.0;

      printf("predict copy %f, sqrt %f, sigmas %f, mjsteps %f, merge %f\n",
          t2-t1, t3-t2, t4-t3, t5-t4, t6-t5);

      // x_t, P_t are outputs
    }

    void correct(double* sensors) {

      std::cout << "\n2pred: "<< x_t.transpose() << std::endl;

      printf("\npredict output:\t");
      for (int i=0; i<L; i++)
        printf("%1.3f ", p_state[i]);
      printf("\n");

      mju_copy(d->qpos, p_state, nq);
      mju_copy(d->qvel, p_state+nq, nv);

      // get state of point 0
      double t1 = omp_get_wtime()*1000.0;

      //#pragma omp parallel for
      for (int i=1; i<N; i++) {
        mju_copy(raw[i], d->qpos, nq);
        mju_copy(raw[i]+nq, d->qvel, nv);
      }
      double t2 = omp_get_wtime()*1000.0;

      // get the matrix square root
      LLT<MatrixXd> chol((L+lambda)*(P_t));
      MatrixXd sqrt = chol.matrixL(); // chol

      double t3 = omp_get_wtime()*1000.0;

      // perturb x with covariance values => make sigma point vectors
      //#pragma omp parallel for
      for (int i=1; i<=L; i++) {
        x[i+0] += sqrt.col(i-1);
        x[i+L] -= sqrt.col(i-1);

        // x is set, backed in raw
        mju_copy(sigma_states[i]->qpos, raw[i], nq);
        mju_copy(sigma_states[i]->qvel, raw[i]+nq, nv);
        mju_copy(sigma_states[i+L]->qpos, raw[i+L], nq);
        mju_copy(sigma_states[i+L]->qvel, raw[i+L]+nq, nv);
      }
      double t4 = omp_get_wtime()*1000.0;

      // copy back to sigma_states and step forward in time
      //#pragma omp parallel for
      for (int i=0; i<N; i++) {
        mj_forward(m, sigma_states[i]);
        mj_sensor(m, sigma_states[i]);
        gamma[i] = Map<VectorXd>(sigma_states[i]->sensordata, m->nsensordata);
      }
      double t5 = omp_get_wtime()*1000.0;

      VectorXd z_k = VectorXd::Zero(m->nsensordata);
      for (int i=0; i<N; i++) {
        z_k += W_s[i] * gamma[i];
      }

      P_z.setZero();
      //MatrixXd Pxz = MatrixXd::Zero(L, m->nsensordata);
      Pxz.setZero();

      for (int i=0; i<N; i++) {
        VectorXd z(gamma[i] - z_k);
        VectorXd x_i(x[i] - x[0]);
        //if (i==0){
        //  std::cout << "\ngamma:\n"<< gamma[i].transpose() << std::endl;
        //  std::cout << "\nz_k:\n"<< z_k.transpose() << std::endl;
        //  std::cout << "\nz:\n"<< z.transpose() << std::endl;
        //}
        P_z += W_c[i] * (z * z.transpose());

        Pxz += W_c[i] * (x_i * z.transpose());
      }

      MatrixXd K = Pxz * P_z.inverse();
      VectorXd s = Map<VectorXd>(sensors, m->nsensordata); // map our real z to vector
      x_t = x_t + K*(s-z_k);
      P_t = P_t - K * P_z * K.transpose();

      double t6 = omp_get_wtime()*1000.0;

      printf("correct copy %f, sqrt %f, sigmas %f, mjsteps %f, merge %f\n",
          t2-t1, t3-t2, t4-t3, t5-t4, t6-t5);

      //std::cout << "\nPxz:\n"<< Pxz << std::endl;
      //std::cout << "\nP_z inverse:\n"<< P_z.inverse() << std::endl;
      //std::cout << "\nK:\n"<< K << std::endl;
      //std::cout << "\ndelta:\n"<< s-z_k << std::endl;
      //std::cout << "\n3pred: "<< x_t.transpose() << std::endl;
    }

    mjData* get_state() {
      return this->d;
    }

  private:
    int L;
    int N;
    double alpha;
    double beta;
    double lambda;
    double * W_s;
    double * W_c;
    double * p_state;
    std::vector<double *> raw; // raw data storage pointed to by x
    //std::vector<VectorXd *> x;
    VectorXd * x;
    VectorXd * gamma;
    VectorXd x_t;
    //MatrixXd * P;
    MatrixXd P_t;
    MatrixXd P_z;
    MatrixXd Pxz;

    std::vector<mjData *> sigma_states;
};


