#include "mujoco.h"

#include <string.h>
#include <iostream>
#include <random>

#include <omp.h>

#ifdef USE_EIGEN_MKL
#define EIGEN_USE_MKL_ALL
#endif

#define EIGEN_DONT_PARALLELIZE

//#include <Eigen/StdVector>
#include <vector>
#include <Eigen/Dense>


using namespace Eigen;

class Estimator {
  public:
    Estimator(mjModel *m, mjData * d) {
      this->m = mj_copyModel(NULL, m); // make local copy of model and data
      this->d = mj_makeData(this->m);
      mj_copyData(this->d, m, d); // get current state

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
      printf("Alpha %f\nBeta %f\nLambda %f\n", alpha, beta, lambda);

      W_s = new double[N];
      W_c = new double[N];
      sigma_states.resize(N);
      //raw.resize(N);
      x = new VectorXd[N];
      gamma = new VectorXd[N];
      for (int i=0; i<N; i++) {
        if (i==0) {
          sigma_states[i] = this->d;
          W_s[i] = lambda / ((double) L + lambda);

          W_c[i] = W_s[i] + (1-(alpha*alpha)+beta);
          //W_c[i] = W_s[i];
        }
        else {
          sigma_states[i] = mj_makeData(this->m);
          mj_copyData(sigma_states[i], this->m, this->d);

          W_s[i] = 1.0 / (2.0 * ((double)L + lambda));
          W_c[i] = W_s[i];
        }
        x[i] = VectorXd::Zero(L); // point into raw buffer, not mjdata

        //x[i] = VectorXd::Zero(L); // point into raw buffer, not mjdata

        //gamma[i] = Map<VectorXd>(sigma_states[i]->sensordata, m->nsensordata);
        gamma[i] = VectorXd::Zero(m->nsensordata);
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

      //P = new MatrixXd::Identity(L,L);
      P_t = MatrixXd::Identity(L,L);
      P_z = MatrixXd::Zero(m->nsensordata,m->nsensordata);
      Pxz = MatrixXd::Zero(L,m->nsensordata);

      x_t = VectorXd::Zero(L);

      mju_copy(&x_t(0), sigma_states[0]->qpos, nq);
      mju_copy(&x_t(nq), sigma_states[0]->qvel, nv);

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
    };

    void predict(double * ctrl, double dt) {

      m->opt.timestep = dt; // smoother way of doing this?
      IOFormat CleanFmt(2, 0, ", ", "\n", "[", "]");

      printf("Predict t-1 = %f seconds\n", sigma_states[0]->time);

      double t1 = omp_get_wtime()*1000.0;

      mju_copy(d->ctrl, ctrl, nu); // set controls for this t
      mj_forward(m, d);

      // d should have our previous correction
      // copy to sigma_states
      mju_copy(d->qpos, &(x_t(0)), nq);
      mju_copy(d->qvel, &(x_t(nq)), nv);
      // copy to our data vectors
      mju_copy(&x[0](0), d->qpos, nq);
      mju_copy(&x[0](nq), d->qvel, nv);

      double t2 = omp_get_wtime()*1000.0;

      // get the matrix square root
      LLT<MatrixXd> chol((L+lambda)*(P_t));
      MatrixXd sqrt = chol.matrixL(); // chol
      std::cout<<"\n\nPrediction sqrt(P_T):\n"<<sqrt.diagonal().transpose().format(CleanFmt)<<"\n\n";

      double t3 = omp_get_wtime()*1000.0;
//#define NUMBER_CHECK

      // perturb x with covariance values => make sigma point vectors
#pragma omp parallel for
      for (int i=1; i<=L; i++) {
        //x[i+0] += sqrt.col(i-1);
        //x[i+L] -= sqrt.col(i-1);
        x[i+0] = x[0]+sqrt.col(i-1);
        x[i+L] = x[0]-sqrt.col(i-1);

        mju_copy(sigma_states[i]->qpos,   &x[i](0), nq);
        mju_copy(sigma_states[i]->qvel,   &x[i](nq), nv);
        mju_copy(sigma_states[i]->ctrl,   ctrl, nu); // set controls for this t
        //mju_copy(sigma_states[i]->qact, d->qact, nv);
        mju_copy(sigma_states[i+L]->qpos, &x[i+L](0), nq);
        mju_copy(sigma_states[i+L]->qvel, &x[i+L](nq), nv);
        mju_copy(sigma_states[i+L]->ctrl, ctrl, nu); // set controls for this t
        //mju_copy(sigma_states[i+L]->qact, d->qact, nv);
      }
      double t4 = omp_get_wtime()*1000.0;

#ifdef NUMBER_CHECK
      std::cout << "Prediction sqrt(P_T):\n"<< sqrt.format(CleanFmt) << std::endl;
      printf("Before Step:\n");
      for (int i=0; i<N; i++) {
        printf("%d ", i);
        for (int j=0; j<nq; j++)
          printf("%1.5f ", sigma_states[i]->qpos[j]);
        for (int j=0; j<nv; j++)
          printf("%1.5f ", sigma_states[i]->qvel[j]);
        printf("\t:: ");
        for (int j=0; j<m->nsensordata; j++)
          printf("%f ", sigma_states[i]->sensordata[j]);

        printf("\n");
      }
#endif

#pragma omp parallel for
      for (int i=0; i<N; i++) {
        mj_step(m, sigma_states[i]);
        
        // TODO is the forward and sensor necessary?
        mj_forward(m, sigma_states[i]);
        mj_sensor(m, sigma_states[i]);

        mju_copy(&x[i](0), sigma_states[i]->qpos, nq);
        mju_copy(&x[i](nq), sigma_states[i]->qvel, nv);
        // TODO can put the W_s multiplication here
      }

#ifdef NUMBER_CHECK
      printf("Predict t-0 = %f seconds\n", sigma_states[0]->time);
      printf("After Step:\n");
      for (int i=0; i<N; i++) {
        printf("%d ", i);
        for (int j=0; j<nq; j++)
          printf("%1.5f ", sigma_states[i]->qpos[j]);
        for (int j=0; j<nv; j++)
          printf("%1.5f ", sigma_states[i]->qvel[j]);
        printf("\t:: ");
        for (int j=0; j<m->nsensordata; j++)
          printf("%f ", sigma_states[i]->sensordata[j]);

        printf("\n");
      }
#endif

      double t5 = omp_get_wtime()*1000.0;

      x_t.setZero();
      //std::cout << "\n0pred: "<< x_t.transpose() << std::endl;
      for (int i=0; i<N; i++) {
        x_t += W_s[i] * x[i];
      }

      std::cout << "\n1pred: "<< x_t.transpose() << std::endl;
      //P_t = P_t + 0.0001;
      //P_t.setIdentity() * 0.0001;
      // TODO random noise in the covariance, with symmetry
      //P_t.setIdentity();
      P_t.setZero();
      for (int i=0; i<N; i++) {
        VectorXd x_i(x[i] - x_t);
        P_t += W_c[i] * (x_i * x_i.transpose());
      }
      std::cout<<"\n\nPrediction P_T:\n"<<P_t.diagonal().transpose().format(CleanFmt)<<"\n\n";
      for (int i=0; i<L; i++) {
        if (P_t.row(i).isZero())
          P_t(i,i) = 1.0;
      }
      std::cout<<"\n\nFixed?? P_T:\n"<<P_t.diagonal().transpose().format(CleanFmt)<<"\n\n";

      double t6 = omp_get_wtime()*1000.0;

      printf("\npredict copy %f, sqrt %f, sigmas %f, mjsteps %f, merge %f\n",
          t2-t1, t3-t2, t4-t3, t5-t4, t6-t5);

      // x_t, P_t are outputs
    }

    void correct(double* sensors) {

      IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

      // todo 
      mju_copy(d->qpos, &(x_t(0)), nq);
      mju_copy(d->qvel, &(x_t(nq)), nv);

      mj_forward(m, d);

      mju_copy(&x[0](0), d->qpos, nq);
      mju_copy(&x[0](nq), d->qvel, nv);

      // get state of point 0
      double t1 = omp_get_wtime()*1000.0;

#pragma omp parallel for
      for (int i=0; i<N; i++) {
        // TODO change to repropagate the sigma points
        //mju_copy(&(x[i](0)), d->qpos, nq);
        //mju_copy(&(x[i](nq)), d->qvel, nv);
      }
      double t2 = omp_get_wtime()*1000.0;

      // get the matrix square root
      LLT<MatrixXd> chol((L+lambda)*(P_t));
      MatrixXd sqrt = chol.matrixL(); // chol

      std::cout<<"\n\nCorrection sqrt(P_T):\n"<< sqrt.diagonal().transpose().format(CleanFmt) <<"\n\n";
      double t3 = omp_get_wtime()*1000.0;

      // perturb x with covariance values => make sigma point vectors
#ifdef NUMBER_CHECK
      std::cout<<"\n\nCorrection sqrt(P_T):\n"<< sqrt.format(CleanFmt) <<"\n\n";
      printf("Sensor Before Step at time %f\n", sigma_states[0]->time);
      for (int i=0; i<N; i++) {
        printf("%d\t", i);
        for (int j=0; j<nq; j++)
          printf("%1.5f ", sigma_states[i]->qpos[j]);
        for (int j=0; j<nv; j++)
          printf("%1.5f ", sigma_states[i]->qvel[j]);
        printf("\t:: ");
        for (int j=0; j<m->nsensordata; j++)
          printf("%f ", sigma_states[i]->sensordata[j]);
        printf("\n");
      }
#endif

#pragma omp parallel for
      for (int i=1; i<=L; i++) {
        //x[i+0] += sqrt.col(i-1);
        //x[i+L] -= sqrt.col(i-1);

        x[i+0] = x[0]+sqrt.col(i-1);
        x[i+L] = x[0]-sqrt.col(i-1);

        mju_copy(sigma_states[i]->qpos,   &x[i](0), nq);
        mju_copy(sigma_states[i]->qvel,   &x[i](nq), nv);
        //mju_copy(sigma_states[i]->qact, d->qact, nv);
        mju_copy(sigma_states[i+L]->qpos, &x[i+L](0), nq);
        mju_copy(sigma_states[i+L]->qvel, &x[i+L](nq), nv);
        //mju_copy(sigma_states[i+L]->qact, d->qact, nv);
      }
      double t4 = omp_get_wtime()*1000.0;

      // copy back to sigma_states and step forward in time
#pragma omp parallel for
      for (int i=0; i<N; i++) {
        // to get sensor values at the current time
        mj_forward(m, sigma_states[i]);
        mj_sensor(m, sigma_states[i]);

        gamma[i] = Map<VectorXd>(sigma_states[i]->sensordata, m->nsensordata);
        //mju_copy(&gamma[i](0), sigma_states[i]->sensordata, m->nsensordata);
        // TODO can do W_s multiply here
      }

#ifdef NUMBER_CHECK
      printf("Sensor After Step time %f\n", sigma_states[0]->time);
      for (int i=0; i<N; i++) {
        printf("%d\t", i);
        for (int j=0; j<nq; j++)
          printf("%1.5f ", sigma_states[i]->qpos[j]);
        for (int j=0; j<nv; j++)
          printf("%1.5f ", sigma_states[i]->qvel[j]);
        printf("\t:: ");
        for (int j=0; j<m->nsensordata; j++)
          printf("%f ", sigma_states[i]->sensordata[j]);
        printf("\n");
      }
#endif
      printf("\n");
      double t5 = omp_get_wtime()*1000.0;

      VectorXd z_k = VectorXd::Zero(m->nsensordata);
      for (int i=0; i<N; i++) {
        z_k += W_s[i] * gamma[i];
      }

      P_z.setZero();
      Pxz.setZero();
      //P_z.setIdentity() * 0.0001;
      //Pxz.setIdentity() * 0.0001;
      //P_z.setIdentity();
      //Pxz.setIdentity();

      for (int i=0; i<N; i++) {
        VectorXd z(gamma[i] - z_k);
        VectorXd x_i(x[i] - x_t);
        
        P_z += W_c[i] * (z * z.transpose());
        Pxz += W_c[i] * (x_i * z.transpose());
      }
      for (int i=0; i<m->nsensordata; i++) {
        if (P_z.row(i).isZero())
          P_z(i,i) = 1.0;
      }

      MatrixXd K = Pxz * P_z.inverse();
      //for (int i=0; i<m->nsensordata; i++) {
      //  if (K(i,i) == 0)
      //    K(i,i) = 1.0;
      //}
      VectorXd s = Map<VectorXd>(sensors, m->nsensordata); // map our real z to vector
      x_t = x_t + (K*(s-z_k));
      P_t = P_t - (K * P_z * K.transpose());

      double t6 = omp_get_wtime()*1000.0;

      printf("correct copy %f, sqrt %f, sigmas %f, mjsteps %f, merge %f\n",
          t2-t1, t3-t2, t4-t3, t5-t4, t6-t5);

#ifdef NUMBER_CHECK
      std::cout << "Kalman Gain:\n"<< K.format(CleanFmt) << std::endl;
      std::cout << "\nPz  :\n"<< P_z.format(CleanFmt) << std::endl;
      std::cout << "\nPz^-1:\n"<< P_z.inverse().format(CleanFmt) << std::endl;
      std::cout << "\nPxz :\n"<< Pxz.format(CleanFmt) << std::endl;
      std::cout << "\ncorrect p_t :\n"<< P_t.format(CleanFmt) << std::endl;
      std::cout << "\nraw sensors:\n"<< (s).transpose().format(CleanFmt) << std::endl;
      std::cout << "\nz_k hat:\n"<< (z_k).transpose().format(CleanFmt) << std::endl;
      std::cout << "\ndelta:\n"<< (s-z_k).transpose().format(CleanFmt) << std::endl;
#endif

      // set our ideal state to be our calculated estimate
      mju_copy(d->qpos, &(x_t(0)), nq);
      mju_copy(d->qvel, &(x_t(nq)), nv);
    }

    mjData* get_state() {
      //return this->d;
      return sigma_states[0];
    }

    std::vector<mjData*> get_sigmas() {
      //return this->d;
      return sigma_states;
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


