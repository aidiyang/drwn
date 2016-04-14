#include "mujoco.h"

#include <string.h>
#include <random>

#include <Eigen/StdVector>


using namespace Eigen;

class Estimator {
   public:
      Estimator(mjModel *m, mjData * d) {
         this->m = mj_copyModel(NULL, m); // make local copy of model and data
         this->d = mj_makeData(this->m);

         nq = this->m->nq;
         nv = this->m->nv;
         nu = this->m->nu;
      }

      ~Estimator() {
         mj_deleteData(d);
         mj_deleteModel(m);
      }

      void get_state(mjData * d, double * state) {
         mju_copy(state, d->qpos);
         mju_copy(state+nq, d->qvel);
      }
      void set_state(mjData * d, double * state) {
         mju_copy(d->qpos, state);
         mju_copy(d->qvel, state+nq);
      }
      virtual void init();
      virtual void predict(double * u, double dt);
      virtual void correct(double* j_qpos, double* j_qvel, double* sensors);

   private:
      mjModel* m;
      mjData* d;
      int nq;
      int nv;
      int nu;
};

class EKF : public Estimator {
   public:
      EKF(mjModel *m, mjData * d) : Estimator() {
      };
      ~EKF() : ~Estimator() {};

};

class UKF : public Estimator {
   public:
      ~UKF(mjModel *m, mjData * d,
            double _alpha, double _beta, double _kappa) : ~Estimator() {
         L = nq + nv; // size of state dimensions
         N = 2*L + 1;

         alpha = _alpha;
         beta = _beta;
         lambda = (alpha*alpha)*((double)L + _kappa) - (double)L;

         // keep copies of mjData for sigma points
         // keep copies of vectors for maths
         sigma_states.resize(N);
         raw.resize(N);
         x.resize(N);
         for (int i=0; i<N; i++) {
            if (i==0) {
               sigma_states[i] = d;
               W_s[i] = lamda / ((double) L + lambda);
               W_c[i] = W_s[i] + (1-(alpha*alpha)+beta);
            }
            else {
               sigma_states[i] = mj_makeData(this->m);
               W_s[i] = 1.0 / (2.0 * ((double)L + lambda));
               W_c[i] = W_s[i];
            }
            raw[i] = new double[L];
            Map<VectorXd> v(raw[i], L);
            x[i] = &v; // TODO double check allocated right
         }

         P = new MatrixXd::Identity(L,L);
      };

      ~UKF() : ~Estimator() {};

      void predict(double * u, double dt) {

         m->opt.timestep = dt; // smoother way of doing this?

         //mju_copy(d->ctrl, u, nu); // set controls for this t

         // get state of point 0
         for (int i=0; i<N; i++) {
            get_state(this->d, raw[i]);
            mju_copy(sigma_states[i]->ctrl, u, nu); // set controls for this t
         } // should be accessible from x now

         // get the matrix square root
         MatrixXd sqrt( ((L+lambda)*P).llt().matrixL() ); // chol

         // perturb x with covariance values => make sigma point vectors
         for (int i=0; i<L; i++) {
            x[i+0] += sqrt.col(i);
            x[i+L] -= sqrt.col(i);
         }

         // copy back to sigma_states and step forward in time
         // #pragma omp parallel for
         for (int i=0; i<N; i++) {
            //mj_copyData(sigma_states[i], this->m, this->d);
            set_state(sigma_states[i], raw[i]);
            mj_step(m, sigma_statesi]);
         }

         for (int i=0; i<N; i++) {
            x_t += W_s[i] * x[i];
         }
         for (int i=0; i<N; i++) {
            VectorXd x_i(x[i] - x_t);
            P_t += W_c[i] * (x_i * x_i.transpose());
         }

         // x_t, P_t are outputs
         x[0] = x_t;
         P = P_t;
      }

      // TODO changes this? joint qpos, not 'actual'
      void correct(double* j_qpos, double* j_qvel, double* sensors) {
         m->opt.timestep = dt; // smoother way of doing this?

         //mju_copy(d->ctrl, u, nu); // set controls for this t

         // get state of point 0
         for (int i=0; i<N; i++) {
            get_state(this->d, raw[i]);
            mju_copy(sigma_states[i]->ctrl, u, nu); // set controls for this t
         } // should be accessible from x now

         // get the matrix square root
         MatrixXd sqrt( ((L+lambda)*P).llt().matrixL() ); // chol

         // perturb x with covariance values => make sigma point vectors
         for (int i=1; i<=L; i++) { // TODO check bounds
            x[i+0] += sqrt.col(i);
            x[i+L] -= sqrt.col(i);
         }

         // copy back to sigma_states and step forward in time
         // #pragma omp parallel for
         for (int i=0; i<N; i++) {
            //mj_copyData(sigma_states[i], this->m, this->d);
            set_state(sigma_states[i], raw[i]);
            mj_step(m, sigma_statesi]);
         }

         for (int i=0; i<N; i++) {
            x_t += W_s[i] * x[i];
         }
         for (int i=0; i<N; i++) {
            VectorXd x_i(x[i] - x_t);
            P_t += W_c[i] * (x_i * x_i.transpose());
         }

      }

   private:
      int L;
      int N;
      double alpha;
      double beta;
      double lambda;
      std::vector<double *> raw; // raw data storage pointed to by x
      std::vector<VectorXd *> x;
      VectorXd * x_t;
      //std::vector<MatrixXd *> P;
      MatrixXd * P;
      MatrixXd * P_t;
      std::vector<mjData *> sigma_states;
};


