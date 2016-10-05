#include "mujoco.h"
#include <string.h>
#include <iostream>
#include <random>
#include <functional>
#include <vector>
#include <stdlib.h>

#ifndef __APPLE__
//#include <omp.h>
#endif
#include <math.h>

#include <Eigen/Dense>


//#include <boost/random.hpp>
//#include <boost/random/normal_distribution.hpp>

#include "estimator.h"
//#include "derivative.cpp"

using namespace Eigen;

/*const int nq;
const int nv;
double** deriv[nq+nv][nq+nv] = 0;*/
//std::vector<std::vector<double>> deriv;
//mjtNum* deriv = 0;          // dynamics derivatives (6*nv*nv):
double eps = 1e-6;          // finite-difference epsilon

class EKF : public Estimator {
   public:
      int L;
      MatrixXd sigma;
      mjData* d2;
      //std::vector<mjData*> sigma_states;


      EKF(mjModel *m, mjData * d, double _noise, double _dnoise, double _tol, double _diag,
            bool debug = false, int threads = 1) : Estimator(m, d) {
         ctrl_state = false;
         nq = this->m->nq;
         nv = this->m->nv;
         
         //nsensordata = m->nsensordata;
         L = nq+nv;
         diag = _diag;
         noise = _noise;
         dnoise = _dnoise;
         stddev = mj_makeData(this->m);
         covar = mj_makeData(this->m);
         this->NUMBER_CHECK = debug;
         sigma_states.push_back(mj_makeData(this->m));
         predmu = mj_makeData(this->m);
         mj_copyData(predmu, this->m, this->d);
         //predmu = this->d;
         d2 = mj_makeData(this->m);
         mj_copyData(d2, this->m, this->d);

         Rt = MatrixXd::Identity(ns, ns)*diag;
         sigma = MatrixXd::Identity(L, L);
         // sigma.setIdentity(l,l);
         std::random_device rd;
         rng = std::mt19937(rd());
         nd = std::normal_distribution<>(0, noise);
         printf("Sensor noise: %f\n", noise);
         dnd = std::normal_distribution<>(0, dnoise); //Normal dist for derivatives
         printf("Derivative sensor noise: %f\n", dnoise);
      };
      
      ~EKF() {
         mj_deleteData(stddev);
         mj_deleteData(covar);
         mj_deleteData(d2);
         for (int i = 0; i < sigma_states.size(); i++){
            mj_deleteData(sigma_states[i]);
         }
      };

      void predict_correct(double * ctrl, double dt, double* sensors, double* conf) {
         //Add noise to sensors:
         VectorXd sensor_noise(m->nsensordata);
         if (noise > 0) {
            for (int i =0; i<m->nsensordata; i++) {
               sensor_noise[i] = nd(rng);
               sensors[i] += sensor_noise[i];//nd(rng);
            }
         }

         //Prediction
         mju_copy(predmu->ctrl, ctrl, nu);

         m->opt.timestep = dt;
         //mj_step(m, predmu);
         mj_forward(m, predmu);
         mj_Euler(m, predmu);
         bool print_stuff = false;
         if (print_stuff) {
            printf("predMu\n");
            for (int i=0; i<nq; i++) { printf("%1.4f ", predmu->qpos[i]); }
            for (int i=0; i<nv; i++) { printf("%1.4f ", predmu->qvel[i]); }
            printf("\n"); 
         }

         m->opt.timestep = eps;
         MatrixXd Gt = get_deriv(m, predmu, d2);
         MatrixXd Vt = get_ctrlderiv(m, predmu, d2);

         m->opt.timestep = dt;

         //Process noise just identity? Or use Qt = Vt*Mt*Vt.transpose(), where Vt is jacob of state w.r.t control, Mt is "motion noise"
         MatrixXd Qt = MatrixXd::Identity(nu, nu)*1e-6;
         MatrixXd predSigma = Gt*sigma*Gt.transpose() + Vt*Qt*Vt.transpose();//MatrixXd::Identity(L,L)*diag;
         //Initial covar = identity

         MatrixXd testMu = MatrixXd::Zero(L, 1);
         for (int i = 0; i < nq; i++) {
               testMu(i) = predmu->qpos[i];
         }
         for (int i = 0; i < nv; i++) {
               testMu(i+nq) = predmu->qvel[i];
         }
         MatrixXd testGt = Gt * testMu;

         //Correction
         mj_forward(m, predmu);
         //mj_sensor(m, predmu);

         //Add sensor noise
         if (noise > 0){     
            for (int i =0; i<m->nsensordata; i++) {
               predmu->sensordata[i] += sensor_noise[i];;
            }
         }
         
         //Set predsens
         Map<VectorXd> s(sensors, ns);
         Map<VectorXd> pred_s(predmu->sensordata, ns);

         //Compute sens deriv
         MatrixXd Ht = get_sensderiv(m, predmu, d2);

         //Rt = MatrixXd::Identity(m->nsensordata, m->nsensordata)*diag; //Observation (sensor) noise just identity?
         MatrixXd St = (Ht*predSigma*Ht.transpose()+Rt); // pzadd
         //MatrixXd tmp = MatrixXd::Identity(m->nsensordata, m->nsensordata);
         //HouseholderQR<MatrixXd> qr(St);
         //MatrixXd Stinv = qr.solve(tmp);  //Compute inverse of St using QR solver
         //MatrixXd Stinv = St.colPivHouseholderQr().solve(tmp);
         MatrixXd Stinv = St.inverse();
         MatrixXd Kt = predSigma*Ht.transpose()*Stinv;
         sigma = (MatrixXd::Identity(L, L) - Kt*Ht)*predSigma;
         sigma = sigma + MatrixXd::Identity(L, L)*diag;

         if (print_stuff) {
            IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");
            std::cout<<"PredSigma:\n"<<predSigma.format(CleanFmt)<<"\n";
            std::cout<<"Sigma:\n"<<sigma.format(CleanFmt)<<"\n";
            std::cout<<"Gt:\n"<<Gt.format(CleanFmt)<<"\n";
            std::cout<<"Vt:\n"<<Vt.format(CleanFmt)<<"\n";
            std::cout<<"Ht:\n"<<Ht.format(CleanFmt)<<"\n";
            std::cout<<"Kt:\n"<<Kt.format(CleanFmt)<<"\n";

            std::cout<<"Gt Test:\n"<<testGt.format(CleanFmt)<<"\n";

            std::cout<<"Rt:\n"<<Rt.format(CleanFmt)<<"\n";
            std::cout<<"St:\n"<<St.format(CleanFmt)<<"\n";
            std::cout<<"St_inv:\n"<<Stinv.format(CleanFmt)<<"\n";

            std::cout<<"St*St_inv:\n"<<(St*Stinv).format(CleanFmt)<<"\n";

            std::cout<<"S-Sensors:\n"<<s.transpose().format(CleanFmt)<<"\n";
            std::cout<<pred_s.transpose().format(CleanFmt)<<"\n";
            std::cout<<(s-pred_s).transpose().format(CleanFmt)<<"\n";
            std::cout<<"Added sensor noise"<<sensor_noise.transpose().format(CleanFmt)<<"\n";
         }

         VectorXd mu(L);
         mu = Kt*(s-pred_s);
         for (int i = 0; i<nq; i++) {
            mu[i] += predmu->qpos[i];
         }
         for (int i = 0; i<nv; i++) {
            mu[i+nq] += predmu->qvel[i];
         }

         set_data(predmu, &mu);
         
         mj_copyData(sigma_states[0], m, predmu);
         /*mju_copy(sigma_states[0]->qpos, predmu->qpos, nq);
         mju_copy(sigma_states[0]->qvel, predmu->qvel, nv);*/
         //std::cout << "Rt\t" << Rt.rows() << "\t"<< Rt.cols() << std::endl;
         //std::cout << Rt.diagonal().transpose() << std::endl;
      }

      MatrixXd get_sensderiv(const mjModel* m, const mjData* dmain, mjData* d) {
         int nv = m->nv;
         int nq = m->nq;
         int nsensordata = m->nsensordata;

         // allocate stack space for result at center
         mjtNum* center = mj_stackAlloc(d, nsensordata);

         // copy state and control from dmain to thread-specific d
         d->time = dmain->time;
         mju_copy(d->qpos, dmain->qpos, m->nq);
         mju_copy(d->qvel, dmain->qvel, m->nv);
         //mju_copy(d->qacc, dmain->qacc, m->nv);
         //mju_copy(d->qfrc_applied, dmain->qfrc_applied, m->nv);
         //mju_copy(d->xfrc_applied, dmain->xfrc_applied, 6*m->nbody);
         mju_copy(d->ctrl, dmain->ctrl, m->nu);

         //Compute and copy center point
         mj_forward(m, d);
         //mj_forwardSkip(m, d, 0);
         //mj_sensor(m, d);
         mju_copy(center, d->sensordata, nsensordata);

         //Reset data for perturb
         d->time = dmain->time;
         mju_copy(d->qpos, dmain->qpos, m->nq);
         mju_copy(d->qvel, dmain->qvel, m->nv);
         mju_copy(d->ctrl, dmain->ctrl, m->nu);

         MatrixXd deriv(nsensordata, L);

         // finite-difference over position: skip = 2
         for( int i=0; i<nq; i++ ) {
            d->qpos[i] += eps;
            d->time = dmain->time;

            // compute column i of derivative 0
            mj_forwardSkip(m, d, 0, 0);
            //mj_sensor(m, d);
            for( int j=0; j<ns; j++ ) {
               deriv(j, i) = (d->sensordata[j] - center[j])/eps;
            }
            d->qpos[i] = dmain->qpos[i];
            //d->qpos[i] -= eps;
         }

         // finite-difference over velocity: skip = 1
         for( int i=0; i<nv; i++ ) {
            // perturb velocity
            d->qvel[i] += eps;
            d->time = dmain->time;

            // compute column i of derivative 1
            mj_forwardSkip(m, d, 0, 0);
            //mj_sensor(m, d);
            for( int j=0; j<ns; j++ ) {
               deriv(j, i+nq) = (d->sensordata[j] - center[j])/eps;
            }
            d->qvel[i] = dmain->qvel[i];
         }
         return deriv;
      }

      MatrixXd get_deriv(const mjModel* m, const mjData* dmain, mjData* d) {
         int nv = m->nv;
         int nq = m->nq;
         // allocate stack space for result at center
         mjtNum* center = mj_stackAlloc(d, L);

         // copy state and control from dmain to thread-specific d
         d->time = dmain->time;
         mju_copy(d->qpos, dmain->qpos, m->nq);
         mju_copy(d->qvel, dmain->qvel, m->nv);
         //mju_copy(d->qacc, dmain->qacc, m->nv);
         //mju_copy(d->qfrc_applied, dmain->qfrc_applied, m->nv);
         //mju_copy(d->xfrc_applied, dmain->xfrc_applied, 6*m->nbody);
         mju_copy(d->ctrl, dmain->ctrl, m->nu);

         //Compute and copy center point
         mj_forward(m, d);
         mj_Euler(m, d);
         mju_copy(center,    d->qpos, nq);
         mju_copy(center+nq, d->qvel, nv);
      
         //Reset data for perturb
         d->time = dmain->time;
         mju_copy(d->qpos, dmain->qpos, m->nq);
         mju_copy(d->qvel, dmain->qvel, m->nv);
         mju_copy(d->ctrl, dmain->ctrl, m->nu);

         MatrixXd deriv(L, L);
        
         // finite-difference over position: skip = 2
         for( int i=0; i<nq; i++ ) {
            d->qpos[i] += eps;
            d->time = dmain->time;
            //mj_forwardSkip(m, d, 0);

            // compute column i of derivative 0
            mj_forwardSkip(m, d, 0, 1);
            mj_Euler(m, d);
            for( int j=0; j<nq; j++ ) {
               deriv(j, i) = (d->qpos[j] - center[j])/eps;
            }
            for( int j=0; j<nv; j++ ) {
               deriv(j+nq, i) = (d->qvel[j] - center[j+nq])/eps;
            }
            // undo perturbation
            mju_copy(d->qpos, dmain->qpos, m->nq);
            mju_copy(d->qvel, dmain->qvel, m->nv);
         }

         // finite-difference over velocity: skip = 1
         for( int i=0; i<nv; i++ ) {
            // perturb velocity
            d->qvel[i] += eps;
            d->time = dmain->time;
            
            // compute column i of derivative 1
            mj_forwardSkip(m, d, 0, 1);
            mj_Euler(m, d);  
            for( int j=0; j<nq; j++ ) {
               deriv(j, i+nq) = (d->qpos[j] - center[j])/eps;
            }
            for( int j=0; j<nv; j++ ) {
               deriv(j+nq, i+nq) = (d->qvel[j] - center[j+nq])/eps;
            }
            // undo perturbation
            mju_copy(d->qpos, dmain->qpos, m->nq);
            mju_copy(d->qvel, dmain->qvel, m->nv);
         }
         return deriv;
      }

      MatrixXd get_ctrlderiv(const mjModel* m, const mjData* dmain, mjData* d) {
         int nv = m->nv;
         int nq = m->nq;
         int nu = m->nu;

         // allocate stack space for result at center
         mjtNum* center = mj_stackAlloc(d, L);

         // copy state and control from dmain to thread-specific d
         d->time = dmain->time;
         mju_copy(d->qpos, dmain->qpos, m->nq);
         mju_copy(d->qvel, dmain->qvel, m->nv);
         //mju_copy(d->qacc, dmain->qacc, m->nv);
         //mju_copy(d->qfrc_applied, dmain->qfrc_applied, m->nv);
         //mju_copy(d->xfrc_applied, dmain->xfrc_applied, 6*m->nbody);
         mju_copy(d->ctrl, dmain->ctrl, m->nu);

         //Copy and compute center point
         mj_forward(m, d);
         mj_Euler(m, d);
         mju_copy(center,    d->qpos, nq);
         mju_copy(center+nq, d->qvel, nv);

         //Reset data for perturbd->time = dmain->time;
         mju_copy(d->qpos, dmain->qpos, m->nq);
         mju_copy(d->qvel, dmain->qvel, m->nv);
         mju_copy(d->ctrl, dmain->ctrl, m->nu);

         MatrixXd deriv(L, nu);

         for( int i=0; i<nu; i++ ) {
            d->ctrl[i] += eps;
            d->time = dmain->time;
            mj_forwardSkip(m, d, 0, 1);   //Or just do mj_forward?
            mj_Euler(m, d);
            // compute column i of derivative 0
            for( int j=0; j<nq; j++ ) {
               deriv(j, i) = (d->qpos[j] - center[j])/eps;
            }
            for( int j=0; j<nv; j++ ) {
               deriv(j+nq, i) = (d->qvel[j] - center[j+nq])/eps;
            } 

            // undo perturbation
            //mju_copy(d->ctrl, dmain->ctrl, nu);
            mju_copy(d->qpos, dmain->qpos, m->nq);
            mju_copy(d->qvel, dmain->qvel, m->nv);
            d->ctrl[i] = dmain->ctrl[i];
         }

         return deriv;
      }

      void set_data(mjData* data, VectorXd *x) {
         mju_copy(data->qpos,   &x[0](0), nq);
         mju_copy(data->qvel,   &x[0](nq), nv);
      }

      mjData* get_state() {
            return this->predmu;
      }

      mjData* get_stddev() {
         VectorXd var = sigma.diagonal();
         if (NUMBER_CHECK) {
            std::cout<<"sigma diag:\n";
            std::cout<< var.transpose() << std::endl;
            std::cout<<"\nRt diag:\n";
            std::cout<< Rt.diagonal().transpose() << std::endl;
         }
         mju_copy(stddev->qpos, &(var(0)), nq);
         mju_copy(stddev->qvel, &(var(nq)), nv);
         if (ctrl_state) mju_copy(stddev->ctrl, &(var(nq+nv)), nu);
         else mju_copy(stddev->ctrl, d->ctrl, nu);
         
         var = Rt.diagonal();
         mju_copy(stddev->sensordata, &(var(0)), ns);

         return stddev;
      }

      MatrixXd get_covar() {
         return sigma;
      }

      std::vector<mjData*> get_sigmas() {
         return sigma_states;
      }


   private:
      //MatrixXd sigma;
      double diag;
      double noise;
      double dnoise;
      MatrixXd Rt;
      mjData* stddev;
      mjData* covar;
      mjData* predmu;
      //mjData* d2;
      bool NUMBER_CHECK;
      bool ctrl_state;
      std::vector<mjData *> sigma_states;
      //std::random_device rd;
      std::mt19937 rng;
      std::normal_distribution<> nd;
      std::normal_distribution<> dnd;





};







