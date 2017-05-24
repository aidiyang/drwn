#include "mujoco.h"
// #include "kNN.h"
#include <string.h>
#include <iostream>
#include <random>
#include <functional>
#include <vector>
#include <stdlib.h>
#include <set>
#include "util_func.h"
#include <algorithm>

#include <math.h>

#include <Eigen/Dense>

#include "estimator.h"

using namespace Eigen;

kNNPF::kNNPF(mjModel *m, mjData * d, double _eps, int _numPart, int _nResamp, double _diag, double _snoise,
        double _cnoise, double* P_covar, int _render, int _threads, bool _debug) : Estimator(m, d) { 
    //Random Uniform initial particles
    std::default_random_engine gen;
    eps = _eps;
    rand = std::uniform_real_distribution<double> (-eps, eps);
    std::uniform_real_distribution<double> partRandqpos = std::uniform_real_distribution<double> (-_diag, _diag);
    std::uniform_real_distribution<double> partRandqvel = std::uniform_real_distribution<double> (-2*_diag, 2*_diag);

    numPart = _numPart;
    snoise = _snoise;
    cnoise = _cnoise;
    debug = _debug;
    nResamp = _nResamp;
    num_diag = _diag;
    render = _render;
   
    printf("nq: %i \n", nq);
    printf("nv: %i \n", nv);
    printf("ns: %i \n", ns);
    printf("nu: %i \n", nu);
    printf("Crtl noise: %f \n", cnoise);

    kNNdata = new kNN(nq, nv, ns, 100000);

    //Set up sigma_states (for rendering in viewer)
    sigma_states.resize(render+1);
    for (int i = 0; i < render + 1; i++) {
        sigma_states[i] = mj_makeData(m);
    }

    //Set up sensor noise
    rd_vec = new std::normal_distribution<>[ns]; 
    for (int i=0; i<ns; i++) {
        rd_vec[i] = std::normal_distribution<>(0, snoise); 
    }

    //Set up control noise
    std::normal_distribution<double> crand(0, cnoise);

    kNN_states = new double[(numPart) * (nq + nv)];
    kNN_sensors = new double[(numPart) * (ns)];
    //Put in rest of particles (perturbed)
    //This is actually kinda pointless since the kNN_states are gonna be set in predict_correct anyway
    //But kept for fear of breaking working code
    for (int i = 0; i < numPart; i++){
        //Make rand vector
        for (int j = 0; j < nq; j++) {
            kNN_states[(nq+nv)*i + j] = this->d->qpos[j] + rand(gen);
        }
        for (int j = 0; j < nv; j++) {
            kNN_states[(nq+nv)*i + nq + j] = this->d->qvel[j] + rand(gen);
        }
    }

    kNN_mu = VectorXd::Zero(nq+nv);
    for (int i = 0; i < nq; i++) { 
        kNN_mu(i) = this->d->qpos[i];
    }
    for (int i = 0; i < nv; i++) {
        kNN_mu(nq+i) = this->d->qvel[i];
    }
    kNN_covar = MatrixXd::Zero(nq+nv, nq+nv);
    for (int i = 0; i < nq+nv; i++) {
        kNN_covar(i, i) = 1e-1;
    }

    kNN_sensCovar = MatrixXd::Zero(ns, ns);
    kNN_crossCovar = MatrixXd::Zero(nq+nv, ns);
    kNN_Kgain = MatrixXd::Zero(nq+nv, ns);
    kNN_estSens = VectorXd::Zero(ns);
    kNNest = VectorXd::Zero(nq+nv);
    sensor = VectorXd::Zero(ns);
    
    //Initial weights
    kNN_weights = VectorXd::Constant(numPart, 1, 0);
    sensDiff = MatrixXd::Zero(ns, numPart);
    snsr_weights = new double[ns];
    for (int i=0; i<ns; i++) {
        snsr_weights[i] = 1.0;
    }

    //Get sensor weights (Not actually used, but can use when taking sensor differences)
    double* snsr_vec = util::get_numeric_field(m, "snsr_weights", NULL);
    if (snsr_vec) {
        util::fill_sensor_vector(m, snsr_weights, snsr_vec);
        if (debug) {    
            printf("Weights for Sensors:\n");
            for (int i=0; i<ns; i++) printf("%f ", snsr_weights[i]);
            printf("\n");
        }
    }
    
    P_add = VectorXd::Zero(nq+nv);
    for (int i = 0; i < nq+nv; i++) {
        P_add[i] = P_covar[i];      //covar_diag
    }
    
    S_add = MatrixXd::Zero(ns, ns);     //regularization for inverse of snsr_cov
    double* s_add_arr = util::get_numeric_field(m, "snsr_add", NULL);
    if (s_add_arr) {
        for (int i = 0; i < ns; i++) {
            S_add(i, i) = 1e-6;//s_add_arr[i];
        }
    }

    s_rng = std::mt19937(494949);

    kNN_diff = VectorXd::Zero(ns);
    kNN_rms = VectorXd::Zero(numPart);

    // hella faster
    // m->opt.iterations = 5; 
    // m->opt.tolerance = 1e-5; 

    //Build kNN database
    set_data(d, &kNN_mu);
    printf("Building knn Database\n");
    double t1 = util::now_t();

    kNNdata->readFile("small_traj_darwin.h5");

    printf("Time to build database: %f ms\n", util::now_t() - t1);
    printf("Database size: %d\n", kNNdata->getSize());
    // kNNdata->printSmall();

};

kNNPF::~kNNPF() {
    for (int i = 0; i < render + 1; i++) {
        mj_deleteData(sigma_states[i]);
    }
    delete[] rd_vec;
    delete[] kNN_states;
    delete[] kNN_sensors;
};


void kNNPF::predict_correct(double * ctrl, double dt, double* sensors, double* conf) {
    //PREDICT
    std::cout<<"predict correct"<<std::endl;

    IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");

    m->opt.timestep = dt;
    //Add control noise
    // for (int i = 0; i < numPart; i++) {
    //     for (int j = 0; j < nu; j++) {
            // kNN_ctrl[(nu)*i + j] = ctrl[j] + crand(gen);
    //     }
    // }
    //Step forward real_data
    mju_copy(this->d->ctrl, ctrl, nu);
    mj_step(m, this->d);
    mj_forward(m, this->d);
    for (int i = 0; i < nq; i++) {      //Set kNN_mu
        kNN_mu(i) = this->d->qpos[i];
    }
    for (int i = 0; i < nv; i++) {
        kNN_mu(nq + i) = this->d->qvel[i];
    }
    //Get numPart closest particles to real_data sensors
    std::vector<double*> closeData = kNNdata->getClose(sensors, numPart);
    double* closeStates = closeData[0];
    double* closeSens = closeData[1];
    for (int i = 0; i < numPart; i++) {
        for (int j = 0; j < nq+nv; j++) {
            kNN_states[(nq+nv)*i + j] = closeStates[(nq+nv)*i + j];
        }
        for (int j = 0; j < ns; j++) {
            kNN_sensors[(ns)*i + j] = closeSens[ns*i + j];
        }
    }

    // Add noise to predicted sensors
    // for (int i = 0; i < numPart; i++) {
    //     for (int j = 0; j < ns; j++) {
    //         kNN_sensors[ns*i + j] += rd_vec[j](s_rng);
    //     }
    // }

    //Compare with predicted sensors with observation
    for (int i = 0; i < numPart; i++) {
        for (int j = 0; j < ns; j++) {
            kNN_diff(j) = sensors[j] - kNN_sensors[ns*i + j];
            // kNN_diff(j) = snsr_weights[j]*(sensors[j] - kNN_sensors[ns*i + j]);
        }
        kNN_rms(i) = kNN_diff.norm();
        sensDiff.block(0, i, ns, 1) = kNN_diff;
    }
    
    //Calc weights and normalize
    kNN_weights = kNN_rms.cwiseInverse();
    clampWeights(kNN_weights);
    kNN_weights /= kNN_weights.sum();

    kNN_resample.clear();
    //Get n biggest rms particles (equiv to n smallest weights)
    for (int i = 0; i < nResamp; i++) {
        int kNN_min;
        kNN_weights.minCoeff(&kNN_min);
        kNN_weights[kNN_min] = 10;
        //resample[i] = min;
        kNN_resample.insert(kNN_min);
    }
    for (std::set<int>::iterator it = kNN_resample.begin(); it!=kNN_resample.end(); ++it) {
        kNN_weights[*it] = 0;
    }
    
    //Renormalize weights
    kNN_weights /= kNN_weights.sum();

    //Calc predicted covar
    kNN_covar.setZero();
    for (int i = 0; i < numPart; i++) {
        VectorXd temp = VectorXd::Zero(nq+nv);
        for (int j = 0; j < nq; j++) {
            temp(j) = kNN_states[(nq+nv)*i + j];
        }
        for (int j = 0; j < nv; j++) {
            temp(j + nq) = kNN_states[(nq+nv)*i + nq + j];
        }
        kNN_covar += kNN_weights[i] * (temp - kNN_mu) * ((temp - kNN_mu).transpose());
    }

    //CORRECTION UPDATE

    //Calc covariances
    kNN_sensCovar.setZero();
    kNN_crossCovar.setZero();
    kNN_estSens.setZero();
    for (int j=0; j < ns; j++) {
        sensor(j) = sensors[j];
        kNN_estSens(j) = this->d->sensordata[j];
    }
    for (int i = 0; i < numPart; i++) {
        VectorXd kNN_temp = VectorXd::Zero(nq+nv);
        for (int j = 0; j < nq+nv; j++) {
            kNN_temp[j] = kNN_states[(nq+nv)*i + j];
        }        
        VectorXd kNN_tempsens = VectorXd::Zero(ns);
        for (int j=0; j < ns; j++) {
            kNN_tempsens[j] = kNN_sensors[ns*i + j];
        }
        kNN_sensCovar += kNN_weights[i]*(kNN_tempsens - kNN_estSens)*((kNN_tempsens - kNN_estSens).transpose());
        kNN_crossCovar += kNN_weights[i]*(kNN_temp-kNN_mu)*((kNN_tempsens - kNN_estSens).transpose());
    }

    kNN_Kgain = kNN_crossCovar * (kNN_sensCovar + S_add).inverse();     
    kNN_covar = kNN_covar - kNN_Kgain*kNN_sensCovar*kNN_Kgain.transpose();
    VectorXd kNN_diag = kNN_covar.diagonal() + P_add;
    kNN_mu = kNN_mu + kNN_Kgain*(sensor - kNN_estSens);
    set_data(this->d, &kNN_mu);

    if(debug) {
        //std::cout << "sensDiff\n" << sensDiff.format(CleanFmt) << "\n";
        // std::cout << "rms\n" << rms.transpose().format(CleanFmt) << "\n";
        // std::cout << "Weights\n" << weights.transpose().format(CleanFmt) << "\n";
        // std::cout << "Kgain\n" << Kgain.transpose().format(CleanFmt) << "\n";
        // std::cout << "Covariance\n" << covar.format(CleanFmt) << "\n";
        // std::cout << "Sensor Covariance\n" << (sensCovar+S_add).format(CleanFmt) << "\n";
        // std::cout << "Inverse SensCovar\n" << ((sensCovar+S_add).inverse()).format(CleanFmt) << "\n";
        //std::cout << "KgainSnsrCovarKgain\n" << (Kgain*sensCovar*Kgain.transpose()).format(CleanFmt) << "\n";]
        // std::cout << "Cross Covariance\n" << crossCovar.format(CleanFmt) << "\n";    
        // std::cout << "diag\n" << diag.format(CleanFmt) << "\n";
        // std::cout << "kNN_Kgain\n" << kNN_Kgain.transpose().format(CleanFmt) << "\n";
        // std::cout << "kNN_Sensor Covariance\n" << (kNN_sensCovar+S_add).format(CleanFmt) << "\n";
        // std::cout << "kNN_Cross Covariance\n" << kNN_crossCovar.format(CleanFmt) << "\n";
        // std::cout << "mu \n" << mu.format(CleanFmt) << "\n";
        // std::cout << "kNN_mu \n" << kNN_mu.format(CleanFmt) << "\n";
        // std::cout << "kNN_est \n" << kNNest.format(CleanFmt) << "\n";
        // std::cout << "kNN_mu correction \n" << (kNN_Kgain*(sensor - kNN_estSens)).format(CleanFmt) << "\n";
        // std::cout << "kNN Innovation\n" << (sensor - kNN_estSens).norm() << "\n";

    }

};

void kNNPF::printState() {
    for (int j = 0; j < nq; j++) {
        printf("[");
        for(int i = 0; i < numPart; i++) {
            printf("%f ,", kNN_states[(nq+nv)*i + j]);
        }
        printf("]");
        printf("\n");
    }
    for (int j = 0; j < nv; j++) {
        printf("[");
        for(int i = 0; i < numPart; i++) {
            printf("%f ,", kNN_states[(nq+nv)*i + nv + j]);
        }
        printf("]");
        printf("\n");
    }
}

void kNNPF::set_data(mjData* data, VectorXd *x) {
    mju_copy(data->qpos,   &x[0](0), nq);
    mju_copy(data->qvel,   &x[0](nq), nv);
};

void kNNPF::clampWeights(VectorXd &weights){
    for (int i = 0; i < numPart; i++) {
        if (weights(i) > 1000000) {
            weights(i) = 1000000;
        }
    }
};

double kNNPF::getDiff(VectorXd part, double* est) {
    double rms = 0;
    for (int i = 0; i < nq+nv; i++) {
        rms += pow(est[i] - part[i], 2);
    }
    return sqrt(rms);
};

mjData* kNNPF::get_state() {
    return this->d;
};

std::vector<mjData*> kNNPF::get_sigmas() {
    mju_copy(sigma_states[0]->qpos, this->d->qpos, nq);
    mju_copy(sigma_states[0]->qvel, this->d->qvel, nv);
    for (int i = 1; i < render + 1; i++) {
        mju_copy(sigma_states[i]->qpos, kNN_states + (nq+nv)*i, nq);
        mju_copy(sigma_states[i]->qvel, kNN_states + (nq+nv)*i + nq, nv);
    }
    return sigma_states;
};

