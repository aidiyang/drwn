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

PF::PF(mjModel *m, mjData * d, double _eps, int _numPart, int _nResamp, double _diag, double _snoise,
        double _cnoise, double* P_covar, int _render, int _threads, bool _debug) : Estimator(m, d) { 
    //Random Uniform initial particles
    std::default_random_engine gen;
    eps = _eps;
    rand = std::uniform_real_distribution<double> (-eps, eps);

    numPart = _numPart;
    snoise = _snoise;
    cnoise = _cnoise;
    debug = _debug;
    nResamp = _nResamp;
    num_diag = _diag;
    render = _render;
    threads = _threads;
   
    printf("nq: %i \n", nq);
    printf("nv: %i \n", nv);
    printf("ns: %i \n", ns);
    printf("nu: %i \n", nu);
    printf("Crtl noise: %f \n", cnoise);

    //Set up sigma_states
    sigma_states.resize(render + 1);
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

    p_states = new double[(numPart) * (nq + nv)];
    p_sensors = new double[(numPart) * (ns)];
    p_ctrl = new double[(numPart) * (nu)];
    //Put in rest of particles (perturbed)
    for (int i = 0; i < numPart; i++){
        //Make rand vector
        for (int j = 0; j < nq; j++) {
            p_states[(nq+nv)*i + j] = this->d->qpos[j] + rand(gen);
        }
        for (int j = 0; j < nv; j++) {
            p_states[(nq+nv)*i + nq + j] = this->d->qvel[j] + rand(gen);
        }
    }

    mu = VectorXd::Zero(nq+nv);
    for (int i = 0; i < nq; i++) { 
        mu(i) = this->d->qpos[i];
    }
    for (int i = 0; i < nv; i++) {
        mu(nq+i) = this->d->qvel[i];
    }
    covar = MatrixXd::Zero(nq+nv, nq+nv);
    for (int i = 0; i < nq+nv; i++) {
        covar(i, i) = 1e-1;
    }
    sensCovar = MatrixXd::Zero(ns, ns);
    crossCovar = MatrixXd::Zero(nq+nv, ns);
    Kgain = MatrixXd::Zero(nq+nv, ns);
    estSens = VectorXd::Zero(ns);
    sensor = VectorXd::Zero(ns);
    
    //Initial weights
    weights = VectorXd::Constant(numPart, 1, 0);    
    sensDiff = MatrixXd::Zero(ns, numPart);
    snsr_weights = new double[ns];
    S_cov = new double[ns];
    for (int i=0; i<ns; i++) {
        snsr_weights[i] = 1.0;
        S_cov[i] = 1.0;
    }

    //Get sensor weights
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

    partCount = VectorXd::Zero(numPart);
    // resample = new std::set<int>;

    qposResamp = new std::normal_distribution<double>[nq];
    qvelResamp = new std::normal_distribution<double>[nv];
    s_rng = std::mt19937(494949);

    diff = VectorXd::Zero(ns);
    rms = VectorXd::Zero(numPart);

    thread_handles = new std::future<double>[threads];
    thread_datas = new mjData*[threads];
    for (int i = 0; i < threads; i++) {
        thread_datas[i] = mj_makeData(m);   
    }

    // hella faster
    // m->opt.iterations = 5; 
    // m->opt.tolerance = 1e-5; 

    minQfrc = 100;
    maxQfrc = 0;

};

PF::~PF() {
    for (int i = 0; i < threads; i++) {
        mj_deleteData(thread_datas[i]);
    }
    for (int i = 0; i < render + 1; i++) {
        mj_deleteData(sigma_states[i]);
    }
    delete[] rd_vec;
    delete[] qposResamp;
    delete[] qvelResamp;
    delete[] p_states;
    delete[] p_sensors;
    delete[] p_ctrl;
    delete[] thread_handles;
};

//Parallel function to step for the particles
double PF::forward_particles(mjModel* m, mjData* data, double* ctrl, int s, int e, int num) {
    //printf("Launching thread num %i \n", num);
    double t1 = util::now_t();
    mju_copy(data->ctrl, ctrl, nu);
    for (int i = s; i < e; i++) {
        mju_copy(data->qpos, p_states + (nq+nv)*i, nq);
        mju_copy(data->qvel, p_states + (nq+nv)*i + nq, nv);
        mju_copy(data->ctrl, p_ctrl + (nu)*i, nu);
        // mj_forwardSkip(m, d, 0, 1);
        // mj_Euler(m, data);
        mj_step(m, data);
        mj_forward(m, data);
        mju_copy(p_sensors + (ns)*(i), data->sensordata, ns);
        mju_copy(p_states + (nq+nv)*i, data->qpos, nq);
        mju_copy(p_states + (nq+nv)*i + nq, data->qvel, nv);
    }
    return util::now_t() - t1;
}

void PF::predict_correct(double * ctrl, double dt, double* sensors, double* conf) {
    //PREDICT
    std::cout<<"predict correct"<<std::endl;

    IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");

    m->opt.timestep = dt;
    //Add control noise
    for (int i = 0; i < numPart; i++) {
        for (int j = 0; j < nu; j++) {
            p_ctrl[(nu)*i + j] = ctrl[j] + crand(gen);
        }
    }

    // if(debug) {
    //     std::cout << "ctrl" << "\n";
    //     for (int i = 0; i < nu; i++) {
    //         std::cout << ctrl[i] << "\n";
    //     }
    // }

    //Step particles forward
    double t1 = util::now_t();
    for (int i = 0; i < threads; i++) {
        int s = i * numPart / threads;
        int e = (i+1) * numPart / threads;
        if (i == threads - 1) {
            e = numPart;
        }
        thread_handles[i] = std::async(std::launch::async, &PF::forward_particles, this, m, thread_datas[i], ctrl, s, e, i);
    }

    for (int i = 0; i < threads; i++) {
        int s = i * numPart / threads;
        int e = (i+1) * numPart / threads;
        if (i == threads - 1) {
            e = numPart;
        }
        double t = thread_handles[i].get();
        printf("Thread %d: %d - %d took %f\n", i, s, e, t);
    }
    printf("Total time: %f\n", util::now_t()-t1);

    // Add noise to predicted sensors
    // for (int i = 0; i < numPart; i++) {
    //     for (int j = 0; j < ns; j++) {
    //         p_sensors[ns*i + j] += rd_vec[j](s_rng);
    //     }
    // }

    //Compare with predicted sensors with observation
    for (int i = 0; i < numPart; i++) {
        for (int j = 0; j < ns; j++) {
            diff(j) = sensors[j] - p_sensors[ns*i + j];
            // diff(j) = std::abs(snsr_weights[j]*(p_sensors[ns*i + j] - sensors[j]));
        }
        rms(i) = diff.norm();
        sensDiff.block(0, i, ns, 1) = diff;
    }
    
    //Calc weights and normalize
    weights = rms.cwiseInverse();
    clampWeights(weights);      //Clamp weights
    weights /= weights.sum();

    resample.clear();
    //Get n biggest rms particles (equiv to n smallest weights)
    for (int i = 0; i < nResamp; i++) {
        int min;
        weights.minCoeff(&min);
        weights[min] = 10;
        //resample[i] = min;
        resample.insert(min);
    }
    //Particles being resampled are not used in mu calculations and correction calculations
    for (std::set<int>::iterator it = resample.begin(); it!=resample.end(); ++it) {
        weights[*it] = 0;
    }
    weights /= weights.sum();       //Renormalize weights
    sumWeights = VectorXd::Zero(numPart);
    sumWeights(0) = weights(0);
    for (int i = 1; i < numPart; i++) {
        sumWeights(i) = sumWeights(i-1) + weights(i); 
    }    

    //Posterior is weighted average of particles
    mu.setZero();
    for (int i = 0; i < numPart; i++) {
        for (int j = 0; j < nq; j++) {
            mu(j) += weights[i] * p_states[(nq+nv)*i + j];
        }
        for (int j = 0; j < nv; j++) {
            mu(j+nq) += weights[i] * p_states[(nq+nv)*i + nq + j];
        }        
    }

    //Calc predicted covar
    covar.setZero();
    for (int i = 0; i < numPart; i++) {
        VectorXd temp = VectorXd::Zero(nq+nv);
        for (int j = 0; j < nq; j++) {
            temp(j) = p_states[(nq+nv)*i + j];
        }
        for (int j = 0; j < nv; j++) {
            temp(j + nq) = p_states[(nq+nv)*i + nq + j];
        }
        covar += weights[i] * (temp - mu) * ((temp - mu).transpose());
    }

    //CORRECTION UPDATE

    //Calc covariances
    sensCovar.setZero();
    crossCovar.setZero();
    estSens.setZero();
    for (int j=0; j < ns; j++) {
        sensor(j) = sensors[j];
    }
    for (int i = 0; i < numPart; i++) {
        for (int j=0; j < ns; j++) {
            estSens(j) += weights[i] * p_sensors[ns*i + j];
        }
    }
    for (int i = 0; i < numPart; i++) {
        VectorXd temp = VectorXd::Zero(nq+nv); 
        for (int j = 0; j < nq+nv; j++) {
            temp[j] = p_states[(nq+nv)*i + j];
        }        
        VectorXd tempsens = VectorXd::Zero(ns);
        for (int j=0; j < ns; j++) {
            tempsens[j] = p_sensors[ns*i + j];
        }
        sensCovar += weights[i]*(tempsens - estSens)*((tempsens - estSens).transpose());
        crossCovar += weights[i]*(temp-mu)*((tempsens-estSens).transpose());
    }
    Kgain = crossCovar * (sensCovar + S_add).inverse();      
    covar = covar - Kgain*sensCovar*Kgain.transpose();
    VectorXd diag = covar.diagonal() + P_add;
    mu = mu + Kgain*(sensor - estSens);
    set_data(d, &mu);

    // Make distributions to resample from
    for (int i=0; i<nq; i++) {
        qposResamp[i].param(std::normal_distribution<>::param_type(0, diag[i]));
    }
    for (int i=0; i<nv; i++) {
        qvelResamp[i].param(std::normal_distribution<>::param_type(0, diag[nq+i]));
    }

    if(debug) {
        //std::cout << "sensDiff\n" << sensDiff.format(CleanFmt) << "\n";
        // std::cout << "rms\n" << rms.transpose().format(CleanFmt) << "\n";
        // std::cout << "Weights\n" << weights.transpose().format(CleanFmt) << "\n";
        // std::cout << "SumWeights\n" << sumWeights.transpose().format(CleanFmt) << "\n";
        // std::cout << "Kgain\n" << Kgain.transpose().format(CleanFmt) << "\n";
        // std::cout << "Covariance\n" << covar.format(CleanFmt) << "\n";
        // std::cout << "Sensor Covariance\n" << (sensCovar+S_add).format(CleanFmt) << "\n";
        // std::cout << "Inverse SensCovar\n" << ((sensCovar+S_add).inverse()).format(CleanFmt) << "\n";
        //std::cout << "KgainSnsrCovarKgain\n" << (Kgain*sensCovar*Kgain.transpose()).format(CleanFmt) << "\n";]
        // std::cout << "Cross Covariance\n" << crossCovar.format(CleanFmt) << "\n";
        // std::cout << "Innovation\n" << (sensor - estSens).format(CleanFmt) << "\n";
        // std::cout << "mu Change\n" << (Kgain*(sensor - estSens)).format(CleanFmt) << "\n";        
        // std::cout << "diag\n" << diag.format(CleanFmt) << "\n";
        // std::cout << "mu \n" << mu.format(CleanFmt) << "\n";
        // std::cout << "Innovation\n" << (sensor - estSens).norm() << "\n";

    }

    //Resample
    for (std::set<int>::iterator it = resample.begin(); it != resample.end(); it++) {
        resampPart(*it);
    }

};

void PF::resampPart(int index) {
    for (int i = 0; i<nq; i++) {
        this->p_states[(nq+nv)*(index) + i] = d->qpos[i] + qposResamp[i](gen);
    }
    for (int i = 0; i<nv; i++) {
        this->p_states[(nq+nv)*(index) + nq + i] = d->qvel[i] + qvelResamp[i](gen);
    }
}

void PF::printState() {
    for (int j = 0; j < nq; j++) {
        printf("[");
        for(int i = 0; i < numPart; i++) {
            printf("%f ,", p_states[(nq+nv)*i + j]);
        }
        printf("]");
        printf("\n");
    }
    for (int j = 0; j < nv; j++) {
        printf("[");
        for(int i = 0; i < numPart; i++) {
            printf("%f ,", p_states[(nq+nv)*i + nv + j]);
        }
        printf("]");
        printf("\n");
    }
}

void PF::set_data(mjData* data, VectorXd *x) {
    mju_copy(data->qpos,   &x[0](0), nq);
    mju_copy(data->qvel,   &x[0](nq), nv);
};

void PF::clampWeights(VectorXd &weights){
    for (int i = 0; i < numPart; i++) {
        if (weights(i) > 1000000) {
            weights(i) = 1000000;
        }
    }
};

double PF::getDiff(VectorXd part, double* est) {
    double rms = 0;
    for (int i = 0; i < nq+nv; i++) {
        rms += pow(est[i] - part[i], 2);
    }
    return sqrt(rms);
};

mjData* PF::get_state() {
    return this->d;
};

std::vector<mjData*> PF::get_sigmas() {
    mju_copy(sigma_states[0]->qpos, d->qpos, nq);
    mju_copy(sigma_states[0]->qvel, d->qvel, nv);
    for (int i = 1; i < render + 1; i++) {
        mju_copy(sigma_states[i]->qpos, p_states + (nq+nv)*i, nq);
        mju_copy(sigma_states[i]->qvel, p_states + (nq+nv)*i + nq, nv);
    }
    return sigma_states;
};

