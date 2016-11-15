#include "mujoco.h"
#include <string.h>
#include <iostream>
#include <random>
#include <functional>
#include <vector>
#include <stdlib.h>

#include <math.h>

#include <Eigen/Dense>

#include "estimator.h"

using namespace Eigen;

PF::PF(mjModel *m, mjData * d, double _eps, int _numPart, double _snoise, double _cnoise, double* P_covar, double _std_dev, bool _debug) : Estimator(m, d) { 
    //Random Uniform initial particles
    std::default_random_engine gen;
    eps = _eps;
    rand = std::uniform_real_distribution<double> (-eps, eps);
    resamp= std::uniform_real_distribution<double> (0.0, 1.0);

    numPart = _numPart;
    snoise = _snoise;
    cnoise = _cnoise;
    debug = _debug;
    std_dev = _std_dev;
   
    P_add = VectorXd::Zero(nq+nv);
    for (int i = 0; i < nq+nv; i++) {
        P_add[i] = P_covar[i];
    }

    //sigma_states.push_back(d);
    sigma_states.resize(numPart+1);
    // for (int i=0; i<numPart; i++) {
    //     mjData* n_d = mj_makeData(this->m);
    //     mj_forward(m, n_d);
    //     sigma_states.push_back(n_d);
    // }

    // for (int i=0; i<numPart; i++) {
    //     sigma_states[i] = mj_makeData(this->m);
    // }

    //Set up sensor noise
    rd_vec = new std::normal_distribution<>[ns]; 
    for (int i=0; i<ns; i++) {
        rd_vec[i] = std::normal_distribution<>(0, snoise);
    }

    //Set up control noise
    std::normal_distribution<double> crand(0, cnoise);


    particles.resize(numPart);
    newParticles.resize(numPart);
    for (int i = 0; i<numPart; i++){
        mjData* temp = mj_makeData(m);
        //Make rand vector
        for (int i = 0; i<nq; i++) {
            temp->qpos[i] = this->d->qpos[i] + rand(gen);
        }
        for (int i = 0; i<nv; i++) {
            temp->qvel[i] = this->d->qvel[i] + rand(gen);
        }
        particles[i] = temp;
    }

    mu = VectorXd::Zero(nq+nv);
    covar = MatrixXd::Zero(nq+nv, nq+nv);

    //Initial weights
    weights = VectorXd::Constant(numPart, 1, 1/numPart);     
    sensDiff = MatrixXd::Zero(nu, numPart);
    sensNorm = VectorXd::Zero(numPart);
    
    //newParticles = MatrixXd::Zero(nq+nv, numPart);
    partCount = VectorXd::Zero(numPart);

    //qposResamp = new std::uniform_real_distribution<double>[nq];
    //qvelResamp = new std::uniform_real_distribution<double>[nv];
    qposResamp = new std::normal_distribution<double>[nq];
    qvelResamp = new std::normal_distribution<double>[nv];
    s_rng = std::mt19937(494949);

    diff = VectorXd::Zero(ns);
    rms = VectorXd::Zero(numPart);
};

PF::~PF() {
    mj_deleteData(d);
    mj_deleteModel(m);
    mj_deleteData(sigma_states[0]);
    for (int i = 0; i<numPart; i++) {
        mj_deleteData(particles[i]);
        mj_deleteData(newParticles[i]);
        mj_deleteData(sigma_states[i]);
    }
    mj_deleteData(sigma_states[numPart+1]);
    delete[] rd_vec;
    delete[] qposResamp;
    delete[] qvelResamp;
    //delete[] particles;
};

void PF::predict_correct(double * ctrl, double dt, double* sensors, double* conf) {
    std::cout<<"predict correct"<<std::endl;

    IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");

    m->opt.timestep = dt;
    //Add control noise
    // for (int i = 0; i < nu; i++) {
    //     ctrl[i] += crand(gen);
    // }
    // for (int i = 0; i < numPart; i++) {
    //     mju_copy(particles[i]->ctrl, ctrl, nu);
    // }

    if(debug) {
        std::cout << "ctrl" << "\n";
        for (int i = 0; i < nu; i++) {
            std::cout << ctrl[i] << "\n";
        }
    }

    //Step particles forward
    for (int i = 0; i < numPart; i++) {
        mj_forward(m, particles[i]);
        mj_Euler(m, particles[i]);
        mj_forward(m, particles[i]);
    }
    
    //Add noise to predicted sensors
    for (int i = 0; i < numPart; i++) {
        for (int j=0; j<ns; j++) {
            double r = rd_vec[j](s_rng);
            particles[i]->sensordata[j] += r;
        }
    }

    //Compare with predicted sensors with observation
    for (int i = 0; i < numPart; i++) {
        for (int j = 0; j < ns; j++) {
            diff(j) = (particles[i]->sensordata[j] - sensors[j]);
        }
        rms[i] = diff.norm();
        //sensDiff.block(0, i, ns, 1) = diff;
    }
    
    //Calc weights and normalize
    weights = rms.cwiseInverse();
    clampWeights(weights);      //Clamp weights
    weights /= weights.sum();       //Normalize weights
   
    sumWeights = VectorXd::Zero(numPart);
    sumWeights(0) = weights(0);
    for (int i = 1; i < numPart; i++) {
        sumWeights(i) = sumWeights(i-1) + weights(i); 
    }
    
    //Calc most likely particle
    // partCount = VectorXd::Zero(numPart);
    // for (int i = 0; i < numPart; i++) {
    //     sample = resamp(gen);
    //     for (int j = 0; j < numPart; j++) {
    //         if (sample <= sumWeights(j)) {
    //             partCount(j) += 1;
    //             j = numPart;        //Change to break;
    //         }
    //     }
    // }
    
    // //Calc posterior estimate
    // int max;
    // partCount.maxCoeff(&max);
    // particles[max]->time = d->time;
    // mju_copy(d->qpos, particles[max]->qpos, nq);
    // mju_copy(d->qvel, particles[max]->qvel, nv);

    // //Posterior is closest particle (smallest rms)
    // int close;
    // rms.minCoeff(&close);
    // mju_copy(d->qpos, particles[close]->qpos, nq);
    // mju_copy(d->qvel, particles[close]->qvel, nv);
 
    //Posterior is weighted average of particles
    mu.setZero();
    for (int i = 0; i < numPart; i++) {
        for (int j = 0; j < nq; j++) {
            mu[j] += weights[i] * particles[i]->qpos[j];
        }
        for (int j = 0; j < nv; j++) {
            mu[j+nq] += weights[i] * particles[i]->qvel[j];
        }
    }
    set_data(d, &mu);

    //Set up rand dists. for resampling
    // int big;
    // rms.maxCoeff(&big);
    // mjData* maxPart = particles[big];
    // for (int i=0; i<nq; i++) {
    //     double temp = maxPart->qpos[i];// - d->qpos[i];
    //     qposResamp[i].param(std::uniform_real_distribution<>::param_type(-temp, temp));
    //     //qposResamp[i] = std::uniform_real_distribution<>(-temp, temp);
    // }
    // for (int i=0; i<nv; i++) {
    //     double temp = maxPart->qvel[i];// - d->qvel[i];
    //     qvelResamp[i].param(std::uniform_real_distribution<>::param_type(-temp, temp));
    //     //qvelResamp[i] = std::uniform_real_distribution<>(-temp, temp);
    // }
    
    //Calc covariance
    covar.setZero();
    for (int i = 0; i < numPart; i++) {
        VectorXd temp = get_posvel(particles[i]);
        //covar += weights[i]*temp*temp.transpose();
        covar += weights[i]*(mu-temp)*((mu-temp).transpose());
    }
    VectorXd diag = covar.diagonal() + P_add;

    for (int i=0; i<nq; i++) {
        //qposResamp[i].param(std::uniform_real_distribution<>::param_type(-diag[i], diag[i]));
        qposResamp[i].param(std::normal_distribution<>::param_type(0,diag[i]));
        //qposResamp[i] = std::uniform_real_distribution<>(-diag[i], diag[i]);
    }
    for (int i=0; i<nv; i++) {
        //qvelResamp[i].param(std::uniform_real_distribution<>::param_type(-diag[i], diag[i]));
        qvelResamp[i].param(std::normal_distribution<>::param_type(0, diag[i]));
        //qvelResamp[i] = std::uniform_real_distribution<>(-diag[i], diag[i]);
    }

    //Resample
    for (int j = 0; j < numPart; j++) {
        if (partCount(j) <= 1) {
            for (int i = 0; i<nq; i++) {
                // double num = rand(gen);
                // std::cout<<"num "<< num <<std::endl;
                particles[j]->qpos[i] = d->qpos[i] + qposResamp[i](gen);
            }
            for (int i = 0; i<nv; i++) {
                // double num2 = rand(gen);
                // std::cout<<"num2 "<< num2 <<std::endl;
                particles[j]->qvel[i] = d->qvel[i] + qvelResamp[i](gen);
            }
        }
    }
    
    //Update mu
    for (int i = 0; i<nq; i++){
        mu[i] = d->qpos[i];
    }
    for (int i = 0; i<nv; i++){
        mu[i+nq] = d->qvel[i];
    }

    if(debug) {
        std::cout << "partSens\n" << partSens.format(CleanFmt) << "\n";
        std::cout << "sensDiff\n" << sensDiff.format(CleanFmt) << "\n";
        //std::cout << "totalDiff" << totalDiff.format(CleanFmt) << "\n";;
        std::cout << "rms\n" << rms.transpose().format(CleanFmt) << "\n";
        std::cout << "Weights\n" << weights.transpose().format(CleanFmt) << "\n";
        std::cout << "SumWeights\n" << sumWeights.transpose().format(CleanFmt) << "\n";
        std::cout << "Covariance\n" << covar.format(CleanFmt) << "\n";
        std::cout << "Most likely particle\n" << mu.format(CleanFmt) << "\n";
        //std::cout << "Max: " << max << "\n";
        //std::cout << "Close: " << close << "\n";
        //std::cout << "part count: " << partCount.transpose().format(CleanFmt) << "\n";
    }

};

void PF::set_data(mjData* data, VectorXd *x) {
    mju_copy(data->qpos,   &x[0](0), nq);
    mju_copy(data->qvel,   &x[0](nq), nv);
};

VectorXd PF::get_posvel(mjData* data) {
    VectorXd x = VectorXd::Zero(nq+nv);
    for (int i = 0; i < nq; i++){
        x[i] = data->qpos[i];
    }
    for (int i = 0; i < nv; i++){
        x[i+nq] = data->qvel[i];
    }
    return x;
}

void PF::clampWeights(VectorXd &weights){
    for (int i = 0; i < numPart; i++) {
        if (weights(i) > 1000000) {
            weights(i) = 1000000;
        }
    }
};

mjData* PF::get_state() {
    return this->d;
};

std::vector<mjData*> PF::get_sigmas() {
    sigma_states[0] = d;
    for (int i=0; i<numPart; i++) {
        // mju_copy(sigma_states[i]->qpos,   particles[i]->qpos, nq);
        // mju_copy(sigma_states[i]->qvel,   particles[i]->qvel, nv);
        //set_data(sigma_states[i], particles.col(i));
        sigma_states[i+1] = particles[i];
    }
    return sigma_states;
};

