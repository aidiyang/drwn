#include "mujoco.h"
#include "kNN.h"
#include <string.h>
#include <iostream>
#include <random>
#include <functional>
#include <vector>
#include <stdlib.h>
#include <set>
#include <fstream>
#include <sstream>
#include "util_func.h"

#include <math.h>

#include <boost/program_options.hpp>

namespace po = boost::program_options;

#include <Eigen/Dense>
using namespace Eigen;
#include "H5Cpp.h"
#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

void flatMatrix(MatrixXd data, double* arr, int dim1, int dim2);
void reshapeMatrix(MatrixXd &out, double* data, int dim1, int dim2);
double getDiff(VectorXd part, double* est, int size);
bool checkConstraint(double* data, double* limit, int size);

/*
Makes a database of inputted size and saves it to a output file. If a file is inputted, the database is based off of that
inputted trajectory. Note that num_part input must be large enough to accomidate the input trajectory if one is used. 100 random controls
are applied at each timestep, so if the trajectory is n timesteps long, num_part must be at least 101*n (it also saves the timestep itself)
*/
int main (int argc, const char* argv[]) {

    std::string model_name;
    std::string output_file;
    std::string input_file;
    double eps;
    int num_part;
    bool isDarwin = false;
    IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");
    int nq, nv, ns, nu;

    //Parse arguments
    try {
        po::options_description desc("Allowed options");
        desc.add_options()
        ("help", "Usage guide")
        ("model,m", po::value<std::string>(&model_name)->required(), "Model file to load.")
        ("file,f", po::value<std::string>(&output_file)->default_value("./kNN_data.h5"), "Name of output file.")
        ("eps,e", po::value<double>(&eps)->default_value(1.0), "Gaussian amount of estimator noise to corrupt data with.")
        ("num_part,n", po::value<int>(&num_part)->default_value(10000), "Number of particles to sample")
        ("input_file,i", po::value<std::string>(&input_file)->default_value(""), "File to read if (if any)")
        ;

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    }
    catch(...) {
        std::cerr << "Unknown error!\n";
        return 0;
    }
    
    if (model_name == "../models/darwin.xml") {
        isDarwin = true;
        printf("Is Darwin!!!\n");
    }

    //Print input info
    printf("Model:\t\t\t%s\n", model_name.c_str());
    printf("Output File:\t\t%s\n", output_file.c_str());
    printf("Input File: \t\t%s\n", input_file.c_str());
    printf("Eps:\t\t\t%f\n", eps);
    printf("Number of particles:\t%d\n", num_part);

    //Load model and data
    mj_activate("mjkey.txt");
    printf("MuJoCo Activated.\n");
    char error[1000] = "could not load binary model";
    mjModel* m = mj_loadXML(model_name.c_str(), NULL, error, 1000);
    printf("Model Loaded\n");
    mjData* d = mj_makeData(m);     //Initial data (data we sample from)
    mjData* saveData = mj_makeData(m);      //Data used for mj_steps

    //Intialize data structures
    nq = m->nq;
    nv = m->nv;
    nu = m->nu;
    ns = m->nsensordata;
    int size = 0;
    std::uniform_real_distribution<double> partRandqpos = std::uniform_real_distribution<double> (-eps, eps);
    std::uniform_real_distribution<double> partRandqvel = std::uniform_real_distribution<double> (-2*eps, 2*eps);
    std::uniform_real_distribution<double> randCtrl = std::uniform_real_distribution<double> (-2, 2);
    std::default_random_engine gen;
    double* zeroCtrl = new double[nu];
    kNN* kNNdata = new kNN(nq, nv, ns, num_part);
    double* constraint_limit = new double[nv];

    double* savepos = new double[nq+nv];
    double* savenext = new double[nq+nv];
    double* savesens = new double[ns];
    double* saveSensNext = new double[ns];
    double* qfrc = new double[nv];
    for (int i=0; i<nv; i++) qfrc[i] = 0.0;
    double t1 = util::now_t();
    int counter = 0;
    int check = 0;


    //Set darwin specific fields
    if (isDarwin) {
        //Set in initial position
        printf("Loading initial crouched position\n");
        double init_qpos[26] = {0.0423, 0.0002, -0.0282, 0.0004, 0.2284, -0.0020, 0.0000, -0.4004, 0.7208, 0.2962, -0.5045, -0.8434,
         -0.2962, 0.5045, -0.0005, -0.0084, 0.6314, -0.9293, -0.5251, -0.0115, 0.0005, 0.0084, -0.6314, 0.9293, 0.5251, 0.0115};
        for (int i=0; i<nq; i++) {
            d->qpos[i] = init_qpos[i];
        }
        //Get qfrc constraint limits
        printf("Loading constraint limits\n");
        int qfrc_c_size = 0;
        double* qfrc_vec = util::get_numeric_field(m, "qfrc_constraint", &qfrc_c_size);
        if (qfrc_vec) {
            for (int i = 0; i < nv; i++) {
                constraint_limit[i] = qfrc_vec[i];
            }
        }
    }

    //Build database
    //If there is an input file, use it
    if (input_file != "") {
        std::ifstream myfile(input_file);
        std::string line, stateItem;
        if (myfile.is_open()) {
            std::getline(myfile, line);      //Get rid of header line
            while(std::getline(myfile,line)) {
                if (kNNdata->getSize() == 5000*check) {
                printf("Sampled %i particles so far\n", 5000*check);
                printf("Skipped %i particles so far\n", counter);
                check++;
                }
                // std::getline(myfile, line);     //Get next line of data
                std::stringstream ss(line);
                std::getline(ss, stateItem, ',');      //Get rid of time data
                double* trajState = new double[nq + nv];
                //Set saveData
                for (int j = 0; j < nq; j++) {
                    std::getline(ss, stateItem, ',');
                    try {
                        trajState[j] = std::stod(stateItem.c_str());
                    }catch (const std::invalid_argument&) {
                        std::cout<<"Invalid argument: " << stateItem.c_str() << std::endl;
                    }
                }
                for (int j = 0; j < nv; j++) {
                    std::getline(ss, stateItem, ',');
                    try {
                        trajState[nq + j] = std::stod(stateItem.c_str());
                    }catch (const std::invalid_argument&) {
                        std::cout<<"Invalid argument: " << stateItem.c_str() << std::endl;
                    }
                }

                //Set random controls and step forward to make new particles
                for (int j = 0; j < 100; j++) {
                    //Set random control
                    for (int k = 9; k < nu; k++) {
                        saveData->ctrl[k] = randCtrl(gen);
                    }
                    mju_copy(saveData->qpos, trajState, nq);
                    mju_copy(saveData->qvel, trajState + nq, nv);
                    mj_step(m, saveData);
                    mj_forward(m, saveData);
                    mju_copy(savepos, saveData->qpos, nq);    
                    mju_copy(savepos + nq, saveData->qvel, nv);
                    mju_copy(savesens, saveData->sensordata, ns);
                    mju_copy(saveData->ctrl, zeroCtrl, nu);     //Set ctrl to zero before stepping for nextPart
                    mj_Euler(m, saveData);
                    mj_forward(m, saveData);
                    mju_copy(savenext, saveData->qpos, nq);    
                    mju_copy(savenext + nq, saveData->qvel, nv);
                    mju_copy(saveSensNext, saveData->sensordata, ns);
                    kNNdata->savePart(savepos, savenext, savesens, saveSensNext);
                }
            }
        }
        printf("Size of database: %i\n", kNNdata->getSize());
    //Otherwise make database normally by randomly sampling in space. 
    }else {
        //Build Database
        printf("Building knn Database\n");
        // for (int i = 0; i < num_part; i++) {
        while(kNNdata->getSize() < num_part) {
            if (kNNdata->getSize() == 5000*check) {
                printf("Sampled %i particles so far\n", 5000*check);
                printf("Skipped %i particles so far\n", counter);
                check++;
            }
            for (int j = 0; j < nq; j++) {
                savepos[j] = d->qpos[j] + partRandqpos(gen);
            }
            for (int j = 0; j < nv; j++) {
                savepos[j + nq] = d->qvel[j] + partRandqvel(gen);
            }
            mju_copy(saveData->qpos, savepos, nq);
            mju_copy(saveData->qvel, savepos + nq, nv);
            mj_forward(m, saveData);
            mju_copy(savesens, saveData->sensordata, ns);
            mj_step(m, saveData); // mj_Euler(m, saveData);
            mj_forward(m, saveData);

            //If qfrc_constraint is not as expected, resample. Doesn't seem to work very well.
            if (isDarwin) {
                // if (*std::max_element(savesens, savesens + ns) > 5 || *std::max_element(saveSensNext, saveSensNext + ns) > 5) {
                //     counter++;
                //     continue;
                mju_copy(qfrc, saveData->qfrc_constraint, nv);
                double qfrc_sum = 0;
                double maxqfrc = 1e-6;
                for (int i = 0; i < nv; i++) {
                    qfrc_sum += fabs(qfrc[i]);
                    if (qfrc[i] > maxqfrc)
                    maxqfrc = qfrc[i];
                }
                // printf("max: %f %f\n", maxqfrc, qfrc_sum);
                // if (checkConstraint(qfrc, constraint_limit, nv)) {
                if (qfrc_sum > 500) {
                    counter++;
                } else {
                    mju_copy(savenext, saveData->qpos, nq);
                    mju_copy(savenext + nq, saveData->qvel, nv);
                    mju_copy(saveSensNext, saveData->sensordata, ns);
                    kNNdata->savePart(savepos, savenext, savesens, saveSensNext);
                }
            } else {
                mju_copy(savenext, saveData->qpos, nq);
                mju_copy(savenext + nq, saveData->qvel, nv);
                mju_copy(saveSensNext, saveData->sensordata, ns);
                kNNdata->savePart(savepos, savenext, savesens, saveSensNext);

                //Code to ensure particle diversity. If generated particle is too closed to one that is already saved,
                //throw it away and generate another one.
                // if (getDiff(kNNdata->findPart(savesens), savepos, nq+nv) > 0.3) {
                //     kNNdata->savePart(savepos, savenext, savesens, saveSensNext);
                // } else {
                //     counter++;
                //     eps += .01;
                //     partRandqpos.param(std::uniform_real_distribution<>::param_type(-eps, eps));
                //     partRandqvel.param(std::uniform_real_distribution<>::param_type(-eps, eps));
                //     printf("Didn't sample particle\n");
                // }   
            }
        }
    }
    printf("Number of particles skipped: %i\n", counter);
    printf("Number of particles saved: %i\n", kNNdata->getSize());
    printf("Time to build database: %f ms\n", util::now_t() - t1);
    //Save database to h5 file
    kNNdata->saveData(output_file);
    printf("Saved data to file\n\n");
}

double getDiff(VectorXd part, double* est, int size) {
    double rms = 0;
    for (int i = 0; i < size; i++) {
        rms += pow(est[i] - part(i), 2);
    }
    return sqrt(rms);
}

//Checks if the inputted violates the inputted limits (data is greater than limits). Returns true if
//constraint is violated.
bool checkConstraint(double* data, double* limits, int size) {
    for (int i = 0; i < size; i++) {
        if (std::abs(data[i]) > limits[i]) {
            return true;
        }
    }
    return false;
}

