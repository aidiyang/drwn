#include "kNN.h"
#include "mujoco.h"
#include <string.h>
#include <iostream>
#include <random>
#include <functional>
#include <vector>
#include <stdlib.h>
#include <set>
#include <future>
#include "util_func.h"

#include <math.h>

#include <Eigen/Dense>
#include <Eigen/Eigen>

#include "H5Cpp.h"
#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

using namespace Eigen;

kNN::kNN(int _nq, int _nv, int _ns, int MAX_SIZE) {
    //MAX_SIZE is the maximum number of particles the database can store. Needed to initialize the size of matricies
    //Though could theoretically be done with dynamically sized matricies, so then no need for max size.
    this->nq = _nq;
    this->nv = _nv;
    this->ns = _ns;
    this->size = 0;
    states = MatrixXd::Zero(this->nq + this->nv, MAX_SIZE);
    nextStates = MatrixXd::Zero(this->nq + this->nv, MAX_SIZE);
    sensors = MatrixXd::Zero(this->ns, MAX_SIZE);
    nextSens = MatrixXd::Zero(this->ns, MAX_SIZE);

};

kNN::~kNN() {
    
    
}

//Save a particle in the database
bool kNN::savePart(double* particle, double* nextPart, double* sensPart, double* nextSensPart) {
    for (int i = 0; i < nq+nv; i++) {
        this->states(i, this->size) = particle[i];
        this->nextStates(i, this->size) = nextPart[i];
    }
    for (int i = 0; i < ns; i++) {
        this->sensors(i, this->size) = sensPart[i];
        this->nextSens(i, this->size) = nextSensPart[i];
    }
    this->size++;
    return true;
};

//Parallel helper function for findPart. Returns index of minimum of the inputted partition.
int kNN::parallel_findPart(double* sensorsData, int s, int e) {
    int min = 0;
    double minDiff = 0;
    for (int i = s; i < e; i++) {
        double diff = 0;
        for (int j = 0; j < this->ns; j++) {
            diff += pow((sensorsData[j] - this->sensors(j, i)), 2);
        }
        diff = sqrt(diff);
        if (i == s) {
            minDiff = diff;
            min = i;
        } else if (diff < minDiff) {
            minDiff = diff;
            min = i;
        }
    }
    return min;
}

//Returns the vector corresponding to the particle with the closest sensor values to the input
Eigen::VectorXd kNN::findPart(double* sensorsData) {
    //Splits the database into threads number of partitions, and finds the minimum of each partition. Then finds the minimum of those to find
    //the overall minimum
    int threads = 4;
    std::future<int>* thread_handles = new std::future<int>[threads];
    //Launch threads
    for (int i = 0; i < threads; i++) {
        int s = i * this->size / threads;
        int e = (i+1) * this->size / threads;
        if (i == threads - 1) {
            e = this->size;
        }
        thread_handles[i] = std::async(std::launch::async, &kNN::parallel_findPart, this, sensorsData, s, e);
    }
    //Collect threads
    int threadMin[threads];
    for (int i = 0; i < threads; i++) {
        threadMin[i] = thread_handles[i].get();
    }
    int min = 0;    //index of closest particle
    double minDiff = 0;     //minimum distance
    //Find the minimum of the threadMins
    for (int i = 0; i < threads; i++) {
        double diff = 0;
        for (int j = 0; j < this->ns; j++) {
            diff += pow((sensorsData[j] - this->sensors(j, threadMin[i])), 2);
        }
        diff = sqrt(diff);
        if (i == 0) {
            minDiff = diff;
            min = threadMin[i];        
        } else if (diff < minDiff) {
            minDiff = diff;
            min = threadMin[i];
        }
    }
    //Get particle corresponding to min
    VectorXd output = VectorXd::Zero(nq+nv);
    for (int i = 0; i < this->nq+this->nv; i++) {
        output(i) = this->states(i, min);
    }
    return output;
};

//Helper function for findNextPart. Returns an n long list of diffIndex objects (stores index of particle and its corresponding diff)
//Uses the c++ sort list function, probably pretty slow. 
std::list<diffIndex> kNN::parallel_findNext(double* particle, double* sensPart, int index, int n, int s, int e) {
    double diff;
    std::vector<diffIndex> vecSortList(n);
    int count = 0;
    for (int i = s; i < e; i++) {
        diff = 0;
        for (int j = 0; j < this->nq + this->nv; j++) {
            diff += pow((particle[j + index] - this->states(j, i)), 2);
        }
        for (int j = 0; j < this->ns; j++) {
            diff += pow((sensPart[j + index] - this->sensors(j, i)), 2);
        }
        diff = sqrt(diff);
        diffIndex temp(diff, i);
        if (count == 0) {
            vecSortList[count] = temp;
            count++;
        } else if (count < n) {
            if (diff >= vecSortList[count - 1].getDiff()) {     //diff is bigger than current biggest diff, put at end of vec
                vecSortList[count] = temp;
            }
            else {
                sort(vecSortList, temp, count + 1);     //count + 1 because adding to vecSortList
            }
            count++;
        } else {
            if (diff < vecSortList[n - 1].getDiff()) {
                sort(vecSortList, temp, n);
            }
        }
    }
    std::list<diffIndex> vecSortOut(vecSortList.begin(), vecSortList.end());
    return vecSortOut;
}

/*
Propagates the inputted particle forward in time using the database. Particle is approximated by a weighted average of the n closest saved particles
by sensor data.
NOTE: Depricated function, no longer used in the kNNPF
*/
void kNN::findNextPart(double* particle, double* sensPart, int index, int n) {
    double t1 = util::now_t();
    VectorXd nextPart = VectorXd::Zero(nq+nv);      //These can be double arrays, don't need to be vectors
    VectorXd nextSensors = VectorXd::Zero(ns);
    int threads = 4;
    std::future<std::list<diffIndex>>* thread_handles = new std::future<std::list<diffIndex>>[threads];
    std::list<diffIndex> diffList;
    //Launch threads
    for (int i = 0; i < threads; i++) {
        int s = i * this->size / threads;
        int e = (i+1) * this->size / threads;
        if (i == threads - 1) {
            e = this->size;
        }
        thread_handles[i] = std::async(std::launch::async, &kNN::parallel_findNext,
                                        this, particle, sensPart, index, n, s, e);
    }
    //Collect threads, merge all the lists together
    for (int i = 0; i < threads; i++) {
        std::list<diffIndex> temp = thread_handles[i].get();
        diffList.merge(temp);
    }
    double minRMS[n];
    VectorXd partRMS = VectorXd::Zero(n);
    //The merge function merges the lists so that the resulting list is still sorted, just need to pop off the smallest (front) particle
    //n times.
    for (int i = 0; i < n; i++) {
        diffIndex temp = diffList.front();
        diffList.pop_front();
        minRMS[i] = temp.getIndex();
        partRMS(i) = temp.getDiff();
    }
    partRMS = partRMS.cwiseInverse();   //Weights are inverse of distance
    partRMS = partRMS / partRMS.sum();  //Normalize weights
    for (int i = 0; i < n; i++) {
        nextPart += partRMS(i) * this->nextStates.col(minRMS[i]);
        nextSensors += partRMS(i) * this->nextSens.col(minRMS[i]);
    }
    //Set particle to next state
    for (int i = 0; i < this->nq+this->nv; i++) {
        particle[index + i] = nextPart(i);
    }
    for (int i = 0; i < this->ns; i++) {
        sensPart[index + i] = nextSensors(i);
    }    
}

//Helper function for getClose and getEst. Returns list of n diffIndex objects 
std::list<diffIndex> kNN::parallel_Est(double* sensors, VectorXd *rms, int n, int s, int e) {
    double t1 = util::now_t();
    double time_sort = 0;
    double my_time_sort = 0;
    double diff_time = 0;
    double diff;
    std::list<diffIndex> sortList;
    std::vector<diffIndex> vecSortList(n);
    int count = 0;
    for (int i = s; i < e; i++) {
        double diff_t1 = util::now_t();
        diff = 0;
        for (int j = 0; j < this->ns; j++) {
            diff += pow((sensors[j] - this->sensors(j, i)), 2);
        }
        diff = sqrt(diff);
        diff_time += util::now_t() - diff_t1;
        diffIndex temp(diff, i);
        //LEGACY CODE: for sorting using c++ list sort function. Kept in case own sorting function breaks
        // sortList.push_back(diffIndex(diff, i));
        // double t2 = util::now_t();
        // sortList.sort();
        // time_sort += util::now_t() - t2;
        // // printf("Sorting took %f ms\n", util::now_t() - t2);
        // if (sortList.size() > n) {
        //     sortList.pop_back();
        // }
        double t3 = util::now_t();
        if (count == 0) {
            vecSortList[count] = temp;
            count++;
        } else if (count < n) {
            if (diff >= vecSortList[count - 1].getDiff()) {     //diff is bigger than current biggest diff, put at end of vec
                vecSortList[count] = temp;
            }
            else {
                sort(vecSortList, temp, count + 1);     //count + 1 because adding to vecSortList
            }
            count++;
        } else {
            if (diff < vecSortList[n - 1].getDiff()) {
                sort(vecSortList, temp, n);
            }
        }
        my_time_sort += util::now_t() - t3;
        (*rms)(i) = diff;
    }
    std::list<diffIndex> vecSortOut(vecSortList.begin(), vecSortList.end());
    //Debugging Print Stuff
    // printf("Check vecSortList:\n");
    // int index = 0;
    // for (std::list<diffIndex>::iterator it = sortList.begin(); it != sortList.end(); ++it) {
    //     // std::cout<<vecSortList[index].getIndex()<<" ";
    //     std::cout<<vecSortList[index].getIndex() - (*it).getIndex()<<" ";
    //     index++;  
    // }
    // std::cout<<"\n";
    // printf("Total Sorting time: %f\n", time_sort);
    // printf("Total my sorting time: %f\n", my_time_sort);
    // printf("Total diff time: %f\n", diff_time);
    // printf("Thread %d took %f ms\n", s, util::now_t() - t1);
    // return sortList;
    return vecSortOut;
}

//Own sorting function for parallel Est. Faster than default c++ sort list function, i think
//Check end case before calling sort function
void kNN::sort(std::vector<diffIndex> &data, diffIndex input, int size) {
    for (int i = 0; i < size; i++) {
        if (input.getDiff() <= data[i].getDiff()) {
            //Put in input and shift rest of data
            for (int j = i; j < size; j++) {
                diffIndex temp = data[j];
                data[j] = input;
                input = temp;
            }
            break;
        }
    }
}

//Returns a state estimate based off the inputted sensors. Returns a weighted average of the n closest
//particles to the inputted sensors.
Eigen::VectorXd kNN::getEst(double* sensors, int n) {
    IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");
    double t1 = util::now_t();
    //Calculate diffs of each particle
    VectorXd checkRMS = VectorXd::Zero(this->size);
    VectorXd parallel_est = VectorXd::Zero(nq+nv);
    int threads = 4;
    std::future<std::list<diffIndex>>* thread_handles = new std::future<std::list<diffIndex>>[threads];
    std::list<diffIndex> diffList;
    for (int i = 0; i < threads; i++) {
        int s = i * this->size / threads;
        int e = (i+1) * this->size / threads;
        if (i == threads - 1) {
            e = this->size;
        }
        thread_handles[i] = std::async(std::launch::async, &kNN::parallel_Est,
                                        this, sensors, &checkRMS, n, s, e);
    }
    for (int i = 0; i < threads; i++) {
        std::list<diffIndex> temp = thread_handles[i].get();
        diffList.merge(temp);
    }
    double parallel_minRMS[n];
    VectorXd parallel_estRMS = VectorXd::Zero(n);
    for (int i = 0; i < n; i++) {
        diffIndex temp = diffList.front();
        diffList.pop_front();
        parallel_minRMS[i] = temp.getIndex();
        parallel_estRMS(i) = temp.getDiff();
    }
    parallel_estRMS = parallel_estRMS.cwiseInverse();
    parallel_estRMS = parallel_estRMS / parallel_estRMS.sum();
    for (int i = 0; i < n; i++) {
        parallel_est += parallel_estRMS(i) * states.col(parallel_minRMS[i]);
    }
    // printf("Finding the particle took %f ms\n", util::now_t() - t1);
    
    return parallel_est;
}

//The actual important function that kNNPF actually uses. Returns a vector of n double pointers of the n closest 
//particles to the inputted sensors
std::vector<double*> kNN::getClose(double* sensors, int n) {
    IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");
    double t1 = util::now_t();
    double* closeStates = new double[n*(this->nq + this->nv)];
    double* closeSens = new double[n*(this->ns)];
    std::vector<double*> data;
    //Calculate diffs of each particle
    VectorXd checkRMS = VectorXd::Zero(this->size);
    VectorXd parallel_est = VectorXd::Zero(nq+nv);
    int threads = 4;
    std::future<std::list<diffIndex>>* thread_handles = new std::future<std::list<diffIndex>>[threads];
    std::list<diffIndex> diffList;
    for (int i = 0; i < threads; i++) {
        int s = i * this->size / threads;
        int e = (i+1) * this->size / threads;
        if (i == threads - 1) {
            e = this->size;
        }
        thread_handles[i] = std::async(std::launch::async, &kNN::parallel_Est,
                                        this, sensors, &checkRMS, n, s, e);
    }
    for (int i = 0; i < threads; i++) {
        std::list<diffIndex> temp = thread_handles[i].get();
        diffList.merge(temp);
    }
    diffList.resize(n);
    for (int i = 0; i < n; i++) {
        diffIndex temp = diffList.front();
        diffList.pop_front();
        for (int j = 0; j < this->nq; j++) {
            closeStates[i*(this->nq+this->nv) + j] = this->states(j, temp.getIndex());
        }
        for (int j = 0; j < this->nv; j++) {
            closeStates[i*(this->nq+this->nv) + nq + j] = this->states(j + nq, temp.getIndex());
        }
        for (int j = 0; j < this->ns; j++) {
            closeSens[i*(this->ns) + j] = this->sensors(j, temp.getIndex());
        }
    }
    data.push_back(closeStates);
    data.push_back(closeSens);
    return data;
}

int kNN::getSize() {
    return this->size;
}

//Save database to h5 file
void kNN::saveData(std::string FILE_NAME) {
    //FLATTEN DATA MATRICES TO ARRAYS
    double* statesArr = new double[this->states.size()];
    double* nextStatesArr = new double[this->nextStates.size()];
    double* sensorsArr = new double[this->sensors.size()];
    double* nextSensArr = new double[this->nextSens.size()];
    this->flatMatrix(this->states, statesArr, this->states.rows(), this->states.cols());
    this->flatMatrix(this->nextStates, nextStatesArr, this->nextStates.rows(), this->nextStates.cols());
    this->flatMatrix(this->sensors, sensorsArr, this->sensors.rows(), this->sensors.cols());
    this->flatMatrix(this->nextSens, nextSensArr, this->nextSens.rows(), this->nextSens.cols());

    //SAVE DATA TO FILE
    hsize_t statesDimsf[1], nextStatesDimsf[1], sensorsDimsf[1], nextSensDimsf[1];      //Only 1 dimension b/c flattened to an array
    herr_t ret;
    hid_t file;
    hid_t statesPlist, nextStatesPlist, sensorsPlist, nextSensPlist; //plist = Dataset creation property list   
    hid_t statesSid, nextStatesSid, sensorsSid, nextSensSid; //Sid = dataspace ID
    hid_t statesDset, nextStatesDset, sensorsDset, nextSensDset; //Dset = dataset ID
    statesDimsf[0] = this->states.size();       
    nextStatesDimsf[0] = this->nextStates.size();
    sensorsDimsf[0] = this->sensors.size();
    nextSensDimsf[0] = this->nextSens.size();
    //Create/open file
    file = H5Fcreate(FILE_NAME.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);  
    //Create dataspace IDs
    statesSid = H5Screate_simple(1, statesDimsf, NULL);     
    nextStatesSid = H5Screate_simple(1, nextStatesDimsf, NULL);
    sensorsSid = H5Screate_simple(1, sensorsDimsf, NULL);
    nextSensSid = H5Screate_simple(1, nextSensDimsf, NULL);
    //Create dataset creation property lists
    statesPlist = H5Pcreate(H5P_DATASET_CREATE);
    nextStatesPlist = H5Pcreate(H5P_DATASET_CREATE);
    sensorsPlist = H5Pcreate(H5P_DATASET_CREATE);
    nextSensPlist = H5Pcreate(H5P_DATASET_CREATE);
    //Create datasets
    statesDset = H5Dcreate(file, "statesDset", H5T_NATIVE_DOUBLE, statesSid, H5P_DEFAULT, statesPlist, H5P_DEFAULT);
    nextStatesDset = H5Dcreate(file, "nextStatesDset", H5T_NATIVE_DOUBLE, nextStatesSid, H5P_DEFAULT, nextStatesPlist, H5P_DEFAULT);
    sensorsDset = H5Dcreate(file, "sensorsDset", H5T_NATIVE_DOUBLE, sensorsSid, H5P_DEFAULT, sensorsPlist, H5P_DEFAULT);
    nextSensDset = H5Dcreate(file, "nextSensDset", H5T_NATIVE_DOUBLE, nextSensSid, H5P_DEFAULT, nextSensPlist, H5P_DEFAULT);
    //Write data to file
    ret = H5Dwrite(statesDset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &statesArr[0]);
    ret = H5Dwrite(nextStatesDset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &nextStatesArr[0]);
    ret = H5Dwrite(sensorsDset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &sensorsArr[0]);
    ret = H5Dwrite(nextSensDset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &nextSensArr[0]);
    //Close dataspaces
    ret = H5Dclose(statesDset);
    ret = H5Dclose(nextStatesDset);
    ret = H5Dclose(sensorsDset);
    ret = H5Dclose(nextSensDset);
    //Close file
    ret = H5Fclose(file);
}

//Read in data from h5 file 
void kNN::readFile(std::string FILE_NAME) {
    herr_t ret;
    hid_t rfile;
    hid_t statesSid, nextStatesSid, sensorsSid, nextSensSid; //Sid = dataspace ID
    hid_t statesDset, nextStatesDset, sensorsDset, nextSensDset; //Dset = dataset ID
    // Get datasets of file.
    rfile = H5Fopen(FILE_NAME.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    statesDset = H5Dopen(rfile, "statesDset", H5P_DEFAULT);
    nextStatesDset = H5Dopen(rfile, "nextStatesDset", H5P_DEFAULT);
    sensorsDset = H5Dopen(rfile, "sensorsDset", H5P_DEFAULT);
    nextSensDset = H5Dopen(rfile, "nextSensDset", H5P_DEFAULT);
    //Get dataspaces of datasets
    statesSid = H5Dget_space(statesDset);
    nextStatesSid = H5Dget_space(nextStatesDset);
    sensorsSid = H5Dget_space(sensorsDset);
    nextSensSid = H5Dget_space(nextSensDset);
    //Get rank of dataspaces
    int statesRank = H5Sget_simple_extent_ndims(statesSid);
    int nextStatesRank = H5Sget_simple_extent_ndims(nextStatesSid);
    int sensorsRank = H5Sget_simple_extent_ndims(sensorsSid);
    int nextSensRank = H5Sget_simple_extent_ndims(nextSensSid);
    //Get dimensions of dataspaces
    hsize_t* statesDims = new hsize_t[statesRank];
    hsize_t* nextStatesDims = new hsize_t[nextStatesRank];
    hsize_t* sensorsDims = new hsize_t[sensorsRank];
    hsize_t* nextSensDims = new hsize_t[nextSensRank];
    ret = H5Sget_simple_extent_dims(statesSid, statesDims, NULL);
    ret = H5Sget_simple_extent_dims(nextStatesSid, nextStatesDims, NULL);
    ret = H5Sget_simple_extent_dims(sensorsSid, sensorsDims, NULL);    
    ret = H5Sget_simple_extent_dims(nextSensSid, nextSensDims, NULL);
    //Read data
    double* statesArr = new double[statesDims[0]];
    double* nextStatesArr = new double[nextStatesDims[0]];
    double* sensorsArr = new double[sensorsDims[0]];
    double* nextSensArr = new double[nextSensDims[0]];
    ret = H5Dread(statesDset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &statesArr[0]);
    ret = H5Dread(nextStatesDset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &nextStatesArr[0]);
    ret = H5Dread(sensorsDset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &sensorsArr[0]);
    ret = H5Dread(nextSensDset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &nextSensArr[0]);
    //Copy data to kNN matrices
    this->states = MatrixXd::Zero(this->nq + this->nv, statesDims[0] / (this->nq + this->nv));
    this->nextStates = MatrixXd::Zero(this->nq + this->nv, nextStatesDims[0] / (this->nq + this->nv));
    this->sensors = MatrixXd::Zero(this->ns, sensorsDims[0] / (this->ns));
    this->nextSens = MatrixXd::Zero(this->ns, nextSensDims[0] / (this->ns));
    reshapeMatrix(this->states, statesArr, this->states.rows(), this->states.cols());
    reshapeMatrix(this->nextStates, nextStatesArr, this->nextStates.rows(), this->nextStates.cols());
    reshapeMatrix(this->sensors, sensorsArr, this->sensors.rows(), this->sensors.cols());
    reshapeMatrix(this->nextSens, nextSensArr, this->nextSens.rows(), this->nextSens.cols());
    this->size = statesDims[0] / (this->nq + this->nv);
}

//Print a small portion of the current database. For debugging purposes
void kNN::printSmall() {
    IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");
    std::cout << (this->states.block(0, 0, nq+nv, 10)).format(CleanFmt) << "\n";
    printf("Size: %d\n", this->size);

}

//Flattens matrix into an array COLUMN MAJOR
//dim1 = # rows, dim2 = # columns
void kNN::flatMatrix(MatrixXd data, double* arr, int dim1, int dim2) {
    for(int i = 0; i < dim2; i++) {
        for (int j = 0; j < dim1; j++) {
            arr[j + i*dim1] = data(j, i);
        }
    }
}

//Reshape arr into Matrix of given dimensions
//dim1 = # rows, dim2 = # columns
void kNN::reshapeMatrix(MatrixXd &out, double* data, int dim1, int dim2) {
    for(int i = 0; i < dim2; i++) {
        for (int j = 0; j < dim1; j++) {
            out(j, i) = data[j + i*dim1];
        }
    }
}


//diffIndex class. Used to keep track of a particles index and difference. 
diffIndex::diffIndex(double diff, int index) {
    this->diff = diff;
    this->index = index;
}

diffIndex::~diffIndex() {

}

//Used in order to use c++ list sort function
bool operator<(const diffIndex &data1, const diffIndex &data2) {
    return data1.diff < data2.diff;
}

//TODO: PROTECT ABSTRACTION? Probably pointless
double diffIndex::getDiff() {
    return this->diff;
}

int diffIndex::getIndex() {
    return this->index;
}
