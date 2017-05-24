#include "mujoco.h"
#include <string.h>
#include <iostream>
#include <random>
#include <functional>
#include <vector>
#include <stdlib.h>
#include <set>
#include "util_func.h"

#include <math.h>

#include <Eigen/Dense>
#include <Eigen/Eigen>
#include "H5Cpp.h"
#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

using namespace Eigen;

class diffIndex {
        public:
            diffIndex(double diff = 0, int index = 0);

            ~diffIndex();

            friend bool operator<(const diffIndex &data1, const diffIndex &data2);
            double getDiff();
            int getIndex();

        private:
            double diff;
            int index;

};

class kNN {
    public:
        kNN(int _nq, int _nv, int _ns, int MAX_SIZE);

        ~kNN();

        bool savePart(double* particles, double* nextPart, double* sensPart, double* nextSensPart);
        Eigen::VectorXd findPart(double* sensorsData);
        int parallel_findPart(double* sensorsData, int s, int e);
        void findNextPart(double* particle, double* sensPart, int index, int n);
        std::list<diffIndex> parallel_findNext(double* particle, double* sensPart, int index, int n, int s, int e);
        int getSize();
        Eigen::VectorXd getEst(double* sensors, int n);
        std::vector<double*> getClose(double* sensors, int n);
        std::list<diffIndex> parallel_Est(double* sensors, VectorXd *rms, int n, int s, int e);
        void saveData(std::string FILE_NAME);
        void readFile(std::string FILE_NAME);
        void flatMatrix(MatrixXd data, double* arr, int dim1, int dim2);
        void reshapeMatrix(MatrixXd &out, double* data, int dim1, int dim2);
        void sort(std::vector<diffIndex> &data, diffIndex input, int size);
        void printSmall();

    private:
        Eigen::MatrixXd states;
        Eigen::MatrixXd nextStates;
        Eigen::MatrixXd sensors;
        Eigen::MatrixXd nextSens;
        int size;
        int nq;
        int nv;
        int ns;
        

    
};


// class diffIndex {
//         public:
//             diffIndex(double diff, int index);

//             ~diffIndex();

//             friend bool operator<(diffIndex &data1, diffIndex &data2);
//             double getDiff();
//             int getIndex();

//         private:
//             double diff;
//             int index;

// };