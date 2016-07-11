#include <stdio.h>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace Eigen;

int main (int argc, char* argv[]) {

    MatrixXd R(3,3);
    R << 0.6754,  0.7334,  0.0771,
      0.5765, -0.4599, -0.6754,
      -0.4599,  0.5007, -0.7334;

    std::cout<<"Matrix:\n"<< R << "\n\n";

    Vector3d va1 = R.row(1) - R.row(0);
    Vector3d va2 = R.row(2) - R.row(0);

    std::cout<<"Vec 1:\n"<< va1.transpose() << "\n";
    std::cout<<"Vec 2:\n"<< va2.transpose() << "\n\n";

    Vector3d cross1 = va1.cross(va2);

    std::cout<<"Cross Product:\n"<< cross1.transpose() << "\n";

    cross1.normalize();
    std::cout<<"Cross Product normalized:\n"<< cross1.transpose() << "\n";

    //MatrixXd ker = cross1.fullPivLu().kernel();
    //std::cout<<"LU Kernel: "<< ker << std::endl;

    JacobiSVD<MatrixXd> svd(cross1, Eigen::ComputeFullU);
    //JacobiSVD<MatrixXd> svd(cross1, Eigen::ComputeFullV | Eigen::ComputeFullU);
    //std::cout<<"SVD Decomp V:\n"<< svd.matrixV() << std::endl;
    std::cout<<"SVD Decomp U:\n"<< svd.matrixU() << std::endl;
    MatrixXd U = svd.matrixU();

    //JacobiSVD<MatrixXd> svd2(cross1, Eigen::ComputeThinV | Eigen::ComputeThinU);
    //std::cout<<"SVD Thin V: "<< svd2.matrixV() << std::endl;
    //std::cout<<"SVD Thin U: "<< svd2.matrixU() << std::endl;

    MatrixXd rot(3, 3);
    rot.col(0) = U.col(1);
    rot.col(1) = U.col(2);
    rot.col(2) = U.col(0);

    std::cout<<"\nRotation:\n"<< rot << std::endl;
    //rot =
    //   -0.4470    0.7688    0.4573
    //    0.8629    0.2358    0.4470
    //    0.2358    0.5944   -0.7688
    std::cout<<"\nRotated:\n"<< R*rot << std::endl;
    //xyz*rot =
    //    0.3491    0.7381    0.5774
    //   -0.8138   -0.0667    0.5774
    //    0.4646   -0.6714    0.5774
    //
    double data[] = {1,2,3,4,5,6};
    double * p = data;
    Map<Vector3d> v(p);
    v = v*2;
    printf("Mapped: ");
    for (int i=0; i<6; i++){
        printf("%1.4f ", data[i]);
    }
    printf("\n");

    return 0;
}

