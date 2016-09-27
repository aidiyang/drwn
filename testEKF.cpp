#include "mujoco.h"
#include <string.h>
#include <iostream>
#include <random>
#include <functional>
#include <omp.h>

#include <math.h>
#include <vector>
#include <Eigen/Dense>
#include <cmath>

//#include "estimator.h"
#include "ekf_estimator.h"

mjModel* m; // defined in viewer_lib.h to capture perturbations
mjData* d;

int main(int argc, const char** argv) {

	printf("start\n");
	//string model_name = argv[1];
	mj_activate("mjkey.txt");
	m = mj_loadXML(argv[1], 0, 0, 0);
	d = mj_makeData(m);

	printf("Loaded model and made data\n");

	d->qpos[0] = 1;
	d->qpos[1] = 1;

	EKF* ekf = new EKF(m, d, 1.0, 1.0, 1.0, 1e-3);

	mjData* d2 = mj_makeData(m);
	std::vector<mjData*> frame;


	MatrixXd deriv = ekf->get_deriv(m, d, d2);
	printf("Calc deriv done\n");
	printf("nu: %d\n", m->nu);
	printf("L: %d\n", m->nq+m->nv);

	IOFormat CleanFmt(3, 0, ", ", "\n", "[", "]");

	std::cout << deriv.rows() << "\t" << deriv.cols() << std::endl;
	std::cout << deriv.block(0, 0, m->nq+m->nv, m->nq+m->nv).format(CleanFmt) << std::endl;

	//MatrixXd sigma = MatrixXd::Identity(m->nq+m->nv+m->nv, m->nq+m->nv+m->nv);
	//printf("%f\n", d->qpos[10]);
	//double dt = m->opt.timestep;
	//ekf->predict_correct(d->ctrl, dt, d->sensordata, conf);



}