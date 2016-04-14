
#include "viewer_lib.h"
#include "darwin_hw/sim_interface.h"


std::string model_name;

extern mjModel* m; // defined in viewer_lib.h to capture perturbations
extern mjData* d;

int main(int argc, const char** argv) {

  if( argc==2 )
	model_name = argv[1];

  if (init_viz(model_name)) { return 1; }

  // init darwin 'robot'
  double s_noise = 0.0;
  double s_time_noise = 0.0;
  SimDarwin *robot = new SimDarwin(m, d, s_noise, s_time_noise);

  int nq = m->nq;
  int nv = m->nv;
  int nu = m->nu;
  int nsensordata = m->nsensordata;
  double time = 0.0;
  double *qpos = new double[nq];
  double *qvel = new double[nv];
  double *ctrl = new double[nu];
  double *sensors = new double[nsensordata];

  // init estimator from darwin 'robot'
  
  printf("DT is set %f\n", m->opt.timestep);
  while( !closeViewer() ) {
	// simulate and render
   //printf("time: %f\t", d->time);
	if (robot->get_state(&time, qpos, qvel, sensors)) {
	  // we got sensor values

	  //robot->set_controls(ctrl);
	  printf("got new sensors at %f \n", time);
	}
	else {
	  printf("no  new sensors at %f \n", time);
	  // we just advanced the sim, nothing else

	}
	render(window, NULL); // get state updated model / data

	finalize();
  }

  end_viz();
  // end_estimator
  // end darwin 'robot'
  //for(int id = 0; id < nq; id++) {
  //  printf("%f ", qpos[id]);
  //}

  printf("\n");
  delete[] qpos;
  delete[] qvel;
  delete[] ctrl;
  delete[] sensors;

  return 0;
}
