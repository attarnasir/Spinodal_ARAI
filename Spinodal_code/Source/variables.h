#ifndef VARIABLES_H
#define VARIABLES_H

#include <AMReX_Utility.H>
using namespace amrex;

int n_cell{0};
int max_grid_size{0};
int nsteps{0};
int plot_int;
Real dt;
Real c0;
Real Kap;
Real M;
int var_mob{0};

void (*advance)(MultiFab& phi_old,
              MultiFab& phi_new,
	      MultiFab& DFDC,
          MultiFab& mob,
	      Real& K,
	      Real& M,              
              Real& dt,
              Geometry const& geom);


#endif