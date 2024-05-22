#ifndef INIT_H
#define INIT_H

#include <AMReX_Utility.H>
#include "variables.h"
#include "func.h"

using namespace amrex;

void init_phi(amrex::MultiFab& phi_new, amrex::Geometry const& geom, amrex::Real c0){

    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
    GpuArray<Real,AMREX_SPACEDIM> prob_lo = geom.ProbLoArray();
    srand(time(0));
    // =======================================
    // Initialize phi_new by calling a Fortran routine.
    // MFIter = MultiFab Iterator
    for (MFIter mfi(phi_new); mfi.isValid(); ++mfi)
    {
        const Box& vbx = mfi.validbox();
        auto const& phiNew = phi_new.array(mfi);
        amrex::ParallelFor(vbx,
        [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {   
            Real noise = 0.02;
            amrex::RandomEngine rd;
            phiNew(i,j,k) = c0 + noise*(0.5 - amrex::Random(rd));

        });
    }
}

void init_function(){
     
     if(var_mob == 0){
        advance = advance_const_mob;
     }

     if(var_mob == 1){
        advance = advance_var_mob;
     }
}



#endif