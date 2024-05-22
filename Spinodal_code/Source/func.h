#ifndef FUNC_H_
#define FUNC_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BCRec.H>
#include <AMReX_BCUtil.H>
#include <AMReX_BC_TYPES.H>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <AMReX_ParallelDescriptor.H>

#include "variables.h"
#include "kernel.h"

using namespace amrex;


void advance_const_mob (amrex::MultiFab& phi_old, amrex::MultiFab& phi_new, amrex::MultiFab& DFDC, amrex::MultiFab& mob, amrex::Real& Kap, amrex::Real& M, amrex::Real& dt, amrex::Geometry const& geom){

    phi_old.FillBoundary(geom.periodicity());

    // Advance the solution one grid at a time
    for ( MFIter mfi(phi_old); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        
        auto const& phiOld = phi_old.array(mfi);
        auto const& phiNew = phi_new.array(mfi);
        auto const& dfdc = DFDC.array(mfi);

        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
            amrex::ParallelFor(vbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {  
                dGdc(i,j,k,phiOld,dfdc,dx[0],dx[1],Kap);
        
            });
        }

        DFDC.FillBoundary(geom.periodicity());

        for ( MFIter mfi(phi_old); mfi.isValid(); ++mfi )
        {
            const Box& validbx = mfi.validbox();
            
            auto const& phiOLD = phi_old.array(mfi);
            auto const& phiNEW = phi_new.array(mfi);
            auto const& dfdc = DFDC.array(mfi);
            
            GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
                amrex::ParallelFor(validbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                
                    update_phi(i,j,k,phiOLD,phiNEW,dfdc,M,dt,dx[0],dx[1]);
                
                });
        }
    
              }

void advance_var_mob (amrex::MultiFab& phi_old, amrex::MultiFab& phi_new,  amrex::MultiFab& DFDC, amrex::MultiFab& mob, amrex::Real& Kap, amrex::Real& M, amrex::Real& dt, amrex::Geometry const& geom){
    phi_old.FillBoundary(geom.periodicity());

    // Advance the solution one grid at a time
    for ( MFIter mfi(phi_old); mfi.isValid(); ++mfi )
    {
        const Box& vbx = mfi.validbox();
        
        auto const& phiOld = phi_old.array(mfi);
        auto const& phiNew = phi_new.array(mfi);
        auto const& dfdc = DFDC.array(mfi);
        auto const& mobil = mob.array(mfi);
        
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
            amrex::ParallelFor(vbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                dGdc(i,j,k,phiOld,dfdc,dx[0],dx[1],Kap); 
                
                mobility(i,j,k,phiOld,mobil);     
            });
        }

        DFDC.FillBoundary(geom.periodicity());
        mob.FillBoundary(geom.periodicity());
        

        for ( MFIter mfi(phi_old); mfi.isValid(); ++mfi )
        {
            const Box& validbx = mfi.validbox();
            
            auto const& phiOLD = phi_old.array(mfi);
            auto const& phiNEW = phi_new.array(mfi);
            auto const& mobil = mob.array(mfi);
            auto const& dfdc = DFDC.array(mfi);

            GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
                amrex::ParallelFor(validbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {   
                    update_phi(i,j,k,phiOLD,phiNEW,dfdc,mobil,dt,dx[0],dx[1]);
                });
        }
}

#endif
