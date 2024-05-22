#ifndef KERNEL_H_
#define KERNEL_H_

#include <AMReX_FArrayBox.H>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <AMReX_ParallelDescriptor.H>
#include <time.h>



AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void update_phi (int i, int j, int k,
                 amrex::Array4<amrex::Real const> const& phiold,
                 amrex::Array4<amrex::Real      > const& phinew,
				amrex::Array4<amrex::Real      > const& dfdc,
				amrex::Array4<amrex::Real      > const& mobil,
                 amrex::Real dt, amrex::Real dx, amrex::Real dy)
{
    
	Real mob_iph = 0.5*(mobil(i+1,j,k) + mobil(i,j,k));
	Real mob_imh = 0.5*(mobil(i-1,j,k) + mobil(i,j,k));
	Real mob_jph = 0.5*(mobil(i,j+1,k) + mobil(i,j,k));
	Real mob_jmh = 0.5*(mobil(i,j-1,k) + mobil(i,j,k));

	Real der_iph = (dfdc(i+1,j,k) - dfdc(i,j,k))/dx;
	Real der_imh = (dfdc(i,j,k) - dfdc(i-1,j,k))/dx;
	Real der_jph = (dfdc(i,j+1,k) - dfdc(i,j,k))/dy;
	Real der_jmh = (dfdc(i,j,k) - dfdc(i,j-1,k))/dy;

	phinew(i,j,k) = phiold(i,j,k) + (dt*((mob_iph*der_iph - mob_imh*der_imh)/dx + 
										  (mob_jph*der_jph - mob_jmh*der_jmh)/dy ));
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void update_phi (int i, int j, int k,
                 amrex::Array4<amrex::Real const> const& phiold,
                 amrex::Array4<amrex::Real      > const& phinew,
				amrex::Array4<amrex::Real      > const& dfdc,
				amrex::Real M,
                 amrex::Real dt, amrex::Real dx, amrex::Real dy)
{
    phinew(i,j,k) = phiold(i,j,k) + (dt * M * ((dfdc(i-1,j,k) + dfdc(i+1,j,k) + dfdc(i,j-1,k) + dfdc(i,j+1,k) - 4.0 * dfdc(i,j,k))/(dx*dy)));

}


// AMREX_GPU_DEVICE AMREX_FORCE_INLINE
// void der_free_energy(int i, int j, int k,
// 		amrex::Array4<amrex::Real const> const& phiOld,
// 		amrex::Array4<amrex::Real      > const& deriv)
// {
		
//         deriv(i,j,k) = (6.214e18/6e23)*(-2345.293*phiOld(i,j,k)*(1-phiOld(i,j,k))*(8*phiOld(i,j,k)-4) + 5137.25*phiOld(i,j,k)*(1-phiOld(i,j,k)) + 
//                                         2345.293*phiOld(i,j,k)*(2*phiOld(i,j,k)-1)*(2*phiOld(i,j,k)-1) - 2568.625*phiOld(i,j,k)*(2*phiOld(i,j,k)-1) -  
//                                         24179.86*phiOld(i,j,k) - 2345.293*(1-phiOld(i,j,k))*(2*phiOld(i,j,k)-1)*(2*phiOld(i,j,k)-1) +
//                                         2568.625*(1-phiOld(i,j,k)*(2*phiOld(i,j,k)-1) + 4167.994*log(phiOld(i,j,k))) - 
//                                         7052.907*log(1-phiOld(i,j,k)) + 20064.944 - (7052.907-7052.907*phiOld(i,j,k))/(1-phiOld(i,j,k)));

// 		//deriv(i,j,k) = (6.214e18/6e23)*(2.0*phiOld(i,j,k)*(1.0-phiOld(i,j,k))*(1.0-phiOld(i,j,k)) - 2.0*phiOld(i,j,k)*phiOld(i,j,k)*(1.0 - phiOld(i,j,k)));

// }


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void dGdc(int i, int j, int k, amrex::Array4<amrex::Real const> const& phiOld, amrex::Array4<amrex::Real> const& dfdc, Real dx, Real dy, Real kappa){
		
		Real g = (6.214e18/6e23)*(-2345.293*phiOld(i,j,k)*(1-phiOld(i,j,k))*(8*phiOld(i,j,k)-4) + 5137.25*phiOld(i,j,k)*(1-phiOld(i,j,k)) + 
                                        2345.293*phiOld(i,j,k)*(2*phiOld(i,j,k)-1)*(2*phiOld(i,j,k)-1) - 2568.625*phiOld(i,j,k)*(2*phiOld(i,j,k)-1) -  
                                        24179.86*phiOld(i,j,k) - 2345.293*(1-phiOld(i,j,k))*(2*phiOld(i,j,k)-1)*(2*phiOld(i,j,k)-1) +
                                        2568.625*(1-phiOld(i,j,k)*(2*phiOld(i,j,k)-1) + 4167.994*log(phiOld(i,j,k))) - 
                                        7052.907*log(1-phiOld(i,j,k)) + 20064.944 - (7052.907-7052.907*phiOld(i,j,k))/(1-phiOld(i,j,k)));

		//Real g = (2.0*phiOld(i,j,k)*(1.0-phiOld(i,j,k))*(1.0-phiOld(i,j,k)) - 2.0*phiOld(i,j,k)*phiOld(i,j,k)*(1.0 - phiOld(i,j,k)));

		Real lapl = (phiOld(i-1,j,k) + phiOld(i+1,j,k) + phiOld(i,j-1,k) + phiOld(i,j+1,k) - 4.0 * phiOld(i,j,k))/(dx*dy);

		dfdc(i,j,k) = g - kappa*lapl;
}


// AMREX_GPU_DEVICE AMREX_FORCE_INLINE
// void laplacian(int i, int j, int k,
// 		amrex::Array4<amrex::Real const> const& phiOld,
// 		amrex::Array4<amrex::Real      > const& lap,
// 		Real dx, Real dy)
// {
	
//         lap(i,j,k) = (phiOld(i-1,j,k) + phiOld(i+1,j,k) + phiOld(i,j-1,k) + phiOld(i,j+1,k) - 4.0 * phiOld(i,j,k))/(dx*dy);
        		
	
// }


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void mobility(int i, int j, int k,
		amrex::Array4<amrex::Real const> const& phiOld,
		amrex::Array4<amrex::Real      > const& mob
		)
{
		mob(i,j,k) = (1e18*6e23/6.214e18)*(pow((1-phiOld(i,j,k)),2)*phiOld(i,j,k)*pow(10,((-32.770*phiOld(i,j,k)) + (-25.818*(1-phiOld(i,j,k))) + (-3.29*phiOld(i,j,k)*log(phiOld(i,j,k))) + 
																			 (17.66*(1-phiOld(i,j,k))*log(1-phiOld(i,j,k))) + (37.619*phiOld(i,j,k)*(1-phiOld(i,j,k)))  +
																			 (20.69*phiOld(i,j,k)*(1-phiOld(i,j,k))*(2*phiOld(i,j,k)-1)) + (10.8*phiOld(i,j,k)*(1-phiOld(i,j,k))*(pow((2*phiOld(i,j,k)-1),2)))))
										   +
										   pow(phiOld(i,j,k),2)*(1-phiOld(i,j,k))*pow(10,((-31.68*phiOld(i,j,k)) + (-26.02*(1-phiOld(i,j,k))) + (0.22*phiOld(i,j,k)*log(phiOld(i,j,k))) + 
										   									              (24.36*(1-phiOld(i,j,k))*log(1-phiOld(i,j,k))) + (44.33*phiOld(i,j,k)*(1-phiOld(i,j,k))) + 
																						  (8.73*phiOld(i,j,k)*(1-phiOld(i,j,k))*(2*phiOld(i,j,k)-1)) + (-20.95*phiOld(i,j,k)*(1-phiOld(i,j,k))*pow((2*phiOld(i,j,k)),2)))) );   		

}


#endif