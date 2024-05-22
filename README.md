# Spinodal_ARAI

Steps to run the code.

1) Place the GP_AMReX_GUI.sif file adjacent to the Spinodal_code folder

2) Open the path of the sif file in terminal (right click in the same folder where sif file is present and click open in terminal)

3) Type the following command to enter the singularity "singularity shell --bind /run/user,./Spinodal_code:/mnt GP_AMReX_GUI.sif"  

4) Locate to the Exec folder inside Spinodal code and type make (one can also type make -j4, this will use 4 processors to compile the code and will be faster)

5) Once make is complete an executable will be generated. To run the code of 4 processors, type mpirun -n 4 ./main2d.gnu.MPI.ex inputs 

6) View the plotfile in paraview  
