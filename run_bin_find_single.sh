#!/bin/bash 


#!/bin/sh
#SBATCH -N 1 # Number of nodes
#SBATCH -n 1 # Number of cores
#SBATCH -p small # Queue
#SBATCH -J R10_bin # Job name
#SBATCH -o bin_%j.o # Name of stdout output file 
#SBATCH -e bin_%j.e # Name of stderr error file 
#SBATCH -A AST23034
#SBATCH -t 48:00:00 # time in hours:minutes:sec ibrun ./hyperion input output

module load hdf5
module unload impi
module load python3/3.9.2


python3 run_batch.py 0 -1 8.0


    


