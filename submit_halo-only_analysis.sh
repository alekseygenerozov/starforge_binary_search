#!/bin/sh
#SBATCH -N 1 # Number of nodes
#SBATCH -n 16 # Number of cores
#SBATCH -p small # Queue
#SBATCH -J R10_halo # Job name
#SBATCH -o halo.o%j # Name of stdout output file 
#SBATCH -e halo.e%j # Name of stderr error file 
#SBATCH -A AST21002
#SBATCH -t 48:00:00 # time in hours:minutes:sec ibrun ./hyperion input output

##  sbatch submit_halo-only_analysis.sh
##  Need to set n = OMP_NUM_THREADS = NTASKS on line 280:
##   with multiprocessing.Pool(NTASKS) as pool: in halo*par.py

export OMP_NUM_THREADS=16

module load launcher
module unload intel/19.1.1
module unload impi/19.0.9
module load gcc/9.1.0
module load python3/3.9.2
source $HOME/_find_multiples/multenv/bin/activate
#module unload python3/3.9.2

# n < 100 (prepend a 0)
python run_batch_halo_only_par.py 100 489

#python run_batch_halo_only_par0.py 100 254




