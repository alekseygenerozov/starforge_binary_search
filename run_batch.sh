#!/bin/bash 

#SBATCH -J gas_halo
#SBATCH --output=job.%J.out.txt
#SBATCH --error=job.%J.err.txt
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=NTASKS

#SBATCH --time=2-00:00:00

python gas_batch.py $SLURM_ARRAY_TASK_ID
    


