#!/bin/sh
#SBATCH -N 1 # Number of nodes
#SBATCH -n 9 # Number of cores
#SBATCH -p small # Queue
#SBATCH -J R10_sink # Job name
#SBATCH -o sink.o%j # Name of stdout output file 
#SBATCH -e sink.e%j # Name of stderr error file 
#SBATCH -A AST21002
#SBATCH -t 3:00:00 # time in hours:minutes:sec ibrun ./hyperion input output

## To run this file: sbatch submit_radmc.sh

export OMP_NUM_THREADS=9

module load launcher
export LAUNCHER_RMI=SLURM
export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins

#                         # JOB_FILE is a list of executions to run

export LAUNCHER_JOB_FILE=`pwd`/commands_sink
export LAUNCHER_SCHED=interleaved
export LAUNCHER_WORKDIR=`pwd`

$LAUNCHER_DIR/paramrun
