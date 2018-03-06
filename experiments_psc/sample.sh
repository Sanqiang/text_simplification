#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --ntasks-per-node 7
#SBATCH --gres=gpu:p100:1
#SBATCH -t 5:00:00

#echo commands to stdout
set -x

#move to working directory
cd $SCRATCH

#copy input file to working directory
cp $PROJECT/input.data .

#run GPU program
./mygpu

#copy output file to persistent storage
cp output.data $PROJECT