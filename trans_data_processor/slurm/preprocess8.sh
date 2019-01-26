#!/usr/bin/env bash


#SBATCH --cluster=smp
#SBATCH --partition=high-mem
#SBATCH --job-name=p8
#SBATCH --output=p8.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --qos=short
#SBATCH --mem=32g

# Load modules
module restore

# Run the job
srun python ../preprocess.py