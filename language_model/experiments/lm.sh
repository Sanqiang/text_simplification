#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=gtx1080
#SBATCH --job-name=lm
#SBATCH --output=lm.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00 # This is the max
#SBATCH --qos=long

# Load modules
module restore

# Run the job
srun python ../../language_model/train.py