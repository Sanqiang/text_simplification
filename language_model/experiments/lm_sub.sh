#!/usr/bin/env bash
#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titan
#SBATCH --job-name=lm_sub50k
#SBATCH --output=lm_sub50k.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00 # This is the max
#SBATCH --qos=long

# Load modules
module restore

# Run the job
srun python ../../language_model/train.py -ns 5000 -dim 512 -nh 8 -maxlen 300 -nhl 6 -svoc_size 50000 -out lm_sub50k -bsize 15