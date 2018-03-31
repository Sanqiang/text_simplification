#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titan
#SBATCH --job-name=dressnew_adagrad_eval
#SBATCH --output=dressnew_adagrad_eval.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --qos=long
#SBATCH --mem=16g

# Load modules
module restore

# Run the job
srun python ../../model/eval.py -ngpus 1 -bsize 32 -fw transformer -out dn_adagrad -layer_drop 0.0 -op adagrad -lr 0.01 --mode dressnew -nhl 4 -nel 4 -ndl 4 -lc True --pointer_mode ptr --min_count 4

