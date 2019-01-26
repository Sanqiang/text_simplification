#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titan
#SBATCH --job-name=dmass_eval
#SBATCH --output=dmass_eval.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --qos=long
#SBATCH --mem=16g

# Load modules
module restore

# Run the job
srun python ../../model/eval.py -ngpus 1 -bsize 128 -fw transformer -out dmass -layer_drop 0.2 -op adagrad -lr 0.1 --mode dress -dim 512 -nh 8 -nhl 4 -nel 4 -ndl 4 -lc True --min_count 5 -eval_freq 0 --memory rule -memcfg mofinal --memory_prepare_step 50000
