#!/usr/bin/env bash


#SBATCH --cluster=gpu
#SBATCH --gres=gpu:1
#SBATCH --partition=titan
#SBATCH --job-name=dmass_pt_eval04
#SBATCH --output=dmass_pt_eval04.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-00:00:00
#SBATCH --qos=long
#SBATCH --mem=16g

# Load modules
module restore

# Run the job
srun python ../../model/eval.py -pt glove -ngpus 1 -bsize 128 -fw transformer -out dmass_pt04 -layer_drop 0.2 -op adagrad -lr 0.1 --mode dress -dim 300 -nh 5 -nhl 4 -nel 4 -ndl 4 -lc True --min_count 5 -eval_freq 0 --memory rule -memcfg mofinal --rule_threshold 0.4 --memory_prepare_step 50000
